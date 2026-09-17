"""Small numerical and artifact checks for the producer-only embedded layout."""
import gzip
import json
import pickle

import hist
import numpy as np
import pytest
from coffea.processor.accumulator import accumulate
from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist

from analysis.topeft_run2.analysis_processor import (
    AnalysisProcessor, prepare_eft_coefficients,
)
from topeft.modules.axes import info, info_2d
from topeft.modules.embedded_sumw2 import (
    COVERAGE_KEY, EMBEDDED_SCHEMA_VERSION, EMBEDDED_LAYOUT,
    fill_embedded_nominal, coverage_manifest,
)
from topeft.modules.nominal_schema import (
    scalar_nominal_key, eft_nominal_key, validate_nominal_mapping,
    merge_nominal_mappings,
)
from topeft.modules.histogram_artifact import (
    write_histogram_artifact, validate_histogram_artifact,
    metadata_sidecar_path, histogram_artifact_error,
)
from topeft.modules.sumw2_policy import (
    resolve_sumw2_storage_policy, resolved_policy_from_provenance,
)
from sumw2_profile_test_helpers import certify_test_profile


SAMPLES = {
    "selected": {"histAxisName": "shared", "isData": False, "WCnames": []},
    "unselected": {"histAxisName": "shared", "isData": False, "WCnames": []},
    "eft": {"histAxisName": "signal", "isData": False, "WCnames": ["ctG"]},
}


def policy():
    return resolve_sumw2_storage_policy(
        {"mode": "full_custom", "rules": [{"dataset_names": ["selected", "eft"]}]},
        samples=SAMPLES, runtime_families=("njets",), axes_info=info,
        axes_info_2d=info_2d, sumw2_storage_present=True,
    )


def axes(name="njets"):
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.Regular(2, 0, 2, name=name),
    )


def payload():
    # Exercise the backend independently of optional raw-count dependency support.
    return {
        COVERAGE_KEY: set(),
        scalar_nominal_key("njets"): HistEFT(*axes(), wc_names=[], use_multicell=True, store_sumw2=True),
        eft_nominal_key("njets"): HistEFT(*axes(), wc_names=["ctG"], use_multicell=True, store_sumw2=True),
    }


def fill(output, dataset, weights, coefficients=None, systematic="nominal", values=None):
    component = "eft_nominal" if dataset == "eft" else "scalar_nominal"
    histogram = output[f"njets__{component}"]
    weights = np.asarray(weights, dtype=float)
    fill_embedded_nominal(
        histogram, output[COVERAGE_KEY], family="njets", component=component,
        dataset=dataset, selected=policy().selects(dataset, SAMPLES[dataset]["histAxisName"], "njets"),
        process=SAMPLES[dataset]["histAxisName"], channel="3l", appl="isSR",
        systematic=systematic, njets=np.full(len(weights), .5) if values is None else values,
        weight=weights, eft_coeff=coefficients,
    )


def totals(output, component="scalar_nominal", systematic="nominal"):
    histogram = output[f"njets__{component}"]
    key = ("signal" if component == "eft_nominal" else "shared", "3l", systematic, "isSR")
    return np.sum(histogram.eval({})[key]), np.sum(histogram.nominal_sumw2(flow=True)[key])


def validate(output):
    validate_nominal_mapping(output, runtime_families=("njets",),
                             schema_version=EMBEDDED_SCHEMA_VERSION, policy=policy())


@pytest.mark.parametrize("dataset,expected_variance", [("selected", 13), ("unselected", 0)])
def test_selected_and_unselected_scalar_targets(dataset, expected_variance):
    output = payload()
    fill(output, dataset, [2, -3])
    assert totals(output) == (-1, expected_variance)
    validate(output)
    assert not any(name.endswith("_sumw2") for name in output)


def test_datasets_sharing_process_do_not_share_selection():
    output = payload()
    fill(output, "selected", [2, -3])
    fill(output, "unselected", [100, -20])
    assert totals(output) == (79, 13)
    states = {record[2]: record[-1] for record in coverage_manifest(output[COVERAGE_KEY])}
    assert states == {"selected": "selected_nonzero", "unselected": "unavailable"}
    validate(output)


@pytest.mark.parametrize("systematic", ["nominal", "JERUp", "triggerSF_2022Up"])
@pytest.mark.parametrize("dataset", ["selected", "unselected", "eft"])
def test_nominal_versus_systematic_and_eft_constant(dataset, systematic):
    output = payload()
    coeff = np.array([[1.5, 10, 2], [-2, -5, 1]]) if dataset == "eft" else None
    fill(output, dataset, [2, -3], coeff, systematic)
    component = "eft_nominal" if dataset == "eft" else "scalar_nominal"
    expected_yield = 9 if dataset == "eft" else -1
    variance = (45 if dataset == "eft" else 13) if systematic == "nominal" and dataset != "unselected" else 0
    assert totals(output, component, systematic) == (expected_yield, variance)
    validate(output)


@pytest.mark.parametrize("weights", [[0, 0], []])
def test_selected_zero_is_distinct_from_unavailable(weights):
    output = payload()
    fill(output, "selected", weights)
    fill(output, "unselected", weights)
    assert totals(output) == (0, 0)
    assert {r[2]: r[-1] for r in coverage_manifest(output[COVERAGE_KEY])} == {
        "selected": "selected_zero", "unselected": "unavailable",
    }
    validate(output)


def test_masking_remapping_sm_only_and_negative_cancellation():
    output = payload()
    coefficients = np.array([[1., 10, 2], [-1., -5, 1], [20., 3, 2]])
    coefficients = prepare_eft_coefficients(coefficients, ["ctW"], ["ctG"], "sm_only")
    mask = np.array([True, True, False])
    fill(output, "eft", np.array([2., 2., 100.])[mask], coefficients[mask])
    assert totals(output, "eft_nominal") == (0, 8)
    # The projected polynomial remains constant away from the SM.
    assert sum(np.sum(v) for v in output[eft_nominal_key("njets")].eval({"ctG": 2}).values()) == 0
    validate(output)


def test_accumulation_pickle_and_schema_merge():
    first, second = payload(), payload()
    fill(first, "selected", [0])
    fill(second, "selected", [-3])
    fill(second, "unselected", [10])
    merged = accumulate([first, second])
    assert totals(merged) == (7, 9)
    restored = pickle.loads(pickle.dumps(merged))
    validate(restored)
    assert coverage_manifest(restored[COVERAGE_KEY]) == coverage_manifest(merged[COVERAGE_KEY])
    assert {r[2]: r[-1] for r in coverage_manifest(restored[COVERAGE_KEY])}["selected"] == "selected_nonzero"
    schema_merged = merge_nominal_mappings([first, second], runtime_families=("njets",),
                                          schema_version=EMBEDDED_SCHEMA_VERSION, policy=policy())
    assert totals(schema_merged) == (7, 9)


def write(path, output):
    resolved = policy()
    return write_histogram_artifact(
        path, histograms=output, artifact_kind="processor_output",
        sumw2_storage_provenance=resolved.to_provenance(),
        production_sample_contract=certify_test_profile(resolved, SAMPLES),
    )


def test_artifact_policy_and_coverage_round_trip(tmp_path):
    output = payload()
    fill(output, "selected", [0])
    fill(output, "unselected", [10])
    fill(output, "eft", [2], np.array([[1.5, 10, 2]]))
    path = tmp_path / "embedded.pkl.gz"
    sidecar = write(path, output)
    with gzip.open(path, "rb") as stream:
        restored = pickle.load(stream)
    validated = validate_histogram_artifact(path)
    assert validated["schema"] == EMBEDDED_LAYOUT
    assert sidecar["artifact"]["nominal_container_schema_version"] == 3
    assert resolved_policy_from_provenance(validated["metadata"]["sumw2_storage_provenance"]) == policy()
    assert restored[COVERAGE_KEY] == output[COVERAGE_KEY]
    assert sidecar["sumw2_content_manifest"]["embedded_coverage"] == coverage_manifest(output[COVERAGE_KEY])
    assert totals(restored, "eft_nominal") == (3, 9)
    assert "njets_sumw2" not in restored


def test_only_unselected_dataset_of_shared_process_can_be_published(tmp_path):
    output = payload()
    fill(output, "unselected", [10])
    sidecar = write(tmp_path / "unselected.pkl.gz", output)
    assert sidecar["sumw2_content_manifest"]["families"]["njets"]["required_sumw2_processes"] == []


@pytest.mark.parametrize("corruption", ["missing_coverage", "wrong_selection", "companion", "missing_storage", "numerical"])
def test_embedded_schema_rejects_invalid_payload(corruption):
    output = payload()
    fill(output, "selected", [2])
    if corruption == "missing_coverage":
        output[COVERAGE_KEY].clear()
    elif corruption == "wrong_selection":
        record = next(iter(output[COVERAGE_KEY]))
        output[COVERAGE_KEY] = {record[:-1] + ("unavailable",)}
    elif corruption == "companion":
        output["njets_sumw2"] = SparseHist(*axes("njets_sumw2"), storage="Double")
    elif corruption == "missing_storage":
        output[scalar_nominal_key("njets")] = HistEFT(*axes(), wc_names=[], use_multicell=True)
    else:
        histogram = output[scalar_nominal_key("njets")]
        histogram.fill(process="shared", channel="3l", appl="isSR", systematic="JERUp",
                       njets=np.array([.5]), weight=np.array([3]), fill_sumw2=True)
    with pytest.raises((ValueError, TypeError)):
        validate(output)


def test_artifact_rejects_tampered_coverage_and_missing_sidecar(tmp_path):
    output = payload()
    fill(output, "selected", [2])
    path = tmp_path / "embedded.pkl.gz"
    write(path, output)
    metadata_path = metadata_sidecar_path(path)
    metadata = json.loads(metadata_path.read_text())
    metadata["sumw2_content_manifest"]["embedded_coverage"][0][-1] = "selected_zero"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(histogram_artifact_error, match="coverage"):
        validate_histogram_artifact(path)
    metadata_path.unlink()
    with pytest.raises(histogram_artifact_error, match="sidecar"):
        validate_histogram_artifact(path)


def test_main_producer_storage_and_unchanged_2d():
    processor = AnalysisProcessor(samples=SAMPLES, wc_names_lst=["ctG"],
                                  hist_lst=["njets", "lepton_pt_vs_eta"])
    for key in (scalar_nominal_key("njets"), eft_nominal_key("njets")):
        histogram = processor.accumulator[key]
        assert type(histogram) is HistEFT and histogram._use_multicell and histogram.store_sumw2
    assert "njets_sumw2" not in processor.accumulator
    for key in ("lepton_pt_vs_eta", "lepton_pt_vs_eta_sumw2"):
        assert type(processor.accumulator[key]) is SparseHist
        assert processor.accumulator[key]._init_args["storage"] == "Double"


@pytest.mark.parametrize("legacy_backend", [True, False])
def test_historical_uniform_artifacts_remain_readable(tmp_path, legacy_backend):
    histogram = HistEFT(*axes(), wc_names=["ctG"], use_multicell=not legacy_backend)
    histogram.fill(process="signal", channel="3l", appl="isSR", systematic="nominal",
                   njets=np.array([.5]), weight=np.array([-2.]),
                   eft_coeff=np.array([[1.5, 2, 3]]))
    path = tmp_path / "historical.pkl.gz"
    with gzip.open(path, "wb") as stream:
        pickle.dump({"njets": histogram}, stream)
    assert validate_histogram_artifact(path)["schema"] == "legacy_uniform"
    with gzip.open(path, "rb") as stream:
        restored = pickle.load(stream)["njets"]
    assert sum(np.sum(v) for v in restored.eval({}).values()) == -3


def test_split_sibling_v2_artifact_still_requires_physical_companion(tmp_path):
    histogram = SparseHist(*axes(), storage="Double")
    histogram.fill(process="shared", channel="3l", appl="isSR", systematic="nominal",
                   njets=np.array([.5]), weight=np.array([2.]))
    companion = SparseHist(*axes("njets_sumw2"), storage="Double")
    companion.fill(process="shared", channel="3l", appl="isSR", systematic="nominal",
                   njets_sumw2=np.array([.5]), weight=np.array([4.]))
    output = {scalar_nominal_key("njets"): histogram, "njets_sumw2": companion}
    path = tmp_path / "split.pkl.gz"
    sidecar = write(path, output)
    assert sidecar["artifact"]["nominal_container_schema_version"] == 2
    assert validate_histogram_artifact(path)["schema"] == "split_sibling_v1"
    del output["njets_sumw2"]
    with pytest.raises((ValueError, histogram_artifact_error), match="companion|sumw2"):
        write(tmp_path / "broken_split.pkl.gz", output)


@pytest.mark.parametrize("record_raw_count", [False, True])
def test_producer_fill_with_upstream_raw_count_contract(record_raw_count):
    processor = AnalysisProcessor(samples=SAMPLES, wc_names_lst=["ctG"],
                                  hist_lst=["ptz"], record_raw_count=record_raw_count)
    output = processor.accumulator
    histogram = output[scalar_nominal_key("ptz")]
    classification = processor._raw_count_fill_classification(
        histogram, is_data=False, wgt_fluct="nominal")
    extra = {} if classification is None else {"record_raw_count": classification}
    fill_embedded_nominal(histogram, output[COVERAGE_KEY], family="ptz",
                          component="scalar_nominal", dataset="selected", selected=True,
                          process="shared", channel="3l", appl="isSR", systematic="nominal",
                          ptz=np.array([25., 75.]), weight=np.array([-2., 3.]), **extra)
    assert sum(np.sum(v) for v in histogram.nominal_sumw2(flow=True).values()) == 13
    if record_raw_count:
        restored = pickle.loads(pickle.dumps(histogram))
        assert sum(np.sum(v) for v in restored.raw_counts(flow=True).values()) == 2
