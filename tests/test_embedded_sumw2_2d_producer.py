"""Main producer 2D numerical, policy, coverage, and persisted layout contract."""
import copy
import gzip
import pickle

import cloudpickle
import hist
import awkward as ak
import numpy as np
import pytest
from coffea.processor.accumulator import accumulate
from topcoffea.modules.sparseHist import SparseHist

from analysis.topeft_run2.analysis_processor import (
    AnalysisProcessor, prepare_eft_coefficients, flatten_jagged_jet_eta_phi_weights,
    evaluate_eft_coefficients_at_sm,
)
from topeft.modules.axes import info, info_2d
from topeft.modules.embedded_sumw2 import (
    COVERAGE_KEY, EMBEDDED_LAYOUT, coverage_manifest, embedded_sumw2_view, scale_embedded_family,
    fill_embedded_nominal,
)
from topeft.modules.nominal_schema import validate_nominal_mapping, merge_nominal_mappings
from topeft.modules.histogram_artifact import (
    write_histogram_artifact, validate_histogram_artifact, merge_histogram_sidecars,
    histogram_artifact_error,
)
from topeft.modules.sumw2_policy import resolve_sumw2_storage_policy, resolved_policy_from_provenance
from sumw2_profile_test_helpers import certify_test_profile

FAMILY = "lepton_pt_vs_eta"
SAMPLES = {
    "selected": {"histAxisName": "shared", "isData": False, "WCnames": []},
    "unselected": {"histAxisName": "shared", "isData": False, "WCnames": []},
    "eft": {"histAxisName": "signal", "isData": False, "WCnames": ["ctG"]},
}


def policy(variables=(FAMILY,), mode="full_custom"):
    block = {"mode": mode}
    if mode == "full_custom":
        block["rules"] = [{"dataset_names": ["selected", "eft"], "variables": list(variables)}]
    return resolve_sumw2_storage_policy(
        block, samples=SAMPLES, runtime_families=(FAMILY,), axes_info=info,
        axes_info_2d=info_2d, sumw2_storage_present=True,
    )


def processor(*, resolved=None, tracked=False):
    return AnalysisProcessor(samples=SAMPLES, wc_names_lst=["ctG"], hist_lst=[FAMILY],
                             sumw2_policy=policy() if resolved is None else resolved,
                             record_raw_count=tracked)


def fill(p, dataset, weights, coeff=None, systematic="nominal", xy=None, output=None):
    weights = np.asarray(weights, dtype=float)
    output = p.accumulator if output is None else output
    names = output[FAMILY].dense_axes.name
    x, y = (np.full(weights.shape, 25.), np.full(weights.shape, .5)) if xy is None else xy
    p._fill_2d_histogram(
        output, FAMILY, dataset, process=SAMPLES[dataset]["histAxisName"], channel="3l",
        appl="isSR", systematic=systematic, weight=weights, eft_coeff=coeff,
        **{names[0]: x, names[1]: y},
    )
    return output


def totals(output, process="shared", systematic="nominal"):
    cell = output[FAMILY].view(flow=True)[(process, "3l", systematic, "isSR")]
    return cell.value.sum(), cell.variance.sum()


def validate(output, resolved=None):
    validate_nominal_mapping(output, runtime_families=(FAMILY,), schema_version=3,
                             policy=policy() if resolved is None else resolved)


def write(path, output, resolved=None, **kwargs):
    resolved = policy() if resolved is None else resolved
    return write_histogram_artifact(
        path, histograms=output, artifact_kind="processor_output",
        sumw2_storage_provenance=resolved.to_provenance(),
        production_sample_contract=certify_test_profile(resolved, SAMPLES), **kwargs,
    )


@pytest.mark.parametrize("dataset,variance", [("selected", 13), ("unselected", 0), ("eft", 45)])
@pytest.mark.parametrize("systematic", ["nominal", "JERUp", "triggerSF_2022Up"])
def test_scalar_eft_negative_values_and_nominal_gate(dataset, variance, systematic):
    p = processor()
    coeff = np.array([[1.5, 10, 2], [-2., -5, 1]]) if dataset == "eft" else None
    output = fill(p, dataset, [2, -3], coeff, systematic)
    assert totals(output, SAMPLES[dataset]["histAxisName"], systematic) == (
        -1, variance if systematic == "nominal" else 0)
    validate(output)
    assert set(output) == {FAMILY, COVERAGE_KEY}
    assert {r[-1] for r in output[COVERAGE_KEY]} == {
        "selected_nonzero" if systematic == "nominal" and variance else "unavailable"}


def test_selected_and_unselected_datasets_share_process_without_variance_leakage():
    p = processor()
    fill(p, "selected", [2, -3])
    output = fill(p, "unselected", [100, -20])
    assert totals(output) == (79, 13)
    assert {r[2]: r[-1] for r in coverage_manifest(output[COVERAGE_KEY])} == {
        "selected": "selected_nonzero", "unselected": "unavailable"}
    validate(output)


@pytest.mark.parametrize("mode", ["production", "production_central", "taufitter",
                                  "full_diagnostics", "disabled", "full_custom"])
def test_all_policy_modes_control_2d_moments_without_gating_values(mode):
    block = {"mode": mode}
    if mode not in {"full_diagnostics", "disabled"}:
        block["rules"] = [{"dataset_names": ["selected"], "variables": [FAMILY]}]
    resolved = resolve_sumw2_storage_policy(
        block, samples=SAMPLES, runtime_families=(FAMILY,), axes_info=info,
        axes_info_2d=info_2d, sumw2_storage_present=True,
        analysis_mode="taufitter" if mode == "taufitter" else "standard",
    )
    p = processor(resolved=resolved)
    for dataset in ("selected", "unselected"):
        fill(p, dataset, [-2, 3])
    expected_variance = 13 * sum(
        resolved.selects(dataset, "shared", FAMILY)
        for dataset in ("selected", "unselected")
    )
    assert totals(p.accumulator) == (2, expected_variance)
    validate(p.accumulator, resolved)
    assert resolved_policy_from_provenance(resolved.to_provenance()) == resolved


@pytest.mark.parametrize("selected_family", ["njets", FAMILY])
def test_mixed_1d_2d_artifact_keeps_family_selection_and_provenance(tmp_path, selected_family):
    resolved = resolve_sumw2_storage_policy(
        {"mode": "full_custom", "rules": [{"variables": [selected_family]}]},
        samples=SAMPLES, runtime_families=("njets", FAMILY), axes_info=info,
        axes_info_2d=info_2d, sumw2_storage_present=True,
    )
    p = AnalysisProcessor(samples=SAMPLES, wc_names_lst=["ctG"],
                          hist_lst=["njets", FAMILY], sumw2_policy=resolved)
    output = fill(p, "selected", [-2, 3])
    fill_embedded_nominal(
        output["njets__scalar_nominal"], output[COVERAGE_KEY], family="njets",
        component="scalar_nominal", dataset="selected",
        selected=resolved.selects("selected", "shared", "njets"),
        process="shared", channel="3l", appl="isSR", systematic="nominal",
        njets=np.array([2., 3.]), weight=np.array([-2., 3.]),
    )
    path = tmp_path / "mixed.pkl.gz"
    write(path, output, resolved)
    validated = validate_histogram_artifact(path)
    assert validated["metadata"]["sumw2_storage_provenance"] == resolved.to_provenance()
    with gzip.open(path, "rb") as stream:
        restored = pickle.load(stream)
    for family in ("njets", FAMILY):
        view = embedded_sumw2_view(restored, family, provenance=resolved.to_provenance())
        component = view["components"]["scalar_nominal"]
        assert sum(cell.sum() for cell in component["values"].values()) == 1
        assert sum(cell.sum() for cell in component["variances"].values()) == (
            13 if family == selected_family else 0)
        assert {record[-1] for record in view["coverage"]} == {
            "selected_nonzero" if family == selected_family else "unavailable"}
    assert not any(name.endswith("_sumw2") for name in restored)


@pytest.mark.parametrize("weights,coeff", [([], None), ([0, 0], None),
                                          ([2, -3], np.zeros((2, 3)))])
def test_selected_zero_is_distinct_from_unavailable(weights, coeff):
    p = processor()
    dataset = "eft" if coeff is not None else "selected"
    output = fill(p, dataset, weights, coeff)
    fill(p, "unselected", weights)
    assert {r[2]: r[-1] for r in output[COVERAGE_KEY]} == {
        dataset: "selected_zero", "unselected": "unavailable"}
    assert totals(output, SAMPLES[dataset]["histAxisName"])[1] == 0
    validate(output)


def test_top_level_family_selection_and_explicit_moment_api(monkeypatch):
    p = processor()
    original = SparseHist.fill_with_moments
    calls = []
    def record(self, **kwargs):
        calls.append(kwargs.copy())
        return original(self, **kwargs)
    monkeypatch.setattr(SparseHist, "fill_with_moments", record)
    fill(p, "eft", [-2], np.array([[3., 1, 2]]))
    np.testing.assert_array_equal(calls[0]["value_weight"], [-2])
    np.testing.assert_array_equal(calls[0]["second_moment"], [36])
    fill(p, "eft", [-2], np.array([[3., 1, 2]]), systematic="JERUp")
    assert calls[1]["second_moment"] == 0
    assert policy().selects("eft", "signal", FAMILY)
    for name in p.accumulator[FAMILY].dense_axes.name:
        assert not policy().selects("eft", "signal", name)
        with pytest.raises(ValueError, match="SUMW2-E"):
            policy(variables=[name])


def test_remapped_sm_only_coefficients_and_masked_inputs():
    p = processor()
    coeff = prepare_eft_coefficients(np.array([[2., 20, 4], [-3., 10, 1], [99., 1, 2]]),
                                     ["ctW"], ["ctG"], "sm_only")
    mask = np.array([True, True, False])
    output = fill(p, "eft", np.array([-2, 3, 100])[mask], coeff[mask])
    assert totals(output, "signal") == (1, 97)
    validate(output)


@pytest.mark.parametrize("dataset", ["selected", "unselected", "eft"])
@pytest.mark.parametrize("systematic", ["nominal", "JERUp"])
def test_regular_awkward_coordinates_preserve_aligned_masks(dataset, systematic):
    p = processor()
    x = ak.Array([25., None, 75., 125.])
    y = ak.Array([.5, 1., None, 1.5])
    weights = ak.Array([-2., 100., 200., 3.])
    coeff = np.array([[3., 1., 2.], [99., 1., 2.], [99., 1., 2.], [-2., 1., 2.]])
    mask = ~ak.is_none(x) & ~ak.is_none(y)
    names = p.accumulator[FAMILY].dense_axes.name
    p._fill_2d_histogram(
        p.accumulator, FAMILY, dataset,
        process=SAMPLES[dataset]["histAxisName"], channel="3l", appl="isSR",
        systematic=systematic, weight=weights[mask],
        eft_coeff=coeff[mask] if dataset == "eft" else None,
        **{names[0]: x[mask], names[1]: y[mask]},
    )
    expected_variance = {"selected": 13, "unselected": 0, "eft": 72}[dataset]
    assert totals(p.accumulator, SAMPLES[dataset]["histAxisName"], systematic) == (
        1, expected_variance if systematic == "nominal" else 0,
    )
    validate(p.accumulator)


@pytest.mark.parametrize("tracked", [False, True])
def test_flow_bins_categories_copy_slice_and_raw_count_restriction(tracked):
    p = processor(tracked=tracked)
    output = fill(p, "selected", [-2, 3, 4], xy=([-1, 25, 300], [-1, .5, 3]))
    assert totals(output) == (5, 29)
    h = output[FAMILY]
    cell = h.view(flow=True)[("shared", "3l", "nominal", "isSR")]
    assert (cell.value[0, 0], cell.variance[0, 0]) == (-2, 4)
    assert (cell.value[-1, -1], cell.variance[-1, -1]) == (4, 16)
    assert h.view(flow=False)[("shared", "3l", "nominal", "isSR")].variance.sum() == 9
    assert not h.track_raw_counts
    with pytest.raises(RuntimeError, match="disabled"):
        h.raw_counts()
    for clone in (h.copy(), copy.deepcopy(h), h[{"process": ["shared"]}]):
        np.testing.assert_array_equal(clone.view(flow=True)[("shared", "3l", "nominal", "isSR")], cell)
    validate(output)


def test_accumulation_and_schema_merge_preserve_independent_moments():
    p, q = processor(), processor()
    first = fill(p, "selected", [0])
    second = fill(q, "selected", [-3])
    fill(q, "unselected", [10])
    for output in (accumulate([first, second]), merge_nominal_mappings(
            [first, second], runtime_families=(FAMILY,), schema_version=3, policy=policy())):
        assert totals(output) == (7, 9)
        validate(output)
        assert {r[2]: r[-1] for r in coverage_manifest(output[COVERAGE_KEY])} == {
            "selected": "selected_nonzero", "unselected": "unavailable"}


@pytest.mark.parametrize("factor", [2, -3, 0])
def test_scaling_preserves_variance_and_coverage(factor):
    p = processor()
    output = fill(p, "eft", [2, -3], np.array([[1.5, 10, 2], [-2., -5, 1]]))
    records = output[COVERAGE_KEY].copy()
    scale_embedded_family(output, FAMILY, factor)
    assert totals(output, "signal") == (-factor, 45 * factor**2)
    if factor:
        assert output[COVERAGE_KEY] == records
    else:
        assert {r[-1] for r in output[COVERAGE_KEY]} == {"selected_zero"}
    validate(output)


@pytest.mark.parametrize("serializer", [pickle, cloudpickle])
def test_serialization_provenance_and_shared_adapter(tmp_path, serializer):
    p = processor()
    output = fill(p, "eft", [2, -3], np.array([[1.5, 10, 2], [-2., -5, 1]]))
    fill(p, "selected", [0])
    fill(p, "unselected", [7])
    restored = serializer.loads(serializer.dumps(output))
    validate(restored)
    path = tmp_path / "embedded_2d.pkl.gz"
    sidecar = write(path, restored)
    validated = validate_histogram_artifact(path)
    assert validated["schema"] == EMBEDDED_LAYOUT
    provenance = validated["metadata"]["sumw2_storage_provenance"]
    assert resolved_policy_from_provenance(provenance) == policy()
    with gzip.open(path, "rb") as stream:
        reopened = pickle.load(stream)
    assert totals(reopened, "signal") == (-1, 45)
    exposed = embedded_sumw2_view(reopened, FAMILY, provenance=provenance, flow=False)
    component = exposed["components"]["scalar_nominal"]
    key = ("signal", "3l", "nominal", "isSR")
    assert component["values"][key].shape == (25, 25)
    assert component["values"][key].sum() == -1
    assert component["variances"][key].sum() == 45
    assert exposed["coverage"] == coverage_manifest(output[COVERAGE_KEY])
    assert exposed["provenance"] == policy().to_provenance()
    assert sidecar["sumw2_content_manifest"]["families"][FAMILY]["sumw2_processes"] == ["shared", "signal"]
    component["variances"][key][...] = 0
    assert totals(reopened, "signal")[1] == 45
    assert not any(name.endswith("_sumw2") for name in reopened)


@pytest.mark.parametrize("kind", ["empty", "only_unselected"])
def test_empty_and_unselected_only_outputs_publish_without_claiming_coverage(tmp_path, kind):
    p = processor()
    output = p.accumulator
    if kind == "only_unselected":
        fill(p, "unselected", [10])
    sidecar = write(tmp_path / "output.pkl.gz", output)
    content = sidecar["sumw2_content_manifest"]["families"][FAMILY]
    assert content["sumw2_processes"] == content["required_sumw2_processes"] == []
    validate_histogram_artifact(tmp_path / "output.pkl.gz")


@pytest.mark.parametrize("corruption", ["missing_coverage", "wrong_selection", "wrong_state", "companion",
                                        "double_storage", "internal_family", "eft_component", "negative_variance"])
def test_schema_v3_rejects_invalid_2d_payload(corruption):
    p = processor()
    output = fill(p, "selected", [2])
    record = next(iter(output[COVERAGE_KEY]))
    if corruption == "missing_coverage":
        output[COVERAGE_KEY].clear()
    elif corruption in {"wrong_selection", "wrong_state"}:
        state = "unavailable" if corruption == "wrong_selection" else "selected_zero"
        output[COVERAGE_KEY] = {record[:-1] + (state,)}
    elif corruption == "companion":
        output[FAMILY + "_sumw2"] = output[FAMILY].copy()
    elif corruption == "double_storage":
        output[FAMILY] = SparseHist(*output[FAMILY].axes, storage="Double")
    elif corruption == "internal_family":
        output[COVERAGE_KEY] = {(output[FAMILY].dense_axes.name[0],) + record[1:]}
    elif corruption == "eft_component":
        output[COVERAGE_KEY] = {(record[0], "eft_nominal") + record[2:]}
    else:
        next(iter(output[FAMILY].view(flow=True).values())).variance[...] = -1
    with pytest.raises((ValueError, TypeError)):
        validate(output)


def historical_payload():
    h = processor().accumulator[FAMILY]
    scalar = SparseHist(*h.axes, storage="Double")
    companion = SparseHist(*h.categorical_axes, *(hist.axis.Regular(
        len(a), a.edges[0], a.edges[-1], name=a.name + "_sumw2") for a in h.dense_axes), storage="Double")
    cats = dict(process="shared", channel="3l", systematic="nominal", appl="isSR")
    coords = dict(zip(h.dense_axes.name, ([25.], [.5])))
    scalar.fill(**cats, **coords, weight=[-2])
    companion.fill(**cats, **{k + "_sumw2": v for k, v in coords.items()}, weight=[4])
    return {FAMILY: scalar, FAMILY + "_sumw2": companion}


def test_historical_split_sibling_2d_artifact_and_legacy_reading(tmp_path):
    old = historical_payload()
    path = tmp_path / "split_2d.pkl.gz"
    sidecar = write(path, old)
    assert sidecar["artifact"]["nominal_container_schema_version"] == 2
    assert validate_histogram_artifact(path)["schema"] == "split_sibling_v1"
    with gzip.open(path, "rb") as stream:
        restored = pickle.load(stream)
    assert sum(a.sum() for a in restored[FAMILY].view(flow=True).values()) == -2
    assert sum(a.sum() for a in restored[FAMILY + "_sumw2"].view(flow=True).values()) == 4
    del old[FAMILY + "_sumw2"]
    with pytest.raises((ValueError, histogram_artifact_error), match="companion|sumw2"):
        write(tmp_path / "broken.pkl.gz", old)
    path = tmp_path / "legacy.pkl.gz"
    with gzip.open(path, "wb") as stream:
        pickle.dump(old, stream)
    assert validate_histogram_artifact(path)["schema"] == "legacy_uniform"


def test_incompatible_schema_layout_and_policy_merges_fail(tmp_path):
    p = processor()
    output = fill(p, "selected", [-2])
    sidecar = write(tmp_path / "new.pkl.gz", output)
    historical = write(tmp_path / "old.pkl.gz", historical_payload())
    with pytest.raises(histogram_artifact_error, match="schemas/layouts"):
        merge_histogram_sidecars([sidecar, historical])
    disabled = policy(mode="disabled")
    other = processor(resolved=disabled)
    alternate = write(tmp_path / "disabled.pkl.gz", fill(other, "selected", [-2]), disabled)
    with pytest.raises(histogram_artifact_error):
        merge_histogram_sidecars([sidecar, alternate])
    broken = copy.deepcopy(output)
    old_axis = broken[FAMILY].dense_axes[0]
    broken[FAMILY] = SparseHist(*broken[FAMILY].categorical_axes,
        hist.axis.Regular(3, 0, 250, name=old_axis.name), broken[FAMILY].dense_axes[1], storage="Weight")
    broken[COVERAGE_KEY].clear()
    with pytest.raises(ValueError, match="Dense axes"):
        merge_nominal_mappings([output, broken], runtime_families=(FAMILY,), schema_version=3, policy=policy())


def test_compatible_artifact_merge_preserves_coverage_and_provenance(tmp_path):
    p, q = processor(), processor()
    first = fill(p, "selected", [0])
    second = fill(q, "unselected", [10])
    sidecars = [write(tmp_path / f"part{i}.pkl.gz", output)
                for i, output in enumerate((first, second))]
    merged = merge_nominal_mappings([first, second], runtime_families=(FAMILY,),
                                    schema_version=3, policy=policy())
    context = merge_histogram_sidecars(sidecars)
    path = tmp_path / "merged.pkl.gz"
    write_histogram_artifact(path, histograms=merged, **context)
    validated = validate_histogram_artifact(path)
    assert validated["schema"] == EMBEDDED_LAYOUT
    assert totals(merged) == (10, 0)
    assert validated["metadata"]["sumw2_content_manifest"]["embedded_coverage"] == coverage_manifest(merged[COVERAGE_KEY])


@pytest.mark.parametrize("selected", [False, True])
def test_flattened_jet_diagnostic_uses_aligned_eft_constants(selected):
    family = "jet_eta_phi_before_veto"
    resolved = resolve_sumw2_storage_policy(
        {"mode": "full_diagnostics" if selected else "disabled"}, samples=SAMPLES,
        runtime_families=(family,), axes_info=info, axes_info_2d=info_2d,
        sumw2_storage_present=True,
    )
    p = AnalysisProcessor(samples=SAMPLES, wc_names_lst=["ctG"], hist_lst=[family],
                          sumw2_policy=resolved)
    jets = ak.Array([[{"eta": .1, "phi": .2}, {"eta": .3, "phi": .4}],
                     [{"eta": -.2, "phi": -.3}], [{"eta": .2, "phi": .3}]])
    mask = np.array([True, True, False])
    coeff = np.array([[3., 1, 2], [-2., 1, 2], [99., 1, 2]])
    eta, phi, weights = flatten_jagged_jet_eta_phi_weights(jets, mask, np.array([-2., 3., 100.]))
    _, _, constants = flatten_jagged_jet_eta_phi_weights(jets, mask, evaluate_eft_coefficients_at_sm(coeff))
    names = p.accumulator[family].dense_axes.name
    p._fill_2d_histogram(p.accumulator, family, "eft", process="signal", channel="2lOS",
        appl="isSR", systematic="nominal", weight=weights,
        eft_coeff=np.asarray(constants)[:, None], **{names[0]: eta, names[1]: phi})
    cell = next(iter(p.accumulator[family].view(flow=True).values()))
    assert cell.value.sum() == -1
    assert cell.variance.sum() == (108 if selected else 0)
    validate_nominal_mapping(p.accumulator, runtime_families=(family,), schema_version=3, policy=resolved)


def test_artifact_rejects_sidecar_coverage_and_policy_identity_tampering(tmp_path):
    import json
    from topeft.modules.histogram_artifact import metadata_sidecar_path
    p = processor()
    output = fill(p, "selected", [2])
    path = tmp_path / "tampered.pkl.gz"
    sidecar = write(path, output)
    broken = copy.deepcopy(sidecar)
    broken["sumw2_content_manifest"]["embedded_coverage"][0][1] = "eft_nominal"
    metadata_sidecar_path(path).write_text(json.dumps(broken))
    with pytest.raises(histogram_artifact_error, match="coverage"):
        validate_histogram_artifact(path)
    broken = copy.deepcopy(sidecar)
    broken["sumw2_storage_provenance"]["resolved_targets"] = []
    metadata_sidecar_path(path).write_text(json.dumps(broken))
    with pytest.raises((histogram_artifact_error, ValueError)):
        validate_histogram_artifact(path)
