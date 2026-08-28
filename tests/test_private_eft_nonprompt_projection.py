from __future__ import annotations

import copy
import json

import hist
import numpy as np
import pytest

from analysis.topeft_run2 import run_data_driven
from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from topcoffea.modules.utils import get_hist_from_pkl
from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.dataDrivenEstimation import DataDrivenProducer
from topeft.modules.data_driven_products import (
    certify_data_driven_preflight,
    resolve_data_driven_products,
)
from topeft.modules.histogram_artifact import (
    histogram_artifact_error,
    lineage_input_from_sidecar,
    metadata_sidecar_path,
    read_histogram_sidecar,
    validate_histogram_artifact,
    write_histogram_artifact,
)
from topeft.modules.nominal_schema import eft_nominal_key, scalar_nominal_key
from topeft.modules.nonprompt_policy import certify_active_nonprompt_policy
from topeft.modules.production_sample_profile import (
    build_active_sample_universe,
    certify_production_sample_contract,
)
from topeft.modules.sumw2_policy import (
    resolve_sumw2_storage_mode,
    resolve_sumw2_storage_policy,
)


PRIVATE_PROCESS = "tllq_privateUL18"
CENTRAL_EQUIVALENT_PROCESS = "tZq_centralUL18"
UNSELECTED_EFT_PROCESS = "WWTo2L2Nu_centralUL18"
TTH_PRIVATE_PROCESS = "ttH_private2022"


def _axes(dense_name, *, bins=1):
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.Regular(bins, 0.0, 1.0, name=dense_name),
    )


def _fill_sparse(dense_name, entries):
    output = SparseHist(*_axes(dense_name), storage="Double")
    for process, appl, value in entries:
        output.fill(
            process=process,
            channel="3l",
            systematic="nominal",
            appl=appl,
            **{dense_name: np.asarray([0.5])},
            weight=np.asarray([value]),
        )
    return output


def _fill_eft(entries, *, bins=1):
    output = HistEFT(
        *_axes("njets", bins=bins),
        wc_names=["ctW"],
        label="Events",
    )
    for process, appl, weight, coefficients in entries:
        output.fill(
            process=process,
            channel="3l",
            systematic="nominal",
            appl=appl,
            njets=np.asarray([0.5]),
            weight=np.asarray([weight]),
            eft_coeff=np.asarray([coefficients]),
        )
    return output


def _total(histogram, process, wc_values=None):
    selected = histogram.integrate("process", process)
    values = (
        selected.eval(wc_values or {})
        if isinstance(selected, HistEFT)
        else selected.view(flow=True, as_dict=True)
    )
    return sum(float(np.asarray(value).sum()) for value in values.values())


def _samples(prompt_process, *, private):
    is_run3 = prompt_process.endswith(("2022", "2022EE", "2023", "2023BPix"))
    data_process = "data2022" if is_run3 else "dataUL18"
    baseline_prompt = (
        "TTto2L2Nu_central2022" if is_run3 else "TTTo2L2Nu_centralUL18"
    )
    samples = {
        "data_dataset": {
            "histAxisName": data_process,
            "isData": True,
            "WCnames": [],
        },
        "prompt_dataset": {
            "histAxisName": baseline_prompt,
            "isData": False,
            "WCnames": [],
        },
        "projection_dataset": {
            "histAxisName": prompt_process,
            "isData": False,
            "WCnames": ["ctW"] if private else [],
        },
    }
    return samples


def _contracts(prompt_process, *, private):
    samples = _samples(prompt_process, private=private)
    data_process = samples["data_dataset"]["histAxisName"]
    baseline_prompt = samples["prompt_dataset"]["histAxisName"]
    mode = "production" if private else "full_diagnostics"
    storage = {"mode": mode}
    if private:
        storage["rules"] = [
            {
                "process_names": [
                    data_process,
                    baseline_prompt,
                    prompt_process,
                ],
                "variables": ["njets"],
            }
        ]
    mode_resolution = resolve_sumw2_storage_mode(
        storage,
        sumw2_storage_present=True,
    )
    universe = build_active_sample_universe(samples, wrapper_identity="pytest")
    products = resolve_data_driven_products(
        {
            "nonprompt": {
                "enabled": True,
                "source_contributors": {
                    "data": {"process_names": [data_process]},
                    "prompt_mc": {
                        "process_names": [
                            baseline_prompt,
                            prompt_process,
                        ]
                    },
                },
            },
            "flips": {"enabled": False},
        },
        data_driven_products_present=True,
        legacy_do_np=False,
        samples=samples,
        runtime_families=("njets",),
        metadata_path="private_eft_projection.yml",
    )
    policy = resolve_sumw2_storage_policy(
        storage,
        samples=samples,
        runtime_families=("njets",),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
        mode_resolution=mode_resolution,
    )
    requested, resolved = certify_data_driven_preflight(products, policy)
    profile = certify_production_sample_contract(universe, policy, products)
    return policy, requested, resolved, profile


def _payload(*, private=True):
    prompt_process = PRIVATE_PROCESS if private else CENTRAL_EQUIVALENT_PROCESS
    scalar_entries = [
        ("dataUL18", "isAR_3l", 10.0),
        ("TTTo2L2Nu_centralUL18", "isAR_3l", 3.0),
    ]
    if not private:
        scalar_entries.extend(
            [
                (prompt_process, "isAR_3l", 6.0),
                (prompt_process, "isAR_3l", -2.0),
            ]
        )
    eft_entries = [
        (UNSELECTED_EFT_PROCESS, "isAR_3l", 2.0, [2.0, -1.0, 0.5]),
        (UNSELECTED_EFT_PROCESS, "isAR_2lSS_OS", 11.0, [2.0, -1.0, 0.5]),
        (UNSELECTED_EFT_PROCESS, "isSR_3l", 3.0, [2.0, -1.0, 0.5]),
    ]
    if private:
        eft_entries.extend(
            [
                (prompt_process, "isAR_3l", 4.0, [1.5, 2.0, 3.0]),
                (prompt_process, "isAR_3l", -4.0 / 3.0, [1.5, -0.5, 1.0]),
                (prompt_process, "isAR_2lSS_OS", 7.0, [1.5, 2.0, 3.0]),
                (prompt_process, "isSR_3l", 5.0, [1.5, 2.0, 3.0]),
            ]
        )
    companion_entries = [
        ("dataUL18", "isAR_3l", 100.0),
        ("TTTo2L2Nu_centralUL18", "isAR_3l", 9.0),
        (prompt_process, "isAR_3l", 36.0),
        (prompt_process, "isAR_3l", 4.0),
    ]
    if private:
        companion_entries.append((prompt_process, "isSR_3l", 56.25))
    return {
        scalar_nominal_key("njets"): _fill_sparse("njets", scalar_entries),
        eft_nominal_key("njets"): _fill_eft(eft_entries),
        "njets_sumw2": _fill_sparse("njets_sumw2", companion_entries),
    }


def _representation_payload(prompt_process, representation):
    samples = _samples(prompt_process, private=representation == "eft")
    data_process = samples["data_dataset"]["histAxisName"]
    baseline_prompt = samples["prompt_dataset"]["histAxisName"]
    excluded_process = (
        "WWTo2L2Nu_central2022"
        if prompt_process.endswith("2022")
        else UNSELECTED_EFT_PROCESS
    )
    scalar_entries = [
        (data_process, "isAR_3l", 10.0),
        (baseline_prompt, "isAR_3l", 3.0),
    ]
    eft_entries = [
        (excluded_process, "isAR_3l", 2.0, [1.0, 0.0, 0.0]),
    ]
    if representation == "scalar":
        scalar_entries.append((prompt_process, "isAR_3l", 4.0))
    else:
        eft_entries.append(
            (prompt_process, "isAR_3l", 4.0, [1.0, 0.0, 0.0])
        )
        eft_entries.append(
            (prompt_process, "isSR_3l", 1.0, [1.0, 0.0, 0.0])
        )
    companion_entries = [
        (data_process, "isAR_3l", 100.0),
        (baseline_prompt, "isAR_3l", 9.0),
        (prompt_process, "isAR_3l", 16.0),
    ]
    if representation == "eft":
        companion_entries.append((prompt_process, "isSR_3l", 1.0))
    return {
        scalar_nominal_key("njets"): _fill_sparse("njets", scalar_entries),
        eft_nominal_key("njets"): _fill_eft(eft_entries),
        "njets_sumw2": _fill_sparse("njets_sumw2", companion_entries),
    }


def _write_processor(path, *, private=True, payload=None):
    prompt_process = PRIVATE_PROCESS if private else CENTRAL_EQUIVALENT_PROCESS
    policy, requested, resolved, profile = _contracts(
        prompt_process,
        private=private,
    )
    sidecar = write_histogram_artifact(
        path,
        histograms=payload or _payload(private=private),
        artifact_kind="processor_output",
        sumw2_storage_provenance=policy.to_provenance(),
        production_sample_contract=profile,
        requested_data_driven_products=requested,
        resolved_data_driven_contract=resolved,
    )
    return policy, sidecar


def _transform(tmp_path, *, private=True):
    source_path = tmp_path / ("private.pkl.gz" if private else "central.pkl.gz")
    output_path = tmp_path / (
        "private_nonprompt.pkl.gz" if private else "central_nonprompt.pkl.gz"
    )
    policy, source_sidecar = _write_processor(source_path, private=private)
    producer = DataDrivenProducer(
        str(source_path),
        "",
        artifact_kind="nonprompt_output",
    )
    transformed = producer.getDataDrivenHistogram()
    context = producer.get_transformation_context("nonprompt_output")
    output_sidecar = write_histogram_artifact(
        output_path,
        histograms=transformed,
        artifact_kind="nonprompt_output",
        sumw2_storage_provenance=policy.to_provenance(),
        lineage_inputs=[lineage_input_from_sidecar(source_sidecar)],
        input_sidecar=producer.get_effective_input_sidecar(),
        transformation_context=context,
    )
    return {
        "source_path": source_path,
        "output_path": output_path,
        "policy": policy,
        "source_sidecar": source_sidecar,
        "producer": producer,
        "context": context,
        "histograms": transformed,
        "sidecar": output_sidecar,
    }


def _write_representation_processor(source_path, prompt_process, representation):
    policy, requested, resolved, profile = _contracts(
        prompt_process,
        private=representation == "eft",
    )
    source_sidecar = write_histogram_artifact(
        source_path,
        histograms=_representation_payload(prompt_process, representation),
        artifact_kind="processor_output",
        sumw2_storage_provenance=policy.to_provenance(),
        production_sample_contract=profile,
        requested_data_driven_products=requested,
        resolved_data_driven_contract=resolved,
    )
    return policy, source_sidecar


def _transform_representation(tmp_path, prompt_process, representation):
    source_path = tmp_path / f"{representation}_source.pkl.gz"
    output_path = tmp_path / f"{representation}_nonprompt.pkl.gz"
    policy, source_sidecar = _write_representation_processor(
        source_path,
        prompt_process,
        representation,
    )
    producer = DataDrivenProducer(
        str(source_path),
        "",
        artifact_kind="nonprompt_output",
    )
    transformed = producer.getDataDrivenHistogram()
    effective_sidecar = producer.get_effective_input_sidecar()
    output_sidecar = write_histogram_artifact(
        output_path,
        histograms=transformed,
        artifact_kind="nonprompt_output",
        sumw2_storage_provenance=policy.to_provenance(),
        lineage_inputs=[lineage_input_from_sidecar(source_sidecar)],
        input_sidecar=effective_sidecar,
        transformation_context=producer.get_transformation_context(
            "nonprompt_output"
        ),
    )
    return {
        "source_sidecar": source_sidecar,
        "effective_sidecar": effective_sidecar,
        "histograms": transformed,
        "sidecar": output_sidecar,
        "execution": producer.get_prompt_subtraction_execution_evidence(),
    }


def test_private_eft_projection_is_sm_only_and_preserves_passthrough(tmp_path):
    result = _transform(tmp_path, private=True)
    output = result["histograms"]
    scalar = output[scalar_nominal_key("njets")]
    companion = output["njets_sumw2"]
    eft = output[eft_nominal_key("njets")]

    assert _total(scalar, "nonpromptUL18") == pytest.approx(3.0)
    assert _total(companion, "nonpromptUL18") == pytest.approx(149.0)
    assert _total(companion, "nonpromptUL18") != pytest.approx(
        100.0 + 9.0 + 4.0**2
    )
    assert _total(eft, PRIVATE_PROCESS, {}) == pytest.approx(7.5)
    assert _total(eft, PRIVATE_PROCESS, {"ctW": 1.0}) == pytest.approx(32.5)
    assert _total(eft, UNSELECTED_EFT_PROCESS, {}) == pytest.approx(6.0)
    assert _total(
        eft, UNSELECTED_EFT_PROCESS, {"ctW": 1.0}
    ) == pytest.approx(4.5)
    assert "nonpromptUL18" not in [str(value) for value in eft.axes["process"]]
    assert "appl" not in [axis.name for axis in eft.axes]
    assert "quadratic_term" not in [axis.name for axis in companion.axes]

    for wc_values in ({"ctW": 1.0}, {"ctW": -2.0}):
        assert _total(scalar, "nonpromptUL18", wc_values) == pytest.approx(3.0)

    projection = result["sidecar"]["transformation_contract"][
        "eft_prompt_projection"
    ]
    assert projection == {
        "mode": "sm_point",
        "required_processes": [PRIVATE_PROCESS],
        "generated_nonprompt_eft_dependence": False,
    }
    assert read_histogram_sidecar(result["output_path"]) == result["sidecar"]
    assert validate_histogram_artifact(result["output_path"])["metadata"] == result[
        "sidecar"
    ]
    execution = result["producer"].get_prompt_subtraction_execution_evidence()[
        "families"
    ]["njets"]
    assert execution["executed_processes"] == execution[
        "selected_present_processes"
    ]
    assert execution["nominal_evaluation_route"][PRIVATE_PROCESS] == "eft_sm_point"
    assert (
        execution["nominal_evaluation_route"]["TTTo2L2Nu_centralUL18"]
        == "scalar_nominal"
    )


@pytest.mark.parametrize(
    "prompt_process",
    [TTH_PRIVATE_PROCESS, "WZTo3LNu_central2022"],
)
@pytest.mark.parametrize("representation", ["scalar", "eft"])
def test_prompt_membership_and_execution_are_representation_independent(
    tmp_path,
    prompt_process,
    representation,
):
    result = _transform_representation(
        tmp_path,
        prompt_process,
        representation,
    )
    source_contract = result["source_sidecar"]["resolved_data_driven_contract"]
    execution = result["execution"]["families"]["njets"]

    assert prompt_process in source_contract["resolved_prompt_process_set"]
    assert prompt_process in execution["selected_processes"]
    assert prompt_process in execution["executed_processes"]
    assert execution["nominal_evaluation_route"][prompt_process] == (
        "eft_sm_point" if representation == "eft" else "scalar_nominal"
    )
    assert execution["executed_processes"] == execution[
        "selected_present_processes"
    ]
    excluded_process = "WWTo2L2Nu_central2022"
    assert excluded_process in execution["excluded_processes"]
    assert excluded_process not in execution["executed_processes"]
    assert _total(
        result["histograms"][scalar_nominal_key("njets")],
        "nonprompt2022",
    ) == pytest.approx(3.0)

    if representation == "eft":
        assert "required_prompt_signal_processes" not in source_contract
        assert "required_prompt_signal_processes" not in result[
            "effective_sidecar"
        ]["resolved_data_driven_contract"]
        assert result["sidecar"]["transformation_contract"][
            "eft_prompt_projection"
        ]["required_processes"] == [prompt_process]


def test_selected_absent_is_distinct_from_unhandled_and_ambiguity_fails_closed():
    certificate = certify_active_nonprompt_policy(
        [TTH_PRIVATE_PROCESS],
        configuration_source="focused_execution_contract",
    )
    selected = certificate.resolved_prompt_process_set
    excluded = certificate.explicit_exclusions
    absent = DataDrivenProducer._build_prompt_subtraction_execution_plan(
        selected,
        excluded,
        {"njets": {"scalar": (), "eft": (), "sumw2": ()}},
    )["njets"]
    assert absent["selected_absent_processes"] == {TTH_PRIVATE_PROCESS}
    assert absent["unhandled_processes"] == set()

    with pytest.raises(RuntimeError, match="without a supported scalar or EFT"):
        DataDrivenProducer._build_prompt_subtraction_execution_plan(
            selected,
            excluded,
            {
                "njets": {
                    "scalar": (),
                    "eft": (),
                    "sumw2": (TTH_PRIVATE_PROCESS,),
                }
            },
        )
    with pytest.raises(RuntimeError, match="ambiguous scalar and EFT"):
        DataDrivenProducer._build_prompt_subtraction_execution_plan(
            selected,
            excluded,
            {
                "njets": {
                    "scalar": (TTH_PRIVATE_PROCESS,),
                    "eft": (TTH_PRIVATE_PROCESS,),
                    "sumw2": (TTH_PRIVATE_PROCESS,),
                }
            },
        )


def test_tth_private_streaming_path_uses_canonical_selection(tmp_path):
    source_path = tmp_path / "tth_private_source.pkl.gz"
    output_path = tmp_path / "tth_private_nonprompt.pkl.gz"
    _policy, source_sidecar = _write_representation_processor(
        source_path,
        TTH_PRIVATE_PROCESS,
        "eft",
    )
    source_contract = source_sidecar["resolved_data_driven_contract"]
    assert TTH_PRIVATE_PROCESS in source_contract["resolved_prompt_process_set"]
    assert "required_prompt_signal_processes" not in source_contract

    assert run_data_driven.main(
        [
            "--input-pkl",
            str(source_path),
            "--output-pkl",
            str(output_path),
            "--quiet",
        ]
    ) == 0
    output = get_hist_from_pkl(str(output_path))
    assert _total(
        output[scalar_nominal_key("njets")],
        "nonprompt2022",
    ) == pytest.approx(3.0)
    sidecar = read_histogram_sidecar(output_path)
    assert sidecar["transformation_contract"]["eft_prompt_projection"][
        "required_processes"
    ] == [TTH_PRIVATE_PROCESS]


def test_ar_only_eft_input_produces_empty_no_appl_sibling():
    eft_key = eft_nominal_key("njets")
    source = _fill_eft(
        [
            (PRIVATE_PROCESS, "isAR_3l", 4.0, [1.5, 2.0, 3.0]),
            (
                UNSELECTED_EFT_PROCESS,
                "isAR_2lSS_OS",
                11.0,
                [2.0, -1.0, 0.5],
            ),
        ]
    )

    output = DataDrivenProducer({eft_key: source}, "").getDataDrivenHistogram()[
        eft_key
    ]

    assert isinstance(output, HistEFT)
    assert output.empty()
    assert "appl" not in [axis.name for axis in output.axes]


def test_private_eft_and_equivalent_central_scalar_agree_at_sm(tmp_path):
    private = _transform(tmp_path, private=True)["histograms"]
    central_result = _transform(tmp_path, private=False)
    central = central_result["histograms"]
    assert _total(
        private[scalar_nominal_key("njets")], "nonpromptUL18"
    ) == pytest.approx(
        _total(central[scalar_nominal_key("njets")], "nonpromptUL18")
    )
    assert _total(private["njets_sumw2"], "nonpromptUL18") == pytest.approx(
        _total(central["njets_sumw2"], "nonpromptUL18")
    )
    assert central_result["sidecar"]["transformation_contract"][
        "eft_prompt_projection"
    ] == {
        "mode": "sm_point",
        "required_processes": [],
        "generated_nonprompt_eft_dependence": False,
    }
    assert "nonpromptUL18" not in [
        str(value)
        for value in central[eft_nominal_key("njets")].axes["process"]
    ]


def test_private_eft_projection_streaming_roundtrip(tmp_path):
    source_path = tmp_path / "streaming_private.pkl.gz"
    output_path = tmp_path / "streaming_nonprompt.pkl.gz"
    _write_processor(source_path, private=True)
    assert run_data_driven.main(
        [
            "--input-pkl",
            str(source_path),
            "--output-pkl",
            str(output_path),
            "--quiet",
        ]
    ) == 0
    output = get_hist_from_pkl(str(output_path))
    assert _total(
        output[scalar_nominal_key("njets")], "nonpromptUL18"
    ) == pytest.approx(3.0)
    assert _total(output["njets_sumw2"], "nonpromptUL18") == pytest.approx(
        149.0
    )
    projection = read_histogram_sidecar(output_path)["transformation_contract"][
        "eft_prompt_projection"
    ]
    assert projection["required_processes"] == [PRIVATE_PROCESS]
    assert projection["generated_nonprompt_eft_dependence"] is False


def test_private_eft_nominal_reference_uses_sm_prompt_subtraction_without_sumw2(
    tmp_path,
):
    source_path = tmp_path / "reference_private.pkl.gz"
    output_path = tmp_path / "reference_private_output.pkl.gz"
    policy, source_sidecar = _write_processor(source_path, private=True)
    producer = DataDrivenProducer(
        str(source_path),
        "",
        artifact_kind="nonprompt_nominal_reference_output",
    )
    output = producer.getDataDrivenHistogram()

    # The independent scalar closure is data (10) minus TT (3) minus the
    # private EFT prompt at the SM point (4), not a WC-dependent subtraction.
    assert _total(output[scalar_nominal_key("njets")], "nonpromptUL18") == pytest.approx(3.0)
    assert "nonpromptUL18" not in [
        str(value) for value in output["njets_sumw2"].axes["process"]
    ]
    sidecar = write_histogram_artifact(
        output_path,
        histograms=output,
        artifact_kind="nonprompt_nominal_reference_output",
        sumw2_storage_provenance=policy.to_provenance(),
        lineage_inputs=[lineage_input_from_sidecar(source_sidecar)],
        input_sidecar=producer._resolved_input_sidecar,
        transformation_context=producer.get_transformation_context(
            "nonprompt_nominal_reference_output"
        ),
    )
    assert sidecar["nominal_reference_contract"]["statistically_complete"] is False
    assert sidecar["nominal_reference_contract"]["card_ready"] is False
    assert validate_histogram_artifact(output_path)["metadata"] == sidecar


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            "missing_eft_source",
            "missing_source_nominal=.*tllq_privateUL18",
        ),
        ("duplicate_source", "duplicate process labels.*tllq_privateUL18"),
        (
            "missing_companion",
            "missing_required_companions=.*tllq_privateUL18",
        ),
        ("incompatible_axes", "different dense axes"),
    ],
)
def test_private_eft_input_errors_are_actionable(tmp_path, mutation, match):
    payload = _payload(private=True)
    if mutation == "missing_eft_source":
        payload[eft_nominal_key("njets")] = payload[eft_nominal_key("njets")].remove(
            "process", [PRIVATE_PROCESS]
        )
    elif mutation == "duplicate_source":
        payload[scalar_nominal_key("njets")].fill(
            process=PRIVATE_PROCESS,
            channel="3l",
            systematic="nominal",
            appl="isAR_3l",
            njets=np.asarray([0.5]),
            weight=np.asarray([4.0]),
        )
    elif mutation == "missing_companion":
        payload["njets_sumw2"] = payload["njets_sumw2"].remove(
            "process", [PRIVATE_PROCESS]
        )
    else:
        payload[eft_nominal_key("njets")] = _fill_eft(
            [(PRIVATE_PROCESS, "isAR_3l", 1.0, [1.5, 2.0, 3.0])],
            bins=2,
        )
    with pytest.raises((histogram_artifact_error, ValueError), match=match):
        _write_processor(tmp_path / f"{mutation}.pkl.gz", payload=payload)


def test_generated_eft_nonprompt_and_tampered_projection_are_rejected(tmp_path):
    result = _transform(tmp_path, private=True)
    unexpected = copy.deepcopy(result["histograms"])
    unexpected[eft_nominal_key("njets")].fill(
        process="nonpromptUL18",
        channel="3l",
        systematic="nominal",
        njets=np.asarray([0.5]),
        weight=np.asarray([1.0]),
        eft_coeff=np.asarray([[1.0, 0.0, 0.0]]),
    )
    with pytest.raises(
        histogram_artifact_error,
        match="unexpected generated nonprompt EFT component|EFT nominal roles differ",
    ):
        write_histogram_artifact(
            tmp_path / "unexpected_eft_nonprompt.pkl.gz",
            histograms=unexpected,
            artifact_kind="nonprompt_output",
            sumw2_storage_provenance=result["policy"].to_provenance(),
            lineage_inputs=[lineage_input_from_sidecar(result["source_sidecar"])],
            input_sidecar=result["source_sidecar"],
            transformation_context=result["context"],
        )

    sidecar_path = metadata_sidecar_path(result["output_path"])
    tampered = json.loads(sidecar_path.read_text(encoding="utf-8"))
    tampered["transformation_contract"]["eft_prompt_projection"][
        "required_processes"
    ] = []
    sidecar_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        histogram_artifact_error,
        match="projection provenance is tampered or inconsistent",
    ):
        validate_histogram_artifact(result["output_path"])
