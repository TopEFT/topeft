from __future__ import annotations

import hist
import numpy as np
import pytest

from topcoffea.modules.sparseHist import SparseHist
from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.dataDrivenEstimation import DataDrivenProducer
from topeft.modules.data_driven_products import (
    certify_data_driven_preflight,
    resolve_data_driven_products,
)
from topeft.modules.histogram_artifact import (
    histogram_artifact_error,
    metadata_sidecar_path,
    write_histogram_artifact,
)
from topeft.modules.nominal_schema import scalar_nominal_key, sumw2_key
from topeft.modules.production_sample_profile import (
    build_active_sample_universe,
    certify_production_sample_contract,
)
from topeft.modules.sumw2_policy import resolve_sumw2_storage_policy


def _axes(dense_name):
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name=dense_name),
    )


def _histogram(dense_name, entries):
    output = SparseHist(*_axes(dense_name), storage="Double")
    for process, weight in entries:
        output.fill(
            process=process,
            channel="3l",
            systematic="nominal",
            appl="isAR_3l",
            **{dense_name: np.asarray([0.5])},
            weight=np.asarray([weight]),
        )
    return output


def _total(histogram, process):
    selected = histogram.integrate("process", process)
    return sum(
        float(np.asarray(value).sum())
        for value in selected.view(flow=True, as_dict=True).values()
    )


def _contracts(*, run_era):
    if run_era == "run2":
        data_process = "dataUL18"
        prompt_processes = (
            "TTTo2L2Nu_centralUL18",
            "WZTo3LNu_centralUL18",
        )
    else:
        data_process = "data2022"
        prompt_processes = (
            "TTto2L2Nu_central2022",
            "WZTo3LNu_central2022",
        )
    samples = {
        "data_dataset": {
            "histAxisName": data_process,
            "isData": True,
            "WCnames": [],
        },
        **{
            f"prompt_dataset_{index}": {
                "histAxisName": process,
                "isData": False,
                "WCnames": [],
            }
            for index, process in enumerate(prompt_processes)
        },
    }
    runtime_families = ("njets", "lt")
    policy = resolve_sumw2_storage_policy(
        {"mode": "full_diagnostics"},
        samples=samples,
        runtime_families=runtime_families,
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    products = resolve_data_driven_products(
        {
            "nonprompt": {
                "enabled": True,
                "source_contributors": {
                    "data": {"process_names": [data_process]},
                    "prompt_mc": {"process_names": list(prompt_processes)},
                },
            },
            "flips": {"enabled": False},
        },
        data_driven_products_present=True,
        legacy_do_np=False,
        samples=samples,
        runtime_families=runtime_families,
        metadata_path="family_applicability.yml",
    )
    requested, resolved = certify_data_driven_preflight(products, policy)
    profile = certify_production_sample_contract(
        build_active_sample_universe(samples, wrapper_identity="pytest"),
        policy,
        products,
    )
    return data_process, prompt_processes, policy, requested, resolved, profile


def _payload(data_process, prompt_processes, *, omit_present_companion=False):
    prompt_a, prompt_b = prompt_processes
    lt_companions = [(data_process, 49.0)]
    if not omit_present_companion:
        lt_companions.append((prompt_a, 1.0))
    return {
        scalar_nominal_key("njets"): _histogram(
            "njets",
            [(data_process, 10.0), (prompt_a, 3.0), (prompt_b, 2.0)],
        ),
        sumw2_key("njets"): _histogram(
            sumw2_key("njets"),
            [(data_process, 100.0), (prompt_a, 9.0), (prompt_b, 4.0)],
        ),
        scalar_nominal_key("lt"): _histogram(
            "lt", [(data_process, 7.0), (prompt_a, 1.0)]
        ),
        sumw2_key("lt"): _histogram(sumw2_key("lt"), lt_companions),
    }


@pytest.mark.parametrize("run_era", ["run2", "run3"])
def test_family_absent_prompt_process_is_valid_and_consumable(tmp_path, run_era):
    data_process, prompt_processes, policy, requested, resolved, profile = (
        _contracts(run_era=run_era)
    )
    source_path = tmp_path / f"{run_era}_processor.pkl.gz"
    write_histogram_artifact(
        source_path,
        histograms=_payload(data_process, prompt_processes),
        artifact_kind="processor_output",
        sumw2_storage_provenance=policy.to_provenance(),
        production_sample_contract=profile,
        requested_data_driven_products=requested,
        resolved_data_driven_contract=resolved,
    )

    producer = DataDrivenProducer(
        str(source_path), "", artifact_kind="nonprompt_output"
    )
    output = producer.getDataDrivenHistogram()
    execution = producer.get_prompt_subtraction_execution_evidence()["families"]
    prompt_a, prompt_b = prompt_processes
    nonprompt_process = "nonpromptUL18" if run_era == "run2" else "nonprompt2022"

    assert execution["njets"]["executed_processes"] == sorted(prompt_processes)
    assert prompt_b in execution["lt"]["selected_absent_processes"]
    assert execution["lt"]["executed_processes"] == [prompt_a]
    assert _total(output[scalar_nominal_key("njets")], nonprompt_process) == pytest.approx(5.0)
    assert _total(output[sumw2_key("njets")], nonprompt_process) == pytest.approx(113.0)
    assert _total(output[scalar_nominal_key("lt")], nonprompt_process) == pytest.approx(6.0)
    assert _total(output[sumw2_key("lt")], nonprompt_process) == pytest.approx(50.0)


def test_family_present_prompt_process_without_sumw2_remains_fail_closed(tmp_path):
    data_process, prompt_processes, policy, requested, resolved, profile = (
        _contracts(run_era="run2")
    )
    source_path = tmp_path / "missing_present_companion.pkl.gz"

    with pytest.raises(
        histogram_artifact_error,
        match="Manifest requires sumw2 processes absent from artifact content",
    ):
        write_histogram_artifact(
            source_path,
            histograms=_payload(
                data_process,
                prompt_processes,
                omit_present_companion=True,
            ),
            artifact_kind="processor_output",
            sumw2_storage_provenance=policy.to_provenance(),
            production_sample_contract=profile,
            requested_data_driven_products=requested,
            resolved_data_driven_contract=resolved,
        )

    assert not source_path.exists()
    assert not metadata_sidecar_path(source_path).exists()
