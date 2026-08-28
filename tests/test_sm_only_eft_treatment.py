from __future__ import annotations

import copy
import json
import runpy
import sys
from pathlib import Path
from unittest import mock

import hist
import numpy as np
import pytest

from analysis.topeft_run2 import analysis_processor
from analysis.topeft_run2.analysis_processor import (
    calculate_sm_sumw2_weights,
    prepare_eft_coefficients,
    project_eft_coefficients_for_treatment,
    resolve_eft_treatment,
)
from topcoffea.modules import eft_helper
from topcoffea.modules.histEFT import HistEFT
from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.data_driven_products import resolve_data_driven_products
from topeft.modules.datacard_tools import DatacardMaker
from topeft.modules.sumw2_policy import (
    resolve_nominal_component_availability,
    resolve_sumw2_storage_policy,
)


def _metadata(*, role="sm_only", is_data=False, wc_names=None):
    payload = {
        "isData": is_data,
        "WCnames": list(["ctW"] if wc_names is None else wc_names),
    }
    if role is not None:
        payload["eft_treatment"] = role
    return payload


def _axes():
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name="njets"),
    )


def _fill_eft(output, coefficients):
    output.fill(
        process="ttll_private2022",
        channel="3l",
        systematic="nominal",
        appl="isSR",
        njets=np.asarray([0.5]),
        weight=np.asarray([1.0]),
        eft_coeff=np.asarray([coefficients]),
    )


def _total(output, wc_value):
    evaluated = output.eval({"ctW": wc_value})
    return sum(float(np.asarray(values).sum()) for values in evaluated.values())


@pytest.mark.parametrize("role", ["SM", "constant", "disabled", "", 1, ["sm_only"]])
def test_unknown_eft_treatment_is_rejected(role):
    with pytest.raises(ValueError, match="EFT-TREATMENT-E001"):
        resolve_eft_treatment(_metadata(role=role), sample_name="sample")


def test_unknown_eft_treatment_fails_run_analysis_preflight(tmp_path):
    payload = {
        "files": ["/store/test/2022/file.root"],
        "year": "2022",
        "xsec": 1.0,
        "nEvents": 1,
        "nGenEvents": 1,
        "nSumOfWeights": 1.0,
        "isData": False,
        "histAxisName": "test_private2022",
        "treeName": "Events",
        "options": "",
        "WCnames": ["ctW"],
        "eft_treatment": "unknown",
    }
    metadata_path = tmp_path / "sample.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    script_path = (
        Path(__file__).resolve().parents[1]
        / "analysis/topeft_run2/run_analysis.py"
    )
    argv = [
        "run_analysis.py",
        str(metadata_path),
        "--pretend",
        "--skip-topcoffea-data-check",
    ]
    original_sys_path = list(sys.path)
    sys.path.insert(0, str(script_path.parent))
    try:
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(ValueError, match="EFT-TREATMENT-E001"):
                runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.path = original_sys_path


def test_sm_only_is_rejected_for_data():
    with pytest.raises(ValueError, match="EFT-TREATMENT-E002"):
        resolve_eft_treatment(_metadata(is_data=True), sample_name="data")


@pytest.mark.parametrize("wc_names", [[], ["ctW", "ctW"], ["ctW", ""]])
def test_sm_only_requires_nonempty_unique_wc_names(wc_names):
    with pytest.raises(ValueError, match="EFT-TREATMENT-E003"):
        resolve_eft_treatment(
            _metadata(wc_names=wc_names),
            sample_name="low_mass",
        )


def test_sm_only_requires_the_coefficient_branch_at_runtime():
    with pytest.raises(RuntimeError, match="EFT-TREATMENT-E004"):
        project_eft_coefficients_for_treatment(
            None,
            "sm_only",
            sample_name="low_mass",
        )


def test_absent_role_preserves_scalar_and_eft_classification():
    scalar = _metadata(role=None, wc_names=[])
    varied = _metadata(role=None, wc_names=["ctW"])
    assert resolve_eft_treatment(scalar, sample_name="scalar") is None
    assert resolve_eft_treatment(varied, sample_name="varied") is None
    assert prepare_eft_coefficients(None, [], [], None) is None
    assert resolve_nominal_component_availability(
        {
            "scalar": {**scalar, "histAxisName": "background"},
            "varied": {**varied, "histAxisName": "signal"},
        }
    ) == {"scalar": True, "eft": True}


@pytest.mark.parametrize("sample_name", ["ordinary_ttll", "other_eft_signal"])
def test_absent_role_is_exactly_backward_compatible_for_eft_vectors_and_sumw2(
    sample_name,
):
    native_wc_names = ["ctW", "ctZ"]
    global_wc_names = ["ctZ", "ctW"]
    coefficient_sets = {
        "nominal": np.asarray([[2.0, 3.0, 5.0, 7.0, 11.0, 13.0]]),
        "renormUp": np.asarray([[17.0, 19.0, 23.0, 29.0, 31.0, 37.0]]),
        "JESUp": np.asarray([[41.0, 43.0, 47.0, 53.0, 59.0, 61.0]]),
    }
    scalar_weights = np.asarray([1.75])

    for coefficients in coefficient_sets.values():
        legacy = eft_helper.remap_coeffs(
            native_wc_names,
            global_wc_names,
            coefficients,
        )
        observed = prepare_eft_coefficients(
            coefficients,
            native_wc_names,
            global_wc_names,
            None,
            sample_name=sample_name,
        )
        np.testing.assert_array_equal(observed, legacy)
        np.testing.assert_array_equal(
            calculate_sm_sumw2_weights(scalar_weights, observed),
            calculate_sm_sumw2_weights(scalar_weights, legacy),
        )


def test_sm_only_projection_occurs_after_native_to_global_remapping(monkeypatch):
    remapped = np.asarray([[7.0, 11.0, 13.0, 17.0, 19.0, 23.0]])
    calls = []

    def fake_remap(native_wc_names, global_wc_names, coefficients):
        calls.append((list(native_wc_names), list(global_wc_names), coefficients.copy()))
        return remapped.copy()

    monkeypatch.setattr(analysis_processor.efth, "remap_coeffs", fake_remap)
    source = np.asarray([[101.0, 103.0, 107.0]])
    observed = prepare_eft_coefficients(
        source,
        ["ctW"],
        ["ctW", "ctZ"],
        "sm_only",
        sample_name="low_mass",
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(observed, [[7.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


def test_projected_vectors_retain_only_the_sm_constant():
    coefficients = np.asarray(
        [[7.0, 11.0, 13.0], [2.0, -5.0, 4.0]],
        dtype=np.float64,
    )
    observed = project_eft_coefficients_for_treatment(
        coefficients,
        "sm_only",
        sample_name="low_mass",
    )
    np.testing.assert_array_equal(observed, [[7.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert observed.dtype == coefficients.dtype


def test_low_mass_wc_metadata_keeps_eft_nominal_storage():
    sample = {
        "histAxisName": "ttll_private2022",
        "isData": False,
        "WCnames": ["ctW"],
        "eft_treatment": "sm_only",
    }
    assert resolve_eft_treatment(sample, sample_name="low_mass") == "sm_only"
    assert resolve_nominal_component_availability({"low_mass": sample}) == {
        "scalar": False,
        "eft": True,
    }


def test_same_process_accumulation_preserves_ordinary_eft_dependence():
    ordinary = HistEFT(*_axes(), wc_names=["ctW"], label="Events")
    low_mass = HistEFT(*_axes(), wc_names=["ctW"], label="Events")
    _fill_eft(ordinary, [6.0, 8.0, 10.0])
    _fill_eft(low_mass, [35.0, 0.0, 0.0])
    merged = ordinary + low_mass

    assert list(merged.axes["process"]) == ["ttll_private2022"]
    assert _total(merged, 0.0) == 41.0
    assert _total(merged, 1.0) == 59.0
    assert _total(merged, 2.0) == 97.0
    selected = merged.integrate("process", "ttll_private2022")
    selected = selected.integrate("channel", "3l")
    selected = selected.integrate("systematic", "nominal")
    selected = selected.integrate("appl", "isSR")
    np.testing.assert_allclose(
        selected.make_scaling()[1],
        np.asarray([1.0, 4.0 / 41.0, 10.0 / 41.0]),
    )


def test_sumw2_adds_complete_event_squares_before_aggregation():
    ordinary_weights = calculate_sm_sumw2_weights(
        np.asarray([1.0]),
        np.asarray([[6.0, 8.0, 10.0]]),
    )
    low_mass_weights = calculate_sm_sumw2_weights(
        np.asarray([1.0]),
        np.asarray([[35.0, 0.0, 0.0]]),
    )
    assert float(np.sum(ordinary_weights) + np.sum(low_mass_weights)) == 1261.0
    assert 1261.0 != 41.0**2


def test_two_source_datasets_target_one_prompt_process_and_both_sumw2_inputs():
    process = "ttll_private2022"
    samples = {
        "ordinary_ttll": {
            "histAxisName": process,
            "isData": False,
            "WCnames": ["ctW"],
        },
        "low_mass_ttll": {
            "histAxisName": process,
            "isData": False,
            "WCnames": ["ctW"],
            "eft_treatment": "sm_only",
        },
        "data": {"histAxisName": "data2022", "isData": True, "WCnames": []},
    }
    products = resolve_data_driven_products(
        {
            "nonprompt": {
                "enabled": True,
                "source_contributors": {
                    "data": {"process_names": ["data2022"]},
                    "prompt_mc": {"process_names": [process]},
                },
            },
            "flips": {"enabled": False},
        },
        data_driven_products_present=True,
        legacy_do_np=False,
        samples=samples,
        runtime_families=("njets",),
        metadata_path="test.yml",
    )
    assert products.product("nonprompt").contributors_for("prompt_mc") == (process,)

    policy = resolve_sumw2_storage_policy(
        {
            "mode": "production",
            "rules": [{"process_names": [process], "variables": ["njets"]}],
        },
        samples=samples,
        runtime_families=("njets",),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    selected = {
        (target.dataset, target.process, target.family)
        for target in policy.resolved_targets
    }
    assert selected == {
        ("ordinary_ttll", process, "njets"),
        ("low_mass_ttll", process, "njets"),
    }


def test_datacard_classification_keeps_one_eft_ttll_signal():
    assert DatacardMaker.get_process("ttll_private2022") == "ttll"
    assert DatacardMaker.is_signal("ttll_private2022") is True
    assert DatacardMaker.get_process("ttll_private2022EE") == "ttll"


def test_resolver_does_not_mutate_metadata():
    payload = _metadata()
    before = copy.deepcopy(payload)
    assert resolve_eft_treatment(payload, sample_name="low_mass") == "sm_only"
    assert payload == before
