from __future__ import annotations

import json

import hist
import numpy as np
import pytest

uproot = pytest.importorskip("uproot")

from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.datacard_tools import load_and_merge_histogram_pkls
from topeft.modules.histogram_artifact import write_histogram_artifact
from topeft.modules.nominal_schema import (
    eft_nominal_key,
    materialize_legacy_histogram_dict,
    scalar_nominal_key,
)
from topeft.modules.sumw2_policy import resolve_sumw2_storage_policy
from tests.sumw2_profile_test_helpers import certify_test_profile


def _categorical_axes():
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
    )


def _fill_coordinates(output, process, weight, coefficients=None):
    values = {
        "process": process,
        "channel": "3l_onZ_1b",
        "systematic": "nominal",
        "appl": "isSR",
        "njets": np.asarray([0.5]),
        "weight": np.asarray([weight]),
    }
    if coefficients is not None:
        values["eft_coeff"] = np.asarray([coefficients])
    output.fill(**values)


def _evaluated_total(histogram, wc_values):
    evaluated = histogram.eval(wc_values)
    return sum(
        (np.asarray(values, dtype=float) for values in evaluated.values()),
        np.zeros(4),
    )


def test_datacard_transient_view_preserves_rates_shapes_coefficients_scalings_and_root_txt(tmp_path):
    scalar = SparseHist(
        *_categorical_axes(),
        hist.axis.Regular(2, 0.0, 2.0, name="njets"),
        storage="Double",
    )
    _fill_coordinates(scalar, "background", 4.0)
    eft = HistEFT(
        *_categorical_axes(),
        hist.axis.Regular(2, 0.0, 2.0, name="njets"),
        wc_names=["ctG"],
        label="Events",
    )
    _fill_coordinates(eft, "signal", 2.0, [1.5, 2.0, 3.0])
    companion = SparseHist(
        *_categorical_axes(),
        hist.axis.Regular(2, 0.0, 2.0, name="njets_sumw2"),
        storage="Double",
    )
    for process, weight in (("background", 16.0), ("signal", 9.0)):
        companion.fill(
            process=process,
            channel="3l_onZ_1b",
            systematic="nominal",
            appl="isSR",
            njets_sumw2=np.asarray([0.5]),
            weight=np.asarray([weight]),
        )

    baseline = HistEFT(
        *_categorical_axes(),
        hist.axis.Regular(2, 0.0, 2.0, name="njets"),
        wc_names=["ctG"],
        label="Events",
    )
    _fill_coordinates(baseline, "background", 4.0)
    _fill_coordinates(baseline, "signal", 2.0, [1.5, 2.0, 3.0])
    split = {
        scalar_nominal_key("njets"): scalar,
        eft_nominal_key("njets"): eft,
        "njets_sumw2": companion,
    }
    samples = {
        "background_dataset": {
            "histAxisName": "background",
            "isData": False,
            "WCnames": [],
        },
        "signal_dataset": {
            "histAxisName": "signal",
            "isData": False,
            "WCnames": ["ctG"],
        },
    }
    policy = resolve_sumw2_storage_policy(
        {"mode": "full_diagnostics"},
        samples=samples,
        runtime_families=("njets",),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    source_path = tmp_path / "processor.pkl.gz"
    write_histogram_artifact(
        source_path,
        histograms=split,
        artifact_kind="processor_output",
        sumw2_storage_provenance=policy.to_provenance(),
        production_sample_contract=certify_test_profile(policy, samples),
    )
    split, merge_report = load_and_merge_histogram_pkls([str(source_path)])
    assert merge_report["artifact_kind"] == "processor_output"
    transient = materialize_legacy_histogram_dict(
        split,
        runtime_families=("njets",),
        require_companions=("njets",),
    )
    assert tuple(transient) == ("njets", "njets_sumw2")
    assert list(transient["njets"].wc_names) == ["ctG"]
    assert [axis.name for axis in transient["njets"].axes] == [
        axis.name for axis in baseline.axes
    ]
    assert set(transient["njets"].axes["process"]) == set(baseline.axes["process"])
    for point in ({}, {"ctG": 1.0}, {"ctG": -0.5}):
        np.testing.assert_allclose(
            _evaluated_total(transient["njets"], point),
            _evaluated_total(baseline, point),
        )
    np.testing.assert_allclose(
        transient["njets"].make_scaling(), baseline.make_scaling()
    )

    baseline_values = _evaluated_total(baseline, {})[1:-1]
    transient_values = _evaluated_total(transient["njets"], {})[1:-1]
    edges = np.asarray([0.0, 1.0, 2.0])
    baseline_root = tmp_path / "baseline.root"
    transient_root = tmp_path / "transient.root"
    with uproot.recreate(baseline_root) as output:
        output["njets"] = (baseline_values, edges)
    with uproot.recreate(transient_root) as output:
        output["njets"] = (transient_values, edges)
    with uproot.open(baseline_root) as baseline_file, uproot.open(transient_root) as transient_file:
        np.testing.assert_allclose(
            baseline_file["njets"].values(flow=True),
            transient_file["njets"].values(flow=True),
        )

    baseline_txt = tmp_path / "baseline.txt"
    transient_txt = tmp_path / "transient.txt"
    baseline_payload = {
        "rate": baseline_values.tolist(),
        "wc_plus_one": _evaluated_total(baseline, {"ctG": 1.0})[1:-1].tolist(),
    }
    transient_payload = {
        "rate": transient_values.tolist(),
        "wc_plus_one": _evaluated_total(transient["njets"], {"ctG": 1.0})[1:-1].tolist(),
    }
    baseline_txt.write_text(json.dumps(baseline_payload, sort_keys=True))
    transient_txt.write_text(json.dumps(transient_payload, sort_keys=True))
    assert baseline_txt.read_bytes() == transient_txt.read_bytes()

    transient.clear()
    assert not transient
    assert tuple(split) == (
        scalar_nominal_key("njets"),
        eft_nominal_key("njets"),
        "njets_sumw2",
    )
