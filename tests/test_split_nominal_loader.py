from __future__ import annotations

import gzip
import pickle

import hist
import numpy as np
import pytest

from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.datacard_tools import load_and_merge_histogram_pkls
from topeft.modules.histogram_artifact import write_histogram_artifact
from topeft.modules.nominal_schema import (
    eft_nominal_key,
    evaluate_nominal_at_wc,
    scalar_nominal_key,
)
from topeft.modules.sumw2_policy import resolve_sumw2_storage_policy
from analysis.topeft_run2 import make_cards
from tests.sumw2_profile_test_helpers import certify_test_profile


_SAMPLES = {
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


def _axes(dense_name):
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.Regular(2, 0.0, 2.0, name=dense_name),
    )


def _scalar(process, weight, *, companion=False):
    dense_name = "njets_sumw2" if companion else "njets"
    output = SparseHist(*_axes(dense_name), storage="Double")
    output.fill(
        process=process,
        channel="3l",
        systematic="nominal",
        appl="isSR",
        **{dense_name: np.asarray([0.5])},
        weight=np.asarray([weight]),
    )
    return output


def _eft(process, weight):
    output = HistEFT(*_axes("njets"), wc_names=["ctG"], label="Events")
    output.fill(
        process=process,
        channel="3l",
        systematic="nominal",
        appl="isSR",
        njets=np.asarray([0.5]),
        weight=np.asarray([weight]),
        eft_coeff=np.asarray([[1.25, 2.0, 3.0]]),
    )
    return output


@pytest.fixture
def policy():
    return resolve_sumw2_storage_policy(
        {"mode": "full_diagnostics"},
        samples=_SAMPLES,
        runtime_families=("njets",),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )


def _write_versioned(path, payload, policy, samples=_SAMPLES):
    write_histogram_artifact(
        path,
        histograms=payload,
        artifact_kind="processor_output",
        sumw2_storage_provenance=policy.to_provenance(),
        production_sample_contract=certify_test_profile(policy, samples),
    )


def _replace_payload(path, payload):
    with gzip.open(path, "wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _total_sparse(histogram):
    return sum(
        float(np.asarray(values).sum())
        for values in histogram.view(flow=True, as_dict=True).values()
    )


def test_split_roundtrip_same_policy_optional_components_and_post_reopen_merge(tmp_path, policy):
    scalar_path = tmp_path / "scalar.pkl.gz"
    eft_path = tmp_path / "eft.pkl.gz"
    _write_versioned(
        scalar_path,
        {
            scalar_nominal_key("njets"): _scalar("background", 2.0),
            "njets_sumw2": _scalar("background", 4.0, companion=True),
        },
        policy,
    )
    _write_versioned(
        eft_path,
        {
            eft_nominal_key("njets"): _eft("signal", 3.0),
            "njets_sumw2": _scalar("signal", 14.0625, companion=True),
        },
        policy,
    )

    merged, report = load_and_merge_histogram_pkls(
        [str(scalar_path), str(eft_path)],
        require_sumw2=True,
        consumer_required_families=("njets",),
    )
    assert tuple(merged) == (
        scalar_nominal_key("njets"),
        eft_nominal_key("njets"),
        "njets_sumw2",
    )
    assert report["schema"] == "split_sibling_v1"
    assert _total_sparse(evaluate_nominal_at_wc(merged, "njets", {})) == pytest.approx(5.75)
    assert _total_sparse(merged["njets_sumw2"]) == pytest.approx(18.0625)

    cached_path = make_cards._cache_merged_histograms(
        merged, "cached_split", str(tmp_path), report
    )
    assert (tmp_path / "cached_split.pkl.gz.metadata.json").is_file()
    reopened, reopened_report = load_and_merge_histogram_pkls([cached_path])
    assert reopened_report["schema"] == "split_sibling_v1"
    assert tuple(reopened) == tuple(merged)


def test_required_missing_partial_orphan_and_present_unselected_are_rejected(tmp_path, policy):
    missing_path = tmp_path / "missing.pkl.gz"
    _write_versioned(
        missing_path,
        {
            scalar_nominal_key("njets"): _scalar("background", 2.0),
            "njets_sumw2": _scalar("background", 4.0, companion=True),
        },
        policy,
    )
    _replace_payload(
        missing_path,
        {scalar_nominal_key("njets"): _scalar("background", 2.0)},
    )
    with pytest.raises(RuntimeError, match="content mismatch|missing required"):
        load_and_merge_histogram_pkls([str(missing_path)])

    orphan_path = tmp_path / "orphan.pkl.gz"
    _write_versioned(
        orphan_path,
        {
            scalar_nominal_key("njets"): _scalar("background", 2.0),
            "njets_sumw2": _scalar("background", 4.0, companion=True),
        },
        policy,
    )
    _replace_payload(
        orphan_path,
        {"njets_sumw2": _scalar("background", 4.0, companion=True)},
    )
    with pytest.raises(RuntimeError, match="content mismatch"):
        load_and_merge_histogram_pkls([str(orphan_path)])

    disabled = resolve_sumw2_storage_policy(
        {"mode": "disabled"},
        samples={
            "background_dataset": {
                "histAxisName": "background",
                "isData": False,
                "WCnames": [],
            }
        },
        runtime_families=("njets",),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    unselected_path = tmp_path / "unselected.pkl.gz"
    _write_versioned(
        unselected_path,
        {scalar_nominal_key("njets"): _scalar("background", 2.0)},
        disabled,
        {
            "background_dataset": {
                "histAxisName": "background",
                "isData": False,
                "WCnames": [],
            }
        },
    )
    _replace_payload(
        unselected_path,
        {
            scalar_nominal_key("njets"): _scalar("background", 2.0),
            "njets_sumw2": _scalar("background", 4.0, companion=True),
        },
    )
    with pytest.raises(RuntimeError, match="content mismatch"):
        load_and_merge_histogram_pkls([str(unselected_path)])


def test_split_without_sidecar_and_policy_identity_mismatch_are_rejected(tmp_path, policy):
    no_metadata = tmp_path / "no_metadata.pkl.gz"
    with gzip.open(no_metadata, "wb") as stream:
        pickle.dump({scalar_nominal_key("njets"): _scalar("background", 2.0)}, stream)
    with pytest.raises(RuntimeError, match="expected_sidecar_path=.*no_metadata"):
        load_and_merge_histogram_pkls([str(no_metadata)])

    first = tmp_path / "first.pkl.gz"
    second = tmp_path / "second.pkl.gz"
    payload = {
        scalar_nominal_key("njets"): _scalar("background", 2.0),
        "njets_sumw2": _scalar("background", 4.0, companion=True),
    }
    _write_versioned(first, payload, policy)
    altered = policy.to_provenance()
    altered["warnings"] = ["different"]
    write_histogram_artifact(
        second,
        histograms=payload,
        artifact_kind="processor_output",
        sumw2_storage_provenance=altered,
        production_sample_contract=certify_test_profile(policy, _SAMPLES),
    )
    with pytest.raises(RuntimeError, match="source-allocation provenance|policy identities"):
        load_and_merge_histogram_pkls(
            [str(first), str(second)], on_process_collision="allow"
        )


def test_legacy_uniform_dispatch_and_histeft_wc_zero_scalar_access(tmp_path):
    path = tmp_path / "legacy.pkl.gz"
    nominal = _eft("signal", 2.0)
    companion = _eft("signal", 3.125)
    with gzip.open(path, "wb") as stream:
        pickle.dump({"njets": nominal, "njets_sumw2": companion}, stream)
    merged, report = load_and_merge_histogram_pkls([str(path)], require_sumw2=True)
    assert report["schema"] == "legacy_uniform"
    scalar = evaluate_nominal_at_wc(merged, "njets", {}, schema_version=None)
    assert _total_sparse(scalar) == pytest.approx(2.5)
