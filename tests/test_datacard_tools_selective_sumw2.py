from __future__ import annotations

import numpy as np
import pytest

hist = pytest.importorskip("hist")
uproot = pytest.importorskip("uproot")

from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.datacard_tools import (
    DatacardMaker,
    process_retains_stat_uncertainty,
)
from topeft.modules.histogram_artifact import write_histogram_artifact
from topeft.modules.nominal_schema import (
    eft_nominal_key,
    materialize_legacy_histogram_dict,
    scalar_nominal_key,
)
from topeft.modules.sumw2_policy import resolve_sumw2_storage_policy
from tests.sumw2_profile_test_helpers import certify_test_profile


_CHANNEL = "3l_onZ_1b"


def _axes(dense_name):
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name=dense_name),
    )


def _fill_scalar(histogram, process, dense_name, weight, systematic="nominal"):
    histogram.fill(
        process=process,
        channel=_CHANNEL,
        systematic=systematic,
        **{dense_name: np.asarray([0.5])},
        weight=np.asarray([weight]),
    )


def _fill_eft(histogram, process, weight, coefficients):
    histogram.fill(
        process=process,
        channel=_CHANNEL,
        systematic="nominal",
        njets=np.asarray([0.5]),
        weight=np.asarray([weight]),
        eft_coeff=np.asarray([coefficients]),
    )


def _txt_rates(txt_path):
    process_names = None
    rates = None
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "process" and any(name.endswith("_sm") for name in fields[1:]):
            process_names = fields[1:]
        if fields[0] == "rate":
            rates = [float(rate) for rate in fields[1:]]
    assert process_names is not None
    assert rates is not None
    return dict(zip(process_names, rates))


def _selective_split_fixture(tmp_path):
    scalar = SparseHist(*_axes("njets"), storage="Double")
    _fill_scalar(scalar, "nonpromptUL18", "njets", 10.0)

    eft = HistEFT(*_axes("njets"), wc_names=["ctW"], label="Events")
    _fill_eft(eft, "tllq_privateUL18", 5.0, [2.0, 3.0, 4.0])
    _fill_eft(eft, "unselected_eftUL18", 4.0, [1.5, 2.0, 3.0])

    companion = SparseHist(*_axes("njets_sumw2"), storage="Double")
    _fill_scalar(companion, "nonpromptUL18", "njets_sumw2", 16.0)
    _fill_scalar(companion, "tllq_privateUL18", "njets_sumw2", 25.0)

    samples = {
        "fakes_dataset": {
            "histAxisName": "nonpromptUL18",
            "isData": False,
            "WCnames": [],
        },
        "private_signal_dataset": {
            "histAxisName": "tllq_privateUL18",
            "isData": False,
            "WCnames": ["ctW"],
        },
        "unselected_signal_dataset": {
            "histAxisName": "unselected_eftUL18",
            "isData": False,
            "WCnames": ["ctW"],
        },
    }
    policy = resolve_sumw2_storage_policy(
        {
            "mode": "full_custom",
            "rules": [
                {
                    "process_names": ["nonpromptUL18", "tllq_privateUL18"],
                    "variables": ["njets"],
                }
            ],
        },
        samples=samples,
        runtime_families=("njets",),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    source_path = tmp_path / "selective_split.pkl.gz"
    write_histogram_artifact(
        source_path,
        histograms={
            scalar_nominal_key("njets"): scalar,
            eft_nominal_key("njets"): eft,
            "njets_sumw2": companion,
        },
        artifact_kind="processor_output",
        sumw2_storage_provenance=policy.to_provenance(),
        production_sample_contract=certify_test_profile(policy, samples),
    )

    from topeft.modules.datacard_tools import load_and_merge_histogram_pkls

    reopened, report = load_and_merge_histogram_pkls([str(source_path)])
    assert report["schema"] == "split_sibling_v1"
    materialized = materialize_legacy_histogram_dict(
        reopened,
        runtime_families=("njets",),
        require_companions=("njets",),
    )
    assert tuple(materialized) == ("njets", "njets_sumw2")
    return materialized


def _make_legacy_hists(process, companion_process):
    nominal = HistEFT(*_axes("njets"), wc_names=[], label="Events")
    _fill_scalar(nominal, process, "njets", 3.0)
    companion = HistEFT(*_axes("njets_sumw2"), wc_names=[], label="Events")
    _fill_scalar(companion, companion_process, "njets_sumw2", 7.0)
    return {"njets": nominal, "njets_sumw2": companion}


@pytest.mark.parametrize(
    ("process_name", "expected"),
    [
        ("fakes", True),
        ("example_close_process", True),
        ("tllq", False),
        ("unselected", False),
    ],
)
def test_process_retains_stat_uncertainty_current_policy(process_name, expected):
    assert process_retains_stat_uncertainty(process_name) is expected


def test_selective_companion_terminal_writes_root_and_txt_without_synthetic_variance(tmp_path):
    materialized = _selective_split_fixture(tmp_path)
    maker = DatacardMaker(
        hists=materialized,
        out_dir=str(tmp_path),
        var_lst=["njets"],
        do_nuisance=False,
        verbose=False,
        use_AAC=True,
    )
    maker.analyze(
        "njets",
        _CHANNEL,
        {"fakes": [], "tllq": ["ctW"], "unselected": ["ctW"]},
        True,
        {},
    )

    root_path = tmp_path / "ttx_multileptons-3l_onZ_1b_njets.root"
    txt_path = tmp_path / "ttx_multileptons-3l_onZ_1b_njets.txt"
    assert root_path.is_file()
    assert txt_path.is_file()
    expected_integrals = {
        "fakes_sm": 10.0,
        "tllq_sm": 10.0,
        "tllq_lin_ctW": 45.0,
        "tllq_quad_ctW": 20.0,
        "unselected_sm": 6.0,
        "unselected_lin_ctW": 26.0,
        "unselected_quad_ctW": 12.0,
    }
    with uproot.open(root_path) as root_file:
        object_names = [key.split(";", 1)[0] for key in root_file.keys()]
        assert len(object_names) == len(set(object_names))
        for name, expected_integral in expected_integrals.items():
            template = root_file[name]
            assert float(np.asarray(template.values(), dtype=float).sum()) == pytest.approx(
                expected_integral
            )
            expected_variance = 16.0 if name == "fakes_sm" else 0.0
            assert float(np.asarray(template.variances(), dtype=float).sum()) == pytest.approx(
                expected_variance
            )

    assert "ttx_multileptons-3l_onZ_1b_njets.root" in txt_path.read_text(
        encoding="utf-8"
    )
    rates = _txt_rates(txt_path)
    assert rates == pytest.approx(expected_integrals)


def test_datacard_emits_independent_renorm_and_fact_shapes(tmp_path):
    histogram = HistEFT(*_axes("njets"), wc_names=[], label="Events")
    for systematic, weight in (
        ("nominal", 10.0),
        ("renormUp", 11.0),
        ("renormDown", 9.0),
        ("factUp", 12.0),
        ("factDown", 8.0),
    ):
        _fill_scalar(
            histogram,
            "ttH_centralUL18",
            "njets",
            weight,
            systematic=systematic,
        )

    maker = DatacardMaker(
        hists={"njets": histogram},
        out_dir=str(tmp_path),
        var_lst=["njets"],
        do_nuisance=True,
        skip_missing_parton_rate_syst=True,
        verbose=False,
    )
    maker.analyze("njets", _CHANNEL, {"ttH": []}, True, {})

    root_path = tmp_path / "ttx_multileptons-3l_onZ_1b_njets.root"
    txt_path = tmp_path / "ttx_multileptons-3l_onZ_1b_njets.txt"
    with uproot.open(root_path) as root_file:
        object_names = {key.split(";", 1)[0] for key in root_file.keys()}
    for template_name in (
        "ttH_sm_renorm_ttHUp",
        "ttH_sm_renorm_ttHDown",
        "ttH_sm_fact_ttHUp",
        "ttH_sm_fact_ttHDown",
    ):
        assert template_name in object_names
    assert not any("renormfact" in name for name in object_names)

    card_text = txt_path.read_text(encoding="utf-8")
    assert "renorm" in card_text
    assert "fact" in card_text
    assert "renormfact" not in card_text


@pytest.mark.parametrize(
    ("nominal_process", "selected_process"),
    [
        ("nonpromptUL18", "fakes"),
        ("example_close_processUL18", "example_close"),
    ],
)
def test_missing_retained_variance_companion_fails_clearly(
    tmp_path, nominal_process, selected_process
):
    maker = DatacardMaker(
        hists=_make_legacy_hists(nominal_process, "tllq_privateUL18"),
        out_dir=str(tmp_path),
        var_lst=["njets"],
        do_nuisance=False,
        verbose=False,
    )
    with pytest.raises(RuntimeError, match="requires a process companion") as error_info:
        maker.analyze("njets", _CHANNEL, {selected_process: []}, True, {})

    message = str(error_info.value)
    assert selected_process in message
    assert "njets_sumw2" in message


@pytest.mark.parametrize(
    ("raw_process", "final_process"),
    [
        ("nonpromptUL18", "fakes"),
        ("example_close_processUL18", "example_close"),
    ],
)
def test_retained_variance_root_templates_keep_companion_variance(
    tmp_path, raw_process, final_process
):
    maker = DatacardMaker(
        hists=_make_legacy_hists(raw_process, raw_process),
        out_dir=str(tmp_path),
        var_lst=["njets"],
        do_nuisance=False,
        verbose=False,
    )
    maker.analyze("njets", _CHANNEL, {final_process: []}, True, {})

    root_path = tmp_path / "ttx_multileptons-3l_onZ_1b_njets.root"
    with uproot.open(root_path) as root_file:
        template = root_file[f"{final_process}_sm"]
        assert float(np.asarray(template.values(), dtype=float).sum()) == pytest.approx(
            3.0
        )
        assert float(np.asarray(template.variances(), dtype=float).sum()) == pytest.approx(
            7.0
        )


def test_absent_companion_object_uses_zero_variance_legacy_path(tmp_path):
    nominal = HistEFT(*_axes("njets"), wc_names=[], label="Events")
    _fill_scalar(nominal, "tllq_privateUL18", "njets", 3.0)
    maker = DatacardMaker(
        hists={"njets": nominal},
        out_dir=str(tmp_path),
        var_lst=["njets"],
        do_nuisance=False,
        verbose=False,
    )
    maker.analyze("njets", _CHANNEL, {"tllq": []}, True, {})

    root_path = tmp_path / "ttx_multileptons-3l_onZ_1b_njets.root"
    with uproot.open(root_path) as root_file:
        template = root_file["tllq_sm"]
        assert float(np.asarray(template.variances(), dtype=float).sum()) == pytest.approx(
            0.0
        )
