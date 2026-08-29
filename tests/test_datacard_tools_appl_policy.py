from __future__ import annotations

from collections import namedtuple

import pytest

hist = pytest.importorskip("hist")
np = pytest.importorskip("numpy")
uproot = pytest.importorskip("uproot")

from topcoffea.modules.histEFT import HistEFT
from topeft.modules import datacard_tools
from topeft.modules.datacard_tools import DatacardMaker


def _make_hist(axis_name, channel, appl_yields=None, process="tZqUL18"):
    axes = [
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
    ]
    if appl_yields is not None:
        axes.append(hist.axis.StrCategory([], name="appl", growth=True))
    axes.append(hist.axis.Regular(1, 0.0, 1.0, name=axis_name))
    histogram = HistEFT(*axes, wc_names=[], label="Events")
    histogram.metadata = {}

    values = appl_yields if appl_yields is not None else {None: 100.0}
    for appl_label, value in values.items():
        fill_args = {
            "process": process,
            "channel": channel,
            "systematic": "nominal",
            axis_name: np.array([0.5]),
            "weight": np.array([float(value)]),
        }
        if appl_label is not None:
            fill_args["appl"] = appl_label
        histogram.fill(**fill_args)
    return histogram


def _evaluated_total(histogram):
    return sum(
        float(np.asarray(values, dtype=float).sum())
        for values in histogram.eval({}).values()
    )


def _write_card(tmp_path, channel, appl_yields):
    hists = {
        "njets": _make_hist("njets", channel, appl_yields),
        "njets_sumw2": _make_hist("njets_sumw2", channel, appl_yields),
    }
    card_maker = DatacardMaker(
        hists=hists,
        out_dir=str(tmp_path),
        var_lst=["njets"],
        do_nuisance=False,
        verbose=False,
    )
    card_maker.analyze("njets", channel, {"tZq": []}, True, {})
    stem = f"ttx_multileptons-{channel}_njets"
    return tmp_path / f"{stem}.root", tmp_path / f"{stem}.txt"


def _txt_rates(txt_path):
    process_names = None
    rates = None
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "process" and any(field.endswith("_sm") for field in fields[1:]):
            process_names = fields[1:]
        if fields[0] == "rate":
            rates = [float(field) for field in fields[1:]]
    return dict(zip(process_names, rates))


def _root_integrals(root_path, object_name):
    integrals = {}
    with uproot.open(root_path) as root_file:
        for key in root_file.keys():
            if key.split(";")[0] == object_name:
                integrals[key] = float(
                    np.asarray(root_file[key].values(), dtype=float).sum()
                )
    return integrals


def test_with_appl_selects_only_expected_sr_category():
    maker = object.__new__(DatacardMaker)
    source = _make_hist(
        "njets",
        "3l_onZ_1b",
        {"isSR_3l": 100.0, "isAR_3l": 3.0},
    )

    selected = maker.select_final_sr_appl(source, "3l_onZ_1b", process="tZq")

    assert "appl" not in maker._axis_names(selected)
    assert _evaluated_total(selected) == pytest.approx(100.0)


def test_final_channel_resolves_through_exact_base_channel_registry():
    maker = object.__new__(DatacardMaker)
    source = _make_hist(
        "ht",
        "3l_onZ_1b_2j",
        {"isSR_3l": 17.0, "isAR_3l": 5.0},
    )

    selected = maker.select_final_sr_appl(source, "3l_onZ_1b_2j")

    assert _evaluated_total(selected) == pytest.approx(17.0)


def test_missing_expected_sr_label_fails():
    maker = object.__new__(DatacardMaker)
    source = _make_hist("njets", "3l_onZ_1b", {"isAR_3l": 3.0})

    with pytest.raises(ValueError) as exc_info:
        maker.select_final_sr_appl(source, "3l_onZ_1b", process="tZq")

    message = str(exc_info.value)
    assert "recognized channel '3l_onZ_1b', process 'tZq'" in message
    assert "metadata-defined SR channel" in message
    assert "exact expected appl label 'isSR_3l' is missing" in message
    assert "Available appl labels are ['isAR_3l']" in message
    assert "No SR/AR integration, label guessing, or fallback was performed" in message


@pytest.mark.parametrize(
    "channel",
    [
        "3l_CR",
        "3l_onZ_1b_AR",
        "custom_3l",
    ],
)
def test_non_sr_channel_with_appl_fails_with_explicit_support_boundary(channel):
    maker = object.__new__(DatacardMaker)
    source = _make_hist("njets", channel, {"isSR_3l": 3.0})

    with pytest.raises(ValueError) as exc_info:
        maker.select_final_sr_appl(source, channel, process="tZq")

    message = str(exc_info.value)
    assert "supports only metadata-defined SR channels" in message
    assert f"Requested channel {channel!r}, process 'tZq'" in message
    assert "is not in the ALL_CH_LST_SR contract" in message
    assert "CR/AR application-axis card production is not implemented" in message
    assert "No SR/AR integration, label guessing, or fallback was performed" in message
    assert "already projected/no-appl input" in message
    assert "separately reviewed supported workflow" in message


def test_analyze_rejects_cr_zero_jet_before_unrelated_channel_parsing(tmp_path):
    channel = "3l_CR_0j"
    hists = {
        "ht": _make_hist(
            "ht",
            channel,
            {"isSR_3l": 17.0, "isAR_3l": 5.0},
        ),
        "ht_sumw2": _make_hist(
            "ht_sumw2",
            channel,
            {"isSR_3l": 19.0, "isAR_3l": 7.0},
        ),
    }
    maker = DatacardMaker(
        hists=hists,
        out_dir=str(tmp_path),
        var_lst=["ht"],
        do_nuisance=False,
        verbose=False,
    )

    with pytest.raises(ValueError, match="supports only metadata-defined SR channels"):
        maker.analyze("ht", channel, {"tZq": []}, True, {})

    assert list(tmp_path.iterdir()) == []


def test_no_appl_input_is_identity_and_does_not_load_registry(monkeypatch):
    maker = object.__new__(DatacardMaker)
    source = _make_hist("njets", "legacy_channel")

    def fail_if_loaded():
        raise AssertionError("no-appl compatibility path loaded SR metadata")

    monkeypatch.setattr(
        datacard_tools,
        "load_missing_parton_channel_contract",
        fail_if_loaded,
    )

    assert maker.select_final_sr_appl(source, "legacy_channel") is source


def test_nominal_and_sumw2_appl_selection_remain_aligned():
    maker = object.__new__(DatacardMaker)
    nominal = _make_hist(
        "njets",
        "2lss_p",
        {"isSR_2lSS": 11.0, "isAR_2lSS": 101.0},
    )
    sumw2 = _make_hist(
        "njets_sumw2",
        "2lss_p",
        {"isSR_2lSS": 13.0, "isAR_2lSS": 103.0},
    )

    selected_nominal = maker.select_final_sr_appl(nominal, "2lss_p")
    selected_sumw2 = maker.select_final_sr_appl(sumw2, "2lss_p")

    assert _evaluated_total(selected_nominal) == pytest.approx(11.0)
    assert _evaluated_total(selected_sumw2) == pytest.approx(13.0)
    assert tuple(ax.name for ax in selected_nominal.categorical_axes) == tuple(
        ax.name for ax in selected_sumw2.categorical_axes
    )
    assert "appl" not in maker._axis_names(selected_nominal)
    assert "appl" not in maker._axis_names(selected_sumw2)


def test_card_writer_does_not_sum_sr_and_ar(tmp_path):
    root_path, txt_path = _write_card(
        tmp_path,
        "3l_onZ_1b",
        {"isSR_3l": 100.0, "isAR_3l": 3.0},
    )

    assert _root_integrals(root_path, "tZq_sm") == {
        "tZq_sm;1": pytest.approx(100.0)
    }
    assert _txt_rates(txt_path)["tZq_sm"] == pytest.approx(100.0)


def test_no_appl_card_preserves_public_rate_semantics(tmp_path):
    root_path, txt_path = _write_card(tmp_path, "3l_onZ_1b", None)

    assert _root_integrals(root_path, "tZq_sm") == {
        "tZq_sm;1": pytest.approx(100.0)
    }
    assert _txt_rates(txt_path)["tZq_sm"] == pytest.approx(100.0)


def test_unresolved_sparse_axis_guard_reports_duplicate_name_risk():
    sparse_key = namedtuple("sparse_key", ["systematic", "appl"])
    templates = {
        sparse_key("nominal", "isSR_3l"): [
            np.array([0.0, 100.0, 0.0]),
            np.array([0.0, 100.0, 0.0]),
        ],
        sparse_key("nominal", "isAR_3l"): [
            np.array([0.0, 3.0, 0.0]),
            np.array([0.0, 3.0, 0.0]),
        ],
    }

    with pytest.raises(ValueError, match="Unresolved sparse axis 'appl'") as exc_info:
        DatacardMaker.validate_sparse_axes_for_card(
            templates,
            channel="3l_onZ_1b",
            process="tZq_sm",
        )

    assert "duplicate ROOT template names" in str(exc_info.value)
