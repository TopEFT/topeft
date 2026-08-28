import numpy as np
import pytest

hist = pytest.importorskip("hist")
uproot = pytest.importorskip("uproot")

from topcoffea.modules.histEFT import HistEFT
from topeft.modules.datacard_tools import DatacardMaker


_CHANNEL = "3l_onZ_1b"


def _axes():
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name="njets"),
    )


def _fill_template(histogram, systematic, weight):
    histogram.fill(
        process="ttH_centralUL18",
        channel=_CHANNEL,
        systematic=systematic,
        njets=np.asarray([0.5]),
        weight=np.asarray([weight]),
    )


def _make_card(tmp_path, variations):
    histogram = HistEFT(*_axes(), wc_names=[], label="Events")
    _fill_template(histogram, "nominal", 10.0)
    for systematic, weights in variations.items():
        for weight in weights:
            _fill_template(histogram, systematic, weight)

    maker = DatacardMaker(
        hists={"njets": histogram},
        out_dir=str(tmp_path),
        var_lst=["njets"],
        do_nuisance=True,
        skip_missing_parton_rate_syst=True,
        verbose=False,
    )
    maker.analyze("njets", _CHANNEL, {"ttH": []}, True, {})
    stem = tmp_path / "ttx_multileptons-3l_onZ_1b_njets"
    return stem.with_suffix(".root"), stem.with_suffix(".txt")


def _shape_tokens(txt_path):
    rows = [line.split() for line in txt_path.read_text(encoding="utf-8").splitlines()]
    process_row = next(
        row
        for row in rows
        if row and row[0] == "process" and any(name.endswith("_sm") for name in row[1:])
    )
    processes = process_row[1:]
    return {
        row[0]: dict(zip(processes, row[2:]))
        for row in rows
        if len(row) >= 3 and row[1] == "shape"
    }


def _root_names(root_path):
    with uproot.open(root_path) as root_file:
        return set(root_file.keys(cycle=False))


def _assert_active_shape_columns_resolve(root_path, txt_path):
    root_names = _root_names(root_path)
    for nuisance, tokens in _shape_tokens(txt_path).items():
        for process, token in tokens.items():
            if token == "1":
                assert f"{process}_{nuisance}Up" in root_names
                assert f"{process}_{nuisance}Down" in root_names


def test_zero_up_nonzero_down_is_locally_inactive(tmp_path):
    root_path, txt_path = _make_card(
        tmp_path,
        {"renormUp": [1.0, -1.0], "renormDown": [9.0]},
    )

    root_names = _root_names(root_path)
    tokens = _shape_tokens(txt_path)
    assert "ttH_sm_renorm_ttHUp" not in root_names
    assert "ttH_sm_renorm_ttHDown" in root_names
    assert tokens["renorm_ttH"]["ttH_sm"] == "-"
    _assert_active_shape_columns_resolve(root_path, txt_path)


def test_complete_nonzero_decorrelated_pair_is_active(tmp_path):
    root_path, txt_path = _make_card(
        tmp_path,
        {"renormUp": [11.0], "renormDown": [9.0]},
    )

    root_names = _root_names(root_path)
    tokens = _shape_tokens(txt_path)
    assert "ttH_sm_renorm_ttHUp" in root_names
    assert "ttH_sm_renorm_ttHDown" in root_names
    assert tokens["renorm_ttH"]["ttH_sm"] == "1"
    _assert_active_shape_columns_resolve(root_path, txt_path)


def test_one_direction_absent_is_locally_inactive(tmp_path):
    root_path, txt_path = _make_card(tmp_path, {"FSRDown": [9.0]})

    root_names = _root_names(root_path)
    tokens = _shape_tokens(txt_path)
    assert "ttH_sm_FSRUp" not in root_names
    assert "ttH_sm_FSRDown" in root_names
    assert tokens["FSR"]["ttH_sm"] == "-"
    _assert_active_shape_columns_resolve(root_path, txt_path)
