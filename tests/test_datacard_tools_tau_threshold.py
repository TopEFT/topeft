from collections import namedtuple
import json

import numpy as np
import pytest

hist = pytest.importorskip("hist")
uproot = pytest.importorskip("uproot")

from topcoffea.modules.histEFT import HistEFT
from topeft.modules.datacard_tools import (
    DatacardMaker,
    _nominal_like_tau_shape_bases,
)


key = namedtuple("key", "systematic")
tau = "CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2022"
tes_run2 = "CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2016"


def _pair(nominal, up, down, nuisance=tau, underflow=(0, 0, 0)):
    templates = {
        key("nominal"): [np.asarray([underflow[0], *nominal], dtype=float), None],
        key(f"{nuisance}Up"): [np.asarray([underflow[1], *up], dtype=float), None],
        key(f"{nuisance}Down"): [np.asarray([underflow[2], *down], dtype=float), None],
    }
    return templates


def _suppressed(templates, raw=None, extent=None, run_decorrelate=()):
    return _nominal_like_tau_shape_bases(
        templates, templates if raw is None else raw,
        extent or len(next(iter(templates.values()))[0]),
        "_run3", run_decorrelate, "per-run",
    )


@pytest.mark.parametrize(
    ("nominal", "up", "down", "expected"),
    [
        ([100000.0], [np.nextafter(100001.0, 100000.0)], [100000.0], True),
        ([100000.0], [100001.0], [100000.0], False),
        ([100000.0], [np.nextafter(100001.0, np.inf)], [100000.0], False),
        ([10.0], [10.0], [11.0], False),
        ([0.0, 10.0], [0.0, 10.0], [0.0, 10.0], True),
        ([0.0, 10.0], [1e-12, 10.0], [0.0, 10.0], False),
        ([1e-12], [1e-12 * (1 + 1e-6)], [1e-12], True),
        ([1e-300, 10.0], [1e300, 10.0], [1e-300, 10.0], False),
        ([10.0], [np.nan], [10.0], False),
        ([10.0], [np.inf], [10.0], False),
    ],
)
def test_strict_pair_comparator(nominal, up, down, expected):
    assert (tau in _suppressed(_pair(nominal, up, down))) is expected


def test_flow_geometry_and_quarantine():
    assert tau not in _suppressed(_pair([10, 10], [10, 11], [10, 10]))
    assert tau in _suppressed(_pair([10], [10], [10], underflow=(3, 3, 3)))
    assert tau not in _suppressed(_pair([10], [10], [10], underflow=(3, 4, 3)))
    assert tau not in _suppressed(_pair([10], [10], [10]), extent=3)
    assert tes_run2 not in _suppressed(_pair([10], [10], [10], nuisance=tes_run2))
    run3_energy = "CMS_scale_t_DeepTau2018v2p5_DM1_genTau_2022"
    assert run3_energy not in _suppressed(_pair([10], [11], [9], nuisance=run3_energy))
    ambiguous_energy = "CMS_scale_t_DeepTau2018v2p5_DM1_genTau_unknown"
    assert ambiguous_energy not in _suppressed(_pair([10], [10], [10], nuisance=ambiguous_energy))


def test_missing_partner_duplicate_and_raw_negative_retain():
    pair = _pair([10], [10], [10])
    assert tau not in _suppressed({k: v for k, v in pair.items() if not k.systematic.endswith("Down")})
    raw = _pair([10], [10], [10])
    raw[key(f"{tau}Up")][0][1] = -1
    assert tau not in _suppressed(pair, raw=raw)
    pair[key(f"{tau}Down")][0][1] = 0
    assert tau not in _suppressed(pair)

    alias = _pair([10], [10], [10], nuisance="lepSF_taus_fake_run3")
    alias[key("lepSF_taus_fakeUp")] = [np.asarray([0, 10.0]), None]
    assert "lepSF_taus_fake_run3" not in _suppressed(alias, run_decorrelate=["lepSF_taus_fake"])


def test_cli_default_and_aac_rejection(tmp_path):
    from analysis.topeft_run2.make_cards import build_arg_parser

    parser = build_arg_parser()
    assert parser.parse_args([]).suppress_nominal_like_tau_shapes is False
    assert parser.parse_args(["--suppress-nominal-like-tau-shapes"]).suppress_nominal_like_tau_shapes is True
    with pytest.raises(ValueError, match="unsupported with use_AAC"):
        DatacardMaker(
            hists={}, use_AAC=True, suppress_nominal_like_tau_shapes=True,
            out_dir=str(tmp_path),
        )


def _card(tmp_path, enabled, *, survivor=True):
    tmp_path.mkdir()
    channel = "3l_onZ_1b"
    histogram = HistEFT(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0, 1, name="njets"),
        wc_names=["ctW"], label="Events",
    )
    for process, weights in (
        ("Diboson_central2022", (10, 10, 10)),
        ("tllq_private2022", (5, 5, 5)),
        ("ttH_central2022", (10, 11, 9) if survivor else (10, 10, 10)),
    ):
        for systematic, weight in zip(("nominal", f"{tau}Up", f"{tau}Down"), weights):
            histogram.fill(
                process=process, channel=channel, systematic=systematic,
                njets=np.asarray([0.5]), weight=np.asarray([weight]),
                eft_coeff=np.asarray([[1.0, 0.2, 0.1]]),
            )
    options = {} if enabled is None else {"suppress_nominal_like_tau_shapes": enabled}
    maker = DatacardMaker(
        hists={"njets": histogram}, out_dir=str(tmp_path),
        var_lst=["njets"], do_nuisance=True,
        skip_missing_parton_rate_syst=True, verbose=False,
        **options,
    )
    maker.analyze("njets", channel, {"Diboson": [], "tllq": ["ctW"], "ttH": []}, True, {})
    stem = tmp_path / "ttx_multileptons-3l_onZ_1b_njets"
    root_path, txt_path = stem.with_suffix(".root"), stem.with_suffix(".txt")
    with uproot.open(root_path) as root_file:
        shapes = {name: (root_file[name].values(), root_file[name].variances()) for name in root_file.keys(cycle=False)}
    rows = [line.split() for line in txt_path.read_text().splitlines()]
    processes = next(row[1:] for row in rows if row[0] == "process" and "tllq_sm" in row)
    tau_row = next(row[2:] for row in rows if row[0] == tau and row[1] == "shape")
    rate_row = next(row for row in rows if row[0] == "rate")
    return (
        shapes, dict(zip(processes, tau_row)), txt_path.read_text(),
        rate_row, json.dumps(maker.scalings, sort_keys=True),
    )


def test_default_off_and_local_signal_background_suppression(tmp_path):
    off, off_cells, off_text, off_rates, off_scalings = _card(tmp_path / "off", False)
    omitted, omitted_cells, omitted_text, omitted_rates, omitted_scalings = _card(tmp_path / "omitted", None)
    on, on_cells, on_text, on_rates, on_scalings = _card(tmp_path / "on", True)
    assert off_rates == omitted_rates == on_rates
    assert off_scalings == omitted_scalings == on_scalings
    assert omitted_cells == off_cells
    assert omitted_text == off_text
    assert set(omitted) == set(off)
    for name in off:
        assert np.array_equal(omitted[name][0], off[name][0])
        assert np.array_equal(omitted[name][1], off[name][1])
    for process in ("Diboson_sm", "tllq_sm"):
        assert off_cells[process] == "1"
        assert on_cells[process] == "-"
        for direction in ("Up", "Down"):
            assert f"{process}_{tau}{direction}" in off
            assert f"{process}_{tau}{direction}" not in on
        assert np.array_equal(off[process][0], on[process][0])
        assert np.array_equal(off[process][1], on[process][1])
    assert on_cells["ttH_sm"] == "1"
    for direction in ("Up", "Down"):
        name = f"ttH_sm_{tau}{direction}"
        assert np.array_equal(off[name][0], on[name][0])
    assert "tllq_sm" in on_text and "rate" in on_text
    assert "tllq_sm" in off_text


def test_all_inactive_row_is_explicit(tmp_path):
    shapes, cells, text, _, _ = _card(tmp_path / "all_inactive", True, survivor=False)
    assert set(cells.values()) == {"-"}
    assert f"{tau}" in text
    assert not any(name.endswith((f"{tau}Up", f"{tau}Down")) for name in shapes)
