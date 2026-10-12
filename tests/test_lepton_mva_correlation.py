"""Focused card-time identity and numerical controls for lepton MVA correlation."""

import numpy as np
import pytest

hist = pytest.importorskip("hist")
uproot = pytest.importorskip("uproot")

from analysis.topeft_run2 import make_cards
from analysis.topeft_run2.make_cards import build_arg_parser
from topcoffea.modules.histEFT import HistEFT
from topeft.modules.datacard_tools import (
    DatacardMaker, cross_run_mva_identities, resolve_shape_nuisance_identity,
)


@pytest.mark.parametrize("base", tuple(cross_run_mva_identities))
@pytest.mark.parametrize("run_suffix,domain", (("_run2", "13TeV"), ("_run3", "13p6TeV")))
@pytest.mark.parametrize("direction", ("Up", "Down"))
def test_resolver_exact_mva_identities(base, run_suffix, domain, direction):
    for mode, expected in (("per-run", f"{base}_{domain}"),
                           ("cross-run", base)):
        assert resolve_shape_nuisance_identity(
            f"{base}{direction}", run_suffix, [base], mode,
        ) == (expected, f"{expected}{direction}")
    assert resolve_shape_nuisance_identity(
        f"{base}{direction}", run_suffix, [base],
    ) == (f"{base}_{domain}", f"{base}_{domain}{direction}")


@pytest.mark.parametrize("run_suffix", ("_run2", "_run3"))
@pytest.mark.parametrize("base", (
    "lepSF_elec_non_mva", "lepSF_muon_non_mva", "lepSF_elec", "FES", "nominal",
))
def test_non_mva_identity_is_mode_invariant(base, run_suffix):
    run_decorrelate = [base] if base != "nominal" else []
    reference = resolve_shape_nuisance_identity(f"{base}Up", run_suffix, run_decorrelate)
    assert resolve_shape_nuisance_identity(
        f"{base}Up", run_suffix, run_decorrelate, "cross-run",
    ) == reference


def test_invalid_mode_rejected_before_card_output(tmp_path):
    with pytest.raises(ValueError, match="Invalid lepton MVA correlation"):
        DatacardMaker(hists={}, out_dir=str(tmp_path / "cards"), lepton_mva_correlation="shared")
    assert not (tmp_path / "cards").exists()
    with pytest.raises(ValueError, match="Invalid lepton MVA correlation"):
        resolve_shape_nuisance_identity("nominal", "_run3", [], "shared")
    parser = build_arg_parser()
    assert parser.parse_args([]).lepton_mva_correlation == "per-run"
    for mode in ("per-run", "cross-run"):
        assert parser.parse_args(["--lepton-mva-correlation", mode]).lepton_mva_correlation == mode
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--lepton-mva-correlation", "shared"])
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ("per-run", "cross-run"))
def test_direct_cli_forwards_one_mode_to_maker(tmp_path, monkeypatch, mode):
    observed = []
    class maker_spy:
        def __init__(self, **kwargs):
            observed.append(kwargs)
            self.hists = {"njets": object()}

        def channels(self, distribution):
            return []

        def get_selected_wcs(self, distribution, channels):
            return {}

        def is_signal(self, process):
            return False

    monkeypatch.setattr(make_cards.datacard_tools, "load_and_merge_histogram_pkls",
                        lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(make_cards.datacard_tools, "DatacardMaker", maker_spy)
    monkeypatch.setattr(make_cards, "_emit_merge_report", lambda *args: None)
    args = build_arg_parser().parse_args([
        "synthetic.pkl.gz", "--out-dir", str(tmp_path / "cards"),
        "--var-lst", "njets", "--select-only", "--skip-selected-wcs-check",
        "--lepton-mva-correlation", mode,
    ])
    make_cards._run_producer(args, ["synthetic.pkl.gz"])
    assert len(observed) == 1
    assert observed[0]["lepton_mva_correlation"] == mode


def _make_card(directory, mode, years):
    directory.mkdir()
    channel = "3l_onZ_1b"
    histogram = HistEFT(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name="njets"),
        wc_names=[], label="Events",
    )
    weights = {"nominal": 10.0, "lepSF_elec_mvaUp": 12.0,
               "lepSF_elec_mvaDown": 8.0, "lepSF_muon_mvaUp": 11.0,
               "lepSF_muon_mvaDown": 9.0, "lepSF_elec_non_mvaUp": 13.0,
               "lepSF_elec_non_mvaDown": 7.0}
    for systematic, weight in weights.items():
        histogram.fill(process="ttH_centralUL18", channel=channel,
                       systematic=systematic, njets=np.asarray([0.5]),
                       weight=np.asarray([weight]))
    maker = DatacardMaker(
        hists={"njets": histogram}, out_dir=str(directory), var_lst=["njets"],
        year_lst=years,
        do_nuisance=True, skip_missing_parton_rate_syst=True,
        lepton_mva_correlation=mode, verbose=False,
    )
    maker.analyze("njets", channel, {"ttH": []}, True, {})
    stem = directory / "ttx_multileptons-3l_onZ_1b_njets"
    return stem.with_suffix(".txt"), stem.with_suffix(".root"), maker.scalings


def _card_contents(card, root):
    rows = [line.split() for line in card.read_text().splitlines()]
    shape_names = {row[0] for row in rows if len(row) >= 2 and row[1] == "shape"}
    rate_row = next(row for row in rows if row and row[0] == "rate")
    with uproot.open(root) as source:
        templates = {key: (source[key].values(flow=True), source[key].variances(flow=True))
                     for key in source.keys(cycle=False)}
    return shape_names, rate_row, templates


@pytest.mark.parametrize("years,domain", (([], "13p6TeV"), (["UL18"], "13TeV")))
def test_card_time_txt_root_and_numeric_ab_control(tmp_path, years, domain):
    per_card, per_root, per_scalings = _make_card(tmp_path / "per", "per-run", years)
    cross_card, cross_root, cross_scalings = _make_card(tmp_path / "cross", "cross-run", years)
    per_names, per_rates, per_templates = _card_contents(per_card, per_root)
    cross_names, cross_rates, cross_templates = _card_contents(cross_card, cross_root)
    mva_names = lambda names: {
        name for name in names
        if any(name == base or name.startswith(f"{base}_13")
               for base in cross_run_mva_identities)
    }
    assert mva_names(per_names) == {f"{base}_{domain}" for base in cross_run_mva_identities}
    assert mva_names(cross_names) == set(cross_run_mva_identities)
    for base in cross_run_mva_identities:
        shared = base
        assert f"{base}_{domain}" in per_names
        assert shared in cross_names
        for direction in ("Up", "Down"):
            assert f"ttH_sm_{base}_{domain}{direction}" in per_templates
            assert f"ttH_sm_{shared}{direction}" in cross_templates
    def normalize(name):
        for base in cross_run_mva_identities:
            name = name.replace(base, f"{base}_{domain}")
        return name
    assert {normalize(name) for name in cross_names} == per_names
    assert per_rates == cross_rates
    assert per_scalings == cross_scalings
    normalized_templates = {normalize(name): data for name, data in cross_templates.items()}
    assert normalized_templates.keys() == per_templates.keys()
    for name, (values, variances) in per_templates.items():
        np.testing.assert_array_equal(values, normalized_templates[name][0])
        np.testing.assert_array_equal(variances, normalized_templates[name][1])
