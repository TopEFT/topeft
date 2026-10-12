from __future__ import annotations

import hist
import numpy as np
import pytest
import uproot

from analysis.topeft_run2.standardize_systematics import (
    SYST_NAMING_MAP,
    process_cards,
    process_templates,
    remap_name,
    resolve_nuisance_name,
)
from topcoffea.modules.sparseHist import SparseHist
from topcoffea.modules.histEFT import HistEFT
from topeft.modules.datacard_tools import DatacardMaker


def make_histogram(source_base, year_values):
    histogram = SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name="x"),
        storage="Double",
    )
    for year, (nominal, up, down) in year_values.items():
        process = f"sample{year}"
        systematic_year = "2016APV" if year == "UL16APV" else "2016"
        for systematic, value in (
            ("nominal", nominal),
            (f"{source_base}_{systematic_year}Up", up),
            (f"{source_base}_{systematic_year}Down", down),
        ):
            histogram.fill(
                process=process,
                channel="2lss_p_1tau_offZ_6j",
                systematic=systematic,
                x=0.5,
                weight=value,
            )
    return histogram


def value_for(histogram, systematic):
    matches = [
        value
        for key, value in histogram.view(as_dict=True, flow=True).items()
        if key.process == "sample"
        and key.channel == "2lss_p_1tau_offZ_6j"
        and key.systematic == systematic
    ]
    assert len(matches) == 1
    return float(matches[0][1])


def registered_maker(monkeypatch):
    observed = {}

    def capture_base_read(self, fpath=None, hists=None):
        observed.update(self.syst_year_corr)

    monkeypatch.setattr(DatacardMaker, "read", capture_base_read)
    maker = DatacardMaker(
        hists={}, year_lst=["UL16", "UL16APV"], do_nuisance=False,
        skip_missing_parton_rate_syst=True, verbose=False,
    )
    assert observed == maker.syst_year_corr
    maker.do_nuisance = True
    return maker


@pytest.mark.parametrize(
    "source_base",
    ("FFcloseEl", "CMS_scale_t_DeepTau2017v2p1_DM0_genElectron"),
)
def test_ffclose_and_tau_use_same_native_correlation(source_base, monkeypatch):
    maker = registered_maker(monkeypatch)
    histogram = make_histogram(
        source_base,
        {"UL16": (10, 12, 8), "UL16APV": (20, 22, 18)},
    )
    merged = maker.correlate_years(histogram, add_luminosity_shapes=False)
    assert set(merged.axes["process"]) == {"sample"}
    assert value_for(merged, "nominal") == pytest.approx(30)
    assert value_for(merged, f"{source_base}_2016Up") == pytest.approx(34)
    assert value_for(merged, f"{source_base}_2016Down") == pytest.approx(26)
    assert f"{source_base}_2016APVUp" not in set(merged.axes["systematic"])
    assert f"{source_base}_2016APVDown" not in set(merged.axes["systematic"])


def test_run2_tau_standardizer_stale_and_era_defined_identities(monkeypatch):
    maker = registered_maker(monkeypatch)
    stale_bases = {
        name.removesuffix("_2016")
        for name in SYST_NAMING_MAP["run2"]
        if name.startswith((
            "CMS_fake_t_DeepTau2017v2p1_VSe_",
            "CMS_fake_t_DeepTau2017v2p1_VSmu_",
            "CMS_scale_t_DeepTau2017v2p1_",
        )) and name.endswith("_2016")
    }
    assert len(stale_bases) == 13
    for source_base in stale_bases:
        assert maker.syst_year_corr[source_base] == {
            "2016": ["2016APV"],
            "2016APV": ["2016"],
        }
        native_name = f"{source_base}_2016"
        assert SYST_NAMING_MAP["run2"][native_name] == native_name
        assert f"{source_base}_2016APV" not in SYST_NAMING_MAP["run2"]
        assert resolve_nuisance_name(native_name, "run2") == native_name
        with pytest.raises(ValueError, match="regenerate"):
            resolve_nuisance_name(f"{source_base}_2016APV", "run2")

    vsjet_bases = {
        name.removesuffix("_2016")
        for name in SYST_NAMING_MAP["run2"]
        if name.startswith("CMS_eff_t_DeepTau2017v2p1_VSjet_dm_")
        and name.endswith("_2016")
    }
    assert len(vsjet_bases) == 13
    assert vsjet_bases.isdisjoint(maker.syst_year_corr)
    for source_base in vsjet_bases:
        native = f"{source_base}_2016APV"
        assert resolve_nuisance_name(native, "run2") == native
        assert remap_name(f"ttH_sm_{native}Up", "run2") == f"ttH_sm_{native}Up"


@pytest.mark.parametrize("run,domain", (("run2", "13TeV"), ("run3", "13p6TeV")))
def test_standardizer_owns_exact_mva_and_non_mva_names(run, domain):
    for source in (
        "lepSF_elec_mva", "lepSF_muon_mva",
        "lepSF_elec_non_mva", "lepSF_muon_non_mva",
    ):
        target = f"{source}_{domain}"
        assert SYST_NAMING_MAP[run][source] == target
        assert resolve_nuisance_name(source, run) == target
        assert resolve_nuisance_name(target, run) == target
        assert remap_name(f"ttH_sm_{source}Up", run) == f"ttH_sm_{target}Up"
        assert remap_name(f"ttH_sm_{target}Up", run) == f"ttH_sm_{target}Up"
        assert not target.startswith("CMS_TOP26006_")


def test_standardizer_rejects_stale_tau_template_and_accepts_native_name():
    native_name = "CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2016"
    assert remap_name(f"ttH_sm_{native_name}Up", "run2") == f"ttH_sm_{native_name}Up"
    with pytest.raises(ValueError, match="regenerate"):
        remap_name(
            "ttH_sm_CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2016APVUp",
            "run2",
        )


@pytest.mark.parametrize("run,domain", (("run2", "13TeV"), ("run3", "13p6TeV")))
def test_standardizer_writes_final_lepton_names_in_cards_and_templates(tmp_path, run, domain):
    source = "lepSF_elec_non_mva"
    old_name = f"ttH_sm_{source}Up"
    target = f"{source}_{domain}"
    card = tmp_path / "ttx_multileptons-demo.txt"
    card.write_text(f"{source} shape 1\n")
    template = tmp_path / "ttx_multileptons-demo.root"
    histogram = hist.Hist(hist.axis.Regular(1, 0, 1), storage=hist.storage.Weight())
    histogram.fill(0.5)
    with uproot.recreate(template) as root_file:
        root_file[old_name] = histogram
    output = tmp_path / "output"
    process_cards([str(card)], run, str(output))
    process_templates([str(template)], run, str(output))
    final_card = output / run / f"ttx_multileptons-{run}_demo.txt"
    final_template = output / run / f"ttx_multileptons-{run}_demo.root"
    assert target in final_card.read_text()
    assert source + " shape" not in final_card.read_text()
    with uproot.open(final_template) as root_file:
        assert f"ttH_sm_{target}Up" in root_file.keys(cycle=False)
        assert old_name not in root_file.keys(cycle=False)


def test_stale_tau_card_fails_before_output_is_created(tmp_path):
    source = "CMS_scale_t_DeepTau2017v2p1_DM0_genTau_2016APV"
    card = tmp_path / "ttx_multileptons-stale.txt"
    card.write_text(f"{source} shape 1\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="regenerate"):
        process_cards([str(card)], "run2", str(output))
    assert not output.exists()


@pytest.mark.parametrize(
    "year,domain",
    (("UL18", "13TeV"), ("2022", "13p6TeV")),
)
def test_raw_card_writer_uses_lepton_run_domain_names(tmp_path, year, domain):
    sources = (
        "lepSF_elec_mva", "lepSF_muon_mva",
        "lepSF_elec_non_mva", "lepSF_muon_non_mva",
    )
    histogram = HistEFT(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0, 1, name="njets"),
        wc_names=[], label="Events",
    )
    for systematic, value in [("nominal", 10.0)] + [
        (f"{source}{direction}", 11.0 if direction == "Up" else 9.0)
        for source in (*sources, "JES_Absolute")
        for direction in ("Up", "Down")
    ]:
        histogram.fill(
            process=f"ttH_central{year}", channel="3l_onZ_1b",
            systematic=systematic, njets=np.asarray([0.5]),
            weight=np.asarray([value]),
        )
    maker = DatacardMaker(
        hists={"njets": histogram}, year_lst=[year],
        out_dir=str(tmp_path), var_lst=["njets"], do_nuisance=True,
        skip_missing_parton_rate_syst=True, verbose=False,
    )
    maker.analyze("njets", "3l_onZ_1b", {"ttH": []}, True, {})
    stem = tmp_path / "ttx_multileptons-3l_onZ_1b_njets"
    card_text = stem.with_suffix(".txt").read_text()
    shape_names = {
        words[0] for words in (line.split() for line in card_text.splitlines())
        if len(words) >= 2 and words[1] == "shape"
    }
    expected = {f"{source}_{domain}" for source in sources}
    assert expected | {"JES_Absolute"} <= shape_names
    assert not set(sources) & shape_names
    with uproot.open(stem.with_suffix(".root")) as root_file:
        root_names = set(root_file.keys(cycle=False))
        for source in sources:
            target = f"{source}_{domain}"
            for direction, value in (("Up", 11.0), ("Down", 9.0)):
                key = f"ttH_sm_{target}{direction}"
                assert key in root_names
                assert f"ttH_sm_{source}{direction}" not in root_names
                assert np.sum(root_file[key].values()) == pytest.approx(value)
                assert np.count_nonzero(root_file[key].values()) == 1
        for direction, value in (("Up", 11.0), ("Down", 9.0)):
            assert np.sum(root_file[f"ttH_sm_JES_Absolute{direction}"].values()) == pytest.approx(value)
