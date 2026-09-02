from pathlib import Path

import numpy as np
import pytest

hist = pytest.importorskip("hist")
uproot = pytest.importorskip("uproot")

from topcoffea.modules.histEFT import HistEFT
from topeft.modules.corrections import (
    get_supported_tau_energy_systematics,
    get_tau_eta_variation_specs,
    get_tau_vsjet_variation_specs,
    get_tau_weight_variation_names,
)
from topeft.modules.datacard_tools import (
    DatacardMaker,
    resolve_shape_nuisance_identity,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSOR_SOURCE = REPO_ROOT / "analysis/topeft_run2/analysis_processor.py"

SELECTED_DMS = (0, 1, 10, 11)
RUN3_PAYLOAD_ERAS = ("2022", "2022EE", "2023", "2023BPix")
RUN3_VSE_ETA_TOKENS = (
    "abseta0to1p46",
    "abseta1p46to1p56",
    "abseta1p56to2p5",
)


def _expected_vsjet_names(tagger, year, run):
    names = {
        f"CMS_eff_t_{tagger}_VSjet_dm_{stat}_DM{dm}_{year}"
        for stat in ("stat1", "stat2")
        for dm in SELECTED_DMS
    }
    names.add(f"CMS_eff_t_{tagger}_VSjet_dm_syst_{year}")
    names.update(
        f"CMS_eff_t_{tagger}_VSjet_dm_syst_DM{dm}_{year}"
        for dm in SELECTED_DMS
    )
    names.add(f"CMS_eff_t_{tagger}_VSjet_dm_syst_alleras_{run}")
    return names


@pytest.mark.parametrize(
    ("year", "tagger", "run"),
    [
        ("2016APV", "DeepTau2017v2p1", "Run2"),
        ("2016", "DeepTau2017v2p1", "Run2"),
        ("2017", "DeepTau2017v2p1", "Run2"),
        ("2018", "DeepTau2017v2p1", "Run2"),
        ("2022", "DeepTau2018v2p5", "Run3"),
        ("2022EE", "DeepTau2018v2p5", "Run3"),
        ("2023", "DeepTau2018v2p5", "Run3"),
        ("2023BPix", "DeepTau2018v2p5", "Run3"),
    ],
)
def test_vsjet_exact_five_component_identity_set(year, tagger, run):
    observed = set(get_tau_vsjet_variation_specs(year))
    assert observed == _expected_vsjet_names(tagger, year, run)
    assert len(observed) == 14
    assert not any(name in observed for name in ("lepSF_taus_real", "TES", "FES"))


def test_vsjet_payload_component_spelling_maps_to_canonical_names():
    run2 = get_tau_vsjet_variation_specs("2018")
    run3 = get_tau_vsjet_variation_specs("2022")
    assert run2[
        "CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_DM10_2018"
    ] == "syst_2018_dm10_{direction}"
    assert run3[
        "CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_DM10_2022"
    ] == "syst_TES_2022_preEE_dm10_{direction}"
    assert run2[
        "CMS_eff_t_DeepTau2017v2p1_VSjet_dm_syst_alleras_Run2"
    ] == "syst_alleras_{direction}"
    assert run3[
        "CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_alleras_Run3"
    ] == "syst_alleras_{direction}"


@pytest.mark.parametrize(
    ("year", "tagger", "period", "vse_tokens", "vsmu_tokens"),
    [
        (
            "2018",
            "DeepTau2017v2p1",
            "2018",
            ("abseta0to1p5", "abseta1p5to2p3"),
            (
                "abseta0to0p4",
                "abseta0p4to0p8",
                "abseta0p8to1p2",
                "abseta1p2to1p7",
                "abseta1p7to2p3",
            ),
        ),
        (
            "2022EE",
            "DeepTau2018v2p5",
            "2022",
            ("abseta0to1p46", "abseta1p46to1p56", "abseta1p56to2p5"),
            (
                "abseta0to0p4",
                "abseta0p4to0p8",
                "abseta0p8to1p2",
                "abseta1p2to1p7",
                "abseta1p7to2p4",
            ),
        ),
        (
            "2023BPix",
            "DeepTau2018v2p5",
            "2023",
            ("abseta0to1p46", "abseta1p46to1p56", "abseta1p56to2p5"),
            (
                "abseta0to0p4",
                "abseta0p4to0p8",
                "abseta0p8to1p2",
                "abseta1p2to1p7",
                "abseta1p7to2p4",
            ),
        ),
    ],
)
def test_eta_nuisance_names_have_exact_tokens_and_periods(
    year, tagger, period, vse_tokens, vsmu_tokens
):
    observed = get_tau_eta_variation_specs(year)
    if year.startswith("201"):
        expected = {
            f"CMS_fake_t_{tagger}_VSe_{token}_{period}" for token in vse_tokens
        }
    else:
        expected = {
            f"CMS_fake_t_{tagger}_VSe_DM{dm}_{token}_{year}"
            for token in vse_tokens
            for dm in SELECTED_DMS
        }
    expected.update(
        f"CMS_fake_t_{tagger}_VSmu_{token}_{period}" for token in vsmu_tokens
    )
    assert set(observed) == expected
    assert len(observed) == len(set(observed))
    assert all(not name.endswith(("Up", "Down")) for name in observed)


@pytest.mark.parametrize("year", RUN3_PAYLOAD_ERAS)
def test_run3_vse_exact_payload_era_eta_dm_identity_partition(year):
    expected = {
        f"CMS_fake_t_DeepTau2018v2p5_VSe_DM{dm}_{eta_token}_{year}"
        for eta_token in RUN3_VSE_ETA_TOKENS
        for dm in SELECTED_DMS
    }
    observed = {
        name
        for name, spec in get_tau_eta_variation_specs(year).items()
        if spec["source"] == "VSe"
    }
    superseded = {
        f"CMS_fake_t_DeepTau2018v2p5_VSe_{eta_token}_{year[:4]}"
        for eta_token in RUN3_VSE_ETA_TOKENS
    }

    assert observed == expected
    assert len(observed) == len(RUN3_VSE_ETA_TOKENS) * len(SELECTED_DMS)
    assert observed.isdisjoint(superseded)


def test_run3_vse_payload_era_identities_do_not_collapse_suberas():
    identities_by_era = {
        year: {
            name
            for name, spec in get_tau_eta_variation_specs(year).items()
            if spec["source"] == "VSe"
        }
        for year in RUN3_PAYLOAD_ERAS
    }

    assert identities_by_era["2022"].isdisjoint(identities_by_era["2022EE"])
    assert identities_by_era["2023"].isdisjoint(identities_by_era["2023BPix"])


def test_eta_masks_keep_run2_vse_nuisance_bins_distinct_from_payload_edges():
    specs = get_tau_eta_variation_specs("2018")
    vse = [spec for spec in specs.values() if spec["source"] == "VSe"]
    assert [(spec["low"], spec["high"]) for spec in vse] == [
        (0.0, 1.5),
        (1.5, 2.3),
    ]
    assert (1.46, 1.558) not in {
        (spec["low"], spec["high"]) for spec in vse
    }


@pytest.mark.parametrize(
    ("year", "tagger", "run", "electron_dms"),
    [
        ("2018", "DeepTau2017v2p1", "Run2", (0, 1)),
        ("2022EE", "DeepTau2018v2p5", "Run3", SELECTED_DMS),
    ],
)
def test_energy_nuisance_exact_source_dm_period_set(year, tagger, run, electron_dms):
    expected_bases = {
        f"CMS_scale_t_{tagger}_DM{dm}_genTau_{year}" for dm in SELECTED_DMS
    }
    expected_bases.update(
        f"CMS_scale_t_{tagger}_DM{dm}_genElectron_{year}"
        for dm in electron_dms
    )
    expected_bases.update(
        f"CMS_scale_t_{tagger}_DM{dm}_genMuon_{run}" for dm in SELECTED_DMS
    )
    expected = {
        f"{base}{direction}"
        for base in expected_bases
        for direction in ("Up", "Down")
    }
    observed = set(get_supported_tau_energy_systematics(year, isData=False))
    assert observed == expected
    assert len(observed) == len(expected)


def test_weight_nuisance_set_is_unique_and_has_only_independent_families():
    observed = get_tau_weight_variation_names("2022", include_jet_fake=True)
    assert len(observed) == len(set(observed)) == 32
    assert observed[-1] == "lepSF_taus_fake_run3"
    assert "lepSF_taus_real" not in observed
    assert "lepSF_taus_fake" not in observed


@pytest.mark.parametrize(
    "base",
    [
        "CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat1_DM0_2022",
        "CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2022",
        "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4_2022",
        "CMS_scale_t_DeepTau2018v2p5_DM10_genMuon_Run3",
        "lepSF_taus_fake_run3",
    ],
)
def test_canonical_tau_up_down_map_to_one_unsuffixed_final_base(base):
    up = resolve_shape_nuisance_identity(f"{base}Up", "_run3", ["TES", "FES"])
    down = resolve_shape_nuisance_identity(
        f"{base}Down", "_run3", ["TES", "FES"]
    )
    assert up == (base, f"{base}Up")
    assert down == (base, f"{base}Down")


def test_unrelated_legacy_run_decorrelation_mapping_is_unchanged():
    assert resolve_shape_nuisance_identity(
        "lepSF_muonUp", "_run3", ["lepSF_muon"]
    ) == ("lepSF_muon_run3", "lepSF_muon_run3Up")


def _axes():
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name="njets"),
    )


def test_datacard_readback_preserves_exact_tau_shape_identities(tmp_path):
    bases = (
        "CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_alleras_Run3",
        "CMS_fake_t_DeepTau2018v2p5_VSe_DM0_abseta0to1p46_2022",
        "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4_2022",
        "CMS_scale_t_DeepTau2018v2p5_DM10_genMuon_Run3",
        "lepSF_taus_fake_run3",
    )
    histogram = HistEFT(*_axes(), wc_names=[], label="Events")
    for systematic, weight in [("nominal", 10.0)] + [
        (f"{base}{direction}", 11.0 if direction == "Up" else 9.0)
        for base in bases
        for direction in ("Up", "Down")
    ]:
        histogram.fill(
            process="ttH_central2022",
            channel="3l_onZ_1b",
            systematic=systematic,
            njets=np.asarray([0.5]),
            weight=np.asarray([weight]),
        )
    maker = DatacardMaker(
        hists={"njets": histogram},
        out_dir=str(tmp_path),
        var_lst=["njets"],
        do_nuisance=True,
        skip_missing_parton_rate_syst=True,
        verbose=False,
    )
    maker.analyze("njets", "3l_onZ_1b", {"ttH": []}, True, {})
    stem = tmp_path / "ttx_multileptons-3l_onZ_1b_njets"
    card_text = stem.with_suffix(".txt").read_text(encoding="utf-8")
    shape_bases = {
        row.split()[0]
        for row in card_text.splitlines()
        if len(row.split()) >= 2 and row.split()[1] == "shape"
    }
    assert shape_bases == set(bases)
    with uproot.open(stem.with_suffix(".root")) as root_file:
        root_names = set(root_file.keys(cycle=False))
    for base in bases:
        assert f"ttH_sm_{base}Up" in root_names
        assert f"ttH_sm_{base}Down" in root_names
    assert not any("_run3_run3" in name for name in root_names)
    assert "TES_run3" not in card_text
    assert "FES_run3" not in card_text
    assert "lepSF_taus_real_run3" not in card_text


def test_processor_emits_canonical_tau_families_without_legacy_aggregates():
    source = PROCESSOR_SOURCE.read_text(encoding="utf-8")
    assert "get_tau_weight_variation_names" in source
    assert "get_supported_tau_energy_systematics" in source
    assert "lepSF_taus_realUp" not in source
    assert "lepSF_taus_realDown" not in source
    assert '"TESUp"' not in source
    assert '"FESUp"' not in source
