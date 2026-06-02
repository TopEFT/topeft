from pathlib import Path
from types import SimpleNamespace

import pytest

from topeft.modules.corrections import (
    ApplyMETSystematics,
    get_corr_t1_met_jets,
    get_jerc_keys,
    get_selected_met,
    get_selected_raw_met,
    get_supported_met_systematics,
    is_met_unclustered_systematic,
    use_type1_met,
)


def _processor_source():
    return (
        Path(__file__).parents[1]
        / "analysis"
        / "topeft_run2"
        / "analysis_processor.py"
    ).read_text()


def _processor_type1_block():
    source = _processor_source()
    return source[
        source.index("# Build the Type-1 MET correction"):
        source.index("# Loop over the list of systematic variations")
    ]


@pytest.mark.parametrize(
    "year",
    ["2016APV", "2016", "2017", "2018", "2022", "2022EE", "2023", "2023BPix"],
)
def test_type1_met_policy_covers_supported_run2_and_run3_years(year):
    assert use_type1_met(year)


def test_type1_met_policy_rejects_unknown_years():
    assert not use_type1_met("2024")
    assert not use_type1_met("unknown")


def test_run2_selected_met_uses_legacy_met():
    events = SimpleNamespace(MET=object(), PuppiMET=object())

    assert get_selected_met(events, "2018") is events.MET


def test_run3_selected_met_uses_puppimet():
    events = SimpleNamespace(MET=object(), PuppiMET=object())

    assert get_selected_met(events, "2022") is events.PuppiMET


def test_run3_selected_met_missing_puppimet_fails_clearly():
    events = SimpleNamespace(MET=object())

    with pytest.raises(RuntimeError, match="requires events.PuppiMET"):
        get_selected_met(events, "2022")


def test_run3_type1_raw_met_policy_uses_raw_puppimet():
    events = SimpleNamespace(MET=object(), PuppiMET=object(), RawPuppiMET=object())

    assert use_type1_met("2022")
    assert get_selected_met(events, "2022") is events.PuppiMET
    assert get_selected_raw_met(events, "2022") is events.RawPuppiMET


def test_run2_type1_raw_met_policy_uses_raw_met():
    events = SimpleNamespace(MET=object(), RawMET=object(), PuppiMET=object())

    assert use_type1_met("2018")
    assert get_selected_met(events, "2018") is events.MET
    assert get_selected_raw_met(events, "2018") is events.RawMET


def test_run2_type1_raw_met_missing_rawmet_fails_clearly():
    events = SimpleNamespace(MET=object(), PuppiMET=object())

    with pytest.raises(RuntimeError, match="requires events.RawMET"):
        get_selected_raw_met(events, "2018")


def test_run3_type1_raw_met_missing_rawpuppimet_fails_clearly():
    events = SimpleNamespace(MET=object(), PuppiMET=object())

    with pytest.raises(RuntimeError, match="requires events.RawPuppiMET"):
        get_selected_raw_met(events, "2022")


@pytest.mark.parametrize("year", ["2018", "2022"])
def test_type1_requires_corr_t1_met_jet_collection(year):
    corr_t1 = object()
    events = SimpleNamespace(CorrT1METJet=corr_t1)

    assert get_corr_t1_met_jets(events, year) is corr_t1

    with pytest.raises(RuntimeError, match="requires events.CorrT1METJet"):
        get_corr_t1_met_jets(SimpleNamespace(), year)


@pytest.mark.parametrize("year", ["2016APV", "2016", "2017", "2018"])
def test_run2_type1_mc_jec_levels_use_full_l1l2l3(year):
    _, _, levels, _, _ = get_jerc_keys(year, isdata=False, corr_type="type1_met")

    assert levels == ["L1FastJet", "L2Relative", "L3Absolute"]


@pytest.mark.parametrize(
    ("year", "era"),
    [("2016APV", "B"), ("2016", "F"), ("2017", "B"), ("2018", "A")],
)
def test_run2_type1_data_jec_levels_include_residuals(year, era):
    _, _, levels, _, _ = get_jerc_keys(year, isdata=True, era=era, corr_type="type1_met")

    assert levels == ["L1FastJet", "L2Relative", "L3Absolute", "L2L3Residual"]


@pytest.mark.parametrize("year", ["2016APV", "2016", "2017", "2018"])
def test_regular_run2_analysis_jet_jec_levels_are_unchanged(year):
    _, _, levels, _, _ = get_jerc_keys(year, isdata=False, corr_type="jets")

    assert levels == ["L1FastJet", "L2Relative"]


def test_run3_type1_jec_levels_are_unchanged():
    _, _, mc_levels, _, _ = get_jerc_keys("2022", isdata=False, corr_type="type1_met")
    _, _, data_levels, _, _ = get_jerc_keys("2022", isdata=True, era="C", corr_type="type1_met")

    assert mc_levels == ["L1FastJet", "L2Relative", "L3Absolute", "L2L3Residual"]
    assert data_levels == ["L1FastJet", "L2Relative", "L3Absolute", "L2L3Residual"]


def test_met_unclustered_systematics_are_public_generic_labels():
    assert get_supported_met_systematics("2022", isData=False) == [
        "MET_UnclusteredEnergyUp",
        "MET_UnclusteredEnergyDown",
    ]
    assert get_supported_met_systematics("2022", isData=True) == []
    assert is_met_unclustered_systematic("MET_UnclusteredEnergyUp")
    assert is_met_unclustered_systematic("MET_UnclusteredEnergyDown")
    assert not is_met_unclustered_systematic("JER_2022Up")


def test_apply_met_systematics_selects_unclustered_shift_only():
    nominal = object()
    up = object()
    down = object()
    met = SimpleNamespace(
        MET_UnclusteredEnergy=SimpleNamespace(up=up, down=down)
    )

    assert ApplyMETSystematics(met, "nominal") is met
    assert ApplyMETSystematics(met, "MET_UnclusteredEnergyUp") is up
    assert ApplyMETSystematics(met, "MET_UnclusteredEnergyDown") is down
    assert ApplyMETSystematics(nominal, "JER_2022Up") is nominal


def test_apply_met_systematics_selects_type1_jet_variations_when_present():
    nominal = object()
    jer_up = object()
    jer_down = object()
    jes_up = object()
    jes_down = object()
    met = SimpleNamespace(
        JER=SimpleNamespace(up=jer_up, down=jer_down),
        JES_Total=SimpleNamespace(up=jes_up, down=jes_down),
    )

    assert ApplyMETSystematics(met, "nominal") is met
    assert ApplyMETSystematics(met, "JER_2022Up") is jer_up
    assert ApplyMETSystematics(met, "JER_2022Down") is jer_down
    assert ApplyMETSystematics(met, "JES_TotalUp") is jes_up
    assert ApplyMETSystematics(met, "JES_TotalDown") is jes_down
    assert ApplyMETSystematics(nominal, "JES_TotalUp") is nominal


def test_processor_type1_met_does_not_precorrect_temporary_jets():
    block = _processor_type1_block()

    assert "type1Jets = copy.copy" not in block
    assert "corrT1METJets = copy.copy" not in block
    assert "ak.with_field" in block
    assert "corr_type='jets'" not in block
    assert "corr_type='type1_met'" in block


def test_processor_type1_met_passes_full_jets_corrt1_and_correction_options():
    block = _processor_type1_block()

    assert "type1Jets = jets" in block
    assert "corrT1METJets = get_corr_t1_met_jets(events, year)" in block
    assert "CorrT1METJet has no per-object rho branch" in block
    assert "ak.broadcast_arrays(jetsRho, corrT1METJets.rawPt)[0]" in block
    assert "met,\n                raw_met,\n                type1Jets,\n                corrT1METJets" in block
    assert "suppress_forward_eta_stochastic_jer=effective_suppress_forward_eta_stochastic_jer" in block
    assert "del type1Jets" in block
    assert "del corrT1METJets" in block


def test_processor_type1_met_build_policy_is_not_run3_only():
    block = _processor_type1_block()

    assert "if use_type1_met(year):" in block
    assert "use_run3_type1_met" not in block


def test_processor_keeps_cleaned_analysis_jets_separate():
    source = _processor_source()

    assert "cleanedJets = jets[~ak.any(tmp.slot0 == tmp.slot1, axis=-1)]" in source
    assert "cleanedJets = ApplyJetCorrections(" in source
    assert "goodJets = cleanedJets[cleanedJets.isGood]" in source
    assert "fwdJets  = cleanedJets[cleanedJets.isFwd]" in source


def test_processor_keeps_public_met_and_lt_semantics():
    processor_source = _processor_source()

    assert "type1_met = ApplyJetCorrections(" in processor_source
    assert "corr_type='type1_met'" in processor_source
    assert "met = ApplyMETSystematics(type1_met, syst_var)" in processor_source
    assert "lt = ak.sum(l_fo_conept_sorted_padded.pt, axis=-1) + met.pt" in processor_source
    assert 'varnames["met"]     = met.pt' in processor_source
