from topcoffea.modules.CorrectedJetsFactory import get_jec_uncertainty_label
from topeft.modules import corrections as cor


def _jes_bases_from_variations(variations):
    bases = []
    for syst_var in variations:
        if not syst_var.startswith("JES_"):
            continue
        if syst_var.endswith("Up"):
            bases.append(syst_var[len("JES_") : -len("Up")])
        elif syst_var.endswith("Down"):
            bases.append(syst_var[len("JES_") : -len("Down")])
    return sorted(set(bases))


def _expected_jes_bases_from_jecstack(year):
    jet_algo, jec_tag, _, _, junc_types = cor.get_jerc_keys(year, isdata=False, era=None)
    bases = []
    for junc_type in junc_types:
        full_unc_name = f"{jec_tag}_{junc_type}_{jet_algo}"
        bases.append(get_jec_uncertainty_label(full_unc_name, jec_tag, jet_algo))
    return bases


def test_run2_ul17_jes_contract_matches_fields():
    variations = cor.get_supported_jet_systematics("2017", isData=False, era=None)
    producer_bases = _jes_bases_from_variations(variations)
    field_bases = _expected_jes_bases_from_jecstack("2017")
    assert producer_bases == sorted(set(field_bases))
    assert len(field_bases) == len(set(field_bases))
    assert "BBEC1" in producer_bases
    assert "Regrouped_BBEC1" not in producer_bases


def test_run3_2022_jes_contract_matches_fields_without_collisions():
    variations = cor.get_supported_jet_systematics("2022", isData=False, era=None)
    producer_bases = _jes_bases_from_variations(variations)
    field_bases = _expected_jes_bases_from_jecstack("2022")
    assert producer_bases == sorted(set(field_bases))
    assert len(field_bases) == len(set(field_bases))
    assert "Regrouped_Absolute_2022" in producer_bases
    assert "Regrouped_BBEC1_2022" in producer_bases
    assert "2022" not in producer_bases


def test_run3_2022ee_jes_tokens_use_regrouped_names():
    variations = cor.get_supported_jet_systematics("2022EE", isData=False, era=None)
    producer_bases = _jes_bases_from_variations(variations)
    field_bases = _expected_jes_bases_from_jecstack("2022EE")
    assert producer_bases == sorted(set(field_bases))
    assert len(field_bases) == len(set(field_bases))
    assert "JES_Regrouped_Absolute_2022EEUp" in variations
    assert "JES_Absolute_2022EEUp" not in variations
    assert "Regrouped_Absolute_2022EE" in producer_bases
    assert "Absolute_2022EE" not in producer_bases


def test_run3_2023_uses_regrouped_sources_without_total_tokens():
    variations = cor.get_supported_jet_systematics("2023", isData=False, era=None)
    producer_bases = _jes_bases_from_variations(variations)
    field_bases = _expected_jes_bases_from_jecstack("2023")
    assert producer_bases == sorted(set(field_bases))
    assert len(field_bases) == len(set(field_bases))
    assert "Regrouped_Absolute_2023" in producer_bases
    assert "Total" not in producer_bases
    assert "Regrouped_Total" not in producer_bases
