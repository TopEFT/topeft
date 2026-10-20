"""Small common-domain checks against live card definitions."""

import numpy as np
import pytest
from topcoffea.modules.utils import regex_match

from topeft.modules import plot_uncertainty_contract as contract
from topeft.modules.datacard_tools import (
    DatacardMaker, run2_luminosity_shape_factor,
    run3_luminosity_shape_factor,
)


@pytest.mark.parametrize("process, expected", [
    ("TTTo2L2Nu_centralUL18", "ttll"),
    ("ttllJet_privateUL16", "ttll"),
    ("WZTo3LNu_central2022", "Diboson"),
    ("TTGJets_central2023", "convs"),
    ("DY50_centralUL18", "DY50"),
    ("TTto4Q_central2023BPix", "TTto4Q"),
])
def test_card_group_identity(process, expected):
    assert contract.card_process_name(process) == expected
    assert expected in {
        *[group.rstrip("_") for group in DatacardMaker.GROUP],
        "DY50", "TTto4Q",
    }


@pytest.mark.parametrize("process, base, expected", [
    ("TTTo2L2Nu_centralUL18", "ISR", "ISR_gg"),
    ("ttllJet_privateUL17", "ISR", "ISR_gg"),
    ("WZTo3LNu_central2022", "ISR", "ISR_qq"),
    ("tHq_private2023", "ISR", "ISR_qg"),
    ("TTTo2L2Nu_centralUL18", "renorm", "renorm_ttll"),
    ("ttllJet_privateUL17", "fact", "fact_ttll"),
    ("TTto4Q_central2022", "renorm", "renorm_TTto4Q"),
    ("ttH_private2023", "TES", "TES_run3"),
])
def test_shape_identity_matches_card_rules(process, base, expected):
    assert contract.shape_nuisance_name(base, process) == expected


def test_isr_partitions_are_card_instance_definitions():
    card = DatacardMaker(hists={}, year_lst=["2022"], do_nuisance=False,
                         skip_missing_parton_rate_syst=True, verbose=False)
    for process in ("ttll_private2022", "WZTo3LNu_central2022",
                    "tHq_private2022", "TTto4Q_central2022"):
        card_name = contract.card_process_name(process)
        groups = [rule["group"] for rule in card.syst_shape_decorrelate["ISR"]
                  if regex_match([card_name], rule["matches"])]
        assert len(groups) <= 1
        expected = "ISR_" + groups[0] if groups else "ISR"
        assert contract.shape_nuisance_name("ISR", process) == expected


def test_tau_year_correlation_follows_card_configuration():
    card = DatacardMaker(hists={}, year_lst=["2022"], do_nuisance=False,
                         skip_missing_parton_rate_syst=True, verbose=False)
    run2_base = "CMS_scale_t_DeepTau2017v2p1_DM0_genTau"
    assert card.syst_year_corr[run2_base]["2016APV"] == ["2016"]
    assert contract.shape_nuisance_name(
        run2_base + "_2016APV", "ttllJet_privateUL16APV",
    ) == contract.shape_nuisance_name(
        run2_base + "_2016", "ttllJet_privateUL16",
    )
    run3_base = "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4"
    assert card.syst_year_corr[run3_base]["2022EE"] == ["2022"]
    assert contract.shape_nuisance_name(
        run3_base + "_2022EE", "ttH_private2022EE",
    ) == contract.shape_nuisance_name(
        run3_base + "_2022", "ttH_private2022",
    )


def test_copied_card_identity_tables_match_the_live_card():
    card = DatacardMaker(hists={}, year_lst=["2022"], do_nuisance=False,
                         skip_missing_parton_rate_syst=True, verbose=False)
    assert set(contract.RUN_DECORRELATE) == set(card.run_decorrelate)
    assert contract.YEAR_CORRELATE_2016 == {
        base for base, years in card.syst_year_corr.items()
        if "2016APV" in years.get("2016", ())
    }
    assert contract.YEAR_CORRELATE_RUN3 == {
        base for base, years in card.syst_year_corr.items()
        if "2022EE" in years.get("2022", ())
    }
    for base, rules in card.syst_shape_decorrelate.items():
        for process in ("ttll_private2022", "WZTo3LNu_central2022",
                        "tHq_private2022", "TTto4Q_central2022"):
            card_name = contract.card_process_name(process)
            groups = [rule["group"] for rule in rules
                      if regex_match([card_name], rule["matches"])]
            assert len(groups) <= 1
            expected = base if not groups else (
                f"{base}_{card_name}" if groups == [""] else f"{base}_{groups[0]}"
            )
            assert contract.shape_nuisance_name(base, process) == expected


def test_two_sided_coherent_response_before_quadrature():
    nominal = np.array([10.0, 0.0])
    responses = [
        ("ISR_gg", np.array([-3.0, 0]), np.array([3.0, 0])),
        ("ISR_gg", np.array([2.0, 0]), np.array([-2.0, 0])),
        ("ISR_qq", np.array([4.0, 0]), np.array([-1.0, 0])),
    ]
    low, high = contract.combine_responses(responses, nominal)
    np.testing.assert_allclose(low, [2.0, 0.0])
    np.testing.assert_allclose(high, [17.0, 0.0])
    asymmetric = [("qcd_scale_ttll", np.array([-1.02]), np.array([1.17]))]
    low, high = contract.combine_responses(asymmetric, np.array([10.0]))
    np.testing.assert_allclose(low, [1.02 ** 2])
    np.testing.assert_allclose(high, [1.17 ** 2])


@pytest.mark.parametrize("process, nuisance, pair", [
    ("TTTo2L2Nu_centralUL18", "qcd_scale_ttll", (0.898, 1.117)),
    ("ttll_private2022", "qcd_scale_ttll", (0.898, 1.117)),
    ("tHq_private2023", "pdf_scale_qg", (1 / 1.051, 1.051)),
    ("ttH_privateUL18", "qcd_scale_ttH", (0.929, 1.076)),
])
def test_rate_factors_are_card_factors(process, nuisance, pair):
    responses = contract.rate_responses(process, np.array([10.0]))
    down, up = responses[nuisance]
    np.testing.assert_allclose(down, [10 * (pair[0] - 1)])
    np.testing.assert_allclose(up, [10 * (pair[1] - 1)])


@pytest.mark.parametrize("process, nuisance, factor", [
    ("TTTo2L2Nu_centralUL18", "lumi_13TeV_15161718_l", run2_luminosity_shape_factor),
    ("TTto4Q_central2022EE", "lumi_1", run3_luminosity_shape_factor),
])
def test_luminosity_uses_card_annual_components(process, nuisance, factor):
    year = DatacardMaker.get_year(process)
    down, up = contract.luminosity_responses(process, np.array([10.0]))[nuisance]
    np.testing.assert_allclose(down, [10 * (factor(nuisance, year, "Down") - 1)])
    np.testing.assert_allclose(up, [10 * (factor(nuisance, year, "Up") - 1)])
    assert contract.luminosity_responses("nonpromptUL18", np.array([10.0])) == {}
    assert contract.rate_responses("dataUL18", np.array([10.0])) == {}


def test_auxiliary_mapping_has_no_invented_rate():
    assert contract.card_process_name("DY50_centralUL18") == "DY50"
    assert contract.rate_responses("DY50_centralUL18", np.array([4.0])) == {}
    assert contract.card_process_name("TTto4Q_central2022") == "TTto4Q"
    assert contract.rate_responses("TTto4Q_central2022", np.array([4.0])) == {}
    for process, nuisance in (("flipsUL18", "charge_flips"),
                              ("flips2022", "charge_flips_run3")):
        down, up = contract.rate_responses(process, np.array([10.0]))[nuisance]
        np.testing.assert_allclose([down[0], up[0]],
                                   [10 * (1 / 1.3 - 1), 3.0])
        assert contract.luminosity_responses(process, np.array([10.0])) == {}


def test_absent_auxiliary_rate_is_distinct_from_unknown_identity():
    process = "DY50_central2022"
    assert contract.rate_responses(process, np.array([4.0])) == {}
    assert contract.luminosity_responses(process, np.array([4.0]))
    with pytest.raises(ValueError, match="Unknown SR card process identity"):
        contract.rate_responses("mystery_central2022", np.array([4.0]))
    with pytest.raises(ValueError, match="no era"):
        contract.rate_responses("mystery", np.array([4.0]))
