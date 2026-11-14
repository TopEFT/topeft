"""Small common-domain checks against live card definitions."""

import numpy as np
import pytest
from topcoffea.modules.utils import regex_match

from topeft.modules import plot_uncertainty_contract as contract
from topeft.modules.datacard_tools import (
    DatacardMaker, run2_luminosity_shape_factor,
    run3_luminosity_shape_factor,
    resolve_missing_parton_response_index,
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


@pytest.mark.parametrize("run, process, channel, jet_count, high", [
    ("run2", "WZTo3LNu_centralUL18", "3l_onZ_1b_4j", 4, 1.77),
    ("run3", "WZTo3LNu_central2022", "3l_onZ_1b_4j", 4, 1.43),
    ("run2", "WZTo3LNu_centralUL18", "2los_onZ_1tau_5j", 5, 2.05),
    ("run3", "WZTo3LNu_central2022", "2los_onZ_1tau_5j", 5, 1.72),
])
def test_non_njets_diboson_rate_uses_card_jet_factor(run, process, channel, jet_count, high):
    card_factor = contract._card_rates(run, True)["diboson_njets"].get_process("Diboson", jet_count)
    low_text, high_text = card_factor.split("/")
    assert float(high_text) == high
    low, up = contract.rate_responses(
        process, np.array([10.0]), channel=channel, variable="lj0pt",
    )["diboson_njets"]
    np.testing.assert_allclose(low, [10 * (float(low_text) - 1)])
    np.testing.assert_allclose(up, [10 * (high - 1)])
    assert "diboson_njets" not in contract.rate_responses(process, np.array([10.0]))


@pytest.mark.parametrize("run, year", [("run2", "UL18"), ("run3", "2022")])
@pytest.mark.parametrize("card_name", ["tllq", "tHq"])
@pytest.mark.parametrize("channel, response_index", [
    ("2los_onZ_1tau_5j", 3),
    ("2los_onZ_1tau_3j", 3),
    ("3l_onZ_1b_3j", 3),
    ("3l_onZ_1b_4j", 4),
    ("3l_onZ_1b_5j", 5),
])
def test_non_njets_missing_parton_uses_card_channel_and_jet(
    run, year, card_name, channel, response_index,
):
    process = f"{card_name}_private{year}"
    card = object.__new__(DatacardMaker)
    card.do_nuisance = True
    card.skip_missing_parton_rate_syst = False
    card.year_lst = [year]
    payload = card.resolve_missing_parton_payload_path(card.year_lst)
    rates = card.load_systematics(f"params/rate_systs_{run}.json", payload)
    channel_key = channel.rsplit("_", 1)[0]
    channel_rates = rates["missing_parton"].get_process(card_name)[channel_key]
    if channel == "2los_onZ_1tau_5j":
        assert len(channel_rates) == 4  # No stored index 5 to fall back to.
    factor_high = float(channel_rates[response_index])
    factor_low = max(0.01, 2 - factor_high)
    expected_low = float(f"{factor_low:.6f}")
    expected_high = float(f"{factor_high:.6f}")
    down, up = contract.rate_responses(
        process, np.array([7.0]), channel=channel, variable="ptz",
    )["missing_parton"]
    np.testing.assert_allclose(down, [7 * (expected_low - 1)])
    np.testing.assert_allclose(up, [7 * (expected_high - 1)])
    assert "missing_parton" not in contract.rate_responses(process, np.array([7.0]))


def test_missing_parton_adapter_calls_maintained_resolver(monkeypatch):
    assert contract.resolve_missing_parton_response_index is resolve_missing_parton_response_index
    calls = []

    def record_resolution(channel, natural_index):
        calls.append((channel, natural_index))
        return resolve_missing_parton_response_index(channel, natural_index)

    monkeypatch.setattr(contract, "resolve_missing_parton_response_index", record_resolution)
    contract.rate_responses(
        "tllq_privateUL18", np.array([7.0]),
        channel="2los_onZ_1tau_5j", variable="lj0pt",
    )
    assert calls == [("2los_onZ_1tau_5j", 5)]


@pytest.mark.parametrize("run, year", [("run2", "UL18"), ("run3", "2022")])
def test_missing_parton_remains_inapplicable_to_ttll(run, year):
    source = contract._card_rates(run, True)["missing_parton"]
    assert set(source.corrs) == {"tllq", "tHq"}
    assert not source.has_process("ttll")
    assert "missing_parton" not in contract.rate_responses(
        f"ttll_private{year}", np.array([7.0]),
        channel="2los_onZ_1tau_5j", variable="lj0pt",
    )


@pytest.mark.parametrize("year", ["UL18", "2022"])
@pytest.mark.parametrize("card_name", ["tllq", "tHq"])
@pytest.mark.parametrize("variable", ["njets", None])
def test_missing_parton_njets_skips_special_rate_lookup(monkeypatch, year, card_name, variable):
    def unexpected_lookup(*args):
        pytest.fail("njets must not request a channel/jet-specific rate")

    monkeypatch.setattr(contract, "_special_rate_factor", unexpected_lookup)
    assert "missing_parton" not in contract.rate_responses(
        f"{card_name}_private{year}", np.array([7.0]),
        channel="2los_onZ_1tau_5j", variable=variable,
    )


def test_non_njets_special_rate_requires_physical_jet_channel():
    with pytest.raises(ValueError, match="cannot resolve jets"):
        contract.rate_responses(
            "WZTo3LNu_central2022", np.array([5.0]),
            channel="3l_onZ_1b", variable="lj0pt",
        )


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
