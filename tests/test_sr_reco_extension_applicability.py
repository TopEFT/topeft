import pytest

from analysis.topeft_run2 import analysis_processor


@pytest.fixture(scope="module")
def channel_config():
    return analysis_processor.load_category_config()


@pytest.mark.parametrize(
    ("family", "lepton_channel", "expected"),
    [
        ("fwd0pt", "2lss_fwd_p", True),
        ("fwd0eta", "2lss_fwd_m", True),
        ("fwd0pt", "3l_onZ_1b_fwd", True),
        ("fwd0eta", "3l_m_offZ_2b_fwd", True),
        ("fwd0pt", "2lss_p", False),
        ("fwd0eta", "3l_1tau_1b", False),
        ("tau0Tpt", "2lss_m_1tau_onZ", True),
        ("tau0Tpt", "2los_onZ_1tau", True),
        ("tau0Tpt", "3l_1tau_1b", True),
        ("tau0Tpt", "3l_1tau_2b", True),
        ("tau0Tpt", "2lss_p", False),
        ("tau0Tpt", "3l_onZ_1b", False),
        ("tau0Tpt", "4l", False),
    ],
)
def test_sr_reco_families_follow_explicit_channel_semantics(
    family, lepton_channel, expected
):
    producer = object.__new__(analysis_processor.AnalysisProcessor)
    producer._analysis_mode = "all"
    assert producer.histogram_fill_is_applicable(
        family, lepton_channel, lepton_channel, producer_region="SR"
    ) is expected


@pytest.mark.parametrize("block_names", [("2l", "2lss_1tau", "2los_1tau", "4l"), ("3l_onZ_tau",)])
def test_sr_sidecar_matches_runtime_for_mixed_blocks(channel_config, block_names):
    categories = {
        name: channel_config["ALL_CH_LST_SR"][name] for name in block_names
    }
    families = ("fwd0pt", "fwd0eta", "tau0Tpt")
    declaration = analysis_processor.AnalysisProcessor.build_histogram_applicability(
        analysis_mode="all",
        runtime_families=families,
        selected_category_dicts=(categories,),
        selected_category_regions=("SR",),
    )
    producer = object.__new__(analysis_processor.AnalysisProcessor)
    producer._analysis_mode = "all"
    for family in families:
        states = declaration["families"][family]["channels"]
        for category in categories.values():
            for lepton_channel, *_ in category["lep_chan_lst"]:
                for jet_label in category["jet_lst"]:
                    jet_mode, jet_threshold, _ = (
                        analysis_processor.parse_analysis_njet_token(jet_label)
                    )
                    channel = analysis_processor.construct_cat_name(
                        lepton_channel,
                        njet_str=f"{jet_mode}_{jet_threshold}j",
                    )
                    expected = producer.histogram_fill_is_applicable(
                        family, channel, lepton_channel, producer_region="SR"
                    )
                    assert states[channel] == (
                        "applicable" if expected else "not_applicable"
                    )
    if block_names == ("3l_onZ_tau",):
        tau_states = declaration["families"]["tau0Tpt"]["channels"]
        assert any(
            state == "applicable" for channel, state in tau_states.items()
            if channel.startswith("3l_1tau_")
        )
        assert all(
            state == "not_applicable" for channel, state in tau_states.items()
            if channel.startswith("3l_onZ_")
        )


@pytest.mark.parametrize("family", ("fwd0pt", "fwd0eta", "tau0Tpt"))
def test_cr_reco_applicability_matches_prior_unscoped_decision(channel_config, family):
    category = channel_config["CH_LST_CR"]["2l_CR"]
    lepton_channel = category["lep_chan_lst"][0][0]
    producer = object.__new__(analysis_processor.AnalysisProcessor)
    producer._analysis_mode = "all"
    prior = producer.histogram_fill_is_applicable(
        family, lepton_channel, lepton_channel
    )
    current = producer.histogram_fill_is_applicable(
        family, lepton_channel, lepton_channel, producer_region="CR"
    )
    assert current is prior
    declaration = analysis_processor.AnalysisProcessor.build_histogram_applicability(
        analysis_mode="all",
        runtime_families=(family,),
        selected_category_dicts=({"2l_CR": category},),
        selected_category_regions=("CR",),
    )
    assert set(declaration["families"][family]["channels"].values()) == {
        "applicable" if prior else "not_applicable"
    }
