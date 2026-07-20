from __future__ import annotations

import json

import pytest

from analysis.topeft_run2.analysis_processor import construct_cat_name
from topeft.modules.missing_parton_contract import (
    build_channel_appl_contract,
    load_missing_parton_channel_contract,
)
from topeft.modules.paths import topeft_path


EXPECTED_CHANNELS_BY_APPL = {
    "isSR_2lSS": {
        "2lss_p",
        "2lss_m",
        "2lss_4t_p",
        "2lss_4t_m",
        "2lss_fwd_p",
        "2lss_fwd_m",
        "2lss_m_1tau_onZ",
        "2lss_p_1tau_offZ",
        "2lss_m_1tau_offZ",
        "2lss_p_1tau_onZ",
    },
    "isSR_2lOS": {"2los_onZ_1tau"},
    "isSR_3l": {
        "3l_m_offZ_low_1b",
        "3l_m_offZ_high_1b",
        "3l_m_offZ_none_1b",
        "3l_m_offZ_low_2b",
        "3l_m_offZ_high_2b",
        "3l_m_offZ_none_2b",
        "3l_p_offZ_low_1b",
        "3l_p_offZ_high_1b",
        "3l_p_offZ_none_1b",
        "3l_p_offZ_low_2b",
        "3l_p_offZ_high_2b",
        "3l_p_offZ_none_2b",
        "3l_onZ_1b",
        "3l_onZ_2b",
        "3l_1tau_1b",
        "3l_1tau_2b",
        "3l_onZ_1b_fwd",
        "3l_onZ_2b_fwd",
        "3l_m_offZ_1b_fwd",
        "3l_p_offZ_1b_fwd",
        "3l_m_offZ_2b_fwd",
        "3l_p_offZ_2b_fwd",
    },
    "isSR_4l": {"4l"},
}


def _family(channel, appl_labels=("isSR_3l",), jet_labels=("=2",)):
    return {
        "lep_chan_lst": [[channel]],
        "appl_lst": list(appl_labels),
        "jet_lst": list(jet_labels),
    }


def test_all_34_base_channels_use_exact_metadata_defined_sr_appl():
    contract = load_missing_parton_channel_contract()

    expected_mapping = {
        channel: appl
        for appl, channels in EXPECTED_CHANNELS_BY_APPL.items()
        for channel in channels
    }
    assert dict(contract.base_to_sr_appl) == expected_mapping
    assert len(contract.base_to_sr_appl) == 34
    assert len(contract.final_to_base) == 132

    for final_channel, base_channel in contract.final_to_base.items():
        assert contract.expected_sr_appl(final_channel) == expected_mapping[base_channel]


def test_final_channel_resolver_matches_processor_channel_constructor():
    with open(topeft_path("channels/ch_lst.json"), encoding="utf-8") as config_stream:
        sr_config = json.load(config_stream)["ALL_CH_LST_SR"]
    expected_final_to_base = {}
    for family_config in sr_config.values():
        for channel_definition in family_config["lep_chan_lst"]:
            base_channel = channel_definition[0]
            for source_label in family_config["jet_lst"]:
                prefix = "exactly_" if source_label.startswith("=") else "atleast_"
                normalized_label = f"{prefix}{source_label[1:]}j"
                final_channel = construct_cat_name(
                    base_channel,
                    njet_str=normalized_label,
                )
                expected_final_to_base[final_channel] = base_channel

    contract = load_missing_parton_channel_contract()

    assert dict(contract.final_to_base) == expected_final_to_base


def test_duplicate_base_channel_metadata_fails():
    config = {
        "family_a": _family("3l_onZ_1b"),
        "family_b": _family("3l_onZ_1b"),
    }

    with pytest.raises(ValueError, match="Duplicate base channel"):
        build_channel_appl_contract(config)


def test_contradictory_sr_appl_metadata_fails():
    config = {
        "family": _family(
            "3l_onZ_1b",
            appl_labels=("isSR_3l", "isSR_4l", "isAR_3l"),
        ),
    }

    with pytest.raises(ValueError, match="exactly one authoritative isSR_"):
        build_channel_appl_contract(config)


def test_duplicate_sr_appl_metadata_fails():
    config = {
        "family": _family(
            "3l_onZ_1b",
            appl_labels=("isSR_3l", "isSR_3l"),
        ),
    }

    with pytest.raises(ValueError, match="exactly one authoritative isSR_"):
        build_channel_appl_contract(config)


def test_unknown_channel_fails_without_substring_fallback():
    contract = load_missing_parton_channel_contract()

    with pytest.raises(ValueError, match="Unknown missing-parton channel"):
        contract.expected_sr_appl("custom_3l_channel")
