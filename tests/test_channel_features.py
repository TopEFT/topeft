from pathlib import Path

import pytest

import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.topeft_run2.workflow import ChannelPlanner, normalize_jet_category
from topeft.modules.channel_metadata import ChannelMetadataHelper


MINIMAL_CHANNEL_METADATA = {
    "groups": {
        "TOP22_006_CH_LST_SR": {
            "description": "Baseline signal regions",
            "regions": [
                {
                    "lepton_category": "3l",
                    "lepton_flavors": ["eee"],
                    "jet_bins": ["=2"],
                    "application_tags": {"mc": ["isSR_3l"]},
                    "histogram_variables": {"exclude": ["ptz_wtau", "tau0pt"]},
                    "region_definitions": [
                        {
                            "name": "3l_p_offZ_1b",
                            "channel": "3l_p",
                            "subchannel": "3l_offZ",
                            "tags": ["bmask_exactly1m"],
                        }
                    ],
                }
            ],
        },
        "OFFZ_SPLIT_CH_LST_SR": {
            "description": "Off-Z split variant of the trilepton channel",
            "features": ["offz_split"],
            "regions": [
                {
                    "lepton_category": "3l",
                    "lepton_flavors": ["eee"],
                    "jet_bins": ["=2"],
                    "application_tags": {"mc": ["isSR_3l"]},
                    "histogram_variables": {"include": ["lt"]},
                    "region_definitions": [
                        {
                            "name": "3l_p_offZ_low_1b",
                            "channel": "3l_p",
                            "subchannel": "3l_offZ_low",
                            "tags": ["bmask_exactly1m"],
                        }
                    ],
                }
            ],
        },
        "TAU_CH_LST_SR": {
            "description": "Tau-enriched signal regions",
            "features": ["requires_tau"],
            "regions": [
                {
                    "lepton_category": "2lss",
                    "lepton_flavors": ["ee"],
                    "jet_bins": ["=4"],
                    "application_tags": {"mc": ["isSR_2lSS"]},
                    "region_definitions": [
                        {
                            "name": "2lss_p",
                            "channel": "2lss",
                            "subchannel": "2l_p",
                            "tags": ["bmask_atleast1m2l", "0tau"],
                        }
                    ],
                }
            ],
        },
        "TAU_CH_LST_CR": {
            "description": "Tau control regions",
            "regions": [
                {
                    "lepton_category": "2los_1tau",
                    "lepton_flavors": ["ee"],
                    "jet_bins": ["=2"],
                    "application_tags": {"mc": ["isSR_2lOS"]},
                    "region_definitions": [
                        {
                            "name": "2los_1tau_Ftau",
                            "channel": "2los",
                            "subchannel": "2l_nozeeveto",
                            "tags": ["1Ftau"],
                        }
                    ],
                }
            ],
        },
        "FWD_CH_LST_SR": {
            "description": "Forward-enriched signal regions",
            "features": ["requires_forward"],
            "regions": [
                {
                    "lepton_category": "2l",
                    "lepton_flavors": ["ee"],
                    "jet_bins": ["=4"],
                    "application_tags": {"mc": ["isSR_2lSS"]},
                    "region_definitions": [
                        {
                            "name": "2lss_fwd_p",
                            "channel": "2lss_fwd",
                            "subchannel": "2l_fwd_p",
                            "tags": ["bmask_atleast1m2l"],
                        }
                    ],
                }
            ],
        },
        "CH_LST_CR": {
            "description": "Shared control regions",
            "regions": [
                {
                    "lepton_category": "2l_CR",
                    "lepton_flavors": ["ee"],
                    "jet_bins": ["=1"],
                    "application_tags": {"mc": ["isSR_2lSS"]},
                    "region_definitions": [
                        {
                            "name": "2lss_CR",
                            "channel": "chargedl0",
                            "subchannel": "2lss",
                            "tags": [],
                        }
                    ],
                }
            ],
        },
    },
    "scenarios": [
        {
            "name": "TOP_22_006",
            "groups": ["TOP22_006_CH_LST_SR", "OFFZ_SPLIT_CH_LST_SR", "CH_LST_CR"],
        },
        {
            "name": "tau_analysis",
            "groups": ["TAU_CH_LST_SR", "TAU_CH_LST_CR", "CH_LST_CR"],
        },
        {
            "name": "fwd_analysis",
            "groups": ["FWD_CH_LST_SR", "CH_LST_CR"],
        },
    ],
}


@pytest.fixture
def channel_helper():
    return ChannelMetadataHelper(MINIMAL_CHANNEL_METADATA)


def build_dict(
    channel_helper,
    channel,
    application,
    *,
    is_data,
    skip_sr,
    skip_cr,
    scenario_names=None,
):
    planner = ChannelPlanner(
        channel_helper,
        skip_sr=skip_sr,
        skip_cr=skip_cr,
        scenario_names=scenario_names,
    )
    return planner.build_channel_dict(channel, application, is_data=is_data)


def test_build_channel_dict_includes_offz_features(channel_helper):
    channel_dict = build_dict(
        channel_helper,
        "3l_p_offZ_low_1b_2j",
        "isSR_3l",
        is_data=False,
        skip_sr=False,
        skip_cr=False,
        scenario_names=["TOP_22_006"],
    )
    assert "offz_split" in channel_dict["features"]


def test_build_channel_dict_includes_tau_features_for_control(channel_helper):
    channel_dict = build_dict(
        channel_helper,
        "2los_1tau_Ftau_2j",
        "isSR_2lOS",
        is_data=False,
        skip_sr=False,
        skip_cr=False,
        scenario_names=["tau_analysis"],
    )
    assert "requires_tau" in channel_dict["features"]


def test_build_channel_dict_preserves_features_when_sr_skipped(channel_helper):
    channel_dict = build_dict(
        channel_helper,
        "2los_1tau_Ftau_2j",
        "isSR_2lOS",
        is_data=False,
        skip_sr=True,
        skip_cr=False,
        scenario_names=["tau_analysis"],
    )
    assert "requires_tau" in channel_dict["features"]


def test_build_channel_dict_includes_forward_features(channel_helper):
    channel_dict = build_dict(
        channel_helper,
        "2lss_fwd_p_4j",
        "isSR_2lSS",
        is_data=False,
        skip_sr=False,
        skip_cr=False,
        scenario_names=["fwd_analysis"],
    )
    assert "requires_forward" in channel_dict["features"]


def test_resolve_channel_groups_infers_tau_control_regions(channel_helper):
    planner = ChannelPlanner(
        channel_helper,
        skip_sr=False,
        skip_cr=False,
        scenario_names=["tau_analysis"],
    )
    sr_groups, cr_groups, features = planner.resolve_groups()

    assert any(group.name == "TAU_CH_LST_SR" for group in sr_groups)
    assert any(group.name == "TAU_CH_LST_CR" for group in cr_groups)
    assert "requires_tau" in features


def test_build_channel_dict_respects_histogram_filters(channel_helper):
    sr_channel = build_dict(
        channel_helper,
        "3l_p_offZ_1b_2j",
        "isSR_3l",
        is_data=False,
        skip_sr=False,
        skip_cr=False,
        scenario_names=["TOP_22_006"],
    )
    assert "tau0pt" in set(sr_channel.get("channel_var_blacklist", ()))

    fwd_channel = build_dict(
        channel_helper,
        "3l_p_offZ_low_1b_2j",
        "isSR_3l",
        is_data=False,
        skip_sr=False,
        skip_cr=False,
        scenario_names=["TOP_22_006"],
    )
    assert "lt" in set(fwd_channel.get("channel_var_whitelist", ()))


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("=2", "exactly_2j"),
        (">4", "atleast_4j"),
        ("<3", "atmost_3j"),
    ],
)
def test_normalize_jet_category(raw, expected):
    assert normalize_jet_category(raw) == expected

