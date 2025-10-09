import ast
import re
from pathlib import Path

import pytest

import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from topeft.modules.channel_metadata import ChannelMetadataHelper

_RUN_ANALYSIS_PATH = _REPO_ROOT / "analysis" / "topeft_run2" / "run_analysis.py"

with _RUN_ANALYSIS_PATH.open("r", encoding="utf-8") as _run_analysis_file:
    _RUN_ANALYSIS_SOURCE = _run_analysis_file.read()

_RUN_ANALYSIS_AST = ast.parse(_RUN_ANALYSIS_SOURCE)

_EXPORTED_FUNCTIONS = {}
for node in _RUN_ANALYSIS_AST.body:
    if isinstance(node, ast.FunctionDef) and node.name in {
        "resolve_channel_groups",
        "normalize_jet_category",
        "build_channel_dict",
    }:
        _EXPORTED_FUNCTIONS[node.name] = ast.get_source_segment(
            _RUN_ANALYSIS_SOURCE, node
        )

exec(_EXPORTED_FUNCTIONS["resolve_channel_groups"], globals())
exec(_EXPORTED_FUNCTIONS["normalize_jet_category"], globals())
exec(_EXPORTED_FUNCTIONS["build_channel_dict"], globals())


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


def test_build_channel_dict_includes_offz_features(channel_helper):
    channel_dict = build_channel_dict(
        "3l_p_offZ_low_1b_2j",
        "isSR_3l",
        isData=False,
        skip_sr=False,
        skip_cr=False,
        channel_helper=channel_helper,
        scenario_names=["TOP_22_006"],
    )
    assert "offz_split" in channel_dict["features"]


def test_build_channel_dict_includes_tau_features_for_control(channel_helper):
    channel_dict = build_channel_dict(
        "2los_1tau_Ftau_2j",
        "isSR_2lOS",
        isData=False,
        skip_sr=False,
        skip_cr=False,
        channel_helper=channel_helper,
        scenario_names=["tau_analysis"],
    )
    assert "requires_tau" in channel_dict["features"]


def test_build_channel_dict_preserves_features_when_sr_skipped(channel_helper):
    channel_dict = build_channel_dict(
        "2los_1tau_Ftau_2j",
        "isSR_2lOS",
        isData=False,
        skip_sr=True,
        skip_cr=False,
        channel_helper=channel_helper,
        scenario_names=["tau_analysis"],
    )
    assert "requires_tau" in channel_dict["features"]


def test_build_channel_dict_includes_forward_features(channel_helper):
    channel_dict = build_channel_dict(
        "2lss_fwd_p_4j",
        "isSR_2lSS",
        isData=False,
        skip_sr=False,
        skip_cr=False,
        channel_helper=channel_helper,
        scenario_names=["fwd_analysis"],
    )
    assert "requires_forward" in channel_dict["features"]


def test_resolve_channel_groups_infers_tau_control_regions(channel_helper):
    sr_groups, cr_groups, features = resolve_channel_groups(
        channel_helper,
        skip_sr=False,
        skip_cr=False,
        scenario_names=None,
        required_features=["requires_tau"],
    )

    assert any(group.name == "TAU_CH_LST_SR" for group in sr_groups)
    assert any(group.name == "TAU_CH_LST_CR" for group in cr_groups)
    assert "requires_tau" in features


