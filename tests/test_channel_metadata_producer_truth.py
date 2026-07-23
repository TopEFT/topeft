import inspect
import json

import pytest

from analysis.topeft_run2 import make_cr_and_sr_plots as plots
from analysis.topeft_run2.analysis_processor import (
    AnalysisProcessor,
    construct_cat_name,
    resolve_category_dict_names,
)
from topeft.modules.paths import topeft_path as te_topeft_path


def _normalize_jet_token(raw_token):
    if raw_token.startswith("="):
        prefix = "exactly_"
    elif raw_token.startswith("<"):
        prefix = "atmost_"
    elif raw_token.startswith(">"):
        prefix = "atleast_"
    else:
        raise ValueError(f"Unsupported jet token '{raw_token}' in channel metadata.")
    return f"{prefix}{raw_token[1:]}j"


def _expand_producer_channel_sets(channel_config):
    base_channels = set()
    leaf_channels = set()
    for _, category_cfg in channel_config.items():
        jet_tokens = [_normalize_jet_token(token) for token in category_cfg.get("jet_lst", [])]
        for lep_channel_def in category_cfg.get("lep_chan_lst", []):
            base_name = lep_channel_def[0]
            base_channels.add(base_name)
            for jet_token in jet_tokens:
                leaf_channels.add(construct_cat_name(base_name, njet_str=jet_token))
    return base_channels, leaf_channels


def _load_channel_json():
    with open(te_topeft_path("channels/ch_lst.json"), encoding="utf-8") as handle:
        return json.load(handle)


def _tokens_for_category(category_cfg):
    return {
        lep_channel_def[0]: set(lep_channel_def[1:])
        for lep_channel_def in category_cfg.get("lep_chan_lst", [])
    }


def test_sr_metadata_covers_default_producer_truth():
    channel_json = _load_channel_json()
    sr_dict_name, _ = resolve_category_dict_names(False, False, False, False)
    producer_base, producer_leaves = _expand_producer_channel_sets(channel_json[sr_dict_name])

    namespace, _ = plots._resolve_region_channel_namespace("SR")
    metadata_base = set(namespace["base_to_leaves"].keys())
    metadata_leaves = set(namespace["leaf_to_base"].keys())

    assert producer_base <= metadata_base
    assert producer_leaves <= metadata_leaves


def test_standard_cr_categories_use_existing_forward_gate_token():
    """Standard CR groups intentionally leave forward-jet gating unapplied."""

    channel_json = _load_channel_json()
    standard_cr_categories = [
        "2l_CR",
        "2l_CRflip",
        "2los_CRZ",
        "2los_CRtt",
        "3l_CR",
    ]

    for block_name in ("CH_LST_CR", "TAU_CH_LST_CR"):
        for category_name in standard_cr_categories:
            tokens_by_channel = _tokens_for_category(channel_json[block_name][category_name])
            assert tokens_by_channel
            for tokens in tokens_by_channel.values():
                assert "fwdjet_mask" not in tokens
                assert "~fwdjet_mask" not in tokens


def test_tau_specific_cr_categories_do_not_inherit_forward_gate_token():
    channel_json = _load_channel_json()
    tau_specific_categories = [
        "2los_1tau",
        "1l_1tau_CRtt",
        "1l_1tau_CRDY",
    ]

    for category_name in tau_specific_categories:
        tokens_by_channel = _tokens_for_category(channel_json["TAU_CH_LST_CR"][category_name])
        assert tokens_by_channel
        for tokens in tokens_by_channel.values():
            assert "fwdjet_mask" not in tokens


def test_forward_gate_token_is_defined_in_processor_preselections():
    source = "".join(inspect.getsource(AnalysisProcessor.process).split())
    assert 'preselections.add("fwdjet_mask",(fwdjet_mask))' in source


def test_cr_tau_mode_metadata_covers_producer_truth_after_declared_transforms():
    channel_json = _load_channel_json()
    _, cr_dict_name = resolve_category_dict_names(False, True, False, False)
    producer_base, producer_leaves = _expand_producer_channel_sets(channel_json[cr_dict_name])

    namespace, _ = plots._resolve_region_channel_namespace("CR")
    metadata_base = set(namespace["base_to_leaves"].keys())
    metadata_leaves = set(namespace["leaf_to_base"].keys())

    transformed_leaves = {
        plots._apply_channel_transforms(channel_name, ["lepflav"])
        for channel_name in metadata_leaves
    }
    transformed_bases = {
        plots._apply_channel_transforms(channel_name, ["lepflav", "njets"])
        for channel_name in metadata_leaves
    }

    assert producer_leaves <= transformed_leaves
    assert producer_base <= transformed_bases
    assert producer_base <= metadata_base


def test_region_channel_config_sets_cr_and_sr_lepflav_flags():
    cr_cfg = plots._resolve_region_channel_config("CR")
    sr_cfg = plots._resolve_region_channel_config("SR")

    assert cr_cfg["is_lepton_flavor_in_pkl"] is True
    assert sr_cfg["is_lepton_flavor_in_pkl"] is False


def test_strict_leaf_overlap_policy_remains_enforced():
    with pytest.raises(ValueError, match="leaf overlap"):
        plots._build_channel_namespace(
            {
                "2los_CRZ": {"leaves": ["2los_ee_CRZ_0j"], "alias": "cr_2los_Z"},
                "2los_CRZ_duplicate": {
                    "leaves": ["2los_ee_CRZ_0j"],
                    "alias": "cr_2los_Z_dup",
                },
            },
            region_label="CR_CHAN_DICT",
        )
