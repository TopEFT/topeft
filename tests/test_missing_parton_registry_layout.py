from __future__ import annotations

import json
from copy import deepcopy

import pytest

from topeft.modules.missing_parton_contract import (
    LEGACY_MISSING_PARTON_PAYLOAD_LENGTHS,
    SUPPORTED_SR_REGISTRIES,
    build_registry_payload_layout,
    load_registry_payload_layout,
    parse_sr_njet_token,
)
from topeft.modules.paths import topeft_path


def _family(base_categories, jet_lst):
    return {
        "lep_chan_lst": [[category] for category in base_categories],
        "jet_lst": list(jet_lst),
    }


@pytest.mark.parametrize("registry", SUPPORTED_SR_REGISTRIES)
def test_real_registry_layout_is_deterministic_and_physical_indexed(registry):
    config_path = topeft_path("channels/ch_lst.json")
    with open(config_path, encoding="utf-8") as config_stream:
        selected_config = json.load(config_stream)[registry]

    expected_categories = []
    expected_by_category = {}
    for sr_family, family_config in selected_config.items():
        jet_lst = tuple(family_config["jet_lst"])
        parsed = tuple(parse_sr_njet_token(token) for token in jet_lst)
        for channel_definition in family_config["lep_chan_lst"]:
            base_category = channel_definition[0]
            if base_category not in expected_by_category:
                expected_categories.append(base_category)
                expected_by_category[base_category] = (sr_family, jet_lst, parsed)
            else:
                assert expected_by_category[base_category][1] == jet_lst

    first = load_registry_payload_layout(registry)
    second = load_registry_payload_layout(registry)

    assert first == second
    assert list(first.ordered_base_categories) == expected_categories
    assert len(first.categories) == len(expected_categories)
    for category in first.categories:
        sr_family, jet_lst, parsed = expected_by_category[
            category.base_sr_category
        ]
        modes = [mode for mode, _, _ in parsed]
        thresholds = [threshold for _, threshold, _ in parsed]
        assert category.registry == registry
        assert category.sr_family == sr_family
        assert category.jet_lst == jet_lst
        assert [token.mode for token in category.parsed_tokens] == modes
        assert [token.threshold for token in category.parsed_tokens] == thresholds
        assert category.terminal_threshold == thresholds[-1]
        assert category.public_array_length == thresholds[-1] + 1
        assert category.final_suffixes == tuple(
            suffix for _, _, suffix in parsed
        )
        assert category.consumed_physical_indices == tuple(thresholds)
        assert category.leading_compatibility_indices == tuple(
            range(thresholds[0])
        )
        assert max(range(category.public_array_length)) == thresholds[-1]
        assert category.terminal_mode == (
            "inclusive" if modes[-1] == "atleast" else "exact"
        )


def test_identical_duplicate_base_category_is_deduplicated_in_first_seen_order():
    layout = build_registry_payload_layout(
        "ALL_CH_LST_SR",
        {
            "family_a": _family(("a", "b"), ("=2", ">3")),
            "family_b": _family(("a", "c"), ("=2", ">3")),
        },
    )

    assert layout.ordered_base_categories == ("a", "b", "c")
    assert layout.categories_by_name["a"].sr_family == "family_a"


def test_layout_derivation_does_not_mutate_selected_json_block():
    config = {
        "family": _family(("a", "b"), ("=2", "=3", ">4")),
    }
    original = deepcopy(config)

    build_registry_payload_layout("ALL_CH_LST_SR", config)

    assert config == original


def test_conflicting_duplicate_base_category_fails_clearly():
    with pytest.raises(ValueError, match="Conflicting payload layouts"):
        build_registry_payload_layout(
            "ALL_CH_LST_SR",
            {
                "family_a": _family(("a",), ("=2", ">3")),
                "family_b": _family(("a",), ("=2", ">4")),
            },
        )


@pytest.mark.parametrize(
    "jet_lst,message",
    (
        ((), "empty jet_lst"),
        (("jets3",), "Invalid jet_lst token"),
        ((">3", "=4"), "must be terminal"),
        ((">3", ">4"), "more than one inclusive"),
        (("=4", "=3"), "increase strictly"),
        (("=3", "=3"), "increase strictly"),
    ),
)
def test_invalid_jet_contracts_fail(jet_lst, message):
    with pytest.raises(ValueError, match=message):
        build_registry_payload_layout(
            "ALL_CH_LST_SR",
            {"family": _family(("a",), jet_lst)},
        )


def test_exact_terminal_keeps_direct_physical_indices_without_inventing_tail():
    layout = build_registry_payload_layout(
        "ALL_CH_LST_SR",
        {"family": _family(("a",), ("=2", "=3", "=4"))},
    ).categories[0]

    assert layout.terminal_mode == "exact"
    assert layout.terminal_threshold == 4
    assert layout.public_array_length == 5
    assert layout.consumed_physical_indices == (2, 3, 4)
    assert layout.leading_compatibility_indices == (0, 1)


def test_all_registry_changes_only_the_two_accepted_forward_lengths():
    derived = load_registry_payload_layout("ALL_CH_LST_SR").public_lengths
    changes = {
        category: (LEGACY_MISSING_PARTON_PAYLOAD_LENGTHS[category], length)
        for category, length in derived.items()
        if LEGACY_MISSING_PARTON_PAYLOAD_LENGTHS[category] != length
    }

    assert changes == {
        "3l_m_offZ_1b_fwd": (6, 5),
        "3l_p_offZ_1b_fwd": (6, 5),
    }


@pytest.mark.parametrize(
    "base_category",
    ("3l_m_offZ_1b_fwd", "3l_p_offZ_1b_fwd"),
)
def test_forward_regression_has_complete_four_and_higher_public_tail(base_category):
    category = load_registry_payload_layout(
        "ALL_CH_LST_SR"
    ).categories_by_name[base_category]

    assert category.jet_lst == ("=1", "=2", "=3", ">4")
    assert category.public_array_length == 5
    assert category.terminal_threshold == 4
    assert 5 not in range(category.public_array_length)
