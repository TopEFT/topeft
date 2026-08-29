from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from topeft.modules.missing_parton_contract import (
    DEFAULT_SR_REGISTRY,
    SUPPORTED_SR_REGISTRIES,
    load_or_validate_selected_registry,
    normalize_sr_registry,
    parse_analysis_njet_token,
    parse_sr_njet_token,
)
from topeft.modules.paths import topeft_path


CHANNEL_CONFIG_PATH = Path(topeft_path("channels/ch_lst.json"))


def _registry_tokens():
    config = json.loads(CHANNEL_CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        str(token)
        for registry in SUPPORTED_SR_REGISTRIES
        for family in config[registry].values()
        for token in family["jet_lst"]
    }


def test_registry_normalization_preserves_only_canonical_values():
    assert normalize_sr_registry() == DEFAULT_SR_REGISTRY
    assert normalize_sr_registry(None) == DEFAULT_SR_REGISTRY
    for registry in SUPPORTED_SR_REGISTRIES:
        assert normalize_sr_registry(registry) == registry

    with pytest.raises(ValueError, match="Unsupported SR registry"):
        normalize_sr_registry("NOT_A_REGISTRY")


@pytest.mark.parametrize("sr_registry", SUPPORTED_SR_REGISTRIES)
def test_supported_registry_exists_and_requested_block_is_returned(sr_registry):
    config = json.loads(CHANNEL_CONFIG_PATH.read_text(encoding="utf-8"))

    observed_registry, observed_block = load_or_validate_selected_registry(
        sr_registry
    )

    assert observed_registry == sr_registry
    assert observed_block == config[sr_registry]


def test_supported_constant_missing_from_config_is_rejected(tmp_path):
    incomplete_config = tmp_path / "ch_lst.json"
    incomplete_config.write_text(
        json.dumps({DEFAULT_SR_REGISTRY: {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="absent from"):
        load_or_validate_selected_registry(
            "FWD_CH_LST_SR",
            config_path=incomplete_config,
        )


def test_processor_uses_the_canonical_sr_token_parser():
    from analysis.topeft_run2 import analysis_processor

    assert analysis_processor.parse_analysis_njet_token is parse_analysis_njet_token


def test_analysis_parser_preserves_atmost_support_outside_sr_registries():
    assert parse_analysis_njet_token("<4") == ("atmost", 4, "_4j")
    with pytest.raises(ValueError, match="expected an exact '=N' or inclusive '>N'"):
        parse_sr_njet_token("<4")


def test_every_registry_token_has_processor_equivalent_boolean_semantics():
    tokens = _registry_tokens()
    assert {">3", ">4", ">5", ">6", ">7"} <= tokens

    for token in tokens | {">1"}:
        mode, threshold, suffix = parse_sr_njet_token(token)
        assert (mode, threshold, suffix) == parse_analysis_njet_token(token)
        njet = np.asarray([max(threshold - 1, 0), threshold, threshold + 1])
        if mode == "exactly":
            selected = njet == threshold
            assert selected.tolist() == [threshold == 0, True, False]
        else:
            assert mode == "atleast"
            selected = njet >= threshold
            assert selected.tolist() == [threshold == 0, True, True]
        assert suffix == f"_{threshold}j"
        assert f"{mode}_{threshold}j" in {
            f"exactly_{threshold}j",
            f"atleast_{threshold}j",
        }


@pytest.mark.parametrize("token", (">1", ">3", ">4", ">5", ">6", ">7"))
def test_greater_than_tokens_include_the_threshold(token):
    mode, threshold, suffix = parse_sr_njet_token(token)

    assert mode == "atleast"
    assert np.asarray([threshold]) >= threshold
    assert bool((np.asarray([threshold]) >= threshold)[0]) is True
    assert suffix == f"_{threshold}j"
