from __future__ import annotations

import pytest

from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.sumw2_policy import (
    resolve_nominal_component_availability,
    resolve_sumw2_storage_policy,
    resolved_policy_from_provenance,
)


@pytest.fixture
def samples():
    return {
        "data_run": {"histAxisName": "data", "isData": True, "WCnames": []},
        "background_a": {
            "histAxisName": "background",
            "isData": False,
            "WCnames": [],
        },
        "signal_eft": {
            "histAxisName": "signal",
            "isData": False,
            "WCnames": ["ctG"],
        },
    }


def _resolve(block, samples, families=("njets", "met"), **kwargs):
    return resolve_sumw2_storage_policy(
        block,
        samples=samples,
        runtime_families=families,
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        **kwargs,
    )


def test_component_classification_uses_data_and_validated_wc_metadata(samples):
    assert resolve_nominal_component_availability(samples) == {
        "scalar": True,
        "eft": True,
    }
    broken = dict(samples)
    broken["data_run"] = dict(broken["data_run"], WCnames=["ctG"])
    with pytest.raises(ValueError, match="Data sample"):
        resolve_nominal_component_availability(broken)


@pytest.mark.parametrize(
    "mode,block,analysis_mode,expected_count",
    [
        ("full_diagnostics", {"mode": "full_diagnostics"}, "standard", 6),
        ("disabled", {"mode": "disabled"}, "standard", 0),
        (
            "production",
            {"mode": "production", "rules": [{"process_prefixes": ["back"]}]},
            "standard",
            2,
        ),
        (
            "production_central",
            {
                "mode": "production_central",
                "rules": [{"process_prefixes": ["back"]}],
            },
            "standard",
            2,
        ),
        (
            "taufitter",
            {"mode": "taufitter", "rules": [{"variables": ["njets"]}]},
            "taufitter",
            3,
        ),
        (
            "full_custom",
            {"mode": "full_custom", "rules": [{"variables": ["njets"]}]},
            "standard",
            3,
        ),
    ],
)
def test_all_six_modes(mode, block, analysis_mode, expected_count, samples):
    policy = _resolve(
        block,
        samples,
        sumw2_storage_present=True,
        analysis_mode=analysis_mode,
    )
    assert policy.requested_mode == mode
    assert policy.resolved_mode == mode
    assert policy.signal_sample_profile == {
        "production": "private",
        "production_central": "central",
    }.get(mode, "unrestricted")
    assert len(policy.resolved_targets) == expected_count
    assert resolved_policy_from_provenance(policy.to_provenance()) == policy


@pytest.mark.parametrize("legacy_value", [True, False])
def test_new_block_conflicts_with_explicit_legacy_presence(samples, legacy_value):
    with pytest.raises(ValueError, match="cannot be combined"):
        _resolve(
            {"mode": "full_diagnostics"},
            samples,
            sumw2_storage_present=True,
            legacy_no_sumw2_present=True,
            legacy_no_sumw2_value=legacy_value,
        )


def test_new_block_with_unset_parser_default_false_uses_explicit_block(samples):
    policy = _resolve(
        {"mode": "disabled"},
        samples,
        sumw2_storage_present=True,
        legacy_no_sumw2_present=False,
        legacy_no_sumw2_value=False,
    )
    assert policy.source == "explicit"
    assert policy.requested_mode == "disabled"


@pytest.mark.parametrize(
    "legacy_present,legacy_value,expected_source,expected_mode",
    [
        (True, True, "legacy_no_sumw2", "disabled"),
        (True, False, "legacy_no_sumw2_false", "full_diagnostics"),
        (False, False, "implicit_production_default", "production"),
    ],
)
def test_legacy_source_presence_truth_table(
    samples, legacy_present, legacy_value, expected_source, expected_mode
):
    with pytest.warns(UserWarning, match="SUMW2-W001"):
        policy = _resolve(
            None,
            samples,
            sumw2_storage_present=False,
            legacy_no_sumw2_present=legacy_present,
            legacy_no_sumw2_value=legacy_value,
        )
    assert policy.source == expected_source
    assert policy.requested_mode == expected_mode


def test_modern_block_with_omitted_mode_uses_production_default(samples):
    policy = _resolve(
        {"rules": [{"process_names": ["background"]}]},
        samples,
        sumw2_storage_present=True,
    )
    assert policy.source == "implicit_production_default"
    assert policy.requested_mode == "production"
    assert policy.selected_processes("njets") == ("background",)


def test_absent_modern_and_legacy_configuration_production_can_use_requirements(samples):
    with pytest.warns(UserWarning, match="production default"):
        policy = _resolve(
            None,
            samples,
            sumw2_storage_present=False,
            implicit_production_requirements=[
                ("data_run", "data", "njets"),
            ],
        )
    assert policy.source == "implicit_production_default"
    assert policy.requested_mode == "production"
    assert [target.to_dict() for target in policy.resolved_targets] == [
        {"dataset": "data_run", "process": "data", "family": "njets"}
    ]


def test_selector_union_with_dataset_process_and_variable_and(samples):
    policy = _resolve(
        {
            "mode": "full_custom",
            "rules": [
                {
                    "dataset_prefixes": ["background"],
                    "process_names": ["background"],
                    "variables": ["njets"],
                },
                {
                    "dataset_names": ["signal_eft"],
                    "variables": ["met"],
                },
            ],
        },
        samples,
        sumw2_storage_present=True,
    )
    assert [(target.dataset, target.process, target.family) for target in policy.resolved_targets] == [
        ("background_a", "background", "njets"),
        ("signal_eft", "signal", "met"),
    ]


@pytest.mark.parametrize(
    "block,error_code",
    [
        ({"mode": "full_custom", "rules": [{"variables": ["unknown"]}]}, "SUMW2-E004"),
        ({"mode": "full_custom", "rules": [{"variables": ["ptz"]}]}, "SUMW2-E005"),
        (
            {
                "mode": "full_custom",
                "rules": [{"variables": ["lepton_pt_vs_eta_pt"]}],
            },
            "SUMW2-E006",
        ),
        (
            {"mode": "full_custom", "rules": [{"dataset_names": ["missing"]}]},
            "SUMW2-E007",
        ),
        (
            {
                "mode": "full_custom",
                "rules": [
                    {"process_names": ["signal"]},
                    {"process_prefixes": ["sig"]},
                ],
            },
            "SUMW2-E008",
        ),
    ],
)
def test_unknown_unselected_internal_unmatched_and_overlap_errors(samples, block, error_code):
    with pytest.raises(ValueError, match=error_code):
        _resolve(block, samples, sumw2_storage_present=True)


def test_duplicate_rules_zero_target_and_consumer_requirements_are_rejected(samples):
    duplicate = {
        "mode": "full_custom",
        "rules": [{"process_names": ["signal"]}, {"process_names": ["signal"]}],
    }
    with pytest.raises(ValueError, match="SUMW2-E003"):
        _resolve(duplicate, samples, sumw2_storage_present=True)

    with pytest.raises(ValueError, match="SUMW2-E007"):
        _resolve(
            {
                "mode": "full_custom",
                "rules": [
                    {
                        "dataset_names": ["background_a"],
                        "process_names": ["signal"],
                    }
                ],
            },
            samples,
            sumw2_storage_present=True,
        )

    with pytest.raises(ValueError, match="SUMW2-E010"):
        _resolve(
            {"mode": "disabled"},
            samples,
            sumw2_storage_present=True,
            consumer_requirements=(("data_run", "data", "njets"),),
        )


def test_missing_parton_contract_is_standard_full_custom_njets(samples):
    policy = _resolve(
        {"mode": "full_custom", "rules": [{"variables": ["njets"]}]},
        samples,
        sumw2_storage_present=True,
        analysis_mode="standard",
    )
    assert policy.selected_families() == ("njets",)
    assert policy.requested_mode == "full_custom"


def test_unknown_mode_lists_all_six_values(samples):
    with pytest.raises(ValueError) as error_info:
        _resolve(
            {"mode": "unknown"},
            samples,
            sumw2_storage_present=True,
        )
    message = str(error_info.value)
    for mode in (
        "production",
        "production_central",
        "taufitter",
        "full_diagnostics",
        "disabled",
        "full_custom",
    ):
        assert mode in message


def test_two_production_modes_share_ordinary_allocation(samples):
    rule = [{"process_prefixes": ["back"], "variables": ["njets"]}]
    private = _resolve(
        {"mode": "production", "rules": rule},
        samples,
        sumw2_storage_present=True,
    )
    central = _resolve(
        {"mode": "production_central", "rules": rule},
        samples,
        sumw2_storage_present=True,
    )
    assert private.resolved_targets == central.resolved_targets
    assert private.signal_sample_profile == "private"
    assert central.signal_sample_profile == "central"
