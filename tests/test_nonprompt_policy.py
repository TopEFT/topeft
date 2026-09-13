from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from topeft.modules.nonprompt_policy import (
    DEFAULT_NONPROMPT_ALIAS_DEFINITIONS,
    EXPLICIT_NONPROMPT_EXCLUSION,
    PROMPT_SUBTRACTION_MEMBER,
    RUN2_YEARS,
    RUN3_YEARS,
    canonical_prompt_aliases,
    certify_active_nonprompt_policy,
    explicit_exclusion_aliases,
    nonprompt_alias_definition,
    nonprompt_policy_error,
    split_year_qualified_process,
)


RUN2_ACTIVE_BASES = (
    "TTGJets_central",
    "TTGamma_central",
    "TTTo2L2Nu_central",
    "TTToSemiLeptonic_central",
    "TTZToLL_M1to10_central",
    "TWZToLL_central",
    "WWTo2L2Nu_central",
    "WWW_4F_central",
    "WWZ_4F_central",
    "WZTo3LNu_central",
    "WZZ_central",
    "ZZTo4L_central",
    "ZZZ_central",
    "tHq_private",
    "tllq_private",
    "ttHJet_private",
    "ttllJet_private",
    "ttlnuJet_private",
    "tttt_private",
)

RUN3_ACTIVE_BASES = (
    "TTG-1Jets_PTG-10to100_central",
    "TTG-1Jets_PTG-100to200_central",
    "TTG-1Jets_PTG-200_central",
    "TTto2L2Nu_central",
    "TTtoLNu2Q_central",
    "TWZ_Tto2Q_WtoLNu_Zto2L_central",
    "TWZ_TtoLNu_Wto2Q_Zto2L_central",
    "TWZ_TtoLNu_WtoLNu_Zto2L_central",
    "WWTo2L2Nu_central",
    "WWW_central",
    "WWZ_central",
    "WZTo3LNu_central",
    "WZZ_central",
    "WZto3LNu-2Jets_central",
    "ZZTo4L_central",
    "ZZZ_central",
    "ggToZZTo2e2mu_central",
    "ggToZZTo2e2nu_central",
    "ggToZZTo2e2tau_central",
    "ggToZZTo2mu2nu_central",
    "ggToZZTo2mu2tau_central",
    "ggToZZTo4e_central",
    "ggToZZTo4mu_central",
    "ggToZZTo4tau_central",
    "tHq_private",
    "tllq_private",
    "ttH_private",
    "ttll_private",
    "ttlnu_private",
    "tttt_private",
)

RUN3_NEW_PROMPT_BASES = (
    "TTto2L2Nu_central",
    "ttH_private",
    "ggToZZTo2e2mu_central",
    "ggToZZTo2e2nu_central",
    "ggToZZTo2e2tau_central",
    "ggToZZTo2mu2tau_central",
    "ggToZZTo4e_central",
    "ggToZZTo4mu_central",
    "ggToZZTo4tau_central",
)

EXCLUDED_BASES = (
    "WZto3LNu-2Jets_central",
    "WWTo2L2Nu_central",
    "ggToZZTo2mu2nu_central",
)

# Exact aliases recorded by maintained Run-2 CR configuration/JSON inspection.
# Update this fixture when that route's configured histAxisName values change.
RUN2_CR_ONLY_ACTIVE_ALIASES = (
    "DY10to50_centralUL16APV",
    "DY10to50_centralUL16",
    "DY10to50_centralUL18",
    "DY50_centralUL16APV",
    "DY50_centralUL16",
    "DY50_centralUL18",
    "DYJetsToLL_centralUL17",
    "ST_antitop_t-channel_centralUL16APV",
    "ST_antitop_t-channel_centralUL16",
    "ST_antitop_t-channel_centralUL17",
    "ST_antitop_t-channel_centralUL18",
    "ST_top_s-channel_centralUL16APV",
    "ST_top_s-channel_centralUL16",
    "ST_top_s-channel_centralUL17",
    "ST_top_s-channel_centralUL18",
    "ST_top_t-channel_centralUL16APV",
    "ST_top_t-channel_centralUL16",
    "ST_top_t-channel_centralUL17",
    "ST_top_t-channel_centralUL18",
    "WJetsToLNu_centralUL16APV",
    "WJetsToLNu_centralUL16",
    "WJetsToLNu_centralUL17",
    "WJetsToLNu_centralUL18",
    "ZGToLLG_centralUL16APV",
    "ZGToLLG_centralUL16",
    "ZGToLLG_centralUL17",
    "ZGToLLG_centralUL18",
    "tW_centralUL16APV",
    "tW_centralUL16",
    "tW_centralUL17",
    "tW_centralUL18",
    "tbarW_centralUL16APV",
    "tbarW_centralUL16",
    "tbarW_centralUL17",
    "tbarW_centralUL18",
)

RUN3_CR_ONLY_ACTIVE_BASES = (
    "DYJetsToLL_MLL-10to50_central",
    "DYJetsToLL_MLL-50_central",
    "ST_antitop_t-channel_central",
    "ST_top_s-channel_central",
    "ST_top_t-channel_central",
    "ST_tW_Leptonic_central",
    "ST_tW_Semileptonic_central",
    "ST_tbarW_Leptonic_central",
    "ST_tbarW_Semileptonic_central",
    "TTto4Q_central",
    "WJetsToLNu_central",
    "ZG_MLL-4to50_PTG-10to100_central",
    "ZG_MLL-4to50_PTG-100to200_central",
    "ZG_MLL-4to50_PTG-200_central",
    "ZG_MLL-50_PTG-10to100_central",
    "ZG_MLL-50_PTG-100to200_central",
    "ZG_MLL-50_PTG-200to400_central",
    "ZG_MLL-50_PTG-400to600_central",
    "ZG_MLL-50_PTG-600_central",
)

HISTORICAL_PROMPT_ENTRIES = (
    "TTTo2L2Nu_central",
    "TTToSemiLeptonic_central",
    "TTtoLNu2Q_central",
    "TTZToLL_M1to10_central",
    "TWZToLL_central",
    "TWZ_Tto2Q_WtoLNu_Zto2L_central",
    "TWZ_TtoLNu_Wto2Q_Zto2L_central",
    "TWZ_TtoLNu_WtoLNu_Zto2L_central",
    "TTGamma_central",
    "TTGJets_central",
    "TTG-1Jets_PTG-10to100_central",
    "TTG-1Jets_PTG-100to200_central",
    "TTG-1Jets_PTG-200_central",
    "WWW_central",
    "WWZ_4F_central",
    "WWZ_central",
    "WZTo3LNu_central",
    "WZZ_central",
    "ZZTo4L_central",
    "ZZZ_central",
    "WWW_4F_central",
    "WWZ_central",
    "WZZ_ext_central",
    "tHq_private",
    "tllq_private",
    "ttHJet_private",
    "ttllJet_private",
    "ttlnuJet_private",
    "tttt_private",
    "tZq_central",
    "ttHJet_central",
    "ttH_central",
    "ttW_central",
    "ttZ_central",
    "tttt_central",
)


def _labels(bases, years):
    return tuple(f"{base}{year}" for base in bases for year in years)


RUN2_CR_ACTIVE_ALIASES = (
    _labels(RUN2_ACTIVE_BASES, RUN2_YEARS)
    + _labels(("data",), RUN2_YEARS)
    + RUN2_CR_ONLY_ACTIVE_ALIASES
)

RUN3_CR_ONLY_ACTIVE_ALIASES = _labels(RUN3_CR_ONLY_ACTIVE_BASES, RUN3_YEARS)
RUN3_CR_ACTIVE_ALIASES = (
    _labels(RUN3_ACTIVE_BASES, RUN3_YEARS)
    + _labels(("data",), RUN3_YEARS)
    + RUN3_CR_ONLY_ACTIVE_ALIASES
)


def _historical_prompt_set(active_aliases):
    params_path = Path(__file__).parents[1] / "topeft/params/params.json"
    historical_entries = set(
        json.loads(params_path.read_text(encoding="utf-8"))["prompt_subtraction_samples"]
    )
    return {
        alias
        for alias in active_aliases
        if split_year_qualified_process(alias)[0] in historical_entries
    }


def _certificate(bases, years):
    return certify_active_nonprompt_policy(
        _labels(bases, years),
        configuration_source="synthetic_maintained_universe",
    )


def test_run2_active_prompt_set_is_exactly_the_accepted_72_labels():
    certificate = _certificate(RUN2_ACTIVE_BASES, RUN2_YEARS)
    expected = set(_labels(set(RUN2_ACTIVE_BASES) - {"WWTo2L2Nu_central"}, RUN2_YEARS))
    assert len(certificate.resolved_prompt_process_set) == 72
    assert set(certificate.resolved_prompt_process_set) == expected
    assert set(certificate.explicit_exclusions) == set(
        _labels(("WWTo2L2Nu_central",), RUN2_YEARS)
    )


def test_run2_cr_active_alias_universe_has_complete_explicit_policy_coverage():
    certificate = certify_active_nonprompt_policy(
        RUN2_CR_ACTIVE_ALIASES,
        configuration_source="run2_cr_cfg_json_active_alias_universe",
    )

    assert len(RUN2_CR_ACTIVE_ALIASES) == 115
    assert {row.raw_process_label for row in certificate.resolutions} == set(
        RUN2_CR_ACTIVE_ALIASES
    )
    cr_only_rows = [
        row
        for row in certificate.resolutions
        if row.raw_process_label in RUN2_CR_ONLY_ACTIVE_ALIASES
    ]
    assert len(cr_only_rows) == 35
    assert {row.policy_role for row in cr_only_rows} == {
        EXPLICIT_NONPROMPT_EXCLUSION
    }
    assert not set(RUN2_CR_ONLY_ACTIVE_ALIASES) & set(
        certificate.resolved_prompt_process_set
    )
    assert set(RUN2_CR_ONLY_ACTIVE_ALIASES) <= set(certificate.explicit_exclusions)
    assert set(certificate.resolved_prompt_process_set) == _historical_prompt_set(
        RUN2_CR_ACTIVE_ALIASES
    )


def test_run3_cr_active_alias_universe_has_complete_historical_policy_coverage():
    certificate = certify_active_nonprompt_policy(
        RUN3_CR_ACTIVE_ALIASES,
        configuration_source="run3_cr_cfg_json_active_alias_universe",
    )

    assert len(RUN3_CR_ACTIVE_ALIASES) == 200
    assert {row.raw_process_label for row in certificate.resolutions} == set(
        RUN3_CR_ACTIVE_ALIASES
    )
    cr_only_rows = [
        row
        for row in certificate.resolutions
        if row.raw_process_label in RUN3_CR_ONLY_ACTIVE_ALIASES
    ]
    assert len(cr_only_rows) == 76
    assert {row.policy_role for row in cr_only_rows} == {
        EXPLICIT_NONPROMPT_EXCLUSION
    }
    assert set(certificate.resolved_prompt_process_set) == _historical_prompt_set(
        RUN3_CR_ACTIVE_ALIASES
    )


def test_certification_remains_bounded_to_the_active_sr_style_subset():
    active_aliases = ("TTTo2L2Nu_centralUL18", "WWTo2L2Nu_centralUL18")
    certificate = certify_active_nonprompt_policy(
        active_aliases,
        configuration_source="run2_sr_style_active_alias_subset",
    )

    assert {row.raw_process_label for row in certificate.resolutions} == set(
        active_aliases
    )
    assert certificate.resolved_prompt_process_set == ("TTTo2L2Nu_centralUL18",)
    assert certificate.explicit_exclusions == ("WWTo2L2Nu_centralUL18",)
    assert not set(RUN2_CR_ONLY_ACTIVE_ALIASES) & {
        row.raw_process_label for row in certificate.resolutions
    }


def test_run3_certification_remains_bounded_to_the_active_sr_style_subset():
    active_aliases = ("TTto2L2Nu_central2022", "WWTo2L2Nu_central2022")
    certificate = certify_active_nonprompt_policy(
        active_aliases,
        configuration_source="run3_sr_style_active_alias_subset",
    )

    assert {row.raw_process_label for row in certificate.resolutions} == set(
        active_aliases
    )
    assert certificate.resolved_prompt_process_set == ("TTto2L2Nu_central2022",)
    assert certificate.explicit_exclusions == ("WWTo2L2Nu_central2022",)
    assert not set(RUN3_CR_ONLY_ACTIVE_ALIASES) & {
        row.raw_process_label for row in certificate.resolutions
    }


@pytest.mark.parametrize(
    "unknown_alias",
    ("futureRenamedPrompt_centralUL18", "futureRenamedPrompt_central2022"),
)
def test_unknown_active_alias_is_fail_closed_for_both_run_eras(unknown_alias):
    with pytest.raises(nonprompt_policy_error, match="NONPROMPT-POLICY-E006"):
        certify_active_nonprompt_policy(
            (unknown_alias,), configuration_source="unknown_run_era_test"
        )


def test_run3_additions_exclusions_and_ggzz_labels_are_exact():
    certificate = _certificate(RUN3_ACTIVE_BASES, RUN3_YEARS)
    prompt = set(certificate.resolved_prompt_process_set)
    assert len(prompt) == 108
    assert set(_labels(RUN3_NEW_PROMPT_BASES, RUN3_YEARS)) <= prompt
    assert set(_labels(("ttll_private", "ttlnu_private"), RUN3_YEARS)) <= prompt
    assert set(certificate.explicit_exclusions) == set(
        _labels(EXCLUDED_BASES, RUN3_YEARS)
    )
    ggzz_rows = [
        row
        for row in certificate.resolutions
        if row.process_base.startswith("ggToZZ") and row.is_prompt_member
    ]
    assert len(ggzz_rows) == 28
    assert {row.canonical_family for row in ggzz_rows} == {
        "zz_to_4l_prompt_family"
    }
    assert len({row.raw_process_label for row in ggzz_rows}) == 28


@pytest.mark.parametrize(
    "run2_base,run3_base,family",
    [
        ("TTTo2L2Nu_central", "TTto2L2Nu_central", "dileptonic_ttbar"),
        ("ttHJet_private", "ttH_private", "private_tth"),
        ("ttllJet_private", "ttll_private", "private_ttll"),
        ("ttlnuJet_private", "ttlnu_private", "private_ttlnu"),
    ],
)
def test_run_era_aliases_are_additive_and_exact(run2_base, run3_base, family):
    run2 = _certificate((run2_base,), RUN2_YEARS)
    run3 = _certificate((run3_base,), RUN3_YEARS)
    assert {row.canonical_family for row in run2.resolutions} == {family}
    assert {row.canonical_family for row in run3.resolutions} == {family}


@pytest.mark.parametrize(
    "near_miss",
    [
        "ttto2l2nu_central2022",
        "TTto2L2Nu2022",
        "TTto2L2Nu_central_extra2022",
        "ttHJet_private2022",
        "ttH_privateUL18",
        "ggToZZTo4M_central2023",
        "WWTo2L2Nu_central2024",
    ],
)
def test_case_suffix_jet_private_central_and_year_near_misses_fail(near_miss):
    with pytest.raises(nonprompt_policy_error, match="NONPROMPT-POLICY-E00[16]"):
        certify_active_nonprompt_policy(
            (near_miss,),
            configuration_source="near_miss_test",
        )


def test_unknown_duplicate_alias_family_role_conflict_and_missing_required_fail():
    with pytest.raises(nonprompt_policy_error, match="NONPROMPT-POLICY-E006"):
        certify_active_nonprompt_policy(
            ("futureRenamedPrompt_central2022",),
            configuration_source="unknown_test",
        )

    duplicate = DEFAULT_NONPROMPT_ALIAS_DEFINITIONS + (
        DEFAULT_NONPROMPT_ALIAS_DEFINITIONS[1],
    )
    with pytest.raises(nonprompt_policy_error, match="ambiguous duplicate alias"):
        certify_active_nonprompt_policy(
            ("TTTo2L2Nu_centralUL18",),
            configuration_source="duplicate_test",
            alias_definitions=duplicate,
        )

    conflicting = DEFAULT_NONPROMPT_ALIAS_DEFINITIONS + (
        nonprompt_alias_definition(
            raw_process_base="conflicting_alias_central",
            canonical_family="dileptonic_ttbar",
            run_eras=("run3",),
            policy_role=EXPLICIT_NONPROMPT_EXCLUSION,
            policy_reason="test_conflict",
            source_of_alias="test",
        ),
    )
    with pytest.raises(nonprompt_policy_error, match="conflicting policy assignments"):
        certify_active_nonprompt_policy(
            ("TTto2L2Nu_central2022",),
            configuration_source="family_conflict_test",
            alias_definitions=conflicting,
        )

    with pytest.raises(nonprompt_policy_error, match="without a resolvable active"):
        certify_active_nonprompt_policy(
            ("TTto2L2Nu_central2022",),
            configuration_source="required_family_test",
            required_canonical_families=("private_tth",),
        )


def test_params_legacy_surface_is_additive_and_cannot_override_policy():
    params_path = Path(__file__).parents[1] / "topeft/params/params.json"
    entries = json.loads(params_path.read_text(encoding="utf-8"))[
        "prompt_subtraction_samples"
    ]
    for historical_entry in HISTORICAL_PROMPT_ENTRIES:
        assert historical_entry in entries
    assert set(canonical_prompt_aliases()) <= set(entries)
    assert not set(explicit_exclusion_aliases()) & set(entries)

    configured = certify_active_nonprompt_policy(
        ("ttH_private2022", "WWTo2L2Nu_central2022"),
        configuration_source="legacy_non_orchestrated_source",
        configured_prompt_aliases=entries,
    )
    direct = certify_active_nonprompt_policy(
        ("ttH_private2022", "WWTo2L2Nu_central2022"),
        configuration_source="current_orchestrated_source",
    )
    assert configured.resolved_prompt_process_set == direct.resolved_prompt_process_set
    assert configured.explicit_exclusions == direct.explicit_exclusions
    serialized = configured.to_dict()
    assert "eft_prompt_processes" not in serialized
    assert all("eft_sm_point" not in row for row in serialized["resolutions"])


def test_prompt_membership_ignores_signal_background_role_metadata():
    resolved_sets = []
    for analysis_role in ("signal", "background"):
        certificate = certify_active_nonprompt_policy(
            {
                "sample": {
                    "histAxisName": "ttH_private2022",
                    "isData": False,
                    "WCnames": ["ctW"],
                    "analysis_role": analysis_role,
                }
            },
            configuration_source=f"role_{analysis_role}",
        )
        resolved_sets.append(certificate.resolved_prompt_process_set)
    assert resolved_sets == [
        ("ttH_private2022",),
        ("ttH_private2022",),
    ]


def test_invalid_policy_call_precedes_processor_and_executor_construction():
    source_path = Path(__file__).parents[1] / "analysis/topeft_run2/run_analysis.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    calls = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in {
            "_certify_nonprompt_policy_before_executor",
            "AnalysisProcessor",
            "FuturesExecutor",
            "WorkQueueExecutor",
            "TaskVineExecutor",
        }:
            calls.setdefault(name, []).append(node.lineno)
    policy_line = min(calls["_certify_nonprompt_policy_before_executor"])
    executor_lines = [
        line
        for name, lines in calls.items()
        if name != "_certify_nonprompt_policy_before_executor"
        for line in lines
    ]
    assert policy_line < min(executor_lines)
    with pytest.raises(nonprompt_policy_error, match="unknown active process alias"):
        certify_active_nonprompt_policy(
            ("renamed_unknown_central2022",),
            configuration_source="pre_executor_injected_invalid",
        )
