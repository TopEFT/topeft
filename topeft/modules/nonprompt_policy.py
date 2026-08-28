"""Exact canonical physics-family authority for nonprompt prompt subtraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


NONPROMPT_POLICY_SCHEMA_VERSION = 1
PROMPT_SUBTRACTION_MEMBER = "prompt_subtraction_member"
EXPLICIT_NONPROMPT_EXCLUSION = "explicit_nonprompt_exclusion"
DATA_OR_NON_MC = "data_or_non_mc"

RUN2_YEARS = ("UL16APV", "UL16", "UL17", "UL18")
RUN3_YEARS = ("2022", "2022EE", "2023", "2023BPix")
CANONICAL_NONPROMPT_YEARS = RUN2_YEARS + RUN3_YEARS
_YEAR_PATTERN = re.compile(
    r"^(?P<base>.*?)(?P<year>UL16APV|UL16|UL17|UL18|2022EE|2022|2023BPix|2023)$"
)


class nonprompt_policy_error(ValueError):
    """The active process universe cannot be certified for nonprompt use."""


@dataclass(frozen=True)
class nonprompt_alias_definition:
    raw_process_base: str
    canonical_family: str
    run_eras: tuple[str, ...]
    policy_role: str
    policy_reason: str
    source_of_alias: str


@dataclass(frozen=True)
class resolved_nonprompt_process:
    raw_process_label: str
    process_base: str
    canonical_family: str
    year: str
    run_era: str
    policy_role: str
    policy_reason: str
    source_of_alias: str

    @property
    def is_prompt_member(self) -> bool:
        return self.policy_role == PROMPT_SUBTRACTION_MEMBER

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_process_label": self.raw_process_label,
            "normalized_or_year_stripped_label": self.process_base,
            "canonical_family": self.canonical_family,
            "run_era": self.run_era,
            "year": self.year,
            "policy_role": self.policy_role,
            "policy_reason": self.policy_reason,
            "is_prompt_member": self.is_prompt_member,
            "source_of_alias": self.source_of_alias,
            "resolution_status": "resolved",
            "ambiguity_count": 1,
        }


@dataclass(frozen=True)
class certified_nonprompt_policy:
    configuration_source: str
    resolutions: tuple[resolved_nonprompt_process, ...]

    @property
    def resolved_prompt_process_set(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                resolution.raw_process_label
                for resolution in self.resolutions
                if resolution.is_prompt_member
            )
        )

    @property
    def explicit_exclusions(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                resolution.raw_process_label
                for resolution in self.resolutions
                if resolution.policy_role == EXPLICIT_NONPROMPT_EXCLUSION
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NONPROMPT_POLICY_SCHEMA_VERSION,
            "configuration_source": self.configuration_source,
            "resolved_prompt_process_set": list(self.resolved_prompt_process_set),
            "explicit_exclusions": list(self.explicit_exclusions),
            "resolutions": [resolution.to_dict() for resolution in self.resolutions],
        }


def _alias(
    raw_process_base: str,
    canonical_family: str,
    run_eras: Sequence[str],
    *,
    role: str = PROMPT_SUBTRACTION_MEMBER,
    reason: str = "historical_selective_prompt_policy",
    source: str = "maintained_nonprompt_policy",
) -> nonprompt_alias_definition:
    return nonprompt_alias_definition(
        raw_process_base=raw_process_base,
        canonical_family=canonical_family,
        run_eras=tuple(run_eras),
        policy_role=role,
        policy_reason=reason,
        source_of_alias=source,
    )


_BOTH_ERAS = ("run2", "run3")
_RUN2_ONLY = ("run2",)
_RUN3_ONLY = ("run3",)


DEFAULT_NONPROMPT_ALIAS_DEFINITIONS = (
    _alias("data", "collision_data", _BOTH_ERAS, role=DATA_OR_NON_MC, reason="observed_collision_data"),
    _alias("TTTo2L2Nu_central", "dileptonic_ttbar", _RUN2_ONLY, reason="run2_historical_prompt_member"),
    _alias("TTto2L2Nu_central", "dileptonic_ttbar", _RUN3_ONLY, reason="run3_additive_alias_of_run2_dileptonic_ttbar"),
    _alias("TTToSemiLeptonic_central", "semileptonic_ttbar", _RUN2_ONLY),
    _alias("TTtoLNu2Q_central", "semileptonic_ttbar", _RUN3_ONLY, reason="run3_additive_alias_of_run2_semileptonic_ttbar"),
    _alias("TTZToLL_M1to10_central", "low_mass_ttz", _RUN2_ONLY),
    _alias("TWZToLL_central", "twz", _RUN2_ONLY),
    _alias("TWZ_Tto2Q_WtoLNu_Zto2L_central", "twz", _RUN3_ONLY, reason="run3_resolved_twz_decay_mode"),
    _alias("TWZ_TtoLNu_Wto2Q_Zto2L_central", "twz", _RUN3_ONLY, reason="run3_resolved_twz_decay_mode"),
    _alias("TWZ_TtoLNu_WtoLNu_Zto2L_central", "twz", _RUN3_ONLY, reason="run3_resolved_twz_decay_mode"),
    _alias("TTGamma_central", "ttgamma", _RUN2_ONLY),
    _alias("TTGJets_central", "ttgamma", _RUN2_ONLY),
    _alias("TTG-1Jets_PTG-10to100_central", "ttgamma", _RUN3_ONLY, reason="run3_resolved_ttgamma_pt_bin"),
    _alias("TTG-1Jets_PTG-100to200_central", "ttgamma", _RUN3_ONLY, reason="run3_resolved_ttgamma_pt_bin"),
    _alias("TTG-1Jets_PTG-200_central", "ttgamma", _RUN3_ONLY, reason="run3_resolved_ttgamma_pt_bin"),
    _alias("WWW_4F_central", "www", _RUN2_ONLY),
    _alias("WWW_central", "www", _RUN3_ONLY, reason="run3_additive_alias_of_run2_www"),
    _alias("WWZ_4F_central", "wwz", _RUN2_ONLY),
    _alias("WWZ_central", "wwz", _RUN3_ONLY, reason="run3_additive_alias_of_run2_wwz"),
    _alias("WZTo3LNu_central", "inclusive_wz", _BOTH_ERAS),
    _alias("WZZ_central", "wzz", _BOTH_ERAS),
    _alias("WZZ_ext_central", "wzz", _RUN2_ONLY),
    _alias("ZZTo4L_central", "zz_to_4l_prompt_family", _BOTH_ERAS),
    _alias("ZZZ_central", "zzz", _BOTH_ERAS),
    _alias("tHq_private", "private_thq", _BOTH_ERAS),
    _alias("tllq_private", "private_tllq", _BOTH_ERAS),
    _alias("ttHJet_private", "private_tth", _RUN2_ONLY),
    _alias("ttH_private", "private_tth", _RUN3_ONLY, reason="run3_additive_alias_of_run2_private_tth"),
    _alias("ttllJet_private", "private_ttll", _RUN2_ONLY),
    _alias("ttll_private", "private_ttll", _RUN3_ONLY, reason="run3_additive_alias_of_run2_private_ttll"),
    _alias("ttlnuJet_private", "private_ttlnu", _RUN2_ONLY),
    _alias("ttlnu_private", "private_ttlnu", _RUN3_ONLY, reason="run3_additive_alias_of_run2_private_ttlnu"),
    _alias("tttt_private", "private_tttt", _BOTH_ERAS),
    _alias("tZq_central", "private_tllq", _RUN2_ONLY),
    _alias("ttHJet_central", "private_tth", _RUN2_ONLY),
    _alias("ttH_central", "private_tth", _RUN2_ONLY),
    _alias("ttW_central", "private_ttlnu", _RUN2_ONLY),
    _alias("ttZ_central", "private_ttll", _RUN2_ONLY),
    _alias("tttt_central", "private_tttt", _RUN2_ONLY),
    _alias("ggToZZTo2e2mu_central", "zz_to_4l_prompt_family", _RUN3_ONLY, reason="inherits_run2_zzto4l_prompt_role"),
    _alias("ggToZZTo2e2nu_central", "zz_to_4l_prompt_family", _RUN3_ONLY, reason="inherits_run2_zzto4l_prompt_role"),
    _alias("ggToZZTo2e2tau_central", "zz_to_4l_prompt_family", _RUN3_ONLY, reason="inherits_run2_zzto4l_prompt_role"),
    _alias("ggToZZTo2mu2tau_central", "zz_to_4l_prompt_family", _RUN3_ONLY, reason="inherits_run2_zzto4l_prompt_role"),
    _alias("ggToZZTo4e_central", "zz_to_4l_prompt_family", _RUN3_ONLY, reason="inherits_run2_zzto4l_prompt_role"),
    _alias("ggToZZTo4mu_central", "zz_to_4l_prompt_family", _RUN3_ONLY, reason="inherits_run2_zzto4l_prompt_role"),
    _alias("ggToZZTo4tau_central", "zz_to_4l_prompt_family", _RUN3_ONLY, reason="inherits_run2_zzto4l_prompt_role"),
    _alias("WWTo2L2Nu_central", "ww_dilepton", _BOTH_ERAS, role=EXPLICIT_NONPROMPT_EXCLUSION, reason="intentional_nonprompt_policy_exclusion"),
    _alias("WZto3LNu-2Jets_central", "electroweak_wz_two_jets", _RUN3_ONLY, role=EXPLICIT_NONPROMPT_EXCLUSION, reason="intentional_distinct_wz_two_jet_exclusion"),
    _alias("ggToZZTo2mu2nu_central", "ggzz_two_mu_two_nu", _RUN3_ONLY, role=EXPLICIT_NONPROMPT_EXCLUSION, reason="intentional_ggzz_two_mu_two_nu_exclusion"),
)


def split_year_qualified_process(process_label: str) -> tuple[str, str, str]:
    """Split one exact maintained process label into base, year, and run era."""

    if not isinstance(process_label, str) or not process_label:
        raise nonprompt_policy_error(
            f"NONPROMPT-POLICY-E001: process label must be a nonempty string; observed={process_label!r}."
        )
    match = _YEAR_PATTERN.fullmatch(process_label)
    if match is None:
        raise nonprompt_policy_error(
            "NONPROMPT-POLICY-E001: active process lacks an exact maintained year suffix; "
            f"process={process_label!r} supported_years={list(CANONICAL_NONPROMPT_YEARS)}."
        )
    base = match.group("base")
    year = match.group("year")
    run_era = "run2" if year in RUN2_YEARS else "run3"
    return base, year, run_era


def _validated_alias_index(
    definitions: Sequence[nonprompt_alias_definition],
) -> dict[str, nonprompt_alias_definition]:
    aliases: dict[str, nonprompt_alias_definition] = {}
    family_roles: dict[str, set[str]] = {}
    for definition in definitions:
        if definition.raw_process_base in aliases:
            previous = aliases[definition.raw_process_base]
            raise nonprompt_policy_error(
                "NONPROMPT-POLICY-E002: ambiguous duplicate alias definition; "
                f"alias={definition.raw_process_base!r} "
                f"assignments={[previous.canonical_family, definition.canonical_family]}."
            )
        if not definition.run_eras or not set(definition.run_eras) <= {"run2", "run3"}:
            raise nonprompt_policy_error(
                "NONPROMPT-POLICY-E003: alias has invalid run-era coverage; "
                f"alias={definition.raw_process_base!r} run_eras={definition.run_eras}."
            )
        if definition.policy_role not in {
            PROMPT_SUBTRACTION_MEMBER,
            EXPLICIT_NONPROMPT_EXCLUSION,
            DATA_OR_NON_MC,
        }:
            raise nonprompt_policy_error(
                "NONPROMPT-POLICY-E003: alias has no explicit supported nonprompt role; "
                f"alias={definition.raw_process_base!r} role={definition.policy_role!r}."
            )
        aliases[definition.raw_process_base] = definition
        if definition.policy_role != DATA_OR_NON_MC:
            family_roles.setdefault(definition.canonical_family, set()).add(
                definition.policy_role
            )
    conflicts = {
        family: sorted(roles)
        for family, roles in family_roles.items()
        if len(roles) != 1
    }
    if conflicts:
        raise nonprompt_policy_error(
            "NONPROMPT-POLICY-E003: conflicting policy assignments for canonical family; "
            f"conflicts={conflicts}."
        )
    return aliases


def _active_process_metadata(
    samples_or_processes: Mapping[str, Mapping[str, Any]] | Iterable[str],
) -> tuple[tuple[str, ...], dict[str, bool | None]]:
    if not isinstance(samples_or_processes, Mapping):
        processes = tuple(sorted(set(str(value) for value in samples_or_processes)))
        return processes, {process: None for process in processes}

    process_data_flags: dict[str, set[bool]] = {}
    for dataset, sample in samples_or_processes.items():
        if not isinstance(sample, Mapping):
            raise nonprompt_policy_error(
                f"NONPROMPT-POLICY-E004: sample metadata must be a mapping; dataset={dataset!r}."
            )
        process = sample.get("histAxisName")
        is_data = sample.get("isData")
        if not isinstance(process, str) or not process or not isinstance(is_data, bool):
            raise nonprompt_policy_error(
                "NONPROMPT-POLICY-E004: sample requires exact histAxisName and boolean isData; "
                f"dataset={dataset!r} histAxisName={process!r} isData={is_data!r}."
            )
        process_data_flags.setdefault(process, set()).add(is_data)
    conflicts = {
        process: sorted(flags)
        for process, flags in process_data_flags.items()
        if len(flags) != 1
    }
    if conflicts:
        raise nonprompt_policy_error(
            "NONPROMPT-POLICY-E005: one raw process label is assigned conflicting data/MC identities; "
            f"conflicts={conflicts}."
        )
    return (
        tuple(sorted(process_data_flags)),
        {process: next(iter(flags)) for process, flags in process_data_flags.items()},
    )


def certify_active_nonprompt_policy(
    samples_or_processes: Mapping[str, Mapping[str, Any]] | Iterable[str],
    *,
    configuration_source: str,
    required_canonical_families: Iterable[str] = (),
    configured_prompt_aliases: Sequence[str] | None = None,
    alias_definitions: Sequence[nonprompt_alias_definition] = DEFAULT_NONPROMPT_ALIAS_DEFINITIONS,
) -> certified_nonprompt_policy:
    """Resolve every exact active identity and reject all silent fall-through."""

    aliases = _validated_alias_index(alias_definitions)
    configured_prompt_alias_set = None
    if configured_prompt_aliases is not None:
        validate_legacy_prompt_compatibility(configured_prompt_aliases)
        configured_prompt_alias_set = set(configured_prompt_aliases)
    processes, data_flags = _active_process_metadata(samples_or_processes)
    resolutions = []
    for process in processes:
        process_base, year, run_era = split_year_qualified_process(process)
        definition = aliases.get(process_base)
        if definition is None:
            raise nonprompt_policy_error(
                "NONPROMPT-POLICY-E006: unknown active process alias has no explicit nonprompt role; "
                f"process={process!r} process_base={process_base!r} "
                f"configuration_source={configuration_source!r}."
            )
        if (
            definition.policy_role == PROMPT_SUBTRACTION_MEMBER
            and configured_prompt_alias_set is not None
            and process_base not in configured_prompt_alias_set
        ):
            raise nonprompt_policy_error(
                "NONPROMPT-POLICY-E008: configured prompt policy omits a canonical "
                f"prompt alias; process={process!r} process_base={process_base!r}."
            )
        if run_era not in definition.run_eras:
            raise nonprompt_policy_error(
                "NONPROMPT-POLICY-E006: active alias is not valid for its exact run era; "
                f"process={process!r} canonical_family={definition.canonical_family!r} "
                f"run_era={run_era!r} allowed_run_eras={list(definition.run_eras)} "
                f"configuration_source={configuration_source!r}."
            )
        is_data = data_flags[process]
        if is_data is True and definition.policy_role != DATA_OR_NON_MC:
            raise nonprompt_policy_error(
                "NONPROMPT-POLICY-E005: data sample resolves to an MC nonprompt role; "
                f"process={process!r} role={definition.policy_role!r}."
            )
        if is_data is False and definition.policy_role == DATA_OR_NON_MC:
            raise nonprompt_policy_error(
                "NONPROMPT-POLICY-E005: MC sample resolves to data_or_non_mc; "
                f"process={process!r}."
            )
        resolutions.append(
            resolved_nonprompt_process(
                raw_process_label=process,
                process_base=process_base,
                canonical_family=definition.canonical_family,
                year=year,
                run_era=run_era,
                policy_role=definition.policy_role,
                policy_reason=definition.policy_reason,
                source_of_alias=definition.source_of_alias,
            )
        )
    active_families = {resolution.canonical_family for resolution in resolutions}
    missing_families = sorted(set(required_canonical_families) - active_families)
    if missing_families:
        raise nonprompt_policy_error(
            "NONPROMPT-POLICY-E007: configured source universe requires prompt families "
            "without a resolvable active process label; "
            f"missing_canonical_families={missing_families} "
            f"active_processes={list(processes)} configuration_source={configuration_source!r}."
        )
    return certified_nonprompt_policy(
        configuration_source=configuration_source,
        resolutions=tuple(resolutions),
    )


def canonical_prompt_aliases() -> tuple[str, ...]:
    """Return exact base aliases on the maintained prompt side of the policy."""

    _validated_alias_index(DEFAULT_NONPROMPT_ALIAS_DEFINITIONS)
    return tuple(
        sorted(
            definition.raw_process_base
            for definition in DEFAULT_NONPROMPT_ALIAS_DEFINITIONS
            if definition.policy_role == PROMPT_SUBTRACTION_MEMBER
        )
    )


def explicit_exclusion_aliases() -> tuple[str, ...]:
    return tuple(
        sorted(
            definition.raw_process_base
            for definition in DEFAULT_NONPROMPT_ALIAS_DEFINITIONS
            if definition.policy_role == EXPLICIT_NONPROMPT_EXCLUSION
        )
    )


def validate_legacy_prompt_compatibility(entries: Sequence[str]) -> None:
    """Keep params.json additive and subordinate to the canonical authority."""

    if any(not isinstance(entry, str) or not entry for entry in entries):
        raise nonprompt_policy_error(
            "NONPROMPT-POLICY-E008: prompt_subtraction_samples contains a non-string entry."
        )
    missing = sorted(set(canonical_prompt_aliases()) - set(entries))
    forbidden = sorted(set(explicit_exclusion_aliases()) & set(entries))
    if missing or forbidden:
        raise nonprompt_policy_error(
            "NONPROMPT-POLICY-E008: legacy prompt_subtraction_samples is not aligned "
            "with the canonical family authority; "
            f"missing_prompt_aliases={missing} explicit_exclusions_present={forbidden}."
        )
