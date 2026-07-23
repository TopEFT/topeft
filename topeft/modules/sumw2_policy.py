"""Metadata-driven selective statistical-companion policy resolution."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


SUMW2_PROVENANCE_SCHEMA_VERSION = 2
LEGACY_SUMW2_PROVENANCE_SCHEMA_VERSION = 1
SUMW2_MODES = frozenset(
    {
        "production",
        "production_central",
        "taufitter",
        "full_diagnostics",
        "disabled",
        "full_custom",
    }
)
RULE_MODES = frozenset(
    {"production", "production_central", "taufitter", "full_custom"}
)
PRODUCTION_SIGNAL_SAMPLE_PROFILES = {
    "production": "private",
    "production_central": "central",
}
RULE_KEYS = frozenset(
    {
        "dataset_names",
        "dataset_prefixes",
        "process_names",
        "process_prefixes",
        "variables",
    }
)
POLICY_KEYS = frozenset({"mode", "rules"})


@dataclass(frozen=True, order=True)
class sumw2_target:
    dataset: str
    process: str
    family: str

    def to_dict(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "process": self.process,
            "family": self.family,
        }


@dataclass(frozen=True)
class sumw2_mode_resolution:
    source: str
    requested_mode: str
    resolved_mode: str
    signal_sample_profile: str
    sumw2_storage: Mapping[str, Any]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class normalized_sumw2_rule:
    dataset_names: tuple[str, ...] = ()
    dataset_prefixes: tuple[str, ...] = ()
    process_names: tuple[str, ...] = ()
    process_prefixes: tuple[str, ...] = ()
    variables: tuple[str, ...] = ()
    variables_wildcard: bool = True

    def to_dict(self) -> dict[str, list[str]]:
        output = {}
        for field_name in (
            "dataset_names",
            "dataset_prefixes",
            "process_names",
            "process_prefixes",
        ):
            values = getattr(self, field_name)
            if values:
                output[field_name] = list(values)
        if not self.variables_wildcard:
            output["variables"] = list(self.variables)
        return output

    def identity(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class resolved_sumw2_policy:
    source: str
    requested_mode: str
    resolved_mode: str
    signal_sample_profile: str
    normalized_rules: tuple[normalized_sumw2_rule, ...]
    runtime_histogram_families: tuple[str, ...]
    resolved_datasets: tuple[str, ...]
    resolved_processes: tuple[str, ...]
    resolved_targets: tuple[sumw2_target, ...]
    warnings: tuple[str, ...] = ()
    schema_version: int = SUMW2_PROVENANCE_SCHEMA_VERSION
    _target_set: frozenset[sumw2_target] = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_target_set", frozenset(self.resolved_targets))

    def selects(self, dataset: str, process: str, family: str) -> bool:
        return sumw2_target(dataset, process, family) in self._target_set

    def selects_family(self, family: str) -> bool:
        return any(target.family == family for target in self.resolved_targets)

    def selected_families(self) -> tuple[str, ...]:
        selected = {target.family for target in self.resolved_targets}
        return tuple(
            family
            for family in self.runtime_histogram_families
            if family in selected
        )

    def selected_processes(self, family: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    target.process
                    for target in self.resolved_targets
                    if target.family == family
                }
            )
        )

    def to_provenance(self) -> dict[str, Any]:
        provenance = {
            "schema_version": self.schema_version,
            "source": self.source,
            "requested_mode": self.requested_mode,
            "normalized_rules": [rule.to_dict() for rule in self.normalized_rules],
            "runtime_histogram_families": list(self.runtime_histogram_families),
            "resolved_datasets": list(self.resolved_datasets),
            "resolved_processes": list(self.resolved_processes),
            "resolved_targets": [target.to_dict() for target in self.resolved_targets],
            "warnings": list(self.warnings),
        }
        if self.schema_version == SUMW2_PROVENANCE_SCHEMA_VERSION:
            provenance["resolved_mode"] = self.resolved_mode
            provenance["signal_sample_profile"] = self.signal_sample_profile
        return provenance

    def identity(self) -> str:
        return json.dumps(self.to_provenance(), sort_keys=True, separators=(",", ":"))


def _require_unique_string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"SUMW2-E003: '{field_name}' must be a nonempty list of strings."
        )
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(
            f"SUMW2-E003: '{field_name}' entries must be nonempty strings."
        )
    if len(value) != len(set(value)):
        raise ValueError(f"SUMW2-E003: '{field_name}' contains duplicate values.")
    return tuple(value)


def _normalize_rule(
    raw_rule: Any,
    *,
    runtime_families: tuple[str, ...],
    registered_families: frozenset[str],
    internal_2d_axes: frozenset[str],
) -> normalized_sumw2_rule:
    if not isinstance(raw_rule, Mapping):
        raise ValueError("SUMW2-E003: every sumw2_storage rule must be a mapping.")
    unknown = sorted(set(raw_rule) - RULE_KEYS)
    if unknown:
        raise ValueError(
            "SUMW2-E001: unknown sumw2_storage rule field(s): " + ", ".join(unknown)
        )

    normalized = {}
    for field_name in RULE_KEYS - {"variables"}:
        normalized[field_name] = (
            _require_unique_string_list(raw_rule[field_name], field_name)
            if field_name in raw_rule
            else ()
        )

    variables_wildcard = "variables" not in raw_rule
    variables = ()
    if not variables_wildcard:
        variables = _require_unique_string_list(raw_rule["variables"], "variables")
        for family in variables:
            if family in internal_2d_axes and family not in registered_families:
                raise ValueError(
                    f"SUMW2-E006: '{family}' is an internal 2D component axis; "
                    "select its top-level family instead."
                )
            if family not in registered_families:
                raise ValueError(f"SUMW2-E004: unknown histogram family '{family}'.")
            if family not in runtime_families:
                raise ValueError(
                    f"SUMW2-E005: histogram family '{family}' is registered but not "
                    "selected for this run."
                )

    return normalized_sumw2_rule(
        dataset_names=normalized["dataset_names"],
        dataset_prefixes=normalized["dataset_prefixes"],
        process_names=normalized["process_names"],
        process_prefixes=normalized["process_prefixes"],
        variables=variables,
        variables_wildcard=variables_wildcard,
    )


def _validate_sample_metadata(samples: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    if not isinstance(samples, Mapping):
        raise TypeError("samples must be a dataset-keyed mapping.")
    dataset_processes = {}
    for dataset, sample in samples.items():
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("Dataset keys must be nonempty strings.")
        if not isinstance(sample, Mapping):
            raise TypeError(f"Sample metadata for '{dataset}' must be a mapping.")
        process = sample.get("histAxisName")
        is_data = sample.get("isData")
        wc_names = sample.get("WCnames")
        if not isinstance(process, str) or not process:
            raise ValueError(f"Sample '{dataset}' has invalid histAxisName metadata.")
        if not isinstance(is_data, bool):
            raise ValueError(f"Sample '{dataset}' has invalid isData metadata.")
        if not isinstance(wc_names, list) or any(
            not isinstance(wc, str) or not wc for wc in wc_names
        ):
            raise ValueError(f"Sample '{dataset}' has invalid WCnames metadata.")
        if len(wc_names) != len(set(wc_names)):
            raise ValueError(f"Sample '{dataset}' has duplicate WCnames metadata.")
        if is_data and wc_names:
            raise ValueError(
                f"Data sample '{dataset}' cannot declare WC-dependent content."
            )
        dataset_processes[dataset] = process
    return dataset_processes


def resolve_nominal_component_availability(
    samples: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    _validate_sample_metadata(samples)
    has_scalar = False
    has_eft = False
    for sample in samples.values():
        if sample["isData"] or not sample["WCnames"]:
            has_scalar = True
        else:
            has_eft = True
    return {"scalar": has_scalar, "eft": has_eft}


def _selector_matches(
    value: str,
    *,
    exact: Sequence[str],
    prefixes: Sequence[str],
) -> bool:
    if not exact and not prefixes:
        return True
    return value in exact or any(value.startswith(prefix) for prefix in prefixes)


def _validate_selector_coverage(
    rule: normalized_sumw2_rule,
    *,
    datasets: tuple[str, ...],
    processes: tuple[str, ...],
) -> None:
    for exact in rule.dataset_names:
        if exact not in datasets:
            raise ValueError(f"SUMW2-E007: dataset selector '{exact}' matched nothing.")
    for prefix in rule.dataset_prefixes:
        if not any(value.startswith(prefix) for value in datasets):
            raise ValueError(f"SUMW2-E007: dataset prefix '{prefix}' matched nothing.")
    for exact in rule.process_names:
        if exact not in processes:
            raise ValueError(f"SUMW2-E007: process selector '{exact}' matched nothing.")
    for prefix in rule.process_prefixes:
        if not any(value.startswith(prefix) for value in processes):
            raise ValueError(f"SUMW2-E007: process prefix '{prefix}' matched nothing.")


def _normalize_consumer_requirements(
    requirements: Iterable[sumw2_target | Sequence[str]],
) -> frozenset[sumw2_target]:
    output = set()
    for requirement in requirements:
        if isinstance(requirement, sumw2_target):
            target = requirement
        else:
            values = tuple(requirement)
            if len(values) != 3:
                raise ValueError("Consumer requirement entries must contain dataset/process/family.")
            target = sumw2_target(*values)
        output.add(target)
    return frozenset(output)


def resolve_sumw2_storage_mode(
    sumw2_storage: Mapping[str, Any] | None,
    *,
    sumw2_storage_present: bool | None = None,
    legacy_no_sumw2_present: bool = False,
    legacy_no_sumw2_value: bool = False,
) -> sumw2_mode_resolution:
    """Resolve the public mode and its signal profile without selecting samples."""

    if sumw2_storage_present is None:
        sumw2_storage_present = sumw2_storage is not None
    policy_warnings = []
    if sumw2_storage_present:
        if legacy_no_sumw2_present:
            raise ValueError(
                "Explicit sumw2_storage and explicit legacy no_sumw2 cannot be combined."
            )
        if not isinstance(sumw2_storage, Mapping):
            raise ValueError("SUMW2-E001: sumw2_storage must be a mapping.")
        unknown = sorted(set(sumw2_storage) - POLICY_KEYS)
        if unknown:
            raise ValueError(
                "SUMW2-E001: unknown sumw2_storage field(s): " + ", ".join(unknown)
            )
        mode = sumw2_storage.get("mode", "production")
        source = (
            "explicit" if "mode" in sumw2_storage else "implicit_production_default"
        )
        normalized_storage = sumw2_storage
    elif legacy_no_sumw2_present and legacy_no_sumw2_value:
        mode = "disabled"
        source = "legacy_no_sumw2"
        policy_warnings.append(
            "SUMW2-W001: explicit legacy no_sumw2=true maps to disabled and is deprecated."
        )
        normalized_storage = {}
    elif legacy_no_sumw2_present:
        mode = "full_diagnostics"
        source = "legacy_no_sumw2_false"
        policy_warnings.append(
            "SUMW2-W001: explicit legacy no_sumw2=false maps to full_diagnostics and is deprecated."
        )
        normalized_storage = {}
    else:
        mode = "production"
        source = "implicit_production_default"
        policy_warnings.append(
            "SUMW2-W001: sumw2_storage is absent; using the production default."
        )
        normalized_storage = {}

    if mode not in SUMW2_MODES:
        raise ValueError(
            f"SUMW2-E001: unknown sumw2_storage mode {mode!r}; expected one of "
            + ", ".join(sorted(SUMW2_MODES))
        )
    return sumw2_mode_resolution(
        source=source,
        requested_mode=mode,
        resolved_mode=mode,
        signal_sample_profile=PRODUCTION_SIGNAL_SAMPLE_PROFILES.get(
            mode, "unrestricted"
        ),
        sumw2_storage=normalized_storage,
        warnings=tuple(policy_warnings),
    )


def resolve_sumw2_storage_policy(
    sumw2_storage: Mapping[str, Any] | None,
    *,
    samples: Mapping[str, Mapping[str, Any]],
    runtime_families: Sequence[str],
    axes_info: Mapping[str, Any],
    axes_info_2d: Mapping[str, Any],
    analysis_mode: str = "standard",
    sumw2_storage_present: bool | None = None,
    legacy_no_sumw2_present: bool = False,
    legacy_no_sumw2_value: bool = False,
    consumer_requirements: Iterable[sumw2_target | Sequence[str]] = (),
    implicit_production_requirements: Iterable[
        sumw2_target | Sequence[str]
    ] = (),
    mode_resolution: sumw2_mode_resolution | None = None,
) -> resolved_sumw2_policy:
    dataset_processes = _validate_sample_metadata(samples)
    datasets = tuple(sorted(dataset_processes))
    processes = tuple(sorted(set(dataset_processes.values())))
    runtime_families = tuple(runtime_families)
    if len(runtime_families) != len(set(runtime_families)):
        raise ValueError("Runtime histogram families must be unique and ordered.")

    registered_families = frozenset(axes_info) | frozenset(axes_info_2d)
    unknown_runtime = sorted(set(runtime_families) - registered_families)
    if unknown_runtime:
        raise ValueError(
            "Runtime histogram families are not registered in axes.py: "
            + ", ".join(unknown_runtime)
        )
    internal_2d_axes = frozenset(
        axis_cfg["name"]
        for family_cfg in axes_info_2d.values()
        for axis_cfg in family_cfg["axes"]
    )

    independently_resolved_mode = resolve_sumw2_storage_mode(
        sumw2_storage,
        sumw2_storage_present=sumw2_storage_present,
        legacy_no_sumw2_present=legacy_no_sumw2_present,
        legacy_no_sumw2_value=legacy_no_sumw2_value,
    )
    if mode_resolution is None:
        mode_resolution = independently_resolved_mode
    elif mode_resolution != independently_resolved_mode:
        raise ValueError(
            "SUMW2-E001: supplied mode resolution disagrees with the active "
            "sumw2_storage and legacy flags."
        )
    mode = mode_resolution.resolved_mode
    source = mode_resolution.source
    policy_warnings = list(mode_resolution.warnings)
    sumw2_storage = mode_resolution.sumw2_storage

    rules_present = "rules" in sumw2_storage
    raw_rules = sumw2_storage.get("rules")
    implicit_production = (
        mode == "production" and source == "implicit_production_default"
    )
    if mode in RULE_MODES:
        if (
            not implicit_production
            and (not rules_present or not isinstance(raw_rules, list) or not raw_rules)
        ):
            raise ValueError(f"SUMW2-E002: mode '{mode}' requires nonempty rules.")
        if implicit_production and rules_present and (
            not isinstance(raw_rules, list) or not raw_rules
        ):
            raise ValueError(
                "SUMW2-E002: implicit production rules, when provided, must be nonempty."
            )
    elif rules_present:
        raise ValueError(f"SUMW2-E002: mode '{mode}' forbids rules.")

    if (analysis_mode == "taufitter") != (mode == "taufitter"):
        raise ValueError(
            "SUMW2-E009: analysis_mode=taufitter and sumw2_storage.mode=taufitter "
            "must be selected together."
        )

    normalized_rules = []
    if mode in RULE_MODES and rules_present:
        for raw_rule in raw_rules:
            normalized_rules.append(
                _normalize_rule(
                    raw_rule,
                    runtime_families=runtime_families,
                    registered_families=registered_families,
                    internal_2d_axes=internal_2d_axes,
                )
            )
        identities = [rule.identity() for rule in normalized_rules]
        if len(identities) != len(set(identities)):
            raise ValueError("SUMW2-E003: structurally duplicate rules are forbidden.")
        normalized_rules.sort(key=normalized_sumw2_rule.identity)

    targets = set()
    if mode == "full_diagnostics":
        for dataset in datasets:
            process = dataset_processes[dataset]
            for family in runtime_families:
                targets.add(sumw2_target(dataset, process, family))
    elif mode in RULE_MODES and rules_present:
        for rule in normalized_rules:
            _validate_selector_coverage(rule, datasets=datasets, processes=processes)
            families = runtime_families if rule.variables_wildcard else rule.variables
            rule_targets = set()
            for dataset in datasets:
                process = dataset_processes[dataset]
                if not _selector_matches(
                    dataset,
                    exact=rule.dataset_names,
                    prefixes=rule.dataset_prefixes,
                ):
                    continue
                if not _selector_matches(
                    process,
                    exact=rule.process_names,
                    prefixes=rule.process_prefixes,
                ):
                    continue
                for family in families:
                    rule_targets.add(sumw2_target(dataset, process, family))
            if not rule_targets:
                raise ValueError("SUMW2-E007: a sumw2_storage rule resolved zero targets.")
            overlap = targets & rule_targets
            if overlap:
                examples = ", ".join(
                    f"{target.dataset}/{target.process}/{target.family}"
                    for target in sorted(overlap)[:5]
                )
                raise ValueError(
                    "SUMW2-E008: sumw2_storage rules overlap on concrete targets: "
                    + examples
                )
            targets.update(rule_targets)

    if implicit_production and not rules_present:
        targets.update(
            _normalize_consumer_requirements(implicit_production_requirements)
        )

    required_targets = _normalize_consumer_requirements(consumer_requirements)
    missing_requirements = sorted(required_targets - targets)
    if missing_requirements:
        examples = ", ".join(
            f"{target.dataset}/{target.process}/{target.family}"
            for target in missing_requirements[:5]
        )
        raise ValueError(
            "SUMW2-E010: active consumer requirements are not covered: " + examples
        )

    family_order = {family: index for index, family in enumerate(runtime_families)}
    ordered_targets = tuple(
        sorted(
            targets,
            key=lambda target: (
                target.dataset,
                target.process,
                family_order[target.family],
            ),
        )
    )
    for message in policy_warnings:
        warnings.warn(message, UserWarning, stacklevel=2)
    return resolved_sumw2_policy(
        source=source,
        requested_mode=mode,
        resolved_mode=mode_resolution.resolved_mode,
        signal_sample_profile=mode_resolution.signal_sample_profile,
        normalized_rules=tuple(normalized_rules),
        runtime_histogram_families=runtime_families,
        resolved_datasets=datasets,
        resolved_processes=processes,
        resolved_targets=ordered_targets,
        warnings=tuple(policy_warnings),
    )


def resolved_policy_from_provenance(
    provenance: Mapping[str, Any],
) -> resolved_sumw2_policy:
    if not isinstance(provenance, Mapping):
        raise ValueError("sumw2_storage_provenance must be a mapping.")
    common_required = {
        "schema_version",
        "source",
        "requested_mode",
        "normalized_rules",
        "runtime_histogram_families",
        "resolved_datasets",
        "resolved_processes",
        "resolved_targets",
        "warnings",
    }
    schema_version = provenance.get("schema_version")
    if schema_version == SUMW2_PROVENANCE_SCHEMA_VERSION:
        required = common_required | {"resolved_mode", "signal_sample_profile"}
    elif schema_version == LEGACY_SUMW2_PROVENANCE_SCHEMA_VERSION:
        required = common_required
    else:
        raise ValueError(
            "Unsupported sumw2 provenance schema version "
            f"{schema_version!r}."
        )
    missing = sorted(required - set(provenance))
    unknown = sorted(set(provenance) - required)
    if missing or unknown:
        raise ValueError(
            "Invalid sumw2_storage_provenance fields; missing={} unknown={}.".format(
                missing, unknown
            )
        )
    if provenance["source"] not in {
        "explicit",
        "implicit_production_default",
        "legacy_no_sumw2",
        "legacy_no_sumw2_false",
    }:
        raise ValueError("Invalid sumw2 provenance source.")
    if provenance["requested_mode"] not in SUMW2_MODES:
        raise ValueError("Invalid requested_mode in sumw2 provenance.")
    if (
        schema_version == LEGACY_SUMW2_PROVENANCE_SCHEMA_VERSION
        and provenance["requested_mode"] == "production_central"
    ):
        raise ValueError(
            "Legacy sumw2 provenance predates production_central and cannot encode it."
        )
    if schema_version == SUMW2_PROVENANCE_SCHEMA_VERSION:
        if provenance["resolved_mode"] != provenance["requested_mode"]:
            raise ValueError("sumw2 provenance requested_mode/resolved_mode mismatch.")
        expected_profile = PRODUCTION_SIGNAL_SAMPLE_PROFILES.get(
            provenance["resolved_mode"], "unrestricted"
        )
        if provenance["signal_sample_profile"] != expected_profile:
            raise ValueError("Invalid signal_sample_profile in sumw2 provenance.")

    rules = []
    for raw_rule in provenance["normalized_rules"]:
        if not isinstance(raw_rule, Mapping) or set(raw_rule) - RULE_KEYS:
            raise ValueError("Invalid normalized rule in sumw2 provenance.")
        rules.append(
            normalized_sumw2_rule(
                dataset_names=tuple(raw_rule.get("dataset_names", ())),
                dataset_prefixes=tuple(raw_rule.get("dataset_prefixes", ())),
                process_names=tuple(raw_rule.get("process_names", ())),
                process_prefixes=tuple(raw_rule.get("process_prefixes", ())),
                variables=tuple(raw_rule.get("variables", ())),
                variables_wildcard="variables" not in raw_rule,
            )
        )
    targets = []
    for raw_target in provenance["resolved_targets"]:
        if not isinstance(raw_target, Mapping) or set(raw_target) != {
            "dataset",
            "process",
            "family",
        }:
            raise ValueError("Invalid resolved target in sumw2 provenance.")
        targets.append(
            sumw2_target(
                raw_target["dataset"], raw_target["process"], raw_target["family"]
            )
        )
    policy = resolved_sumw2_policy(
        source=provenance["source"],
        requested_mode=provenance["requested_mode"],
        resolved_mode=provenance.get("resolved_mode", provenance["requested_mode"]),
        signal_sample_profile=provenance.get(
            "signal_sample_profile",
            PRODUCTION_SIGNAL_SAMPLE_PROFILES.get(
                provenance["requested_mode"], "unrestricted"
            ),
        ),
        normalized_rules=tuple(rules),
        runtime_histogram_families=tuple(provenance["runtime_histogram_families"]),
        resolved_datasets=tuple(provenance["resolved_datasets"]),
        resolved_processes=tuple(provenance["resolved_processes"]),
        resolved_targets=tuple(targets),
        warnings=tuple(provenance["warnings"]),
        schema_version=schema_version,
    )
    if policy.to_provenance() != dict(provenance):
        raise ValueError("sumw2 provenance is not in canonical deterministic form.")
    return policy


def validate_policy_identity(
    first: resolved_sumw2_policy | Mapping[str, Any],
    second: resolved_sumw2_policy | Mapping[str, Any],
) -> None:
    first_policy = (
        first
        if isinstance(first, resolved_sumw2_policy)
        else resolved_policy_from_provenance(first)
    )
    second_policy = (
        second
        if isinstance(second, resolved_sumw2_policy)
        else resolved_policy_from_provenance(second)
    )
    if first_policy.identity() != second_policy.identity():
        raise ValueError("Input files have different resolved sumw2 policy identities.")
