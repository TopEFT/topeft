"""Helpers for loading Run 2 scenario definitions and channel groups."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple

import yaml

from topeft.modules.paths import topeft_path

RUN2_SCENARIOS_PATH = Path(topeft_path("../analysis/metadata/run2_scenarios.yaml"))
GROUP_METADATA_PATHS = (
    Path(topeft_path("../analysis/metadata/metadata_TOP_22_006.yaml")),
    Path(topeft_path("../analysis/metadata/metadata_tau_analysis.yaml")),
    Path(topeft_path("../analysis/metadata/metadata_run2_groups.yaml")),
)


@dataclass(frozen=True)
class ScenarioDefinition:
    """Immutable container describing a Run 2 scenario."""

    name: str
    group_names: Tuple[str, ...]

    @property
    def groups(self) -> Tuple[str, ...]:
        """Return the ordered group names for this scenario."""

        return self.group_names


def load_run2_scenarios() -> Dict[str, ScenarioDefinition]:
    """Return the canonical Run 2 scenarios keyed by name."""

    return dict(_load_run2_scenarios())


def resolve_scenario_groups(name: str) -> ScenarioDefinition:
    """Return the :class:`ScenarioDefinition` matching ``name``."""

    scenarios = load_run2_scenarios()
    try:
        return scenarios[name]
    except KeyError as exc:
        known = ", ".join(sorted(scenarios))
        raise KeyError(
            f"Scenario {name!r} not found in analysis/metadata/run2_scenarios.yaml. "
            f"Available scenarios: {known or '<none>'}."
        ) from exc


def known_run2_scenarios() -> Tuple[str, ...]:
    """Return the scenario names enumerated in ``run2_scenarios.yaml``."""

    return tuple(_load_run2_scenarios().keys())


def is_run2_scenario(name: str) -> bool:
    """Return ``True`` when ``name`` is defined in ``run2_scenarios.yaml``."""

    if not name:
        return False
    return name in _load_run2_scenarios()


def load_run2_channels_for_scenario(name: str) -> Mapping[str, object]:
    """Return metadata suitable for :class:`ChannelMetadataHelper`.

    The returned mapping follows the ``metadata['channels']`` structure and
    contains only the groups requested by ``name``.  Scenario information is
    included so callers can still rely on ``ChannelMetadataHelper`` helpers that
    need the scenario → group map.
    """

    scenario = resolve_scenario_groups(name)
    available_groups = _load_group_metadata()

    selected_groups: Dict[str, Mapping[str, object]] = {}
    for group_name in scenario.groups:
        try:
            metadata = available_groups[group_name]
        except KeyError as exc:
            raise KeyError(
                f"Scenario {scenario.name!r} references unknown group {group_name!r}."
            ) from exc
        if group_name not in selected_groups:
            selected_groups[group_name] = metadata

    return {
        "groups": selected_groups,
        "scenarios": [
            {
                "name": scenario.name,
                "groups": scenario.groups,
            }
        ],
    }


@lru_cache(maxsize=1)
def _load_run2_scenarios() -> Mapping[str, ScenarioDefinition]:
    payload = _read_yaml_mapping(RUN2_SCENARIOS_PATH)
    scenarios_section = payload.get("scenarios") or {}
    if not isinstance(scenarios_section, Mapping):
        raise TypeError(
            f"'scenarios' in {RUN2_SCENARIOS_PATH} must be a mapping of scenario definitions"
        )

    scenarios: MutableMapping[str, ScenarioDefinition] = {}
    for scenario_name, definition in scenarios_section.items():
        if not isinstance(scenario_name, str):
            raise TypeError("Scenario names must be strings")
        if not isinstance(definition, Mapping):
            raise TypeError(
                f"Scenario definition for {scenario_name!r} must be a mapping"
            )
        raw_groups = definition.get("groups", [])
        if raw_groups is None:
            raw_groups = []
        if isinstance(raw_groups, str):
            raw_groups = [raw_groups]
        if not isinstance(raw_groups, (list, tuple)):
            raise TypeError(
                f"Scenario {scenario_name!r} groups must be a sequence of group names"
            )

        normalized_groups = _normalize_group_names(raw_groups)
        scenarios[scenario_name] = ScenarioDefinition(
            name=scenario_name,
            group_names=normalized_groups,
        )

    return scenarios


def _normalize_group_names(group_names: Sequence[object]) -> Tuple[str, ...]:
    seen = set()
    ordered = []
    for name in group_names:
        if not isinstance(name, str):
            raise TypeError("Scenario group names must be strings")
        stripped = name.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        ordered.append(stripped)
    return tuple(ordered)


@lru_cache(maxsize=1)
def _load_group_metadata() -> Dict[str, Mapping[str, object]]:
    """Return the canonical channel-group metadata keyed by group name."""

    groups: Dict[str, Mapping[str, object]] = {}
    for metadata_path in GROUP_METADATA_PATHS:
        payload = _read_yaml_mapping(metadata_path)
        channels = payload.get("channels") or {}
        if not isinstance(channels, Mapping):
            raise TypeError(
                f"'channels' in {metadata_path} must be a mapping with 'groups'"
            )
        available = channels.get("groups") or {}
        if not isinstance(available, Mapping):
            raise TypeError(
                f"'channels.groups' in {metadata_path} must be a mapping of group definitions"
            )
        for group_name, metadata in available.items():
            if not isinstance(group_name, str):
                raise TypeError(
                    f"Channel group names in {metadata_path} must be strings"
                )
            if not isinstance(metadata, Mapping):
                raise TypeError(
                    f"Channel group {group_name!r} in {metadata_path} must be a mapping"
                )
            if group_name in groups:
                # Some groups (e.g. shared control regions) appear in multiple
                # metadata bundles with cosmetic differences. Retain the first
                # encounter to provide deterministic behaviour.
                continue
            groups[group_name] = metadata
    return groups


def _read_yaml_mapping(path: Path) -> Mapping[str, object]:
    """Return ``yaml.safe_load`` output ensuring it is a mapping."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} must contain a YAML mapping at the top level")
    return payload
