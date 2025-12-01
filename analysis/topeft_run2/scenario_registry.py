"""Registry mapping scenario names to metadata YAML files.

This module centralises where the Run 2 CLI should look for the metadata
describing each supported scenario.  Keeping the mapping in one location makes
it straightforward to audit which YAML bundles are considered production ready
while preventing ad-hoc path strings from spreading throughout the workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Paths are stored relative to the topeft repository root.  The CLI converts
# them into absolute paths via ``topeft.modules.paths.topeft_path`` before
# passing them to the workflow.
_SCENARIO_REGISTRY: Dict[str, str] = {
    "TOP_22_006": "analysis/metadata/metadata_TOP_22_006.yaml",
    "tau_analysis": "analysis/metadata/metadata_tau_analysis.yaml",
    "fwd_analysis": "analysis/metadata/metadata_fwd_analysis.yaml",
    # NOTE: intentionally not exposing "all_analysis" yet.
}

_REPO_ROOT = Path(__file__).resolve().parents[2]


def known_scenarios() -> Iterable[str]:
    """Return the scenario names wired into the registry."""

    return _SCENARIO_REGISTRY.keys()


@dataclass(frozen=True)
class ScenarioResolution:
    """Describes the metadata path lookup for a scenario."""

    metadata_path: str
    known_scenarios: Sequence[str]


def resolve_scenario_choice(name: str) -> ScenarioResolution:
    """Return the metadata path and known names for ``name``.

    Args:
        name: Scenario identifier supplied via CLI/YAML.

    Raises:
        ValueError: if ``name`` is unknown. The exception message includes the
            available scenarios so the caller can surface a user-friendly hint.
    """

    known: List[str] = list(known_scenarios())
    rel_path = _SCENARIO_REGISTRY.get(name)
    if rel_path is None:  # pragma: no cover - simple guard
        available = ", ".join(known)
        raise ValueError(
            f"Unknown scenario '{name}'. Known scenarios: {available}"
        )
    metadata_path = str((_REPO_ROOT / rel_path).resolve())
    return ScenarioResolution(metadata_path=metadata_path, known_scenarios=known)


def resolve_scenario_path(name: str) -> str:
    """Return only the metadata path for ``name`` for backwards compatibility."""

    return resolve_scenario_choice(name).metadata_path


__all__ = [
    "ScenarioResolution",
    "known_scenarios",
    "resolve_scenario_choice",
    "resolve_scenario_path",
]
