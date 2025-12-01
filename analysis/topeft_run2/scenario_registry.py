"""Registry mapping scenario names to metadata YAML files.

This module centralises where the Run 2 CLI should look for the metadata
describing each supported scenario.  Keeping the mapping in one location makes
it straightforward to audit which YAML bundles are considered production ready
while preventing ad-hoc path strings from spreading throughout the workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

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


def resolve_scenario_path(name: str) -> str:
    """Return the metadata YAML path for ``name``.

    Raises:
        ValueError: if ``name`` is not registered.  The exception message lists
            the available scenarios to help steer the user.
    """

    try:
        rel_path = _SCENARIO_REGISTRY[name]
        return str((_REPO_ROOT / rel_path).resolve())
    except KeyError as exc:  # pragma: no cover - simple guard
        available = ", ".join(sorted(known_scenarios()))
        raise ValueError(
            f"Unknown scenario '{name}'. Known scenarios: {available}"
        ) from exc


__all__ = ["known_scenarios", "resolve_scenario_path"]
