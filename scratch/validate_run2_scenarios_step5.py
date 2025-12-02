"""Validate run2_scenarios.yaml by summarizing group counts per scenario."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import yaml

TOPEFT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = TOPEFT_ROOT / "analysis" / "metadata"
SCENARIO_FILE = METADATA_DIR / "run2_scenarios.yaml"

GROUP_SOURCES = {
    "TOP22_006_CH_LST_SR": METADATA_DIR / "metadata_TOP_22_006.yaml",
    "CH_LST_CR": METADATA_DIR / "metadata_TOP_22_006.yaml",
    "TAU_CH_LST_SR": METADATA_DIR / "metadata_tau_analysis.yaml",
    "TAU_CH_LST_CR": METADATA_DIR / "metadata_tau_analysis.yaml",
    "OFFZ_TAU_SPLIT_CH_LST_SR": METADATA_DIR / "metadata_run2_groups.yaml",
    "ALL_CH_LST_SR": METADATA_DIR / "metadata_run2_groups.yaml",
    "OFFZ_FWD_SPLIT_CH_LST_SR": METADATA_DIR / "metadata_run2_groups.yaml",
}

_CACHE: Dict[Path, Mapping[str, object]] = {}


def load_yaml(path: Path) -> Mapping[str, object]:
    cached = _CACHE.get(path)
    if cached is not None:
        return cached
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    _CACHE[path] = data
    return data


def fetch_group(name: str) -> Mapping[str, object]:
    source = GROUP_SOURCES.get(name)
    if source is None:
        raise KeyError(f"No metadata source registered for group {name}")
    metadata = load_yaml(source)
    channels = metadata.get("channels") or {}
    groups = channels.get("groups") or {}
    group = groups.get(name)
    if group is None:
        raise KeyError(f"Group {name} not found in {source}")
    return group


def count_group_channels(group: Mapping[str, object]) -> int:
    total = 0
    for category in group.get("regions", []):
        regions: List[Mapping[str, object]] = category.get("region_definitions", [])
        flavors: Iterable[str] = category.get("lepton_flavors", [])
        jet_bins: List[str] = category.get("jet_bins") or [None]
        flavor_count = max(len(list(flavors)), 1)
        jet_count = len(jet_bins)
        total += len(regions) * flavor_count * jet_count
    return total


def load_scenarios() -> Mapping[str, Mapping[str, object]]:
    data = load_yaml(SCENARIO_FILE)
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, Mapping):
        raise TypeError("run2_scenarios.yaml must define a mapping under 'scenarios'")
    return scenarios


def main() -> None:
    scenarios = load_scenarios()
    print(f"Loaded scenarios from {SCENARIO_FILE}")
    print("SCENARIO                 GROUP_COUNT   APPROX_CHANNELS")

    for scenario_name, scenario_def in scenarios.items():
        group_names: List[str] = list(scenario_def.get("groups", []))
        group_contributions: List[tuple[str, int]] = []
        total = 0
        for group_name in group_names:
            group = fetch_group(group_name)
            group_count = count_group_channels(group)
            group_contributions.append((group_name, group_count))
            total += group_count
        print(f"{scenario_name:<24}{len(group_contributions):>12}{total:>18}")
        print(f"Scenario {scenario_name}:")
        for group_name, count in group_contributions:
            print(f"  - {group_name}: {count}")
        print(f"  → total: {total}\n")

if __name__ == "__main__":
    main()
