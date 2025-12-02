"""Validation helper comparing canonical Run-2 groups vs legacy JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import yaml

TOPEFT_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = TOPEFT_ROOT / "scratch" / "ch_lst_step5.json"

METADATA_FILES = {
    "TOP22_006_CH_LST_SR": TOPEFT_ROOT / "analysis" / "metadata" / "metadata_TOP_22_006.yaml",
    "CH_LST_CR": TOPEFT_ROOT / "analysis" / "metadata" / "metadata_TOP_22_006.yaml",
    "TAU_CH_LST_SR": TOPEFT_ROOT / "analysis" / "metadata" / "metadata_tau_analysis.yaml",
    "TAU_CH_LST_CR": TOPEFT_ROOT / "analysis" / "metadata" / "metadata_tau_analysis.yaml",
    "OFFZ_TAU_SPLIT_CH_LST_SR": TOPEFT_ROOT / "analysis" / "metadata" / "metadata_run2_groups.yaml",
    "ALL_CH_LST_SR": TOPEFT_ROOT / "analysis" / "metadata" / "metadata_run2_groups.yaml",
}

DERIVED_ONLY_GROUP = "OFFZ_FWD_SPLIT_CH_LST_SR"
DERIVED_FILE = TOPEFT_ROOT / "analysis" / "metadata" / "metadata_run2_groups.yaml"

def normalize_jet_category(jet_cat: str | None) -> str:
    if not jet_cat:
        return ""
    jet_cat = str(jet_cat).strip()
    if jet_cat.startswith("="):
        tag = "exactly_"
    elif jet_cat.startswith("<"):
        tag = "atmost_"
    elif jet_cat.startswith(">"):
        tag = "atleast_"
    else:
        raise ValueError(f"jet_cat {jet_cat} misses =,<,> !")
    return f"{tag}{jet_cat[1:]}j"

def load_yaml(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}

def fetch_yaml_group(path: Path, group_name: str) -> Mapping[str, object]:
    data = load_yaml(path)
    channels = (data.get("channels") or {})
    groups = channels.get("groups") or {}
    if group_name not in groups:
        raise KeyError(f"Group {group_name} not found in {path}")
    return groups[group_name]

def count_from_yaml(group: Mapping[str, object]) -> int:
    total = 0
    for category in group.get("regions", []):
        region_defs: List[Mapping[str, object]] = category.get("region_definitions", [])
        flavors: Iterable[str] = category.get("lepton_flavors", [])
        jet_bins: List[str] = category.get("jet_bins") or [None]
        flavor_count = max(len(list(flavors)), 1)
        jet_count = len(jet_bins)
        total += len(region_defs) * flavor_count * jet_count
    return total

def sample_channel_names(group: Mapping[str, object], limit: int = 8) -> List[str]:
    samples: List[str] = []
    for category in group.get("regions", []):
        jet_bins = category.get("jet_bins") or [None]
        for region in category.get("region_definitions", []):
            for jet in jet_bins:
                suffix = normalize_jet_category(jet)
                entry = region.get("name")
                if suffix:
                    entry = f"{entry}_{suffix.split('_')[-1]}"
                samples.append(entry)
                if len(samples) >= limit:
                    return samples
    return samples

def count_from_json(group_data: Mapping[str, object]) -> int:
    total = 0
    for block in group_data.values():
        lep_chans = block.get("lep_chan_lst", [])
        lep_flavs = block.get("lep_flav_lst", [])
        jet_bins = block.get("jet_lst", []) or [None]
        flavor_count = max(len(lep_flavs), 1)
        total += len(lep_chans) * flavor_count * len(jet_bins)
    return total

def main() -> None:
    with JSON_PATH.open("r", encoding="utf-8") as handle:
        json_data: Dict[str, Mapping[str, object]] = json.load(handle)

    print(f"Using JSON reference: {JSON_PATH}")
    print("GROUP                          JSON_COUNT    YAML_COUNT    MATCH")

    for group_name in (
        "TOP22_006_CH_LST_SR",
        "TAU_CH_LST_SR",
        "TAU_CH_LST_CR",
        "OFFZ_TAU_SPLIT_CH_LST_SR",
        "CH_LST_CR",
        "ALL_CH_LST_SR",
    ):
        json_count = count_from_json(json_data[group_name])
        yaml_group = fetch_yaml_group(METADATA_FILES[group_name], group_name)
        yaml_count = count_from_yaml(yaml_group)
        match = "YES" if json_count == yaml_count else "NO"
        print(f"{group_name:<30}{json_count:>12}{yaml_count:>13}{match:>8}")

        if group_name in {"TOP22_006_CH_LST_SR", "ALL_CH_LST_SR"}:
            samples = sample_channel_names(yaml_group)
            print(f"  sample channels: {', '.join(samples[:6])}...")

    derived_group = fetch_yaml_group(DERIVED_FILE, DERIVED_ONLY_GROUP)
    derived_count = count_from_yaml(derived_group)
    print(f"\nDerived group {DERIVED_ONLY_GROUP}: YAML_COUNT={derived_count}")
    derived_samples = sample_channel_names(derived_group)
    if derived_samples:
        print(f"  sample channels: {', '.join(derived_samples[:6])}...")

if __name__ == "__main__":
    main()
