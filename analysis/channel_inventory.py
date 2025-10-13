"""Utility to expand channel definitions into fully qualified names.

This script loads the JSON data from ``topeft/channels/ch_lst.json`` and
expands every combination of lepton channel, flavour, and jet bin.  It then
splits the resulting names by the available application masks so we can check
which entries are only selected with ``isSR_*`` and which also enable
``isAR_*`` masks.

The output is written both as JSON (for programmatic use) and as Markdown (for
quick inspection) under ``analysis/``.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
CHANNEL_JSON = REPO_ROOT / "topeft" / "channels" / "ch_lst.json"
OUTPUT_JSON = Path(__file__).with_name("channel_inventory.json")
OUTPUT_MD = Path(__file__).with_name("channel_inventory.md")


def load_channel_config() -> Mapping[str, Mapping[str, Mapping[str, object]]]:
    with CHANNEL_JSON.open() as handle:
        return json.load(handle)


def jet_spec_to_label(spec: str) -> str:
    """Convert a jet bin specifier into the suffix used in channel names."""
    if spec.startswith("="):
        return f"{spec[1:]}j"
    if spec.startswith(">"):
        try:
            lower = int(spec[1:])
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Cannot parse jet specifier '{spec}'") from exc
        return f"{lower + 1}j"
    if spec.startswith("<"):
        return f"lt{spec[1:]}j"
    raise ValueError(f"Unhandled jet specifier '{spec}'")


def build_channel_name(base: str, flavour: str, jet_spec: str) -> str:
    """Insert the flavour after the leading token and append the jet label."""
    tokens = base.split("_")
    # keep the first element untouched, inject the flavour, and reattach the rest
    tokens = [tokens[0], flavour, *tokens[1:]] if len(tokens) > 1 else [base, flavour]
    return "_".join(tokens + [jet_spec_to_label(jet_spec)])


def classify_application_masks(appl_lst: Iterable[str]) -> str:
    sr = [mask for mask in appl_lst if mask.startswith("isSR_")]
    ar = [mask for mask in appl_lst if mask.startswith("isAR_")]
    if sr and not ar:
        return "sr_only"
    if sr and ar:
        return "sr_plus_ar"
    if not sr and ar:
        return "ar_only"
    return "unclassified"


def expand_section(section_name: str, section_cfg: Mapping[str, Mapping[str, object]]):
    section_result: Dict[str, object] = {
        "categories": {},
        "by_application": defaultdict(set),
    }

    for category, cfg in section_cfg.items():
        lep_chan_lst = cfg.get("lep_chan_lst") or []
        lep_flav_lst = cfg.get("lep_flav_lst") or [""]
        jet_lst = cfg.get("jet_lst") or [""]
        appl_lst = cfg.get("appl_lst") or []
        appl_class = classify_application_masks(appl_lst)

        channels: List[str] = []
        for lep_chan in lep_chan_lst:
            if not lep_chan:
                continue
            base_name = lep_chan[0]
            for flavour in lep_flav_lst:
                for jet_spec in jet_lst:
                    if not jet_spec:
                        continue
                    channel_name = build_channel_name(base_name, flavour, jet_spec)
                    channels.append(channel_name)
                    section_result["by_application"][appl_class].add(channel_name)

        section_result["categories"][category] = {
            "appl_lst": appl_lst,
            "appl_lst_data": cfg.get("appl_lst_data", []),
            "application_class": appl_class,
            "channels": sorted(channels),
        }

    # convert the aggregated sets to sorted lists for serialisation
    section_result["by_application"] = {
        key: sorted(values)
        for key, values in section_result["by_application"].items()
    }
    return section_result


def build_inventory(sections: Iterable[str]) -> Dict[str, object]:
    full_cfg = load_channel_config()
    inventory = {}
    for section in sections:
        section_cfg = full_cfg.get(section)
        if section_cfg is None:
            raise KeyError(f"Section '{section}' not found in channel config")
        inventory[section] = expand_section(section, section_cfg)
    return inventory


def write_json_inventory(inventory: Mapping[str, object]) -> None:
    OUTPUT_JSON.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")


def write_markdown_inventory(inventory: Mapping[str, object]) -> None:
    lines: List[str] = ["# Channel inventory", ""]
    for section, section_info in inventory.items():
        lines.append(f"## {section}")
        lines.append("")
        by_application = section_info.get("by_application", {})
        if by_application:
            lines.append("### Aggregated by application mask")
            for cls_key, channels in by_application.items():
                if not channels:
                    continue
                cls_label = {
                    "sr_only": "Only isSR_* masks",
                    "sr_plus_ar": "isSR_* and isAR_* masks",
                    "ar_only": "Only isAR_* masks",
                    "unclassified": "Unclassified masks",
                }.get(cls_key, cls_key)
                lines.append(f"- **{cls_label}**")
                for name in channels:
                    lines.append(f"  - {name}")
            lines.append("")

        lines.append("### Per-category breakdown")
        for category, category_info in section_info.get("categories", {}).items():
            lines.append(f"#### {category}")
            lines.append(
                f"- Application class: {category_info.get('application_class', 'n/a')}"
            )
            appl_lst = category_info.get("appl_lst") or []
            if appl_lst:
                lines.append("- Application masks: " + ", ".join(appl_lst))
            appl_lst_data = category_info.get("appl_lst_data") or []
            if appl_lst_data:
                lines.append("- Data application masks: " + ", ".join(appl_lst_data))
            lines.append("- Channels:")
            for name in category_info.get("channels", []):
                lines.append(f"  - {name}")
            lines.append("")
        lines.append("")

    OUTPUT_MD.write_text("\n".join(lines))


def main() -> None:
    sections = ["TAU_CH_LST_SR", "TAU_CH_LST_CR"]
    inventory = build_inventory(sections)
    write_json_inventory(inventory)
    write_markdown_inventory(inventory)


if __name__ == "__main__":
    main()
