"""Validate Run 2 scenario composition by enumerating datacard channels."""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Mapping, Sequence

from topeft.modules.channel_metadata import ChannelGroup, ChannelMetadataHelper
from topeft.modules.run2_scenarios import (
    load_run2_channels_for_scenario,
    load_run2_scenarios,
)

DEFAULT_SAMPLE_SCENARIOS = (
    "TOP_22_006",
    "offz_tau_analysis",
    "offz_fwd_analysis",
    "all_analysis",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        default=None,
        help="Scenario name to validate. Defaults to all known scenarios.",
    )
    parser.add_argument(
        "--sample-scenario",
        action="append",
        dest="sample_scenarios",
        default=None,
        help=(
            "Scenario name to include in the sample channel list. "
            "Defaults to TOP_22_006, offz_tau_analysis, offz_fwd_analysis, all_analysis."
        ),
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=8,
        help="Number of channel names to print for each sample scenario.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    scenario_map = load_run2_scenarios()

    if not scenario_map:
        raise SystemExit("No scenarios are defined in analysis/metadata/run2_scenarios.yaml.")

    selected_names = _resolve_selected_scenarios(args.scenarios, scenario_map)
    summaries: List[ScenarioSummary] = []
    channels_cache: Dict[str, List[str]] = {}

    for scenario_name in selected_names:
        metadata = load_run2_channels_for_scenario(scenario_name)
        helper = ChannelMetadataHelper(metadata)
        sr_channels = collect_datacard_channels(helper, scenario_name)
        scenario_def = scenario_map[scenario_name]
        summaries.append(
            ScenarioSummary(
                name=scenario_name,
                total_groups=len(scenario_def.groups),
                sr_channels=len(sr_channels),
            )
        )
        channels_cache[scenario_name] = sr_channels

    _print_summary_table(summaries)
    _print_sample_channels(
        channels_cache,
        desired_scenarios=args.sample_scenarios,
        fallback_order=selected_names,
        sample_count=args.sample_count,
    )


def _resolve_selected_scenarios(
    requested: Iterable[str] | None, scenario_map: Mapping[str, object]
) -> List[str]:
    if not requested:
        return list(scenario_map.keys())

    ordered: List[str] = []
    for name in requested:
        normalized = (name or "").strip()
        if not normalized:
            continue
        if normalized not in scenario_map:
            known = ", ".join(scenario_map.keys())
            raise SystemExit(
                f"Scenario {normalized!r} is not defined. Known scenarios: {known}."
            )
        ordered.append(normalized)
    return ordered


def _print_summary_table(summaries: Sequence["ScenarioSummary"]) -> None:
    header = (
        "SCENARIO".ljust(18)
        + "  "
        + "GROUPS".rjust(8)
        + "  "
        + "SR_CHANNELS".rjust(13)
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f"{summary.name.ljust(18)}  "
            f"{summary.total_groups:>8}  "
            f"{summary.sr_channels:>13}"
        )
    print()


def _print_sample_channels(
    channels_cache: Mapping[str, List[str]],
    *,
    desired_scenarios: Iterable[str] | None,
    fallback_order: Sequence[str],
    sample_count: int,
) -> None:
    sample_targets = _resolve_sample_targets(desired_scenarios, fallback_order)
    if not sample_targets:
        return

    print("Sample channel names:")
    for scenario_name in sample_targets:
        channels = channels_cache.get(scenario_name)
        if channels is None:
            continue
        sample = ", ".join(channels[:sample_count])
        print(f"  {scenario_name}: {sample}")


def _resolve_sample_targets(
    requested: Iterable[str] | None, fallback_order: Sequence[str]
) -> List[str]:
    if requested:
        candidates = list(requested)
    else:
        candidates = list(DEFAULT_SAMPLE_SCENARIOS)
    resolved: List[str] = []
    seen = set()
    for candidate in candidates:
        normalized = (candidate or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(normalized)
    if not resolved:
        resolved = list(fallback_order)
    return resolved


class ScenarioSummary:
    """Lightweight record bundling summary values for display."""

    __slots__ = ("name", "total_groups", "sr_channels")

    def __init__(self, name: str, total_groups: int, sr_channels: int):
        self.name = name
        self.total_groups = total_groups
        self.sr_channels = sr_channels


def collect_datacard_channels(
    helper: ChannelMetadataHelper, scenario_name: str
) -> List[str]:
    """Return the canonical datacard channel list for ``scenario_name``."""

    sr_channels: List[str] = []
    group_names = helper.selected_group_names([scenario_name])
    for group_name in group_names:
        if group_name.endswith("_CR"):
            continue
        group = helper.group(group_name)
        sr_channels.extend(_collect_group_channels(group, scenario_name))

    seen = set()
    ordered = []
    for name in sorted(sr_channels):
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _collect_group_channels(group: ChannelGroup, scenario_name: str) -> List[str]:
    analysis_mode = _analysis_mode_for_group(group, scenario_name)
    channels: List[str] = []
    for category in group.categories():
        jet_bins = _normalize_jet_bins(category.jet_bins)
        if not jet_bins:
            continue
        for region in category.region_definitions:
            for jet in jet_bins:
                suffix = determine_histogram_suffix(region, jet, analysis_mode)
                channels.append(f"{region.name}_{jet}j_{suffix}")
    return channels


def _normalize_jet_bins(values: Iterable[object]) -> List[str]:
    jets: List[str] = []
    for value in values:
        jet = extract_number(value)
        if jet:
            jets.append(jet)
    return jets


def extract_number(item: object) -> str:
    """Return the digit substring used in legacy jet binning."""

    if item is None:
        return ""
    text = str(item)
    digits = "".join(char for char in text if char.isdigit())
    return digits


def determine_histogram_suffix(region, jet_value, analysis_mode):
    """Infer the histogram name suffix for a region based on metadata tags."""

    tags = set(region.tags)
    subchannel = region.subchannel or ""
    channel = region.channel or ""
    name = region.name
    jet_value = str(jet_value) if jet_value is not None else ""

    if "onZ_tau" in tags:
        return "ptz_wtau"

    if analysis_mode == "fwd":
        if channel and "2lss_fwd" in channel:
            return "lt"
        if "~fwdjet_mask" in tags and channel == "2lss":
            return "lt"

    if subchannel == "3l_onZ" or channel == "3l_onZ":
        if name == "3l_onZ_2b" and jet_value not in {"4", "5"}:
            return "lj0pt"
        return "ptz"

    if analysis_mode == "offZdivision" and (
        "offZ_low" in subchannel or "offZ_high" in subchannel
    ):
        return "ptz"

    if "offZ_tau" in tags:
        return "ptz"

    if "2l_onZ" in tags or "2l_onZ_as" in tags:
        return "ptz"

    if analysis_mode == "tau" and channel == "2los":
        return "ptz"

    return "lj0pt"


def _analysis_mode_for_group(group: ChannelGroup, scenario_name: str) -> str:
    """Return the legacy analysis-mode tag for ``group``."""

    features = set(group.features)
    if "offz_split" in features:
        return "offZdivision"
    if "requires_tau" in features or scenario_name == "tau_analysis":
        return "tau"
    if "requires_forward" in features or scenario_name == "fwd_analysis":
        return "fwd"
    return "top22006"


if __name__ == "__main__":
    main()
