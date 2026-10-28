"""Canonical paths for SR campaign figures and machine evidence.

This module is deliberately independent of the renderer and campaign runners.
It does not read histograms or create directories.
"""

from __future__ import annotations

from pathlib import Path


_periods = {
    ("run2", "combined"): "run2",
    ("run2", "UL16APV"): "2016APV",
    ("run2", "UL16"): "2016",
    ("run2", "UL17"): "2017",
    ("run2", "UL18"): "2018",
    ("run3", "combined"): "run3",
    ("run3", "2022"): "2022",
    ("run3", "2022EE"): "2022EE",
    ("run3", "2023"): "2023",
    ("run3", "2023BPix"): "2023BPix",
}


def _component(value: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value or "/" in value:
        raise ValueError(f"Invalid plot path component: {value!r}")
    return value


def normalize_plot_period(run: str, view: str) -> str:
    try:
        return _periods[(run, view)]
    except KeyError as error:
        raise ValueError(f"Unsupported SR plot period: {run}/{view}") from error


def canonical_period_directory(root: Path, period: str) -> Path:
    if period not in _periods.values():
        raise ValueError(f"Unsupported canonical plot period: {period!r}")
    return Path(root) / period / "merged"


def canonical_plot_destinations_for_period(
    root: Path, period: str, category: str, filename: str, variable: str,
) -> tuple[Path, ...]:
    _component(category)
    _component(filename)
    if not filename.endswith(".png"):
        raise ValueError("Canonical SR campaign figures must be PNG")
    merged = canonical_period_directory(root, period) / category / filename
    if variable == "njets":
        return merged, merged.parents[2] / "merged_njets" / category / filename
    return (merged,)


def canonical_diagnostics_directory(root: Path) -> Path:
    return Path(root) / "_metadata" / "diagnostics"


def canonical_observed_path(root: Path, relative: str) -> Path:
    """Resolve an observed PNG through the same authority used by the renderer."""
    parts = Path(relative).parts
    if Path(relative).is_absolute() or len(parts) != 4:
        raise ValueError(f"Invalid canonical observed path: {relative!r}")
    period, view, category, filename = parts
    destinations = canonical_plot_destinations_for_period(
        root, period, category, filename,
        "njets" if filename.endswith("_njets_fitting.png") else "other",
    )
    target = Path(root) / relative
    if view not in {"merged", "merged_njets"} or target not in destinations:
        raise ValueError(f"Noncanonical observed path: {relative!r}")
    return target
