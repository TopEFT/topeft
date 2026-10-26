"""Canonical paths for SR campaign figures and machine evidence.

This module is deliberately independent of the renderer and campaign runners.
It does not read histograms or create directories.
"""

from __future__ import annotations

import argparse
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


def canonical_row_directory(root: Path, run: str, view: str) -> Path:
    return canonical_period_directory(root, normalize_plot_period(run, view))


def canonical_period_directory(root: Path, period: str) -> Path:
    if period not in _periods.values():
        raise ValueError(f"Unsupported canonical plot period: {period!r}")
    return Path(root) / period / "merged"


def canonical_plot_destinations(
    root: Path, run: str, view: str, category: str, filename: str, variable: str,
) -> tuple[Path, ...]:
    return canonical_plot_destinations_for_period(
        root, normalize_plot_period(run, view), category, filename, variable,
    )


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


def canonical_receipt_path(root: Path, unit_id: str) -> Path:
    return Path(root) / "_metadata" / "receipts" / (_component(unit_id) + ".receipt.json")


def canonical_diagnostics_path(root: Path, *components: str) -> Path:
    if not components:
        raise ValueError("A diagnostics filename is required")
    return canonical_diagnostics_directory(root) / Path(*(_component(part) for part in components))


def canonical_diagnostics_directory(root: Path) -> Path:
    return Path(root) / "_metadata" / "diagnostics"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("run", choices=("run2", "run3"))
    parser.add_argument("view")
    args = parser.parse_args()
    period = normalize_plot_period(args.run, args.view)
    print(f"{period}\t{canonical_row_directory(args.root, args.run, args.view)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
