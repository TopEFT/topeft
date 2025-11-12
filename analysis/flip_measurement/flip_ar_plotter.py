"""Plotter for tuple-keyed flip application region histograms."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Tuple

import argparse
import gzip

import matplotlib.pyplot as plt
import hist
import cloudpickle

from topeft.modules.runner_output import SUMMARY_KEY


def load_histograms(path: str) -> Mapping[Tuple[str, str, str, str], hist.Hist]:
    with gzip.open(path, "rb") as fin:
        payload = cloudpickle.load(fin)
    if not isinstance(payload, Mapping):
        raise TypeError("Histogram payload must be a mapping")
    result: Dict[Tuple[str, str, str, str], hist.Hist] = {}
    for key, value in payload.items():
        if key == SUMMARY_KEY:
            continue
        if not isinstance(key, tuple) or len(key) != 4:
            continue
        if not isinstance(value, hist.Hist):
            continue
        result[key] = value
    if not result:
        raise ValueError("No tuple-keyed histograms found in payload")
    return result


def group_by_variable(
    histograms: Mapping[Tuple[str, str, str, str], hist.Hist]
) -> Mapping[str, Dict[str, Dict[str, hist.Hist]]]:
    grouped: Dict[str, Dict[str, Dict[str, hist.Hist]]] = defaultdict(lambda: defaultdict(dict))
    for key, histogram in histograms.items():
        variable, channel, sample, _systematic = key
        grouped[variable][sample][channel] = histogram.copy()
    return grouped


def make_fig(
    histo1: hist.Hist,
    histo2: hist.Hist | None = None,
    hup: hist.Hist | None = None,
    hdo: hist.Hist | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    if hup is not None and hdo is not None:
        hist.plot1d(hup, ax=ax, stack=False, line_opts={"color": "lightgrey"})
        hist.plot1d(hdo, ax=ax, stack=False, line_opts={"color": "lightgrey"})

    hist.plot1d(histo1, ax=ax, stack=False)
    if histo2 is not None:
        hist.plot1d(histo2, ax=ax, stack=False)

    ax.autoscale(axis="y")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot flip application histograms")
    parser.add_argument(
        "filepath",
        default="histos/flipTopEFT.pkl.gz",
        help="path of file with histograms",
    )
    parser.add_argument("--outpath", "-o", default=".", help="Path to the output directory")
    args = parser.parse_args()

    tuple_histograms = load_histograms(args.filepath)
    grouped = group_by_variable(tuple_histograms)

    outpath = Path(args.outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    for variable, sample_map in grouped.items():
        print(f"\nVariable: {variable}")
        for sample, channel_map in sample_map.items():
            ssz_hist = channel_map.get("ssz")
            osz_hist = channel_map.get("osz")
            if ssz_hist is None or osz_hist is None:
                print(f"  Missing channel histograms for sample {sample}, skipping")
                continue

            print(f"  sample_name {sample}")

            ssz_plot = ssz_hist.copy()
            osz_plot = osz_hist.copy()

            h_up = osz_plot.copy()
            h_up *= 1.3
            h_do = osz_plot.copy()
            h_do *= 0.7

            fig = make_fig(ssz_plot, histo2=osz_plot, hup=h_up, hdo=h_do)
            savename = outpath / f"{sample}_{variable}.png"
            fig.savefig(savename)
            plt.close(fig)


if __name__ == "__main__":
    main()
