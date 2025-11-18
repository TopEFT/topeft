"""Plotter for the flip measurement tuple-keyed histogram output."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, MutableMapping

import argparse
import gzip

import matplotlib.pyplot as plt
import numpy as np
import cloudpickle
import hist

from .plot_utils import (
    TupleHistogramEntry,
    load_tuple_histogram_entries,
    summarise_by_variable,
)


PT_BINS = (0.0, 30.0, 45.0, 60.0, 100.0, 200.0)
ABSETA_BINS = (0.0, 0.4, 0.8, 1.1, 1.4, 1.6, 1.9, 2.2, 2.5)

SCALE_DICT = {
    "UL16APV": 1.0,
    "UL16": 1.0,
    "UL17": 1.0,
    "UL18": 1.0,
}


def load_histograms(path: str) -> Iterable[TupleHistogramEntry]:
    return load_tuple_histogram_entries(path)


def determine_year(sample: str) -> str | None:
    if "UL16APV" in sample:
        return "UL16APV"
    for year in ("UL16", "UL17", "UL18"):
        if year in sample and "APV" not in sample:
            return year
    return None


def group_by_year(
    entries: Iterable[TupleHistogramEntry],
) -> Dict[str, Dict[str, hist.Hist]]:
    grouped: Dict[str, Dict[str, hist.Hist]] = defaultdict(dict)

    # Aggregate histograms per sample/flip status first so that duplicate entries
    # for the same tuple accumulate before we project to the year level.
    variable_map = summarise_by_variable(entries, systematic="nominal", application=None)
    application_map: MutableMapping[str, MutableMapping[str, MutableMapping[str, hist.Hist]]] = variable_map.get("ptabseta", {})
    sample_map: MutableMapping[str, MutableMapping[str, hist.Hist]] = application_map.get(
        "flip_measurement", application_map.get("", {})
    )

    for sample, flip_map in sample_map.items():
        year = determine_year(sample)
        if year is None:
            continue
        for flipstatus, histogram in flip_map.items():
            if flipstatus not in ("truthFlip", "truthNoFlip"):
                continue
            hist_copy = histogram.copy()
            existing = grouped[year].get(flipstatus)
            if existing is None:
                grouped[year][flipstatus] = hist_copy
            else:
                grouped[year][flipstatus] = existing + hist_copy
    return grouped


def make_ratio_hist(ratio_arr: np.ndarray) -> hist.Hist:
    ratio_hist = hist.Hist(
        hist.axis.Variable(PT_BINS, name="pt", label="pt"),
        hist.axis.Variable(ABSETA_BINS, name="abseta", label="abseta"),
        storage=hist.storage.Double(),
    )
    ratio_hist[...] = ratio_arr
    return ratio_hist


def make_2d_fig(histo: hist.Hist, xaxis_var: str, save_name: str, title: str | None = None) -> None:
    title_str = title if title is not None else save_name
    histo.plot2d(xaxis=xaxis_var)
    plt.title(title_str)
    plt.savefig(save_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filepath",
        default="histos/flipMR_TopEFT.pkl.gz",
        help="path of file with histograms",
    )
    args = parser.parse_args()

    tuple_histograms = load_histograms(args.filepath)
    grouped = group_by_year(tuple_histograms)

    for year in ("UL16APV", "UL16", "UL17", "UL18"):
        flip_map = grouped.get(year, {})
        if not flip_map:
            print(f"No histograms found for year {year}, skipping.")
            continue
        if "truthFlip" not in flip_map or "truthNoFlip" not in flip_map:
            print(f"Incomplete histogram set for year {year}, skipping.")
            continue

        hist_flip = flip_map["truthFlip"]
        hist_noflip = flip_map["truthNoFlip"]

        flip_values = hist_flip.values()
        noflip_values = hist_noflip.values()
        denom = flip_values + noflip_values
        ratio_values = np.divide(
            flip_values,
            denom,
            out=np.zeros_like(flip_values, dtype=float),
            where=denom != 0,
        )

        hist_ratio = make_ratio_hist(ratio_values)

        make_2d_fig(hist_flip, "pt", f"{year}_truth_flip")
        make_2d_fig(hist_noflip, "pt", f"{year}_truth_noflip")
        make_2d_fig(
            hist_ratio,
            "pt",
            f"{year}_truth_ratio",
            "Flip ratio = flip/(flip+noflip)",
        )

        scaled_ratio = hist_ratio.copy()
        scaled_ratio[...] = ratio_values * SCALE_DICT[year]
        make_2d_fig(
            scaled_ratio,
            "pt",
            f"{year}_truth_ratio_scaled",
            "Flip ratio = flip/(flip+noflip)",
        )

        save_pkl_str = f"flip_probs_topcoffea_{year}.pkl.gz"
        with gzip.open(save_pkl_str, "wb") as fout:
            cloudpickle.dump(scaled_ratio, fout)


if __name__ == "__main__":
    main()
