"""Plotter for the flip measurement tuple-keyed histogram output."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, Tuple

import argparse
import gzip

import matplotlib.pyplot as plt
import numpy as np
import cloudpickle
import hist

from topeft.modules.runner_output import SUMMARY_KEY


PT_BINS = (0.0, 30.0, 45.0, 60.0, 100.0, 200.0)
ABSETA_BINS = (0.0, 0.4, 0.8, 1.1, 1.4, 1.6, 1.9, 2.2, 2.5)

SCALE_DICT = {
    "UL16APV": 1.0,
    "UL16": 1.0,
    "UL17": 1.0,
    "UL18": 1.0,
}


def load_histograms(path: str) -> Mapping[Tuple[str, str, str, str, str], hist.Hist]:
    with gzip.open(path, "rb") as fin:
        payload = cloudpickle.load(fin)
    if not isinstance(payload, Mapping):
        raise TypeError("Histogram payload must be a mapping")
    result: Dict[Tuple[str, str, str, str, str], hist.Hist] = {}
    for key, value in payload.items():
        if key == SUMMARY_KEY:
            continue
        if not isinstance(key, tuple) or len(key) != 5:
            continue
        if not isinstance(value, hist.Hist):
            continue
        result[key] = value
    if not result:
        raise ValueError("No tuple-keyed histograms found in payload")
    return result


def determine_year(sample: str) -> str | None:
    if "UL16APV" in sample:
        return "UL16APV"
    for year in ("UL16", "UL17", "UL18"):
        if year in sample and "APV" not in sample:
            return year
    return None


def group_by_year(
    histograms: Mapping[Tuple[str, str, str, str, str], hist.Hist]
) -> Mapping[str, Dict[str, hist.Hist]]:
    grouped: Dict[str, Dict[str, hist.Hist]] = defaultdict(dict)
    for key, histogram in histograms.items():
        variable, flipstatus, _application, sample, _systematic = key
        if variable != "ptabseta":
            continue
        year = determine_year(sample)
        if year is None:
            continue
        if flipstatus not in ("truthFlip", "truthNoFlip"):
            continue
        hist_copy = histogram.copy()
        if flipstatus in grouped[year]:
            grouped[year][flipstatus] = grouped[year][flipstatus] + hist_copy
        else:
            grouped[year][flipstatus] = hist_copy
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
    hist.plot2d(histo, xaxis=xaxis_var)
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
