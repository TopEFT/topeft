#!/usr/bin/env python3

import argparse
import gzip
import pickle

import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.optimize import curve_fit

matplotlib.use("Agg")
np.seterr(divide="ignore", invalid="ignore", over="ignore")


def ParseArgs():
    parser = argparse.ArgumentParser(description="You can select which file to run over")
    parser.add_argument("fin", default="", help="Variable to run over")
    return parser.parse_args()


def LoadHistsGzPickle(path):
    hists = {}
    with gzip.open(path, "rb") as f:
        hin = pickle.load(f)
    for k, v in hin.items():
        hists[k] = hists.get(k, 0) + v
    return hists


def SelectAndSum(hist, processes, channels, systematic="nominal"):
    return hist[{"process": processes, "channel": channels, "systematic": systematic}][
        {"process": sum, "channel": sum}
    ]


def GetFirstFlattenedEval(hist):
    return list(hist.eval({}).values())[0].flatten()


def FormatArray(arr, precision=6):
    # Full content, no truncation; readable floats.
    with np.printoptions(
        precision=precision,
        suppress=False,
        threshold=np.inf,  # never crop
        linewidth=200,     # keep lines reasonably wide
        floatmode="maxprec_equal",
    ):
        return np.array2string(np.asarray(arr))


def PrintArray(label, arr, precision=6):
    a = np.asarray(arr)
    print(f"\n=== {label} ===")
    print(f"shape={a.shape}, dtype={a.dtype}")
    print(FormatArray(a, precision=precision))


def PrintInfoBlock(title, lines):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    for line in lines:
        print(line)


def main():
    args = ParseArgs()
    fin = args.fin

    hists = LoadHistsGzPickle(fin)

    h = hists["njets"]
    h_sumw2 = hists["njets_sumw2"]

    dibosonPrefixes = ["WWTo", "WZTo", "ZZTo", "WZto"]
    dibosonProcs = [proc for proc in h.axes["process"] if any(p == proc[:4] for p in dibosonPrefixes)]
    dataProcs = [proc for proc in h.axes["process"] if "data" in proc]
    bkgProcs = [
        proc
        for proc in h.axes["process"]
        if ("data" not in proc) and (not any(p == proc[:4] for p in dibosonPrefixes))
    ]
    cr3lChannels = [chan for chan in h.axes["channel"] if "3l_CR" in chan]

    PrintInfoBlock(
        "Selection Summary",
        [
            f"Input file: {fin}",
            f"Selected channels (contains '3l_CR'): {cr3lChannels}",
            f"Selected processes: data={len(dataProcs)}, diboson={len(dibosonProcs)}, other-bkg={len(bkgProcs)}",
            f"  data processes: {dataProcs}",
            f"  diboson processes (prefix in {dibosonPrefixes}): {dibosonProcs}",
            f"  other-bkg processes (not data, not diboson): {bkgProcs}",
        ],
    )

    h_diboson = SelectAndSum(h, dibosonProcs, cr3lChannels)
    h_diboson2 = SelectAndSum(h_sumw2, dibosonProcs, cr3lChannels)

    h_data = SelectAndSum(h, dataProcs, cr3lChannels)

    h_bkg = SelectAndSum(h, bkgProcs, cr3lChannels)
    h_bkg2 = SelectAndSum(h_sumw2, bkgProcs, cr3lChannels)

    data = GetFirstFlattenedEval(h_data)
    bkg = GetFirstFlattenedEval(h_bkg)
    diboson = GetFirstFlattenedEval(h_diboson)
    bkg2 = GetFirstFlattenedEval(h_bkg2)
    diboson2 = GetFirstFlattenedEval(h_diboson2)

    # Kept for parity with original script (even if unused later)
    h_nodi = h_data - h_bkg

    bins = h_diboson.axes["njets"].edges

    PrintInfoBlock(
        "Histogram Axis",
        [
            "njets bin edges (h_diboson.axes['njets'].edges):",
            FormatArray(bins, precision=6),
        ],
    )

    PrintArray("DATA counts (summed over selected processes/channels)", data)
    PrintArray("BKG counts (non-data, non-diboson)", bkg)
    PrintArray("DIBOSON counts (WWTo/WZTo/ZZTo prefix)", diboson)

    PrintArray("BKG sumw2 (variance proxy)", bkg2)
    PrintArray("DIBOSON sumw2 (variance proxy)", diboson2)

    # --- Difference of data and background
    data_minus_bkg = data - bkg
    PrintArray("DATA_MINUS_BKG = DATA - BKG", data_minus_bkg)

    # --- Uncertainty ingredients (kept identical, even if odd)
    ediboson = np.sqrt(diboson2) / (diboson)
    ebkg = np.sqrt(bkg2) / (bkg2)
    PrintArray("ediboson = sqrt(diboson2)/diboson", ediboson)
    PrintArray("ebkg = sqrt(bkg2)/bkg2", ebkg)

    # --- Ratios
    ratio = data_minus_bkg / diboson
    PrintArray("RATIO = (DATA - BKG) / DIBOSON", ratio)

    # --- Slice examples (kept, but now labeled)
    PrintInfoBlock(
        "Sanity slices (kept from original)",
        [
            f"ratio[1:-1] = {FormatArray(ratio[1:-1])}",
            f"ratio[3:8]  = {FormatArray(ratio[3:8])}",
        ],
    )

    # --- Totals
    tot_data = np.sum(data_minus_bkg)
    PrintInfoBlock("Totals", [f"tot_data = sum(DATA - BKG) = {tot_data}"])

    # --- Uncertainties (avoid divide-by-zero with np.nan_to_num)
    yerr = np.nan_to_num(np.sqrt(1 / data + 1 / bkg), nan=0)
    yerr2 = np.nan_to_num(np.sqrt((1 / data)), nan=0)
    yerr3 = np.nan_to_num(
        ratio
        * np.sqrt(
            ((np.sqrt(data) + np.sqrt(ebkg)) ** 2) / ((data - bkg) ** 2)
            + (np.sqrt(ediboson) / diboson) ** 2
        ),
        nan=0,
    )

    PrintArray("yerr = sqrt(1/DATA + 1/BKG) (nan->0)", yerr)
    PrintArray("yerr2 = sqrt(1/DATA) (nan->0)", yerr2)
    PrintArray("yerr3 (original expression, nan->0)", yerr3)

    hep.style.use("CMS")

    # Keep original debugging prints, but clearer and full
    PrintInfoBlock(
        "Fit debug (kept from original)",
        [
            f"fit x = np.arange(2, 8) = {FormatArray(np.arange(2, 8), precision=0)}",
            f"fit y shown (ratio[2:-4]) = {FormatArray(ratio[2:-4])}",
            f"initial p0 = [1, 1]",
            f"fit sigma shown (yerr3[2:-4]) = {FormatArray(yerr3[2:-4])}",
        ],
    )
    PrintInfoBlock(
        "More debug (kept from original)",
        [
            f"x (printed) = {FormatArray(np.arange(2, 8), precision=0)}",
            f"y (printed) = {FormatArray(ratio[2:8])}",
            "bins (full):",
            FormatArray(bins),
            "ratio (full):",
            FormatArray(ratio),
            "ratio[2:-4] (full):",
            FormatArray(ratio[2:-4]),
        ],
    )

    # Keep the exact fit range and call signature from original
    popt, pcov = curve_fit(
        lambda x, *p: p[0] * x + p[1],
        np.arange(2, 6),
        ratio[2:6],
        [1, 1],
        yerr3[2:6],
    )

    PrintInfoBlock(
        "Fit result",
        [
            f"popt = {FormatArray(popt)}  (model: p0*x + p1)",
            "polyval(popt, np.arange(2, 8)):",
            FormatArray(np.polyval(popt, np.arange(2, 8))),
        ],
    )

    plt.plot(np.arange(2, 8), np.polyval(popt, np.arange(2, 8)))
    plt.plot(np.arange(2, 8), ratio[2:8], marker="o", linestyle="none")

    plt.xlim([2, 8])
    plt.ylim([0.5, 3.5])
    plt.xlabel(r"$N_{jets}$", loc="right")
    plt.ylabel(r"$\frac{data\;-\;non-diboson}{diboson}$")
    plt.savefig("diboson.pdf")
    plt.savefig("diboson.png")


if __name__ == "__main__":
    main()