#!/usr/bin/env python3

import argparse
import gzip
import pickle

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.optimize import curve_fit

np.seterr(divide="ignore", invalid="ignore", over="ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute diboson scale factors from a gzipped pickle histogram file.")
    parser.add_argument("fin", help="Input .pkl.gz file")
    return parser.parse_args()


def load_hists_gz_pickle(path):
    with gzip.open(path, "rb") as f:
        histograms_in = pickle.load(f)

    histograms = {}
    for key, value in histograms_in.items():
        histograms[key] = histograms.get(key, 0) + value
    return histograms


def select_and_sum(histogram, processes, channels, systematic="nominal"):
    return histogram[
        {"process": processes, "channel": channels, "systematic": systematic}
    ][{"process": sum, "channel": sum}]


def get_first_flattened_eval(histogram):
    return list(histogram.eval({}).values())[0].flatten()


def format_array(array, precision=6):
    with np.printoptions(
        precision=precision,
        suppress=False,
        threshold=np.inf,
        linewidth=200,
        floatmode="maxprec_equal",
    ):
        return np.array2string(np.asarray(array))


def print_array(label, array, precision=6):
    arr = np.asarray(array)
    print(f"\n=== {label} ===")
    print(f"shape={arr.shape}, dtype={arr.dtype}")
    print(format_array(arr, precision=precision))


def print_info_block(title, lines):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    for line in lines:
        print(line)


def safe_divide(numerator, denominator, fill_value=0.0):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full_like(numerator, fill_value, dtype=float)
    np.divide(numerator, denominator, out=out, where=(denominator != 0))
    return out


def safe_sqrt(array):
    array = np.asarray(array, dtype=float)
    return np.sqrt(np.clip(array, 0.0, None))


def linear_model(x, p0, p1):
    return p0 * x + p1


def main():
    args = parse_args()
    fin = args.fin

    hists = load_hists_gz_pickle(fin)

    h = hists["njets"]
    h_sumw2 = hists["njets_sumw2"]

    diboson_prefixes = ["WWTo", "WZTo", "ZZTo", "WZto"]

    diboson_procs = [
        proc for proc in h.axes["process"]
        if any(proc.startswith(prefix) for prefix in diboson_prefixes)
    ]
    data_procs = [proc for proc in h.axes["process"] if "data" in proc.lower()]
    bkg_procs = [
        proc for proc in h.axes["process"]
        if ("data" not in proc.lower()) and (proc not in diboson_procs)
    ]
    cr_3l_channels = [chan for chan in h.axes["channel"] if "3l_CR" in chan]

    print_info_block(
        "selection_summary",
        [
            f"input_file: {fin}",
            f"selected_channels (contains '3l_CR'): {cr_3l_channels}",
            f"n_data_processes={len(data_procs)}, n_diboson_processes={len(diboson_procs)}, n_other_bkg_processes={len(bkg_procs)}",
            f"data_processes: {data_procs}",
            f"diboson_processes: {diboson_procs}",
            f"other_bkg_processes: {bkg_procs}",
        ],
    )

    h_diboson = select_and_sum(h, diboson_procs, cr_3l_channels)
    h_diboson_sumw2 = select_and_sum(h_sumw2, diboson_procs, cr_3l_channels)

    h_data = select_and_sum(h, data_procs, cr_3l_channels)

    h_bkg = select_and_sum(h, bkg_procs, cr_3l_channels)
    h_bkg_sumw2 = select_and_sum(h_sumw2, bkg_procs, cr_3l_channels)

    data = get_first_flattened_eval(h_data).astype(float)
    bkg = get_first_flattened_eval(h_bkg).astype(float)
    diboson = get_first_flattened_eval(h_diboson).astype(float)

    bkg_sumw2 = get_first_flattened_eval(h_bkg_sumw2).astype(float)
    diboson_sumw2 = get_first_flattened_eval(h_diboson_sumw2).astype(float)

    bins = h_diboson.axes["njets"].edges
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    print_info_block(
        "histogram_axis",
        [
            "njets_bin_edges:",
            format_array(bins, precision=6),
            "njets_bin_centers:",
            format_array(bin_centers, precision=6),
        ],
    )

    print_array("data", data)
    print_array("bkg", bkg)
    print_array("diboson", diboson)

    print_array("bkg_sumw2", bkg_sumw2)
    print_array("diboson_sumw2", diboson_sumw2)

    # Absolute uncertainties
    sigma_data = safe_sqrt(data)              # Poisson data uncertainty
    sigma_bkg = safe_sqrt(bkg_sumw2)          # MC stat uncertainty on background
    sigma_diboson = safe_sqrt(diboson_sumw2)  # MC stat uncertainty on diboson

    # Relative uncertainties
    rel_sigma_bkg = safe_divide(sigma_bkg, bkg)
    rel_sigma_diboson = safe_divide(sigma_diboson, diboson)

    print_array("sigma_data = sqrt(data)", sigma_data)
    print_array("sigma_bkg = sqrt(bkg_sumw2)", sigma_bkg)
    print_array("sigma_diboson = sqrt(diboson_sumw2)", sigma_diboson)

    print_array("rel_sigma_bkg = sqrt(bkg_sumw2) / bkg", rel_sigma_bkg)
    print_array("rel_sigma_diboson = sqrt(diboson_sumw2) / diboson", rel_sigma_diboson)

    # Numerator and ratio
    data_minus_bkg = data - bkg
    ratio = safe_divide(data_minus_bkg, diboson)

    print_array("data_minus_bkg = data - bkg", data_minus_bkg)
    print_array("ratio = (data - bkg) / diboson", ratio)

    # Uncertainty on numerator: sigma^2(data - bkg) = sigma_data^2 + sigma_bkg^2
    sigma_num = safe_sqrt(sigma_data**2 + sigma_bkg**2)

    # Propagated uncertainty on ratio:
    # R = N / S
    # sigma_R = |R| * sqrt( (sigma_N / N)^2 + (sigma_S / S)^2 )
    rel_sigma_num = safe_divide(sigma_num, data_minus_bkg)
    ratio_unc = np.abs(ratio) * safe_sqrt(rel_sigma_num**2 + rel_sigma_diboson**2)

    print_array("sigma_num = sqrt(sigma_data^2 + sigma_bkg^2)", sigma_num)
    print_array("rel_sigma_num = sigma_num / (data - bkg)", rel_sigma_num)
    print_array("ratio_unc", ratio_unc)

    total_data_minus_bkg = np.sum(data_minus_bkg)
    total_diboson = np.sum(diboson)
    total_ratio = total_data_minus_bkg / total_diboson if total_diboson != 0 else np.nan

    total_sigma_data = np.sqrt(np.sum(sigma_data**2))
    total_sigma_bkg = np.sqrt(np.sum(sigma_bkg**2))
    total_sigma_num = np.sqrt(total_sigma_data**2 + total_sigma_bkg**2)
    total_sigma_diboson = np.sqrt(np.sum(sigma_diboson**2))

    if total_data_minus_bkg != 0 and total_diboson != 0:
        total_ratio_unc = abs(total_ratio) * np.sqrt(
            (total_sigma_num / total_data_minus_bkg) ** 2
            + (total_sigma_diboson / total_diboson) ** 2
        )
    else:
        total_ratio_unc = np.nan

    print_info_block(
        "totals",
        [
            f"sum(data - bkg) = {total_data_minus_bkg}",
            f"sum(diboson) = {total_diboson}",
            f"global_ratio = {total_ratio}",
            f"global_ratio_unc = {total_ratio_unc}",
        ],
    )

    hep.style.use("CMS")

    # Keep the same nominal fit window as the original script: bins with x = 2,3,4,5
    fit_x = np.arange(2, 6)

    if len(ratio) < 6:
        raise RuntimeError(
            f"Not enough bins to perform the requested fit. Found {len(ratio)} bins, need at least 6."
        )

    fit_y = ratio[2:6]
    fit_sigma = ratio_unc[2:6]

    # curve_fit requires strictly positive sigma values if provided
    valid_fit = np.isfinite(fit_y) & np.isfinite(fit_sigma) & (fit_sigma > 0)

    if np.count_nonzero(valid_fit) < 2:
        raise RuntimeError(
            "Not enough valid points for fit after requiring finite values and positive uncertainties."
        )

    fit_x_valid = fit_x[valid_fit]
    fit_y_valid = fit_y[valid_fit]
    fit_sigma_valid = fit_sigma[valid_fit]

    print_info_block(
        "fit_debug",
        [
            f"fit_x_all = {format_array(fit_x, precision=0)}",
            f"fit_y_all = {format_array(fit_y)}",
            f"fit_sigma_all = {format_array(fit_sigma)}",
            f"fit_x_valid = {format_array(fit_x_valid, precision=0)}",
            f"fit_y_valid = {format_array(fit_y_valid)}",
            f"fit_sigma_valid = {format_array(fit_sigma_valid)}",
            "fit_model = p0 * x + p1",
            "initial_p0 = [1, 1]",
        ],
    )

    popt, pcov = curve_fit(
        linear_model,
        fit_x_valid,
        fit_y_valid,
        p0=[1, 1],
        sigma=fit_sigma_valid,
        absolute_sigma=True,
    )

    fit_eval_x = np.arange(2, 8)
    fit_eval_y = linear_model(fit_eval_x, *popt)

    print_info_block(
        "fit_result",
        [
            f"popt = {format_array(popt)}",
            "pcov =",
            format_array(pcov),
            f"fit_eval_x = {format_array(fit_eval_x, precision=0)}",
            f"fit_eval_y = {format_array(fit_eval_y)}",
        ],
    )

    plt.figure()
    plt.errorbar(
        np.arange(2, 8),
        ratio[2:8],
        yerr=ratio_unc[2:8],
        fmt="o",
        capsize=3,
        label="data / prediction",
    )
    plt.plot(fit_eval_x, fit_eval_y, label="linear fit")

    plt.xlim([2, 8])
    plt.ylim([0.5, 3.5])
    plt.xlabel(r"$N_{jets}$", loc="right")
    plt.ylabel(r"$\frac{\mathrm{data}-\mathrm{non\!-\!diboson}}{\mathrm{diboson}}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("diboson.pdf")
    plt.savefig("diboson.png")


if __name__ == "__main__":
    main()