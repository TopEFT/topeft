'''
        This script is made specifically for run3 flip scale factor calculation based on l0eta distribution.
        Default eta bins are [-3, -1.479, 0, 1.479, 3], for +/- endcap/barrel regions. Hence there will be total of four sfs.

        Run the following command:
        python diboson_sf_run3.py {/path/to/np.pkl.gz} -y {year}

'''

import argparse
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.optimize import curve_fit
import hist
import boost_histogram as bh
import awkward as ak
import json

def load_pkl_file(pkl_file):
    with gzip.open(pkl_file, "rb") as f:
        return pickle.load(f)

def get_yields_in_bins(hin_dict, proc_list, bins, hist_name, channel_name):
    h = hin_dict[hist_name]
    yields = {}
    edges = None

    for proc in proc_list:
        yields[proc] = []

        try:
            # Slice to process and channel
            h_sel = h[{"process": proc, "channel": channel_name}]

            if hist_name not in [ax.name for ax in h_sel.axes]:
                raise ValueError(f"Axis '{hist_name}' not found for histogram '{hist_name}'")

            # Project onto the histogram of interest to make sure we can access
            # a simple 1D array of bin contents and variances.
            if len(h_sel.axes) > 1:
                h_sel = h_sel.project(hist_name)

            axis = h_sel.axes[hist_name]
            edges = axis.edges

            values = np.asarray(h_sel.values(flow=False), dtype=float)
            variances = h_sel.variances(flow=False)
            if variances is None:
                variances = np.zeros_like(values, dtype=float)
            else:
                variances = np.asarray(variances, dtype=float)

            view_flatten = values.reshape(-1)
            var_flatten = variances.reshape(-1)

        except Exception as e:
            print(f"\n\n  Error slicing/integrating for proc {proc}: {e}")
            yields[proc] = [(0.0, 0.0)] * (len(bins) - 1)
            continue

        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            axis_edges = np.asarray(axis.edges, dtype=float)
            start = np.searchsorted(axis_edges, low, side="left")
            stop = np.searchsorted(axis_edges, high, side="left")
            start = max(start, 0)
            stop = min(stop, len(view_flatten))
            val = float(np.sum(view_flatten[start:stop]))
            err = float(np.sqrt(np.sum(var_flatten[start:stop])))
            yields[proc].append((val, err))

    return list(edges) if edges is not None else bins, yields

def make_diboson_sf_json(bins, scale_factors, year):
    if len(bins) != len(scale_factors) + 1:
        raise ValueError("Number of scale factors must be one less than number of bin edges.")
    
    key_name = f"dibosonSF_njets_{year}"
    sf_json = {
        key_name: {
            f"[{bins[i]},{bins[i+1]}]": scale_factors[i]
            for i in range(len(scale_factors))
        }
    }
    with open(f"diboson_sf_{year}.json", "w") as f:
        json.dump(sf_json, f, indent=2)
    print(f"Scaling factors saved to diboson_sf_{year}.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", help="Path to the input pkl.gz file")
    parser.add_argument("--hist-name", default="njets", help="Histogram name")
    parser.add_argument("--channel", default="3l_CR", help="Channel name")
    parser.add_argument("-y", "--year", default="2022", help = "The year of the sample")
    args = parser.parse_args()

    hin_dict = load_pkl_file(args.pkl)

    h = hin_dict[args.hist_name]
    axis = h.axes[args.hist_name]
    bins = axis.edges.tolist()
    proc_list = list(h.axes["process"])

    _, yields = get_yields_in_bins(hin_dict, proc_list, bins, hist_name=args.hist_name, channel_name=args.channel)

    diboson_vals = diboson_errs = None
    data_vals = data_errs = None
    other_vals = other_errs = None

    for proc, vals in yields.items():
        if proc.startswith("flip"):
            continue  # Skip flip samples
        #print("\n\nProcessing proc:", proc) #, vals)
        if proc.startswith("WZTo") or proc.startswith("ZZTo") or proc.startswith("WWTo"):
            #print("  Adding to diboson")
            val_arr = np.array([val for val, _ in vals], dtype=float)
            err_arr = np.array([err for _, err in vals], dtype=float)
            if diboson_vals is None:
                diboson_vals = val_arr.copy()
                diboson_errs = err_arr.copy()
            else:
                diboson_vals += val_arr
                diboson_errs = np.sqrt(diboson_errs**2 + err_arr**2)
        elif "data" in proc:
            #print("  Adding to data")
            data_vals = np.array([val for val, _ in vals], dtype=float)
            data_errs = np.array([err for _, err in vals], dtype=float)
        else:
            #print("  Adding to other")
            val_arr = np.array([val for val, _ in vals], dtype=float)
            err_arr = np.array([err for _, err in vals], dtype=float)
            if other_vals is None:
                other_vals = val_arr.copy()
                other_errs = err_arr.copy()
            else:
                other_vals += val_arr
                other_errs = np.sqrt(other_errs**2 + err_arr**2)

    num_bins = len(bins) - 1
    if diboson_vals is None:
        diboson_vals = np.zeros(num_bins)
        diboson_errs = np.zeros(num_bins)
    if data_vals is None:
        data_vals = np.zeros(num_bins)
        data_errs = np.zeros(num_bins)
    if other_vals is None:
        other_vals = np.zeros(num_bins)
        other_errs = np.zeros(num_bins)

    # Compute (data - other) / diboson
    scale_factors = []
    scale_factor_errs = []
    for d, de, o, oe, f, fe in zip(
        data_vals, data_errs, other_vals, other_errs, diboson_vals, diboson_errs
    ):
        if f != 0:
            numerator = d - o
            sf = numerator / f
            variance = (de**2 + oe**2) / (f**2)
            variance += ((numerator) ** 2 / (f**4)) * (fe**2)
            sf_err = float(np.sqrt(max(variance, 0.0)))
        else:
            sf = float(0)
            sf_err = float(0)
        scale_factors.append(float(sf))
        scale_factor_errs.append(sf_err)
    make_diboson_sf_json(bins, scale_factors, year=args.year)

    # Output
    print("diboson =", diboson_vals.tolist())
    print("data  =", data_vals.tolist())
    print("other =", other_vals.tolist())
    print("SFs   =", scale_factors)
    print("SF errs =", scale_factor_errs)

    edges = np.array(bins, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ratio = np.array(scale_factors, dtype=float)
    yerr = np.array(scale_factor_errs, dtype=float)

    if len(ratio) <= 7:
        print("Not enough bins to apply slice(3, -4); skipping plotting.")
        return

    sel = slice(3, -4)
    sel_centers = centers[sel]
    sel_ratio = ratio[sel]
    sel_err = yerr[sel]

    sel_err_safe = np.where(sel_err > 0, sel_err, 1e-6)

    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ax.errorbar(sel_centers, sel_ratio, yerr=sel_err, fmt="o", label="Scale factor")

    coeffs = np.polyfit(sel_centers, sel_ratio, deg=1, w=1.0 / sel_err_safe)
    ax.plot(sel_centers, np.polyval(coeffs, sel_centers), label="Polyfit")

    def linear(x, m, b):
        return m * x + b

    try:
        popt, pcov = curve_fit(
            linear, sel_centers, sel_ratio, sigma=sel_err_safe, absolute_sigma=True
        )
        ax.plot(sel_centers, linear(sel_centers, *popt), label="Curve fit", linestyle="--")
        print(f"curve_fit parameters: {popt}")
        print(f"curve_fit covariance matrix:\n{pcov}")
    except Exception as exc:
        popt = None
        pcov = None
        print(f"curve_fit failed: {exc}")

    print(f"Polyfit coefficients: {coeffs}")

    ax.set_xlabel("Njets bin center")
    ax.set_ylabel("Scale factor")
    ax.legend()
    fig.tight_layout()

    fig.savefig("output.pdf")
    fig.savefig("output.png")
    plt.close(fig)



if __name__ == "__main__":
    main()

