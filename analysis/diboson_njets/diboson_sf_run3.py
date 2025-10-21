'''
        This script is made specifically for run3 diboson scale factor calculation based on njets distribution.
        Default njet bins are [0, 1, 2, 3, 4, 5, 6].

        Run the following command:
        python diboson_sf_run3.py {/path/to/np.pkl.gz} -y {year}
'''

import argparse
import pickle
import gzip
import numpy as np
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

    for proc in proc_list:
        yields[proc] = []

        try:
        # Slice to process and channel
            h_sel = h[{"process": proc, "channel": channel_name}]

            # #If "l0eta" is not the only axis, integrate over the others
            # for ax in h_sel.axes:
            #     if ax.name != "njets":
            #         h_sel = h_sel.integrate(ax.name)
            axis = h_sel.axes[hist_name]
            view = h_sel.view(flow=False)
            edges = axis.edges
            view_array = list(view.values())[0]  # Extracts the array
            view_flatten = view_array.flatten().tolist()

        except Exception as e:
            print(f"\n\n  Error slicing/integrating for proc {proc}: {e}")
            yields[proc] = [(None, None)] * (len(bins) - 1)
            break

        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            val = 0.0
            err = 0.0

            bin_indices = [
                j for j, (lo, hi) in enumerate(zip(axis.edges[:-1], axis.edges[1:]))
                if hi > low and lo < high
            ]
            val = sum(view_flatten[j] for j in bin_indices)
            yields[proc].append((val, 0.0))
            
    return yields

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

    bins = list(range(0, 7))
    hin_dict = load_pkl_file(args.pkl)

    h = hin_dict[args.hist_name]
    proc_list = list(h.axes["process"])

    yields = get_yields_in_bins(hin_dict, proc_list, bins, hist_name=args.hist_name, channel_name=args.channel)

    diboson = []
    data = []
    other = []
    
    for proc, vals in yields.items():
        if proc.startswith("flip"):
            continue  # Skip flip samples
        #print("\n\nProcessing proc:", proc) #, vals)
        if proc.startswith("WZTo") or proc.startswith("ZZTo") or proc.startswith("WWTo"):
            #print("  Adding to diboson")
            if not diboson:
                diboson = [val for val, _ in vals]
            else:
                diboson = [x + val for x, (val, _) in zip(diboson, vals)]
        elif "data" in proc:
            #print("  Adding to data")
            data = [val for val, _ in vals]
        else:
            #print("  Adding to other")
            if not other:
                other = [val for val, _ in vals]
            else:
                other = [x + val for x, (val, _) in zip(other, vals)]

    # Compute (data - other) / diboson
    scale_factors = []
    for d, o, f in zip(data, other, diboson):
        if f != 0:
            sf = (d - o) / f
        else:
            sf = float(0)  # or 0, or raise an error
        scale_factors.append(sf)

    # Calculate bin centers for plotting and fitting
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

    # Perform a linear fit of the scale factors vs. bin centers
    if bin_centers:
        coeffs = np.polyfit(bin_centers, scale_factors, deg=1)
        slope, intercept = coeffs
        fitted_values = np.polyval(coeffs, bin_centers)
        print(f"Linear fit coefficients: slope = {slope:.6f}, intercept = {intercept:.6f}")

        fit_coefficients = {"slope": float(slope), "intercept": float(intercept)}
        fit_coeff_path = f"diboson_sf_{args.year}_linear_fit.json"
        with open(fit_coeff_path, "w") as f:
            json.dump(fit_coefficients, f, indent=2)
        print(f"Saved linear fit coefficients to {fit_coeff_path}")
    else:
        coeffs = None
        fitted_values = []

    # Plot the scale factors and the fitted line if matplotlib is available
    try:
        import matplotlib.pyplot as plt  # Guarded import to keep unpickling working
    except ImportError:
        plt = None

    if plt is not None and coeffs is not None:
        fig, ax = plt.subplots()
        ax.errorbar(
            bin_centers,
            scale_factors,
            yerr=np.zeros_like(scale_factors, dtype=float),
            fmt="o",
            label="Scale factors",
        )
        ax.plot(bin_centers, fitted_values, label="Linear fit", linestyle="-", marker="")
        ax.set_xlabel("N_{jets} bin center")
        ax.set_ylabel("Scale factor")
        ax.set_title(f"Diboson scale factors ({args.year}, {args.channel})")
        ax.legend()
        plot_path = f"diboson_sf_{args.year}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved scale factor plot to {plot_path}")
    elif plt is None:
        print("matplotlib not available; skipping plot generation.")

    make_diboson_sf_json(bins, scale_factors, year=args.year)

    # Output
    print("diboson =", diboson)
    print("data  =", data)
    print("other =", other)
    print("SFs   =", scale_factors)



if __name__ == "__main__":
    main()

