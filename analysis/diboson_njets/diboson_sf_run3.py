'''
        This script is made specifically for run3 diboson scale factor calculation based on njets distribution.
        Default njet bins are [0, 1, 2, 3, 4, 5, 6].

        Run the following command:
        python diboson_sf_run3.py --pkl {/path/to/np.pkl.gz} -y {year}

        Multiple years can be processed at once by repeating the arguments, e.g.:
        python diboson_sf_run3.py --pkl year_{year}.pkl.gz -y 2022 2023
'''

import argparse
import gzip
import json
import os
import pickle

import numpy as np
import hist
import boost_histogram as bh
import awkward as ak

def load_pkl_file(pkl_file):
    with gzip.open(pkl_file, "rb") as f:
        return pickle.load(f)

def get_yields_in_bins(
    hin_dict,
    proc_list,
    bins,
    hist_name,
    channel_name,
    extra_slices=None,
):
    h = hin_dict[hist_name]
    yields = {}

    print("h axes:")
    for ax in h.axes:
        print(f"  {ax.name}: {list(ax)}")

    available_axes = {ax.name for ax in h.axes}

    for proc in proc_list:
        yields[proc] = []

        try:
            selection = {"process": proc, "channel": channel_name}
            if extra_slices:
                selection.update(extra_slices)

            filtered_selection = {
                key: selection[key]
                for key in selection
                if key in available_axes
            }

            # Slice to process, channel, and any optional axes (e.g., year)
            h_sel = h[filtered_selection]

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
    parser.add_argument(
        "--pkl",
        nargs="+",
        required=True,
        help=(
            "Path(s) to the input pkl.gz file. Provide one path per year, a single "
            "template containing '{year}' to be expanded for each year, or a single "
            "file that contains histograms for all requested years."
        ),
    )
    parser.add_argument("--hist-name", default="njets", help="Histogram name")
    parser.add_argument("--channel", default="3l_CR", help="Channel name")
    parser.add_argument(
        "-y",
        "--year",
        nargs="+",
        default=["2022"],
        help="One or more data-taking years to process",
    )
    parser.add_argument(
        "--year-axis",
        default="year",
        help=(
            "Histogram axis name to use when selecting a specific year from a "
            "shared input file containing multiple years."
        ),
    )
    args = parser.parse_args()

    years = list(args.year)

    if not years:
        parser.error("At least one year must be provided via -y/--year.")

    shared_pkl_for_years = False

    if len(args.pkl) == 1:
        pkl_arg = args.pkl[0]
        if "{year}" in pkl_arg:
            pkl_paths = [pkl_arg.format(year=year) for year in years]
            missing_paths = [path for path in pkl_paths if not os.path.exists(path)]
            if missing_paths:
                parser.error(
                    "The following expanded --pkl paths do not exist: "
                    + ", ".join(missing_paths)
                )
        elif len(years) == 1:
            pkl_paths = [pkl_arg]
        else:
            if not os.path.exists(pkl_arg):
                parser.error(f"Shared --pkl file not found: {pkl_arg}")
            pkl_paths = [pkl_arg] * len(years)
            shared_pkl_for_years = True
    elif len(args.pkl) == len(years):
        pkl_paths = args.pkl
    else:
        parser.error(
            "Number of --pkl paths must match the number of years unless a template is used."
        )

    if not shared_pkl_for_years:
        missing_paths = [path for path in pkl_paths if not os.path.exists(path)]
        if missing_paths:
            parser.error(
                "The following --pkl paths do not exist: " + ", ".join(missing_paths)
            )

    bins = list(range(0, 7))

    cached_inputs = {}

    for year, pkl_path in zip(years, pkl_paths):
        if pkl_path in cached_inputs:
            hin_dict = cached_inputs[pkl_path]
        else:
            hin_dict = load_pkl_file(pkl_path)
            cached_inputs[pkl_path] = hin_dict

        h = hin_dict[args.hist_name]
        proc_list = list(h.axes["process"])

        year_selection = {}
        if shared_pkl_for_years:
            available_axes = {ax.name for ax in h.axes}
            if args.year_axis not in available_axes:
                parser.error(
                    "A shared --pkl file was provided, but histogram "
                    f"'{args.hist_name}' does not contain the '{args.year_axis}' axis. "
                    f"Available axes: {sorted(available_axes)}"
                )

            year_axis = h.axes[args.year_axis]
            axis_values = list(year_axis)
            target_value = None

            if year in axis_values:
                target_value = year
            else:
                for candidate in axis_values:
                    try:
                        converted = type(candidate)(year)
                        if candidate == converted:
                            target_value = converted
                            break
                    except Exception:
                        continue

            if target_value is None:
                parser.error(
                    f"Year '{year}' not found on axis '{args.year_axis}' in histogram "
                    f"'{args.hist_name}'. Available values: {axis_values}"
                )

            year_selection = {args.year_axis: target_value}

        yields = get_yields_in_bins(
            hin_dict,
            proc_list,
            bins,
            hist_name=args.hist_name,
            channel_name=args.channel,
            extra_slices=year_selection,
        )

        diboson = []
        data = []
        other = []

        for proc, vals in yields.items():
            if proc.startswith("flip"):
                continue  # Skip flip samples
            if proc.startswith("WZTo") or proc.startswith("ZZTo") or proc.startswith("WWTo"):
                if not diboson:
                    diboson = [val for val, _ in vals]
                else:
                    diboson = [x + val for x, (val, _) in zip(diboson, vals)]
            elif "data" in proc:
                data = [val for val, _ in vals]
            else:
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
            print(
                f"Linear fit coefficients for {year}: "
                f"slope = {slope:.6f}, intercept = {intercept:.6f}"
            )

            fit_coefficients = {"slope": float(slope), "intercept": float(intercept)}
            fit_coeff_path = f"diboson_sf_{year}_linear_fit.json"
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
            ax.plot(
                bin_centers,
                fitted_values,
                label="Linear fit",
                linestyle="-",
                marker="",
            )
            ax.set_xlabel("N_{jets} bin center")
            ax.set_ylabel("Scale factor")
            ax.set_title(f"Diboson scale factors ({year}, {args.channel})")
            ax.legend()
            plot_path = f"diboson_sf_{year}.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved scale factor plot to {plot_path}")
        elif plt is None:
            print("matplotlib not available; skipping plot generation.")

        make_diboson_sf_json(bins, scale_factors, year=year)

        # Output
        print(f"Results for {year}:")
        print("diboson =", diboson)
        print("data  =", data)
        print("other =", other)
        print("SFs   =", scale_factors)



if __name__ == "__main__":
    main()

