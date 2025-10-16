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
    required_axes = {"process", "channel"}
    missing_required = sorted(required_axes - available_axes)

    if missing_required:
        missing_str = ", ".join(missing_required)
        raise KeyError(
            f"Histogram '{hist_name}' is missing required axis/axes: {missing_str}. "
            f"Available axes: {sorted(available_axes)}"
        )

    optional_slices = {}
    if extra_slices:
        missing_optional = sorted(set(extra_slices) - available_axes)
        if missing_optional:
            print(
                "Warning: ignoring extra_slices axes not present on histogram "
                f"'{hist_name}': {missing_optional}"
            )
        optional_slices = {
            key: extra_slices[key]
            for key in extra_slices
            if key in available_axes
        }

    for proc in proc_list:
        yields[proc] = []

        try:
            selection = {
                "process": proc,
                "channel": channel_name,
                **optional_slices,
            }

            # Slice to process, channel, and any optional axes (e.g., year)
            h_sel = h[selection]

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

def make_diboson_sf_json(bins, scale_factors, year, output_dir="."):
    if len(bins) != len(scale_factors) + 1:
        raise ValueError("Number of scale factors must be one less than number of bin edges.")

    key_name = f"dibosonSF_njets_{year}"
    sf_json = {
        key_name: {
            f"[{bins[i]},{bins[i+1]}]": scale_factors[i]
            for i in range(len(scale_factors))
        }
    }
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"diboson_sf_{year}.json")
    with open(output_path, "w") as f:
        json.dump(sf_json, f, indent=2)
    print(f"Scaling factors saved to {output_path}")


def compute_linear_fit(bin_centers, scale_factors):
    if not bin_centers:
        return None, []

    coeffs = np.polyfit(bin_centers, scale_factors, deg=1)
    slope, intercept = coeffs
    fitted_values = np.polyval(coeffs, bin_centers)
    fitted_values_list = np.atleast_1d(fitted_values).tolist()
    fit_coefficients = {"slope": float(slope), "intercept": float(intercept)}
    return fit_coefficients, fitted_values_list


def save_linear_fit_coefficients(year, fit_coefficients, output_dir="."):
    if not fit_coefficients:
        return

    os.makedirs(output_dir, exist_ok=True)
    fit_coeff_path = os.path.join(output_dir, f"diboson_sf_{year}_linear_fit.json")
    with open(fit_coeff_path, "w") as f:
        json.dump(fit_coefficients, f, indent=2)
    print(
        "Saved linear fit coefficients to "
        f"{fit_coeff_path}: slope = {fit_coefficients['slope']:.6f}, "
        f"intercept = {fit_coefficients['intercept']:.6f}"
    )


def save_scale_factor_plot(
    year,
    channel,
    bin_centers,
    scale_factors,
    fitted_values,
    output_dir=".",
):
    if not bin_centers or not scale_factors:
        print("No bin centers or scale factors available for plotting; skipping plot generation.")
        return

    try:
        import matplotlib.pyplot as plt  # Guarded import to keep unpickling working
    except ImportError:
        print("matplotlib not available; skipping plot generation.")
        return

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots()
    scale_factor_array = np.asarray(scale_factors, dtype=float)
    yerr = np.zeros_like(scale_factor_array)
    ax.errorbar(
        bin_centers,
        scale_factor_array,
        yerr=yerr,
        fmt="o",
        label="Scale factors",
    )
    if fitted_values:
        fitted_array = np.asarray(fitted_values, dtype=float)
        ax.plot(
            bin_centers,
            fitted_array,
            label="Linear fit",
            linestyle="-",
            marker="",
        )
    ax.set_xlabel("N_{jets} bin center")
    ax.set_ylabel("Scale factor")
    ax.set_title(f"Diboson scale factors ({year}, {channel})")
    ax.legend()
    plot_path = os.path.join(output_dir, f"diboson_sf_{year}.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scale factor plot to {plot_path}")

def process_year(
    pkl_path,
    year,
    hist_name,
    channel,
    bins,
    *,
    cache=None,
    year_axis=None,
):
    """Process a single year and return scale-factor and fit information."""

    hin_dict = None
    if cache is not None and pkl_path in cache:
        hin_dict = cache[pkl_path]
    else:
        hin_dict = load_pkl_file(pkl_path)
        if cache is not None:
            cache[pkl_path] = hin_dict

    h = hin_dict[hist_name]
    proc_list = list(h.axes["process"])

    year_selection = {}
    if year_axis is not None:
        available_axes = {ax.name for ax in h.axes}
        if year_axis not in available_axes:
            raise KeyError(
                "A shared --pkl file was provided, but histogram "
                f"'{hist_name}' does not contain the '{year_axis}' axis. "
                f"Available axes: {sorted(available_axes)}"
            )

        year_axis_obj = h.axes[year_axis]
        axis_values = list(year_axis_obj)
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
            raise KeyError(
                f"Year '{year}' not found on axis '{year_axis}' in histogram "
                f"'{hist_name}'. Available values: {axis_values}"
            )

        year_selection = {year_axis: target_value}

    yields = get_yields_in_bins(
        hin_dict,
        proc_list,
        bins,
        hist_name=hist_name,
        channel_name=channel,
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

    fit_coefficients, fitted_values = compute_linear_fit(bin_centers, scale_factors)

    # Output
    print(f"Results for {year}:")
    print("diboson =", diboson)
    print("data  =", data)
    print("other =", other)
    print("SFs   =", scale_factors)

    return {
        "scale_factors": scale_factors,
        "fit_coefficients": fit_coefficients,
        "bin_centers": bin_centers,
        "fitted_values": fitted_values,
        "diboson": diboson,
        "data": data,
        "other": other,
    }


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
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where per-year outputs (JSON, plots) will be written.",
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

    results = {}
    summary = {}
    for year, pkl_path in zip(years, pkl_paths):
        try:
            year_axis = args.year_axis if shared_pkl_for_years else None
            results[year] = process_year(
                pkl_path,
                year,
                args.hist_name,
                args.channel,
                bins,
                cache=cached_inputs,
                year_axis=year_axis,
            )
            year_output_dir = os.path.join(args.output_dir, str(year))
            make_diboson_sf_json(
                bins,
                results[year]["scale_factors"],
                year=year,
                output_dir=year_output_dir,
            )
            save_linear_fit_coefficients(
                year,
                results[year]["fit_coefficients"],
                output_dir=year_output_dir,
            )
            save_scale_factor_plot(
                year,
                args.channel,
                results[year]["bin_centers"],
                results[year]["scale_factors"],
                results[year]["fitted_values"],
                output_dir=year_output_dir,
            )

            sf_values = results[year]["scale_factors"]
            mean_sf = float(np.mean(sf_values)) if sf_values else float("nan")
            fit_coefficients = results[year]["fit_coefficients"] or {}
            summary[year] = {
                "mean_scale_factor": mean_sf,
                "slope": fit_coefficients.get("slope"),
                "intercept": fit_coefficients.get("intercept"),
            }
        except KeyError as exc:
            parser.error(str(exc))

    if summary:
        print("\nSummary of scale factor results:")
        header = f"{'Year':<8}{'Mean SF':>12}{'Slope':>12}{'Intercept':>14}"
        print(header)
        print("-" * len(header))
        for year in years:
            year_summary = summary.get(year, {})
            mean_sf = year_summary.get("mean_scale_factor")
            slope = year_summary.get("slope")
            intercept = year_summary.get("intercept")
            if mean_sf is None or (isinstance(mean_sf, float) and np.isnan(mean_sf)):
                mean_sf_str = "n/a"
            else:
                mean_sf_str = f"{mean_sf:.6f}"
            slope_str = "n/a" if slope is None else f"{slope:.6f}"
            intercept_str = "n/a" if intercept is None else f"{intercept:.6f}"
            print(f"{year:<8}{mean_sf_str:>12}{slope_str:>12}{intercept_str:>14}")

    return results



if __name__ == "__main__":
    main()

