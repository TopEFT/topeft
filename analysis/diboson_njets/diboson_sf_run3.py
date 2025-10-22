"""
Run 3 diboson scale factor calculation based on the ``njets`` distribution.
The default ``njets`` bin edges are ``[0, 1, 2, 3, 4, 5, 6]``.

Shared-pickle workflow
======================

Run 3 ntuple production often yields a *single* histogram pickle that contains
all years.  The recommended workflow for such files is:

1. Produce (or obtain) a combined pickle that stores the Run 3 histograms for
   every year.  The per-year information must be encoded in the process names
   (tokens such as ``central2023`` or ``2022EE`` work well).
2. Invoke the script with ``--pkl`` pointing to the shared file and
   ``--year all``.  The sentinel scans the process names, identifies every
   year-like token, runs the per-year fits, and generates a combined summary for
   the union of processed years.
3. Inspect the per-year artifacts written underneath ``--output-dir``.

Invocation patterns
-------------------

* Provide one pickle per year when the histograms are stored separately::

      python diboson_sf_run3.py --pkl 2022.pkl.gz 2023.pkl.gz --year 2022 2023

* Use a template containing ``{year}`` when the files follow a naming
  convention; the placeholder will be expanded for each requested year::

      python diboson_sf_run3.py --pkl "/path/to/year_{year}.pkl.gz" --year 2022 2023

* Supply a single pickle shared by all years.  The script automatically selects
  the processes whose names encode each requested year.  A conservative regular
  expression is used first; if that does not find a match, the code falls back
  to a simple substring search so unconventional naming schemes are still
  handled gracefully::

      python diboson_sf_run3.py --pkl combined.pkl.gz --year 2022 2023

* Discover every available year in a shared pickle with ``--year all``.  This
  sentinel triggers the scanning behaviour described above, runs the per-year
  fits for every detected token, and writes a combined result that aggregates
  all processed years::

      python diboson_sf_run3.py --pkl combined.pkl.gz --year all

Output layout
-------------

Outputs are written per year (including the combined entry when ``all`` is
used) in subdirectories of ``--output-dir``.  Each directory receives a
``diboson_sf_{year}.json`` summary, the linear-fit JSON, and the PNG diagnostic
plot so the per-year artifacts remain grouped together.  The combined
directory, produced only when ``--year all`` is used, aggregates the fits for
all discovered years.
"""

import argparse
import gzip
import json
import logging
import os
import pickle
import re

import awkward as ak
import numpy as np


logger = logging.getLogger(__name__)

def load_pkl_file(pkl_file):
    with gzip.open(pkl_file, "rb") as f:
        return pickle.load(f)

ALL_YEARS_SENTINEL = "all"


def _derive_process_subset_for_year(proc_list, year):
    """Return processes whose names encode the requested year."""

    year_str = str(year)
    # Require the year digits to appear as a standalone number (not part of a
    # longer integer) but allow alphabetic suffixes such as "2022EE" or
    # "2023BPix".
    pattern = re.compile(rf"(?<!\\d){re.escape(year_str)}(?!\\d)")
    matches = {proc for proc in proc_list if pattern.search(str(proc))}

    if matches:
        return matches

    # Fall back to a simple substring search if the conservative regex above
    # did not find any matches so that unexpected naming conventions are still
    # handled gracefully.
    fallback_matches = {proc for proc in proc_list if year_str in str(proc)}
    return fallback_matches


def _map_year_tokens_to_processes(proc_list):
    """Return a mapping of discovered year tokens to matching process names."""

    # Match a four-digit Run 3 year optionally followed by an alphabetic era
    # designator (e.g., "2022EE", "2023BPix"). Keep the suffix attached to
    # preserve distinct data-taking periods while still catching plain years
    # such as "2022".
    pattern = re.compile(r"(?<!\\d)(20\\d{2}(?:[A-Za-z]+)?)(?!\\d)")
    matches = {}
    for proc in proc_list:
        proc_str = str(proc)
        for token in pattern.findall(proc_str):
            matches.setdefault(token, set()).add(proc_str)
    return matches


def get_yields_in_bins(
    hin_dict,
    proc_list,
    bins,
    hist_name,
    channel_name,
    extra_slices=None,
    process_whitelist=None,
):
    """Return per-process yields for ``hist_name`` bins as (value, uncertainty).

    Parameters mirror the histogram dictionary, processes, and binning
    configuration supplied by callers.  The returned mapping associates each
    process with a list of tuples ``(value, uncertainty)`` for the requested
    ``bins``.
    """
    h = hin_dict[hist_name]
    yields = {}

    logger.debug("Histogram '%s' axes:", hist_name)
    for ax in h.axes:
        logger.debug("  %s: %s", ax.name, list(ax))

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
            logger.warning(
                "Ignoring extra_slices axes not present on histogram '%s': %s",
                hist_name,
                missing_optional,
            )
        optional_slices = {
            key: extra_slices[key]
            for key in extra_slices
            if key in available_axes
        }

    whitelist_set = None
    if process_whitelist is not None:
        whitelist_set = set(process_whitelist)

    if whitelist_set is not None:
        processes_to_scan = whitelist_set
    else:
        processes_to_scan = proc_list

    for proc in processes_to_scan:

        try:
            selection = {
                "process": proc,
                "channel": channel_name,
                **optional_slices,
            }

            # Slice to process, channel, and any optional axes (e.g., year)
            h_sel = h[selection]

            axis_names = [ax.name for ax in h_sel.axes]
            try:
                target_index = axis_names.index(hist_name)
            except ValueError as exc:
                raise KeyError(
                    f"Histogram '{hist_name}' axis not present after slicing for process "
                    f"'{proc}'. Remaining axes: {axis_names}"
                ) from exc

            try:
                values_np = ak.to_numpy(h_sel.values(flow=False))
            except Exception as exc:
                logger.error(
                    "Failed to convert histogram values to a dense NumPy array using "
                    "ak.to_numpy for process '%s' on histogram '%s'.",
                    proc,
                    hist_name,
                    exc_info=exc,
                )
                raise RuntimeError(
                    "Failed to convert histogram values to a dense NumPy array using "
                    f"ak.to_numpy for process '{proc}' on histogram '{hist_name}'."
                ) from exc

            # Reduce all axes except the histogram axis via a single summation.
            sum_axes = tuple(
                axis_idx for axis_idx in range(values_np.ndim) if axis_idx != target_index
            )
            if sum_axes:
                values_np = values_np.sum(axis=sum_axes)

            target_axis = next(
                (ax for ax in h_sel.axes if ax.name == hist_name),
                None,
            )
            if target_axis is None or not hasattr(target_axis, "edges"):
                raise AttributeError(
                    f"Unable to determine bin edges for axis '{hist_name}'."
                )
            edges = target_axis.edges

            expected_bins = len(edges) - 1
            values_np = np.asarray(values_np, dtype=float).reshape(-1)
            if values_np.shape[0] != expected_bins:
                raise RuntimeError(
                    "Reduced histogram values do not match the number of bins for axis "
                    f"'{hist_name}'. Expected {expected_bins}, got {values_np.shape[0]}"
                )

        except Exception as exc:  # pragma: no cover - exercised via tests
            error_message = (
                "Failed to compute yields for process "
                f"'{proc}' from histogram '{hist_name}' in channel "
                f"'{channel_name}'. Encountered error: {exc.__class__.__name__}: {exc}"
            )
            logger.error(error_message, exc_info=exc)
            raise RuntimeError(error_message) from exc

        proc_yields = []
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            bin_indices = [
                j for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))
                if hi > low and lo < high
            ]
            val = float(np.sum(values_np[bin_indices])) if bin_indices else 0.0
            proc_yields.append((val, 0.0))

        yields[proc] = proc_yields

    return yields

def make_diboson_sf_json(bins, scale_factors, year, output_dir="."):
    """Write the per-bin scale factors for ``year`` to a JSON summary.

    Parameters
    ----------
    bins : Sequence[float]
        Monotonically increasing bin edges whose length must exceed the number
        of scale factors by one.
    scale_factors : Sequence[float]
        Scale-factor values computed for the intervals defined by ``bins``.
    year : str or int
        Year label used to key the JSON payload and the output filename.
    output_dir : str, optional
        Destination directory for produced artifacts.  Created if it does not
        already exist.

    Returns
    -------
    str
        Absolute path to the emitted JSON file.

    Notes
    -----
    Writes ``diboson_sf_{year}.json`` containing the scale-factor JSON to
    ``output_dir`` so the README-described summary file can be located directly
    from the code.
    """
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
    return os.path.abspath(output_path)


def compute_linear_fit(bin_centers, scale_factors):
    """Fit a straight line to the supplied scale factors.

    Parameters
    ----------
    bin_centers : Sequence[float]
        Centers of the ``njets`` bins used as the independent variable.
    scale_factors : Sequence[float]
        Scale-factor values serving as the dependent variable.

    Returns
    -------
    Tuple[Optional[Dict[str, float]], List[float]]
        The slope/intercept coefficients and the fitted values evaluated at the
        provided ``bin_centers``.  Returns ``(None, [])`` when the fit cannot be
        computed (e.g., mismatched inputs).
    """
    if bin_centers is None or scale_factors is None:
        return None, []

    if np.size(bin_centers) == 0 or np.size(scale_factors) == 0:
        return None, []

    if len(bin_centers) != len(scale_factors):
        print(
            "Warning: skipping linear fit because the number of bin centers does "
            "not match the number of scale factors."
        )
        return None, []

    coeffs = np.polyfit(bin_centers, scale_factors, deg=1)
    slope, intercept = coeffs
    fitted_values = np.polyval(coeffs, bin_centers)
    fitted_values_list = np.atleast_1d(fitted_values).tolist()
    fit_coefficients = {"slope": float(slope), "intercept": float(intercept)}
    return fit_coefficients, fitted_values_list


def save_linear_fit_coefficients(year, fit_coefficients, output_dir="."):
    """Persist linear-fit coefficients for ``year`` when available.

    Parameters
    ----------
    year : str or int
        Label used to name the output file.
    fit_coefficients : Mapping[str, float]
        Dictionary containing the ``slope`` and ``intercept`` keys produced by
        :func:`compute_linear_fit`.
    output_dir : str, optional
        Directory where the artifact should be written.  Created if missing.

    Notes
    -----
    Writes ``diboson_sf_{year}_linear_fit.json`` containing the linear-fit JSON
    to ``output_dir`` so the README-described coefficients are visible from the
    implementation.
    """
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
    """Generate the diagnostic plot overlaying measured scale factors and the fit.

    Parameters
    ----------
    year : str or int
        Year label included in the plot title and filename.
    channel : str
        Channel label included in the plot title.
    bin_centers : Sequence[float]
        Abscissa of the plotted points corresponding to ``njets`` bin centers.
    scale_factors : Sequence[float]
        Scale-factor values to scatter with uncertainties.
    fitted_values : Sequence[float]
        Linear-fit evaluation to overlay; may be empty when the fit fails.
    output_dir : str, optional
        Directory where the figure is stored.  Created if it does not exist.

    Notes
    -----
    Writes ``diboson_sf_{year}.png`` containing the scale-factor diagnostic plot
    to ``output_dir`` so the README-stated PNG artifact is documented alongside
    its implementation.
    """
    if bin_centers is None or scale_factors is None:
        print("No bin centers or scale factors available for plotting; skipping plot generation.")
        return

    if np.size(bin_centers) == 0 or np.size(scale_factors) == 0:
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
    if fitted_values is not None and np.size(fitted_values) > 0:
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
    allowed_years=None,
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

    year_str = str(year)
    whitelist_set = None

    if allowed_years:
        allowed_tokens = {str(token) for token in allowed_years}
        if year_str.lower() != ALL_YEARS_SENTINEL:
            filter_tokens = {year_str}
        else:
            filter_tokens = allowed_tokens

        whitelist_set = set()
        for token in filter_tokens:
            derived = _derive_process_subset_for_year(proc_list, token)
            if derived:
                whitelist_set.update(derived)

        if not whitelist_set:
            raise KeyError(
                "No processes remain after filtering for requested year(s) "
                f"'{', '.join(sorted(filter_tokens))}'."
            )

    yields = get_yields_in_bins(
        hin_dict,
        proc_list,
        bins,
        hist_name=hist_name,
        channel_name=channel,
        extra_slices=None,
        process_whitelist=whitelist_set,
    )

    num_bins = len(bins) - 1
    diboson = [0.0] * num_bins
    data = [0.0] * num_bins
    other = [0.0] * num_bins

    for proc, vals in yields.items():
        if proc.startswith("flip"):
            continue  # Skip flip samples
        if proc.startswith("WZTo") or proc.startswith("ZZTo") or proc.startswith("WWTo"):
            diboson = [x + (val or 0.0) for x, (val, _) in zip(diboson, vals)]
        elif "data" in proc:
            data = [val or 0.0 for val, _ in vals]
        else:
            other = [x + (val or 0.0) for x, (val, _) in zip(other, vals)]

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
            "file that contains histograms for all requested years. When a shared "
            "file is used, the script filters the process axis for names encoding "
            "each requested year (e.g. tokens like 'central2023' or 'central2022EE'). "
            "Passing '--year all' with a shared file automatically processes every "
            "discovered year and also writes a combined entry that sums them."
        ),
    )
    parser.add_argument("--hist-name", default="njets", help="Histogram name")
    parser.add_argument("--channel", default="3l_CR", help="Channel name")
    parser.add_argument(
        "-y",
        "--year",
        nargs="+",
        default=["2022"],
        help=(
            "One or more data-taking years to process. With a shared file, using "
            "'all' causes the script to discover every year token embedded in the "
            "process names and to include a combined result that sums them."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where per-year outputs (JSON, plots) will be written.",
    )
    args = parser.parse_args()

    years = [str(year) for year in args.year]

    if not years:
        parser.error("At least one year must be provided via -y/--year.")

    has_all_years = any(year.lower() == ALL_YEARS_SENTINEL for year in years)
    requested_specific_years = [
        year for year in years if year.lower() != ALL_YEARS_SENTINEL
    ]

    shared_pkl_for_years = False
    year_process_map = {}

    if len(args.pkl) == 1:
        pkl_arg = args.pkl[0]
        if "{year}" in pkl_arg:
            if has_all_years:
                parser.error(
                    "--year all cannot be used with a template --pkl path. Provide "
                    "a shared pickle containing all years instead."
                )
            pkl_paths = [pkl_arg.format(year=year) for year in years]
            missing_paths = [path for path in pkl_paths if not os.path.exists(path)]
            if missing_paths:
                parser.error(
                    "The following expanded --pkl paths do not exist: "
                    f"{', '.join(missing_paths)}"
                )
        elif len(years) == 1:
            pkl_paths = [pkl_arg]
            if has_all_years:
                shared_pkl_for_years = True
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

    bins = list(range(0, 7))

    cached_inputs = {}

    if shared_pkl_for_years:
        sample_path = pkl_paths[0]
        if sample_path in cached_inputs:
            sample_dict = cached_inputs[sample_path]
        else:
            sample_dict = load_pkl_file(sample_path)
            cached_inputs[sample_path] = sample_dict

        try:
            sample_hist = sample_dict[args.hist_name]
        except KeyError:
            parser.error(
                f"Histogram '{args.hist_name}' not found in shared pickle '{sample_path}'."
            )

        try:
            process_values = list(sample_hist.axes["process"])
        except KeyError:
            parser.error(
                f"Histogram '{args.hist_name}' in shared pickle '{sample_path}' "
                "is missing the 'process' axis required for scale factors."
            )
        token_map = _map_year_tokens_to_processes(process_values)
        discovered_years = sorted(token_map)
        if discovered_years:
            year_process_map = {
                str(year_token): sorted(process_set)
                for year_token, process_set in token_map.items()
            }
            print(
                "Discovered years embedded in process names: "
                f"{', '.join(discovered_years)}"
            )
        else:
            print(
                "No embedded year tokens detected on the process axis of the shared "
                "file; all processes will be treated as combined."
            )

        if has_all_years and years == [ALL_YEARS_SENTINEL] and discovered_years:
            years = discovered_years + [ALL_YEARS_SENTINEL]
            pkl_paths = [sample_path] * len(years)
            print(
                "Automatically expanding '--year all' to process each discovered "
                f"year: {', '.join(discovered_years)}"
            )
            requested_specific_years = [
                year for year in years if year.lower() != ALL_YEARS_SENTINEL
            ]

        if discovered_years:
            missing = [
                year for year in requested_specific_years if year not in year_process_map
            ]
            if missing:
                parser.error(
                    "The shared --pkl file does not contain processes encoding the "
                    "requested year(s): " + ", ".join(missing)
                )
        else:
            if requested_specific_years:
                parser.error(
                    "The shared --pkl file does not embed year tokens in the process "
                    "names, so specific years cannot be selected."
                )

    has_all_years = any(year.lower() == ALL_YEARS_SENTINEL for year in years)

    if has_all_years and not shared_pkl_for_years:
        parser.error(
            "--year all requires providing a single shared --pkl file that contains "
            "all years."
        )

    if not shared_pkl_for_years:
        missing_paths = [path for path in pkl_paths if not os.path.exists(path)]
        if missing_paths:
            parser.error(
                "The following --pkl paths do not exist: " + ", ".join(missing_paths)
            )

    results = {}
    summary = {}
    for year, pkl_path in zip(years, pkl_paths):
        try:
            allowed_years = None

            if shared_pkl_for_years:
                if str(year).lower() != ALL_YEARS_SENTINEL:
                    if year_process_map and str(year) not in year_process_map:
                        parser.error(
                            "No processes encoding the requested year '"
                            f"{year}' were found in the shared file."
                        )
                    allowed_years = [year]
                else:
                    if requested_specific_years:
                        allowed_years = requested_specific_years
                    elif year_process_map:
                        allowed_years = sorted(year_process_map)

            results[year] = process_year(
                pkl_path,
                year,
                args.hist_name,
                args.channel,
                bins,
                cache=cached_inputs,
                allowed_years=allowed_years,
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
