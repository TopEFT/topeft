#!/usr/bin/env python3
'''
    This script is made specifically for Run3 flip scale factor calculation based on l0eta distribution.
    Default eta bins are [-3, -1.479, 0, 1.479, 3], for +/- endcap/barrel regions. Hence there will be total of four SFs.

    Run:
      python flip_sf_run3.py /path/to/np.pkl.gz -y 2022
'''

import argparse
import gzip
import json
import pickle

import numpy as np

import topcoffea.modules.histEFT as tc_histEFT
import topcoffea.modules.sparseHist as tc_sparseHist


def load_pkl_file(pkl_file):
    with gzip.open(pkl_file, "rb") as f:
        return pickle.load(f)


def _has_axis(histogram, axis_name):
    try:
        histogram.axes[axis_name]
        return True
    except Exception:
        return False


def _values_with_flow_or_overflow(hist_slice):
    """
    Lifted (simplified) from make_cr_and_sr_plots.py.
    Returns values including under/over-flow bins when possible.
    """
    if isinstance(hist_slice, tc_histEFT.HistEFT):
        evaluated = hist_slice.eval({})
        if isinstance(evaluated, dict):
            if () in evaluated:
                return np.asarray(evaluated[()])
            return np.asarray(next(iter(evaluated.values())))
        return np.asarray(evaluated)

    values_method = hist_slice.values

    # Try overflow="all" (some impls), else flow=True, else default
    try:
        return np.asarray(values_method(overflow="all"))
    except TypeError:
        pass
    try:
        return np.asarray(values_method(flow=True))
    except TypeError:
        pass

    values = values_method()
    if isinstance(values, dict):
        if () in values:
            return np.asarray(values[()])
        return np.asarray(next(iter(values.values())))
    return np.asarray(values)


def _eval_without_underflow(hist_slice):
    """
    Lifted (simplified) from make_cr_and_sr_plots.py.
    Evaluates and removes the underflow bin along the dense axis (assumed first dim).
    """
    evaluated = hist_slice.eval({})
    if isinstance(evaluated, dict):
        if () in evaluated:
            evaluated = evaluated[()]
        else:
            evaluated = next(iter(evaluated.values()))
    values = np.asarray(evaluated)
    if values.shape[0] == 0:
        return values
    # Drop underflow
    return values[1:]


def _integrate_nominal_axis(histogram):
    """
    Like make_cr_and_sr_plots.py: if there's a 'systematic' axis, integrate to 'nominal'.
    """
    if histogram is None:
        return None
    if not _has_axis(histogram, "systematic"):
        return histogram
    try:
        return histogram.integrate("systematic", "nominal")
    except Exception:
        return histogram


def _integrate_out_axis(histogram, axis_name, value):
    """
    Integrate out an axis by selecting a category/value or summing.
    """
    if histogram is None:
        return None
    if not _has_axis(histogram, axis_name):
        return histogram

    # If value is sum, use [{"axis": sum}] pattern when available (hist-like)
    if value is sum:
        try:
            return histogram[{axis_name: sum}]
        except Exception:
            # fallback: integrate then sum by selecting all bins
            try:
                labels = list(histogram.axes[axis_name])
                return histogram.integrate(axis_name, labels)[{axis_name: sum}]
            except Exception:
                return histogram

    # Else integrate to a specific category/value
    try:
        return histogram.integrate(axis_name, value)
    except Exception:
        # Some objects accept direct slicing
        try:
            return histogram[{axis_name: value}]
        except Exception:
            return histogram


def _reduce_for_eta(histogram, eta_axis_name="l0eta"):
    """
    Replicate the reduction logic from make_cr_and_sr_plots.py style:
      - integrate systematic -> nominal
      - integrate quadratic_term -> sum (if present)
      - after this, eta axis should be the remaining dense axis (or one of them)
    """
    h = histogram

    # nominal systematic
    h = _integrate_nominal_axis(h)

    # quadratic_term: sum over it (your input has it)
    if _has_axis(h, "quadratic_term"):
        h = _integrate_out_axis(h, "quadratic_term", sum)

    # At this point we expect eta axis to exist
    if not _has_axis(h, eta_axis_name):
        raise RuntimeError(
            f"After reduction, missing eta axis '{eta_axis_name}'. "
            f"Available axes: {[getattr(ax,'name',None) for ax in getattr(h,'axes',[])]}"
        )

    return h


def _sum_in_eta_bins(edges, values_no_underflow, eta_bins):
    """
    values_no_underflow are the bin contents excluding underflow.
    edges are the axis edges including under/overflow definition.
    We assume values_no_underflow corresponds to 'visible bins + overflow' like _eval_without_underflow in make_cr_and_sr_plots.
    Here we ignore overflow for binning, unless your axis edges include an overflow edge (they don't).
    """
    edges = np.asarray(edges, dtype=float)

    # values_no_underflow length should match number of visible bins (+ maybe overflow)
    # Use only the visible bins for eta slicing:
    n_visible = len(edges) - 1
    values_vis = np.asarray(values_no_underflow, dtype=float)
    if values_vis.shape[0] > n_visible:
        values_vis = values_vis[:n_visible]
    elif values_vis.shape[0] < n_visible:
        padded = np.zeros(n_visible, dtype=float)
        padded[: values_vis.shape[0]] = values_vis
        values_vis = padded

    out = []
    for low, high in zip(eta_bins[:-1], eta_bins[1:]):
        bin_indices = [
            j for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))
            if hi > low and lo < high
        ]
        val = float(values_vis[bin_indices].sum()) if bin_indices else 0.0
        out.append((val, 0.0))
    return out


def get_yields_in_eta_bins(hin_dict, proc_list, eta_bins, hist_name, channel_name, eta_axis_name="l0eta", verbose=False):
    h = hin_dict[hist_name]
    yields = {}

    for proc in proc_list:
        try:
            # Slice categorical axes like the plotting script does
            h_sel = h[{"process": proc, "channel": channel_name}]

            if verbose:
                axes_dbg = [(getattr(ax, "name", None), getattr(ax, "label", None), type(ax).__name__) for ax in h_sel.axes]
                print(f"[DEBUG] proc={proc} axes after slice: {axes_dbg}")

            # Reduce systematic/quadratic_term like make_cr_and_sr_plots.py does
            h_eta = _reduce_for_eta(h_sel, eta_axis_name=eta_axis_name)

            # Extract eta edges
            eta_axis = h_eta.axes[eta_axis_name]
            edges = eta_axis.edges

            # Evaluate and drop underflow (same convention as make_cr_and_sr_plots)
            values_no_underflow = _eval_without_underflow(h_eta)

            # If still multi-dim (should not happen after the reductions), collapse everything except eta
            # by summing over remaining dims
            if values_no_underflow.ndim != 1:
                # Sum over all but first axis
                values_no_underflow = np.sum(values_no_underflow, axis=tuple(range(1, values_no_underflow.ndim)))

            yields[proc] = _sum_in_eta_bins(edges, values_no_underflow, eta_bins)

        except Exception as e:
            print(f"  Error slicing/reducing for proc {proc}: {e}")
            yields[proc] = [(None, None)] * (len(eta_bins) - 1)

    return yields


def make_flipsf_json(eta_bins, scale_factors, year):
    if len(eta_bins) != len(scale_factors) + 1:
        raise ValueError("Number of scale factors must be one less than number of bin edges.")

    key_name = "FlipSF_eta"
    sf_json = {
        key_name: {
            f"[{eta_bins[i]},{eta_bins[i+1]}]": scale_factors[i]
            for i in range(len(scale_factors))
        }
    }
    outname = f"flip_sf_{year}.json"
    with open(outname, "w") as f:
        json.dump(sf_json, f, indent=2)
    print(f"Scaling factors saved to {outname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file", help="Path to the input pkl.gz file")
    parser.add_argument("--hist-name", default="l0eta", help="Histogram key in the pkl dict")
    parser.add_argument("--channel", default="2lss_ee_CRflip_3j", help="Channel name")
    parser.add_argument("--eta-axis-name", default="l0eta", help="Eta axis name inside the histogram")
    parser.add_argument("--eta-bins", nargs="+", type=float, default=[-3, -1.479, 0, 1.479, 3],
                        help="Eta bin edges, e.g. --eta-bins -3 -1.479 0 1.479 3")
    parser.add_argument("-y", "--year", default="2022", help="The year of the sample (only used for output filename)")
    parser.add_argument("--verbose", action="store_true", help="Print debug info about axes after slicing")
    args = parser.parse_args()

    eta_bins = args.eta_bins
    hin_dict = load_pkl_file(args.pkl_file)

    if args.hist_name not in hin_dict:
        raise KeyError(f"Histogram '{args.hist_name}' not found. Keys: {list(hin_dict.keys())}")

    h = hin_dict[args.hist_name]

    # Build process list from axis 'process'
    try:
        proc_list = list(h.axes["process"])
    except Exception:
        raise RuntimeError("Histogram does not expose a 'process' axis in the expected way.")

    yields = get_yields_in_eta_bins(
        hin_dict,
        proc_list,
        eta_bins,
        hist_name=args.hist_name,
        channel_name=args.channel,
        eta_axis_name=args.eta_axis_name,
        verbose=args.verbose,
    )

    flips = None
    data = None
    other = None

    for proc, vals in yields.items():
        if any(v is None for v, _ in vals):
            continue

        vals_only = [v for v, _ in vals]
        p = proc.lower()

        if "flips" in p:
            flips = vals_only if flips is None else [x + v for x, v in zip(flips, vals_only)]
        elif "data" in p:
            data = vals_only if data is None else [x + v for x, v in zip(data, vals_only)]
        else:
            other = vals_only if other is None else [x + v for x, v in zip(other, vals_only)]

    if flips is None or data is None or other is None:
        raise RuntimeError(
            "Could not build flips/data/other arrays. "
            f"Found flips={flips is not None}, data={data is not None}, other={other is not None}."
        )

    scale_factors = []
    for d, o, f in zip(data, other, flips):
        if f != 0:
            scale_factors.append((d - o) / f)
        else:
            scale_factors.append(float("nan"))

    make_flipsf_json(eta_bins, scale_factors, year=args.year)

    print("eta bins =", eta_bins)
    print("flips =", flips)
    print("data  =", data)
    print("other =", other)
    print("SFs   =", scale_factors)


if __name__ == "__main__":
    main()