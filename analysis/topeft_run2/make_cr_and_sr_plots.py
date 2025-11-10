import numpy as np
import os
import copy
import datetime
import argparse
import re
from collections import OrderedDict
from collections.abc import Mapping

import logging
from decimal import Decimal
import inspect
import math
import warnings
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import FixedFormatter, FixedLocator
from cycler import cycler
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mplhep as hep
import hist
from matplotlib.transforms import Bbox
from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d

from topcoffea.scripts.make_html import make_html
import topcoffea.modules.utils as utils
from topeft.modules.yield_tools import YieldTools


_logger = logging.getLogger(__name__)
_ORIGINAL_SPARSEHIST_READ_FROM_REDUCE = SparseHist._read_from_reduce.__func__
_VALUES_METHOD_CAPS = {}


def _fast_sparsehist_from_reduce(cls, cat_axes, dense_axes, init_args, dense_hists):
    """Fast reconstruction helper used to patch :class:`SparseHist` pickles."""

    try:
        histogram = cls(*cat_axes, *dense_axes, **init_args)

        if dense_hists:
            categorical_axes = histogram.categorical_axes
            if categorical_axes:
                fill_payload = {axis.name: [] for axis in categorical_axes}
                for index_key in dense_hists:
                    categories = histogram.index_to_categories(index_key)
                    for axis, category in zip(categorical_axes, categories):
                        fill_payload[axis.name].append(category)

                hist.Hist.fill(histogram, **fill_payload)

        histogram._dense_hists = (
            dense_hists.copy() if hasattr(dense_hists, "copy") else dict(dense_hists)
        )
        return histogram
    except Exception:  # pragma: no cover - defensive fallback
        _logger.exception("Falling back to the original SparseHist deserializer.")
        return _ORIGINAL_SPARSEHIST_READ_FROM_REDUCE(
            cls, cat_axes, dense_axes, init_args, dense_hists
        )


SparseHist._read_from_reduce = classmethod(_fast_sparsehist_from_reduce)

from topcoffea.modules.paths import topcoffea_path
from topeft.modules.paths import topeft_path
import topeft.modules.get_rate_systs as grs
from topcoffea.modules.get_param_from_jsons import GetParam
get_tc_param = GetParam(topcoffea_path("params/params.json"))
import yaml

with open(topeft_path("params/cr_sr_plots_metadata.yml")) as f:
    _META = yaml.safe_load(f)


def _compile_data_driven_prefixes(raw_specs):
    """Return compiled regex objects for each configured data-driven prefix."""

    matchers = []
    for spec in raw_specs or ():
        if spec is None:
            continue
        if isinstance(spec, str):
            value = spec.strip()
            if not value:
                continue
            matchers.append(re.compile(rf"^{re.escape(value)}"))
        elif isinstance(spec, dict):
            pattern = spec.get("pattern")
            prefix = spec.get("prefix")
            if pattern:
                matchers.append(re.compile(pattern))
            elif prefix:
                matchers.append(re.compile(rf"^{re.escape(prefix)}"))
        else:
            raise TypeError(
                "Unsupported DATA_DRIVEN_PREFIXES entry type '{}'.".format(type(spec).__name__)
            )
    return tuple(matchers)


DATA_DRIVEN_MATCHERS = _compile_data_driven_prefixes(
    _META.get("DATA_DRIVEN_PREFIXES", [])
)
DATA_ERR_OPS = _META["DATA_ERR_OPS"]
MC_ERROR_OPS = _META["MC_ERROR_OPS"]
if isinstance(MC_ERROR_OPS.get("edgecolor"), list):
    MC_ERROR_OPS["edgecolor"] = tuple(MC_ERROR_OPS["edgecolor"])
CR_CHAN_DICT = _META["CR_CHAN_DICT"]
SR_CHAN_DICT = _META["SR_CHAN_DICT"]
CR_GROUP_INFO = _META.get("CR_GRP_MAP", {})
SR_GROUP_INFO = _META.get("SR_GRP_MAP", {})
CR_GRP_PATTERNS = {k: v.get("patterns", []) for k, v in CR_GROUP_INFO.items()}
SR_GRP_PATTERNS = {k: v.get("patterns", []) for k, v in SR_GROUP_INFO.items()}
CR_GRP_MAP = {k: [] for k in CR_GRP_PATTERNS.keys()}
SR_GRP_MAP = {k: [] for k in SR_GRP_PATTERNS.keys()}
SR_SIGNAL_GROUP_KEYS = {"ttH", "ttlnu", "ttll", "tXq", "tttt"}
SIGNAL_WC_MATCHES = ("ttH", "tllq", "ttlnu", "ttll", "tHq", "tttt")
CR_KNOWN_CHANNELS = {chan for chans in CR_CHAN_DICT.values() for chan in chans}
SR_KNOWN_CHANNELS = {chan for chans in SR_CHAN_DICT.values() for chan in chans}
FILL_COLORS = {k: v.get("color") for k, v in {**CR_GROUP_INFO, **SR_GROUP_INFO}.items()}
DEFAULT_STACK_COLORS = (
    "tab:blue",
    "darkgreen",
    "tab:orange",
    "tab:cyan",
    "tab:purple",
    "tab:pink",
    "tan",
    "mediumseagreen",
    "tab:red",
    "brown",
    "goldenrod",
    "yellow",
    "olive",
    "coral",
    "navy",
    "yellowgreen",
    "aquamarine",
    "black",
    "plum",
    "gray",
)
WCPT_EXAMPLE = _META["WCPT_EXAMPLE"]
LUMI_COM_PAIRS = _META["LUMI_COM_PAIRS"]
PROC_WITHOUT_PDF_RATE_SYST = _META["PROC_WITHOUT_PDF_RATE_SYST"]
REGION_PLOTTING = _META.get("REGION_PLOTTING", {})
STACKED_RATIO_STYLE = _META.get("STACKED_RATIO_STYLE", {})

YEAR_TOKEN_RULES = {
    "2016": {
        "mc_wl": ["UL16"],
        "mc_bl": ["UL16APV"],
        "data_wl": ["UL16"],
        "data_bl": ["UL16APV"],
    },
    "2016APV": {
        "mc_wl": ["UL16APV"],
        "data_wl": ["UL16APV"],
    },
    "2017": {"mc_wl": ["UL17"], "data_wl": ["UL17"]},
    "2018": {"mc_wl": ["UL18"], "data_wl": ["UL18"]},
    "2022": {"mc_wl": ["2022"], "data_wl": ["2022"]},
    "2022EE": {"mc_wl": ["2022EE"], "data_wl": ["2022EE"]},
    "2023": {"mc_wl": ["2023"], "data_wl": ["2023"]},
    "2023BPix": {"mc_wl": ["2023BPix"], "data_wl": ["2023BPix"]},
}

YEAR_AGGREGATE_ALIASES = {
    "run2": ("2016", "2016APV", "2017", "2018"),
    "run3": ("2022", "2022EE", "2023", "2023BPix"),
}

_YEAR_TOKEN_CANONICAL = {token.lower(): token for token in YEAR_TOKEN_RULES}

YEAR_WHITELIST_OPTIONALS = set()
for _year_rule in YEAR_TOKEN_RULES.values():
    YEAR_WHITELIST_OPTIONALS.update(_year_rule.get("mc_wl", []))
    YEAR_WHITELIST_OPTIONALS.update(_year_rule.get("data_wl", []))


def _normalize_year_tokens(raw_values):
    """Return canonical year tokens expanded from *raw_values*.

    Aggregate aliases such as ``run2``/``run3`` are expanded, inputs are
    interpreted case-insensitively, and the returned sequence contains only
    tokens known to :data:`YEAR_TOKEN_RULES`.
    """

    normalized = []
    seen = set()
    for raw_value in raw_values or ():
        if raw_value is None:
            continue
        for token in str(raw_value).split(","):
            cleaned = token.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            expansion = YEAR_AGGREGATE_ALIASES.get(lowered, (cleaned,))
            for expanded in expansion:
                canonical = _YEAR_TOKEN_CANONICAL.get(str(expanded).strip().lower())
                if canonical is None or canonical in seen:
                    continue
                seen.add(canonical)
                normalized.append(canonical)
    return normalized


def _hist_has_content(histogram):
    """Return True if *histogram* contains any finite, non-zero entries."""

    hist_view = histogram.view(flow=True)

    def _collect_arrays(view):
        if isinstance(view, Mapping):
            arrays = []
            for subview in view.values():
                arrays.extend(_collect_arrays(subview))
            return arrays
        data = view.value if hasattr(view, "value") else view
        try:
            arr = np.asarray(data, dtype=float)
        except (TypeError, ValueError):
            try:
                arr = np.asarray(data)
            except (TypeError, ValueError):
                return []
        return [arr]

    for values in _collect_arrays(hist_view):
        try:
            finite_mask = np.isfinite(values)
        except (TypeError, ValueError):
            continue
        if not np.any(finite_mask):
            continue
        if np.any(~np.isclose(values[finite_mask], 0.0, atol=1e-12)):
            return True
    return False


logger = logging.getLogger(__name__)

# This script takes an input pkl file that should have both data and background MC included.
# Use the -y option to specify one or more years.
# There are various other options available from the command line.
# For example, to make unit normalized plots for 2017+2018, with the timestamp appended to the directory name, you would run:
#     python make_cr_and_sr_plots.py -f histos/your.pkl.gz -o ~/www/somewhere/in/your/web/dir -n some_dir_name -y 2017 2018 -t -u

yt = YieldTools()

######### Utility functions #########

# Takes a dictionary where the keys are catetory names and keys are lists of bin names in the category, and a string indicating what type of info (njets, or lepflav) to remove
# Returns a dictionary of the same structure, except with njet or lepflav info stripped off of the bin names
# E.g. if a value was ["cat_a_1j","cat_b_1j","cat_b_2j"] and we passed "njets", we should return ["cat_a","cat_b"]
def get_dict_with_stripped_bin_names(in_chan_dict,type_of_info_to_strip):
    out_chan_dict = {}
    for cat,bin_names in in_chan_dict.items():
        out_chan_dict[cat] = []
        for bin_name in bin_names:
            if type_of_info_to_strip == "njets":
                bin_name_no_njet = yt.get_str_without_njet(bin_name)
            elif type_of_info_to_strip == "lepflav":
                bin_name_no_njet = yt.get_str_without_lepflav(bin_name)
            else:
                raise Exception(f"Error: Unknown type of string to remove \"{type_of_info_to_strip}\".")
            if bin_name_no_njet not in out_chan_dict[cat]:
                out_chan_dict[cat].append(bin_name_no_njet)
    return (out_chan_dict)


def _apply_channel_transforms(channel_name, transformations):
    transformed = channel_name
    for transform in transformations:
        if transform == "njets":
            transformed = yt.get_str_without_njet(transformed)
        elif transform == "lepflav":
            transformed = yt.get_str_without_lepflav(transformed)
        else:
            raise ValueError(f"Unsupported channel transformation '{transform}'")
    return transformed


def _apply_secondary_ticks(ax, axis="x"):
    """Install evenly spaced secondary ticks between existing major ticks."""

    if axis not in {"x", "y"}:
        raise ValueError(f"Unsupported axis '{axis}'. Expected 'x' or 'y'.")

    axis_obj = ax.xaxis if axis == "x" else ax.yaxis

    try:
        major_ticks = np.asarray(axis_obj.get_ticklocs(minor=False), dtype=float)
    except Exception:
        return

    if major_ticks.size < 2:
        return

    unique_ticks = np.unique(major_ticks)
    if unique_ticks.size < 2:
        return

    unique_ticks.sort()
    deltas = np.diff(unique_ticks)
    valid_mask = deltas > 0
    if not np.any(valid_mask):
        return

    minor_ticks = []
    for start, delta in zip(unique_ticks[:-1][valid_mask], deltas[valid_mask]):
        step = delta / 5.0
        minor_ticks.extend(start + step * np.arange(1, 5))

    if not minor_ticks:
        return

    axis_obj.set_minor_locator(FixedLocator(sorted(minor_ticks)))


def _integrate_category(histogram, hist_cat, axes_to_integrate):
    """Integrate a histogram over the provided axes, returning None on failure."""

    if histogram is None:
        return None

    try:
        integrated = yt.integrate_out_appl(histogram, hist_cat)
        integrated = yt.integrate_out_cats(integrated, axes_to_integrate)[{"channel": sum}]
    except Exception:
        return None

    return integrated


def _validate_bin_edges(edges):
    """Return a 1D numpy array of strictly increasing bin edges."""

    array = np.asarray(edges, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("Bin edges must be a 1D sequence with at least two entries.")

    deltas = np.diff(array)
    if not np.all(np.isfinite(array)):
        raise ValueError("Bin edges must be finite values.")
    if not np.all(deltas > 0):
        raise ValueError("Bin edges must be strictly increasing.")

    return array


def _build_variable_axis_like(axis, edges):
    """Construct a Variable axis matching the metadata of an existing dense axis."""

    metadata = getattr(axis, "metadata", None)
    label = getattr(axis, "label", "")
    traits = getattr(axis, "traits", None)
    underflow = getattr(traits, "underflow", None)
    overflow = getattr(traits, "overflow", None)
    flow = bool(underflow or overflow)

    return hist.axis.Variable(
        tuple(edges),
        name=getattr(axis, "name", None) or axis.name,
        label=label,
        metadata=metadata,
        flow=flow,
        underflow=underflow,
        overflow=overflow,
    )


def _rebin_flow_content(values_flow, variances_flow, original_edges, target_edges):
    """Aggregate flow-inclusive histogram contents onto a new edge definition."""

    original_edges = np.asarray(original_edges, dtype=float)
    target_edges = np.asarray(target_edges, dtype=float)

    edge_indices = []
    for edge in target_edges:
        matches = np.where(np.isclose(original_edges, edge, rtol=1e-9, atol=1e-12))[0]
        if matches.size == 0:
            raise ValueError(f"Requested edge {edge} not found in source histogram edges.")
        edge_indices.append(int(matches[0]))

    if any(next_idx <= idx for idx, next_idx in zip(edge_indices, edge_indices[1:])):
        raise ValueError("Target bin edges must be strictly increasing and align with source edges.")

    values_flow = np.asarray(values_flow, dtype=float)
    variances_flow = None if variances_flow is None else np.asarray(variances_flow, dtype=float)

    first_idx = edge_indices[0]
    last_idx = edge_indices[-1]

    underflow = values_flow[0] + values_flow[1 : 1 + first_idx].sum(axis=0)
    overflow = values_flow[-1] + values_flow[1 + last_idx : -1].sum(axis=0)

    rebinned_bins = [
        values_flow[1 + start : 1 + stop].sum(axis=0)
        for start, stop in zip(edge_indices[:-1], edge_indices[1:])
    ]

    rebinned_values = np.concatenate(
        [
            underflow[np.newaxis, ...],
            *[bin_values[np.newaxis, ...] for bin_values in rebinned_bins],
            overflow[np.newaxis, ...],
        ],
        axis=0,
    )

    rebinned_variances = None
    if variances_flow is not None:
        under_var = variances_flow[0] + variances_flow[1 : 1 + first_idx].sum(axis=0)
        over_var = variances_flow[-1] + variances_flow[1 + last_idx : -1].sum(axis=0)
        rebinned_vars = [
            variances_flow[1 + start : 1 + stop].sum(axis=0)
            for start, stop in zip(edge_indices[:-1], edge_indices[1:])
        ]
        rebinned_variances = np.concatenate(
            [
                under_var[np.newaxis, ...],
                *[var[np.newaxis, ...] for var in rebinned_vars],
                over_var[np.newaxis, ...],
            ],
            axis=0,
        )

    return rebinned_values, rebinned_variances


def _rebin_dense_histogram(dense_hist, axis_name, target_edges):
    """Return a rebinned copy of a dense hist.Hist along the specified axis."""

    try:
        axis_index = dense_hist.axes.index(axis_name)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Axis '{axis_name}' not found in histogram.") from exc

    original_axis = dense_hist.axes[axis_index]
    new_axes = []
    for idx, axis in enumerate(dense_hist.axes):
        if idx == axis_index:
            new_axes.append(_build_variable_axis_like(axis, target_edges))
        else:
            new_axes.append(axis)

    storage_type = dense_hist.storage_type
    new_hist = hist.Hist(*new_axes, storage=storage_type())

    values_flow = np.asarray(dense_hist.values(flow=True), dtype=float)
    variances_flow_raw = dense_hist.variances(flow=True)
    variances_flow = (
        None
        if variances_flow_raw is None
        else np.asarray(variances_flow_raw, dtype=float)
    )

    values_reordered = np.moveaxis(values_flow, axis_index, 0)
    variances_reordered = (
        None
        if variances_flow is None
        else np.moveaxis(variances_flow, axis_index, 0)
    )

    rebinned_values_reordered, rebinned_variances_reordered = _rebin_flow_content(
        values_reordered, variances_reordered, original_axis.edges, target_edges
    )

    rebinned_values = np.moveaxis(rebinned_values_reordered, 0, axis_index)
    rebinned_variances = (
        None
        if rebinned_variances_reordered is None
        else np.moveaxis(rebinned_variances_reordered, 0, axis_index)
    )

    view = new_hist.view(flow=True)
    if hasattr(view, "value"):
        view.value = rebinned_values
        if hasattr(view, "variance") and rebinned_variances is not None:
            view.variance = rebinned_variances
    else:
        view[...] = rebinned_values

    if hasattr(dense_hist, "label"):
        new_hist.label = dense_hist.label

    return new_hist


def _rebin_sparse_histogram(sparse_hist, axis_name, target_edges):
    """Return a rebinned copy of a SparseHist/HistEFT along a dense axis."""

    dense_axes = []
    replaced = False
    for axis in sparse_hist.dense_axes:
        if axis.name == axis_name:
            dense_axes.append(_build_variable_axis_like(axis, target_edges))
            replaced = True
        else:
            dense_axes.append(axis)

    if not replaced:
        raise ValueError(f"Axis '{axis_name}' not found in histogram dense axes.")

    rebinned_hist = sparse_hist.empty_from_axes(dense_axes=dense_axes)

    for index_key, dense_hist in sparse_hist._dense_hists.items():
        categories = sparse_hist.index_to_categories(index_key)
        new_index = rebinned_hist._fill_bookkeep(*categories)
        rebinned_hist._dense_hists[new_index] = _rebin_dense_histogram(
            dense_hist, axis_name, target_edges
        )

    if hasattr(sparse_hist, "label"):
        rebinned_hist.label = sparse_hist.label

    return rebinned_hist


def _clone_with_rebinned_axis(histogram, axis_name, target_edges):
    """Clone a histogram (dense or sparse) with rebinned dense axis."""

    if histogram is None:
        return None

    if hasattr(histogram, "_dense_hists"):
        return _rebin_sparse_histogram(histogram, axis_name, target_edges)

    if isinstance(histogram, hist.Hist):
        return _rebin_dense_histogram(histogram, axis_name, target_edges)

    raise TypeError(
        f"Unsupported histogram type '{type(histogram).__name__}' for rebinning operations."
    )


def _rebin_uncertainty_array(
    values,
    original_edges,
    target_edges,
    *,
    nominal=None,
    direction=None,
):
    """Aggregate a 1D uncertainty array according to new bin edges.

    When ``nominal`` is provided the ``values`` array is treated as a nominal yield
    shifted by an uncertainty (``direction`` must then be ``"up"`` or ``"down"``).
    The rebinned result preserves the nominal contribution and combines the
    bin-wise deviations in quadrature so uncorrelated uncertainties do not grow
    linearly when bins are merged.
    """

    if values is None:
        return None

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Uncertainty arrays must be one-dimensional for rebinning.")

    original_edges = np.asarray(original_edges, dtype=float)
    if original_edges.ndim != 1:
        raise ValueError("Original bin edges must form a one-dimensional array.")
    n_source_bins = original_edges.size - 1
    if array.size not in {n_source_bins, n_source_bins + 1}:
        raise ValueError(
            "Uncertainty arrays must match the source binning (with or without overflow)."
        )

    includes_overflow = array.size == n_source_bins + 1

    def _to_flow(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == n_source_bins:
            return np.concatenate(([0.0], arr, [0.0]))
        return np.concatenate(([0.0], arr[:-1], [arr[-1]]))

    def _trim_flow(flow_array):
        visible_and_overflow = flow_array[1:]
        if includes_overflow:
            return visible_and_overflow
        return visible_and_overflow[:-1]

    if nominal is None:
        values_flow = _to_flow(array)
        rebinned_values, _ = _rebin_flow_content(
            values_flow, None, original_edges, target_edges
        )
        return _trim_flow(rebinned_values)

    reference = np.asarray(nominal, dtype=float)
    if reference.ndim != 1:
        raise ValueError("Nominal arrays must be one-dimensional for rebinning.")
    if reference.shape != array.shape:
        raise ValueError("Nominal and uncertainty arrays must share the same shape.")

    if direction not in {"up", "down"}:
        raise ValueError(
            "Direction must be 'up' or 'down' when rebinding nominal-shifted uncertainties."
        )

    reference_flow = _to_flow(reference)
    rebinned_reference, _ = _rebin_flow_content(
        reference_flow, None, original_edges, target_edges
    )

    delta = array - reference
    if direction == "up":
        diff = np.clip(delta, a_min=0.0, a_max=None)
        sign = 1.0
    else:
        diff = np.clip(-delta, a_min=0.0, a_max=None)
        sign = -1.0

    diff_sq_flow = _to_flow(diff**2)
    zeros_flow = np.zeros_like(diff_sq_flow)
    _, rebinned_diff_sq = _rebin_flow_content(
        zeros_flow, diff_sq_flow, original_edges, target_edges
    )

    rebinned_reference = _trim_flow(rebinned_reference)
    rebinned_diff = np.sqrt(
        np.clip(_trim_flow(rebinned_diff_sq), a_min=0.0, a_max=None)
    )

    rebinned = rebinned_reference + sign * rebinned_diff
    if direction == "down":
        rebinned = np.clip(rebinned, a_min=0.0, a_max=None)

    return rebinned


def _determine_ratio_window(ratio_arrays, data_ratio_arrays, *, tolerance=1e-12):
    """Return ratio axis limits and warning flags given MC/data ratio samples."""

    ratio_windows = (
        (0.5, 1.5),
        (0.0, 2.0),
        (-1.0, 3.0),
    )
    ratio_window_deviations = (0.5, 1.0, 2.0)
    largest_low, largest_high = ratio_windows[-1]

    def _finite_segments(arrays):
        segments = []
        for arr in arrays or ():
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            finite_mask = np.isfinite(arr)
            if np.any(finite_mask):
                segments.append(arr[finite_mask])
        return segments

    finite_segments = _finite_segments(ratio_arrays)

    ratio_limits = ratio_windows[0]
    exceeds_largest_window = False
    if finite_segments:
        combined = np.concatenate(finite_segments)
        min_val = float(np.min(combined))
        max_val = float(np.max(combined))
        max_abs_deviation = float(np.max(np.abs(combined - 1.0)))

        selected_limits = ratio_windows[-1]
        for (low, high), allowed_deviation in zip(ratio_windows, ratio_window_deviations):
            if (
                max_abs_deviation <= allowed_deviation + tolerance
                and min_val >= low - tolerance
                and max_val <= high + tolerance
            ):
                selected_limits = (low, high)
                break

        ratio_limits = selected_limits

        exceeds_largest_window = (
            min_val < largest_low - tolerance or max_val > largest_high + tolerance
        )

    data_finite_segments = _finite_segments(data_ratio_arrays)
    data_exceeds_largest_window = False
    if data_finite_segments:
        data_combined = np.concatenate(data_finite_segments)
        data_min = float(np.min(data_combined))
        data_max = float(np.max(data_combined))
        data_exceeds_largest_window = (
            data_min < largest_low - tolerance or data_max > largest_high + tolerance
        )

    return ratio_limits, exceeds_largest_window, data_exceeds_largest_window


def _merge_mappings(base, updates):
    if not isinstance(base, dict) or not isinstance(updates, Mapping):
        return base
    for key, value in updates.items():
        if isinstance(value, Mapping):
            nested = base.get(key)
            if not isinstance(nested, dict):
                nested = {}
            base[key] = _merge_mappings(nested, value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _style_get(style, path, default=None):
    current = style
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _resolve_stacked_ratio_style(region_name, overrides=None):
    style_cfg = STACKED_RATIO_STYLE if isinstance(STACKED_RATIO_STYLE, Mapping) else {}
    defaults = style_cfg.get("defaults", {})
    resolved = copy.deepcopy(defaults) if isinstance(defaults, Mapping) else {}

    per_region = style_cfg.get("per_region", {})
    if isinstance(per_region, Mapping) and region_name in per_region:
        resolved = _merge_mappings(resolved, per_region[region_name])

    if overrides and isinstance(overrides, Mapping):
        resolved = _merge_mappings(resolved, overrides)

    return resolved


def _close_figure_payload(fig_payload):
    """Close matplotlib figures contained in *fig_payload*."""

    if fig_payload is None:
        return
    if isinstance(fig_payload, dict):
        for nested in fig_payload.values():
            _close_figure_payload(nested)
        return
    try:
        plt.close(fig_payload)
    except Exception:
        # Fall back to the global close-all safeguard when a payload does not
        # expose the standard matplotlib Figure interface.
        plt.close('all')


_WORKER_RENDER_CONTEXT = None


def _initialize_render_worker(
    region_ctx,
    save_dir_path,
    skip_syst_errs,
    unit_norm_bool,
    unblind_flag,
    stacked_log_y,
    verbose,
    prepared_payloads=None,
):
    """Store shared plotting context inside a worker process."""

    global _WORKER_RENDER_CONTEXT
    _WORKER_RENDER_CONTEXT = {
        "region_ctx": region_ctx,
        "save_dir_path": save_dir_path,
        "skip_syst_errs": skip_syst_errs,
        "unit_norm_bool": unit_norm_bool,
        "unblind_flag": unblind_flag,
        "stacked_log_y": stacked_log_y,
        "verbose": bool(verbose),
        "prepared_variables": dict(prepared_payloads or {}),
    }


def _render_variable_from_worker(task_id, payload):
    """Delegate variable rendering using the worker-local cached context."""

    if _WORKER_RENDER_CONTEXT is None:
        raise RuntimeError(
            "Worker render context is not initialised; expected ProcessPoolExecutor initializer to set it."
        )

    if isinstance(payload, tuple):
        var_name, category = payload
    else:
        var_name, category = payload, None

    ctx = _WORKER_RENDER_CONTEXT
    verbose = ctx.get("verbose", False)

    prepared_cache = ctx.setdefault("prepared_variables", {})
    variable_payload = prepared_cache.get(var_name)
    if var_name not in prepared_cache:
        variable_payload = _prepare_variable_payload(
            var_name,
            ctx["region_ctx"],
            verbose=verbose,
            unblind_flag=ctx["unblind_flag"],
        )
        prepared_cache[var_name] = variable_payload

    if category is None:
        stat_only, stat_and_syst, html_set = _render_variable(
            var_name,
            ctx["region_ctx"],
            ctx["save_dir_path"],
            ctx["skip_syst_errs"],
            ctx["unit_norm_bool"],
            ctx["stacked_log_y"],
            ctx["unblind_flag"],
            verbose=verbose,
            category=category,
            variable_payload=variable_payload,
        )
    else:
        if not variable_payload:
            stat_only, stat_and_syst, html_set = 0, 0, set()
        else:
            region_ctx = ctx["region_ctx"]
            channel_bins = variable_payload["channel_dict"].get(category)
            if channel_bins is None or _should_skip_category(
                region_ctx.category_skip_rules, category, var_name
            ):
                stat_only, stat_and_syst, html_set = 0, 0, set()
            else:
                stat_only, stat_and_syst, html_set = _render_variable_category(
                    var_name,
                    category,
                    channel_bins,
                    region_ctx=region_ctx,
                    channel_transformations=variable_payload["channel_transformations"],
                    hist_mc=variable_payload["hist_mc"],
                    hist_data=variable_payload["hist_data"],
                    hist_mc_sumw2_orig=variable_payload["hist_mc_sumw2_orig"],
                    is_sparse2d=variable_payload["is_sparse2d"],
                    save_dir_path=ctx["save_dir_path"],
                    skip_syst_errs=ctx["skip_syst_errs"],
                    unit_norm_bool=ctx["unit_norm_bool"],
                    stacked_log_y=ctx["stacked_log_y"],
                    unblind_flag=ctx["unblind_flag"],
                    verbose=verbose,
                )
    return task_id, stat_only, stat_and_syst, html_set


def _prepare_variable_payload(
    var_name,
    region_ctx,
    *,
    verbose=False,
    unblind_flag=False,
    metadata_only=False,
):
    """Prepare variable-level plotting inputs shared across categories."""

    histo = region_ctx.dict_of_hists[var_name]
    is_sparse2d = _is_sparse_2d_hist(histo)
    if is_sparse2d and region_ctx.skip_sparse_2d:
        return None
    if is_sparse2d and (var_name not in axes_info_2d) and ("_vs_" not in var_name):
        print(
            f"Warning: Histogram '{var_name}' identified as sparse 2D but lacks metadata; falling back to 1D plotting."
        )
        is_sparse2d = False

    channel_transformations = _resolve_channel_transformations(region_ctx, var_name)
    channel_dict = _apply_channel_dict_transformations(
        region_ctx.channel_map, channel_transformations
    )

    if metadata_only:
        return {
            "channel_dict": channel_dict,
            "channel_transformations": channel_transformations,
            "is_sparse2d": is_sparse2d,
        }

    hist_mc = histo.remove("process", region_ctx.samples_to_remove["mc"])
    hist_data = histo.remove("process", region_ctx.samples_to_remove["data"])
    hist_mc_sumw2_orig = region_ctx.sumw2_hists.get(var_name)

    if hist_mc_sumw2_orig is not None:
        hist_mc_sumw2_orig = hist_mc_sumw2_orig.remove(
            "process", region_ctx.samples_to_remove["mc"]
        )
        if region_ctx.sumw2_remove_signal and region_ctx.signal_samples:
            existing_signal = [
                sample
                for sample in region_ctx.signal_samples
                if sample in yt.get_cat_lables(hist_mc_sumw2_orig, "process")
            ]
            if existing_signal:
                hist_mc_sumw2_orig = hist_mc_sumw2_orig.remove(
                    "process", existing_signal
                )
        if (
            region_ctx.sumw2_remove_signal_when_blinded
            and region_ctx.signal_samples
            and not unblind_flag
        ):
            hist_mc_sumw2_orig = hist_mc_sumw2_orig.remove(
                "process", region_ctx.signal_samples
            )

    if region_ctx.debug_channel_lists and verbose:
        try:
            channels_lst = yt.get_cat_lables(histo, "channel")
        except Exception:
            channels_lst = []
        print("channels:", channels_lst)

    return {
        "channel_dict": channel_dict,
        "channel_transformations": channel_transformations,
        "hist_mc": hist_mc,
        "hist_data": hist_data,
        "hist_mc_sumw2_orig": hist_mc_sumw2_orig,
        "is_sparse2d": is_sparse2d,
    }


def _render_variable(
    var_name,
    region_ctx,
    save_dir_path,
    skip_syst_errs,
    unit_norm_bool,
    stacked_log_y,
    unblind_flag,
    *,
    verbose=False,
    category=None,
    variable_payload=None,
):
    """Render plots for *var_name* and return summary accounting."""

    label = region_ctx.variable_label
    if verbose:
        print(f"\n{label}: {var_name}")

    if variable_payload is None:
        variable_payload = _prepare_variable_payload(
            var_name,
            region_ctx,
            verbose=verbose,
            unblind_flag=unblind_flag,
        )
    if not variable_payload:
        return 0, 0, set()

    channel_dict = variable_payload["channel_dict"]

    stat_only_plots = 0
    stat_and_syst_plots = 0
    html_dirs = set()

    if category is not None:
        channel_items = (
            [(category, channel_dict.get(category))]
            if category in channel_dict
            else []
        )
    else:
        channel_items = list(channel_dict.items())

    for hist_cat, channel_bins in channel_items:
        if channel_bins is None:
            continue
        if _should_skip_category(region_ctx.category_skip_rules, hist_cat, var_name):
            continue

        stat_only, stat_and_syst, html_set = _render_variable_category(
            var_name,
            hist_cat,
            channel_bins,
            region_ctx=region_ctx,
            channel_transformations=variable_payload["channel_transformations"],
            hist_mc=variable_payload["hist_mc"],
            hist_data=variable_payload["hist_data"],
            hist_mc_sumw2_orig=variable_payload["hist_mc_sumw2_orig"],
            is_sparse2d=variable_payload["is_sparse2d"],
            save_dir_path=save_dir_path,
            skip_syst_errs=skip_syst_errs,
            unit_norm_bool=unit_norm_bool,
            stacked_log_y=stacked_log_y,
            unblind_flag=unblind_flag,
            verbose=verbose,
        )
        stat_only_plots += stat_only
        stat_and_syst_plots += stat_and_syst
        html_dirs.update(html_set)

    return stat_only_plots, stat_and_syst_plots, html_dirs


def _render_variable_category(
    var_name,
    hist_cat,
    channel_bins,
    *,
    region_ctx,
    channel_transformations,
    hist_mc,
    hist_data,
    hist_mc_sumw2_orig,
    is_sparse2d,
    save_dir_path,
    skip_syst_errs,
    unit_norm_bool,
    stacked_log_y,
    unblind_flag,
    verbose=False,
):
    """Render a single (variable, category) pair and return bookkeeping totals."""

    validate_channel_group(
        [hist_mc, hist_data],
        channel_bins,
        channel_transformations,
        region=region_ctx.name,
        subgroup=hist_cat,
        variable=var_name,
    )

    base_dir = save_dir_path or ""
    save_dir_path_tmp = os.path.join(base_dir, hist_cat)
    os.makedirs(save_dir_path_tmp, exist_ok=True)

    stat_only_plots = 0
    stat_and_syst_plots = 0
    html_dirs = set()

    if region_ctx.channel_mode == "aggregate":
        if verbose:
            # Category headings are mainly useful when debugging channel regrouping.
            print(f"\n\tCategory: {hist_cat}")

        axes_to_integrate_dict = {"channel": channel_bins}
        hist_mc_integrated = _integrate_category(hist_mc, hist_cat, axes_to_integrate_dict)
        hist_data_integrated = _integrate_category(
            hist_data, hist_cat, axes_to_integrate_dict
        )
        if hist_mc_integrated is None or hist_data_integrated is None:
            return 0, 0, html_dirs
        hist_mc_sumw2_integrated = None
        if hist_mc_sumw2_orig is not None:
            hist_mc_sumw2_integrated = _integrate_category(
                hist_mc_sumw2_orig, hist_cat, axes_to_integrate_dict
            )

        samples_to_rm = _collect_samples_to_remove(
            region_ctx.sample_removal_rules, hist_cat, region_ctx
        )
        hist_mc_integrated = hist_mc_integrated.remove("process", samples_to_rm)
        if hist_mc_sumw2_integrated is not None:
            hist_mc_sumw2_integrated = hist_mc_sumw2_integrated.remove(
                "process", samples_to_rm
            )

        p_err_arr = None
        m_err_arr = None
        p_err_arr_ratio = None
        m_err_arr_ratio = None
        syst_err_mode = False
        if not (is_sparse2d or skip_syst_errs):
            rate_systs_summed_arr_m, rate_systs_summed_arr_p = get_rate_syst_arrs(
                hist_mc_integrated,
                region_ctx.group_map,
                group_type=region_ctx.name,
            )
            shape_systs_summed_arr_m, shape_systs_summed_arr_p = get_shape_syst_arrs(
                hist_mc_integrated,
                group_type=region_ctx.name,
            )
            if var_name == "njets":
                diboson_samples = region_ctx.group_map.get("Diboson", [])
                if diboson_samples:
                    db_hist = _eval_without_underflow(
                        hist_mc_integrated.integrate("process", diboson_samples)[{"process": sum}]
                        .integrate("systematic", "nominal")
                    )
                    diboson_njets_syst = get_diboson_njets_syst_arr(
                        db_hist, bin0_njets=0
                    )
                    shape_systs_summed_arr_p = (
                        shape_systs_summed_arr_p + diboson_njets_syst
                    )
                    shape_systs_summed_arr_m = (
                        shape_systs_summed_arr_m + diboson_njets_syst
                    )
            nom_arr_all = _eval_without_underflow(
                hist_mc_integrated[{"process": sum}].integrate(
                    "systematic", "nominal"
                )
            )
            sqrt_sum_p = np.sqrt(
                np.asarray(shape_systs_summed_arr_p)
                + np.asarray(rate_systs_summed_arr_p)
            )
            sqrt_sum_m = np.sqrt(
                np.asarray(shape_systs_summed_arr_m)
                + np.asarray(rate_systs_summed_arr_m)
            )
            p_err_arr = nom_arr_all + sqrt_sum_p
            m_err_arr = nom_arr_all - sqrt_sum_m
            with np.errstate(divide="ignore", invalid="ignore"):
                p_err_arr_ratio = np.where(
                    nom_arr_all > 0, p_err_arr / nom_arr_all, 1
                )
                m_err_arr_ratio = np.where(
                    nom_arr_all > 0, m_err_arr / nom_arr_all, 1
                )
            syst_err_mode = "total" if unblind_flag else True

        if is_sparse2d:
            hist_mc_nominal = hist_mc_integrated[{"process": sum}].integrate(
                "systematic", "nominal"
            )
            hist_data_nominal = hist_data_integrated[{"process": sum}].integrate(
                "systematic", "nominal"
            )
            if not _hist_has_content(hist_mc_nominal):
                logger.warning(
                    "Empty histogram for hist_cat=%s var_name=%s, skipping 2D plot.",
                    hist_cat,
                    var_name,
                )
                return 0, 0, html_dirs
            if not _hist_has_content(hist_data_nominal):
                logger.warning(
                    "Empty data histogram for hist_cat=%s var_name=%s, skipping 2D plot.",
                    hist_cat,
                    var_name,
                )
                return 0, 0, html_dirs
            fig = make_sparse2d_fig(
                hist_mc_nominal,
                hist_data_nominal,
                var_name,
                channel_name=hist_cat,
                lumitag=region_ctx.lumi_pair[0],
                comtag=region_ctx.lumi_pair[1],
                per_panel=True,
            )
        else:
            hist_mc_integrated = hist_mc_integrated.integrate(
                "systematic", "nominal"
            )
            if hist_mc_sumw2_orig is not None and hist_mc_sumw2_integrated is not None:
                hist_mc_sumw2_integrated = hist_mc_sumw2_integrated.integrate(
                    "systematic", "nominal"
                )
            hist_data_integrated = hist_data_integrated.integrate(
                "systematic", "nominal"
            )
            if not _hist_has_content(hist_mc_integrated):
                logger.warning(
                    "Empty histogram for hist_cat=%s var_name=%s, skipping plot.",
                    hist_cat,
                    var_name,
                )
                return 0, 0, html_dirs
            if not _hist_has_content(hist_data_integrated):
                logger.warning(
                    "Empty data histogram for hist_cat=%s var_name=%s, skipping plot.",
                    hist_cat,
                    var_name,
                )
                return 0, 0, html_dirs
            x_range = (0, 250) if var_name == "ht" else None
            group = {k: v for k, v in region_ctx.group_map.items() if v}
            stacked_kwargs = {
                "h_mc_sumw2": hist_mc_sumw2_integrated,
                "syst_err": syst_err_mode,
                "err_p_syst": p_err_arr,
                "err_m_syst": m_err_arr,
                "err_ratio_p_syst": p_err_arr_ratio,
                "err_ratio_m_syst": m_err_arr_ratio,
                "unblind": unblind_flag,
                "set_x_lim": x_range,
                "log_scale": stacked_log_y,
                "style": region_ctx.stacked_ratio_style,
            }
            bins_override = region_ctx.analysis_bins.get(var_name)
            if bins_override is not None:
                stacked_kwargs["bins"] = bins_override
            fig = make_region_stacked_ratio_fig(
                hist_mc_integrated,
                hist_data_integrated,
                unit_norm_bool,
                var=var_name,
                group=group,
                lumitag=region_ctx.lumi_pair[0] if region_ctx.lumi_pair else None,
                comtag=region_ctx.lumi_pair[1] if region_ctx.lumi_pair else None,
                **stacked_kwargs,
            )
        title = hist_cat + "_" + var_name
        if unit_norm_bool:
            title = title + "_unitnorm"
        has_syst_inputs = any(
            err is not None
            for err in (
                p_err_arr,
                m_err_arr,
                p_err_arr_ratio,
                m_err_arr_ratio,
            )
        )
        if isinstance(fig, dict):
            combined_fig = fig["combined"]
            combined_fig.savefig(
                os.path.join(save_dir_path_tmp, title),
                bbox_inches="tight",
                pad_inches=0.05,
            )
            suffix_map = {"mc": "_MC", "data": "_data", "ratio": "_ratio"}
            for key, panel_fig in fig.items():
                if key == "combined":
                    continue
                suffix = suffix_map.get(key, f"_{key}")
                panel_fig.savefig(
                    os.path.join(save_dir_path_tmp, f"{title}{suffix}"),
                    bbox_inches="tight",
                    pad_inches=0.05,
                )
        else:
            fig.savefig(
                os.path.join(save_dir_path_tmp, title),
                bbox_inches="tight",
                pad_inches=0.05,
            )
        _close_figure_payload(fig)
        if has_syst_inputs:
            stat_and_syst_plots += 1
        else:
            stat_only_plots += 1
    elif region_ctx.channel_mode == "per-channel":
        channels = [
            chan
            for chan in channel_bins
            if chan in hist_mc.axes["channel"]
        ]
        if not channels:
            return 0, 0, html_dirs
        hist_mc_channel = hist_mc.integrate("channel", channels)[{'channel': sum}]
        hist_mc_integrated = hist_mc_channel.integrate(
            "systematic", "nominal"
        )
        hist_mc_sumw2 = None
        if hist_mc_sumw2_orig is not None:
            channels_sumw2 = [
                chan
                for chan in channel_bins
                if chan in hist_mc_sumw2_orig.axes["channel"]
            ]
            if channels_sumw2:
                hist_mc_sumw2 = hist_mc_sumw2_orig.integrate(
                    "channel", channels_sumw2
                )[{'channel': sum}]
                hist_mc_sumw2 = hist_mc_sumw2.integrate(
                    "systematic", "nominal"
                )
        channels_data = [
            chan
            for chan in channel_bins
            if chan in hist_data.axes["channel"]
        ]
        hist_data_channel = hist_data.integrate("channel", channels_data)[{'channel': sum}]
        hist_data_integrated = hist_data_channel.integrate(
            "systematic", "nominal"
        )

        syst_err = False
        err_p_syst = None
        err_m_syst = None
        err_ratio_p_syst = None
        err_ratio_m_syst = None
        if not skip_syst_errs:
            try:
                rate_systs_summed_arr_m, rate_systs_summed_arr_p = get_rate_syst_arrs(
                    hist_mc_channel,
                    region_ctx.group_map,
                    group_type=region_ctx.name,
                )
                shape_systs_summed_arr_m, shape_systs_summed_arr_p = get_shape_syst_arrs(
                    hist_mc_channel,
                    group_type=region_ctx.name,
                )
            except Exception as exc:
                print(
                    f"Warning: Failed to compute {region_ctx.name} systematics for {hist_cat} {var_name}: {exc}"
                )
            else:
                nominal_projection = hist_mc_channel[{"process": sum}].integrate(
                    "systematic", "nominal"
                )
                nom_arr_all = _values_without_flow(
                    nominal_projection, include_overflow=True
                )
                sqrt_sum_p = np.sqrt(
                    shape_systs_summed_arr_p + rate_systs_summed_arr_p
                )
                sqrt_sum_m = np.sqrt(
                    shape_systs_summed_arr_m + rate_systs_summed_arr_m
                )
                err_p_syst = nom_arr_all + sqrt_sum_p
                err_m_syst = nom_arr_all - sqrt_sum_m
                with np.errstate(divide="ignore", invalid="ignore"):
                    err_ratio_p_syst = np.where(
                        nom_arr_all > 0, err_p_syst / nom_arr_all, 1
                    )
                    err_ratio_m_syst = np.where(
                        nom_arr_all > 0, err_m_syst / nom_arr_all, 1
                    )
                syst_err = True

        if not _hist_has_content(hist_mc_integrated):
            print("Warning: empty mc histo, continuing")
            return 0, 0, html_dirs
        if unblind_flag and not _hist_has_content(hist_data_integrated):
            print("Warning: empty data histo, continuing")
            return 0, 0, html_dirs

        hist_data_to_plot = (
            hist_data_integrated
            if (unblind_flag or not region_ctx.use_mc_as_data_when_blinded)
            else hist_mc_integrated
        )
        if region_ctx.years:
            year_str = "_".join(region_ctx.years)
        else:
            year_str = "ULall"
        title = f"{hist_cat}_{var_name}_{year_str}"
        bins_override = region_ctx.analysis_bins.get(var_name)
        default_bins = (
            axes_info[var_name]["variable"] if var_name in axes_info else None
        )
        stacked_kwargs = {
            "group": region_ctx.group_map,
            "lumitag": region_ctx.lumi_pair[0] if region_ctx.lumi_pair else None,
            "comtag": region_ctx.lumi_pair[1] if region_ctx.lumi_pair else None,
            "h_mc_sumw2": hist_mc_sumw2,
            "syst_err": syst_err,
            "err_p_syst": err_p_syst,
            "err_m_syst": err_m_syst,
            "err_ratio_p_syst": err_ratio_p_syst,
            "err_ratio_m_syst": err_ratio_m_syst,
            "unblind": unblind_flag,
            "log_scale": stacked_log_y,
            "style": region_ctx.stacked_ratio_style,
        }
        bins_to_use = bins_override if bins_override is not None else default_bins
        if bins_to_use is not None:
            stacked_kwargs["bins"] = bins_to_use
        fig = make_region_stacked_ratio_fig(
            hist_mc_integrated,
            hist_data_to_plot,
            var=var_name,
            unit_norm_bool=False,
            **stacked_kwargs,
        )
        fig.savefig(
            os.path.join(save_dir_path_tmp, title),
            bbox_inches="tight",
            pad_inches=0.05,
        )
        _close_figure_payload(fig)
        has_syst_inputs = any(
            err is not None
            for err in (
                err_p_syst,
                err_m_syst,
                err_ratio_p_syst,
                err_ratio_m_syst,
            )
        )
        if has_syst_inputs:
            stat_and_syst_plots += 1
        else:
            stat_only_plots += 1
    else:
        raise ValueError(
            f"Unsupported channel_mode '{region_ctx.channel_mode}'"
        )

    if "www" in save_dir_path_tmp:
        html_dirs.add(save_dir_path_tmp)

    return stat_only_plots, stat_and_syst_plots, html_dirs
def _resolve_requested_variables(dict_of_hists, variables, context):
    """Return the ordered list of variables to process for a plotting function."""

    all_variables = list(dict_of_hists.keys())
    if not variables:
        return all_variables

    resolved = []
    missing = []
    for var_name in variables:
        if var_name in dict_of_hists:
            if var_name not in resolved:
                resolved.append(var_name)
        else:
            missing.append(var_name)

    for missing_name in missing:
        print(
            f"Warning: Requested variable '{missing_name}' not found in {context}; skipping."
        )

    return resolved


def validate_channel_group(histos, expected_labels, transformations, region, subgroup, variable):
    if not isinstance(histos, (list, tuple)):
        histos = [histos]

    available_channels = set()
    for histo in histos:
        if not isinstance(histo, (HistEFT, SparseHist)):
            continue
        if "channel" not in yt.get_axis_list(histo):
            continue
        available_channels.update(list(histo.axes["channel"]))

    if not available_channels:
        return

    expected_set = set(expected_labels)
    expected_transformed = {
        _apply_channel_transforms(label, transformations) for label in expected_set
    }

    region_known_channels = {
        "CR": CR_KNOWN_CHANNELS,
        "SR": SR_KNOWN_CHANNELS,
    }.get(region)
    transformed_known_channels = None
    if region_known_channels is not None:
        transformed_known_channels = {
            _apply_channel_transforms(label, transformations)
            for label in region_known_channels
        }

    stray_channels = set()

    for channel in available_channels:
        transformed = _apply_channel_transforms(channel, transformations)
        if transformed in expected_transformed:
            continue
        if region_known_channels is not None and channel in region_known_channels:
            continue
        if transformed_known_channels is not None and transformed in transformed_known_channels:
            continue
        stray_channels.add(channel)

    if stray_channels:
        var_str = f" for variable '{variable}'" if variable is not None else ""
        region_str = f"{region} " if region else ""
        raise ValueError(
            f"Found channel bins {sorted(stray_channels)} in {region_str}subgroup '{subgroup}'{var_str} that are not defined in the YAML configuration."
        )

def populate_group_map(samples, pattern_map):
    out = OrderedDict((k, []) for k in pattern_map)
    fallback_groups = OrderedDict()

    for proc_name in samples:
        matched = False
        for grp, patterns in pattern_map.items():
            for pat in patterns:
                if pat in proc_name:
                    out[grp].append(proc_name)
                    matched = True
                    break
            if matched:
                break
        if not matched:
            if proc_name not in fallback_groups:
                logger.warning(
                    "Process name '%s' does not match any configured group pattern; "
                    "assigning it to fallback group '%s'.",
                    proc_name,
                    proc_name,
                )
                fallback_groups[proc_name] = []
            fallback_groups[proc_name].append(proc_name)

    out.update(fallback_groups)
    return out


def _safe_divide(num, denom, default, zero_over_zero=None):
    """Safely divide two arrays while handling division by zero."""

    num_arr = np.asarray(num, dtype=float)
    denom_arr = np.asarray(denom, dtype=float)
    out = np.full_like(num_arr, default, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        valid = denom_arr != 0
        np.divide(num_arr, denom_arr, out=out, where=valid)
    if zero_over_zero is not None:
        zero_zero_mask = (denom_arr == 0) & (num_arr == 0)
        out[zero_zero_mask] = zero_over_zero
    return out


def _normalize_histograms(
    h_mc,
    h_data,
    unit_norm_bool,
    err_p,
    err_m,
    err_ratio_p,
    err_ratio_m,
    err_p_syst,
    err_m_syst,
    err_ratio_p_syst,
    err_ratio_m_syst,
    variable_name,
):
    """Scale MC and data histograms (and associated uncertainties) for unit-normalised plots."""

    if err_p_syst is None and err_p is not None:
        err_p_syst = np.asarray(err_p, dtype=float)
    if err_m_syst is None and err_m is not None:
        err_m_syst = np.asarray(err_m, dtype=float)
    if err_ratio_p_syst is None and err_ratio_p is not None:
        err_ratio_p_syst = np.asarray(err_ratio_p, dtype=float)
    if err_ratio_m_syst is None and err_ratio_m is not None:
        err_ratio_m_syst = np.asarray(err_ratio_m, dtype=float)

    mc_norm_factor = 1.0
    mc_scaled = False

    if unit_norm_bool:
        mc_eval = h_mc.eval({})
        data_eval = h_data.eval({})

        sum_mc = 0.0
        for values in mc_eval.values():
            sum_mc += float(np.sum(np.asarray(values, dtype=float)))

        sum_data = 0.0
        for values in data_eval.values():
            sum_data += float(np.sum(np.asarray(values, dtype=float)))

        if not np.isfinite(sum_mc) or np.isclose(sum_mc, 0.0, atol=1e-12, rtol=1e-6):
            logger.warning(
                "Skipping MC unit normalization for variable '%s' because the total MC yield is zero.",
                variable_name,
            )
        else:
            mc_norm_factor = 1.0 / sum_mc
            h_mc.scale(mc_norm_factor)
            mc_scaled = True

        if not np.isfinite(sum_data) or np.isclose(sum_data, 0.0, atol=1e-12, rtol=1e-6):
            logger.warning(
                "Skipping data unit normalization for variable '%s' because the total data yield is zero.",
                variable_name,
            )
        else:
            h_data.scale(1.0 / sum_data)

        if mc_scaled:
            if err_p is not None:
                err_p = np.asarray(err_p, dtype=float) * mc_norm_factor
            if err_m is not None:
                err_m = np.asarray(err_m, dtype=float) * mc_norm_factor
            if err_p_syst is not None:
                err_p_syst = np.asarray(err_p_syst, dtype=float) * mc_norm_factor
            if err_m_syst is not None:
                err_m_syst = np.asarray(err_m_syst, dtype=float) * mc_norm_factor

    return {
        "err_p": err_p,
        "err_m": err_m,
        "err_ratio_p": err_ratio_p,
        "err_ratio_m": err_ratio_m,
        "err_p_syst": err_p_syst,
        "err_m_syst": err_m_syst,
        "err_ratio_p_syst": err_ratio_p_syst,
        "err_ratio_m_syst": err_ratio_m_syst,
        "mc_norm_factor": mc_norm_factor,
        "mc_scaled": mc_scaled,
    }


def _prepare_log_scaled_stacks(
    plot_arrays,
    stacked_arrays,
    var,
    log_scale_requested,
):
    """Adjust stacked MC arrays to support log scaling while preserving warnings and fallbacks."""

    log_axis_enabled = False
    log_y_baseline = None
    adjusted_mc_totals = None

    stacked_matrix = np.asarray(plot_arrays, dtype=float)
    if stacked_matrix.ndim == 1:
        if plot_arrays:
            stacked_matrix = stacked_matrix[np.newaxis, :]
        else:
            stacked_matrix = stacked_matrix.reshape(0, 0)

    if stacked_matrix.size:
        totals_for_plot = np.sum(stacked_matrix, axis=0)
    else:
        totals_for_plot = (
            np.zeros_like(stacked_arrays[0], dtype=float)
            if stacked_arrays
            else np.zeros(0, dtype=float)
        )

    positive_totals = totals_for_plot[totals_for_plot > 0]
    epsilon = max(np.min(positive_totals) * 0.01, 1e-6) if positive_totals.size else 1e-6
    nonpositive_mask = totals_for_plot <= 0
    if np.any(nonpositive_mask) and stacked_matrix.size:
        warnings.warn(
            "Stacked MC totals for '%s' contain non-positive bins; "
            "lifting them slightly to enable log scaling." % var,
            RuntimeWarning,
        )
        divisor = max(stacked_matrix.shape[0], 1)
        stacked_matrix[:, nonpositive_mask] = np.where(
            stacked_matrix[:, nonpositive_mask] > 0,
            stacked_matrix[:, nonpositive_mask],
            epsilon / divisor,
        )
        totals_for_plot = np.sum(stacked_matrix, axis=0)
    positive_totals = totals_for_plot[totals_for_plot > 0]
    if positive_totals.size:
        epsilon = max(np.min(positive_totals) * 0.01, epsilon)
    if positive_totals.size == 0:
        logger.warning(
            "Unable to apply log scaling to '%s' stacked panel: no positive MC totals remain after adjustment.",
            var,
        )
        log_scale_requested = False
        plot_arrays = [arr.copy() for arr in stacked_arrays]
    else:
        divisor = max(stacked_matrix.shape[0], 1)
        per_group_floor = epsilon / divisor
        for idx in range(stacked_matrix.shape[1]):
            column = stacked_matrix[:, idx]
            neg_mask = column <= 0
            if not np.any(neg_mask):
                continue
            pos_mask = column > 0
            if not np.any(pos_mask):
                logger.warning(
                    "Unable to apply log scaling to '%s' stacked panel: bin %d has no positive MC contributions after adjustment.",
                    var,
                    idx,
                )
                log_scale_requested = False
                break
            lifted_negatives = np.full(np.count_nonzero(neg_mask), per_group_floor)
            difference = np.sum(lifted_negatives - column[neg_mask])
            positive_sum = np.sum(column[pos_mask])
            if positive_sum <= difference:
                logger.warning(
                    "Unable to apply log scaling to '%s' stacked panel: insufficient positive yield to offset negative contributions in bin %d.",
                    var,
                    idx,
                )
                log_scale_requested = False
                break
            scale = (positive_sum - difference) / positive_sum
            adjusted_column = column.copy()
            adjusted_column[neg_mask] = per_group_floor
            adjusted_column[pos_mask] = column[pos_mask] * scale
            if np.any(adjusted_column[pos_mask] <= 0):
                logger.warning(
                    "Unable to apply log scaling to '%s' stacked panel: rescaled positive contributions became non-positivein bin %d.",
                    var,
                    idx,
                )
                log_scale_requested = False
                break
            stacked_matrix[:, idx] = adjusted_column
        if log_scale_requested:
            plot_arrays = [stacked_matrix[i, :] for i in range(stacked_matrix.shape[0])]
            totals_after_adjustment = np.sum(stacked_matrix, axis=0)
            positive_totals_after = totals_after_adjustment[totals_after_adjustment > 0]
            if positive_totals_after.size == 0:
                logger.warning(
                    "Unable to apply log scaling to '%s' stacked panel: adjustments removed all positive totals.",
                    var,
                )
                log_scale_requested = False
                plot_arrays = [arr.copy() for arr in stacked_arrays]
            else:
                min_positive = np.min(positive_totals_after)
                log_y_baseline = max(min_positive * 0.5, 1e-6)
                adjusted_mc_totals = totals_after_adjustment
                log_axis_enabled = True
    if not log_scale_requested:
        plot_arrays = [arr.copy() for arr in stacked_arrays]

    return (
        plot_arrays,
        log_scale_requested,
        log_axis_enabled,
        log_y_baseline,
        adjusted_mc_totals,
    )


def _draw_stacked_panel(
    h_mc,
    h_data,
    grouping,
    colors,
    axis,
    var,
    bins,
    unit_norm_bool,
    lumitag,
    comtag,
    h_mc_sumw2,
    mc_scaled,
    mc_norm_factor,
    *,
    log_scale=False,
    style=None,
):
    """Render the stacked MC panel and ratio subplot, returning figure objects and MC summaries."""

    style = {} if style is None else style
    figure_style = _style_get(style, ("figure",), {})
    figsize = tuple(figure_style.get("figsize", (10, 8)))
    height_ratios = tuple(figure_style.get("height_ratios", (4, 1)))
    if len(height_ratios) != 2:
        height_ratios = (4, 1)
    hep.style.use("CMS")
    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios},
        sharex=True,
    )
    fig.subplots_adjust(hspace=figure_style.get("hspace", 0.07))

    plt.sca(ax)
    cms_style = _style_get(style, ("cms",), {})
    cms_fontsize = cms_style.get("fontsize", 18.0)
    cms_label = hep.cms.label(lumi=lumitag, com=comtag, fontsize=cms_fontsize)

    summed_mc = h_mc[{"process": sum}]
    summed_data = h_data[{"process": sum}]

    summed_mc_edges = None
    if hasattr(summed_mc, "axes"):
        try:
            summed_mc_edges = summed_mc.axes[var].edges
        except KeyError:
            summed_mc_edges = None

    summed_data_edges = None
    if hasattr(summed_data, "axes"):
        try:
            summed_data_edges = summed_data.axes[var].edges
        except KeyError:
            summed_data_edges = None

    if summed_mc_edges is None:
        summed_mc_edges = summed_data_edges
    if summed_data_edges is None:
        summed_data_edges = summed_mc_edges

    default_bins = summed_data_edges if bins is None else bins
    if default_bins is None:
        raise ValueError("Histogram axis has fewer than two edges; cannot determine binning.")
    bins = np.asarray(default_bins, dtype=float)
    n_bins = max(bins.size - 1, 0)

    axis_traits = None
    axis_obj = None
    for candidate in (summed_mc, summed_data, h_mc, h_data):
        axes = getattr(candidate, "axes", None)
        if axes is None:
            continue
        try:
            axis_obj = axes[var]
        except (KeyError, TypeError):
            continue
        else:
            break

    if axis_obj is not None:
        axis_traits = getattr(axis_obj, "traits", None)

    axis_has_underflow = (
        bool(getattr(axis_traits, "underflow", False)) if axis_traits is not None else None
    )
    axis_has_overflow = (
        bool(getattr(axis_traits, "overflow", False)) if axis_traits is not None else None
    )
    axis_nominal_bins = len(axis_obj) if axis_obj is not None else None
    includes_overflow_hint = (
        None
        if axis_nominal_bins is None
        else bool(n_bins > axis_nominal_bins)
    )

    def _visible_from_flow(
        flow_array,
        n_bins,
        *,
        has_underflow=None,
        has_overflow=None,
        include_overflow_hint=None,
    ):
        flow_values = np.asarray(flow_array, dtype=float)
        if flow_values.ndim == 0:
            return flow_values

        size = flow_values.size
        if size == n_bins:
            return flow_values

        if n_bins <= 0:
            return np.zeros(max(n_bins, 0), dtype=flow_values.dtype)

        drop_front = 0
        drop_back = 0
        target = n_bins

        if has_underflow is True and size > target:
            drop_front = 1

        keep_overflow = include_overflow_hint
        if has_overflow is True:
            if keep_overflow is False:
                if size - drop_front > target:
                    drop_back = 1
            elif keep_overflow is None and size - drop_front > target:
                drop_back = 1
        elif has_overflow is False:
            drop_back = 0

        remaining = size - drop_front - drop_back - target
        if remaining > 0:
            if keep_overflow is True or (keep_overflow is None and has_underflow in (True, None)):
                extra_front = min(remaining, size - drop_front - drop_back)
                drop_front += extra_front
                remaining -= extra_front
        if remaining > 0:
            drop_back += remaining

        start = min(drop_front, size)
        end = size - min(drop_back, max(size - start, 0))
        visible = flow_values[start:end]

        if visible.size > n_bins:
            trim = visible.size - n_bins
            if keep_overflow is True:
                visible = visible[trim:]
            else:
                visible = visible[:n_bins]
        elif visible.size < n_bins and n_bins > 0:
            padded = np.zeros(n_bins, dtype=flow_values.dtype)
            padded[: visible.size] = visible
            visible = padded

        return visible

    summed_mc_values_flow = _values_with_flow_or_overflow(summed_mc)
    summed_data_values_flow = _values_with_flow_or_overflow(summed_data)
    summed_mc_values = _visible_from_flow(
        summed_mc_values_flow,
        n_bins,
        has_underflow=axis_has_underflow,
        has_overflow=axis_has_overflow,
        include_overflow_hint=includes_overflow_hint,
    )
    summed_data_values = _visible_from_flow(
        summed_data_values_flow,
        n_bins,
        has_underflow=axis_has_underflow,
        has_overflow=axis_has_overflow,
        include_overflow_hint=includes_overflow_hint,
    )

    def _get_grouped_vals(hist_obj, grouping_map):
        grouped_values = {}
        for proc_name, members in grouping_map.items():
            grouped_hist = hist_obj[{"process": members}][{"process": sum}]
            flow_vals = _values_with_flow_or_overflow(grouped_hist)
            grouped_values[proc_name] = _visible_from_flow(
                flow_vals,
                n_bins,
                has_underflow=axis_has_underflow,
                has_overflow=axis_has_overflow,
                include_overflow_hint=includes_overflow_hint,
            )
        return grouped_values

    mc_vals = _get_grouped_vals(h_mc, grouping)
    stacked_arrays = [np.asarray(values, dtype=float) for values in mc_vals.values()]
    plot_arrays = [arr.copy() for arr in stacked_arrays]
    mc_sumw2_vals = {}
    if h_mc_sumw2 is not None:
        try:
            available_processes = set(h_mc_sumw2.axes[axis])
        except KeyError:
            available_processes = set()
        template = next(iter(mc_vals.values())) if mc_vals else summed_mc_values
        for proc_name, members in grouping.items():
            valid_members = [m for m in members if m in available_processes]
            missing_members = [m for m in members if m not in available_processes]

            grouped_vals = np.zeros_like(template)
            if valid_members:
                grouped_hist = h_mc_sumw2[{"process": valid_members}][{"process": sum}]
                flow_vals = _values_with_flow_or_overflow(grouped_hist)
                grouped_vals = _visible_from_flow(
                    flow_vals,
                    n_bins,
                    has_underflow=axis_has_underflow,
                    has_overflow=axis_has_overflow,
                    include_overflow_hint=includes_overflow_hint,
                )
                if unit_norm_bool and mc_scaled:
                    grouped_vals = grouped_vals * mc_norm_factor**2

            fallback_vals = np.zeros_like(template)
            if missing_members:
                fallback_hist = h_mc[{"process": missing_members}][{"process": sum}]
                fallback_flow = _values_with_flow_or_overflow(fallback_hist)
                fallback_vals = _visible_from_flow(
                    fallback_flow,
                    n_bins,
                    has_underflow=axis_has_underflow,
                    has_overflow=axis_has_overflow,
                    include_overflow_hint=includes_overflow_hint,
                )
                if unit_norm_bool and mc_scaled:
                    fallback_vals = fallback_vals * mc_norm_factor

            mc_sumw2_vals[proc_name] = grouped_vals + fallback_vals

    log_scale_requested = bool(log_scale)
    log_y_baseline = None
    adjusted_mc_totals = None
    log_axis_enabled = False
    if log_scale_requested and plot_arrays:
        (
            plot_arrays,
            log_scale_requested,
            log_axis_enabled,
            log_y_baseline,
            adjusted_mc_totals,
        ) = _prepare_log_scaled_stacks(
            plot_arrays,
            stacked_arrays,
            var,
            log_scale_requested,
        )
    elif log_scale_requested and not plot_arrays:
        logger.warning(
            "Requested log scaling for '%s' but no MC groups were available; falling back to linear scale.",
            var,
        )
        log_scale_requested = False

    if log_scale_requested and plot_arrays:
        log_axis_enabled = True
        if adjusted_mc_totals is None:
            adjusted_mc_totals = np.sum(plot_arrays, axis=0)

    if log_axis_enabled:
        ax.set_yscale("log", nonpositive="clip")

    hep.histplot(
        plot_arrays if plot_arrays else list(mc_vals.values()),
        ax=ax,
        bins=bins,
        stack=True,
        density=unit_norm_bool,
        label=list(mc_vals.keys()),
        histtype="fill",
        color=colors,
    )
    if log_y_baseline is not None:
        ax.set_ylim(bottom=log_y_baseline)

    hep.histplot(
        summed_data_values,
        ax=ax,
        bins=bins,
        stack=False,
        density=unit_norm_bool,
        label="Data",
        histtype="errorbar",
        **DATA_ERR_OPS,
    )

    data_vals = summed_data_values
    mc_vals_total = summed_mc_values

    ratio_vals = _safe_divide(
        data_vals,
        mc_vals_total,
        default=np.nan,
        zero_over_zero=1.0,
    )
    ratio_yerr = _safe_divide(
        np.sqrt(data_vals),
        mc_vals_total,
        default=0.0,
    )
    ratio_yerr[mc_vals_total == 0] = np.nan

    mc_nonpositive_mask = mc_vals_total <= 0
    zero_over_zero_mask = (mc_vals_total == 0) & (data_vals == 0)
    mask_for_nan = mc_nonpositive_mask & ~zero_over_zero_mask
    if np.any(mask_for_nan):
        ratio_vals = ratio_vals.astype(float, copy=True)
        ratio_yerr = ratio_yerr.astype(float, copy=True)
        ratio_vals[mask_for_nan] = np.nan
        ratio_yerr[mask_for_nan] = np.nan

    hep.histplot(
        ratio_vals,
        yerr=ratio_yerr,
        ax=rax,
        bins=bins,
        stack=False,
        density=unit_norm_bool,
        histtype="errorbar",
        **DATA_ERR_OPS,
    )

    mc_totals = mc_vals_total

    return {
        "fig": fig,
        "ax": ax,
        "rax": rax,
        "bins": bins,
        "cms_label": cms_label,
        "mc_sumw2_vals": mc_sumw2_vals,
        "mc_totals": mc_totals,
        "adjusted_mc_totals": adjusted_mc_totals,
        "log_axis_enabled": log_axis_enabled,
        "log_y_baseline": log_y_baseline,
        "ratio_values": ratio_vals,
        "ratio_errors": ratio_yerr,
    }


def _compute_uncertainty_bands(
    ax,
    rax,
    bins,
    mc_totals,
    mc_sumw2_vals,
    h_mc_sumw2,
    unit_norm_bool,
    mc_scaled,
    mc_norm_factor,
    err_p_syst,
    err_m_syst,
    err_ratio_p_syst,
    err_ratio_m_syst,
    syst_err,
    *,
    display_mc_totals=None,
    log_axis_enabled=False,
    log_y_baseline=None,
    style=None,
):
    """Compute and draw statistical/systematic uncertainty bands for the stacked plot."""

    style = {} if style is None else style

    if mc_totals.size == 0:
        return {"main_band_handles": []}

    if h_mc_sumw2 is not None:
        if mc_sumw2_vals:
            summed_mc_sumw2 = np.sum(list(mc_sumw2_vals.values()), axis=0)
        else:
            summed_mc_sumw2_flow = (
                h_mc_sumw2[{"process": sum}].as_hist({}).values(flow=True)
            )
            summed_mc_sumw2 = np.asarray(summed_mc_sumw2_flow, dtype=float)[1:]
            if summed_mc_sumw2.size > mc_totals.size:
                summed_mc_sumw2 = summed_mc_sumw2[: mc_totals.size]
            elif summed_mc_sumw2.size < mc_totals.size:
                padded = np.zeros_like(mc_totals, dtype=float)
                padded[: summed_mc_sumw2.size] = summed_mc_sumw2
                summed_mc_sumw2 = padded
            if unit_norm_bool and mc_scaled:
                summed_mc_sumw2 = summed_mc_sumw2 * mc_norm_factor**2
    else:
        if unit_norm_bool and mc_scaled:
            summed_mc_sumw2 = mc_totals * mc_norm_factor
        else:
            summed_mc_sumw2 = mc_totals

    mc_stat_unc = np.sqrt(np.clip(summed_mc_sumw2, a_min=0, a_max=None))

    has_syst_arrays = all(
        arr is not None
        for arr in (err_p_syst, err_m_syst, err_ratio_p_syst, err_ratio_m_syst)
    )

    valid_modes = {"stat", "syst", "total"}
    if isinstance(syst_err, str) and syst_err.lower() in valid_modes:
        band_mode = syst_err.lower()
    elif isinstance(syst_err, bool):
        if syst_err and has_syst_arrays:
            band_mode = "total"
        else:
            band_mode = "stat"
    else:
        band_mode = "total" if has_syst_arrays else "stat"

    def _append_last(arr):
        if arr is None or len(arr) == 0:
            return arr
        return np.append(arr, arr[-1])

    mc_stat_up = mc_totals + mc_stat_unc
    mc_stat_down = np.clip(mc_totals - mc_stat_unc, a_min=0, a_max=None)
    stat_fraction = _safe_divide(mc_stat_unc, mc_totals, default=0.0)
    ratio_stat_up = 1 + stat_fraction
    ratio_stat_down = 1 - stat_fraction

    mc_stat_band_up = _append_last(mc_stat_up)
    mc_stat_band_down = _append_last(mc_stat_down)
    ratio_stat_band_up = _append_last(ratio_stat_up)
    ratio_stat_band_down = _append_last(ratio_stat_down)

    syst_up = syst_down = ratio_syst_up = ratio_syst_down = None
    mc_total_band_up = mc_total_band_down = None
    ratio_total_band_up = ratio_total_band_down = None
    if has_syst_arrays:
        syst_up = np.asarray(err_p_syst)
        syst_down = np.asarray(err_m_syst)
        ratio_syst_up = np.asarray(err_ratio_p_syst)
        ratio_syst_down = np.asarray(err_ratio_m_syst)

        def _trim_overflow(arr):
            if arr is None:
                return arr
            arr = np.asarray(arr)
            if arr.ndim == 0:
                return arr
            if arr.shape[0] == mc_totals.shape[0]:
                return arr
            if arr.shape[0] == mc_totals.shape[0] + 1:
                return arr[:-1]
            return arr

        syst_up = _trim_overflow(syst_up)
        syst_down = _trim_overflow(syst_down)
        ratio_syst_up = _trim_overflow(ratio_syst_up)
        ratio_syst_down = _trim_overflow(ratio_syst_down)

        syst_up_diff = np.clip(syst_up - mc_totals, a_min=0, a_max=None)
        syst_down_diff = np.clip(mc_totals - syst_down, a_min=0, a_max=None)

        total_unc_up = np.sqrt(mc_stat_unc**2 + syst_up_diff**2)
        total_unc_down = np.sqrt(mc_stat_unc**2 + syst_down_diff**2)

        mc_total_band_up = _append_last(mc_totals + total_unc_up)
        mc_total_band_down = _append_last(
            np.clip(mc_totals - total_unc_down, a_min=0, a_max=None)
        )

        total_up_fraction = _safe_divide(total_unc_up, mc_totals, default=0.0)
        total_down_fraction = _safe_divide(total_unc_down, mc_totals, default=0.0)
        ratio_total_up = 1 + total_up_fraction
        ratio_total_down = 1 - total_down_fraction
        ratio_total_band_up = _append_last(
            np.clip(ratio_total_up, a_min=0, a_max=None)
        )
        ratio_total_band_down = _append_last(
            np.clip(ratio_total_down, a_min=0, a_max=None)
        )

        ratio_syst_band_up = _append_last(ratio_syst_up)
        ratio_syst_band_down = _append_last(ratio_syst_down)
        mc_syst_band_up = _append_last(np.clip(syst_up, a_min=0, a_max=None))
        mc_syst_band_down = _append_last(np.clip(syst_down, a_min=0, a_max=None))
    else:
        ratio_syst_band_up = ratio_syst_band_down = None
        mc_syst_band_up = mc_syst_band_down = None

    stat_label = "Stat. unc."
    syst_label = "Syst. unc."
    total_label = "Stat. $\\oplus$ syst. unc."

    ratio_band_handles = []
    main_band_handles = []

    if display_mc_totals is None:
        display_mc_totals = mc_totals

    display_mc_totals_appended = _append_last(display_mc_totals)

    def _ensure_log_safe(arr):
        if arr is None or not log_axis_enabled:
            return arr
        baseline = log_y_baseline if log_y_baseline is not None else 1e-6
        safe = np.asarray(arr, dtype=float)
        safe = np.clip(safe, a_min=baseline, a_max=None)
        reference = np.clip(display_mc_totals_appended, a_min=baseline, a_max=None)
        return np.maximum(safe, reference)

    if band_mode == "syst" and has_syst_arrays:
        if mc_syst_band_up is not None and mc_syst_band_down is not None:
            ax.fill_between(
                bins,
                _ensure_log_safe(mc_syst_band_down),
                _ensure_log_safe(mc_syst_band_up),
                step="post",
                facecolor="none",
                edgecolor="gray",
                label=syst_label,
                hatch="////",
            )
        if ratio_syst_band_up is not None and ratio_syst_band_down is not None:
            ratio_syst_handle = rax.fill_between(
                bins,
                ratio_syst_band_down,
                ratio_syst_band_up,
                step="post",
                facecolor="none",
                edgecolor="gray",
                label=syst_label,
                hatch="////",
            )
            ratio_band_handles.append(ratio_syst_handle)
    else:
        if mc_stat_band_up is not None and mc_stat_band_down is not None:
            stat_handle_main = ax.fill_between(
                bins,
                _ensure_log_safe(mc_stat_band_down),
                _ensure_log_safe(mc_stat_band_up),
                step="post",
                facecolor="gray",
                alpha=0.3,
                edgecolor="none",
                label="_nolegend_",
            )
            main_band_handles.append((stat_handle_main, stat_label))
        if ratio_stat_band_up is not None and ratio_stat_band_down is not None:
            ratio_stat_handle = rax.fill_between(
                bins,
                ratio_stat_band_down,
                ratio_stat_band_up,
                step="post",
                facecolor="gray",
                alpha=0.3,
                edgecolor="none",
                label=stat_label,
            )
            ratio_band_handles.append(ratio_stat_handle)

        show_total = band_mode == "total" and has_syst_arrays
        if show_total:
            if mc_total_band_up is not None and mc_total_band_down is not None:
                total_handle_main = ax.fill_between(
                    bins,
                    _ensure_log_safe(mc_total_band_down),
                    _ensure_log_safe(mc_total_band_up),
                    step="post",
                    facecolor="none",
                    edgecolor="gray",
                    label="_nolegend_",
                    hatch="////",
                )
                main_band_handles.append((total_handle_main, total_label))
            if ratio_total_band_up is not None and ratio_total_band_down is not None:
                ratio_total_handle = rax.fill_between(
                    bins,
                    ratio_total_band_down,
                    ratio_total_band_up,
                    step="post",
                    facecolor="none",
                    edgecolor="gray",
                    label=total_label,
                    hatch="////",
                )
                ratio_band_handles.append(ratio_total_handle)

    if ratio_band_handles:
        ratio_legend_style = _style_get(style, ("ratio_band_legend",), {})
        legend_kwargs = {
            "loc": ratio_legend_style.get("loc", "upper left"),
            "fontsize": ratio_legend_style.get("fontsize", 10),
            "frameon": ratio_legend_style.get("frameon", False),
            "ncol": ratio_legend_style.get("ncol", 2),
            "columnspacing": ratio_legend_style.get("columnspacing", 1.0),
        }
        handletextpad = ratio_legend_style.get("handletextpad")
        if handletextpad is not None:
            legend_kwargs["handletextpad"] = handletextpad
        bbox_to_anchor = ratio_legend_style.get("bbox_to_anchor")
        if bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = tuple(bbox_to_anchor)
        rax.legend(handles=ratio_band_handles, **legend_kwargs)

    return {
        "main_band_handles": main_band_handles,
        "ratio_stat_band_up": ratio_stat_band_up,
        "ratio_stat_band_down": ratio_stat_band_down,
        "ratio_syst_band_up": ratio_syst_band_up,
        "ratio_syst_band_down": ratio_syst_band_down,
        "ratio_total_band_up": ratio_total_band_up,
        "ratio_total_band_down": ratio_total_band_down,
    }



def _finalize_layout(
    fig,
    ax,
    rax,
    legend,
    cms_label,
    display_label,
    *,
    label_artist=None,
    events_artist=None,
    ratio_anchor=None,
    events_anchor=None,
    legend_anchor=None,
    legend_is_figure=False,
    style=None,
):
    """Align legends and axis annotations after all plotting calls."""

    legend_anchor_local = list(legend_anchor) if legend_anchor is not None else None
    style = {} if style is None else style
    legend_style = _style_get(style, ("legend",), {})
    cms_style = _style_get(style, ("cms",), {})
    axes_style = _style_get(style, ("axes",), {})
    legend_overlap_margin = legend_style.get("overlap_margin", 0.01)
    top_margin_min = legend_style.get("top_margin_min", 0.01)
    top_margin_scale = legend_style.get("top_margin_scale", 0.25)
    ratio_label_margin = axes_style.get("ratio_label_margin", 0.002)

    def _draw_and_get_renderer():
        fig.canvas.draw()
        return fig.canvas.get_renderer()

    renderer = _draw_and_get_renderer()

    legend_box = None
    if legend is not None:
        legend_bbox = legend.get_window_extent(renderer=renderer)
        legend_box = legend_bbox.transformed(fig.transFigure.inverted())

    cms_artists = ()
    if cms_label is not None:
        cms_artists = cms_label if isinstance(cms_label, (list, tuple)) else (cms_label,)

    cms_box = None
    cms_bboxes = []
    for artist in cms_artists:
        if hasattr(artist, "get_window_extent"):
            cms_bbox = artist.get_window_extent(renderer=renderer)
            cms_bboxes.append(cms_bbox)
    if cms_bboxes:
        cms_box = Bbox.union(cms_bboxes).transformed(fig.transFigure.inverted())

    if legend_box is not None and cms_box is not None:
        if legend_is_figure and legend_anchor_local is not None:
            buffer = max(top_margin_min, top_margin_scale * legend_box.height)
            required_headroom = legend_box.height + buffer
            desired_anchor_y = cms_box.y1 + buffer + legend_box.height
            if not np.isclose(desired_anchor_y, legend_anchor_local[1]):
                legend_anchor_local[1] = desired_anchor_y
                legend.set_bbox_to_anchor(tuple(legend_anchor_local), fig.transFigure)
                renderer = _draw_and_get_renderer()
                legend_bbox = legend.get_window_extent(renderer=renderer)
                legend_box = legend_bbox.transformed(fig.transFigure.inverted())

            subplot_params = fig.subplotpars
            available_top = max(0.0, 1.0 - required_headroom)
            if subplot_params.top > available_top:
                plt.subplots_adjust(
                    bottom=subplot_params.bottom,
                    top=available_top,
                    left=subplot_params.left,
                    right=subplot_params.right,
                    hspace=subplot_params.hspace,
                    wspace=subplot_params.wspace,
                )
                renderer = _draw_and_get_renderer()
                legend_bbox = legend.get_window_extent(renderer=renderer)
                legend_box = legend_bbox.transformed(fig.transFigure.inverted())
        else:
            horizontal_overlap = (
                legend_box.x0 < cms_box.x1 and legend_box.x1 > cms_box.x0
            )
            vertical_overlap = legend_box.y0 < cms_box.y1 and legend_box.y1 > cms_box.y0
            if horizontal_overlap and legend_anchor_local is not None:
                legend_width = legend_box.width
                space_right = 1.0 - legend_overlap_margin - cms_box.x1
                space_left = cms_box.x0 - legend_overlap_margin
                if space_right >= legend_width:
                    new_left = cms_box.x1 + legend_overlap_margin
                elif space_left >= legend_width:
                    new_left = max(
                        legend_overlap_margin,
                        cms_box.x0 - legend_overlap_margin - legend_width,
                    )
                else:
                    if space_right >= space_left:
                        new_left = min(
                            max(legend_overlap_margin, cms_box.x1 + legend_overlap_margin),
                            max(0.0, 1.0 - legend_width),
                        )
                    else:
                        new_left = max(
                            legend_overlap_margin,
                            min(
                                cms_box.x0 - legend_overlap_margin - legend_width,
                                max(0.0, 1.0 - legend_width),
                            ),
                        )
                new_left = np.clip(new_left, 0.0, max(0.0, 1.0 - legend_width))
                legend_anchor_local[0] = new_left + legend_width / 2.0
                legend.set_bbox_to_anchor(tuple(legend_anchor_local), fig.transFigure)
                renderer = _draw_and_get_renderer()
                legend_bbox = legend.get_window_extent(renderer=renderer)
                legend_box = legend_bbox.transformed(fig.transFigure.inverted())

            if vertical_overlap:
                shift = cms_box.y1 - legend_box.y0 + legend_overlap_margin
                if shift > 0:
                    ax_box = ax.get_position()
                    rax_box = rax.get_position()
                    ax.set_position([ax_box.x0, ax_box.y0 - shift, ax_box.width, ax_box.height])
                    rax.set_position([rax_box.x0, rax_box.y0 - shift, rax_box.width, rax_box.height])
                    renderer = _draw_and_get_renderer()
                    legend_bbox = legend.get_window_extent(renderer=renderer)
                    legend_box = legend_bbox.transformed(fig.transFigure.inverted())

    axis_bboxes = []
    for axis_obj in (ax, rax):
        try:
            bbox = axis_obj.get_tightbbox(renderer)
        except Exception:
            bbox = None
        if bbox is None:
            continue
        axis_bboxes.append(bbox.transformed(fig.transFigure.inverted()))
    if axis_bboxes:
        rightmost_extent = max(bbox.x1 for bbox in axis_bboxes)
    else:
        rightmost_extent = max(ax.get_position().x1, rax.get_position().x1)

    subplot_params = fig.subplotpars
    effective_right = min(np.nextafter(1.0, 0.0), rightmost_extent + 0.003)
    if not np.isclose(effective_right, subplot_params.right):
        stored_positions = [ax.get_position().frozen(), rax.get_position().frozen()]
        plt.subplots_adjust(
            bottom=subplot_params.bottom,
            top=subplot_params.top,
            left=subplot_params.left,
            right=effective_right,
            hspace=subplot_params.hspace,
            wspace=subplot_params.wspace,
        )
        renderer = _draw_and_get_renderer()
        for axis_obj, original in zip((ax, rax), stored_positions):
            updated = axis_obj.get_position()
            delta_y = original.y0 - updated.y0
            if not np.isclose(delta_y, 0.0):
                axis_obj.set_position(
                    [updated.x0, updated.y0 + delta_y, updated.width, updated.height]
                )
        renderer = _draw_and_get_renderer()

    def _ratio_axis_min_y(current_renderer):
        bboxes = []
        for tick_label in rax.get_xticklabels():
            if not tick_label.get_visible():
                continue
            text = tick_label.get_text()
            if not text:
                continue
            bbox = tick_label.get_window_extent(renderer=current_renderer)
            bboxes.append(bbox.transformed(fig.transFigure.inverted()))
        axis_label = rax.xaxis.label
        if axis_label and axis_label.get_visible():
            axis_bbox = axis_label.get_window_extent(renderer=current_renderer)
            bboxes.append(axis_bbox.transformed(fig.transFigure.inverted()))
        if bboxes:
            return min(b.y0 for b in bboxes)
        return rax.get_position().y0

    default_label_size = (
        rax.yaxis.label.get_size()
        if rax.yaxis.label
        else plt.rcParams.get("axes.labelsize", 18)
    )
    label_fontsize = axes_style.get("label_fontsize", default_label_size)
    renderer = _draw_and_get_renderer()
    temp = fig.text(0, 0, display_label, fontsize=label_fontsize)
    temp_bbox = temp.get_window_extent(renderer=renderer)
    temp.remove()
    measured_height = temp_bbox.transformed(fig.transFigure.inverted()).height
    label_y = _ratio_axis_min_y(renderer) - measured_height - ratio_label_margin

    subplot_params = fig.subplotpars
    new_bottom = np.clip(max(0.0, label_y - ratio_label_margin), 0.0, 1.0)
    if not np.isclose(new_bottom, subplot_params.bottom):
        plt.subplots_adjust(
            bottom=new_bottom,
            top=subplot_params.top,
            left=subplot_params.left,
            right=subplot_params.right,
            hspace=subplot_params.hspace,
            wspace=subplot_params.wspace,
        )
        renderer = _draw_and_get_renderer()
        label_y = _ratio_axis_min_y(renderer) - measured_height - ratio_label_margin

    renderer = _draw_and_get_renderer()
    ax_box = ax.get_position()
    rax_box = rax.get_position()

    ratio_label_fig = None
    ratio_label = rax.yaxis.label
    if ratio_label is not None:
        try:
            ratio_pos = np.asarray(ratio_label.get_position(), dtype=float)
            ratio_transform = ratio_label.get_transform()
            if ratio_transform is not None:
                ratio_display = ratio_transform.transform([ratio_pos])[0]
                ratio_label_fig = fig.transFigure.inverted().transform(ratio_display)
        except Exception:
            ratio_label_fig = None
    if ratio_label_fig is None and ratio_anchor is not None:
        ratio_label_fig = ratio_anchor

    events_x, events_y = events_anchor if events_anchor is not None else (None, None)
    if ratio_label_fig is not None:
        events_x = ratio_label_fig[0]
    if events_x is None:
        events_x = rax_box.x0 + rax_box.width
    current_events_y = ax_box.y0 + ax_box.height
    if current_events_y is not None:
        events_y = current_events_y
    elif events_y is None:
        events_y = rax_box.y0 + rax_box.height

    if events_artist is None or not isinstance(events_artist, mpl.text.Text):
        events_artist = fig.text(
            events_x,
            events_y,
            "Events",
            ha="right",
            va="bottom",
            fontsize=label_fontsize,
            rotation=90,
        )
    else:
        events_artist.set_position((events_x, events_y))
        events_artist.set_text("Events")
        events_artist.set_fontsize(label_fontsize)
        events_artist.set_rotation(90)
        events_artist.set_ha("right")
        events_artist.set_va("bottom")

    if label_artist is None or not isinstance(label_artist, mpl.text.Text):
        label_artist = fig.text(
            rax_box.x0 + rax_box.width,
            label_y,
            display_label,
            ha="right",
            va="bottom",
            fontsize=label_fontsize,
        )
    else:
        label_artist.set_position((rax_box.x0 + rax_box.width, label_y))
        label_artist.set_text(display_label)
        label_artist.set_fontsize(label_fontsize)
        label_artist.set_ha("right")
        label_artist.set_va("bottom")

    return label_artist, events_artist, legend_anchor_local


def _sample_in_signal_group(sample_name, sample_group_map, group_type):
    if group_type == "CR":
        return sample_name in sample_group_map.get("Signal", [])

    if group_type == "SR":
        for grp_key in SR_SIGNAL_GROUP_KEYS:
            if sample_name in sample_group_map.get(grp_key, []):
                return True

    return False


def _normalize_sequence(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _evaluate_channel_condition(condition, region_ctx):
    if condition == "not_split_by_lepflav":
        return not yt.is_split_by_lepflav(region_ctx.dict_of_hists)
    raise ValueError(
        f"Unsupported channel transformation condition '{condition}'"
    )


def _resolve_channel_transformations(region_ctx, var_name):
    rules = region_ctx.channel_rules
    transformations = []
    transformations.extend(rules.get("default", []))
    transformations.extend(rules.get("variables", {}).get(var_name, []))
    for cond_entry in rules.get("conditional", []):
        condition = cond_entry.get("when")
        if condition is None:
            continue
        if _evaluate_channel_condition(condition, region_ctx):
            transformations.extend(cond_entry.get("apply", []))
    ordered = []
    seen = set()
    for transform in transformations:
        if transform not in seen:
            ordered.append(transform)
            seen.add(transform)
    return ordered


def _apply_channel_dict_transformations(channel_dict, transformations):
    if not transformations:
        return dict(channel_dict)

    transformed_dict = copy.deepcopy(channel_dict)
    for transform in transformations:
        if transform == "njets":
            transformed_dict = get_dict_with_stripped_bin_names(
                transformed_dict, "njets"
            )
        elif transform == "lepflav":
            transformed_dict = get_dict_with_stripped_bin_names(
                transformed_dict, "lepflav"
            )
        else:
            raise ValueError(
                f"Unsupported channel transformation '{transform}'"
            )
    return transformed_dict


def _match_category(hist_cat, categories_cfg):
    if not categories_cfg:
        return True
    prefixes = _normalize_sequence(categories_cfg.get("prefixes"))
    if prefixes and any(hist_cat.startswith(pref) for pref in prefixes):
        return True
    equals = _normalize_sequence(categories_cfg.get("equals"))
    if equals and hist_cat in equals:
        return True
    contains = _normalize_sequence(categories_cfg.get("contains"))
    if contains and any(token in hist_cat for token in contains):
        return True
    return False


def _should_skip_category(rules, hist_cat, var_name):
    for rule in rules:
        if not _match_category(hist_cat, rule.get("categories")):
            continue
        includes = _normalize_sequence(rule.get("variable_includes"))
        if includes and not any(token in var_name for token in includes):
            continue
        excludes = _normalize_sequence(rule.get("variable_excludes"))
        if excludes and any(token in var_name for token in excludes):
            continue
        return True
    return False


def _collect_samples_to_remove(rules, hist_cat, region_ctx):
    samples = []
    for rule in rules:
        if not _match_category(hist_cat, rule.get("categories")):
            continue
        rule_groups = _normalize_sequence(rule.get("groups"))
        for group in rule_groups:
            samples.extend(region_ctx.group_map.get(group, []))
        samples.extend(_normalize_sequence(rule.get("samples")))
    ordered = []
    seen = set()
    for sample in samples:
        if sample not in seen:
            ordered.append(sample)
            seen.add(sample)
    return ordered


def _normalize_channel_rules(raw_rules):
    if raw_rules is None:
        return {"default": [], "variables": {}, "conditional": []}
    normalized = {
        "default": _normalize_sequence(raw_rules.get("default", [])),
        "variables": {
            key: _normalize_sequence(value)
            for key, value in raw_rules.get("variables", {}).items()
        },
        "conditional": [],
    }
    conditional_entries = []
    for entry in raw_rules.get("conditional", []):
        if not entry:
            continue
        when_key = entry.get("when")
        if when_key is None:
            continue
        conditional_entries.append(
            {"when": when_key, "apply": _normalize_sequence(entry.get("apply", []))}
        )
    normalized["conditional"] = conditional_entries
    return normalized


def _find_reference_hist_name(dict_of_hists):
    for hist_name in dict_of_hists:
        if not hist_name.endswith("_sumw2"):
            return hist_name
    raise ValueError("No histogram without '_sumw2' suffix was found.")


class RegionContext(object):
    def __init__(
        self,
        name,
        dict_of_hists,
        years,
        channel_map,
        group_patterns,
        group_map,
        all_samples,
        mc_samples,
        data_samples,
        samples_to_remove,
        sumw2_hists,
        signal_samples,
        unblind_default,
        lumi_pair,
        skip_variables=None,
        analysis_bins=None,
        stacked_ratio_style=None,
        channel_rules=None,
        sample_removal_rules=None,
        category_skip_rules=None,
        skip_sparse_2d=False,
        channel_mode="per-channel",
        variable_label="Variable",
        debug_channel_lists=False,
        sumw2_remove_signal=False,
        sumw2_remove_signal_when_blinded=False,
        use_mc_as_data_when_blinded=False,
    ):
        self.name = name
        self.dict_of_hists = dict_of_hists
        self.years = None if years is None else tuple(years)
        if self.years is None:
            self.year = None
        elif len(self.years) == 1:
            self.year = self.years[0]
        else:
            self.year = self.years
        self.channel_map = channel_map
        self.group_patterns = group_patterns
        self.group_map = group_map
        self.all_samples = all_samples
        self.mc_samples = mc_samples
        self.data_samples = data_samples
        self.samples_to_remove = samples_to_remove
        self.sumw2_hists = sumw2_hists
        self.signal_samples = signal_samples
        self.unblind_default = unblind_default
        self.lumi_pair = lumi_pair
        self.skip_variables = set() if skip_variables is None else set(skip_variables)
        self.analysis_bins = (
            {} if analysis_bins is None else copy.deepcopy(analysis_bins)
        )
        self.stacked_ratio_style = (
            copy.deepcopy(stacked_ratio_style)
            if isinstance(stacked_ratio_style, Mapping)
            else {}
        )
        default_channel_rules = {"default": [], "variables": {}, "conditional": []}
        self.channel_rules = copy.deepcopy(
            channel_rules if channel_rules is not None else default_channel_rules
        )
        self.sample_removal_rules = (
            copy.deepcopy(sample_removal_rules)
            if sample_removal_rules is not None
            else []
        )
        self.category_skip_rules = (
            copy.deepcopy(category_skip_rules)
            if category_skip_rules is not None
            else []
        )
        self.skip_sparse_2d = bool(skip_sparse_2d)
        self.channel_mode = channel_mode
        self.variable_label = variable_label
        self.debug_channel_lists = bool(debug_channel_lists)
        self.sumw2_remove_signal = bool(sumw2_remove_signal)
        self.sumw2_remove_signal_when_blinded = bool(
            sumw2_remove_signal_when_blinded
        )
        self.use_mc_as_data_when_blinded = bool(use_mc_as_data_when_blinded)


def _format_decimal_string(value):
    normalized = value.normalize()
    # Decimal.normalize() may produce scientific notation for integers; format
    # explicitly to keep plain strings such as "101.3".
    formatted = format(normalized, "f")
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _resolve_lumi_pair(year_tokens):
    if not year_tokens:
        return None

    lumi_components = []
    com_tags = set()
    missing_metadata = []

    for token in year_tokens:
        pair = LUMI_COM_PAIRS.get(token)
        if pair is None:
            missing_metadata.append(token)
            continue
        lumi_components.append(Decimal(pair[0]))
        com_tags.add(pair[1])

    if missing_metadata and not lumi_components:
        return None

    if missing_metadata:
        raise KeyError(
            "No luminosity metadata available for year token(s): "
            + ", ".join(sorted(set(missing_metadata)))
        )

    if len(com_tags) != 1:
        raise ValueError(
            "Inconsistent center-of-mass energies encountered while combining "
            "years {}.".format(
                ", ".join(year_tokens)
            )
        )

    combined_lumi = sum(lumi_components, Decimal("0"))
    return (_format_decimal_string(combined_lumi), com_tags.pop())


def build_region_context(region,dict_of_hists,years,unblind=None):
    region_upper = region.upper()
    if region_upper not in ["CR","SR"]:
        raise ValueError(f"Unsupported region '{region}'.")

    mc_wl = []
    mc_bl = ["data"]
    data_wl = ["data"]
    data_bl = []
    if years is None:
        year_tokens = []
    elif isinstance(years, str):
        year_tokens = [years]
    else:
        year_tokens = list(years)

    normalized_year_tokens = []
    seen_years = set()
    for token in year_tokens:
        if token is None:
            continue
        cleaned = str(token).strip()
        if not cleaned or cleaned in seen_years:
            continue
        if cleaned not in YEAR_TOKEN_RULES:
            raise ValueError(
                "Error: Unknown year token '{}' requested. Supported tokens: {}".format(
                    cleaned, ", ".join(sorted(YEAR_TOKEN_RULES))
                )
            )
        rules = YEAR_TOKEN_RULES[cleaned]

        mc_wl_values = rules.get("mc_wl", [])
        if mc_wl_values:
            for value in mc_wl_values:
                if value not in mc_wl:
                    mc_wl.append(value)
            mc_bl[:] = [value for value in mc_bl if value not in mc_wl_values]

        mc_bl_values = rules.get("mc_bl", [])
        if mc_bl_values:
            for value in mc_bl_values:
                if value in mc_wl or value in mc_bl:
                    continue
                mc_bl.append(value)

        data_wl_values = rules.get("data_wl", [])
        if data_wl_values:
            for value in data_wl_values:
                if value not in data_wl:
                    data_wl.append(value)
            data_bl[:] = [value for value in data_bl if value not in data_wl_values]

        data_bl_values = rules.get("data_bl", [])
        if data_bl_values:
            for value in data_bl_values:
                if value in data_wl or value in data_bl:
                    continue
                data_bl.append(value)
        seen_years.add(cleaned)
        normalized_year_tokens.append(cleaned)

    try:
        all_samples = yt.get_cat_lables(dict_of_hists, "process")
    except Exception:
        ref_hist = _find_reference_hist_name(dict_of_hists)
        all_samples = yt.get_cat_lables(dict_of_hists, "process", h_name=ref_hist)

    def _filter_samples(all_labels, whitelist, blacklist, *, allow_data_driven_reinsertion=False):
        """Return samples that satisfy blacklist rules and multi-token requirements."""

        if len(whitelist) <= 1 and not allow_data_driven_reinsertion:
            return utils.filter_lst_of_strs(
                all_labels, substr_whitelist=whitelist, substr_blacklist=blacklist
            )

        must_have_tokens = []
        optional_tokens = []
        for token in whitelist:
            if token is None:
                continue
            if token.lower() == "data" or token not in YEAR_WHITELIST_OPTIONALS:
                must_have_tokens.append(token)
            else:
                optional_tokens.append(token)

        # Remove duplicates while preserving ordering to keep predictable filtering.
        must_have_tokens = list(dict.fromkeys(must_have_tokens))
        optional_tokens = list(dict.fromkeys(optional_tokens))
        optional_token_set = set(optional_tokens)

        year_token_cache = {}

        def _present_year_tokens(label):
            cached = year_token_cache.get(label)
            if cached is not None:
                return cached

            detected_tokens = {
                year_token
                for year_token in YEAR_WHITELIST_OPTIONALS
                if year_token in label
            }
            if len(detected_tokens) <= 1:
                result = frozenset(detected_tokens)
                year_token_cache[label] = result
                return result

            resolved_tokens = set(detected_tokens)
            for token in list(detected_tokens):
                for other_token in detected_tokens:
                    if token == other_token:
                        continue
                    if token in other_token:
                        resolved_tokens.discard(token)
                        break

            result = frozenset(resolved_tokens)
            year_token_cache[label] = result
            return result

        def _label_contains_disallowed_year(present_tokens):
            if not optional_tokens or not present_tokens:
                return False
            return any(token not in optional_token_set for token in present_tokens)

        def _label_passes(label, *, require_optional_tokens):
            if any(token in label for token in blacklist):
                return False
            if must_have_tokens and any(token not in label for token in must_have_tokens):
                return False
            present_tokens = _present_year_tokens(label)
            if _label_contains_disallowed_year(present_tokens):
                return False
            if require_optional_tokens and optional_tokens:
                if not present_tokens.intersection(optional_token_set):
                    return False
            return True

        filtered = [
            label
            for label in all_labels
            if _label_passes(label, require_optional_tokens=True)
        ]

        if allow_data_driven_reinsertion and DATA_DRIVEN_MATCHERS:
            filtered_set = set(filtered)
            for label in all_labels:
                if label in filtered_set:
                    continue
                if not any(matcher.search(label) for matcher in DATA_DRIVEN_MATCHERS):
                    continue
                if not _label_passes(label, require_optional_tokens=False):
                    continue
                filtered.append(label)
                filtered_set.add(label)
        return filtered

    mc_samples = _filter_samples(
        all_samples,
        mc_wl,
        mc_bl,
        allow_data_driven_reinsertion=True,
    )
    data_samples = _filter_samples(
        all_samples,
        data_wl,
        data_bl,
        allow_data_driven_reinsertion=False,
    )
    samples_to_remove = {
        "mc": [sample for sample in all_samples if sample not in mc_samples],
        "data": [sample for sample in all_samples if sample not in data_samples],
    }

    sumw2_hists = {
        hist_name.replace("_sumw2", ""): hist_obj
        for hist_name, hist_obj in dict_of_hists.items()
        if hist_name.endswith("_sumw2") and hist_name.count("sumw2") == 1
    }

    if not sumw2_hists:
        logger.warning(
            "No sumw² histograms found in the input. Statistical uncertainties will default to Poisson counting errors."
        )

    try:
        lumi_pair = _resolve_lumi_pair(normalized_year_tokens)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc

    if unblind is None:
        resolved_unblind = region_upper == "CR"
    else:
        resolved_unblind = bool(unblind)

    region_plot_cfg = REGION_PLOTTING.get(region_upper, {})
    stacked_ratio_style = _resolve_stacked_ratio_style(
        region_upper, region_plot_cfg.get("stacked_ratio_style")
    )

    skip_variables = set(region_plot_cfg.get("skip_variables", []))
    analysis_bins = {}
    for var_name, spec in region_plot_cfg.get("analysis_bins", {}).items():
        if isinstance(spec, str):
            if spec not in axes_info:
                raise KeyError(
                    f"Analysis bin specification '{spec}' is not defined in axes_info."
                )
            analysis_bins[var_name] = axes_info[spec]["variable"]
        else:
            analysis_bins[var_name] = spec

    channel_rules = _normalize_channel_rules(
        region_plot_cfg.get("channel_transformations")
    )
    sample_removal_rules = region_plot_cfg.get("sample_removals", [])
    category_skip_rules = region_plot_cfg.get("category_skips", [])
    skip_sparse_2d = region_plot_cfg.get("skip_sparse_2d", False)
    channel_mode = region_plot_cfg.get("channel_mode", "per-channel")
    variable_label = region_plot_cfg.get("variable_label", "Variable")
    debug_channel_lists = region_plot_cfg.get("debug_channel_lists", False)
    sumw2_remove_signal = region_plot_cfg.get("sumw2_remove_signal", False)
    sumw2_remove_signal_when_blinded = region_plot_cfg.get(
        "sumw2_remove_signal_when_blinded", False
    )
    use_mc_as_data_when_blinded = region_plot_cfg.get(
        "use_mc_as_data_when_blinded", False
    )

    removed_mc_samples = set(samples_to_remove.get("mc", ()))
    removed_data_samples = set(samples_to_remove.get("data", ()))
    filtered_mc_samples = [
        sample for sample in mc_samples if sample not in removed_mc_samples
    ]
    filtered_data_samples = [
        sample for sample in data_samples if sample not in removed_data_samples
    ]
    filtered_group_samples = filtered_mc_samples + [
        sample
        for sample in filtered_data_samples
        if sample not in filtered_mc_samples
    ]

    if region_upper == "CR":
        group_patterns = CR_GRP_PATTERNS
        channel_map = CR_CHAN_DICT
        group_map = populate_group_map(filtered_group_samples, group_patterns)
        signal_samples = sorted(set(group_map.get("Signal", [])))
        unblind_default = resolved_unblind
        global CR_GRP_MAP
        CR_GRP_MAP = group_map
    else:
        group_patterns = SR_GRP_PATTERNS
        channel_map = SR_CHAN_DICT
        group_map = populate_group_map(mc_samples + data_samples, group_patterns)
        signal_samples = sorted(
            {
                proc_name
                for group_name in SR_SIGNAL_GROUP_KEYS
                for proc_name in group_map.get(group_name, [])
            }
        )
        unblind_default = resolved_unblind
        global SR_GRP_MAP
        SR_GRP_MAP = group_map

    return RegionContext(
        region_upper,
        dict_of_hists,
        normalized_year_tokens if normalized_year_tokens else None,
        channel_map,
        group_patterns,
        group_map,
        all_samples,
        mc_samples,
        data_samples,
        samples_to_remove,
        sumw2_hists,
        signal_samples,
        unblind_default,
        lumi_pair,
        skip_variables,
        analysis_bins,
        stacked_ratio_style=stacked_ratio_style,
        channel_rules=channel_rules,
        sample_removal_rules=sample_removal_rules,
        category_skip_rules=category_skip_rules,
        skip_sparse_2d=skip_sparse_2d,
        channel_mode=channel_mode,
        variable_label=variable_label,
        debug_channel_lists=debug_channel_lists,
        sumw2_remove_signal=sumw2_remove_signal,
        sumw2_remove_signal_when_blinded=sumw2_remove_signal_when_blinded,
        use_mc_as_data_when_blinded=use_mc_as_data_when_blinded,
    )



def produce_region_plots(
    region_ctx,
    save_dir_path,
    variables,
    skip_syst_errs,
    unit_norm_bool,
    stacked_log_y,
    unblind=None,
    *,
    workers=1,
    verbose=False,
):
    dict_of_hists = region_ctx.dict_of_hists
    context_label = f"{region_ctx.name} region"
    variables_to_plot = _resolve_requested_variables(
        dict_of_hists, variables, context_label
    )
    if verbose and variables is not None:
        print("Filtered variables:", variables_to_plot)

    if verbose:
        print("\n\nAll samples:", region_ctx.all_samples)
        print("\nMC samples:", region_ctx.mc_samples)
        print("\nData samples:", region_ctx.data_samples)
        print("\nVariables:", list(dict_of_hists.keys()))

    unblind_flag = region_ctx.unblind_default if unblind is None else bool(unblind)

    variable_payload_cache = {}
    variable_categories = {}
    eligible_variables = []
    category_dirs = set(region_ctx.channel_map.keys()) if save_dir_path else set()
    for var_name in variables_to_plot:
        if "sumw2" in var_name:
            continue
        if var_name in region_ctx.skip_variables:
            continue

        variable_metadata = _prepare_variable_payload(
            var_name,
            region_ctx,
            verbose=verbose,
            unblind_flag=unblind_flag,
            metadata_only=True,
        )
        if not variable_metadata:
            continue

        eligible_variables.append(var_name)

        categories = [
            hist_cat
            for hist_cat, channel_bins in variable_metadata["channel_dict"].items()
            if channel_bins is not None
            and not _should_skip_category(
                region_ctx.category_skip_rules, hist_cat, var_name
            )
        ]
        variable_categories[var_name] = categories
        if save_dir_path:
            category_dirs.update(categories)

    stat_only_plots = 0
    stat_and_syst_plots = 0
    html_dirs = set()

    worker_count = max(int(workers or 1), 1)
    tasks = list(eligible_variables)
    if not verbose:
        if tasks:
            print(
                "[{}] Rendering {} variable{}...".format(
                    region_ctx.name,
                    len(tasks),
                    "s" if len(tasks) != 1 else "",
                )
            )
        else:
            print(f"[{region_ctx.name}] No eligible variables to render.")
    if worker_count > 1 and eligible_variables:
        category_tasks = []
        for var_name in eligible_variables:
            categories = variable_categories.get(var_name, [])
            if categories:
                category_tasks.extend((var_name, hist_cat) for hist_cat in categories)
            else:
                category_tasks.append(var_name)
        if len(category_tasks) > len(tasks):
            tasks = category_tasks

    if save_dir_path:
        for hist_cat in sorted(category_dirs):
            os.makedirs(os.path.join(save_dir_path, hist_cat), exist_ok=True)

    total_tasks = len(tasks)
    task_specs = []
    for task_index, payload in enumerate(tasks, start=1):
        if isinstance(payload, tuple):
            var_name, hist_cat = payload
        else:
            var_name, hist_cat = payload, None
        label = f"{var_name} [{hist_cat}]" if hist_cat else var_name
        task_specs.append((task_index, payload, label, var_name, hist_cat))

    progress_total = total_tasks
    progress_done = 0
    progress_enabled = bool(verbose and progress_total)

    def _get_variable_payload(var_name):
        if var_name not in variable_payload_cache:
            variable_payload_cache[var_name] = _prepare_variable_payload(
                var_name,
                region_ctx,
                verbose=verbose,
                unblind_flag=unblind_flag,
            )
        return variable_payload_cache[var_name]

    def _report_progress(task_label):
        nonlocal progress_done
        if not progress_enabled:
            return
        progress_done += 1
        print(
            "[{}] [{}/{}] Completed {}".format(
                region_ctx.name,
                progress_done,
                progress_total,
                task_label,
            )
        )

    if worker_count > 1 and total_tasks > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        max_workers = min(worker_count, total_tasks)

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_initialize_render_worker,
            initargs=(
                region_ctx,
                save_dir_path,
                skip_syst_errs,
                unit_norm_bool,
                unblind_flag,
                stacked_log_y,
                verbose,
                None,
            ),
        ) as executor:
            id_to_label = {
                task_id: label for task_id, _, label, _, _ in task_specs
            }
            futures = [
                executor.submit(
                    _render_variable_from_worker,
                    task_id,
                    payload,
                )
                for task_id, payload, _, _, _ in task_specs
            ]
            for future in as_completed(futures):
                task_id, stat_only, stat_and_syst, html_set = future.result()
                stat_only_plots += stat_only
                stat_and_syst_plots += stat_and_syst
                html_dirs.update(html_set)
                _report_progress(id_to_label.get(task_id, str(task_id)))
    else:
        for _, _, label, var_name, hist_cat in task_specs:
            variable_payload = _get_variable_payload(var_name)
            if hist_cat is None:
                stat_only, stat_and_syst, html_set = _render_variable(
                    var_name,
                    region_ctx,
                    save_dir_path,
                    skip_syst_errs,
                    unit_norm_bool,
                    stacked_log_y,
                    unblind_flag,
                    verbose=verbose,
                    category=hist_cat,
                    variable_payload=variable_payload,
                )
            else:
                if not variable_payload:
                    stat_only, stat_and_syst, html_set = 0, 0, set()
                else:
                    channel_bins = variable_payload["channel_dict"].get(hist_cat)
                    if channel_bins is None or _should_skip_category(
                        region_ctx.category_skip_rules, hist_cat, var_name
                    ):
                        stat_only, stat_and_syst, html_set = 0, 0, set()
                    else:
                        stat_only, stat_and_syst, html_set = _render_variable_category(
                            var_name,
                            hist_cat,
                            channel_bins,
                            region_ctx=region_ctx,
                            channel_transformations=variable_payload["channel_transformations"],
                            hist_mc=variable_payload["hist_mc"],
                            hist_data=variable_payload["hist_data"],
                            hist_mc_sumw2_orig=variable_payload["hist_mc_sumw2_orig"],
                            is_sparse2d=variable_payload["is_sparse2d"],
                            save_dir_path=save_dir_path,
                            skip_syst_errs=skip_syst_errs,
                            unit_norm_bool=unit_norm_bool,
                            stacked_log_y=stacked_log_y,
                            unblind_flag=unblind_flag,
                            verbose=verbose,
                        )
            stat_only_plots += stat_only
            stat_and_syst_plots += stat_and_syst
            html_dirs.update(html_set)
            _report_progress(label)

    for html_dir in sorted(html_dirs):
        try:
            make_html(html_dir)
        except Exception as exc:
            print(f"Warning: Failed to refresh HTML in {html_dir}: {exc}")

    if progress_enabled and progress_done < progress_total:
        progress_done = progress_total

    if progress_total:
        summary_suffix = (
            f" after completing {progress_total} rendering task"
            f"{'s' if progress_total != 1 else ''}"
        )
    else:
        summary_suffix = "; no rendering tasks were executed"

    print(
        f"[{region_ctx.name}] Produced {stat_and_syst_plots} plots with stat⊕syst uncertainties and {stat_only_plots} plots with stat-only bands"
        f"{summary_suffix}",
        end="",
    )
    if save_dir_path:
        print(f" in {save_dir_path}")
    else:
        print()


def _ensure_list(values):
    if isinstance(values, str):
        return [values]
    return list(values)


# Group bins in a hist, returns a new hist
def group_bins(histo, bin_map, axis_name="process", drop_unspecified=False):

    normalized_map = OrderedDict(
        (group, _ensure_list(bins)) for group, bins in bin_map.items()
    )  # _ensure_list copies each sequence to avoid mutating caller data

    axis_categories = list(histo.axes[axis_name])
    axis_category_set = set(axis_categories)

    if not drop_unspecified:
        specified = {item for bins in normalized_map.values() for item in bins}
        for category in axis_categories:
            if category not in specified:
                normalized_map.setdefault(category, [category])

    requested = {item for bins in normalized_map.values() for item in bins}
    missing = sorted(requested - axis_category_set)
    if missing:
        raise ValueError(
            f"Requested {axis_name} bins not found in histogram: {', '.join(missing)}"
        )

    return histo.group(axis_name, normalized_map)


######### Functions for getting info from the systematics json #########

# Match a given sample name to whatever it is called in the json
# Will return None if a match is not found
def get_scale_name(sample_name,sample_group_map,group_type="CR"):
    scale_name_for_json = None
    if sample_name in sample_group_map.get("Conv", []):
        scale_name_for_json = "convs"
    elif sample_name in sample_group_map.get("Diboson", []):
        scale_name_for_json = "Diboson"
    elif sample_name in sample_group_map.get("Triboson", []):
        scale_name_for_json = "Triboson"
    elif _sample_in_signal_group(sample_name, sample_group_map, group_type):
        wc_matches = [proc_str for proc_str in SIGNAL_WC_MATCHES if proc_str in sample_name]
        if group_type == "CR":
            if len(wc_matches) == 1:
                scale_name_for_json = wc_matches[0]
        else:
            if wc_matches:
                # This should only match once, but maybe we should put a check to enforce this
                scale_name_for_json = wc_matches[0]
    return scale_name_for_json

# This function gets the tag that indicates how a particualr systematic is correlated
#   - For pdf_scale this corresponds to the initial state (e.g. gg)
#   - For qcd_scale this corresponds to the process type (e.g. VV)
# For any systemaitc or process that is not included in the correlations json we return None
def get_correlation_tag(uncertainty_name,proc_name,sample_group_map,group_type="CR"):
    proc_name_in_json = get_scale_name(proc_name,sample_group_map,group_type=group_type)
    corr_tag = None
    # Right now we only have two types of uncorrelated rate systematics
    if uncertainty_name in ["qcd_scale","pdf_scale"]:
        if proc_name_in_json is not None:
            if proc_name_in_json == "convs":
                # Special case for conversions since we estimate from LO sample, we do not have qcd uncty
                # Would be better to handle this in a more general way
                corr_tag = None
            else:
                corr_tag = grs.get_correlation_tag(uncertainty_name,proc_name_in_json)
    return corr_tag

# This function gets all of the the rate systematics from the json file
# Returns a dictionary with all of the uncertainties
# If the sample does not have an uncertainty in the json, an uncertainty of 0 is returned for that category
def get_rate_systs(sample_name,sample_group_map,group_type="CR"):

    # Figure out the name of the appropriate sample in the syst rate json (if the proc is in the json)
    scale_name_for_json = get_scale_name(sample_name,sample_group_map,group_type=group_type)

    # Get the lumi uncty for this sample (same for all samles)
    lumi_uncty = grs.get_syst("lumi")

    # Get the flip uncty from the json (if there is not an uncertainty for this sample, return 1 since the uncertainties are multiplicative)
    if sample_name in sample_group_map["Flips"]:
        flip_uncty = grs.get_syst("charge_flips","charge_flips_sm")
    else:
        flip_uncty = [1.0,1,0]

    # Get the scale uncty from the json (if there is not an uncertainty for this sample, return 1 since the uncertainties are multiplicative)
    if scale_name_for_json is not None:
        if scale_name_for_json in PROC_WITHOUT_PDF_RATE_SYST:
            # Special cases for when we do not have a pdf uncty (this is a really brittle workaround)
            # NOTE Someday should fix this, it's a really hardcoded and brittle and bad workaround
            pdf_uncty = [1.0,1,0]
        else:
            pdf_uncty = grs.get_syst("pdf_scale",scale_name_for_json)
        if scale_name_for_json == "convs":
            # Special case for conversions, since we estimate these from a LO sample, so we don't have an NLO uncty here
            # Would be better to handle this in a more general way
            qcd_uncty = [1.0,1,0]
        else:
            # In all other cases, use the qcd scale uncty that we have for the process
            qcd_uncty = grs.get_syst("qcd_scale",scale_name_for_json)
    else:
        pdf_uncty = [1.0,1,0]
        qcd_uncty = [1.0,1,0]

    out_dict = {"pdf_scale":pdf_uncty, "qcd_scale":qcd_uncty, "lumi":lumi_uncty, "charge_flips":flip_uncty}
    return out_dict


# Wrapper for getting plus and minus rate arrs
def get_rate_syst_arrs(base_histo,proc_group_map,group_type="CR"):

    # Fill dictionary with the rate uncertainty arrays (with correlated ones organized together)
    rate_syst_arr_dict = {}
    process_labels = yt.get_cat_lables(base_histo, "process")

    cached_rates = []
    for sample_name in process_labels:
        nominal_hist = (
            base_histo.integrate("process", sample_name)
            .integrate("systematic", "nominal")
        )
        thissample_nom_arr = _eval_without_underflow(nominal_hist)
        rate_syst_dict = get_rate_systs(sample_name, proc_group_map, group_type=group_type)
        cached_rates.append((sample_name, thissample_nom_arr, rate_syst_dict))

    for rate_sys_type in grs.get_syst_lst():
        rate_syst_arr_dict[rate_sys_type] = {}
        for sample_name, thissample_nom_arr, rate_syst_dict in cached_rates:

            # Build the plus and minus arrays from the rate uncertainty number and the nominal arr
            p_arr = thissample_nom_arr * (rate_syst_dict[rate_sys_type][1]) - thissample_nom_arr # Difference between positive fluctuation and nominal
            m_arr = thissample_nom_arr * (rate_syst_dict[rate_sys_type][0]) - thissample_nom_arr # Difference between positive fluctuation and nominal

            # Put the arrays into the correlation dict (organizing correlated ones together)
            correlation_tag = get_correlation_tag(rate_sys_type,sample_name,proc_group_map,group_type=group_type)
            out_key_name = rate_sys_type
            if correlation_tag is not None: out_key_name += "_"+correlation_tag
            if out_key_name not in rate_syst_arr_dict[rate_sys_type]:
                rate_syst_arr_dict[rate_sys_type][out_key_name] = {"p":[],"m":[]}
            rate_syst_arr_dict[rate_sys_type][out_key_name]["p"].append(p_arr)
            rate_syst_arr_dict[rate_sys_type][out_key_name]["m"].append(m_arr)

    # Now sum the linearly correlated ones and then square everything
    all_rates_p_sumw2_lst = []
    all_rates_m_sumw2_lst = []
    for syst_name in rate_syst_arr_dict.keys():
        for correlated_syst_group in rate_syst_arr_dict[syst_name]:
            sum_p_arrs = sum(rate_syst_arr_dict[syst_name][correlated_syst_group]["p"])
            sum_m_arrs = sum(rate_syst_arr_dict[syst_name][correlated_syst_group]["m"])
            all_rates_p_sumw2_lst.append(sum_p_arrs*sum_p_arrs)
            all_rates_m_sumw2_lst.append(sum_m_arrs*sum_m_arrs)

    summed_m = sum(all_rates_m_sumw2_lst) if all_rates_m_sumw2_lst else 0.0
    summed_p = sum(all_rates_p_sumw2_lst) if all_rates_p_sumw2_lst else 0.0

    return [summed_m, summed_p]

# Wrapper for getting plus and minus shape arrs
def get_shape_syst_arrs(base_histo,group_type="CR"):

    # Get the list of systematic base names (i.e. without the up and down tags)
    # Assumes each syst has a "systnameUp" and a "systnameDown" category on the systematic axis
    syst_var_lst = []
    all_syst_var_lst = yt.get_cat_lables(base_histo,"systematic")
    for syst_var_name in all_syst_var_lst:
        if syst_var_name.endswith("Up"):
            syst_name_base = "Up".join(syst_var_name.split("Up")[:-1])
            if syst_name_base not in syst_var_lst:
                syst_var_lst.append(syst_name_base)

    # Sum each systematic's contributions for all samples together (e.g. the ISR for all samples is summed linearly)
    p_arr_rel_lst = []
    m_arr_rel_lst = []
    for syst_name in syst_var_lst:
        # Skip the variation of renorm and fact together, since we're treating as independent
        if syst_name == "renormfact": continue

        relevant_samples_lst = yt.get_cat_lables(base_histo.integrate("systematic",syst_name+"Up"), "process") # The samples relevant to this syst
        proc_projection = base_histo.integrate("process", relevant_samples_lst)[{"process": sum}]
        n_arr = _eval_without_underflow(
            proc_projection.integrate("systematic", "nominal")
        )  # Sum of all samples for nominal variation
        u_arr_sum = _eval_without_underflow(
            proc_projection.integrate("systematic", syst_name + "Up")
        )
        d_arr_sum = _eval_without_underflow(
            proc_projection.integrate("systematic", syst_name + "Down")
        )

        # Special handling of renorm and fact
        # Uncorrelate these systs across the processes (though leave processes in groups like dibosons correlated to be consistent with SR)
        if (syst_name == "renorm") or (syst_name == "fact"):
            grp_map = CR_GRP_MAP if group_type == "CR" else SR_GRP_MAP
            p_arr_rel,m_arr_rel = get_decorrelated_uncty(
                syst_name,
                grp_map,
                relevant_samples_lst,
                base_histo,
                n_arr,
                total_up_arr=u_arr_sum,
                total_down_arr=d_arr_sum,
            )

        # If the syst is not renorm or fact, just treat it normally (correlate across all processes)
        else:
            u_arr_rel = u_arr_sum - n_arr # Diff with respect to nominal
            d_arr_rel = d_arr_sum - n_arr # Diff with respect to nominal
            p_arr_rel = np.maximum.reduce([u_arr_rel, d_arr_rel, 0.0])
            m_arr_rel = np.minimum.reduce([u_arr_rel, d_arr_rel, 0.0])

        # Square and append this syst to the return lists
        p_arr_rel_lst.append(p_arr_rel*p_arr_rel) # Square each element in the arr and append the arr to the out list
        m_arr_rel_lst.append(m_arr_rel*m_arr_rel) # Square each element in the arr and append the arr to the out list

    summed_m = sum(m_arr_rel_lst) if m_arr_rel_lst else 0.0
    summed_p = sum(p_arr_rel_lst) if p_arr_rel_lst else 0.0

    return [summed_m, summed_p]


# Special case for renorm and fact, as these are decorrelated across processes
# Sorry to anyone who tries to read this in the future, this function is very ad hoc and messy and hard to follow
# Just used in get_shape_syst_arrs()
# Here are a few notes:
#   - This is complicated, so I just symmetrized the errors
#   - The processes are generally correlated across groups (e.g. WZ and ZZ) since this is what's done in the datacard maker for the SR
#   - So the grouping generally follows what's in the CR group map, except in the case of signal
#       - Since in the SR all signal processes are uncorrelated for these systs, we also uncorrelate here
#       - Note there are caveats to this:
#           * In the SR, TTZToLL_M1to10 and TTToSemiLeptonic and TTTo2L2Nu are all grouped into ttll
#           * Here in the CR TTZToLL_M1to10 is part of signal group, but TTToSemiLeptonic and TTTo2L2Nu are in their own ttbar group
#           * So there are two differences with respect to how these processes are grouped in the SR:
#               1) Here TTToSemiLeptonic and TTTo2L2Nu are correlated with each other, but not with ttll
#               2) Here TTZToLL_M1to10 is grouped as part of signal (as in SR) but here _all_ signal processes are uncorrleated so here TTZToLL_M1to10 is uncorrelated with ttll while in SR they would be correlated
def _values_with_flow_or_overflow(hist_slice):
    """Return histogram values including overflow bins for different histogram types."""

    if isinstance(hist_slice, HistEFT):
        evaluated = hist_slice.eval({})
        if isinstance(evaluated, dict):
            if () in evaluated:
                return np.asarray(evaluated[()])
            return np.asarray(next(iter(evaluated.values())))
        return np.asarray(evaluated)

    values_method = hist_slice.values

    method_key = getattr(values_method, "__func__", None)
    if method_key is None:
        method_owner = type(hist_slice)
        method_name = getattr(values_method, "__name__", "values")
        method_key = (method_owner, method_name)
    capability = _VALUES_METHOD_CAPS.get(method_key)

    if capability is None:
        try:
            signature = inspect.signature(values_method)
        except (TypeError, ValueError):
            capability = "none"
        else:
            if "overflow" in signature.parameters:
                capability = "overflow"
            elif "flow" in signature.parameters:
                capability = "flow"
            else:
                capability = "none"
        _VALUES_METHOD_CAPS[method_key] = capability

    if capability == "overflow":
        values = values_method(overflow="all")
    elif capability == "flow":
        values = values_method(flow=True)
    else:
        values = values_method()

    if isinstance(values, dict):
        if () in values:
            return np.asarray(values[()])
        return np.asarray(next(iter(values.values())))

    return np.asarray(values)


def _values_without_flow(
    hist_or_values, reference_hist=None, *, include_overflow=False
):
    """Return histogram values without underflow bins.

    When ``include_overflow`` is ``True`` the overflow bin is preserved; otherwise
    it is trimmed as well.
    """

    if isinstance(hist_or_values, np.ndarray):
        values = hist_or_values
        hist_for_axes = reference_hist
    else:
        values = _values_with_flow_or_overflow(hist_or_values)
        hist_for_axes = hist_or_values

    if reference_hist is not None:
        hist_for_axes = reference_hist

    axes = getattr(hist_for_axes, "axes", None)
    if axes is None or values.ndim < len(axes):
        return values

    slices = []
    trimmed = False
    for dim_idx, axis in enumerate(axes):
        traits = getattr(axis, "traits", None)
        has_underflow = bool(getattr(traits, "underflow", False)) if traits else False
        has_overflow = bool(getattr(traits, "overflow", False)) if traits else False
        axis_bins = len(axis)
        dim_size = values.shape[dim_idx]
        if dim_size < axis_bins:
            return values

        start = 0
        stop = None
        effective_size = dim_size
        if has_underflow and effective_size > axis_bins:
            start = 1
            effective_size -= 1
        if has_overflow and effective_size > axis_bins:
            if not include_overflow:
                stop = -1
        if start != 0 or stop is not None:
            trimmed = True
        slices.append(slice(start, stop))

    if not trimmed:
        return values

    return values[tuple(slices)]


def _eval_without_underflow(hist_slice):
    """Return histogram values with the underflow bin removed."""

    evaluated = hist_slice.eval({})
    if isinstance(evaluated, dict):
        if () in evaluated:
            evaluated = evaluated[()]
        else:
            evaluated = next(iter(evaluated.values()))
    values = np.asarray(evaluated)
    if values.shape[0] == 0:
        return values
    return values[1:]


def get_decorrelated_uncty(
    syst_name,
    grp_map,
    relevant_samples_lst,
    base_histo,
    template_zeros_arr,
    *,
    total_up_arr=None,
    total_down_arr=None,
):

    # Initialize the array we will return (ok technically we return sqrt of this arr squared..)
    if total_up_arr is None:
        total_up_arr = template_zeros_arr
    if total_down_arr is None:
        total_down_arr = template_zeros_arr

    result_dtype = np.result_type(template_zeros_arr, total_up_arr, total_down_arr)
    a_arr_sum = np.zeros_like(template_zeros_arr, dtype=result_dtype) # Just using this template_zeros_arr for its size

    # Loop over the groups of processes, generally the processes in the groups will be correlated and the different groups will be uncorrelated
    for proc_grp in grp_map.keys():
        proc_lst = grp_map[proc_grp]
        if proc_grp in ["Nonprompt","Flips","Data"]: continue # Renorm and fact not relevant here
        if proc_lst == []: continue # Nothing here

        # We'll keep all signal processes as uncorrelated, similar to what's done in SR
        if proc_grp == "Signal":
            for proc_name in proc_lst:
                if proc_name not in relevant_samples_lst: continue

                n_arr_proc = _eval_without_underflow(
                    base_histo[{"process": proc_name, "systematic": "nominal"}]
                )
                u_arr_proc = _eval_without_underflow(
                    base_histo[{"process": proc_name, "systematic": syst_name + "Up"}]
                )
                d_arr_proc = _eval_without_underflow(
                    base_histo[{"process": proc_name, "systematic": syst_name + "Down"}]
                )

                u_arr_proc_rel = u_arr_proc - n_arr_proc
                d_arr_proc_rel = d_arr_proc - n_arr_proc
                a_arr_proc_rel = (abs(u_arr_proc_rel) + abs(d_arr_proc_rel))/2.0

                a_arr_sum += a_arr_proc_rel*a_arr_proc_rel

        # Otherwise corrleated across groups (e.g. ZZ and WZ, as datacard maker does in SR)
        else:
            group_projection = base_histo.integrate("process", proc_lst)[{"process": sum}]
            n_arr_grp = _eval_without_underflow(
                group_projection.integrate("systematic", "nominal")
            )
            u_arr_grp = _eval_without_underflow(
                group_projection.integrate("systematic", syst_name + "Up")
            )
            d_arr_grp = _eval_without_underflow(
                group_projection.integrate("systematic", syst_name + "Down")
            )
            u_arr_grp_rel = u_arr_grp - n_arr_grp
            d_arr_grp_rel = d_arr_grp - n_arr_grp
            a_arr_grp_rel = (abs(u_arr_grp_rel) + abs(d_arr_grp_rel))/2.0

            a_arr_sum += a_arr_grp_rel*a_arr_grp_rel

    # Before we move on, need to sqrt the outcome since later we'll square before adding in quadrature with other systs
    p_arr_rel =  np.sqrt(a_arr_sum)
    m_arr_rel = -np.sqrt(a_arr_sum)

    return [p_arr_rel,m_arr_rel]


# Get the squared arr for the jet dependent syst (e.g. for diboson jet dependent syst)
def get_diboson_njets_syst_arr(njets_histo_vals_arr,bin0_njets):

    # Get the list of njets vals for which we have SFs
    sf_int_lst = []
    diboson_njets_dict = grs.get_jet_dependent_syst_dict()
    sf_str_lst = list(diboson_njets_dict.keys())
    for s in sf_str_lst: sf_int_lst.append(int(s))
    min_njets = min(sf_int_lst) # The lowest njets bin we have a SF for
    max_njets = max(sf_int_lst) # The highest njets bin we have a SF for

    # Put the SFs into an array that matches the njets hist array
    sf_lst = []
    jet_idx = bin0_njets
    for idx in range(len(njets_histo_vals_arr)):
        if jet_idx < min_njets:
            # We do not apply the syst for these low jet bins
            sf_lst.append(1.0)
        elif jet_idx > max_njets:
            # For jet bins higher than the highest one in the dict, just use the val of the highest one
            sf_lst.append(diboson_njets_dict[str(max_njets)])
        else:
            # In this case, the exact jet bin should be included in the dict so use it directly
            sf_lst.append(diboson_njets_dict[str(jet_idx)])
        jet_idx = jet_idx + 1
    sf_arr = np.array(sf_lst)

    shift = abs(njets_histo_vals_arr - sf_arr*njets_histo_vals_arr)
    shift_sq = shift*shift # Return shift squared so we can combine with other syts in quadrature
    return shift*shift


def _is_sparse_2d_hist(histo):
    if not isinstance(histo, SparseHist):
        return False

    quadratic_axis = next(
        (ax for ax in histo.dense_axes if getattr(ax, "name", None) == "quadratic_term"),
        None,
    )
    if quadratic_axis is not None:
        try:
            # Skip the sparse 2D path only when the quadratic axis has a single bin.
            if histo.axes["quadratic_term"].size > 1:
                return True
        except (KeyError, AttributeError):
            # If the axis cannot be inspected reliably, keep the conservative 2D
            # classification to avoid mis-shaping 1D projections.
            return True

    dense_axes = [ax for ax in histo.dense_axes if ax is not quadratic_axis]
    return len(dense_axes) > 1


######### Plotting functions #########

def make_sparse2d_fig(
    h_mc,
    h_data,
    var,
    channel_name,
    lumitag="138",
    comtag="13",
    per_panel=False,
):
    axes_meta = axes_info_2d.get(var, {})
    axis_cfgs = axes_meta.get("axes", [])
    if len(axis_cfgs) < 2:
        raise ValueError(f"No 2D axis metadata configured for histogram '{var}'.")
    axis_labels = [cfg.get("label", cfg.get("name", "")) for cfg in axis_cfgs]
    cbar_label = axes_meta.get("cbar_label", "Events")
    ratio_meta = axes_meta.get("ratio", {})
    ratio_cbar_label = ratio_meta.get("cbar_label", "Data/MC")

    def _extract_weighted_values(histo):
        view = histo.view(flow=False, as_dict=True)
        if isinstance(view, dict):
            if len(view) == 1:
                view = next(iter(view.values()))
            else:
                # Fall back to the higher-level values helper when multiple
                # categorical entries remain. This preserves the dense layout
                # while still supporting weighted storages.
                view = histo.values(flow=False)

        if hasattr(view, "dtype") and view.dtype.fields:
            if "value" in view.dtype.fields:
                return np.asarray(view["value"], dtype=float)

        try:
            return np.asarray(view, dtype=float)
        except TypeError:
            return np.asarray(np.array(view), dtype=float)

    def _dense_edges(histo):
        return [np.asarray(ax.edges, dtype=float) for ax in histo.axes]

    mc_vals = _extract_weighted_values(h_mc)
    data_vals = _extract_weighted_values(h_data)
    ratio_vals = np.ones_like(data_vals, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(data_vals, mc_vals, out=ratio_vals, where=mc_vals != 0)
    empty_mask = (mc_vals == 0) & (data_vals == 0)
    data_only_mask = (mc_vals == 0) & (data_vals != 0)
    ratio_vals[empty_mask | data_only_mask] = np.nan
    dense_edges = _dense_edges(h_mc)

    def _norm_from_meta(meta_cfg, values):
        if not meta_cfg:
            return None

        norm_cfg = meta_cfg.get("norm")
        if isinstance(norm_cfg, mpl.colors.Normalize):
            return copy.copy(norm_cfg)
        if callable(norm_cfg):
            generated = norm_cfg(values)
            if isinstance(generated, mpl.colors.Normalize):
                return generated

        zlim = meta_cfg.get("zlim")
        if zlim is not None:
            vmin, vmax = zlim
            finite_vals = values[np.isfinite(values)]
            if vmin is None:
                if finite_vals.size:
                    vmin = float(np.nanmin(finite_vals))
                else:
                    vmin = 0.0
            if vmax is None:
                if finite_vals.size:
                    vmax = float(np.nanmax(finite_vals))
                else:
                    vmax = 1.0
            return mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        return None

    def _build_norm(values, dataset_key):
        dataset_meta = axes_meta.get(dataset_key, {})
        norm = _norm_from_meta(dataset_meta, values)
        if norm is None:
            norm = _norm_from_meta(axes_meta, values)
        if norm is None:
            finite_vals = values[np.isfinite(values)]
            if finite_vals.size:
                vmax = float(np.nanmax(finite_vals))
            else:
                vmax = 0.0
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
        return norm

    mc_norm = _build_norm(mc_vals, "mc")
    data_norm = _build_norm(data_vals, "data")

    finite_ratio = ratio_vals[np.isfinite(ratio_vals)]
    if "zlim" in ratio_meta:
        ratio_low, ratio_high = ratio_meta["zlim"]
        span = max(abs(1.0 - ratio_low), abs(ratio_high - 1.0))
        if not np.isfinite(span) or span <= 0:
            span = 0.5
        ratio_vmin = 1.0 - span
        ratio_vmax = 1.0 + span
    else:
        if finite_ratio.size:
            max_dev = float(np.max(np.abs(finite_ratio - 1.0)))
        else:
            max_dev = 0.0
        half_range = max(max_dev, 0.5)
        ratio_vmin = 1.0 - half_range
        ratio_vmax = 1.0 + half_range
    ratio_norm = mpl.colors.TwoSlopeNorm(vmin=ratio_vmin, vcenter=1.0, vmax=ratio_vmax)

    def _apply_panel_margins(fig):
        fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.08)

    def _make_single_panel(values, norm, title, colorbar_label):
        fig = plt.figure(figsize=(10, 9))
        hep.style.use("CMS")
        ax = fig.add_subplot(111)
        hep.cms.label(ax=ax, lumi=lumitag, com=comtag, fontsize=20.0)
        artists = hep.hist2dplot(
            values,
            ax=ax,
            norm=norm,
            xbins=dense_edges[0],
            ybins=dense_edges[1],
        )
        mesh = getattr(artists, "mesh", None)
        if mesh is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.04)
            cbar = fig.colorbar(mesh, cax=cax, norm=norm)
            cbar.set_label(colorbar_label, fontsize=18)
            cbar.ax.tick_params(labelsize=16)
        ax.set_xlabel(axis_labels[0], fontsize=20)
        ax.set_ylabel(axis_labels[1], fontsize=20)
        ax.set_title(
            f"{channel_name} {title}" if channel_name else title,
            fontsize=22,
        )
        ax.tick_params(axis="both", labelsize=16, width=1.5, length=6)
        _apply_panel_margins(fig)
        return fig

    fig = plt.figure(figsize=(20, 12))
    outer_gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1, 1],
        hspace=0.15,
        left=0.06,
        right=0.98,
        top=0.96,
        bottom=0.08,
    )
    top_gs = outer_gs[0].subgridspec(1, 2, wspace=0.12)

    hep.style.use("CMS")

    ax_mc = fig.add_subplot(top_gs[0])
    ax_data = fig.add_subplot(top_gs[1])
    ax_ratio = fig.add_subplot(outer_gs[1])

    axes_top = [ax_mc, ax_data]

    hep.cms.label(ax=ax_mc, lumi=lumitag, com=comtag, fontsize=20.0)
    for ax, plot_hist, title, norm in zip(
        axes_top,
        (mc_vals, data_vals),
        ("MC", "Data"),
        (mc_norm, data_norm),
    ):
        artists = hep.hist2dplot(
            plot_hist,
            ax=ax,
            norm=norm,
            xbins=dense_edges[0],
            ybins=dense_edges[1],
        )
        mesh = getattr(artists, "mesh", None)
        if mesh is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.04)
            cbar = fig.colorbar(mesh, cax=cax, norm=norm)
            cbar.set_label(cbar_label, fontsize=18)
            cbar.ax.tick_params(labelsize=16)
        ax.set_xlabel(axis_labels[0], fontsize=20)
        ax.set_ylabel(axis_labels[1], fontsize=20)
        ax.set_title(
            f"{channel_name} {title}" if channel_name else title,
            fontsize=22,
        )
        ax.tick_params(axis="both", labelsize=16, width=1.5, length=6)
    ratio_artists = hep.hist2dplot(
        ratio_vals,
        ax=ax_ratio,
        norm=ratio_norm,
        xbins=dense_edges[0],
        ybins=dense_edges[1],
    )
    ratio_mesh = getattr(ratio_artists, "mesh", None)
    if ratio_mesh is not None:
        divider = make_axes_locatable(ax_ratio)
        cax = divider.append_axes("right", size="5%", pad=0.04)
        ratio_cbar = fig.colorbar(ratio_mesh, cax=cax, norm=ratio_norm)
        ratio_cbar.set_label(ratio_cbar_label, fontsize=18)
        ratio_cbar.ax.tick_params(labelsize=16)
    ax_ratio.set_xlabel(axis_labels[0], fontsize=20)
    ax_ratio.set_ylabel(axis_labels[1], fontsize=20)
    ax_ratio.set_title(
        f"{channel_name} Data/MC" if channel_name else "Data/MC",
        fontsize=22,
    )
    ax_ratio.tick_params(axis="both", labelsize=16, width=1.5, length=6)
    for ax in axes_top:
        ax.set_ylabel(axis_labels[1], fontsize=20)
    _apply_panel_margins(fig)

    if not per_panel:
        return fig

    single_panel_figs = {
        "combined": fig,
        "mc": _make_single_panel(mc_vals, mc_norm, "MC", cbar_label),
        "data": _make_single_panel(data_vals, data_norm, "Data", cbar_label),
        "ratio": _make_single_panel(ratio_vals, ratio_norm, "Data/MC", ratio_cbar_label),
    }
    return single_panel_figs


# Takes two histograms and makes a region-level stacked ratio plot (with only one sparse axis, which should be "process").
# One histogram should encode the MC prediction while the other carries the data yields (or MC-substituted data when blinded).
def make_region_stacked_ratio_fig(
    h_mc,
    h_data,
    unit_norm_bool,
    axis='process',
    var='lj0pt',
    bins=None,
    group=None,
    set_x_lim=None,
    err_p=None,
    err_m=None,
    err_ratio_p=None,
    err_ratio_m=None,
    lumitag="138",
    comtag="13",
    h_mc_sumw2=None,
    syst_err=None,
    err_p_syst=None,
    err_m_syst=None,
    err_ratio_p_syst=None,
    err_ratio_m_syst=None,
    log_scale=False,
    unblind=False,
    style=None,
):
    if bins is None:
        bins = []
    else:
        bins = list(bins)
    if group is None:
        group = {}

    recompute_syst_ratio_arrays = False

    if bins:
        target_edges = _validate_bin_edges(bins)

        mc_projection = h_mc[{"process": sum}].as_hist({})
        original_edges = mc_projection.axes[var].edges
        original_mc_totals = _values_without_flow(
            mc_projection, include_overflow=True
        )

        ratio_up_input = None if err_ratio_p_syst is None else np.asarray(err_ratio_p_syst, dtype=float)
        ratio_down_input = None if err_ratio_m_syst is None else np.asarray(err_ratio_m_syst, dtype=float)

        def _ensure_absolute(up_values, down_values, up_ratio, down_ratio):
            up_array = None if up_values is None else np.asarray(up_values, dtype=float)
            down_array = None if down_values is None else np.asarray(down_values, dtype=float)

            if up_array is None and up_ratio is not None:
                up_array = up_ratio * original_mc_totals
            if down_array is None and down_ratio is not None:
                down_array = down_ratio * original_mc_totals

            return up_array, down_array

        err_p_syst, err_m_syst = _ensure_absolute(
            err_p_syst,
            err_m_syst,
            ratio_up_input,
            ratio_down_input,
        )

        target_edges_array = np.asarray(target_edges, dtype=float)
        original_edges_array = np.asarray(original_edges, dtype=float)

        same_binning = False
        if target_edges_array.shape == original_edges_array.shape:
            same_binning = np.allclose(
                target_edges_array,
                original_edges_array,
                rtol=1e-12,
                atol=1e-12,
            )

        if any(arr is not None for arr in (err_p_syst, err_m_syst)):
            recompute_syst_ratio_arrays = True

        if not same_binning:
            err_p = _rebin_uncertainty_array(
                err_p,
                original_edges,
                target_edges,
                nominal=original_mc_totals,
                direction="up",
            )
            err_m = _rebin_uncertainty_array(
                err_m,
                original_edges,
                target_edges,
                nominal=original_mc_totals,
                direction="down",
            )
            err_p_syst = _rebin_uncertainty_array(
                err_p_syst,
                original_edges,
                target_edges,
                nominal=original_mc_totals,
                direction="up",
            )
            err_m_syst = _rebin_uncertainty_array(
                err_m_syst,
                original_edges,
                target_edges,
                nominal=original_mc_totals,
                direction="down",
            )

            recompute_syst_ratio_arrays = any(
                arr is not None for arr in (err_p_syst, err_m_syst)
            )
            if recompute_syst_ratio_arrays:
                err_ratio_p_syst = None
                err_ratio_m_syst = None

            h_mc = _clone_with_rebinned_axis(h_mc, var, target_edges)
            h_data = _clone_with_rebinned_axis(h_data, var, target_edges)
            if h_mc_sumw2 is not None:
                h_mc_sumw2 = _clone_with_rebinned_axis(
                    h_mc_sumw2, var, target_edges
                )
    else:
        target_edges = None

    if style is None:
        style = {}
    axes_style = _style_get(style, ("axes",), {})
    legend_style = _style_get(style, ("legend",), {})
    uncertainty_legend_style = _style_get(style, ("uncertainty_legend",), {})
    legend_top_margin_min = legend_style.get("top_margin_min", 0.01)
    legend_top_margin_scale = legend_style.get("top_margin_scale", 0.25)
    tick_labelsize = axes_style.get("tick_labelsize", 18)
    tick_width = axes_style.get("tick_width", 1.0)
    tick_length = axes_style.get("tick_length", 4)

    default_tick_length = None
    if isinstance(STACKED_RATIO_STYLE, Mapping):
        default_tick_length = _style_get(
            STACKED_RATIO_STYLE, ("defaults", "axes", "tick_length"), None
        )

    raw_minor_tick_length = axes_style.get("minor_tick_length")
    minor_tick_ratio = axes_style.get("minor_tick_ratio")
    if (
        minor_tick_ratio is None
        and raw_minor_tick_length is not None
        and tick_length
    ):
        reference_length = axes_style.get("tick_length")
        if not isinstance(reference_length, (int, float)) or reference_length <= 0:
            reference_length = default_tick_length
        if (
            isinstance(default_tick_length, (int, float))
            and reference_length == default_tick_length
        ):
            reference_length = 6.0
        if not isinstance(reference_length, (int, float)) or reference_length <= 0:
            reference_length = 6.0
        minor_tick_ratio = raw_minor_tick_length / reference_length
    if minor_tick_ratio is None:
        minor_tick_ratio = 0.6
    minor_tick_length = tick_length * minor_tick_ratio if tick_length else 0
    spine_width = axes_style.get("spine_width", tick_width)
    axis_label_fontsize = axes_style.get("label_fontsize", 18)
    ratio_tick_labelsize = axes_style.get("ratio_tick_labelsize", tick_labelsize)
    ratio_label_text = axes_style.get("ratio_label", "Ratio")
    ratio_label_fontsize = axes_style.get(
        "ratio_label_fontsize", axis_label_fontsize
    )
    offset_fontsize = axes_style.get("offset_fontsize", axis_label_fontsize)
    y_offset = axes_style.get("y_offset", -0.07)
    overflow_label = axes_style.get("overflow_label", ">500")
    ticklabel_format_cfg = axes_style.get("ticklabel_format")
    secondary_ticks_cfg = axes_style.get("apply_secondary_ticks", {})

    if h_mc is None or h_data is None:
        return None
    if getattr(h_mc, "empty", False) and h_mc.empty():
        return None
    if getattr(h_data, "empty", False) and h_data.empty():
        return None

    default_colors = DEFAULT_STACK_COLORS

    grouping = OrderedDict()
    axis_collection = getattr(h_mc, "axes", None)
    axis_entries = None
    axis_entry_set = None
    if axis_collection is not None:
        try:
            axis_entries = axis_collection[axis]
            axis_entry_set = set(axis_entries)
        except Exception:
            axis_entries = None
            axis_entry_set = None
    for proc, members in group.items():
        if axis_entry_set is None:
            present_members = list(members)
        else:
            present_members = [p for p in members if p in axis_entry_set]
        if present_members:
            grouping[proc] = present_members
    if not grouping:
        if axis_entries is not None:
            grouping = OrderedDict((proc, [proc]) for proc in axis_entries)
        else:
            grouping = OrderedDict()

    colors = []
    default_color_index = 0
    for proc in grouping:
        c = FILL_COLORS.get(proc)
        if c is None:
            c = default_colors[default_color_index % len(default_colors)]
            default_color_index += 1
        colors.append(c)

    display_label = axes_info.get(var, {}).get("label", var)

    axis_edges = target_edges
    if axis_edges is None:
        try:
            axis_edges = h_data.axes[var].edges
        except KeyError:
            axis_edges = None
        except AttributeError:
            axis_edges = None
    if axis_edges is None:
        try:
            axis_edges = h_mc.axes[var].edges
        except (KeyError, AttributeError):
            axis_edges = None
    axis_edges = np.asarray(axis_edges, dtype=float)
    if axis_edges.size < 2:
        raise ValueError("Histogram axis has fewer than two edges; cannot determine binning.")
    last_width = axis_edges[-1] - axis_edges[-2]
    plot_bins = np.append(axis_edges, [axis_edges[-1] + last_width * 0.3])

    norm_info = _normalize_histograms(
        h_mc,
        h_data,
        unit_norm_bool,
        err_p,
        err_m,
        err_ratio_p,
        err_ratio_m,
        err_p_syst,
        err_m_syst,
        err_ratio_p_syst,
        err_ratio_m_syst,
        var,
    )

    err_p_syst = norm_info["err_p_syst"]
    err_m_syst = norm_info["err_m_syst"]
    err_ratio_p_syst = norm_info["err_ratio_p_syst"]
    err_ratio_m_syst = norm_info["err_ratio_m_syst"]
    mc_norm_factor = norm_info["mc_norm_factor"]
    mc_scaled = norm_info["mc_scaled"]

    panel_info = _draw_stacked_panel(
        h_mc,
        h_data,
        grouping,
        colors,
        axis,
        var,
        plot_bins,
        unit_norm_bool,
        lumitag,
        comtag,
        h_mc_sumw2,
        mc_scaled,
        mc_norm_factor,
        log_scale=log_scale,
        style=style,
    )

    fig = panel_info["fig"]
    ax = panel_info["ax"]
    rax = panel_info["rax"]
    bins = panel_info["bins"]
    cms_label = panel_info["cms_label"]
    mc_sumw2_vals = panel_info["mc_sumw2_vals"]
    mc_totals = panel_info["mc_totals"]
    adjusted_mc_totals = panel_info.get("adjusted_mc_totals")
    log_axis_enabled = panel_info.get("log_axis_enabled", False)
    use_log_y = log_axis_enabled
    log_y_baseline = panel_info.get("log_y_baseline")

    if recompute_syst_ratio_arrays:
        mc_totals_array = np.asarray(mc_totals, dtype=float)

        def _match_visible_bins(values):
            if values is None:
                return None
            array = np.asarray(values, dtype=float)
            if array.size == mc_totals_array.size:
                return array
            if array.size > mc_totals_array.size:
                return array[: mc_totals_array.size]
            padded = np.zeros_like(mc_totals_array, dtype=float)
            padded[: array.size] = array
            return padded

        err_p_syst = _match_visible_bins(err_p_syst)
        err_m_syst = _match_visible_bins(err_m_syst)

        with np.errstate(divide="ignore", invalid="ignore"):
            if err_p_syst is not None:
                err_ratio_p_syst = np.where(
                    mc_totals_array > 0,
                    err_p_syst / mc_totals_array,
                    1.0,
                )
            if err_m_syst is not None:
                err_ratio_m_syst = np.where(
                    mc_totals_array > 0,
                    err_m_syst / mc_totals_array,
                    1.0,
                )

    band_info = _compute_uncertainty_bands(
        ax,
        rax,
        bins,
        mc_totals,
        mc_sumw2_vals,
        h_mc_sumw2,
        unit_norm_bool,
        mc_scaled,
        mc_norm_factor,
        err_p_syst,
        err_m_syst,
        err_ratio_p_syst,
        err_ratio_m_syst,
        syst_err,
        display_mc_totals=adjusted_mc_totals,
        log_axis_enabled=log_axis_enabled,
        log_y_baseline=log_y_baseline,
        style=style,
    )

    main_band_handles = band_info.get("main_band_handles", [])

    ratio_arrays = []
    data_ratio_arrays = []

    ratio_values = panel_info.get("ratio_values")
    ratio_errors = panel_info.get("ratio_errors")
    if ratio_values is not None:
        ratio_arrays.append(np.asarray(ratio_values, dtype=float))
        data_ratio_arrays.append(np.asarray(ratio_values, dtype=float))
        if ratio_errors is not None:
            ratio_lower = np.asarray(ratio_values, dtype=float) - np.asarray(
                ratio_errors, dtype=float
            )
            ratio_upper = np.asarray(ratio_values, dtype=float) + np.asarray(
                ratio_errors, dtype=float
            )
            ratio_arrays.extend([ratio_lower, ratio_upper])
            data_ratio_arrays.extend([ratio_lower, ratio_upper])

    for key in (
        "ratio_stat_band_down",
        "ratio_stat_band_up",
        "ratio_syst_band_down",
        "ratio_syst_band_up",
        "ratio_total_band_down",
        "ratio_total_band_up",
    ):
        arr = band_info.get(key)
        if arr is not None:
            ratio_arrays.append(np.asarray(arr, dtype=float))

    (
        ratio_limits,
        exceeds_largest_window,
        data_exceeds_largest_window,
    ) = _determine_ratio_window(ratio_arrays, data_ratio_arrays)

    if exceeds_largest_window or data_exceeds_largest_window:
        warnings.warn(
            "Ratio data exceed the [-1.0, 3.0] limits; values outside the plotted range will be clipped.",
            RuntimeWarning,
        )

    ax.autoscale(axis="y")
    ax.set_xlabel(None)
    ax.tick_params(axis="both", labelsize=tick_labelsize, width=tick_width, length=tick_length)
    ax.tick_params(axis="both", which="minor", width=tick_width, length=minor_tick_length)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    if not use_log_y:
        if isinstance(ticklabel_format_cfg, Mapping):
            format_kwargs = dict(ticklabel_format_cfg)
            scilimits = format_kwargs.get("scilimits")
            if isinstance(scilimits, (list, tuple)):
                format_kwargs["scilimits"] = tuple(scilimits)
            format_kwargs.setdefault("axis", "y")
            ax.ticklabel_format(**format_kwargs)
        else:
            ax.ticklabel_format(
                axis="y", style="scientific", scilimits=(0, 6), useMathText=True
            )
    else:
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_offset_position("left")
    if y_offset is not None:
        ax.yaxis.offsetText.set_x(y_offset)
    ax.yaxis.offsetText.set_fontsize(offset_fontsize)

    rax.set_ylabel(ratio_label_text, loc="center", fontsize=ratio_label_fontsize)
    rax.set_ylim(*ratio_limits)
    rax.tick_params(
        axis="both", labelsize=ratio_tick_labelsize, width=tick_width, length=tick_length
    )
    rax.tick_params(axis="both", which="minor", width=tick_width, length=minor_tick_length)
    for spine in rax.spines.values():
        spine.set_linewidth(spine_width)

    # Ensure the ratio axis always includes a unity tick while preserving the
    # spacing chosen by the existing locator and enforcing ticks at the bounds.
    ratio_major_locator = rax.yaxis.get_major_locator()
    ratio_major_formatter = rax.yaxis.get_major_formatter()
    ratio_low, ratio_high = rax.get_ylim()
    include_unity = ratio_low <= 1.0 <= ratio_high

    major_ticks = None
    if ratio_major_locator is not None:
        try:
            major_ticks = np.asarray(
                ratio_major_locator.tick_values(ratio_low, ratio_high), dtype=float
            )
        except Exception:
            major_ticks = None
    if major_ticks is None or not np.size(major_ticks):
        major_ticks = np.asarray(rax.get_yticks(), dtype=float)

    ticks = np.asarray(major_ticks, dtype=float)
    finite_mask = np.isfinite(ticks)
    ticks = ticks[finite_mask]
    # Filter to ticks that are compatible with the current display range.
    in_range_mask = (ticks >= ratio_low) & (ticks <= ratio_high)
    ticks = ticks[in_range_mask]

    for bound in (ratio_low, ratio_high):
        if not np.any(np.isclose(ticks, bound, rtol=1e-9, atol=1e-12)):
            ticks = np.append(ticks, bound)

    if include_unity and not np.any(np.isclose(ticks, 1.0, rtol=1e-9, atol=1e-12)):
        ticks = np.append(ticks, 1.0)

    if ticks.size:
        ticks = np.unique(ticks)
        ticks.sort()
        rax.yaxis.set_major_locator(FixedLocator(ticks.tolist()))
        # Reapply the formatter to preserve styling (e.g., mathtext/scientific).
        if ratio_major_formatter is not None:
            rax.yaxis.set_major_formatter(ratio_major_formatter)

    fig.canvas.draw()
    xticks = rax.get_xticks()
    xtick_labels = [tick.get_text() for tick in rax.get_xticklabels()]
    if (
        overflow_label is not None
        and xtick_labels
        and len(xtick_labels) == len(xticks)
    ):
        xtick_labels[-1] = overflow_label
        rax.xaxis.set_major_locator(FixedLocator(xticks))
        rax.xaxis.set_major_formatter(FixedFormatter(xtick_labels))

    apply_minor_x = bool(secondary_ticks_cfg.get("x", True))
    apply_minor_y = bool(secondary_ticks_cfg.get("y", True))
    if apply_minor_x:
        _apply_secondary_ticks(ax, axis="x")
        _apply_secondary_ticks(rax, axis="x")
    if apply_minor_y:
        _apply_secondary_ticks(ax, axis="y")
        _apply_secondary_ticks(rax, axis="y")

    ax_box = ax.get_position()
    rax_box = rax.get_position()
    ratio_label_fig = None
    ratio_label = rax.yaxis.label
    if ratio_label is not None:
        try:
            ratio_label_pos = np.asarray(ratio_label.get_position(), dtype=float)
            ratio_label_transform = ratio_label.get_transform()
            if ratio_label_transform is not None:
                ratio_label_display = ratio_label_transform.transform([ratio_label_pos])[0]
                ratio_label_fig = fig.transFigure.inverted().transform(ratio_label_display)
        except Exception:
            ratio_label_fig = None

    initial_events_anchor = (rax_box.x0 + rax_box.width, ax_box.y0 + ax_box.height)

    # Set the x axis lims
    if set_x_lim: plt.xlim(set_x_lim)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    # Build a figure-anchored legend with a measured inset from the top edge
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend = None
    if legend_handles and legend_labels:
        filtered = OrderedDict()
        for handle, label in zip(legend_handles, legend_labels):
            if label == '_nolegend_':
                continue
            if label not in filtered:
                filtered[label] = handle
        if filtered:
            max_rows = legend_style.get("max_rows", 3)
            ncol = legend_style.get("ncol", 5)
            if not isinstance(ncol, int) or ncol <= 0:
                ncol = 1
            entries = list(filtered.items())
            nrows = math.ceil(len(entries) / ncol)
            if nrows > max_rows:
                warnings.warn(
                    "Legend contains more than 15 entries; truncating to fit a 5x3 layout.",
                    RuntimeWarning,
                )
                entries = entries[: ncol * max_rows]
                nrows = max_rows
            bbox_to_anchor = legend_style.get("bbox_to_anchor", (0.5, 1.0))
            if isinstance(bbox_to_anchor, (list, tuple)):
                bbox_to_anchor = tuple(bbox_to_anchor)
            legend_kwargs = {
                "loc": legend_style.get("loc", "upper center"),
                "bbox_to_anchor": bbox_to_anchor,
                "borderaxespad": legend_style.get("borderaxespad", 0.15),
                "ncol": ncol,
                "fontsize": legend_style.get("fontsize", 16),
                "columnspacing": legend_style.get("columnspacing", 0.8),
                "handletextpad": legend_style.get("handletextpad", 0.6),
            }
            labelspacing = legend_style.get("labelspacing")
            if labelspacing is not None:
                legend_kwargs["labelspacing"] = labelspacing
            frameon = legend_style.get("frameon")
            if frameon is not None:
                legend_kwargs["frameon"] = frameon
            legend = fig.legend(
                [handle for _, handle in entries],
                [label for label, _ in entries],
                **legend_kwargs,
            )
    if main_band_handles:
        unc_handles, unc_labels = zip(*main_band_handles)
        unc_bbox = uncertainty_legend_style.get("bbox_to_anchor", (0.98, 0.98))
        if isinstance(unc_bbox, (list, tuple)):
            unc_bbox = tuple(unc_bbox)
        else:
            unc_bbox = (0.98, 0.98)
        _ = ax.legend(
            handles=list(unc_handles),
            labels=list(unc_labels),
            loc=uncertainty_legend_style.get("loc", "upper right"),
            bbox_to_anchor=unc_bbox,
            frameon=uncertainty_legend_style.get("frameon", False),
            fontsize=uncertainty_legend_style.get("fontsize", 10),
            ncol=uncertainty_legend_style.get("ncol", 2),
            columnspacing=uncertainty_legend_style.get("columnspacing", 1.0),
        )

    fig.canvas.draw()
    required_headroom = None
    legend_is_figure_anchored = False
    top_adjusted = False
    legend_anchor = None
    if legend is not None:
        renderer = fig.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer=renderer)
        legend_box = legend_bbox.transformed(fig.transFigure.inverted())
        measured_height = legend_box.height
        buffer = max(legend_top_margin_min, legend_top_margin_scale * measured_height)
        anchor_y = max(0.0, 1.0 - buffer)
        legend_anchor = [0.5, anchor_y]
        legend.set_bbox_to_anchor(tuple(legend_anchor), fig.transFigure)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer=renderer)
        legend_box = legend_bbox.transformed(fig.transFigure.inverted())
        legend_height = legend_box.height
        buffer = max(buffer, legend_top_margin_min)
        required_headroom = legend_height + buffer
        legend_is_figure_anchored = True
        subplot_params = fig.subplotpars
        available_top = 1.0 - required_headroom
        available_top = np.clip(available_top, 0.0, 1.0)
        if subplot_params.top > available_top:
            plt.subplots_adjust(top=available_top)
            top_adjusted = True
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            legend_bbox = legend.get_window_extent(renderer=renderer)
            legend_box = legend_bbox.transformed(fig.transFigure.inverted())

    label_artist = None
    events_artist = None
    iterations = 3 if top_adjusted else 2
    for _ in range(iterations):
        label_artist, events_artist, legend_anchor = _finalize_layout(
            fig,
            ax,
            rax,
            legend,
            cms_label,
            display_label,
            label_artist=label_artist,
            events_artist=events_artist,
            ratio_anchor=ratio_label_fig,
            events_anchor=initial_events_anchor,
            legend_anchor=legend_anchor,
            legend_is_figure=legend_is_figure_anchored,
            style=style,
        )

    return fig

###################### Region plotting entry point ######################
# Execute the region-agnostic plotting pipeline for the requested region name.
# The caller provides the histogram dictionary that includes data and MC.
def run_plots_for_region(
    region_name,
    dict_of_hists,
    years,
    save_dir_path,
    *,
    skip_syst_errs=False,
    unit_norm_bool=False,
    stacked_log_y=False,
    variables=None,
    unblind=None,
    workers=1,
    verbose=False,
):
    region_ctx = build_region_context(
        region_name,
        dict_of_hists,
        years,
        unblind=unblind,
    )
    produce_region_plots(
        region_ctx,
        save_dir_path,
        variables,
        skip_syst_errs,
        unit_norm_bool,
        stacked_log_y,
        unblind=unblind,
        workers=workers,
        verbose=verbose,
    )

def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    parser.add_argument("-o", "--output-path", default=".", help = "The path the output files should be saved to")
    parser.add_argument("-n", "--output-name", default="plots", help = "A name for the output directory")
    parser.add_argument("-t", "--include-timestamp-tag", action="store_true", help = "Append the timestamp to the out dir name")
    parser.add_argument(
        "-y",
        "--year",
        nargs="+",
        help="One or more year tokens or aggregates to include (e.g. 2017 2018, run2, run3)",
    )
    parser.add_argument("-u", "--unit-norm", action="store_true", help = "Unit normalize the plots")
    parser.add_argument(
        "--log-y",
        dest="log_y",
        action="store_true",
        help="Use a logarithmic y-axis for the stacked (upper) panel; the ratio subplot remains linear.",
    )
    parser.add_argument(
        "-s",
        "--skip-syst",
        default=False,
        action="store_true",
        help="Skip systematic error bands in plots (statistical bands fall back to Poisson when sumw² histograms are absent)",
    )
    parser.add_argument(
        "--unblind",
        dest="unblind",
        action="store_true",
        help="Force plots to include data yields even in normally blinded regions.",
    )
    parser.add_argument(
        "--blind",
        dest="unblind",
        action="store_false",
        help="Force plots to hide data yields even in normally unblinded regions.",
    )
    region_group = parser.add_mutually_exclusive_group()
    region_group.add_argument(
        "--cr",
        dest="region_override",
        action="store_const",
        const="CR",
        help="Force control-region plotting, overriding filename-based detection.",
    )
    region_group.add_argument(
        "--sr",
        dest="region_override",
        action="store_const",
        const="SR",
        help="Force signal-region plotting, overriding filename-based detection.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Optional list of histogram variables to plot",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel variable rendering (default: 1).",
    )
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable detailed diagnostic output (variable lists, channel dumps).",
    )
    verbosity_group.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Limit output to high-level progress messages (default).",
    )
    parser.set_defaults(unblind=None, verbose=False)
    args = parser.parse_args()
    normalized_years = _normalize_year_tokens(args.year)
    if args.year and not normalized_years:
        parser.error(
            "No valid year tokens were provided; expected one or more of: {}".format(
                ", ".join(sorted(YEAR_TOKEN_RULES))
            )
        )
    selected_years = normalized_years

    def _detect_region_from_path(path):
        if not path:
            return None, False
        filename = os.path.basename(path)
        uppercase = filename.upper()
        matched_regions = []
        for region in ("CR", "SR"):
            # Accept filenames where the region token is directly followed by
            # qualifiers such as a year (e.g. "SR2018") or run tag (e.g. "CRRun2").
            # We only guard against being embedded within a longer alphanumeric
            # token by ensuring the preceding character is not an uppercase
            # letter or digit.
            pattern = re.compile(rf"(?<![A-Z0-9]){region}")
            if pattern.search(uppercase):
                matched_regions.append(region)
        if len(matched_regions) == 1:
            return matched_regions[0], False
        if len(matched_regions) > 1:
            return None, True
        return None, False

    detected_region, ambiguous_region = _detect_region_from_path(args.pkl_file_path)
    resolved_region = args.region_override or detected_region or "CR"
    if ambiguous_region and not args.region_override:
        print(
            "Warning: Detected both 'CR' and 'SR' tokens in the input filename. "
            "Defaulting to 'CR'; please pass --cr or --sr to specify explicitly."
        )

    if args.unblind is None:
        resolved_unblind = resolved_region == "CR"
        blinding_source = f"default for {resolved_region} region"
    elif args.unblind:
        resolved_unblind = True
        blinding_source = "command-line --unblind override"
    else:
        resolved_unblind = False
        blinding_source = "command-line --blind override"

    print(f"Resolved plotting region: {resolved_region}")
    print(
        "Resolved blinding mode: {} ({})".format(
            "unblinded" if resolved_unblind else "blinded", blinding_source
        )
    )

    normalized_variables = []
    if args.variables is not None:
        seen_variables = set()
        for value in args.variables:
            if value is None:
                continue
            cleaned = value.strip()
            if not cleaned or cleaned in seen_variables:
                continue
            seen_variables.add(cleaned)
            normalized_variables.append(cleaned)
    selected_variables = normalized_variables if normalized_variables else None

    # Whether or not to unit norm the plots
    unit_norm_bool = args.unit_norm

    # Make a tmp output directory in curren dir a different dir is not specified
    timestamp_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    save_dir_path = args.output_path
    outdir_name = args.output_name
    if args.include_timestamp_tag:
        outdir_name = outdir_name + "_" + timestamp_tag
    save_dir_path = os.path.join(save_dir_path,outdir_name)
    os.mkdir(save_dir_path)

    # Get the histograms
    hin_dict = utils.get_hist_from_pkl(args.pkl_file_path,allow_empty=False)
    # Print info about histos
    #yt.print_hist_info(args.pkl_file_path,"nbtagsl")
    #exit()

    print("\nMaking plots for years:", selected_years if selected_years else "All")
    print("Output dir:",save_dir_path)
    print("Variables to plot:", selected_variables if selected_variables else "All")
    print("\n\n")

    # Make the plots
    run_plots_for_region(
        resolved_region,
        hin_dict,
        selected_years,
        save_dir_path,
        skip_syst_errs=args.skip_syst,
        unit_norm_bool=unit_norm_bool,
        stacked_log_y=args.log_y,
        variables=selected_variables,
        unblind=resolved_unblind,
        workers=args.workers,
        verbose=args.verbose,
    )
if __name__ == "__main__":
    main()
