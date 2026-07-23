##############################################################
# Script for creating the fake tau scale factors
#
# Statistical and systematic uncertainties on the fake-rate points are
# combined in quadrature, matching the expectation for uncorrelated
# sources and avoiding the overly conservative behaviour of the
# historical linear addition.
#
# Active script: analysis/topeft_run2/faketau_sf_fitter.py
# Pass one or more input pkls with -f/--pkl-file-path. Input pkls should
# include the fake- and tight-tau CR histograms and all corrections aside
# from fake tau SFs.
# output is in the form of linear fit y = mx+b
# where m and b are in numerical form, y is the SF, and x is the tau pt
# tau fake-rate fits use regrouped pt edges [20, 30, 40, 50, 60, 200]
# Detailed setup instructions and interpretation notes live in
# analysis/topeft_run2/README_faketau_sf_fitter.md.

import numpy as np
import os
import argparse
import copy
import json
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
import functools
import hist

import sys
import re
from scipy.optimize import curve_fit

from topeft.modules.paths import topeft_path
from topeft.modules.yield_tools import YieldTools
from topeft.modules.datacard_tools import load_and_merge_histogram_pkls
from topeft.modules.nominal_schema import (
    NOMINAL_CONTAINER_SCHEMA_VERSION,
    evaluate_nominal_at_wc,
    is_split_nominal_mapping,
)
from topcoffea.modules.histEFT import HistEFT
import topcoffea.modules.utils as utils

yt = YieldTools()


LOGGER = logging.getLogger(__name__)

_TAU_FAKE_HISTOGRAM_REQUIRED_AXES = ("process", "channel", "systematic", "tau0Fpt")
_TAU_TIGHT_HISTOGRAM_REQUIRED_AXES = ("process", "channel", "systematic", "tau0Tpt")
DEFAULT_INPUT_PKL_PATH = "histos/plotsTopEFT.pkl.gz"
FAKETAU_REQUIRED_HISTOGRAMS = ("tau0Fpt", "tau0Tpt")
FAKETAU_REQUIRED_SUMW2_HISTOGRAMS = ("tau0Fpt_sumw2", "tau0Tpt_sumw2")


YEAR_TOKEN_RULES = {
    "2016": {
        "mc_wl": ["UL16", "2016"],
        "mc_bl": ["UL16APV", "2016APV"],
        "data_wl": ["UL16", "2016"],
        "data_bl": ["UL16APV", "2016APV"],
    },
    "2016APV": {
        "mc_wl": ["UL16APV", "2016APV"],
        "data_wl": ["UL16APV", "2016APV"],
    },
    "2017": {"mc_wl": ["UL17", "2017"], "data_wl": ["UL17", "2017"]},
    "2018": {"mc_wl": ["UL18", "2018"], "data_wl": ["UL18", "2018"]},
    "2022": {"mc_wl": ["2022"], "data_wl": ["2022"]},
    "2022EE": {"mc_wl": ["2022EE"], "data_wl": ["2022EE"]},
    "2023": {"mc_wl": ["2023"], "data_wl": ["2023"]},
    "2023BPix": {"mc_wl": ["2023BPix"], "data_wl": ["2023BPix"]},
}

YEAR_WHITELIST_OPTIONALS = set()
for _year_rule in YEAR_TOKEN_RULES.values():
    YEAR_WHITELIST_OPTIONALS.update(_year_rule.get("mc_wl", []))
    YEAR_WHITELIST_OPTIONALS.update(_year_rule.get("data_wl", []))


class HistogramAxisError(RuntimeError):
    """Exception raised when a histogram is missing required axes."""

    def __init__(self, message, *, missing_axes=None, present_axes=None):
        super().__init__(message)
        self.missing_axes = tuple(missing_axes or ())
        self.present_axes = tuple(present_axes or ())


def _extract_jet_suffix(jet_label):
    jet_digits = "".join(ch for ch in jet_label if ch.isdigit())
    if not jet_digits:
        raise RuntimeError(
            f"Unable to determine jet multiplicity from label '{jet_label}' in tau channel configuration."
        )
    return f"{jet_digits}j"


def _insert_flavor(base_name, flavor):
    if "_" not in base_name:
        return f"{base_name}_{flavor}"
    prefix, remainder = base_name.split("_", 1)
    return f"{prefix}_{flavor}_{remainder}"


def load_tau_control_channels(channels_json_path=None):
    """Build the Ftau and Ttau channel lists from the channel configuration JSON."""

    json_path = channels_json_path or topeft_path("channels/ch_lst.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Tau channel configuration JSON not found at '{json_path}'."
        )

    with open(json_path, "r") as ch_json:
        try:
            channel_config = json.load(ch_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse tau channel configuration at '{json_path}': {exc}."
            ) from exc

    try:
        tau_controls = channel_config["TAU_CH_LST_CR"]["2los_1tau"]
    except KeyError as exc:
        missing_key = exc.args[0] if exc.args else "TAU_CH_LST_CR"
        if missing_key == "TAU_CH_LST_CR":
            missing_path = "TAU_CH_LST_CR"
        else:
            missing_path = f"TAU_CH_LST_CR -> {missing_key}"
        raise RuntimeError(
            "The tau control-region definition is missing from the channel configuration. "
            f"Expected to find '{missing_path}' in '{json_path}'."
        ) from exc

    required_fields = ["lep_chan_lst", "lep_flav_lst", "jet_lst"]
    for field in required_fields:
        if field not in tau_controls:
            raise RuntimeError(
                f"Tau control-region definition in '{json_path}' is missing the '{field}' field."
            )

    lep_chans = tau_controls["lep_chan_lst"]
    lep_flavs = tau_controls["lep_flav_lst"]
    jet_bins = tau_controls["jet_lst"]

    if not lep_chans or not lep_flavs or not jet_bins:
        raise RuntimeError(
            "Tau control-region configuration must define non-empty channel, flavor, and jet lists."
        )

    ftau_channels = []
    ttau_channels = []

    jet_suffixes = {
        jet_label: _extract_jet_suffix(jet_label) for jet_label in jet_bins
    }

    for chan_def in lep_chans:
        if not chan_def:
            continue
        base_name = chan_def[0]
        if base_name.endswith("_Ftau"):
            target_list = ftau_channels
        elif base_name.endswith("_Ttau"):
            target_list = ttau_channels
        else:
            raise RuntimeError(
                f"Unexpected tau control channel name '{base_name}'. "
                "Expected names ending with '_Ftau' or '_Ttau'."
            )

        flavored_bases = [_insert_flavor(base_name, flavor) for flavor in lep_flavs]

        for flavored_base in flavored_bases:
            for jet_label in jet_bins:
                channel_name = f"{flavored_base}_{jet_suffixes[jet_label]}"
                target_list.append(channel_name)

    if not ftau_channels or not ttau_channels:
        raise RuntimeError(
            "Failed to build Ftau/Ttau channel lists from the tau control-region configuration."
        )

    return ftau_channels, ttau_channels


def normalize_input_pkl_paths(paths=None):
    """Return a non-empty list of input pkl paths."""

    if paths is None:
        return [DEFAULT_INPUT_PKL_PATH]
    if isinstance(paths, (str, os.PathLike)):
        normalized = [os.fspath(paths)]
    else:
        normalized = [os.fspath(path) for path in paths if path is not None]

    if not normalized:
        return [DEFAULT_INPUT_PKL_PATH]
    if any(not path for path in normalized):
        raise ValueError("Input pkl paths must be non-empty strings.")
    return normalized


def load_histogram_pkl(path):
    """Load one histogram pkl with the same utility used by the legacy fitter."""

    histograms = utils.get_hist_from_pkl(path, allow_empty=False)
    if not isinstance(histograms, dict):
        raise TypeError(
            f"Input pkl file '{path}' did not contain a histogram dictionary. "
            f"Got: {type(histograms).__name__}."
        )
    return histograms


def _materialize_faketau_scalar_view(histograms):
    """Return only the strict WC=0 tau families and scalar companions."""

    if not is_split_nominal_mapping(histograms):
        return histograms
    output = OrderedDict()
    for family in FAKETAU_REQUIRED_HISTOGRAMS:
        output[family] = evaluate_nominal_at_wc(
            histograms,
            family,
            {},
            schema_version=NOMINAL_CONTAINER_SCHEMA_VERSION,
        )
        companion_key = f"{family}_sumw2"
        if companion_key not in histograms:
            raise RuntimeError(
                f"Fake-tau fitting requires selected companion '{companion_key}'."
            )
        output[companion_key] = copy.deepcopy(histograms[companion_key])
    return output


def _combine_histogram_sequence(hist_name, histograms, input_paths):
    """Combine histograms without using a loaded input object as an accumulator."""

    combined = histograms[0]
    for incoming_hist, input_path in zip(histograms[1:], input_paths[1:]):
        try:
            combined = combined + incoming_hist
        except Exception as exc:
            raise RuntimeError(
                f"Failed to combine histogram '{hist_name}' from input pkl file "
                f"'{input_path}' with earlier input file(s): {exc!r}."
            ) from exc
    return combined


def merge_faketau_histogram_dicts(histogram_dicts, *, input_paths):
    """Merge the fake-tau fitter's required histograms from loaded dictionaries."""

    histogram_dicts = list(histogram_dicts)
    input_paths = normalize_input_pkl_paths(input_paths)
    if len(histogram_dicts) != len(input_paths):
        raise ValueError(
            "The number of histogram dictionaries does not match the number of input paths."
        )
    if not histogram_dicts:
        raise ValueError("No histogram dictionaries were provided for fake-tau merging.")

    for input_path, histograms in zip(input_paths, histogram_dicts):
        if not isinstance(histograms, dict):
            raise TypeError(
                f"Input pkl file '{input_path}' did not contain a histogram dictionary. "
                f"Got: {type(histograms).__name__}."
            )
        missing_required = [
            hist_name
            for hist_name in (
                *FAKETAU_REQUIRED_HISTOGRAMS,
                *FAKETAU_REQUIRED_SUMW2_HISTOGRAMS,
            )
            if hist_name not in histograms
        ]
        if missing_required:
            raise RuntimeError(
                f"Input pkl file '{input_path}' is missing required fake-tau "
                f"histogram(s): {', '.join(missing_required)}."
            )

    combined_histograms = OrderedDict()
    for hist_name in FAKETAU_REQUIRED_HISTOGRAMS:
        combined_histograms[hist_name] = _combine_histogram_sequence(
            hist_name,
            [histograms[hist_name] for histograms in histogram_dicts],
            input_paths,
        )

    sumw2_status = {}
    for hist_name in FAKETAU_REQUIRED_SUMW2_HISTOGRAMS:
        combined_histograms[hist_name] = _combine_histogram_sequence(
            hist_name,
            [histograms[hist_name] for histograms in histogram_dicts],
            input_paths,
        )
        sumw2_status[hist_name] = "present in all input files"

    input_summary = {
        "num_inputs": len(input_paths),
        "input_paths": tuple(input_paths),
        "required_histograms": FAKETAU_REQUIRED_HISTOGRAMS,
        "sumw2_status": sumw2_status,
    }
    return combined_histograms, input_summary


def combine_faketau_histogram_pkls(paths):
    """Load and combine one or more fake-tau fitter input pkls."""

    input_paths = normalize_input_pkl_paths(paths)
    merged, merge_report = load_and_merge_histogram_pkls(
        input_paths,
        on_process_collision="allow",
        require_sumw2=True,
        consumer_required_families=FAKETAU_REQUIRED_HISTOGRAMS,
    )
    scalar_view = _materialize_faketau_scalar_view(merged)
    summary = {
        "num_inputs": len(input_paths),
        "input_paths": tuple(input_paths),
        "required_histograms": FAKETAU_REQUIRED_HISTOGRAMS,
        "sumw2_status": {
            key: "present in all input files"
            for key in FAKETAU_REQUIRED_SUMW2_HISTOGRAMS
        },
        "schema": merge_report["schema"],
    }
    return scalar_view, summary


def print_faketau_input_summary(input_summary):
    """Print the concise input summary used before fake-rate extraction."""

    print(f"Input pkl file(s): {input_summary['num_inputs']}")
    for input_path in input_summary["input_paths"]:
        print(f"  - {input_path}")
    print(
        "Combining required histograms: "
        + ", ".join(input_summary["required_histograms"])
    )
    for hist_name in FAKETAU_REQUIRED_SUMW2_HISTOGRAMS:
        print(f"{hist_name}: {input_summary['sumw2_status'][hist_name]}")


def _gather_axis_alias_tokens(mapping):
    """Recursively collect string tokens from an axis-alias mapping."""

    tokens = set()
    if isinstance(mapping, str):
        tokens.add(mapping)
    elif isinstance(mapping, dict):
        for key, value in mapping.items():
            tokens.update(_gather_axis_alias_tokens(key))
            tokens.update(_gather_axis_alias_tokens(value))
    elif isinstance(mapping, (list, tuple, set)):
        for entry in mapping:
            tokens.update(_gather_axis_alias_tokens(entry))
    return tokens


def _collect_processes(histograms, predicate):
    """Collect unique process names from histograms preserving first-seen order."""

    collected = []
    seen = set()

    def _append_process(process_name):
        if process_name not in seen:
            collected.append(process_name)
            seen.add(process_name)

    for histogram in histograms:
        if histogram is None:
            continue
        for process in histogram.axes["process"]:
            process_name = str(process)
            if predicate(process_name):
                _append_process(process_name)

    return collected


def _maybe_remove_processes(histogram, axis_name, labels):
    """Return the original histogram when no removal is required."""

    if histogram is None or not labels:
        return histogram

    return histogram.remove(axis_name, labels)


def _group_all(histograms, mapping):
    """Regroup histograms in a tuple while preserving order and ``None`` entries."""

    regrouped = []
    for histogram in histograms:
        if histogram is None:
            regrouped.append(None)
        else:
            regrouped.append(
                group_bins(
                    histogram,
                    mapping,
                    "process",
                    drop_unspecified=True,
                )
            )
    return tuple(regrouped)


def _integrate_nominal(histogram):
    """Integrate the nominal systematic axis while preserving ``None`` inputs."""

    if histogram is None:
        return None

    return histogram.integrate("systematic", "nominal")


def _integrate_tau_channels(histogram, channels):
    """Integrate tau histograms over the requested channels with safe None handling."""

    if histogram is None or channels is None:
        return None

    return histogram.integrate("channel", channels)[{"channel": sum}]


def _extend_unique(target, values):
    """Append values to target while preserving order and avoiding duplicates."""

    if not values:
        return target

    for value in values:
        if value not in target:
            target.append(value)
    return target


def _prune_tokens(target, forbidden):
    """Return a list excluding forbidden tokens while preserving order."""

    if not forbidden:
        return list(target)

    forbidden_set = set(forbidden)
    return [token for token in target if token not in forbidden_set]


def _resolve_year_filters(year_args):
    """Normalise CLI year arguments and build per-sample filter lists."""

    mc_wl = []
    mc_bl = ["data"]
    data_wl = ["data"]
    data_bl = []

    if year_args is None:
        normalized = None
    else:
        if isinstance(year_args, str):
            raw_tokens = [year_args]
        else:
            raw_tokens = list(year_args)

        normalized = []
        seen = set()
        for token in raw_tokens:
            if token is None:
                continue
            cleaned = str(token).strip()
            if not cleaned or cleaned in seen:
                continue
            if cleaned not in YEAR_TOKEN_RULES:
                raise ValueError(
                    "Unknown year token '{}' requested. Supported tokens: {}".format(
                        cleaned, ", ".join(sorted(YEAR_TOKEN_RULES))
                    )
                )

            rules = YEAR_TOKEN_RULES[cleaned]

            mc_wl_values = rules.get("mc_wl", [])
            _extend_unique(mc_wl, mc_wl_values)
            mc_bl = _prune_tokens(mc_bl, mc_wl_values)

            mc_bl_values = rules.get("mc_bl", [])
            _extend_unique(mc_bl, mc_bl_values)

            data_wl_values = rules.get("data_wl", [])
            _extend_unique(data_wl, data_wl_values)
            data_bl = _prune_tokens(data_bl, data_wl_values)

            data_bl_values = rules.get("data_bl", [])
            _extend_unique(data_bl, data_bl_values)

            normalized.append(cleaned)
            seen.add(cleaned)

        if not normalized:
            normalized = None

    mc_bl = list(dict.fromkeys(mc_bl))
    data_bl = list(dict.fromkeys(data_bl))

    return {
        "selected_years": normalized,
        "mc_whitelist": mc_wl,
        "mc_blacklist": mc_bl,
        "data_whitelist": data_wl,
        "data_blacklist": data_bl,
    }


def _filter_samples(all_labels, whitelist, blacklist):
    """Return samples that satisfy blacklist rules and multi-token requirements."""

    if len(whitelist) <= 1:
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

    must_have_tokens = list(dict.fromkeys(must_have_tokens))
    optional_tokens = list(dict.fromkeys(optional_tokens))
    optional_token_set = set(optional_tokens)

    @functools.lru_cache(maxsize=None)
    def _present_year_tokens(label):
        detected_tokens = {
            year_token
            for year_token in YEAR_WHITELIST_OPTIONALS
            if year_token in label
        }
        if len(detected_tokens) <= 1:
            result = frozenset(detected_tokens)
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
        return result

    def _label_contains_disallowed_year(present_tokens):
        if not optional_tokens or not present_tokens:
            return False
        return any(token not in optional_token_set for token in present_tokens)

    def _label_passes(label):
        if any(token in label for token in blacklist):
            return False
        if must_have_tokens and any(token not in label for token in must_have_tokens):
            return False
        present_tokens = _present_year_tokens(label)
        if _label_contains_disallowed_year(present_tokens):
            return False
        if optional_tokens and not present_tokens.intersection(optional_token_set):
            return False
        return True

    return [label for label in all_labels if _label_passes(label)]


def _resolve_histogram_axis_names(histogram):
    """Return the actual axis names, recognised aliases, and canonicalised names."""

    present_axes = set()
    recognised_axes = set()
    canonical_axes = set()

    alias_mapping = {}
    metadata = getattr(histogram, "metadata", None)
    if isinstance(metadata, dict):
        alias_mapping = metadata.get("_hist_sumw2_axis_mapping") or {}
        if alias_mapping:
            recognised_axes.update(_gather_axis_alias_tokens(alias_mapping))

    if not isinstance(alias_mapping, dict):
        alias_mapping = {}

    for axis in getattr(histogram, "axes", ()):
        axis_name = getattr(axis, "name", None)
        if not axis_name:
            continue
        present_axes.add(axis_name)
        recognised_axes.add(axis_name)
        if axis_name.endswith("_sumw2"):
            base_name = axis_name[:-6]
            if base_name:
                recognised_axes.add(base_name)

        canonical_name = alias_mapping.get(axis_name)
        if isinstance(canonical_name, str):
            canonical_axes.add(canonical_name)
        elif isinstance(canonical_name, (list, tuple, set)):
            for entry in canonical_name:
                if isinstance(entry, str):
                    canonical_axes.add(entry)

        if not canonical_name:
            if axis_name.endswith("_sumw2") and len(axis_name) > 6:
                canonical_axes.add(axis_name[:-6])
            else:
                canonical_axes.add(axis_name)

    return present_axes, recognised_axes, canonical_axes

def _strip_tau_flow(array, expected_bins):
    """Trim under/overflow entries from a tau histogram projection."""

    arr = np.asarray(array, dtype=float)
    if arr.size == 0:
        return np.zeros(expected_bins, dtype=float)

    squeezed = np.squeeze(arr)
    if squeezed.ndim != 1:
        raise RuntimeError(
            "Unexpected tau flow array shape "
            f"{arr.shape}; unable to resolve a 1D tau-pt spectrum."
        )

    if squeezed.size == expected_bins:
        return squeezed.astype(float, copy=True)

    if squeezed.size == expected_bins + 2:
        if expected_bins <= 0:
            raise RuntimeError(
                "Cannot strip tau flow information when no physical bins are expected."
            )

        physical = squeezed[1:-1].astype(float, copy=True)
        if physical.size != expected_bins:
            raise RuntimeError(
                "Failed to extract the expected number of tau-pt bins after removing "
                "under/overflow entries."
            )

        if physical.size:
            physical[-1] += squeezed[-1]
        elif squeezed[-1] != 0:
            raise RuntimeError(
                "Encountered non-zero overflow content with no physical tau-pt bins."
            )
        return physical

    raise RuntimeError(
        "Unexpected tau flow array length "
        f"{squeezed.size}; expected {expected_bins} (trimmed) or "
        f"{expected_bins + 2} (with flow entries)."
    )


def _fold_tau_overflow(array, expected_bins=None):
    """Merge overflow into the last tau-pt bin of a 1D array."""

    if expected_bins is None:
        raise ValueError("The expected number of tau bins must be provided.")

    arr = np.asarray(array, dtype=float)
    if arr.ndim != 1:
        raise RuntimeError(
            "Overflow folding expects a 1D tau spectrum; received shape "
            f"{arr.shape}."
        )

    return _strip_tau_flow(arr, expected_bins)


def _collapse_quadratic_axis(array, histogram, axis_names):
    if "quadratic_term" not in axis_names:
        return array

    quad_index = axis_names.index("quadratic_term")
    quad_axis = histogram.axes[quad_index]
    coeff_index = 0

    try:
        identifiers = getattr(quad_axis, "identifiers", None)
        if identifiers is not None:
            identifiers_iter = identifiers() if callable(identifiers) else identifiers
            for idx, identifier in enumerate(identifiers_iter):
                ident_value = getattr(identifier, "value", identifier)
                if isinstance(ident_value, str):
                    stripped = ident_value.strip()
                    if stripped.isdigit():
                        ident_value = int(stripped)
                    elif stripped.lower() in {"const", "sm,sm"}:
                        ident_value = 0
                if ident_value == 0:
                    coeff_index = idx
                    break
    except Exception:
        coeff_index = 0

    take_index = coeff_index
    traits = getattr(quad_axis, "traits", None)
    if traits is not None and getattr(traits, "underflow", False):
        take_index += 1

    return np.take(array, take_index, axis=quad_index)


def _ensure_dense_histogram(histogram):
    if histogram is None:
        return None

    working_hist = histogram

    if isinstance(working_hist, HistEFT):
        if hasattr(working_hist, "as_hist"):
            working_hist = working_hist.as_hist({})
        else:
            evaluated = working_hist.eval({})
            if isinstance(evaluated, hist.Hist):
                working_hist = evaluated
            else:
                coeff_axis = getattr(working_hist, "_coeff_axis", None)
                base_axes = [axis for axis in working_hist.axes if axis is not coeff_axis]
                init_args = getattr(working_hist, "_init_args", {})
                dense_hist = hist.Hist(*base_axes, **init_args)
                sparse_names = working_hist.categorical_axes.name
                if isinstance(sparse_names, str):
                    sparse_names = (sparse_names,)
                for sp_val, arrs in evaluated.items():
                    if not isinstance(sp_val, tuple):
                        sp_val = (sp_val,)
                    dense_hist[dict(zip(sparse_names, sp_val))] = arrs
                working_hist = dense_hist

    return working_hist


def _extract_tau_counts(histogram, expected_bins, sumw2_input=None):
    """Evaluate tau yields and variances, keeping only the SM quadratic term."""

    working_hist = _ensure_dense_histogram(histogram)
    if working_hist is None:
        raise RuntimeError("A nominal histogram must be provided to extract tau counts.")

    axis_names = tuple(getattr(axis, "name", None) for axis in getattr(working_hist, "axes", ()))

    values = np.asarray(working_hist.values(flow=True), dtype=float)
    if values.size == 0:
        values = np.zeros(0, dtype=float)

    if "quadratic_term" in axis_names:
        values = _collapse_quadratic_axis(values, working_hist, axis_names)

    if sumw2_input is None:
        raise RuntimeError("A selected scalar sumw2 companion is required for tau fitting.")
    if isinstance(sumw2_input, (np.ndarray, list, tuple)):
        variances = np.asarray(sumw2_input, dtype=float)
    else:
        sumw2_hist = _ensure_dense_histogram(sumw2_input)
        if sumw2_hist is None:
            raise RuntimeError("The tau sumw2 companion could not be evaluated.")
        sumw2_axis_names = tuple(
            getattr(axis, "name", None) for axis in getattr(sumw2_hist, "axes", ())
        )
        variances = np.asarray(sumw2_hist.values(flow=True), dtype=float)
        if "quadratic_term" in sumw2_axis_names:
            variances = _collapse_quadratic_axis(
                variances,
                sumw2_hist,
                sumw2_axis_names,
            )

    values_1d = np.squeeze(values)
    if values_1d.ndim == 0:
        values_1d = values_1d.reshape(1)
    if values_1d.ndim != 1:
        raise RuntimeError(
            "Unexpected tau histogram shape; expected a 1D tau-pt spectrum, "
            f"received {values.shape}."
        )

    variances_1d = None
    if variances is not None:
        variances_1d = np.squeeze(variances)
        if variances_1d.ndim == 0:
            variances_1d = variances_1d.reshape(1)
        if variances_1d.ndim != 1:
            raise RuntimeError(
                "Unexpected tau variance shape; expected a 1D tau-pt spectrum, "
                f"received {variances.shape}."
            )

    if variances_1d is None:
        raise RuntimeError("The selected tau sumw2 companion has no variance values.")

    folded_values = _fold_tau_overflow(values_1d, expected_bins=expected_bins)
    folded_variances = _fold_tau_overflow(variances_1d, expected_bins=expected_bins)

    return folded_values, folded_variances


def _variance_to_errors(variances):
    variances = np.asarray(variances, dtype=float)
    return np.sqrt(np.clip(variances, 0.0, None))


def _collect_grouped_counts(hist, sumw2_hist, expected_bins, *, hist_label="grouped histogram"):
    """Collect tau counts and errors for each process in a grouped histogram."""

    if hist is None:
        raise RuntimeError(
            f"Missing {hist_label}; expected a grouped histogram with a 'process' axis."
        )
    if sumw2_hist is None:
        raise RuntimeError(f"Missing selected scalar companion for {hist_label}.")

    try:
        process_axis = hist.axes["process"]
    except (KeyError, AttributeError, TypeError):
        raise RuntimeError(
            f"The {hist_label} is not grouped by process; expected a histogram with a 'process' axis."
        )

    grouped_counts = {}
    for process in process_axis:
        proc_name = str(process)
        proc_hist = hist[{"process": process}]
        proc_sumw2_hist = sumw2_hist[{"process": process}]
        proc_vals, proc_vars = _extract_tau_counts(
            proc_hist,
            expected_bins,
            sumw2_input=proc_sumw2_hist,
        )
        grouped_counts[proc_name] = (proc_vals, _variance_to_errors(proc_vars))

    return grouped_counts


def _extract_grouped_tau_yields(
    fake_hist,
    tight_hist,
    expected_bins,
    *,
    fake_sumw2_hist=None,
    tight_sumw2_hist=None,
    sample_kind,
):
    """Collect tau yields and uncertainties for a grouped sample."""

    fake_counts = _collect_grouped_counts(
        fake_hist,
        fake_sumw2_hist,
        expected_bins,
        hist_label=f"{sample_kind} fake grouped histogram",
    )
    tight_counts = _collect_grouped_counts(
        tight_hist,
        tight_sumw2_hist,
        expected_bins,
        hist_label=f"{sample_kind} tight grouped histogram",
    )

    if fake_counts.keys() != tight_counts.keys():
        raise RuntimeError(
            f"Inconsistent {sample_kind} processes found between fake and tight histograms: "
            f"{sorted(fake_counts)} vs {sorted(tight_counts)}"
        )
    if len(fake_counts) != 1:
        raise RuntimeError(
            f"Expected a single {sample_kind} process after grouping; found "
            f"{sorted(fake_counts)}."
        )

    proc_name = next(iter(fake_counts))
    fake_vals, fake_err = fake_counts[proc_name]
    tight_vals, tight_err = tight_counts[proc_name]

    return fake_vals, fake_err, tight_vals, tight_err


def _format_bin_edge(edge):
    if math.isinf(edge):
        return "∞"
    if float(edge).is_integer():
        return f"{int(edge)}"
    return f"{edge:.2f}".rstrip("0").rstrip(".")


def _format_bin_label(edges, start_index, stop_index):
    left = _format_bin_edge(edges[start_index])
    right_edge = edges[stop_index] if stop_index < len(edges) else math.inf
    right = _format_bin_edge(right_edge)
    return f"[{left}, {right})"


def _print_section_header(title):
    print(f"\n{title}")
    print("=" * len(title))


def _print_year_filter_summary(summary):
    _print_section_header("Year selection summary")

    selected_years = summary.get("selected_years")
    if selected_years:
        print("Requested year tokens : " + ", ".join(selected_years))
    else:
        print("Requested year tokens : <all available>")

    retained_mc = summary.get("mc_samples") or []
    retained_data = summary.get("data_samples") or []
    removed_mc = summary.get("mc_removed") or []
    removed_data = summary.get("data_removed") or []

    # print("Retained MC processes   : " + (", ".join(retained_mc) if retained_mc else "<none>"))
    # print("Retained data processes : " + (", ".join(retained_data) if retained_data else "<none>"))
    # print("Removed MC processes    : " + (", ".join(removed_mc) if removed_mc else "<none>"))
    # print("Removed data processes  : " + (", ".join(removed_data) if removed_data else "<none>"))


def _print_yield_table(title, bin_labels, yields, errors):
    _print_section_header(title)
    header = f"{'Tau pT bin':<16}{'Yield':>14}{'Stat. err':>14}"
    print(header)
    print("-" * len(header))
    values = np.asarray(yields, dtype=float).reshape(-1)
    errs = np.asarray(errors, dtype=float).reshape(-1)

    if len(values) != len(bin_labels) or len(errs) != len(bin_labels):
        raise RuntimeError(
            "Yield table inputs have mismatched shapes: "
            f"{len(bin_labels)} bin labels, {len(values)} yields, {len(errs)} errors."
        )

    for label, value, err in zip(bin_labels, values, errs):
        print(f"{label:<16}{value:14.2f}{err:14.2f}")


def _print_regroup_summary(title, bin_labels, mc_summary, data_summary):
    _print_section_header(title)
    header = (
        f"{'Tau pT bin':<16}"
        f"{'MC fake sum':>14}{'MC fake err':>14}"
        f"{'MC tight sum':>15}{'MC tight err':>15}"
        f"{'Data fake sum':>16}{'Data fake err':>16}"
        f"{'Data tight sum':>17}{'Data tight err':>17}"
    )
    print(header)
    print("-" * len(header))
    for label, mc_info, data_info in zip(bin_labels, mc_summary, data_summary):
        print(
            f"{label:<16}"
            f"{mc_info['fake_sum']:14.2f}{mc_info['fake_err']:14.2f}"
            f"{mc_info['tight_sum']:15.2f}{mc_info['tight_err']:15.2f}"
            f"{data_info['fake_sum']:16.2f}{data_info['fake_err']:16.2f}"
            f"{data_info['tight_sum']:17.2f}{data_info['tight_err']:17.2f}"
        )


def _print_fake_rate_table(title, bin_labels, mc_rates, mc_errors, data_rates, data_errors):
    _print_section_header(title)
    header = f"{'Tau pT bin':<16}{'MC FR':>12}{'MC err':>12}{'Data FR':>12}{'Data err':>12}"
    print(header)
    print("-" * len(header))
    for label, mc_val, mc_err, data_val, data_err in zip(
        bin_labels, mc_rates, mc_errors, data_rates, data_errors
    ):
        print(
            f"{label:<16}"
            f"{mc_val:12.4f}{mc_err:12.4f}"
            f"{data_val:12.4f}{data_err:12.4f}"
        )


def _filter_and_validate_regroup_summaries(
    valid_mask,
    regroup_labels,
    tau_pt_edges,
    bin_starts,
    mc_summary,
    data_summary,
):
    """Filter regroup summaries with ``valid_mask`` and verify bin metadata consistency."""

    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    regroup_labels = np.asarray(regroup_labels, dtype=object).reshape(-1)

    if regroup_labels.size != valid_mask.size:
        raise RuntimeError(
            "Tau fake-rate regroup labels and validity mask have incompatible lengths."
        )

    bin_starts = np.asarray(bin_starts, dtype=float).reshape(-1)
    if bin_starts.size != valid_mask.size:
        raise RuntimeError(
            "Tau fake-rate bin starts do not match the number of regrouped bins."
        )

    tau_pt_edges = np.asarray(tau_pt_edges, dtype=float).reshape(-1)
    if tau_pt_edges.ndim != 1:
        raise RuntimeError("Tau fake-rate tau pT edges must be a one-dimensional array.")

    filtered_labels = regroup_labels[valid_mask]

    def _filter_summary(summary):
        if not summary:
            return None
        if len(summary) != valid_mask.size:
            raise RuntimeError(
                "Tau fake-rate regroup summary length does not match the number of bins."
            )
        return [dict(entry) for entry, keep in zip(summary, valid_mask) if keep]

    filtered_mc = _filter_summary(mc_summary)
    filtered_data = _filter_summary(data_summary)

    def _labels_from_summary(summary):
        if summary is None:
            return None
        return np.array(
            [_format_bin_label(tau_pt_edges, *entry["slice"]) for entry in summary],
            dtype=object,
        )

    mc_labels = _labels_from_summary(filtered_mc)
    data_labels = _labels_from_summary(filtered_data)

    if mc_labels is not None and not np.array_equal(filtered_labels, mc_labels):
        raise RuntimeError(
            "Filtered MC regroup summary is inconsistent with the regrouped tau pT labels."
        )

    if data_labels is not None and not np.array_equal(filtered_labels, data_labels):
        raise RuntimeError(
            "Filtered data regroup summary is inconsistent with the regrouped tau pT labels."
        )

    reference_summary = filtered_mc or filtered_data
    if reference_summary is not None:
        start_indices = np.array(
            [entry["slice"][0] for entry in reference_summary], dtype=int
        )
        expected_starts = tau_pt_edges[start_indices]
        actual_starts = bin_starts[valid_mask]
        if expected_starts.shape != actual_starts.shape or not np.allclose(
            expected_starts, actual_starts
        ):
            raise RuntimeError(
                "Tau pT edges derived from the regroup summary do not match the fit bin starts."
            )

    if filtered_mc is not None and filtered_data is not None:
        mc_slices = [entry["slice"] for entry in filtered_mc]
        data_slices = [entry["slice"] for entry in filtered_data]
        if mc_slices != data_slices:
            raise RuntimeError(
                "Filtered MC and data regroup summaries describe different tau pT slices."
            )

    return filtered_labels, filtered_mc, filtered_data


def _print_scale_factor_table(title, bin_labels, scale_factors, errors):
    _print_section_header(title)
    header = f"{'Tau pT bin':<16}{'Scale factor':>14}{'Uncertainty':>14}"
    print(header)
    print("-" * len(header))
    for label, value, err in zip(bin_labels, scale_factors, errors):
        print(f"{label:<16}{value:14.4f}{err:14.4f}")


def _print_fit_summary(
    c0,
    c1,
    parameter_errors,
    bin_labels,
    pt_reference,
    nominal_sf,
    sf_up,
    sf_down,
):
    _print_section_header("Scale-factor fit summary")
    param_header = f"{'Parameter':<12}{'Value':>14}{'Uncertainty':>14}"
    print(param_header)
    print("-" * len(param_header))
    print(f"{'c0':<12}{c0:14.6f}{parameter_errors[0]:14.6f}")
    print(f"{'c1':<12}{c1:14.6f}{parameter_errors[1]:14.6f}")

    print("\nPer-bin fitted values")
    print("---------------------")
    value_header = f"{'Tau pT bin':<16}{'pT ref [GeV]':>14}{'Fitted SF':>14}{'SF_up':>14}{'SF_down':>14}"
    print(value_header)
    print("-" * len(value_header))
    for label, pt, nom, up_val, down_val in zip(
        bin_labels, pt_reference, nominal_sf, sf_up, sf_down
    ):
        print(
            f"{label:<16}{pt:14.2f}{nom:14.6f}{up_val:14.6f}{down_val:14.6f}"
        )


def linear(x,a,b):
    return b*x+a

def SF_fit(SF,SF_e,x):

    params, cov = curve_fit(linear,x,SF,sigma=SF_e,absolute_sigma=True)
    return params[0],params[1], cov

def _ensure_list(values):
    if isinstance(values, str):
        return [values]
    return list(values)


def group_bins(histo, bin_map, axis_name="process", drop_unspecified=False):
    normalized_map = OrderedDict(
        (group, _ensure_list(categories))
        for group, categories in bin_map.items()
    )

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

TAU_PT_BIN_EDGES = [20, 30, 40, 50, 60, 200]


def _extract_tau_pt_edges(axis):
    """Return the raw tau-pt bin edges, parsing categorical axes when needed."""

    if axis.name not in {"tau0Fpt", "tau0Tpt"}:
        raise RuntimeError(
            "Tau pt regrouping requested for unexpected axis "
            f"'{axis.name}'. Expected one of: tau0Fpt, tau0Tpt."
        )

    try:
        native_edges = np.asarray(axis.edges, dtype=float)
    except AttributeError:
        # Some histograms encode the tau axis as strings (e.g. "[20, 30)"); in
        # that case parse the numeric boundaries manually to recover the edges.
        pattern = re.compile(r"[-+]?\d*\.?\d+")
        boundaries = []
        for category in axis:
            matches = pattern.findall(str(category))
            if len(matches) < 2:
                raise RuntimeError(
                    "Unable to parse tau pt bin edges from axis label "
                    f"'{category}'."
                )
            lower, upper = map(float, matches[:2])
            if not boundaries:
                boundaries.append(lower)
            elif not np.isclose(boundaries[-1], lower):
                raise RuntimeError(
                    "Tau pt axis labels are not contiguous: expected lower edge "
                    f"{boundaries[-1]} but found {lower}."
                )
            boundaries.append(upper)
        native_edges = np.asarray(boundaries, dtype=float)

    if native_edges.ndim != 1 or native_edges.size < 2:
        raise RuntimeError("Tau pt axis does not define a valid edge array.")

    if not np.all(np.diff(native_edges) > 0):
        raise RuntimeError("Tau pt axis edges must be strictly increasing.")

    return native_edges


def _resolve_tau_pt_bins(axis, target_edges):
    """Translate user-specified regrouping edges into histogram bin slices.

    The fitter expects regrouping requests to align with the native histogram
    boundaries.  This helper verifies that every requested edge coincides with
    an existing boundary and then stores the corresponding ``(start, stop)``
    indices.  These indices are later used to sum the fine-grained histogram
    bins into the coarser working points used for the fit.
    """

    if len(target_edges) < 2:
        raise RuntimeError("Tau pt regrouping requires at least two edge values.")

    native_edges = _extract_tau_pt_edges(axis)

    regroup_slices = []
    for left, right in zip(target_edges[:-1], target_edges[1:]):
        # ``np.where`` returns every index matching the requested boundary; we
        # keep the first match because repeated edges would already constitute
        # an invalid (zero-width) regrouping interval.
        left_idx = np.where(np.isclose(native_edges, left))[0]
        right_idx = np.where(np.isclose(native_edges, right))[0]
        if left_idx.size == 0:
            raise RuntimeError(
                "Requested tau pt regrouping edge {0} GeV does not match a native "
                "histogram boundary.".format(left)
            )
        if right_idx.size == 0:
            raise RuntimeError(
                "Requested tau pt regrouping edge {0} GeV does not match a native "
                "histogram boundary.".format(right)
            )

        start = int(left_idx[0])
        stop = int(right_idx[0])
        if stop <= start:
            raise RuntimeError(
                "Tau pt regrouping edges {0} and {1} do not form a valid bin.".format(
                    left, right
                )
            )
        regroup_slices.append((start, stop))

    return native_edges, regroup_slices


def _finalize_error(total):
    """Return the square-root error while guarding against small negatives."""

    return math.sqrt(total) if total > 0.0 else 0.0


def _combine_ratio_uncertainty(num, num_err, den, den_err):
    """Propagate errors for ``num/den`` using the usual uncorrelated formula."""

    if den == 0.0:
        return 0.0
    return math.sqrt((num_err / den) ** 2 + (num * den_err / (den ** 2)) ** 2)


def _combine_ratio_uncertainty_array(num, num_err, den, den_err):
    """Vectorised wrapper around :func:`_combine_ratio_uncertainty`."""

    num = np.asarray(num, dtype=float)
    num_err = np.asarray(num_err, dtype=float)
    den = np.asarray(den, dtype=float)
    den_err = np.asarray(den_err, dtype=float)

    result = np.empty_like(num, dtype=float)
    result.fill(np.inf)
    valid = den != 0.0
    if not np.any(valid):
        return result

    result[valid] = np.sqrt(
        (num_err[valid] / den[valid]) ** 2
        + (num[valid] * den_err[valid] / (den[valid] ** 2)) ** 2
    )
    return result


def compute_fake_rates(
    fake_vals,
    fake_errs,
    tight_vals,
    tight_errs,
    regroup_slices=None,
):
    """Compute fake rates and their uncertainties after optional regrouping.

    Parameters
    ----------
    fake_vals, tight_vals : array-like
        Arrays containing the fake (loose) and tight yields for each tau-pT bin.
    fake_errs, tight_errs : array-like
        Quadrature-combined uncertainties matching ``fake_vals`` and
        ``tight_vals``.
    regroup_slices : iterable of ``(start, stop)`` pairs, optional
        Pre-computed regrouping slices describing how fine histogram bins should
        be merged.  When omitted the function assumes the arrays are already
        aligned with the working binning and folds overflow away.

    Returns
    -------
    tuple
        ``(rates, errors, summary)`` with per-bin fake rates, propagated
        uncertainties, and book-keeping information about the summed counts.
    """

    fake_vals = np.asarray(fake_vals, dtype=float)
    fake_errs = np.asarray(fake_errs, dtype=float)
    tight_vals = np.asarray(tight_vals, dtype=float)
    tight_errs = np.asarray(tight_errs, dtype=float)

    if regroup_slices is None:
        n_bins = fake_vals.shape[-1]
        expected_physical_bins = len(TAU_PT_BIN_EDGES) - 1

        # Histograms produced by coffea often ship the under/overflow bins.
        # Detect this case and skip the flow entries so we can operate on the
        # physical tau-pt intervals only.
        has_flow_bins = n_bins == expected_physical_bins + 2
        if has_flow_bins:
            first_index = 1
            stop_index = min(n_bins - 1, first_index + expected_physical_bins)
        else:
            first_index = 0
            stop_index = min(n_bins, first_index + expected_physical_bins)

        # Build one-slice-per-bin regrouping instructions so downstream logic
        # can keep a uniform code path regardless of whether explicit regrouping
        # instructions were passed.
        regroup_slices = [
            (index, index + 1)
            for index in range(first_index, stop_index)
        ]

    n_groups = len(regroup_slices)
    ratios = np.empty(n_groups, dtype=float)
    errors = np.empty(n_groups, dtype=float)
    summary = []

    for index, (start, stop) in enumerate(regroup_slices):
        fake_slice = slice(start, stop)
        # Sum yields inside the requested regrouping window.  ``np.sum`` already
        # copes with single-bin windows so no special handling is needed here.
        fake_sum = float(np.sum(fake_vals[fake_slice]))
        tight_sum = float(np.sum(tight_vals[fake_slice]))

        # Add statistical and systematic errors in quadrature, mirroring the
        # logic used when the fit inputs are prepared.
        fake_err_total = _finalize_error(
            np.sum(fake_errs[fake_slice] ** 2, dtype=float)
        )
        tight_err_total = _finalize_error(
            np.sum(tight_errs[fake_slice] ** 2, dtype=float)
        )

        if fake_sum != 0.0:
            ratio = tight_sum / fake_sum
            # Tight and fake selections are assumed uncorrelated, so we can use
            # the standard error propagation formula for a ratio of two yields.
            ratio_err = _combine_ratio_uncertainty(
                tight_sum, tight_err_total, fake_sum, fake_err_total
            )
        else:
            ratio = 0.0
            ratio_err = 0.0

        ratios[index] = ratio
        if ratio > 0.0:
            # Preserve the historical minimum ~2% relative uncertainty by
            # picking the larger of the propagated error and the enforced
            # floor.
            errors[index] = max(ratio_err, 0.02 * ratio)
        else:
            errors[index] = ratio_err

        summary.append(
            {
                "slice": (start, stop),
                "fake_sum": fake_sum,
                "fake_err": fake_err_total,
                "tight_sum": tight_sum,
                "tight_err": tight_err_total,
            }
        )

    return (
        ratios,
        errors,
        summary,
    )


def _validate_histogram_axes(histogram, expected_axes, hist_name):
    """Ensure the histogram contains the expected axes for the tau fake-rate workflow."""

    present_axes, recognised_axes, canonical_axes = _resolve_histogram_axis_names(histogram)

    if recognised_axes and not canonical_axes:
        canonical_axes.update(recognised_axes)

    canonical_expected = set()
    for axis in expected_axes:
        if isinstance(axis, str):
            canonical_expected.add(axis)
            if axis.endswith("_sumw2") and len(axis) > 6:
                canonical_expected.add(axis[:-6])
        else:
            canonical_expected.add(axis)

    missing_axes = [axis for axis in canonical_expected if axis not in canonical_axes]

    if missing_axes:
        available = ", ".join(sorted(present_axes)) if present_axes else "<none>"
        summary = (
            f"The '{hist_name}' histogram is missing required axes: {', '.join(missing_axes)}. "
            f"Available axes: {available}. "
            "Regenerate the histogram pickle with these axes enabled before running the fake-tau SF fitter."
        )
        LOGGER.error(summary)
        raise HistogramAxisError(
            summary,
            missing_axes=missing_axes,
            present_axes=sorted(present_axes),
        )

    return set(canonical_axes)


@dataclass(frozen=True)
class TauChannelResolution:
    selected_bins: tuple
    resolution_mode: str
    missing_flavor_split_bins: tuple
    aggregate_bins_checked: tuple
    available_aggregate_bins: tuple
    missing_aggregate_bins: tuple

    def as_dict(self):
        return {
            "selected_bins": self.selected_bins,
            "resolution_mode": self.resolution_mode,
            "missing_flavor_split_bins": self.missing_flavor_split_bins,
            "aggregate_bins_checked": self.aggregate_bins_checked,
            "available_aggregate_bins": self.available_aggregate_bins,
            "missing_aggregate_bins": self.missing_aggregate_bins,
        }


def _format_tau_resolution_bins(bins):
    return ", ".join(bins) if bins else "<none>"


def _tau_channel_resolution_attempt(allow_aggregate_fallback):
    if allow_aggregate_fallback:
        return "complete flavor-split bins, then aggregate fallback"
    return "complete flavor-split bins (aggregate fallback disabled)"


def format_tau_channel_resolution_error(
    hist_name,
    tau_resolutions,
    available_channels,
    *,
    allow_aggregate_fallback=True,
):
    summary_lines = [
        f"The '{hist_name}' histogram is missing required tau control-region categories.",
        "Missing bins summary:",
    ]
    attempted = _tau_channel_resolution_attempt(allow_aggregate_fallback)
    for tau_family, resolution in tau_resolutions.items():
        summary_lines.extend(
            [
                f"  {tau_family}:",
                f"    resolution attempted: {attempted}",
                "    missing flavor-split bins: "
                + _format_tau_resolution_bins(resolution.missing_flavor_split_bins),
                "    aggregate fallback bins checked: "
                + _format_tau_resolution_bins(resolution.aggregate_bins_checked),
                "    missing aggregate fallback bins: "
                + _format_tau_resolution_bins(resolution.missing_aggregate_bins),
                "    aggregate fallback available: "
                + ("yes" if resolution.available_aggregate_bins else "no"),
            ]
        )

    missing_aggregate_bins = sorted(
        {
            channel
            for resolution in tau_resolutions.values()
            for channel in resolution.missing_aggregate_bins
        }
    )
    summary_lines.extend(
        [
            "Available channel bins: "
            + _format_tau_resolution_bins(sorted(available_channels)),
            "Either regenerate the histogram pickle with complete flavor-split tau bins "
            "or include the aggregate fallback bin(s): "
            + _format_tau_resolution_bins(missing_aggregate_bins)
            + ".",
        ]
    )
    return "\n".join(summary_lines)


def _validate_tau_channel_coverage(
    histogram,
    channel_axis_name,
    ftau_channels,
    ttau_channels,
    hist_name,
    *,
    allow_aggregate_fallback=True,
):
    """Resolve configured tau CR bins against one histogram's channel axis.

    Complete flavor-split bins take priority.  When any configured split bin for
    a tau/jet family is absent, the corresponding aggregate bin is used if it is
    available.  A family never selects split and aggregate bins together.
    """

    channel_axis = None
    for axis in histogram.axes:
        axis_name = getattr(axis, "name", None)
        if not axis_name:
            continue
        if axis_name == channel_axis_name:
            channel_axis = axis
            break
        if axis_name.endswith("_sumw2") and axis_name[:-6] == channel_axis_name:
            channel_axis = axis
            break

    if channel_axis is None:
        raise RuntimeError(
            f"Histogram '{hist_name}' does not define a '{channel_axis_name}' axis."
        )

    available_channels = {str(category) for category in channel_axis}

    resolutions = {}
    for tau_family, configured_channels in (
        ("Ftau", ftau_channels),
        ("Ttau", ttau_channels),
    ):
        if not configured_channels:
            continue
        resolutions[tau_family] = resolve_tau_cr_channel_bins(
            available_channels,
            configured_channels,
            tau_family=tau_family,
            hist_name=hist_name,
            allow_aggregate_fallback=allow_aggregate_fallback,
            raise_on_missing=False,
        )

    unresolved = {
        family: resolution
        for family, resolution in resolutions.items()
        if resolution.missing_aggregate_bins
    }
    if unresolved:
        summary = format_tau_channel_resolution_error(
            hist_name,
            unresolved,
            available_channels,
            allow_aggregate_fallback=allow_aggregate_fallback,
        )
        LOGGER.error(summary)
        raise RuntimeError(summary)

    return resolutions


def _aggregate_tau_channel_name(channel):
    """Return the aggregate name corresponding to an ee/em/mm tau CR bin."""

    for flavor in ("ee", "em", "mm"):
        marker = f"_{flavor}_"
        if marker in channel:
            return channel.replace(marker, "_", 1)
    return channel


def resolve_tau_cr_channel_bins(
    available_bins,
    configured_bins,
    *,
    tau_family,
    hist_name,
    allow_aggregate_fallback=True,
    raise_on_missing=True,
):
    """Select complete split bins or one aggregate fallback per tau/jet family."""

    available = {str(channel) for channel in available_bins}
    grouped_channels = OrderedDict()
    for configured_channel in configured_bins:
        channel = str(configured_channel)
        aggregate_channel = _aggregate_tau_channel_name(channel)
        group = grouped_channels.setdefault(aggregate_channel, [])
        if channel != aggregate_channel and channel not in group:
            group.append(channel)

    selected_bins = []
    group_modes = []
    missing_flavor_split_bins = []
    aggregate_bins_checked = []
    available_aggregate_bins = []
    missing_aggregate_bins = []

    for aggregate_channel, split_channels in grouped_channels.items():
        missing_split = [
            channel for channel in split_channels if channel not in available
        ]
        missing_flavor_split_bins.extend(missing_split)
        aggregate_bins_checked.append(aggregate_channel)

        if split_channels and not missing_split:
            selected_bins.extend(split_channels)
            group_modes.append("flavor_split")
            continue

        aggregate_available = aggregate_channel in available
        if aggregate_available:
            available_aggregate_bins.append(aggregate_channel)

        if aggregate_available and (allow_aggregate_fallback or not split_channels):
            selected_bins.append(aggregate_channel)
            group_modes.append("aggregate")
            continue

        missing_aggregate_bins.append(aggregate_channel)

    if not group_modes:
        resolution_mode = "unresolved"
    elif len(set(group_modes)) == 1:
        resolution_mode = group_modes[0]
    else:
        resolution_mode = "mixed"

    resolution = TauChannelResolution(
        selected_bins=tuple(selected_bins),
        resolution_mode=resolution_mode,
        missing_flavor_split_bins=tuple(missing_flavor_split_bins),
        aggregate_bins_checked=tuple(aggregate_bins_checked),
        available_aggregate_bins=tuple(available_aggregate_bins),
        missing_aggregate_bins=tuple(missing_aggregate_bins),
    )

    if raise_on_missing and missing_aggregate_bins:
        message = format_tau_channel_resolution_error(
            hist_name,
            {tau_family: resolution},
            available,
            allow_aggregate_fallback=allow_aggregate_fallback,
        )
        raise RuntimeError(message)

    return resolution


def getPoints(dict_of_hists, ftau_channels, ttau_channels, *, sample_filters=None):
    if sample_filters is None:
        sample_filters = _resolve_year_filters(None)

    mc_wl = list(sample_filters.get("mc_whitelist", ()))
    mc_bl = list(sample_filters.get("mc_blacklist", ()))
    data_wl = list(sample_filters.get("data_whitelist", ()))
    data_bl = list(sample_filters.get("data_blacklist", ()))

    all_samples = yt.get_cat_lables(dict_of_hists, "process")
    mc_sample_lst = _filter_samples(all_samples, mc_wl, mc_bl)
    data_sample_lst = _filter_samples(all_samples, data_wl, data_bl)

    mc_keep = set(mc_sample_lst)
    data_keep = set(data_sample_lst)

    samples_to_rm_from_mc_hist = [
        sample_name for sample_name in all_samples if sample_name not in mc_keep
    ]
    samples_to_rm_from_data_hist = [
        sample_name for sample_name in all_samples if sample_name not in data_keep
    ]

    fake_key = "tau0Fpt"
    tight_key = "tau0Tpt"

    available_keys = sorted(dict_of_hists)
    available_desc = ", ".join(available_keys) if available_keys else "<none>"
    for required_key in (
        fake_key,
        tight_key,
        f"{fake_key}_sumw2",
        f"{tight_key}_sumw2",
    ):
        if required_key not in dict_of_hists:
            raise RuntimeError(
                f"The histogram pickle is missing the required '{required_key}' histogram. "
                f"Available histograms: {available_desc}."
            )

    tau_fake_hist = dict_of_hists[fake_key]
    tau_tight_hist = dict_of_hists[tight_key]
    tau_fake_sumw2_hist = dict_of_hists[f"{fake_key}_sumw2"]
    tau_tight_sumw2_hist = dict_of_hists[f"{tight_key}_sumw2"]

    tau_fake_canonical_axes = _validate_histogram_axes(
        tau_fake_hist,
        _TAU_FAKE_HISTOGRAM_REQUIRED_AXES,
        fake_key,
    )
    fake_channel_resolutions = _validate_tau_channel_coverage(
        tau_fake_hist,
        "channel",
        ftau_channels,
        (),
        fake_key,
    )

    tau_tight_canonical_axes = _validate_histogram_axes(
        tau_tight_hist,
        _TAU_TIGHT_HISTOGRAM_REQUIRED_AXES,
        tight_key,
    )
    tight_channel_resolutions = _validate_tau_channel_coverage(
        tau_tight_hist,
        "channel",
        (),
        ttau_channels,
        tight_key,
    )

    fake_channel_resolution = fake_channel_resolutions["Ftau"]
    tight_channel_resolution = tight_channel_resolutions["Ttau"]
    resolved_ftau_channels = list(fake_channel_resolution.selected_bins)
    resolved_ttau_channels = list(tight_channel_resolution.selected_bins)
    LOGGER.info(
        "Resolved tau CR channels: Ftau=%s %s; Ttau=%s %s",
        fake_channel_resolution.resolution_mode,
        resolved_ftau_channels,
        tight_channel_resolution.resolution_mode,
        resolved_ttau_channels,
    )
    for tau_family, resolution in (
        ("Ftau", fake_channel_resolution),
        ("Ttau", tight_channel_resolution),
    ):
        if resolution.resolution_mode == "aggregate":
            LOGGER.info(
                "%s flavor-split bins are incomplete (%s); using aggregate fallback %s.",
                tau_family,
                ", ".join(resolution.missing_flavor_split_bins),
                ", ".join(resolution.selected_bins),
            )

    _validate_histogram_axes(
        tau_fake_sumw2_hist,
        tau_fake_canonical_axes,
        f"{fake_key}_sumw2",
    )
    _validate_tau_channel_coverage(
        tau_fake_sumw2_hist,
        "channel",
        resolved_ftau_channels,
        (),
        f"{fake_key}_sumw2",
        allow_aggregate_fallback=False,
    )

    _validate_histogram_axes(
        tau_tight_sumw2_hist,
        tau_tight_canonical_axes,
        f"{tight_key}_sumw2",
    )
    _validate_tau_channel_coverage(
        tau_tight_sumw2_hist,
        "channel",
        (),
        resolved_ttau_channels,
        f"{tight_key}_sumw2",
        allow_aggregate_fallback=False,
    )

    # Remove any processes that should not contribute to the MC or data histograms.
    # When no removals are needed, reuse the existing histogram instead of cloning it.
    hist_mc_fake = _maybe_remove_processes(
        tau_fake_hist, "process", samples_to_rm_from_mc_hist
    )
    hist_data_fake = _maybe_remove_processes(
        tau_fake_hist, "process", samples_to_rm_from_data_hist
    )
    hist_mc_tight = _maybe_remove_processes(
        tau_tight_hist, "process", samples_to_rm_from_mc_hist
    )
    hist_data_tight = _maybe_remove_processes(
        tau_tight_hist, "process", samples_to_rm_from_data_hist
    )

    hist_mc_fake_sumw2 = _maybe_remove_processes(
        tau_fake_sumw2_hist,
        "process",
        samples_to_rm_from_mc_hist,
    )
    hist_data_fake_sumw2 = _maybe_remove_processes(
        tau_fake_sumw2_hist,
        "process",
        samples_to_rm_from_data_hist,
    )

    hist_mc_tight_sumw2 = _maybe_remove_processes(
        tau_tight_sumw2_hist,
        "process",
        samples_to_rm_from_mc_hist,
    )
    hist_data_tight_sumw2 = _maybe_remove_processes(
        tau_tight_sumw2_hist,
        "process",
        samples_to_rm_from_data_hist,
    )

    # Integrate to get the categories we want
    mc_fake = _integrate_tau_channels(hist_mc_fake, resolved_ftau_channels)
    mc_tight = _integrate_tau_channels(hist_mc_tight, resolved_ttau_channels)
    data_fake = _integrate_tau_channels(hist_data_fake, resolved_ftau_channels)
    data_tight = _integrate_tau_channels(hist_data_tight, resolved_ttau_channels)

    mc_fake_sumw2 = _integrate_tau_channels(
        hist_mc_fake_sumw2, resolved_ftau_channels
    )
    mc_tight_sumw2 = _integrate_tau_channels(
        hist_mc_tight_sumw2, resolved_ttau_channels
    )

    data_fake_sumw2 = _integrate_tau_channels(
        hist_data_fake_sumw2, resolved_ftau_channels
    )
    data_tight_sumw2 = _integrate_tau_channels(
        hist_data_tight_sumw2, resolved_ttau_channels
    )

    # Build fresh grouping maps derived from the current histogram contents so we only
    # request bins that are still present after the Ftau/Ttau integrations.  This keeps the
    # MC map free of any ``data{year}`` entries while the data map only keeps those
    # ``data{year}`` labels.
    mc_processes = _collect_processes(
        (mc_fake, mc_tight),
        lambda name: not name.startswith("data"),
    )
    data_processes = _collect_processes(
        (data_fake, data_tight),
        lambda name: name.startswith("data"),
    )

    if not mc_processes:
        raise RuntimeError(
            "No MC processes remain after applying sample filters. "
            f"Filtered MC samples: {sorted(mc_sample_lst)!r}"
        )

    if not data_processes:
        raise RuntimeError(
            "No data processes remain after applying sample filters. "
            f"Filtered data samples: {sorted(data_sample_lst)!r}"
        )

    mc_group_map = OrderedDict((("Ttbar", mc_processes),))
    data_group_map = OrderedDict((("Data", data_processes),))

    mc_fake, mc_tight = _group_all((mc_fake, mc_tight), mc_group_map)
    data_fake, data_tight = _group_all((data_fake, data_tight), data_group_map)

    (
        mc_fake_sumw2,
        mc_tight_sumw2,
    ) = _group_all((mc_fake_sumw2, mc_tight_sumw2), mc_group_map)
    (
        data_fake_sumw2,
        data_tight_sumw2,
    ) = _group_all((data_fake_sumw2, data_tight_sumw2), data_group_map)

    (
        mc_fake,
        mc_tight,
        data_fake,
        data_tight,
    ) = tuple(
        _integrate_nominal(hist)
        for hist in (mc_fake, mc_tight, data_fake, data_tight)
    )

    (
        mc_fake_sumw2,
        mc_tight_sumw2,
        data_fake_sumw2,
        data_tight_sumw2,
    ) = tuple(
        _integrate_nominal(hist)
        for hist in (
            mc_fake_sumw2,
            mc_tight_sumw2,
            data_fake_sumw2,
            data_tight_sumw2,
        )
    )

    tau_pt_edges, regroup_slices = _resolve_tau_pt_bins(
        mc_fake.axes["tau0Fpt"],
        TAU_PT_BIN_EDGES,
    )
    start_indices = np.fromiter(
        (start for start, _ in regroup_slices), dtype=int
    )
    pt_bin_starts = tau_pt_edges[start_indices].astype(float, copy=False)
    expected_bins = len(tau_pt_edges) - 1

    mc_tight_tau_pt_edges = _extract_tau_pt_edges(mc_tight.axes["tau0Tpt"])
    if not np.allclose(tau_pt_edges, mc_tight_tau_pt_edges):
        raise RuntimeError(
            "MC fake and MC tight tau pt axes define different native edges."
        )

    data_fake_tau_pt_edges = _extract_tau_pt_edges(data_fake.axes["tau0Fpt"])
    if not np.allclose(tau_pt_edges, data_fake_tau_pt_edges):
        raise RuntimeError(
            "MC fake and data fake tau pt axes define different native edges."
        )

    data_tight_tau_pt_edges = _extract_tau_pt_edges(data_tight.axes["tau0Tpt"])
    if not np.allclose(tau_pt_edges, data_tight_tau_pt_edges):
        raise RuntimeError(
            "MC fake and data tight tau pt axes define different native edges."
        )

    mc_fake_vals, mc_fake_e, mc_tight_vals, mc_tight_e = _extract_grouped_tau_yields(
        mc_fake,
        mc_tight,
        expected_bins,
        fake_sumw2_hist=mc_fake_sumw2,
        tight_sumw2_hist=mc_tight_sumw2,
        sample_kind="MC",
    )

    (
        data_fake_vals,
        data_fake_e,
        data_tight_vals,
        data_tight_e,
    ) = _extract_grouped_tau_yields(
        data_fake,
        data_tight,
        expected_bins,
        fake_sumw2_hist=data_fake_sumw2,
        tight_sumw2_hist=data_tight_sumw2,
        sample_kind="data",
    )

    mc_y, mc_e, mc_regroup_summary = compute_fake_rates(
        mc_fake_vals,
        mc_fake_e,
        mc_tight_vals,
        mc_tight_e,
        regroup_slices,
    )
    data_y, data_e, data_regroup_summary = compute_fake_rates(
        data_fake_vals,
        data_fake_e,
        data_tight_vals,
        data_tight_e,
        regroup_slices,
    )

    data_x = pt_bin_starts.copy()

    native_bin_labels = [
        _format_bin_label(tau_pt_edges, index, index + 1)
        for index in range(len(tau_pt_edges) - 1)
    ]
    regroup_labels = [
        _format_bin_label(tau_pt_edges, start, stop)
        for start, stop in (entry["slice"] for entry in mc_regroup_summary)
    ]

    year_filter_summary = {
        "selected_years": sample_filters.get("selected_years"),
        "mc_samples": sorted(mc_sample_lst),
        "data_samples": sorted(data_sample_lst),
        "mc_removed": sorted(samples_to_rm_from_mc_hist),
        "data_removed": sorted(samples_to_rm_from_data_hist),
    }

    stage_details = {
        "native_bin_labels": native_bin_labels,
        "native_yields": {
            "MC fake": (mc_fake_vals, mc_fake_e),
            "MC tight": (mc_tight_vals, mc_tight_e),
            "Data fake": (data_fake_vals, data_fake_e),
            "Data tight": (data_tight_vals, data_tight_e),
        },
        "regroup_labels": regroup_labels,
        "mc_regroup_summary": mc_regroup_summary,
        "data_regroup_summary": data_regroup_summary,
        "year_filter": year_filter_summary,
        "tau_pt_bin_starts": tuple(pt_bin_starts.tolist()),
        "tau_pt_edges": tuple(tau_pt_edges.tolist()),
        "tau_channel_resolution": {
            "Ftau": fake_channel_resolution.as_dict(),
            "Ttau": tight_channel_resolution.as_dict(),
        },
    }

    return mc_y, mc_e, data_x, data_y, data_e, stage_details

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--pkl-file-path",
        nargs="+",
        default=[DEFAULT_INPUT_PKL_PATH],
        metavar="PKL",
        help=(
            "Path(s) to input pkl file(s). Pass one path for the legacy single-input "
            "case or multiple paths after one -f to combine inputs before fitting."
        ),
    )
    parser.add_argument(
        "--channels-json",
        default=None,
        help=(
            "Optional path to a channel configuration JSON file."
            " Defaults to topeft/channels/ch_lst.json."
        ),
    )
    parser.add_argument(
        "-y",
        "--year",
        nargs="+",
        default=None,
        help="One or more year tokens to include (e.g. 2017 2018).",
    )
    parser.add_argument(
        "--dump-channels",
        nargs="?",
        const="-",
        metavar="OUTPUT",
        help=(
            "Dump the resolved Ftau/Ttau channel lists to stdout or the specified file."
            " The script continues after dumping."
        ),
    )
    parser.add_argument(
        "--output-json",
        metavar="OUTPUT",
        help=(
            "Optional path to write the fitted tau scale factors as JSON."
            " Use '-' to write to stdout."
        ),
    )
    return parser


def main():
    def _as_flat_float(array):
        return np.asarray(array, dtype=float).reshape(-1)

    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        input_pkl_paths = normalize_input_pkl_paths(args.pkl_file_path)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        sample_filters = _resolve_year_filters(args.year)
    except ValueError as exc:
        parser.error(str(exc))

    ftau_channels, ttau_channels = load_tau_control_channels(args.channels_json)

    if args.dump_channels is not None:
        dump_payload = {"Ftau": ftau_channels, "Ttau": ttau_channels}
        if args.dump_channels in ("-", ""):
            json.dump(dump_payload, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            with open(args.dump_channels, "w") as dump_file:
                json.dump(dump_payload, dump_file, indent=2)
            print(f"Tau channel lists written to {args.dump_channels}")

    hin_dict, input_summary = combine_faketau_histogram_pkls(input_pkl_paths)
    print_faketau_input_summary(input_summary)
    y_mc, yerr_mc, x_data, y_data, yerr_data, stage_details = getPoints(
        hin_dict,
        ftau_channels,
        ttau_channels,
        sample_filters=sample_filters,
    )

    y_mc = _as_flat_float(y_mc)
    yerr_mc = _as_flat_float(yerr_mc)
    y_data = _as_flat_float(y_data)
    yerr_data = _as_flat_float(yerr_data)
    x_data = _as_flat_float(x_data)
    full_x_data = x_data

    year_filter_summary = stage_details.get("year_filter")
    if year_filter_summary:
        _print_year_filter_summary(year_filter_summary)

    for label, (values, errors) in stage_details["native_yields"].items():
        _print_yield_table(
            f"{label} yields (native bins)",
            stage_details["native_bin_labels"],
            values,
            errors,
        )

    _print_regroup_summary(
        "Regrouped fake-rate inputs",
        stage_details["regroup_labels"],
        stage_details["mc_regroup_summary"],
        stage_details["data_regroup_summary"],
    )

    _print_fake_rate_table(
        "Fake rates by tau pT bin",
        stage_details["regroup_labels"],
        y_mc,
        yerr_mc,
        y_data,
        yerr_data,
    )

    regroup_labels = np.array(stage_details["regroup_labels"], dtype=object)

    SF = np.divide(
        y_data,
        y_mc,
        out=np.full_like(y_data, np.nan, dtype=float),
        where=(y_mc != 0),
    )
    SF_e = _combine_ratio_uncertainty_array(
        y_data,
        yerr_data,
        y_mc,
        yerr_mc,
    )

    SF_e = np.where(SF_e <= 0, 1e-3, SF_e)

    valid = (
        np.isfinite(SF)
        & np.isfinite(SF_e)
        & (SF_e > 0)
        & np.isfinite(full_x_data)
        & np.isfinite(y_mc)
        & (y_mc > 0)
        & np.isfinite(y_data)
    )

    if not np.all(valid):
        dropped_bins = full_x_data[~valid]
        if dropped_bins.size:
            LOGGER.warning(
                "Dropping %d tau pT bin(s) from fit due to invalid scale factors: %s",
                dropped_bins.size,
                ", ".join(str(bin_edge) for bin_edge in dropped_bins),
            )

    tau_pt_edges = stage_details.get("tau_pt_edges")
    if tau_pt_edges is None:
        tau_pt_edges = np.asarray(TAU_PT_BIN_EDGES, dtype=float)
    else:
        tau_pt_edges = np.asarray(tau_pt_edges, dtype=float)

    (
        filtered_regroup_labels,
        filtered_mc_summary,
        filtered_data_summary,
    ) = _filter_and_validate_regroup_summaries(
        valid,
        regroup_labels,
        tau_pt_edges,
        full_x_data,
        stage_details.get("mc_regroup_summary"),
        stage_details.get("data_regroup_summary"),
    )

    stage_details["filtered_regroup_labels"] = tuple(
        filtered_regroup_labels.tolist()
    )
    if filtered_mc_summary is not None:
        stage_details["filtered_mc_regroup_summary"] = tuple(filtered_mc_summary)
    if filtered_data_summary is not None:
        stage_details["filtered_data_regroup_summary"] = tuple(filtered_data_summary)

    pt_bin_starts = stage_details.get("tau_pt_bin_starts")
    if pt_bin_starts is None:
        pt_bin_starts = np.asarray(tau_pt_edges[:-1], dtype=float)
    else:
        pt_bin_starts = np.asarray(pt_bin_starts, dtype=float)

    report_pt_values = full_x_data.copy()
    if report_pt_values.size:
        report_pt_values = report_pt_values[valid]
    else:
        report_pt_values = pt_bin_starts.copy()

    SF = SF[valid]
    SF_e = SF_e[valid]
    x_data = x_data[valid]
    y_data = y_data[valid]
    y_mc = y_mc[valid]
    yerr_data = yerr_data[valid]
    yerr_mc = yerr_mc[valid]
    regroup_labels = filtered_regroup_labels

    if SF.size < 2:
        raise RuntimeError(
            "Insufficient valid tau fake-rate points for fitting: "
            f"only {SF.size} bin(s) remain after filtering."
        )

    _print_scale_factor_table(
        "Scale factors (data/MC)",
        regroup_labels,
        SF,
        SF_e,
    )

    c0, c1, cov = SF_fit(SF, SF_e, x_data)

    eigenvalues, eigenvectors = np.linalg.eig(cov)
    lv0 = np.sqrt(abs(eigenvalues.dot(eigenvectors[0])))
    lv1 = np.sqrt(abs(eigenvalues.dot(eigenvectors[1])))
    perr = np.sqrt(np.diag(cov))

    if report_pt_values.size == 0:
        report_pt_values = x_data

    nominal_sf = c1 * report_pt_values + c0
    sf_up = (1 + lv0) * c0 + (1 + lv1) * c1 * report_pt_values
    sf_down = (1 - lv0) * c0 + (1 - lv1) * c1 * report_pt_values

    _print_fit_summary(
        c0,
        c1,
        perr,
        regroup_labels,
        report_pt_values,
        nominal_sf,
        sf_up,
        sf_down,
    )

    if args.output_json is not None:
        regroup_summary = stage_details.get("filtered_mc_regroup_summary")
        if regroup_summary is None:
            regroup_summary = stage_details.get("filtered_data_regroup_summary")
        if regroup_summary is None:
            regroup_summary = (
                stage_details.get("mc_regroup_summary")
                or stage_details.get("data_regroup_summary")
            )
        if regroup_summary is None:
            raise RuntimeError(
                "Unable to build tau scale-factor JSON payload: regroup summary unavailable."
            )

        def _format_json_edge(edge):
            edge = float(edge)
            if math.isinf(edge):
                return "inf"
            return format(edge, ".15g")

        if len(regroup_summary) != len(nominal_sf):
            raise RuntimeError(
                "Tau scale-factor binning metadata does not match the fitted results."
            )

        tau_sf_entries = OrderedDict()
        for entry, value, up_val, down_val in zip(
            regroup_summary,
            nominal_sf,
            sf_up,
            sf_down,
        ):
            start, stop = entry["slice"]
            low_edge = float(tau_pt_edges[start])
            high_edge = float(tau_pt_edges[stop])
            bin_key = f"pt:[{_format_json_edge(low_edge)},{_format_json_edge(high_edge)}]"
            tau_sf_entries[bin_key] = {
                "value": float(value),
                "up": float(up_val),
                "down": float(down_val),
            }

        payload = {"TauSF": {"pt": tau_sf_entries}}

        if args.output_json in ("-", ""):
            json.dump(payload, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            with open(args.output_json, "w", encoding="utf-8") as output_file:
                json.dump(payload, output_file, indent=2)
            print(f"Tau scale factors JSON written to {args.output_json}")

if __name__ == "__main__":
    main()
