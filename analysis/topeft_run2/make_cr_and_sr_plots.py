import numpy as np
import os
import copy
import csv
import datetime
import argparse
import json
import gzip
import pickle
import re
from collections import OrderedDict
from collections.abc import Mapping

import logging
from decimal import Decimal
import inspect
import math
import warnings
import itertools
import multiprocessing
from functools import lru_cache
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
import topcoffea.modules.histEFT as tc_histEFT
import topcoffea.modules.sparseHist as tc_sparseHist
from topeft.modules.axes import info as te_axes_info
from topeft.modules.axes import info_2d as te_axes_info_2d

from topcoffea.scripts.make_html import make_html as tc_make_html
import topcoffea.modules.utils as tc_utils
from topeft.modules.yield_tools import YieldTools as te_YieldTools
from topeft.modules.get_rate_systs import (
    get_correlation_tag as te_get_correlation_tag,
    get_jet_dependent_syst_dict as te_get_jet_dependent_syst_dict,
    get_syst as te_get_syst,
    get_syst_lst as te_get_syst_lst,
)
from topeft.modules.datacard_tools import load_and_merge_histogram_pkls


_logger = logging.getLogger(__name__)
_ORIGINAL_SPARSEHIST_READ_FROM_REDUCE = tc_sparseHist.SparseHist._read_from_reduce.__func__
_VALUES_METHOD_CAPS = {}
_SYSTEMATICS_SUMMARY_EMITTED = set()


def _fast_sparsehist_from_reduce(cls, cat_axes, dense_axes, init_args, dense_hists):
    """Fast reconstruction helper used to patch :class:`SparseHist` pickles."""

    try:
        histogram = cls(*cat_axes, *dense_axes, **init_args)

        if dense_hists:
            categorical_axes = histogram.categorical_axes
            if categorical_axes:
                axis_growth_flags = tuple(
                    getattr(axis.traits, "growth", False) for axis in categorical_axes
                )
                growth_axes = tuple(
                    axis for axis, grows in zip(categorical_axes, axis_growth_flags) if grows
                )
                if growth_axes:
                    axis_names = tuple(axis.name for axis in growth_axes)

                    fill_payload = {name: [] for name in axis_names}
                    convert_index = histogram.index_to_categories
                    max_batch = 500_000

                    def _flush_batch():
                        if fill_payload[axis_names[0]]:
                            hist.Hist.fill(histogram, **fill_payload)
                            for values in fill_payload.values():
                                values.clear()

                    for index_key in dense_hists.keys():
                        categories = tuple(convert_index(index_key))
                        appended = False
                        for axis, category, grows in zip(
                            categorical_axes, categories, axis_growth_flags
                        ):
                            if grows:
                                fill_payload[axis.name].append(category)
                                appended = True

                        if appended and len(fill_payload[axis_names[0]]) >= max_batch:
                            _flush_batch()

                    if fill_payload and axis_names:
                        _flush_batch()

        histogram._dense_hists = (
            dense_hists.copy() if hasattr(dense_hists, "copy") else dict(dense_hists)
        )
        return histogram
    except Exception:  # pragma: no cover - defensive fallback
        _logger.exception("Falling back to the original SparseHist deserializer.")
        return _ORIGINAL_SPARSEHIST_READ_FROM_REDUCE(
            cls, cat_axes, dense_axes, init_args, dense_hists
        )


tc_sparseHist.SparseHist._read_from_reduce = classmethod(_fast_sparsehist_from_reduce)
# Backward-compatible export used by existing tests/helpers.
SparseHist = tc_sparseHist.SparseHist

from topcoffea.modules.paths import topcoffea_path as tc_topcoffea_path
from topeft.modules.paths import topeft_path as te_topeft_path
from topcoffea.modules.get_param_from_jsons import GetParam as tc_GetParam
get_tc_param = tc_GetParam(tc_topcoffea_path("params/params.json"))
import yaml

with open(te_topeft_path("params/cr_sr_plots_metadata.yml")) as f:
    _META = yaml.safe_load(f)

REGION_CHANNEL_CONFIG = _META.get("REGION_CHANNEL_CONFIG", {})
_REGION_CHANNEL_CONFIG_DEFAULTS = {
    "CR": {"is_lepton_flavor_in_pkl": True},
    "SR": {"is_lepton_flavor_in_pkl": False},
}


def _resolve_region_channel_config(region_name):
    """Return normalized region-specific channel metadata settings."""

    region_upper = str(region_name or "").upper()
    defaults = dict(_REGION_CHANNEL_CONFIG_DEFAULTS.get(region_upper, {}))

    raw_cfg = REGION_CHANNEL_CONFIG
    if raw_cfg is None:
        raw_cfg = {}
    if not isinstance(raw_cfg, Mapping):
        raise TypeError(
            "REGION_CHANNEL_CONFIG must be a mapping, got '{}'.".format(
                type(raw_cfg).__name__
            )
        )

    region_cfg = raw_cfg.get(region_upper, {})
    if region_cfg is None:
        region_cfg = {}
    if not isinstance(region_cfg, Mapping):
        raise TypeError(
            "REGION_CHANNEL_CONFIG['{}'] must be a mapping, got '{}'.".format(
                region_upper, type(region_cfg).__name__
            )
        )

    unsupported = sorted(
        key for key in region_cfg.keys() if key not in {"is_lepton_flavor_in_pkl"}
    )
    if unsupported:
        raise ValueError(
            "Unsupported REGION_CHANNEL_CONFIG keys for '{}': {}. Allowed keys: "
            "'is_lepton_flavor_in_pkl'.".format(region_upper, unsupported)
        )

    if "is_lepton_flavor_in_pkl" in region_cfg:
        defaults["is_lepton_flavor_in_pkl"] = bool(
            region_cfg["is_lepton_flavor_in_pkl"]
        )

    return defaults


def _coerce_channel_alias(alias_value, *, region_label, base_key):
    """Return a normalized alias string or ``None`` for *alias_value*."""

    if alias_value is None:
        return None
    if not isinstance(alias_value, str):
        raise TypeError(
            "Invalid alias for base '{}' in {}: expected string or null, got '{}'.".format(
                base_key, region_label, type(alias_value).__name__
            )
        )
    normalized = alias_value.strip()
    if not normalized:
        return None
    return normalized


def _normalize_channel_map_entries(raw_channel_map, *, region_label):
    """Normalize raw channel metadata to ``{base: {'leaves': [...], 'alias': ...}}``."""

    if raw_channel_map is None:
        return OrderedDict()
    if not isinstance(raw_channel_map, Mapping):
        raise TypeError(
            "Channel dictionary for {} must be a mapping, got '{}'.".format(
                region_label, type(raw_channel_map).__name__
            )
        )

    normalized = OrderedDict()
    for base_key, raw_entry in raw_channel_map.items():
        base_name = str(base_key)

        if isinstance(raw_entry, Mapping):
            unsupported = sorted(
                key for key in raw_entry.keys() if key not in {"leaves", "alias"}
            )
            if unsupported:
                raise ValueError(
                    "Unsupported keys {} in channel entry '{}' for {}. "
                    "Allowed keys: 'leaves', 'alias'.".format(
                        unsupported, base_name, region_label
                    )
                )
            raw_leaves = raw_entry.get("leaves", [])
            alias_value = _coerce_channel_alias(
                raw_entry.get("alias"), region_label=region_label, base_key=base_name
            )
        elif isinstance(raw_entry, (list, tuple)):
            raw_leaves = raw_entry
            alias_value = None
        else:
            raise TypeError(
                "Invalid channel entry type for '{}' in {}: expected list/tuple or "
                "mapping, got '{}'.".format(
                    base_name, region_label, type(raw_entry).__name__
                )
            )

        if raw_leaves is None:
            raw_leaves = []
        if not isinstance(raw_leaves, (list, tuple)):
            raise TypeError(
                "Invalid leaves payload for '{}' in {}: expected list/tuple, got '{}'.".format(
                    base_name, region_label, type(raw_leaves).__name__
                )
            )

        deduped_leaves = []
        seen_leaves = set()
        for leaf_value in raw_leaves:
            if isinstance(leaf_value, Mapping):
                raise TypeError(
                    "Ambiguous list entry for '{}' in {}: nested mapping entries are not "
                    "allowed inside 'leaves'.".format(base_name, region_label)
                )
            leaf_name = str(leaf_value).strip()
            if not leaf_name:
                raise ValueError(
                    "Encountered an empty channel leaf under '{}' in {}.".format(
                        base_name, region_label
                    )
                )
            if leaf_name in seen_leaves:
                continue
            seen_leaves.add(leaf_name)
            deduped_leaves.append(leaf_name)

        normalized[base_name] = {"leaves": deduped_leaves, "alias": alias_value}

    return normalized


def _build_channel_namespace(raw_channel_map, *, region_label, alias_overrides=None):
    """Return canonical channel namespace maps derived from *raw_channel_map*."""

    normalized_entries = _normalize_channel_map_entries(
        raw_channel_map, region_label=region_label
    )

    base_to_leaves = OrderedDict()
    base_to_alias = OrderedDict()
    leaf_to_base = {}
    leaf_to_bases = {}
    alias_to_bases = OrderedDict()

    alias_overrides = alias_overrides or {}

    for base_name, entry in normalized_entries.items():
        leaves = list(entry.get("leaves", []))
        alias_value = entry.get("alias")
        if base_name in alias_overrides:
            alias_value = _coerce_channel_alias(
                alias_overrides[base_name], region_label=region_label, base_key=base_name
            )

        base_to_leaves[base_name] = leaves
        base_to_alias[base_name] = alias_value

        for leaf_name in leaves:
            owners = leaf_to_bases.setdefault(leaf_name, [])
            if base_name not in owners:
                owners.append(base_name)

        if alias_value:
            alias_to_bases.setdefault(alias_value, []).append(base_name)

    for leaf_name, owners in leaf_to_bases.items():
        if len(owners) == 1:
            leaf_to_base[leaf_name] = owners[0]
            continue

        raise ValueError(
            "leaf overlap detected in {} for '{}': bases {} all reference the same leaf. "
            "Channel leaves must have exactly one owner; use aliases for output grouping instead.".format(
                region_label, leaf_name, sorted(owners)
            )
        )

    for alias_name, bases in alias_to_bases.items():
        if alias_name in base_to_leaves and any(base != alias_name for base in bases):
            raise ValueError(
                "Alias/base collision in {}: alias '{}' is also a base key and cannot "
                "refer to unrelated base categories {}.".format(
                    region_label, alias_name, sorted(bases)
                )
            )

    output_name_by_base = OrderedDict(
        (base_name, base_to_alias.get(base_name) or base_name)
        for base_name in base_to_leaves
    )

    return {
        "entries": normalized_entries,
        "base_to_leaves": base_to_leaves,
        "leaf_to_base": leaf_to_base,
        "base_to_alias": base_to_alias,
        "alias_to_bases": OrderedDict(
            (alias_name, tuple(bases)) for alias_name, bases in alias_to_bases.items()
        ),
        "output_name_by_base": output_name_by_base,
    }


def _resolve_region_channel_namespace(
    region,
    *,
    channel_map=None,
    channel_aliases=None,
    region_dict_name=None,
):
    region_upper = str(region).upper() if region is not None else ""
    if region_upper == "CR":
        default_map = CR_CHAN_DICT
        default_aliases = CR_CHAN_ALIASES
        default_name = "CR_CHAN_DICT"
    elif region_upper == "SR":
        default_map = SR_CHAN_DICT
        default_aliases = SR_CHAN_ALIASES
        default_name = "SR_CHAN_DICT"
    else:
        default_map = channel_map or {}
        default_aliases = channel_aliases or {}
        default_name = "channel dictionary"

    active_map = default_map if channel_map is None else channel_map
    active_aliases = default_aliases if channel_aliases is None else channel_aliases
    dict_name = region_dict_name or default_name

    namespace = _build_channel_namespace(
        active_map, region_label=dict_name, alias_overrides=active_aliases
    )
    return namespace, dict_name


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
_CR_CHANNEL_NAMESPACE = _build_channel_namespace(
    _META["CR_CHAN_DICT"], region_label="CR_CHAN_DICT"
)
_SR_CHANNEL_NAMESPACE = _build_channel_namespace(
    _META["SR_CHAN_DICT"], region_label="SR_CHAN_DICT"
)
CR_CHAN_DICT = _CR_CHANNEL_NAMESPACE["base_to_leaves"]
SR_CHAN_DICT = _SR_CHANNEL_NAMESPACE["base_to_leaves"]
CR_CHAN_ALIASES = _CR_CHANNEL_NAMESPACE["base_to_alias"]
SR_CHAN_ALIASES = _SR_CHANNEL_NAMESPACE["base_to_alias"]
CHANNEL_REFERENCE_MAP = {**CR_CHAN_DICT, **SR_CHAN_DICT}
CR_GROUP_INFO = _META.get("CR_GRP_MAP", {})
SR_GROUP_INFO = _META.get("SR_GRP_MAP", {})
CR_GRP_PATTERNS = {k: v.get("patterns", []) for k, v in CR_GROUP_INFO.items()}
SR_GRP_PATTERNS = {k: v.get("patterns", []) for k, v in SR_GROUP_INFO.items()}
CR_GRP_MAP = {k: [] for k in CR_GRP_PATTERNS.keys()}
SR_GRP_MAP = {k: [] for k in SR_GRP_PATTERNS.keys()}
SR_SIGNAL_GROUP_KEYS = {"ttH", "ttlnu", "ttll", "tXq", "tttt"}
SIGNAL_WC_MATCHES = ("ttH", "tllq", "ttlnu", "ttll", "tHq", "tttt")
CR_KNOWN_CHANNELS = set(_CR_CHANNEL_NAMESPACE["leaf_to_base"].keys())
SR_KNOWN_CHANNELS = set(_SR_CHANNEL_NAMESPACE["leaf_to_base"].keys())
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


# Cached helpers for rate systematic metadata
@lru_cache(maxsize=None)
def _cached_get_syst(syst_name, proc_name=None, *, literal=False):
    """Fetch a systematic entry from params/rate_systs.json with logging."""

    try:
        return tuple(
            te_get_syst(syst_name, proc_name=proc_name, literal=literal)
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        _logger.error(
            "Unable to retrieve rate systematic '%s' for process '%s': %s",
            syst_name,
            proc_name,
            exc,
        )
        raise


def _get_syst_with_default(
    syst_name, proc_name=None, *, default=(1.0, 1.0), literal=False
):
    try:
        return _cached_get_syst(syst_name, proc_name, literal=literal)
    except Exception:  # pragma: no cover - fallback path
        _logger.warning(
            "Defaulting to %s for systematic '%s' and process '%s'.",
            default,
            syst_name,
            proc_name,
        )
        return default


@lru_cache(maxsize=1)
def _cached_get_syst_lst():
    return tuple(te_get_syst_lst())


@lru_cache(maxsize=None)
def _cached_get_correlation_tag(syst_type, proc_name):
    try:
        return te_get_correlation_tag(syst_type, proc_name)
    except Exception as exc:  # pragma: no cover - defensive logging
        _logger.warning(
            "No correlation tag found for systematic '%s' and process '%s': %s",
            syst_type,
            proc_name,
            exc,
        )
        return None


@lru_cache(maxsize=None)
def _cached_get_jet_dependent_syst_dict(process="Diboson"):
    return te_get_jet_dependent_syst_dict(process=process)


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

DD_YEAR_TOKENS = (
    "UL16",
    "UL16APV",
    "UL17",
    "UL18",
    "2022",
    "2022EE",
    "2023",
    "2023BPix",
)
DD_ALIAS_MAP = {
    "run2": ("UL16", "UL16APV", "UL17", "UL18"),
    "run3": ("2022", "2022EE", "2023", "2023BPix"),
    "2016": ("UL16",),
    "2016apv": ("UL16APV",),
    "2017": ("UL17",),
    "2018": ("UL18",),
}
_DD_YEAR_TOKENS_BY_LENGTH = tuple(sorted(DD_YEAR_TOKENS, key=len, reverse=True))
_DD_YEAR_TOKEN_PATTERNS = {
    token: re.compile(rf"{re.escape(token)}(?=$|[_-])", re.IGNORECASE)
    for token in _DD_YEAR_TOKENS_BY_LENGTH
}

_YEAR_SUFFIX_TOKENS = tuple(sorted(YEAR_TOKEN_RULES, key=len, reverse=True))
_LEPFLAV_TOKENS = (
    "eee",
    "eem",
    "emm",
    "mmm",
    "ee",
    "em",
    "mm",
    "e",
    "m",
)
_NJET_SUFFIX_PATTERN = re.compile(r"_(\d+)j$", re.IGNORECASE)


def _strip_year_token(value):
    """Return *value* with a recognised year token suffix removed."""

    if not isinstance(value, str):
        return value

    for token in _YEAR_SUFFIX_TOKENS:
        for separator in ("_", "-", ""):
            suffix = f"{separator}{token}"
            if not suffix:
                continue
            if not value.endswith(suffix):
                continue
            prefix = value[: -len(suffix)] if suffix else value
            if separator or not prefix or not prefix[-1].isalnum():
                return prefix
    return value


def _token_has_lepflav_component(token):
    """Return ``True`` when *token* contains a lepton-flavour marker."""

    if not isinstance(token, str):
        return False

    lowered = token.strip().lower()
    if not lowered:
        return False
    return ("e" in lowered) or ("m" in lowered)


def _parse_lepflav_token_for_region(
    channel_name,
    *,
    region_name,
    is_lepton_flavor_in_pkl,
):
    """Return the strict lepton-flavour token for *channel_name* when required."""

    if not is_lepton_flavor_in_pkl:
        return None

    if not isinstance(channel_name, str):
        raise ValueError(
            "Invalid channel '{}' for region '{}': expected a string channel name "
            "because REGION_CHANNEL_CONFIG.{}.is_lepton_flavor_in_pkl=true.".format(
                channel_name, region_name, region_name
            )
        )

    components = channel_name.split("_")
    if len(components) < 2:
        raise ValueError(
            "Invalid channel '{}' for region '{}': expected an underscore-separated "
            "lepton-flavour token at position #2 because "
            "REGION_CHANNEL_CONFIG.{}.is_lepton_flavor_in_pkl=true.".format(
                channel_name, region_name, region_name
            )
        )

    token = components[1].strip().lower()
    if not _token_has_lepflav_component(token):
        raise ValueError(
            "Invalid lepton-flavour token '{}' in channel '{}' for region '{}': "
            "expected the second token to include 'e' or 'm' because "
            "REGION_CHANNEL_CONFIG.{}.is_lepton_flavor_in_pkl=true.".format(
                components[1], channel_name, region_name, region_name
            )
        )
    return token


def _extract_lepflav_token(channel_name):
    """Return a best-effort lepton-flavour token detected in *channel_name*."""

    if not isinstance(channel_name, str):
        return None

    components = channel_name.split("_")
    for component in reversed(components):
        normalized = component.strip().lower()
        if normalized in _LEPFLAV_TOKENS:
            return normalized
    return None


def _resolve_output_components(region_ctx, category_name):
    """Return ``(output_base, njet_suffix)`` for the requested category label."""

    raw_label = str(category_name)
    njet_suffix = _extract_njet_suffix(raw_label)
    label_without_njet = _strip_njet_suffix(raw_label) if njet_suffix else raw_label

    base_label = label_without_njet
    lepflav_token = None
    if getattr(region_ctx, "is_lepton_flavor_in_pkl", False):
        if "_" in label_without_njet:
            maybe_base, maybe_token = label_without_njet.rsplit("_", 1)
            if (
                maybe_base in getattr(region_ctx, "channel_output_names", {})
                and _token_has_lepflav_component(maybe_token)
            ):
                base_label = maybe_base
                lepflav_token = maybe_token.strip().lower()

    output_base = region_ctx.channel_output_names.get(base_label, base_label)
    if lepflav_token:
        output_base = f"{output_base}_{lepflav_token}"
    return output_base, njet_suffix


def _resolve_output_category_name(region_ctx, category_name):
    """Return the output-folder category label for *category_name*."""

    output_base, suffix = _resolve_output_components(region_ctx, category_name)
    if suffix:
        return _append_njet_suffix(output_base, suffix)
    return output_base


def _extract_njet_suffix(value):
    """Return the trailing ``_<N>j`` suffix detected in *value*, if present."""

    if not isinstance(value, str):
        return None

    match = _NJET_SUFFIX_PATTERN.search(value)
    if not match:
        return None

    return f"{match.group(1).lower()}j"


def _append_njet_suffix(label, suffix):
    if not suffix:
        return label
    normalized_suffix = suffix.lower()
    if label.lower().endswith(f"_{normalized_suffix}"):
        return label
    return f"{label}_{normalized_suffix}"


def _uses_merged_njets_output_mode(channel_output_mode):
    """Return ``True`` when *channel_output_mode* requires merged-njets naming."""

    normalized = str(channel_output_mode or "").strip().lower()
    return normalized in {"merged-njets", "both-njets"}


def _strip_njet_suffix(label):
    """Return *label* with a trailing ``_<N>j`` suffix removed, if present."""

    suffix = _extract_njet_suffix(label)
    if not suffix:
        return label

    normalized = suffix.lower()
    if label.lower().endswith(f"_{normalized}"):
        return label[: -len(normalized) - 1]

    return label


def _resolve_channel_axis_labels(histogram):
    """Return the tuple of channel labels defined on *histogram*."""

    if histogram is None:
        return ()

    try:
        axis = histogram.axes["channel"]
    except Exception:
        return ()

    try:
        return tuple(str(label) for label in axis)
    except Exception:
        return tuple(axis)


def _channel_axis_has_njet_suffixes(channel_labels):
    """Return ``True`` when any channel label retains a trailing njet suffix."""

    return any(_extract_njet_suffix(label) for label in channel_labels or ())


def _resolve_process_axis_labels(histogram):
    """Return the tuple of process labels defined on *histogram*."""

    if histogram is None:
        return ()

    try:
        axis = histogram.axes["process"]
    except Exception:
        return ()

    try:
        return tuple(str(label) for label in axis)
    except Exception:
        return tuple(axis)


def _resolve_grouped_processes(group_map):
    """Return the ordered tuple of process names covered by *group_map*."""

    grouped = []
    seen = set()
    for members in (group_map or {}).values():
        for name in members or ():
            if name in seen:
                continue
            seen.add(name)
            grouped.append(name)
    return tuple(grouped)


def _filter_process_axis(histogram, allowed_processes):
    """Return *histogram* with any process not in *allowed_processes* removed."""

    if histogram is None or not _has_axis(histogram, "process"):
        return histogram

    allowed_set = set(allowed_processes or ())
    axis_labels = _resolve_process_axis_labels(histogram)
    if not axis_labels:
        return histogram

    if allowed_set:
        to_remove = [proc for proc in axis_labels if proc not in allowed_set]
    else:
        to_remove = list(axis_labels)

    if not to_remove:
        return histogram

    try:
        return histogram.remove("process", to_remove)
    except Exception:
        return histogram


def _has_axis(histogram, axis_name):
    """Return ``True`` when *histogram* exposes *axis_name* as an axis."""

    if histogram is None:
        return False

    try:
        histogram.axes[axis_name]
        return True
    except Exception:
        return False


def _preview_channel_axis_labels(histogram_mapping):
    """Return the first available set of channel labels from *histogram_mapping*."""

    if isinstance(histogram_mapping, Mapping):
        for hist_obj in histogram_mapping.values():
            labels = _resolve_channel_axis_labels(hist_obj)
            if labels:
                return labels
    else:
        return _resolve_channel_axis_labels(histogram_mapping)

    return ()


_SPLIT_WARNING_EXPECTED_PREVIEW_LIMIT = 30


def _resolve_split_warning_reference_map(region_name):
    """Return the region-specific channel map used for split-warning diagnostics."""

    region_key = str(region_name or "").upper()
    if region_key == "CR":
        return CR_CHAN_DICT
    if region_key == "SR":
        return SR_CHAN_DICT
    return CHANNEL_REFERENCE_MAP


def _summarize_expected_split_bins(
    expected_split_channels,
    *,
    preview_limit=_SPLIT_WARNING_EXPECTED_PREVIEW_LIMIT,
):
    """Return a compact summary string for expected split-channel labels."""

    count = len(expected_split_channels or ())
    if count == 0:
        return "0 total (<unspecified>)"

    if preview_limit is None:
        preview_limit = _SPLIT_WARNING_EXPECTED_PREVIEW_LIMIT

    limit = max(int(preview_limit), 0)
    if limit == 0:
        return f"{count} total"

    preview = list(expected_split_channels[:limit])
    summary = f"{count} total; showing first {len(preview)}: {', '.join(preview)}"
    if count > limit:
        summary = f"{summary}, ..."
    return summary


def _warn_missing_split_channels(
    histogram_mapping,
    reference_channel_map=None,
    *,
    region_name=None,
    is_lepton_flavor_in_pkl=True,
    expected_preview_limit=_SPLIT_WARNING_EXPECTED_PREVIEW_LIMIT,
):
    """Emit a diagnostic when lepton-flavour split channels are unavailable."""

    if not is_lepton_flavor_in_pkl:
        return

    reference_channel_map = reference_channel_map or {}
    available_channels = _preview_channel_axis_labels(histogram_mapping)
    region_label = str(region_name or "<unknown>").upper()

    expected_split = []
    for channel_bins in reference_channel_map.values():
        for channel_name in channel_bins or ():
            token = _parse_lepflav_token_for_region(
                channel_name,
                region_name=region_label,
                is_lepton_flavor_in_pkl=is_lepton_flavor_in_pkl,
            )
            if token:
                expected_split.append(channel_name)
    expected_split = sorted(dict.fromkeys(expected_split))

    available_summary = (
        ", ".join(sorted(map(str, available_channels))) if available_channels else "<none>"
    )
    expected_summary = _summarize_expected_split_bins(
        expected_split, preview_limit=expected_preview_limit
    )

    _logger.warning(
        "Split channel output was requested for region=%s but lep-flavour labels were not found on the channel axis. "
        "Available channel bins: %s. Expected flavour-split bins (from configuration): %s.",
        region_label,
        available_summary,
        expected_summary,
    )


def _filter_existing_channel_bins(bin_names, available_channels):
    """Return ``(filtered, missing)`` for *bin_names* against *available_channels*."""

    if not bin_names:
        return [], []

    available_set = None
    if available_channels:
        available_set = {str(label) for label in available_channels}

    filtered = []
    missing = []
    seen = set()
    for bin_name in bin_names:
        if available_set is not None and bin_name not in available_set:
            missing.append(bin_name)
            continue
        if bin_name in seen:
            continue
        seen.add(bin_name)
        filtered.append(bin_name)

    return filtered, missing


def _prune_empty_channel_entries(channel_dict):
    """Drop channel entries that no longer contain any bins."""

    pruned = OrderedDict()
    for key, channel_bins in channel_dict.items():
        if channel_bins is None:
            pruned[key] = None
            continue
        if channel_bins:
            pruned[key] = channel_bins
    return pruned


def _maybe_preserve_njet_bins(
    channel_dict,
    *,
    preserve=False,
    available_channels=None,
):
    """Return *channel_dict* with entries split by njets suffix when requested."""

    if not preserve:
        return channel_dict

    preserved = OrderedDict()
    available_channels = tuple(available_channels or ())

    for key, channel_bins in channel_dict.items():
        if channel_bins is None:
            preserved[key] = None
            continue

        filtered_bins, _ = _filter_existing_channel_bins(
            list(channel_bins), available_channels
        )
        if not filtered_bins:
            continue

        suffix_buckets = OrderedDict()
        suffix_order = []
        for bin_name in filtered_bins:
            suffix = _extract_njet_suffix(bin_name)
            if suffix not in suffix_buckets:
                suffix_buckets[suffix] = []
                suffix_order.append(suffix)
            suffix_buckets[suffix].append(bin_name)

        suffixes = [suffix for suffix in suffix_order if suffix is not None]
        if not suffixes:
            preserved[key] = filtered_bins
            continue

        for suffix in suffix_order:
            bucket = suffix_buckets[suffix]
            if suffix is None:
                filtered_bucket, _ = _filter_existing_channel_bins(
                    bucket, available_channels
                )
                if filtered_bucket:
                    preserved[key] = filtered_bucket
                continue
            filtered_bucket, _ = _filter_existing_channel_bins(
                bucket, available_channels
            )
            if not filtered_bucket:
                continue
            new_key = _append_njet_suffix(key, suffix)
            preserved[new_key] = filtered_bucket

    return preserved


def _augment_split_channel_entries(
    channel_dict,
    *,
    available_channels=None,
    reference_channel_map=None,
    channel_mode=None,
    region_name=None,
    is_lepton_flavor_in_pkl=False,
):
    """Inject split-lepton categories from the reference map when available."""

    if channel_mode != "per-channel" or not is_lepton_flavor_in_pkl:
        return channel_dict

    region_label = str(region_name or "<unknown>").upper()
    available_set = {str(label) for label in available_channels or ()}
    if not available_set:
        return channel_dict

    augmented = OrderedDict(channel_dict)
    for key, bin_names in (reference_channel_map or {}).items():
        if key in augmented:
            continue
        if not bin_names:
            continue
        filtered_bins = [name for name in bin_names if name in available_set]
        if not filtered_bins:
            continue
        for bin_name in filtered_bins:
            _parse_lepflav_token_for_region(
                bin_name,
                region_name=region_label,
                is_lepton_flavor_in_pkl=is_lepton_flavor_in_pkl,
            )
        augmented[key] = filtered_bins

    return augmented


def _build_split_channel_key(base_key, lepflav_token):
    """Return a deterministic split-channel label keeping any jet suffix at the end."""

    njet_suffix = _extract_njet_suffix(base_key)
    core_label = _strip_njet_suffix(base_key) if njet_suffix else base_key
    normalized_token = str(lepflav_token).strip().lower()

    if not core_label.endswith(f"_{normalized_token}"):
        core_label = f"{core_label}_{normalized_token}"

    if njet_suffix:
        return _append_njet_suffix(core_label, njet_suffix)
    return core_label


def _group_channels_by_yearless_label(
    channel_dict,
    *,
    preserve_njets=False,
    available_channels=None,
    region_name=None,
    is_lepton_flavor_in_pkl=False,
):
    """Return grouped channel entries and their display labels."""

    del preserve_njets

    grouped = OrderedDict()
    available_channels = tuple(available_channels or ())
    region_label = str(region_name or "<unknown>").upper()

    for key, channel_bins in channel_dict.items():
        normalized_key = _strip_year_token(key)
        bucket = grouped.setdefault(normalized_key, OrderedDict())
        if channel_bins is None:
            if not bucket:
                grouped[normalized_key] = None
            continue

        if bucket is None:
            bucket = OrderedDict()
            grouped[normalized_key] = bucket

        filtered_bins, _ = _filter_existing_channel_bins(
            list(channel_bins), available_channels
        )
        if not filtered_bins:
            continue

        for bin_name in filtered_bins:
            bucket.setdefault(bin_name, None)

    normalized = OrderedDict()
    display_labels = {}
    for key, bucket in grouped.items():
        if bucket is None:
            normalized[key] = None
            continue

        bin_names = list(bucket.keys())
        if not bin_names:
            continue
        if not is_lepton_flavor_in_pkl:
            normalized[key] = bin_names
            display_labels[key] = key
            continue

        token_groups = OrderedDict()
        for bin_name in bin_names:
            token = _parse_lepflav_token_for_region(
                bin_name,
                region_name=region_label,
                is_lepton_flavor_in_pkl=is_lepton_flavor_in_pkl,
            )
            token_groups.setdefault(token, []).append(bin_name)

        for token, grouped_bins in token_groups.items():
            new_key = _build_split_channel_key(key, token)
            normalized[new_key] = grouped_bins
            display_labels[new_key] = new_key

    return normalized, display_labels

YEAR_AGGREGATE_ALIASES = {
    "run2": ("2016", "2016APV", "2017", "2018"),
    "run3": ("2022", "2022EE", "2023", "2023BPix"),
}

CHANNEL_OUTPUT_CHOICES = {
    "merged": {"modes": ("aggregate",), "preserve_njets": False},
    "split": {"modes": ("per-channel",), "preserve_njets": False},
    "both": {
        "modes": ("aggregate", "per-channel"),
        "preserve_njets": False,
    },
    "merged-njets": {"modes": ("aggregate",), "preserve_njets": True},
    "split-njets": {"modes": ("per-channel",), "preserve_njets": True},
    "both-njets": {
        "modes": ("aggregate", "per-channel"),
        "preserve_njets": True,
    },
}

CHANNEL_MODE_LABELS = {
    "aggregate": "merged",
    "per-channel": "split",
}

_YEAR_TOKEN_CANONICAL = {token.lower(): token for token in YEAR_TOKEN_RULES}
_DD_YEAR_CANONICAL = {token.lower(): token for token in DD_YEAR_TOKENS}

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


def _extract_dd_year_tokens_from_cli_years(year_tokens):
    """Return canonical DD year tokens derived from *year_tokens*."""

    collected = []
    seen = set()

    for raw_value in year_tokens or ():
        if raw_value is None:
            continue
        for token in str(raw_value).split(","):
            cleaned = token.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            canonical = _DD_YEAR_CANONICAL.get(lowered)
            if canonical is None:
                mapped_tokens = DD_ALIAS_MAP.get(lowered)
            else:
                mapped_tokens = (canonical,)
            if not mapped_tokens:
                continue
            for mapped in mapped_tokens:
                if mapped not in seen:
                    seen.add(mapped)
                    collected.append(mapped)

    return tuple(collected) if collected else None


def _is_data_driven_process_label(label):
    """Return ``True`` when *label* belongs to a data-driven process family."""

    if not isinstance(label, str):
        return False
    return any(matcher.search(label) for matcher in DATA_DRIVEN_MATCHERS)


def _detect_dd_year_token(label: str) -> "str | None":
    """Return the canonical DD year token detected in *label*, if any."""

    if not isinstance(label, str):
        return None

    for token in _DD_YEAR_TOKENS_BY_LENGTH:
        if _DD_YEAR_TOKEN_PATTERNS[token].search(label):
            return token
    return None


def _dd_label_matches_selected_years(label, dd_year_tokens):
    """Return ``True`` when DD *label* matches the requested DD year tokens."""

    if dd_year_tokens is None:
        return True
    detected_token = _detect_dd_year_token(label)
    if detected_token is None:
        return False
    return detected_token in dd_year_tokens


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


def _hist_is_empty(histogram):
    """Return True when *histogram* is None or explicitly empty."""

    if histogram is None:
        return True
    empty_fn = getattr(histogram, "empty", None)
    if not callable(empty_fn):
        return False
    try:
        return bool(empty_fn())
    except Exception:
        return True


def _integrate_nominal_axis(histogram):
    """Project *histogram* onto the nominal systematic slice, if present."""

    if histogram is None:
        return None

    if not _has_axis(histogram, "systematic"):
        return histogram

    try:
        return histogram.integrate("systematic", "nominal")
    except Exception:
        return histogram


def _describe_data_driven_matcher(matcher):
    """Return a human-friendly label for a data-driven matcher."""

    pattern = getattr(matcher, "pattern", str(matcher))
    if pattern.startswith("^"):
        pattern = pattern[1:]
    return pattern


def _summarize_zero_yield_processes(
    dict_of_hists,
    *,
    region_name,
    preserve_njets_bins=False,
    region_ctx=None,
    variables=None,
):
    """Return a structured summary of zero-yield processes per channel."""

    if region_ctx is not None:
        return _summarize_zero_yield_processes_by_variable(
            dict_of_hists,
            region_ctx=region_ctx,
            variables=variables,
        )

    summary = {
        "region": region_name,
        "channels_scanned": 0,
        "channel_entries": [],
        "zero_process_total": 0,
        "data_driven_zero_total": 0,
        "missing_data_driven_prefixes": set(),
        "errors": [],
    }

    try:
        reference_hist_name = _find_reference_hist_name(dict_of_hists)
    except Exception as exc:  # pragma: no cover - defensive
        summary["errors"].append(str(exc))
        return summary

    base_hist = dict_of_hists.get(reference_hist_name)
    available_channels = set(_resolve_channel_axis_labels(base_hist))
    available_processes = tuple(_resolve_process_axis_labels(base_hist))

    if not _has_axis(base_hist, "process"):
        summary["errors"].append("No process axis available for zero-yield scan.")
        return summary

    if not available_channels:
        summary["errors"].append("No channel axis labels available for zero-yield scan.")
        return summary

    if not available_processes:
        summary["errors"].append("No process labels available for zero-yield scan.")
        return summary

    channel_map = CR_CHAN_DICT if region_name.upper() == "CR" else SR_CHAN_DICT
    data_driven_availability = {
        _describe_data_driven_matcher(matcher): [
            proc for proc in available_processes if matcher.search(proc)
        ]
        for matcher in DATA_DRIVEN_MATCHERS
    }

    for chan_label, chan_bins in channel_map.items():
        summary["channels_scanned"] += 1
        unique_bins = tuple(dict.fromkeys(chan_bins))
        selected_bins = [bin_name for bin_name in unique_bins if bin_name in available_channels]
        missing_bins = [bin_name for bin_name in unique_bins if bin_name not in available_channels]

        if not selected_bins:
            summary["channel_entries"].append(
                {
                    "label": chan_label,
                    "missing_bins": tuple(missing_bins),
                    "zero_processes": [],
                }
            )
            continue

        zero_processes = []
        for proc in available_processes:
            try:
                proc_hist = base_hist.integrate("process", [proc])
                proc_hist = proc_hist.integrate("channel", selected_bins)
            except Exception:
                continue

            proc_hist = _integrate_nominal_axis(proc_hist)

            if not _hist_has_content(proc_hist):
                is_data_driven = any(
                    matcher.search(proc) for matcher in DATA_DRIVEN_MATCHERS
                )
                zero_processes.append((proc, is_data_driven))

        if zero_processes or missing_bins:
            summary["channel_entries"].append(
                {
                    "label": chan_label,
                    "missing_bins": tuple(missing_bins),
                    "zero_processes": zero_processes,
                }
            )
            summary["zero_process_total"] += len(zero_processes)
            summary["data_driven_zero_total"] += sum(
                1 for _, is_data_driven in zero_processes if is_data_driven
            )

    for pattern_label, matches in data_driven_availability.items():
        if not matches:
            summary["missing_data_driven_prefixes"].add(pattern_label)

    return summary


def _summarize_zero_yield_processes_by_variable(
    dict_of_hists,
    *,
    region_ctx,
    variables=None,
):
    """Return a structured summary of zero-yield processes per channel and variable."""

    summary = {
        "region": region_ctx.name,
        "channels_scanned": 0,
        "channel_entries": [],
        "zero_process_total": 0,
        "data_driven_zero_total": 0,
        "missing_data_driven_prefixes": set(),
        "errors": [],
    }

    variables_to_scan = _resolve_requested_variables(
        dict_of_hists, variables, context="zero-yield scan"
    )

    prepared_cache = {}
    missing_data_driven = {}
    for matcher in DATA_DRIVEN_MATCHERS:
        missing_data_driven[_describe_data_driven_matcher(matcher)] = False

    for var_name in variables_to_scan:
        if "sumw2" in var_name:
            continue
        if var_name in region_ctx.skip_variables:
            continue

        variable_metadata = _prepare_variable_payload(
            var_name,
            region_ctx,
            metadata_only=True,
            prepared_cache=prepared_cache,
        )
        if not variable_metadata:
            prepared_cache.setdefault(var_name, None)
            continue

        histo = dict_of_hists.get(var_name)
        if histo is None:
            continue

        channel_transformations = variable_metadata["channel_transformations"]
        available_channels = set(variable_metadata.get("available_channels") or ())
        available_processes = tuple(_resolve_process_axis_labels(histo))

        if not _has_axis(histo, "process"):
            summary["errors"].append(
                f"No process axis available for zero-yield scan in variable '{var_name}'."
            )
            continue

        allowed_processes = _resolve_grouped_processes(region_ctx.group_map)
        if allowed_processes:
            available_processes = tuple(
                proc for proc in available_processes if proc in allowed_processes
            )
        else:
            available_processes = ()

        if not available_channels:
            summary["errors"].append(
                f"No channel axis labels available for zero-yield scan in variable '{var_name}'."
            )
            continue

        if not available_processes:
            summary["errors"].append(
                f"No grouped process labels available for zero-yield scan in variable '{var_name}'."
            )
            continue

        for matcher in DATA_DRIVEN_MATCHERS:
            label = _describe_data_driven_matcher(matcher)
            if not missing_data_driven.get(label):
                missing_data_driven[label] = any(
                    matcher.search(proc) for proc in available_processes
                )

        for chan_label, chan_bins in region_ctx.channel_map.items():
            summary["channels_scanned"] += 1
            unique_bins = tuple(dict.fromkeys(chan_bins))
            selected_bins = []
            missing_bins = []
            for bin_name in unique_bins:
                transformed = _apply_channel_transforms(
                    bin_name, channel_transformations
                )
                if transformed in available_channels:
                    selected_bins.append(transformed)
                else:
                    missing_bins.append(bin_name)

            if not selected_bins:
                summary["channel_entries"].append(
                    {
                        "label": chan_label,
                        "variable": var_name,
                        "missing_bins": tuple(missing_bins),
                        "zero_processes": [],
                    }
                )
                continue

            selected_bins = list(dict.fromkeys(selected_bins))

            zero_processes = []
            for proc in available_processes:
                try:
                    proc_hist = histo.integrate("process", [proc])
                    proc_hist = proc_hist.integrate("channel", selected_bins)
                except Exception:
                    continue

                proc_hist = _integrate_nominal_axis(proc_hist)

                if not _hist_has_content(proc_hist):
                    is_data_driven = any(
                        matcher.search(proc) for matcher in DATA_DRIVEN_MATCHERS
                    )
                    zero_processes.append((proc, is_data_driven))

            if zero_processes or missing_bins:
                summary["channel_entries"].append(
                    {
                        "label": chan_label,
                        "variable": var_name,
                        "missing_bins": tuple(missing_bins),
                        "zero_processes": zero_processes,
                    }
                )
                summary["zero_process_total"] += len(zero_processes)
                summary["data_driven_zero_total"] += sum(
                    1 for _, is_data_driven in zero_processes if is_data_driven
                )

    for pattern_label, seen in missing_data_driven.items():
        if not seen:
            summary["missing_data_driven_prefixes"].add(pattern_label)

    return summary


def _emit_zero_yield_summary(summary, *, detailed=False):
    """Print a short or detailed zero-yield report for the supplied *summary*."""

    region_label = summary.get("region", "<unknown>")
    flagged_channels = summary.get("channel_entries", [])
    channel_count = summary.get("channels_scanned", 0)
    zero_total = summary.get("zero_process_total", 0)
    data_driven_zero_total = summary.get("data_driven_zero_total", 0)
    missing_data_driven = summary.get("missing_data_driven_prefixes", set())
    errors = summary.get("errors", [])

    if detailed:
        print("\nZero-yield content summary:")
        for entry in flagged_channels:
            entry_label = entry.get("label", "<unknown>")
            variable = entry.get("variable")
            if variable:
                entry_label = f"{entry_label} [{variable}]"
            issues = []
            if entry.get("missing_bins"):
                issues.append(
                    "missing channels: " + ", ".join(sorted(entry["missing_bins"]))
                )
            if entry.get("zero_processes"):
                zero_labels = []
                for proc, is_data_driven in entry["zero_processes"]:
                    label = proc
                    if is_data_driven:
                        label = f"{label} [data-driven]"
                    zero_labels.append(label)
                issues.append("zero-content processes: " + ", ".join(zero_labels))
            if issues:
                print(f"  - {region_label}::{entry_label}: " + "; ".join(issues))

        if missing_data_driven:
            print(
                "  Missing data-driven families: "
                + ", ".join(sorted(missing_data_driven))
            )

    notice = (
        f"Zero-yield scan for {region_label}: {zero_total} zero-content processes"
        f" across {channel_count} channel groups"
    )

    if data_driven_zero_total or missing_data_driven:
        notice += (
            f" (data-driven zeros: {data_driven_zero_total}, missing families:"
            f" {len(missing_data_driven)})"
        )

    has_issues = bool(flagged_channels or missing_data_driven)

    if errors:
        notice += f"; unable to scan fully ({'; '.join(errors)})"
    elif not has_issues:
        notice += "; no issues detected."
    else:
        notice += "."
        if not detailed:
            notice += " Pass --report-zero-yields for details."

    print(notice)


logger = logging.getLogger(__name__)

# This script takes an input pkl file that should have both data and background MC included.
# Use the -y option to specify one or more years.
# There are various other options available from the command line.
# For example, to make unit normalized plots for 2017+2018, with the timestamp appended to the directory name, you would run:
#     python make_cr_and_sr_plots.py -f histos/your.pkl.gz -o ~/www/somewhere/in/your/web/dir -n some_dir_name -y 2017 2018 -t -u

yt = te_YieldTools()

######### Utility functions #########

# Takes a dictionary where the keys are catetory names and keys are lists of bin names in the category, and a string indicating what type of info (njets, or lepflav) to remove
# Returns a dictionary of the same structure, except with njet or lepflav info stripped off of the bin names
# E.g. if a value was ["cat_a_1j","cat_b_1j","cat_b_2j"] and we passed "njets", we should return ["cat_a","cat_b"]
def get_dict_with_stripped_bin_names(in_chan_dict,type_of_info_to_strip):
    out_chan_dict = {}
    for cat,bin_names in in_chan_dict.items():
        if isinstance(bin_names, Mapping):
            bin_names = bin_names.get("leaves", [])
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


def _integrate_category(
    histogram,
    hist_cat,
    axes_to_integrate,
    *,
    region_name=None,
    var_name=None,
    hist_label="histogram",
):
    """Integrate a histogram over the provided axes, returning None on failure."""

    if histogram is None:
        return None

    try:
        integrated = yt.integrate_out_appl(histogram, hist_cat)
        integrated = yt.integrate_out_cats(integrated, axes_to_integrate)[{"channel": sum}]
    except Exception as exc:
        logger.warning(
            "Failed to integrate %s: region=%s hist_cat=%s var_name=%s axes=%s error=%s",
            hist_label,
            region_name,
            hist_cat,
            var_name,
            sorted(axes_to_integrate.keys()),
            exc,
        )
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


def parse_rebin_plot_vars(raw_value):
    """Parse a comma-separated variable-to-integer-factor rebin specification."""

    if raw_value is None:
        return {}
    raw_text = str(raw_value).strip()
    if not raw_text:
        return {}

    parsed = OrderedDict()
    for raw_entry in raw_text.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if ":" in entry:
            var_name, factor_text = entry.split(":", 1)
        elif "=" in entry:
            var_name, factor_text = entry.split("=", 1)
        else:
            raise ValueError(
                f"Malformed rebin entry '{entry}'. Expected '<variable>:<factor>'."
            )
        var_name = var_name.strip()
        factor_text = factor_text.strip()
        if not var_name:
            raise ValueError(f"Malformed rebin entry '{entry}': variable is empty.")
        try:
            factor = int(factor_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid rebin factor for variable '{var_name}': '{factor_text}'."
            ) from exc
        if factor < 2:
            raise ValueError(
                f"Invalid rebin factor for variable '{var_name}': expected integer >= 2."
            )
        parsed[var_name] = factor

    return parsed


def _rebin_factor_slices(n_bins, factor):
    if factor < 2:
        raise ValueError("Rebin factor must be >= 2.")
    if n_bins < 1:
        raise ValueError("Cannot rebin an empty 1D bin array.")
    return [
        slice(start, min(start + factor, n_bins))
        for start in range(0, n_bins, factor)
    ]


def rebin_1d_values(values, factor):
    """Return 1D bin contents rebinned by summing groups of *factor* bins."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("Only one-dimensional arrays can be rebinned.")
    return np.asarray([array[slc].sum() for slc in _rebin_factor_slices(array.size, factor)])


def rebin_1d_variances(variances, factor):
    """Return 1D variances rebinned by summing grouped-bin variances."""

    return rebin_1d_values(variances, factor)


def rebin_1d_edges(edges, factor):
    """Return variable-width edges after grouping visible bins by *factor*.

    If the source bin count is not divisible by *factor*, the leftover bins are
    merged into the final rebinned bin.
    """

    edge_array = _validate_bin_edges(edges)
    n_bins = edge_array.size - 1
    slices = _rebin_factor_slices(n_bins, factor)
    new_edges = [edge_array[0]]
    for slc in slices:
        new_edges.append(edge_array[slc.stop])
    return np.asarray(new_edges, dtype=float)


def _rebin_has_leftover(edges, factor):
    edge_array = _validate_bin_edges(edges)
    return ((edge_array.size - 1) % factor) != 0


def _resolve_rebin_plot_edges(var_name, histogram, rebin_plot_vars, base_edges=None):
    """Resolve plot/report-time edges without mutating source histograms.

    Leftover visible bins are handled by rebin_1d_edges as a final merged bin.
    """

    factor = (rebin_plot_vars or {}).get(var_name)
    if factor is None:
        return None, None, False
    if base_edges is None:
        try:
            base_edges = histogram.axes[var_name].edges
        except (AttributeError, KeyError) as exc:
            raise ValueError(
                f"Cannot apply plot-time rebinning to variable '{var_name}': no compatible dense axis found."
            ) from exc
    target_edges = rebin_1d_edges(base_edges, factor)
    return target_edges, factor, _rebin_has_leftover(base_edges, factor)


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
        axis_index = next(
            idx for idx, axis in enumerate(dense_hist.axes) if axis.name == axis_name
        )
    except StopIteration as exc:  # pragma: no cover - defensive guard
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


def _histogram_dense_axis_names(histogram):
    """Return numeric dense-axis names available for plot-time rebinning."""

    if histogram is None:
        return []
    if hasattr(histogram, "dense_axes"):
        axes = histogram.dense_axes
    elif isinstance(histogram, hist.Hist):
        axes = histogram.axes
    else:
        return []
    return [
        axis.name
        for axis in axes
        if hasattr(axis, "edges") and not axis.__class__.__name__.endswith("Category")
    ]


def _resolve_sumw2_rebin_axis_name(histogram, variable):
    """Resolve the dense axis to use when rebinding a sumw2 companion histogram."""

    if histogram is None:
        return None

    dense_axis_names = _histogram_dense_axis_names(histogram)
    candidates = [variable, f"{variable}_sumw2"]
    for candidate in candidates:
        if candidate in dense_axis_names:
            return candidate

    if len(dense_axis_names) == 1:
        return dense_axis_names[0]

    available = ", ".join(dense_axis_names) if dense_axis_names else "<none>"
    tried = ", ".join(candidates)
    raise ValueError(
        "Cannot resolve sumw2 dense axis for variable "
        f"'{variable}'. Tried: {tried}. Available dense axes: {available}."
    )


def _clone_sumw2_with_rebinned_axis(histogram, variable, target_edges):
    """Clone a sumw2 histogram using its nominal or companion dense axis.

    Sumw2 companion histograms may use the nominal dense-axis name or a
    ``<variable>_sumw2`` dense-axis name.  Keep that policy centralized so
    plot-time rebinning and negative-report collection do not drift apart.
    """

    if histogram is None:
        return None
    sumw2_axis_name = _resolve_sumw2_rebin_axis_name(histogram, variable)
    return _clone_with_rebinned_axis(histogram, sumw2_axis_name, target_edges)


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


_SHARED_REGION_CTX = None
_SHARED_VARIABLE_PAYLOADS = None
_WORKER_RENDER_CONTEXT = None


def _initialize_render_worker(
    save_dir_path,
    skip_syst_errs,
    unit_norm_bool,
    unblind_flag,
    stacked_log_y,
    verbose,
    rebin_plot_vars=None,
    negative_weight_report=True,
    prepared_payloads=None,
    shared_region_ctx=None,
):
    """Store shared plotting context inside a worker process."""

    region_ctx = _SHARED_REGION_CTX or shared_region_ctx
    if region_ctx is None:
        raise RuntimeError(
            "Worker render context is not initialised; shared region context was not set."
        )

    shared_payloads = _SHARED_VARIABLE_PAYLOADS
    if prepared_payloads is not None:
        prepared_variables = dict(prepared_payloads)
    elif shared_payloads is not None:
        prepared_variables = shared_payloads
    else:
        prepared_variables = {}

    global _WORKER_RENDER_CONTEXT
    _WORKER_RENDER_CONTEXT = {
        "region_ctx": region_ctx,
        "save_dir_path": save_dir_path,
        "skip_syst_errs": skip_syst_errs,
        "unit_norm_bool": unit_norm_bool,
        "unblind_flag": unblind_flag,
        "stacked_log_y": stacked_log_y,
        "verbose": bool(verbose),
        "rebin_plot_vars": dict(rebin_plot_vars or {}),
        "negative_weight_report": bool(negative_weight_report),
        "prepared_variables": prepared_variables,
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
    needs_payload = False
    if var_name not in prepared_cache:
        needs_payload = True
    elif variable_payload is None:
        needs_payload = True
    elif isinstance(variable_payload, Mapping) and "hist_mc" not in variable_payload:
        needs_payload = True

    if needs_payload:
        variable_payload = _prepare_variable_payload(
            var_name,
            ctx["region_ctx"],
            verbose=verbose,
            unblind_flag=ctx["unblind_flag"],
        )
        prepared_cache[var_name] = variable_payload

    _ensure_variable_channel_coverage_validated(
        var_name,
        ctx["region_ctx"],
        variable_payload,
    )

    if category is None:
        stat_only, stat_and_syst, html_set, negative_rows = _render_variable(
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
            rebin_plot_vars=ctx["rebin_plot_vars"],
            negative_weight_report=ctx["negative_weight_report"],
        )
    else:
        if not variable_payload:
            stat_only, stat_and_syst, html_set, negative_rows = 0, 0, set(), []
        else:
            region_ctx = ctx["region_ctx"]
            channel_bins = variable_payload["channel_dict"].get(category)
            if channel_bins is None or (
                region_ctx.apply_category_skips
                and _should_skip_category(
                    region_ctx.category_skip_rules, category, var_name
                )
            ):
                stat_only, stat_and_syst, html_set, negative_rows = 0, 0, set(), []
            else:
                stat_only, stat_and_syst, html_set, negative_rows = _render_variable_category(
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
                    channel_display_labels=variable_payload.get(
                        "channel_display_labels", {}
                    ),
                    available_channels=variable_payload.get("available_channels"),
                    rebin_plot_vars=ctx["rebin_plot_vars"],
                    negative_weight_report=ctx["negative_weight_report"],
                )
    return task_id, stat_only, stat_and_syst, html_set, negative_rows


def _prepare_variable_payload(
    var_name,
    region_ctx,
    *,
    verbose=False,
    unblind_flag=False,
    metadata_only=False,
    prepared_cache=None,
):
    """Prepare variable-level plotting inputs shared across categories."""

    if prepared_cache is not None and var_name in prepared_cache:
        cached_payload = prepared_cache[var_name]
        if cached_payload is None:
            return None
        if metadata_only:
            return {
                "channel_dict": cached_payload["channel_dict"],
                "channel_transformations": cached_payload[
                    "channel_transformations"
                ],
                "is_sparse2d": cached_payload["is_sparse2d"],
                "channel_display_labels": cached_payload.get(
                    "channel_display_labels", {}
                ),
                "available_channels": cached_payload.get(
                    "available_channels", ()
                ),
            }
        return cached_payload

    histo = region_ctx.dict_of_hists[var_name]
    is_sparse2d = _is_sparse_2d_hist(histo, var_name=var_name)
    if is_sparse2d and region_ctx.skip_sparse_2d:
        return None
    has_2d_metadata = isinstance(var_name, str) and (
        (var_name in te_axes_info_2d) or ("_vs_" in var_name)
    )
    if is_sparse2d and not has_2d_metadata:
        _logger.debug(
            "Sparse 2D histogram '%s' lacks explicit metadata; ensure axes are configured if 2D plotting is desired.",
            var_name,
        )
        is_sparse2d = False

    available_channels = _resolve_channel_axis_labels(histo)
    channel_transformations = _resolve_channel_transformations(region_ctx, var_name)
    preserve_njets_for_payload = bool(region_ctx.preserve_njets_bins)
    if (
        var_name == "njets"
        and preserve_njets_for_payload
        and available_channels
        and not _channel_axis_has_njet_suffixes(available_channels)
    ):
        # Some CR njets histograms are already producer-aggregated on the channel
        # axis, so there are no per-njet channel labels left to preserve.
        preserve_njets_for_payload = False
        if "njets" not in channel_transformations:
            channel_transformations = [*channel_transformations, "njets"]

    channel_dict = _apply_channel_dict_transformations(
        region_ctx.channel_map, channel_transformations
    )
    channel_dict = _deduplicate_channel_bins(channel_dict)
    channel_dict = _prune_unsplit_flavour_entries(channel_dict, region_ctx)

    channel_dict = _augment_split_channel_entries(
        channel_dict,
        available_channels=available_channels,
        reference_channel_map=region_ctx.channel_map,
        channel_mode=region_ctx.channel_mode,
        region_name=region_ctx.name,
        is_lepton_flavor_in_pkl=region_ctx.is_lepton_flavor_in_pkl,
    )

    channel_dict = _maybe_preserve_njet_bins(
        channel_dict,
        preserve=preserve_njets_for_payload,
        available_channels=available_channels,
    )
    channel_dict = _deduplicate_channel_bins(channel_dict)
    channel_dict = _prune_empty_channel_entries(channel_dict)
    channel_dict = _filter_channel_dict_for_mode(channel_dict, region_ctx)
    channel_dict = _prune_empty_channel_entries(channel_dict)
    channel_display_labels = {}
    if region_ctx.channel_mode == "per-channel":
        channel_dict, channel_display_labels = _group_channels_by_yearless_label(
            channel_dict,
            preserve_njets=preserve_njets_for_payload,
            available_channels=available_channels,
            region_name=region_ctx.name,
            is_lepton_flavor_in_pkl=region_ctx.is_lepton_flavor_in_pkl,
        )
    else:
        channel_display_labels = {key: key for key in channel_dict.keys()}


    if metadata_only:
        return {
            "channel_dict": channel_dict,
            "channel_transformations": channel_transformations,
            "is_sparse2d": is_sparse2d,
            "channel_display_labels": channel_display_labels,
            "available_channels": available_channels,
        }

    mc_to_remove = tuple(region_ctx.samples_to_remove.get("mc") or ())
    data_to_remove = tuple(region_ctx.samples_to_remove.get("data") or ())

    hist_mc = histo if not mc_to_remove else histo.remove("process", mc_to_remove)
    hist_data = histo if not data_to_remove else histo.remove("process", data_to_remove)
    hist_mc_sumw2_orig = region_ctx.sumw2_hists.get(var_name)

    if hist_mc_sumw2_orig is not None:
        if mc_to_remove:
            hist_mc_sumw2_orig = hist_mc_sumw2_orig.remove("process", mc_to_remove)
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

    allowed_processes = _resolve_grouped_processes(region_ctx.group_map)
    hist_mc = _filter_process_axis(hist_mc, allowed_processes)
    hist_data = _filter_process_axis(hist_data, allowed_processes)
    if hist_mc_sumw2_orig is not None:
        hist_mc_sumw2_orig = _filter_process_axis(
            hist_mc_sumw2_orig, allowed_processes
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
        "channel_display_labels": channel_display_labels,
        "available_channels": available_channels,
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
    rebin_plot_vars=None,
    negative_weight_report=True,
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
        return 0, 0, set(), []

    _ensure_variable_channel_coverage_validated(var_name, region_ctx, variable_payload)

    channel_dict = variable_payload["channel_dict"]
    channel_display_labels = variable_payload.get("channel_display_labels", {})

    stat_only_plots = 0
    stat_and_syst_plots = 0
    html_dirs = set()
    negative_rows = []

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
        if region_ctx.apply_category_skips and _should_skip_category(
            region_ctx.category_skip_rules, hist_cat, var_name
        ):
            continue

        stat_only, stat_and_syst, html_set, category_negative_rows = _render_variable_category(
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
            channel_display_labels=channel_display_labels,
            available_channels=variable_payload.get("available_channels"),
            rebin_plot_vars=rebin_plot_vars,
            negative_weight_report=negative_weight_report,
        )
        stat_only_plots += stat_only
        stat_and_syst_plots += stat_and_syst
        html_dirs.update(html_set)
        negative_rows.extend(category_negative_rows)

    return stat_only_plots, stat_and_syst_plots, html_dirs, negative_rows


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
    channel_display_labels=None,
    available_channels=None,
    rebin_plot_vars=None,
    negative_weight_report=True,
):
    """Render a single (variable, category) pair and return bookkeeping totals."""

    negative_rows = []

    def _empty_render_result():
        return 0, 0, html_dirs, negative_rows

    if available_channels is None:
        available_channels = _resolve_channel_axis_labels(hist_mc)
    else:
        available_channels = tuple(available_channels)

    filtered_bins, missing_bins = _filter_existing_channel_bins(
        list(channel_bins or []), available_channels
    )

    if region_ctx.preserve_njets_bins and missing_bins and not filtered_bins:
        _logger.info(
            "Skipping %s/%s for variable '%s': no preserved njet bins overlap with histogram axis (missing=%s, available=%s)",
            region_ctx.name,
            hist_cat,
            var_name,
            sorted(missing_bins),
            sorted(str(label) for label in available_channels),
        )

    if not filtered_bins:
        return 0, 0, set(), negative_rows

    validate_channel_group(
        [hist_mc, hist_data],
        filtered_bins,
        channel_transformations,
        region=region_ctx.name,
        subgroup=hist_cat,
        variable=var_name,
        available_channels=filtered_bins,
    )

    channel_bins = filtered_bins

    base_dir = save_dir_path or ""
    raw_display_label = (channel_display_labels or {}).get(hist_cat, hist_cat)
    display_label = (
        raw_display_label
        if region_ctx.preserve_njets_bins
        else re.sub(r"_(\d+)j$", "", raw_display_label, flags=re.IGNORECASE)
    )
    category_label = str(raw_display_label)
    output_category_name = _resolve_output_category_name(region_ctx, raw_display_label)
    save_dir_path_tmp = os.path.join(base_dir, output_category_name)
    os.makedirs(save_dir_path_tmp, exist_ok=True)

    stat_only_plots = 0
    stat_and_syst_plots = 0
    html_dirs = set()

    def _warn_undrawable_plot(
        *,
        reason,
        mode,
        has_mc,
        has_data_like,
        mc_empty,
        data_empty,
    ):
        logger.warning(
            "Skipping undrawable plot (%s): region=%s hist_cat=%s var_name=%s mode=%s has_mc=%s has_data_like=%s mc_empty=%s data_empty=%s",
            reason,
            region_ctx.name,
            hist_cat,
            var_name,
            mode,
            has_mc,
            has_data_like,
            mc_empty,
            data_empty,
        )

    if region_ctx.channel_mode == "aggregate":
        if verbose:
            # Category headings are mainly useful when debugging channel regrouping.
            print(f"\n\tCategory: {hist_cat}")

        axes_to_integrate_dict = {"channel": channel_bins}
        hist_mc_integrated = _integrate_category(
            hist_mc,
            hist_cat,
            axes_to_integrate_dict,
            region_name=region_ctx.name,
            var_name=var_name,
            hist_label="mc histogram",
        )
        hist_data_integrated = _integrate_category(
            hist_data,
            hist_cat,
            axes_to_integrate_dict,
            region_name=region_ctx.name,
            var_name=var_name,
            hist_label="data histogram",
        )
        if hist_mc_integrated is None or hist_data_integrated is None:
            logger.warning(
                "Skipping plot after integration failure: region=%s hist_cat=%s var_name=%s mode=aggregate mc_integrated=%s data_integrated=%s",
                region_ctx.name,
                hist_cat,
                var_name,
                hist_mc_integrated is not None,
                hist_data_integrated is not None,
            )
            return _empty_render_result()
        hist_mc_sumw2_integrated = None
        if hist_mc_sumw2_orig is not None:
            hist_mc_sumw2_integrated = _integrate_category(
                hist_mc_sumw2_orig,
                hist_cat,
                axes_to_integrate_dict,
                region_name=region_ctx.name,
                var_name=var_name,
                hist_label="mc sumw2 histogram",
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
            rate_syst_keys = _cached_get_syst_lst()
            shape_syst_details = {
                "valid_bases": tuple(),
                "skipped_orphans": tuple(),
                "skipped_failed": tuple(),
                "renormfact_present": False,
            }
            rate_systs_summed_arr_m = 0.0
            rate_systs_summed_arr_p = 0.0
            shape_systs_summed_arr_m = 0.0
            shape_systs_summed_arr_p = 0.0
            rate_calc_ok = False
            shape_calc_ok = False

            try:
                rate_systs_summed_arr_m, rate_systs_summed_arr_p = get_rate_syst_arrs(
                    hist_mc_integrated,
                    region_ctx.group_map,
                    group_type=region_ctx.name,
                    rate_syst_by_sample=region_ctx.rate_syst_by_sample,
                )
                rate_calc_ok = True
            except Exception as exc:
                print(
                    f"Warning: Failed to compute rate systematics for {region_ctx.name} {hist_cat} {var_name}: {exc}"
                )

            try:
                shape_syst_output, shape_syst_details = get_shape_syst_arrs(
                    hist_mc_integrated,
                    group_type=region_ctx.name,
                    return_details=True,
                )
                (
                    shape_systs_summed_arr_m,
                    shape_systs_summed_arr_p,
                ) = shape_syst_output
                shape_calc_ok = True
            except Exception as exc:
                shape_syst_details = {
                    "valid_bases": tuple(),
                    "skipped_orphans": tuple(),
                    "skipped_failed": (
                        {"base": "__global__", "error": str(exc)},
                    ),
                    "renormfact_present": False,
                }
                print(
                    f"Warning: Failed to compute shape systematics for {region_ctx.name} {hist_cat} {var_name}: {exc}"
                )

            _emit_systematics_summary_once(
                region_ctx.name,
                rate_syst_keys,
                shape_syst_details,
                rate_calc_ok=rate_calc_ok,
                shape_calc_ok=shape_calc_ok,
            )

            if rate_calc_ok or shape_calc_ok:
                if var_name == "njets":
                    diboson_samples = region_ctx.group_map.get("Diboson", [])
                    if diboson_samples:
                        db_hist = _eval_without_underflow(
                            hist_mc_integrated.integrate("process", diboson_samples)[
                                {"process": sum}
                            ].integrate("systematic", "nominal")
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
            if not unblind_flag:
                logger.warning(
                    "Skipping 2D plot for hist_cat=%s var_name=%s in blinded mode "
                    "(MC-only rendering is only implemented for 1D stacked panels).",
                    hist_cat,
                    var_name,
                )
                return _empty_render_result()
            hist_mc_nominal = hist_mc_integrated[{"process": sum}].integrate(
                "systematic", "nominal"
            )
            hist_data_nominal = hist_data_integrated[{"process": sum}].integrate(
                "systematic", "nominal"
            )
            has_mc = _hist_has_content(hist_mc_nominal)
            has_data = _hist_has_content(hist_data_nominal)
            if not has_mc and not has_data:
                logger.warning(
                    "Empty data and MC histogram for hist_cat=%s var_name=%s, skipping 2D plot.",
                    hist_cat,
                    var_name,
                )
                return _empty_render_result()
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
            has_mc = _hist_has_content(hist_mc_integrated)
            has_data = _hist_has_content(hist_data_integrated)
            has_data_like = has_data
            mc_empty = _hist_is_empty(hist_mc_integrated)
            data_empty = _hist_is_empty(hist_data_integrated)
            if not has_mc:
                _warn_undrawable_plot(
                    reason="empty-mc-content",
                    mode="aggregate",
                    has_mc=has_mc,
                    has_data_like=has_data_like,
                    mc_empty=mc_empty,
                    data_empty=data_empty,
                )
                return _empty_render_result()
            if unblind_flag and not has_data:
                _warn_undrawable_plot(
                    reason="empty-data-content",
                    mode="aggregate",
                    has_mc=has_mc,
                    has_data_like=has_data_like,
                    mc_empty=mc_empty,
                    data_empty=data_empty,
                )
                return _empty_render_result()
            if mc_empty or (unblind_flag and data_empty):
                _warn_undrawable_plot(
                    reason="empty-or-missing-input",
                    mode="aggregate",
                    has_mc=has_mc,
                    has_data_like=has_data_like,
                    mc_empty=mc_empty,
                    data_empty=data_empty,
                )
                return _empty_render_result()
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
            rebin_report = _prepare_plot_rebin_and_negative_rows(
                variable=var_name,
                hist_mc=hist_mc_integrated,
                hist_mc_sumw2=hist_mc_sumw2_integrated,
                hist_data=hist_data_integrated,
                group_map=group,
                channel_or_region=region_ctx.name,
                category_if_available=hist_cat,
                rebin_plot_vars=rebin_plot_vars,
                base_edges=bins_override,
                negative_weight_report=negative_weight_report,
            )
            if rebin_report["bins"] is not None:
                stacked_kwargs["bins"] = rebin_report["bins"]
            negative_rows.extend(rebin_report["negative_rows"])
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
            if fig is None:
                _warn_undrawable_plot(
                    reason="make_region_stacked_ratio_fig-returned-none",
                    mode="aggregate",
                    has_mc=has_mc,
                    has_data_like=has_data_like,
                    mc_empty=mc_empty,
                    data_empty=data_empty,
                )
                return _empty_render_result()
        title = category_label + "_" + var_name
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
            return _empty_render_result()
        hist_mc_channel = hist_mc.integrate("channel", channels)[{"channel": sum}]
        samples_to_rm = _collect_samples_to_remove(
            region_ctx.sample_removal_rules, hist_cat, region_ctx
        )
        if samples_to_rm:
            hist_mc_channel = hist_mc_channel.remove("process", samples_to_rm)
        hist_mc_integrated = hist_mc_channel.integrate("systematic", "nominal")
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
                )[{"channel": sum}]
                if samples_to_rm:
                    hist_mc_sumw2 = hist_mc_sumw2.remove("process", samples_to_rm)
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
            rate_syst_keys = _cached_get_syst_lst()
            shape_syst_details = {
                "valid_bases": tuple(),
                "skipped_orphans": tuple(),
                "skipped_failed": tuple(),
                "renormfact_present": False,
            }
            rate_systs_summed_arr_m = 0.0
            rate_systs_summed_arr_p = 0.0
            shape_systs_summed_arr_m = 0.0
            shape_systs_summed_arr_p = 0.0
            rate_calc_ok = False
            shape_calc_ok = False
            try:
                rate_systs_summed_arr_m, rate_systs_summed_arr_p = get_rate_syst_arrs(
                    hist_mc_channel,
                    region_ctx.group_map,
                    group_type=region_ctx.name,
                    rate_syst_by_sample=region_ctx.rate_syst_by_sample,
                )
                rate_calc_ok = True
            except Exception as exc:
                print(
                    f"Warning: Failed to compute rate systematics for {region_ctx.name} {hist_cat} {var_name}: {exc}"
                )

            try:
                shape_syst_output, shape_syst_details = get_shape_syst_arrs(
                    hist_mc_channel,
                    group_type=region_ctx.name,
                    return_details=True,
                )
                (
                    shape_systs_summed_arr_m,
                    shape_systs_summed_arr_p,
                ) = shape_syst_output
                shape_calc_ok = True
            except Exception as exc:
                shape_syst_details = {
                    "valid_bases": tuple(),
                    "skipped_orphans": tuple(),
                    "skipped_failed": (
                        {"base": "__global__", "error": str(exc)},
                    ),
                    "renormfact_present": False,
                }
                print(
                    f"Warning: Failed to compute shape systematics for {region_ctx.name} {hist_cat} {var_name}: {exc}"
                )

            _emit_systematics_summary_once(
                region_ctx.name,
                rate_syst_keys,
                shape_syst_details,
                rate_calc_ok=rate_calc_ok,
                shape_calc_ok=shape_calc_ok,
            )

            if rate_calc_ok or shape_calc_ok:
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

        has_mc = _hist_has_content(hist_mc_integrated)
        has_data = _hist_has_content(hist_data_integrated)
        has_data_like = has_data
        if not has_mc:
            _warn_undrawable_plot(
                reason="empty-mc-content",
                mode="per-channel",
                has_mc=has_mc,
                has_data_like=has_data_like,
                mc_empty=_hist_is_empty(hist_mc_integrated),
                data_empty=_hist_is_empty(hist_data_integrated),
            )
            return _empty_render_result()
        if unblind_flag and not has_data:
            _warn_undrawable_plot(
                reason="empty-data-content",
                mode="per-channel",
                has_mc=has_mc,
                has_data_like=has_data_like,
                mc_empty=_hist_is_empty(hist_mc_integrated),
                data_empty=_hist_is_empty(hist_data_integrated),
            )
            return _empty_render_result()
        mc_empty = _hist_is_empty(hist_mc_integrated)
        data_empty = _hist_is_empty(hist_data_integrated)
        if mc_empty or (unblind_flag and data_empty):
            _warn_undrawable_plot(
                reason="empty-or-missing-input",
                mode="per-channel",
                has_mc=has_mc,
                has_data_like=has_data_like,
                mc_empty=mc_empty,
                data_empty=data_empty,
            )
            return _empty_render_result()
        title = f"{category_label}_{var_name}"
        if unit_norm_bool:
            title = f"{title}_unitnorm"
        bins_override = region_ctx.analysis_bins.get(var_name)
        axis_meta = te_axes_info.get(var_name, {})
        default_bins = axis_meta.get("variable")
        stacked_kwargs = {
            "group": {k: v for k, v in region_ctx.group_map.items() if v},
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
        rebin_report = _prepare_plot_rebin_and_negative_rows(
            variable=var_name,
            hist_mc=hist_mc_integrated,
            hist_mc_sumw2=hist_mc_sumw2,
            hist_data=hist_data_integrated,
            group_map=stacked_kwargs["group"],
            channel_or_region=region_ctx.name,
            category_if_available=hist_cat,
            rebin_plot_vars=rebin_plot_vars,
            base_edges=bins_to_use,
            negative_weight_report=negative_weight_report,
        )
        if rebin_report["bins"] is not None:
            stacked_kwargs["bins"] = rebin_report["bins"]
        negative_rows.extend(rebin_report["negative_rows"])
        fig = make_region_stacked_ratio_fig(
            hist_mc_integrated,
            hist_data_integrated,
            var=var_name,
            unit_norm_bool=unit_norm_bool,
            **stacked_kwargs,
        )
        if fig is None:
            _warn_undrawable_plot(
                reason="make_region_stacked_ratio_fig-returned-none",
                mode="per-channel",
                has_mc=has_mc,
                has_data_like=has_data_like,
                mc_empty=mc_empty,
                data_empty=data_empty,
            )
            return _empty_render_result()
        save_path = os.path.join(save_dir_path_tmp, f"{title}.png")
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
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

    return stat_only_plots, stat_and_syst_plots, html_dirs, negative_rows
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


def _collect_available_channels(histos):
    if not isinstance(histos, (list, tuple)):
        histos = [histos]

    available_channels = set()
    for histo in histos:
        available_channels.update(_resolve_channel_axis_labels(histo))
    return available_channels


def validate_variable_channel_coverage(
    histos,
    region_known_channels,
    transformations,
    *,
    region,
    variable,
    region_dict_name=None,
):
    available_channels = _collect_available_channels(histos)
    if not available_channels:
        return

    known_channels = {str(channel) for channel in (region_known_channels or ())}
    transformed_known_channels = {
        _apply_channel_transforms(channel, transformations) for channel in known_channels
    }

    unknown_raw = []
    unknown_transformed = []
    seen_transformed = set()
    for channel in sorted(str(chan) for chan in available_channels):
        transformed = _apply_channel_transforms(channel, transformations)
        if channel in known_channels or transformed in transformed_known_channels:
            continue
        unknown_raw.append(channel)
        if transformed not in seen_transformed:
            unknown_transformed.append(transformed)
            seen_transformed.add(transformed)

    if not unknown_raw:
        return

    region_label = region or "plotting region"
    dict_label = region_dict_name or f"{region_label}_CHAN_DICT"
    msg = (
        f"Global channel coverage mismatch for variable '{variable}' in region '{region_label}'. "
        f"Unknown channels not present in {dict_label}: {unknown_raw}."
    )
    if transformations:
        msg += (
            " Unknown transformed channel names after applying configured channel transformations "
            f"{transformations}: {unknown_transformed}."
        )
    msg += (
        " Update the YAML channel dictionary or add a channel transformation/alias so the histogram "
        "channel axis and metadata stay in sync."
    )
    raise ValueError(msg)


def _resolve_region_known_channels(
    region,
    *,
    variable=None,
    channel_transformations=None,
    channel_map=None,
    channel_aliases=None,
    region_dict_name=None,
    namespace_kind=None,
):
    del channel_transformations  # deterministic namespace selection does not widen by observations

    region_upper = str(region).upper() if region is not None else ""
    namespace, dict_name = _resolve_region_channel_namespace(
        region_upper,
        channel_map=channel_map,
        channel_aliases=channel_aliases,
        region_dict_name=region_dict_name,
    )
    if namespace_kind is None:
        if region_upper == "SR" and variable == "njets":
            namespace_kind = "base"
        else:
            namespace_kind = "leaf"

    if namespace_kind == "base":
        return set(namespace["base_to_leaves"].keys()), dict_name
    if namespace_kind == "leaf":
        return set(namespace["leaf_to_base"].keys()), dict_name

    raise ValueError(
        "Unsupported namespace_kind '{}'. Expected 'base' or 'leaf'.".format(
            namespace_kind
        )
    )


def _expected_channel_namespace_kind(region_ctx, var_name):
    """Return deterministic namespace selection for global channel validation."""

    if var_name != "njets":
        return "leaf"

    region_upper = str(region_ctx.name).upper()
    if region_upper == "SR":
        return "base"

    # CR njets in merged-njets style is validated against producer-style base
    # labels obtained by deterministic transforms of leaf metadata.
    if region_upper == "CR" and _uses_merged_njets_output_mode(
        getattr(region_ctx, "channel_output_mode", None)
    ):
        return "leaf"

    return "leaf"


def _resolve_validation_channel_transformations(
    region_ctx,
    var_name,
    channel_transformations,
):
    """Return deterministic transforms used for global channel validation."""

    ordered = []
    seen = set()
    for transform in channel_transformations or ():
        if transform in seen:
            continue
        ordered.append(transform)
        seen.add(transform)

    if (
        var_name == "njets"
        and str(region_ctx.name).upper() == "CR"
        and _uses_merged_njets_output_mode(
            getattr(region_ctx, "channel_output_mode", None)
        )
        and "njets" not in seen
    ):
        # CR merged-njets njets histograms may expose producer base labels on
        # the channel axis; include the deterministic njets transform for global
        # validation while preserving plotting payload behaviour.
        ordered.append("njets")

    return ordered


def _ensure_variable_channel_coverage_validated(var_name, region_ctx, variable_payload):
    if not variable_payload:
        return
    if variable_payload.get("_global_channel_coverage_validated"):
        return

    histos = [variable_payload.get("hist_mc"), variable_payload.get("hist_data")]
    channel_transformations = variable_payload.get("channel_transformations", [])
    validation_transformations = _resolve_validation_channel_transformations(
        region_ctx,
        var_name,
        channel_transformations,
    )
    namespace_kind = _expected_channel_namespace_kind(region_ctx, var_name)
    region_known_channels, region_dict_name = _resolve_region_known_channels(
        region_ctx.name,
        variable=var_name,
        channel_transformations=validation_transformations,
        channel_map=region_ctx.channel_map,
        channel_aliases=region_ctx.channel_base_to_alias,
        region_dict_name=region_ctx.channel_dict_name,
        namespace_kind=namespace_kind,
    )
    validate_variable_channel_coverage(
        histos,
        region_known_channels,
        validation_transformations,
        region=region_ctx.name,
        variable=var_name,
        region_dict_name=region_dict_name,
    )
    variable_payload["_global_channel_coverage_validated"] = True


def validate_channel_group(
    histos,
    expected_labels,
    transformations,
    region,
    subgroup,
    variable,
    *,
    available_channels,
):
    if not isinstance(histos, (list, tuple)):
        histos = [histos]

    available_channels = {str(channel) for channel in available_channels}

    if not available_channels:
        return

    expected_set = {str(label) for label in expected_labels}
    expected_transformed = {
        _apply_channel_transforms(label, transformations) for label in expected_set
    }

    stray_channels = set()

    for channel in available_channels:
        transformed = _apply_channel_transforms(channel, transformations)
        if transformed in expected_transformed:
            continue
        stray_channels.add(channel)

    if stray_channels:
        region_str = f" in region '{region}'" if region else ""
        var_str = f" for variable '{variable}'" if variable is not None else ""
        raise ValueError(
            f"Subgroup '{subgroup}'{region_str}{var_str} references channels not found in the subgroup-local channel selection: {sorted(stray_channels)}."
        )

def populate_group_map(samples, pattern_map):
    out = OrderedDict((k, []) for k in pattern_map)
    unmatched = []

    for proc_name in samples:
        canonical_name = tc_utils.canonicalize_process_name(proc_name)
        matched = False
        for grp, patterns in pattern_map.items():
            for pat in patterns:
                if pat in canonical_name or pat in proc_name:
                    out[grp].append(proc_name)
                    matched = True
                    break
            if matched:
                break
        if not matched:
            unmatched.append(proc_name)

    if unmatched:
        logger.warning(
            "Process names did not match any configured group pattern; skipping: %s",
            ", ".join(sorted(unmatched)),
        )

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


NEGATIVE_WEIGHT_REPORT_COLUMNS = [
    "variable",
    "channel_or_region",
    "category_if_available",
    "stage",
    "level",
    "process",
    "group",
    "bin_index",
    "bin_low",
    "bin_high",
    "yield",
    "sumw2",
    "error",
    "effective_entries",
    "total_mc_yield",
    "data_yield",
    "yield_over_total_mc",
    "abs_yield_over_total_mc",
    "is_compatible_with_zero_1sigma",
    "is_single_effective_entry_like",
    "is_low_effective_entries",
]


def effective_entries(yield_value, sumw2_value):
    """Return yield²/sumw², or NaN when sumw² is unavailable/non-positive."""

    if sumw2_value is None or not np.isfinite(sumw2_value) or sumw2_value <= 0:
        return np.nan
    return (float(yield_value) * float(yield_value)) / float(sumw2_value)


def _negative_row_flags(yield_value, sumw2_value):
    error = np.nan
    if sumw2_value is not None and np.isfinite(sumw2_value) and sumw2_value >= 0:
        error = math.sqrt(float(sumw2_value))
    eff_entries = effective_entries(yield_value, sumw2_value)
    return {
        "error": error,
        "effective_entries": eff_entries,
        "is_compatible_with_zero_1sigma": bool(
            np.isfinite(error) and abs(float(yield_value)) <= error
        ),
        "is_single_effective_entry_like": bool(
            np.isfinite(eff_entries) and eff_entries <= 1.05
        ),
        "is_low_effective_entries": bool(
            np.isfinite(eff_entries) and eff_entries <= 5.0
        ),
    }


def _hist_1d_edges(hist_obj, var_name):
    try:
        return np.asarray(hist_obj.axes[var_name].edges, dtype=float)
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"Histogram has no dense axis '{var_name}'.") from exc


def _hist_1d_visible_values(hist_obj, var_name):
    values = np.asarray(_values_without_flow(hist_obj, include_overflow=False), dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError(
            f"Expected a one-dimensional histogram projection for '{var_name}', got shape {values.shape}."
        )
    return values


def _process_values_from_hist(hist_obj, processes, var_name, template):
    if hist_obj is None:
        return np.full_like(template, np.nan, dtype=float)
    try:
        available = set(hist_obj.axes["process"])
    except (AttributeError, KeyError):
        return np.full_like(template, np.nan, dtype=float)
    if processes is sum:
        selected = hist_obj[{"process": sum}]
        return _hist_1d_visible_values(selected, var_name)
    present = [proc for proc in processes if proc in available]
    if not present:
        return np.zeros_like(template, dtype=float)
    selected = hist_obj[{"process": present}][{"process": sum}]
    return _hist_1d_visible_values(selected, var_name)


def _make_negative_contribution_row(
    *,
    variable,
    channel_or_region,
    category_if_available,
    stage,
    level,
    process,
    group,
    bin_index,
    bin_low,
    bin_high,
    yield_value,
    sumw2_value,
    total_mc_yield,
    data_yield,
):
    flags = _negative_row_flags(yield_value, sumw2_value)
    yield_over_total = np.nan
    if np.isfinite(total_mc_yield) and not np.isclose(total_mc_yield, 0.0):
        yield_over_total = float(yield_value) / float(total_mc_yield)
    row = {
        "variable": variable,
        "channel_or_region": channel_or_region,
        "category_if_available": category_if_available or "",
        "stage": stage,
        "level": level,
        "process": process or "",
        "group": group or "",
        "bin_index": int(bin_index),
        "bin_low": float(bin_low),
        "bin_high": float(bin_high),
        "yield": float(yield_value),
        "sumw2": float(sumw2_value) if sumw2_value is not None else np.nan,
        "total_mc_yield": float(total_mc_yield),
        "data_yield": float(data_yield),
        "yield_over_total_mc": yield_over_total,
        "abs_yield_over_total_mc": abs(yield_over_total)
        if np.isfinite(yield_over_total)
        else np.nan,
    }
    row.update(flags)
    return row


def collect_negative_contribution_rows(
    *,
    variable,
    channel_or_region,
    category_if_available,
    stage,
    hist_mc,
    hist_mc_sumw2,
    hist_data,
    group_map,
):
    """Collect process-level and group-level negative MC contribution rows."""

    if hist_mc is None:
        return []

    edges = _hist_1d_edges(hist_mc, variable)
    template = _hist_1d_visible_values(hist_mc[{"process": sum}], variable)
    total_mc_values = template
    data_values = (
        _process_values_from_hist(hist_data, sum, variable, template)
        if hist_data is not None
        else np.zeros_like(template, dtype=float)
    )

    process_labels = list(hist_mc.axes["process"])
    process_to_group = {}
    for group_name, members in (group_map or {}).items():
        for process_name in members:
            process_to_group.setdefault(process_name, group_name)

    rows = []

    for process_name in process_labels:
        values = _process_values_from_hist(hist_mc, [process_name], variable, template)
        sumw2_values = _process_values_from_hist(
            hist_mc_sumw2, [process_name], variable, template
        )
        for bin_index, yield_value in enumerate(values):
            if yield_value >= 0:
                continue
            rows.append(
                _make_negative_contribution_row(
                    variable=variable,
                    channel_or_region=channel_or_region,
                    category_if_available=category_if_available,
                    stage=stage,
                    level="process",
                    process=process_name,
                    group=process_to_group.get(process_name, ""),
                    bin_index=bin_index,
                    bin_low=edges[bin_index],
                    bin_high=edges[bin_index + 1],
                    yield_value=yield_value,
                    sumw2_value=sumw2_values[bin_index],
                    total_mc_yield=total_mc_values[bin_index],
                    data_yield=data_values[bin_index],
                )
            )

    available_processes = set(process_labels)
    for group_name, members in (group_map or {}).items():
        present_members = [member for member in members if member in available_processes]
        if not present_members:
            continue
        values = _process_values_from_hist(hist_mc, present_members, variable, template)
        sumw2_values = _process_values_from_hist(
            hist_mc_sumw2, present_members, variable, template
        )
        for bin_index, yield_value in enumerate(values):
            if yield_value >= 0:
                continue
            rows.append(
                _make_negative_contribution_row(
                    variable=variable,
                    channel_or_region=channel_or_region,
                    category_if_available=category_if_available,
                    stage=stage,
                    level="group",
                    process="",
                    group=group_name,
                    bin_index=bin_index,
                    bin_low=edges[bin_index],
                    bin_high=edges[bin_index + 1],
                    yield_value=yield_value,
                    sumw2_value=sumw2_values[bin_index],
                    total_mc_yield=total_mc_values[bin_index],
                    data_yield=data_values[bin_index],
                )
            )

    return rows


def collect_negative_rows_for_plot_stage(
    *,
    variable,
    channel_or_region,
    category_if_available,
    stage,
    hist_mc,
    hist_mc_sumw2,
    hist_data,
    group_map,
    target_edges=None,
):
    """Collect negative rows after optionally applying a plot/report-time rebin."""

    if target_edges is not None:
        hist_mc = _clone_with_rebinned_axis(hist_mc, variable, target_edges)
        hist_data = _clone_with_rebinned_axis(hist_data, variable, target_edges)
        hist_mc_sumw2 = _clone_sumw2_with_rebinned_axis(
            hist_mc_sumw2, variable, target_edges
        )
    return collect_negative_contribution_rows(
        variable=variable,
        channel_or_region=channel_or_region,
        category_if_available=category_if_available,
        stage=stage,
        hist_mc=hist_mc,
        hist_mc_sumw2=hist_mc_sumw2,
        hist_data=hist_data,
        group_map=group_map,
    )


def _prepare_plot_rebin_and_negative_rows(
    *,
    variable,
    hist_mc,
    hist_mc_sumw2,
    hist_data,
    group_map,
    channel_or_region,
    category_if_available,
    rebin_plot_vars,
    base_edges=None,
    negative_weight_report=True,
):
    """Resolve plot-time bins and matching negative-report stage rows.

    This is intentionally plot/report-time only: it returns the bin edges to
    pass into the plotting helper and collects rows with the existing stage
    labels ``nominal_no_rebin``, ``pre_rebin``, and ``post_rebin`` without
    mutating the input histograms or changing CLI behavior.
    """

    negative_rows = []
    target_edges, rebin_factor, rebin_leftover = _resolve_rebin_plot_edges(
        variable,
        hist_mc,
        rebin_plot_vars,
        base_edges=base_edges,
    )

    if target_edges is not None:
        if rebin_leftover:
            logger.warning(
                "Plot-time rebinning for variable '%s' by factor %d leaves leftover bins; merging leftovers into the final bin.",
                variable,
                rebin_factor,
            )
        if negative_weight_report:
            common_kwargs = {
                "variable": variable,
                "channel_or_region": channel_or_region,
                "category_if_available": category_if_available,
                "hist_mc": hist_mc,
                "hist_mc_sumw2": hist_mc_sumw2,
                "hist_data": hist_data,
                "group_map": group_map,
            }
            negative_rows.extend(
                collect_negative_rows_for_plot_stage(
                    stage="pre_rebin",
                    **common_kwargs,
                )
            )
            negative_rows.extend(
                collect_negative_rows_for_plot_stage(
                    stage="post_rebin",
                    target_edges=target_edges,
                    **common_kwargs,
                )
            )
        return {
            "target_edges": target_edges,
            "bins": target_edges,
            "negative_rows": negative_rows,
            "rebin_factor": rebin_factor,
            "rebin_leftover": rebin_leftover,
        }

    if negative_weight_report:
        negative_rows.extend(
            collect_negative_rows_for_plot_stage(
                variable=variable,
                channel_or_region=channel_or_region,
                category_if_available=category_if_available,
                stage="nominal_no_rebin",
                hist_mc=hist_mc,
                hist_mc_sumw2=hist_mc_sumw2,
                hist_data=hist_data,
                group_map=group_map,
            )
        )
    return {
        "target_edges": None,
        "bins": base_edges,
        "negative_rows": negative_rows,
        "rebin_factor": None,
        "rebin_leftover": False,
    }


def _format_report_float(value):
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(numeric):
        return ""
    return "{:.12g}".format(numeric)


def write_negative_weight_report(rows, output_dir, summary_limit=20):
    """Write negative MC contribution CSV and Markdown summary."""

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "negative_weight_contribution_report.csv")
    md_path = os.path.join(output_dir, "negative_weight_contribution_summary.md")

    normalized_rows = list(rows or [])
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=NEGATIVE_WEIGHT_REPORT_COLUMNS)
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow(
                {
                    column: _format_report_float(row.get(column))
                    for column in NEGATIVE_WEIGHT_REPORT_COLUMNS
                }
            )

    process_rows = [row for row in normalized_rows if row.get("level") == "process"]
    group_rows = [row for row in normalized_rows if row.get("level") == "group"]
    single_eff_group_rows = [
        row for row in group_rows if row.get("is_single_effective_entry_like")
    ]
    low_eff_group_rows = [row for row in group_rows if row.get("is_low_effective_entries")]
    negative_total_bins = {
        (
            row.get("variable"),
            row.get("channel_or_region"),
            row.get("category_if_available"),
            row.get("stage"),
            row.get("bin_index"),
        )
        for row in normalized_rows
        if row.get("total_mc_yield") is not None
        and np.isfinite(row.get("total_mc_yield"))
        and row.get("total_mc_yield") < 0
    }

    def _top_rows(level_rows):
        return sorted(level_rows, key=lambda row: abs(row.get("yield", 0.0)), reverse=True)[
            :summary_limit
        ]

    def _row_label(row):
        owner = row.get("process") or row.get("group") or "<unknown>"
        return (
            f"{row.get('variable')} {row.get('category_if_available') or row.get('channel_or_region')} "
            f"{row.get('stage')} bin {row.get('bin_index')} [{row.get('bin_low')}, {row.get('bin_high')}): "
            f"{owner} yield={_format_report_float(row.get('yield'))}, "
            f"sumw2={_format_report_float(row.get('sumw2'))}, "
            f"neff={_format_report_float(row.get('effective_entries'))}"
        )

    with open(md_path, "w") as md_file:
        md_file.write("# Negative MC Contribution Summary\n\n")
        md_file.write(f"- total negative process bins: {len(process_rows)}\n")
        md_file.write(f"- total negative group bins: {len(group_rows)}\n")
        md_file.write(
            "- negative group bins with effective_entries <= 1.05: "
            f"{len(single_eff_group_rows)}\n"
        )
        md_file.write(
            "- negative group bins with effective_entries <= 5: "
            f"{len(low_eff_group_rows)}\n"
        )
        md_file.write(f"- bins with negative total MC: {len(negative_total_bins)}\n\n")

        md_file.write("## Top Negative Process Contributions\n\n")
        if process_rows:
            for row in _top_rows(process_rows):
                md_file.write(f"- {_row_label(row)}\n")
        else:
            md_file.write("- none\n")

        md_file.write("\n## Top Negative Group Contributions\n\n")
        if group_rows:
            for row in _top_rows(group_rows):
                md_file.write(f"- {_row_label(row)}\n")
        else:
            md_file.write("- none\n")

    return {"csv": csv_path, "markdown": md_path}


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
    include_ratio_panel=True,
):
    """Render stacked MC content, optionally with data and ratio subpanels."""

    style = {} if style is None else style
    axes_style = _style_get(style, ("axes",), {})
    axis_label_fontsize = axes_style.get("label_fontsize", 18)
    figure_style = _style_get(style, ("figure",), {})
    figsize = tuple(figure_style.get("figsize", (10, 8)))
    height_ratios = tuple(figure_style.get("height_ratios", (4, 1)))
    if len(height_ratios) != 2:
        height_ratios = (4, 1)
    hep.style.use("CMS")
    if include_ratio_panel:
        fig, (ax, rax) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=figsize,
            gridspec_kw={"height_ratios": height_ratios},
            sharex=True,
        )
        fig.subplots_adjust(hspace=figure_style.get("hspace", 0.07))
    else:
        single_panel_figsize = tuple(figure_style.get("single_panel_figsize", figsize))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=single_panel_figsize)
        rax = None

    plt.sca(ax)
    cms_style = _style_get(style, ("cms",), {})
    cms_fontsize = cms_style.get("fontsize", 18.0)
    cms_label = hep.cms.label(lumi=lumitag, com=comtag, fontsize=cms_fontsize)
    ax.set_ylabel("Events", fontsize=axis_label_fontsize)

    summed_mc = h_mc[{"process": sum}]
    summed_data = h_data[{"process": sum}] if h_data is not None else None

    summed_mc_edges = None
    if hasattr(summed_mc, "axes"):
        try:
            summed_mc_edges = summed_mc.axes[var].edges
        except KeyError:
            summed_mc_edges = None

    summed_data_edges = None
    if summed_data is not None and hasattr(summed_data, "axes"):
        try:
            summed_data_edges = summed_data.axes[var].edges
        except KeyError:
            summed_data_edges = None

    if summed_mc_edges is None:
        summed_mc_edges = summed_data_edges
    if summed_data_edges is None:
        summed_data_edges = summed_mc_edges

    default_bins = (summed_mc_edges if summed_mc_edges is not None else summed_data_edges)
    if bins is not None:
        default_bins = bins
    if default_bins is None:
        raise ValueError("Histogram axis has fewer than two edges; cannot determine binning.")
    bins = np.asarray(default_bins, dtype=float)
    n_bins = max(bins.size - 1, 0)

    axis_traits = None
    axis_obj = None
    for candidate in (summed_mc, summed_data, h_mc, h_data):
        if candidate is None:
            continue
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
    summed_data_values_flow = (
        _values_with_flow_or_overflow(summed_data)
        if summed_data is not None
        else np.zeros_like(summed_mc_values_flow)
    )
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

    ratio_vals = None
    ratio_yerr = None
    mc_totals = summed_mc_values
    if include_ratio_panel:
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


def _draw_stacked_panel_only(
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
    return _draw_stacked_panel(
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
        log_scale=log_scale,
        style=style,
        include_ratio_panel=False,
    )


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
    has_ratio_axis = rax is not None

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

    has_main_syst_arrays = all(arr is not None for arr in (err_p_syst, err_m_syst))
    has_ratio_syst_arrays = all(
        arr is not None for arr in (err_ratio_p_syst, err_ratio_m_syst)
    )
    has_syst_arrays = has_main_syst_arrays and (
        has_ratio_syst_arrays or not has_ratio_axis
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
    if has_main_syst_arrays:
        syst_up = np.asarray(err_p_syst)
        syst_down = np.asarray(err_m_syst)
        ratio_syst_up = (
            np.asarray(err_ratio_p_syst) if has_ratio_syst_arrays else None
        )
        ratio_syst_down = (
            np.asarray(err_ratio_m_syst) if has_ratio_syst_arrays else None
        )

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
        if (
            has_ratio_axis
            and ratio_syst_band_up is not None
            and ratio_syst_band_down is not None
        ):
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
        if (
            has_ratio_axis
            and ratio_stat_band_up is not None
            and ratio_stat_band_down is not None
        ):
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
            if (
                has_ratio_axis
                and ratio_total_band_up is not None
                and ratio_total_band_down is not None
            ):
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

    if has_ratio_axis and ratio_band_handles:
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


def _anchor_figure_legend_above_axes(
    fig,
    legend,
    *,
    legend_top_margin_min,
    legend_top_margin_scale,
):
    if legend is None:
        return {
            "legend_box": None,
            "legend_anchor": None,
            "required_headroom": None,
            "top_adjusted": False,
            "legend_is_figure_anchored": False,
        }

    fig.canvas.draw()
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

    subplot_params = fig.subplotpars
    available_top = np.clip(1.0 - required_headroom, 0.0, 1.0)
    available_top = np.clip(min(available_top, legend_box.y0), 0.0, 1.0)
    top_adjusted = False
    if subplot_params.top > available_top:
        plt.subplots_adjust(top=available_top)
        top_adjusted = True
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer=renderer)
        legend_box = legend_bbox.transformed(fig.transFigure.inverted())

    return {
        "legend_box": legend_box,
        "legend_anchor": legend_anchor,
        "required_headroom": required_headroom,
        "top_adjusted": top_adjusted,
        "legend_is_figure_anchored": True,
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
    legend_anchor=None,
    legend_is_figure=False,
    style=None,
):
    """Align legends and axis annotations after all plotting calls."""

    axis_objects = [ax]
    if rax is not None:
        axis_objects.append(rax)
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
                    for axis_obj in axis_objects:
                        axis_box = axis_obj.get_position()
                        axis_obj.set_position(
                            [
                                axis_box.x0,
                                axis_box.y0 - shift,
                                axis_box.width,
                                axis_box.height,
                            ]
                        )
                    renderer = _draw_and_get_renderer()
                    legend_bbox = legend.get_window_extent(renderer=renderer)
                    legend_box = legend_bbox.transformed(fig.transFigure.inverted())

    axis_bboxes = []
    for axis_obj in axis_objects:
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
        rightmost_extent = max(axis_obj.get_position().x1 for axis_obj in axis_objects)

    subplot_params = fig.subplotpars
    effective_right = min(np.nextafter(1.0, 0.0), rightmost_extent + 0.003)
    if not np.isclose(effective_right, subplot_params.right):
        stored_positions = [axis_obj.get_position().frozen() for axis_obj in axis_objects]
        plt.subplots_adjust(
            bottom=subplot_params.bottom,
            top=subplot_params.top,
            left=subplot_params.left,
            right=effective_right,
            hspace=subplot_params.hspace,
            wspace=subplot_params.wspace,
        )
        renderer = _draw_and_get_renderer()
        for axis_obj, original in zip(axis_objects, stored_positions):
            updated = axis_obj.get_position()
            delta_y = original.y0 - updated.y0
            if not np.isclose(delta_y, 0.0):
                axis_obj.set_position(
                    [updated.x0, updated.y0 + delta_y, updated.width, updated.height]
                )
        renderer = _draw_and_get_renderer()

    axis_for_bottom = rax if rax is not None else ax

    def _label_axis_min_y(current_renderer):
        bboxes = []
        for tick_label in axis_for_bottom.get_xticklabels():
            if not tick_label.get_visible():
                continue
            text = tick_label.get_text()
            if not text:
                continue
            bbox = tick_label.get_window_extent(renderer=current_renderer)
            bboxes.append(bbox.transformed(fig.transFigure.inverted()))
        axis_label = axis_for_bottom.xaxis.label
        if axis_label and axis_label.get_visible():
            axis_bbox = axis_label.get_window_extent(renderer=current_renderer)
            bboxes.append(axis_bbox.transformed(fig.transFigure.inverted()))
        if bboxes:
            return min(b.y0 for b in bboxes)
        return axis_for_bottom.get_position().y0

    reference_label = axis_for_bottom.yaxis.label
    default_label_size = (
        reference_label.get_size()
        if reference_label
        else plt.rcParams.get("axes.labelsize", 18)
    )
    label_fontsize = axes_style.get("label_fontsize", default_label_size)
    renderer = _draw_and_get_renderer()
    temp = fig.text(0, 0, display_label, fontsize=label_fontsize)
    temp_bbox = temp.get_window_extent(renderer=renderer)
    temp.remove()
    measured_height = temp_bbox.transformed(fig.transFigure.inverted()).height
    label_y = _label_axis_min_y(renderer) - measured_height - ratio_label_margin

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
        label_y = _label_axis_min_y(renderer) - measured_height - ratio_label_margin

    _axes_bbox_for_labeling = rax.get_position() if rax is not None else ax.get_position()

    if label_artist is None or not isinstance(label_artist, mpl.text.Text):
        label_artist = fig.text(
            _axes_bbox_for_labeling.x0 + _axes_bbox_for_labeling.width,
            label_y,
            display_label,
            ha="right",
            va="bottom",
            fontsize=label_fontsize,
        )
    else:
        label_artist.set_position(
            (_axes_bbox_for_labeling.x0 + _axes_bbox_for_labeling.width, label_y)
        )
        label_artist.set_text(display_label)
        label_artist.set_fontsize(label_fontsize)
        label_artist.set_ha("right")
        label_artist.set_va("bottom")

    return label_artist, legend_anchor_local


def _sample_in_group(sample_name, candidates, canonical_sample=None):
    canonical_sample = canonical_sample or tc_utils.canonicalize_process_name(sample_name)
    return any(
        canonical_sample == tc_utils.canonicalize_process_name(candidate)
        for candidate in (candidates or [])
    )


def _sample_in_signal_group(sample_name, sample_group_map, group_type):
    canonical_sample = tc_utils.canonicalize_process_name(sample_name)

    if group_type == "CR":
        return _sample_in_group(
            sample_name, sample_group_map.get("Signal", []), canonical_sample
        )

    if group_type == "SR":
        for grp_key in SR_SIGNAL_GROUP_KEYS:
            if _sample_in_group(
                sample_name, sample_group_map.get(grp_key, []), canonical_sample
            ):
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
        return not region_ctx.channels_split_by_lepflav
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
        if transform == "njets" and region_ctx.preserve_njets_bins:
            continue
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


def _deduplicate_channel_bins(channel_dict):
    """Return *channel_dict* with duplicate channel names removed per category."""

    deduped = OrderedDict()
    for key, channel_bins in channel_dict.items():
        if channel_bins is None:
            deduped[key] = None
            continue
        seen = set()
        cleaned = []
        for bin_name in channel_bins:
            if bin_name in seen:
                continue
            seen.add(bin_name)
            cleaned.append(bin_name)
        deduped[key] = cleaned
    return deduped


def _category_name_has_lepflav(category_name):
    cleaned = yt.get_str_without_lepflav(category_name)
    return cleaned != category_name


def _prune_unsplit_flavour_entries(channel_dict, region_ctx):
    """Drop per-flavour categories that collapse onto aggregate bins when unsplit."""

    if region_ctx.channels_split_by_lepflav:
        return channel_dict

    grouped = {}
    for key, channel_bins in channel_dict.items():
        bins_key = tuple(channel_bins or [])
        grouped.setdefault(bins_key, []).append(key)

    to_remove = set()
    for _, categories in grouped.items():
        if len(categories) <= 1:
            continue
        non_flavour = [
            category
            for category in categories
            if not _category_name_has_lepflav(category)
        ]
        keeper = non_flavour[0] if non_flavour else categories[0]
        for category in categories:
            if category == keeper:
                continue
            to_remove.add(category)

    if not to_remove:
        return channel_dict

    pruned = OrderedDict()
    for key, channel_bins in channel_dict.items():
        if key in to_remove:
            continue
        pruned[key] = channel_bins
    return pruned


def _categorize_channel_dict_entries(channel_dict):
    """Return the sets of aggregate and per-channel keys for *channel_dict*."""

    normalized = []
    for key, channel_bins in channel_dict.items():
        if channel_bins is None:
            bin_values = ()
        else:
            bin_values = tuple(channel_bins)
        normalized.append((key, frozenset(bin_values)))

    aggregate_keys = set()
    per_channel_keys = set()

    for key, bin_set in normalized:
        is_subset = False
        is_superset = False
        for other_key, other_set in normalized:
            if key == other_key:
                continue
            if bin_set < other_set:
                is_subset = True
            if bin_set > other_set:
                is_superset = True
            if is_subset and is_superset:
                break
        if not is_subset:
            aggregate_keys.add(key)
        if not is_superset:
            per_channel_keys.add(key)

    return aggregate_keys, per_channel_keys


def _filter_channel_dict_for_mode(channel_dict, region_ctx):
    """Return *channel_dict* filtered according to the region channel mode."""

    channel_mode = region_ctx.channel_mode
    if channel_mode not in {"aggregate", "per-channel"}:
        return dict(channel_dict)

    aggregate_keys, per_channel_keys = _categorize_channel_dict_entries(channel_dict)
    if channel_mode == "aggregate":
        allowed_keys = aggregate_keys
    else:
        allowed_keys = per_channel_keys

    if not allowed_keys:
        return dict(channel_dict)

    return {
        key: channel_dict[key]
        for key in channel_dict
        if key in allowed_keys
    }


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
        apply_category_skips=False,
        skip_sparse_2d=False,
        channel_mode="per-channel",
        variable_label="Variable",
        debug_channel_lists=False,
        sumw2_remove_signal=False,
        sumw2_remove_signal_when_blinded=False,
        rate_syst_by_sample=None,
        preserve_njets_bins=False,
        channel_output_mode="merged",
        channel_aliases=None,
        channel_dict_name=None,
        is_lepton_flavor_in_pkl=False,
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
        self.channel_dict_name = (
            channel_dict_name
            if channel_dict_name is not None
            else "{}_CHAN_DICT".format(str(name).upper())
        )
        channel_namespace = _build_channel_namespace(
            channel_map,
            region_label=self.channel_dict_name,
            alias_overrides=channel_aliases,
        )
        self.channel_namespace = channel_namespace
        self.channel_map = channel_namespace["base_to_leaves"]
        self.channel_base_to_alias = channel_namespace["base_to_alias"]
        self.channel_alias_to_bases = channel_namespace["alias_to_bases"]
        self.channel_output_names = channel_namespace["output_name_by_base"]
        self.is_lepton_flavor_in_pkl = bool(is_lepton_flavor_in_pkl)
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
        self.channels_split_by_lepflav = bool(
            self.is_lepton_flavor_in_pkl
            and yt.is_split_by_lepflav(
                dict_of_hists, reference_channel_map=self.channel_map
            )
        )
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
        self.apply_category_skips = bool(apply_category_skips)
        self.category_skip_rules = (
            copy.deepcopy(category_skip_rules)
            if self.apply_category_skips and category_skip_rules is not None
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
        self.rate_syst_by_sample = rate_syst_by_sample
        self.preserve_njets_bins = bool(preserve_njets_bins)
        self.channel_output_mode = str(channel_output_mode or "merged")


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


def build_region_context(
    region,
    dict_of_hists,
    years,
    unblind=None,
    *,
    channel_mode_override=None,
    preserve_njets_bins=False,
    channel_output_mode="merged",
    enable_category_skips=False,
):
    region_upper = region.upper()
    if region_upper not in ["CR","SR"]:
        raise ValueError(f"Unsupported region '{region}'.")
    region_channel_cfg = _resolve_region_channel_config(region_upper)
    is_lepton_flavor_in_pkl = bool(
        region_channel_cfg.get("is_lepton_flavor_in_pkl", False)
    )

    mc_wl = []
    mc_bl = ["data"]
    data_wl = ["data"]
    data_bl = []
    if years is None:
        raw_year_tokens = []
    elif isinstance(years, str):
        raw_year_tokens = [years]
    else:
        raw_year_tokens = list(years)

    invalid_tokens = []
    for token in raw_year_tokens:
        if token is None:
            continue
        cleaned = str(token).strip().lower()
        if not cleaned:
            continue
        if cleaned in _YEAR_TOKEN_CANONICAL:
            continue
        if cleaned in YEAR_AGGREGATE_ALIASES:
            continue
        invalid_tokens.append(token)
    if invalid_tokens:
        raise ValueError(
            "Error: Unknown year token(s) {} requested. Supported tokens: {}".format(
                ", ".join(str(token) for token in invalid_tokens),
                ", ".join(sorted(YEAR_TOKEN_RULES)),
            )
        )

    normalized_year_tokens = _normalize_year_tokens(raw_year_tokens)
    seen_years = set()
    for cleaned in normalized_year_tokens:
        if cleaned in seen_years:
            continue
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

    try:
        all_samples = yt.get_cat_lables(dict_of_hists, "process")
    except Exception:
        ref_hist = _find_reference_hist_name(dict_of_hists)
        all_samples = yt.get_cat_lables(dict_of_hists, "process", h_name=ref_hist)

    def _filter_samples(
        all_labels,
        whitelist,
        blacklist,
        *,
        allow_data_driven_reinsertion=False,
        dd_year_tokens=None,
    ):
        """Return samples that satisfy blacklist rules and multi-token requirements."""

        if len(whitelist) <= 1 and not allow_data_driven_reinsertion:
            return tc_utils.filter_lst_of_strs(
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
        dd_year_token_set = (
            frozenset(dd_year_tokens) if dd_year_tokens is not None else None
        )

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
            if (
                dd_year_token_set is not None
                and _is_data_driven_process_label(label)
                and not _dd_label_matches_selected_years(label, dd_year_token_set)
            ):
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
                if not _is_data_driven_process_label(label):
                    continue
                if any(token in label for token in blacklist):
                    continue
                if must_have_tokens and any(token not in label for token in must_have_tokens):
                    continue
                if not _dd_label_matches_selected_years(label, dd_year_token_set):
                    continue
                filtered.append(label)
                filtered_set.add(label)
        return filtered

    dd_year_tokens = _extract_dd_year_tokens_from_cli_years(
        raw_year_tokens if raw_year_tokens else normalized_year_tokens
    )

    mc_samples = _filter_samples(
        all_samples,
        mc_wl,
        mc_bl,
        allow_data_driven_reinsertion=True,
        dd_year_tokens=dd_year_tokens,
    )
    data_samples = _filter_samples(
        all_samples,
        data_wl,
        data_bl,
        allow_data_driven_reinsertion=False,
        dd_year_tokens=dd_year_tokens,
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
            if spec not in te_axes_info:
                raise KeyError(
                    f"Analysis bin specification '{spec}' is not defined in axes_info."
                )
            analysis_bins[var_name] = te_axes_info[spec]["variable"]
        else:
            analysis_bins[var_name] = spec

    channel_rules = _normalize_channel_rules(
        region_plot_cfg.get("channel_transformations")
    )
    sample_removal_rules = region_plot_cfg.get("sample_removals", [])
    category_skip_rules = (
        region_plot_cfg.get("category_skips", []) if enable_category_skips else []
    )
    skip_sparse_2d = (
        region_plot_cfg.get("skip_sparse_2d", False) if enable_category_skips else False
    )
    channel_mode = region_plot_cfg.get("channel_mode", "per-channel")
    if channel_mode_override is not None:
        if channel_mode_override not in ("aggregate", "per-channel"):
            raise ValueError(
                "Unsupported channel_mode_override '{}'. Expected 'aggregate' or 'per-channel'.".format(
                    channel_mode_override
                )
            )
        channel_mode = channel_mode_override
    variable_label = region_plot_cfg.get("variable_label", "Variable")
    debug_channel_lists = region_plot_cfg.get("debug_channel_lists", False)
    sumw2_remove_signal = region_plot_cfg.get("sumw2_remove_signal", False)
    sumw2_remove_signal_when_blinded = region_plot_cfg.get(
        "sumw2_remove_signal_when_blinded", False
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
        channel_aliases = CR_CHAN_ALIASES
        channel_dict_name = "CR_CHAN_DICT"
        group_map = populate_group_map(filtered_group_samples, group_patterns)
        signal_samples = sorted(set(group_map.get("Signal", [])))
        unblind_default = resolved_unblind
        global CR_GRP_MAP
        CR_GRP_MAP = group_map
    else:
        group_patterns = SR_GRP_PATTERNS
        channel_map = SR_CHAN_DICT
        channel_aliases = SR_CHAN_ALIASES
        channel_dict_name = "SR_CHAN_DICT"
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

    rate_syst_by_sample = {
        sample_name: get_rate_systs(
            sample_name, group_map, group_type=region_upper
        )
        for sample_name in dict.fromkeys(filtered_mc_samples or [])
    }

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
        apply_category_skips=enable_category_skips,
        skip_sparse_2d=skip_sparse_2d,
        channel_mode=channel_mode,
        variable_label=variable_label,
        debug_channel_lists=debug_channel_lists,
        sumw2_remove_signal=sumw2_remove_signal,
        sumw2_remove_signal_when_blinded=sumw2_remove_signal_when_blinded,
        rate_syst_by_sample=rate_syst_by_sample,
        preserve_njets_bins=preserve_njets_bins,
        channel_output_mode=channel_output_mode,
        channel_aliases=channel_aliases,
        channel_dict_name=channel_dict_name,
        is_lepton_flavor_in_pkl=is_lepton_flavor_in_pkl,
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
    rebin_plot_vars=None,
    negative_weight_report=True,
):
    """Render requested variables and return negative-report rows from the sweep."""

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
    category_dirs = set()
    for var_name in variables_to_plot:
        if "sumw2" in var_name:
            continue
        if region_ctx.apply_category_skips and var_name in region_ctx.skip_variables:
            continue

        variable_metadata = _prepare_variable_payload(
            var_name,
            region_ctx,
            verbose=verbose,
            unblind_flag=unblind_flag,
            metadata_only=True,
            prepared_cache=variable_payload_cache,
        )
        if not variable_metadata:
            variable_payload_cache.setdefault(var_name, None)
            continue

        if var_name not in variable_payload_cache:
            variable_payload_cache[var_name] = _prepare_variable_payload(
                var_name,
                region_ctx,
                verbose=verbose,
                unblind_flag=unblind_flag,
                prepared_cache=variable_payload_cache,
            )

        variable_payload = variable_payload_cache.get(var_name)
        if not variable_payload:
            continue

        variable_metadata = {
            "channel_dict": variable_payload["channel_dict"],
            "channel_transformations": variable_payload["channel_transformations"],
            "is_sparse2d": variable_payload["is_sparse2d"],
        }

        eligible_variables.append(var_name)

        categories = [
            hist_cat
            for hist_cat, channel_bins in variable_metadata["channel_dict"].items()
            if channel_bins is not None
            and not (
                region_ctx.apply_category_skips
                and _should_skip_category(
                    region_ctx.category_skip_rules, hist_cat, var_name
                )
            )
        ]
        variable_categories[var_name] = categories
        if save_dir_path:
            category_dirs.update(
                _resolve_output_category_name(region_ctx, category)
                for category in categories
            )

    stat_only_plots = 0
    stat_and_syst_plots = 0
    html_dirs = set()
    negative_rows = []

    worker_count = max(int(workers or 1), 1)
    tasks = list(eligible_variables)
    base_task_count = len(tasks)
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
        if worker_count > base_task_count and len(category_tasks) > base_task_count:
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

        start_method = multiprocessing.get_start_method(allow_none=True)
        shared_region_ctx = None if start_method in (None, "fork") else region_ctx

        if start_method in (None, "fork"):
            prepared_payloads = None
        else:
            prepared_payloads = {
                var_name: {
                    "channel_dict": payload["channel_dict"],
                    "channel_transformations": payload["channel_transformations"],
                    "is_sparse2d": payload["is_sparse2d"],
                }
                for var_name, payload in variable_payload_cache.items()
                if payload
            }

        global _SHARED_REGION_CTX, _SHARED_VARIABLE_PAYLOADS
        _SHARED_REGION_CTX = region_ctx
        _SHARED_VARIABLE_PAYLOADS = (
            variable_payload_cache if start_method in (None, "fork") else None
        )
        try:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_initialize_render_worker,
                initargs=(
                    save_dir_path,
                    skip_syst_errs,
                    unit_norm_bool,
                    unblind_flag,
                    stacked_log_y,
                    verbose,
                    rebin_plot_vars,
                    negative_weight_report,
                    prepared_payloads,
                    shared_region_ctx,
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
                    (
                        task_id,
                        stat_only,
                        stat_and_syst,
                        html_set,
                        task_negative_rows,
                    ) = future.result()
                    stat_only_plots += stat_only
                    stat_and_syst_plots += stat_and_syst
                    html_dirs.update(html_set)
                    negative_rows.extend(task_negative_rows)
                    _report_progress(id_to_label.get(task_id, str(task_id)))
        finally:
            _SHARED_REGION_CTX = None
            _SHARED_VARIABLE_PAYLOADS = None
    else:
        for _, _, label, var_name, hist_cat in task_specs:
            variable_payload = _get_variable_payload(var_name)
            if hist_cat is None:
                stat_only, stat_and_syst, html_set, task_negative_rows = _render_variable(
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
                    rebin_plot_vars=rebin_plot_vars,
                    negative_weight_report=negative_weight_report,
                )
            else:
                if not variable_payload:
                    stat_only, stat_and_syst, html_set, task_negative_rows = 0, 0, set(), []
                else:
                    _ensure_variable_channel_coverage_validated(
                        var_name, region_ctx, variable_payload
                    )
                    channel_bins = variable_payload["channel_dict"].get(hist_cat)
                    if channel_bins is None or (
                        region_ctx.apply_category_skips
                        and _should_skip_category(
                            region_ctx.category_skip_rules, hist_cat, var_name
                        )
                    ):
                        stat_only, stat_and_syst, html_set, task_negative_rows = 0, 0, set(), []
                    else:
                        (
                            stat_only,
                            stat_and_syst,
                            html_set,
                            task_negative_rows,
                        ) = _render_variable_category(
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
                            channel_display_labels=variable_payload.get(
                                "channel_display_labels", {}
                            ),
                            rebin_plot_vars=rebin_plot_vars,
                            negative_weight_report=negative_weight_report,
                        )
            stat_only_plots += stat_only
            stat_and_syst_plots += stat_and_syst
            html_dirs.update(html_set)
            negative_rows.extend(task_negative_rows)
            _report_progress(label)

    for html_dir in sorted(html_dirs):
        try:
            tc_make_html(html_dir)
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

    return negative_rows


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
    canonical_sample = tc_utils.canonicalize_process_name(sample_name)
    if _sample_in_group(sample_name, sample_group_map.get("Conv", []), canonical_sample):
        scale_name_for_json = "convs"
    elif _sample_in_group(sample_name, sample_group_map.get("Diboson", []), canonical_sample):
        scale_name_for_json = "Diboson"
    elif _sample_in_group(sample_name, sample_group_map.get("Triboson", []), canonical_sample):
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
                corr_tag = _cached_get_correlation_tag(
                    uncertainty_name, proc_name_in_json
                )
    return corr_tag

# This function gets all of the the rate systematics from the json file
# Returns a dictionary with all of the uncertainties
# If the sample does not have an uncertainty in the json, an uncertainty of 0 is returned for that category
def get_rate_systs(sample_name,sample_group_map,group_type="CR"):

    # Figure out the name of the appropriate sample in the syst rate json (if the proc is in the json)
    scale_name_for_json = get_scale_name(sample_name,sample_group_map,group_type=group_type)
    canonical_sample = tc_utils.canonicalize_process_name(sample_name)

    # Get the lumi uncty for this sample (same for all samles)
    lumi_uncty = _get_syst_with_default("lumi")

    # Get the flip uncty from the json (if there is not an uncertainty for this sample, return 1 since the uncertainties are multiplicative)
    if _sample_in_group(sample_name, sample_group_map["Flips"], canonical_sample):
        flip_uncty = _get_syst_with_default(
            "charge_flips", "charge_flips_sm"
        )
    else:
        flip_uncty = (1.0, 1.0)

    # Get the scale uncty from the json (if there is not an uncertainty for this sample, return 1 since the uncertainties are multiplicative)
    if scale_name_for_json is not None:
        if scale_name_for_json in PROC_WITHOUT_PDF_RATE_SYST:
            # Special cases for when we do not have a pdf uncty (this is a really brittle workaround)
            # NOTE Someday should fix this, it's a really hardcoded and brittle and bad workaround
            pdf_uncty = (1.0, 1.0)
        else:
            pdf_uncty = _get_syst_with_default("pdf_scale", scale_name_for_json)
        if scale_name_for_json == "convs":
            # Special case for conversions, since we estimate these from a LO sample, so we don't have an NLO uncty here
            # Would be better to handle this in a more general way
            qcd_uncty = (1.0, 1.0)
        else:
            # In all other cases, use the qcd scale uncty that we have for the process
            qcd_uncty = _get_syst_with_default("qcd_scale", scale_name_for_json)
    else:
        pdf_uncty = (1.0, 1.0)
        qcd_uncty = (1.0, 1.0)

    out_dict = {"pdf_scale":pdf_uncty, "qcd_scale":qcd_uncty, "lumi":lumi_uncty, "charge_flips":flip_uncty}
    return out_dict


# Wrapper for getting plus and minus rate arrs
def get_rate_syst_arrs(
    base_histo,
    proc_group_map,
    group_type="CR",
    rate_syst_by_sample=None,
):

    # Fill dictionary with the rate uncertainty arrays (with correlated ones organized together)
    rate_syst_arr_dict = {}
    process_labels = yt.get_cat_lables(base_histo, "process")

    nominal_projection = base_histo.integrate("systematic", "nominal")
    cached_rates = []
    for sample_name in process_labels:
        thissample_nom_arr = _eval_without_underflow(
            nominal_projection[{"process": sample_name}]
        )
        if rate_syst_by_sample and sample_name in rate_syst_by_sample:
            rate_syst_dict = rate_syst_by_sample[sample_name]
        else:
            rate_syst_dict = get_rate_systs(
                sample_name, proc_group_map, group_type=group_type
            )
        cached_rates.append((sample_name, thissample_nom_arr, rate_syst_dict))

    for rate_sys_type in _cached_get_syst_lst():
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

def _match_variation_length(nominal, variation):
    """Return *variation* resized to match the length of *nominal*.

    Any overflow entries beyond the nominal length are trimmed. When the
    variation is shorter, the remainder is zero-padded so downstream
    operations can rely on consistent shapes.
    """

    nominal = np.asarray(nominal)
    variation = np.asarray(variation)

    if variation.shape == nominal.shape:
        return variation

    target_len = nominal.shape[0]
    trimmed = variation[:target_len]

    if trimmed.shape[0] == target_len:
        return trimmed

    result = np.zeros(target_len, dtype=np.result_type(nominal, variation))
    result[: trimmed.shape[0]] = trimmed
    return result


def _format_syst_preview(values, max_items=20):
    values = [str(value) for value in values]
    if not values:
        return "none"
    if len(values) <= max_items:
        return ", ".join(values)
    return ", ".join(values[:max_items]) + f", +{len(values) - max_items} more"


def filter_existing_processes(requested_processes, available_processes):
    """Return requested process labels that exist on the current process axis."""

    available_set = set(available_processes)
    present = []
    missing = []
    for process_name in requested_processes:
        if process_name in available_set:
            present.append(process_name)
        else:
            missing.append(process_name)
    return present, missing


def _process_axis_labels(histo):
    """Return labels on the process axis, raising for non-process axis problems."""

    axis_names = [getattr(axis, "name", None) for axis in getattr(histo, "axes", ())]
    if "process" not in axis_names:
        raise KeyError("Histogram has no 'process' axis.")
    return tuple(yt.get_cat_lables(histo, "process"))



def _discover_shape_systematics(all_syst_var_lst):
    axis_labels = [str(label) for label in all_syst_var_lst]
    axis_label_set = set(axis_labels)

    candidate_bases = []
    seen_bases = set()
    for syst_var_name in axis_labels:
        if syst_var_name.endswith("Up"):
            syst_name_base = syst_var_name[:-2]
        elif syst_var_name.endswith("Down"):
            syst_name_base = syst_var_name[:-4]
        else:
            continue
        if syst_name_base in seen_bases:
            continue
        seen_bases.add(syst_name_base)
        candidate_bases.append(syst_name_base)

    valid_bases = []
    skipped_orphans = []
    for syst_name_base in candidate_bases:
        expected_up = syst_name_base + "Up"
        expected_down = syst_name_base + "Down"
        missing_partners = []
        if expected_up not in axis_label_set:
            missing_partners.append(expected_up)
        if expected_down not in axis_label_set:
            missing_partners.append(expected_down)
        if missing_partners:
            skipped_orphans.append(
                {"base": syst_name_base, "missing": tuple(missing_partners)}
            )
            continue
        valid_bases.append(syst_name_base)

    renormfact_present = "renormfact" in seen_bases
    renormfact_skipped = "renormfact" in valid_bases
    valid_bases = [base for base in valid_bases if base != "renormfact"]

    return {
        "axis_labels": tuple(axis_labels),
        "candidate_bases": tuple(candidate_bases),
        "valid_bases": tuple(valid_bases),
        "skipped_orphans": tuple(skipped_orphans),
        "skipped_failed": tuple(),
        "renormfact_present": renormfact_present,
        "renormfact_skipped": renormfact_skipped,
    }


def _emit_systematics_summary_once(
    region_name,
    rate_syst_keys,
    shape_details,
    *,
    rate_calc_ok,
    shape_calc_ok,
):
    summary_key = str(region_name or "UNKNOWN").upper()
    if summary_key in _SYSTEMATICS_SUMMARY_EMITTED:
        return
    _SYSTEMATICS_SUMMARY_EMITTED.add(summary_key)

    details = shape_details or {}
    valid_shape_bases = tuple(details.get("valid_bases", ()))
    skipped_orphans = tuple(details.get("skipped_orphans", ()))
    skipped_failed = tuple(details.get("skipped_failed", ()))
    renormfact_present = bool(details.get("renormfact_present", False))
    rate_keys = tuple(str(key) for key in (rate_syst_keys or ()))
    rate_text = _format_syst_preview(rate_keys)

    if shape_calc_ok and valid_shape_bases:
        print(
            f"[{summary_key}] Systematics summary: discovered {len(valid_shape_bases)} "
            f"shape systematic base(s): {_format_syst_preview(valid_shape_bases)}"
        )
    elif shape_calc_ok:
        print(f"[{summary_key}] No shape systematics found on pkl axis.")
    else:
        print(
            f"[{summary_key}] Shape systematic computation failed; shape uncertainties will be omitted."
        )

    if skipped_orphans:
        for orphan in skipped_orphans[:20]:
            missing = ", ".join(orphan.get("missing", ()))
            print(
                f"[{summary_key}] Warning: Skipping shape systematic '{orphan.get('base')}' "
                f"because missing partner(s): {missing}"
            )
        if len(skipped_orphans) > 20:
            print(
                f"[{summary_key}] Warning: {len(skipped_orphans) - 20} additional orphan shape "
                "systematics were skipped."
            )

    if skipped_failed:
        for failure in skipped_failed[:20]:
            print(
                f"[{summary_key}] Warning: Skipping shape systematic '{failure.get('base')}' "
                f"after evaluation failure: {failure.get('error')}"
            )
        if len(skipped_failed) > 20:
            print(
                f"[{summary_key}] Warning: {len(skipped_failed) - 20} additional shape systematics "
                "were skipped after evaluation failures."
            )

    print(f"[{summary_key}] Rate systematics from rate_systs.json: {rate_text}")
    if rate_calc_ok:
        print(f"[{summary_key}] Rate systematic computation succeeded.")
    else:
        print(
            f"[{summary_key}] Rate systematic computation failed; rate uncertainties will be omitted."
        )

    if shape_calc_ok:
        print(f"[{summary_key}] Shape systematic computation succeeded.")

    if renormfact_present:
        print(
            f"[{summary_key}] Note: 'renormfact' present on axis and explicitly skipped by design."
        )


# Wrapper for getting plus and minus shape arrs
def get_shape_syst_arrs(base_histo,group_type="CR",return_details=False):
    """Compute aggregate shape arrays while tolerating absent process labels.

    Process labels missing from the current process axis are filtered rather
    than treated as global shape-systematic failures.
    """

    # Get the list of systematic base names (i.e. without the up and down tags),
    # and keep only complete Up/Down pairs.
    all_syst_var_lst = yt.get_cat_lables(base_histo,"systematic")
    shape_details = _discover_shape_systematics(all_syst_var_lst)
    syst_var_lst = list(shape_details["valid_bases"])
    skipped_failed = []

    # Sum each systematic's contributions for all samples together (e.g. the ISR for all samples is summed linearly)
    p_arr_rel_lst = []
    m_arr_rel_lst = []
    for syst_name in syst_var_lst:
        try:
            relevant_samples_lst = yt.get_cat_lables(base_histo.integrate("systematic",syst_name+"Up"), "process") # The samples relevant to this syst
            relevant_samples_lst, missing_relevant_samples = filter_existing_processes(
                relevant_samples_lst,
                _process_axis_labels(base_histo),
            )
            if missing_relevant_samples:
                _logger.debug(
                    "Shape systematic '%s' ignored process labels absent from the current process axis: %s",
                    syst_name,
                    ", ".join(missing_relevant_samples),
                )
            if not relevant_samples_lst:
                continue
            proc_projection = base_histo.integrate("process", relevant_samples_lst)[{"process": sum}]
            n_arr = _eval_without_underflow(
                proc_projection.integrate("systematic", "nominal")
            )  # Sum of all samples for nominal variation
            u_arr_sum = _match_variation_length(
                n_arr,
                _eval_without_underflow(
                    proc_projection.integrate("systematic", syst_name + "Up")
                ),
            )
            d_arr_sum = _match_variation_length(
                n_arr,
                _eval_without_underflow(
                    proc_projection.integrate("systematic", syst_name + "Down")
                ),
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
                p_arr_rel = np.maximum(np.maximum(u_arr_rel, d_arr_rel), 0.0)
                m_arr_rel = np.minimum(np.minimum(u_arr_rel, d_arr_rel), 0.0)

            # Square and append this syst to the return lists
            p_arr_rel_lst.append(p_arr_rel*p_arr_rel) # Square each element in the arr and append the arr to the out list
            m_arr_rel_lst.append(m_arr_rel*m_arr_rel) # Square each element in the arr and append the arr to the out list
        except Exception as exc:
            skipped_failed.append({"base": syst_name, "error": str(exc)})
            continue

    summed_m = sum(m_arr_rel_lst) if m_arr_rel_lst else 0.0
    summed_p = sum(p_arr_rel_lst) if p_arr_rel_lst else 0.0

    if skipped_failed:
        shape_details = dict(shape_details)
        shape_details["skipped_failed"] = tuple(skipped_failed)
        failed_bases = {entry["base"] for entry in skipped_failed}
        shape_details["valid_bases"] = tuple(
            base for base in shape_details["valid_bases"] if base not in failed_bases
        )

    out = [summed_m, summed_p]
    if return_details:
        return out, shape_details
    return out


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

    if isinstance(hist_slice, tc_histEFT.HistEFT):
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
        fallback_hist = hist_for_axes if hasattr(hist_for_axes, "eval") else None
        if fallback_hist is not None:
            try:
                trimmed = _eval_without_underflow(fallback_hist)
            except Exception:
                return values
            if include_overflow or trimmed.size == 0:
                return trimmed
            return trimmed[:-1]
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
    """Combine decorrelated group uncertainties for present process labels only."""

    # Initialize the array we will return (ok technically we return sqrt of this arr squared..)
    if total_up_arr is None:
        total_up_arr = template_zeros_arr
    if total_down_arr is None:
        total_down_arr = template_zeros_arr

    result_dtype = np.result_type(template_zeros_arr, total_up_arr, total_down_arr)
    a_arr_sum = np.zeros_like(template_zeros_arr, dtype=result_dtype) # Just using this template_zeros_arr for its size
    available_processes = _process_axis_labels(base_histo)
    relevant_samples = set(relevant_samples_lst)

    # Loop over the groups of processes, generally the processes in the groups will be correlated and the different groups will be uncorrelated
    for proc_grp in grp_map.keys():
        proc_lst = grp_map[proc_grp]
        if proc_grp in ["Nonprompt","Flips","Data"]: continue # Renorm and fact not relevant here
        if proc_lst == []: continue # Nothing here
        present_proc_lst, missing_proc_lst = filter_existing_processes(
            proc_lst,
            available_processes,
        )
        if missing_proc_lst:
            _logger.debug(
                "Shape systematic '%s' ignored process labels absent from the current process axis for group '%s': %s",
                syst_name,
                proc_grp,
                ", ".join(missing_proc_lst),
            )
        proc_lst = [proc_name for proc_name in present_proc_lst if proc_name in relevant_samples]
        if proc_lst == []: continue # No process in this group contributes to this systematic in the current histogram

        # We'll keep all signal processes as uncorrelated, similar to what's done in SR
        if proc_grp == "Signal":
            for proc_name in proc_lst:
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
    diboson_njets_dict = _cached_get_jet_dependent_syst_dict()
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


def _is_sparse_2d_hist(histo, *, var_name=None):
    if not isinstance(histo, tc_sparseHist.SparseHist):
        return False

    variable_label = var_name if isinstance(var_name, str) else getattr(histo, "name", None)
    has_2d_metadata = isinstance(variable_label, str) and (
        (variable_label in te_axes_info_2d) or ("_vs_" in variable_label)
    )

    quadratic_axis = next(
        (ax for ax in histo.dense_axes if getattr(ax, "name", None) == "quadratic_term"),
        None,
    )
    quadratic_multibin = False
    if quadratic_axis is not None:
        try:
            quadratic_multibin = histo.axes["quadratic_term"].size > 1
        except (KeyError, AttributeError):  # pragma: no cover - defensive logging
            _logger.debug(
                "Unable to inspect the quadratic_term axis for '%s'; assuming it is single-bin for sparse 2D checks.",
                variable_label,
            )

    dense_axes = [ax for ax in histo.dense_axes if ax is not quadratic_axis]
    has_multiple_dense_axes = len(dense_axes) > 1

    if quadratic_multibin:
        return True

    if has_2d_metadata and has_multiple_dense_axes:
        return True

    if has_multiple_dense_axes and not has_2d_metadata:
        _logger.debug(
            "Histogram '%s' has multiple dense axes but no 2D metadata; treating it as 1D until metadata is provided.",
            variable_label,
        )

    return False


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
    axes_meta = te_axes_info_2d.get(var, {})
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


# Takes two histograms and makes a region-level stacked plot.
# In unblinded mode it includes a data/MC ratio panel; in blinded mode it renders MC only.
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

        try:
            mc_projection = h_mc[{"process": sum}].as_hist({})
        except (TypeError, AttributeError):
            mc_projection = None
        if mc_projection is not None:
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
                    h_mc_sumw2 = _clone_sumw2_with_rebinned_axis(
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

    if h_mc is None:
        return None
    if unblind and h_data is None:
        return None
    if getattr(h_mc, "empty", False) and h_mc.empty():
        return None
    if unblind and getattr(h_data, "empty", False) and h_data.empty():
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

    display_label = te_axes_info.get(var, {}).get("label", var)

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

    if unblind:
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
    else:
        panel_info = _draw_stacked_panel_only(
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
    has_ratio_axis = rax is not None
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

    ax.autoscale(axis="y")
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

    apply_minor_x = bool(secondary_ticks_cfg.get("x", True))
    apply_minor_y = bool(secondary_ticks_cfg.get("y", True))

    if has_ratio_axis:
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

        ax.set_xlabel(None)

        rax.set_ylabel(ratio_label_text, loc="center", fontsize=ratio_label_fontsize)
        rax.set_ylim(*ratio_limits)
        rax.tick_params(
            axis="both", labelsize=ratio_tick_labelsize, width=tick_width, length=tick_length
        )
        rax.tick_params(
            axis="both", which="minor", width=tick_width, length=minor_tick_length
        )
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

        if apply_minor_x:
            _apply_secondary_ticks(ax, axis="x")
            _apply_secondary_ticks(rax, axis="x")
        if apply_minor_y:
            _apply_secondary_ticks(ax, axis="y")
            _apply_secondary_ticks(rax, axis="y")

    else:
        fig.canvas.draw()
        xticks = ax.get_xticks()
        xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        if (
            overflow_label is not None
            and xtick_labels
            and len(xtick_labels) == len(xticks)
        ):
            xtick_labels[-1] = overflow_label
            ax.xaxis.set_major_locator(FixedLocator(xticks))
            ax.xaxis.set_major_formatter(FixedFormatter(xtick_labels))
        if apply_minor_x:
            _apply_secondary_ticks(ax, axis="x")
        if apply_minor_y:
            _apply_secondary_ticks(ax, axis="y")

    # Set the x axis limits.
    if set_x_lim:
        ax.set_xlim(set_x_lim)
        if has_ratio_axis:
            rax.set_xlim(set_x_lim)
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

    legend_layout = _anchor_figure_legend_above_axes(
        fig,
        legend,
        legend_top_margin_min=legend_top_margin_min,
        legend_top_margin_scale=legend_top_margin_scale,
    )
    required_headroom = legend_layout["required_headroom"]
    legend_is_figure_anchored = legend_layout["legend_is_figure_anchored"]
    top_adjusted = legend_layout["top_adjusted"]
    legend_anchor = legend_layout["legend_anchor"]

    label_artist = None
    iterations = 3 if top_adjusted else 2
    for _ in range(iterations):
        label_artist, legend_anchor = _finalize_layout(
            fig,
            ax,
            rax,
            legend,
            cms_label,
            display_label,
            label_artist=label_artist,
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
    channel_output="merged",
    enable_category_skips=False,
    report_zero_yields=False,
    rebin_plot_vars=None,
    negative_weight_report=True,
):
    """Run one CR/SR plotting pass and write optional zero/negative reports."""

    _SYSTEMATICS_SUMMARY_EMITTED.clear()

    channel_output_cfg = CHANNEL_OUTPUT_CHOICES.get(channel_output)
    if channel_output_cfg is None:
        raise ValueError(
            "Unsupported channel_output '{}' requested. Expected one of: {}".format(
                channel_output, ", ".join(sorted(CHANNEL_OUTPUT_CHOICES))
            )
        )

    requested_channel_modes = channel_output_cfg["modes"]
    preserve_njets_bins = channel_output_cfg.get("preserve_njets", False)
    warning_reference_channel_map = _resolve_split_warning_reference_map(region_name)
    region_channel_cfg = _resolve_region_channel_config(region_name)
    is_lepton_flavor_in_pkl = bool(
        region_channel_cfg.get("is_lepton_flavor_in_pkl", False)
    )

    multi_mode = len(requested_channel_modes) > 1
    split_channels_available = True
    if is_lepton_flavor_in_pkl:
        split_channels_available = yt.is_split_by_lepflav(
            dict_of_hists, reference_channel_map=CHANNEL_REFERENCE_MAP
        )
    restored_channel_labels = False
    summary_region_ctx = None
    all_negative_rows = []

    if (
        is_lepton_flavor_in_pkl
        and not split_channels_available
        and "per-channel" in requested_channel_modes
    ):
        restored_channel_labels = yt.restore_split_channel_labels(
            dict_of_hists, reference_channel_map=CHANNEL_REFERENCE_MAP
        )
        if restored_channel_labels:
            split_channels_available = yt.is_split_by_lepflav(
                dict_of_hists, reference_channel_map=CHANNEL_REFERENCE_MAP
            )

        if not split_channels_available:
            _warn_missing_split_channels(
                dict_of_hists,
                reference_channel_map=warning_reference_channel_map,
                region_name=region_name,
                is_lepton_flavor_in_pkl=is_lepton_flavor_in_pkl,
            )

    for channel_mode in requested_channel_modes:
        region_ctx = build_region_context(
            region_name,
            dict_of_hists,
            years,
            unblind=unblind,
            channel_mode_override=channel_mode,
            preserve_njets_bins=preserve_njets_bins,
            channel_output_mode=channel_output,
            enable_category_skips=enable_category_skips,
        )
        if summary_region_ctx is None:
            summary_region_ctx = region_ctx

        if region_ctx.channel_mode == "per-channel" and not split_channels_available:
            mode_label = CHANNEL_MODE_LABELS.get(channel_mode, channel_mode)
            warnings.warn(
                (
                    f"Skipping {mode_label} channel output for {region_ctx.name}: "
                    "input histograms are not split by lepton flavour."
                ),
                RuntimeWarning,
            )
            continue

        if multi_mode:
            mode_label = CHANNEL_MODE_LABELS.get(channel_mode, channel_mode)
            print(f"\n[{region_ctx.name}] Channel output mode: {mode_label}")

        negative_rows = produce_region_plots(
            region_ctx,
            save_dir_path,
            variables,
            skip_syst_errs,
            unit_norm_bool,
            stacked_log_y,
            unblind=unblind,
            workers=workers,
            verbose=verbose,
            rebin_plot_vars=rebin_plot_vars,
            negative_weight_report=negative_weight_report,
        )
        if negative_rows:
            all_negative_rows.extend(negative_rows)

    zero_yield_summary = _summarize_zero_yield_processes(
        dict_of_hists,
        region_name=region_name,
        preserve_njets_bins=preserve_njets_bins,
        region_ctx=summary_region_ctx,
        variables=variables,
    )
    _emit_zero_yield_summary(
        zero_yield_summary,
        detailed=bool(report_zero_yields),
    )

    if negative_weight_report:
        report_paths = write_negative_weight_report(
            all_negative_rows,
            save_dir_path,
        )
        print(
            "Wrote negative MC contribution report: {csv}; summary: {markdown}".format(
                **report_paths
            )
        )
    else:
        print("Negative MC contribution report disabled by --no-negative-weight-report.")

    return zero_yield_summary

def _running_in_condor():
    condor_env_vars = (
        "_CONDOR_SCRATCH_DIR",
        "_CONDOR_SLOT",
        "CONDOR_JOB_AD",
        "CONDOR_JOBID",
    )
    return any(os.environ.get(var) for var in condor_env_vars)


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


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--pkl-file-path",
        action="append",
        default=[],
        help="Path to an input pkl file. Repeat for multiple inputs.",
    )
    parser.add_argument(
        "--pkl-list-file",
        default="",
        help="Optional text file with one pkl path per line (blank lines and # comments ignored).",
    )
    parser.add_argument("-o", "--output-path", default=".", help = "The path the output files should be saved to")
    parser.add_argument("-n", "--output-name", default="plots", help = "A name for the output directory")
    parser.add_argument("-t", "--include-timestamp-tag", action="store_true", help = "Append the timestamp to the out dir name")
    parser.add_argument(
        "-y",
        "--year",
        nargs="+",
        help="One or more year tokens or aggregates to include (e.g. 2017 2018, run2, run3)",
    )
    parser.add_argument(
        "--channel-output",
        choices=(
            "merged",
            "split",
            "both",
            "merged-njets",
            "split-njets",
            "both-njets",
        ),
        default="merged",
        help=(
            "Control how channel categories are rendered: 'merged' integrates each category before plotting, "
            "'split' keeps the individual channels when flavour-split inputs are available, and 'both' renders "
            "the two sets back-to-back. The '-njets' variants preserve the per-njet bins defined in cr_sr_plots_metadata.yml "
            "instead of collapsing them into the combined templates (default: merged)."
        ),
    )
    parser.add_argument(
        "--enable-category-skips",
        action="store_true",
        help=(
            "Opt back into metadata-driven category filtering. When set, category_skip rules from "
            "cr_sr_plots_metadata.yml are applied to drop matching variable/category combinations."
        ),
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
    parser.add_argument(
        "--report-zero-yields",
        action="store_true",
        help=(
            "Emit a detailed region/channel summary of processes with zero or missing yields"
            " after plotting."
        ),
    )
    parser.add_argument(
        "--rebin-plot-vars",
        default="",
        help=(
            "Comma-separated plot-time integer rebin factors by variable, e.g. "
            "'j0pt:2,l1conept=2'. Leftover bins are merged into the final bin."
        ),
    )
    parser.add_argument(
        "--no-negative-weight-report",
        dest="negative_weight_report",
        action="store_false",
        help="Disable the end-of-run negative MC contribution CSV/Markdown report.",
    )
    parser.add_argument(
        "--on-process-collision",
        choices=["error", "warn", "allow"],
        default="error",
        help=(
            "Policy for process-label overlaps when merging multiple input pkl files. "
            "Default is strict `error`. Expert-only escape hatches: `warn`/`allow`, "
            "to be used only when overlaps are intentional (e.g. chunked outputs)."
        ),
    )
    parser.add_argument(
        "--merge-report",
        default="-",
        help="Path for merge diagnostic report JSON, or '-' for stdout.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only load+merge+validate input histograms and exit.",
    )
    parser.add_argument(
        "--cache-merged-pkl",
        default="",
        help="Optional output path for merged histogram dictionary (.pkl.gz).",
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
    parser.set_defaults(unblind=None, verbose=False, negative_weight_report=True)
    return parser


def _resolve_pkl_paths(args, parser):
    pkl_paths = list(args.pkl_file_path or [])
    if args.pkl_list_file:
        if pkl_paths:
            parser.error("Specify either repeated -f/--pkl-file-path or --pkl-list-file, not both.")
        with open(args.pkl_list_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pkl_paths.append(line)
    if not pkl_paths:
        parser.error("No input pkl files were provided. Use -f/--pkl-file-path or --pkl-list-file.")
    return pkl_paths


def _emit_merge_report(report_obj, report_path, out_dir):
    if report_path == "-":
        print("Merge report:")
        print(json.dumps(report_obj, indent=2, sort_keys=True))
        return

    report_fpath = report_path
    if not os.path.isabs(report_fpath):
        report_fpath = os.path.join(out_dir, report_fpath)
    report_parent = os.path.dirname(report_fpath)
    if report_parent:
        os.makedirs(report_parent, exist_ok=True)
    with open(report_fpath, "w") as f:
        json.dump(report_obj, f, indent=2, sort_keys=True)
    print(f"Wrote merge report: {report_fpath}")


def _cache_merged_histograms(merged_hists, cache_path, out_dir):
    out_fpath = cache_path
    if not os.path.isabs(out_fpath):
        out_fpath = os.path.join(out_dir, out_fpath)
    if not out_fpath.endswith(".pkl.gz"):
        out_fpath += ".pkl.gz"
    out_parent = os.path.dirname(out_fpath)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    print(f"Caching merged histograms to {out_fpath}")
    with gzip.open(out_fpath, "wb") as fout:
        pickle.dump(merged_hists, fout, protocol=pickle.HIGHEST_PROTOCOL)
    return out_fpath


def run_with_args(args, parser):
    """Normalize CLI arguments, load histograms, and dispatch region plotting."""

    pkl_paths = _resolve_pkl_paths(args, parser)

    normalized_years = _normalize_year_tokens(args.year)
    if args.year and not normalized_years:
        parser.error(
            "No valid year tokens were provided; expected one or more of: {}".format(
                ", ".join(sorted(YEAR_TOKEN_RULES))
            )
        )
    selected_years = normalized_years

    detected_region, ambiguous_region = _detect_region_from_path(pkl_paths[0])
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
    print(f"Channel output selection: {args.channel_output}")

    try:
        rebin_plot_vars = parse_rebin_plot_vars(args.rebin_plot_vars)
    except ValueError as exc:
        parser.error(str(exc))
    if rebin_plot_vars:
        print(
            "Plot-time rebinning requested: "
            + ", ".join(
                f"{var_name}:{factor}" for var_name, factor in rebin_plot_vars.items()
            )
        )
    print(
        "Negative MC contribution report: {}".format(
            "enabled" if args.negative_weight_report else "disabled"
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
    auto_timestamp = bool(
        _running_in_condor() and args.channel_output and not args.include_timestamp_tag
    )
    if auto_timestamp:
        print(
            "Condor environment detected; enabling timestamp tagging to reduce output collisions."
        )
    if args.include_timestamp_tag or auto_timestamp:
        outdir_name = outdir_name + "_" + timestamp_tag
    save_dir_path = os.path.join(save_dir_path,outdir_name)
    dir_preexists = os.path.exists(save_dir_path)
    os.makedirs(save_dir_path, exist_ok=True)
    if dir_preexists:
        print(f"Reusing existing output directory: {save_dir_path}")
    else:
        print(f"Created output directory: {save_dir_path}")

    # Get and merge histograms from one or more input pkl files
    load_start_time = datetime.datetime.now()
    if args.verbose:
        path_preview = pkl_paths[:3]
        preview_msg = ", ".join(f"'{path}'" for path in path_preview)
        if len(pkl_paths) > 3:
            preview_msg += ", ..."
        print(
            f"[{load_start_time:%H:%M:%S}] Loading histograms from {len(pkl_paths)} input file(s): {preview_msg}"
        )
    hin_dict, merge_report = load_and_merge_histogram_pkls(
        pkl_paths,
        on_process_collision=args.on_process_collision,
        require_sumw2=True,
    )
    _emit_merge_report(merge_report, args.merge_report, save_dir_path)
    if args.cache_merged_pkl:
        _cache_merged_histograms(hin_dict, args.cache_merged_pkl, save_dir_path)
    if args.verbose:
        load_finish_time = datetime.datetime.now()
        print(
            "[{}] Histogram load+merge completed in {:.2f}s".format(
                load_finish_time.strftime("%H:%M:%S"),
                (load_finish_time - load_start_time).total_seconds(),
            )
        )
    if args.merge_only:
        print("Merge-only mode enabled, stopping after successful merge validation.")
        return 0

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
        channel_output=args.channel_output,
        enable_category_skips=args.enable_category_skips,
        report_zero_yields=args.report_zero_yields,
        rebin_plot_vars=rebin_plot_vars,
        negative_weight_report=args.negative_weight_report,
    )
    return 0


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_with_args(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
