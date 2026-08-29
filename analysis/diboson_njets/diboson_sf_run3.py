"""Run 3 diboson scale factors from nominal and second-moment histograms.

The consumer computes ``(data - background) / diboson`` after aggregating
source ``njets`` bins into the requested final bins.  Statistical propagation
is enabled by default and consumes the paired ``njets_sumw2`` histogram.  The
process partition is exact, exhaustive, and supplied only by YAML configuration.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import pickle
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import awkward as ak
import numpy as np
import yaml


logger = logging.getLogger(__name__)

ALL_YEARS_SENTINEL = "all"
FORMULA_IDENTIFIER = "independent_data_minus_background_over_diboson_v1"
MEMBERSHIP_MAP_VERSION = "source_bin_overlap_v1"
FLOW_POLICY = "exclude_underflow_overflow"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("diboson_sf_run3_config.yml")
ROLE_NAMES = ("data", "background", "diboson", "ignored")


class DibosonContractError(RuntimeError):
    """The configured diboson statistical contract is not satisfied."""


class resolved_diboson_input(NamedTuple):
    """One calculation year paired with its exact input and role config."""

    year: str
    pkl_path: str
    config_path: str
    shared_input: bool


def load_pkl_file(pkl_file: str) -> dict[str, Any]:
    with gzip.open(pkl_file, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise DibosonContractError(
            f"Diboson input must be a histogram dictionary: input={pkl_file!r} "
            f"observed_type={type(payload).__name__}."
        )
    return payload


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_identity(path: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def load_diboson_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, Mapping) or not isinstance(
        loaded.get("diboson"), Mapping
    ):
        raise DibosonContractError(
            f"Configuration {os.fspath(path)!r} must contain a 'diboson' mapping."
        )
    return dict(loaded["diboson"])


def resolve_propagation_state(
    diboson_config: Mapping[str, Any],
    cli_value: bool | None,
) -> tuple[bool, str]:
    """Resolve CLI > config > default propagation precedence."""

    if cli_value is not None:
        return bool(cli_value), "cli"
    if "propagate_statistical_uncertainties" in diboson_config:
        value = diboson_config["propagate_statistical_uncertainties"]
        if not isinstance(value, bool):
            raise DibosonContractError(
                "diboson.propagate_statistical_uncertainties must be boolean."
            )
        return value, "config"
    return True, "default"


def _normalize_process_roles(
    diboson_config: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    raw_roles = diboson_config.get("process_roles")
    if not isinstance(raw_roles, Mapping):
        raise DibosonContractError("diboson.process_roles must be a mapping.")
    unknown_roles = sorted(set(raw_roles) - set(ROLE_NAMES))
    missing_roles = sorted(set(ROLE_NAMES[:-1]) - set(raw_roles))
    if unknown_roles or missing_roles:
        raise DibosonContractError(
            "diboson.process_roles has invalid role keys: "
            f"missing={missing_roles} unknown={unknown_roles}."
        )

    normalized: dict[str, tuple[str, ...]] = {}
    for role in ROLE_NAMES:
        values = raw_roles.get(role, [])
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise DibosonContractError(
                f"diboson.process_roles.{role} must be a list of nonempty exact labels."
            )
        if len(values) != len(set(values)):
            duplicates = sorted(
                value for value in set(values) if values.count(value) > 1
            )
            raise DibosonContractError(
                f"diboson.process_roles.{role} contains duplicate labels: {duplicates}."
            )
        normalized[role] = tuple(values)

    for required_role in ROLE_NAMES[:-1]:
        if not normalized[required_role]:
            raise DibosonContractError(
                f"diboson.process_roles.{required_role} must not be empty."
            )

    memberships: dict[str, list[str]] = {}
    for role, processes in normalized.items():
        for process in processes:
            memberships.setdefault(process, []).append(role)
    overlaps = {
        process: roles for process, roles in memberships.items() if len(roles) > 1
    }
    if overlaps:
        raise DibosonContractError(
            "Diboson process roles must be pairwise disjoint: "
            + json.dumps(overlaps, sort_keys=True)
        )
    return normalized


def resolve_process_roles(
    diboson_config: Mapping[str, Any],
    *,
    available_processes: Sequence[str],
    selected_processes: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    roles = _normalize_process_roles(diboson_config)
    available = set(map(str, available_processes))
    selected = set(map(str, selected_processes))
    configured = {process for processes in roles.values() for process in processes}

    unknown = sorted(configured - available)
    if unknown:
        raise DibosonContractError(
            "Configured process-role labels are absent from the nominal histogram: "
            + ", ".join(unknown)
        )
    unclassified = sorted(selected - configured)
    if unclassified:
        raise DibosonContractError(
            "Selected nominal processes are unclassified: " + ", ".join(unclassified)
        )

    selected_roles = {
        role: tuple(process for process in processes if process in selected)
        for role, processes in roles.items()
    }
    empty_included = [
        role for role in ROLE_NAMES[:-1] if not selected_roles[role]
    ]
    if empty_included:
        raise DibosonContractError(
            "Selected input has no process for required role(s): "
            + ", ".join(empty_included)
        )
    return selected_roles


def _derive_process_subset_for_year(proc_list: Sequence[str], year: str) -> set[str]:
    year_str = str(year)
    pattern = re.compile(rf"(?<!\d){re.escape(year_str)}(?!\d)")
    matches = {str(proc) for proc in proc_list if pattern.search(str(proc))}
    if matches:
        return matches
    return {str(proc) for proc in proc_list if year_str in str(proc)}


def _map_year_tokens_to_processes(
    proc_list: Sequence[str],
) -> dict[str, set[str]]:
    pattern = re.compile(r"(?<!\d)(20\d{2}(?:[A-Za-z]+)?)(?!\d)")
    matches: dict[str, set[str]] = {}
    for process in proc_list:
        process_str = str(process)
        for token in pattern.findall(process_str):
            matches.setdefault(token, set()).add(process_str)
    return matches


def _axis_category(axis: Any, requested: str, *, axis_name: str) -> Any:
    for value in axis:
        if str(value) == str(requested):
            return value
    raise DibosonContractError(
        f"Axis {axis_name!r} does not contain requested category {requested!r}; "
        f"available={[str(value) for value in axis]}."
    )


def _prepare_histogram(histogram: Any, *, histogram_key: str) -> Any:
    wc_names = getattr(histogram, "wc_names", None)
    if wc_names is not None:
        if list(wc_names):
            raise DibosonContractError(
                f"Histogram {histogram_key!r} contains nonconstant EFT coefficients "
                f"{list(wc_names)}; the diboson consumer accepts scalar content only."
            )
        if hasattr(histogram, "as_hist"):
            histogram = histogram.as_hist({})
    return histogram


def _axis_signature(axis: Any) -> dict[str, Any]:
    traits = getattr(axis, "traits", None)
    common = {
        "name": axis.name,
        "type": type(axis).__name__,
        "underflow": bool(getattr(traits, "underflow", False)),
        "overflow": bool(getattr(traits, "overflow", False)),
        "growth": bool(getattr(traits, "growth", False)),
        "circular": bool(getattr(traits, "circular", False)),
    }
    if hasattr(axis, "edges"):
        common["edges"] = np.asarray(axis.edges, dtype=float).tolist()
    else:
        common["categories"] = [str(value) for value in axis]
    return common


def histogram_signature(histogram: Any, *, histogram_key: str) -> list[dict[str, Any]]:
    prepared = _prepare_histogram(histogram, histogram_key=histogram_key)
    return [_axis_signature(axis) for axis in prepared.axes]


def validate_histogram_pair(
    nominal_histogram: Any,
    second_moment_histogram: Any,
    *,
    nominal_key: str,
    second_moment_key: str,
) -> tuple[Any, Any]:
    nominal = _prepare_histogram(nominal_histogram, histogram_key=nominal_key)
    second_moment = _prepare_histogram(
        second_moment_histogram, histogram_key=second_moment_key
    )
    nominal_signature = [_axis_signature(axis) for axis in nominal.axes]
    second_moment_signature = [_axis_signature(axis) for axis in second_moment.axes]
    for signature in nominal_signature:
        if signature["name"] == nominal_key:
            signature["name"] = "__diboson_dense_axis__"
    for signature in second_moment_signature:
        if signature["name"] == second_moment_key:
            signature["name"] = "__diboson_dense_axis__"
    if nominal_signature != second_moment_signature:
        raise DibosonContractError(
            "Nominal and second-moment histogram axes/categories/edges/flow differ: "
            f"nominal_key={nominal_key!r} second_moment_key={second_moment_key!r} "
            f"nominal_signature={nominal_signature} "
            f"second_moment_signature={second_moment_signature}."
        )
    return nominal, second_moment


def build_bin_membership_map(
    histogram: Any,
    bins: Sequence[float],
    *,
    histogram_key: str,
) -> dict[str, Any]:
    if len(bins) < 2 or any(
        not np.isfinite(value) for value in np.asarray(bins, dtype=float)
    ):
        raise DibosonContractError("Final bin edges must be finite and nonempty.")
    if any(high <= low for low, high in zip(bins[:-1], bins[1:])):
        raise DibosonContractError("Final bin edges must be strictly increasing.")
    try:
        axis = histogram.axes[histogram_key]
    except Exception as error:
        raise DibosonContractError(
            f"Histogram {histogram_key!r} lacks its required dense axis."
        ) from error
    bin_edges_cache = np.asarray(axis.edges, dtype=float)
    if not np.all(np.isfinite(bin_edges_cache)):
        raise DibosonContractError(
            f"Histogram {histogram_key!r} has nonfinite source-bin edges."
        )
    unmatched_edges = [
        float(edge)
        for edge in bins
        if not np.any(np.isclose(bin_edges_cache, edge, rtol=0.0, atol=1e-12))
    ]
    if unmatched_edges:
        raise DibosonContractError(
            "Final bin boundaries must align with nominal source-bin edges: "
            f"unmatched={unmatched_edges} source_edges={bin_edges_cache.tolist()}."
        )
    low_edges = bin_edges_cache[:-1]
    high_edges = bin_edges_cache[1:]
    bin_index_cache = [
        np.nonzero((high_edges > low) & (low_edges < high))[0].tolist()
        for low, high in zip(bins[:-1], bins[1:])
    ]
    if any(not indices for indices in bin_index_cache):
        raise DibosonContractError(
            f"Final binning {list(bins)} has bins without nominal source coverage."
        )
    flattened = [index for indices in bin_index_cache for index in indices]
    if len(flattened) != len(set(flattened)):
        raise DibosonContractError("A nominal source bin maps to multiple final bins.")
    return {
        "version": MEMBERSHIP_MAP_VERSION,
        "flow_policy": FLOW_POLICY,
        "source_edges": bin_edges_cache.tolist(),
        "final_edges": [float(value) for value in bins],
        "final_bin_source_indices": bin_index_cache,
    }


def _extract_source_values(
    histogram: Any,
    *,
    histogram_key: str,
    process: str,
    channel_name: str,
    extra_slices: Mapping[str, Any] | None,
) -> np.ndarray:
    available_axes = {axis.name for axis in histogram.axes}
    required_axes = {"process", "channel", histogram_key}
    missing_axes = sorted(required_axes - available_axes)
    if missing_axes:
        raise DibosonContractError(
            f"Histogram {histogram_key!r} is missing required axes {missing_axes}; "
            f"available={sorted(available_axes)}."
        )

    selection: dict[str, Any] = {
        "process": _axis_category(
            histogram.axes["process"], process, axis_name="process"
        ),
        "channel": _axis_category(
            histogram.axes["channel"], channel_name, axis_name="channel"
        ),
    }
    for axis_name, value in (extra_slices or {}).items():
        if axis_name not in available_axes:
            raise DibosonContractError(
                f"Histogram {histogram_key!r} lacks required selection axis "
                f"{axis_name!r}."
            )
        selection[axis_name] = _axis_category(
            histogram.axes[axis_name], str(value), axis_name=axis_name
        )
    if "systematic" in available_axes and "systematic" not in selection:
        selection["systematic"] = _axis_category(
            histogram.axes["systematic"], "nominal", axis_name="systematic"
        )

    try:
        selected = histogram[selection]
        axis_names = [axis.name for axis in selected.axes]
        target_index = axis_names.index(histogram_key)
        values = ak.to_numpy(selected.values(flow=False))
    except Exception as error:
        raise DibosonContractError(
            f"Failed histogram extraction: histogram={histogram_key!r} "
            f"process={process!r} channel={channel_name!r} selection={selection}."
        ) from error
    sum_axes = tuple(index for index in range(values.ndim) if index != target_index)
    if sum_axes:
        values = values.sum(axis=sum_axes)
    values = np.asarray(values, dtype=float).reshape(-1)
    expected_bins = len(histogram.axes[histogram_key].edges) - 1
    if values.shape != (expected_bins,):
        raise DibosonContractError(
            f"Reduced histogram shape mismatch: histogram={histogram_key!r} "
            f"process={process!r} expected_bins={expected_bins} "
            f"observed_shape={values.shape}."
        )
    return values


def get_yields_in_bins(
    hin_dict: Mapping[str, Any],
    proc_list: Sequence[str],
    bins: Sequence[float],
    hist_name: str,
    channel_name: str,
    extra_slices: Mapping[str, Any] | None = None,
    process_whitelist: Sequence[str] | None = None,
    second_moment_hist_name: str | None = None,
    *,
    return_metadata: bool = False,
) -> Any:
    """Extract per-process final-bin nominal values and optional second moments.

    The tuple's second element is a second moment, not an uncertainty.  It is
    ``None`` when statistical propagation is explicitly disabled.
    """

    if hist_name not in hin_dict:
        if second_moment_hist_name and second_moment_hist_name in hin_dict:
            raise DibosonContractError(
                f"Orphan second-moment histogram {second_moment_hist_name!r}: "
                f"nominal histogram {hist_name!r} is absent."
            )
        raise DibosonContractError(
            f"Missing required nominal histogram key {hist_name!r}."
        )
    nominal = _prepare_histogram(hin_dict[hist_name], histogram_key=hist_name)
    second_moment = None
    if second_moment_hist_name is not None:
        if second_moment_hist_name not in hin_dict:
            raise DibosonContractError(
                f"Missing required second-moment histogram key "
                f"{second_moment_hist_name!r}."
            )
        nominal, second_moment = validate_histogram_pair(
            nominal,
            hin_dict[second_moment_hist_name],
            nominal_key=hist_name,
            second_moment_key=second_moment_hist_name,
        )

    available_processes = [str(process) for process in nominal.axes["process"]]
    requested = [str(process) for process in proc_list]
    if process_whitelist is not None:
        whitelist = set(map(str, process_whitelist))
        missing_whitelist = sorted(whitelist - set(available_processes))
        if missing_whitelist:
            raise DibosonContractError(
                "Requested process whitelist labels are absent: "
                + ", ".join(missing_whitelist)
            )
        requested = [process for process in requested if process in whitelist]
    missing_requested = sorted(set(requested) - set(available_processes))
    if missing_requested:
        raise DibosonContractError(
            "Requested nominal processes are absent: " + ", ".join(missing_requested)
        )
    if not requested:
        raise DibosonContractError("No processes remain for diboson extraction.")

    membership = build_bin_membership_map(
        nominal, bins, histogram_key=hist_name
    )
    bin_edges_cache = np.asarray(membership["source_edges"], dtype=float)
    bin_index_cache = membership["final_bin_source_indices"]
    if second_moment is not None:
        second_edges = np.asarray(
            second_moment.axes[second_moment_hist_name].edges, dtype=float
        )
        if not np.array_equal(bin_edges_cache, second_edges):
            raise DibosonContractError(
                "Nominal and second-moment source-bin caches disagree."
            )

    yields: dict[str, list[tuple[float, float | None]]] = {}
    for process in requested:
        nominal_source = _extract_source_values(
            nominal,
            histogram_key=hist_name,
            process=process,
            channel_name=channel_name,
            extra_slices=extra_slices,
        )
        if not np.all(np.isfinite(nominal_source)):
            raise DibosonContractError(
                f"Nonfinite nominal values: histogram={hist_name!r} "
                f"process={process!r} channel={channel_name!r}."
            )
        second_source = None
        if second_moment is not None:
            second_source = _extract_source_values(
                second_moment,
                histogram_key=second_moment_hist_name,
                process=process,
                channel_name=channel_name,
                extra_slices=extra_slices,
            )
            if not np.all(np.isfinite(second_source)) or np.any(second_source < 0):
                raise DibosonContractError(
                    f"Invalid second moments: histogram={second_moment_hist_name!r} "
                    f"process={process!r} channel={channel_name!r} "
                    f"values={second_source.tolist()}."
                )

        process_values = []
        for indices in bin_index_cache:
            nominal_value = float(np.sum(nominal_source[indices]))
            second_value = (
                None
                if second_source is None
                else float(np.sum(second_source[indices]))
            )
            process_values.append((nominal_value, second_value))
        yields[process] = process_values

    if return_metadata:
        return yields, membership
    return yields


def _component_arrays(
    yields: Mapping[str, Sequence[tuple[float, float | None]]],
    roles: Mapping[str, Sequence[str]],
    *,
    propagation_enabled: bool,
) -> dict[str, list[float] | None]:
    num_bins = len(next(iter(yields.values())))
    output: dict[str, list[float] | None] = {}
    for role, nominal_name, variance_name in (
        ("data", "data", "var_data"),
        ("background", "background", "var_background"),
        ("diboson", "diboson", "var_diboson"),
    ):
        nominal = [0.0] * num_bins
        variance = [0.0] * num_bins if propagation_enabled else None
        for process in roles[role]:
            for index, (value, second_moment) in enumerate(yields[process]):
                nominal[index] += value
                if propagation_enabled:
                    if second_moment is None:
                        raise DibosonContractError(
                            f"Missing enabled second moment for process {process!r}."
                        )
                    variance[index] += second_moment
        output[nominal_name] = nominal
        output[variance_name] = variance
    return output


def compute_scale_factor_statistics(
    components: Mapping[str, Sequence[float] | None],
    bins: Sequence[float],
    *,
    input_path: str,
    year: str,
    channel: str,
    propagation_enabled: bool,
) -> tuple[list[float], list[float] | None, list[float] | None]:
    scale_factors: list[float] = []
    variances: list[float] | None = [] if propagation_enabled else None
    uncertainties: list[float] | None = [] if propagation_enabled else None
    data = components["data"]
    background = components["background"]
    diboson = components["diboson"]
    var_data = components["var_data"]
    var_background = components["var_background"]
    var_diboson = components["var_diboson"]

    for index, (d, b, v) in enumerate(zip(data, background, diboson)):
        bounds = [float(bins[index]), float(bins[index + 1])]
        if not all(np.isfinite(value) for value in (d, b, v)) or v <= 0:
            raise DibosonContractError(
                "Invalid diboson denominator: "
                f"input={input_path!r} year={year!r} channel={channel!r} "
                f"final_bin={bounds} data={d} background={b} diboson={v}."
            )
        numerator = d - b
        scale_factors.append(float(numerator / v))
        if propagation_enabled:
            vd = var_data[index]
            vb = var_background[index]
            vv = var_diboson[index]
            if not all(np.isfinite(value) and value >= 0 for value in (vd, vb, vv)):
                raise DibosonContractError(
                    "Invalid aggregated second moments: "
                    f"input={input_path!r} year={year!r} channel={channel!r} "
                    f"final_bin={bounds} var_data={vd} var_background={vb} "
                    f"var_diboson={vv}."
                )
            variance = (vd + vb) / (v**2) + ((numerator**2) / (v**4)) * vv
            if not np.isfinite(variance) or variance < 0:
                raise DibosonContractError(
                    f"Computed invalid scale-factor variance {variance} for "
                    f"input={input_path!r} year={year!r} channel={channel!r} "
                    f"final_bin={bounds}."
                )
            variances.append(float(variance))
            uncertainties.append(float(np.sqrt(variance)))
    return scale_factors, variances, uncertainties


def compute_linear_fit(
    bin_centers: Sequence[float], scale_factors: Sequence[float]
) -> tuple[dict[str, float] | None, list[float]]:
    if bin_centers is None or scale_factors is None:
        return None, []
    if np.size(bin_centers) == 0 or np.size(scale_factors) == 0:
        return None, []
    if len(bin_centers) != len(scale_factors):
        return None, []
    slope, intercept = np.polyfit(bin_centers, scale_factors, deg=1)
    fitted_values = np.polyval([slope, intercept], bin_centers)
    return (
        {"slope": float(slope), "intercept": float(intercept)},
        np.atleast_1d(fitted_values).tolist(),
    )


def process_year(
    pkl_path: str,
    year: str,
    hist_name: str,
    channel: str,
    bins: Sequence[float],
    *,
    process_roles: Mapping[str, Sequence[str]],
    propagation_enabled: bool,
    configuration_source: str,
    cache: dict[str, dict[str, Any]] | None = None,
    allowed_years: Sequence[str] | None = None,
) -> dict[str, Any]:
    if hist_name != "njets":
        raise DibosonContractError(
            "The maintained diboson contract requires hist_name='njets'."
        )
    if cache is not None and pkl_path in cache:
        histograms = cache[pkl_path]
    else:
        histograms = load_pkl_file(pkl_path)
        if cache is not None:
            cache[pkl_path] = histograms

    second_moment_key = f"{hist_name}_sumw2"
    if hist_name not in histograms:
        if propagation_enabled and second_moment_key in histograms:
            raise DibosonContractError(
                f"Orphan second-moment histogram {second_moment_key!r}: "
                f"nominal histogram {hist_name!r} is absent."
            )
        raise DibosonContractError(
            f"Missing required nominal histogram key {hist_name!r}."
        )
    nominal = _prepare_histogram(histograms[hist_name], histogram_key=hist_name)
    available_processes = [str(value) for value in nominal.axes["process"]]

    if allowed_years:
        filter_tokens = (
            set(map(str, allowed_years))
            if str(year).lower() == ALL_YEARS_SENTINEL
            else {str(year)}
        )
        selected_set: set[str] = set()
        for token in filter_tokens:
            selected_set.update(
                _derive_process_subset_for_year(available_processes, token)
            )
        if not selected_set:
            raise DibosonContractError(
                f"No processes remain after year filtering: year={year!r}."
            )
        selected_processes = [
            process for process in available_processes if process in selected_set
        ]
    else:
        selected_processes = available_processes

    selected_roles = resolve_process_roles(
        {
            "process_roles": {
                role: list(processes) for role, processes in process_roles.items()
            }
        },
        available_processes=available_processes,
        selected_processes=selected_processes,
    )
    extra_slices: dict[str, Any] = {}
    available_axes = {axis.name for axis in nominal.axes}
    if "year" in available_axes and str(year).lower() != ALL_YEARS_SENTINEL:
        extra_slices["year"] = str(year)

    logger.info(
        "diboson_configuration enabled=%s configuration_source=%s "
        "roles=%s nominal_key=%s second_moment_key=%s input=%s year=%s "
        "channel=%s final_binning=%s",
        propagation_enabled,
        configuration_source,
        {role: list(values) for role, values in selected_roles.items()},
        hist_name,
        second_moment_key,
        pkl_path,
        year,
        channel,
        list(bins),
    )

    try:
        yields, membership = get_yields_in_bins(
            histograms,
            available_processes,
            bins,
            hist_name,
            channel,
            extra_slices=extra_slices,
            process_whitelist=selected_processes,
            second_moment_hist_name=(
                second_moment_key if propagation_enabled else None
            ),
            return_metadata=True,
        )
        components = _component_arrays(
            yields, selected_roles, propagation_enabled=propagation_enabled
        )
        scale_factors, variances, uncertainties = compute_scale_factor_statistics(
            components,
            bins,
            input_path=pkl_path,
            year=str(year),
            channel=channel,
            propagation_enabled=propagation_enabled,
        )
    except Exception:
        logger.exception(
            "diboson_validation outcome=failed input=%s year=%s channel=%s",
            pkl_path,
            year,
            channel,
        )
        raise

    logger.info(
        "diboson_validation outcome=passed statistical_inputs_consumed=%s",
        propagation_enabled,
    )
    bin_centers = [
        (float(bins[index]) + float(bins[index + 1])) / 2
        for index in range(len(bins) - 1)
    ]
    fit_coefficients, fitted_values = compute_linear_fit(
        bin_centers, scale_factors
    )
    provenance = {
        "input_identity": _input_identity(pkl_path),
        "year": str(year),
        "channel": channel,
        "process_roles": {
            role: list(processes) for role, processes in selected_roles.items()
        },
        "final_binning": [float(value) for value in bins],
        "source_to_final_bin_membership": membership,
        "statistical_inputs_consumed": propagation_enabled,
        "validation_outcome": "passed",
        "fit_weighting": "unweighted_linear_fit_preserved",
    }
    return {
        "scale_factors": scale_factors,
        "scale_factor_statistical_variances": variances,
        "scale_factor_statistical_uncertainties": uncertainties,
        "fit_coefficients": fit_coefficients,
        "bin_centers": bin_centers,
        "fitted_values": fitted_values,
        "diboson": components["diboson"],
        "data": components["data"],
        "other": components["background"],
        "var_data": components["var_data"],
        "var_background": components["var_background"],
        "var_diboson": components["var_diboson"],
        "propagation_enabled": propagation_enabled,
        "configuration_source": configuration_source,
        "provenance": provenance,
    }


def make_diboson_sf_json(
    bins: Sequence[float],
    result: Mapping[str, Any],
    year: str,
    output_dir: str = ".",
) -> str:
    scale_factors = result["scale_factors"]
    if len(bins) != len(scale_factors) + 1:
        raise ValueError(
            "Number of scale factors must be one less than number of bin edges."
        )
    enabled = bool(result["propagation_enabled"])
    key_name = f"dibosonSF_njets_{year}"
    propagation = {
        "enabled": enabled,
        "formula": FORMULA_IDENTIFIER if enabled else None,
        "nominal_histogram_key": "njets",
        "second_moment_histogram_key": "njets_sumw2",
        "configuration_source": result["configuration_source"],
        **result["provenance"],
    }
    payload = {
        key_name: {
            f"[{bins[index]},{bins[index + 1]}]": scale_factors[index]
            for index in range(len(scale_factors))
        },
        "statistical_uncertainty_propagation": propagation,
        "scale_factor_statistical_variances": result[
            "scale_factor_statistical_variances"
        ],
        "scale_factor_statistical_uncertainties": result[
            "scale_factor_statistical_uncertainties"
        ],
    }
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"diboson_sf_{year}.json")
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    logger.info("wrote_json path=%s", output_path)
    return os.path.abspath(output_path)


def save_linear_fit_coefficients(
    year: str,
    fit_coefficients: Mapping[str, float] | None,
    output_dir: str = ".",
) -> str | None:
    if not fit_coefficients:
        return None
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"diboson_sf_{year}_linear_fit.json"
    )
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(dict(fit_coefficients), stream, indent=2, sort_keys=True)
    logger.info("wrote_fit_json path=%s", output_path)
    return os.path.abspath(output_path)


def save_scale_factor_plot(
    year: str,
    channel: str,
    bin_centers: Sequence[float],
    scale_factors: Sequence[float],
    fitted_values: Sequence[float],
    statistical_uncertainties: Sequence[float] | None,
    *,
    propagation_enabled: bool,
    output_dir: str = ".",
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    if (
        bin_centers is None
        or scale_factors is None
        or np.size(bin_centers) == 0
        or np.size(scale_factors) == 0
    ):
        raise DibosonContractError("No scale-factor points are available for plotting.")
    os.makedirs(output_dir, exist_ok=True)
    fig, axis = plt.subplots()
    centers = np.asarray(bin_centers, dtype=float)
    values = np.asarray(scale_factors, dtype=float)
    annotation = None
    y_errors = None
    if propagation_enabled:
        if statistical_uncertainties is None:
            raise DibosonContractError(
                "Enabled plotting requires statistical uncertainties."
            )
        y_errors = np.asarray(statistical_uncertainties, dtype=float)
        if y_errors.shape != values.shape:
            raise DibosonContractError(
                "Plot uncertainty ordering does not match central values."
            )
        axis.errorbar(
            centers, values, yerr=y_errors, fmt="o", label="Scale factors"
        )
    else:
        axis.plot(centers, values, "o", label="Scale factors")
        annotation = "statistical uncertainties disabled"
        axis.text(
            0.5,
            0.04,
            annotation,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
        )
    if fitted_values is not None and np.size(fitted_values) > 0:
        axis.plot(centers, fitted_values, label="Linear fit")
    axis.set_xlabel("N_{jets} bin center")
    axis.set_ylabel("Scale factor")
    axis.set_title(f"Diboson scale factors ({year}, {channel})")
    axis.legend()
    output_path = os.path.join(output_dir, f"diboson_sf_{year}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "path": os.path.abspath(output_path),
        "statistical_error_bars": propagation_enabled,
        "y_errors": None if y_errors is None else y_errors.tolist(),
        "annotation": annotation,
    }
    logger.info("wrote_plot metadata=%s", metadata)
    return metadata


def _has_year_template(
    path: str,
    *,
    option_name: str,
    parser: argparse.ArgumentParser,
) -> bool:
    has_year = "{year}" in path
    unresolved = path.replace("{year}", "")
    if "{" in unresolved or "}" in unresolved:
        parser.error(
            f"{option_name} template supports only the literal '{{year}}' "
            f"placeholder; received {path!r}."
        )
    return has_year


def _resolve_config_paths(
    raw_config_paths: Sequence[str],
    years: Sequence[str],
    *,
    shared_input: bool,
    has_all_years: bool,
    parser: argparse.ArgumentParser,
) -> list[str]:
    config_paths = [str(path) for path in raw_config_paths]
    if not config_paths:
        parser.error("At least one path must be provided via --config.")
    template_flags = [
        _has_year_template(
            path,
            option_name="--config",
            parser=parser,
        )
        for path in config_paths
    ]
    if has_all_years and any(template_flags):
        parser.error("--year all cannot be used with a template --config path.")
    if any(template_flags) and (len(config_paths) != 1 or sum(template_flags) != 1):
        parser.error(
            "A template --config must be the only config argument and contain "
            "the literal {year} placeholder."
        )

    if shared_input:
        if len(config_paths) != 1 or template_flags[0]:
            parser.error("A shared input requires exactly one non-template --config path.")
        resolved = [config_paths[0]] * len(years)
    elif template_flags[0]:
        resolved = [config_paths[0].format(year=year) for year in years]
    elif len(config_paths) == len(years):
        resolved = list(config_paths)
    elif len(config_paths) == 1 and len(years) > 1:
        parser.error(
            "Multiple independent input files require one matching --config per "
            "input, or a --config path containing {year}."
        )
    else:
        parser.error(
            "Number of --config paths must match the number of independent "
            "input/year records."
        )

    missing = list(dict.fromkeys(path for path in resolved if not os.path.exists(path)))
    if missing:
        parser.error(
            "Resolved config path(s) do not exist: " + ", ".join(missing)
        )
    return [str(Path(path).resolve()) for path in resolved]


def _resolve_cli_inputs(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[
    list[resolved_diboson_input],
    dict[str, list[str]],
    list[str],
    dict[str, dict[str, Any]],
]:
    years = [str(year) for year in args.year]
    if not years:
        parser.error("At least one year must be provided via -y/--year.")
    has_all_years = any(year.lower() == ALL_YEARS_SENTINEL for year in years)
    requested_specific_years = [
        year for year in years if year.lower() != ALL_YEARS_SENTINEL
    ]
    shared = False
    year_process_map: dict[str, list[str]] = {}
    raw_config_paths = (
        [str(args.config)]
        if isinstance(args.config, (str, os.PathLike))
        else [str(path) for path in args.config]
    )
    config_template_flags = [
        _has_year_template(
            path,
            option_name="--config",
            parser=parser,
        )
        for path in raw_config_paths
    ]
    if has_all_years and any(config_template_flags):
        parser.error("--year all cannot be used with a template --config path.")

    if len(args.pkl) == 1:
        pkl_arg = args.pkl[0]
        pkl_is_template = _has_year_template(
            pkl_arg,
            option_name="--pkl",
            parser=parser,
        )
        if pkl_is_template:
            if has_all_years:
                parser.error("--year all cannot be used with a template --pkl path.")
            pkl_paths = [pkl_arg.format(year=year) for year in years]
        elif len(years) == 1:
            pkl_paths = [pkl_arg]
            shared = has_all_years
        else:
            pkl_paths = [pkl_arg] * len(years)
            shared = True
    elif len(args.pkl) == len(years):
        pkl_paths = list(args.pkl)
    else:
        parser.error(
            "Number of --pkl paths must match years unless one shared/template path is used."
        )
    missing = [path for path in set(pkl_paths) if not os.path.exists(path)]
    if missing:
        parser.error("Input pickle path(s) do not exist: " + ", ".join(sorted(missing)))

    cache: dict[str, dict[str, Any]] = {}
    if shared:
        sample = load_pkl_file(pkl_paths[0])
        cache[pkl_paths[0]] = sample
        if args.hist_name not in sample:
            parser.error(f"Histogram {args.hist_name!r} is absent from the shared input.")
        try:
            process_values = list(sample[args.hist_name].axes["process"])
        except Exception:
            parser.error("Shared nominal histogram lacks the process axis.")
        token_map = _map_year_tokens_to_processes(process_values)
        year_process_map = {
            token: sorted(processes) for token, processes in token_map.items()
        }
        discovered = sorted(year_process_map)
        if has_all_years and years == [ALL_YEARS_SENTINEL] and discovered:
            years = discovered + [ALL_YEARS_SENTINEL]
            pkl_paths = [pkl_paths[0]] * len(years)
            requested_specific_years = discovered
        missing_years = [
            year for year in requested_specific_years if year not in year_process_map
        ]
        if missing_years:
            parser.error(
                "Shared input lacks requested year token(s): "
                + ", ".join(missing_years)
            )
    if any(year.lower() == ALL_YEARS_SENTINEL for year in years) and not shared:
        parser.error("--year all requires one shared input pickle.")

    config_paths = _resolve_config_paths(
        raw_config_paths,
        years,
        shared_input=shared,
        has_all_years=has_all_years,
        parser=parser,
    )
    records = [
        resolved_diboson_input(
            year=year,
            pkl_path=pkl_path,
            config_path=config_path,
            shared_input=shared,
        )
        for year, pkl_path, config_path in zip(years, pkl_paths, config_paths)
    ]
    return records, year_process_map, requested_specific_years, cache


def _load_resolved_configs(
    records: Sequence[resolved_diboson_input],
    cli_value: bool | None,
) -> dict[str, dict[str, Any]]:
    config_cache: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.config_path in config_cache:
            continue
        diboson_config = load_diboson_config(record.config_path)
        process_roles = _normalize_process_roles(diboson_config)
        propagation_enabled, configuration_source = resolve_propagation_state(
            diboson_config,
            cli_value,
        )
        config_cache[record.config_path] = {
            "diboson_config": diboson_config,
            "process_roles": process_roles,
            "propagation_enabled": propagation_enabled,
            "configuration_source": configuration_source,
        }

    if cli_value is None:
        resolved_states = {
            bool(config["propagation_enabled"])
            for config in config_cache.values()
        }
        if len(resolved_states) > 1:
            details = [
                {
                    "config_path": path,
                    "enabled": bool(config["propagation_enabled"]),
                    "configuration_source": str(config["configuration_source"]),
                }
                for path, config in config_cache.items()
            ]
            raise DibosonContractError(
                "Assigned diboson configs resolve inconsistent statistical "
                "propagation states without a CLI override: "
                + json.dumps(details, sort_keys=True)
            )
    return config_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", nargs="+", required=True)
    parser.add_argument(
        "--config",
        nargs="+",
        default=[str(DEFAULT_CONFIG_PATH)],
        metavar="CONFIG",
    )
    parser.add_argument("--hist-name", default="njets")
    parser.add_argument("--channel", default="3l_CR")
    parser.add_argument(
        "-y",
        "--year",
        nargs="+",
        default=["2022", "2022EE", "2023", "2023BPix"],
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument(
        "--propagate-statistical-uncertainties",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.hist_name != "njets":
        parser.error("The maintained contract accepts only --hist-name njets.")
    records, year_process_map, requested_years, cache = _resolve_cli_inputs(
        args,
        parser,
    )
    config_cache = _load_resolved_configs(
        records,
        args.propagate_statistical_uncertainties,
    )
    for record in records:
        if record.pkl_path not in cache:
            cache[record.pkl_path] = load_pkl_file(record.pkl_path)
    bins = [0, 1, 2, 3, 4, 5, 6]

    results: dict[str, dict[str, Any]] = {}
    for record in records:
        resolved_config = config_cache[record.config_path]
        propagation_enabled = bool(resolved_config["propagation_enabled"])
        configuration_source = str(resolved_config["configuration_source"])
        process_roles = resolved_config["process_roles"]
        allowed_years = None
        if record.shared_input:
            if record.year.lower() != ALL_YEARS_SENTINEL:
                allowed_years = [record.year]
            else:
                allowed_years = requested_years or sorted(year_process_map)
        logger.info(
            "resolved_diboson_input year=%s pkl_path=%s config_path=%s "
            "shared_input=%s propagation_enabled=%s configuration_source=%s "
            "role_counts=%s roles=%s",
            record.year,
            record.pkl_path,
            record.config_path,
            record.shared_input,
            propagation_enabled,
            configuration_source,
            {role: len(processes) for role, processes in process_roles.items()},
            {role: list(processes) for role, processes in process_roles.items()},
        )
        try:
            results[record.year] = process_year(
                record.pkl_path,
                record.year,
                args.hist_name,
                args.channel,
                bins,
                process_roles=process_roles,
                propagation_enabled=propagation_enabled,
                configuration_source=configuration_source,
                cache=cache,
                allowed_years=allowed_years,
            )
        except DibosonContractError as error:
            raise DibosonContractError(
                "Diboson input/config pair failed: "
                f"year={record.year!r} pkl_path={record.pkl_path!r} "
                f"config_path={record.config_path!r}: {error}"
            ) from error

    # All validation and numerical calculation is complete before any final output.
    for record in records:
        result = results[record.year]
        year_output_dir = os.path.join(args.output_dir, record.year)
        make_diboson_sf_json(bins, result, record.year, year_output_dir)
        save_linear_fit_coefficients(
            record.year, result["fit_coefficients"], year_output_dir
        )
        result["plot_metadata"] = save_scale_factor_plot(
            record.year,
            args.channel,
            result["bin_centers"],
            result["scale_factors"],
            result["fitted_values"],
            result["scale_factor_statistical_uncertainties"],
            propagation_enabled=bool(result["propagation_enabled"]),
            output_dir=year_output_dir,
        )

    if results:
        print("\nSummary of scale factor results:")
        header = f"{'Year':<8}{'Mean SF':>12}{'Slope':>12}{'Intercept':>14}"
        print(header)
        print("-" * len(header))
        for record in records:
            result = results[record.year]
            coefficients = result["fit_coefficients"] or {}
            print(
                f"{record.year:<8}{float(np.mean(result['scale_factors'])):>12.6f}"
                f"{coefficients.get('slope', float('nan')):>12.6f}"
                f"{coefficients.get('intercept', float('nan')):>14.6f}"
            )
    return results


if __name__ == "__main__":
    main()
