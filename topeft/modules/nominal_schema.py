"""Access, validation, merge, and transient views for nominal histogram schemas."""

from __future__ import annotations

import copy
import warnings
from collections import OrderedDict
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from itertools import product
from typing import Any

import boost_histogram as bh
import numpy as np

from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist

from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.data_driven_products import (
    CANONICAL_DATA_DRIVEN_YEARS,
    data_driven_product_error,
    parse_process_name,
)
from topeft.modules.sumw2_policy import resolved_sumw2_policy


NOMINAL_CONTAINER_SCHEMA_VERSION = 2
NOMINAL_CONTAINER_LAYOUT = "split_sibling_v1"
# Schema version 3 keeps the split-sibling key layout of version 2 but stores
# nominal statistical sumw2 inside each histogram (MultiCell HistEFT for 1D,
# Weight-storage SparseHist for 2D) instead of in ``<family>_sumw2`` companions.
# It is validated by ``validate_embedded_nominal_mapping``; the active producer
# keeps writing version 2 until the multicell migration switches it.
EMBEDDED_SUMW2_SCHEMA_VERSION = 3
EMBEDDED_SUMW2_LAYOUT = "split_sibling_embedded_sumw2_v1"
_SPLIT_SIBLING_SCHEMA_VERSIONS = frozenset(
    {NOMINAL_CONTAINER_SCHEMA_VERSION, EMBEDDED_SUMW2_SCHEMA_VERSION}
)
SCALAR_NOMINAL_SUFFIX = "__scalar_nominal"
EFT_NOMINAL_SUFFIX = "__eft_nominal"
SUMW2_SUFFIX = "_sumw2"
YEAR_COVERAGE_POLICIES = frozenset({"warn", "error", "off"})


def scalar_nominal_key(family: str) -> str:
    return f"{family}{SCALAR_NOMINAL_SUFFIX}"


def eft_nominal_key(family: str) -> str:
    return f"{family}{EFT_NOMINAL_SUFFIX}"


def sumw2_key(family: str) -> str:
    return f"{family}{SUMW2_SUFFIX}"


def family_from_component_key(key: str) -> str | None:
    for suffix in (SCALAR_NOMINAL_SUFFIX, EFT_NOMINAL_SUFFIX, SUMW2_SUFFIX):
        if key.endswith(suffix) and len(key) > len(suffix):
            return key[: -len(suffix)]
    return None


def canonical_process_year(process_name: str) -> str | None:
    """Return the maintained canonical year for one structural process label."""

    try:
        _base_name, year = parse_process_name(str(process_name))
    except data_driven_product_error:
        return None
    return year


def year_independent_process(process_name: str) -> str:
    """Match the logical process identity used by ``DatacardMaker.get_process``."""

    value = str(process_name)
    try:
        value, _year = parse_process_name(value)
    except data_driven_product_error:
        pass
    if "_lin_" in value or "_quad_" in value:
        value = value.split("_", 1)[0]
    if "_" in value:
        value = value.rsplit("_", 1)[0]
    return value


def histogram_categorical_support(histogram: Any) -> frozenset[tuple[tuple[str, Any], ...]]:
    """Return structurally materialized categorical cells without reading yields."""

    categorical_axes = tuple(getattr(histogram, "categorical_axes", ()))
    axis_names = tuple(axis.name for axis in categorical_axes)
    if not axis_names:
        return frozenset()
    categorical_keys = getattr(histogram, "categorical_keys", None)
    if categorical_keys is None:
        categorical_keys = product(*(tuple(axis) for axis in categorical_axes))
    return frozenset(
        tuple(zip(axis_names, tuple(key))) for key in categorical_keys
    )


def histogram_contribution_support(
    histograms: Mapping[str, Any],
) -> frozenset[tuple[str, tuple[tuple[str, Any], ...]]]:
    """Identify exact payload contributions merged by histogram addition."""

    return frozenset(
        (key, categorical_cell)
        for key, histogram in histograms.items()
        for categorical_cell in histogram_categorical_support(histogram)
    )


def _canonical_year_sort_key(year: str) -> tuple[int, str]:
    try:
        return (CANONICAL_DATA_DRIVEN_YEARS.index(year), year)
    except ValueError:
        return (len(CANONICAL_DATA_DRIVEN_YEARS), year)


def year_coverage_mismatches(
    histograms: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Collect structural year coverage by nominal family/channel/process slice."""

    slice_process_years: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for key, histogram in histograms.items():
        if key.endswith(SUMW2_SUFFIX):
            continue
        family = family_from_component_key(key) or key
        for categorical_cell in histogram_categorical_support(histogram):
            coordinates = dict(categorical_cell)
            if "process" not in coordinates or "channel" not in coordinates:
                continue
            year = canonical_process_year(str(coordinates["process"]))
            if year is None:
                continue
            logical_process = year_independent_process(str(coordinates["process"]))
            slice_process_years[(family, str(coordinates["channel"]))][
                logical_process
            ].add(year)

    mismatches = []
    for (family, channel), process_years in sorted(slice_process_years.items()):
        slice_years = set().union(*process_years.values())
        for process, represented_years in sorted(process_years.items()):
            if represented_years == slice_years:
                continue
            mismatches.append(
                {
                    "family": family,
                    "channel": channel,
                    "process": process,
                    "slice_years": sorted(slice_years, key=_canonical_year_sort_key),
                    "process_years": sorted(
                        represented_years, key=_canonical_year_sort_key
                    ),
                    "missing_years": sorted(
                        slice_years - represented_years,
                        key=_canonical_year_sort_key,
                    ),
                }
            )
    return mismatches


def validate_year_coverage(
    histograms: Mapping[str, Any],
    *,
    policy: str = "warn",
) -> list[dict[str, Any]]:
    """Apply the shared non-mutating structural year-coverage policy."""

    if policy not in YEAR_COVERAGE_POLICIES:
        raise ValueError(
            f"Unknown year coverage policy {policy!r}; expected one of "
            f"{sorted(YEAR_COVERAGE_POLICIES)!r}."
        )
    if policy == "off":
        return []
    mismatches = year_coverage_mismatches(histograms)
    if not mismatches:
        return []
    mismatches_by_slice = defaultdict(list)
    for mismatch in mismatches:
        mismatches_by_slice[(mismatch["family"], mismatch["channel"])].append(
            mismatch
        )
    sections = []
    for (family, channel), slice_mismatches in mismatches_by_slice.items():
        lines = [
            f"Year-coverage mismatch for {family} / {channel}:",
            "  slice years: " + ", ".join(slice_mismatches[0]["slice_years"]),
        ]
        for mismatch in slice_mismatches:
            lines.extend(
                (
                    "  {process} years: {years}".format(
                        process=mismatch["process"],
                        years=", ".join(mismatch["process_years"]),
                    ),
                    "    missing: " + ", ".join(mismatch["missing_years"]),
                )
            )
        lines.append(f"  policy: {policy}")
        sections.append("\n".join(lines))
    message = "\n\n".join(sections)
    if policy == "error":
        raise ValueError(message)
    warnings.warn(message, UserWarning, stacklevel=2)
    return mismatches


def is_split_nominal_mapping(histograms: Mapping[str, Any]) -> bool:
    return any(
        key.endswith(SCALAR_NOMINAL_SUFFIX) or key.endswith(EFT_NOMINAL_SUFFIX)
        for key in histograms
    )


def _dimensionality(family: str) -> int:
    return 2 if family in axes_info_2d else 1


def _require_histogram_mapping(histograms: Mapping[str, Any]) -> None:
    if not isinstance(histograms, Mapping):
        raise TypeError("Histogram payload must be a mapping.")
    if any(not isinstance(key, str) for key in histograms):
        raise TypeError("Histogram payload keys must be strings.")


def _dense_axes(histogram: Any) -> tuple[Any, ...]:
    if type(histogram) is HistEFT:
        return (histogram.dense_axis,)
    if type(histogram) is SparseHist:
        return tuple(histogram.dense_axes)
    raise TypeError(
        "Nominal schema supports exact SparseHist or HistEFT objects; got "
        f"{type(histogram).__name__}."
    )


def _categorical_axes(histogram: Any) -> tuple[Any, ...]:
    if type(histogram) not in {SparseHist, HistEFT}:
        raise TypeError(
            "Nominal schema supports exact SparseHist or HistEFT objects; got "
            f"{type(histogram).__name__}."
        )
    return tuple(histogram.categorical_axes)


def _axis_flow(axis: Any) -> tuple[bool, bool]:
    traits = getattr(axis, "traits", None)
    return (
        bool(getattr(traits, "underflow", False)),
        bool(getattr(traits, "overflow", False)),
    )


def _axis_edges(axis: Any) -> np.ndarray | None:
    try:
        return np.asarray(axis.edges, dtype=float)
    except Exception:
        try:
            return np.asarray(axis.edges(), dtype=float)
        except Exception:
            return None


def _normalized_dense_name(name: str) -> str:
    return name[: -len(SUMW2_SUFFIX)] if name.endswith(SUMW2_SUFFIX) else name


def _axis_signature(axis: Any, *, normalize_sumw2_name: bool = False) -> tuple[Any, ...]:
    name = axis.name
    if normalize_sumw2_name:
        name = _normalized_dense_name(name)
    edges = _axis_edges(axis)
    return (
        type(axis),
        name,
        tuple(edges.tolist()) if edges is not None else None,
        _axis_flow(axis),
    )


def validate_histogram_compatibility(
    first: Any,
    second: Any,
    *,
    key: str,
    normalize_sumw2_name: bool = False,
) -> None:
    if type(first) is not type(second):
        raise ValueError(
            f"Concrete histogram type mismatch for '{key}': "
            f"{type(first).__name__} != {type(second).__name__}."
        )
    first_categories = _categorical_axes(first)
    second_categories = _categorical_axes(second)
    first_category_signature = tuple((type(axis), axis.name) for axis in first_categories)
    second_category_signature = tuple((type(axis), axis.name) for axis in second_categories)
    if first_category_signature != second_category_signature:
        raise ValueError(f"Categorical axes differ for '{key}'.")
    first_dense = tuple(
        _axis_signature(axis, normalize_sumw2_name=normalize_sumw2_name)
        for axis in _dense_axes(first)
    )
    second_dense = tuple(
        _axis_signature(axis, normalize_sumw2_name=normalize_sumw2_name)
        for axis in _dense_axes(second)
    )
    if first_dense != second_dense:
        raise ValueError(f"Dense axes differ for '{key}'.")
    if type(first) is HistEFT and list(first.wc_names) != list(second.wc_names):
        raise ValueError(f"WC ordering differs for '{key}'.")


def _validate_base_companion_axes(base: Any, companion: SparseHist, family: str) -> None:
    base_categories = tuple((type(axis), axis.name) for axis in _categorical_axes(base))
    companion_categories = tuple(
        (type(axis), axis.name) for axis in _categorical_axes(companion)
    )
    if base_categories != companion_categories:
        raise ValueError(f"Categorical axes differ for '{family}' and its companion.")
    base_dense = tuple(
        _axis_signature(axis, normalize_sumw2_name=True) for axis in _dense_axes(base)
    )
    companion_dense = tuple(
        _axis_signature(axis, normalize_sumw2_name=True)
        for axis in _dense_axes(companion)
    )
    if base_dense != companion_dense:
        raise ValueError(f"Dense axes differ for '{family}' and its companion.")


def _process_labels(histogram: Any) -> frozenset[str]:
    try:
        return frozenset(str(value) for value in histogram.axes["process"])
    except Exception:
        return frozenset()


def _channel_labels(histogram: Any) -> frozenset[str]:
    try:
        return frozenset(str(value) for value in histogram.axes["channel"])
    except Exception:
        return frozenset()


def _family_channel_applicability(
    histogram_applicability: Mapping[str, Any],
    *,
    runtime_families: Iterable[str],
) -> dict[str, dict[str, str]]:
    if not isinstance(histogram_applicability, Mapping):
        raise TypeError("Histogram applicability must be a mapping.")
    families = histogram_applicability.get("families")
    if not isinstance(families, Mapping):
        raise ValueError("Histogram applicability lacks a families mapping.")
    runtime_families = tuple(runtime_families)
    if list(families) != list(runtime_families):
        raise ValueError(
            "Histogram applicability family order must match runtime families: "
            f"expected={list(runtime_families)} observed={list(families)}."
        )
    normalized = {}
    for family, family_contract in families.items():
        if not isinstance(family_contract, Mapping) or set(family_contract) != {
            "channels"
        }:
            raise ValueError(
                f"Histogram applicability family {family!r} must contain only channels."
            )
        channels = family_contract["channels"]
        if not isinstance(channels, Mapping):
            raise ValueError(
                f"Histogram applicability channels for family {family!r} must be a mapping."
            )
        normalized_channels = {}
        for channel, state in channels.items():
            if not isinstance(channel, str) or not channel:
                raise ValueError("Histogram applicability channel names must be nonempty strings.")
            if state not in {"applicable", "not_applicable"}:
                raise ValueError(
                    "Histogram applicability states are binary; "
                    f"family={family!r} channel={channel!r} state={state!r}."
                )
            normalized_channels[channel] = state
        if list(normalized_channels) != sorted(normalized_channels):
            raise ValueError(
                f"Histogram applicability channels for family {family!r} must be sorted."
            )
        normalized[family] = normalized_channels
    return normalized


def validate_histogram_applicability_structure(
    histograms: Mapping[str, Any],
    *,
    runtime_families: Iterable[str],
    histogram_applicability: Mapping[str, Any],
    schema_version: int | None = NOMINAL_CONTAINER_SCHEMA_VERSION,
) -> None:
    """Validate category-family structure without inspecting numerical content."""

    runtime_families = tuple(runtime_families)
    applicability = _family_channel_applicability(
        histogram_applicability,
        runtime_families=runtime_families,
    )
    for family in runtime_families:
        components = get_nominal_components(
            histograms,
            family,
            schema_version=schema_version,
        )
        observed_channels = frozenset().union(
            *(_channel_labels(component) for component in components.values())
        ) if components else frozenset()
        expected_applicable = {
            channel
            for channel, state in applicability[family].items()
            if state == "applicable"
        }
        expected_not_applicable = set(applicability[family]) - expected_applicable
        missing = sorted(expected_applicable - observed_channels)
        inconsistent = sorted(expected_not_applicable & observed_channels)
        if missing:
            raise ValueError(
                f"Family {family!r} is structurally absent for applicable channel(s): "
                + ", ".join(missing)
            )
        if inconsistent:
            raise ValueError(
                f"Family {family!r} is structurally present for not-applicable channel(s): "
                + ", ".join(inconsistent)
            )


def _validate_sparse_double(histogram: SparseHist, key: str) -> None:
    storage_name = getattr(histogram, "_init_args", {}).get("storage")
    if storage_name != "Double":
        raise TypeError(
            f"Schema version 2 SparseHist '{key}' must use storage='Double'; "
            f"got {storage_name!r}."
        )


def get_nominal_components(
    histograms: Mapping[str, Any],
    family: str,
    *,
    schema_version: int | None = NOMINAL_CONTAINER_SCHEMA_VERSION,
) -> OrderedDict[str, Any]:
    _require_histogram_mapping(histograms)
    scalar_key = scalar_nominal_key(family)
    eft_key = eft_nominal_key(family)
    components = OrderedDict()

    if schema_version in _SPLIT_SIBLING_SCHEMA_VERSIONS:
        if _dimensionality(family) == 2:
            if scalar_key in histograms or eft_key in histograms:
                raise ValueError(f"2D family '{family}' cannot contain split siblings.")
            if family in histograms:
                components["scalar_nominal"] = histograms[family]
            return components
        if family in histograms and (scalar_key in histograms or eft_key in histograms):
            raise ValueError(
                f"Family '{family}' contains duplicate authoritative nominal content."
            )
        if family in histograms:
            raise ValueError(
                f"Original 1D family key '{family}' is forbidden in schema version "
                f"{schema_version}."
            )
        if scalar_key in histograms:
            components["scalar_nominal"] = histograms[scalar_key]
        if eft_key in histograms:
            components["eft_nominal"] = histograms[eft_key]
        return components

    if scalar_key in histograms or eft_key in histograms:
        raise ValueError("Split sibling keys require nominal container schema version 2.")
    if family in histograms:
        components["uniform_nominal"] = histograms[family]
    return components


def get_scalar_nominal(
    histograms: Mapping[str, Any],
    family: str,
    *,
    schema_version: int | None = NOMINAL_CONTAINER_SCHEMA_VERSION,
) -> Any | None:
    components = get_nominal_components(
        histograms, family, schema_version=schema_version
    )
    if "scalar_nominal" in components:
        return components["scalar_nominal"]
    uniform = components.get("uniform_nominal")
    return uniform if type(uniform) is SparseHist else None


def get_eft_nominal(
    histograms: Mapping[str, Any],
    family: str,
    *,
    schema_version: int | None = NOMINAL_CONTAINER_SCHEMA_VERSION,
) -> Any | None:
    components = get_nominal_components(
        histograms, family, schema_version=schema_version
    )
    if "eft_nominal" in components:
        return components["eft_nominal"]
    uniform = components.get("uniform_nominal")
    return uniform if type(uniform) is HistEFT else None


def iter_nominal_components(
    histograms: Mapping[str, Any],
    family: str,
    *,
    schema_version: int | None = NOMINAL_CONTAINER_SCHEMA_VERSION,
) -> Iterable[tuple[str, Any]]:
    yield from get_nominal_components(
        histograms, family, schema_version=schema_version
    ).items()


def validate_nominal_family(
    histograms: Mapping[str, Any],
    family: str,
    *,
    schema_version: int | None = NOMINAL_CONTAINER_SCHEMA_VERSION,
    companion_selected: bool | None = None,
    selected_processes: Iterable[str] = (),
) -> None:
    components = get_nominal_components(
        histograms, family, schema_version=schema_version
    )
    companion = histograms.get(sumw2_key(family))

    if not components:
        if companion is not None:
            raise ValueError(f"Family '{family}' has an orphan statistical companion.")
        raise ValueError(f"Family '{family}' has no nominal component.")

    if schema_version == NOMINAL_CONTAINER_SCHEMA_VERSION:
        if _dimensionality(family) == 2:
            scalar = components["scalar_nominal"]
            if type(scalar) is not SparseHist:
                raise TypeError(f"2D family '{family}' must be an exact SparseHist.")
            _validate_sparse_double(scalar, family)
        else:
            scalar = components.get("scalar_nominal")
            eft = components.get("eft_nominal")
            if scalar is not None and type(scalar) is not SparseHist:
                raise TypeError(
                    f"Scalar sibling '{scalar_nominal_key(family)}' must be an exact SparseHist."
                )
            if scalar is not None:
                _validate_sparse_double(scalar, scalar_nominal_key(family))
            if eft is not None and type(eft) is not HistEFT:
                raise TypeError(
                    f"EFT sibling '{eft_nominal_key(family)}' must be an exact HistEFT."
                )
            if scalar is not None and eft is not None:
                scalar_categories = tuple(
                    (type(axis), axis.name) for axis in _categorical_axes(scalar)
                )
                eft_categories = tuple(
                    (type(axis), axis.name) for axis in _categorical_axes(eft)
                )
                if scalar_categories != eft_categories:
                    raise ValueError(f"Nominal siblings have different categorical axes for '{family}'.")
                scalar_dense = tuple(
                    _axis_signature(axis) for axis in _dense_axes(scalar)
                )
                eft_dense = tuple(_axis_signature(axis) for axis in _dense_axes(eft))
                if scalar_dense != eft_dense:
                    raise ValueError(f"Nominal siblings have different dense axes for '{family}'.")
                overlap = _process_labels(scalar) & _process_labels(eft)
                if overlap:
                    raise ValueError(
                        f"Nominal siblings for '{family}' duplicate process labels: "
                        + ", ".join(sorted(overlap))
                    )

    if companion_selected is False and companion is not None:
        raise ValueError(f"Family '{family}' has a policy-unselected companion.")
    if companion_selected is True and companion is None:
        raise ValueError(f"Family '{family}' is missing its required companion.")
    if companion is None:
        return
    if type(companion) is not SparseHist:
        raise TypeError(f"Companion '{sumw2_key(family)}' must be an exact SparseHist.")
    _validate_sparse_double(companion, sumw2_key(family))
    for base in components.values():
        _validate_base_companion_axes(base, companion, family)

    allowed_processes = frozenset(str(value) for value in selected_processes)
    companion_processes = _process_labels(companion)
    if allowed_processes and not companion_processes <= allowed_processes:
        extras = sorted(companion_processes - allowed_processes)
        raise ValueError(
            f"Companion '{sumw2_key(family)}' contains unselected processes: "
            + ", ".join(extras)
        )
    if allowed_processes:
        nominal_processes = frozenset().union(
            *(_process_labels(component) for component in components.values())
        )
        missing = sorted((nominal_processes & allowed_processes) - companion_processes)
        if missing:
            raise ValueError(
                f"Companion '{sumw2_key(family)}' has partial required coverage: "
                + ", ".join(missing)
            )


def validate_nominal_mapping(
    histograms: Mapping[str, Any],
    *,
    runtime_families: Iterable[str],
    schema_version: int | None = NOMINAL_CONTAINER_SCHEMA_VERSION,
    policy: resolved_sumw2_policy | None = None,
    histogram_applicability: Mapping[str, Any] | None = None,
) -> None:
    _require_histogram_mapping(histograms)
    runtime_families = tuple(runtime_families)
    family_applicability = (
        None
        if histogram_applicability is None
        else _family_channel_applicability(
            histogram_applicability,
            runtime_families=runtime_families,
        )
    )
    known_keys = set()
    for family in runtime_families:
        if schema_version == NOMINAL_CONTAINER_SCHEMA_VERSION and _dimensionality(family) == 1:
            known_keys.update({scalar_nominal_key(family), eft_nominal_key(family)})
        else:
            known_keys.add(family)
        known_keys.add(sumw2_key(family))
        companion_selected = None if policy is None else policy.selects_family(family)
        selected_processes = () if policy is None else policy.selected_processes(family)
        if family_applicability is not None:
            components = get_nominal_components(
                histograms,
                family,
                schema_version=schema_version,
            )
            companion = histograms.get(sumw2_key(family))
            family_is_applicable = any(
                state == "applicable"
                for state in family_applicability[family].values()
            )
            if not components and companion is None and not family_is_applicable:
                continue
        validate_nominal_family(
            histograms,
            family,
            schema_version=schema_version,
            companion_selected=companion_selected,
            selected_processes=selected_processes,
        )

    unknown_components = sorted(
        key
        for key in histograms
        if (
            key.endswith(SCALAR_NOMINAL_SUFFIX)
            or key.endswith(EFT_NOMINAL_SUFFIX)
            or key.endswith(SUMW2_SUFFIX)
        )
        and key not in known_keys
    )
    if unknown_components:
        raise ValueError(
            "Histogram payload contains orphan or unresolved schema keys: "
            + ", ".join(unknown_components)
        )
    if histogram_applicability is not None:
        validate_histogram_applicability_structure(
            histograms,
            runtime_families=runtime_families,
            histogram_applicability=histogram_applicability,
            schema_version=schema_version,
        )


def canonicalize_nominal_keys(
    histograms: Mapping[str, Any],
    *,
    runtime_families: Iterable[str],
    schema_version: int | None = NOMINAL_CONTAINER_SCHEMA_VERSION,
) -> dict[str, Any]:
    _require_histogram_mapping(histograms)
    # The streaming pickle loader recognizes a built-in dict as the root
    # mapping. Python dicts preserve insertion order, so deterministic schema
    # order does not require serializing an OrderedDict root.
    output = {}
    consumed = set()
    for family in runtime_families:
        if schema_version == NOMINAL_CONTAINER_SCHEMA_VERSION and _dimensionality(family) == 1:
            ordered_keys = (
                scalar_nominal_key(family),
                eft_nominal_key(family),
                sumw2_key(family),
            )
        else:
            ordered_keys = (family, sumw2_key(family))
        for key in ordered_keys:
            if key in histograms:
                output[key] = histograms[key]
                consumed.add(key)
    for key, value in histograms.items():
        if key not in consumed:
            output[key] = value
    return output


def _materialized_runtime_families(
    histograms: Mapping[str, Any],
    *,
    runtime_families: Iterable[str],
    schema_version: int | None,
) -> tuple[str, ...]:
    """Return runtime families represented by at least one schema component."""

    materialized = []
    for family in runtime_families:
        if schema_version == NOMINAL_CONTAINER_SCHEMA_VERSION and _dimensionality(family) == 1:
            family_keys = (
                scalar_nominal_key(family),
                eft_nominal_key(family),
                sumw2_key(family),
            )
        else:
            family_keys = (family, sumw2_key(family))
        if any(key in histograms for key in family_keys):
            materialized.append(family)
    return tuple(materialized)


def merge_nominal_mappings(
    histogram_mappings: Iterable[Mapping[str, Any]],
    *,
    runtime_families: Iterable[str],
    schema_version: int | None = NOMINAL_CONTAINER_SCHEMA_VERSION,
    policy: resolved_sumw2_policy | None = None,
    histogram_applicabilities: Iterable[Mapping[str, Any] | None] | None = None,
    histogram_applicability: Mapping[str, Any] | None = None,
) -> OrderedDict[str, Any]:
    histogram_mappings = tuple(histogram_mappings)
    if not histogram_mappings:
        raise ValueError("No histogram mappings were provided for merge.")
    runtime_families = tuple(runtime_families)
    if histogram_applicabilities is None:
        histogram_applicabilities = (None,) * len(histogram_mappings)
    else:
        histogram_applicabilities = tuple(histogram_applicabilities)
        if len(histogram_applicabilities) != len(histogram_mappings):
            raise ValueError(
                "Histogram applicability declarations must align one-to-one with "
                "histogram mappings."
            )
    merged = OrderedDict()
    for mapping, input_applicability in zip(
        histogram_mappings, histogram_applicabilities
    ):
        input_families = (
            tuple(input_applicability["families"])
            if input_applicability is not None
            else _materialized_runtime_families(
                mapping,
                runtime_families=runtime_families,
                schema_version=schema_version,
            )
        )
        validate_nominal_mapping(
            mapping,
            runtime_families=input_families,
            schema_version=schema_version,
            policy=policy,
            histogram_applicability=input_applicability,
        )
        for key, incoming in mapping.items():
            if key not in merged:
                merged[key] = copy.deepcopy(incoming)
                continue
            validate_histogram_compatibility(merged[key], incoming, key=key)
            merged[key] += incoming
    merged = canonicalize_nominal_keys(
        merged,
        runtime_families=runtime_families,
        schema_version=schema_version,
    )
    validate_nominal_mapping(
        merged,
        runtime_families=runtime_families,
        schema_version=schema_version,
        policy=policy,
        histogram_applicability=histogram_applicability,
    )
    return merged


def evaluate_eft_histogram_at_wc(
    eft_histogram: HistEFT,
    wc_values: Any = None,
) -> SparseHist:
    """Evaluate one EFT histogram into a scalar histogram without changing axes."""

    if type(eft_histogram) is not HistEFT:
        raise TypeError("EFT evaluation requires an exact HistEFT object.")
    output = SparseHist(
        *list(eft_histogram.categorical_axes),
        eft_histogram.dense_axis,
        storage="Double",
    )
    for categories, values in eft_histogram.eval(wc_values).items():
        output[tuple(categories)] = np.asarray(values)
    if eft_histogram.track_raw_counts:
        output.copy_raw_counts_from(eft_histogram)
    return output


def mark_sparse_histogram_raw_counts_unrecorded(
    histogram: SparseHist,
) -> SparseHist:
    """Copy generated content while classifying every raw-count cell unrecorded."""

    if type(histogram) is not SparseHist:
        raise TypeError("Raw-count reclassification requires an exact SparseHist object.")
    if not histogram.track_raw_counts:
        return histogram
    return histogram.with_raw_counts_unrecorded()


def _constant_histeft_from_sparse(
    scalar_histogram: SparseHist,
    *,
    wc_names: Iterable[str],
) -> HistEFT:
    dense_axes = tuple(scalar_histogram.dense_axes)
    if len(dense_axes) != 1:
        raise ValueError("A transient HistEFT compatibility view supports only 1D families.")
    output = HistEFT(
        *list(scalar_histogram.categorical_axes),
        dense_axes[0],
        wc_names=list(wc_names),
        label="Events",
    )
    coefficient_extent = output.axes["quadratic_term"].extent
    for categories, values in scalar_histogram.view(flow=True, as_dict=True).items():
        scalar_values = np.asarray(values)
        coefficients = np.zeros(scalar_values.shape + (coefficient_extent,))
        coefficients[..., 1] = scalar_values
        output[tuple(categories)] = coefficients
    return output


def evaluate_nominal_at_wc(
    histograms: Mapping[str, Any],
    family: str,
    wc_values: Any = None,
    *,
    schema_version: int | None = NOMINAL_CONTAINER_SCHEMA_VERSION,
) -> SparseHist:
    components = get_nominal_components(
        histograms, family, schema_version=schema_version
    )
    if not components:
        raise ValueError(f"Family '{family}' has no nominal component.")
    if "uniform_nominal" in components:
        uniform = components["uniform_nominal"]
        if type(uniform) is HistEFT:
            return evaluate_eft_histogram_at_wc(uniform, wc_values)
        if type(uniform) is SparseHist:
            return copy.deepcopy(uniform)
        raise TypeError(f"Unsupported legacy nominal type for '{family}'.")

    scalar = components.get("scalar_nominal")
    eft = components.get("eft_nominal")
    output = copy.deepcopy(scalar) if scalar is not None else None
    if eft is not None:
        evaluated_eft = evaluate_eft_histogram_at_wc(eft, wc_values)
        if output is None:
            output = evaluated_eft
        else:
            validate_histogram_compatibility(output, evaluated_eft, key=family)
            output += evaluated_eft
    if output is None:
        raise ValueError(f"Family '{family}' has no evaluable nominal component.")
    return output


def map_nominal_components(
    histograms: Mapping[str, Any],
    family: str,
    operation: Callable[[Any], Any],
    *,
    schema_version: int | None = NOMINAL_CONTAINER_SCHEMA_VERSION,
) -> OrderedDict[str, Any]:
    output = OrderedDict(histograms)
    components = get_nominal_components(
        histograms, family, schema_version=schema_version
    )
    if schema_version == NOMINAL_CONTAINER_SCHEMA_VERSION and _dimensionality(family) == 1:
        component_keys = {
            "scalar_nominal": scalar_nominal_key(family),
            "eft_nominal": eft_nominal_key(family),
        }
    else:
        component_keys = {next(iter(components), "uniform_nominal"): family}
    for component_name, histogram in components.items():
        output[component_keys[component_name]] = operation(histogram)
    companion_key = sumw2_key(family)
    if companion_key in output:
        output[companion_key] = operation(output[companion_key])
    return output


def _all_wc_names(histograms: Mapping[str, Any]) -> tuple[str, ...]:
    wc_names = None
    for histogram in histograms.values():
        if type(histogram) is not HistEFT:
            continue
        current = tuple(histogram.wc_names)
        if wc_names is None:
            wc_names = current
        elif wc_names != current:
            raise ValueError("Histogram payload contains incompatible WC orderings.")
    return () if wc_names is None else wc_names


def materialize_nominal_family(
    histograms: Mapping[str, Any],
    family: str,
    *,
    schema_version: int | None = NOMINAL_CONTAINER_SCHEMA_VERSION,
    wc_names: Iterable[str] | None = None,
) -> Any:
    components = get_nominal_components(
        histograms, family, schema_version=schema_version
    )
    if "uniform_nominal" in components:
        return copy.deepcopy(components["uniform_nominal"])
    if _dimensionality(family) == 2:
        return copy.deepcopy(components["scalar_nominal"])
    scalar = components.get("scalar_nominal")
    eft = components.get("eft_nominal")
    resolved_wc_names = tuple(wc_names or (eft.wc_names if eft is not None else ()))
    output = (
        _constant_histeft_from_sparse(scalar, wc_names=resolved_wc_names)
        if scalar is not None
        else None
    )
    if eft is not None:
        if tuple(eft.wc_names) != resolved_wc_names:
            raise ValueError(f"WC ordering differs while materializing '{family}'.")
        if output is None:
            output = copy.deepcopy(eft)
        else:
            validate_histogram_compatibility(output, eft, key=family)
            output += eft
    if output is None:
        raise ValueError(f"Family '{family}' has no nominal component.")
    return output


def materialize_legacy_histogram_dict(
    histograms: Mapping[str, Any],
    *,
    runtime_families: Iterable[str] | None = None,
    schema_version: int | None = NOMINAL_CONTAINER_SCHEMA_VERSION,
    require_companions: Iterable[str] = (),
) -> OrderedDict[str, Any]:
    """Build a bounded consumer-only uniform view without retaining split sources."""

    _require_histogram_mapping(histograms)
    if schema_version != NOMINAL_CONTAINER_SCHEMA_VERSION or not is_split_nominal_mapping(
        histograms
    ):
        return OrderedDict((key, copy.deepcopy(value)) for key, value in histograms.items())
    if runtime_families is None:
        discovered = []
        seen = set()
        for key in histograms:
            family = family_from_component_key(key)
            if family is not None and family not in seen:
                discovered.append(family)
                seen.add(family)
        runtime_families = discovered
    runtime_families = tuple(runtime_families)
    required = frozenset(require_companions)
    wc_names = _all_wc_names(histograms)
    output = OrderedDict()
    consumed = set()
    for family in runtime_families:
        components = get_nominal_components(histograms, family, schema_version=schema_version)
        if not components:
            continue
        output[family] = materialize_nominal_family(
            histograms,
            family,
            schema_version=schema_version,
            wc_names=wc_names,
        )
        consumed.update({scalar_nominal_key(family), eft_nominal_key(family), family})
        companion_key = sumw2_key(family)
        companion = histograms.get(companion_key)
        if family in required and companion is None:
            raise ValueError(f"Family '{family}' is missing its required companion.")
        if companion is not None:
            if _dimensionality(family) == 1:
                output[companion_key] = _constant_histeft_from_sparse(
                    companion, wc_names=wc_names
                )
            else:
                output[companion_key] = copy.deepcopy(companion)
            consumed.add(companion_key)
    for key, value in histograms.items():
        if key not in consumed:
            output[key] = copy.deepcopy(value)
    return output


def materialize_scalar_histogram_dict(
    histograms: Mapping[str, Any],
    *,
    runtime_families: Iterable[str],
    wc_values: Any = None,
    schema_version: int | None = NOMINAL_CONTAINER_SCHEMA_VERSION,
    require_companions: Iterable[str] = (),
) -> OrderedDict[str, Any]:
    """Build a consumer-local scalar view by evaluating every nominal family."""

    _require_histogram_mapping(histograms)
    runtime_families = tuple(runtime_families)
    required = frozenset(require_companions)
    output = OrderedDict()
    consumed = set()
    for family in runtime_families:
        components = get_nominal_components(
            histograms, family, schema_version=schema_version
        )
        if not components:
            continue
        output[family] = evaluate_nominal_at_wc(
            histograms,
            family,
            wc_values,
            schema_version=schema_version,
        )
        consumed.update({scalar_nominal_key(family), eft_nominal_key(family), family})
        companion_key = sumw2_key(family)
        companion = histograms.get(companion_key)
        if family in required and companion is None:
            raise ValueError(f"Family '{family}' is missing its required companion.")
        if companion is not None:
            output[companion_key] = copy.deepcopy(companion)
            consumed.add(companion_key)
    for key, value in histograms.items():
        if key not in consumed:
            output[key] = copy.deepcopy(value)
    return output


# ---------------------------------------------------------------------------
# Schema version 3: embedded nominal statistical sumw2
# ---------------------------------------------------------------------------
#
# Storage contract
#   * Exactly two storage modes exist: EMBEDDED and NONE. One mode applies to
#     every nominal component of a payload; mixed payloads are rejected.
#   * 1D scalar siblings are MultiCell HistEFT objects with ``wc_names == []``;
#     1D EFT siblings are MultiCell HistEFT objects with their WC list. Under
#     EMBEDDED both carry ``store_sumw2=True``; under NONE both carry False.
#   * 2D families are SparseHist objects with ``storage="Weight"`` under
#     EMBEDDED and ``storage="Double"`` under NONE.
#   * Separate ``<family>_sumw2`` companion histograms are not accepted, and no
#     selective sumw2 policy or content manifest participates in validation.
#
# Recording convention (storage available vs. value recorded)
#   * Embedded storage is allocated for every bin, but nominal statistical
#     sumw2 is *recorded* only in sparse bins whose ``systematic`` label is
#     ``"nominal"``. It is recorded for every process: Data (where unit
#     weights make it the event count), MC, and EFT MC.
#   * EFT sumw2 is the second moment of the SM-point weight (all WCs zero).
#     Its Wilson-coefficient dependence is neither computed nor stored.
#   * In non-nominal systematic bins the sumw2 slot means "not recorded",
#     never "measured variance of zero". Its numerical content is not
#     authoritative (a Weight-storage fill accumulates w**2 there), so
#     ``get_nominal_sumw2`` omits those bins instead of returning them.
#   * A histogram without a ``systematic`` axis (for example after selecting
#     the nominal slice) is treated as nominal content.


class Sumw2Storage(str, Enum):
    """Statistical-moment storage of a schema-version-3 nominal payload."""

    EMBEDDED = "embedded"
    NONE = "none"


SUMW2_RECORDED_SYSTEMATIC = "nominal"
SYSTEMATIC_AXIS_NAME = "systematic"


class Sumw2NotStoredError(RuntimeError):
    """Raised when nominal sumw2 is requested from NONE storage."""


def parse_sumw2_storage(value: Any) -> Sumw2Storage:
    """Return the storage mode named by ``value``; reject every other layout."""

    if isinstance(value, Sumw2Storage):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for mode in Sumw2Storage:
            if normalized == mode.value:
                return mode
        if normalized == "companion":
            raise ValueError(
                "Companion '<family>_sumw2' storage is not supported by schema "
                f"version {EMBEDDED_SUMW2_SCHEMA_VERSION}; use 'embedded' or 'none'."
            )
    raise ValueError(
        f"Unknown sumw2 storage {value!r}; expected one of "
        + ", ".join(repr(mode.value) for mode in Sumw2Storage)
        + "."
    )


def sumw2_storage_from_fill_flag(fill_sumw2: bool) -> Sumw2Storage:
    """Map the single producer boolean onto the storage mode."""

    if not isinstance(fill_sumw2, (bool, np.bool_)):
        raise TypeError("fill_sumw2 must be a boolean.")
    return Sumw2Storage.EMBEDDED if fill_sumw2 else Sumw2Storage.NONE


def _storage_kind(storage: Any) -> str | None:
    """Classify a SparseHist storage declaration as 'weight', 'double' or None."""

    if isinstance(storage, str):
        lowered = storage.lower()
        return lowered if lowered in {"weight", "double"} else None
    for kind, storage_type in (
        ("weight", bh.storage.Weight),
        ("double", bh.storage.Double),
    ):
        if isinstance(storage, storage_type) or (
            isinstance(storage, type) and issubclass(storage, storage_type)
        ):
            return kind
    return None


def _sparse_storage_kind(histogram: SparseHist, key: str) -> str:
    declared = _storage_kind(getattr(histogram, "_init_args", {}).get("storage"))
    if declared is None:
        raise TypeError(
            f"Schema version {EMBEDDED_SUMW2_SCHEMA_VERSION} SparseHist '{key}' must "
            "declare storage='Weight' or storage='Double'."
        )
    # Deserialized payloads carry their own dense histograms; their physical
    # storage must agree with the declaration.
    expected_type = bh.storage.Weight if declared == "weight" else bh.storage.Double
    for dense in histogram._dense_hists.values():
        if dense.storage_type is not expected_type:
            raise TypeError(
                f"SparseHist '{key}' declares {declared!r} storage but holds "
                f"{dense.storage_type.__name__} content."
            )
    return declared


def sumw2_storage_of_histogram(histogram: Any, *, key: str = "<histogram>") -> Sumw2Storage:
    """Return the storage mode of one schema-version-3 nominal histogram."""

    if type(histogram) is HistEFT:
        if not getattr(histogram, "_use_multicell", False):
            raise TypeError(
                f"HistEFT '{key}' uses the legacy quadratic_term layout; schema "
                f"version {EMBEDDED_SUMW2_SCHEMA_VERSION} requires MultiCell storage."
            )
        return Sumw2Storage.EMBEDDED if histogram.store_sumw2 else Sumw2Storage.NONE
    if type(histogram) is SparseHist:
        kind = _sparse_storage_kind(histogram, key)
        return Sumw2Storage.EMBEDDED if kind == "weight" else Sumw2Storage.NONE
    raise TypeError(
        "Nominal schema supports exact SparseHist or HistEFT objects; got "
        f"{type(histogram).__name__} for '{key}'."
    )


def _systematic_label(histogram: Any, sparse_key: Any) -> str | None:
    names = tuple(histogram.categorical_axes.name)
    if SYSTEMATIC_AXIS_NAME not in names:
        return None
    return str(tuple(sparse_key)[names.index(SYSTEMATIC_AXIS_NAME)])


def is_nominal_sumw2_recorded(histogram: Any, sparse_key: Any) -> bool:
    """Return whether ``sparse_key`` carries recorded nominal statistical sumw2.

    Storage availability alone is not enough: the bin must also belong to the
    nominal systematic (or the histogram must have no systematic axis).
    """

    if sumw2_storage_of_histogram(histogram) is not Sumw2Storage.EMBEDDED:
        return False
    label = _systematic_label(histogram, sparse_key)
    return label is None or label == SUMW2_RECORDED_SYSTEMATIC


def get_nominal_sumw2(histogram: Any, *, flow: bool = False) -> dict[Any, np.ndarray]:
    """Return recorded nominal statistical sumw2 per sparse bin.

    Only sparse bins with recorded sumw2 are returned (see
    ``is_nominal_sumw2_recorded``); non-nominal systematic bins are omitted so
    that "not recorded" can never be read as a zero variance. Raises
    ``Sumw2NotStoredError`` for NONE storage.
    """

    storage = sumw2_storage_of_histogram(histogram)
    if storage is Sumw2Storage.NONE:
        raise Sumw2NotStoredError(
            "This histogram has no embedded nominal sumw2 storage (storage='none')."
        )
    if type(histogram) is HistEFT:
        moments = histogram.nominal_sumw2(flow=flow)
    else:
        moments = {
            sparse_key: np.asarray(view.variance)
            for sparse_key, view in histogram.view(flow=flow, as_dict=True).items()
        }
    return {
        sparse_key: np.array(values, copy=True)
        for sparse_key, values in moments.items()
        if is_nominal_sumw2_recorded(histogram, sparse_key)
    }


def _validate_embedded_component(
    histogram: Any,
    *,
    family: str,
    key: str,
    component: str,
) -> Sumw2Storage:
    if _dimensionality(family) == 2:
        if type(histogram) is not SparseHist:
            raise TypeError(f"2D family '{key}' must be an exact SparseHist.")
        return sumw2_storage_of_histogram(histogram, key=key)
    if type(histogram) is not HistEFT:
        raise TypeError(
            f"1D nominal sibling '{key}' must be an exact MultiCell HistEFT in "
            f"schema version {EMBEDDED_SUMW2_SCHEMA_VERSION}; got "
            f"{type(histogram).__name__}."
        )
    storage = sumw2_storage_of_histogram(histogram, key=key)
    if component == "scalar_nominal" and list(histogram.wc_names):
        raise ValueError(f"Scalar sibling '{key}' must have no Wilson coefficients.")
    if component == "eft_nominal" and not list(histogram.wc_names):
        raise ValueError(f"EFT sibling '{key}' must declare its Wilson coefficients.")
    return storage


def _validate_embedded_siblings(scalar: HistEFT, eft: HistEFT, family: str) -> None:
    scalar_categories = tuple((type(axis), axis.name) for axis in _categorical_axes(scalar))
    eft_categories = tuple((type(axis), axis.name) for axis in _categorical_axes(eft))
    if scalar_categories != eft_categories:
        raise ValueError(f"Nominal siblings have different categorical axes for '{family}'.")
    if tuple(map(_axis_signature, _dense_axes(scalar))) != tuple(
        map(_axis_signature, _dense_axes(eft))
    ):
        raise ValueError(f"Nominal siblings have different dense axes for '{family}'.")
    overlap = _process_labels(scalar) & _process_labels(eft)
    if overlap:
        raise ValueError(
            f"Nominal siblings for '{family}' duplicate process labels: "
            + ", ".join(sorted(overlap))
        )


def validate_embedded_nominal_family(
    histograms: Mapping[str, Any],
    family: str,
) -> Sumw2Storage | None:
    """Validate one family and return its storage (None when it is absent)."""

    _require_histogram_mapping(histograms)
    if sumw2_key(family) in histograms:
        raise ValueError(
            f"Companion '{sumw2_key(family)}' is not supported by schema version "
            f"{EMBEDDED_SUMW2_SCHEMA_VERSION}; nominal sumw2 must be embedded. "
            "Regenerate this artifact with the embedded-sumw2 producer."
        )
    components = get_nominal_components(
        histograms, family, schema_version=EMBEDDED_SUMW2_SCHEMA_VERSION
    )
    if not components:
        return None
    keys = {
        "scalar_nominal": family if _dimensionality(family) == 2 else scalar_nominal_key(family),
        "eft_nominal": eft_nominal_key(family),
    }
    storages = {
        _validate_embedded_component(
            histogram, family=family, key=keys[component], component=component
        )
        for component, histogram in components.items()
    }
    if len(storages) != 1:
        raise ValueError(f"Family '{family}' mixes embedded and none sumw2 storage.")
    if "scalar_nominal" in components and "eft_nominal" in components:
        _validate_embedded_siblings(
            components["scalar_nominal"], components["eft_nominal"], family
        )
    return storages.pop()


def validate_embedded_nominal_mapping(
    histograms: Mapping[str, Any],
    *,
    runtime_families: Iterable[str],
    sumw2_storage: Sumw2Storage | str | None = None,
    histogram_applicability: Mapping[str, Any] | None = None,
) -> Sumw2Storage:
    """Validate a schema-version-3 payload and return its single storage mode.

    ``sumw2_storage`` is the declared mode (for example from a sidecar); when
    given, every component must match it. When omitted, the mode is inferred
    from the content and must still be uniform across the payload.
    """

    _require_histogram_mapping(histograms)
    runtime_families = tuple(runtime_families)
    declared = None if sumw2_storage is None else parse_sumw2_storage(sumw2_storage)

    companions = sorted(key for key in histograms if key.endswith(SUMW2_SUFFIX))
    if companions:
        raise ValueError(
            "Companion sumw2 histograms are not supported by schema version "
            f"{EMBEDDED_SUMW2_SCHEMA_VERSION}: " + ", ".join(companions)
            + ". Regenerate this artifact with the embedded-sumw2 producer."
        )

    known_keys = set()
    observed = {}
    for family in runtime_families:
        if _dimensionality(family) == 1:
            known_keys.update({scalar_nominal_key(family), eft_nominal_key(family)})
        else:
            known_keys.add(family)
        storage = validate_embedded_nominal_family(histograms, family)
        if storage is not None:
            observed[family] = storage

    unknown_components = sorted(
        key
        for key in histograms
        if (key.endswith(SCALAR_NOMINAL_SUFFIX) or key.endswith(EFT_NOMINAL_SUFFIX))
        and key not in known_keys
    )
    if unknown_components:
        raise ValueError(
            "Histogram payload contains orphan or unresolved schema keys: "
            + ", ".join(unknown_components)
        )

    modes = set(observed.values())
    if len(modes) > 1:
        by_mode = {
            mode.value: sorted(f for f, m in observed.items() if m is mode)
            for mode in modes
        }
        raise ValueError(f"Histogram payload mixes sumw2 storage modes: {by_mode}.")
    if declared is not None and modes and modes != {declared}:
        raise ValueError(
            f"Declared sumw2 storage {declared.value!r} disagrees with the "
            f"histogram content ({modes.pop().value!r})."
        )
    if not modes and declared is None:
        raise ValueError(
            "Cannot infer sumw2 storage from a payload without nominal components; "
            "declare it explicitly."
        )
    if histogram_applicability is not None:
        validate_histogram_applicability_structure(
            histograms,
            runtime_families=runtime_families,
            histogram_applicability=histogram_applicability,
            schema_version=EMBEDDED_SUMW2_SCHEMA_VERSION,
        )
    return declared if declared is not None else modes.pop()


def require_embedded_sumw2(
    histograms: Mapping[str, Any],
    *,
    runtime_families: Iterable[str],
) -> None:
    """Fail unless the payload carries embedded nominal sumw2 storage."""

    storage = validate_embedded_nominal_mapping(
        histograms, runtime_families=runtime_families
    )
    if storage is not Sumw2Storage.EMBEDDED:
        raise Sumw2NotStoredError(
            "This consumer requires embedded nominal sumw2, but the payload was "
            "produced with sumw2 storage 'none'."
        )
