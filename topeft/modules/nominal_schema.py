"""Access, validation, merge, and transient views for nominal histogram schemas."""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist

from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.sumw2_policy import resolved_sumw2_policy


NOMINAL_CONTAINER_SCHEMA_VERSION = 2
NOMINAL_CONTAINER_LAYOUT = "split_sibling_v1"
SCALAR_NOMINAL_SUFFIX = "__scalar_nominal"
EFT_NOMINAL_SUFFIX = "__eft_nominal"
SUMW2_SUFFIX = "_sumw2"


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

    if schema_version == NOMINAL_CONTAINER_SCHEMA_VERSION:
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
                f"Original 1D family key '{family}' is forbidden in schema version 2."
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
) -> None:
    _require_histogram_mapping(histograms)
    runtime_families = tuple(runtime_families)
    known_keys = set()
    for family in runtime_families:
        if schema_version == NOMINAL_CONTAINER_SCHEMA_VERSION and _dimensionality(family) == 1:
            known_keys.update({scalar_nominal_key(family), eft_nominal_key(family)})
        else:
            known_keys.add(family)
        known_keys.add(sumw2_key(family))
        companion_selected = None if policy is None else policy.selects_family(family)
        selected_processes = () if policy is None else policy.selected_processes(family)
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


def merge_nominal_mappings(
    histogram_mappings: Iterable[Mapping[str, Any]],
    *,
    runtime_families: Iterable[str],
    schema_version: int | None = NOMINAL_CONTAINER_SCHEMA_VERSION,
    policy: resolved_sumw2_policy | None = None,
) -> OrderedDict[str, Any]:
    histogram_mappings = tuple(histogram_mappings)
    if not histogram_mappings:
        raise ValueError("No histogram mappings were provided for merge.")
    runtime_families = tuple(runtime_families)
    merged = OrderedDict()
    for mapping in histogram_mappings:
        validate_nominal_mapping(
            mapping,
            runtime_families=runtime_families,
            schema_version=schema_version,
            policy=policy,
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
    return output


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
