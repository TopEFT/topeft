"""Utilities for normalising coffea runner outputs.

This module centralises the logic used to convert the accumulator returned by
``coffea.processor.Runner`` into a serialisable mapping keyed by histogram
tuples.  The canonical tuple ordering is
``(variable, channel, application, sample, systematic)`` and legacy 4-tuple
structures are rejected to prevent silent ambiguities in downstream tools.  The
helpers are intentionally lightweight so that they can be reused by the
analysis scripts as well as the training utilities.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency during some tests
    from hist import Hist, axis
except Exception:  # pragma: no cover - fallback when histogram extras missing
    Hist = None  # type: ignore[assignment]
    axis = None  # type: ignore[assignment]

try:  # pragma: no cover - topcoffea is optional for a subset of the tests
    from topcoffea.modules.HistEFT import HistEFT
except Exception:  # pragma: no cover - fallback when HistEFT is unavailable
    HistEFT = None  # type: ignore[assignment]


TupleKey = Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]
SUMMARY_KEY = "__tuple_summary__"


def _ensure_numpy(array: Any) -> np.ndarray:
    """Return *array* as a :class:`numpy.ndarray` without copying when possible."""

    if isinstance(array, np.ndarray):
        return array
    return np.asarray(array)


def _hist_like(instance: Any) -> bool:
    """Return ``True`` when *instance* behaves like a histogram object."""

    hist_classes: Tuple[type, ...] = tuple(
        cls
        for cls in (Hist, HistEFT)
        if isinstance(cls, type)
    )
    if not hist_classes:
        return False
    return isinstance(instance, hist_classes)


def _summarise_histogram(histogram: Any) -> Dict[str, Any]:
    """Create a deterministic summary payload for *histogram*."""

    values: Optional[np.ndarray] = None
    variances: Optional[np.ndarray] = None

    if HistEFT is not None and isinstance(histogram, HistEFT):
        entries = histogram.values(sumw2=True)
        for payload in entries.values():
            if isinstance(payload, tuple):
                current_values = _ensure_numpy(payload[0])
                raw_variances = payload[1]
                current_variances = None if raw_variances is None else _ensure_numpy(raw_variances)
            else:
                current_values = _ensure_numpy(payload)
                current_variances = None

            values = current_values if values is None else values + current_values
            if current_variances is None:
                variances = None if variances is None else None
            else:
                variances = (
                    current_variances
                    if variances is None
                    else variances + current_variances
                )
    elif Hist is not None and isinstance(histogram, Hist):
        if axis is not None:
            for axis_obj in histogram.axes:
                if isinstance(axis_obj, axis.StrCategory):
                    raise ValueError(
                        "Categorical histogram axes are unsupported for tuple-keyed outputs"
                    )
        values = _ensure_numpy(histogram.values(flow=True))
        raw_variances = histogram.variances(flow=True)
        variances = None if raw_variances is None else _ensure_numpy(raw_variances)
    else:
        raise TypeError(f"Unsupported histogram type: {type(histogram)!r}")

    if values is None:
        values = np.array([])

    summary: Dict[str, Any] = {
        "sumw": float(np.sum(values)) if values.size else 0.0,
        "sumw2": float(np.sum(variances)) if variances is not None else None,
        "values": values,
        "variances": variances,
    }
    return summary


def materialise_tuple_dict(hist_store: Mapping[TupleKey, Any]) -> "OrderedDict[TupleKey, Dict[str, Any]]":
    """Return an :class:`OrderedDict` keyed by sorted histogram tuple identifiers."""

    ordered_items = []
    for key, histogram in sorted(hist_store.items(), key=lambda item: item[0]):
        if not isinstance(key, tuple) or len(key) != 5:
            raise ValueError(
                "Histogram accumulator keys must be 5-tuples of (variable, channel, "
                "application, sample, systematic)."
            )
        summary = _summarise_histogram(histogram)
        ordered_items.append((key, summary))

    return OrderedDict(ordered_items)


def _tuple_entries(payload: Mapping[Any, Any]) -> Dict[TupleKey, Any]:
    """Extract histogram-like entries keyed by tuple identifiers from *payload*."""

    result: Dict[TupleKey, Any] = {}
    for key, value in payload.items():
        if isinstance(key, tuple):
            if len(key) != 5:
                raise ValueError(
                    "Histogram accumulator keys must be 5-tuples of (variable, channel, "
                    "application, sample, systematic); legacy 4-tuples are unsupported."
                )
            if _hist_like(value):
                result[key] = value
    return result


def normalise_runner_output(payload: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """Return a tuple-keyed ordered mapping preserving histogram payloads.

    Tuple-keyed histogram entries are emitted in lexicographic order to provide
    deterministic serialisation while their original histogram objects remain
    untouched.  Non-histogram entries are preserved in their original insertion
    order.  Consumers that need a deterministic, serialisable summary of the
    histogram contents can call :func:`materialise_tuple_dict` on the returned
    mapping as required.
    """

    if not isinstance(payload, Mapping):
        return payload

    tuple_histograms = _tuple_entries(payload)
    if not tuple_histograms:
        return payload

    ordered: "OrderedDict[Any, Any]" = OrderedDict()
    for key, histogram in sorted(tuple_histograms.items(), key=lambda item: item[0]):
        ordered[key] = histogram
    for key, value in payload.items():
        if key not in tuple_histograms:
            ordered[key] = value
    return ordered


def tuple_dict_stats(tuple_dict: Mapping[Any, Any]) -> Tuple[int, int]:
    """Return the total and non-zero bin counts for *tuple_dict* entries."""

    total_bins = 0
    filled_bins = 0
    summaries: Optional[Mapping[TupleKey, Mapping[str, Any]]] = None

    if isinstance(tuple_dict, Mapping):
        candidate = tuple_dict.get(SUMMARY_KEY)
        if isinstance(candidate, Mapping):
            summaries = candidate  # type: ignore[assignment]

    if summaries is None:
        summaries = OrderedDict(
            (
                key,
                _summarise_histogram(value),
            )
            for key, value in tuple_dict.items()
            if isinstance(key, tuple) and _hist_like(value)
        )

    for summary in summaries.values():
        values = summary.get("values")
        if values is None:
            continue
        array = _ensure_numpy(values)
        total_bins += int(array.size)
        filled_bins += int(np.count_nonzero(array))
    return total_bins, filled_bins


__all__ = [
    "TupleKey",
    "SUMMARY_KEY",
    "materialise_tuple_dict",
    "normalise_runner_output",
    "tuple_dict_stats",
]

