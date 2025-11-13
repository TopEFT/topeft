"""Shared helpers for flip measurement plotting scripts.

The flip measurement workflows serialise histograms using tuple keys rather
than categorical axes.  Plotting scripts therefore need small utilities to
iterate over those tuples, apply filters, and regroup the histograms for
display.  This module centralises that logic so the plotting entrypoints stay
focused on presentation concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional

import cloudpickle
import gzip
import hist

from topeft.modules.runner_output import SUMMARY_KEY


@dataclass(frozen=True)
class TupleHistogramEntry:
    """Container pairing a tuple key with its histogram payload."""

    variable: str
    region: str
    sample: str
    systematic: str
    histogram: hist.Hist


def _normalise_component(value: object, fallback: str) -> str:
    """Convert *value* to ``str`` while honouring ``fallback`` for ``None``."""

    if value is None:
        return fallback
    return str(value)


def tuple_histogram_entries(payload: Mapping[object, object]) -> Iterator[TupleHistogramEntry]:
    """Yield :class:`TupleHistogramEntry` objects from *payload*.

    Only entries keyed by 4-tuples are considered.  The tuple components are
    coerced to strings (with sensible defaults for ``None``) so downstream code
    can construct labels without needing categorical axes on the histograms.
    """

    for key, value in payload.items():
        if key == SUMMARY_KEY:
            continue
        if not isinstance(key, tuple) or len(key) != 4:
            continue
        if not isinstance(value, hist.Hist):
            continue

        variable = _normalise_component(key[0], "")
        region = _normalise_component(key[1], "")
        sample = _normalise_component(key[2], "")
        systematic = _normalise_component(key[3], "nominal")

        yield TupleHistogramEntry(
            variable=variable,
            region=region,
            sample=sample,
            systematic=systematic,
            histogram=value,
        )


def load_tuple_histogram_entries(path: str) -> Iterable[TupleHistogramEntry]:
    """Return tuple histogram entries stored at *path*."""

    with gzip.open(path, "rb") as fin:
        payload = cloudpickle.load(fin)

    if not isinstance(payload, Mapping):
        raise TypeError("Histogram payload must be a mapping")

    entries = list(tuple_histogram_entries(payload))
    if not entries:
        raise ValueError("No tuple-keyed histograms found in payload")
    return entries


def filter_entries(
    entries: Iterable[TupleHistogramEntry],
    *,
    variable: Optional[str] = None,
    region: Optional[str] = None,
    sample: Optional[str] = None,
    systematic: Optional[str] = None,
) -> Iterator[TupleHistogramEntry]:
    """Yield entries matching the requested filters."""

    for entry in entries:
        if variable is not None and entry.variable != variable:
            continue
        if region is not None and entry.region != region:
            continue
        if sample is not None and entry.sample != sample:
            continue
        if systematic is not None and entry.systematic != systematic:
            continue
        yield entry


def accumulate_entries(
    entries: Iterable[TupleHistogramEntry],
    *attributes: str,
) -> Dict[str, MutableMapping]:
    """Group *entries* by *attributes*, summing histograms within each bucket."""

    if not attributes:
        raise ValueError("At least one attribute must be provided for grouping")

    root: Dict[str, MutableMapping] = {}

    for entry in entries:
        cursor: MutableMapping = root
        for name in attributes[:-1]:
            key = getattr(entry, name)
            cursor = cursor.setdefault(key, {})  # type: ignore[assignment]

        final_key = getattr(entry, attributes[-1])
        histogram = entry.histogram.copy()

        existing = cursor.get(final_key)
        if existing is None:
            cursor[final_key] = histogram
        else:
            cursor[final_key] = existing + histogram  # type: ignore[operator]

    return root


def summarise_by_variable(
    entries: Iterable[TupleHistogramEntry],
    *,
    systematic: str | None = "nominal",
) -> Dict[str, MutableMapping[str, MutableMapping[str, hist.Hist]]]:
    """Return ``variable -> sample -> region`` groupings for *entries*."""

    filtered = filter_entries(entries, systematic=systematic) if systematic else entries
    grouped = accumulate_entries(filtered, "variable", "sample", "region")

    # ``accumulate_entries`` returns ``Dict[str, MutableMapping]`` at each level,
    # but the innermost mapping still stores generic ``MutableMapping`` values.
    # Normalise the type hints here for clarity.
    result: Dict[str, MutableMapping[str, MutableMapping[str, hist.Hist]]] = {}
    for variable, sample_map in grouped.items():
        sample_mapping: MutableMapping[str, MutableMapping[str, hist.Hist]] = {}
        for sample, region_map in sample_map.items():
            # Region maps already hold histograms directly.
            sample_mapping[sample] = region_map  # type: ignore[assignment]
        result[variable] = sample_mapping
    return result


__all__ = [
    "TupleHistogramEntry",
    "accumulate_entries",
    "filter_entries",
    "load_tuple_histogram_entries",
    "summarise_by_variable",
    "tuple_histogram_entries",
]

