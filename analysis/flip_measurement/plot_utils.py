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
    channel: str
    application: str
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

    Only entries keyed by tuple identifiers are considered.  The tuple
    components are coerced to strings (with sensible defaults for ``None``) so
    downstream code can construct labels without needing categorical axes on the
    histograms.
    """

    for key, value in payload.items():
        if key == SUMMARY_KEY:
            continue
        if not isinstance(key, tuple) or len(key) not in (4, 5):
            continue
        if not isinstance(value, hist.Hist):
            continue

        variable = _normalise_component(key[0], "")
        channel = _normalise_component(key[1], "")
        if len(key) == 5:
            application = _normalise_component(key[2], "")
            sample = _normalise_component(key[3], "")
            systematic = _normalise_component(key[4], "nominal")
        else:
            application = ""
            sample = _normalise_component(key[2], "")
            systematic = _normalise_component(key[3], "nominal")

        yield TupleHistogramEntry(
            variable=variable,
            channel=channel,
            application=application,
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
    application: Optional[str] = None,
    region: Optional[str] = None,
    sample: Optional[str] = None,
    systematic: Optional[str] = None,
) -> Iterator[TupleHistogramEntry]:
    """Yield entries matching the requested filters."""

    for entry in entries:
        if variable is not None and entry.variable != variable:
            continue
        if application is not None and entry.application not in (application, ""):
            continue
        if region is not None and entry.channel != region:
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
    application: str | None = None,
) -> Dict[str, MutableMapping[str, MutableMapping[str, MutableMapping[str, hist.Hist]]]]:
    """Return ``variable -> application -> sample -> channel`` groupings for *entries*."""

    if systematic is not None or application is not None:
        filtered = filter_entries(
            entries, systematic=systematic, application=application
        )
    else:
        filtered = entries
    grouped = accumulate_entries(filtered, "variable", "application", "sample", "channel")

    result: Dict[str, MutableMapping[str, MutableMapping[str, MutableMapping[str, hist.Hist]]]] = {}
    for variable, application_map in grouped.items():
        application_mapping: MutableMapping[str, MutableMapping[str, MutableMapping[str, hist.Hist]]] = {}
        for application_name, sample_map in application_map.items():
            target_application = (
                application if application is not None and application_name == "" else application_name
            )

            target_sample_mapping = application_mapping.setdefault(target_application, {})
            for sample, channel_map in sample_map.items():
                existing_channels = target_sample_mapping.get(sample)
                if existing_channels is None:
                    target_sample_mapping[sample] = channel_map  # type: ignore[assignment]
                else:
                    # Prefer existing (application-tagged) entries when both are present while
                    # keeping legacy-only channels.
                    for channel, histogram in channel_map.items():
                        target_sample_mapping[sample].setdefault(channel, histogram)

        result[variable] = application_mapping
    return result


__all__ = [
    "TupleHistogramEntry",
    "accumulate_entries",
    "filter_entries",
    "load_tuple_histogram_entries",
    "summarise_by_variable",
    "tuple_histogram_entries",
]

