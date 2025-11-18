"""Utilities for working with tuple-keyed histogram payloads.

The MC validation plotting scripts historically consumed histogram
collections keyed by the variable name with categorical axes describing the
dataset, analysis channel, and systematic variation.  The processing
pipeline now serialises per-histogram tuples instead, so the plotting
utilities need to rebuild a convenient structure from those tuples.  This
module centralises that logic so the plotting scripts can remain focussed on
presentation concerns.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import hist

try:  # pragma: no cover - HistEFT is optional in some environments
    from topcoffea.modules.HistEFT import HistEFT
except Exception:  # pragma: no cover - fallback when HistEFT is unavailable
    HistEFT = None  # type: ignore[assignment]


TupleKey = Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]

_COMPONENT_INDEX = {
    "variable": 0,
    "channel": 1,
    "application": 2,
    "sample": 3,
    "systematic": 4,
}


def _normalise_key(key: tuple) -> TupleKey:
    if len(key) != 5:
        raise ValueError(
            "Tuple histogram keys must have five elements: (variable, channel, application, sample, systematic)"
        )
    return key  # type: ignore[return-value]


def _label_components(key: TupleKey) -> Tuple[str, str, str, str, str]:
    """Return *key* components as strings with sensible defaults."""

    variable, channel, application, sample, systematic = key
    variable_label = str(variable)
    channel_label = "" if channel is None else str(channel)
    application_label = "" if application is None else str(application)
    sample_label = "" if sample is None else str(sample)
    systematic_label = "nominal" if systematic is None else str(systematic)
    return (
        variable_label,
        channel_label,
        application_label,
        sample_label,
        systematic_label,
    )


def tuple_histogram_items(hist_store: Mapping[Any, Any]) -> Dict[TupleKey, Any]:
    """Return a mapping of tuple-keyed histogram entries within *hist_store*."""

    entries: Dict[TupleKey, Any] = {}
    for key, value in hist_store.items():
        if isinstance(key, tuple):
            entries[_normalise_key(key)] = value
    return entries


def component_values(tuple_entries: Mapping[TupleKey, Any], component: str) -> Sequence[str]:
    """Return sorted unique values for *component* within *tuple_entries*."""

    index = _COMPONENT_INDEX[component]
    values = {
        _label_components(key)[index]
        for key in tuple_entries.keys()
        if key[index] is not None
    }
    return sorted(values)


def component_labels(
    tuple_entries: Mapping[TupleKey, Any],
    component: str,
    *,
    include_application: bool = False,
) -> Sequence[str]:
    """Return sorted labels for *component* optionally tagged by application.

    When ``include_application`` is ``True`` and the tuple key carries an application
    value, the application tag is appended in parentheses so legends or printed
    selections retain the application region context.
    """

    labels = set()
    for key in tuple_entries.keys():
        variable, channel, application, sample, systematic = _label_components(key)
        index = _COMPONENT_INDEX[component]
        value = (variable, channel, application, sample, systematic)[index]
        if not value:
            continue
        if include_application and application and component not in {"application", "systematic"}:
            labels.add(f"{value} ({application})")
        else:
            labels.add(value)
    return sorted(labels)


def filter_tuple_histograms(
    tuple_entries: Mapping[TupleKey, Any],
    *,
    variable: Optional[str] = None,
    channel: Optional[str] = None,
    application: Optional[str] = None,
    sample: Optional[str] = None,
    systematic: Optional[str] = None,
) -> Dict[TupleKey, Any]:
    """Return the subset of *tuple_entries* matching the requested filters."""

    filters = {
        "variable": variable,
        "channel": channel,
        "application": application,
        "sample": sample,
        "systematic": systematic,
    }
    result: Dict[TupleKey, Any] = {}
    for key, value in tuple_entries.items():
        include = True
        for name, criterion in filters.items():
            if criterion is None:
                continue
            index = _COMPONENT_INDEX[name]
            if key[index] != criterion:
                include = False
                break
        if include:
            result[key] = value
    return result


def _copy_histogram(histogram: Any) -> Any:
    """Return a shallow copy of *histogram* preserving its concrete type."""

    copier = getattr(histogram, "copy", None)
    if callable(copier):
        return copier()
    return copy.deepcopy(histogram)


def _aggregate_variable_entries(
    tuple_entries: Mapping[TupleKey, Any]
) -> Dict[str, MutableMapping[Tuple[str, str, str, str], Any]]:
    """Group histogram entries by variable and tuple components."""

    grouped: Dict[str, MutableMapping[Tuple[str, str, str, str], Any]] = defaultdict(dict)
    for key, histogram in tuple_entries.items():
        variable, channel, application, sample, systematic = _label_components(key)
        dataset = sample
        channel_label = channel or "inclusive"
        application_label = application
        systematic_label = systematic or "nominal"
        aggregate_key = (dataset, application_label, channel_label, systematic_label)

        variable_entries = grouped[variable]
        if aggregate_key in variable_entries:
            variable_entries[aggregate_key] = variable_entries[aggregate_key] + histogram
        else:
            variable_entries[aggregate_key] = _copy_histogram(histogram)
    return grouped


def _build_hist_like(
    template: Any,
    dataset_labels: Sequence[str],
    application_labels: Sequence[str],
    channel_labels: Sequence[str],
    systematic_labels: Sequence[str],
):
    """Create an empty histogram matching *template* with categorical axes."""

    application_axis = hist.axis.StrCategory(application_labels, name="application")
    dataset_axis = hist.axis.StrCategory(dataset_labels, name="dataset")
    channel_axis = hist.axis.StrCategory(channel_labels, name="channel")
    systematic_axis = hist.axis.StrCategory(systematic_labels, name="systematic")

    if HistEFT is not None and isinstance(template, HistEFT):
        dense_axis = template.dense_axis
        return HistEFT(
            dataset_axis,
            application_axis,
            channel_axis,
            systematic_axis,
            dense_axis,
            wc_names=getattr(template, "wc_names", []),
            label=getattr(template, "label", None),
        )

    axes = list(getattr(template, "axes", ()))
    storage = template.storage_type() if hasattr(template, "storage_type") else "Double"
    return hist.Hist(
        dataset_axis,
        application_axis,
        channel_axis,
        systematic_axis,
        *axes,
        storage=storage,
    )


def build_dataset_histograms(hist_store: Mapping[Any, Any]) -> Dict[str, Any]:
    """Reconstruct histograms with categorical axes from tuple-keyed inputs."""

    tuple_entries = tuple_histogram_items(hist_store)
    if not tuple_entries:
        return {}

    grouped = _aggregate_variable_entries(tuple_entries)
    rebuilt: Dict[str, Any] = {}

    for variable, aggregates in grouped.items():
        first_hist = next(iter(aggregates.values()))
        dataset_labels = sorted({key[0] for key in aggregates})
        application_labels = sorted({key[1] for key in aggregates})
        channel_labels = sorted({key[2] for key in aggregates})
        systematic_labels = sorted({key[3] for key in aggregates})

        summary_hist = _build_hist_like(
            first_hist,
            dataset_labels,
            application_labels,
            channel_labels,
            systematic_labels,
        )

        for (dataset, application, channel, systematic), histogram in aggregates.items():
            index = {
                "dataset": dataset,
                "application": application,
                "channel": channel,
                "systematic": systematic,
            }
            try:
                summary_hist[index] = histogram
            except Exception:
                summary_hist[index] = histogram.view()

        rebuilt[variable] = summary_hist

    return rebuilt


__all__ = [
    "TupleKey",
    "build_dataset_histograms",
    "component_labels",
    "component_values",
    "filter_tuple_histograms",
    "tuple_histogram_items",
]

