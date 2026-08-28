"""Canonical processing/fitting axis resolution and exact aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import hist
import numpy as np


LEGACY_AXIS_KEYS = frozenset({"regular", "variable", "variable_multi"})
BINNING_MODES = ("processing", "fitting")


def _strict_edges(edges, *, context):
    array = np.asarray(edges, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{context} edges must be a one-dimensional sequence with at least two values.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{context} edges must all be finite.")
    if not np.all(np.diff(array) > 0):
        raise ValueError(f"{context} edges must be strictly increasing.")
    return array


def processing_edges(axis_config):
    """Validate and expand one typed ``processing`` definition."""

    legacy = LEGACY_AXIS_KEYS.intersection(axis_config)
    if legacy:
        raise ValueError(f"Legacy axis schema keys are not supported: {sorted(legacy)}.")
    if "processing" not in axis_config:
        raise ValueError("Axis configuration is missing the required 'processing' definition.")

    processing = axis_config["processing"]
    if not isinstance(processing, Mapping):
        raise TypeError("Axis 'processing' definition must be a mapping.")
    kind = processing.get("kind")
    if kind == "uniform":
        expected = {"kind", "bins", "start", "stop"}
        if set(processing) != expected:
            raise ValueError(f"Uniform processing definition requires exactly {sorted(expected)}.")
        bins = processing["bins"]
        start = processing["start"]
        stop = processing["stop"]
        if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins <= 0:
            raise ValueError("Uniform processing 'bins' must be a positive integer.")
        if not np.isfinite(start) or not np.isfinite(stop) or not start < stop:
            raise ValueError("Uniform processing requires finite start < stop.")
        return np.linspace(float(start), float(stop), int(bins) + 1)
    if kind == "edges":
        if set(processing) != {"kind", "edges"}:
            raise ValueError("Edge processing definition requires exactly 'kind' and 'edges'.")
        return _strict_edges(processing["edges"], context="Processing")
    raise ValueError(f"Unknown processing kind {kind!r}; expected 'uniform' or 'edges'.")


def validate_axis_config(axis_config, *, known_channels=None):
    """Validate one canonical axis configuration and return processing edges."""

    source_edges = processing_edges(axis_config)
    fitting = axis_config.get("fitting")
    if fitting is None:
        return source_edges
    if not isinstance(fitting, Mapping):
        raise TypeError("Axis 'fitting' definition must be a mapping.")
    if "default" not in fitting:
        raise ValueError("Axis 'fitting' definition requires 'default'.")
    unexpected = set(fitting) - {"default", "channels"}
    if unexpected:
        raise ValueError(f"Unknown fitting keys: {sorted(unexpected)}.")

    targets = [("default", fitting["default"])]
    channels = fitting.get("channels", {})
    if not isinstance(channels, Mapping):
        raise TypeError("Axis 'fitting.channels' must be a mapping.")
    known = None if known_channels is None else {str(channel) for channel in known_channels}
    for channel, edges in channels.items():
        if not isinstance(channel, str) or not channel:
            raise ValueError("Fitting channel override keys must be non-empty exact strings.")
        if known is not None and channel not in known:
            raise ValueError(f"Unknown exact fitting channel override {channel!r}.")
        targets.append((f"channel {channel!r}", edges))

    for target_name, edges in targets:
        target_edges = _strict_edges(edges, context=f"Fitting {target_name}")
        build_aggregation_map(source_edges, target_edges)
    return source_edges


def validate_axis_registry(registry, *, known_channels=None):
    """Validate all one-dimensional canonical axis definitions."""

    for family, axis_config in registry.items():
        validate_axis_config(axis_config, known_channels=known_channels)


def resolve_axis_edges(family, *, mode="fitting", channel=None, registry=None):
    """Resolve processing or fitting edges using exact channel lookup only."""

    if mode not in BINNING_MODES:
        raise ValueError(f"Unknown binning mode {mode!r}; expected one of {BINNING_MODES}.")
    if registry is None:
        from topeft.modules.axes import info as registry

    try:
        axis_config = registry[family]
    except KeyError as error:
        raise KeyError(f"Unknown histogram family {family!r}.") from error
    source_edges = validate_axis_config(axis_config)
    if mode == "processing" or "fitting" not in axis_config:
        return source_edges.copy()

    fitting = axis_config["fitting"]
    channel_overrides = fitting.get("channels", {})
    target = channel_overrides.get(channel, fitting["default"])
    target_edges = _strict_edges(target, context=f"Fitting family {family!r}")
    build_aggregation_map(source_edges, target_edges)
    return target_edges


def resolve_common_axis_edges(family, *, mode, channels, registry=None):
    """Resolve one common target for a group of exact channel names."""

    channel_names = tuple(str(channel) for channel in channels)
    if not channel_names:
        raise ValueError("At least one exact channel is required for grouped binning resolution.")
    resolved = [
        resolve_axis_edges(family, mode=mode, channel=channel, registry=registry)
        for channel in channel_names
    ]
    reference = resolved[0]
    conflicts = [
        (channel, edges.tolist())
        for channel, edges in zip(channel_names[1:], resolved[1:])
        if not np.array_equal(reference, edges)
    ]
    if conflicts:
        raise ValueError(
            f"Channels resolve to incompatible {mode} axes for {family!r}: "
            f"{channel_names[0]!r}={reference.tolist()}, conflicts={conflicts}."
        )
    return reference.copy()


def build_aggregation_map(source_edges, target_edges):
    """Return the exact flow-inclusive source-bin to target-bin index map."""

    source = _strict_edges(source_edges, context="Source")
    target = _strict_edges(target_edges, context="Target")
    if target[0] != source[0]:
        raise ValueError("Target and source must have the same first finite boundary.")
    if target[-1] > source[-1]:
        raise ValueError("Target final boundary cannot exceed the source final boundary.")
    missing = [edge for edge in target if not np.any(source == edge)]
    if missing:
        raise ValueError(f"Target boundaries are not exactly representable by the source: {missing}.")

    target_bins = target.size - 1
    mapping = np.empty(source.size + 1, dtype=int)
    mapping[0] = 0
    mapping[-1] = target_bins + 1
    for source_index, lower_edge in enumerate(source[:-1], start=1):
        if lower_edge >= target[-1]:
            mapping[source_index] = target_bins + 1
            continue
        target_index = int(np.searchsorted(target, lower_edge, side="right") - 1)
        mapping[source_index] = target_index + 1
    return mapping


def aggregate_array(values, aggregation_map, *, axis=0):
    """Sum a flow-inclusive array along one source-bin axis."""

    array = np.asarray(values)
    mapping = np.asarray(aggregation_map, dtype=int)
    if array.shape[axis] != mapping.size:
        raise ValueError(
            f"Aggregation map length {mapping.size} does not match array axis length {array.shape[axis]}."
        )
    output_shape = list(array.shape)
    output_shape[axis] = int(mapping.max()) + 1
    output = np.zeros(output_shape, dtype=array.dtype)
    moved_source = np.moveaxis(array, axis, 0)
    moved_output = np.moveaxis(output, axis, 0)
    for source_index, target_index in enumerate(mapping):
        moved_output[target_index] += moved_source[source_index]
    return output


def _axis_edges(axis):
    edges = axis.edges
    if callable(edges):
        edges = edges()
    return np.asarray(edges, dtype=float)


def _numeric_dense_axes(histogram):
    categorical_names = set(histogram.categorical_axes.name)
    return [
        axis
        for axis in histogram.axes
        if axis.name not in categorical_names and axis.name != "quadratic_term"
    ]


def histogram_dense_edges(histogram):
    """Return the sole physical dense-axis edges from a SparseHist/HistEFT."""

    dense_axes = _numeric_dense_axes(histogram)
    if len(dense_axes) != 1:
        raise ValueError(f"Expected exactly one physical dense axis, found {len(dense_axes)}.")
    return _axis_edges(dense_axes[0])


def validate_matching_histogram_edges(left, right, *, context="histograms"):
    """Reject equal-length payloads whose physical dense edges differ."""

    left_edges = histogram_dense_edges(left)
    right_edges = histogram_dense_edges(right)
    if not np.array_equal(left_edges, right_edges):
        raise ValueError(
            f"Physical dense-axis mismatch for {context}: {left_edges.tolist()} != {right_edges.tolist()}."
        )


def rebin_histogram(histogram, target_edges):
    """Exactly aggregate every sparse payload and EFT slot onto target edges."""

    if histogram is None:
        return None
    dense_axes = _numeric_dense_axes(histogram)
    if len(dense_axes) != 1:
        raise ValueError(f"Exact late rebinning requires one physical dense axis, found {len(dense_axes)}.")
    source_axis = dense_axes[0]
    source_edges = _axis_edges(source_axis)
    target = _strict_edges(target_edges, context="Target")
    if np.array_equal(source_edges, target):
        return histogram
    traits = source_axis.traits
    if not traits.underflow or not traits.overflow:
        raise ValueError(
            "Exact late rebinning requires the live underflow-and-overflow axis contract."
        )
    aggregation_map = build_aggregation_map(source_edges, target)

    target_axis = hist.axis.Variable(
        target,
        name=source_axis.name,
        label=source_axis.label,
        metadata=getattr(source_axis, "metadata", None),
        underflow=traits.underflow,
        overflow=traits.overflow,
    )
    rebinned = histogram.empty_from_axes(dense_axes=[target_axis])
    for source_index, source_dense in histogram._dense_hists.items():
        categories = histogram.index_to_categories(source_index)
        target_index = rebinned._fill_bookkeep(*categories)
        target_dense = rebinned._dense_hists[target_index]
        source_values = source_dense.view(flow=True)
        target_values = aggregate_array(source_values, aggregation_map, axis=0)
        target_dense.view(flow=True)[...] = target_values
    return rebinned


def resolve_and_rebin_histogram(histogram, family, *, mode="fitting", channel=None, registry=None):
    """Resolve one canonical view and apply exact aggregation when required."""

    target_edges = resolve_axis_edges(
        family,
        mode=mode,
        channel=channel,
        registry=registry,
    )
    if registry is None:
        from topeft.modules.axes import info as registry
    if mode == "processing" or "fitting" not in registry[family]:
        return histogram
    return rebin_histogram(histogram, target_edges)


def make_processing_axis(axis_config, *, name, label, suffix="", label_suffix=""):
    """Construct a histogram axis from canonical processing metadata only."""

    edges = processing_edges(axis_config)
    processing = axis_config["processing"]
    axis_name = f"{name}{suffix}"
    axis_label = f"{label}{label_suffix}"
    if processing["kind"] == "uniform":
        return hist.axis.Regular(
            processing["bins"],
            processing["start"],
            processing["stop"],
            name=axis_name,
            label=axis_label,
        )
    return hist.axis.Variable(edges, name=axis_name, label=axis_label)
