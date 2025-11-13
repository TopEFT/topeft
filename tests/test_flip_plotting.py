"""Tests for tuple-keyed flip plotting helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from hist import Hist, axis, storage

from analysis.flip_measurement import flip_ar_plotter
from analysis.flip_measurement.plot_utils import tuple_histogram_entries


def _build_histogram(values: np.ndarray) -> Hist:
    histogram = Hist(axis.Regular(4, 0.0, 4.0, name="observable"), storage=storage.Double())
    histogram.fill(observable=values)
    return histogram


def test_flip_application_plot_labels():
    payload = {
        ("observable", "ssz", "SampleA", "nominal"): _build_histogram(np.asarray([0.5, 1.5])),
        ("observable", "osz", "SampleA", "nominal"): _build_histogram(np.asarray([0.5, 2.5])),
    }

    entries = list(tuple_histogram_entries(payload))
    grouped = flip_ar_plotter.group_by_variable(entries)

    histograms = grouped["observable"]["SampleA"]
    ssz_label = flip_ar_plotter.build_channel_label("SampleA", "ssz")
    osz_label = flip_ar_plotter.build_channel_label("SampleA", "osz")

    fig = flip_ar_plotter.make_fig(
        histograms["ssz"],
        ssz_label,
        histo2=histograms["osz"],
        label2=osz_label,
    )

    legend_texts = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert legend_texts == [ssz_label, osz_label]
    assert fig.axes[0].get_xlabel() == "observable"

    plt.close(fig)
