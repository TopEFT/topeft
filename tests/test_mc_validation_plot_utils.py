import gzip
from pathlib import Path

import cloudpickle
import numpy as np
from hist import Hist, axis, storage

from analysis.mc_validation.plot_utils import (
    build_dataset_histograms,
    component_labels,
    component_values,
    filter_tuple_histograms,
    tuple_histogram_items,
)

try:  # pragma: no cover - optional dependency in CI
    from topcoffea.modules.HistEFT import HistEFT as _HistEFT
except ModuleNotFoundError:  # pragma: no cover - fallback used when topcoffea is absent
    _HistEFT = None


def _build_histogram(values):
    dense_axis = axis.Regular(4, 0.0, 4.0, name="observable")
    if _HistEFT is not None:
        histogram = _HistEFT(dense_axis, wc_names=[], label="Events")
    else:
        histogram = Hist(dense_axis, storage=storage.Double(), label="Events")
    histogram.fill(observable=np.asarray(values), weight=np.ones(len(values)))
    return histogram


def _constant_histogram(weight: float) -> Hist:
    histogram = Hist(axis.Regular(1, 0.0, 1.0, name="x"), storage=storage.Double())
    histogram.view()[...] = weight
    return histogram


def test_tuple_histograms_roundtrip(tmp_path):
    central_hist = _build_histogram([0.5, 1.5])
    private_hist = _build_histogram([0.5, 2.5])

    payload = {
        ("observable", "inclusive", "inclusive", "ttH_centralUL18", "nominal"): central_hist,
        ("observable", "3l_onZ_1b", "inclusive", "ttHJet_privateUL18", "nominal"): private_hist,
    }

    destination = Path(tmp_path) / "tuple_payload.pkl.gz"
    with gzip.open(destination, "wb") as fout:
        cloudpickle.dump(payload, fout)

    with gzip.open(destination, "rb") as fin:
        restored = cloudpickle.load(fin)

    tuple_entries = tuple_histogram_items(restored)
    assert component_values(tuple_entries, "sample") == [
        "ttHJet_privateUL18",
        "ttH_centralUL18",
    ]

    rebuilt = build_dataset_histograms(restored)
    assert "observable" in rebuilt

    histogram = rebuilt["observable"]
    assert list(histogram.axes["dataset"]) == [
        "ttHJet_privateUL18",
        "ttH_centralUL18",
    ]
    assert list(histogram.axes["application"]) == ["inclusive"]
    assert list(histogram.axes["channel"]) == [
        "3l_onZ_1b",
        "inclusive",
    ]

    private_entry = histogram[
        {
            "dataset": "ttHJet_privateUL18",
            "application": "inclusive",
            "channel": "3l_onZ_1b",
            "systematic": "nominal",
        }
    ]
    np.testing.assert_allclose(private_entry.values(), private_hist.values())


def test_application_axis_preserved_and_labelled():
    hist_store = {
        ("observable", "chan", "appA", "sampleA", "nominal"): _constant_histogram(1.0),
        ("observable", "chan", "appB", "sampleA", "nominal"): _constant_histogram(2.0),
    }

    rebuilt = build_dataset_histograms(hist_store)
    rebuilt_hist = rebuilt["observable"]

    selector = {"dataset": "sampleA", "channel": "chan", "systematic": "nominal"}
    assert rebuilt_hist[{**selector, "application": "appA"}].values()[0] == 1.0
    assert rebuilt_hist[{**selector, "application": "appB"}].values()[0] == 2.0


def test_component_labels_include_application_region():
    tuple_entries = tuple_histogram_items(
        {
            ("observable", "chan", "appA", "sampleA", "nominal"): _constant_histogram(1.0),
            ("observable", "chan", "appB", "sampleA", "nominal"): _constant_histogram(2.0),
        }
    )

    labels = component_labels(tuple_entries, "sample", include_application=True)
    assert labels == ["sampleA (appA)", "sampleA (appB)"]


def test_filter_tuple_histograms_supports_application_component():
    tuple_entries = tuple_histogram_items(
        {
            ("observable", "chan", "appA", "sampleA", "nominal"): _constant_histogram(1.0),
            ("observable", "chan", "appB", "sampleB", "nominal"): _constant_histogram(2.0),
        }
    )

    filtered = filter_tuple_histograms(tuple_entries, application="appA")
    assert set(filtered.keys()) == {("observable", "chan", "appA", "sampleA", "nominal")}

