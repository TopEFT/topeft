import gzip
from pathlib import Path

import cloudpickle
import numpy as np
from hist import axis

from analysis.mc_validation.plot_utils import (
    build_dataset_histograms,
    component_values,
    tuple_histogram_items,
)
from topcoffea.modules.histEFT import HistEFT


def _build_histogram(values):
    histogram = HistEFT(axis.Regular(4, 0.0, 4.0, name="observable"), wc_names=[], label="Events")
    histogram.fill(observable=np.asarray(values), weight=np.ones(len(values)))
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
    assert list(histogram.axes["channel"]) == [
        "3l_onZ_1b",
        "inclusive",
    ]

    private_entry = histogram[{"dataset": "ttHJet_privateUL18", "channel": "3l_onZ_1b", "systematic": "nominal"}]
    np.testing.assert_allclose(private_entry.values(), private_hist.values())

