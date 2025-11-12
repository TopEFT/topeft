import cloudpickle
import gzip
from pathlib import Path

import numpy as np
import pytest
from hist import Hist

from topeft.modules.runner_output import normalise_runner_output, tuple_dict_stats


def _build_histogram() -> Hist:
    histogram = Hist.new.Regular(2, 0.0, 2.0, name="observable").Weight()
    histogram.fill(observable=[0.25, 0.75, 1.5], weight=[1.0, 1.0, 1.0])
    return histogram


def test_normalise_runner_output_preserves_tuple_keys(tmp_path):
    hist = _build_histogram()
    tuple_key = ("observable", "chan", "app", "Sample", "nominal")
    payload = {
        tuple_key: hist,
        "metadata": {"note": "retained"},
    }

    serialised = normalise_runner_output(payload)
    total_bins, filled_bins = tuple_dict_stats(serialised)
    assert total_bins == hist.values(flow=True).size
    assert filled_bins == int(np.count_nonzero(hist.values(flow=True)))

    destination = Path(tmp_path) / "artifact.pkl.gz"
    with gzip.open(destination, "wb") as fout:
        cloudpickle.dump(serialised, fout)

    with gzip.open(destination, "rb") as fin:
        restored = cloudpickle.load(fin)

    keys = [key for key in restored.keys() if isinstance(key, tuple)]
    assert keys == [tuple_key]

    summary = restored[tuple_key]
    assert pytest.approx(summary["sumw"]) == pytest.approx(hist.values(flow=True).sum())
    assert summary["values"].shape == hist.values(flow=True).shape
    assert summary["variances"] is None or summary["variances"].shape == summary["values"].shape
    assert restored["metadata"] == {"note": "retained"}
