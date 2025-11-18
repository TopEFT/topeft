import cloudpickle
import gzip
from pathlib import Path

import numpy as np
import pytest
from hist import Hist, axis

from topeft.modules.runner_output import (
    SUMMARY_KEY,
    materialise_tuple_dict,
    normalise_runner_output,
    tuple_dict_stats,
)


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

    restored_hist = restored[tuple_key]
    assert type(restored_hist) is type(hist)
    np.testing.assert_allclose(
        restored_hist.values(flow=True),
        hist.values(flow=True),
    )

    assert SUMMARY_KEY not in restored
    assert restored["metadata"] == {"note": "retained"}


def test_normalise_runner_output_rejects_legacy_tuple(tmp_path):
    hist = _build_histogram()
    bad_key = ("observable", "chan", "Sample", "nominal")
    payload = {bad_key: hist}

    with pytest.raises(ValueError):
        normalise_runner_output(payload)


def test_materialise_tuple_dict_rejects_categorical_axis():
    categorical_hist = Hist.new.StrCategory(["a", "b"], name="category").Weight()
    histogram_key = ("observable", "chan", "app", "Sample", "nominal")

    with pytest.raises(ValueError):
        materialise_tuple_dict({histogram_key: categorical_hist})
