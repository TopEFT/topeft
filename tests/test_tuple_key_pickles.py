from __future__ import annotations

import cloudpickle
import gzip

import pytest
from hist import Hist, axis, storage

from analysis.flip_measurement.plot_utils import load_tuple_histogram_entries
from topeft.modules.dataDrivenEstimation import _validate_tuple_histograms


def _write_payload(tmp_path, payload):
    out = tmp_path / "payload.pkl.gz"
    with gzip.open(out, "wb") as fout:
        cloudpickle.dump(payload, fout)
    return out


def _basic_hist():
    histogram = Hist(axis.Regular(2, 0.0, 2.0, name="var"), storage=storage.Double())
    histogram.fill(var=[0.5, 1.5])
    return histogram


def test_tuple_histogram_entries_require_application(tmp_path):
    histogram = _basic_hist()
    payload = {
        ("var", "chan", None, "sample", "nominal"): histogram,
    }
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match="application region"):
        list(load_tuple_histogram_entries(str(path)))


def test_tuple_histogram_entries_reject_legacy_keys(tmp_path):
    histogram = _basic_hist()
    payload = {
        ("var", "chan", "sample", "nominal"): histogram,
    }
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match="5-tuples"):
        list(load_tuple_histogram_entries(str(path)))


def test_tuple_histogram_entries_accept_application_tag(tmp_path):
    histogram = _basic_hist()
    payload = {
        ("var", "chan", "isAR", "sample", "nominal"): histogram,
        "meta": {"note": "representative"},
    }
    path = _write_payload(tmp_path, payload)

    entries = list(load_tuple_histogram_entries(str(path)))
    assert entries
    assert all(entry.application == "isAR" for entry in entries)
    assert all(entry.sample == "sample" for entry in entries)


def test_data_driven_validator_requires_application(tmp_path):
    payload = {
        ("var", "chan", "", "sample", "nominal"): object(),
    }
    path = _write_payload(tmp_path, payload)
    with gzip.open(path, "rb") as fin:
        loaded = cloudpickle.load(fin)

    with pytest.raises(ValueError, match="application region"):
        _validate_tuple_histograms(loaded)
