from __future__ import annotations

import gzip

import cloudpickle
import hist
import numpy as np
import pytest

from analysis.topeft_run2 import run_data_driven
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules import dataDrivenEstimation
from topeft.modules.dataDrivenEstimation import DataDrivenProducer
from topeft.modules.utils import get_hist_from_pkl


@pytest.fixture
def sparse_hist_axes():
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name="pt"),
    )


def _fill_histogram(entries, axes):
    histogram = SparseHist(*axes)
    for entry in entries:
        histogram.fill(
            process=entry["process"],
            appl=entry["appl"],
            systematic=entry.get("systematic", "nominal"),
            pt=entry.get("pt", 0.5),
            weight=entry["weight"],
        )
    return histogram


def _build_hist_dict(axes):
    entries = [
        {"process": "dataUL16", "appl": "isAR_3l", "weight": 10.0},
        {"process": "TTTo2L2Nu_centralUL16", "appl": "isAR_3l", "weight": 3.0},
        {"process": "dataUL16", "appl": "isAR_2lSS_OS", "weight": 4.0},
        {"process": "TTTo2L2Nu_centralUL16", "appl": "isSR_3l", "weight": 1.0},
    ]
    main_hist = _fill_histogram(entries, axes)

    sumw2_entries = [dict(entry, weight=entry["weight"] ** 2) for entry in entries]
    sumw2_hist = _fill_histogram(sumw2_entries, axes)

    return {"nominal": main_hist, "nominal_sumw2": sumw2_hist}


def test_data_driven_producer_streams_histograms(monkeypatch, tmp_path):
    input_path = tmp_path / "input.pkl.gz"
    input_path.write_bytes(b"placeholder")

    calls = {}

    def fake_iterate(path, *, allow_empty, materialize=False):
        calls["args"] = (path, allow_empty, materialize)
        return iter(())

    monkeypatch.setattr(dataDrivenEstimation, "iterate_hist_from_pkl", fake_iterate)

    producer = DataDrivenProducer(str(input_path), "")
    assert producer.getDataDrivenHistogram() == {}
    assert calls["args"] == (str(input_path), True, False)


def test_run_data_driven_matches_inline_output(tmp_path, sparse_hist_axes):
    expected_histograms = DataDrivenProducer(_build_hist_dict(sparse_hist_axes), "").getDataDrivenHistogram()

    input_histograms = _build_hist_dict(sparse_hist_axes)
    input_path = tmp_path / "input.pkl.gz"
    with gzip.open(input_path, "wb") as stream:
        cloudpickle.dump(input_histograms, stream)

    output_path = tmp_path / "output_np.pkl.gz"
    run_data_driven._finalize_histograms(
        str(input_path),
        str(output_path),
        only_flips=False,
        apply_envelope=False,
    )

    streamed_histograms = get_hist_from_pkl(str(output_path))

    assert set(streamed_histograms) == set(expected_histograms)
    for key, expected_hist in expected_histograms.items():
        streamed_hist = streamed_histograms[key]
        np.testing.assert_allclose(
            np.asarray(streamed_hist.values(flow=True)),
            np.asarray(expected_hist.values(flow=True)),
        )
        assert list(streamed_hist.axes["process"]) == list(expected_hist.axes["process"])
