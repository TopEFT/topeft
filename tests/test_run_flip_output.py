from __future__ import annotations

import cloudpickle
import gzip
import json
from collections import OrderedDict
from pathlib import Path
import types
import importlib
import sys

import hist
import pytest

from topeft.modules.runner_output import SUMMARY_KEY


class DummyRunner:
    def __init__(self, *_args, **_kwargs):
        pass

    def __call__(self, *_args, **_kwargs):
        histogram = hist.Hist(hist.axis.Regular(2, 0.0, 2.0, name="var"))
        histogram.fill(var=[0.5])
        another = histogram.copy()
        another.fill(var=[1.5])
        return OrderedDict(
            {
                ("ptabseta", "truthFlip", "sample", "nominal"): histogram,
                ("ptabseta", "truthNoFlip", "sample", "nominal"): another,
            }
        )


class DummyProcessor:
    def __init__(self, samples):
        self.samples = samples


@pytest.fixture()
def run_flip_module(monkeypatch):
    mr_module = types.SimpleNamespace(AnalysisProcessor=DummyProcessor)
    ar_module = types.SimpleNamespace(AnalysisProcessor=DummyProcessor)
    monkeypatch.setitem(sys.modules, "analysis.flip_measurement.flip_mr_processor", mr_module)
    monkeypatch.setitem(sys.modules, "analysis.flip_measurement.flip_ar_processor", ar_module)
    monkeypatch.setitem(sys.modules, "flip_mr_processor", mr_module)
    monkeypatch.setitem(sys.modules, "flip_ar_processor", ar_module)
    module = importlib.reload(importlib.import_module("analysis.flip_measurement.run_flip"))
    return module


@pytest.fixture()
def sample_json(tmp_path: Path) -> Path:
    payload = {
        "files": ["file.root"],
        "histAxisName": "sample",
        "year": "2018",
        "isData": False,
        "xsec": 1.0,
        "nEvents": 1,
        "nGenEvents": 1,
        "nSumOfWeights": 1.0,
    }
    output = tmp_path / "sample.json"
    output.write_text(json.dumps(payload))
    return output


@pytest.mark.parametrize("processor_name", ["flip_mr_processor", "flip_ar_processor"])
def test_run_flip_writes_tuple_keyed_pickle(monkeypatch, tmp_path, sample_json, processor_name, run_flip_module):
    output_dir = tmp_path / "histos"
    output_dir.mkdir()

    monkeypatch.setattr(run_flip_module.processor, "Runner", DummyRunner)
    monkeypatch.setattr(run_flip_module, "build_futures_executor", lambda *_, **__: object())

    args = [
        str(sample_json),
        "--executor",
        "futures",
        "--outpath",
        str(output_dir),
        "--outname",
        "tuple_output",
        "--processor_name",
        processor_name,
        "--environment-file",
        "none",
    ]

    assert run_flip_module.main(args) == 0

    stored = output_dir / "tuple_output.pkl.gz"
    assert stored.exists()

    with gzip.open(stored, "rb") as handle:
        payload = cloudpickle.load(handle)

    assert isinstance(payload, OrderedDict)
    assert payload

    histogram_entries = [
        value for key, value in payload.items() if isinstance(key, tuple)
    ]
    assert histogram_entries
    assert all(isinstance(histogram, hist.Hist) for histogram in histogram_entries)

    assert SUMMARY_KEY in payload
    tuple_summaries = payload[SUMMARY_KEY]
    assert isinstance(tuple_summaries, OrderedDict)
    assert tuple_summaries

    summary_keys = list(tuple_summaries.keys())
    histogram_keys = [key for key in payload.keys() if isinstance(key, tuple)]
    assert summary_keys == histogram_keys

    first_summary = next(iter(tuple_summaries.values()))
    assert set(first_summary.keys()) >= {"sumw", "values"}
