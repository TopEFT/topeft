import gzip
import importlib.util
import json
from pathlib import Path

import cloudpickle
import pytest


def _load_run_data_driven_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "analysis" / "topeft_run2" / "run_data_driven.py"
    spec = importlib.util.spec_from_file_location("run_data_driven", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


run_data_driven = _load_run_data_driven_module()


class FakeHist:
    def __init__(self, processes):
        self._processes = list(processes)
        self.axes = {"process": tuple(self._processes)}

    def remove(self, axis_name, labels):
        assert axis_name == "process"
        keep = [p for p in self._processes if p not in set(labels)]
        return FakeHist(keep)


class DummyProducer:
    output_hist = {}
    calls = []

    def __init__(self, inputHist, outputName):
        self.inputHist = inputHist
        self.outputName = outputName
        DummyProducer.calls.append((inputHist, outputName))

    def getDataDrivenHistogram(self):
        return DummyProducer.output_hist


@pytest.fixture(autouse=True)
def clear_dummy_state():
    DummyProducer.calls.clear()
    DummyProducer.output_hist = {}
    yield
    DummyProducer.calls.clear()
    DummyProducer.output_hist = {}


def _write_metadata(tmp_path: Path, *, input_path: Path, output_path: Path) -> Path:
    metadata = {
        "metadata_version": 1,
        "do_np": True,
        "resolved_years": ["16", "17"],
        "sample_years": ["16", "17", "18"],
        "input_histogram": str(input_path),
        "output_histogram": str(output_path),
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    return metadata_path


def _load_pkl(pkl_path: Path):
    with gzip.open(pkl_path, "rb") as stream:
        return cloudpickle.load(stream)


def test_run_data_driven_from_metadata(tmp_path, monkeypatch):
    input_path = tmp_path / "input.pkl.gz"
    input_path.write_bytes(b"content")
    output_path = tmp_path / "output.pkl.gz"
    metadata_path = _write_metadata(tmp_path, input_path=input_path, output_path=output_path)

    DummyProducer.output_hist = {"njets": FakeHist(["flipsUL17"])}
    monkeypatch.setattr(run_data_driven, "DataDrivenProducer", DummyProducer)

    run_data_driven.main(["--metadata-json", str(metadata_path)])

    assert DummyProducer.calls == [(str(input_path), str(output_path))]
    result = _load_pkl(output_path)
    assert list(result["njets"].axes["process"]) == ["flipsUL17"]


def test_run_data_driven_only_flips_and_envelope(tmp_path, monkeypatch):
    input_path = tmp_path / "input.pkl.gz"
    input_path.write_bytes(b"content")
    output_path = tmp_path / "output.pkl.gz"
    metadata_path = _write_metadata(tmp_path, input_path=input_path, output_path=output_path)

    DummyProducer.output_hist = {
        "njets": FakeHist(["flipsUL18", "nonpromptUL18", "ttbarUL18"])
    }
    monkeypatch.setattr(run_data_driven, "DataDrivenProducer", DummyProducer)

    envelope_calls = {}

    def fake_envelope(hist_dict):
        envelope_calls["value"] = hist_dict
        return hist_dict

    monkeypatch.setattr(run_data_driven, "get_renormfact_envelope", fake_envelope)

    run_data_driven.main(
        [
            "--metadata-json",
            str(metadata_path),
            "--only-flips",
            "--apply-renormfact-envelope",
        ]
    )

    assert "value" in envelope_calls
    result = _load_pkl(output_path)
    assert list(result["njets"].axes["process"]) == ["flipsUL18"]


def test_run_data_driven_heartbeat(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "input.pkl.gz"
    input_path.write_bytes(b"content")
    output_path = tmp_path / "output.pkl.gz"

    DummyProducer.output_hist = {
        "njets": FakeHist(["flipsUL17"]),
        "ht": FakeHist(["flipsUL17"]),
    }
    monkeypatch.setattr(run_data_driven, "DataDrivenProducer", DummyProducer)

    run_data_driven.main(
        [
            "--input-pkl",
            str(input_path),
            "--output-pkl",
            str(output_path),
            "--heartbeat-seconds",
            "0",
        ]
    )

    captured = capsys.readouterr().out
    assert "[run_data_driven] Processed" in captured
    assert "Finalized 2 histograms" in captured


def test_run_data_driven_quiet(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "input.pkl.gz"
    input_path.write_bytes(b"content")
    output_path = tmp_path / "output.pkl.gz"

    DummyProducer.output_hist = {"njets": FakeHist(["flipsUL17"])}
    monkeypatch.setattr(run_data_driven, "DataDrivenProducer", DummyProducer)

    run_data_driven.main(
        [
            "--input-pkl",
            str(input_path),
            "--output-pkl",
            str(output_path),
            "--heartbeat-seconds",
            "0",
            "--quiet",
        ]
    )

    captured = capsys.readouterr().out
    assert "[run_data_driven]" not in captured
