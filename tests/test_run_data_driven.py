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
    get_calls = 0
    iter_calls = 0

    def __init__(self, inputHist, outputName, iterator_mode=False):
        self.inputHist = inputHist
        self.outputName = outputName
        self.iterator_mode = iterator_mode
        DummyProducer.calls.append((inputHist, outputName, iterator_mode))

    def getDataDrivenHistogram(self):
        DummyProducer.get_calls += 1
        return DummyProducer.output_hist

    def iter_data_driven_histograms(self):
        DummyProducer.iter_calls += 1
        yield from DummyProducer.output_hist.items()


@pytest.fixture(autouse=True)
def clear_dummy_state():
    DummyProducer.calls.clear()
    DummyProducer.output_hist = {}
    DummyProducer.get_calls = 0
    DummyProducer.iter_calls = 0
    yield
    DummyProducer.calls.clear()
    DummyProducer.output_hist = {}
    DummyProducer.get_calls = 0
    DummyProducer.iter_calls = 0


def _write_metadata(tmp_path: Path, *, input_path: Path, output_path: Path) -> Path:
    metadata = {
        "metadata_version": 2,
        "do_np": True,
        "np_postprocess": "defer",
        "pretend_mode": False,
        "apply_renormfact_envelope": False,
        "resolved_years": ["16", "17"],
        "sample_years": ["16", "17", "18"],
        "input_histogram": str(input_path),
        "output_histogram": str(output_path),
        "metadata_path": str(tmp_path / "metadata.json"),
        "followup_command": "python analysis/topeft_run2/run_data_driven.py --metadata-json metadata.json",
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

    assert DummyProducer.calls == [(str(input_path), str(output_path), True)]
    assert DummyProducer.iter_calls == 1
    assert DummyProducer.get_calls == 0
    result = _load_pkl(output_path)
    assert list(result["njets"].axes["process"]) == ["flipsUL17"]


def test_run_data_driven_only_flips_and_envelope(tmp_path, monkeypatch):
    input_path = tmp_path / "input.pkl.gz"
    input_path.write_bytes(b"content")
    output_path = tmp_path / "output.pkl.gz"
    metadata_path = _write_metadata(tmp_path, input_path=input_path, output_path=output_path)
    payload = json.loads(metadata_path.read_text())
    payload["apply_renormfact_envelope"] = True
    metadata_path.write_text(json.dumps(payload))

    DummyProducer.output_hist = {
        "njets": FakeHist(["flipsUL18", "nonpromptUL18", "ttbarUL18"])
    }
    monkeypatch.setattr(run_data_driven, "DataDrivenProducer", DummyProducer)

    envelope_calls = {}

    def fake_envelope(histo, **_kwargs):
        envelope_calls["value"] = histo
        return histo

    monkeypatch.setattr(
        run_data_driven, "apply_renormfact_envelope_to_histogram", fake_envelope
    )

    run_data_driven.main(
        [
            "--metadata-json",
            str(metadata_path),
            "--only-flips",
        ]
    )

    assert "value" in envelope_calls
    assert DummyProducer.calls == [(str(input_path), str(output_path), True)]
    assert DummyProducer.iter_calls == 1
    assert DummyProducer.get_calls == 0
    result = _load_pkl(output_path)
    assert list(result["njets"].axes["process"]) == ["flipsUL18"]


def test_run_data_driven_rejects_missing_required_metadata_keys(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "metadata_version": 2,
                "do_np": True,
                "np_postprocess": "defer",
            }
        )
    )

    with pytest.raises(ValueError, match="missing required keys"):
        run_data_driven.main(["--metadata-json", str(metadata_path)])


def test_run_data_driven_rejects_inconsistent_metadata_years(tmp_path):
    input_path = tmp_path / "input.pkl.gz"
    input_path.write_bytes(b"content")
    output_path = tmp_path / "output.pkl.gz"
    metadata_path = _write_metadata(tmp_path, input_path=input_path, output_path=output_path)
    payload = json.loads(metadata_path.read_text())
    payload["resolved_years"] = ["16", "2022"]
    metadata_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="requested years"):
        run_data_driven.main(["--metadata-json", str(metadata_path)])


def test_run_data_driven_legacy_dict_mode(tmp_path, monkeypatch):
    input_path = tmp_path / "input.pkl.gz"
    input_path.write_bytes(b"content")
    output_path = tmp_path / "output.pkl.gz"

    DummyProducer.output_hist = {"njets": FakeHist(["flipsUL18", "ttbarUL18"])}
    monkeypatch.setattr(run_data_driven, "DataDrivenProducer", DummyProducer)

    monkeypatch.setattr(
        run_data_driven.utils,
        "dump_dict_streaming",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("streaming writer should not be used in legacy mode")
        ),
    )

    def _fake_dump_to_pkl(path, payload):
        with gzip.open(path, "wb") as stream:
            cloudpickle.dump(payload, stream)

    monkeypatch.setattr(run_data_driven.utils, "dump_to_pkl", _fake_dump_to_pkl)

    run_data_driven.main(
        [
            "--input-pkl",
            str(input_path),
            "--output-pkl",
            str(output_path),
            "--legacy-dict-mode",
        ]
    )

    assert DummyProducer.calls == [(str(input_path), str(output_path), False)]
    assert DummyProducer.get_calls == 1
    assert DummyProducer.iter_calls == 0
    result = _load_pkl(output_path)
    assert list(result["njets"].axes["process"]) == ["flipsUL18", "ttbarUL18"]


def test_run_data_driven_metadata_can_force_legacy_mode(tmp_path, monkeypatch):
    input_path = tmp_path / "input.pkl.gz"
    input_path.write_bytes(b"content")
    output_path = tmp_path / "output.pkl.gz"
    metadata_path = _write_metadata(tmp_path, input_path=input_path, output_path=output_path)

    DummyProducer.output_hist = {"njets": FakeHist(["flipsUL18", "ttbarUL18"])}
    monkeypatch.setattr(run_data_driven, "DataDrivenProducer", DummyProducer)
    monkeypatch.setattr(
        run_data_driven.utils,
        "dump_dict_streaming",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("streaming writer should not be used in legacy mode")
        ),
    )

    def _fake_dump_to_pkl(path, payload):
        with gzip.open(path, "wb") as stream:
            cloudpickle.dump(payload, stream)

    monkeypatch.setattr(run_data_driven.utils, "dump_to_pkl", _fake_dump_to_pkl)

    run_data_driven.main(
        [
            "--metadata-json",
            str(metadata_path),
            "--legacy-dict-mode",
        ]
    )

    assert DummyProducer.calls == [(str(input_path), str(output_path), False)]
    assert DummyProducer.get_calls == 1
    assert DummyProducer.iter_calls == 0


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
