import builtins
import pathlib
import sys

import pytest

from analysis.topeft_run2 import tauFitter


def test_tau_fitter_remains_importable_for_legacy_parity():
    assert callable(tauFitter.prepare_taufitter_histograms)


def test_direct_tau_fitter_execution_fails_before_work(tmp_path, monkeypatch):
    script_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "analysis"
        / "topeft_run2"
        / "tauFitter.py"
    )
    source = script_path.read_text(encoding="utf-8")
    code = compile(source, str(script_path), "exec")
    sentinel_input = tmp_path / "must_not_be_opened.pkl.gz"
    sentinel_output = tmp_path / "must_not_be_created"
    open_calls = []
    original_open = builtins.open

    def recording_open(file, *args, **kwargs):
        open_calls.append(file)
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_open)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script_path), "--pkl-file-path", str(sentinel_input)],
    )

    namespace = {
        "__builtins__": builtins,
        "__file__": str(script_path),
        "__name__": "__main__",
    }
    with pytest.raises(SystemExit) as exc_info:
        exec(code, namespace)

    message = str(exc_info.value)
    assert "tauFitter.py is deprecated" in message
    assert "faketau_sf_fitter.py" in message
    assert "No fit or output was produced" in message
    assert open_calls == []
    assert "main" not in namespace
    assert "np" not in namespace
    assert not sentinel_input.exists()
    assert not sentinel_output.exists()
