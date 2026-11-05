import ast
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.topeft_run2 import analysis_processor as module


def test_linux_memory_uses_one_status_observation(monkeypatch, capsys):
    reads = []
    def status(path):
        assert path == "/proc/self/status"
        reads.append(path)
        return io.StringIO("VmHWM:\t32768 kB\nName:\tfixture\nVmRSS:\t16384 kB\n")
    monkeypatch.setattr(module, "open", status, raising=False)
    monkeypatch.setattr(module.resource, "getrusage",
                        lambda *_: pytest.fail("rusage used despite VmHWM"))
    module._PhaseTelemetry().mark("processor_start")
    record = json.loads(capsys.readouterr().out.split("] ", 1)[1])
    assert (record["rss_mib"], record["peak_rss_mib"]) == (16., 32.)
    assert len(reads) == 1


@pytest.mark.parametrize("status,current,peak", [
    ("VmRSS: 8192 kB\n", 8., 8.),  # rusage can lag current accounting
    ("VmRSS: 1024 kB\n", 1., 4.),
    ("VmHWM: 8192 kB\n", None, 8.),
    ("Name: fixture\n", None, 4.),
    ("VmRSS: unavailable\nVmHWM: unavailable\n", None, 4.),
])
def test_missing_memory_fields_are_honest_and_fallback_is_bounded(
    monkeypatch, status, current, peak
):
    monkeypatch.setattr(module, "open", lambda *_: io.StringIO(status), raising=False)
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.resource, "getrusage",
                        lambda *_: SimpleNamespace(ru_maxrss=4096))
    assert module._phase_memory_mib() == (current, peak)


@pytest.mark.parametrize("platform,raw", [("linux", 4096), ("darwin", 4194304)])
def test_no_proc_has_no_current_rss(monkeypatch, platform, raw):
    def unavailable(*_):
        raise OSError("no proc")
    monkeypatch.setattr(module, "open", unavailable, raising=False)
    monkeypatch.setattr(module.sys, "platform", platform)
    monkeypatch.setattr(module.resource, "getrusage",
                        lambda *_: SimpleNamespace(ru_maxrss=raw))
    assert module._phase_memory_mib() == (None, 4.)
    monkeypatch.setattr(module.resource, "getrusage", unavailable)
    assert module._phase_memory_mib() == (None, None)


def test_runtime_phase_memory_invariant(capsys):
    reporter = module._PhaseTelemetry()
    for _ in range(3):
        reporter.mark("processor_complete")
    for line in capsys.readouterr().out.splitlines():
        record = json.loads(line.split("] ", 1)[1])
        if record["rss_mib"] is not None and record["peak_rss_mib"] is not None:
            assert record["peak_rss_mib"] >= record["rss_mib"]


def test_phase_reporter_retains_only_fixed_scalar_context(capsys):
    reporter = module._PhaseTelemetry(sample="sample", year="2023BPix",
                                      filename="input.root", entry_start=2,
                                      entry_stop=5, event_count=3)
    before = dict(reporter._context)
    for _ in range(3):
        reporter.mark("processor_start", rss_mb=12.5, peak_rss_mb=20.)
    records = [json.loads(line.split("] ", 1)[1])
               for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 3
    assert reporter._context == before
    assert not hasattr(reporter, "__dict__")
    assert type(reporter._wall_start_ns) is int
    assert type(reporter._cpu_start_ns) is int
    for record in records:
        assert all(type(value) in (str, int, float, bool, type(None))
                   for value in record.values())
        assert record["rss_mib"] == 12.5
        assert record["peak_rss_mib"] == 20.
        assert record["wall_elapsed_ns"] >= 0
        assert record["cpu_elapsed_ns"] >= 0
    assert records[-1]["wall_ns"] >= records[0]["wall_ns"]


def test_reporter_rejects_scientific_objects():
    with pytest.raises(TypeError, match="only scalars"):
        module._PhaseTelemetry(event_count=[1, 2, 3])


def test_processor_default_and_fixed_phase_sites():
    signature = inspect.signature(module.AnalysisProcessor)
    assert signature.parameters["phase_telemetry"].default is False
    tree = ast.parse(Path(module.__file__).read_text())
    process = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "process")
    marks = [node for node in ast.walk(process)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "telemetry" and node.func.attr == "mark"]
    phases = {node.args[0].value for node in marks}
    assert phases == {
        "processor_start", "processor_inputs_prepared",
        "processor_base_weights_and_met_ready", "processor_nominal_objects_ready",
        "processor_nominal_weights_ready", "processor_nominal_selections_ready",
        "processor_nominal_variables_ready", "processor_nominal_fills_complete",
        "processor_complete",
    }
    assert len(marks) == len(phases)
    # Every loop-local record is guarded by the single nominal variation.
    for loop in (node for node in ast.walk(process) if isinstance(node, (ast.For, ast.While))):
        loop_marks = [mark for mark in marks if mark in list(ast.walk(loop))]
        if loop_marks:
            assert isinstance(loop, ast.For)
            assert isinstance(loop.target, ast.Name) and loop.target.id == "syst_var"
            nominal_guards = [node for node in ast.walk(loop)
                              if isinstance(node, ast.If)
                              and 'syst_var == \'nominal\'' in ast.unparse(node.test)]
            for mark in loop_marks:
                assert any(mark in list(ast.walk(guard)) for guard in nominal_guards)


def test_run_cr_flag_is_forwarded_without_production_execution():
    root = Path(module.__file__).resolve().parents[2]
    source = (root / "analysis/topeft_run2/run_cr.sh").read_text()
    assert "profile_phase_telemetry=false" in source
    assert "--dry-run|--resume|--phase-telemetry)" in source
    assert "cmd_ref+=(--phase-telemetry)" in source
    assert "data_driven_cmd+=(--phase-telemetry)" in source
    wrapper = (root / "analysis/topeft_run2/fullR3_run.sh").read_text()
    assert 'EXTRA_ARGS+=("$1")' in wrapper
