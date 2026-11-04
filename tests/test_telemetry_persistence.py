"""Bounded transport tests; no manager, network, worker or scheduler required."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from analysis.topeft_run2 import analysis_processor


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setitem(sys.modules, "analysis_processor", analysis_processor)
    source = Path(__file__).parents[1] / "analysis/topeft_run2/run_analysis.py"
    spec = importlib.util.spec_from_file_location("telemetry_run_analysis", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stdout_record(phase="processor_complete"):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        analysis_processor._PhaseTelemetry(sample="not duplicated").mark(phase)
    return output.getvalue()


def test_selective_append_and_native_task_join(transport, tmp_path):
    source = tmp_path / "stdout.log"
    source.write_text('ordinary full stdout must disappear\n[phase_telemetry] invalid\n'
                      '[phase_telemetry] []\n[phase_telemetry] {}\n' + stdout_record())
    sidecar = tmp_path / "phase_telemetry.jsonl"
    for task_id in (21, 22):
        transport._persist_task_phase_telemetry(
            SimpleNamespace(id=task_id, outfile_stdout=source), sidecar)
    records = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert [r["task_id"] for r in records] == [21, 22]
    assert set(records[0]) == {"task_id", "schema_version", "phase", "wall_ns", "cpu_ns",
                              "wall_elapsed_ns", "cpu_elapsed_ns", "rss_mib", "peak_rss_mib"}
    assert "ordinary" not in sidecar.read_text()
    assert "sample" not in sidecar.read_text()


@pytest.mark.parametrize("content", ["", "ordinary stdout\n", '[phase_telemetry] {}\n'])
def test_no_records_no_sidecar(transport, tmp_path, content):
    source = tmp_path / "stdout.log"
    source.write_text(content)
    sidecar = tmp_path / "phase_telemetry.jsonl"
    transport._persist_task_phase_telemetry(SimpleNamespace(id=1, outfile_stdout=source), sidecar)
    assert not sidecar.exists()


def test_optional_io_failure_warns_and_preserves_task(transport, tmp_path, capsys):
    source = tmp_path / "stdout.log"
    source.write_text(stdout_record())
    task = SimpleNamespace(id=7, category="processing", outfile_stdout=source, successful=lambda: True)
    queue = SimpleNamespace(wait=lambda timeout=None: task,
                            executor=SimpleNamespace(tasks_accum_log=str(tmp_path / "absent/tasks.log")))
    transport._install_phase_telemetry_persistence(queue)
    assert queue.wait() is task
    assert "optional phase telemetry persistence failed" in capsys.readouterr().err


def test_wait_composition_is_idempotent_and_uses_current_executor(transport, tmp_path):
    source = tmp_path / "stdout.log"
    source.write_text(stdout_record())
    task = SimpleNamespace(id=7, category="processing", outfile_stdout=source, successful=lambda: True)
    queue = SimpleNamespace(wait=lambda timeout=None: task,
                            executor=SimpleNamespace(tasks_accum_log=str(tmp_path / "tasks.log")))
    transport._install_phase_telemetry_persistence(queue)
    original = queue.wait
    transport._install_phase_telemetry_persistence(queue)
    assert queue.wait is original
    assert queue.wait() is task
    assert len((tmp_path / "phase_telemetry.jsonl").read_text().splitlines()) == 1
    other = tmp_path / "other"
    other.mkdir()
    queue.executor.tasks_accum_log = str(other / "tasks.log")
    queue.wait()
    assert (other / "phase_telemetry.jsonl").exists()


@pytest.mark.parametrize("category,success", [("preprocessing", True), ("accumulating", True), ("processing", False)])
def test_nonprocessing_or_failed_tasks_are_not_copied(transport, tmp_path, category, success):
    task = SimpleNamespace(category=category, successful=lambda: success)
    queue = SimpleNamespace(wait=lambda timeout=None: task)
    transport._install_phase_telemetry_persistence(queue)
    assert queue.wait() is task


@pytest.mark.parametrize("enabled", [False, True])
def test_custom_init_opt_in_and_admission_composition(transport, monkeypatch, enabled):
    calls = []
    queue = SimpleNamespace(wait=lambda timeout=None: None)

    def admission(q, **kwargs):
        previous = q.wait
        def admitted(timeout=None):
            calls.append("admission")
            return previous(timeout)
        q.wait = admitted

    monkeypatch.setattr(transport, "install_processing_admission", admission)
    transport._configure_wq_categories(queue, admission_base="fixture", phase_telemetry=enabled)
    assert getattr(queue, "_phase_telemetry_installed", False) is enabled
    assert queue.wait() is None
    assert calls == ["admission"]
