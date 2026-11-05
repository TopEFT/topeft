"""Bounded transport tests; no manager, network, worker or scheduler required."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
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


def archive_shell(tmp_path, action, *, fail_copy=False):
    """Run maintained shell lifecycle functions only; no campaign launcher.

    The state receipt writer is a no-op: these tests qualify file separation,
    not campaign state transitions or production recovery.
    """
    source = (Path(__file__).parents[1] / "analysis/topeft_run2/run_cr.sh").read_text()
    functions = ""
    for name in ("archive_native_wq_logs", "assert_native_wq_logs_clean"):
        body = source.split(name + "() {", 1)[1].split("\n}\n", 1)[0]
        functions += name + "() {" + body + "\n}\n"
    shell = '''set -euo pipefail
output_dir="$1"
native_wq_log_dir="$1/work_queue_logs/fixture/unit/source_native"
production_component=fixture
production_state_path=unused
validation_backend=""
native_wq_log_names=(debug.log tr.log stats.log tasks.log)
production_state_tool() { return 0; }
'''
    if fail_copy:
        shell += '''cp() {
  if [[ "$2" == */phase_telemetry.jsonl ]]; then return 1; fi
  command cp "$@"
}
'''
    return subprocess.run(["/bin/bash", "--noprofile", "--norc", "-c",
                           shell + functions + action, "archive_fixture", str(tmp_path)],
                          text=True, capture_output=True)


def test_processing_attempt_archives_separate_reused_native_ids(transport, tmp_path):
    native = tmp_path / "work_queue_logs/fixture/unit/source_native"
    native.mkdir(parents=True)
    stdout = tmp_path / "stdout.log"
    snapshots = []
    for phase in ("processor_start", "processor_complete"):
        assert archive_shell(tmp_path, "assert_native_wq_logs_clean unit").returncode == 0
        stdout.write_text(stdout_record(phase))
        transport._persist_task_phase_telemetry(SimpleNamespace(id=1, outfile_stdout=stdout),
                                               native / "phase_telemetry.jsonl")
        (native / "tasks.log").write_text("1,processing,done\n")
        result = archive_shell(tmp_path, "archive_native_wq_logs unit source")
        assert result.returncode == 0, result.stderr
        snapshots.append(phase)
        assert not (native / "phase_telemetry.jsonl").exists()
    for attempt, phase in zip(("source", "source_resume_1"), snapshots):
        archive = native.parent / attempt
        rows = [json.loads(line) for line in (archive / "phase_telemetry.jsonl").read_text().splitlines()]
        assert [(row["task_id"], row["phase"]) for row in rows] == [(1, phase)]
        assert (archive / "tasks.log").read_text() == "1,processing,done\n"


def test_failed_optional_archive_preserves_original_and_prevents_next_mix(tmp_path):
    native = tmp_path / "work_queue_logs/fixture/unit/source_native"
    native.mkdir(parents=True)
    sidecar = native / "phase_telemetry.jsonl"
    original = stdout_record()
    sidecar.write_text(original)
    (native / "tasks.log").write_text("1,processing,done\n")
    archived = archive_shell(tmp_path, "archive_native_wq_logs unit source", fail_copy=True)
    assert archived.returncode == 0  # original scientific stage stays successful
    assert "optional phase telemetry archive failed" in archived.stderr
    assert sidecar.read_text() == original
    assert not (native / "tasks.log").exists()
    refused = archive_shell(tmp_path, "assert_native_wq_logs_clean unit", fail_copy=True)
    assert refused.returncode != 0
    assert "refusing a new invocation" in refused.stderr
    assert sidecar.read_text() == original
    # Once copying works, the existing preflight archives the orphan before any
    # new invocation; no registry, deletion fallback or new hierarchy is used.
    separated = archive_shell(tmp_path, "assert_native_wq_logs_clean unit")
    assert separated.returncode == 0, separated.stderr
    assert not sidecar.exists()
    assert (native.parent / "source_resume_2/phase_telemetry.jsonl").read_text() == original


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
