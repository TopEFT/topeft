"""Focused contract tests for the external node-pressure supervisor."""

import json
import os
import signal

import pytest

from topeft.modules import node_memory_guard as guard


def _meminfo(path, *, total=100, available=50):
    path.write_text(f"MemTotal: {total} kB\nMemAvailable: {available} kB\n")
    return path


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_meminfo_formula_and_invalid_metric(tmp_path):
    path = _meminfo(tmp_path / "meminfo", total=100, available=40)
    assert guard.read_node_pressure(path) == {
        "mem_total_kb": 100, "mem_available_kb": 40,
        "pressure_used_bytes": 60 * 1024,
    }
    path.write_text("MemTotal: 100 kB\n")
    with pytest.raises(ValueError, match="both required"):
        guard.read_node_pressure(path)
    path.write_text("MemTotal: 100 kB\nMemAvailable: 101 kB\n")
    with pytest.raises(ValueError, match="relationship"):
        guard.read_node_pressure(path)
    assert guard.DEFAULT_SAMPLING_INTERVAL_SECONDS == 0.5


@pytest.mark.parametrize("limit, expected", [
    (60 * 1024 + 1, "normal"),
    (60 * 1024, "threshold_killed"),
    (60 * 1024 - 1, "threshold_killed"),
])
def test_threshold_boundary_and_owned_group(tmp_path, monkeypatch, limit, expected):
    meminfo = _meminfo(tmp_path / "meminfo", total=100, available=40)
    evidence = tmp_path / "evidence.jsonl"
    actual_killpg = os.killpg
    actual_read_identity = guard._read_process_identity
    signals = []
    identity_reads = []

    def observe_identity(pid):
        identity_reads.append(pid)
        return actual_read_identity(pid)

    def observe_killpg(pgid, sig):
        events = _events(evidence)
        assert events[-1]["event"] == "threshold"
        assert events[-1]["sample"]["pressure_used_bytes"] == 60 * 1024
        assert pgid == events[0]["identity"]["pid"]
        assert pgid != os.getpgrp()
        assert sig == signal.SIGKILL
        assert identity_reads[-1] == pgid
        assert len(identity_reads) >= 2
        signals.append((pgid, sig))
        actual_killpg(pgid, sig)

    monkeypatch.setattr(guard, "_read_process_identity", observe_identity)
    monkeypatch.setattr(guard.os, "killpg", observe_killpg)
    result = guard.run_guarded(
        ["/bin/sleep", "0.15"], hard_limit_bytes=limit,
        evidence_path=evidence, meminfo_path=meminfo,
        sampling_interval_seconds=0.02,
    )
    assert result["termination_kind"] == expected
    assert result["peak_pressure_used_bytes"] == 60 * 1024
    assert result["minimum_mem_available_kb"] == 40
    assert result["pgid"] == result["pid"] != os.getpgrp()
    assert result["start_ticks"] > 0
    assert (len(signals), result["signal"]) == (
        (0, None) if expected == "normal" else (1, signal.SIGKILL)
    )
    assert [event["event"] for event in _events(evidence)].count("start") == 1


def test_identity_mismatch_refuses_to_signal(monkeypatch):
    recorded = {"pid": 987654, "pgid": 987654, "session": 987654,
                "state": "S", "start_ticks": 12}
    monkeypatch.setattr(
        guard, "_read_process_identity", lambda _pid: {**recorded, "start_ticks": 13}
    )
    monkeypatch.setattr(
        guard.os, "killpg", lambda *_args: pytest.fail("identity mismatch signaled")
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        guard._kill_owned_group(recorded["pid"], recorded)


def test_unreadable_metric_fails_closed_and_kills_owned_child(tmp_path):
    evidence = tmp_path / "metric_failure.jsonl"
    result = guard.run_guarded(
        ["/bin/sleep", "10"], hard_limit_bytes=1,
        evidence_path=evidence, meminfo_path=tmp_path / "missing",
        sampling_interval_seconds=0.02,
    )
    assert result["termination_kind"] == "metric_failed"
    assert result["signal"] == signal.SIGKILL
    assert result["returncode"] == -signal.SIGKILL
    assert [event["event"] for event in _events(evidence)] == [
        "start", "metric_failure", "result",
    ]


def test_evidence_write_failure_stops_identified_child(tmp_path, monkeypatch):
    evidence = tmp_path / "unwritable_after_start.jsonl"
    real_write_event = guard._write_event
    kill_calls = []
    real_killpg = os.killpg

    def fail_threshold_write(handle, event):
        if event["event"] == "threshold":
            raise OSError("synthetic evidence write failure")
        real_write_event(handle, event)

    def observe_killpg(pgid, sig):
        assert _events(evidence)[0]["identity"]["pid"] == pgid
        assert pgid != os.getpgrp()
        kill_calls.append((pgid, sig))
        real_killpg(pgid, sig)

    monkeypatch.setattr(guard, "_write_event", fail_threshold_write)
    monkeypatch.setattr(guard.os, "killpg", observe_killpg)
    with pytest.raises(OSError, match="evidence write failure"):
        guard.run_guarded(
            ["/bin/sleep", "10"], hard_limit_bytes=1,
            evidence_path=evidence,
            meminfo_path=_meminfo(tmp_path / "meminfo"),
        )
    assert len(kill_calls) == 1
    assert kill_calls[0][1] == signal.SIGKILL


def test_normal_completion_is_recorded_once(tmp_path):
    evidence = tmp_path / "normal.jsonl"
    result = guard.run_guarded(
        ["/bin/true"], hard_limit_bytes=10**30,
        evidence_path=evidence, meminfo_path=_meminfo(tmp_path / "meminfo"),
    )
    assert result["termination_kind"] == "normal"
    assert result["returncode"] == 0
    assert result["signal"] is None
    assert [event["event"] for event in _events(evidence)] == ["start", "result"]


def test_module_cli_runs_one_child_with_explicit_evidence(tmp_path):
    evidence = tmp_path / "cli.jsonl"
    meminfo = _meminfo(tmp_path / "meminfo")
    assert guard.main([
        "--hard-limit-bytes", str(10**30),
        "--evidence-path", str(evidence),
        "--meminfo-path", str(meminfo),
        "--", "/bin/true",
    ]) == 0
    assert [event["event"] for event in _events(evidence)] == ["start", "result"]
