"""Focused read-only behavior checks for the manifest-driven dashboard."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest


dashboard_source = Path(__file__).resolve().parents[1] / "analysis/topeft_run2/monitoring/datacard_matrix_dashboard.sh"


def dashboard_case(tmp_path):
    script_dir = tmp_path / "analysis/topeft_run2/monitoring"
    script_dir.mkdir(parents=True)
    dashboard = script_dir / "datacard_matrix_dashboard.sh"
    shutil.copy2(dashboard_source, dashboard)
    runner = script_dir.parent / "run_datacard_matrix_resumable.sh"
    runner.write_text('#!/bin/sh\n[ "$1" = "--status" ] || exit 90\ncat "' + str(tmp_path / "status.json") + '"\n')
    runner.chmod(0o755)

    control_root = tmp_path / "control"
    control_root.mkdir()
    rows = [
        {
            "row_id": f"unit_{index}",
            "logical_row_id": "logical_a" if index < 3 else "logical_b",
            "attempt_id": "attempt_01",
            "physical_channels": [f"channel_{index}_{part}" for part in range(index)],
            "output_root": str(tmp_path / "cards" / f"unit_{index}"),
            "log_path": str(control_root / f"unit_{index}.log"),
        }
        for index in (1, 2, 3)
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "topeft_datacard_matrix_v3", "control_root": str(control_root), "rows": rows}))
    status = {
        "schema": "topeft_datacard_matrix_read_only_result_v2",
        "mode": "status",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "mutated": False,
        "lock_held": True,
        "owner_metadata": {
            "current_row_id": "unit_2",
            "current_attempt_id": "attempt_01",
            "current_row_started_at": "2026-01-01T00:00:00Z",
        },
        "rows": [
            {"row_id": row["row_id"], "attempt_id": row["attempt_id"], "status": state}
            for row, state in zip(rows, ("complete", "active", "not_started"))
        ],
    }
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps(status))
    return dashboard, manifest, status_path, control_root


def run_once(dashboard, manifest):
    return subprocess.run([str(dashboard), "--once", "--log-lines", "1", str(manifest)],
                          capture_output=True, text=True, check=False)


def test_dashboard_counts_targets_and_reads_only_current_log(tmp_path):
    dashboard, manifest, _, control_root = dashboard_case(tmp_path)
    current_log = control_root / "unit_2.log"
    current_log.write_text("older\nlatest\n")
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in control_root.iterdir()}

    result = run_once(dashboard, manifest)

    assert result.returncode == 0, result.stderr
    assert "Execution units: 1 / 3" in result.stdout
    assert "Physical targets: 1 / 6" in result.stdout
    assert "Active units   : 1 (2 targets)  Pending: 1  Problems: 0" in result.stdout
    assert "logical_a, logical_b" in result.stdout
    assert "channel_3_0,channel_3_1,channel_3_2" in result.stdout
    assert "2026-01-01T00:00:00Z" in result.stdout
    assert "latest" in result.stdout
    assert "older" not in result.stdout
    assert before == {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in control_root.iterdir()}


def test_dashboard_waits_for_absent_active_log(tmp_path):
    dashboard, manifest, _, control_root = dashboard_case(tmp_path)
    result = run_once(dashboard, manifest)
    assert result.returncode == 0, result.stderr
    assert f"[waiting for log] {control_root / 'unit_2.log'}" in result.stdout
    assert list(control_root.iterdir()) == []


@pytest.mark.parametrize("state,label", [
    ("stale_owner_free_lock", "[STALE OWNER]"),
    ("partial_staging_attempt", "[PARTIAL]"),
    ("complete_validated_staging_not_promoted", "[STAGED READY]"),
    ("promoted_incomplete", "[PUBLISHING]"),
    ("promoted_complete_without_receipt", "[RECEIPT PENDING]"),
    ("held_lock_active_or_unknown", "[LOCK UNKNOWN]"),
    ("invalid_receipt", "[INVALID RECEIPT]"),
    ("canonical_collision_or_ambiguous_state", "[COLLISION]"),
])
def test_dashboard_renders_runner_recovery_state_without_claiming_completion(tmp_path, state, label):
    dashboard, manifest, status_path, _ = dashboard_case(tmp_path)
    payload = json.loads(status_path.read_text())
    payload["rows"][1]["status"] = state
    status_path.write_text(json.dumps(payload))
    observed = run_once(dashboard, manifest)
    assert observed.returncode == 0, observed.stderr
    assert label in observed.stdout
    assert "Execution units: 1 / 3" in observed.stdout
    assert "does not establish physics validation" in observed.stdout


def test_dashboard_uses_runner_active_log_path(tmp_path):
    dashboard, manifest, status_path, control_root = dashboard_case(tmp_path)
    staged_log = control_root / "staged.log"
    staged_log.write_text("staged producer log\n")
    payload = json.loads(status_path.read_text())
    payload["rows"][1]["active_log_path"] = str(staged_log)
    status_path.write_text(json.dumps(payload))
    observed = run_once(dashboard, manifest)
    assert observed.returncode == 0, observed.stderr
    assert "staged producer log" in observed.stdout


@pytest.mark.parametrize("change", ["wrong_schema", "row_mismatch", "unknown_state", "mutated"])
def test_dashboard_rejects_inconsistent_status(tmp_path, change):
    dashboard, manifest, status_path, _ = dashboard_case(tmp_path)
    status = json.loads(status_path.read_text())
    if change == "wrong_schema":
        status["schema"] = "wrong"
    elif change == "row_mismatch":
        status["rows"][0]["row_id"] = "other"
    elif change == "unknown_state":
        status["rows"][0]["status"] = "mystery"
    else:
        status["mutated"] = True
    status_path.write_text(json.dumps(status))
    result = run_once(dashboard, manifest)
    assert result.returncode == 2
    assert "invalid JSON, unknown state, or rows inconsistent" in result.stderr


def test_dashboard_rejects_wrong_manifest_schema(tmp_path):
    dashboard, manifest, _, _ = dashboard_case(tmp_path)
    content = json.loads(manifest.read_text())
    content["schema"] = "topeft_datacard_matrix_v2"
    manifest.write_text(json.dumps(content))
    result = run_once(dashboard, manifest)
    assert result.returncode == 2
    assert "not readable v3 JSON" in result.stderr


def test_dashboard_rejects_invalid_status_json(tmp_path):
    dashboard, manifest, status_path, _ = dashboard_case(tmp_path)
    status_path.write_text("not JSON")
    result = run_once(dashboard, manifest)
    assert result.returncode == 2
    assert "invalid JSON" in result.stderr


def test_dashboard_reports_missing_runner(tmp_path):
    dashboard, manifest, _, _ = dashboard_case(tmp_path)
    (dashboard.parent.parent / "run_datacard_matrix_resumable.sh").unlink()
    result = run_once(dashboard, manifest)
    assert result.returncode == 2
    assert "runner is missing or not executable" in result.stderr


def test_dashboard_renders_multiple_active_units_and_logs(tmp_path):
    dashboard, manifest, status_path, control_root = dashboard_case(tmp_path)
    status = json.loads(status_path.read_text())
    status["rows"][2]["status"] = "active"
    for row in status["rows"][1:]:
        row["owner_metadata"] = {"backend": "condor", "scheduler_id": row["row_id"], "current_row_started_at": "synthetic"}
        (control_root / (row["row_id"] + ".log")).write_text(row["row_id"] + " log\n")
    status_path.write_text(json.dumps(status))
    before = {path: path.read_bytes() for path in control_root.iterdir()}
    result = run_once(dashboard, manifest)
    assert result.returncode == 0, result.stderr
    assert "Active units   : 2 (5 targets)" in result.stdout
    assert "unit_2 log" in result.stdout and "unit_3 log" in result.stdout
    assert result.stdout.count("Backend: condor") == 2
    assert before == {path: path.read_bytes() for path in control_root.iterdir()}


def test_dashboard_displays_observations_without_changing_completion(tmp_path):
    dashboard, manifest, status_path, _ = dashboard_case(tmp_path)
    status = json.loads(status_path.read_text())
    status["rows"][0]["scheduler_metadata"] = {"scheduler_id": "1.0", "state": "removed", "observed_at": "synthetic", "completion_authority": False}
    status_path.write_text(json.dumps(status))
    result = run_once(dashboard, manifest)
    assert result.returncode == 0, result.stderr
    assert "Execution units: 1 / 3" in result.stdout
    assert "removed at synthetic (operational only)" in result.stdout


@pytest.mark.parametrize("state,label", [("HELD", "HELD terminal"), ("UNKNOWN", "BLOCKED"),
                                         ("GONE", "automatically reconciles H2")])
def test_dashboard_prominently_surfaces_reservation_anomalies_even_with_receipt(tmp_path, state, label):
    dashboard, manifest, status_path, _ = dashboard_case(tmp_path)
    status = json.loads(status_path.read_text())
    status["rows"][0]["reservation"] = {"backend": "condor", "scheduler_id": "123.7", "state": state}
    status_path.write_text(json.dumps(status))
    before = status_path.read_bytes()
    result = run_once(dashboard, manifest)
    assert result.returncode == 0, result.stderr
    assert f"[{state}] Logical reservation: unit_1 Backend: condor Scheduler: 123.7" in result.stdout
    assert label in result.stdout
    assert "Execution units: 1 / 3" in result.stdout
    assert status_path.read_bytes() == before


def test_dashboard_displays_exact_scheduler_error_and_hold_evidence(tmp_path):
    dashboard, manifest, status_path, _ = dashboard_case(tmp_path)
    status = json.loads(status_path.read_text())
    status["rows"][0]["reservation"] = {
        "state": "UNKNOWN", "backend": "condor", "error_code": "SCHEDULER_OBSERVABILITY_UNAVAILABLE",
        "reason": "query=['condor_q']; observed_at=2026-10-07; error=authentication failed",
        "hold_evidence": {"HoldReason": "synthetic disk failure", "HoldReasonCode": 34, "HoldReasonSubCode": 2},
    }
    status_path.write_text(json.dumps(status))
    result = run_once(dashboard, manifest)
    assert result.returncode == 0, result.stderr
    assert "SCHEDULER_OBSERVABILITY_UNAVAILABLE" in result.stdout
    assert "observed_at=2026-10-07" in result.stdout and "authentication failed" in result.stdout
    assert "synthetic disk failure" in result.stdout and "HoldReasonSubCode" in result.stdout
