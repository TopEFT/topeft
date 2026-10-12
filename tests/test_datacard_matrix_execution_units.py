"""Synthetic execution-unit tests; no real producer or scheduler is invoked."""
import json
import os
from unittest.mock import patch
from pathlib import Path
import subprocess
import sys
import time

import pytest

from analysis.topeft_run2 import datacard_matrix_runner as runner
from test_datacard_matrix_resumable_runner import (
    fake_script, make_row, manifest, sha, wait_for_owner, ENGINE, FAKE,
)


def test_descriptors_preserve_parent_identity_and_detect_drift(tmp_path, fake_script):
    rows = [make_row(tmp_path, fake_script, name) for name in ("a", "b")]
    path, original = manifest(tmp_path, fake_script, rows)
    units = runner.derive_units(path)
    assert units == runner.derive_units(path)
    assert [unit["row_id"] for unit in units] == ["a", "b"]
    parent, digest, row = runner.resolve_unit(units[0])
    assert parent == original and digest == sha(path) and row == rows[0]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(runner.RunnerError, match="bytes changed"):
        runner.resolve_unit(units[0])


def test_unit_executes_only_selected_row_and_switches_backend(tmp_path, fake_script):
    rows = [make_row(tmp_path, fake_script, name) for name in ("a", "b")]
    path, data = manifest(tmp_path, fake_script, rows)
    parent, digest, row = runner.resolve_unit(runner.derive_units(path)[0])
    assert runner.execute_unit(parent, digest, row)["action"] == "executed"
    assert not Path(rows[1]["output_root"]).exists()
    before = runner.receipt_path(data, row).read_bytes()
    assert runner.execute_unit(parent, digest, row, backend="manual")["action"] == "skipped_valid_receipt"
    assert runner.receipt_path(data, row).read_bytes() == before


def test_independent_leases_coexist_and_alias_manifest_blocks(tmp_path, fake_script):
    rows = [make_row(tmp_path, fake_script, name) for name in ("a", "b")]
    _, data = manifest(tmp_path, fake_script, rows)
    with runner.unit_lease(data, rows[0]), runner.unit_lease(data, rows[1]):
        with pytest.raises(runner.RunnerError) as error:
            runner.execute_unit(data, "other_manifest", {**rows[0], "row_id": "alias"}, backend="condor")
        assert error.value.code == "execution_lock_held"


def test_two_active_units_status_and_parent_kill_exclusion(tmp_path, fake_script):
    rows = [make_row(tmp_path, fake_script, name, extra=["--wait-for", str(tmp_path / "release"), "--counter", str(tmp_path / (name + "_started"))]) for name in ("a", "b")]
    path, data = manifest(tmp_path, fake_script, rows)
    processes = [subprocess.Popen([sys.executable, str(ENGINE), str(path), "--row-id", row["row_id"], "--backend", "local"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for row in rows]
    try:
        for row in rows:
            wait_for_owner(runner.unit_owner_path(row), row["row_id"])
        deadline = time.monotonic() + 10
        while not all((tmp_path / (name + "_started")).exists() for name in ("a", "b")) and time.monotonic() < deadline:
            time.sleep(.01)
        assert all((tmp_path / (name + "_started")).exists() for name in ("a", "b"))
        status = runner.read_only("status", data, sha(path))
        assert [state["status"] for state in status["rows"]] == ["active", "active"]
        assert all(state["owner_metadata"]["backend"] == "local" for state in status["rows"])
        processes[0].kill()
        processes[0].wait()
        with pytest.raises(runner.RunnerError) as error:
            runner.execute_unit(data, sha(path), rows[0], backend="local")
        assert error.value.code == "execution_lock_held"
        assert not runner.receipt_path(data, rows[0]).exists()
        (tmp_path / "release").touch()
        processes[1].communicate(timeout=10)
        assert processes[1].returncode == 0
        deadline = time.monotonic() + 10
        while runner.probe_lock(runner.unit_lock_path(rows[0]))[0] and time.monotonic() < deadline:
            time.sleep(.02)
        assert runner.classify(data, sha(path), rows[0])["status"] == "partial_staging_attempt"
        assert not runner.receipt_path(data, rows[0]).exists()
    finally:
        (tmp_path / "release").touch()
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)


def test_whole_workload_composes_unit_executor(tmp_path, fake_script, monkeypatch):
    rows = [make_row(tmp_path, fake_script, name) for name in ("a", "b")]
    path, data = manifest(tmp_path, fake_script, rows)
    seen = []
    def execute_unit(parent, digest, row, **metadata):
        assert parent is data and digest == sha(path)
        seen.append(row["row_id"])
        return {"row_id": row["row_id"]}
    monkeypatch.setattr(runner, "execute_unit", execute_unit)
    assert runner.execute(data, sha(path))["rows"] == [{"row_id": "a"}, {"row_id": "b"}]
    assert seen == ["a", "b"]


def test_descriptor_cli_and_worker_integrity(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    unit_path = tmp_path / "unit.json"
    unit_path.write_text(json.dumps(runner.derive_units(path)[0]))
    command = [sys.executable, str(ENGINE), "--unit", str(unit_path), "--runner-integrity"]
    bad = subprocess.run([*command, "0" * 64], capture_output=True, text=True)
    assert bad.returncode == 2 and "stale_dispatch_input" in bad.stderr
    assert not Path(row["output_root"]).exists()
    good = subprocess.run([*command, sha(ENGINE)], capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    assert runner.validate_receipt(json.loads(runner.receipt_path(data, row).read_text()), data, sha(path), row)[0]


def test_nested_cross_manifest_domains_fail_closed(tmp_path, fake_script):
    outer = make_row(tmp_path, fake_script, "outer")
    path, data = manifest(tmp_path, fake_script, [outer])
    inner = {**outer, "output_root": str(Path(outer["output_root"]) / "inner")}
    with runner.unit_lease(data, outer):
        with pytest.raises(runner.RunnerError, match="nested"):
            with runner.unit_lease(data, inner):
                pytest.fail("nested domain acquired")
        assert not Path(outer["output_root"]).exists()
    assert not runner.receipt_path(data, outer).exists()


def test_legacy_manifest_layout_still_loads(tmp_path, fake_script):
    rows = [make_row(tmp_path, fake_script, name) for name in ("a", "b")]
    # Historically valid distinct targets can share an output root.
    rows[1]["output_root"] = rows[0]["output_root"]
    args = rows[1]["producer_args"]
    args[args.index("--out-dir") + 1] = rows[0]["output_root"]
    rows[1]["expected_output_paths"] = [str(Path(rows[0]["output_root"]) / "other.txt")]
    path, data = manifest(tmp_path, fake_script, rows)
    assert runner.load_manifest(path)[0] == data
    with pytest.raises(runner.RunnerError, match="disjoint"):
        runner.derive_units(path)


@pytest.fixture
def scheduler(monkeypatch):
    """Only native queue observations are synthetic; H2 runs its real engine."""
    observations = {}
    def query(reservation):
        return observations.get(reservation["owner_id"], {
            "state": "GONE",
            "scheduler_id": reservation["scheduler_id"]})
    monkeypatch.setattr(runner, "query_reservation", query)
    return observations


def reserve(data, path, row):
    with runner.reserve_condor(data, sha(path), [row], {
            "pool": "synthetic.pool", "schedd": "synthetic.schedd", "query_executable": "/bin/false"}) as transaction:
        transaction.mark_submission_started()
        return transaction.reservations[0]


def worker(data, path, row, reservation, scheduler_id="123.7"):
    job_ad = path.parent / "worker.ad"
    cluster, proc = map(int, scheduler_id.split("."))
    job_ad.write_text(f'DatacardUnit = {json.dumps(row["row_id"])}\nDatacardReservation = {json.dumps(reservation["owner_id"])}\nClusterId = {cluster}\nProcId = {proc}\n')
    with patch.dict(os.environ, {"_CONDOR_JOB_AD": str(job_ad)}):
        return runner.execute_unit(data, sha(path), row, backend="condor", scheduler_id="999.999",
                                   reservation_token=reservation["owner_id"])


@pytest.mark.parametrize("state", list(runner.SCHEDULER_STATES.values()))
def test_all_live_queue_states_preserve_logical_owner_without_physical_lease(tmp_path, fake_script, scheduler, state):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    reservation = reserve(data, path, row)
    scheduler[reservation["owner_id"]] = {"state": state, "scheduler_id": "123.7"}
    before = runner.reservation_path(row).read_bytes()
    status = runner.read_only("status", data, sha(path))["rows"][0]
    assert status["reservation"]["state"] == state
    assert not status["lock_held"]
    assert runner.reservation_path(row).read_bytes() == before
    for backend in ("local", "manual"):
        with pytest.raises(runner.RunnerError) as error:
            runner.execute_unit(data, sha(path), row, backend=backend)
        assert error.value.code == "row_reserved"
    with pytest.raises(runner.RunnerError, match="logical owner"):
        reserve(data, path, row)
    assert runner.read_reservation(row)["scheduler_id"] == "123.7"
    assert not Path(row["output_root"]).exists()
    assert runner.read_only("plan_only", data, sha(path))["rows"][0]["action"] == "block"


def test_ambiguous_query_fail_closed_but_worker_never_queries(tmp_path, fake_script, scheduler, monkeypatch):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    reservation = reserve(data, path, row)
    def unavailable(_):
        raise runner.RunnerError("SCHEDULER_OBSERVABILITY_UNAVAILABLE", "synthetic query timeout")
    monkeypatch.setattr(runner, "query_reservation", unavailable)
    with pytest.raises(runner.RunnerError, match="timeout"):
        runner.execute_unit(data, sha(path), row)
    status = runner.read_only("status", data, sha(path))["rows"][0]["reservation"]
    assert status["state"] == "UNKNOWN"
    assert status["error_code"] == "SCHEDULER_OBSERVABILITY_UNAVAILABLE"
    assert runner.read_reservation(row)["owner_id"] == reservation["owner_id"]
    assert worker(data, path, row, reservation)["action"] == "executed"
    assert runner.read_reservation(row)["scheduler_id"] == "123.7"


def test_eviction_keeps_owner_then_same_job_restarts_and_receipt_completes(tmp_path, fake_script, scheduler):
    row = make_row(tmp_path, fake_script, extra=["--partial-once", str(tmp_path / "once")])
    other = make_row(tmp_path, fake_script, "other")
    path, data = manifest(tmp_path, fake_script, [row, other])
    reservation = reserve(data, path, row)
    scheduler[reservation["owner_id"]] = {"state": "RUNNING", "scheduler_id": "123.7"}
    with pytest.raises(runner.RunnerError) as error:
        worker(data, path, row, reservation)
    assert error.value.code == "row_command_failed"
    assert not runner.probe_lock(runner.unit_lock_path(row))[0]
    scheduler[reservation["owner_id"]]["state"] = "IDLE"
    with pytest.raises(runner.RunnerError):
        runner.execute(data, sha(path))
    assert runner.execute_unit(data, sha(path), other)["action"] == "executed"
    assert runner.read_reservation(row)["owner_id"] == reservation["owner_id"]
    scheduler[reservation["owner_id"]]["state"] = "RUNNING"
    assert worker(data, path, row, reservation)["physical_attempt_id"] == "attempt_000002"
    receipt = runner.receipt_path(data, row).read_bytes()
    assert worker(data, path, row, reservation)["action"] == "skipped_valid_receipt"
    scheduler.pop(reservation["owner_id"])
    assert runner.execute_unit(data, sha(path), row, backend="manual")["action"] == "skipped_valid_receipt"
    assert runner.read_reservation(row) is None
    assert runner.receipt_path(data, row).read_bytes() == receipt


def test_disappearance_reconciles_partial_h2_before_new_backend_claim(tmp_path, fake_script, scheduler):
    row = make_row(tmp_path, fake_script, extra=["--partial-once", str(tmp_path / "once")])
    path, data = manifest(tmp_path, fake_script, [row])
    reservation = reserve(data, path, row)
    scheduler[reservation["owner_id"]] = {"state": "RUNNING", "scheduler_id": "123.7"}
    with pytest.raises(runner.RunnerError):
        worker(data, path, row, reservation)
    scheduler.pop(reservation["owner_id"])
    assert runner.execute_unit(data, sha(path), row)["physical_attempt_id"] == "attempt_000002"
    assert runner.read_reservation(row) is None
    assert runner.validate_receipt(json.loads(runner.receipt_path(data, row).read_text()), data, sha(path), row)[0]


def test_sequential_scheduler_ids_are_operational_not_row_or_attempt_identity(tmp_path, fake_script, scheduler):
    row = make_row(tmp_path, fake_script, extra=["--partial-once", str(tmp_path / "once")])
    path, data = manifest(tmp_path, fake_script, [row])
    descriptor = runner.derive_units(path)[0]
    first = reserve(data, path, row)
    scheduler[first["owner_id"]] = {"state": "RUNNING", "scheduler_id": "123.7"}
    with pytest.raises(runner.RunnerError):
        worker(data, path, row, first)
    scheduler.pop(first["owner_id"])
    second = reserve(data, path, row)
    assert first["owner_id"] != second["owner_id"]
    scheduler[second["owner_id"]] = {"state": "RUNNING", "scheduler_id": "456.2"}
    assert worker(data, path, row, second, "456.2")["physical_attempt_id"] == "attempt_000002"
    assert runner.derive_units(path)[0] == descriptor


def test_disappearance_reconciles_valid_staging_without_new_producer(tmp_path, fake_script, scheduler, monkeypatch):
    counter = tmp_path / "counter"
    row = make_row(tmp_path, fake_script, extra=["--counter", str(counter)])
    path, data = manifest(tmp_path, fake_script, [row])
    reservation = reserve(data, path, row)
    scheduler[reservation["owner_id"]] = {"state": "RUNNING", "scheduler_id": "123.7"}
    with monkeypatch.context() as patch:
        def crash(*_):
            raise RuntimeError("before publication")
        patch.setattr(runner, "reconcile_row", crash)
        with pytest.raises(RuntimeError):
            worker(data, path, row, reservation)
    scheduler.pop(reservation["owner_id"])
    with runner.unit_lease(data, row):
        assert runner.reconcile_reservation(data, sha(path), row) is None
    assert runner.classify(data, sha(path), row)["status"] == "complete"
    assert counter.read_text() == "1"


def test_disappearance_with_ambiguous_h2_or_live_physical_lease_keeps_reservation(tmp_path, fake_script, scheduler):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    reservation = reserve(data, path, row)
    scheduler[reservation["owner_id"]] = {"state": "IDLE", "scheduler_id": "123.7"}
    with runner.unit_lease(data, row):
        runner.reconcile_reservation(data, sha(path), row)
    scheduler.pop(reservation["owner_id"])
    with runner.unit_lease(data, row):
        with pytest.raises(runner.RunnerError) as error:
            runner.execute_unit(data, sha(path), row)
        assert error.value.code == "execution_lock_held"
    Path(row["output_root"]).mkdir()
    with pytest.raises(runner.RunnerError) as error:
        runner.execute_unit(data, sha(path), row)
    assert error.value.code == "canonical_collision_or_ambiguous_state"
    assert runner.read_reservation(row)["owner_id"] == reservation["owner_id"]


@pytest.mark.parametrize("contents", ["{broken", "[]", '{}'])
def test_corrupt_reservation_never_grants_execution(tmp_path, fake_script, contents):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    reservation = runner.reservation_path(row)
    reservation.parent.mkdir(parents=True)
    reservation.write_text(contents)
    with pytest.raises(runner.RunnerError) as error:
        runner.execute_unit(data, sha(path), row)
    assert error.value.code == "reservation_ambiguous"
    assert reservation.read_text() == contents


def test_token_query_accepts_changed_locator_and_reports_failures(tmp_path, fake_script, monkeypatch):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    reservation = reserve(data, path, row)
    reservation["scheduler_id"] = "123.7"
    ad = {"ClusterId": 456, "ProcId": 2, "JobStatus": 1, "DatacardUnit": row["row_id"],
          "DatacardReservation": reservation["owner_id"]}
    monkeypatch.setattr(runner, "scheduler_snapshot", lambda scheduler: {"observed_at": "synthetic"})
    def install(ads, returncode=0):
        def query(argv, **kwargs):
            assert argv[argv.index("-constraint") + 1] == f'DatacardReservation == "{reservation["owner_id"]}"'
            assert kwargs["timeout"] == 30
            if argv[0] != "/bin/false":
                return subprocess.CompletedProcess(argv, 0, "[]", "")
            if "-name" in argv:
                assert argv[:5] == ["/bin/false", "-pool", "synthetic.pool", "-name", "synthetic.schedd"]
                return subprocess.CompletedProcess(argv, returncode, json.dumps(ads), "specific synthetic failure")
            return subprocess.CompletedProcess(argv, 0, "[]", "")
        monkeypatch.setattr(runner.subprocess, "run", query)
    install([ad])
    observation = runner.query_reservation(reservation)
    assert observation["state"] == "IDLE" and observation["ownership"] == "ACTIVE"
    assert observation["scheduler_id"] == "456.2"
    install([ad, {**ad, "ClusterId": 789, "ProcId": 8}])
    with pytest.raises(runner.RunnerError) as error:
        runner.query_reservation(reservation)
    assert error.value.code == "DUPLICATE_SCHEDULER_OWNERSHIP"
    assert "456.2" in str(error.value) and "789.8" in str(error.value)
    install([ad, ad])
    with pytest.raises(runner.RunnerError) as error:
        runner.query_reservation(reservation)
    assert error.value.code == "DUPLICATE_SCHEDULER_OWNERSHIP"
    before = runner.reservation_path(row).read_bytes()
    with pytest.raises(runner.RunnerError) as error:
        runner.execute_unit(data, sha(path), row)
    assert error.value.code == "DUPLICATE_SCHEDULER_OWNERSHIP"
    assert runner.reservation_path(row).read_bytes() == before
    for ads in ([{**ad, "DatacardUnit": "other"}], [{**ad, "DatacardReservation": "other"}],
                [{**ad, "JobStatus": 99}], [{**ad, "ClusterId": True}], [None]):
        install(ads)
        with pytest.raises(runner.RunnerError) as error:
            runner.query_reservation(reservation)
        assert error.value.code == "scheduler_observation_invalid"
    install([], 1)
    before = runner.reservation_path(row).read_bytes()
    with pytest.raises(runner.RunnerError) as error:
        runner.execute_unit(data, sha(path), row)
    assert error.value.code == "SCHEDULER_OBSERVABILITY_INCOMPLETE"
    assert all(text in str(error.value) for text in ("specific synthetic failure", "observed_at=", "-constraint", "exit=1"))
    assert runner.reservation_path(row).read_bytes() == before
    install([])
    for cached_locator in (None, "123.7", "obsolete locator"):
        reservation["scheduler_id"] = cached_locator
        with pytest.raises(runner.RunnerError, match="no terminal proof"):
            runner.query_reservation(reservation)


@pytest.mark.parametrize("failure", [OSError("missing condor_q"), subprocess.TimeoutExpired(["condor_q"], 30), "{bad", "{}"])
def test_unobservable_scheduler_preserves_reservation(tmp_path, fake_script, monkeypatch, failure):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    reserve(data, path, row)
    before = runner.reservation_path(row).read_bytes()
    def query(argv, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return subprocess.CompletedProcess(argv, 0, failure, "")
    monkeypatch.setattr(runner.subprocess, "run", query)
    with pytest.raises(runner.RunnerError) as error:
        runner.execute_unit(data, sha(path), row)
    assert error.value.code == "SCHEDULER_OBSERVABILITY_INCOMPLETE"
    assert "query=" in str(error.value) and "observed_at=" in str(error.value)
    assert runner.reservation_path(row).read_bytes() == before


@pytest.mark.parametrize("evidence", ["fallback_live", "no_proof", "history", "event_log", "covered_zero", "contradiction"])
def test_scheduler_hint_failure_uses_independent_token_evidence(tmp_path, fake_script, monkeypatch, evidence):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    reservation = reserve(data, path, row)
    reservation["scheduler_observation"] = {
        "scheduler_id": "123.7", "schedd": "synthetic.schedd", "pool": "synthetic.pool",
        "source": "hint", "observed_at": "synthetic", "global_job_id": None}
    reservation["scheduler_id"] = "123.7"
    reservation["scheduler"]["dispatch_directory"] = str(tmp_path / "dispatch")
    runner.atomic_json(reservation, runner.reservation_path(row), True)
    base_ad = {"DatacardUnit": row["row_id"], "DatacardReservation": reservation["owner_id"], "JobStatus": 2}
    live_ad = {**base_ad, "ClusterId": 456, "ProcId": 2,
               "ScheddName": "new.schedd", "GlobalJobId": "new.schedd#456.2#1"}
    history_ad = {**base_ad, "ClusterId": 123, "ProcId": 7,
                  "JobStatus": 4, "GlobalJobId": "synthetic.schedd#123.7#1"}
    monkeypatch.setattr(runner, "scheduler_snapshot", lambda scheduler: {
        "collector": "different.pool" if evidence == "no_proof" else "synthetic.pool",
        "default_schedd": "synthetic.schedd"})
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/bin/condor_history")
    if evidence == "event_log":
        logs = tmp_path / "dispatch" / "logs"
        logs.mkdir(parents=True)
        (logs / f'{row["row_id"]}__123.7.events.log').write_text("009 (123.7.0) 10/08 00:00:00 Job was aborted.\n")

    def query(argv):
        if argv[0] == "/bin/condor_history":
            return ([history_ad] if evidence in {"history", "contradiction"} else []), None
        if "-name" in argv:
            if evidence == "covered_zero":
                return [], None
            return None, "hinted endpoint unreachable"
        if evidence in {"fallback_live", "contradiction"}:
            return [live_ad], None
        return [], None

    monkeypatch.setattr(runner, "_query_ads", query)
    before = runner.reservation_path(row).read_bytes()
    if evidence == "no_proof":
        with pytest.raises(runner.RunnerError) as error:
            runner.reconcile_reservation(data, sha(path), row)
        assert error.value.code == "SCHEDULER_OBSERVABILITY_INCOMPLETE"
    elif evidence == "contradiction":
        with pytest.raises(runner.RunnerError) as error:
            runner.reconcile_reservation(data, sha(path), row)
        assert error.value.code == "SCHEDULER_EVIDENCE_CONTRADICTION"
    elif evidence == "fallback_live":
        observation = runner.reconcile_reservation(data, sha(path), row)
        assert observation["ownership"] == "ACTIVE"
        assert observation["scheduler_id"] == "456.2"
        assert runner.read_reservation(row)["scheduler_observation"]["schedd"] == "new.schedd"
    else:
        observation = runner.query_reservation(reservation)
        assert observation["ownership"] == "GONE"
        assert observation["terminal_proof"]["source"] == {
            "history": "history", "event_log": "event_log_remove", "covered_zero": "covered_zero"}[evidence]
        assert runner.reconcile_reservation(data, sha(path), row) is None
        assert runner.read_reservation(row) is None
    if evidence in {"no_proof", "contradiction"}:
        assert runner.reservation_path(row).read_bytes() == before


@pytest.mark.parametrize("token", [None, "", "abc", "A" * 32, '" || true || "', "0" * 33, "0" * 32 + "\n"])
def test_unsafe_token_rejected_before_query(tmp_path, fake_script, monkeypatch, token):
    row = make_row(tmp_path, fake_script)
    reservation = runner.new_reservation(row, "condor", {})
    reservation["owner_id"] = token
    def forbidden(*a, **k):
        pytest.fail("unsafe token reached scheduler")
    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    with pytest.raises(runner.RunnerError) as error:
        runner.query_reservation(reservation)
    assert error.value.code == "invalid_reservation_token"


@pytest.mark.parametrize("path_kind", ["unit", "full", "condor", "maintenance"])
def test_never_observed_job_disappearance_automatically_reconciles(tmp_path, fake_script, scheduler, monkeypatch, path_kind):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    old = reserve(data, path, row)
    assert old["scheduler_id"] is None
    calls = []
    original = runner.reconcile_h2
    def observe(*args):
        calls.append(args[2]["row_id"])
        return original(*args)
    monkeypatch.setattr(runner, "reconcile_h2", observe)
    if path_kind == "unit":
        assert runner.execute_unit(data, sha(path), row)["action"] == "executed"
    elif path_kind == "full":
        assert runner.execute(data, sha(path))["rows"][0]["action"] == "executed"
    elif path_kind == "condor":
        assert reserve(data, path, row)["owner_id"] != old["owner_id"]
    else:
        assert runner.main([str(path), "--reconcile-reservations"]) == 0
        assert runner.read_reservation(row) is None
    assert row["row_id"] in calls



def test_missing_scheduler_id_is_ambiguous_reservation_not_status_crash(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    reservation = reserve(data, path, row)
    del reservation["scheduler_id"]
    runner.reservation_path(row).write_text(json.dumps(reservation))
    assert runner.read_only("status", data, sha(path))["rows"][0]["reservation"]["state"] == "UNKNOWN"
    with pytest.raises(runner.RunnerError) as error:
        runner.execute_unit(data, sha(path), row)
    assert error.value.code == "reservation_ambiguous"


def test_local_stale_reservation_reconciles_only_under_free_inherited_lease(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    with runner.unit_lease(data, row):
        reservation = runner.new_reservation(row, "local")
        runner.atomic_json(reservation, runner.reservation_path(row), False)
        with pytest.raises(runner.RunnerError) as error:
            runner.execute_unit(data, sha(path), row)
        assert error.value.code == "execution_lock_held"
    assert runner.execute_unit(data, sha(path), row)["action"] == "executed"
    assert runner.read_reservation(row) is None


def test_h2_private_producer_domain_supports_direct_guard(tmp_path, fake_script):
    source = (
        "import sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(ENGINE.parents[2])!r})\n"
        "from analysis.topeft_run2.datacard_matrix_runner import direct_output_lease\n"
        "with direct_output_lease(Path(sys.argv[sys.argv.index('--out-dir') + 1])):\n"
        f"    exec({FAKE!r})\n"
    )
    fake_script.write_text(source)
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    assert runner.execute_unit(data, sha(path), row)["action"] == "executed"
    assert runner.validate_receipt(json.loads(runner.receipt_path(data, row).read_text()), data, sha(path), row)[0]


def test_live_hold_reason_evidence_and_changed_locator_cache(tmp_path, fake_script, monkeypatch):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    reservation = reserve(data, path, row)
    reservation["scheduler_id"] = "123.7"
    runner.reservation_path(row).write_text(json.dumps(reservation))
    hold = {"HoldReason": "synthetic site hold", "HoldReasonCode": 1, "HoldReasonSubCode": 17}
    ad = {"ClusterId": 456, "ProcId": 2, "JobStatus": 5, "DatacardUnit": row["row_id"],
          "DatacardReservation": reservation["owner_id"], **hold}
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps([ad]), ""))
    before = runner.reservation_path(row).read_bytes()
    status = runner.read_only("status", data, sha(path))["rows"][0]["reservation"]
    assert status["state"] == "HELD" and status["ownership"] == "ACTIVE"
    assert status["hold_evidence"] == hold
    assert runner.reservation_path(row).read_bytes() == before
    with pytest.raises(runner.RunnerError) as error:
        runner.execute_unit(data, sha(path), row)
    assert error.value.code == "row_reserved"
    assert runner.read_reservation(row)["scheduler_id"] == "456.2"
    # Removal does not itself mutate the reservation; a later claim invokes H2.
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "[]", ""))
    with runner.unit_lease(data, row):
        assert runner.reconcile_reservation(data, sha(path), row) is None
    assert runner.classify(data, sha(path), row)["status"] == "not_started"


def test_transaction_holds_leases_before_submit_so_zero_query_cannot_reclaim(tmp_path, fake_script, monkeypatch):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    scheduler = {"pool": "synthetic.pool", "schedd": "synthetic.schedd", "query_executable": "/bin/false"}
    def forbidden(*args):
        pytest.fail("pending transaction reached scheduler observation")
    with runner.reserve_condor(data, sha(path), [row], scheduler) as transaction:
        monkeypatch.setattr(runner, "query_reservation", forbidden)
        before = runner.reservation_path(row).read_bytes()
        with pytest.raises(runner.RunnerError) as error:
            runner.execute_unit(data, sha(path), row)
        assert error.value.code == "execution_lock_held"
        assert runner.reservation_path(row).read_bytes() == before
        assert transaction.lease_fds
    assert runner.read_reservation(row) is None


def test_reservation_publication_failure_rolls_back_exact_owned_token(tmp_path, fake_script, monkeypatch):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    original = runner.atomic_json
    def fail_after_publication(payload, destination, overwrite):
        original(payload, destination, overwrite)
        if destination == runner.reservation_path(row):
            raise OSError("synthetic failure after reservation publication")
    monkeypatch.setattr(runner, "atomic_json", fail_after_publication)
    with pytest.raises(OSError, match="after reservation publication"):
        with runner.reserve_condor(data, sha(path), [row], {
                "pool": "synthetic.pool", "schedd": "synthetic.schedd", "query_executable": "/bin/false"}):
            pytest.fail("failed reservation transaction yielded")
    assert runner.read_reservation(row) is None


def test_proven_spawn_failure_rolls_back_even_after_possible_start_marker(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    with runner.reserve_condor(data, sha(path), [row], {
            "pool": "synthetic.pool", "schedd": "synthetic.schedd", "query_executable": "/bin/false"}) as transaction:
        transaction.mark_submission_started()
        try:
            subprocess.Popen([str(tmp_path / "nonexistent_synthetic_submitter")], pass_fds=transaction.lease_fds)
        except OSError as error:
            transaction.record_definite_spawn_failure(error)
        else:
            pytest.fail("nonexistent executable started")
    assert runner.read_reservation(row) is None


def test_timeout_is_not_a_definite_spawn_failure(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    with pytest.raises(runner.RunnerError) as error:
        with runner.reserve_condor(data, sha(path), [row], {
                "pool": "synthetic.pool", "schedd": "synthetic.schedd", "query_executable": "/bin/false"}) as transaction:
            transaction.mark_submission_started()
            transaction.record_definite_spawn_failure(subprocess.TimeoutExpired("synthetic_submit", 1))
    assert error.value.code == "submission_outcome_error"
    assert runner.read_reservation(row) is not None
