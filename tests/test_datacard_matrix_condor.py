"""Condor frontend tests use synthetic producers and never submit real jobs."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from analysis.topeft_run2 import datacard_matrix_condor as condor
from analysis.topeft_run2 import datacard_matrix_runner as runner
from analysis.topeft_run2 import make_cards
from test_datacard_matrix_resumable_runner import fake_script, make_row, manifest, sha


resources = dict(request_cpus=1, request_memory="1GB", request_disk="1GB",
                 max_retries=2, max_starts=3)


@pytest.fixture
def fake_query(tmp_path):
    ads = tmp_path / "live_ads.json"
    ads.write_text("[]")
    query = tmp_path / "condor_q"
    query.write_text(f'#!/bin/sh\nexec /bin/cat "{ads}"\n')
    query.chmod(0o755)
    return query, ads


def running_ad(row_id, token, cluster=123, proc=7):
    return {"DatacardUnit": row_id, "DatacardReservation": token,
            "ClusterId": cluster, "ProcId": proc, "JobStatus": 2}


def write_worker_ad(path, row_id, token, **overrides):
    fields = {**running_ad(row_id, token), **overrides}
    path.write_text("".join(f"{key} = {json.dumps(value)}\n" for key, value in fields.items()))
    return {"PATH": "/usr/bin:/bin", "LANG": "C", "_CONDOR_JOB_AD": str(path)}


def test_material_identity_worker_command_and_no_submit(tmp_path, fake_script, fake_query, monkeypatch):
    rows = [make_row(tmp_path, fake_script, name) for name in ("a", "b")]
    path, data = manifest(tmp_path, fake_script, rows)
    destination = tmp_path / "dispatch"
    def forbidden(*args, **kwargs):
        pytest.fail("material preparation invoked a subprocess")
    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "run", forbidden)
        result = condor.prepare_material(path, destination, **resources, max_runtime=21600,
                                        query_executable=str(fake_query[0]))
    assert result["submitted"] is False
    units = [json.loads((destination / "units" / (row["row_id"] + ".json")).read_text()) for row in rows]
    assert units == runner.derive_units(path)
    items = (destination / "items.txt").read_text().splitlines()
    assert [item.split()[0] for item in items] == ["a", "b"]
    submit = (destination / "datacards.sub").read_text()
    assert "queue row_id, unit_integrity, reservation_token from " in submit
    assert 'allowed_execute_duration = 21600' in submit and "should_transfer_files = NO" in submit
    assert "getenv = false" in submit and "+MaxRuntime" not in submit
    assert "transfer_output_files" not in submit and "on_exit_hold" not in submit
    assert "condor_submit" not in (destination / "worker.sh").read_text()
    assert not any(Path(row["output_root"]).exists() for row in rows)
    assert all(item.split()[2] == "unreserved" for item in items)
    assert all(runner.read_reservation(row) is None for row in rows)
    with condor.submission_transaction(path, destination, result["scheduler"]) as transaction:
        assert len(transaction.reservations) == 2
        assert transaction.lease_fds
        assert all(runner.probe_lock(runner.unit_lock_path(row))[0] for row in rows)
        transaction.mark_submission_started()
    row_id, integrity, token = (destination / "items.txt").read_text().splitlines()[0].split()
    reservation = runner.read_reservation(rows[0])
    assert reservation["owner_id"] == token and reservation["scheduler_id"] is None
    reservation["scheduler_id"] = "42.0"
    runner.atomic_json(reservation, runner.reservation_path(rows[0]), True)
    # Execute nodes have no scheduler control-plane capability.
    fake_query[0].write_text('#!/bin/sh\necho forbidden_worker_query >&2\nexit 93\n')
    worker_env = write_worker_ad(tmp_path / "worker.ad", row_id, token)
    worker = subprocess.run([str(destination / "worker.sh"), row_id, integrity, "999.9", token],
                            env=worker_env, text=True, capture_output=True)
    assert worker.returncode == 0, worker.stderr
    assert runner.validate_receipt(json.loads(runner.receipt_path(data, rows[0]).read_text()), data, sha(path), rows[0])[0]
    assert not Path(rows[1]["output_root"]).exists()
    assert "forbidden_worker_query" not in worker.stderr
    assert runner.read_reservation(rows[0])["owner_id"] == token
    assert runner.read_reservation(rows[0])["scheduler_id"] == "123.7"
    with pytest.raises(runner.RunnerError, match="exists"):
        condor.prepare_material(path, destination, **resources)


@pytest.mark.parametrize("field,value", [("request_cpus", None), ("request_cpus", 0), ("request_memory", None), ("request_memory", "1GB\nqueue 9"), ("request_disk", ""), ("max_runtime", 0), ("max_runtime", True), ("max_retries", None), ("max_retries", -1), ("max_retries", True), ("max_starts", None), ("max_starts", 0), ("max_starts", True), ("query_executable", "relative_condor_q")])
def test_resources_are_explicit_and_fail_before_output(tmp_path, fake_script, fake_query, field, value):
    path, _ = manifest(tmp_path, fake_script, [make_row(tmp_path, fake_script)])
    with pytest.raises(runner.RunnerError):
        condor.prepare_material(path, tmp_path / "dispatch", **{**resources, "query_executable": str(fake_query[0]), field: value})
    assert not (tmp_path / "dispatch").exists()


def test_frontend_bypasses_all_parent_producer_work(tmp_path, fake_script, fake_query, monkeypatch, capsys):
    path, _ = manifest(tmp_path, fake_script, [make_row(tmp_path, fake_script)])
    def forbidden(*args, **kwargs):
        pytest.fail("Condor frontend entered parent producer work or submitted")
    monkeypatch.setattr(make_cards, "_resolve_pkl_paths", forbidden)
    monkeypatch.setattr(make_cards.datacard_tools, "load_and_merge_histogram_pkls", forbidden)
    monkeypatch.setattr(make_cards.datacard_tools, "DatacardMaker", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(condor.shutil, "which", lambda name: str(fake_query[0]))
    monkeypatch.setattr(sys, "argv", ["make_cards.py", "--condor", "--prepare-only", "--matrix-manifest", str(path), "--condor-output", str(tmp_path / "dispatch"), "--request-cpus", "1", "--request-memory", "1GB", "--request-disk", "1GB", "--max-retries", "2", "--max-starts", "3"])
    make_cards.main()
    assert json.loads(capsys.readouterr().out)["submitted"] is False
    assert (tmp_path / "dispatch" / "datacards.sub").is_file()
    assert (tmp_path / "dispatch" / "items.txt").read_text().split()[2] == "unreserved"
    assert runner.read_reservation(make_row(tmp_path, fake_script)) is None


@pytest.mark.parametrize("outcome", ["success", "spawn_error", "nonzero", "timeout", "interrupted", "io_error"])
def test_canonical_frontend_classifies_submit_outcome(tmp_path, fake_script, fake_query,
                                                      monkeypatch, capsys, outcome):
    row = make_row(tmp_path, fake_script)
    path, _ = manifest(tmp_path, fake_script, [row])
    destination = tmp_path / "dispatch"
    observed = []
    original_mark = runner.submission_transaction.mark_submission_started

    def mark_started(transaction):
        observed.append("marked")
        original_mark(transaction)

    def executable(name):
        return str(fake_query[0])

    class FakeProcess:
        returncode = 0 if outcome == "success" else 7

        def __init__(self):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1 and outcome == "timeout":
                raise subprocess.TimeoutExpired("synthetic", 1)
            if self.calls == 1 and outcome == "interrupted":
                raise KeyboardInterrupt()
            if self.calls == 1 and outcome == "io_error":
                raise OSError("synthetic communication failure")
            return "1 job(s) submitted to cluster 123.\n", "synthetic stderr"

        def kill(self):
            observed.append("killed")

    def spawn(argv, **kwargs):
        assert argv == [str(fake_query[0]), str(destination / "datacards.sub")]
        assert kwargs["pass_fds"] and kwargs["text"] is True
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.PIPE
        assert runner.probe_lock(runner.unit_lock_path(row))[0] is True
        token = (destination / "items.txt").read_text().split()[2]
        assert runner.read_reservation(row)["owner_id"] == token
        observed.append("spawned")
        if outcome == "spawn_error":
            raise OSError("synthetic process creation failure")
        return FakeProcess()

    monkeypatch.setattr(condor.shutil, "which", executable)
    monkeypatch.setattr(runner, "scheduler_snapshot", lambda scheduler: {"observed_at": "synthetic"})
    def unavailable(reservation):
        raise runner.RunnerError("SCHEDULER_OBSERVABILITY_INSUFFICIENT", "synthetic immediate observation failure")
    monkeypatch.setattr(runner, "query_reservation", unavailable)
    monkeypatch.setattr(runner.submission_transaction, "mark_submission_started", mark_started)
    monkeypatch.setattr(condor.subprocess, "Popen", spawn)
    monkeypatch.setattr(sys, "argv", ["make_cards.py", "--condor", "--matrix-manifest", str(path),
                                  "--condor-output", str(destination), "--request-cpus", "1",
                                  "--request-memory", "1GB", "--request-disk", "1GB",
                                  "--max-retries", "2", "--max-starts", "3"])
    if outcome == "success":
        make_cards.main()
    else:
        with pytest.raises(SystemExit) as error:
            make_cards.main()
        assert error.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert observed[:2] == ["marked", "spawned"]
    assert result["argv"] == [str(fake_query[0]), str(destination / "datacards.sub")]
    assert result["start_time"] and result["stdout"] == ("" if outcome == "spawn_error" else "1 job(s) submitted to cluster 123.\n")
    assert result["exit_status"] == (None if outcome == "spawn_error" else FakeProcess.returncode)
    if outcome == "spawn_error":
        assert runner.read_reservation(row) is None
    else:
        assert runner.read_reservation(row)["owner_id"] == (destination / "items.txt").read_text().split()[2]
    assert runner.probe_lock(runner.unit_lock_path(row))[0] is False
    if outcome == "success":
        assert result["submission_outcome"] == "SUBMIT_COMMAND_SUCCEEDED"
        assert result["submitted"] is True
        assert runner.read_reservation(row)["scheduler_id"] is None
    elif outcome == "spawn_error":
        assert result["submission_outcome"] == "DEFINITE_SPAWN_FAILURE"
    else:
        assert result["submission_outcome"] == "AMBIGUOUS_SUBMISSION_OUTCOME"
        assert result["submitted"] is False


@pytest.mark.parametrize("batch_name", [None, "operator group"])
@pytest.mark.parametrize("observation_outcome", ["active", "query_error", "record_error"])
def test_submit_success_keeps_reservation_and_optional_batch_name(
        tmp_path, fake_script, fake_query, monkeypatch, batch_name, observation_outcome):
    row = make_row(tmp_path, fake_script)
    path, _ = manifest(tmp_path, fake_script, [row])
    destination = tmp_path / "dispatch"
    material = condor.prepare_material(path, destination, **resources,
                                       query_executable=str(fake_query[0]))
    monkeypatch.setattr(condor.shutil, "which", lambda name: str(fake_query[0]))
    monkeypatch.setattr(runner, "scheduler_snapshot", lambda scheduler: {
        "observed_at": "synthetic", "collector": "synthetic.pool"})
    expected_argv = [str(fake_query[0])]
    if batch_name is not None:
        expected_argv += ["-batch-name", batch_name]
    expected_argv += [str(destination / "datacards.sub")]

    class FakeProcess:
        returncode = 0

        def communicate(self, timeout=None):
            return "1 job(s) submitted to cluster 123.\n", ""

    def spawn(argv, **kwargs):
        assert argv == expected_argv
        assert kwargs["pass_fds"]
        return FakeProcess()

    monkeypatch.setattr(condor.subprocess, "Popen", spawn)

    def observe(reservation):
        if observation_outcome == "query_error":
            raise runner.RunnerError("SCHEDULER_OBSERVABILITY_INSUFFICIENT", "synthetic query failure")
        return {"state": "RUNNING", "ownership": "ACTIVE", "scheduler_id": "123.0",
                "source": "default", "schedd": "synthetic.schedd", "pool": "synthetic.pool",
                "global_job_id": "synthetic.schedd#123.0#1", "observed_at": "synthetic"}

    monkeypatch.setattr(runner, "query_reservation", observe)
    if observation_outcome == "record_error":
        monkeypatch.setattr(runner, "record_scheduler_observation", lambda *args: (_ for _ in ()).throw(OSError("synthetic write failure")))
    result = condor.submit_material(path, destination, material["scheduler"], batch_name=batch_name)
    assert result["submitted"] is True
    assert result["submission_outcome"] == "SUBMIT_COMMAND_SUCCEEDED"
    assert result["argv"] == expected_argv
    assert result["reported_cluster_id"] == 123
    assert result["pre_submit_scheduler_snapshot"]["collector"] == "synthetic.pool"
    assert runner.read_reservation(row)["owner_id"] == (destination / "items.txt").read_text().split()[2]
    if observation_outcome == "active":
        assert runner.read_reservation(row)["scheduler_observation"]["global_job_id"] == "synthetic.schedd#123.0#1"
        assert result["post_submit_observations"][0]["ownership"] == "ACTIVE"
    else:
        assert runner.read_reservation(row)["scheduler_id"] is None
        assert result["post_submit_observations"][0]["state"] == "UNKNOWN"


def test_worker_rejects_descriptor_mutation(tmp_path, fake_script, fake_query):
    rows = [make_row(tmp_path, fake_script, name) for name in ("a", "b")]
    path, _ = manifest(tmp_path, fake_script, rows)
    destination = tmp_path / "dispatch"
    condor.prepare_material(path, destination, **resources, query_executable=str(fake_query[0]))
    item = (destination / "items.txt").read_text().splitlines()[0].split()
    (destination / "units" / "a.json").write_text((destination / "units" / "b.json").read_text())
    result = subprocess.run([str(destination / "worker.sh"), *item[:2], "123.7", item[2]], capture_output=True, text=True)
    assert result.returncode == 2 and "stale_dispatch_input" in result.stderr
    assert not any(Path(row["output_root"]).exists() for row in rows)


@pytest.mark.parametrize("policy", ["warn", "error", "off"])
@pytest.mark.parametrize("binning", ["fitting", "processing"])
@pytest.mark.parametrize("skip", [False, True])
def test_worker_preserves_canonical_producer_options(tmp_path, fake_script, fake_query, policy, binning, skip):
    extras = ["--year-coverage-policy", policy, "--rate-syst-json", "custom/rates.json"]
    if skip:
        extras.append("--skip-missing-parton-rate-syst")
    row = make_row(tmp_path, fake_script, extra=extras)
    row["producer_args"][row["producer_args"].index("--binning") + 1] = binning
    path, data = manifest(tmp_path, fake_script, [row])
    destination = tmp_path / "dispatch"
    condor.prepare_material(path, destination, **resources, query_executable=str(fake_query[0]))
    unit = json.loads((destination / "units" / (row["row_id"] + ".json")).read_text())
    parent, _, selected = runner.resolve_unit(unit)
    assert runner.resolved_argv(parent, selected) == runner.resolved_argv(data, row)
    assert selected["producer_args"] == row["producer_args"]
    assert "--sr-registry" not in selected["producer_args"]
    assert "--condor" not in selected["producer_args"]


def test_scheduler_overlay_has_no_completion_or_retry_authority(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    status = runner.read_only("status", data, sha(path))
    overlay = {"schema": "topeft_datacard_scheduler_status_v1", "parent_integrity": sha(path), "observed_at": "synthetic", "units": [{"row_id": row["row_id"], "scheduler_id": "123.0", "state": "completed_exit_0"}]}
    runner.overlay_scheduler(status, overlay)
    assert status["rows"][0]["status"] == "not_started"
    assert status["rows"][0]["scheduler_metadata"]["completion_authority"] is False
    runner.execute_unit(data, sha(path), row)
    status = runner.read_only("status", data, sha(path))
    overlay["units"][0]["state"] = "removed"
    runner.overlay_scheduler(status, overlay)
    assert status["rows"][0]["status"] == "complete"


def test_frontend_requires_manifest_and_rejects_producer_options(monkeypatch):
    parser_options = {option for action in make_cards.build_arg_parser()._actions
                      for option in action.option_strings}
    assert "--condor-schedd" not in parser_options
    assert "--condor-pool" not in parser_options
    assert {"--condor-batch-name", "--prepare-only"} <= parser_options
    for args in (["--condor"], ["--condor", "--matrix-manifest", "/manifest.json", "--condor-output", "/dispatch", "--year", "2022"], ["--condor", "--matrix-manifest", "/manifest.json", "--condor-output", "/dispatch", "--binning", "fitting"], ["--condor", "--matrix-manifest", "/manifest.json", "--condor-output", "/dispatch", "--year-coverage-policy", "warn"], ["--chunks", "2"], ["--request-cpus", "1"], ["--prepare-only"]):
        monkeypatch.setattr(sys, "argv", ["make_cards.py", *args])
        with pytest.raises(SystemExit) as error:
            make_cards.main()
        assert error.value.code == 2


def test_scheduler_overlay_rejects_wrong_parent_or_duplicate(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    observation = {"row_id": row["row_id"], "scheduler_id": "1.0", "state": "idle"}
    overlay = {"schema": "topeft_datacard_scheduler_status_v1", "parent_integrity": sha(path), "observed_at": "synthetic", "units": [observation, observation]}
    with pytest.raises(runner.RunnerError, match="duplicate"):
        runner.overlay_scheduler(runner.read_only("status", data, sha(path)), overlay)
    overlay.update(parent_integrity="0" * 64, units=[observation])
    with pytest.raises(runner.RunnerError, match="binding"):
        runner.overlay_scheduler(runner.read_only("status", data, sha(path)), overlay)


def test_prepared_material_is_ownership_free_and_permits_local_execution(tmp_path, fake_script, fake_query):
    counter = tmp_path / "producer_count"
    row = make_row(tmp_path, fake_script, extra=["--counter", str(counter)])
    path, data = manifest(tmp_path, fake_script, [row])
    condor.prepare_material(path, tmp_path / "dispatch", **resources, query_executable=str(fake_query[0]))
    condor.prepare_material(path, tmp_path / "duplicate", **resources, query_executable=str(fake_query[0]))
    assert runner.read_reservation(row) is None
    runner.execute_unit(data, sha(path), row)
    assert counter.read_text() == "1"
    assert runner.read_reservation(row) is None


@pytest.mark.parametrize("outcome", ["no_submit", "local_failure", "success", "timeout", "nonzero", "interrupted"])
def test_submission_transaction_outcomes(tmp_path, fake_script, fake_query, outcome):
    rows = [make_row(tmp_path, fake_script, name) for name in ("a", "b")]
    path, _ = manifest(tmp_path, fake_script, rows)
    destination = tmp_path / "dispatch"
    material = condor.prepare_material(path, destination, **resources, query_executable=str(fake_query[0]))
    started = outcome in {"success", "timeout", "nonzero", "interrupted"}
    failure = {"local_failure": RuntimeError("synthetic pre-submit failure"),
               "timeout": subprocess.TimeoutExpired("synthetic_submit", 1),
               "nonzero": subprocess.CalledProcessError(1, "synthetic_submit"),
               "interrupted": KeyboardInterrupt()}.get(outcome)

    def exercise():
        with condor.submission_transaction(path, destination, material["scheduler"]) as transaction:
            assert len(transaction.reservations) == len(rows)
            assert all(runner.probe_lock(runner.unit_lock_path(row))[0] for row in rows)
            if started:
                transaction.mark_submission_started()
            if failure is not None:
                raise failure

    if failure is None:
        exercise()
    else:
        with pytest.raises(type(failure)):
            exercise()
    assert all((runner.read_reservation(row) is not None) == started for row in rows)
    assert all(runner.probe_lock(runner.unit_lock_path(row))[0] is False for row in rows)
    assert not any(Path(row["output_root"]).exists() for row in rows)


def test_submission_transaction_rolls_back_partial_row_acquisition(tmp_path, fake_script, fake_query):
    rows = [make_row(tmp_path, fake_script, name) for name in ("a", "b")]
    path, data = manifest(tmp_path, fake_script, rows)
    destination = tmp_path / "dispatch"
    material = condor.prepare_material(path, destination, **resources, query_executable=str(fake_query[0]))
    with runner.unit_lease(data, rows[1]):
        with pytest.raises(runner.RunnerError) as error:
            with condor.submission_transaction(path, destination, material["scheduler"]):
                pytest.fail("transaction acquired occupied second row")
        assert error.value.code == "execution_lock_held"
        assert all(runner.read_reservation(row) is None for row in rows)
        assert runner.probe_lock(runner.unit_lock_path(rows[1]))[0] is True
    assert all(line.split()[2] == "unreserved" for line in (destination / "items.txt").read_text().splitlines())


def test_submission_transaction_itemdata_failure_rolls_back(tmp_path, fake_script, fake_query, monkeypatch):
    row = make_row(tmp_path, fake_script)
    path, _ = manifest(tmp_path, fake_script, [row])
    destination = tmp_path / "dispatch"
    material = condor.prepare_material(path, destination, **resources, query_executable=str(fake_query[0]))
    original = Path.write_text

    def fail_items(target, *args, **kwargs):
        if target == destination / "items.txt":
            raise OSError("synthetic itemdata write failure")
        return original(target, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_items)
    with pytest.raises(OSError, match="itemdata"):
        with condor.submission_transaction(path, destination, material["scheduler"]):
            pytest.fail("transaction yielded before itemdata publication")
    assert runner.read_reservation(row) is None


def test_transaction_tokens_are_safe_unique_and_bind_itemdata(tmp_path, fake_script, fake_query):
    rows = [make_row(tmp_path, fake_script, name) for name in ("a", "b")]
    path, _ = manifest(tmp_path, fake_script, rows)
    destination = tmp_path / "dispatch"
    material = condor.prepare_material(path, destination, **resources, query_executable=str(fake_query[0]))
    seen = set()
    for _ in range(2):
        with condor.submission_transaction(path, destination, material["scheduler"]) as transaction:
            tokens = {item["owner_id"] for item in transaction.reservations}
            assert len(tokens) == len(rows) and tokens.isdisjoint(seen)
            assert all(re.fullmatch(r"[0-9a-f]{32}", token) for token in tokens)
            assert {line.split()[2] for line in (destination / "items.txt").read_text().splitlines()} == tokens
            seen.update(tokens)


@pytest.mark.parametrize("fault,code", [
    ("missing", "worker_job_ad_missing"),
    ("unreadable", "worker_job_ad_malformed"),
    ("expression", "worker_job_ad_malformed"),
    ("duplicate", "worker_job_ad_malformed"),
    ("row", "worker_job_ad_identity_conflict"),
    ("token", "worker_job_ad_identity_conflict"),
])
def test_worker_requires_local_job_ad_identity(tmp_path, fake_script, fake_query, fault, code):
    row = make_row(tmp_path, fake_script)
    path, _ = manifest(tmp_path, fake_script, [row])
    destination = tmp_path / "dispatch"
    material = condor.prepare_material(path, destination, **resources, query_executable=str(fake_query[0]))
    with condor.submission_transaction(path, destination, material["scheduler"]) as transaction:
        transaction.mark_submission_started()
    row_id, integrity, token = (destination / "items.txt").read_text().split()
    ad_path = tmp_path / "worker.ad"
    worker_env = write_worker_ad(ad_path, row_id, token)
    if fault == "missing":
        del worker_env["_CONDOR_JOB_AD"]
    elif fault == "unreadable":
        ad_path.unlink()
    elif fault == "expression":
        ad_path.write_text(ad_path.read_text().replace("ClusterId = 123", "ClusterId = 100 + 23"))
    elif fault == "duplicate":
        ad_path.write_text(ad_path.read_text() + "clusterid = 456\n")
    else:
        worker_env = write_worker_ad(ad_path, "other_row" if fault == "row" else row_id,
                                    "0" * 32 if fault == "token" else token)
    fake_query[0].write_text('#!/bin/sh\necho forbidden_worker_query >&2\nexit 93\n')
    result = subprocess.run([str(destination / "worker.sh"), row_id, integrity, "999.9", token],
                            env=worker_env, capture_output=True, text=True)
    assert result.returncode == 2 and code in result.stderr
    assert "forbidden_worker_query" not in result.stderr
    assert not Path(row["output_root"]).exists()
    assert runner.read_reservation(row)["owner_id"] == token


def test_direct_make_cards_refuses_reserved_output_before_producer(tmp_path, fake_script, fake_query, monkeypatch, capsys):
    row = make_row(tmp_path, fake_script)
    path, _ = manifest(tmp_path, fake_script, [row])
    destination = tmp_path / "dispatch"
    material = condor.prepare_material(path, destination, **resources, query_executable=str(fake_query[0]))
    with condor.submission_transaction(path, destination, material["scheduler"]) as transaction:
        transaction.mark_submission_started()
    reservation_before = runner.reservation_path(row).read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("reserved direct invocation reached producer or scheduler")

    monkeypatch.setattr(make_cards, "_run_producer", forbidden)
    monkeypatch.setattr(runner, "query_reservation", forbidden)
    monkeypatch.setattr(sys, "argv", ["make_cards.py", row["input_pkl"], "--out-dir", row["output_root"]])
    with pytest.raises(SystemExit) as error:
        make_cards.main()
    assert error.value.code == 2
    assert "logical reservation" in capsys.readouterr().err
    assert runner.reservation_path(row).read_bytes() == reservation_before
    assert not Path(row["output_root"]).exists()


def test_direct_make_cards_holds_domain_lease_until_producer_returns(tmp_path, fake_script, fake_query, monkeypatch):
    rows = [make_row(tmp_path, fake_script, name) for name in ("direct", "independent")]
    path, data = manifest(tmp_path, fake_script, rows)
    scheduler = {"schedd": "synthetic_schedd", "pool": "synthetic_pool", "query_executable": str(fake_query[0])}
    observed = []

    def synthetic_producer(args, pkl_files):
        assert args.out_dir == rows[0]["output_root"]
        assert pkl_files == [rows[0]["input_pkl"]]
        assert runner.probe_lock(runner.unit_lock_path(rows[0]))[0] is True
        with pytest.raises(runner.RunnerError) as error:
            with runner.reserve_condor(data, sha(path), [rows[0]], scheduler):
                pytest.fail("reserved row accepted")
        assert error.value.code == "execution_lock_held"
        with runner.reserve_condor(data, sha(path), [rows[1]], scheduler) as independent:
            assert independent.reservations[0]["row_id"] == rows[1]["row_id"]
        observed.append("producer_entered_with_exclusion")

    monkeypatch.setattr(make_cards, "_run_producer", synthetic_producer)
    monkeypatch.setattr(sys, "argv", ["make_cards.py", rows[0]["input_pkl"], "--out-dir", rows[0]["output_root"]])
    make_cards.main()
    assert observed == ["producer_entered_with_exclusion"]
    assert runner.probe_lock(runner.unit_lock_path(rows[0]))[0] is False
    with runner.reserve_condor(data, sha(path), [rows[0]], scheduler) as transaction:
        assert transaction.reservations[0]["row_id"] == rows[0]["row_id"]


def test_direct_make_cards_preserves_completed_h2_outputs_without_reservation(tmp_path, fake_script, monkeypatch, capsys):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    runner.execute_unit(data, sha(path), row)
    assert runner.read_reservation(row) is None
    receipt_before = runner.receipt_path(data, row).read_bytes()
    outputs_before = {name: Path(name).read_bytes() for name in row["expected_output_paths"]}

    def forbidden(*args, **kwargs):
        pytest.fail("direct producer entered an H2-published output domain")

    monkeypatch.setattr(make_cards, "_run_producer", forbidden)
    monkeypatch.setattr(sys, "argv", ["make_cards.py", row["input_pkl"], "--out-dir", row["output_root"]])
    with pytest.raises(SystemExit) as error:
        make_cards.main()
    assert error.value.code == 2
    assert "matrix runner" in capsys.readouterr().err
    assert runner.receipt_path(data, row).read_bytes() == receipt_before
    assert {name: Path(name).read_bytes() for name in row["expected_output_paths"]} == outputs_before
    assert runner.validate_receipt(json.loads(receipt_before), data, sha(path), row)[0]


def test_real_topeft_runtime_imports_with_sanitized_environment(tmp_path, record_property):
    # The maintained interpreter resolves editable TopEFT/topcoffea packages
    # without the interactive shell's PYTHONPATH, CONDA_PREFIX or library paths.
    result = subprocess.run(
        [sys.executable, "-B", str(Path(make_cards.__file__).resolve()), "--help"],
        cwd=tmp_path, env={"PATH": "/usr/bin:/bin", "LANG": "C"},
        capture_output=True, text=True, timeout=60,
    )
    record_property("sanitized_runtime_probe", json.dumps({
        "argv": result.args, "returncode": result.returncode,
        "environment": {"PATH": "/usr/bin:/bin", "LANG": "C"},
        "stdout": result.stdout, "stderr": result.stderr,
    }, sort_keys=True))
    assert result.returncode == 0, result.stderr
    assert "AllowedExecuteDuration seconds per activation" in " ".join(result.stdout.split())
    assert "--max-retries" in result.stdout and "--max-starts" in result.stdout


def test_installed_parser_and_native_policy_evaluation(tmp_path, fake_script, fake_query, record_property):
    submit_executable = shutil.which("condor_submit")
    evaluator = shutil.which("classad_eval")
    if not submit_executable or not evaluator:
        pytest.skip("installed HTCondor submit parser and ClassAd evaluator required")
    rows = [make_row(tmp_path, fake_script, name) for name in ("z_last", "a_first")]
    path, _ = manifest(tmp_path, fake_script, rows)
    destination = tmp_path / "dispatch"
    material = condor.prepare_material(path, destination, **resources, max_runtime=21600,
                                       query_executable=str(fake_query[0]))
    with condor.submission_transaction(path, destination, material["scheduler"]):
        # Binding is tested without any possible submission side effect.
        pass
    assert all(runner.read_reservation(row) is None for row in rows)
    # This is parser input only. No submitter variables or real credentials enter
    # generated material; the loopback override needs no network connection.
    parser_env = {"PATH": "/usr/bin:/bin", "LANG": "C", "_CONDOR_NETWORK_INTERFACE": "127.0.0.1"}
    site_config = Path(submit_executable).resolve().parents[2] / "etc" / "condor_config"
    if site_config.is_file():
        parser_env["CONDOR_CONFIG"] = str(site_config)
    elif os.environ.get("CONDOR_CONFIG"):
        parser_env["CONDOR_CONFIG"] = os.environ["CONDOR_CONFIG"]
    ads_path = tmp_path / "parsed.ads"
    parsed = subprocess.run([submit_executable, "-dry-run", str(ads_path), str(destination / "datacards.sub")],
                            env=parser_env, text=True, capture_output=True, timeout=30)
    record_property("native_parser_probe", json.dumps({
        "argv": parsed.args, "returncode": parsed.returncode, "environment": parser_env,
        "stdout": parsed.stdout, "stderr": parsed.stderr,
    }, sort_keys=True))
    assert parsed.returncode == 0, parsed.stderr
    batch_ads = tmp_path / "batch_parsed.ads"
    batch_parsed = subprocess.run([submit_executable, "-dry-run", str(batch_ads),
                                   "-batch-name", "operator_group", str(destination / "datacards.sub")],
                                  env=parser_env, text=True, capture_output=True, timeout=30)
    record_property("native_batch_parser_probe", json.dumps({
        "argv": batch_parsed.args, "returncode": batch_parsed.returncode,
        "stdout": batch_parsed.stdout, "stderr": batch_parsed.stderr,
    }, sort_keys=True))
    assert batch_parsed.returncode == 0, batch_parsed.stderr
    assert 'JobBatchName="operator_group"' in batch_ads.read_text()
    # The long format stores common fields in the first ad and subsequent proc
    # deltas. Reopen the emitted bytes and apply inheritance for each queued row.
    raw_ads = ads_path.read_text()
    record_property("native_parser_raw_ads", raw_ads)
    blocks = re.split(r"\n\s*\n", raw_ads.strip())
    ads = []
    common = {}
    for block in blocks:
        fields = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if not fields:
            continue
        if not ads:
            common = fields
        ads.append({**common, **fields})
    record_property("native_parser_materialized_ads", json.dumps(ads, sort_keys=True))
    assert len(ads) == len(rows)
    items = [line.split() for line in (destination / "items.txt").read_text().splitlines()]
    for ad, row, item in zip(ads, rows, items):
        assert ad["AllowedExecuteDuration"] == "21600"
        assert ad["ShouldTransferFiles"] == '"NO"'
        assert ad.get("Environment", '""') == '""'
        assert ad["OnExitHold"] == "false"
        assert ad["JobMaxRetries"] == "2" and ad["JobSuccessExitCode"] == "0"
        assert ad["NumJobStarts"] == "0"
        assert json.loads(ad["DatacardUnit"]) == row["row_id"]
        assert json.loads(ad["DatacardReservation"]) == item[2]
        job_id = f'{ad["ClusterId"]}.{ad["ProcId"]}'
        assert json.loads(ad["Arguments"]).split() == [item[0], item[1], job_id, item[2]]
        for attribute, suffix in (("Out", "out"), ("Err", "err"), ("UserLog", "events.log")):
            assert json.loads(ad[attribute]) == str(destination / "logs" / f'{row["row_id"]}__{job_id}.{suffix}')
        assert "MaxRuntime" not in ad and "TransferOutput" not in ad
    assert len({ad["UserLog"] for ad in ads}) == len(rows)
    assert "MY.NumJobStarts < 3" in ads[0]["Requirements"]
    assert "NumJobCompletions > JobMaxRetries" in ads[0]["OnExitRemove"]

    evaluator_probe = subprocess.run([evaluator, "-quiet", "[X=1]", "X"], env=parser_env,
                                     text=True, capture_output=True, timeout=10)
    if evaluator_probe.returncode and "No module named 'classad'" in evaluator_probe.stderr:
        record_property("native_policy_evaluator_unavailable", evaluator_probe.stderr)
        return
    assert evaluator_probe.returncode == 0, evaluator_probe.stderr

    policy_evaluations = []

    def evaluate(expression, **values):
        attributes = {"JobMaxRetries": 2, "JobSuccessExitCode": 0, "NumJobCompletions": 0,
                      "NumJobStarts": 0, "JobStatus": 1, "ExitCode": 1, **values}
        context = "[" + ";".join(f"{key}={value}" for key, value in attributes.items()) + "]"
        result = subprocess.run([evaluator, "-quiet", context, expression], env=parser_env,
                                text=True, capture_output=True, timeout=10)
        policy_evaluations.append({"expression": expression, "context": attributes,
                                   "returncode": result.returncode, "result": result.stdout.strip(),
                                   "stderr": result.stderr})
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    remove = ads[0]["OnExitRemove"]
    assert evaluate(remove, NumJobCompletions=1, NumJobStarts=1) == "false"
    assert evaluate(remove, NumJobCompletions=2, NumJobStarts=2) == "false"
    assert evaluate(remove, NumJobCompletions=3, NumJobStarts=2) == "true"
    assert evaluate(remove, ExitCode=0, NumJobCompletions=1, NumJobStarts=1) == "true"
    assert evaluate(remove, ExitCode="undefined", NumJobCompletions=1, NumJobStarts=1) == "false"
    assert evaluate(remove, ExitCode="undefined", NumJobCompletions=3, NumJobStarts=2) == "true"
    assert evaluate(remove, NumJobCompletions=1, NumJobStarts=3) == "true"
    periodic = ads[0]["PeriodicRemove"]
    assert evaluate(periodic, JobStatus=1, NumJobStarts=2) == "false"
    assert evaluate(periodic, JobStatus=1, NumJobStarts=3) == "true"
    for code in (0, 1, 3, 6, 12, 13, 26, 32, 33, 34, 46, 47, 999, "undefined"):
        assert evaluate(periodic, JobStatus=5, NumJobStarts=0, HoldReasonCode=code) == "true"
        assert evaluate(periodic, JobStatus=5, NumJobStarts="undefined", HoldReasonCode=code) == "true"
    for status in (2, 6, 7):
        assert evaluate(periodic, JobStatus=status, NumJobStarts=3) == "false"
    assert ads[0].get("PeriodicRelease", "false") == "false"
    # Final permitted activation may finish; no further match is eligible.
    assert evaluate("MY.NumJobStarts < 3", NumJobStarts=2) == "true"
    assert evaluate("MY.NumJobStarts < 3", NumJobStarts=3) == "false"
    assert evaluate("MY.NumJobStarts < 3", NumJobStarts="undefined") == "undefined"
    record_property("native_policy_evaluations", json.dumps(policy_evaluations, sort_keys=True))
