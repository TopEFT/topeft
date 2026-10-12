import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import pytest

from analysis.topeft_run2 import datacard_matrix_runner as runner_engine


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "analysis/topeft_run2/run_datacard_matrix_resumable.sh"
ENGINE = ROOT / "analysis/topeft_run2/datacard_matrix_runner.py"

FAKE = r'''
from pathlib import Path
import json, sys, time
args = sys.argv[1:]
input_pkl = args[0]
def one(name): return args[args.index(name) + 1]
def many(name):
    result = []
    for item in args[args.index(name) + 1:]:
        if item.startswith("--"): break
        result.append(item)
    return result
if "--record-argv" in args: Path(one("--record-argv")).write_text(json.dumps(args))
if "--counter" in args:
    p = Path(one("--counter")); p.write_text(str(int(p.read_text()) + 1 if p.exists() else 1))
if "--order" in args:
    with Path(one("--order")).open("a") as h: h.write(one("--token") + "\n")
if "--write-path" in args:
    p = Path(one("--write-path")); p.parent.mkdir(parents=True, exist_ok=True); p.write_text("external\n")
if "--mutate-path" in args: Path(one("--mutate-path")).write_text("runtime drift\n")
if "--wait-for" in args:
    deadline = time.monotonic() + 30
    while not Path(one("--wait-for")).exists():
        if time.monotonic() > deadline: raise SystemExit(8)
        time.sleep(.01)
if "--sleep" in args: time.sleep(float(one("--sleep")))
if "--fail" in args: raise SystemExit(7)
root = Path(one("--out-dir")); root.mkdir(parents=True, exist_ok=True)
report = Path(one("--merge-report")); report.parent.mkdir(parents=True, exist_ok=True); report.write_text("{}\n")
(root / "selectedWCs.txt").write_text("selected\n")
(root / "scalings-preselect.json").write_text("scalings\n")
for name in many("--expected-output"):
    p = Path(name); p.parent.mkdir(parents=True, exist_ok=True); p.write_text("synthetic " + input_pkl + "\n")
    if "--partial-once" in args:
        marker = Path(one("--partial-once"))
        if not marker.exists():
            marker.write_text("failed\n")
            raise SystemExit(7)
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def fake_script(tmp_path):
    script = tmp_path / "runtime" / "make_cards.py"
    script.parent.mkdir()
    script.write_text(FAKE)
    return script


def make_row(tmp_path, fake, row_id="row_01", attempt="attempt_01", extra=None):
    root = tmp_path / "cards" / f"{row_id}_{attempt}"
    evidence = tmp_path / "evidence" / f"{row_id}_{attempt}"
    input_pkl = tmp_path / "inputs" / f"{row_id}_{attempt}.pkl.gz"
    input_pkl.parent.mkdir(exist_ok=True); input_pkl.write_text("input\n")
    missing = tmp_path / "inputs" / f"{row_id}_{attempt}_missing.root"; missing.write_text("missing\n")
    outputs = [root / "card.txt", root / "card.root"]
    args = ["--out-dir", str(root), "--var-lst", "lj0pt", "--ch-lst", "channel_a", "channel_b", "--binning", "fitting", "--year", "2022", "2022EE", "--miss-parton-file", str(missing), "--merge-report", str(evidence / "merge.json"), "--expected-output", *map(str, outputs), *(extra or [])]
    return {"row_id": row_id, "logical_row_id": row_id, "attempt_id": attempt, "era": "run3", "working_directory": str(fake.parent), "input_pkl": str(input_pkl), "output_root": str(root), "distribution": "lj0pt", "physical_channels": ["channel_a", "channel_b"], "years": ["2022", "2022EE"], "missing_parton_path": str(missing), "merge_report_path": str(evidence / "merge.json"), "snapshot_directory": str(tmp_path / "control" / "snapshots" / f"{row_id}_{attempt}"), "log_path": str(evidence / "row.log"), "expected_output_paths": list(map(str, outputs)), "producer_args": args}


def manifest(tmp_path, fake, rows, name="manifest.json"):
    data = {"schema": "topeft_datacard_matrix_v3", "control_root": str(tmp_path / "control"), "lock_path": str(tmp_path / "control" / "runner.lock"), "runtime_contract": {"contract_id": "synthetic-runtime", "python_executable": sys.executable, "make_cards_path": str(fake), "fingerprints": [{"path": str(fake), "sha256": sha(fake)}]}, "rows": rows}
    path = tmp_path / name; path.write_text(json.dumps(data, indent=2) + "\n")
    return path, data


def invoke(path, mode=None):
    return subprocess.run([str(RUNNER), *( [mode] if mode else []), str(path)], text=True, capture_output=True)


def result(stream):
    for start in reversed([i for i, char in enumerate(stream) if char == "{"]):
        try: return json.loads(stream[start:])
        except json.JSONDecodeError: pass
    raise AssertionError(stream)


def receipt(tmp_path, row):
    return json.loads((tmp_path / "control" / "receipts" / f"{row['row_id']}__{row['attempt_id']}.json").read_text())


def replace_option_value(row, option, value):
    row["producer_args"][row["producer_args"].index(option) + 1] = str(value)


def wait_for_owner(path, row_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            owner = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(.01)
            continue
        if owner.get("current_row_id") == row_id:
            return owner
        time.sleep(.01)
    raise AssertionError(f"owner metadata never reached {row_id}")


def test_public_wrapper_is_portable_and_has_no_shell_exit(tmp_path):
    source = RUNNER.read_text()
    assert re.search(r"\bexit\b", source) is None
    assert "/users/apiccine/work/correction-lib" not in source
    relocated = tmp_path / "different-checkout" / "analysis" / "topeft_run2"
    relocated.mkdir(parents=True)
    relocated_wrapper = relocated / RUNNER.name
    shutil.copy2(RUNNER, relocated_wrapper)
    shutil.copy2(ENGINE, relocated / ENGINE.name)
    completed = subprocess.run(
        [str(relocated_wrapper), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0
    assert "Run a prequalified datacard matrix" in completed.stdout
    no_interpreter = subprocess.run(
        ["/bin/bash", str(relocated_wrapper), "--help"],
        cwd=tmp_path,
        env={"PATH": ""},
        text=True,
        capture_output=True,
    )
    assert no_interpreter.returncode == 127
    assert "python or python3 is required for the runner engine" in no_interpreter.stderr


def test_schema_and_identity_rejections(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script); row.pop("era")
    path, _ = manifest(tmp_path, fake_script, [row]); assert result(invoke(path).stderr)["status"] == "manifest_schema_error"
    first, second = make_row(tmp_path, fake_script), make_row(tmp_path, fake_script)
    path, _ = manifest(tmp_path, fake_script, [first, second], "duplicate.json"); assert result(invoke(path, "--plan-only").stderr)["status"] == "manifest_schema_error"


def test_historical_v2_rejected_before_runtime_mutation(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script, extra=["--counter", str(tmp_path / "counter")])
    path, data = manifest(tmp_path, fake_script, [row])
    data["schema"] = "topeft_datacard_matrix_v2"
    path.write_text(json.dumps(data))
    completed = invoke(path)
    assert completed.returncode != 0
    error = result(completed.stderr)
    assert error["status"] == "manifest_schema_error"
    message = error["message"]
    assert all(anchor in message for anchor in (
        "Historical", "topeft_datacard_matrix_v2", "current datacard workflow",
        "topeft_datacard_matrix_v3", "make_datacard_matrix_manifest.py",
    ))
    assert "new runtime execution" not in message
    assert not Path(data["control_root"]).exists()
    assert not (tmp_path / "counter").exists()


def test_v3_requires_logical_row_id_and_rejects_old_consumer_option(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script)
    row.pop("logical_row_id")
    path, _ = manifest(tmp_path, fake_script, [row])
    assert result(invoke(path, "--plan-only").stderr)["status"] == "manifest_schema_error"
    row = make_row(tmp_path, fake_script)
    row["producer_args"].extend(["--sr-registry", "ALL_CH_LST_SR"])
    path, _ = manifest(tmp_path, fake_script, [row], "old-option.json")
    assert result(invoke(path, "--plan-only").stderr)["status"] == "manifest_schema_error"
    row = make_row(tmp_path, fake_script, extra=["--cache-merged-pkl", str(tmp_path / "external.pkl.gz")])
    path, _ = manifest(tmp_path, fake_script, [row], "external-cache.json")
    assert result(invoke(path, "--plan-only").stderr)["status"] == "manifest_schema_error"


def test_plan_and_status_are_nonmutating(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script, extra=["--counter", str(tmp_path / "counter")]); path, data = manifest(tmp_path, fake_script, [row])
    assert result(invoke(path, "--plan-only").stdout)["rows"][0]["action"] == "execute"
    assert result(invoke(path, "--status").stdout)["rows"][0]["status"] == "not_started"
    assert not Path(data["control_root"]).exists() and not (tmp_path / "counter").exists()


def test_plan_only_reports_missing_execution_input_but_skips_complete_history(tmp_path, fake_script):
    missing = make_row(tmp_path, fake_script, "missing")
    Path(missing["missing_parton_path"]).unlink()
    missing_path, _ = manifest(tmp_path, fake_script, [missing], "missing-plan.json")
    missing_plan = invoke(missing_path, "--plan-only")
    missing_result = result(missing_plan.stdout)
    assert missing_plan.returncode != 0
    assert not missing_result["launch_ready"]
    assert missing_result["rows"][0]["action"] == "block"
    assert not missing_result["rows"][0]["execution_input_availability"]["checks"]["missing_parton_path"]["available"]

    complete = make_row(tmp_path, fake_script, "complete")
    complete_path, _ = manifest(tmp_path, fake_script, [complete], "complete-plan.json")
    assert invoke(complete_path).returncode == 0
    Path(complete["input_pkl"]).unlink()
    Path(complete["missing_parton_path"]).unlink()
    complete_plan = invoke(complete_path, "--plan-only")
    complete_result = result(complete_plan.stdout)
    assert complete_plan.returncode == 0
    assert complete_result["launch_ready"]
    assert complete_result["rows"][0]["status"] == "complete"
    assert complete_result["rows"][0]["action"] == "skip"
    assert not complete_result["rows"][0]["execution_input_availability"]["checked"]


def test_active_interrupted_and_stale_owner_statuses(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script); path, data = manifest(tmp_path, fake_script, [row])
    free = result(invoke(path, "--status").stdout)
    assert free["lock_held"] is False and free["lock_probe_error"] is None
    lock = Path(data["lock_path"]); lock.parent.mkdir(parents=True); lock.touch()
    stale_owner = {"schema_version": "topeft_datacard_runner_owner_v1", "manifest_sha256": sha(path), "current_row_id": row["row_id"], "current_attempt_id": row["attempt_id"]}
    (tmp_path / "control" / "runner_owner.json").write_text(json.dumps(stale_owner))
    log = Path(row["log_path"]); log.parent.mkdir(parents=True); log.write_text("interrupted\n")
    stale = result(invoke(path, "--status").stdout)
    assert stale["lock_held"] is False and stale["rows"][0]["status"] == "canonical_collision_or_ambiguous_state"
    with lock.open("a+b") as handle:
        runner_engine.fcntl.flock(handle.fileno(), runner_engine.fcntl.LOCK_EX | runner_engine.fcntl.LOCK_NB)
        active = result(invoke(path, "--status").stdout)
        assert active["lock_held"] is True and active["lock_probe_error"] is None and active["rows"][0]["status"] == "active"
        wrong_owner = {**stale_owner, "current_row_id": "different_row"}
        (tmp_path / "control" / "runner_owner.json").write_text(json.dumps(wrong_owner))
        mismatched = result(invoke(path, "--status").stdout)
        assert mismatched["lock_held"] is True and mismatched["rows"][0]["status"] == "held_lock_active_or_unknown"
    (tmp_path / "control" / "runner_owner.json").write_text(json.dumps(stale_owner))
    final = result(invoke(path, "--status").stdout)
    assert final["lock_held"] is False and final["rows"][0]["status"] == "canonical_collision_or_ambiguous_state"


def test_lock_probe_oserror_is_explicit_unknown(tmp_path, fake_script, monkeypatch):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    lock = Path(data["lock_path"]); lock.parent.mkdir(parents=True); lock.touch()
    log = Path(row["log_path"]); log.parent.mkdir(parents=True); log.write_text("interrupted\n")
    loaded, manifest_hash = runner_engine.load_manifest(path)

    def probe_error(*_args):
        raise OSError("synthetic lock probe failure")

    monkeypatch.setattr(runner_engine.fcntl, "flock", probe_error)
    status = runner_engine.read_only("status", loaded, manifest_hash)
    assert status["lock_held"] is None
    assert status["lock_probe_error"] == "OSError: synthetic lock probe failure"
    assert status["rows"][0]["status"] == "held_lock_active_or_unknown"


def test_receipt_byte_binds_outputs_and_control_artifacts(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script); path, _ = manifest(tmp_path, fake_script, [row]); assert invoke(path).returncode == 0
    data = receipt(tmp_path, row); assert all(set(item) == {"path", "size_bytes", "sha256"} for item in data["primary_outputs"])
    for path_to_mutate in [row["expected_output_paths"][0], row["expected_output_paths"][1], data["artifacts"]["selected_wcs_snapshot"]["path"], data["artifacts"]["scalings_snapshot"]["path"]]:
        Path(path_to_mutate).write_text("tampered\n")
        assert result(invoke(path, "--status").stdout)["rows"][0]["status"] == "invalid_receipt"
        Path(path_to_mutate).write_text("synthetic " + row["input_pkl"] + "\n" if str(path_to_mutate).endswith((".txt", ".root")) and "snapshots" not in str(path_to_mutate) else "selected\n" if str(path_to_mutate).endswith("selectedWCs.txt") else "scalings\n")
    Path(row["expected_output_paths"][0]).unlink()
    assert result(invoke(path, "--status").stdout)["rows"][0]["status"] == "invalid_receipt"


def test_resume_failures_and_new_attempt(tmp_path, fake_script):
    counter = tmp_path / "counter"; row = make_row(tmp_path, fake_script, extra=["--counter", str(counter)]); path, _ = manifest(tmp_path, fake_script, [row])
    assert invoke(path).returncode == 0 and invoke(path).returncode == 0 and counter.read_text() == "1"
    bad = make_row(tmp_path, fake_script, "bad", extra=["--fail"]); later = make_row(tmp_path, fake_script, "later", extra=["--counter", str(tmp_path / "later")]); bad_path, _ = manifest(tmp_path, fake_script, [bad, later], "bad.json")
    assert result(invoke(bad_path).stderr)["status"] == "row_command_failed" and not (tmp_path / "later").exists()
    bad_attempts = list(runner_engine.attempt_root(sha(bad_path), bad).iterdir())
    assert len(bad_attempts) == 1 and (bad_attempts[0] / "producer.log").exists()
    assert result(invoke(bad_path, "--status").stdout)["rows"][0]["status"] == "partial_staging_attempt"
    assert not (tmp_path / "control" / "runner_owner.json").exists()
    retry = make_row(tmp_path, fake_script, "bad", "attempt_02"); retry_path, _ = manifest(tmp_path, fake_script, [retry], "retry.json"); assert invoke(retry_path).returncode == 0
    orphan = make_row(tmp_path, fake_script, "orphan"); Path(orphan["expected_output_paths"][0]).parent.mkdir(parents=True); Path(orphan["expected_output_paths"][0]).write_text("orphan\n"); orphan_path, _ = manifest(tmp_path, fake_script, [orphan], "orphan.json")
    assert result(invoke(orphan_path).stderr)["status"] == "canonical_collision_or_ambiguous_state"


def test_partial_attempt_is_preserved_and_explicit_rerun_uses_new_namespace(tmp_path, fake_script):
    marker = tmp_path / "failed-once"
    row = make_row(tmp_path, fake_script, extra=["--partial-once", str(marker)])
    path, _ = manifest(tmp_path, fake_script, [row])
    assert result(invoke(path).stderr)["status"] == "row_command_failed"
    attempt_root = runner_engine.attempt_root(sha(path), row)
    first = attempt_root / "attempt_000001"
    assert (first / "cards" / "card.txt").exists()
    assert not Path(row["output_root"]).exists()
    assert result(invoke(path, "--status").stdout)["rows"][0]["status"] == "partial_staging_attempt"
    assert invoke(path).returncode == 0
    assert (attempt_root / "attempt_000002" / "intent.json").exists()
    assert (first / "cards" / "card.txt").exists()
    assert receipt(tmp_path, row)["physical_attempt_id"] == "attempt_000002"


def test_validated_staging_reconciles_without_reexecution(tmp_path, fake_script, monkeypatch):
    counter = tmp_path / "counter"
    row = make_row(tmp_path, fake_script, extra=["--counter", str(counter)])
    path, data = manifest(tmp_path, fake_script, [row])
    loaded, manifest_hash = runner_engine.load_manifest(path)
    original = runner_engine.reconcile_row
    with monkeypatch.context() as patcher:
        patcher.setattr(runner_engine, "reconcile_row", lambda *_: (_ for _ in ()).throw(RuntimeError("crash before promotion")))
        with pytest.raises(RuntimeError, match="crash before promotion"):
            runner_engine.execute(loaded, manifest_hash)
    assert result(invoke(path, "--status").stdout)["rows"][0]["status"] == "complete_validated_staging_not_promoted"
    assert not Path(row["output_root"]).exists()
    assert original is runner_engine.reconcile_row
    assert invoke(path).returncode == 0
    assert counter.read_text() == "1"
    assert result(invoke(path, "--status").stdout)["rows"][0]["status"] == "complete"
    assert Path(data["control_root"], "receipts", f"{row['row_id']}__{row['attempt_id']}.json").exists()


def test_validated_staging_tamper_blocks_promotion(tmp_path, fake_script, monkeypatch):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    loaded, manifest_hash = runner_engine.load_manifest(path)
    with monkeypatch.context() as patcher:
        patcher.setattr(runner_engine, "reconcile_row", lambda *_: (_ for _ in ()).throw(RuntimeError("crash before promotion")))
        with pytest.raises(RuntimeError, match="crash before promotion"):
            runner_engine.execute(loaded, manifest_hash)
    staged = runner_engine.attempt_root(manifest_hash, row) / "attempt_000001" / "cards" / "card.txt"
    staged.write_text("tampered\n")
    assert result(invoke(path, "--status").stdout)["rows"][0]["status"] == "canonical_collision_or_ambiguous_state"
    assert result(invoke(path).stderr)["status"] == "canonical_collision_or_ambiguous_state"
    assert not Path(row["output_root"]).exists()
    assert not runner_engine.receipt_path(data, row).exists()


def test_promoted_without_receipt_reconciles_exact_bytes(tmp_path, fake_script, monkeypatch):
    counter = tmp_path / "counter"
    row = make_row(tmp_path, fake_script, extra=["--counter", str(counter)])
    path, data = manifest(tmp_path, fake_script, [row])
    loaded, manifest_hash = runner_engine.load_manifest(path)
    final = runner_engine.receipt_path(data, row)
    original = runner_engine.atomic_json

    def crash_at_receipt(payload, destination, overwrite):
        if destination == final:
            raise RuntimeError("crash before receipt")
        return original(payload, destination, overwrite)

    with monkeypatch.context() as patcher:
        patcher.setattr(runner_engine, "atomic_json", crash_at_receipt)
        with pytest.raises(RuntimeError, match="crash before receipt"):
            runner_engine.execute(loaded, manifest_hash)
    assert result(invoke(path, "--status").stdout)["rows"][0]["status"] == "promoted_complete_without_receipt"
    assert not final.exists()
    assert invoke(path).returncode == 0
    assert counter.read_text() == "1"
    assert final.exists()


def test_incomplete_marked_publication_reconciles_without_reexecution(tmp_path, fake_script, monkeypatch):
    counter = tmp_path / "counter"
    row = make_row(tmp_path, fake_script, extra=["--counter", str(counter)])
    path, data = manifest(tmp_path, fake_script, [row])
    loaded, manifest_hash = runner_engine.load_manifest(path)
    first_output = Path(row["expected_output_paths"][0])
    original = runner_engine.publish_matching

    def crash_after_first_output(source, destination, source_record):
        if destination != first_output and first_output.exists():
            raise RuntimeError("crash during publication")
        return original(source, destination, source_record)

    with monkeypatch.context() as patcher:
        patcher.setattr(runner_engine, "publish_matching", crash_after_first_output)
        with pytest.raises(RuntimeError, match="crash during publication"):
            runner_engine.execute(loaded, manifest_hash)
    assert result(invoke(path, "--status").stdout)["rows"][0]["status"] == "promoted_incomplete"
    assert invoke(path).returncode == 0
    assert counter.read_text() == "1"
    assert runner_engine.receipt_path(data, row).exists()


def test_unmarked_canonical_directory_is_ambiguous(tmp_path, fake_script, monkeypatch):
    row = make_row(tmp_path, fake_script)
    path, _ = manifest(tmp_path, fake_script, [row])
    loaded, manifest_hash = runner_engine.load_manifest(path)
    original = runner_engine.os.link

    def crash_before_marker(source, destination):
        if str(destination).endswith(runner_engine.INTENT_MARKER):
            raise RuntimeError("crash before marker")
        return original(source, destination)

    with monkeypatch.context() as patcher:
        patcher.setattr(runner_engine.os, "link", crash_before_marker)
        with pytest.raises(RuntimeError, match="crash before marker"):
            runner_engine.execute(loaded, manifest_hash)
    assert Path(row["output_root"]).is_dir()
    assert result(invoke(path, "--status").stdout)["rows"][0]["status"] == "canonical_collision_or_ambiguous_state"
    assert result(invoke(path).stderr)["status"] == "canonical_collision_or_ambiguous_state"


def test_promoted_mismatch_fails_closed(tmp_path, fake_script, monkeypatch):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    loaded, manifest_hash = runner_engine.load_manifest(path)
    final = runner_engine.receipt_path(data, row)
    original = runner_engine.atomic_json
    with monkeypatch.context() as patcher:
        patcher.setattr(runner_engine, "atomic_json", lambda payload, destination, overwrite: (_ for _ in ()).throw(RuntimeError("receipt crash")) if destination == final else original(payload, destination, overwrite))
        with pytest.raises(RuntimeError, match="receipt crash"):
            runner_engine.execute(loaded, manifest_hash)
    Path(row["expected_output_paths"][0]).write_text("tampered\n")
    assert result(invoke(path, "--status").stdout)["rows"][0]["status"] == "canonical_collision_or_ambiguous_state"
    assert result(invoke(path).stderr)["status"] == "canonical_collision_or_ambiguous_state"
    assert not final.exists()


def test_status_filesystem_read_error_fails_closed(tmp_path, fake_script, monkeypatch):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    assert invoke(path).returncode == 0
    runner_engine.receipt_path(data, row).unlink()
    loaded, manifest_hash = runner_engine.load_manifest(path)
    with monkeypatch.context() as patcher:
        patcher.setattr(runner_engine.os.path, "samefile", lambda *_: (_ for _ in ()).throw(OSError("synthetic NFS read error")))
        status = runner_engine.read_only("status", loaded, manifest_hash)
    assert status["rows"][0]["status"] == "canonical_collision_or_ambiguous_state"
    assert "synthetic NFS read error" in status["rows"][0]["reason"]


def test_stale_owner_with_free_lock_is_not_active(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    control = Path(data["control_root"])
    control.mkdir()
    Path(data["lock_path"]).touch()
    (control / "runner_owner.json").write_text(json.dumps({"schema_version": runner_engine.OWNER_SCHEMA, "manifest_sha256": sha(path), "current_row_id": row["row_id"], "current_attempt_id": row["attempt_id"]}))
    assert result(invoke(path, "--status").stdout)["rows"][0]["status"] == "stale_owner_free_lock"
    assert invoke(path).returncode == 0
    assert result(invoke(path, "--status").stdout)["rows"][0]["status"] == "complete"


def test_current_v3_receipt_without_physical_fields_remains_valid(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    assert invoke(path).returncode == 0
    final = runner_engine.receipt_path(data, row)
    payload = json.loads(final.read_text())
    del payload["physical_attempt_id"]
    del payload["execution_argv"]
    final.write_text(json.dumps(payload))
    assert result(invoke(path, "--status").stdout)["rows"][0]["status"] == "complete"
    assert result(invoke(path).stdout)["rows"][0]["action"] == "skipped_valid_receipt"


def test_malformed_receipt_has_no_authority(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    final = runner_engine.receipt_path(data, row)
    final.parent.mkdir(parents=True)
    final.write_text("{broken")
    assert result(invoke(path, "--status").stdout)["rows"][0]["status"] == "invalid_receipt"
    assert result(invoke(path).stderr)["status"] == "invalid_receipt"
    assert not Path(row["output_root"]).exists()


def test_held_lock_with_unmatched_owner_and_attempt_evidence_is_unknown(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    root = runner_engine.attempt_root(sha(path), row)
    (root / "attempt_000001").mkdir(parents=True)
    lock = Path(data["lock_path"])
    lock.parent.mkdir(parents=True)
    with lock.open("a+b") as handle:
        runner_engine.fcntl.flock(handle.fileno(), runner_engine.fcntl.LOCK_EX | runner_engine.fcntl.LOCK_NB)
        state = result(invoke(path, "--status").stdout)
        assert state["rows"][0]["status"] == "held_lock_active_or_unknown"
        assert result(invoke(path).stderr)["status"] == "execution_lock_held"


def test_unexpected_exception_cleans_active_owner(tmp_path, fake_script, monkeypatch):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    loaded, manifest_hash = runner_engine.load_manifest(path)

    def unexpected(*_args, **_kwargs):
        raise RuntimeError("synthetic unexpected failure")

    monkeypatch.setattr(runner_engine, "execute_row", unexpected)
    with pytest.raises(RuntimeError, match="synthetic unexpected failure"):
        runner_engine.execute(loaded, manifest_hash)
    assert not (Path(data["control_root"]) / "runner_owner.json").exists()


def test_completed_row_skips_unavailable_historical_inputs_before_next_row(tmp_path, fake_script):
    counter = tmp_path / "second-counter"
    first = make_row(tmp_path, fake_script, "first", extra=["--mutate-path", str(fake_script)])
    second = make_row(tmp_path, fake_script, "second", extra=["--counter", str(counter)])
    path, _ = manifest(tmp_path, fake_script, [first, second], "resume.json")
    assert result(invoke(path).stderr)["status"] == "runtime_contract_mismatch"
    assert receipt(tmp_path, first)["row_id"] == "first"
    fake_script.write_text(FAKE)
    Path(first["input_pkl"]).unlink()
    Path(first["missing_parton_path"]).unlink()
    resumed = invoke(path)
    resumed_result = result(resumed.stdout)
    assert resumed.returncode == 0
    assert resumed_result["rows"][0]["action"] == "skipped_valid_receipt"
    assert resumed_result["rows"][1]["action"] == "executed"
    assert counter.read_text() == "1"


def test_missing_parton_binding_and_execution_availability(tmp_path, fake_script):
    missing = make_row(tmp_path, fake_script, "missing")
    Path(missing["missing_parton_path"]).unlink()
    path, _ = manifest(tmp_path, fake_script, [missing], "missing.json")
    missing_result = invoke(path)
    assert result(missing_result.stderr)["status"] == "runtime_preflight_error"
    assert not Path(missing["log_path"]).exists()

    nonregular = make_row(tmp_path, fake_script, "nonregular")
    nonregular_path = tmp_path / "nonregular-missing-parton"
    nonregular_path.mkdir()
    nonregular["missing_parton_path"] = str(nonregular_path)
    replace_option_value(nonregular, "--miss-parton-file", nonregular_path)
    path, _ = manifest(tmp_path, fake_script, [nonregular], "nonregular.json")
    nonregular_result = invoke(path)
    assert result(nonregular_result.stderr)["status"] == "runtime_preflight_error"
    assert not Path(nonregular["log_path"]).exists()

    valid = make_row(tmp_path, fake_script, "valid")
    path, _ = manifest(tmp_path, fake_script, [valid], "valid.json")
    assert invoke(path).returncode == 0

    relative = make_row(tmp_path, fake_script, "relative")
    relative["missing_parton_path"] = "relative.root"
    replace_option_value(relative, "--miss-parton-file", "relative.root")
    path, _ = manifest(tmp_path, fake_script, [relative], "relative.json")
    assert result(invoke(path, "--plan-only").stderr)["status"] == "manifest_schema_error"

    mismatch = make_row(tmp_path, fake_script, "mismatch")
    other = tmp_path / "inputs" / "other-missing.root"; other.write_text("other\n")
    mismatch["missing_parton_path"] = str(other)
    path, _ = manifest(tmp_path, fake_script, [mismatch], "mismatch.json")
    assert result(invoke(path, "--plan-only").stderr)["status"] == "manifest_schema_error"

    duplicate = make_row(tmp_path, fake_script, "duplicate")
    duplicate["producer_args"].extend(["--miss-parton-file", duplicate["missing_parton_path"]])
    path, _ = manifest(tmp_path, fake_script, [duplicate], "duplicate-option.json")
    assert result(invoke(path, "--plan-only").stderr)["status"] == "manifest_schema_error"


def test_runtime_contract_blocks_before_and_between_rows(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script); path, data = manifest(tmp_path, fake_script, [row]); fake_script.write_text("changed\n")
    assert result(invoke(path).stderr)["status"] == "runtime_contract_mismatch" and not Path(row["log_path"]).exists()
    fake_script.write_text(FAKE); first = make_row(tmp_path, fake_script, "first", extra=["--mutate-path", str(fake_script)]); second = make_row(tmp_path, fake_script, "second", extra=["--counter", str(tmp_path / "second")]); path, _ = manifest(tmp_path, fake_script, [first, second], "drift.json")
    assert result(invoke(path).stderr)["status"] == "runtime_contract_mismatch" and not (tmp_path / "second").exists()


def test_fresh_classification_blocks_future_external_output(tmp_path, fake_script):
    future = make_row(tmp_path, fake_script, "future", extra=["--counter", str(tmp_path / "future")])
    first = make_row(tmp_path, fake_script, "first", extra=["--write-path", future["expected_output_paths"][0]])
    path, _ = manifest(tmp_path, fake_script, [first, future])
    assert result(invoke(path).stderr)["status"] == "canonical_collision_or_ambiguous_state" and not (tmp_path / "future").exists()


def test_input_binding_literal_argv_and_direct_execution(tmp_path, fake_script):
    record_argv = tmp_path / "argv.json"; channel = "channel;touch should_not_exist"
    row = make_row(tmp_path, fake_script, extra=["--record-argv", str(record_argv)])
    row["physical_channels"] = [channel]; start = row["producer_args"].index("--ch-lst") + 1; end = row["producer_args"].index("--binning"); row["producer_args"][start:end] = [channel]
    path, _ = manifest(tmp_path, fake_script, [row]); assert invoke(path).returncode == 0
    argv = json.loads(record_argv.read_text()); assert argv[0] == row["input_pkl"] and argv[argv.index("--ch-lst") + 1] == channel
    source = ENGINE.read_text(); assert "subprocess.run(" in source and "shell=True" not in source and "codex-run.sh" not in source
    assert "codex-run.sh" not in RUNNER.read_text() and "/bin/bash --noprofile" not in RUNNER.read_text()
    row = make_row(tmp_path, fake_script, "negative"); original = row["input_pkl"]; row["input_pkl"] = str(tmp_path / "missing-input.pkl"); row["producer_args"].extend(["--unrelated", original]); path, _ = manifest(tmp_path, fake_script, [row], "negative.json")
    assert result(invoke(path).stderr)["status"] == "runtime_preflight_error"


def test_lock_blocks_second_owner_and_no_finalizer_behavior(tmp_path, fake_script):
    row = make_row(tmp_path, fake_script, extra=["--sleep", "1"]); path, _ = manifest(tmp_path, fake_script, [row]); first = subprocess.Popen([str(RUNNER), str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 10
    while not list(runner_engine.attempt_root(sha(path), row).glob("attempt_*/producer.log")) and time.monotonic() < deadline: time.sleep(.02)
    assert result(invoke(path).stderr)["status"] == "execution_lock_held"
    first.communicate(timeout=10); assert first.returncode == 0
    source = ENGINE.read_text(); assert "datacards_post_processing" not in source and '"scalings.json"' not in source


def test_owner_lifetime_is_per_unit(tmp_path, fake_script):
    first = make_row(tmp_path, fake_script, "first", extra=["--sleep", ".4"])
    second = make_row(tmp_path, fake_script, "second", extra=["--sleep", ".4"])
    path, data = manifest(tmp_path, fake_script, [first, second], "owner.json")
    process = subprocess.Popen([str(RUNNER), str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    owner_path = runner_engine.unit_owner_path(first)
    first_owner = wait_for_owner(owner_path, "first")
    second_owner = wait_for_owner(runner_engine.unit_owner_path(second), "second")
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, stdout + stderr
    assert first_owner["started_at"] != second_owner["started_at"]
    assert first_owner["current_row_started_at"] != second_owner["current_row_started_at"]
    assert not owner_path.exists()
