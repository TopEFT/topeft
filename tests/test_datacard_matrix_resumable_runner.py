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
if "--sleep" in args: time.sleep(float(one("--sleep")))
if "--fail" in args: raise SystemExit(7)
root = Path(one("--out-dir")); root.mkdir(parents=True, exist_ok=True)
report = Path(one("--merge-report")); report.parent.mkdir(parents=True, exist_ok=True); report.write_text("{}\n")
(root / "selectedWCs.txt").write_text("selected\n")
(root / "scalings-preselect.json").write_text("scalings\n")
for name in many("--expected-output"):
    p = Path(name); p.parent.mkdir(parents=True, exist_ok=True); p.write_text("synthetic " + input_pkl + "\n")
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
    assert stale["lock_held"] is False and stale["rows"][0]["status"] == "interrupted_requires_external_reconciliation"
    with lock.open("a+b") as handle:
        runner_engine.fcntl.flock(handle.fileno(), runner_engine.fcntl.LOCK_EX | runner_engine.fcntl.LOCK_NB)
        active = result(invoke(path, "--status").stdout)
        assert active["lock_held"] is True and active["lock_probe_error"] is None and active["rows"][0]["status"] == "active"
        wrong_owner = {**stale_owner, "current_row_id": "different_row"}
        (tmp_path / "control" / "runner_owner.json").write_text(json.dumps(wrong_owner))
        mismatched = result(invoke(path, "--status").stdout)
        assert mismatched["lock_held"] is True and mismatched["rows"][0]["status"] == "interrupted_requires_external_reconciliation"
    (tmp_path / "control" / "runner_owner.json").write_text(json.dumps(stale_owner))
    final = result(invoke(path, "--status").stdout)
    assert final["lock_held"] is False and final["rows"][0]["status"] == "interrupted_requires_external_reconciliation"


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
    assert status["rows"][0]["status"] == "interrupted_requires_external_reconciliation"
    assert status["rows"][0]["reason"].startswith("lock state is unknown")


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
    assert Path(bad["log_path"]).exists()
    assert not (tmp_path / "control" / "runner_owner.json").exists()
    retry = make_row(tmp_path, fake_script, "bad", "attempt_02"); retry_path, _ = manifest(tmp_path, fake_script, [retry], "retry.json"); assert invoke(retry_path).returncode == 0
    orphan = make_row(tmp_path, fake_script, "orphan"); Path(orphan["expected_output_paths"][0]).parent.mkdir(parents=True); Path(orphan["expected_output_paths"][0]).write_text("orphan\n"); orphan_path, _ = manifest(tmp_path, fake_script, [orphan], "orphan.json")
    assert result(invoke(orphan_path).stderr)["status"] == "interrupted_requires_external_reconciliation"


def test_unexpected_exception_cleans_active_owner(tmp_path, fake_script, monkeypatch):
    row = make_row(tmp_path, fake_script)
    path, data = manifest(tmp_path, fake_script, [row])
    loaded, manifest_hash = runner_engine.load_manifest(path)

    def unexpected(*_args):
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
    assert result(invoke(path).stderr)["status"] == "interrupted_requires_external_reconciliation" and not (tmp_path / "future").exists()


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
    while not Path(row["log_path"]).exists() and time.monotonic() < deadline: time.sleep(.02)
    assert result(invoke(path).stderr)["status"] == "execution_lock_held"
    first.communicate(timeout=10); assert first.returncode == 0
    source = ENGINE.read_text(); assert "datacards_post_processing" not in source and '"scalings.json"' not in source


def test_owner_started_at_is_stable_across_row_updates(tmp_path, fake_script):
    first = make_row(tmp_path, fake_script, "first", extra=["--sleep", ".4"])
    second = make_row(tmp_path, fake_script, "second", extra=["--sleep", ".4"])
    path, data = manifest(tmp_path, fake_script, [first, second], "owner.json")
    process = subprocess.Popen([str(RUNNER), str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    owner_path = Path(data["control_root"]) / "runner_owner.json"
    first_owner = wait_for_owner(owner_path, "first")
    second_owner = wait_for_owner(owner_path, "second")
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, stdout + stderr
    assert first_owner["started_at"] == second_owner["started_at"]
    assert first_owner["current_row_started_at"] != second_owner["current_row_started_at"]
    assert not owner_path.exists()
