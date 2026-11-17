"""Synthetic launcher handoff checks without a producer or scheduler."""

import hashlib
import copy
import fcntl
import importlib.util
import json
import os
import shutil
import select
import shlex
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager

import pytest

from analysis.topeft_run2 import campaign_manifest as manifest
from analysis.topeft_run2 import wq_staging_guard as staging_guard


RUN_DIRECTORY = Path(__file__).resolve().parents[1] / "analysis" / "topeft_run2"
RUN_CR = RUN_DIRECTORY / "run_cr.sh"
RUN_ANALYSIS = RUN_DIRECTORY / "run_analysis.py"
ARCHIVE = RUN_DIRECTORY / "topeft-envs" / "env_spec_1707ade2a411bb14.tar.gz"


def test_native_tree_reduction_interface_accepts_run2_t50():
    """Execute the actual argument declaration without launching a manager."""
    import argparse
    import ast
    tree = ast.parse(RUN_ANALYSIS.read_text())
    declaration = next(n for n in ast.walk(tree) if isinstance(n, ast.Expr)
                       and isinstance(n.value, ast.Call) and n.value.args
                       and isinstance(n.value.args[0], ast.Constant)
                       and n.value.args[0].value == '--wq-treereduction')
    parser = argparse.ArgumentParser()
    exec(compile(ast.Module(body=[declaration], type_ignores=[]), 'native_treereduction', 'exec'),
         {'parser': parser})
    assert parser.parse_args(['--wq-treereduction', '50']).wq_treereduction == 50
    assert parser.parse_args(['--wq-treereduction', '20']).wq_treereduction == 20
    with pytest.raises(SystemExit):
        parser.parse_args(['--wq-treereduction', '21'])


@pytest.fixture
def backend(tmp_path):
    script = tmp_path / "synthetic_backend.sh"
    script.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
action="$1"
scenario="$2"
shift 2
if [[ "$action" == validate_environment || "$action" == prepare_environment ]]; then
  if [[ -n "${CAMPAIGN_FAIL_ENV_CALL:-}" ]]; then
    counter="$CAMPAIGN_VALIDATION_ROOT/environment_calls"
    count=0
    [[ ! -f "$counter" ]] || read -r count < "$counter"
    count=$((count + 1))
    printf '%s\\n' "$count" > "$counter"
    [[ "$count" != "$CAMPAIGN_FAIL_ENV_CALL" ]] || exit 71
  fi
  archive="${1:-$CAMPAIGN_DEFAULT_ARCHIVE}"
  printf 'env_file: %s\\n' "$archive"
  printf 'env_file_sha256: %s\\n' "$(sha256sum "$archive" | cut -d ' ' -f1)"
  printf 'environment_fingerprint: %064d\\n' 9
  printf 'environment_validation_status: valid\\n'
  if [[ "${CAMPAIGN_OMIT_TOPCOFFEA_PROVENANCE:-false}" != true ]]; then
    printf 'topcoffea_git_commit: %040d\\n' "${CAMPAIGN_TOPCOFFEA_GIT_COMMIT:-8}"
    printf 'topcoffea_relevant_source_fingerprint: %064d\\n' "${CAMPAIGN_TOPCOFFEA_SOURCE_FINGERPRINT:-7}"
  fi
  exit 0
fi
if [[ "$action" == run_block ]]; then
  unit="$1"; archive="$2"; stage="$3"; source="$4"; downstream="$5"; logs="$6"
  printf '%q ' "$@" >> "$CAMPAIGN_VALIDATION_ROOT/source_argv.log"
  printf '\\n' >> "$CAMPAIGN_VALIDATION_ROOT/source_argv.log"
  printf '%s\\n' "$unit" >> "$CAMPAIGN_VALIDATION_ROOT/source_order.log"
  if [[ -n "${CAMPAIGN_EXPECT_LOCK_HELD:-}" ]]; then
    if flock -n "$CAMPAIGN_EXPECT_LOCK_HELD" true; then exit 72; fi
  fi
  if [[ -n "${CAMPAIGN_EXPECT_LOCK_FREE:-}" ]]; then
    flock -n "$CAMPAIGN_EXPECT_LOCK_FREE" true || exit 72
  fi
  if [[ "$unit" == "${CAMPAIGN_GATE_UNIT:-}" ]]; then
    printf '%s\\n' "$BASHPID" > "$CAMPAIGN_GATE_ROOT/ready"
    read -r release < "$CAMPAIGN_GATE_ROOT/release"
  fi
  mkdir -p "$logs"
  if [[ "$unit" == "${CAMPAIGN_FAIL_UNIT:-}" \
    || "$scenario" == source_fail_first && "$unit" == run2_prod_a \
    || "$scenario" == source_fail_run3_tau && "$unit" == run3_prod_2los_1tau_a \
    || "$scenario" == source_fail_both_tau ]]; then
    exit 23
  fi
  printf 'synthetic source\\n' > "$source"
  artifact="$source"
  source_sha=""
elif [[ "$action" == run_nonprompt ]]; then
  unit="$1"; source="$2"; downstream="$3"
  printf '%q ' "$@" >> "$CAMPAIGN_VALIDATION_ROOT/dd_argv.log"
  printf '\\n' >> "$CAMPAIGN_VALIDATION_ROOT/dd_argv.log"
  if [[ "$scenario" == downstream_fail_once && ! -e "$CAMPAIGN_VALIDATION_ROOT/failed_once" ]]; then
    : > "$CAMPAIGN_VALIDATION_ROOT/failed_once"
    exit 137
  fi
  printf 'synthetic downstream\\n' > "$downstream"
  artifact="$downstream"
  source_sha="$(sha256sum "$source" | cut -d ' ' -f1)"
else
  exit 64
fi
if [[ -n "$source_sha" ]]; then
  inputs="{\\"pkl_sha256\\":\\"$source_sha\\"}"
else
  inputs=""
fi
printf '{"artifact":{"pkl_basename":"%s","pkl_size_bytes":%s,"pkl_sha256":"%s"},"lineage":{"inputs":[%s]}}\\n' \
  "$(basename "$artifact")" "$(stat -c %s "$artifact")" \
  "$(sha256sum "$artifact" | cut -d ' ' -f1)" "$inputs" > "$artifact.metadata.json"
""",
        encoding="utf-8",
    )
    script.chmod(0o700)
    environment = os.environ.copy()
    environment.update({
        "CAMPAIGN_VALIDATION_BACKEND": str(script),
        "CAMPAIGN_VALIDATION_ROOT": str(tmp_path),
        "CAMPAIGN_DEFAULT_ARCHIVE": str(ARCHIVE),
        "CAMPAIGN_VALIDATION_SCENARIO": "success",
        "TMPDIR": str(tmp_path),
    })
    return environment


def launch(tmp_path, backend, *extra, profile="run2_prod", archive=ARCHIVE):
    command = [str(RUN_CR), "--production-profile", profile,
               "--output-dir", str(tmp_path / "output"), "--campaign-tag", "synthetic"]
    if archive is not None:
        command.extend(("--env-file", str(archive)))
    command.extend(extra)
    def snapshot():
        paths = sorted((tmp_path / "output").glob("**/*_campaign_state_v9.json"))
        return {str(path.relative_to(tmp_path)): json.loads(path.read_text()) for path in paths}

    before = snapshot()
    result = subprocess.run(
        command,
        cwd=RUN_DIRECTORY, env=backend, capture_output=True, text=True, check=False,
    )
    evidence = tmp_path / f"invocation_{len(list(tmp_path.glob('invocation_*.json'))):03d}.json"
    evidence.write_text(json.dumps({"command": command, "returncode": result.returncode,
                                   "stdout": result.stdout, "stderr": result.stderr,
                                   "before": before, "after": snapshot()}, indent=2))
    return result


def state_path(tmp_path, profile="run2_prod"):
    return tmp_path / "output" / f".{profile}_campaign_state_v9.json"


@contextmanager
def gated_process(tmp_path, environment, command, name):
    """A private FIFO gate; always release and join before fixture teardown."""
    gate_root = tmp_path / name
    gate_root.mkdir()
    for fifo in ("ready", "release"):
        os.mkfifo(gate_root / fifo, 0o600)
    ready_fd = os.open(gate_root / "ready", os.O_RDWR | os.O_NONBLOCK)
    release_fd = os.open(gate_root / "release", os.O_RDWR)
    environment = dict(environment, CAMPAIGN_GATE_ROOT=str(gate_root))
    with (gate_root / "stdout").open("w") as stdout, (gate_root / "stderr").open("w") as stderr:
        process = subprocess.Popen(command, cwd=RUN_DIRECTORY, env=environment,
                                   stdout=stdout, stderr=stderr)
        try:
            assert select.select([ready_fd], [], [], 300)[0], gate_root
            child_pid = int(os.read(ready_fd, 128).strip())
            yield process, child_pid, gate_root
        finally:
            os.write(release_fd, b"release\n")
            try:
                process.wait(timeout=300)
            finally:
                os.close(ready_fd)
                os.close(release_fd)


def lock_is_free(path):
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def process_fd_targets(pid):
    targets = set()
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            targets.add(os.readlink(fd))
        except FileNotFoundError:
            pass
    return targets


def summary_shell(tmp_path, extra="", invocation="publish_combined_summary"):
    """Execute the actual maintained helper, including inside an if context."""
    source = RUN_CR.read_text()
    definition = "publish_combined_summary() (" + source.split(
        "publish_combined_summary() (", 1
    )[1].split("\n)\n", 1)[0] + "\n)\n"
    values = {"script_dir": str(RUN_DIRECTORY), "output_dir": str(tmp_path),
              "production_profile": "run2_run3_prod", "campaign_tag": "synthetic",
              "run2_profile": "run2_prod", "run3_profile": "run3_prod",
              "run2_exit": "0", "run3_exit": "0"}
    for scope in ("run2", "run3"):
        root = tmp_path / scope
        root.mkdir(exist_ok=True)
        values[f"{scope}_root"] = str(root)
        values[f"{scope}_state"] = str(root / f".{scope}_prod_campaign_state_v9.json")
    setup = "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items())
    return ["/bin/bash", "--noprofile", "--norc", "-c",
            "set -euo pipefail\n" + setup + "\n" + definition + extra + "\n" + invocation]


def component_lock(tmp_path, scope):
    return tmp_path / scope / f".{scope}_prod_campaign.lock"


def test_nominal_manager_name_default_and_exact_override(monkeypatch):
    monkeypatch.syspath_prepend(str(RUN_DIRECTORY))
    monkeypatch.setenv("USER", "synthetic_user")
    spec = importlib.util.spec_from_file_location("run_analysis_manager_name_test", RUN_ANALYSIS)
    run_analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_analysis)

    assert run_analysis._resolve_wq_manager_name(None, "work_queue", "sample") == (
        "synthetic_user-workqueue-sample"
    )
    assert run_analysis._resolve_wq_manager_name(
        "top26006-test-manager", "work_queue", "sample"
    ) == "top26006-test-manager"
    assert run_analysis._resolve_wq_manager_name("name.with_ordinary-chars", "work_queue", "sample") == (
        "name.with_ordinary-chars"
    )
    with pytest.raises(ValueError, match="non-empty"):
        run_analysis._resolve_wq_manager_name("", "work_queue", "sample")
    with pytest.raises(ValueError, match="work_queue executor"):
        run_analysis._resolve_wq_manager_name("exact-name", "futures", "sample")
    with pytest.raises(ValueError, match="work_queue executor"):
        run_analysis._resolve_wq_manager_name("exact-name", "taskvine", "sample")


@pytest.mark.parametrize(("executor_name", "manager_name", "expected_error"), (
    ("futures", "exact-name", "requires the work_queue executor"),
    ("taskvine", "exact-name", "requires the work_queue executor"),
    ("work_queue", "", "requires a non-empty name"),
))
def test_nominal_manager_name_cli_rejects_invalid_use_before_input_read(
    executor_name, manager_name, expected_error
):
    wrap_path = RUN_DIRECTORY.parents[2] / "codex-run.sh"
    result = subprocess.run(
        [str(wrap_path), sys.executable, str(RUN_ANALYSIS), "--executor", executor_name,
         "--wq-manager-name", manager_name],
        cwd=RUN_DIRECTORY, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert expected_error in result.stderr


def test_selected_fragment_persists_complete_plan_and_receipts(tmp_path, backend):
    result = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a")
    assert result.returncode == 0, result.stdout + result.stderr
    state = manifest.load_state(state_path(tmp_path))
    assert state["schema_version"] == 9
    assert len(state["units"]) == 5
    assert [manifest.overall_status(unit) for unit in state["units"]] == [
        "success", "planned", "planned", "planned", "planned",
    ]
    unit = state["units"][0]
    assert unit["stages"]["source"]["receipt"]["binding"]["scientific_coverage_sha256"] == state["scientific_identity"]["scientific_coverage_sha256"]
    assert unit["stages"]["data_driven"]["receipt"]["binding"]["source_artifact_sha256"] == unit["stages"]["source"]["receipt"]["artifact"]["sha256"]


def test_capacity_options_keep_plan_owned_output_paths(tmp_path, backend):
    result = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a",
                    "--pkl-min-free-bytes", "2", "--pkl-min-free-inodes", "4",
                    "--pkl-safety-reserve-bytes", "1", "--pkl-safety-reserve-inodes", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    state = manifest.load_state(state_path(tmp_path))
    unit = state["units"][0]
    for stage in ("source", "data_driven"):
        assert unit["stages"][stage]["receipt"]["artifact"]["path"] == unit["outputs"][stage]
        command = unit["stages"][stage]["attempts"][0]["command_argv"]
        assert "--pkl-no-clobber" in command
        assert manifest.output_safety_from_command(command) == (2, 4, 1, 1)


def test_launcher_forwards_staging_without_filesystem_capability(tmp_path, backend):
    login_user = subprocess.run(["id", "-un"], capture_output=True, text=True,
                                check=True).stdout.strip()
    backend["USER"] = "untrusted_environment_name"
    selected_base = f"/tmpscratch/users/{login_user}/selected"
    result = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a",
                    "--wq-staging-base", selected_base,
                    "--wq-free-warning-bytes", "20000000000",
                    "--wq-min-free-bytes", "10000000000",
                    "--wq-poll-seconds", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    command = manifest.load_state(state_path(tmp_path))["units"][0][
        "stages"]["source"]["attempts"][0]["command_argv"]
    assert command[command.index("--wq-filepath") + 1] == selected_base
    assert "--wq-root-backed-local-statvfs" not in command
    assert "--wq-quota-root" not in command
    assert "--pkl-no-clobber" in command


def test_launcher_forwards_arbitrary_selected_staging_path(tmp_path, backend):
    result = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a",
                    "--wq-staging-base", "/synthetic-root-backed/selected",
                    "--wq-free-warning-bytes", "20000000000",
                    "--wq-min-free-bytes", "10000000000",
                    "--wq-poll-seconds", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    command = manifest.load_state(state_path(tmp_path))["units"][0][
        "stages"]["source"]["attempts"][0]["command_argv"]
    assert "--wq-root-backed-local-statvfs" not in command
    assert "--wq-quota-root" not in command


def test_internal_staging_and_output_options_stay_off_public_help(tmp_path):
    launcher_help = subprocess.run([str(RUN_CR), "--help"], cwd=RUN_DIRECTORY,
                                   capture_output=True, text=True, check=True).stdout
    assert "--wq-staging-base" in launcher_help
    assert "--wq-root-backed-local-statvfs" not in launcher_help

    wrap_path = RUN_DIRECTORY.parents[2] / "codex-run.sh"
    command = [str(wrap_path), sys.executable, str(RUN_ANALYSIS)]
    analysis_help = subprocess.run(command + ["--help"], cwd=RUN_DIRECTORY,
                                   capture_output=True, text=True, check=True).stdout
    internal_options = ("--pkl-no-clobber",
                        "--pkl-min-free-bytes", "--pkl-min-free-inodes",
                        "--pkl-safety-reserve-bytes", "--pkl-safety-reserve-inodes",
                        "--pkl-pending-id")
    assert all(option not in analysis_help for option in internal_options)

    ordinary = subprocess.run(command + ["-p", str(tmp_path), "--debug-year-scan"],
                              cwd=RUN_DIRECTORY, capture_output=True, text=True, check=False)
    assert ordinary.returncode == 0, ordinary.stdout + ordinary.stderr
    internal = subprocess.run(command + [
        "-p", str(tmp_path), "--debug-year-scan", "--wq-guard",
        "--pkl-no-clobber",
        "--pkl-min-free-bytes", "2", "--pkl-min-free-inodes", "4",
        "--pkl-safety-reserve-bytes", "1", "--pkl-safety-reserve-inodes", "1",
        "--pkl-pending-id", "synthetic_attempt",
    ], cwd=RUN_DIRECTORY, capture_output=True, text=True, check=False)
    assert internal.returncode == 0, internal.stdout + internal.stderr


def test_resume_accepts_chunksize_and_relocated_identical_archive(tmp_path, backend):
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "source_fail_first"
    first = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a")
    assert first.returncode != 0
    original = manifest.load_state(state_path(tmp_path))
    original_locator = original["provenance"]["initial"]["environment_file"]
    relocated = tmp_path / "relocated.tar.gz"
    shutil.copyfile(ARCHIVE, relocated)
    assert hashlib.sha256(relocated.read_bytes()).hexdigest() == hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "success"
    backend["CAMPAIGN_OMIT_TOPCOFFEA_PROVENANCE"] = "true"
    second = launch(tmp_path, backend, "--resume", "--selected-fragment", "run2_prod_a",
                    "--chunksize", "98765", archive=relocated)
    assert second.returncode == 0, second.stdout + second.stderr
    state = manifest.load_state(state_path(tmp_path))
    attempts = state["units"][0]["stages"]["source"]["attempts"]
    assert [attempt["number"] for attempt in attempts] == [1, 2]
    assert attempts[1]["provenance"]["environment_file"] == str(relocated)
    assert state["provenance"]["initial"]["environment_file"] == original_locator
    assert state["provenance"]["last_resolved"]["environment_file"] == str(relocated)
    assert state["provenance"]["initial"]["environment_file"] != str(relocated)
    assert state["provenance"]["last_resolved"]["topcoffea_git_commit"] == ""
    assert state["provenance"]["last_resolved"]["topcoffea_relevant_source_fingerprint"] == ""


def test_resume_does_not_authorize_from_live_git_or_manager_source_provenance(tmp_path, backend):
    first = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a")
    assert first.returncode == 0, first.stdout + first.stderr
    state = manifest.load_state(state_path(tmp_path))
    provenance = dict(state["provenance"]["last_resolved"])
    provenance.update({
        "topeft_git_commit": "f" * 40,
        "topcoffea_git_commit": "e" * 40,
        "topcoffea_relevant_source_fingerprint": "c" * 64,
        "environment_fingerprint": "d" * 64,
    })
    resumed = manifest.validate_resume(
        state, state["campaign_identity"], state["profile_contract"], provenance,
        state["scientific_identity"]["scientific_coverage"],
    )
    assert resumed is state


def test_launcher_dry_resume_accepts_historical_dd_wrapper_binding(tmp_path, backend):
    first = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a")
    assert first.returncode == 0, first.stdout + first.stderr
    path = state_path(tmp_path)
    state = manifest.load_state(path)
    assert state["units"][0]["stages"]["data_driven"]["status"] == "success"
    assert state["units"][1]["stages"]["source"]["status"] == "pending"
    wrapper = "analysis/topeft_run2/run_data_driven.py"
    historical_sha = "724a46b07e1d9d7b17888fa79acd2fddb1a8cd057f9989a12029c516c4538746"
    state["scientific_identity"]["data_driven_stage_scientific_contract"]["scientific_files"][wrapper] = historical_sha
    stage_sha = manifest.canonical_digest(state["scientific_identity"]["data_driven_stage_scientific_contract"])
    state["scientific_identity"]["data_driven_stage_scientific_contract_sha256"] = stage_sha
    state["units"][0]["stages"]["data_driven"]["receipt"]["binding"]["stage_scientific_contract_sha256"] = stage_sha
    path.write_text(json.dumps(state), encoding="utf-8")
    before = path.read_bytes()
    assert manifest.load_state(path)["scientific_identity"]["data_driven_stage_scientific_contract_sha256"] == stage_sha

    resumed = launch(tmp_path, backend, "--resume", "--dry-run", "--selected-fragment", "run2_prod_b")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "data_driven_scientific_contract_incompatible" not in resumed.stderr
    assert "no campaign state, scheduler, or production action was created" in resumed.stdout
    assert path.read_bytes() == before


def test_current_resolved_resume_retains_manager_source_compatibility_gate(tmp_path, backend):
    first = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a", archive=None)
    assert first.returncode == 0, first.stdout + first.stderr
    state = manifest.load_state(state_path(tmp_path))
    assert state["provenance"]["initial"]["environment_mode"] == "current_resolved"
    provenance = dict(state["provenance"]["last_resolved"])
    provenance["topcoffea_relevant_source_fingerprint"] = "c" * 64
    with pytest.raises(ValueError, match="frozen topcoffea source fingerprint differs"):
        manifest.validate_resume(
            state, state["campaign_identity"], state["profile_contract"], provenance,
            state["scientific_identity"]["scientific_coverage"],
        )

    corrupted_mode = copy.deepcopy(state)
    corrupted_mode["provenance"]["initial"]["environment_mode"] = "explicit_snapshot"
    with pytest.raises(ValueError, match="initial environment mode contradicts first attempt provenance"):
        manifest.validate_resume(
            corrupted_mode, state["campaign_identity"], state["profile_contract"], provenance,
            state["scientific_identity"]["scientific_coverage"],
        )

    unsupported_mode = copy.deepcopy(state)
    unsupported_mode["provenance"]["initial"]["environment_mode"] = "unsupported"
    with pytest.raises(ValueError, match="unsupported environment mode"):
        manifest.validate_state(unsupported_mode)

    backend["CAMPAIGN_OMIT_TOPCOFFEA_PROVENANCE"] = "true"
    resume = launch(tmp_path, backend, "--resume", "--selected-fragment", "run2_prod_a")
    assert resume.returncode != 0
    assert "current environment resolution did not return complete TOPCOFFEA provenance" in resume.stderr


def test_resume_keeps_frozen_sha_campaign_science_state_and_receipt_gates(
    tmp_path, backend, monkeypatch
):
    first = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a")
    assert first.returncode == 0, first.stdout + first.stderr
    state = manifest.load_state(state_path(tmp_path))
    provenance = dict(state["provenance"]["last_resolved"])
    approved = state["scientific_identity"]["scientific_coverage"]

    wrong_sha = dict(provenance, environment_file_sha256="0" * 64)
    with pytest.raises(ValueError, match="frozen environment tarball differs"):
        manifest.validate_resume(
            state, state["campaign_identity"], state["profile_contract"], wrong_sha, approved,
        )

    wrong_identity = dict(state["campaign_identity"], campaign_tag="other-campaign")
    with pytest.raises(ValueError, match="campaign identity or profile contract differs"):
        manifest.validate_resume(
            state, wrong_identity, state["profile_contract"], provenance, approved,
        )

    with pytest.raises(ValueError, match="scientific_coverage_incompatible"):
        manifest.validate_resume(
            state, state["campaign_identity"], state["profile_contract"], provenance, approved[:-1],
        )

    original_scientific_files = manifest.scientific_files

    def changed_scientific_files(contract, stage):
        files = original_scientific_files(contract, stage)
        if stage == "source":
            files = {**files, "input_samples/cfgs/changed.cfg": "a" * 64}
        return files

    monkeypatch.setattr(manifest, "scientific_files", changed_scientific_files)
    with pytest.raises(ValueError, match="scientific_contract_incompatible"):
        manifest.validate_resume(
            state, state["campaign_identity"], state["profile_contract"], provenance, approved,
        )

    corrupt_state = copy.deepcopy(state)
    corrupt_state["schema_version"] = 8
    with pytest.raises(ValueError, match="normal maintained resume requires schema 9"):
        manifest.validate_state(corrupt_state)

    contradictory_receipt = copy.deepcopy(state)
    contradictory_receipt["units"][0]["stages"]["source"]["receipt"]["artifact"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact sidecar identity mismatch"):
        manifest.validate_state(contradictory_receipt)


def test_failed_downstream_reuses_source_on_resume(tmp_path, backend):
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "downstream_fail_once"
    first = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a")
    assert first.returncode != 0
    state = manifest.load_state(state_path(tmp_path))
    unit = state["units"][0]
    assert unit["stages"]["source"]["status"] == "success"
    assert unit["stages"]["data_driven"]["status"] == "failed"
    second = launch(tmp_path, backend, "--resume", "--selected-fragment", "run2_prod_a", "--chunksize", "54321")
    assert second.returncode == 0, second.stdout + second.stderr
    assert "Reusing verified source" in second.stdout
    unit = manifest.load_state(state_path(tmp_path))["units"][0]
    assert len(unit["stages"]["source"]["attempts"]) == 1
    assert len(unit["stages"]["data_driven"]["attempts"]) == 2
    assert unit["stages"]["data_driven"]["status"] == "success"


def test_manager_override_rejects_reusable_source_before_state_transition(tmp_path, backend):
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "downstream_fail_once"
    first = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a")
    assert first.returncode != 0
    before = state_path(tmp_path).read_bytes()
    second = launch(tmp_path, backend, "--resume", "--selected-fragment", "run2_prod_a",
                    "--wq-manager-name", "exact-name")
    assert second.returncode != 0
    assert "resolved zero" in second.stderr
    assert state_path(tmp_path).read_bytes() == before
    assert "Running the following command:" not in second.stdout


def test_manager_override_rejects_multiple_sources_before_campaign_creation(tmp_path, backend):
    result = launch(tmp_path, backend, "--wq-manager-name", "exact-name")
    assert result.returncode != 0
    assert "resolved multiple" in result.stderr
    assert not (tmp_path / "output").exists()
    assert "Running the following command:" not in result.stdout


def test_manager_exact_override_remains_single_source(tmp_path, backend):
    result = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a",
                    "--wq-manager-name", "exact-name")
    assert result.returncode == 0, result.stdout + result.stderr
    state = manifest.load_state(state_path(tmp_path))
    command = state["units"][0]["stages"]["source"]["attempts"][0]["command_argv"]
    assert command[command.index("--wq-manager-name") + 1] == "exact-name"
    assert all(unit["stages"]["source"]["status"] == "pending" for unit in state["units"][1:])


def test_manager_prefix_rejects_exact_override(tmp_path, backend):
    mixed = launch(tmp_path, backend, "--resume", "--wq-manager-name", "exact-name",
                   "--wq-manager-prefix", "test-")
    assert mixed.returncode != 0
    assert "mutually exclusive" in mixed.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(("profile", "prefix"), (
    ("run2_prod_2los_1tau", "top26006-l12-tau5j-run2-c261024-001-"),
    ("run3_prod_2los_1tau", "top26006-l12-tau5j-run3-c261024-001-"),
))
def test_manager_prefix_fresh_component_uses_initialized_state(tmp_path, backend, profile, prefix):
    result = launch(tmp_path, backend, "--wq-manager-prefix", prefix, profile=profile)
    assert result.returncode == 0, result.stdout + result.stderr
    state = manifest.load_state(state_path(tmp_path, profile))
    assert state["schema_version"] == 9
    assert len(state["units"]) == 1
    unit = state["units"][0]
    source_attempts = unit["stages"]["source"]["attempts"]
    assert [attempt["number"] for attempt in source_attempts] == [1]
    command = source_attempts[0]["command_argv"]
    assert command[command.index("--wq-manager-name") + 1] == f"{prefix}{unit['id']}-a1"


def test_manager_prefix_fresh_collision_refuses_before_source(tmp_path, backend):
    first = launch(tmp_path, backend, "--wq-manager-prefix", "fresh-",
                   profile="run2_prod_2los_1tau")
    assert first.returncode == 0, first.stdout + first.stderr
    state_before = state_path(tmp_path, "run2_prod_2los_1tau").read_bytes()
    second = launch(tmp_path, backend, "--wq-manager-prefix", "fresh-",
                    profile="run2_prod_2los_1tau")
    assert second.returncode != 0
    assert "refusing overwrite" in second.stderr
    assert "Running the following command:" not in second.stdout
    assert state_path(tmp_path, "run2_prod_2los_1tau").read_bytes() == state_before


def test_manager_prefix_combined_profile_still_rejected(tmp_path, backend):
    result = launch(tmp_path, backend, "--wq-manager-prefix", "fresh-",
                    profile="run2_run3_prod_2los_1tau")
    assert result.returncode != 0
    assert "requires one component campaign" in result.stderr
    assert not (tmp_path / "output").exists()


def test_manager_prefix_combined_profile_uses_distinct_child_prefixes(tmp_path, backend):
    selected_base = str(tmp_path / "staging")
    Path(selected_base).mkdir()
    result = launch(
        tmp_path, backend,
        "--run2-wq-manager-prefix", "top26006-l12-tau5j-run2-c261024-001-",
        "--run3-wq-manager-prefix", "top26006-l12-tau5j-run3-c261024-001-",
        "--wq-staging-base", selected_base,
        "--wq-free-warning-bytes", "20000000000",
        "--wq-min-free-bytes", "10000000000",
        "--wq-poll-seconds", "1",
        profile="run2_run3_prod_2los_1tau",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "source_order.log").read_text().splitlines() == [
        "run2_prod_2los_1tau_a", "run3_prod_2los_1tau_a",
    ]
    output = tmp_path / "output"
    for run_scope, prefix in (
        ("run2", "top26006-l12-tau5j-run2-c261024-001-"),
        ("run3", "top26006-l12-tau5j-run3-c261024-001-"),
    ):
        profile = f"{run_scope}_prod_2los_1tau"
        state = manifest.load_state(output / run_scope / f".{profile}_campaign_state_v9.json")
        assert state["schema_version"] == 9
        assert state["campaign_identity"]["production_profile"] == profile
        assert state["campaign_identity"]["campaign_tag"] == f"synthetic_{run_scope}"
        assert len(state["units"]) == 1
        source_attempts = state["units"][0]["stages"]["source"]["attempts"]
        assert [attempt["number"] for attempt in source_attempts] == [1]
        command = source_attempts[0]["command_argv"]
        assert command[command.index("--wq-filepath") + 1] == selected_base
        assert "--wq-root-backed-local-statvfs" not in command
        assert "--wq-quota-root" not in command
        assert command[command.index("--wq-manager-name") + 1] == (
            f"{prefix}{profile}_a-a1"
        )
    assert (output / "campaign_summary.tsv").is_file()


@pytest.mark.parametrize("extra", (
    ("--run2-wq-manager-prefix", "run2-"),
    ("--run3-wq-manager-prefix", "run3-"),
    ("--run2-wq-manager-prefix", "", "--run3-wq-manager-prefix", "run3-"),
    ("--run2-wq-manager-prefix", "run2-", "--run3-wq-manager-prefix", "bad prefix"),
    ("--run2-wq-manager-prefix", "run2-", "--run3-wq-manager-prefix", "run3-",
     "--run2-wq-manager-prefix", "again-"),
    ("--run2-wq-manager-prefix", "run2-", "--run3-wq-manager-prefix", "run3-",
     "--wq-manager-prefix", "legacy-"),
    ("--run2-wq-manager-prefix", "run2-", "--run3-wq-manager-prefix", "run3-",
     "--wq-manager-name", "exact-name"),
    ("--dry-run", "--run2-wq-manager-prefix", "run2-",
     "--run3-wq-manager-prefix", "run3-"),
))
def test_manager_prefix_combined_invalid_pair_rejected_before_output(tmp_path, backend, extra):
    result = launch(tmp_path, backend, *extra, profile="run2_run3_prod_2los_1tau")
    assert result.returncode != 0
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "source_order.log").exists()


def test_manager_prefix_paired_options_rejected_on_component(tmp_path, backend):
    result = launch(
        tmp_path, backend,
        "--run2-wq-manager-prefix", "run2-",
        "--run3-wq-manager-prefix", "run3-",
        profile="run2_prod_2los_1tau",
    )
    assert result.returncode != 0
    assert "require a combined profile" in result.stderr
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "source_order.log").exists()


def test_manager_prefix_combined_resume_uses_next_run3_attempt(tmp_path, backend):
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "source_fail_run3_tau"
    prefix_args = (
        "--run2-wq-manager-prefix", "run2-",
        "--run3-wq-manager-prefix", "run3-",
    )
    first = launch(tmp_path, backend, *prefix_args, profile="run2_run3_prod_2los_1tau")
    assert first.returncode != 0
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "success"
    resumed = launch(tmp_path, backend, "--resume", "--wq-port", "0", *prefix_args,
                     profile="run2_run3_prod_2los_1tau")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    output = tmp_path / "output"
    run2 = manifest.load_state(output / "run2" / ".run2_prod_2los_1tau_campaign_state_v9.json")
    run3 = manifest.load_state(output / "run3" / ".run3_prod_2los_1tau_campaign_state_v9.json")
    run2_attempts = run2["units"][0]["stages"]["source"]["attempts"]
    run3_attempts = run3["units"][0]["stages"]["source"]["attempts"]
    assert [attempt["number"] for attempt in run2_attempts] == [1]
    assert [attempt["number"] for attempt in run3_attempts] == [1, 2]
    assert "--port" not in run2_attempts[0]["command_argv"]
    assert run3_attempts[1]["command_argv"].count("--port") == 1
    assert run3_attempts[1]["command_argv"][run3_attempts[1]["command_argv"].index("--port") + 1] == "0"
    assert run2_attempts[0]["command_argv"][
        run2_attempts[0]["command_argv"].index("--wq-manager-name") + 1
    ] == "run2-run2_prod_2los_1tau_a-a1"
    assert run3_attempts[1]["command_argv"][
        run3_attempts[1]["command_argv"].index("--wq-manager-name") + 1
    ] == "run3-run3_prod_2los_1tau_a-a2"
    assert (tmp_path / "source_order.log").read_text().splitlines() == [
        "run2_prod_2los_1tau_a", "run3_prod_2los_1tau_a", "run3_prod_2los_1tau_a",
    ]


def test_wq_port_combined_resume_retries_both_failed_sources(tmp_path, backend):
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "source_fail_both_tau"
    prefix_args = (
        "--run2-wq-manager-prefix", "run2-",
        "--run3-wq-manager-prefix", "run3-",
    )
    first = launch(tmp_path, backend, *prefix_args, profile="run2_run3_prod_2los_1tau")
    assert first.returncode != 0
    output = tmp_path / "output"
    paths = [output / scope / f".{scope}_prod_2los_1tau_campaign_state_v9.json"
             for scope in ("run2", "run3")]
    before = [manifest.load_state(path)["units"][0]["stages"]["source"]["attempts"][0]
              for path in paths]
    assert all(attempt["result"] == "failed" for attempt in before)
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "success"
    resumed = launch(tmp_path, backend, "--resume", "--wq-port", "0", *prefix_args,
                     profile="run2_run3_prod_2los_1tau")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    for scope, path, original in zip(("run2", "run3"), paths, before):
        attempts = manifest.load_state(path)["units"][0]["stages"]["source"]["attempts"]
        assert [attempt["number"] for attempt in attempts] == [1, 2]
        assert attempts[0] == original
        assert attempts[1]["result"] == "success"
        command = attempts[1]["command_argv"]
        assert command.count("--port") == 1
        assert command[command.index("--port") + 1] == "0"
        assert command[command.index("--wq-manager-name") + 1] == (
            f"{scope}-{scope}_prod_2los_1tau_a-a2"
        )
    assert (tmp_path / "source_order.log").read_text().splitlines() == [
        "run2_prod_2los_1tau_a", "run3_prod_2los_1tau_a",
        "run2_prod_2los_1tau_a", "run3_prod_2los_1tau_a",
    ]


@pytest.mark.parametrize("port", ("0", "1", "65535"))
def test_wq_port_component_forwarding(tmp_path, backend, port):
    result = launch(tmp_path, backend, "--wq-port", port,
                    profile="run2_prod_2los_1tau")
    assert result.returncode == 0, result.stdout + result.stderr
    command = manifest.load_state(state_path(tmp_path, "run2_prod_2los_1tau"))["units"][0][
        "stages"]["source"]["attempts"][0]["command_argv"]
    assert command.count("--port") == 1
    assert command[command.index("--port") + 1] == port


@pytest.mark.parametrize("extra", (
    ("--wq-port",),
    ("--wq-port", "-1"),
    ("--wq-port", "65536"),
    ("--wq-port", "1.5"),
    ("--wq-port", "abc"),
    ("--wq-port", "0", "--wq-port", "1"),
))
def test_wq_port_rejects_invalid_value_before_output(tmp_path, backend, extra):
    result = launch(tmp_path, backend, *extra, profile="run2_prod_2los_1tau")
    assert result.returncode != 0
    assert "--wq-port requires one integer" in result.stderr
    assert not (tmp_path / "output").exists()


def test_manager_prefix_fresh_dry_run_requires_initialized_state(tmp_path, backend):
    result = launch(tmp_path, backend, "--dry-run", "--wq-manager-prefix", "fresh-",
                    profile="run2_prod_2los_1tau")
    assert result.returncode != 0
    assert "requires initialized campaign state" in result.stderr
    assert not (tmp_path / "output").exists()


def test_manager_prefix_resumes_all_units_with_persisted_unit_names(tmp_path, backend):
    first = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a")
    assert first.returncode == 0, first.stdout + first.stderr
    resumed = launch(tmp_path, backend, "--resume", "--wq-manager-prefix", "run3-resume-")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    state = manifest.load_state(state_path(tmp_path))
    assert manifest.classify(state) == "success"
    assert [unit["id"] for unit in state["units"]] == [
        "run2_prod_a", "run2_prod_b", "run2_prod_c", "run2_prod_d", "run2_prod_e",
    ]
    assert len(state["units"][0]["stages"]["source"]["attempts"]) == 1
    for unit in state["units"][1:]:
        command = unit["stages"]["source"]["attempts"][0]["command_argv"]
        index = command.index("--wq-manager-name")
        assert command[index + 1] == f"run3-resume-{unit['id']}-a1"


def test_manager_prefix_retry_uses_next_attempt_and_selected_unit(tmp_path, backend):
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "source_fail_first"
    failed = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a")
    assert failed.returncode != 0
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "success"
    resumed = launch(tmp_path, backend, "--resume", "--selected-fragment", "run2_prod_a",
                     "--wq-manager-prefix", "run3-resume-")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    state = manifest.load_state(state_path(tmp_path))
    source_attempts = state["units"][0]["stages"]["source"]["attempts"]
    assert [attempt["number"] for attempt in source_attempts] == [1, 2]
    command = source_attempts[1]["command_argv"]
    assert command[command.index("--wq-manager-name") + 1] == "run3-resume-run2_prod_a-a2"
    assert all(unit["stages"]["source"]["status"] == "pending" for unit in state["units"][1:])


@pytest.mark.parametrize("blocker", (None, "owner", "reader", "next_root", "lock"))
def test_normal_guarded_resume_recovers_before_synthetic_source_and_dd(tmp_path, backend, blocker):
    # This backend writes inert text artifacts; it cannot start a manager or submit.
    sibling = launch(tmp_path, backend, "--selected-fragment", "run2_prod_b")
    assert sibling.returncode == 0, sibling.stdout + sibling.stderr
    base = tmp_path / "staging"
    base.mkdir()
    args = ("--wq-staging-base", str(base), "--wq-free-warning-bytes", "30000000000",
            "--wq-min-free-bytes", "10000000000", "--wq-poll-seconds", "30",
            "--wq-manager-prefix", "synthetic-resume-")
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "source_fail_first"
    failed = launch(tmp_path, backend, "--resume", "--selected-fragment", "run2_prod_a", *args)
    assert failed.returncode != 0
    path = state_path(tmp_path)
    state = manifest.load_state(path)
    unit = state["units"][0]
    record = unit["stages"]["source"]
    assert len(record["attempts"]) == 1
    record["status"] = "running"
    record["attempts"][0].update(result="running", ended_at_utc=None, exit_code=None)
    manifest.atomic_json(path, state)  # Synthetic reboot fixture only.
    preserved_siblings = copy.deepcopy(state["units"][1:])
    old = base / "synthetic_run2_prod_a_source_attempt_1"
    old.mkdir(mode=0o700)
    payload = old / "tmpsyntheticwq-tmp-" / "nested"
    payload.mkdir(parents=True)
    (payload / "accum_large.p").write_bytes(b"transient accumulation")
    marker = {"path": str(old), "invocation": old.name, "status": "running",
              "owner_uid": os.getuid(), "owner_inode": old.stat().st_ino,
              "owner_pid": os.getpid() if blocker == "owner" else 1_000_000_000,
              "owner_start": staging_guard._process_start(os.getpid()) if blocker == "owner" else "1"}
    (old / "guard.json").write_text(json.dumps(marker))
    if blocker == "next_root":
        (base / "synthetic_run2_prod_a_source_attempt_2").mkdir()
    script = Path(backend["CAMPAIGN_VALIDATION_BACKEND"])
    script.write_text(script.read_text().replace(
        'if [[ "$action" == run_block ]]; then',
        'if [[ "$action" == run_block ]]; then\n  [[ ! -e "$OLD_STAGING_ROOT" ]] || exit 99'))
    backend["OLD_STAGING_ROOT"] = str(old)
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "success"
    reader = (payload / "accum_large.p").open("rb") if blocker == "reader" else None
    try:
        if blocker == "lock":
            with manifest.campaign_lock(path):
                resumed = launch(tmp_path, backend, "--resume", "--selected-fragment", "run2_prod_a", *args)
        else:
            resumed = launch(tmp_path, backend, "--resume", "--selected-fragment", "run2_prod_a", *args)
    finally:
        if reader is not None:
            reader.close()
    durable = manifest.load_state(path)
    assert durable["units"][1:] == preserved_siblings
    attempts = durable["units"][0]["stages"]["source"]["attempts"]
    if blocker:
        assert resumed.returncode != 0, resumed.stdout + resumed.stderr
        assert len(attempts) == 1 and (payload / "accum_large.p").exists()
        if blocker in ("owner", "reader", "lock"):
            assert attempts[0]["result"] == "running"
    else:
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        assert not old.exists()
        assert [attempt["result"] for attempt in attempts] == ["interrupted", "success"]
        assert [attempt["number"] for attempt in attempts] == [1, 2]
        command = attempts[1]["command_argv"]
        assert command[command.index("--wq-invocation-id") + 1] == "synthetic_run2_prod_a_source_attempt_2"
        assert command[command.index("--wq-manager-name") + 1] == "synthetic-resume-run2_prod_a-a2"
        assert durable["units"][0]["stages"]["data_driven"]["status"] == "success"


def test_manager_prefix_does_not_name_reused_source(tmp_path, backend):
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "downstream_fail_once"
    first = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a")
    assert first.returncode != 0
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "success"
    resumed = launch(tmp_path, backend, "--resume", "--selected-fragment", "run2_prod_a",
                     "--wq-manager-prefix", "run3-resume-")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "Reusing verified source" in resumed.stdout
    state = manifest.load_state(state_path(tmp_path))
    source_attempts = state["units"][0]["stages"]["source"]["attempts"]
    assert len(source_attempts) == 1
    assert "--wq-manager-name" not in source_attempts[0]["command_argv"]


def test_manager_prefix_full_resume_reuses_source_and_continues(tmp_path, backend):
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "downstream_fail_once"
    first = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a")
    assert first.returncode != 0
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "success"
    resumed = launch(tmp_path, backend, "--resume", "--wq-manager-prefix", "run3-resume-")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    state = manifest.load_state(state_path(tmp_path))
    assert manifest.classify(state) == "success"
    assert len(state["units"][0]["stages"]["source"]["attempts"]) == 1
    for unit in state["units"][1:]:
        command = unit["stages"]["source"]["attempts"][0]["command_argv"]
        assert command[command.index("--wq-manager-name") + 1] == (
            f"run3-resume-{unit['id']}-a1"
        )


def test_combined_child_failure_retains_run2_evidence_and_runs_run3(tmp_path, backend):
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "source_fail_first"
    result = launch(tmp_path, backend, profile="run2_run3_prod")
    assert result.returncode != 0
    run2 = manifest.load_state(tmp_path / "output" / "run2" / ".run2_prod_campaign_state_v9.json")
    run3 = manifest.load_state(tmp_path / "output" / "run3" / ".run3_prod_campaign_state_v9.json")
    assert manifest.classify(run2) == "complete_with_known_failures"
    assert manifest.classify(run3) == "success"
    assert run2["units"][0]["stages"]["source"]["attempts"][0]["exit_code"] == 23
    assert (tmp_path / "output" / "campaign_summary.tsv").is_file()


def combined_states(tmp_path):
    return [manifest.load_state(
        tmp_path / "output" / scope / f".{scope}_prod_campaign_state_v9.json"
    ) for scope in ("run2", "run3")]


@pytest.mark.parametrize("scope", ("run2", "run3"))
def test_combined_selected_preserves_full_plans_and_native_resume(tmp_path, backend, scope):
    selected = f"{scope}_prod_a"
    prefixes = ("--run2-wq-manager-prefix", "run2-", "--run3-wq-manager-prefix", "run3-")
    result = launch(tmp_path, backend, "--selected-fragment", selected, *prefixes,
                    profile="run2_run3_prod")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "resume: true" in result.stdout
    before = combined_states(tmp_path)
    ids = [{unit["id"] for unit in state["units"]} for state in before]
    assert len(ids[0]) == len(ids[1]) == 5
    assert not ids[0] & ids[1]
    assert (tmp_path / "source_order.log").read_text().splitlines() == [selected]
    for state in before:
        for unit in state["units"]:
            assert manifest.overall_status(unit) == ("success" if unit["id"] == selected else "planned")
            for stage in unit["stages"].values():
                assert len(stage["attempts"]) == (1 if unit["id"] == selected else 0)
    resumed = launch(tmp_path, backend, "--resume", "--selected-fragment", selected,
                     *prefixes, profile="run2_run3_prod")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    after = combined_states(tmp_path)
    for old, new in zip(before, after):
        assert old["plan"] == new["plan"]
        assert old["scientific_identity"] == new["scientific_identity"]
        assert old["units"] == new["units"]
    assert (tmp_path / "source_order.log").read_text().splitlines() == [selected]
    assert (tmp_path / "output" / "campaign_summary.tsv").is_file()


def test_combined_selected_retry_uses_native_eligibility(tmp_path, backend):
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "source_fail_first"
    first = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a", profile="run2_run3_prod")
    assert first.returncode != 0
    before = combined_states(tmp_path)
    backend["CAMPAIGN_VALIDATION_SCENARIO"] = "success"
    resumed = launch(tmp_path, backend, "--resume", "--selected-fragment", "run2_prod_a",
                     profile="run2_run3_prod")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    run2, run3 = combined_states(tmp_path)
    assert [a["number"] for a in run2["units"][0]["stages"]["source"]["attempts"]] == [1, 2]
    assert run3 == before[1]
    assert (tmp_path / "source_order.log").read_text().splitlines() == ["run2_prod_a", "run2_prod_a"]


def test_combined_selected_resume_routes_planned_other_component(tmp_path, backend):
    first = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a", profile="run2_run3_prod")
    assert first.returncode == 0, first.stdout + first.stderr
    before = combined_states(tmp_path)
    result = launch(tmp_path, backend, "--resume", "--selected-fragment", "run3_prod_a",
                    profile="run2_run3_prod")
    assert result.returncode == 0, result.stdout + result.stderr
    after = combined_states(tmp_path)
    assert after[0] == before[0]
    assert after[1]["plan"] == before[1]["plan"]
    assert manifest.overall_status(after[1]["units"][0]) == "success"
    assert (tmp_path / "source_order.log").read_text().splitlines() == ["run2_prod_a", "run3_prod_a"]


def test_combined_native_id_namespaces_cannot_be_ambiguous():
    # The maintained constructor prefixes even adversarial supplied IDs with
    # their component profile; the parent never guesses ownership by ordering.
    specs = []
    for profile in ("run2_prod", "run3_prod"):
        units = [{"id": name, "years": ["2018" if profile.startswith("run2") else "2022"],
                  "category_groups": ["2lss"], "histograms": ["njets"]}
                 for name in ("shared", "run2_prod_shared", "run3_prod_shared")]
        specs.append([manifest.make_specs([unit], profile, "tag", "/synthetic", "r1_123456789abc")[0]
                      for unit in units])
    assert all(unit["id"].startswith("run2_prod_") for unit in specs[0])
    assert all(unit["id"].startswith("run3_prod_") for unit in specs[1])
    assert not {u["id"] for u in specs[0]} & {u["id"] for u in specs[1]}


def test_combined_selected_unknown_and_missing_resume_fail_without_work(tmp_path, backend):
    missing = launch(tmp_path, backend, "--resume", "--selected-fragment", "run3_prod_a",
                     profile="run2_run3_prod")
    assert missing.returncode != 0
    result = launch(tmp_path, backend, "--selected-fragment", "unknown", profile="run2_run3_prod")
    assert result.returncode != 0
    assert "exactly one component owner (found 0)" in result.stderr
    for state in combined_states(tmp_path):
        assert len(state["units"]) == 5
        assert all(not stage["attempts"] for unit in state["units"] for stage in unit["stages"].values())
    assert not (tmp_path / "source_order.log").exists()
    resumed = launch(tmp_path, backend, "--resume", "--selected-fragment", "unknown",
                     profile="run2_run3_prod")
    assert resumed.returncode != 0
    assert "found 0" in resumed.stderr


@pytest.mark.parametrize("extra", (
    ("--plan", "unused.json"),
    ("--resume", "--activate-pending-plan", "unused.json", "--expected-plan-revision", "r2"),
))
def test_combined_selected_keeps_revision_restrictions(tmp_path, backend, extra):
    result = launch(tmp_path, backend, "--selected-fragment", "run3_prod_a", *extra,
                    profile="run2_run3_prod")
    assert result.returncode != 0
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "source_order.log").exists()


def test_combined_fresh_full_and_full_resume_keep_native_plans(tmp_path, backend):
    result = launch(tmp_path, backend, profile="run2_run3_prod")
    assert result.returncode == 0, result.stdout + result.stderr
    before = combined_states(tmp_path)
    assert all(len(state["units"]) == 5 for state in before)
    assert all(manifest.classify(state) == "success" for state in before)
    assert len((tmp_path / "source_order.log").read_text().splitlines()) == 10
    resumed = launch(tmp_path, backend, "--resume", profile="run2_run3_prod")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert [s["units"] for s in before] == [s["units"] for s in combined_states(tmp_path)]
    assert len((tmp_path / "source_order.log").read_text().splitlines()) == 10


def test_combined_selected_dry_run_creates_no_state_or_attempts(tmp_path, backend):
    result = launch(tmp_path, backend, "--dry-run", "--selected-fragment", "run3_prod_a",
                    profile="run2_run3_prod")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "source_order.log").exists()


@pytest.mark.parametrize("scope", ("run2", "run3"))
def test_combined_selected_telemetry_then_full_resume(tmp_path, backend, scope):
    selected = f"{scope}_prod_a"
    args = ("--phase-telemetry",)
    first = launch(tmp_path, backend, *args, "--selected-fragment", selected,
                   profile="run2_run3_prod")
    assert first.returncode == 0, first.stdout + first.stderr
    before = combined_states(tmp_path)
    frozen = next(unit for state in before for unit in state["units"] if unit["id"] == selected)
    artifacts = {path: (Path(path).read_bytes(), Path(path).stat().st_mtime_ns)
                 for stage in frozen["stages"].values()
                 for path in (stage["receipt"]["artifact"]["path"],
                              stage["receipt"]["artifact"]["path"] + ".metadata.json")}
    assert all(len(state["units"]) == 5 for state in before)
    assert (tmp_path / "source_order.log").read_text().splitlines() == [selected]
    assert "successful_count: 1" in (tmp_path / "output/campaign_summary.md").read_text()
    for log in ("source_argv.log", "dd_argv.log"):
        assert "--phase-telemetry" in (tmp_path / log).read_text()
    for extra in (("--selected-fragment", selected), (), ()):
        resumed = launch(tmp_path, backend, "--resume", *args, *extra,
                         profile="run2_run3_prod")
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        after = combined_states(tmp_path)
        for old, new in zip(before, after):
            assert old["plan"] == new["plan"]
            assert old["scientific_identity"] == new["scientific_identity"]
            assert old["campaign_identity"] == new["campaign_identity"]
        assert next(u for s in after for u in s["units"] if u["id"] == selected) == frozen
        assert all((Path(path).read_bytes(), Path(path).stat().st_mtime_ns) == value
                   for path, value in artifacts.items())
    order = (tmp_path / "source_order.log").read_text().splitlines()
    assert order == [selected] + [f"{s}_prod_{letter}" for s in ("run2", "run3")
                                  for letter in "abcde" if f"{s}_prod_{letter}" != selected]
    assert all(manifest.classify(state) == "success" for state in after)
    assert all(len(stage["attempts"]) == 1 and "--phase-telemetry" in stage["attempts"][0]["command_argv"]
               for state in after for unit in state["units"] for stage in unit["stages"].values())
    assert all("--phase-telemetry" in line for line in (tmp_path / "dd_argv.log").read_text().splitlines())
    assert "final_classification: success" in (tmp_path / "output/campaign_summary.md").read_text()


@pytest.mark.parametrize("interruption", ("second_prepare", "prepared", "child_environment"))
@pytest.mark.parametrize("selected", ("run2_prod_a", "run3_prod_a"))
def test_combined_resume_completes_only_missing_preparation(tmp_path, backend, interruption, selected):
    if interruption != "prepared":
        backend["CAMPAIGN_FAIL_ENV_CALL"] = "2" if interruption == "second_prepare" else "3"
    first = launch(tmp_path, backend, "--selected-fragment",
                   "unknown" if interruption == "prepared" else selected, profile="run2_run3_prod")
    assert first.returncode != 0
    paths = sorted((tmp_path / "output").glob("**/*_campaign_state_v9.json"))
    assert len(paths) == (1 if interruption == "second_prepare" else 2)
    before = {path: json.loads(path.read_text()) for path in paths}
    assert all(not stage["attempts"] for state in before.values()
               for unit in state["units"] for stage in unit["stages"].values())
    backend.pop("CAMPAIGN_FAIL_ENV_CALL", None)
    resumed = launch(tmp_path, backend, "--resume", "--selected-fragment", selected,
                     profile="run2_run3_prod")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    for path, old in before.items():
        new = manifest.load_state(path)
        assert old["plan"] == new["plan"]
        assert old["created_at_utc"] == new["created_at_utc"]
        assert old["scientific_identity"] == new["scientific_identity"]
    assert (tmp_path / "source_order.log").read_text().splitlines() == [selected]


@pytest.mark.parametrize("scope", ("run2", "run3"))
@pytest.mark.parametrize("condition", ("known_failure", "running", "lock"))
def test_combined_selected_checks_sibling_safety(tmp_path, backend, scope, condition):
    sibling = "run3" if scope == "run2" else "run2"
    backend["CAMPAIGN_FAIL_UNIT"] = f"{sibling}_prod_a"
    first = launch(tmp_path, backend, "--selected-fragment", f"{sibling}_prod_a",
                   profile="run2_run3_prod")
    assert first.returncode != 0
    path = tmp_path / "output" / sibling / f".{sibling}_prod_campaign_state_v9.json"
    state = manifest.load_state(path)
    if condition == "running":
        record = state["units"][0]["stages"]["source"]
        record["status"] = "running"
        record["attempts"][0].update(result="running", ended_at_utc=None, exit_code=None)
        manifest.atomic_json(path, state)
    before = path.read_bytes()
    selected_path = tmp_path / "output" / scope / f".{scope}_prod_campaign_state_v9.json"
    selected_before = selected_path.read_bytes()
    if condition == "known_failure":
        backend["CAMPAIGN_EXPECT_LOCK_FREE"] = str(path.with_name(f".{sibling}_prod_campaign.lock"))
    if condition == "lock":
        with manifest.campaign_lock(path):
            result = launch(tmp_path, backend, "--resume", "--selected-fragment", f"{scope}_prod_a",
                            profile="run2_run3_prod")
    else:
        result = launch(tmp_path, backend, "--resume", "--selected-fragment", f"{scope}_prod_a",
                        profile="run2_run3_prod")
    assert path.read_bytes() == before
    order = (tmp_path / "source_order.log").read_text().splitlines()
    if condition == "known_failure":
        assert result.returncode == 0, result.stdout + result.stderr
        assert order == [f"{sibling}_prod_a", f"{scope}_prod_a"]
    else:
        assert result.returncode != 0
        assert selected_path.read_bytes() == selected_before
        assert order == [f"{sibling}_prod_a"]


@pytest.mark.parametrize("damage", ("identity", "plan", "environment"))
def test_combined_prepare_refuses_incompatible_state(tmp_path, backend, damage):
    prepared = launch(tmp_path, backend, "--selected-fragment", "unknown", profile="run2_run3_prod")
    assert prepared.returncode != 0
    path = tmp_path / "output/run2/.run2_prod_campaign_state_v9.json"
    state = json.loads(path.read_text())
    if damage == "identity":
        state["campaign_identity"]["campaign_tag"] = "other"
    elif damage == "plan":
        state["plan"]["revisions"][0]["units"][0]["histograms"] = ["other"]
    else:
        (tmp_path / "other.tar.gz").write_bytes(b"different environment")
    if damage != "environment":
        manifest.atomic_json(path, state)
    before = path.read_bytes()
    resumed = launch(tmp_path, backend, "--resume", "--selected-fragment", "run3_prod_a",
                     profile="run2_run3_prod",
                     archive=tmp_path / "other.tar.gz" if damage == "environment" else ARCHIVE)
    assert resumed.returncode != 0
    if damage == "environment":
        assert "frozen environment tarball differs" in resumed.stderr
    assert path.read_bytes() == before
    assert not (tmp_path / "source_order.log").exists()


def test_combined_selected_rejects_duplicate_recorded_owner(tmp_path, backend):
    prepared = launch(tmp_path, backend, "--selected-fragment", "unknown", profile="run2_run3_prod")
    assert prepared.returncode != 0
    path = tmp_path / "output/run3/.run3_prod_campaign_state_v9.json"
    state = manifest.load_state(path)
    # A self-consistent recorded plan can use any valid snake_case ID. The
    # parent must count membership even if a supplied state breaks prefixes.
    state["units"][0]["id"] = "run2_prod_a"
    specs = [manifest.unit_spec(unit) for unit in state["units"]]
    units = [{key: unit[key] for key in ("id", "years", "category_groups", "histograms")}
             for unit in state["units"]]
    partition = manifest.plan_partition_sha256(units)
    state["plan"]["plan_partition_sha256"] = partition
    state["plan"]["revisions"][0].update(units=specs, partition_sha256=partition,
                                           snapshot_sha256=manifest.canonical_digest(specs))
    manifest.atomic_json(path, manifest.validate_state(state))
    before = [path.read_bytes() for path in sorted((tmp_path / "output").glob("**/*_state_v9.json"))]
    result = launch(tmp_path, backend, "--resume", "--selected-fragment", "run2_prod_a",
                    profile="run2_run3_prod")
    assert result.returncode != 0 and "found 2" in result.stderr
    assert [path.read_bytes() for path in sorted((tmp_path / "output").glob("**/*_state_v9.json"))] == before
    assert not (tmp_path / "source_order.log").exists()


def test_combined_selected_stat_diagnostic_prepares_sumw2_options(tmp_path, backend):
    result = launch(tmp_path, backend, "--selected-fragment", "run3_stat_diagnostic_a",
                    "--phase-telemetry", profile="run2_run3_stat_diagnostic")
    assert result.returncode == 0, result.stdout + result.stderr
    for scope in ("run2", "run3"):
        root = tmp_path / "output" / scope
        state = manifest.load_state(root / f".{scope}_stat_diagnostic_campaign_state_v9.json")
        assert len(state["units"]) == 5
        assert (root / "sumw2_full_diagnostics.yml").read_text() == "sumw2_storage:\n  mode: full_diagnostics\n"
    assert (tmp_path / "source_order.log").read_text().splitlines() == ["run3_stat_diagnostic_a"]


@pytest.mark.parametrize(("profile", "expected_years"), (
    ("run2_prod_2los_1tau", {"2016", "2016APV", "2017", "2018"}),
    ("run3_prod_2los_1tau", {"2022", "2022EE", "2023", "2023BPix"}),
))
def test_restricted_tau_component_preserves_production_contract(
    tmp_path, backend, profile, expected_years
):
    result = launch(tmp_path, backend, profile=profile)
    assert result.returncode == 0, result.stdout + result.stderr
    state = manifest.load_state(state_path(tmp_path, profile))
    assert len(state["units"]) == 1
    unit = state["units"][0]
    assert unit["category_groups"] == ["2los_1tau"]
    assert unit["histograms"] == ["njets", "lj0pt", "ptz"]
    assert set(manifest.expanded_years(unit["years"])) == expected_years
    assert set(state["scientific_identity"]["scientific_coverage"]) == {
        f"{year}|2los_1tau|{variable}"
        for year in expected_years
        for variable in ("njets", "lj0pt", "ptz")
    }
    policy = state["profile_contract"]
    assert policy["systematics"] is True
    assert policy["sumw2_policy"] == "implicit_production"
    assert policy["data_driven_policy"] == "nonprompt_charge_flip"
    assert policy["stages"] == ["source", "data_driven"]
    assert set(unit["stages"]) == {"source", "data_driven"}
    assert manifest.overall_status(unit) == "success"
    assert "split_lep_flavor: false" in result.stdout


def test_combined_restricted_tau_profile_reuses_component_states(tmp_path, backend):
    result = launch(tmp_path, backend, profile="run2_run3_prod_2los_1tau")
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "output"
    for run_scope in ("run2", "run3"):
        profile = f"{run_scope}_prod_2los_1tau"
        state = manifest.load_state(
            output / run_scope / f".{profile}_campaign_state_v9.json"
        )
        assert state["campaign_identity"]["production_profile"] == profile
        assert len(state["units"]) == 1
        assert state["units"][0]["category_groups"] == ["2los_1tau"]
        assert state["units"][0]["histograms"] == ["njets", "lj0pt", "ptz"]
        assert manifest.overall_status(state["units"][0]) == "success"
    assert (output / "campaign_summary.tsv").is_file()


@pytest.mark.parametrize("profile", (
    "run2_prod_2los_1tau_CR",
    "run3_prod_2los_1tau_CR",
))
def test_restricted_tau_component_rejects_cr_alias(profile):
    with pytest.raises(ValueError, match="unsupported component profile"):
        manifest.profile_contract(profile)


def test_symlinked_campaign_lock_is_rejected(tmp_path, backend):
    output = tmp_path / "output"
    output.mkdir()
    lock = output / ".run2_prod_campaign.lock"
    lock.symlink_to(tmp_path / "outside.lock")
    result = launch(tmp_path, backend, "--resume")
    assert result.returncode != 0
    assert "campaign" in result.stderr.lower() or "campaign" in result.stdout.lower()
    assert not (tmp_path / "outside.lock").exists()


def test_module_lock_refuses_concurrent_writer(tmp_path):
    state = tmp_path / ".run2_prod_campaign_state_v9.json"
    with manifest.campaign_lock(state):
        command = [
            "/users/apiccine/work/correction-lib/codex-run.sh",
            "/bin/bash", "--noprofile", "--norc", "-c",
            '"$1" "$2" reconcile "$3"', "campaign_lock_test",
            "/users/apiccine/work/miniconda3/envs/clib-env/bin/python",
            str(RUN_DIRECTORY / "campaign_manifest.py"), str(state),
        ]
        child = subprocess.run(command, capture_output=True, text=True, check=False)
        assert child.returncode != 0
        assert "exclusive campaign writer lock evidence missing" in child.stderr
def test_staging_guard_rejects_root_and_symlink_paths(tmp_path):
    with pytest.raises(staging_guard.staging_guard_error, match="root temporary"):
        staging_guard._validate_base(Path("/tmp"))
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(staging_guard.staging_guard_error, match="symlink"):
        staging_guard._validate_base(link)


@pytest.mark.parametrize(("measurement_error", "error_text"),
                         ((OSError("quota unavailable"), "OSError: quota unavailable"),
                          (subprocess.TimeoutExpired(["quota"], 15), "TimeoutExpired: Command")))
def test_staging_guard_warns_and_keeps_running_after_telemetry_failures(
    tmp_path, monkeypatch, capsys, measurement_error, error_text
):
    path = tmp_path / "owned"
    path.mkdir()
    pid = os.getpid()
    monkeypatch.setattr(staging_guard.os, "getppid", lambda: pid)
    start = staging_guard._process_start(pid)
    marker = path / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(path),
                                  "owner_uid": os.getuid(), "owner_pid": pid,
                                  "owner_start": start, "owner_inode": path.stat().st_ino}))
    identity = (path.stat().st_ino, marker.stat().st_ino)
    signalled = []
    monkeypatch.setattr(staging_guard.os, "kill", lambda target, sig: signalled.append((target, sig)))
    def no_tree_scan(*_):
        raise AssertionError("runtime monitoring must not enumerate staging")
    monkeypatch.setattr(staging_guard, "_charged_tree", no_tree_scan, raising=False)
    real_scandir = staging_guard.os.scandir
    monkeypatch.setattr(staging_guard.os, "scandir", lambda current: no_tree_scan()
                        if os.fspath(current) == str(path) else real_scandir(current))
    measurements = iter((measurement_error, (20000000000, 0), (40000000000, 2000)))
    def measure(*_):
        value = next(measurements)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(staging_guard, "_space", measure)
    class stop_event:
        polls = 0
        def is_set(self):
            return self.polls >= 3
        def wait(self, _):
            self.polls += 1
            return self.is_set()
    degraded = SimpleNamespace(value=False)
    degraded.set = lambda: setattr(degraded, "value", True)
    staging_guard._watch(path, tmp_path, "/groups/klannon", pid, start,
                         30000000000, 10000000000, 1, stop_event(), *identity, degraded)
    assert signalled == []
    assert degraded.value
    output = capsys.readouterr().out
    assert error_text in output
    assert output.count("WARNING: Work Queue staging free space or inodes are low") == 1
    assert "available=20.000 GB free_inodes=0" in output
    assert "RECOVERY: Work Queue staging free space and inodes recovered; " \
           "available=40.000 GB free_inodes=2000" in output
    assert "measurement_failed" not in output
    assert json.loads(marker.read_text())["status"] == "running"


def test_staging_guard_deduplicates_each_active_reason_independently(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "owned"
    path.mkdir()
    pid = os.getpid()
    monkeypatch.setattr(staging_guard.os, "getppid", lambda: pid)
    start = staging_guard._process_start(pid)
    marker = path / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(path),
                                  "owner_uid": os.getuid(), "owner_pid": pid,
                                  "owner_start": start, "owner_inode": path.stat().st_ino}))
    observed_start = iter((None, None, start, None, None, start, None))
    monkeypatch.setattr(staging_guard, "_process_start", lambda _: next(observed_start))
    measurements = iter((None, None, None, (40000000000, 2000),
                         (40000000000, 2000), (40000000000, 2000), None))

    def measure(*_):
        value = next(measurements)
        if value is None:
            raise OSError("capacity unavailable")
        return value

    monkeypatch.setattr(staging_guard, "_space", measure)
    signals = []
    monkeypatch.setattr(staging_guard.os, "kill", lambda target, sig: signals.append((target, sig)))

    class stop_event:
        polls = 0
        def is_set(self):
            return self.polls >= 7
        def wait(self, _):
            self.polls += 1
            return self.is_set()

    staging_guard._watch(path, tmp_path, "/groups/klannon", pid, start,
                         30000000000, 10000000000, 1, stop_event(),
                         path.stat().st_ino, marker.stat().st_ino)
    output = capsys.readouterr().out
    assert output.count("manager identity telemetry unavailable") == 3
    assert output.count("staging telemetry unavailable") == 2
    assert "CRITICAL:" not in output
    assert signals == []


def test_staging_guard_emits_only_health_state_transitions(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "owned"
    path.mkdir()
    pid = os.getpid()
    monkeypatch.setattr(staging_guard.os, "getppid", lambda: pid)
    start = staging_guard._process_start(pid)
    marker = path / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(path),
                                  "owner_uid": os.getuid(), "owner_pid": pid,
                                  "owner_start": start, "owner_inode": path.stat().st_ino}))
    measurements = iter((
        (40000000000, 2000),
        (40000000000, 2000),
        (29000000000, 2000),
        (29000000000, 2000),
        (40000000000, 2000),
        (40000000000, 2000),
        (29000000000, 2000),
        (10000000000, 2000),
        (10000000000, 2000),
    ))
    monkeypatch.setattr(staging_guard, "_space", lambda *_: next(measurements))
    signalled = []
    monkeypatch.setattr(staging_guard.os, "kill",
                        lambda target, sig: signalled.append((target, sig)))

    class stop_event:
        polls = 0
        def is_set(self):
            return self.polls >= 9
        def wait(self, _):
            self.polls += 1
            return self.is_set()

    staging_guard._watch(path, tmp_path, "/groups/klannon", pid, start,
                         30000000000, 10000000000, 1, stop_event(),
                         path.stat().st_ino, marker.stat().st_ino)
    output = capsys.readouterr().out
    assert output.splitlines() == [
        "WARNING: Work Queue staging free space or inodes are low; "
        "available=29.000 GB free_inodes=2000; manager remains running",
        "RECOVERY: Work Queue staging free space and inodes recovered; "
        "available=40.000 GB free_inodes=2000",
        "WARNING: Work Queue staging free space or inodes are low; "
        "available=29.000 GB free_inodes=2000; manager remains running",
        "CRITICAL: Work Queue staging effective available space 10.000 GB "
        "is at or below 10.000 GB; verifying manager identity before SIGTERM",
    ]
    assert signalled == [(pid, staging_guard.signal.SIGTERM)]


def test_staging_guard_uses_decimal_gb_and_keeps_inode_count_exact(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "owned"
    path.mkdir()
    pid = os.getpid()
    monkeypatch.setattr(staging_guard.os, "getppid", lambda: pid)
    start = staging_guard._process_start(pid)
    marker = path / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(path),
                                  "owner_uid": os.getuid(), "owner_pid": pid,
                                  "owner_start": start, "owner_inode": path.stat().st_ino}))
    measurements = iter(((30000000000, 999), (30000000000, 999)))
    monkeypatch.setattr(staging_guard, "_space", lambda *_: next(measurements))

    class stop_event:
        polls = 0
        def is_set(self):
            return self.polls >= 2
        def wait(self, _):
            self.polls += 1
            return self.is_set()

    staging_guard._watch(path, tmp_path, "/groups/klannon", pid, start,
                         30000000000, 10000000000, 1, stop_event(),
                         path.stat().st_ino, marker.stat().st_ino)
    output = capsys.readouterr().out
    assert output.splitlines() == [
        "WARNING: Work Queue staging free space or inodes are low; "
        "available=30.000 GB free_inodes=999; manager remains running",
    ]


@pytest.mark.parametrize(("available_bytes", "should_signal"), (
    (10000000000, True),
    (0, True),
    (10000000001, False),
))
def test_staging_guard_signals_only_at_or_below_critical_free_space(
    tmp_path, monkeypatch, capsys, available_bytes, should_signal
):
    path = tmp_path / "owned"
    path.mkdir()
    pid = os.getpid()
    monkeypatch.setattr(staging_guard.os, "getppid", lambda: pid)
    start = staging_guard._process_start(pid)
    marker = path / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(path),
                                  "owner_uid": os.getuid(), "owner_pid": pid,
                                  "owner_start": start, "owner_inode": path.stat().st_ino}))
    signals = []
    monkeypatch.setattr(staging_guard.os, "kill",
                        lambda target, sig: signals.append((target, sig)))
    monkeypatch.setattr(staging_guard, "_space", lambda *_: (available_bytes, 2000))

    class stop_event:
        polls = 0
        def is_set(self):
            return self.polls >= 2
        def wait(self, _):
            self.polls += 1
            return self.is_set()

    staging_guard._watch(path, tmp_path, "/groups/klannon", pid, start,
                         30000000000, 10000000000, 1, stop_event(),
                         path.stat().st_ino, marker.stat().st_ino)
    assert signals == ([(pid, staging_guard.signal.SIGTERM)] if should_signal else [])
    metadata = json.loads(marker.read_text())
    assert metadata["status"] == "running"
    if should_signal:
        assert metadata["reason"] == "critical_free_space"
        assert metadata["critical_free_space"]["available_bytes"] == available_bytes
        assert metadata["critical_free_space"]["min_free_bytes"] == 10000000000
        assert metadata["critical_free_space"]["observed_at"]
        assert metadata["critical_free_space"]["observed_at_unix_ns"] > 0
        output = capsys.readouterr().out
        assert output.count("CRITICAL:") == 1
        assert "30.000 GB" not in output
        assert "10.000 GB" in output
        assert "SIGTERM" in output
    else:
        assert "reason" not in metadata
        output = capsys.readouterr().out
        assert output.count("WARNING:") == 1


def test_staging_guard_does_not_signal_when_critical_marker_logging_fails(
    tmp_path, monkeypatch
):
    path = tmp_path / "owned"
    path.mkdir()
    pid = os.getpid()
    monkeypatch.setattr(staging_guard.os, "getppid", lambda: pid)
    start = staging_guard._process_start(pid)
    marker = path / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(path),
                                  "owner_uid": os.getuid(), "owner_pid": pid,
                                  "owner_start": start, "owner_inode": path.stat().st_ino}))
    signals = []
    monkeypatch.setattr(staging_guard.os, "kill",
                        lambda target, sig: signals.append((target, sig)))
    monkeypatch.setattr(staging_guard, "_space", lambda *_: (0, 2000))
    monkeypatch.setattr(staging_guard.os, "fsync",
                        lambda _: (_ for _ in ()).throw(OSError("marker sync failed")))

    class stop_event:
        def is_set(self):
            return False

    with pytest.raises(staging_guard.staging_guard_error,
                       match="critical free-space signal was not safely completed"):
        staging_guard._watch(path, tmp_path, "/groups/klannon", pid, start,
                             30000000000, 10000000000, 1, stop_event(),
                             path.stat().st_ino, marker.stat().st_ino)
    assert signals == []
    assert json.loads(marker.read_text())["reason"] == "critical_free_space"


def test_staging_guard_missing_telemetry_file_does_not_signal(tmp_path, monkeypatch, capsys):
    path = tmp_path / "owned"
    path.mkdir()
    pid = os.getpid()
    monkeypatch.setattr(staging_guard.os, "getppid", lambda: pid)
    start = staging_guard._process_start(pid)
    marker = path / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(path),
                                  "owner_uid": os.getuid(), "owner_pid": pid,
                                  "owner_start": start, "owner_inode": path.stat().st_ino}))
    signalled = []
    monkeypatch.setattr(staging_guard.os, "kill", lambda target, sig: signalled.append((target, sig)))
    monkeypatch.setattr(staging_guard, "_space", lambda *_: (_ for _ in ()).throw(FileNotFoundError("quota output vanished")))

    class stop_event:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, _):
            self.stopped = True
            return True

    staging_guard._watch(path, tmp_path, "/groups/klannon", pid, start,
                         20000000000, 10000000000, 1, stop_event(), path.stat().st_ino,
                         marker.stat().st_ino)
    assert signalled == []
    assert json.loads(marker.read_text())["status"] == "running"
    assert "quota output vanished" in capsys.readouterr().out


@pytest.mark.parametrize("failure_point", ("marker_read", "proc_read"))
def test_staging_guard_transient_identity_read_does_not_signal(
    tmp_path, monkeypatch, capsys, failure_point
):
    path = tmp_path / "owned"
    path.mkdir()
    pid = os.getpid()
    monkeypatch.setattr(staging_guard.os, "getppid", lambda: pid)
    start = staging_guard._process_start(pid)
    marker = path / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(path),
                                  "owner_uid": os.getuid(), "owner_pid": pid,
                                  "owner_start": start, "owner_inode": path.stat().st_ino}))
    original = (staging_guard._verify_owned_tree if failure_point == "marker_read"
                else staging_guard._process_start)
    calls = SimpleNamespace(count=0)
    def transient(*args):
        calls.count += 1
        if calls.count == 1:
            if failure_point == "marker_read":
                raise PermissionError("marker read race")
            raise staging_guard.staging_guard_error("manager process identity cannot be read")
        return original(*args)
    monkeypatch.setattr(staging_guard,
                        "_verify_owned_tree" if failure_point == "marker_read" else "_process_start",
                        transient)
    monkeypatch.setattr(staging_guard, "_space", lambda *_: (0, 2000))
    signalled = []
    monkeypatch.setattr(staging_guard.os, "kill", lambda pid, sig: signalled.append((pid, sig)))
    class stop_event:
        polls = 0
        def is_set(self):
            return self.polls >= 1
        def wait(self, _):
            self.polls += 1
            return self.is_set()
    degraded = SimpleNamespace(value=False)
    degraded.set = lambda: setattr(degraded, "value", True)
    staging_guard._watch(path, tmp_path, "/groups/klannon", pid, start,
                         30000000000, 10000000000, 1, stop_event(), path.stat().st_ino,
                         marker.stat().st_ino, degraded)
    assert degraded.value and signalled == []
    assert json.loads(marker.read_text())["status"] == "running"
    assert ("marker read race" if failure_point == "marker_read" else
            "manager process identity cannot be read") in capsys.readouterr().out


def test_staging_cleanup_is_owned_and_preserves_ambiguous_files(tmp_path, monkeypatch):
    class fake_event:
        def set(self):
            pass
    class fake_watchdog:
        exitcode = 0
        def join(self, timeout):
            pass
        def is_alive(self):
            return False
    monkeypatch.setattr(staging_guard, "_local_readers", lambda _: False)
    owned = tmp_path / "owned"
    owned.mkdir()
    metadata = {"status": "running", "path": str(owned), "owner_uid": os.getuid(),
                "owner_pid": 99999999, "owner_start": "never",
                "owner_inode": owned.stat().st_ino}
    (owned / "guard.json").write_text(json.dumps(metadata))
    (owned / "out_1.p").write_bytes(b"unfinished")
    staging_guard.finish_guard(owned, owned.stat().st_ino, fake_event(), fake_watchdog(), True)
    with pytest.raises(staging_guard.staging_guard_error, match="unexpected"):
        staging_guard.cleanup_owned_staging(owned)
    assert (owned / "out_1.p").exists()
    assert json.loads((owned / "guard.json").read_text())["status"] == "cleanup_needed"
    clean = tmp_path / "clean"
    clean.mkdir()
    metadata.update(path=str(clean), owner_inode=clean.stat().st_ino)
    (clean / "guard.json").write_text(json.dumps(metadata))
    staging_guard.finish_guard(clean, clean.stat().st_ino, fake_event(), fake_watchdog(), True)
    staging_guard.cleanup_owned_staging(clean)
    assert not clean.exists()

    completed = tmp_path / "completed"
    completed.mkdir()
    metadata.update(path=str(completed), owner_inode=completed.stat().st_ino)
    (completed / "guard.json").write_text(json.dumps(metadata))
    temporary = completed / "tmpabcwq-tmp-"
    temporary.mkdir(mode=0o700)
    (temporary / "out_p_1.p").write_bytes(b"intermediate")
    staging_guard.finish_guard(completed, completed.stat().st_ino, fake_event(), fake_watchdog(), True)
    staging_guard.cleanup_owned_staging(completed)
    assert not completed.exists()

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    metadata.update(path=str(foreign), owner_inode=foreign.stat().st_ino)
    (foreign / "guard.json").write_text(json.dumps(metadata))
    temporary = foreign / "tmpabcwq-tmp-"
    temporary.mkdir(mode=0o700)
    (temporary / "out_p_1.p").write_bytes(b"owned")
    (temporary / "unrecognized.txt").write_bytes(b"preserve")
    staging_guard.finish_guard(foreign, foreign.stat().st_ino, fake_event(), fake_watchdog(), True)
    with pytest.raises(staging_guard.staging_guard_error, match="unknown or foreign"):
        staging_guard.cleanup_owned_staging(foreign)
    assert (temporary / "out_p_1.p").exists()
    assert (temporary / "unrecognized.txt").exists()
    assert json.loads((foreign / "guard.json").read_text())["status"] == "cleanup_needed"

    failed = tmp_path / "failed"
    failed.mkdir()
    metadata.update(path=str(failed), owner_inode=failed.stat().st_ino)
    (failed / "guard.json").write_text(json.dumps(metadata))
    (failed / "out_p_1.p").write_bytes(b"unfinished")
    staging_guard.finish_guard(failed, failed.stat().st_ino, fake_event(), fake_watchdog(), False)
    with pytest.raises(staging_guard.staging_guard_error, match="completed invocation"):
        staging_guard.cleanup_owned_staging(failed)
    assert (failed / "out_p_1.p").exists()


def test_staging_preflight_rejects_insufficient_free_space_and_collision(tmp_path, monkeypatch):
    monkeypatch.setattr(staging_guard, "_validate_base", lambda _: tmp_path)
    monkeypatch.setattr(staging_guard.os.path, "commonpath", lambda _: "/groups/klannon")
    for available_bytes in (9999999999, 10000000000):
        monkeypatch.setattr(staging_guard, "_space", lambda *_, value=available_bytes: (value, 2000))
        with pytest.raises(staging_guard.staging_guard_error, match="minimum free space"):
            staging_guard.start_guard(tmp_path, "attempt", "/groups/klannon",
                                      20000000000, 10000000000, 1)
    for free_inodes in (0, 999):
        monkeypatch.setattr(staging_guard, "_space", lambda *_, value=free_inodes:
                            (20000000000, value))
        with pytest.raises(staging_guard.staging_guard_error, match="inode reserve"):
            staging_guard.start_guard(tmp_path, "attempt", "/groups/klannon",
                                      20000000000, 10000000000, 1)
    assert not (tmp_path / "attempt").exists()
    monkeypatch.setattr(staging_guard, "_space", lambda *_: (20000000000, 2000))
    (tmp_path / "attempt").mkdir()
    with pytest.raises(FileExistsError):
        staging_guard.start_guard(tmp_path, "attempt", "/groups/klannon", 20000000000, 10000000000, 1)


def test_staging_preflight_rejects_unknown_space_and_invalid_ownership(tmp_path, monkeypatch):
    monkeypatch.setattr(staging_guard, "_validate_base", lambda _: tmp_path)
    monkeypatch.setattr(staging_guard.os.path, "commonpath", lambda _: "/groups/klannon")
    monkeypatch.setattr(staging_guard, "_space", lambda *_: (_ for _ in ()).throw(OSError("quota unavailable")))
    with pytest.raises(OSError, match="quota unavailable"):
        staging_guard.start_guard(tmp_path, "attempt", "/groups/klannon",
                                  20000000000, 10000000000, 1)
    assert not (tmp_path / "attempt").exists()
    with pytest.raises(staging_guard.staging_guard_error, match="quota root"):
        staging_guard.start_guard(tmp_path, "attempt", "/invalid",
                                  20000000000, 10000000000, 1)
    monkeypatch.setattr(staging_guard, "_validate_base", lambda _: (_ for _ in ()).throw(
        staging_guard.staging_guard_error("staging base ownership or write access is invalid")))
    with pytest.raises(staging_guard.staging_guard_error, match="ownership"):
        staging_guard.start_guard(tmp_path, "attempt", "/groups/klannon",
                                  20000000000, 10000000000, 1)
    assert not (tmp_path / "attempt").exists()


@pytest.mark.parametrize(("watchdog_exitcode", "reason"),
                         ((0, "before requested shutdown"), (1, "watchdog exited")))
def test_lost_watchdog_blocks_completion_without_signalling_manager(
    tmp_path, monkeypatch, watchdog_exitcode, reason
):
    monkeypatch.setattr(staging_guard, "_validate_base", lambda _: tmp_path)
    monkeypatch.setattr(staging_guard, "_space", lambda *_: (20000000000, 2000))
    signalled = []
    monkeypatch.setattr(staging_guard.os, "kill", lambda pid, sig: signalled.append((pid, sig)))
    class fake_event:
        def set(self):
            pass
        def is_set(self):
            return False
        def wait(self, timeout):
            return False
    class fake_process:
        checks = 0
        exitcode = watchdog_exitcode
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
        def join(self, timeout):
            pass
        def is_alive(self):
            self.checks += 1
            return self.checks == 1
    class fake_context:
        def Event(self):
            return fake_event()
        def Process(self, **kwargs):
            return fake_process(**kwargs)
    monkeypatch.setattr(staging_guard.multiprocessing, "get_context", lambda _: fake_context())
    state = staging_guard.start_guard(tmp_path, "attempt", None,
                                      20000000000, 10000000000, 1)
    with pytest.raises(staging_guard.staging_guard_error, match=reason):
        staging_guard.finish_guard(*state, succeeded=True)
    assert signalled == []
    assert json.loads((tmp_path / "attempt" / "guard.json").read_text())["status"] == "running"


def test_finish_guard_waits_for_slow_valid_quota_poll_and_acknowledges_shutdown(tmp_path):
    owned = tmp_path / "owned"
    owned.mkdir()
    pid = os.getpid()
    start = staging_guard._process_start(pid)
    marker = owned / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(owned),
                                  "owner_uid": os.getuid(), "owner_pid": pid,
                                  "owner_start": start, "owner_inode": owned.stat().st_ino}))

    class event:
        def __init__(self, value=False):
            self.value = value
        def set(self):
            self.value = True
        def is_set(self):
            return self.value

    class slow_watchdog:
        exitcode = 0
        owner_pid = pid
        owner_start = start
        marker_inode = marker.stat().st_ino
        shutdown_ack = event()
        telemetry_degraded = event()
        def join(self, timeout):
            self.join_timeout = timeout
            # A bounded in-flight quota call may consume 15 seconds; model its
            # successful completion without sleeping in the test.
            assert timeout > 15
            self.shutdown_ack.set()
        def is_alive(self):
            return False

    watcher = slow_watchdog()
    staging_guard.finish_guard(owned, owned.stat().st_ino, event(), watcher, True)
    assert watcher.join_timeout == 30
    completed = json.loads(marker.read_text())
    assert completed["status"] == "completed_pending_receipt"
    assert completed["telemetry_degraded"] is False


@pytest.mark.parametrize("watcher_state", ("missing_ack", "still_alive"))
def test_finish_guard_rejects_missing_ack_or_stuck_watchdog(tmp_path, watcher_state):
    owned = tmp_path / "owned"
    owned.mkdir()
    pid = os.getpid()
    start = staging_guard._process_start(pid)
    marker = owned / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(owned),
                                  "owner_uid": os.getuid(), "owner_pid": pid,
                                  "owner_start": start, "owner_inode": owned.stat().st_ino}))

    class event:
        def set(self):
            pass
        def is_set(self):
            return watcher_state != "missing_ack"

    class watcher:
        exitcode = 0
        owner_pid = pid
        owner_start = start
        marker_inode = marker.stat().st_ino
        shutdown_ack = event()
        telemetry_degraded = event()
        def join(self, timeout):
            assert timeout == 30
        def is_alive(self):
            return watcher_state == "still_alive"

    reason = "did not stop cleanly" if watcher_state == "still_alive" else "shutdown"
    with pytest.raises(staging_guard.staging_guard_error, match=reason):
        staging_guard.finish_guard(owned, owned.stat().st_ino, event(), watcher(), True)
    assert json.loads(marker.read_text())["status"] == "running"


def test_guard_marker_mutation_blocks_completion_and_cleanup(tmp_path, monkeypatch):
    owned = tmp_path / "owned"
    owned.mkdir()
    marker = owned / "guard.json"
    pid = os.getpid()
    monkeypatch.setattr(staging_guard.os, "getppid", lambda: pid)
    start = staging_guard._process_start(pid)
    metadata = {"status": "running", "path": str(owned), "owner_uid": os.getuid(),
                "owner_pid": pid, "owner_start": start, "owner_inode": owned.stat().st_ino}
    marker.write_text(json.dumps(metadata))
    metadata["owner_inode"] += 1
    marker.write_text(json.dumps(metadata))
    signalled = []
    monkeypatch.setattr(staging_guard.os, "kill", lambda pid, sig: signalled.append((pid, sig)))
    monkeypatch.setattr(staging_guard, "_space", lambda *_: (20000000000, 2000))
    class stop_event:
        def is_set(self):
            return False
    with pytest.raises(staging_guard.staging_guard_error, match="owner marker changed"):
        staging_guard._watch(owned, tmp_path, "/groups/klannon", pid, start,
                         30000000000, 10000000000, 1, stop_event(), owned.stat().st_ino,
                             marker.stat().st_ino)
    class fake_watchdog:
        exitcode = 1
        owner_pid = pid
        owner_start = start
        marker_inode = marker.stat().st_ino
        def join(self, timeout):
            pass
        def is_alive(self):
            return False
    with pytest.raises(staging_guard.staging_guard_error, match="owner marker changed"):
        staging_guard.finish_guard(owned, owned.stat().st_ino,
                                   SimpleNamespace(set=lambda: None), fake_watchdog(), True)
    with pytest.raises(staging_guard.staging_guard_error, match="completed invocation"):
        staging_guard.cleanup_owned_staging(owned)
    assert owned.exists() and signalled == []


def test_transient_telemetry_degradation_keeps_completed_marker_distinct(tmp_path):
    owned = tmp_path / "owned"
    owned.mkdir()
    marker = owned / "guard.json"
    metadata = {"status": "running", "path": str(owned), "owner_uid": os.getuid(),
                "owner_pid": os.getpid(),
                "owner_start": staging_guard._process_start(os.getpid()),
                "owner_inode": owned.stat().st_ino}
    marker.write_text(json.dumps(metadata))
    class fake_event:
        def set(self):
            pass
        def is_set(self):
            return True
    class fake_watchdog:
        exitcode = 0
        telemetry_degraded = fake_event()
        def join(self, timeout):
            pass
        def is_alive(self):
            return False
    staging_guard.finish_guard(owned, owned.stat().st_ino, fake_event(), fake_watchdog(), True)
    completed = json.loads(marker.read_text())
    assert completed["status"] == "completed_pending_receipt"
    assert completed["telemetry_degraded"] is True
    assert "reason" not in completed


def test_cleanup_refusal_does_not_relabel_source_receipt_as_failed():
    launcher = RUN_CR.read_text(encoding="utf-8")
    assert "source failed none staging_cleanup_needed" not in launcher
    assert '"source" "failed" "none" "staging_cleanup_needed"' not in launcher
    assert launcher.count("guarded staging cleanup requires review; source receipt remains valid") == 1
    assert launcher.index('source ready 0 source_child_completed') < launcher.index('complete_guarded_staging "${block_id}"')


def test_cleanup_accepts_more_than_100000_recognized_files_without_materializing_them(tmp_path, monkeypatch):
    owned = tmp_path / "owned"
    owned.mkdir()
    temporary = owned / "tmpabcwq-tmp-"
    temporary.mkdir(mode=0o700)
    metadata = {"status": "completed_pending_receipt", "path": str(owned),
                "owner_uid": os.getuid(), "owner_pid": 99999999,
                "owner_start": "never", "owner_inode": owned.stat().st_ino}
    (owned / "guard.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(staging_guard, "_local_readers", lambda _: False)
    real_scandir = staging_guard.os.scandir
    real_unlink = Path.unlink
    removed = SimpleNamespace(count=0)
    class fake_entry:
        def __init__(self, index):
            self.name = f"out_p_{index}.p"
            self.path = str(temporary / self.name)
        def stat(self, follow_symlinks=False):
            return SimpleNamespace(st_mode=stat.S_IFREG | 0o600,
                                   st_uid=os.getuid(), st_nlink=1)
    class fake_scan:
        def __enter__(self):
            return (fake_entry(index) for index in range(100001))
        def __exit__(self, *_):
            return False
    monkeypatch.setattr(staging_guard.os, "scandir", lambda path: fake_scan()
                        if os.fspath(path) == str(temporary) else real_scandir(path))
    def unlink(path, *args, **kwargs):
        if path.parent == temporary:
            removed.count += 1
        else:
            real_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", unlink)
    staging_guard.cleanup_owned_staging(owned)
    assert removed.count == 100001
    assert not owned.exists()
def test_processing_core_callback_only_sets_processing_category(monkeypatch):
    from coffea import processor
    from coffea.processor import work_queue_tools

    monkeypatch.syspath_prepend(str(RUN_DIRECTORY))
    spec = importlib.util.spec_from_file_location("run_analysis_processing_core_test", RUN_ANALYSIS)
    run_analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_analysis)
    calls = []
    class fake_queue:
        executor = SimpleNamespace(cores=None, memory=None, disk=None, gpus=None,
                                   resource_monitor="measure", resources_mode="auto",
                                   fast_terminate_workers=0)
        def tune(self, *_):
            pass
        def enable_monitoring(self, *, watchdog):
            assert watchdog is False
        def specify_category_max_resources(self, category, resources):
            calls.append((category, dict(resources)))
        def specify_category_mode(self, *_):
            pass
    queue = fake_queue()
    work_queue_tools.CoffeaWQ._declare_resources(queue)
    assert calls == [(category, {}) for category in
                     ("default", "preprocessing", "processing", "accumulating")]
    assert processor.WorkQueueExecutor.__dataclass_fields__["custom_init"].default is None
    run_analysis._request_one_processing_core(queue)
    assert calls[-1] == ("processing", {"cores": 1})
    assert calls[:4] == [(category, {}) for category in
                         ("default", "preprocessing", "processing", "accumulating")]


def test_accumulating_minimum_composes_after_category_declarations(monkeypatch):
    from coffea.processor import work_queue_tools

    monkeypatch.syspath_prepend(str(RUN_DIRECTORY))
    spec = importlib.util.spec_from_file_location("run_analysis_accumulating_test", RUN_ANALYSIS)
    run_analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_analysis)
    calls = []

    class fake_queue:
        executor = SimpleNamespace(cores=None, memory=None, disk=None, gpus=None,
                                   resource_monitor="measure", resources_mode="auto",
                                   fast_terminate_workers=0)
        def tune(self, *_):
            pass
        def enable_monitoring(self, *, watchdog):
            assert watchdog is False
        def specify_category_max_resources(self, category, resources):
            calls.append(("max", category, dict(resources)))
        def specify_category_min_resources(self, category, resources):
            calls.append(("min", category, dict(resources)))
        def specify_category_mode(self, *_):
            pass

    queue = fake_queue()
    work_queue_tools.CoffeaWQ._declare_resources(queue)
    declaration_count = len(calls)
    run_analysis._configure_wq_categories(
        queue, processing_cores=1, accumulating_memory_mb=80000, accumulating_cores=2,
    )
    assert calls[declaration_count:] == [
        ("max", "processing", {"cores": 1}),
        ("min", "accumulating", {"memory": 80000, "cores": 2}),
    ]


def test_selected_run3_and_native_run2_own_distinct_locks(tmp_path, backend):
    prepared = launch(tmp_path, backend, "--selected-fragment", "unknown",
                      profile="run2_run3_prod")
    assert prepared.returncode != 0 and "found 0" in prepared.stderr
    root = tmp_path / "output"
    states = {s: root / s / f".{s}_prod_campaign_state_v9.json" for s in ("run2", "run3")}
    run2_before = states["run2"].read_bytes()
    old_summaries = {root / f"campaign_summary.{suffix}": b"previous summary\n"
                     for suffix in ("tsv", "md")}
    for path, data in old_summaries.items():
        path.write_bytes(data)

    def command(scope):
        return [str(RUN_CR), "--production-profile", f"{scope}_prod", "--output-dir",
                str(root / scope), "--campaign-tag", f"synthetic_{scope}", "--env-file",
                str(ARCHIVE), "--resume", "--selected-fragment", f"{scope}_prod_a"]

    combined = [str(RUN_CR), "--production-profile", "run2_run3_prod", "--output-dir",
                str(root), "--campaign-tag", "synthetic", "--env-file", str(ARCHIVE),
                "--resume", "--selected-fragment", "run3_prod_a"]
    with gated_process(tmp_path, dict(backend, CAMPAIGN_GATE_UNIT="run3_prod_a"),
                       combined, "run3_gate") as (parent, child_pid, gate):
        run2_lock, run3_lock = (component_lock(root, s) for s in ("run2", "run3"))
        assert states["run2"].read_bytes() == run2_before
        assert lock_is_free(run2_lock) and not lock_is_free(run3_lock)
        assert str(run2_lock) not in process_fd_targets(parent.pid)
        # Walk the actual backend -> selected component -> combined parent chain.
        pid = child_pid
        observed_selected = False
        while pid != parent.pid:
            targets = process_fd_targets(pid)
            assert str(run2_lock) not in targets
            assert str(run3_lock) in targets
            environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            assert not any(item.startswith(b"CAMPAIGN_LOCK_FD=") for item in environment)
            argv = Path(f"/proc/{pid}/cmdline").read_bytes()
            observed_selected |= b"run_cr.sh" in argv and b"run3_prod" in argv
            pid = int(Path(f"/proc/{pid}/status").read_text().split("PPid:\t")[1].splitlines()[0])
        assert observed_selected
        run3_active = states["run3"].read_bytes()
        contender = subprocess.run(command("run3"), cwd=RUN_DIRECTORY, env=backend,
                                   capture_output=True, text=True, timeout=300)
        assert contender.returncode != 0 and "writer lock is held" in contender.stderr
        assert states["run3"].read_bytes() == run3_active
        with gated_process(tmp_path, dict(backend, CAMPAIGN_GATE_UNIT="run2_prod_a"),
                           command("run2"), "run2_gate") as (run2, run2_pid, _):
            assert parent.poll() is None and run2.poll() is None
            assert not lock_is_free(run2_lock) and not lock_is_free(run3_lock)
            assert str(run2_lock) in process_fd_targets(run2_pid)
            assert str(run3_lock) not in process_fd_targets(run2_pid)
            assert states["run3"].read_bytes() == run3_active
        assert run2.returncode == 0
        assert lock_is_free(run2_lock) and not lock_is_free(run3_lock)
        assert states["run3"].read_bytes() == run3_active
        run2_after = states["run2"].read_bytes()
        # Contention introduced after admission must defer only the parent report.
        with manifest.campaign_lock(states["run2"]):
            with (gate / "release").open("w") as release:
                release.write("release\n")
            assert parent.wait(timeout=300) == 0
            assert "derived combined summary deferred" in (gate / "stderr").read_text()
            assert all(path.read_bytes() == data for path, data in old_summaries.items())
        assert states["run2"].read_bytes() == run2_after
    assert lock_is_free(run2_lock) and lock_is_free(run3_lock)
    assert manifest.overall_status(manifest.load_state(states["run3"])["units"][0]) == "success"


@pytest.mark.parametrize("parent_exits", (False, True))
def test_sibling_admission_closes_before_child_and_parent_exit(tmp_path, parent_exits):
    root = tmp_path / "run2"
    root.mkdir()
    source = RUN_CR.read_text()
    admission = source.split('      (\n        sibling_scope=run2', 1)[1].split(
        '\n    if "$0" --production-profile "${!selected_profile_variable}"', 1
    )[0]
    admission = '      (\n        sibling_scope=run2' + admission.rsplit('\n    fi', 1)[0]
    sibling = root / ".run2_prod_campaign.lock"
    own = tmp_path / ".run3_prod_campaign.lock"
    read_fd, write_fd = os.pipe()
    child = ('exec {own_fd}>>"$1"; flock -n "$own_fd" || exit 20; '
             'printf "ready\\n"; read -r -u "$2" release; printf "child_done\\n"')
    setup = (f"selected_scope=run3\nrun2_root={shlex.quote(str(root))}\n"
             "run2_profile=run2_prod\nrun2_state=unused\nscript_dir=unused\n"
             "python() { printf 'planned\\n'; }\n")
    invocation = shlex.join(["/bin/bash", "--noprofile", "--norc", "-c", child,
                             "fd_child", str(own), str(read_fd)])
    tail = invocation + (" &\nexit 0" if parent_exits else '\nprintf "parent_alive\\n"; read -r -u ' + str(read_fd) + ' release')
    process = subprocess.Popen(["/bin/bash", "-c", "set -euo pipefail\n" + setup + admission + "\n" + tail],
                               pass_fds=(read_fd,), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    os.close(read_fd)
    try:
        assert select.select([process.stdout], [], [], 30)[0]
        assert process.stdout.readline() == "ready\n"
        assert lock_is_free(sibling) and not lock_is_free(own)
        if parent_exits:
            assert process.wait(timeout=30) == 0
        os.write(write_fd, b"release\n")
        assert process.stdout.readline() == "child_done\n"
        if not parent_exits:
            assert process.stdout.readline() == "parent_alive\n"
            assert process.poll() is None
            assert lock_is_free(sibling) and lock_is_free(own)
            os.write(write_fd, b"release\n")
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0 and not stderr and not stdout
        assert lock_is_free(sibling) and lock_is_free(own)
    finally:
        os.close(write_fd)


@pytest.mark.parametrize("scope", ("run2", "run3"))
def test_combined_summary_contention_defers_both_and_releases(tmp_path, scope):
    command = summary_shell(tmp_path)
    old = {tmp_path / f"campaign_summary.{suffix}": b"old\n" for suffix in ("tsv", "md")}
    for path, data in old.items():
        path.write_bytes(data)
    with component_lock(tmp_path, scope).open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0
        assert f"component-lock contention ({scope})" in result.stderr
        assert all(path.read_bytes() == data for path, data in old.items())
        other = "run3" if scope == "run2" else "run2"
        assert lock_is_free(component_lock(tmp_path, other))
    assert all(lock_is_free(component_lock(tmp_path, s)) for s in ("run2", "run3"))


@pytest.mark.parametrize("scope", ("run2", "run3"))
@pytest.mark.parametrize("damage", ("symlink", "open", "flock", "state", "io"))
def test_combined_summary_real_errors_propagate_in_conditional_context(tmp_path, scope, damage):
    extra = ""
    command = summary_shell(tmp_path)
    lock = component_lock(tmp_path, scope)
    old_tsv = tmp_path / "campaign_summary.tsv"
    old_tsv.write_text("old\n")
    if damage == "symlink":
        lock.symlink_to(tmp_path / "outside")
    elif damage == "open":
        lock.mkdir()
    elif damage == "flock":
        extra = f'''flock() {{
  if [[ "$(readlink /proc/$BASHPID/fd/"${{@: -1}}")" == {shlex.quote(str(lock))} ]]; then
    return 74
  fi
  command flock "$@"
}}'''
    elif damage == "state":
        (tmp_path / scope / f".{scope}_prod_campaign_state_v9.json").write_text("invalid json")
    else:
        # Exercise both atomic replacements; failure of the second can leave the
        # first updated. The maintained summary pair is deliberately nontransactional.
        target = tmp_path / ("campaign_summary.tsv" if scope == "run2" else "campaign_summary.md")
        if target.exists():
            target.unlink()
        target.mkdir()
    command = summary_shell(tmp_path, extra,
                            'if publish_combined_summary; then exit 0; else exit "$?"; fi')
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert result.returncode != 0 and result.stderr
    assert "deferred" not in result.stderr
    if damage == "flock":
        assert result.returncode == 74
    if damage != "io":
        assert old_tsv.read_text() == "old\n"
    for other in ("run2", "run3"):
        if other != scope or damage not in ("symlink", "open"):
            assert lock_is_free(component_lock(tmp_path, other))


def test_combined_summary_success_releases_both_locks(tmp_path):
    result = subprocess.run(summary_shell(tmp_path), capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "completed_with_component_blockers" in (tmp_path / "campaign_summary.md").read_text()
    assert "not_attempted" in (tmp_path / "campaign_summary.tsv").read_text()
    assert all(lock_is_free(component_lock(tmp_path, s)) for s in ("run2", "run3"))


def test_competing_combined_summary_writers_share_order_and_do_not_interleave(tmp_path, backend):
    extra = r'''
flock() {
  local result=0
  command flock "$@" || result=$?
  printf '%s %s %s\n' "$writer" "$(readlink /proc/$BASHPID/fd/"${@: -1}")" "$result" >> "$output_dir/order"
  return "$result"
}
python() {
  printf '%s enter\n' "$writer" >> "$output_dir/events"
  printf '%s\n' "$writer" > "$output_dir/campaign_summary.tsv"
  if [[ "$writer" == first ]]; then
    printf '%s\n' "$BASHPID" > "$CAMPAIGN_GATE_ROOT/ready"
    read -r release < "$CAMPAIGN_GATE_ROOT/release"
  fi
  printf '%s\n' "$writer" > "$output_dir/campaign_summary.md"
  printf '%s exit\n' "$writer" >> "$output_dir/events"
}
'''
    command = summary_shell(tmp_path, "writer=first\n" + extra)
    with gated_process(tmp_path, backend, command, "summary_gate") as (first, _, _):
        assert all(not lock_is_free(component_lock(tmp_path, s)) for s in ("run2", "run3"))
        second = subprocess.run(summary_shell(tmp_path, "writer=second\n" + extra),
                                capture_output=True, text=True, timeout=30)
        assert second.returncode == 0 and "contention (run2)" in second.stderr
        assert (tmp_path / "events").read_text().splitlines() == ["first enter"]
        assert (tmp_path / "campaign_summary.tsv").read_text() == "first\n"
        assert not (tmp_path / "campaign_summary.md").exists()
    assert first.returncode == 0
    third = subprocess.run(summary_shell(tmp_path, "writer=third\n" + extra),
                           capture_output=True, text=True, timeout=30)
    assert third.returncode == 0, third.stderr
    assert (tmp_path / "events").read_text().splitlines() == [
        "first enter", "first exit", "third enter", "third exit"]
    assert (tmp_path / "order").read_text().splitlines() == [
        f"first {component_lock(tmp_path, 'run2')} 0",
        f"first {component_lock(tmp_path, 'run3')} 0",
        f"second {component_lock(tmp_path, 'run2')} 75",
        f"third {component_lock(tmp_path, 'run2')} 0",
        f"third {component_lock(tmp_path, 'run3')} 0"]
    assert all((tmp_path / f"campaign_summary.{s}").read_text() == "third\n" for s in ("tsv", "md"))
    assert all(lock_is_free(component_lock(tmp_path, s)) for s in ("run2", "run3"))


@pytest.mark.parametrize("ambiguous", (False, True))
def test_combined_incomplete_resume_keeps_run2_before_run3(tmp_path, backend, ambiguous):
    backend["CAMPAIGN_FAIL_UNIT"] = "run2_prod_a"
    first = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a",
                   profile="run2_run3_prod")
    assert first.returncode != 0
    before = combined_states(tmp_path)
    if ambiguous:
        path = tmp_path / "output/run2/.run2_prod_campaign_state_v9.json"
        state = manifest.load_state(path)
        record = state["units"][0]["stages"]["source"]
        record["status"] = "running"
        record["attempts"][0].update(result="running", ended_at_utc=None, exit_code=None)
        manifest.atomic_json(path, state)
        Path(state["units"][0]["outputs"]["source"]).write_bytes(b"ambiguous output")
    else:
        backend.pop("CAMPAIGN_FAIL_UNIT")
    result = launch(tmp_path, backend, "--resume", profile="run2_run3_prod")
    after = combined_states(tmp_path)
    order = (tmp_path / "source_order.log").read_text().splitlines()
    if ambiguous:
        assert result.returncode != 0 and "Run 3 was not started" in result.stderr
        assert after[1] == before[1]
        assert order == ["run2_prod_a"]
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert order == ["run2_prod_a"] + [f"{scope}_prod_{letter}" for scope in ("run2", "run3") for letter in "abcde"]
        assert all(manifest.classify(state) == "success" for state in after)
        assert all(old["plan"] == new["plan"] for old, new in zip(before, after))
