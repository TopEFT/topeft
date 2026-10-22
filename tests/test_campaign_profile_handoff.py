"""Synthetic launcher handoff checks without a producer or scheduler."""

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.topeft_run2 import campaign_manifest as manifest
from analysis.topeft_run2 import wq_staging_guard as staging_guard


RUN_DIRECTORY = Path(__file__).resolve().parents[1] / "analysis" / "topeft_run2"
RUN_CR = RUN_DIRECTORY / "run_cr.sh"
RUN_ANALYSIS = RUN_DIRECTORY / "run_analysis.py"
ARCHIVE = RUN_DIRECTORY / "topeft-envs" / "env_spec_1707ade2a411bb14.tar.gz"


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
  archive="$1"
  printf 'env_file: %s\\n' "$archive"
  printf 'env_file_sha256: %s\\n' "$(sha256sum "$archive" | cut -d ' ' -f1)"
  printf 'environment_fingerprint: %064d\\n' 9
  printf 'environment_validation_status: valid\\n'
  printf 'topcoffea_git_commit: %040d\\n' 8
  printf 'topcoffea_relevant_source_fingerprint: %064d\\n' 7
  exit 0
fi
if [[ "$action" == run_block ]]; then
  unit="$1"; archive="$2"; stage="$3"; source="$4"; downstream="$5"; logs="$6"
  mkdir -p "$logs"
  if [[ "$scenario" == source_fail_first && "$unit" == run2_prod_a ]]; then
    exit 23
  fi
  printf 'synthetic source\\n' > "$source"
  artifact="$source"
  source_sha=""
elif [[ "$action" == run_nonprompt ]]; then
  unit="$1"; source="$2"; downstream="$3"
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
        "CAMPAIGN_VALIDATION_SCENARIO": "success",
        "TMPDIR": str(tmp_path),
    })
    return environment


def launch(tmp_path, backend, *extra, profile="run2_prod", archive=ARCHIVE):
    return subprocess.run(
        [str(RUN_CR), "--production-profile", profile,
         "--output-dir", str(tmp_path / "output"), "--campaign-tag", "synthetic",
         "--env-file", str(archive), *extra],
        cwd=RUN_DIRECTORY, env=backend, capture_output=True, text=True, check=False,
    )


def state_path(tmp_path, profile="run2_prod"):
    return tmp_path / "output" / f".{profile}_campaign_state_v9.json"


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


def test_resume_accepts_chunksize_and_relocated_identical_archive(tmp_path, backend):
    first = launch(tmp_path, backend, "--selected-fragment", "run2_prod_a")
    assert first.returncode == 0, first.stdout + first.stderr
    relocated = tmp_path / "relocated.tar.gz"
    shutil.copyfile(ARCHIVE, relocated)
    assert hashlib.sha256(relocated.read_bytes()).hexdigest() == hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    second = launch(tmp_path, backend, "--resume", "--selected-fragment", "run2_prod_a",
                    "--chunksize", "98765", archive=relocated)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "Skipping verified completed unit" in second.stdout
    state = manifest.load_state(state_path(tmp_path))
    assert state["provenance"]["last_resolved"]["environment_file"] == str(relocated)
    assert len(state["units"][0]["stages"]["source"]["attempts"]) == 1


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
    measurements = iter((measurement_error, (20000000000, 0), (20000000000, 2000)))
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
    assert "available_bytes=20000000000 free_inodes=0" in output
    assert "available_bytes=20000000000" in output
    assert "measurement_failed" not in output
    assert json.loads(marker.read_text())["status"] == "running"


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
        assert "requesting SIGTERM for verified manager pid" in capsys.readouterr().out
    else:
        assert "reason" not in metadata
        capsys.readouterr()


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
    monkeypatch.setattr(staging_guard.os.path, "commonpath", lambda _: "/groups/klannon")
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
    state = staging_guard.start_guard(tmp_path, "attempt", "/groups/klannon",
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
