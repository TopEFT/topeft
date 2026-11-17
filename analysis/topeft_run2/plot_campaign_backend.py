"""Execute or observe one immutable argv. No campaign plan, state or science.

The controller interprets execution evidence. The worker CLI consumes only a
single execution descriptor, never a public control wrapper or campaign plan.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from topeft.modules import node_memory_guard

REPO_ROOT = Path(__file__).resolve().parents[2]

class CampaignError(ValueError):
    """A bounded plan, state, receipt, or output contract failed."""


class ActiveChildUnresolved(CampaignError):
    """Terminal evidence is absent, so another heavy child cannot start."""


class ChildDefinitelyNotStarted(CampaignError):
    """Prelaunch setup or the first process creation failed without a child."""


class ChildDefinitelyTerminal(CampaignError):
    """The guard was reaped and its evidence proves the managed child terminal."""


def require(condition, message):
    if not condition:
        raise CampaignError(message)


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def error_description(error):
    try:
        return str(error)
    except BaseException:
        return type(error).__name__


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def exclusive_json(path, value):
    """Durably create a no-repeat boundary before its associated external effect."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def lifecycle_receipt_path(attempt_root):
    return attempt_root / "_metadata" / "lifecycle_receipt_v1.json"


def guard_terminal(path):
    require(path.is_file() and not path.is_symlink(), "guard terminal evidence missing")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError("guard evidence malformed") from error
    require(all(isinstance(row, dict) for row in rows), "guard evidence malformed")
    results = [row for row in rows if row.get("event") == "result"]
    require(len(results) == 1 and rows[-1] is results[0], "guard terminal evidence ambiguous")
    result = results[0]
    require({"termination_kind", "returncode", "signal", "peak_pressure_used_bytes",
             "minimum_mem_available_kb", "pid", "pgid", "start_ticks"} <= set(result)
            and isinstance(result["termination_kind"], str)
            and result["termination_kind"] in {"normal", "threshold_killed", "metric_failed"}
            and type(result["returncode"]) is int
            and all(type(result[key]) is int and result[key] > 0
                    for key in ("pid", "pgid", "start_ticks"))
            and (result["signal"] is None or type(result["signal"]) is int)
            and all(result[key] is None or type(result[key]) is int
                    for key in ("peak_pressure_used_bytes", "minimum_mem_available_kb")),
            "guard terminal result malformed")
    if result.get("observer") == "cgroup_v2":
        require(result.get("peak_pressure_used_bytes") is None
                and result.get("minimum_mem_available_kb") is None
                and (result.get("peak_cgroup_current_bytes") is None
                     or type(result["peak_cgroup_current_bytes"]) is int),
                "cgroup evidence is mislabeled as node pressure")
    return result


def run_guarded_child(argv, attempt_root, *, observer="node_pressure", hard_limit_bytes=245000000000):
    try:
        if observer == "cgroup_v2":
            sample = node_memory_guard.observe_memory(observer)
            require(0 < hard_limit_bytes <= sample["cgroup_limit_bytes"],
                    "hard limit exceeds allocation memory.max")
        metadata = attempt_root / "_metadata"
        metadata.mkdir(mode=0o700, exist_ok=True)
        guard_path = metadata / "guard.jsonl"
        log_path = metadata / "renderer.log"
        guard = [sys.executable, "-m", "topeft.modules.node_memory_guard",
                 "--hard-limit-bytes", str(hard_limit_bytes), "--sampling-interval-seconds", "0.5",
                 "--observer", observer,
                 "--evidence-path", str(guard_path), "--", *argv]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO_ROOT) + (":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
        environment["PYTHON_BIN"] = sys.executable
        with log_path.open("xb"):
            pass
        guard_process = subprocess.Popen(guard, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         env=environment)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise ChildDefinitelyNotStarted(
            f"guard process was not created: {error_description(error)}") from error
    try:
        tee_process = subprocess.Popen(["tee", "-a", str(log_path)],
                                       stdin=guard_process.stdout, env=environment)
    except Exception as error:
        try:
            guard_terminal(guard_path)
        except Exception:
            raise ActiveChildUnresolved(
                f"tee setup failed with unresolved child: {error_description(error)}") from error
        raise ChildDefinitelyTerminal(
            f"tee setup failed after guard terminal: {error_description(error)}") from error
    try:
        guard_process.stdout.close()
        tee_result = tee_process.wait()
        guard_result = guard_process.wait()
        terminal = guard_terminal(guard_path)
    except Exception as error:
        try:
            guard_terminal(guard_path)
        except Exception:
            raise ActiveChildUnresolved(
                f"guard terminal lifecycle is unresolved: {error_description(error)}") from error
        raise ChildDefinitelyTerminal(
            f"guard terminal but supervisor failed: {error_description(error)}") from error
    try:
        child_result = terminal.get("returncode")
        if terminal.get("termination_kind") != "normal":
            child_result = None
        lifecycle_path = lifecycle_receipt_path(attempt_root)
        require(not lifecycle_path.exists() and not lifecycle_path.is_symlink(),
                "lifecycle receipt already exists")
        atomic_json(lifecycle_path, {
            "schema_version": 1, "attempt_id": attempt_root.name,
            "command_identity": identity(argv), "private_unit_root": str(attempt_root),
            "terminal_guard_evidence": terminal, "guard_evidence_sha256": sha256_file(guard_path),
            "guard_result": guard_result, "child_result": child_result,
            "tee_result": tee_result, "ended_at_utc": utc_now(),
        })
    except Exception as error:
        raise ChildDefinitelyTerminal(
            f"terminal lifecycle postprocessing failed: {error_description(error)}") from error
    return guard_result, child_result, tee_result


def execution_descriptor(argv, attempt_root, contract):
    require(contract.get("version") == 1 and contract.get("argv") == argv
            and contract.get("backend") in {"local", "condor"}
            and contract.get("observer") in {"cgroup_v2", "node_pressure"}
            and type(contract.get("hard_limit_bytes")) is int
            and contract["hard_limit_bytes"] > 0,
            "invalid execution contract")
    require(len(argv) >= 2 and Path(argv[0]).is_absolute()
            and argv[1] == str(REPO_ROOT / "analysis/topeft_run2/make_cr_and_sr_plots.py"),
            "worker requires a direct renderer argv")
    body = {"version": 1, "argv": argv, "attempt_root": str(attempt_root),
            "observer": contract["observer"], "hard_limit_bytes": contract["hard_limit_bytes"]}
    return {**body, "token": identity(body)}


def run_direct(argv):
    """Execute one ordinary CLI renderer command, preserving its unguarded interface."""
    return subprocess.call(argv)


def execute(argv, attempt_root, contract):
    descriptor = execution_descriptor(argv, attempt_root, contract)
    if contract["backend"] == "local":
        return run_guarded_child(argv, attempt_root, observer=contract["observer"],
                                 hard_limit_bytes=contract["hard_limit_bytes"])
    from analysis.topeft_run2 import plot_campaign_condor
    metadata = attempt_root / "_metadata"
    try:
        metadata.mkdir(mode=0o700, exist_ok=True)
        path = metadata / "execution.json"
        exclusive_json(path, descriptor)
    except Exception as error:
        raise ActiveChildUnresolved("execution descriptor occupied or unavailable") from error
    plot_campaign_condor.submit(path, descriptor)
    raise ActiveChildUnresolved("Condor execution submitted; resume observes this execution only")


def observe(argv, attempt_root, contract, *, read_only=False):
    from analysis.topeft_run2 import plot_campaign_condor
    expected = execution_descriptor(argv, attempt_root, contract)
    path = attempt_root / "_metadata/execution.json"
    require(path.is_file() and not path.is_symlink(), "execution descriptor missing or unsafe")
    require(json.loads(path.read_text()) == expected, "execution descriptor changed")
    if read_only:
        return plot_campaign_condor.observe(path, expected, read_only=True)
    return plot_campaign_condor.observe(path, expected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args(argv)
    path = args.worker
    require(path.is_file() and not path.is_symlink(), "unsafe worker descriptor")
    descriptor = json.loads(path.read_text())
    attempt_root = Path(descriptor["attempt_root"])
    expected = execution_descriptor(descriptor["argv"], attempt_root, {
        "version": 1, "backend": "condor", "argv": descriptor["argv"],
        "observer": descriptor["observer"], "hard_limit_bytes": descriptor["hard_limit_bytes"],
    })
    require(path == attempt_root / "_metadata/execution.json" and descriptor == expected,
            "worker descriptor identity contradiction")
    # Condor may restart an evicted vanilla job. The persistent exclusive marker
    # prevents a second renderer even if the scheduler launches the worker again.
    exclusive_json(path.parent / "worker_started.json", {"token": descriptor["token"]})
    result = run_guarded_child(descriptor["argv"], attempt_root,
                               observer=descriptor["observer"],
                               hard_limit_bytes=descriptor["hard_limit_bytes"])
    return 0 if result == (0, 0, 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
