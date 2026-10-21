"""Run one command under a fail-closed Linux node-memory pressure guard.

The supervisor owns the evidence file and is outside the child's new session.
One invocation runs one child only; callers must stop their campaign on any
result other than a successful normal completion.

CLI: python -m topeft.modules.node_memory_guard --hard-limit-bytes N
     --evidence-path PATH -- COMMAND [ARG ...]
The evidence path must be absent. JSON-lines records are fsynced before any
signal. Exit 124 means threshold kill, 125 metric failure, and 126 guard
failure; otherwise the child's exit status is returned.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


DEFAULT_MEMINFO_PATH = Path("/proc/meminfo")
DEFAULT_SAMPLING_INTERVAL_SECONDS = 0.5


def read_node_pressure(meminfo_path=DEFAULT_MEMINFO_PATH):
    """Return the authoritative node-pressure sample in bytes."""
    values = {}
    with Path(meminfo_path).open(encoding="ascii") as source:
        for line in source:
            parts = line.split()
            if not parts or parts[0] not in {"MemTotal:", "MemAvailable:"}:
                continue
            if len(parts) != 3 or parts[2] != "kB" or parts[0] in values:
                raise ValueError(f"Invalid meminfo field: {line.strip()!r}")
            values[parts[0]] = int(parts[1])
    if set(values) != {"MemTotal:", "MemAvailable:"}:
        raise ValueError("MemTotal and MemAvailable are both required")
    total_kb, available_kb = values["MemTotal:"], values["MemAvailable:"]
    if total_kb <= 0 or not 0 <= available_kb <= total_kb:
        raise ValueError("Invalid MemTotal/MemAvailable relationship")
    return {
        "mem_total_kb": total_kb,
        "mem_available_kb": available_kb,
        "pressure_used_bytes": (total_kb - available_kb) * 1024,
    }


def _read_process_identity(pid):
    """Read group, session, state and start ticks without trusting a PID alone."""
    stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    closing_paren = stat_text.rfind(")")
    if closing_paren < 0:
        raise ValueError(f"Invalid /proc/{pid}/stat")
    fields = stat_text[closing_paren + 2:].split()
    if len(fields) < 20:
        raise ValueError(f"Incomplete /proc/{pid}/stat")
    return {
        "pid": pid,
        "state": fields[0],
        "pgid": int(fields[2]),
        "session": int(fields[3]),
        "start_ticks": int(fields[19]),
    }


def _child_exited_without_reaping(pid):
    return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _kill_owned_group(pid, recorded_identity):
    """Recheck the unreaped session leader immediately before group SIGKILL."""
    current_identity = _read_process_identity(pid)
    stable_fields = ("pid", "pgid", "session", "start_ticks")
    if (
        any(current_identity[key] != recorded_identity[key] for key in stable_fields)
        or current_identity["state"] in {"Z", "X"}
        or recorded_identity["pgid"] != pid
        or recorded_identity["session"] != pid
        or recorded_identity["pgid"] == os.getpgrp()
    ):
        raise RuntimeError("Child process-group identity changed; refusing to signal")
    # The child is not reaped until after this call. Its PID cannot be reused
    # between the identity check and killpg.
    os.killpg(recorded_identity["pgid"], signal.SIGKILL)


def _write_event(handle, event):
    handle.write(json.dumps(event, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _persist_or_stop(evidence, event, child, identity):
    """Stop an identified child if the supervisor cannot persist evidence."""
    try:
        _write_event(evidence, event)
    except OSError:
        try:
            _kill_owned_group(child.pid, identity)
        except (OSError, RuntimeError, ValueError):
            pass  # An unverified target must never be signaled.
        else:
            child.wait()
        raise


def run_guarded(
    command, *, hard_limit_bytes, evidence_path,
    meminfo_path=DEFAULT_MEMINFO_PATH,
    sampling_interval_seconds=DEFAULT_SAMPLING_INTERVAL_SECONDS,
):
    """Supervise one child; persist threshold/failure evidence before signaling."""
    if not command or hard_limit_bytes <= 0 or sampling_interval_seconds <= 0:
        raise ValueError("A command, positive threshold, and positive interval are required")
    evidence_path = Path(evidence_path)
    # Exclusive creation prevents overwriting evidence from a previous unit.
    with evidence_path.open("x", encoding="utf-8") as evidence:
        try:
            child = subprocess.Popen(list(command), start_new_session=True, close_fds=True)
        except OSError as exc:
            _write_event(evidence, {"event": "launch_failure", "error": str(exc)})
            raise
        try:
            identity = _read_process_identity(child.pid)
        except (OSError, ValueError) as exc:
            if not _child_exited_without_reaping(child.pid):
                # Popen still owns this unreaped PID, even without a readable
                # /proc identity. Do not signal a process group here.
                os.kill(child.pid, signal.SIGKILL)
            child.wait()
            _write_event(evidence, {"event": "identity_failure", "pid": child.pid,
                                    "error": str(exc)})
            raise RuntimeError("Could not establish child identity") from exc
        if identity["pgid"] != child.pid or identity["session"] != child.pid:
            if not _child_exited_without_reaping(child.pid):
                os.kill(child.pid, signal.SIGKILL)
            child.wait()
            _write_event(evidence, {"event": "identity_failure", "identity": identity})
            raise RuntimeError("Child did not enter its own process group/session")
        _persist_or_stop(evidence, {"event": "start", "identity": identity,
                                    "hard_limit_bytes": hard_limit_bytes,
                                    "sampling_interval_seconds": sampling_interval_seconds},
                         child, identity)
        peak_pressure = None
        minimum_available = None
        while True:
            if _child_exited_without_reaping(child.pid):
                returncode = child.wait()
                result = {
                    "termination_kind": "normal", "returncode": returncode,
                    "signal": None, "peak_pressure_used_bytes": peak_pressure,
                    "minimum_mem_available_kb": minimum_available,
                    "pid": child.pid, "pgid": identity["pgid"],
                    "start_ticks": identity["start_ticks"],
                }
                _write_event(evidence, {"event": "result", **result})
                return result
            try:
                sample = read_node_pressure(meminfo_path)
            except (OSError, ValueError) as exc:
                termination_kind = "metric_failed"
                event = {"event": "metric_failure", "error": str(exc),
                         "identity": identity}
            else:
                peak_pressure = max(peak_pressure or 0, sample["pressure_used_bytes"])
                minimum_available = min(
                    minimum_available if minimum_available is not None else sample["mem_available_kb"],
                    sample["mem_available_kb"],
                )
                if sample["pressure_used_bytes"] < hard_limit_bytes:
                    time.sleep(sampling_interval_seconds)
                    continue
                termination_kind = "threshold_killed"
                event = {"event": "threshold", "sample": sample,
                         "hard_limit_bytes": hard_limit_bytes, "identity": identity}
            _persist_or_stop(evidence, event, child, identity)
            try:
                _kill_owned_group(child.pid, identity)
            except (OSError, RuntimeError, ValueError) as exc:
                _write_event(evidence, {"event": "identity_failure", "error": str(exc),
                                        "identity": identity})
                raise RuntimeError("Refusing to signal unverified child group") from exc
            returncode = child.wait()
            result = {
                "termination_kind": termination_kind, "returncode": returncode,
                "signal": signal.SIGKILL, "peak_pressure_used_bytes": peak_pressure,
                "minimum_mem_available_kb": minimum_available,
                "pid": child.pid, "pgid": identity["pgid"],
                "start_ticks": identity["start_ticks"],
            }
            _write_event(evidence, {"event": "result", **result})
            return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-limit-bytes", type=int, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("--meminfo-path", type=Path, default=DEFAULT_MEMINFO_PATH)
    parser.add_argument("--sampling-interval-seconds", type=float,
                        default=DEFAULT_SAMPLING_INTERVAL_SECONDS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a child command is required after --")
    try:
        result = run_guarded(
            command, hard_limit_bytes=args.hard_limit_bytes,
            evidence_path=args.evidence_path, meminfo_path=args.meminfo_path,
            sampling_interval_seconds=args.sampling_interval_seconds,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(126, f"node-memory guard failed: {exc}\n")
    if result["termination_kind"] == "threshold_killed":
        return 124
    if result["termination_kind"] == "metric_failed":
        return 125
    return result["returncode"] if result["returncode"] >= 0 else 128 - result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
