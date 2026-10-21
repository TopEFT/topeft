"""Prepare and submit shared-filesystem HTCondor datacard dispatch material.

The matrix runner owns topology, leases, reconciliation and completion. This
module serializes resource requests and dispatch, then owns the submit process.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

try:
    from . import datacard_matrix_runner as runner
except ImportError:  # Direct make_cards.py entrypoint.
    import datacard_matrix_runner as runner


_safe_path = re.compile(r"^/[A-Za-z0-9_./+-]+$")
_resource = re.compile(r"^[1-9][0-9]*(?:[KMGTP]B?|B)?$", re.IGNORECASE)


def submit_path(path: Path) -> str:
    value = str(path.resolve())
    if not _safe_path.fullmatch(value):
        raise runner.RunnerError("condor_configuration_error", "submit paths must be absolute and contain only letters, digits, _ . / + -")
    return value


def selected_units(manifest_path: Path, row_ids: list[str] | None) -> list[dict]:
    units = runner.derive_units(manifest_path)
    if row_ids is None:
        return units
    if not row_ids or len(row_ids) != len(set(row_ids)):
        raise runner.RunnerError("row_selection_error", "row IDs must be nonempty and unique")
    known = {unit["row_id"]: unit for unit in units}
    unknown = set(row_ids) - set(known)
    if unknown:
        raise runner.RunnerError("row_selection_error", f"unknown row IDs: {sorted(unknown)}")
    return [known[row_id] for row_id in row_ids]


def prepare_material(manifest_path: Path, destination: Path, *, request_cpus: int,
                     request_memory: str, request_disk: str,
                     max_retries: int, max_starts: int,
                     query_executable: str | None = None,
                     max_runtime: int | None = None,
                     row_ids: list[str] | None = None) -> dict:
    """Write fresh dispatch material with one item per canonical unit.

    Paths and runtime must be visible at the same absolute locations on workers.
    Memory/disk values use HTCondor submit units (or explicit unit suffixes).
    No resource values, concurrency cap, transfer model or retries are guessed.
    """
    if type(request_cpus) is not int or request_cpus <= 0:
        raise runner.RunnerError("condor_configuration_error", "request_cpus must be an explicit positive integer")
    if any(not isinstance(value, str) or not _resource.fullmatch(value) for value in (request_memory, request_disk)):
        raise runner.RunnerError("condor_configuration_error", "request_memory and request_disk require explicit positive quantities")
    if max_runtime is not None and (type(max_runtime) is not int or max_runtime <= 0):
        raise runner.RunnerError("condor_configuration_error", "max_runtime must be positive seconds when supplied")
    if type(max_retries) is not int or max_retries < 0 or type(max_starts) is not int or max_starts <= 0:
        raise runner.RunnerError("condor_configuration_error", "explicit max_retries >= 0 and max_starts > 0 are required")
    query_executable = query_executable or shutil.which("condor_q")
    if (not query_executable or not Path(query_executable).is_absolute()
            or not Path(query_executable).is_file() or not os.access(query_executable, os.X_OK)):
        raise runner.RunnerError("condor_configuration_error", "an absolute maintained condor_q executable is required")
    scheduler = {"query_executable": submit_path(Path(query_executable)),
                 "dispatch_directory": submit_path(destination)}
    units = selected_units(manifest_path, row_ids)
    manifest, _, _ = runner.resolve_unit(units[0])
    destination = Path(submit_path(destination))
    engine = Path(runner.__file__).resolve()
    python = manifest["runtime_contract"]["python_executable"]
    submit_path(engine)
    submit_path(Path(python))
    # Validate before creating output; exclusive mkdir provides no-overwrite.
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise runner.RunnerError("condor_output_exists", f"dispatch destination exists: {destination}") from exc
    (destination / "units").mkdir()
    (destination / "logs").mkdir()
    items = []
    for unit in units:
        unit_path = destination / "units" / (unit["row_id"] + ".json")
        runner.atomic_json(unit, unit_path, False)
        items.append(f"{unit['row_id']} {runner.hash_file(unit_path)} unreserved")
    (destination / "items.txt").write_text("\n".join(items) + "\n")
    worker = destination / "worker.sh"
    command = " ".join(shlex.quote(value) for value in (python, str(engine)))
    worker.write_text(
        '#!/bin/sh\nset -eu\n'
        + f'exec {command} --unit {shlex.quote(str(destination / "units"))}/"$1".json '
        + '--unit-integrity "$2" --backend condor --scheduler-id "$3" '
        + '--reservation-token "$4" '
        + f'--runner-integrity {runner.hash_file(engine)}\n'
    )
    worker.chmod(0o755)
    lines = [
        "universe = vanilla", f"executable = {worker}",
        'arguments = "$(row_id) $(unit_integrity) $(Cluster).$(Process) $(reservation_token)"',
        f"initialdir = {destination}", "should_transfer_files = NO",
        "getenv = false", f"request_cpus = {request_cpus}",
        f"request_memory = {request_memory}", f"request_disk = {request_disk}",
        f"output = {destination}/logs/$(row_id)__$(Cluster).$(Process).out",
        f"error = {destination}/logs/$(row_id)__$(Cluster).$(Process).err",
        f"log = {destination}/logs/$(row_id)__$(Cluster).$(Process).events.log",
        '+DatacardUnit = "$(row_id)"',
        '+DatacardParentIntegrity = "' + units[0]["parent_integrity"] + '"',
        '+DatacardReservation = "$(reservation_token)"',
        f"max_retries = {max_retries}", "success_exit_code = 0",
        f"requirements = MY.NumJobStarts < {max_starts}",
        f"on_exit_remove = NumJobStarts >= {max_starts}",
        f"periodic_remove = (JobStatus == 5) || (JobStatus == 1 && NumJobStarts >= {max_starts})",
    ]
    if max_runtime is not None:
        lines.append(f"allowed_execute_duration = {max_runtime}")
    lines.append(f"queue row_id, unit_integrity, reservation_token from {destination}/items.txt")
    # Publish the submit description last; failed preparation leaves no ready job.
    pending = destination / ".datacards.sub.pending"
    with pending.open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    runner.copy_once(pending, destination / "datacards.sub")
    pending.unlink()
    return {"submitted": False, "submit_file": str(destination / "datacards.sub"),
            "unit_ids": [unit["row_id"] for unit in units],
            "reservation_state": "unreserved_material_only",
            "scheduler": scheduler,
            "storage_model": "shared_filesystem", "receipt_authority": "matrix_runner"}


@contextmanager
def submission_transaction(manifest_path: Path, destination: Path, scheduler: dict, *,
                           row_ids: list[str] | None = None,
                           rescue_ambiguous: bool = False):
    """Bind prepared itemdata inside the maintained reservation transaction.

    The submit caller must mark_submission_started immediately before process
    creation, inherit transaction.lease_fds into the submit subprocess, and
    wait for it to terminate before exiting. Exceptions,
    timeout and nonzero exit after that marker preserve all reservations.
    """
    units = selected_units(manifest_path, row_ids)
    manifest, manifest_hash, _ = runner.resolve_unit(units[0])
    runner.validate_runtime(manifest["runtime_contract"])
    if rescue_ambiguous and (row_ids is None or len(units) != 1):
        raise runner.RunnerError("row_selection_error", "ambiguous rescue requires exactly one selected row")
    items = []
    for unit in units:
        unit_path = destination / "units" / (unit["row_id"] + ".json")
        if json.loads(unit_path.read_text()) != unit:
            raise runner.RunnerError("stale_dispatch_input", "prepared descriptor differs from canonical unit")
        items.append((unit["row_id"], runner.hash_file(unit_path)))
    rows = [runner.select_row(manifest, unit["row_id"]) for unit in units]
    with runner.reserve_condor(manifest, manifest_hash, rows, scheduler,
                               rescue_ambiguous=rescue_ambiguous) as transaction:
        tokens = {item["row_id"]: item["owner_id"] for item in transaction.reservations}
        (destination / "items.txt").write_text("".join(
            f"{row_id} {integrity} {tokens[row_id]}\n" for row_id, integrity in items))
        yield transaction


def submit_material(manifest_path: Path, destination: Path, scheduler: dict, *,
                    submit_executable: str | None = None,
                    batch_name: str | None = None,
                    timeout: float | None = None,
                    row_ids: list[str] | None = None,
                    rescue_ambiguous: bool = False) -> dict:
    """Submit prepared material while holding every row's reservation lease.

    A zero exit reports only command success. Scheduler ownership is later
    observed through each reservation token, never inferred from stdout.
    """
    submit_executable = submit_executable or shutil.which("condor_submit")
    if (not submit_executable or not Path(submit_executable).is_absolute()
            or not Path(submit_executable).is_file()
            or not os.access(submit_executable, os.X_OK)):
        raise runner.RunnerError("condor_configuration_error", "an absolute maintained condor_submit executable is required")
    if batch_name is not None and (not isinstance(batch_name, str) or not batch_name or "\x00" in batch_name):
        raise runner.RunnerError("condor_configuration_error", "batch_name must be nonempty and contain no NUL")
    argv = [submit_path(Path(submit_executable))]
    if batch_name is not None:
        argv.extend(["-batch-name", batch_name])
    argv.append(submit_path(destination / "datacards.sub"))
    pre_submit_snapshot = runner.scheduler_snapshot(scheduler)
    with submission_transaction(manifest_path, destination, scheduler,
                                row_ids=row_ids, rescue_ambiguous=rescue_ambiguous) as transaction:
        result = {"argv": argv, "start_time": datetime.now(timezone.utc).isoformat(),
                  "exit_status": None, "stdout": "", "stderr": "",
                  "pre_submit_scheduler_snapshot": pre_submit_snapshot}
        transaction.mark_submission_started()
        try:
            process = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True,
                                       pass_fds=transaction.lease_fds)
        except OSError as exc:
            # This exception came from process creation itself: no executable ran.
            transaction.record_definite_spawn_failure(exc)
            return {**result, "submitted": False,
                    "submission_outcome": "DEFINITE_SPAWN_FAILURE", "error": str(exc)}
        except KeyboardInterrupt as exc:
            # Popen may have created a child before the interruption.
            return {**result, "submitted": False,
                    "submission_outcome": "AMBIGUOUS_SUBMISSION_OUTCOME",
                    "error": type(exc).__name__ + " during process creation"}
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            result.update(exit_status=process.returncode, stdout=stdout, stderr=stderr)
        except (subprocess.TimeoutExpired, KeyboardInterrupt, OSError) as exc:
            result["error"] = type(exc).__name__ + ": " + str(exc)
            # Reap a known child before releasing its inherited row leases.
            try:
                process.kill()
                stdout, stderr = process.communicate()
                result.update(exit_status=process.returncode, stdout=stdout, stderr=stderr)
            except (OSError, KeyboardInterrupt) as cleanup_exc:
                result["cleanup_error"] = type(cleanup_exc).__name__ + ": " + str(cleanup_exc)
            return {**result, "submitted": False,
                    "submission_outcome": "AMBIGUOUS_SUBMISSION_OUTCOME"}
        if process.returncode == 0:
            result["reported_cluster_id"] = runner.reported_cluster_id(result["stdout"])
            observations = []
            for reservation in transaction.reservations:
                try:
                    observation = runner.query_reservation(reservation)
                    if observation["ownership"] == "ACTIVE":
                        runner.record_scheduler_observation(reservation, observation)
                    observations.append({"row_id": reservation["row_id"], **observation})
                except Exception as exc:
                    observations.append({"row_id": reservation["row_id"], "state": "UNKNOWN",
                                         "error_code": getattr(exc, "code", type(exc).__name__),
                                         "reason": str(exc)})
            result["post_submit_observations"] = observations
            return {**result, "submitted": True,
                    "submission_outcome": "SUBMIT_COMMAND_SUCCEEDED"}
        return {**result, "submitted": False,
                "submission_outcome": "AMBIGUOUS_SUBMISSION_OUTCOME"}
