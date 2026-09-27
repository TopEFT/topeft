#!/usr/bin/env python3
"""Run an opaque, prequalified datacard manifest without a shell intermediary."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from typing import Any

MANIFEST_SCHEMA = "topeft_datacard_matrix_v3"
RECEIPT_SCHEMA = "topeft_datacard_execution_receipt_v2"
OWNER_SCHEMA = "topeft_datacard_runner_owner_v1"
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOP_FIELDS = {"schema", "control_root", "lock_path", "runtime_contract", "rows"}
RUNTIME_FIELDS = {"contract_id", "python_executable", "make_cards_path", "fingerprints"}
FINGERPRINT_FIELDS = {"path", "sha256"}
ROW_FIELDS = {"row_id", "logical_row_id", "attempt_id", "era", "working_directory", "input_pkl", "output_root", "distribution", "physical_channels", "years", "missing_parton_path", "merge_report_path", "snapshot_directory", "log_path", "expected_output_paths", "producer_args"}
SNAPSHOTS = {"selected_wcs": "selectedWCs.txt", "scalings": "scalings-preselect.json", "merge_report": "merge_report.json"}


class RunnerError(Exception):
    def __init__(self, code: str, message: str, *, row_id: str | None = None):
        super().__init__(message)
        self.code, self.row_id = code, row_id


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_fields(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RunnerError("manifest_schema_error", f"{label} must be an object")
    missing, extra = sorted(allowed - set(value)), sorted(set(value) - allowed)
    if missing or extra:
        raise RunnerError("manifest_schema_error", f"{label} fields differ: missing={missing}, extra={extra}")
    return value


def text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RunnerError("manifest_schema_error", f"{label} must be a nonempty string")
    return value


def strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise RunnerError("manifest_schema_error", f"{label} must be a nonempty string list")
    return value


def absolute(value: Any, label: str) -> Path:
    path = Path(text(value, label))
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise RunnerError("manifest_schema_error", f"{label} must be an absolute normalized path")
    return path


def inside(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RunnerError("manifest_schema_error", f"{label} must be within control_root") from exc


def option_values(args: list[str], option: str) -> list[str]:
    positions = [index for index, value in enumerate(args) if value == option]
    if len(positions) != 1:
        raise RunnerError("manifest_schema_error", f"producer_args must contain {option} exactly once")
    values: list[str] = []
    for value in args[positions[0] + 1:]:
        if value.startswith("--"):
            break
        values.append(value)
    if not values:
        raise RunnerError("manifest_schema_error", f"producer_args option {option} needs a value")
    return values


def runtime_digest(contract: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_runtime(contract: Any, verify_files: bool = True) -> dict[str, Any]:
    contract = exact_fields(contract, RUNTIME_FIELDS, "runtime_contract")
    text(contract["contract_id"], "runtime_contract.contract_id")
    python = absolute(contract["python_executable"], "runtime_contract.python_executable")
    make_cards = absolute(contract["make_cards_path"], "runtime_contract.make_cards_path")
    fingerprints = contract["fingerprints"]
    if not isinstance(fingerprints, list) or not fingerprints:
        raise RunnerError("manifest_schema_error", "runtime_contract.fingerprints must be nonempty")
    seen: set[str] = set()
    for index, item in enumerate(fingerprints):
        item = exact_fields(item, FINGERPRINT_FIELDS, f"runtime_contract.fingerprints[{index}]")
        path = absolute(item["path"], f"runtime_contract.fingerprints[{index}].path")
        digest = text(item["sha256"], f"runtime_contract.fingerprints[{index}].sha256")
        if not SHA256_RE.fullmatch(digest) or str(path) in seen:
            raise RunnerError("manifest_schema_error", "runtime fingerprints require unique absolute paths and lowercase SHA256")
        seen.add(str(path))
    if verify_files:
        if not python.is_file() or not os.access(python, os.X_OK):
            raise RunnerError("runtime_contract_mismatch", f"python executable unavailable: {python}")
        if not make_cards.is_file():
            raise RunnerError("runtime_contract_mismatch", f"make_cards path unavailable: {make_cards}")
        for item in fingerprints:
            path = Path(item["path"])
            if not path.is_file() or hash_file(path) != item["sha256"]:
                raise RunnerError("runtime_contract_mismatch", f"runtime fingerprint mismatch: {path}")
    return contract


def validate_row(row: Any, index: int, control_root: Path) -> dict[str, Any]:
    row = exact_fields(row, ROW_FIELDS, f"rows[{index}]")
    row_id, attempt = text(row["row_id"], "row_id"), text(row["attempt_id"], "attempt_id")
    logical_row_id = text(row["logical_row_id"], "logical_row_id")
    if not all(IDENTIFIER_RE.fullmatch(value) for value in (row_id, logical_row_id, attempt)):
        raise RunnerError("manifest_schema_error", "row_id, logical_row_id and attempt_id must be portable identifiers")
    for key in ("era", "distribution"):
        text(row[key], key)
    for key in ("working_directory", "input_pkl", "output_root", "missing_parton_path", "merge_report_path", "snapshot_directory", "log_path"):
        absolute(row[key], key)
    channels, years = strings(row["physical_channels"], "physical_channels"), strings(row["years"], "years")
    outputs, args = strings(row["expected_output_paths"], "expected_output_paths"), strings(row["producer_args"], "producer_args")
    if len(set(channels)) != len(channels) or len(set(years)) != len(years) or len(set(outputs)) != len(outputs):
        raise RunnerError("manifest_schema_error", "row repeats a channel, year, or output path")
    for output in outputs:
        try:
            absolute(output, "expected_output_path").relative_to(Path(row["output_root"]))
        except ValueError as exc:
            raise RunnerError("manifest_schema_error", "expected outputs must be below output_root") from exc
    if {"--condor", "-C", "--merge-only", "--select-only"}.intersection(args):
        raise RunnerError("manifest_schema_error", "producer_args selects a non-row mode")
    if "--sr-registry" in args:
        raise RunnerError("manifest_schema_error", "v3 producer_args must not contain --sr-registry")
    required = {"--out-dir": [row["output_root"]], "--var-lst": [row["distribution"]], "--ch-lst": channels, "--year": years, "--miss-parton-file": [row["missing_parton_path"]], "--merge-report": [row["merge_report_path"]]}
    for option, expected in required.items():
        if option_values(args, option) != expected:
            raise RunnerError("manifest_schema_error", f"{option} differs from structured row field")
    inside(Path(row["snapshot_directory"]), control_root, "snapshot_directory")
    return row


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerError("manifest_read_error", str(exc)) from exc
    manifest = exact_fields(manifest, TOP_FIELDS, "manifest")
    if manifest["schema"] == "topeft_datacard_matrix_v2":
        raise RunnerError("manifest_schema_error", "Historical datacard matrix schema `topeft_datacard_matrix_v2` is not accepted by the current datacard workflow. Generate a new `topeft_datacard_matrix_v3` manifest with make_datacard_matrix_manifest.py.")
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise RunnerError("manifest_schema_error", f"schema must be {MANIFEST_SCHEMA}")
    root, lock = absolute(manifest["control_root"], "control_root"), absolute(manifest["lock_path"], "lock_path")
    inside(lock, root, "lock_path")
    validate_runtime(manifest["runtime_contract"], verify_files=False)
    if not isinstance(manifest["rows"], list) or not manifest["rows"]:
        raise RunnerError("manifest_schema_error", "rows must be nonempty")
    rows = [validate_row(row, index, root) for index, row in enumerate(manifest["rows"])]
    row_ids = [row["row_id"] for row in rows]
    attempts = [(row["row_id"], row["attempt_id"]) for row in rows]
    owned = [value for row in rows for value in [*row["expected_output_paths"], row["merge_report_path"], row["log_path"], row["snapshot_directory"]]]
    if len(set(row_ids)) != len(row_ids):
        raise RunnerError("manifest_schema_error", "duplicate row_id")
    if len(set(attempts)) != len(attempts):
        raise RunnerError("manifest_schema_error", "duplicate row/attempt identity")
    if len(set(owned)) != len(owned):
        raise RunnerError("manifest_schema_error", "row-owned paths must be unique")
    return manifest, hashlib.sha256(raw).hexdigest()


def receipt_path(manifest: dict[str, Any], row: dict[str, Any]) -> Path:
    return Path(manifest["control_root"]) / "receipts" / f"{row['row_id']}__{row['attempt_id']}.json"


def owner_path(manifest: dict[str, Any]) -> Path:
    return Path(manifest["control_root"]) / "runner_owner.json"


def snapshots(row: dict[str, Any]) -> dict[str, Path]:
    return {key: Path(row["snapshot_directory"]) / name for key, name in SNAPSHOTS.items()}


def resolved_argv(manifest: dict[str, Any], row: dict[str, Any]) -> list[str]:
    contract = manifest["runtime_contract"]
    return [contract["python_executable"], contract["make_cards_path"], row["input_pkl"], *row["producer_args"]]


def path_availability(path: Path, *, executable: bool = False) -> dict[str, Any]:
    regular_file = path.is_file()
    readable = regular_file and os.access(path, os.R_OK)
    executable_ok = not executable or (regular_file and os.access(path, os.X_OK))
    return {
        "path": str(path),
        "exists": path.exists(),
        "regular_file": regular_file,
        "readable": readable,
        "executable": executable_ok if executable else None,
        "available": regular_file and readable and executable_ok,
    }


def execution_input_availability(manifest: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    runtime = manifest["runtime_contract"]
    checks = {
        "working_directory": {
            "path": row["working_directory"],
            "exists": Path(row["working_directory"]).exists(),
            "directory": Path(row["working_directory"]).is_dir(),
            "available": Path(row["working_directory"]).is_dir(),
        },
        "input_pkl": path_availability(Path(row["input_pkl"])),
        "missing_parton_path": path_availability(Path(row["missing_parton_path"])),
        "python_executable": path_availability(Path(runtime["python_executable"]), executable=True),
        "make_cards_path": path_availability(Path(runtime["make_cards_path"])),
    }
    fingerprint_checks = []
    for item in runtime["fingerprints"]:
        path = Path(item["path"])
        available = path.is_file() and os.access(path, os.R_OK)
        observed_sha256 = hash_file(path) if available else None
        fingerprint_checks.append({
            "path": str(path),
            "available": available,
            "expected_sha256": item["sha256"],
            "observed_sha256": observed_sha256,
            "matches": available and observed_sha256 == item["sha256"],
        })
    checks["runtime_fingerprints"] = fingerprint_checks
    ready = all(check["available"] for key, check in checks.items() if key != "runtime_fingerprints") and all(item["matches"] for item in fingerprint_checks)
    return {"required": True, "checked": True, "ready": ready, "checks": checks}


def validate_row_inputs(row: dict[str, Any]) -> None:
    working_directory = Path(row["working_directory"])
    if not working_directory.is_dir():
        raise RunnerError("runtime_preflight_error", f"working directory unavailable: {working_directory}", row_id=row["row_id"])
    input_pkl = Path(row["input_pkl"])
    if not input_pkl.is_file() or not os.access(input_pkl, os.R_OK):
        raise RunnerError("runtime_preflight_error", f"input_pkl unavailable: {input_pkl}", row_id=row["row_id"])
    missing_parton = Path(row["missing_parton_path"])
    if not missing_parton.is_file() or not os.access(missing_parton, os.R_OK):
        raise RunnerError("runtime_preflight_error", f"missing_parton_path unavailable: {missing_parton}", row_id=row["row_id"])


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": hash_file(path)}


def valid_record(value: Any, path: Path) -> bool:
    return isinstance(value, dict) and set(value) == {"path", "size_bytes", "sha256"} and value.get("path") == str(path) and isinstance(value.get("size_bytes"), int) and isinstance(value.get("sha256"), str) and path.is_file() and path.stat().st_size == value["size_bytes"] and hash_file(path) == value["sha256"]


def validate_receipt(receipt: Any, manifest: dict[str, Any], manifest_hash: str, row: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(receipt, dict):
        return False, "receipt root is not an object"
    runtime = manifest["runtime_contract"]
    expected = {"schema": RECEIPT_SCHEMA, "manifest_sha256": manifest_hash, "row_id": row["row_id"], "attempt_id": row["attempt_id"], "resolved_argv": resolved_argv(manifest, row), "command_return_code": 0, "runtime_contract_id": runtime["contract_id"], "python_executable": runtime["python_executable"], "make_cards_path": runtime["make_cards_path"], "runtime_fingerprints": runtime["fingerprints"], "runtime_contract_digest": runtime_digest(runtime)}
    if any(receipt.get(key) != value for key, value in expected.items()):
        return False, "receipt contract does not match"
    if not all(isinstance(receipt.get(key), str) and receipt[key] for key in ("start_timestamp", "end_timestamp")):
        return False, "receipt timestamps missing"
    outputs = receipt.get("primary_outputs")
    if not isinstance(outputs, list) or len(outputs) != len(row["expected_output_paths"]) or any(not valid_record(item, Path(path)) for item, path in zip(outputs, row["expected_output_paths"])):
        return False, "primary output is missing, size-mismatched, or hash-mismatched"
    artifacts = {"merge_report": Path(row["merge_report_path"]), "selected_wcs_snapshot": snapshots(row)["selected_wcs"], "scalings_snapshot": snapshots(row)["scalings"], "merge_report_snapshot": snapshots(row)["merge_report"], "log": Path(row["log_path"])}
    if not isinstance(receipt.get("artifacts"), dict) or set(receipt["artifacts"]) != set(artifacts) or any(not valid_record(receipt["artifacts"][key], path) for key, path in artifacts.items()):
        return False, "control artifact is missing, size-mismatched, or hash-mismatched"
    return True, "receipt and byte-bound artifacts match"


def read_owner(manifest: dict[str, Any]) -> dict[str, Any] | None:
    try:
        owner = json.loads(owner_path(manifest).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return owner if isinstance(owner, dict) else None


def probe_lock(path: Path) -> tuple[bool | None, str | None]:
    if not path.exists():
        return False, None
    try:
        with path.open("rb") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True, None
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return False, None
    except FileNotFoundError:
        return False, None
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def classify(manifest: dict[str, Any], manifest_hash: str, row: dict[str, Any], held: bool | None = None, owner: dict[str, Any] | None = None) -> dict[str, Any]:
    receipt = receipt_path(manifest, row)
    base = {"row_id": row["row_id"], "attempt_id": row["attempt_id"], "receipt_path": str(receipt)}
    if receipt.exists():
        try:
            valid, reason = validate_receipt(json.loads(receipt.read_text()), manifest, manifest_hash, row)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            valid, reason = False, str(exc)
        return {**base, "status": "complete" if valid else "invalid_receipt", "reason": reason}
    evidence = [str(path) for path in [*[Path(path) for path in row["expected_output_paths"]], Path(row["merge_report_path"]), Path(row["log_path"]), *snapshots(row).values()] if path.exists()]
    if not evidence:
        return {**base, "status": "not_started", "reason": "no row-owned artifacts"}
    held = probe_lock(Path(manifest["lock_path"]))[0] if held is None else held
    owner = read_owner(manifest) if owner is None else owner
    active = held is True and isinstance(owner, dict) and owner.get("schema_version") == OWNER_SCHEMA and owner.get("manifest_sha256") == manifest_hash and owner.get("current_row_id") == row["row_id"] and owner.get("current_attempt_id") == row["attempt_id"]
    reason = "lock and owner identify current row" if active else "lock state is unknown; row-owned artifacts exist without valid receipt" if held is None else "row-owned artifacts exist without valid receipt"
    return {**base, "status": "active" if active else "interrupted_requires_external_reconciliation", "reason": reason, "existing_paths": evidence}


def read_only(mode: str, manifest: dict[str, Any], manifest_hash: str) -> dict[str, Any]:
    held, lock_probe_error = probe_lock(Path(manifest["lock_path"]))
    owner = read_owner(manifest)
    rows = [classify(manifest, manifest_hash, row, held, owner) for row in manifest["rows"]]
    if mode == "plan_only":
        for row, status in zip(manifest["rows"], rows):
            status["action"] = "skip" if status["status"] == "complete" else "execute" if status["status"] == "not_started" else "block"
            status["resolved_argv"] = resolved_argv(manifest, row)
            if status["status"] == "not_started":
                availability = execution_input_availability(manifest, row)
                status["execution_input_availability"] = availability
                if not availability["ready"]:
                    status["action"] = "block"
                    status["reason"] = "execution inputs are not launch-ready"
            else:
                status["execution_input_availability"] = {
                    "required": False,
                    "checked": False,
                    "ready": status["status"] == "complete",
                    "reason": "valid receipt permits skip without historical execution inputs" if status["status"] == "complete" else "row state blocks before execution-input checks",
                }
    result = {"schema": "topeft_datacard_matrix_read_only_result_v2", "mode": mode, "manifest_sha256": manifest_hash, "runtime_contract_digest": runtime_digest(manifest["runtime_contract"]), "lock_held": held, "lock_probe_error": lock_probe_error, "owner_metadata": owner, "mutated": False, "rows": rows}
    if mode == "plan_only":
        result["launch_ready"] = all(row["action"] in {"skip", "execute"} for row in rows)
    return result


def atomic_json(payload: dict[str, Any], destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and destination.exists():
        raise RunnerError("receipt_write_failure", f"destination exists: {destination}")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    try:
        json.loads(temporary.read_text())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    except FileExistsError as exc:
        raise RunnerError("receipt_write_failure", f"destination exists: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def copy_once(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise RunnerError("snapshot_failure", f"control source missing or empty: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RunnerError("snapshot_failure", f"snapshot exists: {destination}")
    with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        with source.open("rb") as input_handle:
            shutil.copyfileobj(input_handle, handle)
        handle.flush(); os.fsync(handle.fileno())
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise RunnerError("snapshot_failure", f"snapshot exists: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def owner_payload(manifest: dict[str, Any], manifest_hash: str, owner_started_at: str, row: dict[str, Any] | None = None) -> dict[str, Any]:
    runtime = manifest["runtime_contract"]
    return {"schema_version": OWNER_SCHEMA, "pid": os.getpid(), "hostname": socket.gethostname(), "started_at": owner_started_at, "manifest_sha256": manifest_hash, "lock_path": manifest["lock_path"], "current_row_id": row["row_id"] if row else None, "current_attempt_id": row["attempt_id"] if row else None, "current_row_started_at": now() if row else None, "python_executable": runtime["python_executable"], "make_cards_path": runtime["make_cards_path"]}


def execute_row(manifest: dict[str, Any], manifest_hash: str, row: dict[str, Any]) -> dict[str, Any]:
    log = Path(row["log_path"]); log.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = log.open("xb")
    except FileExistsError as exc:
        raise RunnerError("interrupted_requires_external_reconciliation", f"log exists: {log}", row_id=row["row_id"]) from exc
    argv, started = resolved_argv(manifest, row), now()
    with handle:
        handle.write((json.dumps({"row_id": row["row_id"], "attempt_id": row["attempt_id"], "resolved_argv": argv, "start_timestamp": started}, sort_keys=True) + "\n").encode())
        handle.flush()
        completed = subprocess.run(argv, cwd=row["working_directory"], stdout=handle, stderr=subprocess.STDOUT, check=False)
        os.fsync(handle.fileno())
    if completed.returncode:
        raise RunnerError("row_command_failed", f"direct producer argv returned {completed.returncode}; no retry was attempted", row_id=row["row_id"])
    output_paths = [Path(path) for path in row["expected_output_paths"]]
    if any(not path.is_file() or path.stat().st_size == 0 for path in output_paths):
        raise RunnerError("expected_output_check_failed", "expected output missing or empty", row_id=row["row_id"])
    root, merge = Path(row["output_root"]), Path(row["merge_report_path"])
    sources = {"selected_wcs": root / "selectedWCs.txt", "scalings": root / "scalings-preselect.json", "merge_report": merge}
    destinations = snapshots(row)
    for key, source in sources.items():
        copy_once(source, destinations[key])
    runtime = manifest["runtime_contract"]
    receipt = {"schema": RECEIPT_SCHEMA, "manifest_sha256": manifest_hash, "row_id": row["row_id"], "attempt_id": row["attempt_id"], "resolved_argv": argv, "start_timestamp": started, "end_timestamp": now(), "command_return_code": completed.returncode, "runtime_contract_id": runtime["contract_id"], "python_executable": runtime["python_executable"], "make_cards_path": runtime["make_cards_path"], "runtime_fingerprints": runtime["fingerprints"], "runtime_contract_digest": runtime_digest(runtime), "primary_outputs": [record(path) for path in output_paths], "artifacts": {"merge_report": record(merge), "selected_wcs_snapshot": record(destinations["selected_wcs"]), "scalings_snapshot": record(destinations["scalings"]), "merge_report_snapshot": record(destinations["merge_report"]), "log": record(log)}}
    final = receipt_path(manifest, row)
    atomic_json(receipt, final, False)
    return {"row_id": row["row_id"], "attempt_id": row["attempt_id"], "action": "executed", "receipt_path": str(final)}


def execute(manifest: dict[str, Any], manifest_hash: str) -> dict[str, Any]:
    root, lock = Path(manifest["control_root"]), Path(manifest["lock_path"])
    root.mkdir(parents=True, exist_ok=True); lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError("execution_lock_held", f"another runner owns lock: {lock}") from exc
        owner_started_at = now()
        try:
            for row in manifest["rows"]:
                state = classify(manifest, manifest_hash, row, True, read_owner(manifest))
                if state["status"] not in {"not_started", "complete"}:
                    raise RunnerError(state["status"], state["reason"], row_id=row["row_id"])
            atomic_json(owner_payload(manifest, manifest_hash, owner_started_at), owner_path(manifest), True)
            results: list[dict[str, Any]] = []
            for row in manifest["rows"]:
                state = classify(manifest, manifest_hash, row, True, read_owner(manifest))
                if state["status"] == "complete":
                    results.append({"row_id": row["row_id"], "attempt_id": row["attempt_id"], "action": "skipped_valid_receipt", "receipt_path": state["receipt_path"]}); continue
                if state["status"] != "not_started":
                    raise RunnerError(state["status"], state["reason"], row_id=row["row_id"])
                validate_runtime(manifest["runtime_contract"])
                validate_row_inputs(row)
                fresh = classify(manifest, manifest_hash, row, True, read_owner(manifest))
                if fresh["status"] != "not_started":
                    raise RunnerError(fresh["status"], fresh["reason"], row_id=row["row_id"])
                atomic_json(owner_payload(manifest, manifest_hash, owner_started_at, row), owner_path(manifest), True)
                results.append(execute_row(manifest, manifest_hash, row))
                atomic_json(owner_payload(manifest, manifest_hash, owner_started_at), owner_path(manifest), True)
            return {"schema": "topeft_datacard_matrix_execution_result_v2", "manifest_sha256": manifest_hash, "rows": results}
        finally:
            owner_path(manifest).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a prequalified datacard matrix with byte-bound receipts.")
    modes = parser.add_mutually_exclusive_group(); modes.add_argument("--plan-only", action="store_true"); modes.add_argument("--status", action="store_true")
    parser.add_argument("manifest", type=Path); args = parser.parse_args(argv)
    try:
        manifest, manifest_hash = load_manifest(args.manifest)
        result = read_only("plan_only" if args.plan_only else "status", manifest, manifest_hash) if (args.plan_only or args.status) else execute(manifest, manifest_hash)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2 if args.plan_only and not result["launch_ready"] else 0
    except RunnerError as exc:
        error = {"schema": "topeft_datacard_matrix_error_v2", "status": exc.code, "message": str(exc)}
        if exc.row_id is not None: error["row_id"] = exc.row_id
        print(json.dumps(error, indent=2, sort_keys=True), file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
