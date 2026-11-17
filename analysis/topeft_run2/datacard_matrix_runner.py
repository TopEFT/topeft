#!/usr/bin/env python3
"""Run an opaque, prequalified datacard manifest without a shell intermediary."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
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
import uuid
from typing import Any

UNIT_SCHEMA = "topeft_datacard_execution_unit_v1"
MANIFEST_SCHEMA = "topeft_datacard_matrix_v3"
RECEIPT_SCHEMA = "topeft_datacard_execution_receipt_v2"
OWNER_SCHEMA = "topeft_datacard_runner_owner_v1"
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOP_FIELDS = {"schema", "control_root", "lock_path", "runtime_contract", "rows"}
RUNTIME_FIELDS = {"contract_id", "python_executable", "make_cards_path", "fingerprints"}
FINGERPRINT_FIELDS = {"path", "sha256"}
RUNTIME_OVERLAY_SCHEMA = "topeft_datacard_runtime_overlay_v1"
RUNTIME_OVERLAY_FIELDS = {"schema", "base_manifest_sha256", "base_runtime_contract_digest", "allowed_row_ids", "replacements"}
RUNTIME_REPLACEMENT_FIELDS = {"path", "from_sha256", "to_sha256"}
RUNTIME_OVERLAY_MANIFEST_SHA256 = "f49d936d682809d0d450ecc829a8801394859ac31ac3ec1e1235ce69b93a1651"
RUNTIME_OVERLAY_BASE_DIGEST = "1728c047d48ab1692d87db65491f09de47ac9305ec07ecff0fd35ecc3f24dc5f"
RUNTIME_OVERLAY_ROWS = {"run3_07_exec_01", "run3_11_exec_01", "run3_05_exec_01"}
RUNTIME_OVERLAY_FROM = {
    "analysis/topeft_run2/make_cards.py": "16bba455a7c27dc07a68ffd990b69c08285189161c89a620fa7eb69605e395a9",
    "topeft/modules/datacard_tools.py": "c9d8b3b7cbdcb43e242ace2b376fbfdb0da86ae425875c828c4668dd4ed81758",
    "analysis/topeft_run2/datacard_matrix_runner.py": "72beb21b3ca47a97e273fbd87f490efd9c03c773254159bd53fc2757fc584374",
}
ROW_FIELDS = {"row_id", "logical_row_id", "attempt_id", "era", "working_directory", "input_pkl", "output_root", "distribution", "physical_channels", "years", "missing_parton_path", "merge_report_path", "snapshot_directory", "log_path", "expected_output_paths", "producer_args"}
SNAPSHOTS = {"selected_wcs": "selectedWCs.txt", "scalings": "scalings-preselect.json", "merge_report": "merge_report.json"}
INTENT_SCHEMA = "topeft_datacard_attempt_intent_v1"
INTENT_MARKER = ".datacard_attempt_intent.json"
ATTEMPT_RE = re.compile(r"^attempt_([0-9]{6})$")
RESERVATION_SCHEMA = "topeft_datacard_reservation_v1"
RESERVATION_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
SCHEDULER_STATES = {1: "IDLE", 2: "RUNNING", 3: "REMOVING", 4: "COMPLETED_IN_QUEUE",
                    5: "HELD", 6: "TRANSFERRING", 7: "SUSPENDED"}


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


def runtime_overlay_digest(overlay: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(overlay, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def effective_runtime(manifest: dict[str, Any], manifest_hash: str, row: dict[str, Any],
                      overlay: Any, *, verify_files: bool = True) -> dict[str, Any]:
    """Validate the frozen campaign's exact three replacements before using them."""
    overlay = exact_fields(overlay, RUNTIME_OVERLAY_FIELDS, "runtime_contract_overlay")
    runtime = manifest["runtime_contract"]
    if (overlay["schema"] != RUNTIME_OVERLAY_SCHEMA
            or manifest_hash != RUNTIME_OVERLAY_MANIFEST_SHA256
            or overlay["base_manifest_sha256"] != manifest_hash
            or runtime_digest(runtime) != RUNTIME_OVERLAY_BASE_DIGEST
            or overlay["base_runtime_contract_digest"] != runtime_digest(runtime)):
        raise RunnerError("runtime_overlay_mismatch", "overlay schema or frozen base identity differs")
    allowed_rows = overlay["allowed_row_ids"]
    if (not isinstance(allowed_rows, list) or len(allowed_rows) != len(RUNTIME_OVERLAY_ROWS)
            or any(not isinstance(item, str) for item in allowed_rows)
            or set(allowed_rows) != RUNTIME_OVERLAY_ROWS or row["row_id"] not in allowed_rows):
        raise RunnerError("runtime_overlay_mismatch", "overlay row scope differs")
    replacements = overlay["replacements"]
    if not isinstance(replacements, list) or len(replacements) != len(RUNTIME_OVERLAY_FROM):
        raise RunnerError("runtime_overlay_mismatch", "overlay requires exactly three replacements")
    repository_root = Path(__file__).resolve().parents[2]
    pinned = {item["path"]: item["sha256"] for item in runtime["fingerprints"]}
    replacement_sha: dict[str, str] = {}
    for index, replacement in enumerate(replacements):
        replacement = exact_fields(replacement, RUNTIME_REPLACEMENT_FIELDS, f"runtime_contract_overlay.replacements[{index}]")
        relative = replacement["path"]
        if relative not in RUNTIME_OVERLAY_FROM or relative in replacement_sha:
            raise RunnerError("runtime_overlay_mismatch", "unknown or duplicate replacement path")
        path = str(repository_root / relative)
        if (pinned.get(path) != RUNTIME_OVERLAY_FROM[relative]
                or replacement["from_sha256"] != pinned[path]
                or not isinstance(replacement["to_sha256"], str)
                or not SHA256_RE.fullmatch(replacement["to_sha256"])):
            raise RunnerError("runtime_overlay_mismatch", f"replacement identity differs: {relative}")
        replacement_sha[relative] = replacement["to_sha256"]
    if set(replacement_sha) != set(RUNTIME_OVERLAY_FROM):
        raise RunnerError("runtime_overlay_mismatch", "replacement set differs")
    replacement_paths = {str(repository_root / relative): digest
                         for relative, digest in replacement_sha.items()}
    effective = {**runtime, "fingerprints": [
        {"path": item["path"], "sha256": replacement_paths.get(item["path"], item["sha256"])}
        for item in runtime["fingerprints"]
    ]}
    validate_runtime(effective, verify_files=verify_files)
    return effective


def overlay_provenance(manifest: dict[str, Any], manifest_hash: str, row: dict[str, Any],
                       overlay: dict[str, Any], *, verify_files: bool = False) -> dict[str, Any]:
    effective = effective_runtime(manifest, manifest_hash, row, overlay, verify_files=verify_files)
    return {"base_manifest_sha256": manifest_hash,
            "base_runtime_contract_digest": runtime_digest(manifest["runtime_contract"]),
            "runtime_contract_overlay": overlay,
            "runtime_contract_overlay_sha256": runtime_overlay_digest(overlay),
            "effective_runtime_contract_digest": runtime_digest(effective),
            "effective_runtime_fingerprints": effective["fingerprints"]}


def validate_overlay_provenance(value: dict[str, Any], manifest: dict[str, Any],
                                manifest_hash: str, row: dict[str, Any]) -> tuple[bool, str]:
    keys = set(overlay_provenance_fields())
    present = keys.intersection(value)
    if not present:
        return True, "legacy base runtime identity"
    if present != keys:
        return False, "overlay provenance is incomplete"
    try:
        expected = overlay_provenance(manifest, manifest_hash, row, value["runtime_contract_overlay"])
    except (RunnerError, TypeError, ValueError) as exc:
        return False, f"overlay provenance differs: {exc}"
    if any(value[key] != expected[key] for key in keys):
        return False, "overlay provenance differs from frozen manifest"
    return True, "embedded overlay and effective runtime identity match"


def overlay_provenance_fields() -> tuple[str, ...]:
    return ("base_manifest_sha256", "base_runtime_contract_digest", "runtime_contract_overlay",
            "runtime_contract_overlay_sha256", "effective_runtime_contract_digest", "effective_runtime_fingerprints")


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
    if {"--condor", "-C", "--merge-only", "--select-only", "--cache-merged-pkl"}.intersection(args):
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


def validate_unit_layout(manifest: dict[str, Any]) -> None:
    roots = [Path(row["output_root"]).resolve() for row in manifest["rows"]]
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(roots) for b in roots[i + 1:]):
        raise RunnerError("manifest_schema_error", "execution units require disjoint output roots")


def derive_units(path: Path) -> list[dict[str, Any]]:
    """Derive dispatch selectors; the original manifest remains workload authority."""
    manifest, manifest_hash = load_manifest(path)
    validate_unit_layout(manifest)
    return [{"schema": UNIT_SCHEMA, "parent_manifest": str(path.resolve()),
             "parent_integrity": manifest_hash, "row_id": row["row_id"]}
            for row in manifest["rows"]]


def resolve_unit(unit: Any) -> tuple[dict[str, Any], str, dict[str, Any]]:
    unit = exact_fields(unit, {"schema", "parent_manifest", "parent_integrity", "row_id"}, "execution_unit")
    if unit["schema"] != UNIT_SCHEMA:
        raise RunnerError("unit_schema_error", "unsupported execution-unit schema")
    manifest, manifest_hash = load_manifest(absolute(unit["parent_manifest"], "parent_manifest"))
    if unit["parent_integrity"] != manifest_hash:
        raise RunnerError("stale_dispatch_input", "parent manifest bytes changed after unit derivation")
    return manifest, manifest_hash, select_row(manifest, unit["row_id"])


def select_row(manifest: dict[str, Any], row_id: str) -> dict[str, Any]:
    for row in manifest["rows"]:
        if row["row_id"] == row_id:
            return row
    raise RunnerError("unit_schema_error", f"unknown execution unit: {row_id}")


def unit_lock_path(row: dict[str, Any]) -> Path:
    output = Path(row["output_root"])
    if output.is_symlink():
        raise RunnerError("canonical_collision_or_ambiguous_state", "output root is a symlink")
    output = output.resolve()
    return output.parent / f".{output.name}.datacard.lock"


def unit_owner_path(row: dict[str, Any]) -> Path:
    return unit_lock_path(row).with_suffix(".owner.json")


def reservation_path(row: dict[str, Any]) -> Path:
    return unit_lock_path(row).with_suffix(".reservation.json")


def read_reservation(row: dict[str, Any]) -> dict[str, Any] | None:
    """A corrupt or foreign domain owner is never equivalent to no owner."""
    path = reservation_path(row)
    try:
        if path.is_symlink():
            raise ValueError("reservation is a symlink")
        value = json.loads(path.read_text())
        required_fields = {"schema", "row_id", "output_root", "backend", "owner_id", "created_at", "scheduler", "scheduler_id"}
        if (not isinstance(value, dict)
                or not required_fields <= set(value)
                or set(value) - required_fields - {"scheduler_observation"}
                or value.get("schema") != RESERVATION_SCHEMA
                or value.get("row_id") != row["row_id"]
                or value.get("output_root") != str(Path(row["output_root"]).resolve())
                or value.get("backend") not in {"local", "manual", "condor"}
                or not isinstance(value.get("created_at"), str) or not value["created_at"]):
            raise ValueError("reservation identity/schema differs")
        validate_reservation_token(value.get("owner_id"))
        if value["backend"] == "condor":
            validate_scheduler(value["scheduler"])
        elif value["scheduler"] is not None:
            raise ValueError("local reservation carries a scheduler endpoint")
        observation = value.get("scheduler_observation")
        if observation is not None:
            keys = {"scheduler_id", "global_job_id", "schedd", "pool", "source", "observed_at"}
            if (not isinstance(observation, dict) or set(observation) != keys
                    or any(item is not None and (not isinstance(item, str) or not item or "\x00" in item)
                           for item in observation.values())
                    or observation["source"] not in {"hint", "default", "global"}
                    or (observation["scheduler_id"] is not None
                        and not re.fullmatch(r"[0-9]+\.[0-9]+", observation["scheduler_id"]))):
                raise ValueError("scheduler observation is malformed")
        return value
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RunnerError("reservation_ambiguous", str(exc), row_id=row["row_id"]) from exc


def validate_reservation_token(token: str) -> None:
    if not isinstance(token, str) or not RESERVATION_TOKEN_RE.fullmatch(token):
        raise RunnerError("invalid_reservation_token", "reservation token must be 32 lowercase hex digits")


def validate_scheduler(scheduler: dict) -> None:
    if (not isinstance(scheduler, dict)
            or not {"query_executable"} <= set(scheduler)
            or set(scheduler) - {"query_executable", "dispatch_directory", "pool", "schedd"}
            or any(not isinstance(value, str) or not value for value in scheduler.values())
            or not Path(scheduler["query_executable"]).is_absolute()
            or ("dispatch_directory" in scheduler and not Path(scheduler["dispatch_directory"]).is_absolute())
            or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]*", scheduler[key]) for key in ("pool", "schedd") if key in scheduler)):
        raise RunnerError("condor_configuration_error", "invalid scheduler observation/query configuration")


def new_reservation(row: dict[str, Any], backend: str, scheduler: dict | None = None) -> dict:
    return {"schema": RESERVATION_SCHEMA, "row_id": row["row_id"],
            "output_root": str(Path(row["output_root"]).resolve()), "backend": backend,
            "owner_id": uuid.uuid4().hex, "created_at": now(),
            # scheduler_id is only the last observed locator, never an equality gate.
            "scheduler": scheduler, "scheduler_id": None}


def reported_cluster_id(stdout: str) -> int | None:
    match = re.search(r"\bsubmitted to cluster ([0-9]+)\b", stdout, re.IGNORECASE)
    return int(match.group(1)) if match else None


def scheduler_snapshot(scheduler: dict) -> dict:
    """Best-effort context immediately before submission; never a target selector."""
    snapshot = {"observed_at": now()}
    try:
        snapshot["submit_host"] = socket.gethostname()
    except OSError:
        pass
    for name, argv in (("collector", ["condor_config_val", "COLLECTOR_HOST"]),
                       ("default_schedd", ["condor_config_val", "SCHEDD_HOST"]),
                       ("htcondor_version", ["condor_version"])):
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=5, check=False)
            if result.returncode == 0 and result.stdout.strip():
                snapshot[name] = result.stdout.strip()
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return snapshot


def _query_ads(argv: list[str]) -> tuple[list[dict] | None, str | None]:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
        if result.returncode:
            raise ValueError(f"exit={result.returncode}; stderr={result.stderr!r}; stdout={result.stdout!r}")
        ads = json.loads(result.stdout)
        if not isinstance(ads, list):
            raise ValueError("scheduler reply is not a JSON list")
        return ads, None
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return None, f"query={argv!r}; error={exc}"


def _terminal_event_log(reservation: dict) -> str | None:
    """A remove event is terminal; an exit event alone may precede a retry."""
    observed = reservation.get("scheduler_observation") or {}
    locator = observed.get("scheduler_id") or reservation.get("scheduler_id")
    directory = reservation["scheduler"].get("dispatch_directory")
    if not directory or not isinstance(locator, str) or not re.fullmatch(r"[0-9]+\.[0-9]+", locator):
        return None
    path = Path(directory) / "logs" / f'{reservation["row_id"]}__{locator}.events.log'
    try:
        if path.is_symlink() or not path.is_file():
            return None
        content = path.read_text()
    except (OSError, UnicodeError):
        return None
    if re.search(rf"(?m)^009 \({re.escape(locator)}\.[0-9]+\)", content):
        return str(path)
    return None


def query_reservation(reservation: dict) -> dict:
    """Submit-side token discovery; a locator is a hint, never ownership authority."""
    validate_reservation_token(reservation["owner_id"])
    scheduler = reservation["scheduler"]
    validate_scheduler(scheduler)
    observed = reservation.get("scheduler_observation") or {}
    constraint = f'DatacardReservation == "{reservation["owner_id"]}"'
    attributes = "ClusterId,ProcId,JobStatus,DatacardUnit,DatacardReservation,GlobalJobId,ScheddName,HoldReason,HoldReasonCode,HoldReasonSubCode"
    base = ["-allusers", "-constraint", constraint, "-json", "-attributes", attributes]
    sources = []
    hint_schedd = observed.get("schedd") or scheduler.get("schedd")
    hint_pool = observed.get("pool") or scheduler.get("pool")
    if hint_schedd and hint_pool:
        sources.append(("hint", [scheduler["query_executable"], "-pool", hint_pool, "-name", hint_schedd, *base]))
    sources.extend((("default", [scheduler["query_executable"], *base]),
                    ("global", [scheduler["query_executable"], "-global", *base])))
    observed_at = now()
    current_snapshot = scheduler_snapshot(scheduler)
    diagnostics = []
    live = {}
    covered_zero = []
    for source, argv in sources:
        ads, error = _query_ads(argv)
        if error:
            diagnostics.append({"source": source, "error": error})
            continue
        if len(ads) > 1:
            job_ids = [f'{ad.get("ClusterId")}.{ad.get("ProcId")}' if isinstance(ad, dict) else repr(ad) for ad in ads]
            raise RunnerError("DUPLICATE_SCHEDULER_OWNERSHIP", f"token={reservation['owner_id']}; jobs={job_ids}; source={source}; observed_at={observed_at}", row_id=reservation["row_id"])
        same_pool = (observed.get("pool") and current_snapshot.get("collector")
                     and observed["pool"] == current_snapshot["collector"])
        same_default = (observed.get("schedd") and current_snapshot.get("default_schedd")
                        and observed["schedd"] == current_snapshot["default_schedd"])
        if not ads and ((source == "hint" and observed.get("schedd") == hint_schedd
                         and observed.get("pool") == hint_pool)
                        or (source == "global" and same_pool)
                        or (source == "default" and same_pool and same_default)):
            covered_zero.append(source)
        for ad in ads:
            if (not isinstance(ad, dict) or ad.get("DatacardUnit") != reservation["row_id"]
                    or ad.get("DatacardReservation") != reservation["owner_id"]
                    or any(type(ad.get(k)) is not int or ad[k] < 0 for k in ("ClusterId", "ProcId"))
                    or type(ad.get("JobStatus")) is not int or ad["JobStatus"] not in SCHEDULER_STATES):
                raise RunnerError("scheduler_observation_invalid", f"invalid token/row/state reply: {ad!r}; query={argv!r}; observed_at={observed_at}", row_id=reservation["row_id"])
            locator = ad.get("GlobalJobId") or (ad.get("ScheddName"), ad["ClusterId"], ad["ProcId"])
            if str(locator) in live and live[str(locator)][0] != ad:
                raise RunnerError("SCHEDULER_EVIDENCE_CONTRADICTION", f"token={reservation['owner_id']}; inconsistent live ads for {locator}; observed_at={observed_at}", row_id=reservation["row_id"])
            live[str(locator)] = (ad, source)
    if len(live) > 1:
        job_ids = [f'{ad["ClusterId"]}.{ad["ProcId"]}' for ad, _ in live.values()]
        raise RunnerError("DUPLICATE_SCHEDULER_OWNERSHIP", f"token={reservation['owner_id']}; jobs={job_ids}; observed_at={observed_at}", row_id=reservation["row_id"])
    terminal_proof = None
    event_log = _terminal_event_log(reservation)
    if event_log:
        terminal_proof = {"source": "event_log_remove", "path": event_log}
    history_executable = shutil.which("condor_history")
    if history_executable:
        history_argv = [history_executable, "-constraint", constraint, "-json", "-attributes", attributes]
        history_ads, history_error = _query_ads(history_argv)
        if history_error:
            diagnostics.append({"source": "history", "error": history_error})
        elif history_ads:
            matches = [ad for ad in history_ads if isinstance(ad, dict)
                       and ad.get("DatacardReservation") == reservation["owner_id"]
                       and ad.get("DatacardUnit") == reservation["row_id"]
                       and ad.get("JobStatus") in (3, 4)]
            if matches:
                terminal_proof = {"source": "history", "global_job_id": matches[0].get("GlobalJobId"),
                                  "scheduler_id": f'{matches[0].get("ClusterId")}.{matches[0].get("ProcId")}' }
    if live and terminal_proof:
        raise RunnerError("SCHEDULER_EVIDENCE_CONTRADICTION", f"live={list(live)}; terminal={terminal_proof}; observed_at={observed_at}", row_id=reservation["row_id"])
    if live:
        ad, source = next(iter(live.values()))
        return {"state": SCHEDULER_STATES[ad["JobStatus"]], "ownership": "ACTIVE",
                "scheduler_id": f'{ad["ClusterId"]}.{ad["ProcId"]}', "observed_at": observed_at,
                "source": source, "global_job_id": ad.get("GlobalJobId"),
                "schedd": ad.get("ScheddName") or current_snapshot.get("default_schedd"),
                "pool": hint_pool if source == "hint" else current_snapshot.get("collector"),
                "hold_evidence": {key: ad[key] for key in ("HoldReason", "HoldReasonCode", "HoldReasonSubCode") if key in ad},
                "diagnostics": diagnostics}
    if terminal_proof or covered_zero:
        return {"state": "GONE", "ownership": "GONE", "scheduler_id": reservation["scheduler_id"],
                "observed_at": observed_at, "terminal_proof": terminal_proof or {"source": "covered_zero", "query_source": covered_zero[0]},
                "diagnostics": diagnostics}
    code = "SCHEDULER_OBSERVABILITY_INCOMPLETE" if any(item["source"] == "hint" for item in diagnostics) else "SCHEDULER_OBSERVABILITY_INSUFFICIENT"
    raise RunnerError(code, f"token={reservation['owner_id']}; observed_at={observed_at}; diagnostics={diagnostics}; no terminal proof or covered zero-result query", row_id=reservation["row_id"])


def record_scheduler_observation(reservation: dict, observation: dict) -> None:
    reservation["scheduler_id"] = observation["scheduler_id"]
    if observation.get("source") in {"hint", "default", "global"}:
        reservation["scheduler_observation"] = {key: observation.get(key) for key in
            ("scheduler_id", "global_job_id", "schedd", "pool", "source", "observed_at")}
    atomic_json(reservation, reservation_path(reservation), True)


def reconcile_h2(manifest: dict, manifest_hash: str, row: dict) -> dict:
    """Called only under the physical lease; never starts a producer."""
    state = classify(manifest, manifest_hash, row, False, unit_owner(row) or read_owner(manifest))
    if state["status"] in {"complete_validated_staging_not_promoted", "promoted_incomplete", "promoted_complete_without_receipt"}:
        reconcile_row(manifest, manifest_hash, row)
        state = classify(manifest, manifest_hash, row, False)
    if state["status"] not in {"complete", "not_started", "stale_owner_free_lock", "partial_staging_attempt"}:
        raise RunnerError(state["status"], state["reason"], row_id=row["row_id"])
    return state


def reconcile_reservation(manifest: dict, manifest_hash: str, row: dict,
                          evidence: dict | None = None) -> dict | None:
    """Under unit_lease, observe ownership or retire it only after safe H2."""
    reservation = read_reservation(row)
    if reservation is None:
        return None
    if reservation["backend"] == "condor":
        observation = query_reservation(reservation)
        if observation["state"] != "GONE":
            record_scheduler_observation(reservation, observation)
            return {**reservation, **observation}
        if evidence is not None:
            evidence.update({"reservation_token": reservation["owner_id"],
                             "terminal_proof": observation.get("terminal_proof"),
                             "observed_at": observation.get("observed_at")})
    reconcile_h2(manifest, manifest_hash, row)
    reservation_path(row).unlink()
    return None


class submission_transaction:
    """No-submit lifecycle context; callers mark possible side effects first.

    Future submit integration must pass lease_fds to its subprocess and await
    its termination before leaving this context. A timeout/nonzero result after
    mark_submission_started is ambiguous, never permission to roll back.
    """
    def __init__(self):
        self.reservations = []
        self.lease_fds = ()
        self.submission_started = False
        self.superseded_tokens = set()

    def mark_submission_started(self):
        self.submission_started = True

    def record_definite_spawn_failure(self, error: OSError):
        """Only for an OSError raised directly by process creation (Popen).

        The future caller must establish that no submit executable started.
        Never use this for a nonzero exit, timeout, interruption, or an error
        from I/O/status handling after launch; those retain possible effects.
        """
        if not isinstance(error, OSError) or not self.submission_started:
            raise RunnerError("submission_outcome_error", "definite spawn failure requires a process-creation OSError after the start marker")
        self.submission_started = False


@contextmanager
def reserve_condor(manifest: dict, manifest_hash: str, rows: list[dict], scheduler: dict, *,
                   rescue_ambiguous: bool = False):
    """Own all row reservations for one transaction; never invoke a submitter."""
    validate_scheduler(scheduler)
    validate_runtime(manifest["runtime_contract"])
    if rescue_ambiguous and len(rows) != 1:
        raise RunnerError("row_selection_error", "ambiguous rescue requires exactly one row")
    transaction = submission_transaction()
    with ExitStack() as leases:
        try:
            for row in sorted(rows, key=lambda item: str(unit_lock_path(item))):
                transaction.lease_fds += leases.enter_context(unit_lease(manifest, row))
                old = read_reservation(row) if rescue_ambiguous else None
                if rescue_ambiguous:
                    if old is None or old["backend"] != "condor":
                        raise RunnerError("rescue_not_applicable", "rescue requires an existing Condor reservation", row_id=row["row_id"])
                    try:
                        observation = query_reservation(old)
                    except RunnerError as exc:
                        if exc.code not in {"SCHEDULER_OBSERVABILITY_INCOMPLETE", "SCHEDULER_OBSERVABILITY_INSUFFICIENT"}:
                            raise
                        observation = None
                    if observation is not None and observation["ownership"] == "ACTIVE":
                        raise RunnerError("row_reserved", "active Condor owner cannot be superseded", row_id=row["row_id"])
                    state = reconcile_h2(manifest, manifest_hash, row)
                    if state["status"] == "complete":
                        raise RunnerError("row_complete", "valid H2 receipt already completes row", row_id=row["row_id"])
                    if (observation is not None and observation.get("terminal_proof", {}).get("source")
                            in {"history", "event_log_remove"}):
                        reservation_path(row).unlink()
                        old = None
                else:
                    if reconcile_reservation(manifest, manifest_hash, row) is not None:
                        raise RunnerError("row_reserved", "row already has a logical owner", row_id=row["row_id"])
                    reconcile_h2(manifest, manifest_hash, row)
                reservation = new_reservation(row, "condor", scheduler)
                transaction.reservations.append(reservation)
                atomic_json(reservation, reservation_path(row), old is not None)
                if old is not None:
                    transaction.superseded_tokens.add(reservation["owner_id"])
            yield transaction
        finally:
            if not transaction.submission_started:
                for reservation in transaction.reservations:
                    current = read_reservation(reservation)
                    if (current and current["owner_id"] == reservation["owner_id"]
                            and reservation["owner_id"] not in transaction.superseded_tokens):
                        reservation_path(reservation).unlink()


def worker_identity(row: dict, reservation_token: str) -> str:
    """Read only literal identity attributes from the starter's local job ad.

    Other ClassAd expressions are deliberately neither evaluated nor imported.
    This backend requires _CONDOR_JOB_AD; missing context is a specific error.
    """
    validate_reservation_token(reservation_token)
    job_ad_path = os.environ.get("_CONDOR_JOB_AD")
    if not job_ad_path:
        raise RunnerError("worker_job_ad_missing", "_CONDOR_JOB_AD is required")
    fields = {}
    required = {"datacardunit", "datacardreservation", "clusterid", "procid"}
    try:
        for line in Path(job_ad_path).read_text().splitlines():
            key, separator, value = line.partition("=")
            key = key.strip().lower()
            if separator and key in required:
                if key in fields:
                    raise ValueError(f"duplicate {key}")
                fields[key] = json.loads(value.strip())
        if set(fields) != required or any(type(fields[k]) is not int or fields[k] < 0 for k in ("clusterid", "procid")):
            raise ValueError("missing attributes or invalid ClusterId/ProcId literals")
        if not all(isinstance(fields[k], str) for k in ("datacardunit", "datacardreservation")):
            raise ValueError("row/token must be string literals")
    except (OSError, ValueError) as exc:
        raise RunnerError("worker_job_ad_malformed", f"{job_ad_path}: {exc}", row_id=row["row_id"]) from exc
    if fields["datacardunit"] != row["row_id"] or fields["datacardreservation"] != reservation_token:
        raise RunnerError("worker_job_ad_identity_conflict", "itemdata row/token differs from local job ad", row_id=row["row_id"])
    return f'{fields["clusterid"]}.{fields["procid"]}'


def unit_owner(row: dict[str, Any]) -> dict[str, Any] | None:
    try:
        value = json.loads(unit_owner_path(row).read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def reject_nested_domain(output: Path) -> None:
    for ancestor in output.parents:
        if ancestor == ancestor.parent:
            continue
        marker = ancestor / INTENT_MARKER
        lock = ancestor.parent / f".{ancestor.name}.datacard.lock"
        if lock.exists() or lock.is_symlink() or marker.exists():
            raise RunnerError("canonical_collision_or_ambiguous_state", f"output is nested in an existing ownership domain: {ancestor}")


def existing_parent(path: Path) -> Path:
    parent = path.parent
    while not parent.is_dir():
        parent = parent.parent
    return parent


@contextmanager
def unit_lease(manifest: dict[str, Any] | None, row: dict[str, Any]):
    """Lease one output domain; initialize paths under a short filesystem mutex.

    The setup mutex prevents a rejected nested unit from creating the active
    owner's canonical directory. It is released before execution. Persistent
    lock inodes are never removed. Producer inheritance keeps ownership alive
    if the manager dies. The shared legacy lock excludes older global runners.
    """
    legacy = Path(manifest["lock_path"]) if manifest else None
    lock = unit_lock_path(row)
    output = Path(row["output_root"]).resolve()
    with ExitStack() as handles:
        while True:
            reject_nested_domain(output)
            parent = existing_parent(output)
            setup_path = parent / ".datacard_lease_setup.lock"
            if setup_path.is_symlink():
                raise RunnerError("canonical_collision_or_ambiguous_state", "setup lock is a symlink")
            with setup_path.open("a+b") as setup:
                fcntl.flock(setup.fileno(), fcntl.LOCK_EX)
                if existing_parent(output) != parent:
                    continue  # Parent creation won a race; use its setup mutex.
                reject_nested_domain(output)
                if legacy:
                    legacy.parent.mkdir(parents=True, exist_ok=True)
                lock.parent.mkdir(parents=True, exist_ok=True)
                if lock.is_symlink() or (legacy and legacy.is_symlink()):
                    raise RunnerError("canonical_collision_or_ambiguous_state", "lock path is a symlink", row_id=row["row_id"])
                legacy_handle = handles.enter_context(legacy.open("a+b")) if legacy else None
                unit_handle = handles.enter_context(lock.open("a+b"))
                try:
                    if legacy_handle:
                        fcntl.flock(legacy_handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                    fcntl.flock(unit_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RunnerError("execution_lock_held", f"execution ownership unavailable: {lock}", row_id=row["row_id"]) from exc
                break
        yield tuple(handle.fileno() for handle in (legacy_handle, unit_handle) if handle is not None)


@contextmanager
def direct_output_lease(output_root: Path):
    """Exclude raw producer calls from reserved canonical output domains.

    Such calls have no manifest with which to reconcile H2. They may only enter
    an unreserved domain and retain its physical lease throughout production.
    An H2-managed producer operates on its private staging domain instead.
    """
    row = {"row_id": "direct", "output_root": str(output_root.absolute())}
    with unit_lease(None, row):
        path = reservation_path(row)
        if path.exists() or path.is_symlink():
            raise RunnerError("row_reserved", "canonical output has a logical reservation; use the matrix runner")
        marker = output_root / INTENT_MARKER
        if marker.exists() or marker.is_symlink():
            raise RunnerError("canonical_collision_or_ambiguous_state", "canonical output belongs to H2; use the matrix runner")
        yield


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


def execution_input_availability(manifest: dict[str, Any], row: dict[str, Any],
                                 effective: dict[str, Any] | None = None) -> dict[str, Any]:
    runtime = effective if effective is not None else manifest["runtime_contract"]
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
    overlay_valid, overlay_reason = validate_overlay_provenance(receipt, manifest, manifest_hash, row)
    if not overlay_valid:
        return False, overlay_reason
    if "physical_attempt_id" in receipt or "execution_argv" in receipt:
        physical = receipt.get("physical_attempt_id")
        if not isinstance(physical, str) or not ATTEMPT_RE.fullmatch(physical):
            return False, "physical attempt identity is invalid"
        if receipt.get("execution_argv") != staged_argv(manifest, row, attempt_root(manifest_hash, row) / physical):
            return False, "execution argv does not match physical attempt"
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


def attempt_root(manifest_hash: str, row: dict[str, Any]) -> Path:
    output = Path(row["output_root"])
    return output.parent / f".{output.name}.{manifest_hash}.{row['attempt_id']}.attempts"


def attempts(manifest_hash: str, row: dict[str, Any]) -> list[Path]:
    root = attempt_root(manifest_hash, row)
    if not root.exists():
        return []
    if not root.is_dir() or root.is_symlink():
        raise RunnerError("canonical_collision_or_ambiguous_state", f"attempt namespace is not a directory: {root}", row_id=row["row_id"])
    entries = sorted(root.iterdir())
    if any(not ATTEMPT_RE.fullmatch(path.name) or not path.is_dir() or path.is_symlink() for path in entries):
        raise RunnerError("canonical_collision_or_ambiguous_state", f"unexpected attempt namespace entry: {root}", row_id=row["row_id"])
    return entries


def staged_paths(row: dict[str, Any], attempt: Path) -> tuple[list[Path], dict[str, Path]]:
    root = Path(row["output_root"])
    outputs = [attempt / "cards" / Path(path).relative_to(root) for path in row["expected_output_paths"]]
    artifacts = {
        "merge_report": attempt / "merge_report.json",
        "selected_wcs_snapshot": attempt / "snapshots" / SNAPSHOTS["selected_wcs"],
        "scalings_snapshot": attempt / "snapshots" / SNAPSHOTS["scalings"],
        "merge_report_snapshot": attempt / "snapshots" / SNAPSHOTS["merge_report"],
        "log": attempt / "producer.log",
    }
    return outputs, artifacts


def canonical_artifacts(row: dict[str, Any]) -> dict[str, Path]:
    return {"merge_report": Path(row["merge_report_path"]), "selected_wcs_snapshot": snapshots(row)["selected_wcs"], "scalings_snapshot": snapshots(row)["scalings"], "merge_report_snapshot": snapshots(row)["merge_report"], "log": Path(row["log_path"])}


def staged_argv(manifest: dict[str, Any], row: dict[str, Any], attempt: Path) -> list[str]:
    # The first three argv entries identify the interpreter, producer and input.
    argv = resolved_argv(manifest, row)
    output_paths, _ = staged_paths(row, attempt)
    destinations = {row["output_root"]: str(attempt / "cards"), row["merge_report_path"]: str(attempt / "merge_report.json")}
    destinations.update(zip(row["expected_output_paths"], map(str, output_paths)))
    return [*argv[:3], *(destinations.get(value, value) for value in argv[3:])]


def valid_intent(manifest: dict[str, Any], manifest_hash: str, row: dict[str, Any], attempt: Path) -> tuple[bool, str, dict[str, Any] | None]:
    path = attempt / "intent.json"
    try:
        intent = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, str(exc), None
    if not isinstance(intent, dict):
        return False, "intent is not an object", None
    expected = {"schema": INTENT_SCHEMA, "manifest_sha256": manifest_hash, "row_id": row["row_id"], "attempt_id": row["attempt_id"], "physical_attempt_id": attempt.name, "resolved_argv": resolved_argv(manifest, row), "execution_argv": staged_argv(manifest, row, attempt), "runtime_contract_digest": runtime_digest(manifest["runtime_contract"]), "command_return_code": 0}
    if any(intent.get(key) != value for key, value in expected.items()):
        return False, "intent identity or execution contract differs", None
    overlay_valid, overlay_reason = validate_overlay_provenance(intent, manifest, manifest_hash, row)
    if not overlay_valid:
        return False, overlay_reason, None
    if not all(isinstance(intent.get(key), str) and intent[key] for key in ("start_timestamp", "end_timestamp")):
        return False, "intent timestamps missing", None
    outputs, artifacts = staged_paths(row, attempt)
    if not isinstance(intent.get("primary_outputs"), list) or len(intent["primary_outputs"]) != len(outputs) or any(path.is_symlink() or not valid_record(item, path) or path.stat().st_size == 0 for item, path in zip(intent["primary_outputs"], outputs)):
        return False, "staged primary outputs differ from intent", None
    if not isinstance(intent.get("artifacts"), dict) or set(intent["artifacts"]) != set(artifacts) or any(path.is_symlink() or not valid_record(intent["artifacts"][key], path) or path.stat().st_size == 0 for key, path in artifacts.items()):
        return False, "staged control artifacts differ from intent", None
    return True, "complete validated attempt", intent


def candidate_attempt(manifest: dict[str, Any], manifest_hash: str, row: dict[str, Any], entries: list[Path]) -> tuple[Path, dict[str, Any]] | None:
    candidates = []
    for attempt in entries:
        if (attempt / "intent.json").exists():
            valid, reason, intent = valid_intent(manifest, manifest_hash, row, attempt)
            if not valid:
                raise RunnerError("canonical_collision_or_ambiguous_state", reason, row_id=row["row_id"])
            candidates.append((attempt, intent))
    if len(candidates) > 1:
        raise RunnerError("canonical_collision_or_ambiguous_state", "multiple complete attempts for one public row", row_id=row["row_id"])
    return candidates[0] if candidates else None


def publication_state(row: dict[str, Any], attempt: Path, intent: dict[str, Any]) -> tuple[str, str]:
    root = Path(row["output_root"])
    marker = root / INTENT_MARKER
    source_outputs, source_artifacts = staged_paths(row, attempt)
    canonical_outputs = [Path(path) for path in row["expected_output_paths"]]
    artifacts = canonical_artifacts(row)
    if root.is_symlink():
        return "canonical_collision_or_ambiguous_state", "canonical root is a symlink"
    if not root.exists():
        if any(path.exists() for path in artifacts.values()):
            return "canonical_collision_or_ambiguous_state", "control artifact exists before canonical publication"
        return "complete_validated_staging_not_promoted", "validated attempt awaits publication"
    if not root.is_dir() or root.is_symlink() or not marker.is_file() or not os.path.samefile(marker, attempt / "intent.json"):
        return "canonical_collision_or_ambiguous_state", "canonical directory has no matching attempt marker"
    allowed = {marker, root / "selectedWCs.txt", root / "scalings-preselect.json", *canonical_outputs}
    if any(path.is_symlink() or (path.is_file() and path not in allowed) or (path.is_dir() and not any(path in expected.parents for expected in allowed)) for path in root.rglob("*")):
        return "canonical_collision_or_ambiguous_state", "unexpected canonical entry"
    pairs = list(zip(canonical_outputs, intent["primary_outputs"]))
    pairs += [(root / "selectedWCs.txt", record(source_artifacts["selected_wcs_snapshot"])), (root / "scalings-preselect.json", record(source_artifacts["scalings_snapshot"]))]
    pairs += [(artifacts[key], intent["artifacts"][key]) for key in artifacts]
    missing = False
    for destination, source_record in pairs:
        if not destination.exists():
            missing = True
        elif destination.is_symlink() or not valid_record({**source_record, "path": str(destination)}, destination):
            return "canonical_collision_or_ambiguous_state", f"published bytes differ: {destination}"
    return ("promoted_incomplete", "matching publication marker; missing canonical files") if missing else ("promoted_complete_without_receipt", "canonical bytes match validated attempt")


def classify(manifest: dict[str, Any], manifest_hash: str, row: dict[str, Any], held: bool | None = None, owner: dict[str, Any] | None = None) -> dict[str, Any]:
    receipt = receipt_path(manifest, row)
    base = {"row_id": row["row_id"], "attempt_id": row["attempt_id"], "receipt_path": str(receipt)}
    if receipt.exists():
        try:
            receipt_value = json.loads(receipt.read_text())
            valid, reason = validate_receipt(receipt_value, manifest, manifest_hash, row)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            valid, reason = False, str(exc)
            receipt_value = None
        provenance = {key: receipt_value[key] for key in overlay_provenance_fields()
                      if valid and isinstance(receipt_value, dict) and key in receipt_value}
        return {**base, "status": "complete" if valid else "invalid_receipt", "reason": reason, **provenance}
    if held is None:
        legacy_held = probe_lock(Path(manifest["lock_path"]))[0]
        unit_held = probe_lock(unit_lock_path(row))[0]
        held = True if True in (legacy_held, unit_held) else None if None in (legacy_held, unit_held) else False
    owner = (unit_owner(row) or read_owner(manifest)) if owner is None else owner
    active = held is True and isinstance(owner, dict) and owner.get("schema_version") == OWNER_SCHEMA and owner.get("manifest_sha256") == manifest_hash and owner.get("current_row_id") == row["row_id"] and owner.get("current_attempt_id") == row["attempt_id"]
    stale_owner = held is False and isinstance(owner, dict) and owner.get("schema_version") == OWNER_SCHEMA and owner.get("manifest_sha256") == manifest_hash and owner.get("current_row_id") == row["row_id"] and owner.get("current_attempt_id") == row["attempt_id"]
    if active:
        return {**base, "status": "active", "reason": "lock and owner identify current row", "active_log_path": owner.get("current_log_path"), "stale_owner_free_lock": False}
    if held is not False:
        return {**base, "status": "held_lock_active_or_unknown", "reason": "output ownership held or unknown", "stale_owner_free_lock": False}
    try:
        entries = attempts(manifest_hash, row)
        candidate = candidate_attempt(manifest, manifest_hash, row, entries)
        if candidate:
            attempt, intent = candidate
            status, reason = publication_state(row, attempt, intent)
            return {**base, "status": status, "reason": reason, "physical_attempt_id": attempt.name, "stale_owner_free_lock": stale_owner}
    except (RunnerError, OSError, ValueError) as exc:
        return {**base, "status": "canonical_collision_or_ambiguous_state", "reason": str(exc), "stale_owner_free_lock": stale_owner}
    canonical = Path(row["output_root"])
    legacy = [str(path) for path in [Path(row["merge_report_path"]), Path(row["log_path"]), *snapshots(row).values()] if path.exists()]
    if canonical.exists() or legacy:
        return {**base, "status": "canonical_collision_or_ambiguous_state", "reason": "canonical or public control bytes have no validated attempt", "existing_paths": ([str(canonical)] if canonical.exists() else []) + legacy, "stale_owner_free_lock": stale_owner}
    if held is None and entries:
        return {**base, "status": "held_lock_active_or_unknown", "reason": "lock state unknown with attempt evidence", "stale_owner_free_lock": False}
    if held is True and entries:
        return {**base, "status": "held_lock_active_or_unknown", "reason": "lock held without matching row owner", "stale_owner_free_lock": False}
    if entries:
        return {**base, "status": "partial_staging_attempt", "reason": "incomplete attempt preserved; explicit rerun may start a fresh physical attempt", "physical_attempt_id": entries[-1].name, "stale_owner_free_lock": stale_owner}
    if stale_owner:
        return {**base, "status": "stale_owner_free_lock", "reason": "owner metadata is stale; OS lock is free", "stale_owner_free_lock": True}
    return {**base, "status": "not_started", "reason": "no row-owned artifacts", "stale_owner_free_lock": False}


def read_only(mode: str, manifest: dict[str, Any], manifest_hash: str, row_id: str | None = None,
              runtime_overlay: dict[str, Any] | None = None) -> dict[str, Any]:
    held, lock_probe_error = probe_lock(Path(manifest["lock_path"]))
    owner = read_owner(manifest)
    selected = [select_row(manifest, row_id)] if row_id else manifest["rows"]
    if runtime_overlay is not None and (mode != "plan_only" or len(selected) != 1):
        raise RunnerError("runtime_overlay_mismatch", "overlay plan requires exactly one selected row")
    effective = (effective_runtime(manifest, manifest_hash, selected[0], runtime_overlay)
                 if runtime_overlay is not None else None)
    rows = []
    for row in selected:
        unit_held, unit_error = probe_lock(unit_lock_path(row))
        row_held = True if True in (held, unit_held) else None if None in (held, unit_held) else False
        row_owner = unit_owner(row) or owner
        state = classify(manifest, manifest_hash, row, row_held, row_owner)
        state.update(lock_path=str(unit_lock_path(row)), lock_held=row_held,
                     lock_probe_error=unit_error or lock_probe_error, owner_metadata=row_owner)
        try:
            reservation = read_reservation(row)
            if reservation:
                observation = query_reservation(reservation) if reservation["backend"] == "condor" else {
                    "state": "LOCAL_ACTIVE" if row_held is True else "LOCAL_RECONCILE" if row_held is False else "UNKNOWN", "scheduler_id": None}
                state["reservation"] = {**observation, "backend": reservation["backend"],
                                        "completion_authority": False}
        except RunnerError as exc:
            state["reservation"] = {"state": "UNKNOWN", "error_code": exc.code, "reason": str(exc), "completion_authority": False}
        rows.append(state)
    if mode == "plan_only":
        for row, status in zip(selected, rows):
            status["action"] = "skip" if status["status"] == "complete" else "execute" if status["status"] in {"not_started", "stale_owner_free_lock", "partial_staging_attempt"} else "reconcile" if status["status"] in {"complete_validated_staging_not_promoted", "promoted_incomplete", "promoted_complete_without_receipt"} else "block"
            if status.get("reservation") and status["status"] != "complete":
                status["action"] = "block"
                status["reason"] = "logical reservation requires owner/H2 reconciliation"
            status["resolved_argv"] = resolved_argv(manifest, row)
            if status["action"] == "execute":
                availability = execution_input_availability(manifest, row, effective)
                status["execution_input_availability"] = availability
                if not availability["ready"]:
                    status["action"] = "block"
                    status["reason"] = "execution inputs are not launch-ready"
            else:
                status["execution_input_availability"] = {
                    "required": False,
                    "checked": False,
                    "ready": status["action"] in {"skip", "reconcile"},
                    "reason": "valid receipt permits skip without historical execution inputs" if status["action"] == "skip" else "validated attempt permits reconciliation without historical execution inputs" if status["action"] == "reconcile" else "row state blocks before execution-input checks",
                }
    result = {"schema": "topeft_datacard_matrix_read_only_result_v2", "mode": mode, "manifest_sha256": manifest_hash, "runtime_contract_digest": runtime_digest(manifest["runtime_contract"]), "lock_held": True if any(row["lock_held"] is True for row in rows) else None if any(row["lock_held"] is None for row in rows) else False, "lock_probe_error": lock_probe_error, "owner_metadata": owner, "mutated": False, "rows": rows}
    if runtime_overlay is not None:
        result.update(overlay_provenance(manifest, manifest_hash, selected[0], runtime_overlay))
    if mode == "plan_only":
        result["launch_ready"] = all(row["action"] in {"skip", "execute", "reconcile"} for row in rows)
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


def owner_payload(manifest: dict[str, Any], manifest_hash: str, owner_started_at: str, row: dict[str, Any] | None = None, log_path: Path | None = None) -> dict[str, Any]:
    runtime = manifest["runtime_contract"]
    return {"schema_version": OWNER_SCHEMA, "pid": os.getpid(), "hostname": socket.gethostname(), "started_at": owner_started_at, "manifest_sha256": manifest_hash, "lock_path": manifest["lock_path"], "current_row_id": row["row_id"] if row else None, "current_attempt_id": row["attempt_id"] if row else None, "current_row_started_at": now() if row else None, "current_log_path": str(log_path) if log_path else None, "python_executable": runtime["python_executable"], "make_cards_path": runtime["make_cards_path"]}


def publish_matching(source: Path, destination: Path, source_record: dict[str, Any]) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            raise RunnerError("canonical_collision_or_ambiguous_state", f"published path is a symlink: {destination}")
        if not valid_record({**source_record, "path": str(destination)}, destination):
            raise RunnerError("canonical_collision_or_ambiguous_state", f"published bytes differ: {destination}")
        return
    if source.is_symlink() or not valid_record(source_record, source) or source.stat().st_size == 0:
        raise RunnerError("canonical_collision_or_ambiguous_state", f"staged source differs: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno != 18:  # EXDEV: control_root may reside on another filesystem.
            raise RunnerError("canonical_collision_or_ambiguous_state", f"cannot publish {destination}: {exc}") from exc
        copy_once(source, destination)
    if not valid_record({**source_record, "path": str(destination)}, destination):
        raise RunnerError("canonical_collision_or_ambiguous_state", f"published bytes differ: {destination}")


def reconcile_row(manifest: dict[str, Any], manifest_hash: str, row: dict[str, Any]) -> dict[str, Any]:
    candidate = candidate_attempt(manifest, manifest_hash, row, attempts(manifest_hash, row))
    if candidate is None:
        raise RunnerError("canonical_collision_or_ambiguous_state", "no validated attempt for reconciliation", row_id=row["row_id"])
    attempt, intent = candidate
    state, reason = publication_state(row, attempt, intent)
    if state not in {"complete_validated_staging_not_promoted", "promoted_incomplete", "promoted_complete_without_receipt"}:
        raise RunnerError(state, reason, row_id=row["row_id"])
    root = Path(row["output_root"])
    if state == "complete_validated_staging_not_promoted":
        try:
            root.mkdir()  # Exclusive creation; a pre-existing directory is a collision.
        except FileExistsError as exc:
            raise RunnerError("canonical_collision_or_ambiguous_state", f"canonical destination exists: {root}", row_id=row["row_id"]) from exc
        try:
            os.link(attempt / "intent.json", root / INTENT_MARKER)
        except OSError as exc:
            raise RunnerError("canonical_collision_or_ambiguous_state", f"cannot mark canonical destination: {exc}", row_id=row["row_id"]) from exc
    source_outputs, source_artifacts = staged_paths(row, attempt)
    for source, destination, source_record in zip(source_outputs, map(Path, row["expected_output_paths"]), intent["primary_outputs"]):
        publish_matching(source, destination, source_record)
    for key, name in (("selected_wcs_snapshot", "selectedWCs.txt"), ("scalings_snapshot", "scalings-preselect.json")):
        source = source_artifacts[key]
        publish_matching(source, root / name, record(source))
    for key, destination in canonical_artifacts(row).items():
        publish_matching(source_artifacts[key], destination, intent["artifacts"][key])
    state, reason = publication_state(row, attempt, intent)
    if state != "promoted_complete_without_receipt":
        raise RunnerError(state, reason, row_id=row["row_id"])
    runtime = manifest["runtime_contract"]
    receipt = {"schema": RECEIPT_SCHEMA, "manifest_sha256": manifest_hash, "row_id": row["row_id"], "attempt_id": row["attempt_id"], "physical_attempt_id": attempt.name, "resolved_argv": resolved_argv(manifest, row), "execution_argv": intent["execution_argv"], "start_timestamp": intent["start_timestamp"], "end_timestamp": intent["end_timestamp"], "command_return_code": 0, "runtime_contract_id": runtime["contract_id"], "python_executable": runtime["python_executable"], "make_cards_path": runtime["make_cards_path"], "runtime_fingerprints": runtime["fingerprints"], "runtime_contract_digest": runtime_digest(runtime), "primary_outputs": [record(Path(path)) for path in row["expected_output_paths"]], "artifacts": {key: record(path) for key, path in canonical_artifacts(row).items()}}
    if "runtime_contract_overlay" in intent:
        receipt.update({key: intent[key] for key in overlay_provenance_fields()})
    valid, reason = validate_receipt(receipt, manifest, manifest_hash, row)
    if not valid:
        raise RunnerError("canonical_collision_or_ambiguous_state", reason, row_id=row["row_id"])
    final = receipt_path(manifest, row)
    atomic_json(receipt, final, False)
    valid, reason = validate_receipt(json.loads(final.read_text()), manifest, manifest_hash, row)
    if not valid:
        raise RunnerError("invalid_receipt", reason, row_id=row["row_id"])
    return {"row_id": row["row_id"], "attempt_id": row["attempt_id"], "physical_attempt_id": attempt.name, "action": "reconciled", "receipt_path": str(final)}


def execute_row(manifest: dict[str, Any], manifest_hash: str, row: dict[str, Any], owner_started_at: str, *, lease_fds: tuple[int, ...] = (), backend: str = "local", scheduler_id: str | None = None, runtime_overlay: dict[str, Any] | None = None) -> dict[str, Any]:
    if runtime_overlay is not None:
        effective_runtime(manifest, manifest_hash, row, runtime_overlay)
    entries = attempts(manifest_hash, row)
    next_index = int(entries[-1].name[-6:]) + 1 if entries else 1
    if next_index > 999999:
        raise RunnerError("canonical_collision_or_ambiguous_state", "physical attempt index exhausted", row_id=row["row_id"])
    root = attempt_root(manifest_hash, row)
    root.mkdir(parents=True, exist_ok=True)
    attempt = root / f"attempt_{next_index:06d}"
    attempt.mkdir()
    log = attempt / "producer.log"
    atomic_json({**owner_payload(manifest, manifest_hash, owner_started_at, row, log), "lock_path": str(unit_lock_path(row)), "backend": backend, "scheduler_id": scheduler_id}, unit_owner_path(row), True)
    try:
        handle = log.open("xb")
    except FileExistsError as exc:
        raise RunnerError("canonical_collision_or_ambiguous_state", f"log exists: {log}", row_id=row["row_id"]) from exc
    argv, started = staged_argv(manifest, row, attempt), now()
    with handle:
        handle.write((json.dumps({"row_id": row["row_id"], "attempt_id": row["attempt_id"], "physical_attempt_id": attempt.name, "resolved_argv": resolved_argv(manifest, row), "execution_argv": argv, "start_timestamp": started}, sort_keys=True) + "\n").encode())
        handle.flush()
        completed = subprocess.run(argv, cwd=row["working_directory"], stdout=handle, stderr=subprocess.STDOUT, check=False, pass_fds=lease_fds)
        os.fsync(handle.fileno())
    if completed.returncode:
        raise RunnerError("row_command_failed", f"direct producer argv returned {completed.returncode}; no retry was attempted", row_id=row["row_id"])
    output_paths, staged_artifacts = staged_paths(row, attempt)
    if any(not path.is_file() or path.stat().st_size == 0 for path in output_paths):
        raise RunnerError("expected_output_check_failed", "expected output missing or empty", row_id=row["row_id"])
    stage_root, merge = attempt / "cards", attempt / "merge_report.json"
    sources = {"selected_wcs": stage_root / "selectedWCs.txt", "scalings": stage_root / "scalings-preselect.json", "merge_report": merge}
    destinations = {key: staged_artifacts[f"{key}_snapshot"] if key != "merge_report" else staged_artifacts["merge_report_snapshot"] for key in sources}
    for key, source in sources.items():
        copy_once(source, destinations[key])
    for path in [*output_paths, *sources.values(), *staged_artifacts.values()]:
        if path.is_symlink() or not path.is_file():
            raise RunnerError("expected_output_check_failed", f"staged artifact is not a regular file: {path}", row_id=row["row_id"])
        with path.open("rb") as source_handle:
            os.fsync(source_handle.fileno())
    intent = {"schema": INTENT_SCHEMA, "manifest_sha256": manifest_hash, "row_id": row["row_id"], "attempt_id": row["attempt_id"], "physical_attempt_id": attempt.name, "resolved_argv": resolved_argv(manifest, row), "execution_argv": argv, "runtime_contract_digest": runtime_digest(manifest["runtime_contract"]), "command_return_code": 0, "start_timestamp": started, "end_timestamp": now(), "primary_outputs": [record(path) for path in output_paths], "artifacts": {key: record(path) for key, path in staged_artifacts.items()}}
    if runtime_overlay is not None:
        intent.update(overlay_provenance(manifest, manifest_hash, row, runtime_overlay))
    atomic_json(intent, attempt / "intent.json", False)
    valid, reason, _ = valid_intent(manifest, manifest_hash, row, attempt)
    if not valid:
        raise RunnerError("canonical_collision_or_ambiguous_state", reason, row_id=row["row_id"])
    result = reconcile_row(manifest, manifest_hash, row)
    result["action"] = "executed"
    return result


def execute_unit(manifest: dict[str, Any], manifest_hash: str, row: dict[str, Any], *, backend: str = "local", scheduler_id: str | None = None, reservation_token: str | None = None, runtime_overlay: dict[str, Any] | None = None) -> dict[str, Any]:
    """The sole mutating primitive: acquire ownership, reconcile, then execute."""
    effective = (effective_runtime(manifest, manifest_hash, row, runtime_overlay)
                 if runtime_overlay is not None else None)
    with unit_lease(manifest, row) as lease_fds:
        if backend == "condor":
            scheduler_id = worker_identity(row, reservation_token)
            reservation = read_reservation(row)
            if (reservation is None or reservation["backend"] != "condor"
                    or reservation["owner_id"] != reservation_token):
                raise RunnerError("row_reserved", "Condor worker requires its row/token reservation", row_id=row["row_id"])
            reservation["scheduler_id"] = scheduler_id
            atomic_json(reservation, reservation_path(row), True)
        else:
            reservation = reconcile_reservation(manifest, manifest_hash, row)
            if backend not in {"local", "manual"} or reservation is not None:
                raise RunnerError("row_reserved", "row has a logical execution owner", row_id=row["row_id"])
            reservation = new_reservation(row, backend)
            atomic_json(reservation, reservation_path(row), False)
        try:
            state = classify(manifest, manifest_hash, row, False, unit_owner(row) or read_owner(manifest))
            if state["status"] == "complete":
                return {"row_id": row["row_id"], "attempt_id": row["attempt_id"], "action": "skipped_valid_receipt", "receipt_path": state["receipt_path"]}
            if state["status"] in {"complete_validated_staging_not_promoted", "promoted_incomplete", "promoted_complete_without_receipt"}:
                return reconcile_row(manifest, manifest_hash, row)
            if state["status"] not in {"not_started", "stale_owner_free_lock", "partial_staging_attempt"}:
                raise RunnerError(state["status"], state["reason"], row_id=row["row_id"])
            validate_unit_layout(manifest)
            validate_runtime(effective if effective is not None else manifest["runtime_contract"])
            validate_row_inputs(row)
            fresh = classify(manifest, manifest_hash, row, False, unit_owner(row) or read_owner(manifest))
            if fresh["status"] not in {"not_started", "stale_owner_free_lock", "partial_staging_attempt"}:
                raise RunnerError(fresh["status"], fresh["reason"], row_id=row["row_id"])
            return execute_row(manifest, manifest_hash, row, now(), lease_fds=lease_fds, backend=backend, scheduler_id=scheduler_id, runtime_overlay=runtime_overlay)
        finally:
            unit_owner_path(row).unlink(missing_ok=True)
            if backend != "condor":
                reservation_path(row).unlink()


def execute(manifest: dict[str, Any], manifest_hash: str, *, backend: str = "local", scheduler_id: str | None = None) -> dict[str, Any]:
    """Sequential convenience orchestration over the canonical unit primitive."""
    if backend == "condor":
        raise RunnerError("row_reserved", "Condor workers must select exactly one reserved unit")
    return {"schema": "topeft_datacard_matrix_execution_result_v2", "manifest_sha256": manifest_hash,
            "rows": [execute_unit(manifest, manifest_hash, row, backend=backend, scheduler_id=scheduler_id) for row in manifest["rows"]]}


def overlay_scheduler(result: dict[str, Any], overlay: Any) -> None:
    """Attach supplied observations without deriving producer completion/retry."""
    overlay = exact_fields(overlay, {"schema", "parent_integrity", "observed_at", "units"}, "scheduler_status")
    if overlay["schema"] != "topeft_datacard_scheduler_status_v1" or overlay["parent_integrity"] != result["manifest_sha256"]:
        raise RunnerError("scheduler_status_error", "scheduler observation parent binding differs")
    text(overlay["observed_at"], "observed_at")
    if not isinstance(overlay["units"], list):
        raise RunnerError("scheduler_status_error", "units must be a list")
    states = {row["row_id"]: row for row in result["rows"]}
    seen, seen_jobs = set(), set()
    for item in overlay["units"]:
        exact_fields(item, {"row_id", "scheduler_id", "state"}, "scheduler_unit")
        row_id = text(item["row_id"], "row_id")
        if row_id not in states or row_id in seen:
            raise RunnerError("scheduler_status_error", "unknown or duplicate scheduler row")
        seen.add(row_id)
        for key in ("scheduler_id", "state"):
            text(item[key], key)
        if item["scheduler_id"] in seen_jobs:
            raise RunnerError("scheduler_status_error", "duplicate scheduler job")
        seen_jobs.add(item["scheduler_id"])
        states[row_id]["scheduler_metadata"] = {**item, "observed_at": overlay["observed_at"], "completion_authority": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a prequalified datacard matrix with byte-bound receipts.")
    modes = parser.add_mutually_exclusive_group(); modes.add_argument("--plan-only", action="store_true"); modes.add_argument("--status", action="store_true")
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--unit", type=Path, help="Derived execution-unit descriptor")
    parser.add_argument("--row-id", help="Select one row from the original manifest")
    modes.add_argument("--derive-units", action="store_true", help="Print deterministic descriptors without writing state")
    modes.add_argument("--reconcile-reservations", action="store_true", help="Reconcile current reservations through H2 proactively; claims also reconcile automatically")
    parser.add_argument("--backend", choices=("local", "condor", "manual"), default="local")
    parser.add_argument("--scheduler-id", help="Submitted scheduler locator for observability; local job ad supplies worker metadata")
    parser.add_argument("--reservation-token", help="Generated per-unit dispatch owner token")
    parser.add_argument("--unit-integrity", help="Dispatch guard for descriptor bytes")
    parser.add_argument("--scheduler-status", type=Path, help="Read-only operational metadata overlay")
    parser.add_argument("--runtime-contract-overlay", type=Path, help="Exact invocation-local runtime fingerprint replacement JSON")
    parser.add_argument("--runner-integrity", help="Dispatch guard for the worker runner bytes")
    args = parser.parse_args(argv)
    if bool(args.manifest) == bool(args.unit) or (args.unit and (args.row_id or args.derive_units)):
        parser.error("supply either MANIFEST [--row-id ROW] or --unit DESCRIPTOR")
    if args.row_id and args.derive_units:
        parser.error("--derive-units describes the whole workload; use --row-id for execution/status")
    if args.scheduler_status and not args.status:
        parser.error("--scheduler-status requires --status")
    if args.unit_integrity and not args.unit:
        parser.error("--unit-integrity requires --unit")
    if args.runtime_contract_overlay and (args.status or args.derive_units or args.reconcile_reservations):
        parser.error("--runtime-contract-overlay requires one selected plan or execution row")
    try:
        if args.runner_integrity and hash_file(Path(__file__)) != args.runner_integrity:
            raise RunnerError("stale_dispatch_input", "worker runner bytes changed")
        if args.derive_units:
            print(json.dumps(derive_units(args.manifest), indent=2, sort_keys=True))
            return 0
        if args.unit:
            raw_unit = args.unit.read_bytes()
            if args.unit_integrity and hashlib.sha256(raw_unit).hexdigest() != args.unit_integrity:
                raise RunnerError("stale_dispatch_input", "execution-unit descriptor bytes changed")
            manifest, manifest_hash, row = resolve_unit(json.loads(raw_unit))
        else:
            manifest, manifest_hash = load_manifest(args.manifest)
            row = select_row(manifest, args.row_id) if args.row_id else None
        runtime_overlay = None
        if args.runtime_contract_overlay:
            if row is None:
                raise RunnerError("runtime_overlay_mismatch", "overlay requires exactly one selected row")
            runtime_overlay = json.loads(args.runtime_contract_overlay.read_text())
            effective_runtime(manifest, manifest_hash, row, runtime_overlay)
        if args.plan_only or args.status:
            result = read_only("plan_only" if args.plan_only else "status", manifest, manifest_hash, row["row_id"] if row else None, runtime_overlay)
        elif args.reconcile_reservations:
            states = []
            for selected in ([row] if row else manifest["rows"]):
                with unit_lease(manifest, selected):
                    evidence = {}
                    reservation = reconcile_reservation(manifest, manifest_hash, selected, evidence)
                    states.append({"row_id": selected["row_id"], "reservation": reservation,
                                   "terminal_evidence": evidence or None,
                                   "h2": classify(manifest, manifest_hash, selected, False)})
            result = {"rows": states, "producer_executed": False}
        elif row:
            result = {"schema": "topeft_datacard_matrix_execution_result_v2", "manifest_sha256": manifest_hash,
                      "rows": [execute_unit(manifest, manifest_hash, row, backend=args.backend, scheduler_id=args.scheduler_id, reservation_token=args.reservation_token, runtime_overlay=runtime_overlay)]}
        else:
            result = execute(manifest, manifest_hash, backend=args.backend, scheduler_id=args.scheduler_id)
        if args.scheduler_status:
            overlay_scheduler(result, json.loads(args.scheduler_status.read_text()))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2 if args.plan_only and not result["launch_ready"] else 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"schema": "topeft_datacard_matrix_error_v2", "status": "dispatch_read_error", "message": str(exc)}), file=sys.stderr)
        return 2
    except RunnerError as exc:
        error = {"schema": "topeft_datacard_matrix_error_v2", "status": exc.code, "message": str(exc)}
        if exc.row_id is not None: error["row_id"] = exc.row_id
        print(json.dumps(error, indent=2, sort_keys=True), file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
