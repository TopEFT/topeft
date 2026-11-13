"""Public plotting control and persistent SR CR-like campaigns.

Plans bind qualified producer sidecars, never a predicted figure inventory. A
successful child is observed in its private root; a receipt then owns the exact
PNG bytes copied to the global canonical root. This module never loads a PKL.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import shlex
import time

import yaml

from analysis.topeft_run2 import plot_campaign_backend as backend
from analysis.topeft_run2.plot_campaign_backend import (
    CampaignError, ActiveChildUnresolved, ChildDefinitelyNotStarted,
    ChildDefinitelyTerminal, require, utc_now, error_description, sha256_file,
    identity, atomic_json, lifecycle_receipt_path, guard_terminal,
)
from topeft.modules.plot_output_layout import normalize_plot_period, canonical_observed_path


RUN_DIRECTORY = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIRECTORY.parents[1]
PROFILES = {
    "sr_crlike_run2": ("run2",),
    "sr_crlike_run3": ("run3",),
    "sr_crlike_run2_run3": ("run2", "run3"),
    "sr_datacard_run2": ("run2",),
    "sr_datacard_run3": ("run3",),
    "sr_datacard_run2_run3": ("run2", "run3"),
}
YEARS = {
    "run2": ("UL16", "UL16APV", "UL17", "UL18"),
    "run3": ("2022", "2022EE", "2023", "2023BPix"),
}
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
terminal_postprocessing_pending = "terminal postprocessing pending"
DEFAULT_EXECUTION_POLICY = {
    "backend": "local", "resource_observer": "cgroup_v2",
    "hard_limit_bytes": 245000000000,
}
SUMMARY_SCHEMA_VERSION = 2
UNIT_FIELDS = {
    "unit_id", "component", "producer_unit_id", "input_path",
    "sidecar_path", "qualification_source", "period_view",
}


class OutputCollision(CampaignError):
    """Canonical output cannot be written without risking another owner's bytes."""


def best_effort_utc_now():
    try:
        return utc_now()
    except Exception:
        return None


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe JSON: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"invalid JSON: {path}") from error


def safe_name(value, label):
    require(isinstance(value, str) and SAFE_NAME.fullmatch(value)
            and value not in {".", ".."}, f"unsafe {label}: {value!r}")
    return value


def safe_relative(value):
    try:
        return canonical_observed_path(Path(), value)
    except ValueError as error:
        raise CampaignError(f"noncanonical PNG path: {value!r}") from error


def component_root(root, profile, component):
    return root / ("components" if len(PROFILES[profile]) > 1 else ".") / component if len(PROFILES[profile]) > 1 else root


def state_path(root, profile, component):
    family = "sr_datacard" if profile.startswith("sr_datacard_") else "sr_crlike"
    name = f".{family}_{component}_plot_campaign_state_v1.json"
    return component_root(root, profile, component) / name


def coordinator_path(root):
    return root / ".plot_campaign_v1.json"


@contextmanager
def campaign_lock(root):
    path = root / ".plot_campaign_v1.lock"
    require(not path.is_symlink(), "campaign lock is a symbolic link")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CampaignError("campaign writer lock is held") from error
        yield
    finally:
        os.close(descriptor)


def private_root(root, profile, unit_id, component, attempt_id):
    return component_root(root, profile, component) / ".campaign_units" / unit_id / attempt_id


def runtime_identity():
    """New campaigns store a non-gating marker; old provenance stays historical."""
    return {"provenance_status": "not_collected"}


def normalize_unit(raw, profile, root, *, check_files=True):
    require(isinstance(raw, dict) and set(raw) == UNIT_FIELDS,
            "plan unit fields must exactly match schema v1")
    unit_id = safe_name(raw["unit_id"], "unit_id")
    component = raw["component"]
    require(component in PROFILES[profile], "unit component disagrees with profile")
    safe_name(raw["producer_unit_id"], "producer_unit_id")
    source = raw["qualification_source"]
    require(isinstance(source, dict)
            and set(source) == {"status", "locator", "input_sha256", "sidecar_sha256"}
            and source["status"] == "accepted"
            and isinstance(source["locator"], str) and source["locator"].strip()
            and all(isinstance(source[key], str) and SHA256.fullmatch(source[key])
                    for key in ("input_sha256", "sidecar_sha256")),
            "qualification_source must bind accepted input and sidecar identities")
    view = raw["period_view"]
    require(view == "combined" or view in YEARS[component], "unsupported period_view")
    input_path = Path(raw["input_path"])
    sidecar_path = Path(raw["sidecar_path"])
    require(input_path.is_absolute() and sidecar_path.is_absolute()
            and ".." not in input_path.parts and ".." not in sidecar_path.parts
            and str(sidecar_path) == str(input_path) + ".metadata.json",
            "input/sidecar path binding is invalid")
    require(sidecar_path.is_file() and not sidecar_path.is_symlink(), "sidecar is missing or unsafe")
    sidecar = read_json(sidecar_path)
    require(sidecar.get("metadata_schema_version") == 2, "unsupported sidecar schema")
    artifact = sidecar.get("artifact")
    provenance = sidecar.get("sumw2_storage_provenance")
    require(isinstance(artifact, dict) and isinstance(provenance, dict), "sidecar lacks artifact provenance")
    size = artifact.get("pkl_size_bytes")
    digest = artifact.get("pkl_sha256")
    require(artifact.get("pkl_basename") == input_path.name
            and isinstance(size, int) and size > 0
            and isinstance(digest, str) and SHA256.fullmatch(digest)
            and digest == source["input_sha256"],
            "sidecar input identity is invalid")
    sidecar_digest = sha256_file(sidecar_path)
    require(sidecar_digest == source["sidecar_sha256"],
            "sidecar contradicts accepted qualification identity")
    variables = provenance.get("runtime_histogram_families")
    require(isinstance(variables, list) and variables
            and len(variables) == len(set(variables))
            and all(isinstance(item, str) and SAFE_NAME.fullmatch(item) for item in variables),
            "sidecar histogram families are invalid")
    if check_files:
        require(input_path.is_file() and not input_path.is_symlink()
                and input_path.stat().st_size == size,
                "input missing or size contradicts bound sidecar")
    return {
        **raw,
        "variables": variables,
        "accepted_input_identity": {"sha256": digest, "size_bytes": size},
        "accepted_sidecar_identity": {"sha256": sidecar_digest},
        "private_unit_root": str(private_root(root, profile, unit_id, component, "attempt_0001")),
    }


def normalize_plan(plan_path, profile, tag, root, existing_plan=None):
    raw = read_json(plan_path)
    require(isinstance(raw, dict) and set(raw) == {"schema_version", "campaign_profile", "campaign_tag", "plan_revision", "units"},
            "plan fields must exactly match schema v1")
    require(raw["schema_version"] == 1 and raw["campaign_profile"] == profile
            and raw["campaign_tag"] == tag and type(raw["plan_revision"]) is int
            and raw["plan_revision"] >= 1 and isinstance(raw["units"], list)
            and raw["units"], "invalid plan identity or revision")
    historical = ({unit["unit_id"]: unit for unit in existing_plan["units"]}
                  if existing_plan is not None else {})
    units = []
    for unit in raw["units"]:
        if isinstance(unit, dict) and unit.get("unit_id") in historical:
            prior = historical[unit["unit_id"]]
            require(unit == {key: prior[key] for key in UNIT_FIELDS},
                    "plan revision attempts to rewrite historical binding")
            units.append(prior)
        else:
            units.append(normalize_unit(unit, profile, root))
    names = [unit["unit_id"] for unit in units]
    require(len(names) == len(set(names)), "duplicate unit_id")
    require(set(unit["component"] for unit in units) == set(PROFILES[profile]),
            "plan lacks a required profile component")
    return {**raw, "units": units}


def year_args(unit):
    view = unit["period_view"]
    return list(YEARS[unit["component"]] if view == "combined" else (view,))


def period(unit):
    return normalize_plot_period(unit["component"], unit["period_view"])


def child_argv(unit, attempt_root, *, historical=False):
    chosen_period = period(unit)
    prefix = ([str(RUN_DIRECTORY / "run_plotter.sh")] if historical else
              [sys.executable, str(RUN_DIRECTORY / "make_cr_and_sr_plots.py")])
    years = year_args(unit) if historical else normalize_year_tokens(year_args(unit))
    return [
        *prefix, "-f", unit["input_path"],
        "-o", str(attempt_root / chosen_period), "-n", "merged",
        "-y", *years, "--sr", "--unblind", "--variables", *unit["variables"],
        "--channel-output", "merged", "--binning", "fitting", "--workers", "1",
        "--output-formats", "png", "--year-coverage-policy", "warn",
        "--suppress-merge-report", "--canonical-campaign-root", str(attempt_root),
        "--canonical-period", chosen_period, "--negative-report-suffix", unit["unit_id"],
    ]


def attempt_argv(unit, attempt):
    """Decode historical identity without ever launching the retired wrapper path."""
    contract = attempt.get("execution_contract")
    if contract is None:
        return child_argv(unit, Path(attempt["private_unit_root"]), historical=True)
    require(contract.get("version") == 1 and contract.get("backend") in {"local", "condor"},
            "unsupported execution contract")
    argv = child_argv(unit, Path(attempt["private_unit_root"]))
    # Executable location is an observed execution identity, not today's interpreter.
    argv[0] = contract["argv"][0]
    require(contract["argv"] == argv, "execution argv contradicts unit binding")
    return argv


def new_component_state(plan, component, runtime):
    units = {}
    for unit in plan["units"]:
        if unit["component"] == component:
            units[unit["unit_id"]] = {
                "binding": unit, "status": "planned", "attempts": [],
                "materializations": {},
            }
    return {
        "schema_version": 1, "campaign_profile": plan["campaign_profile"],
        "campaign_tag": plan["campaign_tag"], "component": component,
        "plan_revision": plan["plan_revision"], "runtime_contract_identity": runtime,
        "units": units, "plan_history": [plan["plan_revision"]],
    }


def validate_state(state, plan, component, runtime):
    require(state.get("schema_version") == 1 and state.get("component") == component
            and state.get("campaign_profile") == plan["campaign_profile"]
            and state.get("campaign_tag") == plan["campaign_tag"]
            and state.get("runtime_contract_identity") == runtime,
            "component state identity or runtime contract contradicts plan")
    plan_units = {unit["unit_id"]: unit for unit in plan["units"] if unit["component"] == component}
    require(set(state.get("units", {})) == set(plan_units), "component state unit set contradicts plan")
    for unit_id, unit_state in state["units"].items():
        require(unit_state.get("binding") == plan_units[unit_id]
                and unit_state.get("status") in {"planned", "running", "success", "failed", "interrupted"},
                f"unit binding/status contradiction: {unit_id}")
        attempts = unit_state.get("attempts")
        status = unit_state["status"]
        require(isinstance(attempts, list) and (
            len(attempts) == 0 if status == "planned" else len(attempts) in {1, 2}),
                f"unit status/attempt history contradiction: {unit_id}")
        if attempts:
            require(all(isinstance(item, dict) for item in attempts),
                    f"unit attempt history is malformed: {unit_id}")
            if len(attempts) == 2:
                require(attempts[0].get("attempt_id") == "attempt_0001"
                        and attempts[0].get("result") == "failed"
                        and attempts[1].get("attempt_id") == "attempt_0002",
                        f"failed-attempt recovery history contradiction: {unit_id}")
            attempt = attempts[-1]
            require(isinstance(attempt, dict) and attempt.get("result") == status,
                    f"unit status/attempt result contradiction: {unit_id}")
            if status == "success":
                require(isinstance(attempt.get("completion_receipt_identity"), str)
                        and SHA256.fullmatch(attempt["completion_receipt_identity"])
                        and attempt.get("materialization_status") == "complete",
                        f"successful unit lacks receipt or complete materialization: {unit_id}")
                if attempt.get("recovery_provenance_identity") is not None:
                    require(isinstance(attempt["recovery_provenance_identity"], str)
                            and SHA256.fullmatch(attempt["recovery_provenance_identity"])
                            and all(attempt.get(key) is None for key in
                                    ("guard_result", "child_result", "tee_result")),
                            f"recovered success invents lifecycle results: {unit_id}")


def inventory_pngs(attempt_root, variables, expected_period):
    records = []
    for path in sorted(attempt_root.rglob("*.png")):
        require(path.is_file() and not path.is_symlink(), f"unsafe observed PNG: {path}")
        relative = path.relative_to(attempt_root).as_posix()
        safe_relative(relative)
        require(Path(relative).parts[0] == expected_period,
                f"PNG period contradicts bound unit: {relative}")
        require(path.stat().st_size > 0, f"empty observed PNG: {path}")
        variable = next((value for value in sorted(variables, key=len, reverse=True)
                         if path.name.endswith(f"_{value}_fitting.png")), None)
        require(variable is not None, f"observed PNG has no bound variable: {relative}")
        records.append({"relative_path": relative, "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path), "variable": variable})
    return records


def completion_receipt_path(attempt_root):
    return attempt_root / "_metadata" / "completion_receipt_v1.json"


def recovery_provenance_path(attempt_root):
    return attempt_root / "_metadata" / "recovery_provenance_v1.json"


def verify_lifecycle_receipt(receipt, attempt, argv):
    require(isinstance(receipt, dict), "lifecycle receipt is not a JSON object")
    attempt_root = Path(attempt["private_unit_root"])
    guard_path = attempt_root / "_metadata" / "guard.jsonl"
    require(receipt.get("schema_version") == 1
            and receipt.get("attempt_id") == attempt["attempt_id"]
            and receipt.get("command_identity") == identity(argv)
            and receipt.get("private_unit_root") == str(attempt_root)
            and receipt.get("terminal_guard_evidence") == guard_terminal(guard_path)
            and receipt.get("guard_evidence_sha256") == sha256_file(guard_path)
            and isinstance(receipt.get("ended_at_utc"), str)
            and type(receipt.get("guard_result")) is int
            and type(receipt.get("tee_result")) is int
            and (receipt.get("child_result") is None
                 or type(receipt.get("child_result")) is int),
            "lifecycle receipt binding or terminal evidence contradiction")
    terminal = receipt["terminal_guard_evidence"]
    require(receipt["child_result"] == (
        terminal.get("returncode") if terminal.get("termination_kind") == "normal" else None),
        "lifecycle receipt child result contradicts guard")
    return receipt


def create_receipt(plan, unit, attempt, runtime, recovery_provenance=None):
    attempt_root = Path(attempt["private_unit_root"])
    if recovery_provenance is None:
        require(attempt.get("guard_result") == 0 and attempt.get("child_result") == 0
                and attempt.get("tee_result") == 0, "terminal lifecycle is not successful")
    else:
        require(all(attempt.get(key) is None for key in
                    ("guard_result", "child_result", "tee_result")),
                "recovery cannot claim unavailable lifecycle results")
    records = inventory_pngs(attempt_root, unit["variables"], period(unit))
    counts = Counter(record["variable"] for record in records)
    log_path = attempt_root / "_metadata" / "renderer.log"
    messages = []
    log_status = "missing"
    log_line_count = 0
    log_message_count = 0
    if log_path.is_file() and not log_path.is_symlink():
        try:
            with log_path.open(encoding="utf-8", errors="replace") as log_stream:
                for line in log_stream:
                    log_line_count += 1
                    if any(token in line.lower() for token in
                           ("warning", "error", "skipp", "missing")):
                        log_message_count += 1
                        if len(messages) < 200:
                            messages.append(line.rstrip("\n")[:500])
            log_status = "present"
        except OSError:
            log_status = "unreadable"
            messages = []
            log_line_count = 0
            log_message_count = 0
    elif log_path.exists() or log_path.is_symlink():
        log_status = "non_regular"
    receipt = {
        "schema_version": 1, "campaign_profile": plan["campaign_profile"],
        "campaign_tag": plan["campaign_tag"], "plan_revision": attempt["plan_revision"],
        "unit_id": unit["unit_id"], "attempt_id": attempt["attempt_id"],
        "input_identity": unit["accepted_input_identity"],
        "sidecar_identity": unit["accepted_sidecar_identity"],
        "runtime_contract_identity": runtime, "private_unit_root": str(attempt_root),
        "guard_result": attempt["guard_result"], "child_result": attempt["child_result"],
        "tee_result": attempt["tee_result"], "started_at_utc": attempt["started_at_utc"],
        "ended_at_utc": attempt["ended_at_utc"],
        "observed_png_relative_paths": [row["relative_path"] for row in records],
        "observed_png_count": len(records),
        "observed_png_sizes": {row["relative_path"]: row["size_bytes"] for row in records},
        "observed_png_sha256": {row["relative_path"]: row["sha256"] for row in records},
        "observed_png_count_by_variable": {value: counts[value] for value in unit["variables"]},
        "renderer_log_status": log_status, "renderer_log_line_count": log_line_count,
        "renderer_message_count": log_message_count, "renderer_messages": messages,
    }
    if recovery_provenance is not None:
        receipt["recovery_provenance_identity"] = identity(recovery_provenance)
    path = completion_receipt_path(attempt_root)
    require(not path.exists() and not path.is_symlink(), "completion receipt already exists")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_json(path, receipt)
    return receipt


def verify_receipt(receipt, plan, unit, attempt, runtime):
    require(isinstance(receipt, dict), "completion receipt is not a JSON object")
    recovery_identity = receipt.get("recovery_provenance_identity")
    if recovery_identity is None:
        lifecycle_valid = all(receipt.get(key) == 0 for key in
                              ("guard_result", "child_result", "tee_result"))
    else:
        provenance = read_json(recovery_provenance_path(Path(attempt["private_unit_root"])))
        lifecycle_valid = (
            isinstance(recovery_identity, str) and SHA256.fullmatch(recovery_identity)
            and identity(provenance) == recovery_identity
            and attempt.get("recovery_provenance_identity") == recovery_identity
            and all(receipt.get(key) is None and attempt.get(key) is None for key in
                    ("guard_result", "child_result", "tee_result"))
            and provenance.get("guard_evidence_sha256") == sha256_file(
                Path(attempt["private_unit_root"]) / "_metadata" / "guard.jsonl")
            and provenance.get("command_identity") == attempt.get("command_identity")
            and provenance.get("observed_png_count") == receipt.get("observed_png_count")
            and provenance.get("observed_png_sizes") == receipt.get("observed_png_sizes")
            and provenance.get("observed_png_sha256") == receipt.get("observed_png_sha256")
            and provenance == observed_recovery_provenance(plan, unit, attempt)
        )
    require(receipt.get("schema_version") == 1
            and receipt.get("campaign_profile") == plan["campaign_profile"]
            and receipt.get("campaign_tag") == plan["campaign_tag"]
            and receipt.get("plan_revision") == attempt["plan_revision"]
            and receipt.get("unit_id") == unit["unit_id"]
            and receipt.get("attempt_id") == attempt["attempt_id"]
            and receipt.get("private_unit_root") == attempt["private_unit_root"]
            and receipt.get("input_identity") == unit["accepted_input_identity"]
            and receipt.get("sidecar_identity") == unit["accepted_sidecar_identity"]
            and receipt.get("runtime_contract_identity") == runtime
            and lifecycle_valid,
            "completion receipt binding/lifecycle contradiction")
    records = inventory_pngs(Path(attempt["private_unit_root"]), unit["variables"], period(unit))
    paths = [row["relative_path"] for row in records]
    require(receipt.get("observed_png_relative_paths") == paths
            and receipt.get("observed_png_count") == len(paths)
            and receipt.get("observed_png_sizes") == {row["relative_path"]: row["size_bytes"] for row in records}
            and receipt.get("observed_png_sha256") == {row["relative_path"]: row["sha256"] for row in records}
            and receipt.get("observed_png_count_by_variable") == dict(
                (value, sum(row["variable"] == value for row in records)) for value in unit["variables"]),
            "completion receipt inventory contradiction")
    receipt_id = identity(receipt)
    require(attempt.get("completion_receipt_identity") is None
            or attempt.get("completion_receipt_identity") == receipt_id,
            "state/receipt identity contradiction")
    return receipt_id


def reject_symlink_ancestors(path, boundary):
    current = Path(path)
    boundary = Path(boundary)
    require(current == boundary or boundary in current.parents,
            "output path escapes campaign root")
    while current != boundary:
        require(not current.is_symlink(), f"symbolic-link output ancestor: {current}")
        current = current.parent
    require(not boundary.is_symlink(), f"symbolic-link campaign root: {boundary}")


def atomic_copy_without_overwrite(source, destination):
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(not destination.parent.is_symlink(), "canonical parent is a symbolic link")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def materialize(plan, component, state, unit_state, attempt, receipt, root, runtime):
    unit = unit_state["binding"]
    receipt_id = verify_receipt(receipt, plan, unit, attempt, runtime)
    for relative in receipt["observed_png_relative_paths"]:
        safe_relative(relative)
        source = Path(attempt["private_unit_root"]) / relative
        destination = canonical_observed_path(root, relative)
        reject_symlink_ancestors(destination.parent, root)
        digest = receipt["observed_png_sha256"][relative]
        binding = {"unit_id": unit["unit_id"], "receipt_identity": receipt_id,
                   "sha256": digest, "size_bytes": receipt["observed_png_sizes"][relative]}
        prior = None
        for other_state in state["units"].values():
            existing = other_state["materializations"].get(relative)
            if existing is not None:
                if prior is not None and prior != existing:
                    raise OutputCollision(f"canonical ownership collision: {relative}")
                prior = existing
        if prior is not None and prior != binding:
            raise OutputCollision(f"canonical ownership collision: {relative}")
        if destination.exists() or destination.is_symlink():
            if not (destination.is_file() and not destination.is_symlink()
                    and destination.stat().st_size == binding["size_bytes"]
                    and sha256_file(destination) == digest):
                raise OutputCollision(f"different-byte canonical destination collision: {destination}")
        if prior is None:
            unit_state["materializations"][relative] = binding
            atomic_json(state_path(root, plan["campaign_profile"], component), state)
        if not destination.exists():
            try:
                atomic_copy_without_overwrite(source, destination)
            except FileExistsError as error:
                raise OutputCollision(f"canonical destination collision: {destination}") from error
        require(destination.stat().st_size == binding["size_bytes"]
                and sha256_file(destination) == digest,
                f"canonical destination changed after copy: {destination}")
    attempt["materialization_status"] = "complete"
    attempt["completion_receipt_identity"] = receipt_id
    attempt["result"] = "success"
    attempt["failure_classification"] = None
    unit_state["status"] = "success"
    unit_state["human_output_root"] = str(root)
    atomic_json(state_path(root, plan["campaign_profile"], component), state)


def execute_unit(plan, component, state, unit_state, root, runtime, execution_options,
                 *, failed_attempt_recovery=False):
    unit = unit_state["binding"]
    if failed_attempt_recovery:
        verify_failed_attempt(plan, component, state, unit_state, root)
    else:
        require(unit_state["status"] == "planned" and not unit_state["attempts"],
                "only unattempted planned units may start")
    current_binding = normalize_unit({key: unit[key] for key in UNIT_FIELDS},
                                     plan["campaign_profile"], root)
    require(current_binding == unit, "input or sidecar identity changed since planning")
    attempt_id = f"attempt_{len(unit_state['attempts']) + 1:04d}"
    attempt_root = private_root(root, plan["campaign_profile"], unit["unit_id"], component, attempt_id)
    require(not attempt_root.exists() and not attempt_root.is_symlink(), "private attempt root is occupied")
    argv = child_argv(unit, attempt_root)
    attempt = {
        "attempt_id": attempt_id, "plan_revision": state["plan_revision"],
        "started_at_utc": utc_now(), "ended_at_utc": None,
        "command_identity": identity(argv), "private_unit_root": str(attempt_root),
        "execution_contract": {"version": 1, "argv": argv, **execution_options},
        "completion_receipt_identity": None, "materialization_status": "pending",
        "guard_result": None, "child_result": None, "tee_result": None,
        "result": "running", "failure_classification": None,
    }
    unit_state["attempts"].append(attempt)
    unit_state["status"] = "running"
    launch_possible = False
    try:
        atomic_json(state_path(root, plan["campaign_profile"], component), state)
        reject_symlink_ancestors(attempt_root.parent,
                                 component_root(root, plan["campaign_profile"], component))
        attempt_root.mkdir(mode=0o700, parents=True)
        launch_possible = True
        guard_result, child_result, tee_result = backend.execute(argv, attempt_root, attempt["execution_contract"])
        attempt.update(guard_result=guard_result, child_result=child_result,
                       tee_result=tee_result, result="interrupted",
                       failure_classification=terminal_postprocessing_pending)
        unit_state["status"] = "interrupted"
        atomic_json(state_path(root, plan["campaign_profile"], component), state)
        attempt["ended_at_utc"] = best_effort_utc_now()
        lifecycle = verify_lifecycle_receipt(read_json(lifecycle_receipt_path(attempt_root)),
                                            attempt, argv)
        require((guard_result, child_result, tee_result) == (
            lifecycle["guard_result"], lifecycle["child_result"], lifecycle["tee_result"]),
            "supervisor return contradicts lifecycle receipt")
        attempt["ended_at_utc"] = lifecycle["ended_at_utc"]
        atomic_json(state_path(root, plan["campaign_profile"], component), state)
        require((guard_result, child_result, tee_result) == (0, 0, 0),
                "terminal guard/child/tee failure")
        # A terminal child is ready for the separate finalize action.
    except Exception as error:
        if (attempt["guard_result"] is None and launch_possible
                and not isinstance(error, (ChildDefinitelyNotStarted, ChildDefinitelyTerminal))):
            attempt["failure_classification"] = error_description(error)
            atomic_json(state_path(root, plan["campaign_profile"], component), state)
            raise ActiveChildUnresolved(
                f"child terminal lifecycle is unresolved: {error_description(error)}") from error
        attempt["result"] = "failed"
        unit_state["status"] = "failed"
        attempt["ended_at_utc"] = attempt["ended_at_utc"] or best_effort_utc_now()
        attempt["failure_classification"] = error_description(error)
        atomic_json(state_path(root, plan["campaign_profile"], component), state)
        if isinstance(error, (CampaignError, OSError, subprocess.SubprocessError)):
            raise
        raise CampaignError(
            f"terminal child postprocessing failed: {error_description(error)}") from error


def reconcile_running(plan, component, state, unit_state, root, runtime):
    require(unit_state["attempts"] and (
        unit_state["status"] == "running" or (
            unit_state["status"] == "interrupted" and
            unit_state["attempts"][-1].get("failure_classification") == terminal_postprocessing_pending)),
        "unit lacks a reconcilable attempt")
    attempt = unit_state["attempts"][-1]
    try:
        attempt_root = Path(attempt["private_unit_root"])
        expected_id = f"attempt_{len(unit_state['attempts']):04d}"
        expected_root = private_root(root, plan["campaign_profile"],
                                     unit_state["binding"]["unit_id"], component, expected_id)
        require(expected_id in {"attempt_0001", "attempt_0002"}
                and attempt.get("attempt_id") == expected_id and attempt_root == expected_root,
                "running attempt root contradicts unit binding")
        guard_path = attempt_root / "_metadata" / "guard.jsonl"
        reject_symlink_ancestors(guard_path,
                                 component_root(root, plan["campaign_profile"], component))
        argv = attempt_argv(unit_state["binding"], attempt)
        require(attempt.get("command_identity") is None
                or attempt.get("command_identity") == identity(argv),
                "running attempt command identity contradiction")
    except Exception as error:
        raise ActiveChildUnresolved(
            f"running child binding unresolved: {error_description(error)}") from error
    scheduler_terminal_success = False
    try:
        if attempt.get("execution_contract", {}).get("backend") == "condor":
            try:
                observation = backend.observe(argv, attempt_root, attempt["execution_contract"])
            except (CampaignError, OSError, subprocess.SubprocessError) as error:
                observation = {"state": "unresolved", "reason": error_description(error)}
            if observation["state"] == "terminal_failure":
                attempt.update(result="failed", failure_classification="scheduler terminal failure",
                               ended_at_utc=best_effort_utc_now())
                unit_state["status"] = "failed"
                atomic_json(state_path(root, plan["campaign_profile"], component), state)
                return
            scheduler_terminal_success = observation["state"] == "terminal_success"
            # OnExitHold can leave a finished, failed worker in a pending job.
            # A bound failure receipt closes that attempt independently of the
            # scheduler; successful publication still needs its existing gates.
            lifecycle_path = lifecycle_receipt_path(attempt_root)
            lifecycle = (verify_lifecycle_receipt(read_json(lifecycle_path), attempt, argv)
                         if lifecycle_path.exists() or lifecycle_path.is_symlink() else None)
            failed_lifecycle = lifecycle is not None and any(
                lifecycle[key] != 0 for key in ("guard_result", "child_result", "tee_result"))
            if not failed_lifecycle:
                require(observation["state"] == "terminal_success",
                        f"scheduler execution unresolved: {observation.get('reason') or observation['state']}")
        guard_terminal(guard_path)
    except Exception as error:
        if scheduler_terminal_success:
            attempt.update(result="interrupted", failure_classification=terminal_postprocessing_pending)
            unit_state["status"] = "interrupted"
            atomic_json(state_path(root, plan["campaign_profile"], component), state)
            raise ChildDefinitelyTerminal(
                f"terminal child postprocessing pending: {error_description(error)}") from error
        raise ActiveChildUnresolved(
            f"terminal guard evidence unavailable: {error_description(error)}") from error
    attempt["result"] = "interrupted"
    attempt["failure_classification"] = terminal_postprocessing_pending
    unit_state["status"] = "interrupted"
    atomic_json(state_path(root, plan["campaign_profile"], component), state)
    try:
        lifecycle_path = lifecycle_receipt_path(attempt_root)
        if not lifecycle_path.exists() and not lifecycle_path.is_symlink():
            attempt["failure_classification"] = "guard terminal but complete lifecycle unavailable"
            attempt["ended_at_utc"] = attempt.get("ended_at_utc") or best_effort_utc_now()
            atomic_json(state_path(root, plan["campaign_profile"], component), state)
            return
        lifecycle = verify_lifecycle_receipt(read_json(lifecycle_path), attempt, argv)
        for key in ("guard_result", "child_result", "tee_result"):
            require(attempt.get(key) is None or attempt.get(key) == lifecycle[key],
                    f"running attempt {key} contradicts lifecycle receipt")
            attempt[key] = lifecycle[key]
        require(attempt.get("ended_at_utc") is None
                or attempt.get("ended_at_utc") == lifecycle["ended_at_utc"],
                "running attempt end time contradicts lifecycle receipt")
        attempt["ended_at_utc"] = lifecycle["ended_at_utc"]
        if (lifecycle["guard_result"], lifecycle["child_result"], lifecycle["tee_result"]) != (0, 0, 0):
            attempt["result"] = "failed"
            attempt["failure_classification"] = "terminal guard/child/tee failure"
            unit_state["status"] = "failed"
            atomic_json(state_path(root, plan["campaign_profile"], component), state)
            return
        atomic_json(state_path(root, plan["campaign_profile"], component), state)
        # Keep the proven terminal attempt pending until --finalize.
    except OutputCollision:
        raise
    except Exception as error:
        attempt["result"] = "interrupted"
        unit_state["status"] = "interrupted"
        attempt["ended_at_utc"] = attempt.get("ended_at_utc") or best_effort_utc_now()
        attempt["failure_classification"] = error_description(error)
        atomic_json(state_path(root, plan["campaign_profile"], component), state)
        raise ChildDefinitelyTerminal(
            f"terminal child postprocessing failed: {error_description(error)}") from error


def require_terminal_process_gone(terminal):
    """Reject a still-live managed identity without launching or reaping anything."""
    process_path = Path("/proc") / str(terminal["pid"]) / "stat"
    try:
        raw = process_path.read_text(encoding="ascii")
    except FileNotFoundError:
        raw = None
    if raw is not None:
        fields = raw[raw.rfind(")") + 2:].split()
        require(len(fields) > 19, "managed process identity is unreadable")
        require(not (int(fields[19]) == terminal["start_ticks"]
                     and int(fields[2]) == terminal["pgid"]),
                "managed child process identity is still live")
    try:
        os.killpg(terminal["pgid"], 0)
    except ProcessLookupError:
        return
    except PermissionError as error:
        raise CampaignError("managed process group liveness is unresolved") from error
    raise CampaignError("managed process group is still live")


def observed_recovery_provenance(plan, unit, attempt):
    require(attempt.get("execution_contract", {}).get("backend", "local") == "local",
            "missing-lifecycle recovery requires local process identity; remote recovery is not inferred")
    attempt_root = Path(attempt["private_unit_root"])
    guard_path = attempt_root / "_metadata" / "guard.jsonl"
    log_path = attempt_root / "_metadata" / "renderer.log"
    require(attempt["attempt_id"] == "attempt_0001"
            and attempt["plan_revision"] <= plan["plan_revision"]
            and attempt["command_identity"] == identity(attempt_argv(unit, attempt))
            and attempt["private_unit_root"] == unit["private_unit_root"],
            "recovery attempt binding contradiction")
    require(not lifecycle_receipt_path(attempt_root).exists()
            and not lifecycle_receipt_path(attempt_root).is_symlink(),
            "recovery requires absent lifecycle receipt")
    require(all(attempt.get(key) is None for key in
                ("guard_result", "child_result", "tee_result")),
            "recovery cannot replace observed lifecycle results")
    terminal = guard_terminal(guard_path)
    guard_rows = [json.loads(line) for line in guard_path.read_text(encoding="utf-8").splitlines()]
    start_identity = guard_rows[0].get("identity") if guard_rows else None
    require(len(guard_rows) >= 2 and guard_rows[0].get("event") == "start"
            and isinstance(start_identity, dict)
            and start_identity.get("pid") == terminal["pid"]
            and start_identity.get("pgid") == terminal["pgid"]
            and start_identity.get("start_ticks") == terminal["start_ticks"],
            "recovery guard start/result identity contradiction")
    require(terminal["termination_kind"] == "normal"
            and terminal["returncode"] == 0 and terminal["signal"] is None,
            "recovery guard is not normal/0")
    require_terminal_process_gone(terminal)
    records = inventory_pngs(attempt_root, unit["variables"], period(unit))
    require(records, "recovery has no observed PNGs")
    require(log_path.is_file() and not log_path.is_symlink(),
            "recovery renderer log is missing or unsafe")
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    summaries = []
    for index, line in enumerate(lines):
        match = re.fullmatch(
            r"\[SR\] Produced (\d+) plots with stat⊕syst uncertainties and "
            r"(\d+) plots with stat-only bands after completing (\d+) rendering tasks in (.+)",
            line)
        if match:
            summaries.append((index, match))
    require(len(summaries) == 1, "recovery renderer completion summary is unavailable")
    summary_index, summary = summaries[0]
    require(int(summary[1]) + int(summary[2]) == len(records)
            and int(summary[3]) > 0
            and summary[4] == str(attempt_root / period(unit) / "merged"),
            "recovery renderer summary contradicts observed PNGs")
    require(not any(re.search(r"Traceback|\bFatal\b|\bError:", line, re.I)
                    for line in lines[summary_index + 1:]),
            "renderer failure follows completion summary")
    return {
        "schema_version": 1, "recovery_kind": "terminal_guard_missing_lifecycle",
        "campaign_profile": plan["campaign_profile"], "campaign_tag": plan["campaign_tag"],
        "unit_id": unit["unit_id"], "attempt_id": attempt["attempt_id"],
        "plan_revision": attempt["plan_revision"],
        "private_unit_root": str(attempt_root),
        "command_identity": attempt["command_identity"],
        "guard_evidence_sha256": sha256_file(guard_path),
        "terminal_guard_evidence": terminal,
        "renderer_log_sha256": sha256_file(log_path),
        "renderer_completion_summary": lines[summary_index],
        "prior_failure_classification": attempt.get(
            "recovery_prior_failure_classification", attempt.get("failure_classification")),
        "interrupted_at_utc": attempt.get("ended_at_utc"),
        "observed_png_count": len(records),
        "observed_png_sizes": {row["relative_path"]: row["size_bytes"] for row in records},
        "observed_png_sha256": {row["relative_path"]: row["sha256"] for row in records},
        "unavailable_lifecycle_fields": ["guard_result", "child_result", "tee_result",
                                         "lifecycle_receipt_v1.json"],
    }


def migrate_successful_unit(plan, component, state, unit_state, root, runtime):
    attempt = unit_state["attempts"][0]
    attempt_root = Path(attempt["private_unit_root"])
    receipt = read_json(completion_receipt_path(attempt_root))
    verify_receipt(receipt, plan, unit_state["binding"], attempt, runtime)
    legacy_root = component_root(root, plan["campaign_profile"], component)
    legacy_paths = []
    old_root = unit_state.get("human_output_root")
    require(old_root in {None, str(root), str(legacy_root)},
            "unit human-output root is unrecognized")
    if legacy_root != root and old_root != str(root):
        candidates = receipt["observed_png_relative_paths"]
    else:
        candidates = unit_state.get("legacy_cleanup_pending", [])
    require(isinstance(candidates, list)
            and len(candidates) == len(set(candidates))
            and set(candidates) <= set(receipt["observed_png_relative_paths"]),
            "legacy cleanup ownership is malformed")
    if legacy_root != root:
        for relative in candidates:
            legacy = legacy_root / safe_relative(relative)
            if legacy.exists() or legacy.is_symlink():
                reject_symlink_ancestors(legacy.parent, legacy_root)
                require(legacy.is_file() and not legacy.is_symlink()
                        and legacy.stat().st_size == receipt["observed_png_sizes"][relative]
                        and sha256_file(legacy) == receipt["observed_png_sha256"][relative],
                        f"legacy human PNG contradicts receipt: {legacy}")
                legacy_paths.append((legacy, root / relative, relative))
    if old_root != str(root):
        unit_state["legacy_cleanup_pending"] = [row[2] for row in legacy_paths]
        atomic_json(state_path(root, plan["campaign_profile"], component), state)
    materialize(plan, component, state, unit_state, attempt, receipt, root, runtime)
    for legacy, canonical, relative in legacy_paths:
        require(unit_state["materializations"].get(relative) is not None
                and canonical.is_file() and not canonical.is_symlink()
                and legacy.is_file() and not legacy.is_symlink()
                and legacy.stat().st_size == canonical.stat().st_size
                and sha256_file(legacy) == sha256_file(canonical),
                f"legacy deletion identity proof failed: {legacy}")
        legacy.unlink()
        unit_state["legacy_cleanup_pending"].remove(relative)
        atomic_json(state_path(root, plan["campaign_profile"], component), state)
    for relative in list(unit_state.get("legacy_cleanup_pending", [])):
        legacy = legacy_root / relative
        canonical = root / relative
        require(not legacy.exists() and not legacy.is_symlink()
                and canonical.is_file() and not canonical.is_symlink()
                and canonical.stat().st_size == receipt["observed_png_sizes"][relative]
                and sha256_file(canonical) == receipt["observed_png_sha256"][relative],
                f"pending legacy cleanup is unresolved: {legacy}")
        unit_state["legacy_cleanup_pending"].remove(relative)
        atomic_json(state_path(root, plan["campaign_profile"], component), state)


def recover_terminal_unit(plan, component, state, unit_state, root, runtime):
    require(unit_state["status"] == "interrupted"
            and len(unit_state["attempts"]) == 1
            and unit_state["attempts"][0].get("failure_classification")
            == "guard terminal but complete lifecycle unavailable",
            "selected unit is not eligible for no-launch recovery")
    unit = unit_state["binding"]
    attempt = unit_state["attempts"][0]
    provenance = observed_recovery_provenance(plan, unit, attempt)
    provenance_path = recovery_provenance_path(Path(attempt["private_unit_root"]))
    if provenance_path.exists() or provenance_path.is_symlink():
        require(read_json(provenance_path) == provenance,
                "existing recovery provenance contradicts observed evidence")
    else:
        atomic_json(provenance_path, provenance)
    attempt["recovery_prior_failure_classification"] = attempt["failure_classification"]
    attempt["recovery_provenance_identity"] = identity(provenance)
    attempt["failure_classification"] = terminal_postprocessing_pending
    atomic_json(state_path(root, plan["campaign_profile"], component), state)


def repair_materializations(root, profile, tag, selected, *, emit=True):
    """Prove one terminal recovery and defer its publication to finalize."""
    require(selected is not None, "no-launch recovery requires a selected unit")
    check_root(root, fresh=False)
    with campaign_lock(root):
        runtime = read_json(coordinator_path(root)).get("runtime_contract_identity")
        _, plan, states = load_campaign(root, profile, tag, runtime)
        require(not any(unit["status"] == "running" for state in states.values()
                        for unit in state["units"].values()),
                "campaign has a running unit; maintenance is unsafe")
        matches = [(component, states[component], states[component]["units"][selected])
                   for component in PROFILES[profile]
                   if selected in states[component]["units"]]
        require(len(matches) == 1, "selected recovery unit is not in the campaign")
        selected_component, selected_state, selected_unit = matches[0]
        require(selected_unit["status"] in {"interrupted", "success"},
                "selected recovery unit is neither interrupted nor recovered")
        if selected_unit["status"] == "success":
            require(selected_unit["attempts"][0].get("recovery_provenance_identity"),
                    "selected success has no recovery provenance")
        else:
            require(selected_unit["attempts"][0].get("failure_classification")
                    == "guard terminal but complete lifecycle unavailable",
                    "selected interruption is not eligible for recovery")
            observed_recovery_provenance(plan, selected_unit["binding"],
                                         selected_unit["attempts"][0])
        if selected_unit["status"] == "interrupted":
            recover_terminal_unit(plan, selected_component, selected_state,
                                  selected_unit, root, runtime)
        if emit:
            print(json.dumps({"recovered_unit": selected, "status": selected_unit["status"],
                              "success_count": sum(u["status"] == "success"
                                                   for s in states.values() for u in s["units"].values())},
                             sort_keys=True))


def recover_rejected_condor_unit(root, profile, tag, selected, *, emit=True):
    """Explicitly rearm one occupied attempt after a local submitfile rejection."""
    require(selected is not None, "Condor rejection recovery requires --selected-unit")
    check_root(root, fresh=False)
    with campaign_lock(root):
        runtime = read_json(coordinator_path(root)).get("runtime_contract_identity")
        _, plan, states = load_campaign(root, profile, tag, runtime)
        matches = [(component, states[component], states[component]["units"][selected])
                   for component in PROFILES[profile]
                   if selected in states[component]["units"]]
        require(len(matches) == 1, "selected recovery unit is not in the campaign")
        component, state, unit_state = matches[0]
        require(unit_state["status"] == "running" and len(unit_state["attempts"]) == 1
                and not unit_state["materializations"],
                "selected unit is not one waiting Condor attempt")
        attempt = unit_state["attempts"][0]
        attempt_root = private_root(root, profile, selected, component, "attempt_0001")
        next_attempt = attempt_root.parent / "attempt_0002"
        require(not next_attempt.exists() and not next_attempt.is_symlink(),
                "another attempt root blocks same-attempt recovery")
        contract = attempt.get("execution_contract")
        require(attempt.get("attempt_id") == "attempt_0001"
                and attempt.get("private_unit_root") == str(attempt_root)
                and attempt.get("result") == "running"
                and attempt.get("failure_classification")
                == "submission response cannot prove a unique execution"
                and attempt.get("ended_at_utc") is None
                and attempt.get("completion_receipt_identity") is None
                and attempt.get("materialization_status") == "pending"
                and all(attempt.get(key) is None for key in
                        ("guard_result", "child_result", "tee_result"))
                and isinstance(contract, dict) and contract.get("backend") == "condor",
                "selected attempt state contradicts rejected Condor submit")
        argv = attempt_argv(unit_state["binding"], attempt)
        require(attempt.get("command_identity") == identity(argv),
                "selected attempt command identity contradiction")
        descriptor = backend.execution_descriptor(argv, attempt_root, contract)
        metadata = attempt_root / "_metadata"
        require(read_json(metadata / "execution.json") == descriptor,
                "selected attempt descriptor identity contradiction")
        for name in ("worker_started.json", "guard.jsonl", "lifecycle_receipt_v1.json",
                     "renderer.log", "completion_receipt_v1.json", "worker.out", "worker.err",
                     "scheduler.log"):
            path = metadata / name
            require(not path.exists() and not path.is_symlink(),
                    f"worker or renderer evidence blocks recovery: {name}")
        require(not any(attempt_root.rglob("*.png")),
                "private PNG evidence blocks recovery")
        from analysis.topeft_run2 import plot_campaign_condor as condor
        result = condor.recover_rejected_submit(metadata / "execution.json", descriptor)
        attempt["failure_classification"] = None
        atomic_json(state_path(root, profile, component), state)
        if emit:
            print(json.dumps({"recovered_unit": selected, "attempt_id": "attempt_0001",
                              "submission_state": result["state"], "job": result["job"]}, sort_keys=True))


def verify_failed_attempt(plan, component, state, unit_state, root):
    """Bind one terminal Condor failure before appending its sole replacement."""
    require(unit_state["status"] == "failed" and len(unit_state["attempts"]) == 1
            and not unit_state["materializations"],
            "recovery requires exactly one failed, unpublished attempt")
    unit = unit_state["binding"]
    attempt = unit_state["attempts"][0]
    attempt_root = private_root(root, plan["campaign_profile"], unit["unit_id"],
                                component, "attempt_0001")
    reject_symlink_ancestors(attempt_root, component_root(root, plan["campaign_profile"], component))
    require(attempt.get("attempt_id") == "attempt_0001"
            and attempt.get("private_unit_root") == str(attempt_root)
            and attempt.get("result") == "failed"
            and attempt.get("plan_revision") == state["plan_revision"] == plan["plan_revision"]
            and attempt.get("completion_receipt_identity") is None,
            "failed attempt identity contradicts campaign")
    require(attempt_root.is_dir()
            and {path.name for path in attempt_root.parent.iterdir()} == {"attempt_0001"},
            "another attempt or recovery path already exists")
    contract = attempt.get("execution_contract")
    require(isinstance(contract, dict) and contract.get("backend") == "condor",
            "failed-attempt recovery requires an existing Condor execution")
    argv = attempt_argv(unit, attempt)
    require(attempt.get("command_identity") == identity(argv),
            "failed attempt command identity contradiction")
    receipt = verify_lifecycle_receipt(
        read_json(lifecycle_receipt_path(attempt_root)), attempt, argv)
    require(all(attempt.get(key) == receipt[key] for key in
                ("guard_result", "child_result", "tee_result", "ended_at_utc"))
            and any(receipt[key] != 0 for key in ("guard_result", "child_result", "tee_result")),
            "failed attempt lacks coherent nonzero lifecycle")
    descriptor = backend.execution_descriptor(argv, attempt_root, contract)
    metadata = attempt_root / "_metadata"
    require(read_json(metadata / "execution.json") == descriptor,
            "failed execution descriptor identity contradiction")
    scheduler = read_json(metadata / "scheduler_identity.json")
    require(scheduler.get("token") == descriptor["token"],
            "failed scheduler token contradiction")
    observation = backend.observe(argv, attempt_root, contract, read_only=True)
    ad = observation.get("ad")
    absent = observation == {"state": "unresolved", "job": scheduler.get("job"),
                             "source": "unresolved", "ad": None}
    retired = (observation.get("state") == "terminal_failure"
               and observation.get("job") == scheduler.get("job")
               and isinstance(ad, dict) and ad.get("JobStatus") in {3, 4}
               and ad.get("NumJobStarts") == 1
               and (ad.get("ExitBySignal") is True or
                    type(ad.get("ExitCode")) is int and ad["ExitCode"] != 0))
    require(absent or retired, "old scheduler execution is not retired or proven absent")
    require(sha256_file(Path(unit["input_path"])) == unit["accepted_input_identity"]["sha256"],
            "producer input bytes changed since qualification")


def recover_failed_unit(root, profile, tag, selected, *, emit=True):
    """One bounded restart; preserve attempt_0001, never create attempt_0003."""
    require(selected is not None, "failed-attempt recovery requires --selected-unit")
    check_root(root, fresh=False)
    with campaign_lock(root):
        runtime = read_json(coordinator_path(root)).get("runtime_contract_identity")
        coordinator, plan, states = load_campaign(root, profile, tag, runtime)
        matches = [(component, states[component], states[component]["units"][selected])
                   for component in PROFILES[profile] if selected in states[component]["units"]]
        require(len(matches) == 1, "selected recovery unit is not in the campaign")
        require(not any(unit["status"] in {"running", "interrupted"}
                        for state in states.values() for unit in state["units"].values()),
                "active or unresolved campaign execution blocks recovery")
        component, state, unit_state = matches[0]
        policy, should_bind = resolve_execution_policy(
            coordinator, dict.fromkeys(DEFAULT_EXECUTION_POLICY), {"can_start_new_attempt": True})
        require(not should_bind and policy["backend"] == "condor",
                "recovery requires the existing bound Condor policy")
        options = {"backend": policy["backend"], "observer": policy["resource_observer"],
                   "hard_limit_bytes": policy["hard_limit_bytes"]}
        warning = None
        try:
            execute_unit(plan, component, state, unit_state, root, runtime, options,
                         failed_attempt_recovery=True)
        except ActiveChildUnresolved as error:
            if len(unit_state["attempts"]) != 2:
                raise
            warning = error_description(error)
        if emit:
            print(json.dumps({"recovered_unit": selected,
                              "attempt_id": unit_state["attempts"][-1]["attempt_id"],
                              "status": unit_state["status"], "warning": warning,
                              "scheduler_readback_required": True}, sort_keys=True))


def resume_recover(root, profile, tag, selected, execution_options):
    """Select existing guarded recovery paths before the ordinary controller pass."""
    coordinator = read_json(coordinator_path(root))
    if "pending_plan" in coordinator:
        # The ordinary locked pass owns completion of a persisted revision intent.
        return [], []
    runtime = coordinator.get("runtime_contract_identity")
    coordinator, plan, states = load_campaign(root, profile, tag, runtime)
    scope = derive_action_scope(plan, states, selected)
    resolve_execution_policy(coordinator, execution_options, scope)
    errors = []
    recovered_ids = []
    for unit in plan["units"]:
        if unit["unit_id"] not in scope["selected_unit_ids"]:
            continue
        unit_state = states[unit["component"]]["units"][unit["unit_id"]]
        attempt = unit_state["attempts"][-1] if unit_state["attempts"] else None
        recovery = None
        if (unit_state["status"] == "running" and attempt
                and attempt.get("failure_classification")
                == "submission response cannot prove a unique execution"):
            recovery = recover_rejected_condor_unit
        elif (unit_state["status"] == "interrupted" and attempt
              and attempt.get("failure_classification")
              == "guard terminal but complete lifecycle unavailable"):
            recovery = repair_materializations
        elif unit_state["status"] == "failed" and len(unit_state["attempts"]) == 1:
            recovery = recover_failed_unit
        if recovery is None:
            continue
        try:
            recovery(root, profile, tag, unit["unit_id"], emit=False)
        except (CampaignError, OSError, subprocess.SubprocessError) as error:
            errors.append(f"{unit['component']}/{unit['unit_id']}: {error_description(error)}")
            if recovery is not repair_materializations:
                recovered_ids.append(unit["unit_id"])
                break
        if recovery is not repair_materializations:
            recovered_ids.append(unit["unit_id"])
            break
    return errors, recovered_ids


def finalize_terminal_unit(plan, component, state, unit_state, root, runtime):
    """Use the existing receipt and materialization path for proven terminal success."""
    attempt = unit_state["attempts"][-1]
    require(unit_state["status"] == "interrupted"
            and attempt.get("result") == "interrupted"
            and attempt.get("failure_classification") == terminal_postprocessing_pending
            and attempt.get("materialization_status") == "pending",
            "unit is not pending terminal-success finalization")
    unit = unit_state["binding"]
    attempt_root = Path(attempt["private_unit_root"])
    argv = attempt_argv(unit, attempt)
    require(attempt.get("command_identity") in {None, identity(argv)},
            "terminal attempt command identity contradiction")
    provenance = None
    if attempt.get("recovery_provenance_identity") is not None:
        provenance = observed_recovery_provenance(plan, unit, attempt)
        require(identity(provenance) == attempt["recovery_provenance_identity"]
                and read_json(recovery_provenance_path(attempt_root)) == provenance,
                "terminal recovery provenance contradiction")
    else:
        lifecycle = verify_lifecycle_receipt(
            read_json(lifecycle_receipt_path(attempt_root)), attempt, argv)
        require(all(lifecycle[key] == 0 and attempt.get(key) == 0 for key in
                    ("guard_result", "child_result", "tee_result"))
                and attempt.get("ended_at_utc") == lifecycle["ended_at_utc"],
                "terminal-success lifecycle evidence is incomplete")
    receipt_path = completion_receipt_path(attempt_root)
    receipt = (read_json(receipt_path) if receipt_path.exists() or receipt_path.is_symlink()
               else create_receipt(plan, unit, attempt, runtime, provenance))
    materialize(plan, component, state, unit_state, attempt, receipt, root, runtime)


def finalize_campaign(root, profile, tag, selected=None, *, output_format="json", emit=True):
    """Finalize only proven terminal attempts; never enter worker dispatch."""
    check_root(root, fresh=False)
    with campaign_lock(root):
        runtime = read_json(coordinator_path(root)).get("runtime_contract_identity")
        coordinator, plan, states = load_campaign(root, profile, tag, runtime)
        scope = derive_action_scope(plan, states, selected)
        for unit in plan["units"]:
            if unit["unit_id"] not in scope["selected_unit_ids"]:
                continue
            state = states[unit["component"]]
            unit_state = state["units"][unit["unit_id"]]
            if unit_state["status"] == "success":
                if (unit_state.get("human_output_root") != str(root)
                        or unit_state.get("legacy_cleanup_pending")):
                    require(len(unit_state["attempts"]) == 1,
                            "legacy successful unit has unsupported attempt history")
                    migrate_successful_unit(plan, unit["component"], state,
                                            unit_state, root, runtime)
                continue
            if unit_state["status"] == "running":
                try:
                    reconcile_running(plan, unit["component"], state, unit_state, root, runtime)
                except (ActiveChildUnresolved, ChildDefinitelyTerminal):
                    if selected is not None:
                        raise
                    continue
            attempt = unit_state["attempts"][-1] if unit_state["attempts"] else None
            eligible = (unit_state["status"] == "interrupted" and attempt is not None
                        and attempt.get("failure_classification")
                        == terminal_postprocessing_pending)
            if not eligible:
                require(selected is None, f"selected unit is not eligible for finalize: {unit['unit_id']}")
                continue
            finalize_terminal_unit(plan, unit["component"], state, unit_state, root, runtime)
        summary = campaign_summary(root, plan, states,
                                   execution_policy=coordinator.get("execution_policy_v1"))
    if emit:
        show_summary(summary, output_format)
    return summary


def check_root(root, *, fresh):
    histos = REPO_ROOT / "histos"
    require(root.is_absolute() and ".." not in root.parts and root != histos
            and histos in root.parents and histos.is_dir(),
            "campaign root must be an absolute child of topeft/histos")
    reject_symlink_ancestors(root.parent, histos)
    if fresh:
        require(not root.exists() and not root.is_symlink(), "fresh campaign root is occupied")
    else:
        require(root.is_dir() and not root.is_symlink(), "campaign root is missing or unsafe")


def dry_run(plan, root, selected):
    if selected:
        require(selected in {unit["unit_id"] for unit in plan["units"]}, "unknown selected unit")
    print(json.dumps({"campaign_profile": plan["campaign_profile"],
                      "campaign_tag": plan["campaign_tag"],
                      "plan_revision": plan["plan_revision"], "units": [
                          {"unit_id": unit["unit_id"], "component": unit["component"],
                           "variables": unit["variables"],
                           "accepted_input_identity": unit["accepted_input_identity"],
                           "accepted_sidecar_identity": unit["accepted_sidecar_identity"],
                           "private_unit_root": unit["private_unit_root"],
                           "child_argv": child_argv(unit, Path(unit["private_unit_root"]))}
                          for unit in plan["units"]]}, indent=2))
    print(f"Selected execution: {selected or 'all planned units'}")


def load_campaign(root, profile, tag, runtime):
    coordinator = read_json(coordinator_path(root))
    require(coordinator.get("schema_version") == 1
            and coordinator.get("campaign_profile") == profile
            and coordinator.get("campaign_tag") == tag,
            "campaign coordinator identity mismatch")
    require(isinstance(coordinator.get("runtime_contract_identity"), dict),
            "campaign coordinator provenance field is malformed")
    require("pending_plan" not in coordinator, "pending plan requires reconciliation")
    if "execution_policy_v1" in coordinator:
        resolve_execution_policy(coordinator, dict.fromkeys(DEFAULT_EXECUTION_POLICY),
                                 {"can_start_new_attempt": False})
    plan = coordinator.get("plan")
    require(isinstance(plan, dict) and plan.get("campaign_profile") == profile
            and plan.get("campaign_tag") == tag, "campaign plan binding mismatch")
    states = {}
    for component in PROFILES[profile]:
        reject_symlink_ancestors(component_root(root, profile, component), root)
        states[component] = read_json(state_path(root, profile, component))
        validate_state(states[component], plan, component, runtime)
        require(states[component]["plan_revision"] == plan["plan_revision"],
                "component revision differs from active coordinator")
    return coordinator, plan, states


def derive_action_scope(plan, states, selected=None):
    """The only units this invocation can reconcile or start."""
    selected_ids = [unit["unit_id"] for unit in plan["units"]
                    if selected is None or unit["unit_id"] == selected]
    require(selected is None or selected_ids, "unknown selected unit")
    reconcilable = []
    launchable = []
    for unit in plan["units"]:
        unit_id = unit["unit_id"]
        if unit_id not in selected_ids:
            continue
        unit_state = states[unit["component"]]["units"][unit_id]
        if unit_state["status"] == "planned":
            launchable.append(unit_id)
        elif unit_state["status"] == "running" or (
                unit_state["status"] == "interrupted" and unit_state["attempts"]
                and unit_state["attempts"][-1].get("failure_classification")
                == terminal_postprocessing_pending
                and unit_state["attempts"][-1].get("recovery_provenance_identity") is None):
            reconcilable.append(unit_id)
    return {"selected_unit_ids": selected_ids,
            "reconcilable_unit_ids": reconcilable,
            "launchable_planned_unit_ids": launchable,
            "can_start_new_attempt": bool(launchable)}


def dispatch_decision(plan, states, action_scope, execution_policy, active_reasons=()):
    """One backend-aware decision for both launches and status presentation."""
    reasons_by_unit = {item.split(":", 1)[0]: item for item in active_reasons}
    active_local = False
    for unit in plan["units"]:
        unit_state = states[unit["component"]]["units"][unit["unit_id"]]
        if unit_state["status"] != "running":
            continue
        label = f"{unit['component']}/{unit['unit_id']}"
        reasons_by_unit.setdefault(label, f"{label}: remains running")
        attempt = unit_state["attempts"][-1]
        contract = attempt.get("execution_contract")
        require(contract is None or (isinstance(contract, dict) and
                                     contract.get("version") == 1 and
                                     contract.get("backend") in {"local", "condor"} and
                                     isinstance(contract.get("argv"), list) and
                                     bool(contract["argv"])),
                f"active attempt backend is invalid: {label}")
        if contract is None or contract["backend"] == "local":
            active_local = True
    reasons = list(reasons_by_unit.values())
    launchable = [unit_id for unit_id in action_scope["launchable_planned_unit_ids"]
                  if any(unit["unit_id"] == unit_id and
                         states[unit["component"]]["units"][unit_id]["status"] == "planned"
                         for unit in plan["units"])]
    blocked = bool(launchable and active_local and
                   (execution_policy or {}).get("backend", "local") == "local")
    return {"active_child_present": bool(reasons),
            "active_child_reasons": reasons,
            "dispatch_blocked": blocked,
            "dispatch_blocked_reasons": reasons if blocked else [],
            "launchable_planned_unit_ids": [] if blocked else launchable}


def resolve_execution_policy(coordinator, supplied, scope, *, fresh=False):
    """Bind future attempts once; historical attempt contracts remain authoritative."""
    require(set(supplied) == set(DEFAULT_EXECUTION_POLICY),
            "future execution policy input is malformed")
    require(supplied["backend"] in {None, "local", "condor"}
            and supplied["resource_observer"] in {None, "cgroup_v2", "node_pressure"}
            and (supplied["hard_limit_bytes"] is None or
                 (isinstance(supplied["hard_limit_bytes"], int)
                  and not isinstance(supplied["hard_limit_bytes"], bool)
                  and supplied["hard_limit_bytes"] > 0)),
            "future execution policy input is invalid")
    bound = coordinator.get("execution_policy_v1")
    if bound is not None:
        require(isinstance(bound, dict) and set(bound) == {
            "backend", "resource_observer", "hard_limit_bytes", "source"}
            and bound["backend"] in {"local", "condor"}
            and bound["resource_observer"] in {"cgroup_v2", "node_pressure"}
            and isinstance(bound["hard_limit_bytes"], int)
            and not isinstance(bound["hard_limit_bytes"], bool)
            and bound["hard_limit_bytes"] > 0
            and bound["source"] in {"fresh_defaults", "explicitly_bound"},
            "future execution policy is malformed")
        for key, value in supplied.items():
            require(value is None or value == bound[key],
                    f"supplied {key} contradicts bound future execution policy")
        return bound, False
    if fresh:
        values = {key: supplied[key] if supplied[key] is not None else default
                  for key, default in DEFAULT_EXECUTION_POLICY.items()}
        return {**values, "source": ("explicitly_bound" if any(
            value is not None for value in supplied.values()) else "fresh_defaults")}, True
    if scope["can_start_new_attempt"]:
        if supplied["backend"] == "condor":
            require(all(value is not None for value in supplied.values()),
                    "legacy Condor launch needs --backend, --resource-observer and --hard-limit-bytes")
        elif supplied["backend"] is None:
            require(False, "legacy planned launch needs --backend to bind future execution policy")
        values = {key: supplied[key] if supplied[key] is not None else default
                  for key, default in DEFAULT_EXECUTION_POLICY.items()}
        return {**values, "source": "explicitly_bound"}, True
    # Reconciliation alone needs no new policy and never mutates the coordinator.
    values = {key: supplied[key] if supplied[key] is not None else default
              for key, default in DEFAULT_EXECUTION_POLICY.items()}
    return values, False


def sha256_canonical_file(path, root):
    """Hash the recorded canonical file without following its final symlink."""
    reject_symlink_ancestors(path.parent, root)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        require(stat.S_ISREG(os.fstat(descriptor).st_mode),
                f"canonical publication is not regular: {path}")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def campaign_summary(root, plan, states, *, unit_errors=(), active_reasons=(),
                     execution_policy=None, action_scope=None, dispatch=None,
                     verify_published=False, observe_scheduler=False, selected=None):
    """Read-only presentation derived from the bound plan and component states."""
    if selected is not None:
        action_scope = derive_action_scope(plan, states, selected)
        require(len(action_scope["selected_unit_ids"]) == 1, "selected unit is not unique")
        # Scope before per-unit reads and deep checks, without altering bound state.
        plan = {**plan, "units": [unit for unit in plan["units"]
                                 if unit["unit_id"] == selected]}
        states = {component: {**state, "units": {
            unit_id: unit for unit_id, unit in state["units"].items()
            if unit_id == selected}} for component, state in states.items()}

    from analysis.topeft_run2 import plot_campaign_condor as condor

    units = []
    active_warnings = dict(item.split(": ", 1) for item in active_reasons if ": " in item)
    for unit in plan["units"]:
        unit_state = states[unit["component"]]["units"][unit["unit_id"]]
        state = unit_state["status"]
        attempt = unit_state["attempts"][-1] if unit_state["attempts"] else None
        execution = {"planned": "pending", "running": "running",
                     "success": "completed", "failed": "failed",
                     "interrupted": "interrupted"}[state]
        publish = ("publish pending" if state == "running" or
                   (state == "interrupted" and attempt and
                    attempt.get("failure_classification") == terminal_postprocessing_pending)
                   else "not published")
        integrity = "not_applicable"
        warning = attempt.get("failure_classification") if attempt else None
        if state == "running":
            warning = active_warnings.get(f"{unit['component']}/{unit['unit_id']}", warning)
        if warning and "Condor execution submitted; resume observes this execution only" in warning:
            warning = None
        job = None
        scheduler = None
        paths = []
        if attempt:
            attempt_root = Path(attempt["private_unit_root"])
            if (attempt.get("execution_contract") or {}).get("backend") == "condor":
                try:
                    descriptor = backend.execution_descriptor(
                        attempt_argv(unit, attempt), attempt_root, attempt["execution_contract"])
                    metadata = attempt_root / "_metadata"
                    intent = condor._read_intent(metadata / "submission_intent.json", descriptor["token"])
                    job = condor._read_identity(metadata / "scheduler_identity.json", intent)["job"]
                except (CampaignError, OSError, KeyError, ValueError) as error:
                    warning = warning or f"scheduler identity unavailable: {error_description(error)}"
                if observe_scheduler:
                    try:
                        observation = backend.observe(
                            attempt_argv(unit, attempt), attempt_root,
                            attempt["execution_contract"], read_only=True)
                    except (CampaignError, OSError, KeyError, ValueError) as error:
                        observation = {"state": "unresolved", "reason": error_description(error)}
                    scheduler = scheduler_status(observation)
                    job = observation.get("job", job)
            if state == "running":
                lifecycle_path = lifecycle_receipt_path(attempt_root)
                if lifecycle_path.is_file() and not lifecycle_path.is_symlink():
                    try:
                        lifecycle = verify_lifecycle_receipt(
                            read_json(lifecycle_path), attempt, attempt_argv(unit, attempt))
                        execution = "completed"
                        if any(lifecycle[key] != 0 for key in
                               ("guard_result", "child_result", "tee_result")):
                            warning = "completed with nonzero result; reconciliation needed"
                    except (CampaignError, OSError, ValueError) as error:
                        warning = f"terminal evidence unavailable: {error_description(error)}"
            if state == "success":
                integrity = "unavailable"
                try:
                    receipt = read_json(completion_receipt_path(attempt_root))
                    require(isinstance(receipt, dict), "completion receipt is malformed")
                    receipt_id = attempt.get("completion_receipt_identity")
                    require(attempt.get("result") == "success"
                            and attempt.get("materialization_status") == "complete"
                            and isinstance(receipt_id, str) and SHA256.fullmatch(receipt_id),
                            "successful publication provenance is incomplete")
                    require(identity(receipt) == receipt_id,
                            "completion receipt identity changed")
                    relatives = receipt.get("observed_png_relative_paths")
                    require(receipt.get("unit_id") == unit["unit_id"]
                            and receipt.get("attempt_id") == attempt["attempt_id"]
                            and isinstance(relatives, list) and relatives
                            and len(relatives) == len(set(relatives))
                            and set(relatives) == set(unit_state["materializations"])
                            and receipt.get("observed_png_count") == len(relatives)
                            and isinstance(receipt.get("observed_png_sha256"), dict)
                            and isinstance(receipt.get("observed_png_sizes"), dict)
                            and set(receipt["observed_png_sha256"]) == set(relatives)
                            and set(receipt["observed_png_sizes"]) == set(relatives),
                            "publication records contradict receipt")
                    for relative in relatives:
                        binding = unit_state["materializations"][relative]
                        path = canonical_observed_path(root, relative)
                        require(binding == {
                            "unit_id": unit["unit_id"],
                            "receipt_identity": receipt_id,
                            "sha256": receipt["observed_png_sha256"][relative],
                            "size_bytes": receipt["observed_png_sizes"][relative],
                        } and isinstance(binding["sha256"], str)
                                and SHA256.fullmatch(binding["sha256"])
                                and isinstance(binding["size_bytes"], int)
                                and not isinstance(binding["size_bytes"], bool)
                                and binding["size_bytes"] > 0,
                                f"publication provenance invalid: {relative}")
                        paths.append(str(path))
                    publish = "published"
                    integrity = "metadata_consistent"
                    for relative in relatives:
                        path = canonical_observed_path(root, relative)
                        try:
                            reject_symlink_ancestors(path.parent, root)
                            require(not path.is_symlink(), f"unsafe canonical symlink: {relative}")
                            require(path.is_file(), f"canonical file unavailable: {relative}")
                            if path.stat().st_size != unit_state["materializations"][relative]["size_bytes"]:
                                integrity = "content_mismatch"
                                warning = f"canonical size mismatch: {relative}"
                                break
                        except (CampaignError, OSError, ValueError) as error:
                            integrity = "unavailable"
                            warning = f"canonical file unavailable: {error_description(error)}"
                            break
                    if verify_published and integrity == "metadata_consistent":
                        integrity = "content_verified"
                        for relative in relatives:
                            path = canonical_observed_path(root, relative)
                            if sha256_canonical_file(path, root) != (
                                    unit_state["materializations"][relative]["sha256"]):
                                integrity = "content_mismatch"
                                warning = f"canonical content mismatch: {relative}"
                                break
                except (CampaignError, OSError, KeyError, TypeError, ValueError) as error:
                    if publish != "published":
                        paths = []
                        warning = f"publication provenance unavailable: {error_description(error)}"
                    else:
                        integrity = "unavailable"
                        warning = f"content verification unavailable: {error_description(error)}"
        units.append({"unit_id": unit["unit_id"], "component": unit["component"],
                      "execution": execution, "publish": publish,
                      "publication": publish, "integrity": integrity, "job": job,
                      "canonical_paths": paths, "warning": warning,
                      "scheduler": scheduler})
    counts = Counter(unit["status"] for state in states.values()
                     for unit in state["units"].values())
    dispatch = dispatch or dispatch_decision(
        plan, states, action_scope or derive_action_scope(plan, states),
        execution_policy, active_reasons)
    dispatch_blocked = dispatch["dispatch_blocked"]
    waiting_reasons = dispatch["dispatch_blocked_reasons"]
    if dispatch_blocked:
        campaign_status = "waiting"
    elif counts["failed"] or counts["interrupted"] or unit_errors:
        campaign_status = "partial"
    elif counts["planned"] or counts["running"]:
        campaign_status = "in_progress"
    else:
        campaign_status = "complete"
    backends = {(attempt.get("execution_contract") or {}).get("backend", "legacy")
                for state in states.values() for item in state["units"].values()
                for attempt in item["attempts"]}
    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "campaign_status": campaign_status,
        "planned_count": sum(counts.values()),
        "success_count": counts["success"], "failed_count": counts["failed"],
        "interrupted_count": counts["interrupted"], "running_count": counts["running"],
        "pending_count": counts["planned"],
        "active_child_present": dispatch["active_child_present"],
        "active_child_reasons": dispatch["active_child_reasons"],
        "dispatch_blocked": dispatch_blocked,
        "dispatch_blocked_reasons": waiting_reasons,
        "waiting_on_active_child": dispatch_blocked,
        "unit_errors": list(unit_errors), "waiting_reasons": list(waiting_reasons),
        "campaign_profile": plan["campaign_profile"], "campaign_tag": plan["campaign_tag"],
        "plan_revision": plan["plan_revision"], "canonical_output_root": str(root),
        "backend": next(iter(backends)) if len(backends) == 1 else ("mixed" if backends else "none"),
        "execution_policy": execution_policy,
        "action_scope": action_scope,
        "published_count": sum(item["publish"] == "published" for item in units),
        "content_verified_count": sum(item["integrity"] == "content_verified" for item in units),
        "content_mismatch_count": sum(item["integrity"] == "content_mismatch" for item in units),
        "units": units,
    }


def scheduler_status(observation):
    """Present objective backend evidence without reconciling campaign state."""
    ad = observation.get("ad") or {}
    label = {1: "IDLE", 2: "RUNNING", 3: "REMOVED", 4: "COMPLETED",
             5: "HELD", 6: "TRANSFERRING", 7: "SUSPENDED"}.get(ad.get("JobStatus"), "UNKNOWN")
    attention = (label in {"HELD", "REMOVED", "UNKNOWN"}
                 or observation["state"] in {"unresolved", "terminal_failure"}
                 or ad.get("ExitCode") not in (None, 0)
                 or ad.get("ExitSignal") is not None or ad.get("ExitBySignal") is True)
    return {"state": label, "backend_state": observation["state"],
            "source": observation.get("source", "unresolved"),
            "observed_at": best_effort_utc_now(), "attention": attention,
            "reason": observation.get("reason") or ad.get("HoldReason") or ad.get("RemoveReason"),
            "mem_mb": ad.get("MemoryUsage"), "rss_kib": ad.get("ResidentSetSize"),
            "raw_classad": observation.get("ad")}


def scheduler_memory_mb(scheduler):
    """Numeric human display; raw scheduler fields remain available to machines."""
    def numeric(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and value >= 0 and (isinstance(value, int) or math.isfinite(value)))

    memory = scheduler.get("mem_mb")
    if numeric(memory):
        return memory
    rss = scheduler.get("rss_kib")
    if numeric(rss):
        # ResidentSetSize is KiB; round up to whole MiB, as MemoryUsage does.
        return (rss + 1023) // 1024 if isinstance(rss, int) else math.ceil(rss / 1024)
    return None


def show_summary(summary, output_format):
    if output_format == "json":
        print(json.dumps(summary, sort_keys=True))
    elif output_format == "yaml":
        print(yaml.safe_dump(summary, sort_keys=False).rstrip())
    else:
        print(f"Campaign: {summary['campaign_profile']}")
        print(f"Tag: {summary['campaign_tag']}")
        print(f"Historical backends: {summary['backend']} | Plan revision: {summary['plan_revision']}")
        policy = summary["execution_policy"]
        if policy is None:
            print("Future attempts: unbound (legacy campaign)")
        else:
            print("Future attempts: "
                  f"{policy['backend']} / {policy['resource_observer']} / "
                  f"{policy['hard_limit_bytes']} bytes")
        print(f"Final output: {summary['canonical_output_root']}")
        published_paths = [path for unit in summary["units"] for path in unit["canonical_paths"]]
        if published_paths:
            print(f"Published plot example: {published_paths[0]} "
                  f"({len(published_paths)} exact paths in --format yaml/json)")
        else:
            print("Plot destinations will appear after receipt-verified publication.")
        print(f"Campaign state (persisted): {summary['published_count']} published / {summary['planned_count']} planned"
              f" | Running: {summary['running_count']} | Pending: {summary['pending_count']}"
              f" | Failed: {summary['failed_count']} | Interrupted: {summary['interrupted_count']}")
        observations = [unit['scheduler'] for unit in summary['units']
                        if unit.get('scheduler') is not None]
        if observations:
            counts = Counter(observation['state'] for observation in observations)
            labels = ('COMPLETED', 'RUNNING', 'IDLE', 'HELD')
            other = sum(count for label, count in counts.items() if label not in labels)
            print("Scheduler observed: " + " | ".join(
                [*(f"{label} {counts[label]}" for label in labels), f"OTHER/UNKNOWN {other}"]))
        print(f"{'UNIT':32} {'CAMPAIGN VIEW':13} {'PUBLICATION':16} {'INTEGRITY':20} {'JOB':12} PLOTS")
        for unit in summary["units"]:
            print(f"{unit['unit_id']:32} {unit['execution']:13} {unit['publish']:16} "
                  f"{unit['integrity']:20} "
                  f"{unit['job'] or '-':12} {len(unit['canonical_paths'])}")
            if unit["warning"]:
                print(f"  {unit['unit_id']}: {unit['warning']}")
            scheduler = unit.get("scheduler")
            if scheduler is not None:
                ad = scheduler["raw_classad"] or {}
                shown = lambda value: "-" if value is None else str(value)
                print(f"  Scheduler: {scheduler['state']} ({scheduler['source']})"
                      f" | Attention: {scheduler['attention']}"
                      f" | MEM_MB={shown(scheduler_memory_mb(scheduler))} RSS_KIB={shown(scheduler['rss_kib'])}"
                      f" | exit={shown(ad.get('ExitCode'))} signal={shown(ad.get('ExitSignal'))}"
                      f" ExitBySignal={shown(ad.get('ExitBySignal'))}")
                if scheduler["reason"]:
                    print(f"  Scheduler reason: {scheduler['reason']}")
        for message in [*summary["unit_errors"], *summary["waiting_reasons"]]:
            if routine_waiting_reason(message):
                continue
            print(f"  Action needed: {message}")


def routine_waiting_reason(message):
    return ("Condor execution submitted; resume observes this execution only" in message
            or "scheduler execution unresolved" in message
            or message.endswith(": remains running"))


def show_monitor_command(root, output_format):
    command = ("./analysis/topeft_run2/run_plotter.sh "
               f"--status -o {shlex.quote(str(root.resolve()))} --observe-scheduler")
    stream = sys.stdout if output_format == "human" else sys.stderr
    print("From the TOPEFT checkout with the intended Conda environment already active:\n"
          f"Monitor with:\n  {command}", file=stream, flush=True)


def presentation_key(summary):
    return (
        summary["campaign_status"], summary["published_count"],
        summary["running_count"], summary["pending_count"],
        summary["failed_count"], summary["interrupted_count"],
        tuple((unit["unit_id"], unit["execution"], unit["publish"],
               unit["job"], unit["warning"]) for unit in summary["units"]),
        tuple(summary["unit_errors"]),
        tuple(reason for reason in summary["waiting_reasons"]
              if not routine_waiting_reason(reason)),
    )


def follow_campaign(summary, root, profile, tag, selected, backend_name, observer,
                    hard_limit_bytes, output_format, poll_seconds):
    """Repeat controller passes with no writer lock held between observations."""
    while (summary["running_count"] if selected is None else
           any(unit["unit_id"] == selected and unit["execution"] == "running"
               for unit in summary["units"])):
        time.sleep(poll_seconds)
        updated = run_campaign(root, profile, tag, action="resume", selected=selected,
                               backend_name=backend_name, observer=observer,
                               hard_limit_bytes=hard_limit_bytes, emit=False)
        if presentation_key(updated) != presentation_key(summary):
            show_summary(updated, output_format)
        summary = updated
    return summary


def finish_pending_revision(root, profile, tag, runtime):
    """Complete only an already validated, atomically persisted revision intent."""
    path = coordinator_path(root)
    coordinator = read_json(path)
    if "pending_plan" not in coordinator:
        return
    require(coordinator.get("schema_version") == 1
            and coordinator.get("campaign_profile") == profile
            and coordinator.get("campaign_tag") == tag,
            "pending revision coordinator identity contradiction")
    require(isinstance(coordinator.get("runtime_contract_identity"), dict),
            "pending revision coordinator provenance field is malformed")
    old_plan = coordinator["plan"]
    new_plan = coordinator["pending_plan"]
    old_revision = old_plan["plan_revision"]
    require(new_plan["plan_revision"] == old_revision + 1,
            "pending revision sequence contradiction")
    old_units = {unit["unit_id"]: unit for unit in old_plan["units"]}
    new_units = {unit["unit_id"]: unit for unit in new_plan["units"]}
    require(set(old_units) < set(new_units)
            and all(new_units[key] == value for key, value in old_units.items()),
            "pending revision historical binding contradiction")
    for component in PROFILES[profile]:
        reject_symlink_ancestors(component_root(root, profile, component), root)
        path_state = state_path(root, profile, component)
        state = read_json(path_state)
        require(state.get("runtime_contract_identity") == runtime
                and state.get("campaign_profile") == profile
                and state.get("campaign_tag") == tag
                and state.get("component") == component,
                "pending revision component binding contradiction")
        if state["plan_revision"] == old_revision:
            validate_state(state, old_plan, component, runtime)
            for unit_id in sorted(set(new_units) - set(old_units)):
                unit = new_units[unit_id]
                if unit["component"] == component:
                    state["units"][unit_id] = {
                        "binding": unit, "status": "planned", "attempts": [],
                        "materializations": {},
                    }
            state["plan_revision"] = new_plan["plan_revision"]
            state["plan_history"].append(new_plan["plan_revision"])
            atomic_json(path_state, state)
        else:
            require(state["plan_revision"] == new_plan["plan_revision"],
                    "pending revision component revision contradiction")
            validate_state(state, new_plan, component, runtime)
    coordinator["plan"] = new_plan
    del coordinator["pending_plan"]
    atomic_json(coordinator_path(root), coordinator)


def activate_revision(root, profile, tag, plan_path, expected, runtime):
    finish_pending_revision(root, profile, tag, runtime)
    coordinator, old_plan, states = load_campaign(root, profile, tag, runtime)
    require(expected == old_plan["plan_revision"], "expected plan revision mismatch")
    new_plan = normalize_plan(plan_path, profile, tag, root, existing_plan=old_plan)
    require(new_plan["plan_revision"] == expected + 1, "new plan revision must increment by one")
    old_units = {unit["unit_id"]: unit for unit in old_plan["units"]}
    new_units = {unit["unit_id"]: unit for unit in new_plan["units"]}
    require(set(old_units) < set(new_units), "plan revision must append qualified units")
    require(all(new_units[key] == value for key, value in old_units.items()),
            "plan revision attempts to rewrite historical binding")
    coordinator["pending_plan"] = new_plan
    atomic_json(coordinator_path(root), coordinator)
    finish_pending_revision(root, profile, tag, runtime)
    print(f"Activated plan revision {new_plan['plan_revision']}")


def run_campaign(root, profile, tag, *, plan_path=None, action, selected=None,
                 backend_name=None, observer=None, hard_limit_bytes=None,
                 output_format="json", emit=True):
    if action == "execute":
        check_root(root, fresh=True)
        prepared_plan = normalize_plan(plan_path, profile, tag, root)
        if selected:
            require(selected in {unit["unit_id"] for unit in prepared_plan["units"]},
                    "unknown selected unit")
        root.mkdir(mode=0o700)
    else:
        check_root(root, fresh=False)
        prepared_plan = None
    execution_options = {"backend": backend_name, "resource_observer": observer,
                         "hard_limit_bytes": hard_limit_bytes}
    recovery_errors, recovered_ids = (resume_recover(root, profile, tag, selected,
                                                    execution_options)
                                      if action == "resume" else ([], []))
    with campaign_lock(root):
        summary = _run_campaign_locked(root, profile, tag, action, selected, prepared_plan,
                                       execution_options, recovery_errors, recovered_ids)
    if emit:
        show_summary(summary, output_format)
    return summary


def _run_campaign_locked(root, profile, tag, action, selected, prepared_plan,
                         execution_options, recovery_errors=(), recovered_ids=()):
    runtime = (runtime_identity() if action == "execute" else
               read_json(coordinator_path(root)).get("runtime_contract_identity"))
    if action == "execute":
        plan = prepared_plan
        states = {}
        for component in PROFILES[profile]:
            component_path = component_root(root, profile, component)
            reject_symlink_ancestors(component_path, root)
            component_path.mkdir(mode=0o700, parents=True, exist_ok=True)
            states[component] = new_component_state(plan, component, runtime)
            atomic_json(state_path(root, profile, component), states[component])
        coordinator = {
            "schema_version": 1, "campaign_profile": profile, "campaign_tag": tag,
            "runtime_contract_identity": runtime, "plan": plan,
        }
    else:
        finish_pending_revision(root, profile, tag, runtime)
        coordinator, plan, states = load_campaign(root, profile, tag, runtime)
    scope = derive_action_scope(plan, states, selected)
    policy, should_bind = resolve_execution_policy(
        coordinator, execution_options, scope, fresh=action == "execute")
    if should_bind:
        coordinator["execution_policy_v1"] = policy
        atomic_json(coordinator_path(root), coordinator)
    effective_options = {
        "backend": policy["backend"], "observer": policy["resource_observer"],
        "hard_limit_bytes": policy["hard_limit_bytes"],
    }
    unit_errors = list(recovery_errors)
    unresolved_running = []

    def reconcile_units(unit_ids):
        for component in PROFILES[profile]:
            state = states[component]
            for unit_id, unit_state in state["units"].items():
                if unit_id not in unit_ids:
                    continue
                unresolved_running[:] = [item for item in unresolved_running
                                         if not item.startswith(f"{component}/{unit_id}:")]
                try:
                    reconcile_running(plan, component, state, unit_state, root, runtime)
                except ActiveChildUnresolved as error:
                    messages = unresolved_running if unit_state["status"] == "running" else unit_errors
                    messages.append(f"{component}/{unit_id}: {error_description(error)}")
                except ChildDefinitelyTerminal as error:
                    unit_errors.append(f"{component}/{unit_id}: {error_description(error)}")
                except OutputCollision as error:
                    attempt = unit_state["attempts"][-1]
                    attempt["result"] = "failed"
                    attempt["failure_classification"] = error_description(error)
                    unit_state["status"] = "failed"
                    atomic_json(state_path(root, profile, component), state)
                    unit_errors.append(f"{component}/{unit_id}: {error_description(error)}")
                if unit_state["status"] == "running" and not any(
                        item.startswith(f"{component}/{unit_id}:") for item in unresolved_running):
                    unresolved_running.append(f"{component}/{unit_id}: remains running")

    if action == "resume":
        reconcile_units([unit_id for unit_id in scope["reconcilable_unit_ids"]
                         if unit_id not in recovered_ids])
    launched_unit_ids = []
    for component in PROFILES[profile]:
        state = states[component]
        for unit in plan["units"]:
            decision = dispatch_decision(plan, states, scope, policy, unresolved_running)
            if unit["component"] != component or unit["unit_id"] not in decision[
                    "launchable_planned_unit_ids"]:
                continue
            unit_state = state["units"][unit["unit_id"]]
            launched_unit_ids.append(unit["unit_id"])
            try:
                execute_unit(plan, component, state, unit_state, root, runtime, effective_options)
            except ActiveChildUnresolved as error:
                unresolved_running.append(
                    f"{component}/{unit['unit_id']}: {error_description(error)}")
            except (CampaignError, OSError, subprocess.SubprocessError) as error:
                unit_errors.append(
                    f"{component}/{unit['unit_id']}: {error_description(error)}")
    if policy["backend"] == "condor":
        reconcilable = derive_action_scope(plan, states, selected)["reconcilable_unit_ids"]
        reconcile_units([unit_id for unit_id in launched_unit_ids if unit_id in reconcilable])
    decision = dispatch_decision(plan, states, scope, policy, unresolved_running)
    return campaign_summary(root, plan, states, unit_errors=unit_errors,
                            active_reasons=unresolved_running,
                            execution_policy=coordinator.get("execution_policy_v1"),
                            action_scope=scope, dispatch=decision)


def normalize_year_tokens(tokens):
    aliases = {"2016": "2016", "ul16": "2016", "2016apv": "2016APV",
               "ul16apv": "2016APV", "2017": "2017", "ul17": "2017",
               "2018": "2018", "ul18": "2018", "2022": "2022",
               "2022ee": "2022EE", "2023": "2023", "2023bpix": "2023BPix"}
    normalized = []
    for raw in tokens:
        for token in raw.split(","):
            token = token.strip()
            for candidate in YEARS.get(token.lower(), (token,)):
                value = aliases.get(candidate.lower(), candidate)
                if value and value not in normalized:
                    normalized.append(value)
    return normalized


def direct_main(argv):
    """Established ordinary plotting CLI; no second campaign language or state."""
    parser = argparse.ArgumentParser(
        description="Plot control: ordinary inputs; for campaigns use --status -o ROOT or --campaign-profile PROFILE.")
    parser.add_argument("-f", "--input", action="append", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("-n", "--name")
    parser.add_argument("-y", "--year", nargs="+", action="extend", required=True)
    parser.add_argument("--variable", "--variables", nargs="+", action="extend", default=[])
    parser.add_argument("--workers", default="1")
    parser.add_argument("--channel-output", type=str.lower,
                        choices=("merged", "split", "both", "merged-njets", "split-njets", "both-njets"))
    parser.add_argument("--cr", dest="region", action="store_const", const="CR")
    parser.add_argument("--sr", dest="region", action="store_const", const="SR")
    parser.add_argument("--blind", dest="blinding", action="store_const", const="blind")
    parser.add_argument("--unblind", dest="blinding", action="store_const", const="unblind")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-t", "--timestamp", action="store_true")
    parser.add_argument("-q", "--quiet", dest="verbosity", action="store_const", const="--quiet")
    parser.add_argument("-v", "--verbose", dest="verbosity", action="store_const", const="--verbose")
    args, extra = parser.parse_known_args([a for a in argv if a != "--"])
    try:
        for path in args.input:
            require(os.access(path, os.R_OK), f"Input pickle '{path}' is missing or unreadable.")
        region = args.region
        if region is None:
            detected = []
            for path in args.input:
                tokens = [r for r in ("CR", "SR")
                          if re.search(r"(^|[^A-Z0-9])" + r, Path(path).name.upper())]
                if len(tokens) > 1:
                    if len(args.input) > 1:
                        raise CampaignError("Ambiguous input filename; use --cr or --sr to override.")
                    print("Warning: both CR and SR tokens; defaulting to CR.", file=sys.stderr)
                detected.append(tokens[0] if len(tokens) == 1 else None)
            if len(args.input) > 1:
                require(len(set(detected)) == 1,
                        "Input filenames resolve to conflicting regions; use --cr or --sr to override.")
            region = detected[0] or "CR"
            if detected[0]:
                print(f"Auto-detected region '{region}' from input filename.")
            else:
                print("No unique region token detected; defaulting to 'CR'.")
        command = [sys.executable, str(RUN_DIRECTORY / "make_cr_and_sr_plots.py")]
        for path in args.input:
            command.extend(("-f", path))
        command.extend(("-o", args.output_dir))
        if args.name:
            command.extend(("-n", args.name))
        command.extend(("-y", *normalize_year_tokens(args.year), "--" + region.lower(),
                        "--" + (args.blinding or ("unblind" if region == "CR" else "blind"))))
        if args.variable:
            command.extend(("--variables", *args.variable))
        command.extend(("--workers", args.workers))
        if args.channel_output:
            command.extend(("--channel-output", args.channel_output))
        command.extend(extra)
        if args.timestamp:
            command.append("-t")
        if args.verbosity:
            command.append(args.verbosity)
        print("Executing make_cr_and_sr_plots.py with command:\n  " + shlex.join(command))
        if args.dry_run:
            print("Dry-run requested; skipping execution.")
            return 0
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        return backend.run_direct(command)
    except CampaignError as error:
        print(str(error), file=sys.stderr)
        return 2


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a.split("=", 1)[0] in {"--campaign-manifest", "--campaign-phase"} for a in argv):
        print("The TSV campaign interface is retired; use a qualified persistent JSON plan.", file=sys.stderr)
        return 2
    campaign_switches = {"--campaign-profile", "--dry-run", "--execute",
                         "--resume", "--finalize", "--status",
                         "--activate-pending-plan"}
    if not any(a.split("=", 1)[0] in campaign_switches for a in argv):
        return direct_main(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-profile", choices=sorted(PROFILES),
                        help="required for a fresh campaign; existing root binds this identity")
    parser.add_argument("--campaign-tag", help="required for a fresh campaign; existing root binds this identity")
    parser.add_argument("-o", "--output-dir", required=True, type=Path,
                        help="final canonical campaign output root")
    parser.add_argument("--plan", type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--resume", action="store_true")
    actions.add_argument("--finalize", action="store_true",
                         help="finalize proven terminal success and publish canonical PNGs")
    actions.add_argument("--status", action="store_true", help="read-only status for an existing root")
    parser.add_argument("--verify-published", action="store_true",
                        help="with --status, hash recorded canonical PNGs against publication SHA-256")
    parser.add_argument("--observe-scheduler", action="store_true",
                        help="with --status, query bound Condor jobs once; no reconciliation or writes")
    actions.add_argument("--activate-pending-plan", type=Path)
    parser.add_argument("--expected-plan-revision", type=int)
    parser.add_argument("--selected-unit")
    parser.add_argument("--backend", choices=("local", "condor"),
                        help=f"fresh default: {DEFAULT_EXECUTION_POLICY['backend']}; "
                             "legacy planned launch requires explicit backend; bound roots reuse policy")
    parser.add_argument("--resource-observer", choices=("cgroup_v2", "node_pressure"),
                        help=f"fresh default: {DEFAULT_EXECUTION_POLICY['resource_observer']}; "
                             "legacy Condor planned launch requires explicit value")
    parser.add_argument("--hard-limit-bytes", type=int,
                        help=f"fresh default: {DEFAULT_EXECUTION_POLICY['hard_limit_bytes']}; "
                             "legacy Condor planned launch requires explicit value")
    parser.add_argument("--format", choices=("human", "yaml", "json"),
                        help="campaign CLI defaults to human; use json for machine output")
    parser.add_argument("--follow", action="store_true",
                        help="opt in to foreground Condor observation")
    parser.add_argument("--poll-seconds", type=int, default=15,
                        help="foreground Condor observation interval (1..300 seconds)")
    args = parser.parse_args(argv)
    try:
        require(1 <= args.poll_seconds <= 300, "poll interval must be 1..300 seconds")
        require(args.hard_limit_bytes is None or args.hard_limit_bytes > 0,
                "hard limit must be positive")
        if args.selected_unit:
            safe_name(args.selected_unit, "selected_unit")
        root = args.output_dir
        profile = args.campaign_profile
        output_format = args.format or "human"
        require(args.status or not args.verify_published,
                "--verify-published requires --status")
        require(args.status or not args.observe_scheduler,
                "--observe-scheduler requires --status")
        if not (args.dry_run or args.execute):
            check_root(root, fresh=False)
            coordinator = read_json(coordinator_path(root))
            require(coordinator.get("schema_version") == 1, "campaign coordinator schema mismatch")
            bound_profile = coordinator.get("campaign_profile")
            bound_tag = coordinator.get("campaign_tag")
            require(bound_profile in PROFILES and isinstance(bound_tag, str),
                    "campaign coordinator identity malformed")
            require(profile is None or profile == bound_profile, "campaign profile contradicts existing root")
            require(args.campaign_tag is None or args.campaign_tag == bound_tag,
                    "campaign tag contradicts existing root")
            profile, args.campaign_tag = bound_profile, bound_tag
        else:
            require(profile is not None and args.campaign_tag is not None,
                    "fresh campaign requires --campaign-profile and --campaign-tag")
        safe_name(args.campaign_tag, "campaign_tag")
        if args.status:
            require(args.plan is None
                    and args.expected_plan_revision is None and not args.follow
                    and args.backend is None and args.resource_observer is None
                    and args.hard_limit_bytes is None,
                    "status accepts only an existing output root and optional identity/selection/format")
            runtime = coordinator.get("runtime_contract_identity")
            coordinator, plan, states = load_campaign(root, profile, args.campaign_tag, runtime)
            summary = campaign_summary(root, plan, states,
                                       execution_policy=coordinator.get("execution_policy_v1"),
                                       verify_published=args.verify_published,
                                       observe_scheduler=args.observe_scheduler,
                                       selected=args.selected_unit)
            show_summary(summary, output_format)
            return 0
        if args.dry_run or args.execute:
            require(args.plan is not None and args.expected_plan_revision is None,
                    "fresh action requires --plan and no expected revision")
            check_root(root, fresh=True)
            if args.dry_run:
                plan = normalize_plan(args.plan, profile, args.campaign_tag, root)
                dry_run(plan, root, args.selected_unit)
            else:
                fresh_policy, _ = resolve_execution_policy(
                    {}, {"backend": args.backend, "resource_observer": args.resource_observer,
                         "hard_limit_bytes": args.hard_limit_bytes},
                    {"can_start_new_attempt": True}, fresh=True)
                backend_name = fresh_policy["backend"]
                use_follow = args.follow
                require(not use_follow or backend_name == "condor", "follow requires Condor backend")
                if output_format == "human":
                    print(f"Final output before execution: {root}", flush=True)
                summary = run_campaign(root, profile, args.campaign_tag, plan_path=args.plan,
                                       action="execute", selected=args.selected_unit,
                                       backend_name=args.backend,
                                       observer=args.resource_observer,
                                       hard_limit_bytes=args.hard_limit_bytes,
                                       output_format=output_format)
                if backend_name == "condor":
                    show_monitor_command(root, output_format)
                if use_follow:
                    try:
                        follow_campaign(summary, root, profile, args.campaign_tag,
                                        args.selected_unit, backend_name,
                                        fresh_policy["resource_observer"],
                                        fresh_policy["hard_limit_bytes"],
                                        output_format, args.poll_seconds)
                    except KeyboardInterrupt:
                        print("Observation stopped; Condor jobs and campaign state remain recoverable.",
                              file=sys.stderr)
                        return 130
        elif args.finalize:
            require(args.plan is None and args.expected_plan_revision is None
                    and not args.follow and args.backend is None
                    and args.resource_observer is None and args.hard_limit_bytes is None,
                    "finalize accepts only existing campaign identity, selection and format")
            finalize_campaign(root, profile, args.campaign_tag, args.selected_unit,
                              output_format=output_format)
        elif args.resume:
            require(args.plan is None and args.expected_plan_revision is None,
                    "resume accepts neither --plan nor expected revision")
            runtime = coordinator.get("runtime_contract_identity")
            coordinator, plan, states = load_campaign(root, profile, args.campaign_tag, runtime)
            scope = derive_action_scope(plan, states, args.selected_unit)
            resume_policy, _ = resolve_execution_policy(
                coordinator, {"backend": args.backend,
                              "resource_observer": args.resource_observer,
                              "hard_limit_bytes": args.hard_limit_bytes}, scope)
            backend_name = resume_policy["backend"]
            use_follow = args.follow
            require(not use_follow or backend_name == "condor", "follow requires Condor backend")
            if output_format == "human":
                print(f"Final output before execution: {root}", flush=True)
            summary = run_campaign(root, profile, args.campaign_tag, action="resume",
                                   selected=args.selected_unit, backend_name=args.backend,
                                   observer=args.resource_observer,
                                   hard_limit_bytes=args.hard_limit_bytes,
                                   output_format=output_format)
            if backend_name == "condor":
                show_monitor_command(root, output_format)
            if use_follow:
                try:
                    follow_campaign(summary, root, profile, args.campaign_tag,
                                args.selected_unit, backend_name,
                                resume_policy["resource_observer"],
                                resume_policy["hard_limit_bytes"],
                                output_format, args.poll_seconds)
                except KeyboardInterrupt:
                    print("Observation stopped; Condor jobs and campaign state remain recoverable.",
                          file=sys.stderr)
                    return 130
        else:
            require(args.plan is None and args.selected_unit is None
                    and args.expected_plan_revision is not None,
                    "revision requires only --activate-pending-plan and expected revision")
            check_root(root, fresh=False)
            with campaign_lock(root):
                activate_revision(root, profile, args.campaign_tag,
                                  args.activate_pending_plan, args.expected_plan_revision,
                                  read_json(coordinator_path(root)).get("runtime_contract_identity"))
        return 0
    except (CampaignError, OSError, TypeError, KeyError) as error:
        print(f"plot campaign: {error_description(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
