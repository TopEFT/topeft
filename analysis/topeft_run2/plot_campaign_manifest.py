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
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import shlex

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
}
YEARS = {
    "run2": ("UL16", "UL16APV", "UL17", "UL18"),
    "run3": ("2022", "2022EE", "2023", "2023BPix"),
}
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
terminal_postprocessing_pending = "terminal postprocessing pending"
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
    name = f".sr_crlike_{component}_plot_campaign_state_v1.json"
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
        require(isinstance(attempts, list) and len(attempts) == (0 if status == "planned" else 1),
                f"unit status/attempt history contradiction: {unit_id}")
        if attempts:
            attempt = attempts[0]
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


def execute_unit(plan, component, state, unit_state, root, runtime, execution_options):
    unit = unit_state["binding"]
    require(unit_state["status"] == "planned", "only planned units may start")
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
        receipt = create_receipt(plan, unit, attempt, runtime)
        attempt["completion_receipt_identity"] = identity(receipt)
        atomic_json(state_path(root, plan["campaign_profile"], component), state)
        materialize(plan, component, state, unit_state, attempt, receipt, root, runtime)
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
        expected_root = private_root(root, plan["campaign_profile"],
                                     unit_state["binding"]["unit_id"], component, "attempt_0001")
        require(attempt.get("attempt_id") == "attempt_0001" and attempt_root == expected_root,
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
    try:
        if attempt.get("execution_contract", {}).get("backend") == "condor":
            observation = backend.observe(argv, attempt_root, attempt["execution_contract"])
            if observation["state"] == "terminal_failure":
                guard_terminal(guard_path)
                attempt.update(result="failed", failure_classification="scheduler terminal failure",
                               ended_at_utc=best_effort_utc_now())
                unit_state["status"] = "failed"
                atomic_json(state_path(root, plan["campaign_profile"], component), state)
                return
            require(observation["state"] == "terminal_success", "scheduler execution unresolved")
        guard_terminal(guard_path)
    except Exception as error:
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
        receipt_path = completion_receipt_path(attempt_root)
        if receipt_path.is_file() and not receipt_path.is_symlink():
            receipt = read_json(receipt_path)
            materialize(plan, component, state, unit_state, attempt, receipt, root, runtime)
            return
        receipt = create_receipt(plan, unit_state["binding"], attempt, runtime)
        materialize(plan, component, state, unit_state, attempt, receipt, root, runtime)
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
    receipt_path = completion_receipt_path(Path(attempt["private_unit_root"]))
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = read_json(receipt_path)
    else:
        receipt = create_receipt(plan, unit, attempt, runtime, provenance)
    materialize(plan, component, state, unit_state, attempt, receipt, root, runtime)


def repair_materializations(root, profile, tag, selected):
    """Maintain existing outputs and one terminal unit; never enter execution."""
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
        for component in PROFILES[profile]:
            state = states[component]
            for unit_state in state["units"].values():
                if unit_state["status"] == "success":
                    migrate_successful_unit(plan, component, state, unit_state, root, runtime)
        if selected_unit["status"] == "interrupted":
            recover_terminal_unit(plan, selected_component, selected_state,
                                  selected_unit, root, runtime)
        print(json.dumps({"recovered_unit": selected, "status": selected_unit["status"],
                          "success_count": sum(u["status"] == "success"
                                               for s in states.values() for u in s["units"].values())},
                         sort_keys=True))


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
                 backend_name="local", observer="cgroup_v2", hard_limit_bytes=245000000000):
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
    with campaign_lock(root):
        return _run_campaign_locked(root, profile, tag, action, selected, prepared_plan,
                                    {"backend": backend_name, "observer": observer,
                                     "hard_limit_bytes": hard_limit_bytes})


def _run_campaign_locked(root, profile, tag, action, selected, prepared_plan, execution_options):
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
        atomic_json(coordinator_path(root), {
            "schema_version": 1, "campaign_profile": profile, "campaign_tag": tag,
            "runtime_contract_identity": runtime, "plan": plan,
        })
    else:
        finish_pending_revision(root, profile, tag, runtime)
        _, plan, states = load_campaign(root, profile, tag, runtime)
        if selected:
            require(selected in {unit["unit_id"] for unit in plan["units"]}, "unknown selected unit")
    unit_errors = []
    unresolved_running = []
    if action == "resume":
        for component in PROFILES[profile]:
            state = states[component]
            for unit_id, unit_state in state["units"].items():
                if unit_state["status"] == "running" or (
                    unit_state["status"] == "interrupted" and unit_state["attempts"]
                    and unit_state["attempts"][-1].get("failure_classification")
                    == terminal_postprocessing_pending
                ):
                    try:
                        reconcile_running(plan, component, state, unit_state, root, runtime)
                    except ActiveChildUnresolved as error:
                        unresolved_running.append(
                            f"{component}/{unit_id}: {error_description(error)}")
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
    if not unresolved_running:
        for component in PROFILES[profile]:
            state = states[component]
            for unit in plan["units"]:
                if unresolved_running:
                    break
                if unit["component"] != component or (selected and unit["unit_id"] != selected):
                    continue
                unit_state = state["units"][unit["unit_id"]]
                if unit_state["status"] != "planned":
                    continue
                try:
                    execute_unit(plan, component, state, unit_state, root, runtime, execution_options)
                except ActiveChildUnresolved as error:
                    unresolved_running.append(
                        f"{component}/{unit['unit_id']}: {error_description(error)}")
                except (CampaignError, OSError, subprocess.SubprocessError) as error:
                    unit_errors.append(
                        f"{component}/{unit['unit_id']}: {error_description(error)}")
    counts = Counter(unit["status"] for state in states.values()
                     for unit in state["units"].values())
    if unresolved_running:
        campaign_status = "waiting"
    elif counts["failed"] or counts["interrupted"] or unit_errors:
        campaign_status = "partial"
    elif counts["planned"] or counts["running"]:
        campaign_status = "in_progress"
    else:
        campaign_status = "complete"
    print(json.dumps({
        "campaign_status": campaign_status,
        "planned_count": sum(counts.values()),
        "success_count": counts["success"], "failed_count": counts["failed"],
        "interrupted_count": counts["interrupted"], "running_count": counts["running"],
        "pending_count": counts["planned"],
        "waiting_on_active_child": bool(unresolved_running),
        "unit_errors": unit_errors, "waiting_reasons": unresolved_running,
    }, sort_keys=True))


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
    parser = argparse.ArgumentParser(description="Plot control: ordinary inputs or --campaign-profile.")
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
    if "--campaign-profile" not in argv and not any(a.startswith("--campaign-profile=") for a in argv):
        return direct_main(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--campaign-tag", required=True)
    parser.add_argument("-o", "--output-dir", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--resume", action="store_true")
    actions.add_argument("--recover-terminal-unit", action="store_true")
    actions.add_argument("--activate-pending-plan", type=Path)
    parser.add_argument("--expected-plan-revision", type=int)
    parser.add_argument("--selected-unit")
    parser.add_argument("--backend", choices=("local", "condor"), default="local")
    parser.add_argument("--resource-observer", choices=("cgroup_v2", "node_pressure"), default="cgroup_v2")
    parser.add_argument("--hard-limit-bytes", type=int, default=245000000000)
    args = parser.parse_args(argv)
    try:
        safe_name(args.campaign_tag, "campaign_tag")
        require(args.hard_limit_bytes > 0, "hard limit must be positive")
        if args.selected_unit:
            safe_name(args.selected_unit, "selected_unit")
        root = args.output_dir
        profile = args.campaign_profile
        if args.dry_run or args.execute:
            require(args.plan is not None and args.expected_plan_revision is None,
                    "fresh action requires --plan and no expected revision")
            check_root(root, fresh=True)
            if args.dry_run:
                plan = normalize_plan(args.plan, profile, args.campaign_tag, root)
                dry_run(plan, root, args.selected_unit)
            else:
                run_campaign(root, profile, args.campaign_tag, plan_path=args.plan,
                             action="execute", selected=args.selected_unit, backend_name=args.backend,
                             observer=args.resource_observer, hard_limit_bytes=args.hard_limit_bytes)
        elif args.recover_terminal_unit:
            require(args.plan is None and args.expected_plan_revision is None
                    and args.selected_unit is not None,
                    "recovery requires only --selected-unit")
            repair_materializations(root, profile, args.campaign_tag, args.selected_unit)
        elif args.resume:
            require(args.plan is None and args.expected_plan_revision is None,
                    "resume accepts neither --plan nor expected revision")
            run_campaign(root, profile, args.campaign_tag, action="resume",
                         selected=args.selected_unit, backend_name=args.backend,
                         observer=args.resource_observer, hard_limit_bytes=args.hard_limit_bytes)
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
