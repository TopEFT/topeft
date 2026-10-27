"""Persistent local SR CR-like plotting campaigns.

Plans bind qualified producer sidecars, never a predicted figure inventory. A
successful child is observed in its private root; a receipt then owns the exact
PNG bytes copied to the canonical component root. This module never loads a PKL.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


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
UNIT_FIELDS = {
    "unit_id", "component", "producer_unit_id", "input_path",
    "sidecar_path", "qualification_source", "period_view",
}


class CampaignError(ValueError):
    """A bounded plan, state, receipt, or output contract failed."""


class ActiveChildUnresolved(CampaignError):
    """Terminal evidence is absent, so another heavy child cannot start."""


class ChildDefinitelyNotStarted(CampaignError):
    """Prelaunch setup or the first process creation failed without a child."""


class OutputCollision(CampaignError):
    """Canonical output cannot be written without risking another owner's bytes."""


def require(condition, message):
    if not condition:
        raise CampaignError(message)


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    path = Path(value)
    require(not path.is_absolute() and len(path.parts) >= 3
            and all(part not in {".", ".."} and SAFE_NAME.fullmatch(part)
                    for part in path.parts), f"unsafe output path: {value!r}")
    require(path.suffix == ".png" and path.parts[1] in {"merged", "merged_njets"},
            f"noncanonical PNG path: {value!r}")
    return path


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
    return unit["component"] if unit["period_view"] == "combined" else (
        "2016APV" if unit["period_view"] == "UL16APV" else
        {"UL16": "2016", "UL17": "2017", "UL18": "2018"}.get(unit["period_view"], unit["period_view"]))


def child_argv(unit, attempt_root):
    chosen_period = period(unit)
    return [
        str(RUN_DIRECTORY / "run_plotter.sh"), "-f", unit["input_path"],
        "-o", str(attempt_root / chosen_period), "-n", "merged",
        "-y", *year_args(unit), "--sr", "--unblind", "--variables", *unit["variables"],
        "--channel-output", "merged", "--binning", "fitting", "--workers", "1",
        "--output-formats", "png", "--year-coverage-policy", "warn",
        "--suppress-merge-report", "--canonical-campaign-root", str(attempt_root),
        "--canonical-period", chosen_period, "--negative-report-suffix", unit["unit_id"],
    ]


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


def lifecycle_receipt_path(attempt_root):
    return attempt_root / "_metadata" / "lifecycle_receipt_v1.json"


def verify_lifecycle_receipt(receipt, attempt, argv):
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


def create_receipt(plan, unit, attempt, runtime):
    attempt_root = Path(attempt["private_unit_root"])
    require(attempt.get("guard_result") == 0 and attempt.get("child_result") == 0
            and attempt.get("tee_result") == 0, "terminal lifecycle is not successful")
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
    path = completion_receipt_path(attempt_root)
    require(not path.exists() and not path.is_symlink(), "completion receipt already exists")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_json(path, receipt)
    return receipt


def verify_receipt(receipt, plan, unit, attempt, runtime):
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
            and all(receipt.get(key) == 0 for key in ("guard_result", "child_result", "tee_result")),
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
    require(attempt.get("completion_receipt_identity") in {None, receipt_id},
            "state/receipt identity contradiction")
    return receipt_id


def reject_symlink_ancestors(path, boundary):
    current = Path(path)
    boundary = Path(boundary)
    require(current == boundary or boundary in current.parents,
            "output path escapes component root")
    while current != boundary:
        require(not current.is_symlink(), f"symbolic-link output ancestor: {current}")
        current = current.parent
    require(not boundary.is_symlink(), f"symbolic-link component root: {boundary}")


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
    component_path = component_root(root, plan["campaign_profile"], component)
    for relative in receipt["observed_png_relative_paths"]:
        safe_relative(relative)
        source = Path(attempt["private_unit_root"]) / relative
        destination = component_path / relative
        reject_symlink_ancestors(destination.parent, component_path)
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
        if prior is None:
            if destination.exists() or destination.is_symlink():
                raise OutputCollision(f"unbound canonical destination collision: {destination}")
            unit_state["materializations"][relative] = binding
            atomic_json(state_path(root, plan["campaign_profile"], component), state)
        if destination.exists() or destination.is_symlink():
            if not (destination.is_file() and not destination.is_symlink()
                    and destination.stat().st_size == binding["size_bytes"]
                    and sha256_file(destination) == digest):
                raise OutputCollision(f"different-byte canonical destination collision: {destination}")
        else:
            try:
                atomic_copy_without_overwrite(source, destination)
            except FileExistsError as error:
                raise OutputCollision(f"canonical destination collision: {destination}") from error
    attempt["materialization_status"] = "complete"
    attempt["completion_receipt_identity"] = receipt_id
    attempt["result"] = "success"
    unit_state["status"] = "success"
    atomic_json(state_path(root, plan["campaign_profile"], component), state)


def guard_terminal(path):
    require(path.is_file() and not path.is_symlink(), "guard terminal evidence missing")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError("guard evidence malformed") from error
    results = [row for row in rows if row.get("event") == "result"]
    require(len(results) == 1 and rows[-1] is results[0], "guard terminal evidence ambiguous")
    return results[0]


def run_guarded_child(argv, attempt_root):
    try:
        metadata = attempt_root / "_metadata"
        metadata.mkdir(mode=0o700, exist_ok=True)
        guard_path = metadata / "guard.jsonl"
        log_path = metadata / "renderer.log"
        guard = [sys.executable, "-m", "topeft.modules.node_memory_guard",
                 "--hard-limit-bytes", "245000000000", "--sampling-interval-seconds", "0.5",
                 "--evidence-path", str(guard_path), "--", *argv]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO_ROOT) + (":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
        environment["PYTHON_BIN"] = sys.executable
        with log_path.open("xb"):
            pass
        guard_process = subprocess.Popen(guard, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         env=environment)
    except (OSError, subprocess.SubprocessError) as error:
        raise ChildDefinitelyNotStarted(f"guard process was not created: {error}") from error
    try:
        tee_process = subprocess.Popen(["tee", "-a", str(log_path)],
                                       stdin=guard_process.stdout, env=environment)
    except BaseException:
        guard_process.wait()
        raise
    guard_process.stdout.close()
    tee_result = tee_process.wait()
    guard_result = guard_process.wait()
    terminal = guard_terminal(guard_path)
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
    return guard_result, child_result, tee_result


def execute_unit(plan, component, state, unit_state, root, runtime):
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
        guard_result, child_result, tee_result = run_guarded_child(argv, attempt_root)
        lifecycle = verify_lifecycle_receipt(read_json(lifecycle_receipt_path(attempt_root)),
                                            attempt, argv)
        require((guard_result, child_result, tee_result) == (
            lifecycle["guard_result"], lifecycle["child_result"], lifecycle["tee_result"]),
            "supervisor return contradicts lifecycle receipt")
        attempt.update(ended_at_utc=lifecycle["ended_at_utc"], guard_result=guard_result,
                       child_result=child_result, tee_result=tee_result)
        atomic_json(state_path(root, plan["campaign_profile"], component), state)
        require((guard_result, child_result, tee_result) == (0, 0, 0),
                "terminal guard/child/tee failure")
        receipt = create_receipt(plan, unit, attempt, runtime)
        attempt["completion_receipt_identity"] = identity(receipt)
        atomic_json(state_path(root, plan["campaign_profile"], component), state)
        materialize(plan, component, state, unit_state, attempt, receipt, root, runtime)
    except (CampaignError, OSError, subprocess.SubprocessError) as error:
        if attempt["guard_result"] is None and launch_possible and not isinstance(error, ChildDefinitelyNotStarted):
            attempt["failure_classification"] = str(error)
            atomic_json(state_path(root, plan["campaign_profile"], component), state)
            raise ActiveChildUnresolved(f"child terminal lifecycle is unresolved: {error}") from error
        attempt["ended_at_utc"] = attempt["ended_at_utc"] or utc_now()
        attempt["result"] = "failed"
        attempt["failure_classification"] = str(error)
        unit_state["status"] = "failed"
        atomic_json(state_path(root, plan["campaign_profile"], component), state)
        raise


def reconcile_running(plan, component, state, unit_state, root, runtime):
    require(unit_state["status"] == "running" and unit_state["attempts"],
            "running unit lacks attempt")
    attempt = unit_state["attempts"][-1]
    attempt_root = Path(attempt["private_unit_root"])
    argv = child_argv(unit_state["binding"], attempt_root)
    require(attempt.get("command_identity") in {None, identity(argv)},
            "running attempt command identity contradiction")
    lifecycle_path = lifecycle_receipt_path(attempt_root)
    if not lifecycle_path.exists() and not lifecycle_path.is_symlink():
        guard_path = attempt_root / "_metadata" / "guard.jsonl"
        if not guard_path.is_file() or guard_path.is_symlink():
            raise ActiveChildUnresolved("terminal guard evidence unavailable")
        guard_terminal(guard_path)
        attempt["ended_at_utc"] = utc_now()
        attempt["result"] = "interrupted"
        attempt["failure_classification"] = "guard terminal but complete lifecycle unavailable"
        unit_state["status"] = "interrupted"
        atomic_json(state_path(root, plan["campaign_profile"], component), state)
        return
    lifecycle = verify_lifecycle_receipt(read_json(lifecycle_path), attempt, argv)
    for key in ("guard_result", "child_result", "tee_result"):
        require(attempt.get(key) in {None, lifecycle[key]},
                f"running attempt {key} contradicts lifecycle receipt")
        attempt[key] = lifecycle[key]
    require(attempt.get("ended_at_utc") in {None, lifecycle["ended_at_utc"]},
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


def run_campaign(root, profile, tag, *, plan_path=None, action, selected=None):
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
        return _run_campaign_locked(root, profile, tag, action, selected, prepared_plan)


def _run_campaign_locked(root, profile, tag, action, selected, prepared_plan):
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
                if unit_state["status"] == "running":
                    try:
                        reconcile_running(plan, component, state, unit_state, root, runtime)
                    except ActiveChildUnresolved as error:
                        unresolved_running.append(f"{component}/{unit_id}: {error}")
                    except OutputCollision as error:
                        attempt = unit_state["attempts"][-1]
                        attempt["result"] = "failed"
                        attempt["failure_classification"] = str(error)
                        unit_state["status"] = "failed"
                        atomic_json(state_path(root, profile, component), state)
                        unit_errors.append(f"{component}/{unit_id}: {error}")
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
                    execute_unit(plan, component, state, unit_state, root, runtime)
                except ActiveChildUnresolved as error:
                    unresolved_running.append(f"{component}/{unit['unit_id']}: {error}")
                except (CampaignError, OSError, subprocess.SubprocessError) as error:
                    unit_errors.append(f"{component}/{unit['unit_id']}: {error}")
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--campaign-tag", required=True)
    parser.add_argument("-o", "--output-dir", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--resume", action="store_true")
    actions.add_argument("--activate-pending-plan", type=Path)
    parser.add_argument("--expected-plan-revision", type=int)
    parser.add_argument("--selected-unit")
    args = parser.parse_args(argv)
    try:
        safe_name(args.campaign_tag, "campaign_tag")
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
                             action="execute", selected=args.selected_unit)
        elif args.resume:
            require(args.plan is None and args.expected_plan_revision is None,
                    "resume accepts neither --plan nor expected revision")
            run_campaign(root, profile, args.campaign_tag, action="resume",
                         selected=args.selected_unit)
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
        print(f"plot campaign: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
