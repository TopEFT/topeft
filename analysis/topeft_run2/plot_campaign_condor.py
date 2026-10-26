"""Prepare shared-filesystem Condor material and run one SR plotting PKL unit.

The accepted logical TSV owns plot coverage. This module derives execution
units from it; descriptors are checked against the TSV again at execution.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Optional

from topeft.modules.plot_output_layout import (
    canonical_row_directory, canonical_receipt_path,
    normalize_plot_period,
)


_header = (
    "unit", "kind", "run", "view", "owner", "years", "variables",
    "expected_figures", "pkl", "historical_pkl_sha256", "sidecar",
    "historical_sidecar_sha256",
)
_safe_name = re.compile(r"^[A-Za-z0-9_.+-]+$")
_safe_path = re.compile(r"^/[A-Za-z0-9_./+-]+$")
_resource = re.compile(r"^[1-9][0-9]*(?:[KMGTP]B?|B)?$", re.IGNORECASE)
_allowed_variables = frozenset(("njets", "lj0pt", "ptz", "ptz_wtau", "ptll", "lt"))
_run_years = {
    "run2": frozenset(("2016APV", "2016", "2017", "2018")),
    "run3": frozenset(("2022", "2022EE", "2023", "2023BPix")),
}
_repo_root = Path(__file__).resolve().parents[2]
_workspace_root = _repo_root.parent
_wrapper = _workspace_root / "codex-run.sh"
_python = Path("/users/apiccine/work/miniconda3/envs/clib-env/bin/python")

# A direct script invocation puts analysis/topeft_run2, not the repository root,
# on sys.path. Bind imports to the selected checkout.
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _safe_absolute(path: Path) -> str:
    value = str(path)
    if not path.is_absolute() or not _safe_path.fullmatch(value) or ".." in path.parts:
        raise ValueError(f"Expected a simple absolute shared-filesystem path: {value!r}")
    return value


def _output_relative(row: dict) -> str:
    return canonical_row_directory(Path("."), row["run"], row["view"]).as_posix()


def read_campaign(manifest_path: Path) -> tuple[list[dict], str]:
    """Validate the logical catalogue without reading its PKL inputs."""
    _safe_absolute(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if tuple(reader.fieldnames or ()) != _header:
            raise ValueError("Unexpected SR campaign TSV header")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty SR campaign")
    seen_ids = set()
    for row in rows:
        if None in row or any(value is None or not value for value in row.values()):
            raise ValueError("Incomplete SR campaign row")
        for key in ("unit", "run", "view", "owner"):
            if not _safe_name.fullmatch(row[key]) or row[key] in {".", ".."}:
                raise ValueError(f"Unsafe campaign {key}: {row[key]!r}")
        if row["kind"] not in {"njets", "other"} or row["run"] not in {"run2", "run3"}:
            raise ValueError("Invalid campaign kind or run")
        variables = row["variables"].split(",")
        years = row["years"].split(",")
        if (not years or not variables or len(variables) != len(set(variables))
                or not set(variables) <= _allowed_variables
                or (row["kind"] == "njets") != (variables == ["njets"])):
            raise ValueError(f"Invalid campaign selection: {row['unit']}")
        if len(years) != len(set(years)) or not set(years) <= _run_years[row["run"]]:
            raise ValueError(f"Invalid campaign years: {row['unit']}")
        if not re.fullmatch(r"[1-9][0-9]*", row["expected_figures"]):
            raise ValueError("Expected figure count must be positive")
        for key in ("pkl", "sidecar"):
            _safe_absolute(Path(row[key]))
        if row["sidecar"] != row["pkl"] + ".metadata.json":
            raise ValueError("Sidecar does not identify principal PKL")
        for key in ("historical_pkl_sha256", "historical_sidecar_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise ValueError("Invalid historical digest")
        if row["unit"] in seen_ids:
            raise ValueError("Duplicate logical row")
        seen_ids.add(row["unit"])
    return rows, _sha256(manifest_path)


def derive_units(rows: list[dict], manifest_sha256: str) -> list[dict]:
    """Partition each logical row exactly once by principal PKL identity."""
    groups = {}
    for row in rows:
        groups.setdefault(row["pkl"], []).append(row)
    units = []
    for pkl_path, group in sorted(groups.items()):
        group = sorted(group, key=lambda row: row["unit"])
        if len({row["sidecar"] for row in group}) != 1:
            raise ValueError(f"Conflicting sidecar paths for {pkl_path}")
        families = sorted({variable for row in group
                           for variable in row["variables"].split(",")})
        # njets-only units can use the established SR physical-channel selector.
        # A mixed unit retains all physical channels needed by its other views.
        channel_selection = "sr_njets_patterns" if families == ["njets"] else "all_present"
        unit = {
            "unit_id": "pkl_" + hashlib.sha256(pkl_path.encode()).hexdigest()[:16],
            "principal_pkl": pkl_path,
            "sidecar": group[0]["sidecar"],
            "historical_pkl_sha256": group[0]["historical_pkl_sha256"],
            "historical_sidecar_sha256": group[0]["historical_sidecar_sha256"],
            "manifest_sha256": manifest_sha256,
            "row_ids": [row["unit"] for row in group],
            "years": sorted({year for row in group for year in row["years"].split(",")}),
            "regions": ["SR"],
            "families": families,
            "channel_selection": channel_selection,
            "expected_figures": sum(int(row["expected_figures"]) for row in group),
            "outputs": [_output_relative(row) for row in group],
            "rows": group,
        }
        units.append(unit)
    if sorted(row_id for unit in units for row_id in unit["row_ids"]) != sorted(
            row["unit"] for row in rows):
        raise ValueError("Execution partition lost or duplicated a logical row")
    return units


def _source_identity() -> dict:
    try:
        result = subprocess.run(
            ["git", "-C", str(_repo_root), "rev-parse", "HEAD"],
            check=False, capture_output=True, text=True,
        )
    except OSError:
        return {"head": None}
    return {"head": result.stdout.strip() if result.returncode == 0 else None}


def _histos_root(output_root: Path) -> str:
    value = _safe_absolute(output_root)
    histos = _repo_root / "histos"
    if not value.startswith(str(histos) + "/") or not histos.is_dir():
        raise ValueError("Output root must be a campaign directory under TOPEFT histos")
    if output_root.resolve(strict=False) != output_root:
        raise ValueError("Output root must not traverse a symbolic link")
    return value


def prepare(manifest_path: Path, destination: Path, output_root: Path, *,
            request_cpus: int, request_memory: str, request_disk: str,
            max_runtime: int, workers: int, unit_id: Optional[str] = None) -> dict:
    rows, manifest_sha256 = read_campaign(manifest_path)
    units = derive_units(rows, manifest_sha256)
    campaign_principal_pkl_paths = sorted({row["pkl"] for row in rows})
    if len(campaign_principal_pkl_paths) != 10:
        raise ValueError("Accepted SR campaign must declare exactly 10 principal PKL paths")
    if unit_id is not None:
        units = [unit for unit in units if unit["unit_id"] == unit_id]
        if len(units) != 1:
            raise ValueError(f"Unknown or ambiguous campaign unit: {unit_id!r}")
    if request_cpus < workers or workers < 1 or max_runtime < 1:
        raise ValueError("Positive workers/runtime and request_cpus >= workers are required")
    if not _resource.fullmatch(request_memory) or not _resource.fullmatch(request_disk):
        raise ValueError("Explicit positive HTCondor memory and disk requests are required")
    _safe_absolute(destination)
    output_path = _histos_root(output_root)
    for path in (destination, manifest_path, Path(output_path), _wrapper, _python, Path(__file__)):
        _safe_absolute(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Dispatch destination already exists: {destination}")
    source_identity = _source_identity()
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    (destination / "units").mkdir(mode=0o700)
    (destination / "logs").mkdir(mode=0o700)
    (destination / "campaign_inputs.json").write_bytes(_json_bytes({
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "campaign_principal_pkl_paths": campaign_principal_pkl_paths,
        "rows_by_principal_pkl": {
            path: sorted(row["unit"] for row in rows if row["pkl"] == path)
            for path in campaign_principal_pkl_paths
        },
    }))
    items = []
    for unit in units:
        descriptor = {
            "schema": "plot_campaign_unit_v1",
            "campaign_id": output_root.name,
            "campaign_manifest": str(manifest_path),
            "output_root": output_path,
            "source_identity": source_identity,
            "campaign_principal_pkl_paths": campaign_principal_pkl_paths,
            "workers": workers,
            **unit,
        }
        unit_path = destination / "units" / (unit["unit_id"] + ".json")
        unit_path.write_bytes(_json_bytes(descriptor))
        items.append(f"{unit['unit_id']} {_sha256(unit_path)}")
    (destination / "items.txt").write_text("\n".join(items) + "\n", encoding="utf-8")
    worker = destination / "worker.sh"
    inner = (
        f"cd {shlex.quote(str(_repo_root))} && exec {shlex.quote(str(_python))} "
        f"{shlex.quote(str(Path(__file__).resolve()))} run-unit "
        f"--descriptor {shlex.quote(str(destination / 'units'))}/\"$1\".json "
        "--descriptor-sha256 \"$2\""
    )
    worker.write_text(
        "#!/bin/sh\nset -eu\nexec " + shlex.quote(str(_wrapper))
        + " /bin/bash --noprofile --norc -c " + shlex.quote(inner)
        + " plot_campaign_unit \"$1\" \"$2\"\n", encoding="utf-8",
    )
    worker.chmod(0o700)
    submit = destination / "plots.sub"
    submit.write_text("\n".join((
        "universe = vanilla",
        f"executable = {worker}",
        'arguments = "$(unit_id) $(unit_sha256)"',
        f"initialdir = {destination}",
        "should_transfer_files = NO",
        "getenv = false",
        f"request_cpus = {request_cpus}",
        f"request_memory = {request_memory}",
        f"request_disk = {request_disk}",
        f"allowed_execute_duration = {max_runtime}",
        f"output = {destination}/logs/$(unit_id)__$(Cluster).$(Process).out",
        f"error = {destination}/logs/$(unit_id)__$(Cluster).$(Process).err",
        f"log = {destination}/logs/$(unit_id)__$(Cluster).$(Process).events.log",
        f"queue unit_id, unit_sha256 from {destination / 'items.txt'}",
    )) + "\n", encoding="utf-8")
    return {"rows": len(rows), "njets_rows": sum(row["kind"] == "njets" for row in rows),
            "other_rows": sum(row["kind"] == "other" for row in rows),
            "units": len(units), "queued_rows": sum(len(unit["rows"]) for unit in units),
            "expected_figures": sum(unit["expected_figures"] for unit in units),
            "manifest_sha256": manifest_sha256, "submit_file": str(submit),
            "submitted": False}


_monitor_ads = (
    "ClusterId", "ProcId", "JobStatus", "NumJobStarts", "RequestCpus",
    "RequestMemory", "RequestDisk", "RemoteHost", "LastRemoteHost",
    "RemoteWallClockTime", "MemoryUsage", "ResidentSetSize", "ExitCode",
    "ExitBySignal", "ExitSignal", "HoldReason", "RemoveReason",
)
_monitor_states = {1: "IDLE", 2: "RUNNING", 3: "REMOVED", 4: "COMPLETED",
                   5: "HELD", 6: "TRANSFERRING", 7: "SUSPENDED"}
_monitor_counts = ("idle", "running", "held", "transferring", "suspended",
                   "resolved_terminal", "unknown")


def _monitor_units(diagnostics_dir: Path) -> tuple[list[dict], Path, int]:
    """Bind submitted jobs to intact prepare descriptors before querying Condor."""
    mapping = json.loads((diagnostics_dir / "campaign_submission_map.json").read_text(
        encoding="utf-8"))
    entries = mapping.get("units")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Missing campaign submission units")
    units = []
    seen_units, seen_jobs, seen_rows = set(), set(), set()
    output_roots, campaign_ids, campaign_manifests = set(), set(), set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Malformed campaign submission unit")
        unit_id = entry.get("unit_id")
        job = entry.get("scheduler_id")
        if (not isinstance(unit_id, str) or not _safe_name.fullmatch(unit_id)
                or not isinstance(job, str) or not re.fullmatch(r"[0-9]+\.[0-9]+", job)
                or unit_id in seen_units or job in seen_jobs):
            raise ValueError("Malformed or duplicate unit/job identity")
        seen_units.add(unit_id)
        seen_jobs.add(job)
        descriptor_path = Path(entry.get("descriptor_path", ""))
        if (descriptor_path != diagnostics_dir / "dispatch" / unit_id / "units" / (unit_id + ".json")
                or descriptor_path.is_symlink()):
            raise ValueError("Descriptor path disagrees with campaign dispatch")
        if _sha256(descriptor_path) != entry.get("descriptor_sha256"):
            raise ValueError("Descriptor digest disagrees with campaign mapping")
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        row_ids = descriptor.get("row_ids")
        if (descriptor.get("schema") != "plot_campaign_unit_v1"
                or descriptor.get("unit_id") != unit_id
                or descriptor.get("principal_pkl") != entry.get("principal_pkl_path")
                or descriptor.get("output_root") != entry.get("output_root")
                or not isinstance(descriptor.get("campaign_manifest"), str)
                or not descriptor["campaign_manifest"]
                or descriptor.get("expected_figures") != entry.get("slots")
                or not isinstance(row_ids, list) or not row_ids
                or len(row_ids) != entry.get("rows")
                or len(set(row_ids)) != len(row_ids)
                or seen_rows.intersection(row_ids)
                or not isinstance(descriptor.get("workers"), int)
                or descriptor["workers"] < 1
                or not isinstance(entry.get("profile"), str)
                or not entry["profile"]
                or not isinstance(entry.get("request_cpus"), int)
                or entry["request_cpus"] < descriptor["workers"]
                or any(not isinstance(entry.get(key), int) or entry[key] < 1
                       for key in ("request_memory_mb", "request_disk_mb", "slots"))):
            raise ValueError("Campaign mapping and descriptor disagree")
        seen_rows.update(row_ids)
        output_roots.add(descriptor["output_root"])
        campaign_ids.add(descriptor.get("campaign_id"))
        campaign_manifests.add(descriptor.get("campaign_manifest"))
        units.append({"unit_id": unit_id, "job": job, "profile": entry["profile"],
                      "workers": descriptor["workers"], "request_cpus": entry["request_cpus"],
                      "request_memory_mb": entry.get("request_memory_mb"),
                      "request_disk_mb": entry.get("request_disk_mb")})
    if len(output_roots) != 1 or len(campaign_ids) != 1 or len(campaign_manifests) != 1:
        raise ValueError("Inconsistent campaign/output identity")
    output_root = Path(next(iter(output_roots)))
    if (not output_root.is_absolute() or output_root.name != next(iter(campaign_ids))
            or not _safe_path.fullmatch(str(output_root))):
        raise ValueError("Invalid campaign output identity")
    planned_slots = sum(entry["slots"] for entry in entries)
    if (mapping.get("planned_units") != len(units)
            or mapping.get("planned_rows") != sum(entry["rows"] for entry in entries)
            or mapping.get("planned_slots") != planned_slots):
        raise ValueError("Campaign totals disagree with descriptors")
    return units, output_root, planned_slots


def _query_monitor_job(job: str, command: str) -> Optional[dict]:
    result = subprocess.run([command, job, "-json", "-attributes",
                             ",".join(_monitor_ads)], capture_output=True, text=True,
                            check=False)
    if result.returncode:
        raise RuntimeError(f"{command} failed for {job}: {result.stderr.strip()}")
    ads = json.loads(result.stdout)
    if not isinstance(ads, list) or len(ads) > 1:
        raise ValueError(f"Ambiguous {command} response for {job}")
    if not ads:
        return None
    ad = ads[0]
    if not isinstance(ad, dict) or f"{ad.get('ClusterId')}.{ad.get('ProcId')}" != job:
        raise ValueError(f"{command} returned another scheduler identity for {job}")
    return {key: ad.get(key) for key in _monitor_ads}


def _monitor_progress(output_root: Path) -> dict:
    progress = {"png_count": 0, "receipt_v2_count": 0, "json_count": 0,
                "size_bytes": 0, "available_bytes": None, "complete_scan": True,
                "label": "informational_only_not_receipt_or_scientific_closure"}
    if output_root.exists():
        scanned = 0
        for directory, names, filenames in os.walk(output_root, followlinks=False):
            names[:] = [name for name in names if not (Path(directory) / name).is_symlink()]
            for filename in filenames:
                scanned += 1
                if scanned > 100000:
                    progress["complete_scan"] = False
                    break
                path = Path(directory) / filename
                if path.is_symlink() or not path.is_file():
                    continue
                suffix = path.suffix.lower()
                progress["png_count"] += suffix == ".png"
                progress["json_count"] += suffix == ".json"
                progress["receipt_v2_count"] += (suffix == ".json" and "receipt" in filename)
                progress["size_bytes"] += path.stat().st_size
            if not progress["complete_scan"]:
                break
    capacity_path = output_root
    while not capacity_path.exists() and capacity_path != capacity_path.parent:
        capacity_path = capacity_path.parent
    capacity = os.statvfs(capacity_path)
    progress["available_bytes"] = capacity.f_bavail * capacity.f_frsize
    return progress


def observe_monitor(diagnostics_dir: Path) -> dict:
    units, output_root, planned_slots = _monitor_units(diagnostics_dir)
    counts = dict.fromkeys(_monitor_counts, 0)
    observed = []
    attention = []
    for unit in units:
        ad = _query_monitor_job(unit["job"], "condor_q")
        source = "queue"
        if ad is None:
            ad = _query_monitor_job(unit["job"], "condor_history")
            source = "history" if ad is not None else "unresolved"
        state = _monitor_states.get(ad.get("JobStatus"), "UNKNOWN") if ad else "UNKNOWN"
        category = ("resolved_terminal" if state in {"REMOVED", "COMPLETED"}
                    else state.lower() if state.lower() in counts else "unknown")
        counts[category] += 1
        exit_code = ad.get("ExitCode") if ad else None
        exit_signal = ad.get("ExitSignal") if ad else None
        if state in {"HELD", "REMOVED", "UNKNOWN"} or (exit_code not in (None, 0)
                or exit_signal is not None or (ad and ad.get("ExitBySignal") is True)):
            attention.append(unit["unit_id"])
        observed.append({**unit, "state": state, "source": source,
                         "ads": ad, "exit_result": ("signal " + str(exit_signal)
                            if exit_signal is not None else "signal unavailable"
                            if ad and ad.get("ExitBySignal") is True else "exit " + str(exit_code)
                            if exit_code is not None else None)})
    if counts["unknown"]:
        campaign_status = "MONITOR_INCOMPLETE_OR_UNRESOLVED"
    elif counts["resolved_terminal"] == len(units):
        campaign_status = "ALL_TERMINAL_READY_FOR_COLLECTION"
    elif attention:
        campaign_status = "ACTIVE_WITH_ATTENTION_REQUIRED"
    else:
        campaign_status = "ACTIVE_NO_IMMEDIATE_ATTENTION"
    return {"observed_at": datetime.now(timezone.utc).isoformat(),
            "diagnostics_dir": str(diagnostics_dir), "output_root": str(output_root),
            "planned_slots": planned_slots, "units": observed, "counts": counts,
            "attention_units": attention, "campaign_status": campaign_status,
            "observation_success": True, "progress": _monitor_progress(output_root)}


def _print_monitor(snapshot: dict, *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(snapshot, sort_keys=True))
        return
    def show(value: object, width: int) -> str:
        value = "-" if value is None else " ".join(str(value).split())
        return value if len(value) <= width else value[:width - 1] + "~"

    counts = snapshot["counts"]
    progress = snapshot["progress"]
    print(f"PLOT CAMPAIGN  {snapshot.get('observed_at', '-')}  "
          f"{Path(snapshot['output_root']).name}  {snapshot['campaign_status']}")
    print("Units: " + "  ".join(f"{key}={counts.get(key, 0)}" for key in _monitor_counts))
    print("Attention: " + (", ".join(snapshot["attention_units"]) or "none"))
    print("Outputs (informational): "
          f"PNG={progress.get('png_count', '-')}  receipt-v2={progress.get('receipt_v2_count', '-')}  "
          f"slots={snapshot['planned_slots']}  bytes={progress.get('size_bytes', '-')}  "
          f"free={progress.get('available_bytes', '-')}  "
          f"scan={'complete' if progress.get('complete_scan', True) else 'partial'}")
    headings = ("STATE", "UNIT", "JOB", "PROFILE/W", "STARTS", "CPU", "REQ_MB",
                "MEM_MB", "RSS_KIB", "HOST/EXIT")
    widths = (12, 22, 12, 12, 6, 4, 8, 16, 10, 24)
    print(" ".join(f"{heading:<{width}}" for heading, width in zip(headings, widths)).rstrip())
    print(" ".join("-" * width for width in widths))
    for unit in snapshot["units"]:
        ad = unit["ads"] or {}
        host_or_exit = unit["exit_result"] or ad.get("RemoteHost") or ad.get("LastRemoteHost")
        values = (unit["state"], unit["unit_id"], unit["job"],
                  f"{unit['profile']}/{unit['workers']}", ad.get("NumJobStarts"),
                  ad.get("RequestCpus") or unit["request_cpus"],
                  ad.get("RequestMemory") or unit["request_memory_mb"],
                  ad.get("MemoryUsage"), ad.get("ResidentSetSize"), host_or_exit)
        print(" ".join(f"{show(value, width):<{width}}" for value, width in zip(values, widths)).rstrip())
        if unit["state"] in {"HELD", "REMOVED"}:
            print("  reason:", show(ad.get("HoldReason") or ad.get("RemoveReason"), 100))


def _selected_view(loaded: dict, report: dict, row: dict) -> tuple[dict, dict]:
    from topeft.modules.nominal_schema import family_from_component_key
    from topcoffea.modules.utils import regex_match

    families = row["variables"].split(",")
    subset = {key: histogram for key, histogram in loaded.items()
              if (family_from_component_key(key) or key) in families}
    for family in families:
        if not any((family_from_component_key(key) or key) == family
                   and not key.endswith("_sumw2") for key in subset):
            raise ValueError(f"Selected union lacks nominal family {family!r}")
    if not subset:
        raise ValueError("Selected logical view is empty")
    if row["kind"] == "njets":
        from analysis.topeft_run2 import make_cr_and_sr_plots as plotter

        patterns = plotter._sr_njets_channel_patterns()
        reduced = {}
        matched = set()
        for key, histogram in subset.items():
            if "channel" not in histogram.categorical_axes.name:
                raise ValueError(f"Selected njets component {key!r} has no channel axis")
            available = list(histogram.axes["channel"])
            channels = regex_match(available, patterns)
            matched.update(channels)
            reduced[key] = (histogram if len(channels) == len(available)
                            else histogram.prune("channel", channels))
        if not matched:
            raise ValueError("Selected njets view has no accepted physical channel")
        subset = reduced
    view_report = {**report, "runtime_histogram_families": families,
                   "selected_families": families}
    return subset, view_report


def _inventory(output_root: Path, planned: list[str]) -> tuple[list[dict], int]:
    from PIL import Image, UnidentifiedImageError

    inventory = []
    unreadable = 0
    for relative in sorted(set(planned)):
        path = output_root / relative
        if path.exists() or path.is_symlink():
            stat = path.lstat()
            invalid = path.is_symlink() or not path.is_file() or stat.st_size == 0
            if not invalid:
                try:
                    with Image.open(path) as figure:
                        if figure.format != "PNG":
                            invalid = True
                        else:
                            figure.verify()
                except (OSError, ValueError, UnidentifiedImageError):
                    invalid = True
            if invalid:
                unreadable += 1
            inventory.append({"path": str(path.relative_to(output_root)),
                              "size": stat.st_size, "valid": not invalid})
    return inventory, unreadable


def _reconcile_outputs(descriptor: dict, planned: list[str],
                       skipped: list[dict], inventory: list[dict],
                       unreadable: int) -> dict:
    """Require one valid PNG or one maintained explicit skip per planned path."""
    from analysis.topeft_run2 import make_cr_and_sr_plots as plotter

    valid_inventory = [{"path": item["path"], "size": item["size"]}
                       for item in inventory if item["valid"]]
    produced = [item["path"] for item in valid_inventory]
    skipped = sorted(skipped, key=lambda record: (record.get("row_id", ""),
                                                   tuple(record.get("paths", ()))))
    skipped_paths = []
    rows_by_id = {row["unit"]: row for row in descriptor.get("rows", [])}
    for record in skipped:
        if (record.get("outcome") != "skipped"
                or record.get("reason") not in plotter.maintained_undrawable_reasons
                or record.get("region") != "SR"
                or record.get("mode") != "aggregate"
                or not record.get("row_id")
                or not record.get("paths")):
            raise RuntimeError("Invalid maintained plot skip outcome")
        if rows_by_id:
            row = rows_by_id.get(record["row_id"])
            if (row is None or record.get("variable") not in row["variables"].split(",")
                    or any(not path.startswith(_output_relative(row) + "/")
                           for path in record["paths"])):
                raise RuntimeError("Plot skip is not bound to its logical row")
        skipped_paths.extend(record["paths"])
    planned_set = set(planned)
    produced_set = set(produced)
    skipped_set = set(skipped_paths)
    unaccounted = planned_set - produced_set - skipped_set
    unexpected_produced = produced_set - planned_set
    unexpected_skips = skipped_set - planned_set
    conflicting_outcomes = produced_set & skipped_set
    duplicate_skips = sorted(path for path, count in Counter(skipped_paths).items()
                             if count > 1)
    if (len(planned) != len(planned_set)
            or len(planned) != descriptor["expected_figures"]
            or len(produced) != len(produced_set)
            or len(skipped_paths) != len(skipped_set)
            or conflicting_outcomes
            or unaccounted or unexpected_produced or unexpected_skips
            or unreadable):
        raise RuntimeError(
            "Plot output inventory differs from the logical manifest: "
            f"planned={len(planned)}, produced={len(produced)}, "
            f"skipped={len(skipped_paths)}, unaccounted={sorted(unaccounted)}, "
            f"unexpected_produced={sorted(unexpected_produced)}, "
            f"unexpected_skips={sorted(unexpected_skips)}, "
            f"conflicting_outcomes={sorted(conflicting_outcomes)}, "
            f"duplicate_skips={duplicate_skips}, invalid_png={unreadable}"
        )
    return {
        "planned_slot_count": len(planned),
        "produced_png_count": len(produced),
        "allowed_skip_count": len(skipped_paths),
        "unaccounted_count": 0,
        "produced_inventory_identity": hashlib.sha256(
            _json_bytes(valid_inventory)).hexdigest(),
        "allowed_skip_inventory_identity": hashlib.sha256(_json_bytes(skipped)).hexdigest(),
        "allowed_skips": skipped,
    }


def run_unit(descriptor_path: Path, descriptor_sha256: str) -> dict:
    """Load the principal PKL once in the parent and render its logical views."""
    if _sha256(descriptor_path) != descriptor_sha256:
        raise ValueError("Unit descriptor integrity mismatch")
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if descriptor.get("schema") != "plot_campaign_unit_v1":
        raise ValueError("Unexpected unit descriptor schema")
    if type(descriptor.get("workers")) is not int or descriptor["workers"] < 1:
        raise ValueError("Invalid descriptor worker count")
    manifest = Path(descriptor["campaign_manifest"])
    rows, manifest_sha256 = read_campaign(manifest)
    units = {unit["unit_id"]: unit for unit in derive_units(rows, manifest_sha256)}
    campaign_principal_pkl_paths = sorted({row["pkl"] for row in rows})
    if descriptor.get("campaign_principal_pkl_paths") != campaign_principal_pkl_paths:
        raise ValueError("Descriptor campaign principal PKL path set differs from manifest")
    unit_id = descriptor["unit_id"]
    if unit_id not in units or any(descriptor.get(key) != value
                                   for key, value in units[unit_id].items()):
        raise ValueError("Descriptor differs from the current campaign partition")
    principal_pkl = descriptor["principal_pkl"]
    if principal_pkl not in campaign_principal_pkl_paths:
        raise ValueError("Unit principal PKL is outside the campaign input set")
    if not Path(principal_pkl).is_file() or not os.access(principal_pkl, os.R_OK):
        raise FileNotFoundError(f"Declared principal PKL is missing or unreadable: {principal_pkl}")
    output_root = Path(descriptor["output_root"])
    _histos_root(output_root)
    if descriptor["campaign_id"] != output_root.name:
        raise ValueError("Campaign output identity mismatch")
    # Logical rows share canonical period/merged directories. Figure creation
    # is exclusive, while receipts retain the technical unit identity.
    occupied = []
    receipt_path = canonical_receipt_path(output_root, unit_id)
    occupied.extend(str(path) for path in [receipt_path]
                    if path.exists() or path.is_symlink())
    if occupied:
        raise FileExistsError(f"Occupied campaign output: {occupied}")
    from analysis.topeft_run2 import make_cr_and_sr_plots as plotter

    patterns = (plotter._sr_njets_channel_patterns()
                if descriptor["channel_selection"] == "sr_njets_patterns" else ())
    start_time = datetime.now(timezone.utc).isoformat()
    loaded, merge_report = plotter.load_and_merge_histogram_pkls(
        [descriptor["principal_pkl"]], require_sumw2=True,
        year_coverage_policy="warn", selected_families=descriptor["families"],
        channel_patterns=patterns,
    )
    # The maintained loader performs its own separate SHA scan before this one
    # selective streaming/deserialization traversal. No view invokes it again.
    output_root.mkdir(parents=True, exist_ok=True)
    parser = plotter.build_arg_parser()
    planned = []
    skipped = []
    for row in descriptor["rows"]:
        variables = row["variables"].split(",")
        relative = _output_relative(row)
        period = normalize_plot_period(row["run"], row["view"])
        row_directory = canonical_row_directory(output_root, row["run"], row["view"])
        parent = row_directory.parent
        args = parser.parse_args([
            "-f", descriptor["principal_pkl"], "-o", str(parent),
            "-n", row_directory.name, "-y", *row["years"].split(","),
            "--sr", "--unblind", "--variables", *variables,
            "--channel-output", "merged", "--binning", "processing",
            "--workers", str(descriptor["workers"]), "--output-formats", "png",
            "--year-coverage-policy", "warn", "--suppress-merge-report",
            "--canonical-campaign-root", str(output_root),
            "--canonical-period", period,
        ])
        outcomes = {"planned": [], "skipped": []}
        result = plotter.run_with_args(
            args, parser, loaded_view=_selected_view(loaded, merge_report, row),
            outcome_collector=outcomes,
            negative_report_suffix=row["unit"],
        )
        if result != 0:
            raise RuntimeError(f"Plot view {row['unit']} returned {result}")
        if len(outcomes["planned"]) != int(row["expected_figures"]):
            raise RuntimeError(f"Plot view {row['unit']} planned inventory differs from manifest")
        planned.extend(str(Path(relative) / path) for path in outcomes["planned"])
        for record in outcomes["skipped"]:
            skipped.append({**record, "row_id": row["unit"],
                            "paths": [str(Path(relative) / path)
                                      for path in record["paths"]]})
    inventory, unreadable = _inventory(output_root, planned)
    accounting = _reconcile_outputs(descriptor, planned, skipped, inventory, unreadable)
    receipt = {
        "schema": "plot_campaign_unit_receipt_v2",
        "campaign_id": descriptor["campaign_id"],
        "unit_id": unit_id,
        "principal_pkl": descriptor["principal_pkl"],
        "historical_pkl_sha256": descriptor["historical_pkl_sha256"],
        "manifest_sha256": manifest_sha256,
        "topeft_source_identity": descriptor["source_identity"],
        "runtime_topeft_source_identity": _source_identity(),
        "worker_count": descriptor["workers"],
        "multiprocessing_start_method": multiprocessing.get_start_method(allow_none=True),
        "start_time": start_time,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "return_code": 0,
        "logical_view_count": len(descriptor["rows"]),
        "expected_output_count": descriptor["expected_figures"],
        "observed_output_count": len(inventory),
        "zero_byte_or_unreadable_output_count": unreadable,
        "inventory_sha256": hashlib.sha256(_json_bytes([
            {"path": item["path"], "size": item["size"]} for item in inventory
        ])).hexdigest(),
        **accounting,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as target:
        target.write(_json_bytes(receipt).decode("utf-8"))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--manifest", type=Path, required=True)
    prepare_parser.add_argument("--dispatch-dir", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--request-cpus", type=int, required=True)
    prepare_parser.add_argument("--request-memory", required=True)
    prepare_parser.add_argument("--request-disk", required=True)
    prepare_parser.add_argument("--max-runtime", type=int, required=True)
    prepare_parser.add_argument("--workers", type=int, required=True)
    prepare_parser.add_argument("--unit-id", help="Queue one manifest-derived principal-PKL unit")
    run_parser = subparsers.add_parser("run-unit")
    run_parser.add_argument("--descriptor", type=Path, required=True)
    run_parser.add_argument("--descriptor-sha256", required=True)
    monitor_parser = subparsers.add_parser(
        "monitor", help="One read-only campaign snapshot",
        epilog=("Refresh with shell watch(1). Output counts are informational, not "
                "receipt reconciliation or scientific acceptance. "
                "ALL_TERMINAL_READY_FOR_COLLECTION marks a separate collection boundary, "
                "not campaign success. MEM_MB is raw MemoryUsage; RSS_KIB is raw "
                "ResidentSetSize. A trailing ~ marks truncated human display text; "
                "--json preserves raw ClassAd values."))
    monitor_parser.add_argument("--diagnostics-dir", type=Path, required=True)
    monitor_parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.command == "monitor":
        _print_monitor(observe_monitor(args.diagnostics_dir), json_mode=args.json)
        return 0
    if args.command == "prepare":
        result = prepare(args.manifest, args.dispatch_dir, args.output_root,
                         request_cpus=args.request_cpus,
                         request_memory=args.request_memory,
                         request_disk=args.request_disk,
                         max_runtime=args.max_runtime, workers=args.workers,
                         unit_id=args.unit_id)
    else:
        result = run_unit(args.descriptor, args.descriptor_sha256)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
