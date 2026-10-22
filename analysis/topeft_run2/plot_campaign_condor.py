"""Prepare shared-filesystem Condor material and run one SR plotting PKL unit.

The accepted logical TSV owns plot coverage. This module derives execution
units from it; descriptors are checked against the TSV again at execution.
"""

from __future__ import annotations

import argparse
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
# on sys.path. Bind imports to the same checkout whose source identity we check.
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
    return "/".join((row["run"], row["view"], row["owner"],
                     row["variables"].replace(",", "_")))


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
    seen_outputs = set()
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
        relative = _output_relative(row)
        if row["unit"] in seen_ids or relative in seen_outputs:
            raise ValueError("Duplicate logical row or output directory")
        seen_ids.add(row["unit"])
        seen_outputs.add(relative)
    return rows, _sha256(manifest_path)


def derive_units(rows: list[dict], manifest_sha256: str) -> list[dict]:
    """Partition each logical row exactly once by principal PKL identity."""
    groups = {}
    for row in rows:
        groups.setdefault(row["pkl"], []).append(row)
    units = []
    for pkl_path, group in sorted(groups.items()):
        group = sorted(group, key=lambda row: row["unit"])
        if len({(row["sidecar"], row["historical_pkl_sha256"],
                 row["historical_sidecar_sha256"]) for row in group}) != 1:
            raise ValueError(f"Conflicting identities for {pkl_path}")
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
    head = subprocess.run(
        ["git", "-C", str(_repo_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return {
        "head": head,
        "frontend_sha256": _sha256(Path(__file__)),
        "plotter_sha256": _sha256(_repo_root / "analysis/topeft_run2/make_cr_and_sr_plots.py"),
    }


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
            max_runtime: int, workers: int) -> dict:
    rows, manifest_sha256 = read_campaign(manifest_path)
    units = derive_units(rows, manifest_sha256)
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
    items = []
    for unit in units:
        descriptor = {
            "schema": "plot_campaign_unit_v1",
            "campaign_id": output_root.name,
            "campaign_manifest": str(manifest_path),
            "output_root": output_path,
            "source_identity": source_identity,
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
            "units": len(units), "expected_figures": sum(unit["expected_figures"] for unit in units),
            "manifest_sha256": manifest_sha256, "submit_file": str(submit),
            "submitted": False}


def _selected_view(loaded: dict, report: dict, row: dict) -> tuple[dict, dict]:
    from topeft.modules.nominal_schema import family_from_component_key

    families = row["variables"].split(",")
    subset = {key: histogram for key, histogram in loaded.items()
              if (family_from_component_key(key) or key) in families}
    for family in families:
        if not any((family_from_component_key(key) or key) == family
                   and not key.endswith("_sumw2") for key in subset):
            raise ValueError(f"Selected union lacks nominal family {family!r}")
    if not subset:
        raise ValueError("Selected logical view is empty")
    view_report = {**report, "runtime_histogram_families": families,
                   "selected_families": families}
    return subset, view_report


def _inventory(output_root: Path, outputs: list[str]) -> tuple[list[dict], int]:
    inventory = []
    unreadable = 0
    for relative in outputs:
        for path in sorted((output_root / relative).rglob("*.png")):
            stat = path.stat()
            if stat.st_size == 0 or not os.access(path, os.R_OK):
                unreadable += 1
            inventory.append({"path": str(path.relative_to(output_root)), "size": stat.st_size})
    return inventory, unreadable


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
    unit_id = descriptor["unit_id"]
    if unit_id not in units or any(descriptor.get(key) != value
                                   for key, value in units[unit_id].items()):
        raise ValueError("Descriptor differs from the current campaign partition")
    if descriptor["source_identity"] != _source_identity():
        raise ValueError("TOPEFT plotting source differs from prepared identity")
    output_root = Path(descriptor["output_root"])
    _histos_root(output_root)
    if descriptor["campaign_id"] != output_root.name:
        raise ValueError("Campaign output identity mismatch")
    occupied = [str(output_root / relative) for relative in descriptor["outputs"]
                if (output_root / relative).exists() or (output_root / relative).is_symlink()]
    receipt_path = output_root / (unit_id + ".receipt.json")
    report_paths = [output_root / (row_id + ".composition.json")
                    for row_id in descriptor["row_ids"]]
    occupied.extend(str(path) for path in [receipt_path, *report_paths]
                    if path.exists() or path.is_symlink())
    if occupied:
        raise FileExistsError(f"Occupied campaign output: {occupied}")
    sidecar = Path(descriptor["sidecar"])
    if _sha256(sidecar) != descriptor["historical_sidecar_sha256"]:
        raise ValueError("Principal sidecar differs from accepted manifest identity")
    sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
    if sidecar_data["artifact"]["pkl_sha256"] != descriptor["historical_pkl_sha256"]:
        raise ValueError("Principal PKL identity differs from accepted manifest")

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
    for row in descriptor["rows"]:
        variables = row["variables"].split(",")
        relative = _output_relative(row)
        parent = output_root / Path(relative).parent
        args = parser.parse_args([
            "-f", descriptor["principal_pkl"], "-o", str(parent),
            "-n", Path(relative).name, "-y", *row["years"].split(","),
            "--sr", "--unblind", "--variables", *variables,
            "--channel-output", "merged", "--binning", "processing",
            "--workers", str(descriptor["workers"]), "--output-formats", "png",
            "--year-coverage-policy", "warn", "--merge-report",
            str(output_root / (row["unit"] + ".composition.json")),
        ])
        result = plotter.run_with_args(args, parser,
                                       loaded_view=_selected_view(loaded, merge_report, row))
        if result != 0:
            raise RuntimeError(f"Plot view {row['unit']} returned {result}")
    inventory, unreadable = _inventory(output_root, descriptor["outputs"])
    if len(inventory) != descriptor["expected_figures"] or unreadable:
        raise RuntimeError("Plot output inventory differs from the logical manifest")
    receipt = {
        "schema": "plot_campaign_unit_receipt_v1",
        "campaign_id": descriptor["campaign_id"],
        "unit_id": unit_id,
        "principal_pkl": descriptor["principal_pkl"],
        "historical_pkl_sha256": descriptor["historical_pkl_sha256"],
        "manifest_sha256": manifest_sha256,
        "topeft_source_identity": descriptor["source_identity"],
        "worker_count": descriptor["workers"],
        "multiprocessing_start_method": multiprocessing.get_start_method(allow_none=True),
        "start_time": start_time,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "return_code": 0,
        "logical_view_count": len(descriptor["rows"]),
        "expected_output_count": descriptor["expected_figures"],
        "observed_output_count": len(inventory),
        "zero_byte_or_unreadable_output_count": unreadable,
        "inventory_sha256": hashlib.sha256(_json_bytes(inventory)).hexdigest(),
    }
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
    run_parser = subparsers.add_parser("run-unit")
    run_parser.add_argument("--descriptor", type=Path, required=True)
    run_parser.add_argument("--descriptor-sha256", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.manifest, args.dispatch_dir, args.output_root,
                         request_cpus=args.request_cpus,
                         request_memory=args.request_memory,
                         request_disk=args.request_disk,
                         max_runtime=args.max_runtime, workers=args.workers)
    else:
        result = run_unit(args.descriptor, args.descriptor_sha256)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
