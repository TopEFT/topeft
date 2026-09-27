"""Materialize a standard Run 2 or Run 3 datacard matrix manifest."""

import argparse
import json
import os
from pathlib import Path
import yaml

from analysis.topeft_run2 import build_per_era_datacard_package as per_era
from analysis.topeft_run2 import datacard_matrix_runner as runner


_card_prefix = "ttx_multileptons-"
_repository_root = Path(__file__).resolve().parents[2]
_registry = _repository_root / "topeft/channels/ch_lst.json"
_profile_path = Path(__file__).with_name("datacard_matrix_profiles.yml")


def _load_profile(era):
    profile = yaml.safe_load(_profile_path.read_text(encoding="utf-8"))
    if profile.get("schema") != "topeft_datacard_matrix_profiles_v1":
        raise ValueError("unsupported datacard matrix profile schema")
    years = profile["eras"][era]["years"]
    rows = profile["rows"]
    targets = [f"{channel}_{row['distribution']}" for row in rows
               for channel in row["physical_channels"]]
    available = per_era._canonical_physical_names(_registry, "ALL_CH_LST_SR")
    if len(targets) != len(set(targets)) or set(targets) != set(available):
        raise ValueError("standard matrix profile differs from ALL_CH_LST_SR")
    if len({row["row_number"] for row in rows}) != len(rows):
        raise ValueError("duplicate standard matrix row number")
    return years, rows, set(targets)


def _input_bindings(values, allowed_roles):
    bindings = {}
    for value in values:
        role, separator, path = value.partition("=")
        if not separator or not path or role not in allowed_roles or role in bindings:
            raise ValueError(f"invalid or duplicate --input-pkl binding: {value}")
        bindings[role] = path
    return bindings


def _execution_channels(profile_row, era):
    """Expand the maintained contiguous partition of one logical row."""
    channels = profile_row["physical_channels"]
    sizes = profile_row.get("execution_unit_sizes", {}).get(era)
    if sizes is None:
        return [channels]
    if (not isinstance(sizes, list) or not sizes
            or any(type(size) is not int or size <= 0 for size in sizes)
            or sum(sizes) != len(channels)):
        raise ValueError("invalid execution unit sizes for standard matrix row")
    groups = []
    start = 0
    for size in sizes:
        groups.append(channels[start:start + size])
        start += size
    return groups


def _runtime_path(value, label, *, executable=False):
    path = runner.absolute(str(value), label)
    if not path.is_file() or (executable and not os.access(path, os.X_OK)):
        raise ValueError(f"{label} must be an existing regular file: {path}")
    return path


def make_manifest(args):
    manifest_output = runner.absolute(str(args.manifest_output), "manifest_output")
    output_root = runner.absolute(str(args.output_root), "output_root")
    control_root = runner.absolute(str(args.control_root), "control_root")
    working_directory = runner.absolute(str(args.working_directory), "working_directory")
    if not working_directory.is_dir():
        raise ValueError(f"working_directory must exist: {working_directory}")
    if not manifest_output.parent.is_dir() or manifest_output.exists():
        raise ValueError(f"manifest output must have an existing parent and be absent: {manifest_output}")
    if not runner.IDENTIFIER_RE.fullmatch(args.attempt_id):
        raise ValueError("attempt_id must be a portable identifier")
    python = _runtime_path(args.python_executable, "python_executable", executable=True)
    make_cards = _runtime_path(args.make_cards_path, "make_cards_path")
    missing_parton = _runtime_path(args.missing_parton_file, "missing_parton_file")
    fingerprints = []
    seen_runtime = set()
    for value in [make_cards, *args.runtime_file]:
        path = _runtime_path(value, "runtime_file")
        if path in seen_runtime:
            continue
        seen_runtime.add(path)
        fingerprints.append({"path": str(path), "sha256": runner.hash_file(path)})
    runtime = {
        "contract_id": args.runtime_contract_id,
        "python_executable": str(python),
        "make_cards_path": str(make_cards),
        "fingerprints": fingerprints,
    }
    runner.validate_runtime(runtime)
    years, profile_rows, profile_targets = _load_profile(args.era)
    physical_names = per_era._selected_physical_names(
        _registry, args.channel_set_key, args.physical_target)
    if not set(physical_names) <= profile_targets:
        raise ValueError("selected channel set contains targets outside the standard profile")
    selected = set(physical_names)
    roles = {row["input_roles"][args.era] for row in profile_rows}
    inputs = _input_bindings(args.input_pkl, roles)
    rows = []
    for profile_row in profile_rows:
        logical_row_id = f"{args.era}_{profile_row['row_number']:02d}"
        distribution = profile_row["distribution"]
        role = profile_row["input_roles"][args.era]
        for unit_index, standard_channels in enumerate(_execution_channels(profile_row, args.era), 1):
            channels = [channel for channel in standard_channels
                        if f"{channel}_{distribution}" in selected]
            if not channels:
                continue
            row_id = f"{logical_row_id}_exec_{unit_index:02d}"
            if role not in inputs:
                raise ValueError(f"{role} PKL is required for row {row_id}")
            input_pkl = _runtime_path(inputs[role], f"{role}_pkl")
            row_output = output_root / row_id
            merge_report = control_root / "merge_reports" / f"{row_id}__{args.attempt_id}.json"
            snapshot = control_root / "snapshots" / f"{row_id}__{args.attempt_id}"
            log = control_root / "logs" / f"{row_id}__{args.attempt_id}.log"
            expected_outputs = [str(row_output / f"{_card_prefix}{channel}_{distribution}.{suffix}")
                                for channel in channels for suffix in ("txt", "root")]
            producer_args = [
                "--out-dir", str(row_output), "--var-lst", distribution,
                "--ch-lst", *channels, "--binning", "fitting",
                "--do-nuisance", "--do-mc-stat", "--skip-selected-wcs-check",
                "--year-coverage-policy", "error", "--year", *years,
                "--miss-parton-file", str(missing_parton),
                "--merge-report", str(merge_report),
            ]
            rows.append({
                "row_id": row_id, "logical_row_id": logical_row_id,
                "attempt_id": args.attempt_id, "era": args.era,
                "working_directory": str(working_directory), "input_pkl": str(input_pkl),
                "output_root": str(row_output), "distribution": distribution,
                "physical_channels": channels, "years": years,
                "missing_parton_path": str(missing_parton),
                "merge_report_path": str(merge_report), "snapshot_directory": str(snapshot),
                "log_path": str(log), "expected_output_paths": expected_outputs,
                "producer_args": producer_args,
            })
    manifest = {
        "schema": runner.MANIFEST_SCHEMA, "control_root": str(control_root),
        "lock_path": str(control_root / "runner.lock"),
        "runtime_contract": runtime, "rows": rows,
    }
    runner.validate_runtime(runtime, verify_files=False)
    for index, row in enumerate(rows):
        runner.validate_row(row, index, control_root)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era", choices=("run2", "run3"), required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--input-pkl", action="append", default=[], metavar="ROLE=PATH",
                        help="Bind each selected profile input role to a current PKL")
    parser.add_argument("--channel-set-key", default="ALL_CH_LST_SR",
                        help="Maintained channel set in topeft/channels/ch_lst.json")
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--make-cards-path", type=Path,
                        default=_repository_root / "analysis/topeft_run2/make_cards.py")
    parser.add_argument("--missing-parton-file", type=Path, required=True)
    parser.add_argument("--runtime-contract-id", required=True)
    parser.add_argument("--runtime-file", type=Path, action="append", default=[])
    parser.add_argument("--physical-target", action="append", default=[],
                        help="Advanced: repeat to filter the selected profile to exact physical targets")
    parser.add_argument("--attempt-id", default="attempt_01")
    parser.add_argument("--working-directory", type=Path, default=_repository_root)
    args = parser.parse_args(argv)
    manifest = make_manifest(args)
    with args.manifest_output.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
