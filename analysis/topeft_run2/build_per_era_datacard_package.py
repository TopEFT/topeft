"""Build one per-era datacard package from current matrix-v3 completions."""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from analysis.topeft_run2 import datacard_matrix_runner as matrix_runner
from topeft.modules import datacard_packaging


_source_record_keys = {"path", "size_bytes", "sha256"}
_unit_keys = {
    "era", "unit_id", "physical_targets", "primary_outputs",
    "selected_wcs_source", "scalings_source",
}
_provenance_keys = {
    "schema", "artifact_type", "analysis", "era", "package_root",
    "packaged_txt_count", "packaged_root_count", "selected_wcs_sha256",
    "scalings_sha256", "physical_to_chN_sha256", "source_manifest_sha256s",
    "source_unit_count", "builder_git_head", "builder_source_sha256",
    "builder_git_dirty",
}
_sha_pattern = re.compile(r"[0-9a-f]{64}\Z")
_git_head_pattern = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_card_prefix = "ttx_multileptons-"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _record(value, expected_path=None):
    _require(isinstance(value, dict) and set(value) == _source_record_keys, "invalid source record")
    path = value["path"]
    _require(isinstance(path, str) and Path(path).is_absolute(), "source path must be absolute")
    _require(expected_path is None or path == str(expected_path), "source path differs from receipt binding")
    _require(type(value["size_bytes"]) is int and value["size_bytes"] >= 0, "invalid source size")
    _require(isinstance(value["sha256"], str) and _sha_pattern.fullmatch(value["sha256"]), "invalid source SHA256")
    return value


def _verify_source(value):
    record = _record(value)
    path = Path(record["path"])
    _require(path.is_file() and path.stat().st_size == record["size_bytes"], f"source size differs: {path}")
    _require(datacard_packaging.sha256_file(path) == record["sha256"], f"source SHA256 differs: {path}")
    return path


def _physical_name(channel, distribution):
    _require(isinstance(channel, str) and isinstance(distribution, str), "invalid physical target")
    name = f"{channel}_{distribution}"
    datacard_packaging.build_per_era_mapping([name])
    return name


def _resolve_v3_manifest_units(manifest_paths, era, *, verify_source_files=True, allow_empty=False):
    """Resolve completed matrix rows and their successful row receipts."""
    _require(era in {"run2", "run3"}, "invalid requested era")
    _require(bool(manifest_paths), "at least one matrix manifest is required")
    units = []
    hashes = []
    unit_ids = set()
    covered = set()
    for manifest_path in manifest_paths:
        manifest, manifest_sha256 = matrix_runner.load_manifest(Path(manifest_path))
        _require(manifest_sha256 not in hashes, f"duplicate manifest identity: {manifest_sha256}")
        hashes.append(manifest_sha256)
        for row in manifest["rows"]:
            if row["era"] != era:
                continue
            unit_id = row["row_id"]
            receipt_path = matrix_runner.receipt_path(manifest, row)
            if not receipt_path.exists():
                continue
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"unreadable current receipt: {receipt_path}") from exc
            runtime = manifest["runtime_contract"]
            binding = {
                "schema": matrix_runner.RECEIPT_SCHEMA,
                "manifest_sha256": manifest_sha256,
                "row_id": unit_id,
                "attempt_id": row["attempt_id"],
                "resolved_argv": matrix_runner.resolved_argv(manifest, row),
                "command_return_code": 0,
                "runtime_contract_id": runtime["contract_id"],
                "python_executable": runtime["python_executable"],
                "make_cards_path": runtime["make_cards_path"],
                "runtime_fingerprints": runtime["fingerprints"],
                "runtime_contract_digest": matrix_runner.runtime_digest(runtime),
            }
            _require(isinstance(receipt, dict) and all(receipt.get(k) == v for k, v in binding.items()),
                     f"receipt is not the selected successful attempt: {unit_id}")
            _require(all(isinstance(receipt.get(k), str) and receipt[k] for k in ("start_timestamp", "end_timestamp")),
                     f"receipt timestamps missing: {unit_id}")
            names = [_physical_name(channel, row["distribution"]) for channel in row["physical_channels"]]
            _require(len(names) == len(set(names)), f"duplicate physical target in unit: {unit_id}")
            expected_outputs = {
                str(Path(row["output_root"]) / f"{_card_prefix}{name}.{suffix}")
                for name in names for suffix in ("txt", "root")
            }
            _require(set(row["expected_output_paths"]) == expected_outputs,
                     f"manifest primary output topology differs: {unit_id}")
            outputs = receipt.get("primary_outputs")
            _require(isinstance(outputs, list) and len(outputs) == len(row["expected_output_paths"]),
                     f"receipt primary outputs incomplete: {unit_id}")
            by_path = {}
            for record, expected_path in zip(outputs, row["expected_output_paths"]):
                _record(record, expected_path)
                by_path[expected_path] = record
            artifacts = receipt.get("artifacts")
            expected_artifacts = {
                "merge_report": Path(row["merge_report_path"]),
                "selected_wcs_snapshot": matrix_runner.snapshots(row)["selected_wcs"],
                "scalings_snapshot": matrix_runner.snapshots(row)["scalings"],
                "merge_report_snapshot": matrix_runner.snapshots(row)["merge_report"],
                "log": Path(row["log_path"]),
            }
            _require(isinstance(artifacts, dict) and set(artifacts) == set(expected_artifacts),
                     f"receipt control artifacts incomplete: {unit_id}")
            for key, expected_path in expected_artifacts.items():
                _record(artifacts[key], expected_path)
            if verify_source_files:
                valid, reason = matrix_runner.validate_receipt(receipt, manifest, manifest_sha256, row)
                _require(valid, f"invalid current receipt for {unit_id}: {reason}")
            _require(unit_id not in unit_ids, f"duplicate unit identity: {unit_id}")
            _require(not covered.intersection(names), f"duplicate physical target coverage: {unit_id}")
            primary_outputs = []
            for name in names:
                base = Path(row["output_root"]) / f"{_card_prefix}{name}"
                primary_outputs.append({
                    "physical_name": name,
                    "txt": by_path[f"{base}.txt"],
                    "root": by_path[f"{base}.root"],
                })
            units.append({
                "era": era,
                "unit_id": unit_id,
                "physical_targets": names,
                "primary_outputs": primary_outputs,
                "selected_wcs_source": artifacts["selected_wcs_snapshot"],
                "scalings_source": artifacts["scalings_snapshot"],
            })
            unit_ids.add(unit_id)
            covered.update(names)
    _require(bool(units) or allow_empty, f"no selected complete {era} units")
    return units, hashes


def _canonical_physical_names(channel_registry, channel_set_key):
    """Derive the accepted ALL_CH_LST_SR physical target set and sorted order."""
    registry = json.loads(Path(channel_registry).read_text(encoding="utf-8"))
    groups = registry[channel_set_key]
    _require(isinstance(groups, dict) and groups, "invalid channel registry selection")
    names = []
    for group in groups.values():
        _require(isinstance(group, dict), "invalid channel group")
        for channel_entry in group["lep_chan_lst"]:
            channel = channel_entry[0]
            for jet_entry in group["jet_lst"]:
                jet_match = re.search(r"[0-9]+", jet_entry)
                _require(jet_match is not None, "invalid registry jet")
                jet = int(jet_match.group())
                if "3l" in channel and jet == 1 and "fwd" not in channel:
                    continue
                if (("3l_onZ_1b" in channel or ("3l_onZ_2b" in channel and jet in (4, 5)))
                        and "fwd" not in channel):
                    distribution = "ptz"
                elif (("3l_onZ_2b" in channel and jet == 1)
                      or ("3l_onZ_1b" in channel and jet == 1 and "fwd" not in channel)
                      or ("offZ_2b_fwd" in channel and jet == 1)):
                    continue
                elif "high" in channel or "low" in channel:
                    distribution = "ptll"
                elif "2los" in channel:
                    distribution = "ptz"
                elif "1tau_onZ" in channel:
                    distribution = "ptz_wtau"
                elif "fwd" in channel:
                    distribution = "lt"
                else:
                    distribution = "lj0pt"
                names.append(_physical_name(f"{channel}_{jet}j", distribution))
    _require(len(names) == len(set(names)), "duplicate registry target")
    return sorted(names)


def _selected_physical_names(channel_registry, channel_set_key, selected_targets):
    available = _canonical_physical_names(channel_registry, channel_set_key)
    if not selected_targets:
        return available
    requested = set(selected_targets)
    _require(len(requested) == len(selected_targets), "duplicate requested physical target")
    _require(requested <= set(available),
             f"requested physical target is outside the channel set: {sorted(requested - set(available))}")
    return sorted(requested)


def _declared_manifest_surface(manifest_paths, era):
    declared = set()
    for path in manifest_paths:
        manifest, _ = matrix_runner.load_manifest(Path(path))
        for row in manifest["rows"]:
            if row["era"] != era:
                continue
            for channel in row["physical_channels"]:
                target = _physical_name(channel, row["distribution"])
                _require(target not in declared, f"duplicate manifest physical target: {target}")
                declared.add(target)
    _require(bool(declared), f"no selected {era} matrix targets")
    return sorted(declared)


def _target_coverage_message(manifest_paths, era, requested, units):
    observed = {name for unit in units for name in unit["physical_targets"]}
    missing = sorted(set(requested) - observed)
    extra = sorted(observed - set(requested))
    if not missing and not extra:
        return None
    lines = [f"{era} physical target set differs: missing={missing}, extra={extra}"]
    rows_by_target = {}
    for path in manifest_paths:
        manifest, _ = matrix_runner.load_manifest(Path(path))
        for row in manifest["rows"]:
            if row["era"] != era:
                continue
            for channel in row["physical_channels"]:
                target = _physical_name(channel, row["distribution"])
                rows_by_target.setdefault(target, []).append((path, row["row_id"]))
    relevant_manifests = set()
    for target in missing:
        rows = rows_by_target.get(target, [])
        if not rows:
            lines.append(f"{target}: no row in the supplied matrix manifests; supply or generate a manifest containing this target")
            continue
        for path, row_id in rows:
            lines.append(f"{target}: manifest={path}, row_id={row_id}")
            relevant_manifests.add(str(path))
    runner = "analysis/topeft_run2/run_datacard_matrix_resumable.sh"
    for path in sorted(relevant_manifests):
        quoted = shlex.quote(path)
        lines.append(f"Check row status: {runner} --status {quoted}")
        lines.append(f"Resume with the runner after reviewing status: {runner} {quoted}")
    return "\n".join(lines)


def _validate_units(units, era, physical_names):
    _require(isinstance(units, list) and units, "normalized units are empty")
    _require(len(physical_names) == len(set(physical_names)), "duplicate canonical target")
    covered = {}
    unit_ids = set()
    for unit in units:
        _require(isinstance(unit, dict) and set(unit) == _unit_keys, "invalid normalized unit shape")
        _require(unit["era"] == era, "normalized era mismatch")
        _require(isinstance(unit["unit_id"], str) and unit["unit_id"] not in unit_ids, "duplicate normalized unit")
        unit_ids.add(unit["unit_id"])
        targets = unit["physical_targets"]
        outputs = unit["primary_outputs"]
        _require(isinstance(targets, list) and isinstance(outputs, list) and len(targets) == len(outputs),
                 "incomplete normalized primary outputs")
        for name, output in zip(targets, outputs):
            _require(isinstance(output, dict) and set(output) == {"physical_name", "txt", "root"},
                     "invalid normalized primary output")
            _require(name == output["physical_name"] and name not in covered, "duplicate or mismatched target")
            for suffix in ("txt", "root"):
                record = _record(output[suffix])
                _require(Path(record["path"]).name == f"{_card_prefix}{name}.{suffix}",
                         "source basename differs from physical target")
            covered[name] = output
        _record(unit["selected_wcs_source"])
        _record(unit["scalings_source"])
    _require(set(covered) == set(physical_names),
             f"normalized target set differs: missing={sorted(set(physical_names)-set(covered))}, extra={sorted(set(covered)-set(physical_names))}")
    return covered


def _builder_identity():
    source = Path(__file__).resolve()
    repository = source.parents[2]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, check=True,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(source)], cwd=repository,
                           check=True, capture_output=True, text=True).stdout.strip()
    return head, datacard_packaging.sha256_file(source), bool(dirty)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def build_per_era_package_from_units(era, analysis, physical_names, units, output,
                                     source_manifest_sha256s):
    """Build from normalized, already-authorized successful units."""
    _require(era in {"run2", "run3"} and isinstance(analysis, str) and analysis.strip(),
             "invalid package identity")
    output = Path(output)
    _require(output.is_absolute() and output.name not in {"", ".", ".."}, "output must be absolute")
    staging = output.parent / f".{output.name}.staging"
    _require(not output.exists() and not output.is_symlink(), "final output already exists")
    _require(not staging.exists() and not staging.is_symlink(), "private staging already exists")
    _require(isinstance(source_manifest_sha256s, list)
             and len(source_manifest_sha256s) == len(set(source_manifest_sha256s))
             and all(isinstance(value, str) and _sha_pattern.fullmatch(value) for value in source_manifest_sha256s),
             "invalid manifest SHA256 identities")
    covered = _validate_units(units, era, physical_names)
    identity = _builder_identity()
    _require(isinstance(identity, tuple) and len(identity) == 3, "invalid builder identity")
    _require(isinstance(identity[0], str) and _git_head_pattern.fullmatch(identity[0])
             and isinstance(identity[1], str) and _sha_pattern.fullmatch(identity[1])
             and type(identity[2]) is bool, "unobservable builder identity")
    selected_sources = []
    scaling_sources = []
    for unit in units:
        selected_sources.append(json.loads(_verify_source(unit["selected_wcs_source"]).read_text(encoding="utf-8")))
        scaling_sources.append(json.loads(_verify_source(unit["scalings_source"]).read_text(encoding="utf-8")))
    for output_pair in covered.values():
        for suffix in ("txt", "root"):
            _verify_source(output_pair[suffix])
    staging.mkdir()
    cards = staging / "cards"
    cards.mkdir()
    for name in physical_names:
        output_pair = covered[name]
        for suffix in ("txt", "root"):
            record = output_pair[suffix]
            destination = cards / Path(record["path"]).name
            shutil.copyfile(record["path"], destination)
            _require(destination.stat().st_size == record["size_bytes"]
                     and datacard_packaging.sha256_file(destination) == record["sha256"],
                     f"copied payload differs: {destination}")
    mapping = datacard_packaging.build_per_era_mapping(physical_names)
    selected_wcs = datacard_packaging.consolidate_selected_wcs(selected_sources)
    scalings = datacard_packaging.consolidate_scaling_records(scaling_sources, mapping)
    _write_json(staging / "physical_to_chN.json", mapping)
    _write_json(staging / "selectedWCs.txt", selected_wcs)
    _write_json(staging / "scalings.json", scalings)
    observed_mapping = json.loads((staging / "physical_to_chN.json").read_text(encoding="utf-8"))
    observed_wcs = json.loads((staging / "selectedWCs.txt").read_text(encoding="utf-8"))
    observed_scalings = json.loads((staging / "scalings.json").read_text(encoding="utf-8"))
    datacard_packaging.verify_per_era_mapping(observed_mapping, physical_names)
    datacard_packaging.verify_per_era_selected_wcs(observed_wcs, selected_sources)
    datacard_packaging.verify_per_era_scalings(observed_scalings, scaling_sources, observed_mapping)
    provenance = {
        "schema": "TOP26006_v1", "artifact_type": "package_provenance",
        "analysis": analysis, "era": era, "package_root": str(output),
        "packaged_txt_count": len(physical_names), "packaged_root_count": len(physical_names),
        "selected_wcs_sha256": datacard_packaging.sha256_file(staging / "selectedWCs.txt"),
        "scalings_sha256": datacard_packaging.sha256_file(staging / "scalings.json"),
        "physical_to_chN_sha256": datacard_packaging.sha256_file(staging / "physical_to_chN.json"),
        "source_manifest_sha256s": source_manifest_sha256s, "source_unit_count": len(units),
        "builder_git_head": identity[0], "builder_source_sha256": identity[1],
        "builder_git_dirty": identity[2],
    }
    _require(set(provenance) == _provenance_keys, "provenance key contract differs")
    _write_json(staging / "package-provenance.json", provenance)
    _require(set(path.name for path in staging.iterdir()) == {
        "cards", "selectedWCs.txt", "scalings.json", "physical_to_chN.json", "package-provenance.json",
    }, "staged package inventory differs")
    _require(set(path.name for path in cards.iterdir()) == {
        f"{_card_prefix}{name}.{suffix}" for name in physical_names for suffix in ("txt", "root")
    }, "staged cards inventory differs")
    for name in physical_names:
        for suffix in ("txt", "root"):
            record = covered[name][suffix]
            destination = cards / Path(record["path"]).name
            _require(destination.stat().st_size == record["size_bytes"]
                     and datacard_packaging.sha256_file(destination) == record["sha256"],
                     "staged payload identity differs")
    _require(json.loads((staging / "package-provenance.json").read_text(encoding="utf-8")) == provenance,
             "staged provenance differs")
    _require(not output.exists() and not output.is_symlink(), "final output appeared before publication")
    staging.rename(output)
    _require(output.is_dir() and not staging.exists(), "publication readback differs")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--era", choices=("run2", "run3"), required=True)
    build.add_argument("--matrix-manifest", type=Path, action="append", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--analysis", required=True)
    args = parser.parse_args(argv)
    units, manifest_hashes = _resolve_v3_manifest_units(args.matrix_manifest, args.era, allow_empty=True)
    physical_names = _declared_manifest_surface(args.matrix_manifest, args.era)
    coverage_message = _target_coverage_message(args.matrix_manifest, args.era, physical_names, units)
    _require(coverage_message is None, coverage_message)
    build_per_era_package_from_units(args.era, args.analysis, physical_names, units,
                                     args.output, manifest_hashes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
