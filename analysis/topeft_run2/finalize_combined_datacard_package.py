#!/usr/bin/env python3
"""Sanitize and independently certify a combined datacard package.

``sanitize`` replaces only package-facing metadata with a consumer-safe
projection. ``certify`` is read-only with respect to the package and checks the
packaged cards, templates, ordering, scalings, and consumer metadata against
an internal build manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any


PACKAGE_SCHEMA = "TOP26006_v1"
MANIFEST_ARTIFACT_TYPE = "combined_mapping_manifest"
PROVENANCE_ARTIFACT_TYPE = "package_provenance"
METADATA_NAMES = (
    "combined_mapping_manifest.json",
    "package-provenance.json",
    "README.md",
)
CONSUMER_ROW_FIELDS = (
    "era",
    "physical_name",
    "per_era_chN",
    "combined_chN",
    "combined_order_index",
    "destination_txt_name",
    "destination_root_name",
)
TEXT_EXTENSIONS = {".json", ".md", ".txt"}
DEFAULT_FORBID_PATTERNS = (
    r"prompt_id",
    r"short_title",
    r"/reports/diagnostics/",
    r"t0_datacards_",
)
SHAPES_LINE_PATTERN = re.compile(r"^([ \t]*shapes[ \t]+\S+[ \t]+\S+[ \t]+)(\S+)")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class FinalizationError(ValueError):
    """Raised when a package fails a bounded finalization contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"cannot read JSON {path}: {exc}") from exc


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    Path(path).write_bytes(json_bytes(value))


def require_new_diagnostics_dir(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FinalizationError(f"diagnostics directory already exists: {path}")
    path.mkdir(parents=True)
    return path


def package_files(package_root: Path) -> dict[str, dict[str, Any]]:
    package_root = Path(package_root)
    if not package_root.is_dir():
        raise FinalizationError(f"package root is not a directory: {package_root}")
    entries = sorted(package_root.iterdir(), key=lambda path: path.name)
    if any(not entry.is_file() for entry in entries):
        raise FinalizationError("package root must contain files only")
    return {
        entry.name: {"sha256": sha256(entry), "size_bytes": entry.stat().st_size}
        for entry in entries
    }


def require_metadata_files(files: dict[str, dict[str, Any]]) -> None:
    missing = sorted(set(METADATA_NAMES) - set(files))
    if missing:
        raise FinalizationError(f"package metadata is missing: {missing}")


def validate_build_manifest(manifest: Any) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict):
        raise FinalizationError("build manifest must be an object")
    if manifest.get("schema") != PACKAGE_SCHEMA:
        raise FinalizationError("unsupported build manifest schema")
    if manifest.get("artifact_type") != MANIFEST_ARTIFACT_TYPE:
        raise FinalizationError("unsupported build manifest artifact type")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise FinalizationError("build manifest must contain nonempty rows")
    required = set(CONSUMER_ROW_FIELDS) | {"source_txt_path", "source_root_path"}
    seen_order, seen_channel, seen_destinations = set(), set(), set()
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise FinalizationError("build manifest row is incomplete")
        index = row["combined_order_index"]
        if not isinstance(index, int) or isinstance(index, bool) or index < 1:
            raise FinalizationError("combined order index must be a positive integer")
        if index in seen_order or row["combined_chN"] in seen_channel:
            raise FinalizationError("build manifest has duplicate channel identity")
        destinations = (row["destination_txt_name"], row["destination_root_name"])
        if destinations[0] in seen_destinations or destinations[1] in seen_destinations:
            raise FinalizationError("build manifest has duplicate destination name")
        seen_order.add(index)
        seen_channel.add(row["combined_chN"])
        seen_destinations.update(destinations)
    ordered = sorted(rows, key=lambda row: row["combined_order_index"])
    if [row["combined_order_index"] for row in ordered] != list(range(1, len(ordered) + 1)):
        raise FinalizationError("combined order indexes are incomplete")
    for era in ("run2", "run3"):
        source = manifest.get("source_scalings", {}).get(era)
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise FinalizationError(f"build manifest lacks {era} source scalings")
    return ordered


def consumer_manifest(build_manifest: dict[str, Any], package_root: Path) -> dict[str, Any]:
    rows = validate_build_manifest(build_manifest)
    return {
        "schema": PACKAGE_SCHEMA,
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "package_root": str(Path(package_root)),
        "rows": [{field: row[field] for field in CONSUMER_ROW_FIELDS} for row in rows],
    }


def require_hash_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise FinalizationError(f"{label} must be a SHA-256 string")
    return value


def source_hashes(build_manifest: dict[str, Any], key: str) -> dict[str, str]:
    result = {}
    for era in ("run2", "run3"):
        source = build_manifest.get(key, {}).get(era)
        if not isinstance(source, dict):
            raise FinalizationError(f"build manifest lacks {key} for {era}")
        result[era] = require_hash_value(source.get("sha256"), f"{key} {era} hash")
    return result


def consumer_provenance(
    build_manifest: dict[str, Any],
    package_root: Path,
    before_files: dict[str, dict[str, Any]],
    analysis: str,
    package_version: str,
    package_date: str,
    assembler_commit: str,
    original_provenance: dict[str, Any],
) -> dict[str, Any]:
    if not all(isinstance(value, str) and value for value in (analysis, package_version, package_date, assembler_commit)):
        raise FinalizationError("analysis, package version/date, and assembler commit must be nonempty")
    if not isinstance(original_provenance, dict):
        raise FinalizationError("original package provenance must be an object")
    assembler_source_sha256 = require_hash_value(
        original_provenance.get("assembler_source_sha256"), "assembler source hash"
    )
    manifest = consumer_manifest(build_manifest, package_root)
    rows = manifest["rows"]
    return {
        "schema": PACKAGE_SCHEMA,
        "artifact_type": PROVENANCE_ARTIFACT_TYPE,
        "analysis": analysis,
        "package_version": package_version,
        "package_date": package_date,
        "package_root": str(Path(package_root)),
        "assembler_commit": assembler_commit,
        "assembler_source_sha256": assembler_source_sha256,
        "manifest_sha256": hashlib.sha256(json_bytes(manifest)).hexdigest(),
        "ordered_card_inputs_sha256": before_files["ordered_card_inputs.txt"]["sha256"],
        "scalings_sha256": before_files["scalings.json"]["sha256"],
        "source_mapping_sha256": source_hashes(build_manifest, "source_mappings"),
        "source_scalings_sha256": source_hashes(build_manifest, "source_scalings"),
        "packaged_txt_count": len(rows),
        "packaged_root_count": len(rows),
    }


def consumer_readme(package_root: Path, analysis: str) -> bytes:
    root = Path(package_root)
    return (
        f"# {analysis} combined Run2+Run3 datacard package\n\n"
        f"Package path: `{root}`\n\n"
        "Contents: combined TXT cards, ROOT templates, `scalings.json`, and "
        "`ordered_card_inputs.txt`.\n\n"
        "Build the combined card from this package in the intended consumer environment:\n\n"
        "```bash\n"
        f"cd {root}\n"
        "mapfile -t cards < ordered_card_inputs.txt\n"
        'combineCards.py "${cards[@]}" > combinedcard.txt\n'
        "```\n\n"
        "Do not use a wildcard/glob directly to build the combined card for this package. "
        "The order in `ordered_card_inputs.txt` is tied to the combined channel mapping "
        "used by `scalings.json`.\n\n"
        "This package does not create or include `combinedcard.txt`, a Combine workspace, "
        "or fit outputs. The WC-population / selected-WC input used for workspace "
        "construction is handled separately in the analysis fit configuration.\n"
    ).encode("utf-8")


def atomic_replace(package_root: Path, name: str, content: bytes) -> None:
    destination = Path(package_root) / name
    sibling = destination.parent / f".{name}.finalize-{uuid.uuid4().hex}"
    if not destination.is_file() or sibling.exists():
        raise FinalizationError(f"metadata destination or temporary sibling invalid: {name}")
    try:
        with sibling.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if sibling.read_bytes() != content:
            raise FinalizationError(f"temporary metadata readback mismatch: {name}")
        os.replace(sibling, destination)
        if destination.read_bytes() != content:
            raise FinalizationError(f"metadata replacement readback mismatch: {name}")
    finally:
        if sibling.exists():
            sibling.unlink()


def build_inventory(package_root: Path, diagnostics_dir: Path) -> dict[str, Any]:
    before_files = package_files(package_root)
    require_metadata_files(before_files)
    backup_dir = diagnostics_dir / "internal_pre_sanitization_metadata"
    backup_dir.mkdir()
    for name in METADATA_NAMES:
        (backup_dir / name).write_bytes((package_root / name).read_bytes())
    non_metadata_names = sorted(set(before_files) - set(METADATA_NAMES))
    inventory = {
        "schema": PACKAGE_SCHEMA,
        "artifact_type": "package_before_after_inventory",
        "package_root": str(package_root),
        "authorized_metadata_names": list(METADATA_NAMES),
        "before_file_count": len(before_files),
        "before_files": before_files,
        "frozen_non_metadata_names": non_metadata_names,
        "pre_sanitization_metadata_sha256": {name: before_files[name]["sha256"] for name in METADATA_NAMES},
        "pre_sanitization_metadata_backup_directory": str(backup_dir),
    }
    write_json(diagnostics_dir / "package_before_after_inventory.json", inventory)
    return inventory


def sanitize(args: argparse.Namespace) -> dict[str, Any]:
    package_root = Path(args.package_root)
    diagnostics_dir = require_new_diagnostics_dir(Path(args.diagnostics_dir))
    inventory = build_inventory(package_root, diagnostics_dir)
    backup_dir = diagnostics_dir / "internal_pre_sanitization_metadata"
    build_manifest = read_json(backup_dir / "combined_mapping_manifest.json")
    original_provenance = read_json(backup_dir / "package-provenance.json")
    manifest = consumer_manifest(build_manifest, package_root)
    provenance = consumer_provenance(
        build_manifest, package_root, inventory["before_files"], args.analysis,
        args.package_version, args.package_date, args.assembler_commit, original_provenance,
    )
    candidates = {
        "combined_mapping_manifest.json": json_bytes(manifest),
        "package-provenance.json": json_bytes(provenance),
        "README.md": consumer_readme(package_root, args.analysis),
    }
    for name in METADATA_NAMES:
        atomic_replace(package_root, name, candidates[name])
    after_files = package_files(package_root)
    changed = sorted(name for name in before_and_after_names(inventory["before_files"], after_files) if inventory["before_files"][name]["sha256"] != after_files[name]["sha256"])
    frozen_ok = (
        set(after_files) == set(inventory["before_files"])
        and all(after_files[name]["sha256"] == inventory["before_files"][name]["sha256"] for name in inventory["frozen_non_metadata_names"])
    )
    inventory.update({
        "after_file_count": len(after_files),
        "after_files": after_files,
        "changed_file_names": changed,
        "only_authorized_metadata_changed": set(changed) == set(METADATA_NAMES),
        "non_metadata_hashes_unchanged": frozen_ok,
        "post_sanitization_metadata_sha256": {name: after_files[name]["sha256"] for name in METADATA_NAMES},
        "temporary_siblings": sorted(path.name for path in package_root.iterdir() if ".finalize-" in path.name),
    })
    write_json(diagnostics_dir / "package_before_after_inventory.json", inventory)
    if not inventory["only_authorized_metadata_changed"] or not frozen_ok or inventory["temporary_siblings"]:
        raise FinalizationError("metadata sanitization changed an unauthorized package surface")
    return {"package_root": str(package_root), "changed_metadata": list(METADATA_NAMES)}


def before_and_after_names(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    if set(before) != set(after):
        raise FinalizationError("package file set changed during sanitization")
    return set(before)


def expected_card_bytes(source_bytes: bytes, source_root_name: str, destination_root_name: str) -> bytes:
    try:
        text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FinalizationError("source card is not UTF-8") from exc
    output, shape_count = [], 0
    for line in text.splitlines(keepends=True):
        if re.match(r"^[ \t]*shapes(?:[ \t]|$)", line):
            match = SHAPES_LINE_PATTERN.match(line)
            if match is None or match.group(2) != source_root_name:
                raise FinalizationError("unexpected source shapes reference")
            line = line[:match.start(2)] + destination_root_name + line[match.end(2):]
            shape_count += 1
        output.append(line)
    if shape_count == 0:
        raise FinalizationError("source card has no shapes reference")
    return "".join(output).encode("utf-8")


def compiled_forbidden_pattern(tokens: list[str], regexes: list[str]) -> re.Pattern[str]:
    parts = [*DEFAULT_FORBID_PATTERNS, *(re.escape(token) for token in tokens), *regexes]
    try:
        return re.compile("|".join(f"(?:{part})" for part in parts), re.IGNORECASE)
    except re.error as exc:
        raise FinalizationError(f"invalid forbidden regex: {exc}") from exc


def scan_internal_references(package_root: Path, diagnostics_dir: Path, tokens: list[str], regexes: list[str]) -> dict[str, Any]:
    pattern = compiled_forbidden_pattern(tokens, regexes)
    files_scanned, top_level_names, matches = [], [], []
    for path in sorted(package_root.iterdir(), key=lambda item: item.name):
        top_level_names.append(path.name)
        for match in pattern.finditer(path.name):
            matches.append({"file": path.name, "surface": "filename", "matched_text": match.group()})
        if path.suffix not in TEXT_EXTENSIONS:
            continue
        files_scanned.append(path.name)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise FinalizationError(f"consumer text file is not UTF-8: {path.name}") from exc
        for line_number, line in enumerate(text.splitlines(), 1):
            for match in pattern.finditer(line):
                matches.append({"file": path.name, "line": line_number, "matched_text": match.group()})
    result = {
        "schema": PACKAGE_SCHEMA,
        "artifact_type": "internal_reference_scan",
        "method": "UTF-8 regex scan of JSON, Markdown, and text package files plus top-level names; ROOT binaries are excluded",
        "default_patterns": list(DEFAULT_FORBID_PATTERNS),
        "forbid_tokens": tokens,
        "forbid_regexes": regexes,
        "files_scanned": files_scanned,
        "top_level_names_scanned": top_level_names,
        "matches": matches,
        "passed": not matches,
    }
    write_json(diagnostics_dir / "internal_reference_scan.json", result)
    return result


def validate_consumer_metadata(package_root: Path, build_manifest: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[bool, bool, bool]:
    manifest = read_json(package_root / "combined_mapping_manifest.json")
    provenance = read_json(package_root / "package-provenance.json")
    readme = (package_root / "README.md").read_text(encoding="utf-8")
    expected_manifest = consumer_manifest(build_manifest, package_root)
    manifest_ok = manifest == expected_manifest
    required_provenance = {
        "schema", "artifact_type", "analysis", "package_version", "package_date", "package_root",
        "assembler_commit", "assembler_source_sha256", "manifest_sha256", "ordered_card_inputs_sha256",
        "scalings_sha256", "source_mapping_sha256", "source_scalings_sha256", "packaged_txt_count", "packaged_root_count",
    }
    provenance_ok = (
        set(provenance) == required_provenance
        and provenance.get("schema") == PACKAGE_SCHEMA
        and provenance.get("artifact_type") == PROVENANCE_ARTIFACT_TYPE
        and provenance.get("package_root") == str(package_root)
        and isinstance(provenance.get("analysis"), str) and bool(provenance["analysis"])
        and isinstance(provenance.get("package_version"), str) and bool(provenance["package_version"])
        and isinstance(provenance.get("package_date"), str) and bool(provenance["package_date"])
        and isinstance(provenance.get("assembler_commit"), str) and bool(provenance["assembler_commit"])
        and SHA256_PATTERN.fullmatch(str(provenance.get("assembler_source_sha256", ""))) is not None
        and provenance.get("manifest_sha256") == sha256(package_root / "combined_mapping_manifest.json")
        and provenance.get("ordered_card_inputs_sha256") == sha256(package_root / "ordered_card_inputs.txt")
        and provenance.get("scalings_sha256") == sha256(package_root / "scalings.json")
        and provenance.get("source_mapping_sha256") == source_hashes(build_manifest, "source_mappings")
        and provenance.get("source_scalings_sha256") == source_hashes(build_manifest, "source_scalings")
        and provenance.get("packaged_txt_count") == len(rows)
        and provenance.get("packaged_root_count") == len(rows)
    )
    command = f"cd {package_root}\nmapfile -t cards < ordered_card_inputs.txt\ncombineCards.py \"${{cards[@]}}\" > combinedcard.txt"
    readme_ok = command in readme and "Do not use a wildcard/glob directly" in readme
    return manifest_ok, provenance_ok, readme_ok


def validate_scalings(build_manifest: dict[str, Any], rows: list[dict[str, Any]], package_root: Path) -> dict[str, Any]:
    channel_map = {(row["era"], row["per_era_chN"]): row["combined_chN"] for row in rows}
    expected, source_duplicates, expected_duplicates = {}, [], []
    for era in ("run2", "run3"):
        source_spec = build_manifest["source_scalings"][era]
        source_path = Path(source_spec["path"])
        if not source_path.is_file() or sha256(source_path) != source_spec.get("sha256"):
            raise FinalizationError(f"{era} source scalings do not match the build manifest")
        records = read_json(source_path)
        if not isinstance(records, list):
            raise FinalizationError(f"{era} source scalings are not a JSON list")
        for record in records:
            if not isinstance(record, dict) or not {"channel", "process", "parameters", "scaling"}.issubset(record):
                raise FinalizationError("source scaling record is incomplete")
            source_key = (era, record["channel"], record["process"])
            if source_key in expected or (era, record["channel"]) not in channel_map:
                source_duplicates.append(source_key)
                continue
            transformed = dict(record)
            transformed["channel"] = channel_map[(era, record["channel"])]
            combined_key = (transformed["channel"], transformed["process"])
            if combined_key in expected:
                expected_duplicates.append(combined_key)
                continue
            expected[combined_key] = transformed
    observed, observed_duplicates = {}, []
    records = read_json(package_root / "scalings.json")
    if not isinstance(records, list):
        raise FinalizationError("packaged scalings are not a JSON list")
    for record in records:
        if not isinstance(record, dict) or "channel" not in record or "process" not in record:
            raise FinalizationError("packaged scaling record is incomplete")
        key = (record["channel"], record["process"])
        if key in observed:
            observed_duplicates.append(key)
            continue
        observed[key] = record
    return {
        "valid": not source_duplicates and not expected_duplicates and not observed_duplicates
        and set(expected) == set(observed) and all(expected[key] == observed[key] for key in expected),
        "source_duplicate_keys": source_duplicates,
        "expected_duplicate_keys": expected_duplicates,
        "observed_duplicate_keys": observed_duplicates,
        "missing_keys": sorted(set(expected) - set(observed)),
        "extra_keys": sorted(set(observed) - set(expected)),
        "payload_mismatch_keys": sorted(key for key in set(expected) & set(observed) if expected[key] != observed[key]),
        "expected_record_count": len(expected),
        "observed_record_count": len(observed),
    }


def certify(args: argparse.Namespace) -> dict[str, Any]:
    package_root = Path(args.package_root)
    diagnostics_dir = require_new_diagnostics_dir(Path(args.diagnostics_dir))
    build_manifest = read_json(Path(args.build_manifest))
    rows = validate_build_manifest(build_manifest)
    entries = package_files(package_root)
    require_metadata_files(entries)
    manifest_ok, provenance_ok, readme_ok = validate_consumer_metadata(package_root, build_manifest, rows)
    ordered_lines = (package_root / "ordered_card_inputs.txt").read_text(encoding="utf-8").splitlines()
    expected_order = [row["destination_txt_name"] for row in rows]
    expected_txt = set(expected_order)
    expected_root = {row["destination_root_name"] for row in rows}
    observed_txt = {name for name in entries if name.endswith(".txt") and name != "ordered_card_inputs.txt"}
    observed_root = {name for name in entries if name.endswith(".root")}
    ordering_valid = (
        ordered_lines == expected_order and len(ordered_lines) == len(set(ordered_lines))
        and all(not Path(line).is_absolute() and Path(line).parent == Path(".") and (package_root / line).is_file() for line in ordered_lines)
    )
    csv_path = diagnostics_dir / "package_file_certification.csv"
    fields = (
        "combined_order_index", "era", "physical_name", "per_era_chN", "combined_chN",
        "destination_txt_name", "destination_txt_sha256", "destination_root_name", "destination_root_sha256",
        "source_txt_sha256", "txt_transform_status", "source_root_sha256", "root_hash_match",
        "ordered_input_line", "ordered_input_match",
    )
    txt_failures, root_failures, order_failures = [], [], []
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            index = row["combined_order_index"]
            source_txt, source_root = Path(row["source_txt_path"]), Path(row["source_root_path"])
            destination_txt = package_root / row["destination_txt_name"]
            destination_root = package_root / row["destination_root_name"]
            if not all(path.is_file() for path in (source_txt, source_root, destination_txt, destination_root)):
                raise FinalizationError(f"card/template pair missing at combined order {index}")
            txt_ok = destination_txt.read_bytes() == expected_card_bytes(source_txt.read_bytes(), source_root.name, destination_root.name)
            root_ok = sha256(source_root) == sha256(destination_root)
            ordered_line = ordered_lines[index - 1] if index <= len(ordered_lines) else ""
            order_ok = ordered_line == row["destination_txt_name"]
            if not txt_ok:
                txt_failures.append(index)
            if not root_ok:
                root_failures.append(index)
            if not order_ok:
                order_failures.append(index)
            writer.writerow({
                "combined_order_index": index, "era": row["era"], "physical_name": row["physical_name"],
                "per_era_chN": row["per_era_chN"], "combined_chN": row["combined_chN"],
                "destination_txt_name": destination_txt.name, "destination_txt_sha256": sha256(destination_txt),
                "destination_root_name": destination_root.name, "destination_root_sha256": sha256(destination_root),
                "source_txt_sha256": sha256(source_txt),
                "txt_transform_status": "exact_approved_template_reference_rewrite_only" if txt_ok else "mismatch",
                "source_root_sha256": sha256(source_root), "root_hash_match": str(root_ok).lower(),
                "ordered_input_line": ordered_line, "ordered_input_match": str(order_ok).lower(),
            })
    scalings = validate_scalings(build_manifest, rows, package_root)
    forbidden_outputs = sorted(name for name in entries if name in {"combinedcard.txt", "workspace.root", "selectedWCs.txt"} or name.startswith(("higgsCombine", "fitDiagnostics", "multidimfit", "impacts")))
    scan = scan_internal_references(package_root, diagnostics_dir, args.forbid_token, args.forbid_regex)
    summary = {
        "schema": PACKAGE_SCHEMA,
        "artifact_type": "combined_package_certification",
        "package_root": str(package_root),
        "row_count": len(rows),
        "manifest_contract": manifest_ok,
        "provenance_contract": provenance_ok,
        "readme_contract": readme_ok,
        "txt_membership_valid": observed_txt == expected_txt,
        "root_membership_valid": observed_root == expected_root,
        "txt_failures": txt_failures,
        "root_failures": root_failures,
        "ordering_valid": ordering_valid and not order_failures,
        "ordering_failures": order_failures,
        "scalings": scalings,
        "forbidden_outputs": forbidden_outputs,
        "internal_reference_scan_passed": scan["passed"],
        "certification_csv": csv_path.name,
        "certification_csv_sha256": sha256(csv_path),
        "certification_csv_rows": len(rows),
    }
    summary["passed"] = (
        summary["manifest_contract"] and summary["provenance_contract"] and summary["readme_contract"]
        and summary["txt_membership_valid"] and summary["root_membership_valid"]
        and not txt_failures and not root_failures and summary["ordering_valid"]
        and scalings["valid"] and not forbidden_outputs and scan["passed"]
    )
    write_json(diagnostics_dir / "combined_package_certification.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    sanitize_parser = commands.add_parser("sanitize", help="atomically sanitize only consumer metadata")
    sanitize_parser.add_argument("--package-root", required=True, type=Path)
    sanitize_parser.add_argument("--diagnostics-dir", required=True, type=Path)
    sanitize_parser.add_argument("--analysis", required=True)
    sanitize_parser.add_argument("--package-version", required=True)
    sanitize_parser.add_argument("--package-date", required=True)
    sanitize_parser.add_argument("--assembler-commit", required=True)
    certify_parser = commands.add_parser("certify", help="read-only package certification")
    certify_parser.add_argument("--package-root", required=True, type=Path)
    certify_parser.add_argument("--build-manifest", required=True, type=Path)
    certify_parser.add_argument("--diagnostics-dir", required=True, type=Path)
    certify_parser.add_argument("--forbid-token", action="append", default=[])
    certify_parser.add_argument("--forbid-regex", action="append", default=[])
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = sanitize(args) if args.command == "sanitize" else certify(args)
    except FinalizationError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
