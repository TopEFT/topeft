"""Build and source-certify a canonical Run2+Run3 datacard package."""

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from topeft.modules import datacard_packaging


_source_names = {"cards", "selectedWCs.txt", "scalings.json", "physical_to_chN.json", "package-provenance.json"}
_source_provenance_names = {
    "schema", "artifact_type", "analysis", "era", "package_root", "packaged_txt_count",
    "packaged_root_count", "selected_wcs_sha256", "scalings_sha256", "physical_to_chN_sha256",
    "source_manifest_sha256s", "source_unit_count", "builder_git_head", "builder_source_sha256",
    "builder_git_dirty",
}
_output_names = {"cards", "scalings.json", "ordered_card_inputs.txt", "combined_mapping_manifest.json", "package-provenance.json", "README.md"}
_provenance_names = {
    "schema", "artifact_type", "analysis", "package_version", "package_date", "package_root",
    "assembler_commit", "assembler_source_sha256", "manifest_sha256", "ordered_card_inputs_sha256",
    "scalings_sha256", "source_mapping_sha256", "source_scalings_sha256", "packaged_txt_count",
    "packaged_root_count",
}
_source_root_prefix = "ttx_multileptons-"
_channel_pattern = re.compile(r"ch[1-9][0-9]*\Z")
_writer_shapes = re.compile(rb"^([ \t]*shapes[ \t]+\S+[ \t]+\S+[ \t]+)(\S+)")
_verifier_shapes = re.compile(rb"^([ \t]*shapes[ \t]+\S+[ \t]+\S+[ \t]+)(\S+)(.*)\Z", re.DOTALL)
_sha_pattern = re.compile(r"[0-9a-f]{64}\Z")
class PublishedButNotCertified(ValueError):
    def __init__(self, certification_result):
        self.certification_result = certification_result
        super().__init__("published_but_not_certified: " + str(certification_result["mismatches"]))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _resolved_trees_overlap(first, second):
    first = Path(first).resolve()
    second = Path(second).resolve()
    return first == second or first in second.parents or second in first.parents


def _validate_build_path_disjointness(run2_package, run3_package, output, staging):
    run2_root = Path(run2_package).resolve(strict=True)
    run3_root = Path(run3_package).resolve(strict=True)
    output_root = output.parent.resolve(strict=True) / output.name
    staging_root = staging.parent.resolve(strict=True) / staging.name
    package_trees = (run2_root, run3_root, output_root, staging_root)
    for index, first in enumerate(package_trees):
        for second in package_trees[index + 1:]:
            _require(not _resolved_trees_overlap(first, second),
                     "source package, output, and staging trees must be disjoint")


def _resolved_report_path(report_path, protected_roots):
    report_path = Path(report_path)
    _require(report_path.is_absolute(), "report path must be absolute")
    _require(report_path.name not in {"", ".", ".."}, "report path must name a file")
    _require(report_path.parent.is_dir(), "report parent must already exist")
    resolved_parent = report_path.parent.resolve(strict=True)
    resolved_report = resolved_parent / report_path.name
    for protected_root in protected_roots:
        protected_root = Path(protected_root).resolve()
        _require(not _resolved_trees_overlap(resolved_report, protected_root),
                 "report path must be outside protected package roots")
    _require(not resolved_report.exists() and not resolved_report.is_symlink(),
             "report path must be absent")
    return resolved_report


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path, value):
    Path(path).write_bytes(_json_bytes(value))


def _plain_file(path):
    return path.is_file() and not path.is_symlink()


def _plain_directory(path):
    return path.is_dir() and not path.is_symlink()


def _source_package(root, era):
    root = Path(root)
    _require(root.is_absolute() and _plain_directory(root), f"{era} package root is invalid")
    _require({entry.name for entry in root.iterdir()} == _source_names, f"{era} package layout differs")
    cards = root / "cards"
    _require(_plain_directory(cards), f"{era} cards directory is invalid")
    for name in _source_names - {"cards"}:
        _require(_plain_file(root / name), f"{era} source metadata member missing: {name}")
    provenance = _read_json(root / "package-provenance.json")
    _require(isinstance(provenance, dict) and set(provenance) == _source_provenance_names
             and provenance.get("schema") == "TOP26006_v1"
             and provenance.get("artifact_type") == "package_provenance"
             and provenance.get("era") == era and provenance.get("package_root") == str(root)
             and isinstance(provenance.get("analysis"), str) and provenance["analysis"].strip(),
             f"{era} source provenance role differs")
    mapping = _read_json(root / "physical_to_chN.json")
    _require(isinstance(mapping, list) and mapping, f"{era} source mapping is invalid")
    datacard_packaging.verify_per_era_mapping(mapping, [row["physical_name"] for row in mapping])
    expected = {
        f"{_source_root_prefix}{row['physical_name']}.{suffix}"
        for row in mapping for suffix in ("txt", "root")
    }
    _require({entry.name for entry in cards.iterdir()} == expected, f"{era} cards inventory differs")
    _require(all(_plain_file(cards / name) for name in expected), f"{era} source card is not a regular file")
    _require(type(provenance["packaged_txt_count"]) is int
             and type(provenance["packaged_root_count"]) is int
             and provenance["packaged_txt_count"] == len(mapping)
             and provenance["packaged_root_count"] == len(mapping), f"{era} source counts differ")
    for key, name in (("selected_wcs_sha256", "selectedWCs.txt"),
                      ("scalings_sha256", "scalings.json"),
                      ("physical_to_chN_sha256", "physical_to_chN.json")):
        _require(provenance[key] == datacard_packaging.sha256_file(root / name),
                 f"{era} source provenance hash differs: {key}")
    scalings = _read_json(root / "scalings.json")
    _require(isinstance(scalings, list), f"{era} source scalings must be a list")
    return {"root": root, "cards": cards, "mapping": mapping, "scalings": scalings,
            "analysis": provenance["analysis"]}


def _require_matching_physical_surface(run2, run3):
    run2_names = {row["physical_name"] for row in run2["mapping"]}
    run3_names = {row["physical_name"] for row in run3["mapping"]}
    missing_from_run2 = sorted(run3_names - run2_names)
    missing_from_run3 = sorted(run2_names - run3_names)
    _require(not missing_from_run2 and not missing_from_run3,
             "Run 2 and Run 3 packages must contain the same physical targets; "
             f"missing_from_run2={missing_from_run2}, missing_from_run3={missing_from_run3}. "
             "Rebuild or select both era packages with the same physical channel and distribution set")


def _scaling_identity(record):
    _require(isinstance(record, dict) and {"channel", "process", "parameters", "scaling"} <= set(record),
             "incomplete scaling record")
    channel = record["channel"]
    process = record["process"]
    _require(isinstance(channel, str) and _channel_pattern.fullmatch(channel) is not None
             and isinstance(process, str) and bool(process.strip()), "invalid scaling identity")
    _require(isinstance(record["parameters"], list) and isinstance(record["scaling"], list),
             "invalid scaling payload")
    return channel, process


def _combine_scalings(run2_records, run3_records, mapping):
    """Relabel source records without mutating or aliasing any source object."""
    labels = {(row["era"], row["per_era_chN"]): row["combined_chN"] for row in mapping}
    result = []
    seen = set()
    for era, records in (("run2", run2_records), ("run3", run3_records)):
        for record in records:
            source_channel, process = _scaling_identity(record)
            combined = labels.get((era, source_channel))
            _require(combined is not None, "unmapped source scaling channel")
            key = (combined, process)
            _require(key not in seen, "duplicate combined scaling identity")
            seen.add(key)
            transformed = copy.deepcopy(record)
            transformed["channel"] = combined
            result.append(transformed)
    return result


def _scalings_bytes(records):
    return ("[\n" + ",\n".join(json.dumps(record, separators=(",", ":"), allow_nan=False)
                             for record in records) + "\n]\n").encode("utf-8")


def _rewrite_card(source_bytes, source_root, destination_root):
    """Writer: replace only exact token four on well-formed shapes lines."""
    source_token = source_root.encode("ascii")
    destination_token = destination_root.encode("ascii")
    rewritten = []
    count = 0
    source_bytes.decode("utf-8")
    for line in source_bytes.splitlines(keepends=True):
        if re.match(rb"^[ \t]*shapes(?:[ \t]|$)", line):
            match = _writer_shapes.match(line)
            _require(match is not None and match.group(2) == source_token,
                     "ambiguous source template reference")
            line = line[:match.start(2)] + destination_token + line[match.end(2):]
            count += 1
        rewritten.append(line)
    _require(count > 0, "source card has no shapes template reference")
    return b"".join(rewritten)


def _verify_card(source_bytes, destination_bytes, source_root, destination_root):
    """Certifier: compare independent token spans and every other byte."""
    source_lines = source_bytes.splitlines(keepends=True)
    destination_lines = destination_bytes.splitlines(keepends=True)
    _require(len(source_lines) == len(destination_lines), "card line count differs")
    count = 0
    for source_line, destination_line in zip(source_lines, destination_lines):
        if re.match(rb"^[ \t]*shapes(?:[ \t]|$)", source_line):
            source_match = _verifier_shapes.fullmatch(source_line)
            destination_match = _verifier_shapes.fullmatch(destination_line)
            _require(source_match is not None and destination_match is not None,
                     "ambiguous packaged shapes line")
            _require(source_match.group(1) == destination_match.group(1)
                     and source_match.group(3) == destination_match.group(3)
                     and source_match.group(2) == source_root.encode("ascii")
                     and destination_match.group(2) == destination_root.encode("ascii"),
                     "card differs outside approved template token")
            count += 1
        else:
            _require(source_line == destination_line, "card differs outside approved shapes line")
    _require(count > 0, "card has no approved shapes reference")


def _verify_scalings(observed, run2_records, run3_records, mapping):
    """Certifier: compare each observed record against its source payload."""
    _require(isinstance(observed, list), "combined scalings must be a list")
    sources = [("run2", record) for record in run2_records] + [("run3", record) for record in run3_records]
    _require(len(observed) == len(sources), "combined scaling count differs")
    labels = {(row["era"], row["per_era_chN"]): row["combined_chN"] for row in mapping}
    seen = set()
    for actual, (era, source) in zip(observed, sources):
        source_channel, process = _scaling_identity(source)
        actual_channel, actual_process = _scaling_identity(actual)
        _require(actual_channel == labels.get((era, source_channel)) and actual_process == process,
                 "combined scaling identity/order differs")
        key = (actual_channel, actual_process)
        _require(key not in seen, "duplicate combined scaling identity")
        seen.add(key)
        _require(set(actual) == set(source) and all(actual[name] == source[name] for name in source if name != "channel"),
                 "combined scaling payload differs")


def _verify_finite_scalings(records, label):
    for record in records:
        _scaling_identity(record)
        for coefficients in record["scaling"]:
            _require(isinstance(coefficients, list), f"{label} scaling coefficients are invalid")
            for coefficient in coefficients:
                _require(type(coefficient) in (int, float) and math.isfinite(coefficient),
                         f"{label} scaling coefficient is nonfinite or invalid")


def _source_inventory_sha256(source):
    root = source["root"]
    paths = [root / name for name in sorted(_source_names - {"cards"})]
    paths.extend(sorted((root / "cards").iterdir()))
    rows = [[path.relative_to(root).as_posix(), datacard_packaging.sha256_file(path)]
            for path in paths]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _scan_forbidden_references(package_root, output, run2_root, run3_root, card_names):
    staging = output.parent / f".{output.name}.staging"
    forbidden = {str(run2_root), str(run3_root), str(staging)}
    if package_root != output:
        forbidden.add(str(package_root))
    names = sorted(_output_names - {"cards"})
    paths = [package_root / name for name in names]
    paths.extend(package_root / "cards" / name for name in sorted(card_names) if name.endswith(".txt"))
    for path in paths:
        raw = path.read_bytes()
        for reference in forbidden:
            reference_bytes = reference.encode("utf-8")
            offset = raw.find(reference_bytes)
            matched = False
            while offset >= 0:
                end = offset + len(reference_bytes)
                if end == len(raw) or raw[end:end + 1] == b"/" or not (
                        raw[end:end + 1].isalnum() or raw[end:end + 1] in (b"_", b"-", b".")):
                    matched = True
                    break
                offset = raw.find(reference_bytes, offset + 1)
            _require(not matched,
                     f"consumer text contains forbidden path: {path.relative_to(package_root)}")


def _readme(analysis, output):
    return (f"# {analysis} combined Run2+Run3 datacard package\n\n"
            f"Package path: `{output}`\n\n"
            "Packaged TXT and ROOT payloads live under `cards/`. "
            "`combined_mapping_manifest.json` records the mapping and order. "
            "`scalings.json` is the combined scaling payload. "
            "`ordered_card_inputs.txt` is the canonical consumer ordering authority.\n\n"
            "```bash\n"
            f"cd {output}\n"
            "mapfile -t cards < ordered_card_inputs.txt\n"
            'combineCards.py "${cards[@]}" > combinedcard.txt\n'
            "```\n\n"
            "Do not use wildcard/glob card discovery.\n")


def _builder_identity():
    source = Path(__file__).resolve()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source.parents[2],
                          check=True, capture_output=True, text=True).stdout.strip()
    _require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", head) is not None,
             "builder commit is unobservable")
    return head, datacard_packaging.sha256_file(source)


def _manifest(mapping, output):
    return {"schema": "TOP26006_v1", "artifact_type": "combined_mapping_manifest",
            "package_root": str(output), "rows": mapping}


def _certify_details(package_root, run2_package, run3_package, output, result):
    """Reopen all decision-relevant source and package content."""
    checks = result["checks"]
    result["current_check"] = "source_packages"
    _require(package_root.is_absolute() and _plain_directory(package_root), "combined package root is invalid")
    _require(output.is_absolute(), "expected package path must be absolute")
    run2 = _source_package(run2_package, "run2")
    run3 = _source_package(run3_package, "run3")
    _require_matching_physical_surface(run2, run3)
    _require(run2["analysis"] == run3["analysis"], "source analyses differ")
    result["run2_source_inventory_sha256"] = _source_inventory_sha256(run2)
    result["run3_source_inventory_sha256"] = _source_inventory_sha256(run3)
    result["observed_counts"].update(run2_cards=len(run2["mapping"]), run3_cards=len(run3["mapping"]))
    checks["source_packages"] = {"result": "pass", "evidence": {
        "run2_cards": len(run2["mapping"]), "run3_cards": len(run3["mapping"])}, "failures": []}
    result["current_check"] = "combined_layout"
    observed_names = {entry.name for entry in package_root.iterdir()}
    result["missing_paths"].extend(sorted(_output_names - observed_names))
    result["extra_paths"].extend(sorted(observed_names - _output_names))
    _require(observed_names == _output_names, "combined package top-level inventory differs")
    cards = package_root / "cards"
    _require(_plain_directory(cards), "combined cards directory is invalid")
    for name in _output_names - {"cards"}:
        _require(_plain_file(package_root / name), f"combined metadata member missing: {name}")
    checks["combined_layout"] = {"result": "pass", "evidence": {"top_level_members": len(observed_names)}, "failures": []}
    result["current_check"] = "mapping"
    manifest = _read_json(package_root / "combined_mapping_manifest.json")
    _require(isinstance(manifest, dict) and set(manifest) == {"schema", "artifact_type", "package_root", "rows"}
             and manifest["schema"] == "TOP26006_v1"
             and manifest["artifact_type"] == "combined_mapping_manifest"
             and manifest["package_root"] == str(output), "combined manifest wrapper differs")
    mapping = manifest["rows"]
    datacard_packaging.verify_combined_mapping(mapping, run2["mapping"], run3["mapping"])
    expected_names = {row[name] for row in mapping for name in ("destination_txt_name", "destination_root_name")}
    observed_card_names = {entry.name for entry in cards.iterdir()}
    result["missing_paths"].extend("cards/" + name for name in sorted(expected_names - observed_card_names))
    result["extra_paths"].extend("cards/" + name for name in sorted(observed_card_names - expected_names))
    _require(observed_card_names == expected_names, "combined cards inventory differs")
    _require(all(_plain_file(cards / name) for name in expected_names), "combined card member is not a regular file")
    result["observed_counts"]["combined_cards"] = len(mapping)
    checks["mapping"] = {"result": "pass", "evidence": {"rows": len(mapping)}, "failures": []}
    result["current_check"] = "card_payloads"
    for row in mapping:
        source = run2 if row["era"] == "run2" else run3
        stem = f"{_source_root_prefix}{row['physical_name']}"
        source_root = source["cards"] / (stem + ".root")
        source_txt = source["cards"] / (stem + ".txt")
        destination_root = cards / row["destination_root_name"]
        destination_txt = cards / row["destination_txt_name"]
        _require(datacard_packaging.sha256_file(source_root) == datacard_packaging.sha256_file(destination_root),
                 "packaged ROOT bytes differ from source")
        destination_bytes = destination_txt.read_bytes()
        _verify_card(source_txt.read_bytes(), destination_bytes, source_root.name, destination_root.name)
    checks["card_payloads"] = {"result": "pass", "evidence": {"txt": len(mapping), "root": len(mapping)}, "failures": []}
    result["current_check"] = "scalings"
    scaling_bytes = (package_root / "scalings.json").read_bytes()
    observed_scalings = json.loads(scaling_bytes)
    _verify_finite_scalings(run2["scalings"], "run2 source")
    _verify_finite_scalings(run3["scalings"], "run3 source")
    _verify_finite_scalings(observed_scalings, "combined")
    _verify_scalings(observed_scalings, run2["scalings"], run3["scalings"], mapping)
    result["observed_counts"]["combined_scalings"] = len(observed_scalings)
    checks["scalings"] = {"result": "pass", "evidence": {"records": len(observed_scalings)}, "failures": []}
    result["current_check"] = "ordered_inputs"
    order_bytes = (package_root / "ordered_card_inputs.txt").read_bytes()
    order = order_bytes.decode("utf-8")
    datacard_packaging.verify_ordered_card_inputs(order, mapping)
    _require(order.endswith("\n") and all((package_root / line).is_file() for line in order.splitlines()),
             "ordered card path is missing")
    checks["ordered_inputs"] = {"result": "pass", "evidence": {"lines": len(order.splitlines())}, "failures": []}
    result["current_check"] = "provenance"
    provenance = _read_json(package_root / "package-provenance.json")
    _require(isinstance(provenance, dict) and set(provenance) == _provenance_names
             and provenance["schema"] == "TOP26006_v1"
             and provenance["artifact_type"] == "package_provenance"
             and provenance["analysis"] == run2["analysis"]
             and provenance["package_root"] == str(output)
             and isinstance(provenance["package_date"], str)
             and re.fullmatch(r"[0-9]{6}", provenance["package_date"]) is not None
             and isinstance(provenance["package_version"], str)
             and re.fullmatch(r"v[1-9][0-9]*", provenance["package_version"]) is not None
             and isinstance(provenance["assembler_commit"], str)
             and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", provenance["assembler_commit"]) is not None
             and isinstance(provenance["assembler_source_sha256"], str)
             and _sha_pattern.fullmatch(provenance["assembler_source_sha256"]) is not None,
             "combined provenance identity differs")
    expected_hashes = {
        "manifest_sha256": "combined_mapping_manifest.json",
        "ordered_card_inputs_sha256": "ordered_card_inputs.txt",
        "scalings_sha256": "scalings.json",
    }
    for key, name in expected_hashes.items():
        _require(provenance[key] == datacard_packaging.sha256_file(package_root / name),
                 f"combined provenance hash differs: {key}")
    for key, name in (("source_mapping_sha256", "physical_to_chN.json"),
                      ("source_scalings_sha256", "scalings.json")):
        _require(provenance[key] == {
            "run2": datacard_packaging.sha256_file(run2["root"] / name),
            "run3": datacard_packaging.sha256_file(run3["root"] / name),
        }, f"combined source hash differs: {key}")
    _require(type(provenance["packaged_txt_count"]) is int
             and type(provenance["packaged_root_count"]) is int
             and provenance["packaged_txt_count"] == len(mapping)
             and provenance["packaged_root_count"] == len(mapping), "packaged counts differ")
    checks["provenance"] = {"result": "pass", "evidence": {"declared_txt": provenance["packaged_txt_count"],
                           "declared_root": provenance["packaged_root_count"]}, "failures": []}
    result["current_check"] = "consumer_references"
    readme = (package_root / "README.md").read_text(encoding="utf-8")
    for clause in ("cards/", "ordered_card_inputs.txt", "combined_mapping_manifest.json", "scalings.json",
                   f"cd {output}", "mapfile -t cards < ordered_card_inputs.txt",
                   'combineCards.py "${cards[@]}" > combinedcard.txt', "wildcard/glob"):
        _require(clause in readme, f"README omits consumer contract: {clause}")
    _scan_forbidden_references(package_root, output, run2["root"], run3["root"], expected_names)
    checks["consumer_references"] = {"result": "pass", "evidence": {"text_members": 5 + len(mapping)}, "failures": []}


def certify_combined_package(package_root, run2_package, run3_package, *, expected_output=None):
    """Return a source-bound PASS or FAIL result from reopened files."""
    package_root = Path(package_root)
    output = Path(expected_output) if expected_output is not None else package_root
    result = {"schema": "topeft_combined_source_certification_v1", "result": "fail",
              "package_root": str(output), "run2_package": str(run2_package), "run3_package": str(run3_package),
              "run2_source_inventory_sha256": None, "run3_source_inventory_sha256": None,
              "observed_counts": {}, "checks": {}, "missing_paths": [], "extra_paths": [],
              "mismatches": [], "certification_observed_at": datetime.now(timezone.utc).isoformat()}
    try:
        _certify_details(package_root, run2_package, run3_package, output, result)
    except (ValueError, KeyError, TypeError, UnicodeError, OSError) as error:
        check_id = result["current_check"]
        result["checks"][check_id] = {"result": "fail", "evidence": {}, "failures": [str(error)]}
        result["mismatches"].append({"check_id": check_id, "detail": str(error)})
    else:
        result["result"] = "pass"
    result.pop("current_check", None)
    return result


def build_combined_package(run2_package, run3_package, output, analysis, package_date, package_version):
    output = Path(output)
    _require(output.is_absolute() and output.name not in {"", ".", ".."}, "output must be absolute")
    _require(_plain_directory(output.parent), "output parent does not exist")
    staging = output.parent / f".{output.name}.staging"
    _validate_build_path_disjointness(run2_package, run3_package, output, staging)
    _require(not output.exists() and not output.is_symlink(), "final output already exists")
    _require(not staging.exists() and not staging.is_symlink(), "private staging already exists")
    _require(isinstance(analysis, str) and bool(analysis.strip())
             and isinstance(package_date, str) and bool(re.fullmatch(r"[0-9]{6}", package_date))
             and isinstance(package_version, str) and bool(re.fullmatch(r"v[1-9][0-9]*", package_version)),
             "invalid package metadata")
    run2 = _source_package(run2_package, "run2")
    run3 = _source_package(run3_package, "run3")
    _require_matching_physical_surface(run2, run3)
    _require(run2["analysis"] == run3["analysis"] == analysis, "source analysis differs")
    mapping = datacard_packaging.build_combined_mapping(run2["mapping"], run3["mapping"])
    combined_scalings = _combine_scalings(run2["scalings"], run3["scalings"], mapping)
    ordered_inputs = datacard_packaging.build_ordered_card_inputs(mapping)
    head, source_hash = _builder_identity()
    staging.mkdir()
    cards = staging / "cards"
    cards.mkdir()
    for row in mapping:
        source = run2 if row["era"] == "run2" else run3
        stem = f"{_source_root_prefix}{row['physical_name']}"
        source_root = source["cards"] / (stem + ".root")
        source_txt = source["cards"] / (stem + ".txt")
        shutil.copyfile(source_root, cards / row["destination_root_name"])
        (cards / row["destination_txt_name"]).write_bytes(
            _rewrite_card(source_txt.read_bytes(), source_root.name, row["destination_root_name"]))
    _write_json(staging / "combined_mapping_manifest.json", _manifest(mapping, output))
    (staging / "scalings.json").write_bytes(_scalings_bytes(combined_scalings))
    (staging / "ordered_card_inputs.txt").write_text("\n".join(ordered_inputs) + "\n", encoding="utf-8")
    (staging / "README.md").write_text(_readme(analysis, output), encoding="utf-8")
    provenance = {
        "schema": "TOP26006_v1", "artifact_type": "package_provenance", "analysis": analysis,
        "package_date": package_date, "package_version": package_version, "package_root": str(output),
        "assembler_commit": head, "assembler_source_sha256": source_hash,
        "manifest_sha256": datacard_packaging.sha256_file(staging / "combined_mapping_manifest.json"),
        "scalings_sha256": datacard_packaging.sha256_file(staging / "scalings.json"),
        "ordered_card_inputs_sha256": datacard_packaging.sha256_file(staging / "ordered_card_inputs.txt"),
        "source_mapping_sha256": {era: datacard_packaging.sha256_file(source["root"] / "physical_to_chN.json")
                                  for era, source in (("run2", run2), ("run3", run3))},
        "source_scalings_sha256": {era: datacard_packaging.sha256_file(source["root"] / "scalings.json")
                                   for era, source in (("run2", run2), ("run3", run3))},
        "packaged_txt_count": len(mapping), "packaged_root_count": len(mapping),
    }
    _write_json(staging / "package-provenance.json", provenance)
    staging_certification = certify_combined_package(staging, run2["root"], run3["root"], expected_output=output)
    _require(staging_certification["result"] == "pass",
             "private staging certification failed: " + str(staging_certification["mismatches"]))
    _require(not output.exists() and not output.is_symlink(), "final output appeared before publication")
    staging.rename(output)
    final_certification = certify_combined_package(output, run2["root"], run3["root"])
    if final_certification["result"] != "pass":
        raise PublishedButNotCertified(final_certification)
    return {"schema": "topeft_combined_package_build_v1", "result": "pass",
            "package_root": str(output), "post_publication_certification": final_certification}


def _write_certification_report(path, result, roots):
    path = _resolved_report_path(path, roots)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as report:
            temporary_path = Path(report.name)
            report.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--run2-package", type=Path, required=True)
    build.add_argument("--run3-package", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--analysis", required=True)
    build.add_argument("--package-date", required=True)
    build.add_argument("--package-version", required=True)
    certify = commands.add_parser("certify")
    certify.add_argument("--package-root", type=Path, required=True)
    certify.add_argument("--run2-package", type=Path, required=True)
    certify.add_argument("--run3-package", type=Path, required=True)
    certify.add_argument("--report-json", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        try:
            result = build_combined_package(args.run2_package, args.run3_package, args.output,
                                            args.analysis, args.package_date, args.package_version)
        except PublishedButNotCertified as error:
            print(json.dumps({"schema": "topeft_combined_package_build_v1", "result": "fail",
                              "state": "published_but_not_certified", "package_root": str(args.output),
                              "post_publication_certification": error.certification_result}, sort_keys=True))
            return 1
    else:
        roots = (args.package_root, args.run2_package, args.run3_package)
        if args.report_json is not None:
            try:
                args.report_json = _resolved_report_path(args.report_json, roots)
            except (OSError, ValueError) as error:
                parser.error(str(error))
        result = certify_combined_package(*roots)
        if args.report_json is not None:
            _write_certification_report(args.report_json, result, roots)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
