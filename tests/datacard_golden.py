"""Portable, test-only comparison with accepted datacard package identities."""

import hashlib
import json
import re
from pathlib import Path


_sha256_pattern = re.compile(r"[0-9a-f]{64}\Z")
_forbidden_pattern = re.compile(r"(?:/groups/|/users/|reports/diagnostics|t0_datacards_|ptz-lj0pt_withSys)")
_result_keys = (
    "payload_mismatches", "semantic_contract_mismatches",
    "allowed_provenance_deltas", "unexpected_provenance_deltas",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _basename(value, suffix):
    return isinstance(value, str) and value == Path(value).name and value.endswith(suffix) and value not in (suffix,)


def _per_era_mapping_issues(rows, expected_count):
    if not isinstance(rows, list):
        return [{"reason": "not_list"}]
    issues = []
    if len(rows) != expected_count:
        issues.append({"reason": "row_count", "expected": expected_count, "observed": len(rows)})
    physical_names = {}
    channel_names = {}
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or set(row) != {"physical_name", "per_era_chN"}
                or any(not isinstance(row[key], str) for key in ("physical_name", "per_era_chN"))):
            issues.append({"reason": "malformed_row", "index": index, "observed": row})
            continue
        for key, seen in (("physical_name", physical_names), ("per_era_chN", channel_names)):
            value = row[key]
            if value in seen:
                issues.append({"reason": "duplicate_" + key, "index": index,
                               "first_index": seen[value], key: value})
            else:
                seen[value] = index
    return issues


def load_golden_fixture(path):
    """Load a strict portable fixture; malformed or site-bound fixtures fail closed."""
    raw = Path(path).read_text(encoding="utf-8")
    if _forbidden_pattern.search(raw):
        raise ValueError("fixture contains an internal path or round label")
    fixture = json.loads(raw)
    if not isinstance(fixture, dict):
        raise ValueError("fixture must be an object")
    common = {"fixture_schema", "analysis", "reference_role",
              "payload", "scalings_sha256", "target_layout_contract", "target_package_naming_contract"}
    combined = {"package_contract_schema", "ordered_card_inputs", "combined_mapping",
                "readme_operational_contract", "provenance_comparison_contract"}
    per_era = {"era", "selected_wcs_sha256", "physical_to_chN"}
    is_combined = "package_contract_schema" in fixture
    if set(fixture) != common | (combined if is_combined else per_era):
        raise ValueError("fixture field set is malformed")
    if (fixture["fixture_schema"] != "top26006_datacard_golden_v1"
            or fixture["analysis"] != "TOP-26-006"
            or fixture["reference_role"] != "development_regression"
            or fixture["target_layout_contract"] != {"payload_subdirectory": "cards"}):
        raise ValueError("fixture contract header is malformed")
    expected_name = ("top26006_combined_package_<date>_vN" if is_combined else
                     "top26006_" + str(fixture.get("era")) + "_package_<date>_vN")
    if fixture["target_package_naming_contract"] != expected_name:
        raise ValueError("fixture naming contract is malformed")
    if not is_combined and fixture["era"] not in ("run2", "run3"):
        raise ValueError("fixture era is malformed")
    payload = fixture["payload"]
    if not isinstance(payload, dict) or set(payload) != {"txt", "root"}:
        raise ValueError("payload classes are malformed")
    for kind, suffix in (("txt", ".txt"), ("root", ".root")):
        values = payload[kind]
        if not isinstance(values, dict) or not values:
            raise ValueError("payload identities are missing")
        if any(not _basename(name, suffix) or not isinstance(value, str) or not _sha256_pattern.fullmatch(value)
               for name, value in values.items()):
            raise ValueError("payload identity is malformed")
    if {name[:-4] for name in payload["txt"]} != {name[:-5] for name in payload["root"]}:
        raise ValueError("payload TXT/ROOT identities do not pair")
    for key in ("scalings_sha256",) + (() if is_combined else ("selected_wcs_sha256",)):
        if not isinstance(fixture[key], str) or not _sha256_pattern.fullmatch(fixture[key]):
            raise ValueError(key + " is malformed")
    if is_combined:
        if fixture["package_contract_schema"] != "TOP26006_v1":
            raise ValueError("consumer schema is malformed")
        ordered = fixture["ordered_card_inputs"]
        if set(ordered) != {"source_sha256", "basenames"} or not _sha256_pattern.fullmatch(ordered["source_sha256"]):
            raise ValueError("ordered-card fixture is malformed")
        if (not isinstance(ordered["basenames"], list) or len(ordered["basenames"]) != len(payload["txt"])
                or len(set(ordered["basenames"])) != len(ordered["basenames"])
                or set(ordered["basenames"]) != set(payload["txt"])):
            raise ValueError("ordered-card identities are malformed")
        if not isinstance(fixture["combined_mapping"], list) or len(fixture["combined_mapping"]) != len(ordered["basenames"]):
            raise ValueError("combined mapping is malformed")
        if fixture["readme_operational_contract"] != {"ordered_list_combine_cards_required": True, "direct_glob_forbidden": True}:
            raise ValueError("README contract is malformed")
        contract = fixture["provenance_comparison_contract"]
        if not isinstance(contract, dict) or set(contract) != {"required_stable_semantic_values", "reference_values", "allowed_variable_keys", "conditional_digest_deltas"}:
            raise ValueError("provenance contract is malformed")
        stable = contract["required_stable_semantic_values"]
        stable_keys = {"schema", "artifact_type", "analysis", "packaged_txt_count", "packaged_root_count",
                       "scalings_sha256"}
        reference_keys = {"assembler_commit", "assembler_source_sha256", "manifest_sha256",
                          "ordered_card_inputs_sha256", "package_date", "package_version", "source_mapping_sha256",
                          "source_scalings_sha256"}
        variable_keys = {"package_root", "package_date", "package_version", "created_at",
                         "assembler_commit", "assembler_source_sha256"}
        conditional = {"manifest_sha256": "combined_mapping_semantics_equal",
                       "source_mapping_sha256": "combined_mapping_semantics_equal",
                       "source_scalings_sha256": "combined_scalings_exact"}
        if (not isinstance(stable, dict) or set(stable) != stable_keys
                or not isinstance(contract["reference_values"], dict)
                or set(contract["reference_values"]) != reference_keys
                or not isinstance(contract["allowed_variable_keys"], list)
                or len(contract["allowed_variable_keys"]) != len(variable_keys)
                or set(contract["allowed_variable_keys"]) != variable_keys
                or contract["conditional_digest_deltas"] != conditional):
            raise ValueError("provenance key contract is malformed")
        if (stable.get("schema") != "TOP26006_v1" or stable.get("artifact_type") != "package_provenance"
                or stable.get("analysis") != "TOP-26-006"
                or stable.get("packaged_txt_count") != len(payload["txt"])
                or stable.get("packaged_root_count") != len(payload["root"])
                or stable.get("scalings_sha256") != fixture["scalings_sha256"]):
            raise ValueError("provenance stable semantics are malformed")
    else:
        rows = fixture["physical_to_chN"]
        if _per_era_mapping_issues(rows, len(payload["txt"])):
            raise ValueError("per-era mapping is malformed")
    return fixture


def _collect_payload(package_root):
    root = Path(package_root)
    cards = root / "cards"
    result = {"txt": {}, "root": {}}
    for kind, suffix in (("txt", ".txt"), ("root", ".root")):
        for path in cards.glob("*" + suffix):
            result[kind][path.name] = _sha256(path)
    return result


def _ordered_basenames(lines):
    basenames = []
    for line in lines:
        if line.startswith("cards/") and _basename(line[6:], ".txt"):
            basenames.append(line[6:])
        else:
            raise ValueError("ordered card path must use cards/: " + line)
    return basenames


def _normalize_mapping(rows):
    normalized = []
    keys = {"era", "physical_name", "per_era_chN", "combined_chN", "combined_order_index",
            "destination_txt_name", "destination_root_name"}
    for row in rows:
        if not isinstance(row, dict) or not keys <= set(row):
            raise ValueError("combined mapping row is malformed")
        value = {key: row[key] for key in keys}
        for key in ("destination_txt_name", "destination_root_name"):
            name = value[key]
            if not _basename(name, ".txt" if key.endswith("txt_name") else ".root"):
                raise ValueError("combined mapping destination must be a basename")
        normalized.append(value)
    return normalized


def _compare_provenance(fixture, package_root, result, mapping_equal, combined_scalings_exact):
    path = Path(package_root) / "package-provenance.json"
    contract = fixture["provenance_comparison_contract"]
    if not path.is_file():
        result["semantic_contract_mismatches"].append({"artifact": path.name, "reason": "missing"})
        return
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        result["semantic_contract_mismatches"].append({"artifact": path.name, "reason": str(exc)})
        return
    if not isinstance(observed, dict):
        result["semantic_contract_mismatches"].append({"artifact": path.name, "reason": "not_object"})
        return
    stable = contract["required_stable_semantic_values"]
    reference = contract["reference_values"]
    known = set(stable) | set(reference) | set(contract["allowed_variable_keys"])
    for key in sorted((set(stable) | set(reference) | {"package_root"}) - set(observed)):
        result["semantic_contract_mismatches"].append({"artifact": path.name, "key": key, "reason": "missing_required_key"})
    for key in sorted(set(observed) - known):
        result["unexpected_provenance_deltas"].append({"key": key, "reason": "unknown_key", "observed": observed[key]})
    for key, expected in stable.items():
        if key in observed and observed[key] != expected:
            result["semantic_contract_mismatches"].append({"artifact": path.name, "key": key, "expected": expected, "observed": observed[key]})
    for key, expected in reference.items():
        if key not in observed or observed[key] == expected:
            continue
        if key in contract["allowed_variable_keys"]:
            result["allowed_provenance_deltas"].append({"key": key, "reference": expected, "observed": observed[key]})
        elif key in ("manifest_sha256", "source_mapping_sha256") and mapping_equal:
            result["allowed_provenance_deltas"].append({"key": key, "reference": expected, "observed": observed[key], "basis": "combined_mapping_semantics_equal"})
        elif key == "source_scalings_sha256" and combined_scalings_exact:
            result["allowed_provenance_deltas"].append({"key": key, "reference": expected, "observed": observed[key], "basis": "combined_scalings_exact"})
        else:
            result["unexpected_provenance_deltas"].append({"key": key, "reference": expected, "observed": observed[key]})
    if "package_root" in observed:
        result["allowed_provenance_deltas"].append({"key": "package_root", "reference": "omitted_for_portability", "observed": observed["package_root"]})
    if "created_at" in observed:
        result["allowed_provenance_deltas"].append({"key": "created_at", "reference": "absent_in_reference", "observed": observed["created_at"]})


def compare_package(fixture, package_root, physical_to_chN=None):
    """Return observed differences by contract class, without a verdict field."""
    result = {key: [] for key in _result_keys}
    package_root = Path(package_root)
    payload = _collect_payload(package_root)
    if not (package_root / "cards").is_dir():
        result["semantic_contract_mismatches"].append({"artifact": "cards", "reason": "missing_payload_directory"})
    for suffix in (".txt", ".root"):
        for path in package_root.glob("*" + suffix):
            if path.name not in {"selectedWCs.txt", "ordered_card_inputs.txt"}:
                result["semantic_contract_mismatches"].append({"artifact": "payload", "reason": "root_level_payload", "basename": path.name})
    for kind in ("txt", "root"):
        expected = fixture["payload"][kind]
        observed = payload[kind]
        for name in sorted(set(expected) | set(observed)):
            if name not in expected:
                result["payload_mismatches"].append({"kind": kind, "basename": name, "reason": "extra"})
            elif name not in observed:
                result["payload_mismatches"].append({"kind": kind, "basename": name, "reason": "missing"})
            elif observed[name] != expected[name]:
                result["payload_mismatches"].append({"kind": kind, "basename": name, "reason": "sha256", "expected": expected[name], "observed": observed[name]})
    scalings_path = package_root / "scalings.json"
    observed_scalings_sha256 = _sha256(scalings_path) if scalings_path.is_file() else None
    combined_scalings_exact = observed_scalings_sha256 == fixture["scalings_sha256"]
    if not combined_scalings_exact:
        result["payload_mismatches"].append({"kind": "scalings.json", "basename": scalings_path.name,
                                              "expected": fixture["scalings_sha256"],
                                              "observed": observed_scalings_sha256})
    if "era" in fixture:
        path = package_root / "selectedWCs.txt"
        observed = _sha256(path) if path.is_file() else None
        if observed != fixture["selected_wcs_sha256"]:
            result["payload_mismatches"].append({"kind": "selected_wcs", "basename": path.name, "expected": fixture["selected_wcs_sha256"], "observed": observed})
        expected_mapping = {row["physical_name"]: row["per_era_chN"] for row in fixture["physical_to_chN"]}
        if physical_to_chN is None:
            result["semantic_contract_mismatches"].append({"artifact": "physical_to_chN", "reason": "missing_candidate_mapping"})
        else:
            issues = _per_era_mapping_issues(physical_to_chN, len(fixture["physical_to_chN"]))
            result["semantic_contract_mismatches"].extend({"artifact": "physical_to_chN", **issue} for issue in issues)
            if not issues:
                observed_mapping = {row["physical_name"]: row["per_era_chN"] for row in physical_to_chN}
                for name in sorted(set(expected_mapping) | set(observed_mapping)):
                    if expected_mapping.get(name) != observed_mapping.get(name):
                        result["semantic_contract_mismatches"].append({"artifact": "physical_to_chN", "physical_name": name, "expected": expected_mapping.get(name), "observed": observed_mapping.get(name)})
        return result

    ordered_path = package_root / "ordered_card_inputs.txt"
    ordered_equal = False
    try:
        lines = ordered_path.read_text(encoding="utf-8").splitlines()
        basenames = _ordered_basenames(lines)
        expected = fixture["ordered_card_inputs"]["basenames"]
        ordered_equal = basenames == expected
        for line in lines:
            if not (package_root / line).is_file():
                result["semantic_contract_mismatches"].append({"artifact": "ordered_card_inputs", "reason": "unresolved_path", "path": line})
        if _sha256(ordered_path) != fixture["ordered_card_inputs"]["source_sha256"]:
            result["semantic_contract_mismatches"].append({"artifact": "ordered_card_inputs", "reason": "sha256"})
        if not ordered_equal:
            for index in range(max(len(expected), len(basenames))):
                want = expected[index] if index < len(expected) else None
                got = basenames[index] if index < len(basenames) else None
                if want != got:
                    result["semantic_contract_mismatches"].append({"artifact": "ordered_card_inputs", "index": index, "expected": want, "observed": got})
    except (ValueError, OSError) as exc:
        result["semantic_contract_mismatches"].append({"artifact": "ordered_card_inputs", "reason": str(exc)})
    manifest_path = package_root / "combined_mapping_manifest.json"
    mapping_equal = False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != fixture["package_contract_schema"] or manifest.get("artifact_type") != "combined_mapping_manifest":
            result["semantic_contract_mismatches"].append({"artifact": manifest_path.name, "reason": "schema_or_artifact_type", "observed": {key: manifest.get(key) for key in ("schema", "artifact_type")}})
        observed_mapping = _normalize_mapping(manifest["rows"])
        expected_mapping = _normalize_mapping(fixture["combined_mapping"])
        mapping_equal = observed_mapping == expected_mapping
        if not mapping_equal:
            for index in range(max(len(expected_mapping), len(observed_mapping))):
                want = expected_mapping[index] if index < len(expected_mapping) else None
                got = observed_mapping[index] if index < len(observed_mapping) else None
                if want != got:
                    result["semantic_contract_mismatches"].append({"artifact": "combined_mapping", "index": index, "expected": want, "observed": got})
    except (ValueError, KeyError, OSError, TypeError) as exc:
        result["semantic_contract_mismatches"].append({"artifact": manifest_path.name, "reason": str(exc)})
    readme = package_root / "README.md"
    try:
        content = readme.read_text(encoding="utf-8")
        normalized = re.sub(r"\s+", " ", content)
        load_command = re.search(r"\bmapfile\s+-t\s+cards\s*<\s*ordered_card_inputs\.txt\b", normalized)
        combine_command = re.search(r'\bcombineCards\.py\s+"\$\{cards\[@\]\}"\s*>\s*combinedcard\.txt\b', normalized)
        if (not load_command or not combine_command or load_command.end() > combine_command.start()
                or not re.search(r"\b(?:do not|never)\b.{0,100}\b(?:wildcard|glob)\b", normalized, re.IGNORECASE)):
            result["semantic_contract_mismatches"].append({"artifact": "README.md", "reason": "ordered_list_or_glob_contract_missing"})
    except OSError as exc:
        result["semantic_contract_mismatches"].append({"artifact": "README.md", "reason": str(exc)})
    _compare_provenance(fixture, package_root, result, mapping_equal, combined_scalings_exact)
    return result
