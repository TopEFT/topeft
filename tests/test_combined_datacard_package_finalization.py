"""Focused coverage for combined-package sanitization and certification."""

import hashlib
import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "topeft_run2"))
from finalize_combined_datacard_package import (  # noqa: E402
    FinalizationError,
    certify,
    parser,
    sanitize,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(path), "sha256": _sha256(path)}


def _fixture(tmp_path):
    source_root = tmp_path / "source"
    package_root = tmp_path / "package"
    package_root.mkdir()
    rows, mappings, source_scalings, expected_scalings = [], {}, {}, []
    for era, offset, prefix in (("run2", 0, "Run2_"), ("run3", 1, "Run3_")):
        source_dir = source_root / era / "cards"
        source_dir.mkdir(parents=True)
        physical_name = f"{era}_region_ptz"
        source_txt = source_dir / f"ttx_multileptons-{physical_name}.txt"
        source_root_file = source_dir / f"ttx_multileptons-{physical_name}.root"
        source_txt.write_text(
            f"shapes * * {source_root_file.name} $PROCESS $PROCESS_$SYSTEMATIC\n"
            f"bin bin_{physical_name}\nprocess ttH_sm\nrate 1\n", encoding="utf-8"
        )
        source_root_file.write_bytes(f"root-{era}".encode("utf-8"))
        index = offset + 1
        destination_txt = f"{prefix}ttx_multileptons-{physical_name}.txt"
        destination_root = f"{prefix}ttx_multileptons-{physical_name}.root"
        (package_root / destination_txt).write_text(
            source_txt.read_text().replace(source_root_file.name, destination_root), encoding="utf-8"
        )
        (package_root / destination_root).write_bytes(source_root_file.read_bytes())
        row = {
            "era": era,
            "physical_name": physical_name,
            "per_era_chN": "ch1",
            "combined_chN": f"ch{index}",
            "combined_order_index": index,
            "destination_txt_name": destination_txt,
            "destination_root_name": destination_root,
            "source_txt_path": str(source_txt),
            "source_root_path": str(source_root_file),
        }
        rows.append(row)
        mappings[era] = _write_json(tmp_path / f"{era}_mapping.json", {"entries": [physical_name]})
        record = {"channel": "ch1", "process": "ttH_sm", "parameters": ["cSM[1]"], "scaling": [[1.0]], "extra": era}
        source_scalings[era] = _write_json(source_dir / "scalings.json", [record])
        expected_scalings.append({**record, "channel": f"ch{index}"})
    build_manifest = {
        "schema": "TOP26006_v1",
        "artifact_type": "combined_mapping_manifest",
        "source_mappings": mappings,
        "source_scalings": source_scalings,
        "rows": list(reversed(rows)),
    }
    manifest_path = tmp_path / "build_manifest.json"
    _write_json(manifest_path, build_manifest)
    _write_json(package_root / "combined_mapping_manifest.json", build_manifest)
    _write_json(package_root / "package-provenance.json", {
        "schema": "TOP26006_v1", "artifact_type": "package_provenance",
        "assembler_source_sha256": "a" * 64, "source_build_path": "/internal/build/path",
    })
    (package_root / "README.md").write_text("internal build instructions\n", encoding="utf-8")
    (package_root / "ordered_card_inputs.txt").write_text(
        "".join(row["destination_txt_name"] + "\n" for row in rows), encoding="utf-8"
    )
    _write_json(package_root / "scalings.json", expected_scalings)
    return package_root, manifest_path, rows


def _sanitize_args(package_root, diagnostics_dir):
    return parser().parse_args([
        "sanitize", "--package-root", str(package_root), "--diagnostics-dir", str(diagnostics_dir),
        "--analysis", "TOP-TEST", "--package-version", "test", "--package-date", "today",
        "--assembler-commit", "deadbeef",
    ])


def _certify_args(package_root, manifest_path, diagnostics_dir, *extra):
    return parser().parse_args([
        "certify", "--package-root", str(package_root), "--build-manifest", str(manifest_path),
        "--diagnostics-dir", str(diagnostics_dir), *extra,
    ])


def _sanitized_fixture(tmp_path):
    package_root, manifest_path, rows = _fixture(tmp_path)
    original = {path.name: _sha256(path) for path in package_root.iterdir()}
    sanitize(_sanitize_args(package_root, tmp_path / "sanitize"))
    return package_root, manifest_path, rows, original


def test_sanitize_builds_inventory_backups_and_consumer_metadata(tmp_path):
    package_root, _, rows = _fixture(tmp_path)
    original_payload = {path.name: _sha256(path) for path in package_root.iterdir() if path.name not in {
        "combined_mapping_manifest.json", "package-provenance.json", "README.md"
    }}
    diagnostics = tmp_path / "sanitize"
    sanitize(_sanitize_args(package_root, diagnostics))
    inventory = json.loads((diagnostics / "package_before_after_inventory.json").read_text())
    assert set(inventory["changed_file_names"]) == {"combined_mapping_manifest.json", "package-provenance.json", "README.md"}
    assert inventory["non_metadata_hashes_unchanged"] is True
    assert (diagnostics / "internal_pre_sanitization_metadata" / "README.md").is_file()
    manifest = json.loads((package_root / "combined_mapping_manifest.json").read_text())
    assert manifest["schema"] == "TOP26006_v1"
    assert manifest["artifact_type"] == "combined_mapping_manifest"
    assert len(manifest["rows"]) == len(rows)
    assert set(manifest["rows"][0]) == {"era", "physical_name", "per_era_chN", "combined_chN", "combined_order_index", "destination_txt_name", "destination_root_name"}
    assert "source_txt_path" not in (package_root / "combined_mapping_manifest.json").read_text()
    provenance = json.loads((package_root / "package-provenance.json").read_text())
    assert provenance["schema"] == "TOP26006_v1"
    assert provenance["artifact_type"] == "package_provenance"
    assert "source_build_path" not in provenance
    assert {path.name: _sha256(path) for path in package_root.iterdir() if path.name in original_payload} == original_payload


def test_sanitize_fails_closed_for_incompatible_schema(tmp_path):
    package_root, _, _ = _fixture(tmp_path)
    manifest_path = package_root / "combined_mapping_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = "unsupported"
    _write_json(manifest_path, manifest)
    with pytest.raises(FinalizationError, match="schema"):
        sanitize(_sanitize_args(package_root, tmp_path / "sanitize"))


def test_sanitize_fails_when_a_frozen_payload_changes_before_replacement(tmp_path, monkeypatch):
    package_root, _, _ = _fixture(tmp_path)
    import finalize_combined_datacard_package as module

    original = module.consumer_manifest

    def mutate_then_build(*args):
        (package_root / "scalings.json").write_text("[]\n", encoding="utf-8")
        return original(*args)

    monkeypatch.setattr(module, "consumer_manifest", mutate_then_build)
    with pytest.raises(FinalizationError, match="unauthorized"):
        sanitize(_sanitize_args(package_root, tmp_path / "sanitize"))


def test_certify_accepts_approved_transformation_and_scalings(tmp_path):
    package_root, manifest_path, rows, _ = _sanitized_fixture(tmp_path)
    result = certify(_certify_args(package_root, manifest_path, tmp_path / "certify"))
    assert result["passed"] is True
    assert result["certification_csv_rows"] == len(rows)
    assert result["scalings"]["valid"] is True


def test_certify_rejects_additional_txt_mutation(tmp_path):
    package_root, manifest_path, rows, _ = _sanitized_fixture(tmp_path)
    card = package_root / rows[0]["destination_txt_name"]
    card.write_text(card.read_text() + "nuisance extra\n", encoding="utf-8")
    result = certify(_certify_args(package_root, manifest_path, tmp_path / "certify"))
    assert result["passed"] is False
    assert result["txt_failures"] == [1]


def test_certify_rejects_root_hash_and_forbidden_outputs(tmp_path):
    package_root, manifest_path, rows, _ = _sanitized_fixture(tmp_path)
    (package_root / rows[1]["destination_root_name"]).write_bytes(b"mutated")
    (package_root / "combinedcard.txt").write_text("forbidden", encoding="utf-8")
    result = certify(_certify_args(package_root, manifest_path, tmp_path / "certify"))
    assert result["passed"] is False
    assert result["root_failures"] == [2]
    assert result["forbidden_outputs"] == ["combinedcard.txt"]


@pytest.mark.parametrize("ordered", [
    "wrong.txt\nRun3_ttx_multileptons-run3_region_ptz.txt\n",
    "Run2_ttx_multileptons-run2_region_ptz.txt\nRun2_ttx_multileptons-run2_region_ptz.txt\n",
    "/absolute.txt\nRun3_ttx_multileptons-run3_region_ptz.txt\n",
])
def test_certify_rejects_ordering_mismatch_and_absolute_inputs(tmp_path, ordered):
    package_root, manifest_path, _, _ = _sanitized_fixture(tmp_path)
    (package_root / "ordered_card_inputs.txt").write_text(ordered, encoding="utf-8")
    result = certify(_certify_args(package_root, manifest_path, tmp_path / "certify"))
    assert result["passed"] is False
    assert result["ordering_valid"] is False


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "mismatch"])
def test_certify_rejects_scaling_identity_or_payload_errors(tmp_path, change):
    package_root, manifest_path, _, _ = _sanitized_fixture(tmp_path)
    scalings_path = package_root / "scalings.json"
    records = json.loads(scalings_path.read_text())
    if change == "missing":
        records.pop()
    elif change == "extra":
        records.append({"channel": "ch999", "process": "x", "parameters": [], "scaling": []})
    elif change == "duplicate":
        records.append(dict(records[0]))
    else:
        records[0]["scaling"] = [[9.0]]
    _write_json(scalings_path, records)
    result = certify(_certify_args(package_root, manifest_path, tmp_path / "certify"))
    assert result["passed"] is False
    assert result["scalings"]["valid"] is False


def test_certify_scans_default_and_caller_supplied_internal_patterns(tmp_path):
    package_root, manifest_path, _, _ = _sanitized_fixture(tmp_path)
    (package_root / "README.md").write_text("prompt_id: internal\nclient-secret-label\n", encoding="utf-8")
    result = certify(_certify_args(package_root, manifest_path, tmp_path / "certify", "--forbid-token", "client-secret", "--forbid-regex", "label$"))
    scan = json.loads((tmp_path / "certify" / "internal_reference_scan.json").read_text())
    assert result["passed"] is False
    assert {match["matched_text"] for match in scan["matches"]} >= {"prompt_id", "client-secret", "label"}


def test_maintained_source_has_no_round_specific_runtime_state():
    source = (Path(__file__).resolve().parents[1] / "analysis" / "topeft_run2" / "finalize_combined_datacard_package.py").read_text()
    assert "top26006_combined_package_260923" not in source
    assert "004J" not in source
    assert "004K2R3" not in source
