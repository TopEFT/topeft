"""Focused mechanical coverage for explicit-order combined card packaging."""

import hashlib
import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "topeft_run2"))
from assemble_combined_datacard_package import (  # noqa: E402
    assemble_package,
    rewrite_card_template,
    validate_manifest,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": _sha256(path)}


def _fixture(tmp_path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "combined"
    rows = []
    scalings = {}
    mappings = {}
    source_cards = {}
    for era, offset, prefix in (("run2", 0, "Run2_"), ("run3", 2, "Run3_")):
        source_dir = source_root / era / "ptz-lj0pt_withSys"
        source_dir.mkdir(parents=True)
        mapping_entries = []
        records = []
        for local_index, physical_channel in ((1, "region_a"), (2, "region_b")):
            physical_name = f"{physical_channel}_ptz"
            basename = f"ttx_multileptons-{physical_name}"
            txt = source_dir / f"{basename}.txt"
            root = source_dir / f"{basename}.root"
            card = (
                f"shapes * * {root.name} $PROCESS $PROCESS_$SYSTEMATIC\n"
                f"bin bin_{physical_name}\n"
                "process ttH_sm\n"
                "nuisance_name shape 1\n"
            )
            txt.write_text(card, encoding="utf-8")
            root.write_bytes(f"ROOT {era} {physical_name}".encode())
            source_cards[(era, physical_name)] = card
            mapping_entries.append({
                "era": era,
                "physical_channel": physical_channel,
                "distribution": "ptz",
                "physical_name": physical_name,
                "per_era_chN": f"ch{local_index}",
            })
            combined_index = offset + local_index
            rows.append({
                "era": era,
                "physical_channel": physical_channel,
                "distribution": "ptz",
                "physical_name": physical_name,
                "per_era_chN": f"ch{local_index}",
                "per_era_ch_index": local_index,
                "combined_chN": f"ch{combined_index}",
                "combined_order_index": combined_index,
                "source_txt_path": str(txt),
                "source_root_path": str(root),
                "destination_txt_name": f"{prefix}{basename}.txt",
                "destination_root_name": f"{prefix}{basename}.root",
                "expected_template_reference_after_packaging": f"{prefix}{basename}.root",
                "destination_naming_policy": "era_prefix_v1: Run2_/Run3_ + exact source basename",
            })
            records.append({
                "channel": f"ch{local_index}",
                "process": "ttH_sm",
                "parameters": ["cSM[1]", f"wc_{era}[0]"],
                "scaling": [[1.0, float(local_index)]],
                "extra": f"unchanged_{era}_{local_index}",
            })
        mappings[era] = _write_json(tmp_path / f"{era}_mapping.json", {
            "era": era,
            "entries": mapping_entries,
        })
        scalings[era] = _write_json(source_dir / "scalings.json", records)
    manifest = {
        "schema": "TOP26006_v1",
        "artifact_type": "combined_mapping_manifest",
        "source_per_era_package_root": str(source_root),
        "destination_package_root": str(output_root),
        "destination_naming_policy": "era_prefix_v1: Run2_/Run3_ + exact source basename",
        "combined_order_policy": "run2: N; run3: 129+N for certified per-era chN",
        "selectedWCs_combined_rule": "outside_current_packaging_boundary",
        "source_mappings": mappings,
        "source_scalings": scalings,
        "rows": list(reversed(rows)),
    }
    manifest_path = tmp_path / "manifest.json"
    ordered_path = tmp_path / "ordered_card_inputs.txt"
    _write_json(manifest_path, manifest)
    ordered_path.write_text(
        "".join(row["destination_txt_name"] + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest_path, ordered_path, output_root, rows, source_cards


def test_assembly_uses_manifest_order_and_preserves_payload(tmp_path):
    manifest_path, ordered_path, output_root, rows, source_cards = _fixture(tmp_path)
    provenance = assemble_package(manifest_path, ordered_path, output_root, expected_per_era_count=2)
    assert provenance["packaged_txt_count"] == 4
    assert provenance["packaged_root_count"] == 4
    assert provenance["source_scaling_record_counts"] == {"run2": 2, "run3": 2}
    assert provenance["schema"] == "TOP26006_v1"
    assert provenance["artifact_type"] == "package_provenance"
    assert provenance["source_per_era_package_root"] == str(tmp_path / "source")
    assert "source_004j_package_root" not in provenance
    assert (output_root / "ordered_card_inputs.txt").read_text() == "".join(
        row["destination_txt_name"] + "\n" for row in rows
    )
    assert not (output_root / "selectedWCs.txt").exists()
    assert not (output_root / "combinedcard.txt").exists()
    for row in rows:
        old = Path(row["source_root_path"]).name
        expected_card = source_cards[(row["era"], row["physical_name"])].replace(
            old, row["destination_root_name"], 1
        )
        assert (output_root / row["destination_txt_name"]).read_text() == expected_card
        assert (output_root / row["destination_root_name"]).read_bytes() == Path(
            row["source_root_path"]
        ).read_bytes()
    output_records = json.loads((output_root / "scalings.json").read_text())
    by_combined = {(record["channel"], record["process"]): record for record in output_records}
    assert set(by_combined) == {(f"ch{index}", "ttH_sm") for index in range(1, 5)}
    for row in rows:
        source_records = json.loads(Path(json.loads(manifest_path.read_text())["source_scalings"][row["era"]]["path"]).read_text())
        source = next(record for record in source_records if record["channel"] == row["per_era_chN"])
        expected = {**source, "channel": row["combined_chN"]}
        assert by_combined[(row["combined_chN"], source["process"])] == expected


@pytest.mark.parametrize("field", ["combined_chN", "destination_txt_name", "destination_root_name"])
def test_duplicate_manifest_identity_fails_closed(tmp_path, field):
    manifest_path, ordered_path, output_root, _, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["rows"][0][field] = manifest["rows"][1][field]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate"):
        assemble_package(manifest_path, ordered_path, output_root, expected_per_era_count=2)
    assert not output_root.exists()


@pytest.mark.parametrize("missing_kind", ["txt", "root"])
def test_missing_source_pair_fails_closed(tmp_path, missing_kind):
    manifest_path, ordered_path, output_root, rows, _ = _fixture(tmp_path)
    Path(rows[0][f"source_{missing_kind}_path"]).unlink()
    with pytest.raises(ValueError, match="missing source"):
        assemble_package(manifest_path, ordered_path, output_root, expected_per_era_count=2)
    assert not output_root.exists()


def test_ambiguous_template_reference_fails_without_publication(tmp_path):
    manifest_path, ordered_path, output_root, rows, _ = _fixture(tmp_path)
    source_card = Path(rows[1]["source_txt_path"])
    source_card.write_text(source_card.read_text().replace(Path(rows[1]["source_root_path"]).name, "other.root"))
    with pytest.raises(ValueError, match="ambiguous card template reference"):
        assemble_package(manifest_path, ordered_path, output_root, expected_per_era_count=2)
    assert not output_root.exists()
    assert not list(tmp_path.glob(".combined.tmp-*"))


def test_multiple_exact_shapes_lines_preserve_other_card_bytes():
    source = b"shapes * * card.root $PROCESS $PROCESS_$SYSTEMATIC\r\nshapes data_obs * card.root data_obs\r\nbin bin_physical\r\n"
    rewritten, count = rewrite_card_template(source, "card.root", "Run2_card.root")
    assert count == 2
    assert rewritten == source.replace(b"card.root", b"Run2_card.root")


def test_existing_output_and_stale_order_fail_closed(tmp_path):
    manifest_path, ordered_path, output_root, _, _ = _fixture(tmp_path)
    ordered_path.write_text("wrong.txt\n")
    with pytest.raises(ValueError, match="ordered card inputs"):
        assemble_package(manifest_path, ordered_path, output_root, expected_per_era_count=2)
    assert not output_root.exists()
    ordered = sorted(json.loads(manifest_path.read_text())["rows"], key=lambda row: row["combined_order_index"])
    ordered_path.write_text("".join(row["destination_txt_name"] + "\n" for row in ordered))
    output_root.mkdir()
    (output_root / "sentinel").write_text("preserve")
    with pytest.raises(FileExistsError, match="already exists"):
        assemble_package(manifest_path, ordered_path, output_root, expected_per_era_count=2)
    assert (output_root / "sentinel").read_text() == "preserve"


def test_duplicate_source_scaling_identity_fails_without_publication(tmp_path):
    manifest_path, ordered_path, output_root, _, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    source_scaling_path = Path(manifest["source_scalings"]["run2"]["path"])
    records = json.loads(source_scaling_path.read_text())
    records.append(dict(records[0]))
    manifest["source_scalings"]["run2"] = _write_json(source_scaling_path, records)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate or unmapped"):
        assemble_package(manifest_path, ordered_path, output_root, expected_per_era_count=2)
    assert not output_root.exists()
    assert not list(tmp_path.glob(".combined.tmp-*"))


def test_manifest_row_must_match_bound_physical_to_ch_mapping(tmp_path):
    manifest_path, ordered_path, output_root, _, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    mapping_path = Path(manifest["source_mappings"]["run2"]["path"])
    mapping = json.loads(mapping_path.read_text())
    mapping["entries"][0]["per_era_chN"] = "ch2"
    manifest["source_mappings"]["run2"] = _write_json(mapping_path, mapping)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from 004J"):
        assemble_package(manifest_path, ordered_path, output_root, expected_per_era_count=2)
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "topeft_combined_datacard_manifest_v1", "schema"),
        ("artifact_type", "wrong_type", "artifact type"),
    ],
)
def test_legacy_schema_and_invalid_artifact_type_fail_closed(tmp_path, field, value, message):
    manifest_path, _, output_root, _, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    with pytest.raises(ValueError, match=message):
        validate_manifest(manifest, output_root, expected_per_era_count=2)


def test_legacy_source_root_key_is_not_a_compatibility_alias(tmp_path):
    manifest_path, _, output_root, _, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["source_004j_package_root"] = manifest.pop("source_per_era_package_root")
    with pytest.raises(ValueError, match="source package root"):
        validate_manifest(manifest, output_root, expected_per_era_count=2)
