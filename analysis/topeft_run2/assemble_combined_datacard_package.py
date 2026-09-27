#!/usr/bin/env python3
"""Assemble an explicit-order Run 2 + Run 3 datacard package.

This tool changes package filenames, card template references, and scaling
channel labels. It does not interpret or alter nuisance or physics content.
"""

import argparse
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path


package_schema = "TOP26006_v1"
manifest_artifact_type = "combined_mapping_manifest"
provenance_artifact_type = "package_provenance"
naming_policy = "era_prefix_v1: Run2_/Run3_ + exact source basename"
order_policy = "run2: N; run3: 129+N for certified per-era chN"
shapes_line_pattern = re.compile(r"^([ \t]*shapes[ \t]+\S+[ \t]+\S+[ \t]+)(\S+)")
channel_label_pattern = re.compile(r"ch([1-9][0-9]*)\Z")
physical_name_pattern = re.compile(r"[A-Za-z0-9_]+\Z")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _require_bound_file(bound_file, label):
    path = Path(bound_file["path"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{label} must be an existing absolute file")
    if _sha256(path) != bound_file["sha256"]:
        raise ValueError(f"{label} SHA-256 differs from the manifest")
    return path


def _require_unique(values, label):
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


def validate_manifest(manifest, output_root, expected_per_era_count=129):
    """Return rows in declared combined order after all input gates pass."""
    if manifest.get("schema") != package_schema:
        raise ValueError("unsupported combined manifest schema")
    if manifest.get("artifact_type") != manifest_artifact_type:
        raise ValueError("unsupported combined manifest artifact type")
    if manifest.get("destination_naming_policy") != naming_policy:
        raise ValueError("unsupported destination naming policy")
    if manifest.get("combined_order_policy") != order_policy:
        raise ValueError("unsupported combined order policy")
    if manifest.get("selectedWCs_combined_rule") != "outside_current_packaging_boundary":
        raise ValueError("unsupported combined selectedWCs policy")
    output_root = Path(output_root)
    if not output_root.is_absolute() or Path(manifest.get("destination_package_root", "")) != output_root:
        raise ValueError("output root differs from the manifest")
    source_root = Path(manifest.get("source_per_era_package_root", ""))
    if not source_root.is_absolute() or not source_root.is_dir():
        raise ValueError("source package root is absent")
    mapping_entries = {}
    for era in ("run2", "run3"):
        mapping_path = _require_bound_file(manifest["source_mappings"][era], f"{era} mapping")
        source_mapping = _read_json(mapping_path)
        if source_mapping.get("era") != era or not isinstance(source_mapping.get("entries"), list):
            raise ValueError(f"{era} source mapping is invalid")
        mapping_entries[era] = {
            entry["physical_name"]: entry for entry in source_mapping["entries"]
        }
        if len(mapping_entries[era]) != expected_per_era_count:
            raise ValueError(f"{era} source mapping has duplicate or missing physical names")
        scaling_path = _require_bound_file(manifest["source_scalings"][era], f"{era} scalings")
        expected_dir = source_root / era / "ptz-lj0pt_withSys"
        if scaling_path.parent.resolve() != expected_dir.resolve():
            raise ValueError(f"{era} scalings outside declared final directory")

    rows = manifest.get("rows")
    total = expected_per_era_count * 2
    if not isinstance(rows, list) or len(rows) != total:
        raise ValueError(f"manifest requires {total} rows")
    by_order = sorted(rows, key=lambda row: row["combined_order_index"])
    if [row["combined_order_index"] for row in by_order] != list(range(1, total + 1)):
        raise ValueError("combined order indexes are incomplete or duplicate")
    _require_unique([row["combined_chN"] for row in by_order], "combined_chN")
    _require_unique([(row["era"], row["physical_name"]) for row in by_order], "physical identity")
    for key in ("source_txt_path", "source_root_path", "destination_txt_name", "destination_root_name"):
        _require_unique([row[key] for row in by_order], key)

    for era in ("run2", "run3"):
        era_rows = [row for row in by_order if row["era"] == era]
        if len(era_rows) != expected_per_era_count:
            raise ValueError(f"{era} row count mismatch")
        if {row["per_era_ch_index"] for row in era_rows} != set(range(1, expected_per_era_count + 1)):
            raise ValueError(f"{era} per-era channel domain mismatch")
        if {row["physical_name"] for row in era_rows} != set(mapping_entries[era]):
            raise ValueError(f"{era} manifest physical domain differs from 004J mapping")

    for row in by_order:
        era = row["era"]
        if era not in ("run2", "run3"):
            raise ValueError("unsupported era")
        local_index = row["per_era_ch_index"]
        if not isinstance(local_index, int) or isinstance(local_index, bool):
            raise ValueError("per-era index must be an integer")
        match = channel_label_pattern.fullmatch(row["per_era_chN"])
        if match is None or int(match.group(1)) != local_index:
            raise ValueError("per-era chN/index mismatch")
        combined_index = local_index if era == "run2" else expected_per_era_count + local_index
        if row["combined_order_index"] != combined_index or row["combined_chN"] != f"ch{combined_index}":
            raise ValueError("combined chN/order differs from the canonical mapping")
        physical_name = row["physical_name"]
        if physical_name_pattern.fullmatch(physical_name) is None:
            raise ValueError("unsafe physical name")
        if physical_name != f"{row['physical_channel']}_{row['distribution']}":
            raise ValueError("physical channel/distribution mismatch")
        source_entry = mapping_entries[era][physical_name]
        if any(
            row[key] != source_entry[key]
            for key in ("physical_name", "physical_channel", "distribution", "per_era_chN")
        ):
            raise ValueError("manifest row differs from 004J physical-to-chN mapping")
        basename = f"ttx_multileptons-{physical_name}"
        source_dir = source_root / era / "ptz-lj0pt_withSys"
        source_txt = Path(row["source_txt_path"])
        source_root_file = Path(row["source_root_path"])
        if source_txt != source_dir / f"{basename}.txt" or source_root_file != source_dir / f"{basename}.root":
            raise ValueError("source card/template path differs from physical identity")
        if not source_txt.is_file() or not source_root_file.is_file():
            raise ValueError("missing source card/template pair")
        prefix = "Run2_" if era == "run2" else "Run3_"
        if row["destination_txt_name"] != f"{prefix}{basename}.txt":
            raise ValueError("destination TXT naming policy mismatch")
        if row["destination_root_name"] != f"{prefix}{basename}.root":
            raise ValueError("destination ROOT naming policy mismatch")
        if row["expected_template_reference_after_packaging"] != row["destination_root_name"]:
            raise ValueError("expected template reference mismatch")
        if row["destination_naming_policy"] != naming_policy:
            raise ValueError("row destination naming policy mismatch")
    return by_order


def rewrite_card_template(source_bytes, source_root_name, destination_root_name):
    """Change only exact `shapes` template path tokens for one source ROOT."""
    try:
        text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("card is not UTF-8") from exc
    result = []
    shape_count = 0
    for line in text.splitlines(keepends=True):
        if re.match(r"^[ \t]*shapes(?:[ \t]|$)", line):
            match = shapes_line_pattern.match(line)
            if match is None or match.group(2) != source_root_name:
                raise ValueError("ambiguous card template reference")
            line = line[:match.start(2)] + destination_root_name + line[match.end(2):]
            shape_count += 1
        result.append(line)
    if shape_count == 0:
        raise ValueError("card contains no source template reference")
    return "".join(result).encode("utf-8"), shape_count


def _write_scalings(manifest, rows, destination):
    channel_map = {
        (row["era"], row["per_era_chN"]): row["combined_chN"]
        for row in rows
    }
    seen_source = set()
    seen_combined = set()
    record_counts = {}
    with destination.open("x", encoding="utf-8") as stream:
        stream.write("[\n")
        first = True
        for era in ("run2", "run3"):
            source_path = Path(manifest["source_scalings"][era]["path"])
            records = _read_json(source_path)
            if not isinstance(records, list):
                raise ValueError(f"{era} scalings must be a JSON array")
            record_counts[era] = 0
            for record in records:
                if not isinstance(record, dict) or not {"channel", "process", "parameters", "scaling"}.issubset(record):
                    raise ValueError("invalid source scaling record")
                channel = record["channel"]
                process = record["process"]
                if not isinstance(process, str) or not process:
                    raise ValueError("invalid scaling process")
                source_key = (era, channel, process)
                if source_key in seen_source or (era, channel) not in channel_map:
                    raise ValueError("duplicate or unmapped source scaling identity")
                combined_channel = channel_map[(era, channel)]
                combined_key = (combined_channel, process)
                if combined_key in seen_combined:
                    raise ValueError("duplicate combined scaling identity")
                seen_source.add(source_key)
                seen_combined.add(combined_key)
                transformed = dict(record)
                transformed["channel"] = combined_channel
                stream.write("" if first else ",\n")
                stream.write(json.dumps(transformed, separators=(",", ":"), allow_nan=False))
                first = False
                record_counts[era] += 1
            del records
        stream.write("\n]\n")
    return record_counts


def _package_readme():
    return (
        "# Combined Run2+Run3 datacard package\n\n"
        "The manifest owns physical identity and combined ch1..ch258 mapping. "
        "The ordered input list owns the card order consumed downstream.\n\n"
        "From this package root, run later in an authorized consumer environment:\n\n"
        "```bash\n"
        "mapfile -t cards < ordered_card_inputs.txt\n"
        'combineCards.py "${cards[@]}" > combinedcard.txt\n'
        "```\n\n"
        "The ordered-input convention is new TOP-26-006 hardening. It is not "
        "attributed to Andrew. Do not use a shell glob for this package. "
        "No combined selectedWCs.txt is defined by this packaging boundary.\n"
    )


def assemble_package(manifest_path, ordered_path, output_root, expected_per_era_count=129):
    """Build and atomically publish one fresh package from an explicit manifest."""
    manifest_path = Path(manifest_path)
    ordered_path = Path(ordered_path)
    output_root = Path(output_root)
    manifest = _read_json(manifest_path)
    rows = validate_manifest(manifest, output_root, expected_per_era_count)
    expected_order = "".join(row["destination_txt_name"] + "\n" for row in rows)
    if ordered_path.read_text(encoding="utf-8") != expected_order:
        raise ValueError("ordered card inputs differ from the manifest")
    if output_root.exists():
        raise FileExistsError(f"combined package root already exists: {output_root}")
    if not output_root.parent.is_dir():
        raise ValueError("combined package parent does not exist")
    temporary_root = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary_root.mkdir()
    try:
        shape_line_count = 0
        for row in rows:
            source_txt = Path(row["source_txt_path"])
            source_root_file = Path(row["source_root_path"])
            rewritten, count = rewrite_card_template(
                source_txt.read_bytes(),
                source_root_file.name,
                row["destination_root_name"],
            )
            (temporary_root / row["destination_txt_name"]).write_bytes(rewritten)
            shutil.copyfile(source_root_file, temporary_root / row["destination_root_name"])
            shape_line_count += count
        record_counts = _write_scalings(manifest, rows, temporary_root / "scalings.json")
        shutil.copyfile(manifest_path, temporary_root / "combined_mapping_manifest.json")
        (temporary_root / "ordered_card_inputs.txt").write_text(expected_order, encoding="utf-8")
        (temporary_root / "README.md").write_text(_package_readme(), encoding="utf-8")
        provenance = {
            "schema": package_schema,
            "artifact_type": provenance_artifact_type,
            "source_per_era_package_root": manifest["source_per_era_package_root"],
            "destination_package_root": str(output_root),
            "destination_naming_policy": naming_policy,
            "combined_order_policy": manifest["combined_order_policy"],
            "selectedWCs_combined_rule": "outside_current_packaging_boundary",
            "historical_glob_replaced": True,
            "ordered_inputs_origin": "new TOP-26-006 hardening; not Andrew-authored",
            "manifest_sha256": _sha256(temporary_root / "combined_mapping_manifest.json"),
            "ordered_card_inputs_sha256": _sha256(temporary_root / "ordered_card_inputs.txt"),
            "combined_scalings_sha256": _sha256(temporary_root / "scalings.json"),
            "source_mappings": manifest["source_mappings"],
            "source_scalings": manifest["source_scalings"],
            "assembler_source_sha256": _sha256(__file__),
            "packaged_txt_count": len(rows),
            "packaged_root_count": len(rows),
            "source_scaling_record_counts": record_counts,
            "rewritten_shapes_line_count": shape_line_count,
        }
        (temporary_root / "package-provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        observed_txt = {path.name for path in temporary_root.iterdir() if re.fullmatch(r"Run[23]_.*\.txt", path.name)}
        observed_root = {path.name for path in temporary_root.iterdir() if path.suffix == ".root"}
        if observed_txt != {row["destination_txt_name"] for row in rows}:
            raise ValueError("staging TXT set differs from manifest")
        if observed_root != {row["destination_root_name"] for row in rows}:
            raise ValueError("staging ROOT set differs from manifest")
        if (temporary_root / "ordered_card_inputs.txt").read_text(encoding="utf-8") != expected_order:
            raise ValueError("staging order readback mismatch")
        if output_root.exists():
            raise FileExistsError(f"combined package root appeared: {output_root}")
        temporary_root.rename(output_root)
        return provenance
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ordered-card-inputs", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    provenance = assemble_package(args.manifest, args.ordered_card_inputs, args.output_root)
    print(json.dumps({"package_root": args.output_root.as_posix(), "provenance": provenance}, sort_keys=True))


if __name__ == "__main__":
    main()
