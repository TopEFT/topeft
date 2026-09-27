#!/usr/bin/env python3
"""Consolidate receipt-bound datacard metadata for one run era."""

import argparse
import hashlib
import json
import math
import shutil
import uuid
from pathlib import Path


REGISTRY_SCHEMA = "topeft_successful_metadata_units_v1"
PROVENANCE_SCHEMA = "topeft_datacard_metadata_consolidation_v1"
PROVENANCE_FILENAME = "consolidation-provenance.json"


def _read_json(path, label):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc


def _sha256(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot read declared source {path}: {exc}") from exc
    return digest.hexdigest()


def _require_bound_file(unit, path_key, hash_key):
    path = Path(unit[path_key])
    observed = _sha256(path)
    expected = unit[hash_key]
    if observed != expected:
        raise ValueError(
            f"declared hash mismatch for {unit['execution_unit_id']} {path_key}: "
            f"expected {expected}, observed {observed}"
        )
    return path


def _validate_registry(registry, era):
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(f"registry schema must be {REGISTRY_SCHEMA}")
    if registry.get("accepted") is not True:
        raise ValueError("registry is not marked accepted")
    units = [unit for unit in registry.get("units", []) if unit.get("era") == era]
    if not units:
        raise ValueError(f"registry contains no accepted units for {era}")
    identities = [
        (unit.get("execution_unit_id"), unit.get("attempt_id")) for unit in units
    ]
    if len(identities) != len(set(identities)):
        raise ValueError(f"registry repeats an execution unit for {era}")
    return units


def _validate_scaling_records(records, unit):
    if not isinstance(records, list):
        raise ValueError(
            f"scaling fragment for {unit['execution_unit_id']} is not a JSON list"
        )
    owned_channels = {
        f"{target['physical_channel']}_{target['distribution']}"
        for target in unit.get("owned_scientific_target_identities", [])
        if target.get("era") == unit["era"]
    }
    validated = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not {
            "channel", "process", "parameters", "scaling"
        }.issubset(record):
            raise ValueError(
                f"invalid scaling record {index} in {unit['execution_unit_id']}"
            )
        if not isinstance(record["channel"], str) or not record["channel"].strip():
            raise ValueError(
                f"invalid scaling channel at record {index} in "
                f"{unit['execution_unit_id']}"
            )
        if (
            not isinstance(record["process"], str)
            or not record["process"].strip()
        ):
            raise ValueError(
                f"invalid scaling process at record {index} in "
                f"{unit['execution_unit_id']}"
            )
        if not isinstance(record["parameters"], list) or not all(
            isinstance(parameter, str) for parameter in record["parameters"]
        ):
            raise ValueError(
                f"invalid scaling parameters at record {index} in "
                f"{unit['execution_unit_id']}"
            )
        scaling = record["scaling"]
        if not isinstance(scaling, list) or not all(
            isinstance(bin_scaling, list) for bin_scaling in scaling
        ):
            raise ValueError(
                f"invalid scaling structure at record {index} in "
                f"{unit['execution_unit_id']}"
            )
        for bin_scaling in scaling:
            for value in bin_scaling:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f"invalid scaling value at record {index} in "
                        f"{unit['execution_unit_id']}"
                    )
        if record["channel"] not in owned_channels:
            raise ValueError(
                f"scaling channel {record['channel']} is not owned by "
                f"{unit['execution_unit_id']}"
            )
        validated.append(record)
    return validated


def _validate_selected_wcs(selected_wcs, unit):
    if not isinstance(selected_wcs, dict):
        raise ValueError(
            f"selectedWCs fragment for {unit['execution_unit_id']} is not an object"
        )
    for process, wcs in selected_wcs.items():
        if (
            not isinstance(process, str)
            or not isinstance(wcs, list)
            or not all(isinstance(wc, str) for wc in wcs)
        ):
            raise ValueError(
                f"invalid selectedWCs entry for {unit['execution_unit_id']}:{process}"
            )
        if len(wcs) != len(set(wcs)):
            raise ValueError(
                f"duplicate selected WC for {unit['execution_unit_id']}:{process}"
            )
    return selected_wcs


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _make_temporary_sibling(output_dir):
    temporary_dir = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    temporary_dir.mkdir()
    return temporary_dir


def _build_provenance(registry_path, registry, era, source_units, outputs):
    return {
        "schema": PROVENANCE_SCHEMA,
        "era": era,
        "input_registry": {
            "path": str(registry_path),
            "sha256": _sha256(registry_path),
            "source_registry_identity_sha256": registry.get(
                "source_registry_identity_sha256"
            ),
        },
        "consumed_units": [
            {
                "execution_unit_id": unit["execution_unit_id"],
                "attempt_id": unit["attempt_id"],
                "receipt_sha256": unit["receipt_sha256"],
                "scalings_snapshot_sha256": unit["scalings_snapshot_sha256"],
                "selectedWCs_snapshot_sha256": unit[
                    "selectedWCs_snapshot_sha256"
                ],
            }
            for unit in source_units
        ],
        "outputs": outputs,
        "scaling_semantic_key": ["physical_channel", "process"],
        "duplicate_policy": "fail_closed",
        "selectedWCs_policy": "deterministic_per_era_process_to_wc_union",
        "tool_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__)),
        },
    }


def consolidate_metadata(registry_path, era, output_dir):
    """Consolidate exactly the registry-listed metadata units for ``era``."""
    registry_path = Path(registry_path)
    output_dir = Path(output_dir)
    registry = _read_json(registry_path, "registry")
    units = _validate_registry(registry, era)

    records = []
    seen_scaling_identities = set()
    selected_union = {}
    source_units = []

    for unit in units:
        receipt_path = _require_bound_file(unit, "receipt_path", "receipt_sha256")
        scalings_path = _require_bound_file(
            unit, "scalings_snapshot_path", "scalings_snapshot_sha256"
        )
        selected_path = _require_bound_file(
            unit, "selectedWCs_snapshot_path", "selectedWCs_snapshot_sha256"
        )
        fragment = _validate_scaling_records(
            _read_json(scalings_path, "scaling fragment"), unit
        )
        for record in fragment:
            identity = (record["channel"], record["process"])
            if identity in seen_scaling_identities:
                raise ValueError(
                    "duplicate scaling identity "
                    f"(physical_channel={identity[0]}, process={identity[1]})"
                )
            seen_scaling_identities.add(identity)
            records.append(record)

        selected = _validate_selected_wcs(
            _read_json(selected_path, "selectedWCs fragment"), unit
        )
        for process, wcs in selected.items():
            process_union = selected_union.setdefault(process, [])
            for wc in wcs:
                if wc not in process_union:
                    process_union.append(wc)

        source_units.append(
            {
                "era": unit["era"],
                "execution_unit_id": unit["execution_unit_id"],
                "attempt_id": unit["attempt_id"],
                "receipt_path": str(receipt_path),
                "receipt_sha256": unit["receipt_sha256"],
                "scalings_snapshot_path": str(scalings_path),
                "scalings_snapshot_sha256": unit["scalings_snapshot_sha256"],
                "selectedWCs_snapshot_path": str(selected_path),
                "selectedWCs_snapshot_sha256": unit[
                    "selectedWCs_snapshot_sha256"
                ],
            }
        )

    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = _make_temporary_sibling(output_dir)
    try:
        temporary_scalings = temporary_dir / "scalings-preselect.json"
        temporary_selected = temporary_dir / "selectedWCs.txt"
        temporary_provenance = temporary_dir / PROVENANCE_FILENAME
        _write_json(temporary_scalings, records)
        _write_json(temporary_selected, selected_union)
        outputs = {
            "scalings-preselect.json": {"sha256": _sha256(temporary_scalings)},
            "selectedWCs.txt": {"sha256": _sha256(temporary_selected)},
        }
        provenance = _build_provenance(
            registry_path, registry, era, source_units, outputs
        )
        _write_json(temporary_provenance, provenance)

        if _read_json(temporary_scalings, "written scaling output") != records:
            raise ValueError("written scaling output failed readback")
        if (
            _read_json(temporary_selected, "written selectedWCs output")
            != selected_union
        ):
            raise ValueError("written selectedWCs output failed readback")
        if (
            _read_json(temporary_provenance, "written provenance output")
            != provenance
        ):
            raise ValueError("written provenance output failed readback")
        if output_dir.exists():
            raise FileExistsError(f"output directory already exists: {output_dir}")
        temporary_dir.rename(output_dir)
    except BaseException as original_error:
        try:
            shutil.rmtree(temporary_dir)
        except OSError as cleanup_error:
            raise original_error from cleanup_error
        raise

    scalings_output = output_dir / "scalings-preselect.json"
    selected_output = output_dir / "selectedWCs.txt"
    provenance_output = output_dir / PROVENANCE_FILENAME

    return {
        "era": era,
        "registry_path": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "source_registry_identity_sha256": registry.get(
            "source_registry_identity_sha256"
        ),
        "source_units": source_units,
        "source_unit_count": len(source_units),
        "scaling_record_count": len(records),
        "duplicate_scaling_identity_count": 0,
        "selected_wcs_process_count": len(selected_union),
        "scalings_output_path": str(scalings_output),
        "scalings_output_sha256": _sha256(scalings_output),
        "selectedWCs_output_path": str(selected_output),
        "selectedWCs_output_sha256": _sha256(selected_output),
        "provenance_output_path": str(provenance_output),
        "provenance_output_sha256": _sha256(provenance_output),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Consolidate receipt-bound datacard metadata for one era."
    )
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--era", required=True, choices=("run2", "run3"))
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    summary = consolidate_metadata(args.registry, args.era, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
