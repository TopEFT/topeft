#!/usr/bin/env python3
"""Repair the three known Run 2 WWW/WWZ process-label metadata defects.

The command is dry-run only unless ``--write`` and ``--output-dir`` are both
provided.  Write mode publishes validated copies and never mutates or
overwrites an input artifact or its adjacent metadata sidecar.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import cloudpickle
import numpy as np

from topcoffea.modules.utils import get_hist_from_pkl
from topeft.modules.histogram_artifact import (
    build_sumw2_content_manifest,
    histogram_artifact_error,
    metadata_sidecar_path,
    validate_histogram_artifact,
)
from topeft.modules.nominal_schema import histogram_categorical_support
from topeft.modules.sumw2_policy import resolved_policy_from_provenance


RUN2_PROCESS_LABEL_REPAIRS = {
    "WWW_centralUL16APV": "WWW_4F_centralUL16APV",
    "WWW_centralUL17": "WWW_4F_centralUL17",
    "WWZ_centralUL17": "WWZ_4F_centralUL17",
}

_PROCESS_SCALAR_FIELDS = frozenset({"process"})
_PROCESS_LIST_FIELDS = frozenset(
    {
        "consumed_source_processes",
        "eft_nominal_processes",
        "generated_flips_processes",
        "generated_nonprompt_processes",
        "process_names",
        "process_prefixes",
        "prompt_mc",
        "required_processes",
        "required_prompt_signal_processes",
        "required_source_sumw2_processes",
        "required_sumw2_processes",
        "resolved_processes",
        "retained_eft_processes",
        "retained_scalar_processes",
        "scalar_nominal_processes",
        "source_eft_processes",
        "source_scalar_processes",
        "sumw2_processes",
    }
)
_known_legacy_metadata_schema_version = 2
_known_legacy_artifact_kind = "nonprompt_output"
_known_legacy_data_driven_contract_version = 3
_known_legacy_sumw2_provenance_schema_version = 2
_known_legacy_transformation_contract_version = 3
_legacy_process_fragments = ("WWW_central", "WWZ_central")
_non_authoritative_warning_path = (
    "requested_data_driven_products",
    "warnings",
)


class repair_error(RuntimeError):
    """An artifact cannot be repaired without violating the safe-copy contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> tuple[int, str]:
    return path.stat().st_size, _sha256(path)


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise repair_error(f"Could not read repair input sidecar {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise repair_error(f"Repair input sidecar must be a JSON object: {path}")
    return dict(value)


def _is_legacy_like_process_label(value: Any) -> bool:
    return isinstance(value, str) and any(
        fragment in value for fragment in _legacy_process_fragments
    )


def _legacy_like_label_paths(
    value: Any, path: tuple[Any, ...] = ()
) -> list[tuple[Any, ...]]:
    paths = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            paths.extend(_legacy_like_label_paths(child, (*path, key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_legacy_like_label_paths(child, (*path, index)))
    elif _is_legacy_like_process_label(value):
        paths.append(path)
    return paths


def _validate_non_authoritative_warning_text(
    value: Any, path: tuple[Any, ...]
) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise repair_error(
            f"Non-authoritative warning text at {path!r} must be a string list."
        )


def _preflight_typed_metadata(
    value: Any, path: tuple[Any, ...] = ()
) -> set[str]:
    found_labels: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = (*path, key)
            if child_path == _non_authoritative_warning_path:
                _validate_non_authoritative_warning_text(child, child_path)
            elif key == "production_sample_contract":
                occurrences = _legacy_like_label_paths(child, child_path)
                if occurrences:
                    raise repair_error(
                        "A legacy-like label occurs in production_sample_contract, "
                        f"which is not repairable: {occurrences!r}."
                    )
            elif key in _PROCESS_SCALAR_FIELDS:
                if not isinstance(child, str):
                    raise repair_error(
                        f"Typed process field {child_path!r} is not a string."
                    )
                if child in RUN2_PROCESS_LABEL_REPAIRS:
                    found_labels.add(child)
                elif _is_legacy_like_process_label(child):
                    raise repair_error(
                        "Unknown or fuzzy legacy process label in typed metadata "
                        f"field {child_path!r}: {child!r}."
                    )
            elif key in _PROCESS_LIST_FIELDS:
                if not isinstance(child, list) or not all(
                    isinstance(item, str) for item in child
                ):
                    raise repair_error(
                        f"Typed process-list field {child_path!r} is not a string list."
                    )
                for item in child:
                    if item in RUN2_PROCESS_LABEL_REPAIRS:
                        found_labels.add(item)
                    elif _is_legacy_like_process_label(item):
                        raise repair_error(
                            "Unknown or fuzzy legacy process label in typed metadata "
                            f"field {child_path!r}: {item!r}."
                        )
            else:
                found_labels.update(_preflight_typed_metadata(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found_labels.update(_preflight_typed_metadata(child, (*path, index)))
    elif isinstance(value, str) and (
        value in RUN2_PROCESS_LABEL_REPAIRS or _is_legacy_like_process_label(value)
    ):
        raise repair_error(
            f"Legacy process label occurs in unsupported metadata field {path!r}: "
            f"{value!r}."
        )
    return found_labels


def _process_labels(histogram: Any) -> set[str]:
    try:
        return {str(label) for label in histogram.axes["process"]}
    except Exception:
        return set()


def _support_without_process(histogram: Any, process: str) -> set[tuple[Any, ...]]:
    return {
        tuple((axis_name, value) for axis_name, value in cell if axis_name != "process")
        for cell in histogram_categorical_support(histogram)
        if dict(cell).get("process") == process
    }


def _check_process_collisions(histograms: Mapping[str, Any]) -> None:
    collisions = []
    for histogram_name, histogram in histograms.items():
        labels = _process_labels(histogram)
        for source, target in RUN2_PROCESS_LABEL_REPAIRS.items():
            if source not in labels or target not in labels:
                continue
            overlap = _support_without_process(histogram, source) & _support_without_process(
                histogram, target
            )
            if overlap:
                collisions.append(
                    {
                        "histogram": histogram_name,
                        "source": source,
                        "target": target,
                        "overlapping_cells": len(overlap),
                    }
                )
    if collisions:
        raise repair_error(
            "Process-label repair would merge existing categorical support: "
            + json.dumps(collisions, sort_keys=True)
        )


def _numerical_snapshot(
    histograms: Mapping[str, Any], *, apply_mapping: bool
) -> dict[tuple[Any, ...], tuple[str, tuple[int, ...], bytes]]:
    snapshot = {}
    for histogram_name, histogram in histograms.items():
        categorical_axes = tuple(getattr(histogram, "categorical_axes", ()))
        axis_names = tuple(axis.name for axis in categorical_axes)
        if "process" not in axis_names:
            continue
        for categorical_key, dense_values in histogram.view(flow=True).items():
            values = list(tuple(categorical_key))
            process_index = axis_names.index("process")
            if apply_mapping:
                values[process_index] = RUN2_PROCESS_LABEL_REPAIRS.get(
                    str(values[process_index]), str(values[process_index])
                )
            normalized_key = (histogram_name, tuple(zip(axis_names, values)))
            dense_array = np.asarray(dense_values)
            dense_identity = (
                dense_array.dtype.str,
                dense_array.shape,
                dense_array.tobytes(),
            )
            if normalized_key in snapshot:
                raise repair_error(
                    "Process-label repair would create a categorical collision in "
                    f"histogram {histogram_name!r}: {normalized_key[1]!r}."
                )
            snapshot[normalized_key] = dense_identity
    return snapshot


def _repair_histograms(
    histograms: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    repaired = {}
    affected = {}
    for histogram_name, histogram in histograms.items():
        labels = _process_labels(histogram)
        found = sorted(labels & set(RUN2_PROCESS_LABEL_REPAIRS))
        if not found:
            repaired[histogram_name] = histogram
            continue
        groups: dict[str, list[str]] = {}
        for label in sorted(labels):
            target = RUN2_PROCESS_LABEL_REPAIRS.get(label, label)
            groups.setdefault(target, []).append(label)
        repaired[histogram_name] = histogram.group("process", groups)
        affected[histogram_name] = found
    return repaired, affected


def _old_label_paths(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    paths = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            paths.extend(_old_label_paths(child, (*path, key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_old_label_paths(child, (*path, index)))
    elif isinstance(value, str) and value in RUN2_PROCESS_LABEL_REPAIRS:
        paths.append(path)
    return paths


def _repair_typed_metadata(value: Any, path: tuple[Any, ...] = ()) -> Any:
    if isinstance(value, Mapping):
        repaired = {}
        for key, child in value.items():
            child_path = (*path, key)
            if child_path == _non_authoritative_warning_path:
                _validate_non_authoritative_warning_text(child, child_path)
                repaired[key] = copy.deepcopy(child)
            elif key == "production_sample_contract":
                occurrences = _old_label_paths(child, child_path)
                if occurrences:
                    raise repair_error(
                        "An affected label occurs in production_sample_contract, whose "
                        "current schema does not expose a repairable background-process "
                        f"field: {occurrences!r}."
                    )
                repaired[key] = copy.deepcopy(child)
            elif key in _PROCESS_SCALAR_FIELDS:
                if not isinstance(child, str):
                    raise repair_error(f"Typed process field {child_path!r} is not a string.")
                repaired[key] = RUN2_PROCESS_LABEL_REPAIRS.get(child, child)
            elif key in _PROCESS_LIST_FIELDS:
                if not isinstance(child, list) or not all(
                    isinstance(item, str) for item in child
                ):
                    raise repair_error(
                        f"Typed process-list field {child_path!r} is not a string list."
                    )
                mapped = [
                    RUN2_PROCESS_LABEL_REPAIRS.get(item, item) for item in child
                ]
                repaired[key] = sorted(set(mapped)) if mapped != child else list(child)
            else:
                repaired[key] = _repair_typed_metadata(child, child_path)
        return repaired
    if isinstance(value, list):
        return [
            _repair_typed_metadata(child, (*path, index))
            for index, child in enumerate(value)
        ]
    if isinstance(value, str) and value in RUN2_PROCESS_LABEL_REPAIRS:
        raise repair_error(
            f"Affected label occurs in unsupported metadata field {path!r}; refusing to guess."
        )
    return copy.deepcopy(value)


def _canonicalize_repaired_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    canonical = copy.deepcopy(dict(provenance))
    canonical["resolved_processes"] = sorted(set(canonical["resolved_processes"]))
    family_order = {
        family: index
        for index, family in enumerate(canonical["runtime_histogram_families"])
    }
    targets = canonical["resolved_targets"]
    target_identities = {
        (target["dataset"], target["process"], target["family"])
        for target in targets
    }
    if len(target_identities) != len(targets):
        raise repair_error(
            "Process-label repair would create duplicate sumw2 resolved targets."
        )
    canonical["resolved_targets"] = sorted(
        targets,
        key=lambda target: (
            target["dataset"],
            target["process"],
            family_order[target["family"]],
        ),
    )
    return canonical


def _repair_sidecar(
    sidecar: Mapping[str, Any], repaired_histograms: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    original_contract = copy.deepcopy(sidecar.get("production_sample_contract"))
    old_manifest = sidecar.get("sumw2_content_manifest")
    if not isinstance(old_manifest, Mapping):
        raise repair_error("Sidecar lacks a valid sumw2_content_manifest mapping.")
    required = {
        family: [
            RUN2_PROCESS_LABEL_REPAIRS.get(process, process)
            for process in family_manifest.get("required_sumw2_processes", [])
        ]
        for family, family_manifest in old_manifest.get("families", {}).items()
    }

    repair_input = copy.deepcopy(dict(sidecar))
    repair_input.pop("sumw2_content_manifest", None)
    repaired = _repair_typed_metadata(repair_input)
    provenance = repaired.get("sumw2_storage_provenance")
    if not isinstance(provenance, Mapping):
        raise repair_error("Sidecar lacks valid sumw2_storage_provenance.")
    repaired["sumw2_storage_provenance"] = resolved_policy_from_provenance(
        _canonicalize_repaired_provenance(provenance)
    ).to_provenance()
    artifact = repaired.get("artifact")
    if not isinstance(artifact, Mapping):
        raise repair_error("Sidecar lacks a valid artifact identity mapping.")
    artifact_kind = artifact.get("artifact_kind")
    repaired["sumw2_content_manifest"] = build_sumw2_content_manifest(
        repaired_histograms,
        sumw2_storage_provenance=repaired["sumw2_storage_provenance"],
        artifact_kind=artifact_kind,
        required_sumw2_processes=required,
    )
    if repaired.get("production_sample_contract") != original_contract:
        raise repair_error("production_sample_contract changed unexpectedly.")
    remaining = _old_label_paths(repaired)
    if remaining:
        raise repair_error(
            "Affected labels remain in unsupported sidecar locations: "
            f"{remaining!r}."
        )
    old_paths = _old_label_paths(sidecar)
    surfaces = sorted({str(path[0]) for path in old_paths if path})
    return repaired, surfaces


def _load_known_legacy_sidecar(
    input_path: Path,
    sidecar_path: Path,
    input_pkl_identity: tuple[int, str],
) -> tuple[dict[str, Any], set[str]]:
    sidecar = _read_json_mapping(sidecar_path)
    expected_sidecar_fields = {
        "metadata_schema_version",
        "artifact",
        "sumw2_storage_provenance",
        "sumw2_content_manifest",
        "lineage",
        "production_sample_contract",
        "requested_data_driven_products",
        "resolved_data_driven_contract",
        "transformation_contract",
    }
    if set(sidecar) != expected_sidecar_fields:
        raise repair_error(
            "Known legacy repair requires the exact supported sidecar field shape: "
            f"missing={sorted(expected_sidecar_fields - set(sidecar))} "
            f"unknown={sorted(set(sidecar) - expected_sidecar_fields)}."
        )
    if sidecar["metadata_schema_version"] != _known_legacy_metadata_schema_version:
        raise repair_error("Known legacy repair requires metadata schema version 2.")

    artifact = sidecar["artifact"]
    expected_artifact_fields = {
        "pkl_basename",
        "pkl_size_bytes",
        "pkl_sha256",
        "artifact_kind",
        "merged",
        "nominal_container_schema_version",
        "nominal_container_layout",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != expected_artifact_fields:
        raise repair_error("Known legacy repair requires the exact artifact identity shape.")
    if artifact["artifact_kind"] != _known_legacy_artifact_kind:
        raise repair_error(
            "Canonical-validation bootstrap is supported only for nonprompt_output artifacts."
        )
    observed_pkl_size, observed_pkl_sha256 = input_pkl_identity
    if (
        artifact["pkl_basename"] != input_path.name
        or artifact["pkl_size_bytes"] != observed_pkl_size
        or artifact["pkl_sha256"] != observed_pkl_sha256
    ):
        raise repair_error(
            "Raw sidecar artifact identity does not match the exact input PKL "
            f"basename/size/SHA: {input_path}."
        )

    provenance = sidecar["sumw2_storage_provenance"]
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("schema_version")
        != _known_legacy_sumw2_provenance_schema_version
    ):
        raise repair_error("Known legacy repair requires sumw2 provenance schema version 2.")
    contract = sidecar["resolved_data_driven_contract"]
    if not isinstance(contract, Mapping) or set(contract) != {
        "contract_version",
        "required_prompt_signal_processes",
        "products",
    }:
        raise repair_error(
            "Known legacy repair requires the exact pre-canonical data-driven contract shape."
        )
    if contract["contract_version"] != _known_legacy_data_driven_contract_version:
        raise repair_error("Known legacy repair requires data-driven contract version 3.")
    products = contract["products"]
    if (
        not isinstance(products, Mapping)
        or not isinstance(products.get("nonprompt"), Mapping)
        or products["nonprompt"].get("enabled") is not True
    ):
        raise repair_error("Known legacy repair requires an enabled nonprompt contract.")
    transformation_contract = sidecar["transformation_contract"]
    if (
        not isinstance(transformation_contract, Mapping)
        or transformation_contract.get("contract_version")
        != _known_legacy_transformation_contract_version
        or transformation_contract.get("artifact_kind")
        != _known_legacy_artifact_kind
    ):
        raise repair_error(
            "Known legacy repair requires a version-3 nonprompt transformation contract."
        )

    metadata_labels = _preflight_typed_metadata(sidecar)
    if not metadata_labels:
        raise repair_error(
            "Canonical validation failed, but the sidecar contains no exact maintained "
            "Run-2 process label eligible for this repair."
        )
    return sidecar, metadata_labels


def _preflight_payload_labels(
    histograms: Mapping[str, Any], metadata_labels: set[str]
) -> None:
    payload_labels = set()
    unknown_legacy_labels = set()
    for histogram in histograms.values():
        for label in _process_labels(histogram):
            if label in RUN2_PROCESS_LABEL_REPAIRS:
                payload_labels.add(label)
            elif _is_legacy_like_process_label(label):
                unknown_legacy_labels.add(label)
    if unknown_legacy_labels:
        raise repair_error(
            "Payload contains unknown or fuzzy legacy process labels: "
            + ", ".join(sorted(unknown_legacy_labels))
        )
    if payload_labels != metadata_labels:
        raise repair_error(
            "Legacy process labels claimed by metadata and materialized by payload "
            "process axes differ: "
            f"metadata={sorted(metadata_labels)} payload={sorted(payload_labels)}."
        )


def _validate_repaired_representation(
    input_path: Path,
    repaired_histograms: Mapping[str, Any],
    repaired_sidecar: Mapping[str, Any],
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".process-label-repair-validation-"
    ) as temporary_root:
        validation_path = Path(temporary_root) / input_path.name
        validation_path.symlink_to(input_path)
        validation_sidecar_path = metadata_sidecar_path(validation_path)
        validation_sidecar_path.write_text(
            json.dumps(repaired_sidecar, indent=2) + "\n",
            encoding="utf-8",
        )
        validation = validate_histogram_artifact(
            validation_path,
            histograms=repaired_histograms,
        )
        if validation["schema"] == "legacy_uniform" or validation["metadata"] is None:
            raise repair_error(
                f"Repaired representation is not a canonical split artifact: {input_path}"
            )


def _load_and_prepare(input_path: Path, output_dir: Path | None) -> dict[str, Any]:
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise repair_error(f"Input histogram artifact does not exist: {input_path}")
    sidecar_path = metadata_sidecar_path(input_path)
    if not sidecar_path.is_file():
        raise repair_error(f"Required adjacent sidecar does not exist: {sidecar_path}")
    input_identity = (_file_identity(input_path), _file_identity(sidecar_path))
    histograms = dict(get_hist_from_pkl(str(input_path)))
    try:
        validation = validate_histogram_artifact(input_path, histograms=histograms)
    except histogram_artifact_error:
        sidecar, metadata_labels = _load_known_legacy_sidecar(
            input_path,
            sidecar_path,
            input_identity[0],
        )
        _preflight_payload_labels(histograms, metadata_labels)
        input_validation_mode = "known_repairable_legacy"
    else:
        if validation["schema"] == "legacy_uniform" or validation["metadata"] is None:
            raise repair_error(f"Unsupported legacy artifact: {input_path}")
        sidecar = validation["metadata"]
        input_validation_mode = "already_canonical"
    _check_process_collisions(histograms)
    expected_numerical = _numerical_snapshot(histograms, apply_mapping=True)
    repaired_histograms, affected = _repair_histograms(histograms)
    observed_numerical = _numerical_snapshot(
        repaired_histograms, apply_mapping=False
    )
    if expected_numerical != observed_numerical:
        raise repair_error(
            f"Numerical content changed while repairing process labels in {input_path}."
        )
    repaired_sidecar, sidecar_surfaces = _repair_sidecar(
        sidecar, repaired_histograms
    )
    _validate_repaired_representation(
        input_path,
        repaired_histograms,
        repaired_sidecar,
    )
    found_labels = sorted(
        {
            label
            for labels in affected.values()
            for label in labels
        }
    )
    absent_labels = sorted(set(RUN2_PROCESS_LABEL_REPAIRS) - set(found_labels))
    output_path = (
        output_dir.resolve() / input_path.name
        if output_dir is not None
        else input_path.parent / "corrected" / input_path.name
    )
    output_sidecar_path = metadata_sidecar_path(output_path)
    if output_path == input_path or output_sidecar_path == sidecar_path:
        raise repair_error(
            "Output path resolves to an input artifact; in-place repair is forbidden."
        )
    return {
        "input_path": input_path,
        "sidecar_path": sidecar_path,
        "input_identity": input_identity,
        "input_validation_mode": input_validation_mode,
        "histograms": repaired_histograms,
        "sidecar": repaired_sidecar,
        "mapping_entries_found": {
            label: RUN2_PROCESS_LABEL_REPAIRS[label] for label in found_labels
        },
        "mapping_entries_absent": absent_labels,
        "payload_histograms_affected": affected,
        "sidecar_surfaces_affected": sidecar_surfaces,
        "repaired_representation_validation": (
            "passed_unchanged_validate_histogram_artifact"
        ),
        "output_path": output_path,
        "output_sidecar_path": output_sidecar_path,
    }


def _summary(prepared: Mapping[str, Any], *, write_performed: bool) -> dict[str, Any]:
    return {
        "input_path": str(prepared["input_path"]),
        "sidecar_path": str(prepared["sidecar_path"]),
        "input_validation_mode": prepared["input_validation_mode"],
        "mapping_entries_found": prepared["mapping_entries_found"],
        "mapping_entries_absent": prepared["mapping_entries_absent"],
        "payload_histograms_affected": prepared["payload_histograms_affected"],
        "sidecar_surfaces_affected": prepared["sidecar_surfaces_affected"],
        "repaired_representation_validation": prepared[
            "repaired_representation_validation"
        ],
        "collision_check": "passed_no_overlapping_categorical_support",
        "numerical_invariance_plan": "exact_remapped_categorical_cell_array_equality",
        "would_write_output_path": str(prepared["output_path"]),
        "would_write_sidecar_path": str(prepared["output_sidecar_path"]),
        "write_performed": write_performed,
    }


def _stage_prepared(prepared: dict[str, Any], staging_root: Path) -> tuple[Path, Path]:
    staged_path = staging_root / prepared["output_path"].name
    staged_sidecar = metadata_sidecar_path(staged_path)
    with gzip.open(staged_path, "wb") as stream:
        cloudpickle.dump(prepared["histograms"], stream)
    repaired_sidecar = copy.deepcopy(prepared["sidecar"])
    repaired_sidecar["artifact"]["pkl_basename"] = prepared["output_path"].name
    repaired_sidecar["artifact"]["pkl_size_bytes"] = staged_path.stat().st_size
    repaired_sidecar["artifact"]["pkl_sha256"] = _sha256(staged_path)
    staged_sidecar.write_text(
        json.dumps(repaired_sidecar, indent=2) + "\n",
        encoding="utf-8",
    )
    validate_histogram_artifact(staged_path, histograms=prepared["histograms"])
    return staged_path, staged_sidecar


def repair_artifacts(
    input_paths: list[str | os.PathLike[str]],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    write: bool = False,
) -> list[dict[str, Any]]:
    """Validate repair plans and optionally publish separate corrected copies."""

    if not input_paths:
        raise repair_error("At least one input artifact is required.")
    if write and output_dir is None:
        raise repair_error("Write mode requires an explicit --output-dir.")
    resolved_output_dir = Path(output_dir).resolve() if output_dir is not None else None
    prepared = [
        _load_and_prepare(Path(input_path), resolved_output_dir)
        for input_path in input_paths
    ]
    output_paths = [item["output_path"] for item in prepared]
    if len(output_paths) != len(set(output_paths)):
        raise repair_error("Multiple inputs resolve to the same output path.")
    if not write:
        return [_summary(item, write_performed=False) for item in prepared]

    assert resolved_output_dir is not None
    for item in prepared:
        if item["output_path"].exists() or item["output_sidecar_path"].exists():
            raise repair_error(
                "Refusing to overwrite existing corrected output: "
                f"{item['output_path']} or {item['output_sidecar_path']}"
            )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    temporary_roots = []
    published: list[Path] = []
    try:
        for item in prepared:
            temporary_root = tempfile.TemporaryDirectory(
                prefix=".process-label-repair-", dir=resolved_output_dir
            )
            temporary_roots.append(temporary_root)
            staged.append(
                _stage_prepared(item, Path(temporary_root.name))
            )
        for item, (staged_path, staged_sidecar) in zip(prepared, staged):
            os.replace(staged_path, item["output_path"])
            published.append(item["output_path"])
            os.replace(staged_sidecar, item["output_sidecar_path"])
            published.append(item["output_sidecar_path"])
        for item in prepared:
            validate_histogram_artifact(
                item["output_path"], histograms=item["histograms"]
            )
            current_identity = (
                _file_identity(item["input_path"]),
                _file_identity(item["sidecar_path"]),
            )
            if current_identity != item["input_identity"]:
                raise repair_error(
                    f"Input artifact changed during repair: {item['input_path']}"
                )
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        for temporary_root in temporary_roots:
            temporary_root.cleanup()
    return [_summary(item, write_performed=True) for item in prepared]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_paths", nargs="+", help="Histogram PKLs to inspect")
    parser.add_argument(
        "--output-dir",
        help="Separate destination directory; required with --write",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Publish validated corrected copies (default: dry-run only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summaries = repair_artifacts(
            args.input_paths,
            output_dir=args.output_dir,
            write=args.write,
        )
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", "artifacts": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
