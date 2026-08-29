"""Automatic sidecars and stage-aware validation for histogram pickle artifacts."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
import uuid
from typing import Any

import cloudpickle

from topeft.modules.data_driven_products import (
    CANONICAL_DATA_DRIVEN_YEARS,
    FLIPS_OUTPUT_ARTIFACT_KIND,
    NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND,
    NONPROMPT_OUTPUT_ARTIFACT_KIND,
    PRECANONICAL_RESOLVED_DATA_DRIVEN_CONTRACT_VERSION,
    RESOLVED_DATA_DRIVEN_CONTRACT_VERSION,
    TRANSFORMED_DATA_DRIVEN_ARTIFACT_KINDS,
    data_driven_product_error,
    generated_output_processes_from_contract,
    resolved_prompt_processes_from_contract,
    validate_requested_product_input,
    validate_serialized_data_driven_contract,
)
from topeft.modules.nonprompt_policy import (
    certify_active_nonprompt_policy,
    nonprompt_policy_error,
)
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.nominal_schema import (
    NOMINAL_CONTAINER_LAYOUT,
    NOMINAL_CONTAINER_SCHEMA_VERSION,
    eft_nominal_key,
    is_split_nominal_mapping,
    scalar_nominal_key,
    sumw2_key,
    validate_nominal_mapping,
)
from topeft.modules.sumw2_policy import resolved_policy_from_provenance
from topeft.modules.sumw2_policy import SUMW2_PROVENANCE_SCHEMA_VERSION
from topeft.modules.production_sample_profile import (
    validate_production_sample_contract,
)


METADATA_SCHEMA_VERSION = 2
SUMW2_CONTENT_MANIFEST_VERSION = 1
PRIOR_TRANSFORMATION_CONTRACT_VERSION = 2
TRANSFORMATION_CONTRACT_VERSION = 3
ARTIFACT_KINDS = frozenset(
    {"processor_output", *TRANSFORMED_DATA_DRIVEN_ARTIFACT_KINDS}
)
NONPROMPT_APPLICATION_REGIONS = frozenset(
    {
        "isAR_1l",
        "isAR_2lSS",
        "isAR_2lOS",
        "isAR_3l",
    }
)
FLIPS_APPLICATION_REGION = "isAR_2lSS_OS"
SUPPORTED_DATA_DRIVEN_PRODUCTS = ("nonprompt", "flips")
_PRODUCER_CONTEXT_TOKEN = object()
NOMINAL_REFERENCE_CONTRACT_VERSION = 1


def derive_data_driven_applicability(
    source_application_regions: Iterable[str],
) -> dict[str, bool]:
    """Derive maintained product applicability from source application labels."""

    regions = {str(region) for region in source_application_regions}
    return {
        "nonprompt": bool(regions & NONPROMPT_APPLICATION_REGIONS),
        "flips": FLIPS_APPLICATION_REGION in regions,
    }


class histogram_artifact_error(RuntimeError):
    """Base error for histogram artifact metadata failures."""


class histogram_sidecar_error(histogram_artifact_error):
    """A sidecar is absent, malformed, or paired with the wrong pickle."""


class histogram_content_error(histogram_artifact_error):
    """A serialized histogram payload disagrees with its generated manifest."""


class histogram_merge_error(histogram_artifact_error):
    """Input sidecars cannot describe one valid merged artifact."""


def metadata_sidecar_path(pkl_path: str | os.PathLike[str]) -> Path:
    """Return the one canonical sidecar path without replacing any suffix."""

    return Path(f"{os.fspath(pkl_path)}.metadata.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(identity_path: Path, *, pkl_basename: str) -> dict[str, Any]:
    return {
        "pkl_basename": pkl_basename,
        "pkl_size_bytes": identity_path.stat().st_size,
        "pkl_sha256": _sha256(identity_path),
    }


def _process_labels(histogram: Any | None) -> list[str]:
    if histogram is None:
        return []
    try:
        return sorted({str(value) for value in histogram.axes["process"]})
    except Exception:
        return []


def _family_process_content(
    histograms: Mapping[str, Any],
    family: str,
) -> dict[str, list[str] | int]:
    dimensionality = 2 if family in axes_info_2d else 1
    if dimensionality == 2:
        scalar = histograms.get(family)
        eft = None
    else:
        scalar = histograms.get(scalar_nominal_key(family))
        eft = histograms.get(eft_nominal_key(family))
    companion = histograms.get(sumw2_key(family))
    return {
        "dimensionality": dimensionality,
        "scalar_nominal_processes": _process_labels(scalar),
        "eft_nominal_processes": _process_labels(eft),
        "sumw2_processes": _process_labels(companion),
    }


def _normalize_required_processes(
    required_sumw2_processes: Mapping[str, Iterable[str]] | None,
) -> dict[str, list[str]]:
    output = {}
    for family, processes in (required_sumw2_processes or {}).items():
        output[str(family)] = sorted({str(process) for process in processes})
    return output


def build_sumw2_content_manifest(
    histograms: Mapping[str, Any],
    *,
    sumw2_storage_provenance: Mapping[str, Any],
    artifact_kind: str,
    required_sumw2_processes: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Describe actual nominal and companion process content deterministically."""

    if artifact_kind not in ARTIFACT_KINDS:
        raise histogram_sidecar_error(
            f"Unknown histogram artifact kind {artifact_kind!r}."
        )
    policy = resolved_policy_from_provenance(sumw2_storage_provenance)
    explicit_required = _normalize_required_processes(required_sumw2_processes)
    unknown_required = sorted(
        set(explicit_required) - set(policy.runtime_histogram_families)
    )
    if unknown_required:
        raise histogram_sidecar_error(
            "Required sumw2 process mapping contains unknown families: "
            + ", ".join(unknown_required)
        )

    families = {}
    for family in policy.runtime_histogram_families:
        content = _family_process_content(histograms, family)
        if family in explicit_required:
            required = explicit_required[family]
        elif artifact_kind == "processor_output":
            nominal_processes = set(content["scalar_nominal_processes"]) | set(
                content["eft_nominal_processes"]
            )
            required = sorted(
                nominal_processes & set(policy.selected_processes(family))
            )
        else:
            raise histogram_sidecar_error(
                f"{artifact_kind} requires independently derived "
                "required_sumw2_processes; observed companion content cannot "
                "define the requirement."
            )
        families[family] = {
            **content,
            "required_sumw2_processes": required,
        }
    return {
        "manifest_version": SUMW2_CONTENT_MANIFEST_VERSION,
        "families": families,
    }


def _normalize_lineage_inputs(
    lineage_inputs: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    normalized = []
    expected_keys = {"pkl_basename", "artifact_kind", "pkl_sha256"}
    for raw_input in lineage_inputs:
        if not isinstance(raw_input, Mapping) or set(raw_input) != expected_keys:
            raise histogram_sidecar_error(
                "Lineage inputs must contain exactly pkl_basename, artifact_kind, and pkl_sha256."
            )
        values = {key: raw_input[key] for key in sorted(expected_keys)}
        if any(not isinstance(value, str) or not value for value in values.values()):
            raise histogram_sidecar_error(
                "Lineage input identity fields must be nonempty strings."
            )
        if values["artifact_kind"] not in ARTIFACT_KINDS:
            raise histogram_sidecar_error(
                f"Unknown lineage artifact kind {values['artifact_kind']!r}."
            )
        normalized.append(values)
    normalized.sort(
        key=lambda item: (
            item["pkl_basename"],
            item["artifact_kind"],
            item["pkl_sha256"],
        )
    )
    if len(normalized) != len(
        {
            (item["pkl_basename"], item["artifact_kind"], item["pkl_sha256"])
            for item in normalized
        }
    ):
        raise histogram_sidecar_error("Lineage inputs must be unique.")
    return normalized


def _build_sidecar_payload(
    pkl_path: Path,
    histograms: Mapping[str, Any],
    *,
    identity_path: Path,
    artifact_kind: str,
    merged: bool,
    sumw2_storage_provenance: Mapping[str, Any],
    lineage_inputs: Iterable[Mapping[str, Any]],
    required_sumw2_processes: Mapping[str, Iterable[str]] | None,
    transformation_contract: Mapping[str, Any] | None,
    requested_data_driven_products: Mapping[str, Any] | None,
    resolved_data_driven_contract: Mapping[str, Any] | None,
    production_sample_contract: Mapping[str, Any] | None,
    nominal_reference_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    identity = _file_identity(identity_path, pkl_basename=pkl_path.name)
    if artifact_kind == "processor_output":
        if transformation_contract is not None:
            raise histogram_sidecar_error(
                "processor_output must not contain a transformation_contract."
            )
        normalized_contract = None
    else:
        if transformation_contract is None:
            raise histogram_sidecar_error(
                f"{artifact_kind} requires an independently generated "
                "transformation_contract. Regenerate the artifact with "
                "run_data_driven."
            )
        normalized_contract = _normalize_transformation_contract(
            transformation_contract,
            sumw2_storage_provenance=sumw2_storage_provenance,
            artifact_kind=artifact_kind,
        )
        derived_required = required_sumw2_processes_from_transformation_contract(
            normalized_contract,
            sumw2_storage_provenance=sumw2_storage_provenance,
            resolved_data_driven_contract=resolved_data_driven_contract,
        )
        if required_sumw2_processes is not None:
            requested_required = _normalize_required_processes(
                required_sumw2_processes
            )
            if requested_required != derived_required:
                raise histogram_sidecar_error(
                    f"{artifact_kind} required_sumw2_processes must be derived from "
                    "the transformation contract; "
                    f"derived={derived_required} requested={requested_required}."
                )
        required_sumw2_processes = derived_required

    payload = {
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "artifact": {
            **identity,
            "artifact_kind": artifact_kind,
            "merged": bool(merged),
            "nominal_container_schema_version": NOMINAL_CONTAINER_SCHEMA_VERSION,
            "nominal_container_layout": NOMINAL_CONTAINER_LAYOUT,
        },
        "sumw2_storage_provenance": copy.deepcopy(
            dict(sumw2_storage_provenance)
        ),
        "sumw2_content_manifest": build_sumw2_content_manifest(
            histograms,
            sumw2_storage_provenance=sumw2_storage_provenance,
            artifact_kind=artifact_kind,
            required_sumw2_processes=required_sumw2_processes,
        ),
        "lineage": {"inputs": _normalize_lineage_inputs(lineage_inputs)},
    }
    policy = resolved_policy_from_provenance(sumw2_storage_provenance)
    if policy.schema_version == SUMW2_PROVENANCE_SCHEMA_VERSION:
        if production_sample_contract is None:
            raise histogram_sidecar_error(
                "Current sumw2 provenance requires a certified "
                "production_sample_contract."
            )
        try:
            validate_production_sample_contract(
                production_sample_contract,
                policy,
            )
        except ValueError as error:
            raise histogram_sidecar_error(str(error)) from error
        payload["production_sample_contract"] = copy.deepcopy(
            dict(production_sample_contract)
        )
    elif production_sample_contract is not None:
        raise histogram_sidecar_error(
            "Legacy sumw2 provenance cannot be upgraded by attaching a current "
            "production_sample_contract; regenerate the processor artifact."
        )
    if (requested_data_driven_products is None) != (
        resolved_data_driven_contract is None
    ):
        raise histogram_sidecar_error(
            "requested_data_driven_products and resolved_data_driven_contract "
            "must be provided together."
        )
    normalized_reference_contract = None
    if artifact_kind == NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND:
        if nominal_reference_contract is None:
            raise histogram_sidecar_error(
                "A nominal-only reference output requires nominal_reference_contract."
            )
        if requested_data_driven_products is None or resolved_data_driven_contract is None:
            raise histogram_sidecar_error(
                "A nominal-only reference output requires data-driven provenance."
            )
    elif nominal_reference_contract is not None:
        raise histogram_sidecar_error(
            "nominal_reference_contract is valid only for nominal-only reference outputs."
        )
    if requested_data_driven_products is not None:
        try:
            normalized_requested, normalized_data_driven_contract = (
                validate_serialized_data_driven_contract(
                    requested_data_driven_products,
                    resolved_data_driven_contract,
                    policy=resolved_policy_from_provenance(
                        sumw2_storage_provenance
                    ),
                    allow_incomplete_nonprompt_sumw2=(
                        artifact_kind
                        == NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND
                    ),
                )
            )
        except data_driven_product_error as error:
            raise histogram_sidecar_error(str(error)) from error
        payload["requested_data_driven_products"] = normalized_requested
        payload["resolved_data_driven_contract"] = normalized_data_driven_contract
        if artifact_kind == NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND:
            normalized_reference_contract = _normalize_nominal_reference_contract(
                nominal_reference_contract,
                policy=policy,
                resolved_data_driven_contract=normalized_data_driven_contract,
            )
            payload["nominal_reference_contract"] = normalized_reference_contract
    if normalized_contract is not None:
        payload["transformation_contract"] = normalized_contract
        if requested_data_driven_products is not None:
            _validate_transformation_against_data_driven_contract(
                normalized_contract,
                normalized_data_driven_contract,
            )
    return payload


def _certified_generated_processes_for_family(
    transformation_contract: Mapping[str, Any],
    data_driven_contract: Mapping[str, Any],
    family: str,
    product: str,
) -> list[str]:
    generated = sorted(
        generated_output_processes_from_contract(
            data_driven_contract,
            product,
        )
    )
    if (
        transformation_contract["contract_version"]
        == TRANSFORMATION_CONTRACT_VERSION
        and not transformation_contract["families"][family][
            "applicable_products"
        ][product]
    ):
        return []
    return generated


def _validate_transformation_against_data_driven_contract(
    transformation_contract: Mapping[str, Any],
    data_driven_contract: Mapping[str, Any],
) -> None:
    artifact_kind = transformation_contract["artifact_kind"]
    projection = transformation_contract["eft_prompt_projection"]
    selected_prompt_processes = set(
        data_driven_contract["resolved_prompt_process_set"]
    )
    expected_projected_processes = set()
    for family, roles in transformation_contract["families"].items():
        source_scalar_processes = set(roles["source_scalar_processes"])
        source_eft_processes = set(roles["source_eft_processes"])
        duplicated_selected = sorted(
            selected_prompt_processes
            & source_scalar_processes
            & source_eft_processes
        )
        if duplicated_selected:
            raise histogram_sidecar_error(
                f"Family '{family}' duplicates selected prompt sources in "
                "scalar and EFT nominal source roles: "
                + ", ".join(duplicated_selected)
            )
        if artifact_kind in {
            NONPROMPT_OUTPUT_ARTIFACT_KIND,
            NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND,
        }:
            expected_projected_processes.update(
                selected_prompt_processes & source_eft_processes
            )
    expected_projected = sorted(expected_projected_processes)
    if projection["required_processes"] != expected_projected:
        raise histogram_sidecar_error(
            "EFT prompt projection provenance is tampered or inconsistent with "
            "the certified source roles: "
            f"expected={expected_projected} "
            f"observed={projection['required_processes']}."
        )
    if artifact_kind == FLIPS_OUTPUT_ARTIFACT_KIND and projection["required_processes"]:
        raise histogram_sidecar_error(
            "flips_output cannot certify private EFT prompt projection."
        )
    for family, roles in transformation_contract["families"].items():
        expected_nonprompt = (
            _certified_generated_processes_for_family(
                transformation_contract,
                data_driven_contract,
                family,
                "nonprompt",
            )
            if artifact_kind
            in {
                NONPROMPT_OUTPUT_ARTIFACT_KIND,
                NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND,
            }
            else []
        )
        expected_flips = _certified_generated_processes_for_family(
            transformation_contract,
            data_driven_contract,
            family,
            "flips",
        )
        if artifact_kind == FLIPS_OUTPUT_ARTIFACT_KIND:
            expected_nonprompt = []
        if roles["generated_nonprompt_processes"] != expected_nonprompt:
            raise histogram_sidecar_error(
                "Transformed nonprompt processes disagree with the certified "
                f"requested-product contract for family {family!r}: "
                f"expected={expected_nonprompt} "
                f"observed={roles['generated_nonprompt_processes']}."
            )
        if roles["generated_flips_processes"] != expected_flips:
            raise histogram_sidecar_error(
                "Transformed flips processes disagree with the certified "
                f"requested-product contract for family {family!r}: "
                f"expected={expected_flips} "
                f"observed={roles['generated_flips_processes']}."
            )


def lineage_input_from_sidecar(sidecar: Mapping[str, Any]) -> dict[str, str]:
    artifact = sidecar["artifact"]
    return {
        "pkl_basename": artifact["pkl_basename"],
        "artifact_kind": artifact["artifact_kind"],
        "pkl_sha256": artifact["pkl_sha256"],
    }


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise histogram_sidecar_error(
            f"Invalid {label} fields; missing={missing} unknown={unknown}."
        )


def _require_sorted_unique_strings(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise histogram_sidecar_error(f"{label} must be a list of nonempty strings.")
    if value != sorted(set(value)):
        raise histogram_sidecar_error(
            f"{label} must be unique and lexically ordered."
        )
    return list(value)


_TRANSFORMATION_ROLE_FIELDS = (
    "source_scalar_processes",
    "source_eft_processes",
    "retained_scalar_processes",
    "retained_eft_processes",
    "generated_nonprompt_processes",
    "generated_flips_processes",
)
_TRANSFORMATION_APPLICABILITY_FIELDS = (
    "source_application_regions",
    "applicable_products",
)
_TRANSFORMATION_CONTEXT_FIELDS = (
    *_TRANSFORMATION_ROLE_FIELDS,
    *_TRANSFORMATION_APPLICABILITY_FIELDS,
)
_PRIOR_TRANSFORMATION_CONTRACT_FAMILY_FIELDS = (
    *_TRANSFORMATION_ROLE_FIELDS,
    "consumed_source_processes",
)
_TRANSFORMATION_CONTRACT_FAMILY_FIELDS = (
    *_TRANSFORMATION_CONTEXT_FIELDS,
    "consumed_source_processes",
)


def _derive_data_driven_applicability(
    source_application_regions: Iterable[str],
) -> dict[str, bool]:
    return derive_data_driven_applicability(source_application_regions)


def _require_producer_transformation_context(
    transformation_context: Any,
) -> Mapping[str, Any]:
    if (
        not isinstance(transformation_context, Mapping)
        or getattr(
            transformation_context,
            "_producer_context_token",
            None,
        )
        is not _PRODUCER_CONTEXT_TOKEN
    ):
        raise histogram_sidecar_error(
            "transformation_context must be generated by DataDrivenProducer; "
            "caller-authored transformation contexts are not accepted."
        )
    return transformation_context


def _normalize_applicable_products(
    value: Any,
    *,
    source_application_regions: list[str],
    label: str,
) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise histogram_sidecar_error(f"{label} must be an object.")
    expected_keys = set(SUPPORTED_DATA_DRIVEN_PRODUCTS)
    _require_exact_keys(value, expected_keys, label=label)
    if any(type(value[product]) is not bool for product in expected_keys):
        raise histogram_sidecar_error(
            f"{label} values must be booleans."
        )
    observed = {product: value[product] for product in sorted(expected_keys)}
    expected = _derive_data_driven_applicability(source_application_regions)
    if observed != expected:
        raise histogram_sidecar_error(
            f"{label} contradicts authoritative source application-region "
            f"evidence: expected={expected} observed={observed}."
        )
    return observed


def _normalize_eft_prompt_projection(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise histogram_sidecar_error(
            "transformation_contract.eft_prompt_projection must be an object."
        )
    _require_exact_keys(
        value,
        {"mode", "required_processes", "generated_nonprompt_eft_dependence"},
        label="transformation_contract.eft_prompt_projection",
    )
    if value["mode"] != "sm_point":
        raise histogram_sidecar_error(
            "EFT prompt projection provenance must use mode='sm_point'."
        )
    required_processes = _require_sorted_unique_strings(
        value["required_processes"],
        label="transformation_contract.eft_prompt_projection.required_processes",
    )
    if value["generated_nonprompt_eft_dependence"] is not False:
        raise histogram_sidecar_error(
            "Generated nonprompt EFT dependence must be false."
        )
    return {
        "mode": "sm_point",
        "required_processes": required_processes,
        "generated_nonprompt_eft_dependence": False,
    }


def _normalize_missing_processes_by_family(
    value: Any,
    *,
    policy: Any,
    label: str,
) -> dict[str, list[str]]:
    if not isinstance(value, Mapping):
        raise histogram_sidecar_error(f"{label} must be an object.")
    unknown = sorted(set(value) - set(policy.runtime_histogram_families))
    if unknown:
        raise histogram_sidecar_error(
            f"{label} contains unknown runtime families: {unknown}."
        )
    return {
        str(family): _require_sorted_unique_strings(
            processes,
            label=f"{label}.{family}",
        )
        for family, processes in value.items()
    }


def _normalize_nominal_reference_contract(
    value: Any,
    *,
    policy: Any,
    resolved_data_driven_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the explicit non-card-ready contract for a nominal reference."""

    if not isinstance(value, Mapping):
        raise histogram_sidecar_error("nominal_reference_contract must be an object.")
    fields = {
        "contract_version",
        "reference_only",
        "card_ready",
        "statistically_complete",
        "migration_status",
        "serialized_contract_provenance",
        "serialized_prompt_process_set",
        "resolved_prompt_process_set",
        "added_prompt_processes",
        "removed_prompt_processes",
        "missing_process_resolved_sumw2",
        "missing_sumw2_policy_selection",
    }
    _require_exact_keys(value, fields, label="nominal_reference_contract")
    if value["contract_version"] != NOMINAL_REFERENCE_CONTRACT_VERSION:
        raise histogram_sidecar_error("Unsupported nominal_reference_contract version.")
    if value["reference_only"] is not True or value["card_ready"] is not False:
        raise histogram_sidecar_error(
            "nominal_reference_contract must be explicitly reference_only and non-card-ready."
        )
    if value["statistically_complete"] is not False:
        raise histogram_sidecar_error(
            "nominal_reference_contract must mark the output statistically incomplete."
        )
    if value["migration_status"] not in {
        "current_contract_revalidated",
        "reresolved_changed",
        "reresolved_unchanged",
    }:
        raise histogram_sidecar_error("Invalid nominal reference migration status.")
    if not isinstance(value["serialized_contract_provenance"], Mapping):
        raise histogram_sidecar_error(
            "nominal_reference_contract.serialized_contract_provenance must be an object."
        )
    serialized = _require_sorted_unique_strings(
        value["serialized_prompt_process_set"],
        label="nominal_reference_contract.serialized_prompt_process_set",
    )
    resolved = _require_sorted_unique_strings(
        value["resolved_prompt_process_set"],
        label="nominal_reference_contract.resolved_prompt_process_set",
    )
    added = _require_sorted_unique_strings(
        value["added_prompt_processes"],
        label="nominal_reference_contract.added_prompt_processes",
    )
    removed = _require_sorted_unique_strings(
        value["removed_prompt_processes"],
        label="nominal_reference_contract.removed_prompt_processes",
    )
    if added != sorted(set(resolved) - set(serialized)) or removed != sorted(
        set(serialized) - set(resolved)
    ):
        raise histogram_sidecar_error(
            "nominal_reference_contract prompt migration differences are inconsistent."
        )
    if resolved != resolved_data_driven_contract["resolved_prompt_process_set"]:
        raise histogram_sidecar_error(
            "nominal_reference_contract resolved prompt set disagrees with the "
            "current canonical data-driven contract."
        )
    migration = resolved_data_driven_contract["policy_migration"]
    if (
        serialized != migration["serialized_prompt_process_set"]
        or added != migration["added_prompt_processes"]
        or removed != migration["removed_prompt_processes"]
    ):
        raise histogram_sidecar_error(
            "nominal_reference_contract migration evidence disagrees with the "
            "resolved data-driven contract."
        )
    return {
        "contract_version": NOMINAL_REFERENCE_CONTRACT_VERSION,
        "reference_only": True,
        "card_ready": False,
        "statistically_complete": False,
        "migration_status": value["migration_status"],
        "serialized_contract_provenance": copy.deepcopy(
            dict(value["serialized_contract_provenance"])
        ),
        "serialized_prompt_process_set": serialized,
        "resolved_prompt_process_set": resolved,
        "added_prompt_processes": added,
        "removed_prompt_processes": removed,
        "missing_process_resolved_sumw2": _normalize_missing_processes_by_family(
            value["missing_process_resolved_sumw2"],
            policy=policy,
            label="nominal_reference_contract.missing_process_resolved_sumw2",
        ),
        "missing_sumw2_policy_selection": _normalize_missing_processes_by_family(
            value["missing_sumw2_policy_selection"],
            policy=policy,
            label="nominal_reference_contract.missing_sumw2_policy_selection",
        ),
    }


def _normalize_transformation_contract(
    transformation_contract: Mapping[str, Any],
    *,
    sumw2_storage_provenance: Mapping[str, Any],
    artifact_kind: str,
) -> dict[str, Any]:
    if not isinstance(transformation_contract, Mapping):
        raise histogram_sidecar_error("transformation_contract must be an object.")
    _require_exact_keys(
        transformation_contract,
        {
            "contract_version",
            "artifact_kind",
            "eft_prompt_projection",
            "families",
        },
        label="transformation_contract",
    )
    contract_version = transformation_contract["contract_version"]
    if contract_version not in {
        PRIOR_TRANSFORMATION_CONTRACT_VERSION,
        TRANSFORMATION_CONTRACT_VERSION,
    }:
        raise histogram_sidecar_error(
            "Unsupported transformed-content contract version "
            f"{contract_version!r}."
        )
    if transformation_contract["artifact_kind"] != artifact_kind:
        raise histogram_sidecar_error(
            "transformation_contract artifact kind does not match the artifact: "
            f"expected={artifact_kind!r} "
            f"observed={transformation_contract['artifact_kind']!r}."
        )
    if artifact_kind == "processor_output":
        raise histogram_sidecar_error(
            "processor_output must not contain a transformation_contract."
        )

    policy = resolved_policy_from_provenance(sumw2_storage_provenance)
    eft_prompt_projection = _normalize_eft_prompt_projection(
        transformation_contract["eft_prompt_projection"]
    )
    if artifact_kind == FLIPS_OUTPUT_ARTIFACT_KIND and eft_prompt_projection[
        "required_processes"
    ]:
        raise histogram_sidecar_error(
            "flips_output cannot certify private EFT prompt projection."
        )
    families = transformation_contract["families"]
    if not isinstance(families, Mapping):
        raise histogram_sidecar_error("transformation_contract.families must be an object.")
    if list(families) != list(policy.runtime_histogram_families):
        raise histogram_sidecar_error(
            "Transformation-contract families must match authoritative runtime "
            f"family order: expected={list(policy.runtime_histogram_families)} "
            f"observed={list(families)}."
        )

    normalized_families = {}
    for family, raw_roles in families.items():
        if not isinstance(raw_roles, Mapping):
            raise histogram_sidecar_error(
                f"Transformation roles for family '{family}' must be an object."
            )
        family_fields = (
            _TRANSFORMATION_CONTRACT_FAMILY_FIELDS
            if contract_version == TRANSFORMATION_CONTRACT_VERSION
            else _PRIOR_TRANSFORMATION_CONTRACT_FAMILY_FIELDS
        )
        _require_exact_keys(
            raw_roles,
            set(family_fields),
            label=f"transformation roles for family '{family}'",
        )
        roles = {
            field_name: _require_sorted_unique_strings(
                raw_roles[field_name],
                label=f"transformation family '{family}' field '{field_name}'",
            )
            for field_name in _PRIOR_TRANSFORMATION_CONTRACT_FAMILY_FIELDS
        }
        if contract_version == TRANSFORMATION_CONTRACT_VERSION:
            source_application_regions = _require_sorted_unique_strings(
                raw_roles["source_application_regions"],
                label=(
                    f"transformation family '{family}' field "
                    "'source_application_regions'"
                ),
            )
            roles["source_application_regions"] = source_application_regions
            roles["applicable_products"] = _normalize_applicable_products(
                raw_roles["applicable_products"],
                source_application_regions=source_application_regions,
                label=(
                    f"transformation family '{family}' field "
                    "'applicable_products'"
                ),
            )
        source_scalar = set(roles["source_scalar_processes"])
        source_eft = set(roles["source_eft_processes"])
        retained_scalar = set(roles["retained_scalar_processes"])
        retained_eft = set(roles["retained_eft_processes"])
        generated_nonprompt = set(roles["generated_nonprompt_processes"])
        generated_flips = set(roles["generated_flips_processes"])
        source_processes = source_scalar | source_eft
        retained_processes = retained_scalar | retained_eft
        generated_processes = generated_nonprompt | generated_flips

        if not retained_scalar <= source_scalar:
            raise histogram_sidecar_error(
                f"Family '{family}' retains scalar processes absent from the input: "
                f"{sorted(retained_scalar - source_scalar)}."
            )
        if not retained_eft <= source_eft:
            raise histogram_sidecar_error(
                f"Family '{family}' retains EFT processes absent from the input: "
                f"{sorted(retained_eft - source_eft)}."
            )
        if generated_nonprompt & generated_flips:
            raise histogram_sidecar_error(
                f"Family '{family}' assigns generated processes to both nonprompt "
                "and flips roles."
            )
        if generated_processes & source_processes:
            raise histogram_sidecar_error(
                f"Family '{family}' classifies source processes as generated: "
                f"{sorted(generated_processes & source_processes)}."
            )
        if artifact_kind == FLIPS_OUTPUT_ARTIFACT_KIND and generated_nonprompt:
            raise histogram_sidecar_error(
                f"flips_output family '{family}' cannot contain generated nonprompt roles."
            )
        expected_consumed = sorted(source_processes - retained_processes)
        if roles["consumed_source_processes"] != expected_consumed:
            raise histogram_sidecar_error(
                f"Family '{family}' consumed-source roles are inconsistent: "
                f"expected={expected_consumed} "
                f"observed={roles['consumed_source_processes']}."
            )
        normalized_families[family] = roles

    return {
        "contract_version": contract_version,
        "artifact_kind": artifact_kind,
        "eft_prompt_projection": eft_prompt_projection,
        "families": normalized_families,
    }


def required_sumw2_processes_from_transformation_contract(
    transformation_contract: Mapping[str, Any],
    *,
    sumw2_storage_provenance: Mapping[str, Any],
    resolved_data_driven_contract: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Derive required transformed companions without reading companion content."""

    artifact_kind = transformation_contract.get("artifact_kind")
    normalized = _normalize_transformation_contract(
        transformation_contract,
        sumw2_storage_provenance=sumw2_storage_provenance,
        artifact_kind=artifact_kind,
    )
    if resolved_data_driven_contract is not None:
        _validate_transformation_against_data_driven_contract(
            normalized,
            resolved_data_driven_contract,
        )
    policy = resolved_policy_from_provenance(sumw2_storage_provenance)
    output = {}
    for family, roles in normalized["families"].items():
        selected_source_processes = set(policy.selected_processes(family))
        retained_selected = selected_source_processes & (
            set(roles["retained_scalar_processes"])
            | set(roles["retained_eft_processes"])
        )
        generated = set(roles["generated_flips_processes"])
        if resolved_data_driven_contract is not None:
            generated = set(
                _certified_generated_processes_for_family(
                    normalized,
                    resolved_data_driven_contract,
                    family,
                    "flips",
                )
            )
        if artifact_kind == NONPROMPT_OUTPUT_ARTIFACT_KIND:
            if resolved_data_driven_contract is None:
                generated |= set(roles["generated_nonprompt_processes"])
            else:
                generated |= set(
                    _certified_generated_processes_for_family(
                        normalized,
                        resolved_data_driven_contract,
                        family,
                        "nonprompt",
                    )
                )
        output[family] = sorted(retained_selected | generated)
    return output


def _expected_nominal_processes_from_transformation_contract(
    transformation_contract: Mapping[str, Any],
    family: str,
    *,
    resolved_data_driven_contract: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    roles = transformation_contract["families"][family]
    generated = set(roles["generated_flips_processes"])
    if resolved_data_driven_contract is not None:
        generated = set(
            _certified_generated_processes_for_family(
                transformation_contract,
                resolved_data_driven_contract,
                family,
                "flips",
            )
        )
    if transformation_contract["artifact_kind"] in {
        NONPROMPT_OUTPUT_ARTIFACT_KIND,
        NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND,
    }:
        if resolved_data_driven_contract is None:
            generated |= set(roles["generated_nonprompt_processes"])
        else:
            generated |= set(
                _certified_generated_processes_for_family(
                    transformation_contract,
                    resolved_data_driven_contract,
                    family,
                    "nonprompt",
                )
            )
    expected_scalar = sorted(set(roles["retained_scalar_processes"]) | generated)
    return expected_scalar, list(roles["retained_eft_processes"])


def derive_transformed_required_sumw2_processes(
    *,
    input_sidecar: Mapping[str, Any],
    transformation_context: Mapping[str, Any],
    artifact_kind: str,
    transformed_histograms: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Build expected transformed roles and their independent companion contract."""

    if artifact_kind not in TRANSFORMED_DATA_DRIVEN_ARTIFACT_KINDS:
        raise histogram_sidecar_error(
            f"Cannot derive transformed requirements for {artifact_kind!r}."
        )
    if input_sidecar.get("artifact", {}).get("artifact_kind") != "processor_output":
        raise histogram_sidecar_error(
            "Transformed requirements require a validated processor_output input sidecar."
        )
    try:
        validate_requested_product_input(
            input_sidecar,
            artifact_kind=artifact_kind,
        )
    except data_driven_product_error as error:
        raise histogram_sidecar_error(str(error)) from error
    data_driven_contract = input_sidecar["resolved_data_driven_contract"]
    input_manifest = input_sidecar["sumw2_content_manifest"]["families"]
    transformation_context = _require_producer_transformation_context(
        transformation_context
    )
    _require_exact_keys(
        transformation_context,
        {"eft_prompt_projection", "families"},
        label="transformation_context",
    )
    context_projection = _normalize_eft_prompt_projection(
        transformation_context["eft_prompt_projection"]
    )
    context_families = transformation_context.get("families")
    if not isinstance(context_families, Mapping):
        raise histogram_sidecar_error(
            "transformation_context.families must be an object generated by "
            "DataDrivenProducer."
        )
    if list(context_families) != list(input_manifest):
        raise histogram_sidecar_error(
            "Transformation context must cover the input runtime families in "
            f"authoritative order: expected={list(input_manifest)} "
            f"observed={list(context_families)}."
        )

    contract_families = {}
    selected_prompt_processes = set(
        data_driven_contract["resolved_prompt_process_set"]
    )
    expected_projected_processes = set()
    for family, input_family in input_manifest.items():
        raw_roles = context_families[family]
        if not isinstance(raw_roles, Mapping):
            raise histogram_sidecar_error(
                f"Transformation context for family '{family}' must be an object."
            )
        _require_exact_keys(
            raw_roles,
            set(_TRANSFORMATION_CONTEXT_FIELDS),
            label=f"transformation context for family '{family}'",
        )
        roles = {
            field_name: _require_sorted_unique_strings(
                list(raw_roles[field_name]),
                label=f"transformation context family '{family}' field '{field_name}'",
            )
            for field_name in _TRANSFORMATION_ROLE_FIELDS
        }
        source_application_regions = _require_sorted_unique_strings(
            list(raw_roles["source_application_regions"]),
            label=(
                f"transformation context family '{family}' field "
                "'source_application_regions'"
            ),
        )
        applicable_products = _normalize_applicable_products(
            raw_roles["applicable_products"],
            source_application_regions=source_application_regions,
            label=(
                f"transformation context family '{family}' field "
                "'applicable_products'"
            ),
        )
        roles["source_application_regions"] = source_application_regions
        roles["applicable_products"] = applicable_products
        expected_source_scalar = input_family["scalar_nominal_processes"]
        expected_source_eft = input_family["eft_nominal_processes"]
        scalar_sources = set(expected_source_scalar)
        eft_sources = set(expected_source_eft)
        duplicated_selected = sorted(
            selected_prompt_processes & scalar_sources & eft_sources
        )
        if duplicated_selected:
            raise histogram_sidecar_error(
                f"Family '{family}' has the same selected prompt source in "
                "scalar and EFT nominal siblings: "
                + ", ".join(duplicated_selected)
            )
        if artifact_kind in {
            NONPROMPT_OUTPUT_ARTIFACT_KIND,
            NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND,
        }:
            expected_projected_processes.update(
                selected_prompt_processes & eft_sources
            )
        if roles["source_scalar_processes"] != expected_source_scalar:
            raise histogram_sidecar_error(
                f"Family '{family}' scalar source roles do not match the validated "
                f"input manifest: expected={expected_source_scalar} "
                f"observed={roles['source_scalar_processes']}."
            )
        if roles["source_eft_processes"] != expected_source_eft:
            raise histogram_sidecar_error(
                f"Family '{family}' EFT source roles do not match the validated input "
                f"manifest: expected={expected_source_eft} "
                f"observed={roles['source_eft_processes']}."
            )
        source_processes = set(expected_source_scalar) | set(expected_source_eft)
        retained_processes = set(roles["retained_scalar_processes"]) | set(
            roles["retained_eft_processes"]
        )
        contract_families[family] = {
            **roles,
            "consumed_source_processes": sorted(
                source_processes - retained_processes
            ),
        }

    expected_projection = {
        "mode": "sm_point",
        "required_processes": sorted(expected_projected_processes),
        "generated_nonprompt_eft_dependence": False,
    }
    if context_projection != expected_projection:
        raise histogram_sidecar_error(
            "EFT prompt projection provenance is tampered or inconsistent with "
            "the validated input nominal siblings: "
            f"expected={expected_projection} observed={context_projection}."
        )

    contract = _normalize_transformation_contract(
        {
            "contract_version": TRANSFORMATION_CONTRACT_VERSION,
            "artifact_kind": artifact_kind,
            "eft_prompt_projection": context_projection,
            "families": contract_families,
        },
        sumw2_storage_provenance=input_sidecar["sumw2_storage_provenance"],
        artifact_kind=artifact_kind,
    )
    _validate_transformation_against_data_driven_contract(
        contract,
        data_driven_contract,
    )
    if transformed_histograms is not None:
        for family in contract["families"]:
            content = _family_process_content(transformed_histograms, family)
            expected_scalar, expected_eft = (
                _expected_nominal_processes_from_transformation_contract(
                    contract,
                    family,
                    resolved_data_driven_contract=data_driven_contract,
                )
            )
            if content["scalar_nominal_processes"] != expected_scalar:
                raise histogram_content_error(
                    f"{artifact_kind} family '{family}' scalar nominal roles differ "
                    f"from the maintained transformation contract: "
                    f"expected={expected_scalar} "
                    f"observed={content['scalar_nominal_processes']}."
                )
            if content["eft_nominal_processes"] != expected_eft:
                raise histogram_content_error(
                    f"{artifact_kind} family '{family}' EFT nominal roles differ from "
                    f"the maintained transformation contract: expected={expected_eft} "
                    f"observed={content['eft_nominal_processes']}."
                )
            generated_nonprompt = set(
                generated_output_processes_from_contract(
                    data_driven_contract,
                    "nonprompt",
                )
            )
            unexpected_generated_eft = sorted(
                generated_nonprompt & set(content["eft_nominal_processes"])
            )
            if unexpected_generated_eft:
                raise histogram_content_error(
                    f"{artifact_kind} family '{family}' contains an unexpected "
                    "generated nonprompt EFT component: "
                    + ", ".join(unexpected_generated_eft)
                )
    required = required_sumw2_processes_from_transformation_contract(
        contract,
        sumw2_storage_provenance=input_sidecar["sumw2_storage_provenance"],
        resolved_data_driven_contract=data_driven_contract,
    )
    return contract, required


def _require_immutable_input_provenance(
    input_sidecar: Mapping[str, Any],
    output_provenance: Mapping[str, Any],
) -> None:
    input_identity = json.dumps(
        input_sidecar["sumw2_storage_provenance"],
        sort_keys=True,
        separators=(",", ":"),
    )
    output_identity = json.dumps(
        output_provenance,
        sort_keys=True,
        separators=(",", ":"),
    )
    if input_identity != output_identity:
        raise histogram_sidecar_error(
            "Transformed output must preserve its validated input "
            "sumw2_storage_provenance unchanged."
        )


def _normalize_certified_precanonical_data_driven_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Upgrade the certified version-3 reader shape without altering sidecars.

    Version 3 stored the complete prompt-MC membership under the nonprompt
    generated-output contributors, but retained a narrower signal-only list as
    ``required_prompt_signal_processes``.  The latter must not be promoted to
    the current prompt-subtraction authority.
    """

    normalized = copy.deepcopy(dict(contract))
    if normalized.get("contract_version") != (
        PRECANONICAL_RESOLVED_DATA_DRIVEN_CONTRACT_VERSION
    ):
        return normalized
    expected_fields = {
        "contract_version",
        "required_prompt_signal_processes",
        "products",
    }
    if set(normalized) != expected_fields:
        raise histogram_sidecar_error(
            "Cannot normalize a pre-canonical resolved_data_driven_contract "
            "with an unrecognized field shape."
        )

    nonprompt = normalized["products"]["nonprompt"]
    prompt_processes = list(resolved_prompt_processes_from_contract(normalized))
    if nonprompt["enabled"]:
        contributor_universe = sorted(
            {
                process
                for output in nonprompt["generated_outputs"].values()
                for processes in output["source_contributors"].values()
                for process in processes
            }
        )
        try:
            certified_policy = certify_active_nonprompt_policy(
                contributor_universe,
                configuration_source=(
                    "certified_legacy_resolved_data_driven_contract_v3"
                ),
            )
        except nonprompt_policy_error as error:
            raise histogram_sidecar_error(
                "Cannot normalize pre-canonical resolved_data_driven_contract "
                f"from its certified contributors: {error}"
            ) from error
        if list(certified_policy.resolved_prompt_process_set) != prompt_processes:
            raise histogram_sidecar_error(
                "Cannot normalize pre-canonical resolved_data_driven_contract: "
                "the canonical policy classification disagrees with the exact "
                "serialized nonprompt prompt_mc contributors."
            )
        nonprompt_policy = certified_policy.to_dict()
    else:
        if prompt_processes:
            raise histogram_sidecar_error(
                "Disabled pre-canonical nonprompt product cannot contain "
                "prompt_mc contributors."
            )
        nonprompt_policy = None

    return {
        "contract_version": RESOLVED_DATA_DRIVEN_CONTRACT_VERSION,
        "nonprompt_policy": nonprompt_policy,
        "resolved_prompt_process_set": prompt_processes,
        "policy_migration": {
            "status": "normalized_certified_legacy_contract",
            "previous_contract_version": (
                PRECANONICAL_RESOLVED_DATA_DRIVEN_CONTRACT_VERSION
            ),
            "serialized_prompt_process_set": prompt_processes,
            "added_prompt_processes": [],
            "removed_prompt_processes": [],
        },
        "products": normalized["products"],
    }


def _validate_sidecar_structure(
    sidecar: Mapping[str, Any],
    *,
    pkl_path: Path,
) -> dict[str, Any]:
    if not isinstance(sidecar, Mapping):
        raise histogram_sidecar_error("Histogram sidecar must be a JSON object.")
    sidecar = copy.deepcopy(dict(sidecar))
    common_fields = {
        "metadata_schema_version",
        "artifact",
        "sumw2_storage_provenance",
        "sumw2_content_manifest",
        "lineage",
    }
    missing_common = sorted(common_fields - set(sidecar))
    if missing_common:
        raise histogram_sidecar_error(
            f"Invalid histogram sidecar fields; missing={missing_common}."
        )
    if sidecar["metadata_schema_version"] != METADATA_SCHEMA_VERSION:
        raise histogram_sidecar_error(
            "Unsupported histogram metadata schema version "
            f"{sidecar['metadata_schema_version']!r}."
        )

    artifact = sidecar["artifact"]
    if not isinstance(artifact, Mapping):
        raise histogram_sidecar_error("Histogram sidecar artifact must be an object.")
    _require_exact_keys(
        artifact,
        {
            "pkl_basename",
            "pkl_size_bytes",
            "pkl_sha256",
            "artifact_kind",
            "merged",
            "nominal_container_schema_version",
            "nominal_container_layout",
        },
        label="artifact",
    )
    if artifact["pkl_basename"] != pkl_path.name:
        raise histogram_sidecar_error(
            "Histogram artifact identity mismatch: "
            f"pkl_path={pkl_path} sidecar_path={metadata_sidecar_path(pkl_path)} "
            f"expected_basename={artifact['pkl_basename']!r} "
            f"observed_basename={pkl_path.name!r}. Regenerate the sidecar with the artifact producer."
        )
    if not isinstance(artifact["pkl_size_bytes"], int) or artifact["pkl_size_bytes"] < 0:
        raise histogram_sidecar_error("artifact.pkl_size_bytes must be a nonnegative integer.")
    if (
        not isinstance(artifact["pkl_sha256"], str)
        or len(artifact["pkl_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in artifact["pkl_sha256"])
    ):
        raise histogram_sidecar_error("artifact.pkl_sha256 must be a SHA-256 hex digest.")
    if artifact["artifact_kind"] not in ARTIFACT_KINDS:
        raise histogram_sidecar_error(
            f"Unknown histogram artifact kind {artifact['artifact_kind']!r}."
        )
    if not isinstance(artifact["merged"], bool):
        raise histogram_sidecar_error("artifact.merged must be a boolean.")
    if artifact["nominal_container_schema_version"] != NOMINAL_CONTAINER_SCHEMA_VERSION:
        raise histogram_sidecar_error("Artifact nominal schema version is incompatible.")
    if artifact["nominal_container_layout"] != NOMINAL_CONTAINER_LAYOUT:
        raise histogram_sidecar_error("Artifact nominal container layout is incompatible.")

    policy = resolved_policy_from_provenance(sidecar["sumw2_storage_provenance"])
    expected_sidecar_fields = set(common_fields)
    has_production_contract = "production_sample_contract" in sidecar
    if policy.schema_version == SUMW2_PROVENANCE_SCHEMA_VERSION:
        if not has_production_contract:
            raise histogram_sidecar_error(
                "Current sumw2 provenance requires production_sample_contract "
                "certification. Regenerate this artifact with run_analysis."
            )
        expected_sidecar_fields.add("production_sample_contract")
        try:
            validate_production_sample_contract(
                sidecar["production_sample_contract"],
                policy,
            )
        except ValueError as error:
            raise histogram_sidecar_error(str(error)) from error
    elif has_production_contract:
        raise histogram_sidecar_error(
            "Legacy sumw2 provenance cannot contain a current production sample contract."
        )
    has_requested_products = "requested_data_driven_products" in sidecar
    has_resolved_contract = "resolved_data_driven_contract" in sidecar
    if has_requested_products != has_resolved_contract:
        raise histogram_sidecar_error(
            "requested_data_driven_products and resolved_data_driven_contract "
            "must be present together."
        )
    if has_requested_products:
        expected_sidecar_fields.update(
            {
                "requested_data_driven_products",
                "resolved_data_driven_contract",
            }
        )
    if artifact["artifact_kind"] == NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND:
        expected_sidecar_fields.add("nominal_reference_contract")
    transformation_contract = None
    if artifact["artifact_kind"] == "processor_output":
        if "transformation_contract" in sidecar:
            raise histogram_sidecar_error(
                "processor_output must not contain a transformation_contract."
            )
    else:
        expected_sidecar_fields.add("transformation_contract")
        if "transformation_contract" not in sidecar:
            raise histogram_sidecar_error(
                f"{artifact['artifact_kind']} metadata schema version 2 predates "
                "independent transformed-companion validation and lacks "
                f"transformation_contract: pkl_path={pkl_path} "
                f"sidecar_path={metadata_sidecar_path(pkl_path)}. Regenerate this "
                "PKL and sidecar with run_data_driven before reuse."
            )
        transformation_contract = _normalize_transformation_contract(
            sidecar["transformation_contract"],
            sumw2_storage_provenance=sidecar["sumw2_storage_provenance"],
            artifact_kind=artifact["artifact_kind"],
        )
    _require_exact_keys(
        sidecar,
        expected_sidecar_fields,
        label="histogram sidecar",
    )

    if has_requested_products:
        try:
            normalized_requested, normalized_contract = (
                validate_serialized_data_driven_contract(
                    sidecar["requested_data_driven_products"],
                    sidecar["resolved_data_driven_contract"],
                    policy=policy,
                    allow_incomplete_nonprompt_sumw2=(
                        artifact["artifact_kind"]
                        == NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND
                    ),
                )
            )
            normalized_contract = (
                _normalize_certified_precanonical_data_driven_contract(
                    normalized_contract
                )
            )
            normalized_requested, normalized_contract = (
                validate_serialized_data_driven_contract(
                    normalized_requested,
                    normalized_contract,
                    policy=policy,
                    allow_incomplete_nonprompt_sumw2=(
                        artifact["artifact_kind"]
                        == NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND
                    ),
                )
            )
        except data_driven_product_error as error:
            raise histogram_sidecar_error(str(error)) from error
        sidecar["requested_data_driven_products"] = normalized_requested
        sidecar["resolved_data_driven_contract"] = normalized_contract
        if transformation_contract is not None:
            _validate_transformation_against_data_driven_contract(
                transformation_contract,
                sidecar["resolved_data_driven_contract"],
            )
        if artifact["artifact_kind"] == NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND:
            _normalize_nominal_reference_contract(
                sidecar["nominal_reference_contract"],
                policy=policy,
                resolved_data_driven_contract=sidecar["resolved_data_driven_contract"],
            )
    manifest = sidecar["sumw2_content_manifest"]
    if not isinstance(manifest, Mapping):
        raise histogram_sidecar_error("sumw2_content_manifest must be an object.")
    _require_exact_keys(
        manifest,
        {"manifest_version", "families"},
        label="sumw2_content_manifest",
    )
    if manifest["manifest_version"] != SUMW2_CONTENT_MANIFEST_VERSION:
        raise histogram_sidecar_error("Unsupported sumw2 content manifest version.")
    families = manifest["families"]
    if not isinstance(families, Mapping):
        raise histogram_sidecar_error("sumw2_content_manifest.families must be an object.")
    if list(families) != list(policy.runtime_histogram_families):
        raise histogram_sidecar_error(
            "Manifest families must match authoritative runtime family order: "
            f"expected={list(policy.runtime_histogram_families)} observed={list(families)}."
        )
    family_fields = {
        "dimensionality",
        "scalar_nominal_processes",
        "eft_nominal_processes",
        "sumw2_processes",
        "required_sumw2_processes",
    }
    for family, family_manifest in families.items():
        if not isinstance(family_manifest, Mapping):
            raise histogram_sidecar_error(f"Manifest family '{family}' must be an object.")
        _require_exact_keys(
            family_manifest,
            family_fields,
            label=f"manifest family '{family}'",
        )
        expected_dimensionality = 2 if family in axes_info_2d else 1
        if family_manifest["dimensionality"] != expected_dimensionality:
            raise histogram_sidecar_error(
                f"Manifest family '{family}' has wrong dimensionality."
            )
        for field_name in family_fields - {"dimensionality"}:
            _require_sorted_unique_strings(
                family_manifest[field_name],
                label=f"manifest family '{family}' field '{field_name}'",
            )
        required = set(family_manifest["required_sumw2_processes"])
        observed = set(family_manifest["sumw2_processes"])
        if not required <= observed:
            raise histogram_sidecar_error(
                "Manifest requires sumw2 processes absent from artifact content: "
                f"pkl_path={pkl_path} "
                f"sidecar_path={metadata_sidecar_path(pkl_path)} "
                f"artifact_kind={artifact['artifact_kind']} family={family} "
                f"expected_processes={sorted(required)} "
                f"observed_processes={sorted(observed)} "
                f"missing_required_companions={sorted(required - observed)} "
                "unexpected_companions=[]. Regenerate it with run_data_driven."
            )

    if transformation_contract is not None:
        derived_required = required_sumw2_processes_from_transformation_contract(
            transformation_contract,
            sumw2_storage_provenance=sidecar["sumw2_storage_provenance"],
            resolved_data_driven_contract=sidecar.get(
                "resolved_data_driven_contract"
            ),
        )
        serialized_required = {
            family: list(family_manifest["required_sumw2_processes"])
            for family, family_manifest in families.items()
        }
        if serialized_required != derived_required:
            raise histogram_sidecar_error(
                f"{artifact['artifact_kind']} required_sumw2_processes disagree "
                "with the independently generated transformation contract: "
                f"expected={derived_required} observed={serialized_required}. "
                "Regenerate the transformed artifact with run_data_driven."
            )

    lineage = sidecar["lineage"]
    if not isinstance(lineage, Mapping):
        raise histogram_sidecar_error("lineage must be an object.")
    _require_exact_keys(lineage, {"inputs"}, label="lineage")
    if not isinstance(lineage["inputs"], list):
        raise histogram_sidecar_error("lineage.inputs must be a list.")
    normalized_lineage = _normalize_lineage_inputs(lineage["inputs"])
    if normalized_lineage != lineage["inputs"]:
        raise histogram_sidecar_error("lineage.inputs must use deterministic ordering.")
    if (
        artifact["artifact_kind"] == "processor_output"
        and not artifact["merged"]
        and lineage["inputs"]
    ):
        raise histogram_sidecar_error(
            "An unmerged processor_output lineage.inputs list must be empty."
        )
    if (
        (artifact["artifact_kind"] != "processor_output" or artifact["merged"])
        and not lineage["inputs"]
    ):
        raise histogram_sidecar_error(
            f"{artifact['artifact_kind']} merged={artifact['merged']} requires generated input lineage."
        )
    return copy.deepcopy(dict(sidecar))


def read_histogram_sidecar(
    pkl_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read the canonical colocated sidecar; callers never pass a JSON path."""

    pkl_path = Path(pkl_path)
    sidecar_path = metadata_sidecar_path(pkl_path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Histogram sidecar not found: {sidecar_path}")
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise histogram_sidecar_error(
            f"Could not read histogram sidecar '{sidecar_path}': {error}"
        ) from error
    return _validate_sidecar_structure(payload, pkl_path=pkl_path)


def _validate_content_manifest(
    pkl_path: Path,
    histograms: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> None:
    artifact_kind = sidecar["artifact"]["artifact_kind"]
    expected_manifest = sidecar["sumw2_content_manifest"]
    actual_manifest = build_sumw2_content_manifest(
        histograms,
        sumw2_storage_provenance=sidecar["sumw2_storage_provenance"],
        artifact_kind=artifact_kind,
        required_sumw2_processes={
            family: family_manifest["required_sumw2_processes"]
            for family, family_manifest in expected_manifest["families"].items()
        },
    )
    for family, expected in expected_manifest["families"].items():
        observed = actual_manifest["families"][family]
        for field_name in (
            "scalar_nominal_processes",
            "eft_nominal_processes",
            "sumw2_processes",
        ):
            if expected[field_name] == observed[field_name]:
                continue
            expected_processes = expected[field_name]
            observed_processes = observed[field_name]
            missing = sorted(set(expected_processes) - set(observed_processes))
            unexpected = sorted(set(observed_processes) - set(expected_processes))
            raise histogram_content_error(
                "Histogram artifact content mismatch: "
                f"pkl_path={pkl_path} sidecar_path={metadata_sidecar_path(pkl_path)} "
                f"artifact_kind={artifact_kind} family={family} field={field_name} "
                f"expected_processes={expected_processes} observed_processes={observed_processes} "
                f"missing_required_companions={missing if field_name == 'sumw2_processes' else []} "
                f"unexpected_companions={unexpected if field_name == 'sumw2_processes' else []}. "
                "Regenerate the PKL and sidecar together with run_analysis, run_data_driven, "
                "or the merged-cache writer."
            )
        required = set(expected["required_sumw2_processes"])
        observed_sumw2 = set(observed["sumw2_processes"])
        missing_required = sorted(required - observed_sumw2)
        if missing_required:
            raise histogram_content_error(
                "Histogram artifact is missing required companions: "
                f"pkl_path={pkl_path} sidecar_path={metadata_sidecar_path(pkl_path)} "
                f"artifact_kind={artifact_kind} family={family} "
                f"expected_processes={sorted(required)} observed_processes={sorted(observed_sumw2)} "
                f"missing_required_companions={missing_required} unexpected_companions=[]. "
                "Regenerate this artifact with its maintained producer."
            )


def _validate_artifact_identity(pkl_path: Path, sidecar: Mapping[str, Any]) -> None:
    expected = sidecar["artifact"]
    observed_size = pkl_path.stat().st_size
    observed_sha256 = _sha256(pkl_path)
    if (
        observed_size != expected["pkl_size_bytes"]
        or observed_sha256 != expected["pkl_sha256"]
    ):
        raise histogram_sidecar_error(
            "Histogram artifact identity mismatch: "
            f"pkl_path={pkl_path} sidecar_path={metadata_sidecar_path(pkl_path)} "
            f"expected_basename={expected['pkl_basename']!r} observed_basename={pkl_path.name!r} "
            f"expected_size={expected['pkl_size_bytes']} observed_size={observed_size} "
            f"expected_sha256={expected['pkl_sha256']} observed_sha256={observed_sha256}. "
            "Regenerate or restore the matching PKL/sidecar pair."
        )


def validate_processor_output(
    pkl_path: str | os.PathLike[str],
    histograms: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> None:
    if sidecar["artifact"]["artifact_kind"] != "processor_output":
        raise histogram_content_error("Expected artifact_kind=processor_output.")
    policy = resolved_policy_from_provenance(sidecar["sumw2_storage_provenance"])
    validate_nominal_mapping(
        histograms,
        runtime_families=policy.runtime_histogram_families,
        schema_version=NOMINAL_CONTAINER_SCHEMA_VERSION,
        policy=policy,
    )
    expected_manifest = build_sumw2_content_manifest(
        histograms,
        sumw2_storage_provenance=sidecar["sumw2_storage_provenance"],
        artifact_kind="processor_output",
    )
    for family in policy.runtime_histogram_families:
        expected_required = expected_manifest["families"][family][
            "required_sumw2_processes"
        ]
        observed_required = sidecar["sumw2_content_manifest"]["families"][family][
            "required_sumw2_processes"
        ]
        if observed_required != expected_required:
            raise histogram_content_error(
                f"processor_output family '{family}' has required_sumw2_processes "
                f"{observed_required}, expected {expected_required} from source allocation."
            )
    if "requested_data_driven_products" in sidecar:
        try:
            requested_products = sidecar["requested_data_driven_products"][
                "products"
            ]
            if (
                policy.schema_version == SUMW2_PROVENANCE_SCHEMA_VERSION
                and
                sidecar["resolved_data_driven_contract"]["contract_version"]
                == RESOLVED_DATA_DRIVEN_CONTRACT_VERSION
            ):
                if requested_products["nonprompt"]["enabled"]:
                    validate_requested_product_input(
                        sidecar,
                        artifact_kind="nonprompt_output",
                    )
                if requested_products["flips"]["enabled"]:
                    validate_requested_product_input(
                        sidecar,
                        artifact_kind="flips_output",
                    )
        except data_driven_product_error as error:
            raise histogram_content_error(str(error)) from error


def _validate_transformed_output(
    pkl_path: str | os.PathLike[str],
    histograms: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    *,
    artifact_kind: str,
) -> None:
    if sidecar["artifact"]["artifact_kind"] != artifact_kind:
        raise histogram_content_error(f"Expected artifact_kind={artifact_kind}.")
    policy = resolved_policy_from_provenance(sidecar["sumw2_storage_provenance"])
    validate_nominal_mapping(
        histograms,
        runtime_families=policy.runtime_histogram_families,
        schema_version=NOMINAL_CONTAINER_SCHEMA_VERSION,
        policy=None,
    )
    manifest_families = sidecar["sumw2_content_manifest"]["families"]
    transformation_contract = _normalize_transformation_contract(
        sidecar["transformation_contract"],
        sumw2_storage_provenance=sidecar["sumw2_storage_provenance"],
        artifact_kind=artifact_kind,
    )
    independently_required = required_sumw2_processes_from_transformation_contract(
        transformation_contract,
        sumw2_storage_provenance=sidecar["sumw2_storage_provenance"],
        resolved_data_driven_contract=sidecar.get(
            "resolved_data_driven_contract"
        ),
    )
    for family in policy.runtime_histogram_families:
        family_manifest = manifest_families[family]
        expected_scalar, expected_eft = (
            _expected_nominal_processes_from_transformation_contract(
                transformation_contract,
                family,
                resolved_data_driven_contract=sidecar.get(
                    "resolved_data_driven_contract"
                ),
            )
        )
        expected_sumw2 = independently_required[family]
        observed_scalar = family_manifest["scalar_nominal_processes"]
        observed_eft = family_manifest["eft_nominal_processes"]
        observed_sumw2 = family_manifest["sumw2_processes"]
        if observed_scalar != expected_scalar or observed_eft != expected_eft:
            raise histogram_content_error(
                f"{artifact_kind} pkl_path={pkl_path} family={family} has nominal "
                "process roles inconsistent with the maintained transformation: "
                f"expected_scalar={expected_scalar} observed_scalar={observed_scalar} "
                f"expected_eft={expected_eft} observed_eft={observed_eft}. "
                "Regenerate it with run_data_driven."
            )
        if observed_sumw2 != expected_sumw2:
            missing = sorted(set(expected_sumw2) - set(observed_sumw2))
            unexpected = sorted(set(observed_sumw2) - set(expected_sumw2))
            raise histogram_content_error(
                "Transformed histogram companion contract mismatch: "
                f"pkl_path={pkl_path} sidecar_path={metadata_sidecar_path(pkl_path)} "
                f"artifact_kind={artifact_kind} family={family} "
                f"expected_processes={expected_sumw2} observed_processes={observed_sumw2} "
                f"missing_required_companions={missing} "
                f"unexpected_companions={unexpected}. Regenerate it with "
                "run_data_driven."
            )


def validate_nonprompt_output(
    pkl_path: str | os.PathLike[str],
    histograms: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> None:
    _validate_transformed_output(
        pkl_path,
        histograms,
        sidecar,
        artifact_kind="nonprompt_output",
    )


def validate_nonprompt_nominal_reference_output(
    pkl_path: str | os.PathLike[str],
    histograms: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> None:
    """Validate a deliberately incomplete non-card-ready nominal reference."""

    _validate_transformed_output(
        pkl_path,
        histograms,
        sidecar,
        artifact_kind=NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND,
    )
    reference_contract = sidecar.get("nominal_reference_contract")
    if not isinstance(reference_contract, Mapping):
        raise histogram_content_error(
            "Nominal-only reference output lacks nominal_reference_contract."
        )
    if reference_contract.get("statistically_complete") is not False:
        raise histogram_content_error(
            "Nominal-only reference output cannot be statistically complete."
        )


def validate_flips_output(
    pkl_path: str | os.PathLike[str],
    histograms: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> None:
    _validate_transformed_output(
        pkl_path,
        histograms,
        sidecar,
        artifact_kind="flips_output",
    )


def _load_histograms(pkl_path: Path) -> dict[str, Any]:
    from topcoffea.modules.utils import get_hist_from_pkl

    loaded = get_hist_from_pkl(str(pkl_path), allow_empty=False)
    if not isinstance(loaded, dict):
        raise histogram_content_error(
            f"Histogram PKL '{pkl_path}' did not contain a dictionary."
        )
    return loaded


def _recognized_legacy_metadata(payload: Any) -> bool:
    return (
        isinstance(payload, Mapping)
        and payload.get("metadata_version") == 1
        and {
            "input_histogram",
            "sumw2_storage_provenance",
            "nominal_container_schema_version",
            "nominal_container_layout",
        }
        <= set(payload)
    )


def validate_histogram_artifact(
    pkl_path: str | os.PathLike[str],
    histograms: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a split artifact or classify an explicit legacy uniform payload."""

    pkl_path = Path(pkl_path)
    loaded = dict(histograms) if histograms is not None else _load_histograms(pkl_path)
    split_payload = is_split_nominal_mapping(loaded)
    sidecar_path = metadata_sidecar_path(pkl_path)
    if not sidecar_path.is_file():
        if split_payload:
            split_keys = sorted(
                key
                for key in loaded
                if key.endswith("__scalar_nominal") or key.endswith("__eft_nominal")
            )
            raise histogram_sidecar_error(
                "Schema-v2 histogram PKL is missing its required automatic sidecar: "
                f"pkl_path={pkl_path} expected_sidecar_path={sidecar_path} "
                f"detected_split_sibling_keys={split_keys}. Expected producer: run_analysis, "
                "run_data_driven, or a merged-cache writer. Regenerate the artifact with its "
                "maintained producer; do not supply a sidecar path manually."
            )
        return {
            "schema": "legacy_uniform",
            "metadata": None,
            "legacy_metadata_present": False,
        }

    try:
        raw_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise histogram_sidecar_error(
            "Could not read automatic histogram sidecar: "
            f"pkl_path={pkl_path} sidecar_path={sidecar_path} error={error}. "
            "Regenerate the PKL and sidecar together with the maintained producer."
        ) from error
    if _recognized_legacy_metadata(raw_payload):
        if split_payload:
            raise histogram_sidecar_error(
                f"Schema-v2 PKL '{pkl_path}' has obsolete version-1 metadata at "
                f"'{sidecar_path}'. Regenerate it with the maintained producer."
            )
        return {
            "schema": "legacy_uniform",
            "metadata": None,
            "legacy_metadata_present": True,
        }
    sidecar = _validate_sidecar_structure(raw_payload, pkl_path=pkl_path)
    _validate_content_manifest(pkl_path, loaded, sidecar)
    artifact_kind = sidecar["artifact"]["artifact_kind"]
    if artifact_kind == "processor_output":
        validate_processor_output(pkl_path, loaded, sidecar)
    elif artifact_kind == NONPROMPT_OUTPUT_ARTIFACT_KIND:
        validate_nonprompt_output(pkl_path, loaded, sidecar)
    elif artifact_kind == NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND:
        validate_nonprompt_nominal_reference_output(pkl_path, loaded, sidecar)
    elif artifact_kind == FLIPS_OUTPUT_ARTIFACT_KIND:
        validate_flips_output(pkl_path, loaded, sidecar)
    else:  # pragma: no cover - structural validation already rejects this
        raise histogram_sidecar_error(f"Unknown artifact kind {artifact_kind!r}.")
    _validate_artifact_identity(pkl_path, sidecar)
    return {
        "schema": NOMINAL_CONTAINER_LAYOUT,
        "metadata": sidecar,
        "legacy_metadata_present": False,
    }


def _canonical_identity(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _ordered_family_union(
    family_sequences: Iterable[Iterable[str]],
) -> list[str]:
    """Compose materialized family coverage without changing producer order."""

    families: list[str] = []
    seen: set[str] = set()
    for sequence in family_sequences:
        for family in sequence:
            if family not in seen:
                seen.add(family)
                families.append(family)
    return families


def _compose_sumw2_storage_provenance(
    provenances: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compose compatible fragments of one source universe."""

    normalized = [
        resolved_policy_from_provenance(provenance).to_provenance()
        for provenance in provenances
    ]
    if not normalized:
        raise histogram_merge_error("No sumw2 provenance records were provided.")
    control_fields = (
        "schema_version",
        "source",
        "requested_mode",
        "resolved_mode",
        "signal_sample_profile",
        "normalized_rules",
        "warnings",
    )
    first = normalized[0]
    for field_name in control_fields:
        if any(record[field_name] != first[field_name] for record in normalized[1:]):
            raise histogram_merge_error(
                "Cannot compose source-allocation provenance with incompatible "
                f"policy-control field '{field_name}'."
            )
    datasets = sorted(
        {
            dataset
            for record in normalized
            for dataset in record["resolved_datasets"]
        }
    )
    processes = sorted(
        {
            process
            for record in normalized
            for process in record["resolved_processes"]
        }
    )
    runtime_histogram_families = _ordered_family_union(
        record["runtime_histogram_families"] for record in normalized
    )
    family_order = {
        family: index for index, family in enumerate(runtime_histogram_families)
    }
    targets_by_identity: dict[str, dict[str, Any]] = {}
    for record in normalized:
        for target in record["resolved_targets"]:
            targets_by_identity.setdefault(
                _canonical_identity(target), copy.deepcopy(target)
            )
    targets = list(targets_by_identity.values())
    targets.sort(
        key=lambda target: (
            target["dataset"],
            target["process"],
            family_order[target["family"]],
        )
    )
    composed = {
        **{field_name: copy.deepcopy(first[field_name]) for field_name in control_fields},
        "runtime_histogram_families": runtime_histogram_families,
        "resolved_datasets": datasets,
        "resolved_processes": processes,
        "resolved_targets": targets,
    }
    try:
        return resolved_policy_from_provenance(composed).to_provenance()
    except ValueError as error:
        raise histogram_merge_error(
            f"Composed source-allocation provenance is invalid: {error}"
        ) from error


def _compose_production_sample_contract(
    contracts: Iterable[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose fragment coverage and recompute the certified contract identity."""

    contracts = tuple(contracts)
    first = contracts[0]
    compatibility_fields = (
        "contract_version",
        "wrapper_identity",
        "resolved_mode",
        "signal_sample_profile",
        "compatibility_validated",
    )
    for field_name in compatibility_fields:
        if any(contract[field_name] != first[field_name] for contract in contracts[1:]):
            raise histogram_merge_error(
                "Same-run histogram fragments have incompatible production sample "
                f"contract field {field_name!r}."
            )
    cfg_identities_by_identity: dict[str, dict[str, Any]] = {}
    active_variants: dict[str, dict[str, Any]] = {}
    for contract in contracts:
        for identity in contract["cfg_identities"]:
            cfg_identities_by_identity.setdefault(
                _canonical_identity(identity), copy.deepcopy(identity)
            )
        for variant_key, variant in contract["active_signal_variants"].items():
            prior = active_variants.get(variant_key)
            if prior is None:
                active_variants[variant_key] = copy.deepcopy(variant)
                continue
            for field_name in ("signal_group", "year", "selected_variant"):
                if prior[field_name] != variant[field_name]:
                    raise histogram_merge_error(
                        "Same-run histogram fragments disagree on active signal variant "
                        f"{variant_key!r} field {field_name!r}."
                    )
            prior["processes"] = sorted(
                set(prior["processes"]) | set(variant["processes"])
            )
    composed = {
        **{
            field_name: copy.deepcopy(first[field_name])
            for field_name in compatibility_fields
        },
        "cfg_identities": sorted(
            cfg_identities_by_identity.values(),
            key=lambda identity: tuple(sorted(identity.items())),
        ),
        "active_signal_variants": {
            key: active_variants[key] for key in sorted(active_variants)
        },
    }
    composed["contract_identity_sha256"] = hashlib.sha256(
        _canonical_identity(composed).encode("utf-8")
    ).hexdigest()
    try:
        validate_production_sample_contract(composed, provenance)
    except Exception as error:
        raise histogram_merge_error(
            f"Composed production sample contract is invalid: {error}"
        ) from error
    return composed


def _compose_resolved_data_driven_contract(
    contracts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Union current generated-output coverage without changing request policy."""

    contracts = tuple(contracts)
    first = contracts[0]
    if any(
        contract["contract_version"] != first["contract_version"]
        for contract in contracts[1:]
    ):
        raise histogram_merge_error(
            "Same-run histogram fragments require one resolved data-driven contract version."
        )
    if first["contract_version"] != RESOLVED_DATA_DRIVEN_CONTRACT_VERSION:
        if any(
            _canonical_identity(contract) != _canonical_identity(first)
            for contract in contracts[1:]
        ):
            raise histogram_merge_error(
                "Legacy resolved data-driven contracts must remain identical across fragments."
            )
        return copy.deepcopy(dict(first))
    if all(
        _canonical_identity(contract) == _canonical_identity(first)
        for contract in contracts[1:]
    ):
        return copy.deepcopy(dict(first))

    products = {}
    for product_name, first_product in first["products"].items():
        if any(
            contract["products"][product_name]["enabled"]
            is not first_product["enabled"]
            for contract in contracts[1:]
        ):
            raise histogram_merge_error(
                "Same-run histogram fragments disagree on requested data-driven "
                f"product {product_name!r}."
            )
        generated_outputs = {}
        for contract in contracts:
            for output_name, output in contract["products"][product_name][
                "generated_outputs"
            ].items():
                prior = generated_outputs.get(output_name)
                if prior is None:
                    generated_outputs[output_name] = copy.deepcopy(output)
                    continue
                if prior["year"] != output["year"]:
                    raise histogram_merge_error(
                        "Same-run histogram fragments disagree on resolved data-driven "
                        f"output year for {product_name}/{output_name}."
                    )
                if list(prior["source_contributors"]) != list(
                    output["source_contributors"]
                ):
                    raise histogram_merge_error(
                        "Same-run histogram fragments disagree on resolved data-driven "
                        f"roles for {product_name}/{output_name}."
                    )
                for role, processes in prior["source_contributors"].items():
                    processes[:] = sorted(
                        set(processes)
                        | set(output["source_contributors"][role])
                    )
                prior["required_source_sumw2_processes"] = sorted(
                    {
                        process
                        for processes in prior["source_contributors"].values()
                        for process in processes
                    }
                )
        ordered_outputs = {
            output_name: generated_outputs[output_name]
            for output_name in sorted(
                generated_outputs,
                key=lambda name: (
                    CANONICAL_DATA_DRIVEN_YEARS.index(
                        generated_outputs[name]["year"]
                    ),
                    name,
                ),
            )
        }
        products[product_name] = {
            "enabled": first_product["enabled"],
            "generated_outputs": ordered_outputs,
            "output_processes": list(ordered_outputs),
        }
    return {
        "contract_version": first["contract_version"],
        "nonprompt_policy": (
            {
                "schema_version": first["nonprompt_policy"]["schema_version"],
                "configuration_source": "merged_histogram_fragments",
                "resolved_prompt_process_set": sorted(
                    {
                        process
                        for contract in contracts
                        for process in contract["resolved_prompt_process_set"]
                    }
                ),
                "explicit_exclusions": sorted(
                    {
                        process
                        for contract in contracts
                        for process in contract["nonprompt_policy"][
                            "explicit_exclusions"
                        ]
                    }
                ),
                "resolutions": sorted(
                    {
                        _canonical_identity(resolution): resolution
                        for contract in contracts
                        for resolution in contract["nonprompt_policy"]["resolutions"]
                    }.values(),
                    key=lambda resolution: resolution["raw_process_label"],
                ),
            }
            if first["nonprompt_policy"] is not None
            else None
        ),
        "resolved_prompt_process_set": sorted(
            {
                process
                for contract in contracts
                for process in contract["resolved_prompt_process_set"]
            }
        ),
        "policy_migration": {
            "status": "merged_resolution",
            "previous_contract_version": None,
            "serialized_prompt_process_set": [],
            "added_prompt_processes": sorted(
                {
                    process
                    for contract in contracts
                    for process in contract["resolved_prompt_process_set"]
                }
            ),
            "removed_prompt_processes": [],
        },
        "products": products,
    }


def _compose_requested_data_driven_products(
    requested_products: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compose compatible requested policy while retaining all diagnostics."""

    requested_products = tuple(requested_products)
    first = requested_products[0]
    compatibility_fields = ("schema_version", "source", "products")
    if any(
        any(
            requested[field_name] != first[field_name]
            for field_name in compatibility_fields
        )
        for requested in requested_products[1:]
    ):
        raise histogram_merge_error(
            "Same-run histogram fragments require identical requested "
            "data-driven product policy."
        )
    return {
        **{
            field_name: copy.deepcopy(first[field_name])
            for field_name in compatibility_fields
        },
        "warnings": sorted(
            {
                warning
                for requested in requested_products
                for warning in requested["warnings"]
            }
        ),
    }


def _compose_merged_contract_set(
    sidecars: Iterable[Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Build one truthful coupled contract set for a maintained sidecar merge."""

    sidecars = tuple(sidecars)
    if len(sidecars) == 1:
        sidecar = sidecars[0]
        return (
            copy.deepcopy(sidecar["sumw2_storage_provenance"]),
            copy.deepcopy(sidecar.get("production_sample_contract")),
            copy.deepcopy(sidecar.get("requested_data_driven_products")),
            copy.deepcopy(sidecar.get("resolved_data_driven_contract")),
        )

    requested_presence = {
        "requested_data_driven_products" in sidecar for sidecar in sidecars
    }
    if len(requested_presence) != 1:
        raise histogram_merge_error(
            "Cannot merge artifacts with mixed requested data-driven contract presence."
        )
    provenance = _compose_sumw2_storage_provenance(
        sidecar["sumw2_storage_provenance"] for sidecar in sidecars
    )
    production_contracts = tuple(
        sidecar.get("production_sample_contract") for sidecar in sidecars
    )
    production_presence = {contract is not None for contract in production_contracts}
    if len(production_presence) != 1:
        raise histogram_merge_error(
            "Cannot merge artifacts with mixed production sample contract presence."
        )
    production_contract = None
    if production_presence == {True}:
        production_contract = _compose_production_sample_contract(
            production_contracts,
            provenance,
        )
    if production_contract is not None:
        try:
            validate_production_sample_contract(production_contract, provenance)
        except Exception as error:
            raise histogram_merge_error(
                f"Composed production sample contract is invalid: {error}"
            ) from error
    requested = None
    resolved = None
    if requested_presence == {True}:
        requested = _compose_requested_data_driven_products(
            sidecar["requested_data_driven_products"] for sidecar in sidecars
        )
        resolved = _compose_resolved_data_driven_contract(
            sidecar["resolved_data_driven_contract"] for sidecar in sidecars
        )
        try:
            requested, resolved = validate_serialized_data_driven_contract(
                requested,
                resolved,
                policy=resolved_policy_from_provenance(provenance),
            )
        except data_driven_product_error as error:
            raise histogram_merge_error(
                f"Composed data-driven contract is invalid: {error}"
            ) from error
    return provenance, production_contract, requested, resolved


def merge_histogram_sidecars(
    sidecars: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate merge compatibility and derive deterministic output metadata inputs."""

    sidecars = tuple(sidecars)
    if not sidecars:
        raise histogram_merge_error("No histogram sidecars were provided for merge.")
    kinds = {sidecar["artifact"]["artifact_kind"] for sidecar in sidecars}
    if len(kinds) != 1:
        raise histogram_merge_error(
            "Cannot merge incompatible histogram artifact kinds: "
            + ", ".join(sorted(kinds))
        )
    kind = next(iter(kinds))
    if kind == NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND:
        raise histogram_merge_error(
            "Nominal-only reference artifacts are non-card-ready comparison "
            "products and cannot be merged."
        )
    layouts = {
        (
            sidecar["artifact"]["nominal_container_schema_version"],
            sidecar["artifact"]["nominal_container_layout"],
        )
        for sidecar in sidecars
    }
    if len(layouts) != 1:
        raise histogram_merge_error("Cannot merge incompatible nominal schemas/layouts.")
    (
        provenance,
        production_sample_contract,
        requested_data_driven_products,
        resolved_data_driven_contract,
    ) = _compose_merged_contract_set(sidecars)
    family_order = tuple(provenance["runtime_histogram_families"])
    for sidecar in sidecars:
        manifest_order = tuple(sidecar["sumw2_content_manifest"]["families"])
        provenance_order = tuple(
            sidecar["sumw2_storage_provenance"]["runtime_histogram_families"]
        )
        if manifest_order != provenance_order:
            raise histogram_merge_error(
                "Cannot merge a sidecar whose content-manifest family order does not "
                "match its runtime histogram-family provenance."
            )
    required = {}
    for family in family_order:
        family_sidecars = [
            sidecar
            for sidecar in sidecars
            if family in sidecar["sumw2_content_manifest"]["families"]
        ]
        dimensions = {
            sidecar["sumw2_content_manifest"]["families"][family]["dimensionality"]
            for sidecar in family_sidecars
        }
        if len(dimensions) != 1:
            raise histogram_merge_error(
                f"Cannot merge incompatible dimensionality for family '{family}'."
            )
        required[family] = sorted(
            {
                process
                for sidecar in family_sidecars
                for process in sidecar["sumw2_content_manifest"]["families"][family][
                    "required_sumw2_processes"
                ]
            }
        )
    merged_contract = None
    if kind != "processor_output":
        try:
            normalized_contracts = tuple(
                _normalize_transformation_contract(
                    sidecar["transformation_contract"],
                    sumw2_storage_provenance=sidecar[
                        "sumw2_storage_provenance"
                    ],
                    artifact_kind=kind,
                )
                for sidecar in sidecars
            )
        except histogram_sidecar_error as error:
            raise histogram_merge_error(
                f"Cannot merge a tampered transformation contract: {error}"
            ) from error
        contract_versions = {
            contract["contract_version"] for contract in normalized_contracts
        }
        if len(contract_versions) != 1:
            raise histogram_merge_error(
                "Cannot merge prior and applicability-aware transformation "
                "contract versions."
            )
        contract_version = next(iter(contract_versions))
        projection_modes = {
            contract["eft_prompt_projection"]["mode"]
            for contract in normalized_contracts
        }
        projection_dependence = {
            contract["eft_prompt_projection"][
                "generated_nonprompt_eft_dependence"
            ]
            for contract in normalized_contracts
        }
        if projection_modes != {"sm_point"} or projection_dependence != {False}:
            raise histogram_merge_error(
                "Cannot merge inconsistent EFT prompt projection provenance."
            )
        merged_projection = {
            "mode": "sm_point",
            "required_processes": sorted(
                {
                    process
                    for contract in normalized_contracts
                    for process in contract["eft_prompt_projection"][
                        "required_processes"
                    ]
                }
            ),
            "generated_nonprompt_eft_dependence": False,
        }
        merged_contract_families = {}
        for family in family_order:
            family_contracts = [
                contract
                for contract in normalized_contracts
                if family in contract["families"]
            ]
            merged_roles = {}
            for field_name in _TRANSFORMATION_ROLE_FIELDS:
                merged_roles[field_name] = sorted(
                    {
                        process
                        for contract in family_contracts
                        for process in contract["families"][family][field_name]
                    }
                )
            if contract_version == TRANSFORMATION_CONTRACT_VERSION:
                source_application_regions = sorted(
                    {
                        region
                        for contract in family_contracts
                        for region in contract["families"][family][
                            "source_application_regions"
                        ]
                    }
                )
                merged_roles[
                    "source_application_regions"
                ] = source_application_regions
                merged_roles[
                    "applicable_products"
                ] = _derive_data_driven_applicability(
                    source_application_regions
                )
            source_processes = set(merged_roles["source_scalar_processes"]) | set(
                merged_roles["source_eft_processes"]
            )
            retained_processes = set(merged_roles["retained_scalar_processes"]) | set(
                merged_roles["retained_eft_processes"]
            )
            merged_roles["consumed_source_processes"] = sorted(
                source_processes - retained_processes
            )
            merged_contract_families[family] = merged_roles
        merged_contract = _normalize_transformation_contract(
            {
                "contract_version": contract_version,
                "artifact_kind": kind,
                "eft_prompt_projection": merged_projection,
                "families": merged_contract_families,
            },
            sumw2_storage_provenance=provenance,
            artifact_kind=kind,
        )
        independently_required = required_sumw2_processes_from_transformation_contract(
            merged_contract,
            sumw2_storage_provenance=provenance,
            resolved_data_driven_contract=resolved_data_driven_contract,
        )
        if required != independently_required:
            raise histogram_merge_error(
                "Merged transformed requirements disagree with the union of "
                "independently validated transformation contracts."
            )
    return {
        "artifact_kind": kind,
        "merged": True,
        "sumw2_storage_provenance": provenance,
        "production_sample_contract": production_sample_contract,
        "required_sumw2_processes": required,
        "transformation_contract": merged_contract,
        "requested_data_driven_products": requested_data_driven_products,
        "resolved_data_driven_contract": resolved_data_driven_contract,
        "lineage_inputs": [lineage_input_from_sidecar(sidecar) for sidecar in sidecars],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def write_histogram_sidecar(
    pkl_path: str | os.PathLike[str],
    *,
    histograms: Mapping[str, Any],
    artifact_kind: str,
    sumw2_storage_provenance: Mapping[str, Any],
    merged: bool = False,
    lineage_inputs: Iterable[Mapping[str, Any]] = (),
    required_sumw2_processes: Mapping[str, Iterable[str]] | None = None,
    input_sidecar: Mapping[str, Any] | None = None,
    transformation_context: Mapping[str, Any] | None = None,
    transformation_contract: Mapping[str, Any] | None = None,
    requested_data_driven_products: Mapping[str, Any] | None = None,
    resolved_data_driven_contract: Mapping[str, Any] | None = None,
    production_sample_contract: Mapping[str, Any] | None = None,
    nominal_reference_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one automatic sidecar for an already finalized PKL."""

    pkl_path = Path(pkl_path)
    if artifact_kind != "processor_output":
        if merged:
            if input_sidecar is not None or transformation_context is not None:
                raise histogram_sidecar_error(
                    "Merged transformed publication accepts only the merged "
                    "transformation_contract derived from validated inputs."
                )
        elif transformation_contract is not None:
            raise histogram_sidecar_error(
                "Original transformed publication cannot accept a caller-authored "
                "transformation_contract; provide its validated input sidecar and "
                "generated transformation context."
            )
        elif input_sidecar is None:
            raise histogram_sidecar_error(
                "Original transformed publication requires its validated input "
                "sidecar and generated transformation context."
            )
    if input_sidecar is not None or transformation_context is not None:
        if input_sidecar is None or transformation_context is None:
            raise histogram_sidecar_error(
                "Original transformed publication requires both input_sidecar and "
                "transformation_context."
            )
        _require_immutable_input_provenance(
            input_sidecar,
            sumw2_storage_provenance,
        )
        derived_contract, derived_required = derive_transformed_required_sumw2_processes(
            input_sidecar=input_sidecar,
            transformation_context=transformation_context,
            artifact_kind=artifact_kind,
            transformed_histograms=histograms,
        )
        if transformation_contract is not None and dict(transformation_contract) != derived_contract:
            raise histogram_sidecar_error(
                "Explicit transformation_contract disagrees with the independently "
                "derived original transformation contract."
            )
        transformation_contract = derived_contract
        if required_sumw2_processes is not None and _normalize_required_processes(
            required_sumw2_processes
        ) != derived_required:
            raise histogram_sidecar_error(
                "Explicit required_sumw2_processes disagree with the independently "
                "derived transformed requirements."
            )
        required_sumw2_processes = derived_required
    if input_sidecar is not None:
        input_requested = input_sidecar.get("requested_data_driven_products")
        input_contract = input_sidecar.get("resolved_data_driven_contract")
        if requested_data_driven_products is not None and dict(
            requested_data_driven_products
        ) != input_requested:
            raise histogram_sidecar_error(
                "Transformed output must preserve requested_data_driven_products unchanged."
            )
        if resolved_data_driven_contract is not None and dict(
            resolved_data_driven_contract
        ) != input_contract:
            raise histogram_sidecar_error(
                "Transformed output must preserve resolved_data_driven_contract unchanged."
            )
        requested_data_driven_products = input_requested
        resolved_data_driven_contract = input_contract
        input_production_contract = input_sidecar.get("production_sample_contract")
        if production_sample_contract is not None and dict(
            production_sample_contract
        ) != input_production_contract:
            raise histogram_sidecar_error(
                "Transformed output must preserve production_sample_contract unchanged."
            )
        production_sample_contract = input_production_contract
        input_reference_contract = input_sidecar.get("nominal_reference_contract")
        if nominal_reference_contract is not None and dict(
            nominal_reference_contract
        ) != input_reference_contract:
            raise histogram_sidecar_error(
                "Transformed output must preserve nominal_reference_contract unchanged."
            )
        nominal_reference_contract = input_reference_contract
    payload = _build_sidecar_payload(
        pkl_path,
        histograms,
        identity_path=pkl_path,
        artifact_kind=artifact_kind,
        merged=merged,
        sumw2_storage_provenance=sumw2_storage_provenance,
        lineage_inputs=lineage_inputs,
        required_sumw2_processes=required_sumw2_processes,
        transformation_contract=transformation_contract,
        requested_data_driven_products=requested_data_driven_products,
        resolved_data_driven_contract=resolved_data_driven_contract,
        production_sample_contract=production_sample_contract,
        nominal_reference_contract=nominal_reference_contract,
    )
    _validate_sidecar_structure(payload, pkl_path=pkl_path)
    temporary_path = metadata_sidecar_path(pkl_path).with_name(
        f".{metadata_sidecar_path(pkl_path).name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        _write_json(temporary_path, payload)
        os.replace(temporary_path, metadata_sidecar_path(pkl_path))
    finally:
        temporary_path.unlink(missing_ok=True)
    return payload


def _default_pickle_writer(path: str, histograms: Mapping[str, Any]) -> None:
    with gzip.open(path, "wb") as stream:
        cloudpickle.dump(histograms, stream)


def write_histogram_artifact(
    pkl_path: str | os.PathLike[str],
    *,
    artifact_kind: str,
    sumw2_storage_provenance: Mapping[str, Any],
    histograms: Mapping[str, Any] | None = None,
    payload_writer: Callable[[str], Mapping[str, Any] | None] | None = None,
    merged: bool = False,
    lineage_inputs: Iterable[Mapping[str, Any]] = (),
    required_sumw2_processes: Mapping[str, Iterable[str]] | None = None,
    input_sidecar: Mapping[str, Any] | None = None,
    transformation_context: Mapping[str, Any] | None = None,
    transformation_contract: Mapping[str, Any] | None = None,
    requested_data_driven_products: Mapping[str, Any] | None = None,
    resolved_data_driven_contract: Mapping[str, Any] | None = None,
    production_sample_contract: Mapping[str, Any] | None = None,
    nominal_reference_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage, validate, and publish a PKL/sidecar pair as one logical output."""

    if (histograms is None) == (payload_writer is None):
        raise ValueError("Provide exactly one of histograms or payload_writer.")
    if artifact_kind != "processor_output":
        if merged:
            if input_sidecar is not None or transformation_context is not None:
                raise histogram_sidecar_error(
                    "Merged transformed publication accepts only the merged "
                    "transformation_contract derived from validated inputs."
                )
        elif transformation_contract is not None:
            raise histogram_sidecar_error(
                "Original transformed publication cannot accept a caller-authored "
                "transformation_contract; provide its validated input sidecar and "
                "generated transformation context."
            )
        elif input_sidecar is None:
            raise histogram_sidecar_error(
                "Original transformed publication requires its validated input "
                "sidecar and generated transformation context."
            )
    pkl_path = Path(pkl_path)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staged_pkl = pkl_path.parent / f".{pkl_path.name}.{token}.pkl.gz"
    staged_sidecar = pkl_path.parent / f".{pkl_path.name}.{token}.metadata.json.tmp"
    final_sidecar = metadata_sidecar_path(pkl_path)
    backup_pkl = pkl_path.parent / f".{pkl_path.name}.{token}.backup"
    backup_sidecar = pkl_path.parent / f".{final_sidecar.name}.{token}.backup"
    had_pkl = pkl_path.exists()
    had_sidecar = final_sidecar.exists()
    published_pkl = False
    published_sidecar = False
    try:
        if payload_writer is not None:
            generated_context = payload_writer(str(staged_pkl))
            if generated_context is not None:
                if transformation_context is not None:
                    raise histogram_sidecar_error(
                        "Transformation context was supplied both directly and by "
                        "the payload writer."
                    )
                transformation_context = generated_context
        else:
            assert histograms is not None
            _default_pickle_writer(str(staged_pkl), histograms)
        manifest_histograms = (
            dict(histograms)
            if histograms is not None
            else _load_histograms(staged_pkl)
        )
        if input_sidecar is not None or transformation_context is not None:
            if input_sidecar is None or transformation_context is None:
                raise histogram_sidecar_error(
                    "Original transformed publication requires both input_sidecar "
                    "and transformation_context."
                )
            _require_immutable_input_provenance(
                input_sidecar,
                sumw2_storage_provenance,
            )
            derived_contract, derived_required = derive_transformed_required_sumw2_processes(
                input_sidecar=input_sidecar,
                transformation_context=transformation_context,
                artifact_kind=artifact_kind,
                transformed_histograms=manifest_histograms,
            )
            if transformation_contract is not None and dict(transformation_contract) != derived_contract:
                raise histogram_sidecar_error(
                    "Explicit transformation_contract disagrees with the independently "
                    "derived original transformation contract."
                )
            transformation_contract = derived_contract
            if required_sumw2_processes is not None and _normalize_required_processes(
                required_sumw2_processes
            ) != derived_required:
                raise histogram_sidecar_error(
                    "Explicit required_sumw2_processes disagree with the independently "
                    "derived transformed requirements."
                )
            required_sumw2_processes = derived_required
        if input_sidecar is not None:
            input_requested = input_sidecar.get("requested_data_driven_products")
            input_contract = input_sidecar.get("resolved_data_driven_contract")
            if requested_data_driven_products is not None and dict(
                requested_data_driven_products
            ) != input_requested:
                raise histogram_sidecar_error(
                    "Transformed output must preserve requested_data_driven_products unchanged."
                )
            if resolved_data_driven_contract is not None and dict(
                resolved_data_driven_contract
            ) != input_contract:
                raise histogram_sidecar_error(
                    "Transformed output must preserve resolved_data_driven_contract unchanged."
                )
            requested_data_driven_products = input_requested
            resolved_data_driven_contract = input_contract
            input_production_contract = input_sidecar.get(
                "production_sample_contract"
            )
            if production_sample_contract is not None and dict(
                production_sample_contract
            ) != input_production_contract:
                raise histogram_sidecar_error(
                    "Transformed output must preserve production_sample_contract unchanged."
                )
            production_sample_contract = input_production_contract
            input_reference_contract = input_sidecar.get("nominal_reference_contract")
            if nominal_reference_contract is not None and dict(
                nominal_reference_contract
            ) != input_reference_contract:
                raise histogram_sidecar_error(
                    "Transformed output must preserve nominal_reference_contract unchanged."
                )
            nominal_reference_contract = input_reference_contract
        sidecar = _build_sidecar_payload(
            pkl_path,
            manifest_histograms,
            identity_path=staged_pkl,
            artifact_kind=artifact_kind,
            merged=merged,
            sumw2_storage_provenance=sumw2_storage_provenance,
            lineage_inputs=lineage_inputs,
            required_sumw2_processes=required_sumw2_processes,
            transformation_contract=transformation_contract,
            requested_data_driven_products=requested_data_driven_products,
            resolved_data_driven_contract=resolved_data_driven_contract,
            production_sample_contract=production_sample_contract,
            nominal_reference_contract=nominal_reference_contract,
        )
        _validate_sidecar_structure(sidecar, pkl_path=pkl_path)
        _validate_content_manifest(pkl_path, manifest_histograms, sidecar)
        if artifact_kind == "processor_output":
            validate_processor_output(pkl_path, manifest_histograms, sidecar)
        elif artifact_kind == NONPROMPT_OUTPUT_ARTIFACT_KIND:
            validate_nonprompt_output(pkl_path, manifest_histograms, sidecar)
        elif artifact_kind == NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND:
            validate_nonprompt_nominal_reference_output(
                pkl_path,
                manifest_histograms,
                sidecar,
            )
        else:
            validate_flips_output(pkl_path, manifest_histograms, sidecar)
        _write_json(staged_sidecar, sidecar)

        if had_pkl:
            os.replace(pkl_path, backup_pkl)
        if had_sidecar:
            os.replace(final_sidecar, backup_sidecar)
        os.replace(staged_pkl, pkl_path)
        published_pkl = True
        os.replace(staged_sidecar, final_sidecar)
        published_sidecar = True
        backup_pkl.unlink(missing_ok=True)
        backup_sidecar.unlink(missing_ok=True)
        return sidecar
    except Exception:
        if published_sidecar:
            final_sidecar.unlink(missing_ok=True)
        if published_pkl:
            pkl_path.unlink(missing_ok=True)
        if backup_pkl.exists():
            os.replace(backup_pkl, pkl_path)
        if backup_sidecar.exists():
            os.replace(backup_sidecar, final_sidecar)
        raise
    finally:
        staged_pkl.unlink(missing_ok=True)
        staged_sidecar.unlink(missing_ok=True)
        backup_pkl.unlink(missing_ok=True)
        backup_sidecar.unlink(missing_ok=True)
