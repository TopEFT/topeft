"""Certify active cfg samples against the selected production signal profile."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from topeft.modules.sumw2_policy import (
    SUMW2_PROVENANCE_SCHEMA_VERSION,
    resolved_policy_from_provenance,
    resolved_sumw2_policy,
    sumw2_mode_resolution,
)


PRODUCTION_SAMPLE_CONTRACT_VERSION = 1


class production_sample_profile_error(ValueError):
    """The active sample universe is incompatible with its declared profile."""


@dataclass(frozen=True)
class signal_variant_group:
    name: str
    years: tuple[str, ...]
    private_bases: tuple[str, ...]
    central_bases: tuple[str, ...]


# These groups are intentionally narrower than the available signal cfgs.  Each
# pairing is supported by maintained cfg membership plus process-role/xsec
# evidence.  Unpaired signals remain ordinary active cfg members.
VALIDATED_SIGNAL_VARIANT_GROUPS = (
    signal_variant_group(
        "tllq",
        ("UL16APV", "UL16", "UL17", "UL18", "2022", "2022EE", "2023", "2023BPix"),
        ("tllq_private",),
        ("tZq_central",),
    ),
    signal_variant_group(
        "tttt_run2",
        ("UL16APV", "UL16", "UL17", "UL18"),
        ("tttt_private",),
        ("tttt_central",),
    ),
    signal_variant_group(
        "tttt_run3",
        ("2022", "2022EE", "2023", "2023BPix"),
        ("tttt_private",),
        ("TTTT_central",),
    ),
    signal_variant_group(
        "tth_run2",
        ("UL16APV", "UL16", "UL17", "UL18"),
        ("ttHJet_private",),
        ("ttHJet_central", "ttH_central"),
    ),
    signal_variant_group(
        "ttlnu_run2",
        ("UL16APV", "UL16", "UL17", "UL18"),
        ("ttlnuJet_private",),
        ("ttW_central",),
    ),
    signal_variant_group(
        "ttll_run2",
        ("UL16APV", "UL16", "UL17", "UL18"),
        ("ttllJet_private",),
        ("ttZ_central",),
    ),
    signal_variant_group(
        "ttlnu_run3",
        ("2022", "2022EE", "2023", "2023BPix"),
        ("ttlnu_private",),
        ("ttLNu_cental",),
    ),
)


@dataclass(frozen=True)
class active_sample_universe:
    wrapper_identity: str
    cfg_identities: tuple[tuple[tuple[str, str], ...], ...]
    datasets: tuple[str, ...]
    processes: tuple[str, ...]

    def serialized_cfg_identities(self) -> list[dict[str, str]]:
        return [dict(identity) for identity in self.cfg_identities]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _cfg_identity(path: str) -> tuple[tuple[str, str], ...]:
    with open(path, "rb") as stream:
        digest = _sha256_bytes(stream.read())
    basename = os.path.basename(path)
    input_kind = "json" if basename.endswith(".json") else "cfg"
    return tuple(
        sorted(
            {
                "basename": basename,
                "content_sha256": digest,
                "input_kind": input_kind,
            }.items()
        )
    )


def build_active_sample_universe(
    samples: Mapping[str, Mapping[str, Any]],
    *,
    input_paths: Sequence[str] = (),
    wrapper_identity: str = "run_analysis.py",
) -> active_sample_universe:
    """Freeze the exact dataset/process universe selected by the active inputs."""

    if not isinstance(wrapper_identity, str) or not wrapper_identity:
        raise production_sample_profile_error(
            "SUMW2-PROFILE-E006: wrapper identity must be a nonempty string."
        )
    datasets = []
    processes = []
    for dataset, sample in samples.items():
        process = sample.get("histAxisName") if isinstance(sample, Mapping) else None
        if (
            not isinstance(dataset, str)
            or not dataset
            or not isinstance(process, str)
            or not process
        ):
            raise production_sample_profile_error(
                "SUMW2-PROFILE-E006: active sample metadata requires nonempty "
                "dataset keys and histAxisName process labels."
            )
        datasets.append(dataset)
        processes.append(process)
    if input_paths:
        cfg_identities = tuple(
            sorted(_cfg_identity(path) for path in input_paths)
        )
    else:
        generated_identity = {
            "basename": "<direct-library-input>",
            "content_sha256": _sha256_bytes(
                _canonical_json(
                    {
                        dataset: samples[dataset]["histAxisName"]
                        for dataset in sorted(samples)
                    }
                ).encode("utf-8")
            ),
            "input_kind": "generated_active_universe",
        }
        cfg_identities = (tuple(sorted(generated_identity.items())),)
    return active_sample_universe(
        wrapper_identity=wrapper_identity,
        cfg_identities=cfg_identities,
        datasets=tuple(sorted(datasets)),
        processes=tuple(sorted(set(processes))),
    )


def _group_active_variants(
    processes: Sequence[str],
    *,
    groups: Sequence[signal_variant_group] = VALIDATED_SIGNAL_VARIANT_GROUPS,
) -> tuple[dict[str, Any], ...]:
    from topeft.modules.data_driven_products import parse_process_name

    assignments: dict[str, tuple[str, str]] = {}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    recognized_bases = {
        base
        for group in groups
        for base in (*group.private_bases, *group.central_bases)
    }
    for process in sorted(set(processes)):
        try:
            base, year = parse_process_name(process)
        except ValueError as error:
            if any(process.startswith(base) for base in recognized_bases):
                raise production_sample_profile_error(
                    "SUMW2-PROFILE-E006: signal variant mapping is ambiguous or unsupported; "
                    f"recognized signal variant process={process!r} lacks a maintained "
                    "canonical year suffix. Recommended correction: use one of UL16APV, "
                    "UL16, UL17, UL18, 2022, 2022EE, 2023, or 2023BPix."
                ) from error
            continue
        matches = []
        for group in groups:
            if year not in group.years:
                continue
            if base in group.private_bases:
                matches.append((group, "private"))
            if base in group.central_bases:
                matches.append((group, "central"))
        if len(matches) > 1:
            raise production_sample_profile_error(
                "SUMW2-PROFILE-E006: signal variant mapping is ambiguous or unsupported; "
                f"process={process!r} matches={[(group.name, side) for group, side in matches]}. "
                "Recommended correction: repair the centralized validated signal-variant catalog."
            )
        if not matches:
            continue
        group, side = matches[0]
        previous = assignments.get(process)
        assignment = (group.name, side)
        if previous is not None and previous != assignment:
            raise production_sample_profile_error(
                "SUMW2-PROFILE-E006: signal variant mapping is ambiguous or unsupported; "
                f"process={process!r} assignments={[previous, assignment]}."
            )
        assignments[process] = assignment
        key = (group.name, year)
        record = records.setdefault(
            key,
            {
                "signal_group": group.name,
                "year": year,
                "private_processes": [],
                "central_processes": [],
                "private_variants": list(group.private_bases),
                "central_variants": list(group.central_bases),
            },
        )
        record[f"{side}_processes"].append(process)

    output = []
    for key in sorted(records):
        record = records[key]
        record["private_processes"].sort()
        record["central_processes"].sort()
        if len(record["private_processes"]) > 1 or len(record["central_processes"]) > 1:
            raise production_sample_profile_error(
                "SUMW2-PROFILE-E006: signal variant mapping is ambiguous or unsupported; "
                f"signal_group={record['signal_group']!r} year={record['year']!r} "
                f"private_processes={record['private_processes']} "
                f"central_processes={record['central_processes']}. "
                "Recommended correction: keep one supported process per variant or refine the catalog."
            )
        output.append(record)
    return tuple(output)


def _format_profile_error(
    error_id: str,
    summary: str,
    *,
    universe: active_sample_universe,
    mode_resolution: sumw2_mode_resolution,
    metadata_path: str,
    signal_group_name: str = "<not-applicable>",
    private_variants: Sequence[str] = (),
    central_variants: Sequence[str] = (),
    contributors: Sequence[str] = (),
    product: str = "<not-applicable>",
    family: str = "<not-applicable>",
    conflicts: Sequence[str] = (),
    metadata_source: str = "<not-provided>",
    correction: str,
) -> str:
    cfg_ids = [identity["basename"] for identity in universe.serialized_cfg_identities()]
    return (
        f"{error_id}: {summary}; metadata_path/source={metadata_path!r}; "
        f"wrapper={universe.wrapper_identity!r}; sr_cfg_identities={cfg_ids}; "
        f"resolved_mode={mode_resolution.resolved_mode!r}; "
        f"expected_signal_profile={mode_resolution.signal_sample_profile!r}; "
        f"signal_group={signal_group_name!r}; private_variant={list(private_variants)}; "
        f"central_variant={list(central_variants)}; "
        f"active_cfg_processes={list(universe.processes)}; "
        f"resolved_contributor_processes={sorted(set(contributors))}; "
        f"metadata_source={metadata_source!r}; "
        f"affected_data_driven_product={product!r}; affected_family={family!r}; "
        f"missing_or_conflicting_targets={list(conflicts)}. "
        f"Recommended correction: {correction}"
    )


def _configured_selector_processes(data_driven_products: Any) -> tuple[str, ...]:
    if not isinstance(data_driven_products, Mapping):
        return ()
    configured = []
    for product in data_driven_products.values():
        if not isinstance(product, Mapping) or not product.get("enabled"):
            continue
        roles = product.get("source_contributors")
        if not isinstance(roles, Mapping):
            continue
        for selector in roles.values():
            if not isinstance(selector, Mapping):
                continue
            names = selector.get("process_names", ())
            if isinstance(names, list):
                configured.extend(name for name in names if isinstance(name, str))
    return tuple(sorted(set(configured)))


def validate_active_sample_profile(
    universe: active_sample_universe,
    mode_resolution: sumw2_mode_resolution,
    *,
    data_driven_products: Mapping[str, Any] | None = None,
    data_driven_products_present: bool = False,
    metadata_path: str | None = None,
) -> None:
    """Validate cfg -> mode and explicit selectors -> cfg before resolution."""

    metadata_source = metadata_path or "<command-line/default>"
    configured = _configured_selector_processes(data_driven_products)
    if data_driven_products_present and isinstance(data_driven_products, Mapping):
        for product_name, product in data_driven_products.items():
            if not isinstance(product, Mapping) or not product.get("enabled"):
                continue
            roles = product.get("source_contributors")
            if not isinstance(roles, Mapping):
                continue
            for role_name, selector in roles.items():
                if not isinstance(selector, Mapping):
                    continue
                process_names = selector.get("process_names", ())
                if not isinstance(process_names, list):
                    process_names = ()
                for process in process_names:
                    if process not in universe.processes:
                        raise production_sample_profile_error(
                            _format_profile_error(
                                "SUMW2-PROFILE-E004",
                                "explicit data-driven contributor is absent from the active cfg universe",
                                universe=universe,
                                mode_resolution=mode_resolution,
                                metadata_path=metadata_source,
                                contributors=configured,
                                product=str(product_name),
                                conflicts=(f"{role_name}:{process}",),
                                correction="correct the explicit contributor selector or add the intended sample to the active cfg",
                            )
                        )
                process_prefixes = selector.get("process_prefixes", ())
                if not isinstance(process_prefixes, list):
                    process_prefixes = ()
                for prefix in process_prefixes:
                    if not any(process.startswith(prefix) for process in universe.processes):
                        raise production_sample_profile_error(
                            _format_profile_error(
                                "SUMW2-PROFILE-E004",
                                "explicit data-driven contributor prefix is absent from the active cfg universe",
                                universe=universe,
                                mode_resolution=mode_resolution,
                                metadata_path=metadata_source,
                                contributors=configured,
                                product=str(product_name),
                                conflicts=(f"{role_name}:prefix:{prefix}",),
                                correction="correct the explicit contributor selector or add the intended sample to the active cfg",
                            )
                        )

    try:
        active_variant_records = _group_active_variants(universe.processes)
    except production_sample_profile_error as error:
        raise production_sample_profile_error(
            _format_profile_error(
                "SUMW2-PROFILE-E006",
                "signal variant mapping is ambiguous or unsupported",
                universe=universe,
                mode_resolution=mode_resolution,
                metadata_path=metadata_source,
                contributors=configured,
                conflicts=(str(error),),
                correction="repair the centralized validated signal-variant catalog before processing",
            )
        ) from error

    for record in active_variant_records:
        private_processes = record["private_processes"]
        central_processes = record["central_processes"]
        if private_processes and central_processes:
            raise production_sample_profile_error(
                _format_profile_error(
                    "SUMW2-PROFILE-E003",
                    "both private and central variants of one validated signal group are active or selected",
                    universe=universe,
                    mode_resolution=mode_resolution,
                    metadata_path=metadata_source,
                    signal_group_name=f"{record['signal_group']}:{record['year']}",
                    private_variants=private_processes,
                    central_variants=central_processes,
                    contributors=configured,
                    conflicts=private_processes + central_processes,
                    correction="remove one duplicate private/central variant from the cfg",
                )
            )
        if mode_resolution.signal_sample_profile == "private" and central_processes:
            raise production_sample_profile_error(
                _format_profile_error(
                    "SUMW2-PROFILE-E001",
                    "production selected but the active cfg contains only the central variant",
                    universe=universe,
                    mode_resolution=mode_resolution,
                    metadata_path=metadata_source,
                    signal_group_name=f"{record['signal_group']}:{record['year']}",
                    private_variants=record["private_variants"],
                    central_variants=central_processes,
                    contributors=configured,
                    conflicts=central_processes,
                    correction="use mode production_central with the central-signal cfg, or restore the private-signal cfg",
                )
            )
        if mode_resolution.signal_sample_profile == "central" and private_processes:
            raise production_sample_profile_error(
                _format_profile_error(
                    "SUMW2-PROFILE-E002",
                    "production_central selected but the active cfg contains only the private variant",
                    universe=universe,
                    mode_resolution=mode_resolution,
                    metadata_path=metadata_source,
                    signal_group_name=f"{record['signal_group']}:{record['year']}",
                    private_variants=private_processes,
                    central_variants=record["central_variants"],
                    contributors=configured,
                    conflicts=private_processes,
                    correction="use mode production with the private-signal cfg, or use the central-signal cfg",
                )
            )


def _resolved_contributors(resolved_products: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                process
                for _product_name, product in resolved_products.products
                if product.enabled
                for _role, processes in product.source_contributors
                for process in processes
            }
        )
    )


def certify_production_sample_contract(
    universe: active_sample_universe,
    policy: resolved_sumw2_policy,
    resolved_products: Any,
) -> dict[str, Any]:
    """Certify contributors and targets, then create deterministic provenance."""

    mode_resolution = sumw2_mode_resolution(
        source=policy.source,
        requested_mode=policy.requested_mode,
        resolved_mode=policy.resolved_mode,
        signal_sample_profile=policy.signal_sample_profile,
        sumw2_storage={},
        warnings=policy.warnings,
    )
    validate_active_sample_profile(
        universe,
        mode_resolution,
        metadata_path=resolved_products.metadata_path,
    )
    if (
        universe.datasets != policy.resolved_datasets
        or universe.processes != policy.resolved_processes
    ):
        raise production_sample_profile_error(
            _format_profile_error(
                "SUMW2-PROFILE-E006",
                "resolved sumw2 universe does not equal the active cfg universe",
                universe=universe,
                mode_resolution=mode_resolution,
                metadata_path=resolved_products.metadata_path,
                conflicts=(
                    f"cfg_datasets={list(universe.datasets)} policy_datasets={list(policy.resolved_datasets)}",
                    f"cfg_processes={list(universe.processes)} policy_processes={list(policy.resolved_processes)}",
                ),
                correction="resolve the sumw2 policy from the unchanged active cfg sample mapping",
            )
        )
    contributors = _resolved_contributors(resolved_products)
    missing_contributors = sorted(set(contributors) - set(universe.processes))
    if missing_contributors:
        raise production_sample_profile_error(
            _format_profile_error(
                "SUMW2-PROFILE-E004",
                "resolved data-driven contributor is absent from the active cfg universe",
                universe=universe,
                mode_resolution=mode_resolution,
                metadata_path=resolved_products.metadata_path,
                contributors=contributors,
                conflicts=missing_contributors,
                correction="correct the explicit contributor selector or restore the intended active cfg sample",
            )
        )

    selected_targets = set(policy.resolved_targets)
    for product_name, product in resolved_products.products:
        if not product.enabled:
            continue
        required_targets = {
            target
            for target in resolved_products.required_targets()
            if target.process in product.required_processes()
        }
        missing_targets = sorted(required_targets - selected_targets)
        if missing_targets:
            first = missing_targets[0]
            formatted = [
                f"{target.dataset}/{target.process}/{target.family}"
                for target in missing_targets
            ]
            raise production_sample_profile_error(
                _format_profile_error(
                    "SUMW2-PROFILE-E005",
                    "required active contributor lacks a dataset/process/family sumw2 target",
                    universe=universe,
                    mode_resolution=mode_resolution,
                    metadata_path=resolved_products.metadata_path,
                    contributors=contributors,
                    product=product_name,
                    family=first.family,
                    conflicts=formatted,
                    correction="add the missing process/family target to the selected sumw2 rules",
                )
            )

    active_variants = {}
    for record in _group_active_variants(universe.processes):
        selected_variant = "private" if record["private_processes"] else "central"
        processes = record[f"{selected_variant}_processes"]
        active_variants[f"{record['signal_group']}:{record['year']}"] = {
            "signal_group": record["signal_group"],
            "year": record["year"],
            "selected_variant": selected_variant,
            "processes": list(processes),
        }
    contract = {
        "contract_version": PRODUCTION_SAMPLE_CONTRACT_VERSION,
        "wrapper_identity": universe.wrapper_identity,
        "cfg_identities": universe.serialized_cfg_identities(),
        "resolved_mode": policy.resolved_mode,
        "signal_sample_profile": policy.signal_sample_profile,
        "active_signal_variants": active_variants,
        "compatibility_validated": True,
    }
    contract["contract_identity_sha256"] = _sha256_bytes(
        _canonical_json(contract).encode("utf-8")
    )
    validate_production_sample_contract(contract, policy)
    return contract


def validate_production_sample_contract(
    contract: Mapping[str, Any],
    policy: resolved_sumw2_policy | Mapping[str, Any],
) -> None:
    """Recompute the active signal map from sumw2 provenance and reject tampering."""

    if not isinstance(contract, Mapping):
        raise production_sample_profile_error(
            "production_sample_contract must be a mapping."
        )
    required = {
        "contract_version",
        "wrapper_identity",
        "cfg_identities",
        "resolved_mode",
        "signal_sample_profile",
        "active_signal_variants",
        "compatibility_validated",
        "contract_identity_sha256",
    }
    if set(contract) != required:
        raise production_sample_profile_error(
            f"Invalid production_sample_contract fields; missing={sorted(required - set(contract))} "
            f"unknown={sorted(set(contract) - required)}."
        )
    if contract["contract_version"] != PRODUCTION_SAMPLE_CONTRACT_VERSION:
        raise production_sample_profile_error(
            f"Unsupported production_sample_contract version {contract['contract_version']!r}."
        )
    unsigned = dict(contract)
    observed_identity = unsigned.pop("contract_identity_sha256")
    expected_identity = _sha256_bytes(_canonical_json(unsigned).encode("utf-8"))
    if observed_identity != expected_identity:
        raise production_sample_profile_error(
            "production_sample_contract identity does not match its content."
        )
    parsed_policy = (
        policy
        if isinstance(policy, resolved_sumw2_policy)
        else resolved_policy_from_provenance(policy)
    )
    if parsed_policy.schema_version != SUMW2_PROVENANCE_SCHEMA_VERSION:
        raise production_sample_profile_error(
            "A production sample contract requires current sumw2 provenance."
        )
    if contract["resolved_mode"] != parsed_policy.resolved_mode or contract[
        "signal_sample_profile"
    ] != parsed_policy.signal_sample_profile:
        raise production_sample_profile_error(
            "production_sample_contract contradicts sumw2 mode/profile provenance."
        )
    if contract["compatibility_validated"] is not True:
        raise production_sample_profile_error(
            "production_sample_contract is not certified compatible."
        )
    if (
        not isinstance(contract["wrapper_identity"], str)
        or not contract["wrapper_identity"]
    ):
        raise production_sample_profile_error(
            "Invalid wrapper identity in production sample contract."
        )
    cfg_identities = contract["cfg_identities"]
    if not isinstance(cfg_identities, list) or not cfg_identities:
        raise production_sample_profile_error(
            "Invalid cfg identities in production sample contract."
        )
    cfg_fields = {"basename", "content_sha256", "input_kind"}
    for identity in cfg_identities:
        if not isinstance(identity, Mapping) or set(identity) != cfg_fields:
            raise production_sample_profile_error(
                "Invalid cfg identity in production sample contract."
            )
        if any(
            not isinstance(identity[field], str) or not identity[field]
            for field in cfg_fields
        ):
            raise production_sample_profile_error(
                "Invalid cfg identity value in production sample contract."
            )
        digest = identity["content_sha256"]
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise production_sample_profile_error(
                "Invalid cfg identity SHA-256 in production sample contract."
            )
    if cfg_identities != sorted(
        cfg_identities,
        key=lambda identity: tuple(sorted(identity.items())),
    ):
        raise production_sample_profile_error(
            "Cfg identities must be in deterministic canonical order."
        )
    if len({_canonical_json(identity) for identity in cfg_identities}) != len(
        cfg_identities
    ):
        raise production_sample_profile_error(
            "Cfg identities must be unique."
        )
    expected_variants = {}
    for record in _group_active_variants(parsed_policy.resolved_processes):
        private_processes = record["private_processes"]
        central_processes = record["central_processes"]
        if private_processes and central_processes:
            raise production_sample_profile_error(
                "SUMW2-PROFILE-E003: sumw2 provenance contains both private and central variants."
            )
        selected_variant = "private" if private_processes else "central"
        expected_variants[f"{record['signal_group']}:{record['year']}"] = {
            "signal_group": record["signal_group"],
            "year": record["year"],
            "selected_variant": selected_variant,
            "processes": list(record[f"{selected_variant}_processes"]),
        }
    if contract["active_signal_variants"] != expected_variants:
        raise production_sample_profile_error(
            "production_sample_contract active signal variants contradict sumw2 provenance."
        )
    if parsed_policy.signal_sample_profile in {"private", "central"}:
        disallowed = [
            key
            for key, record in expected_variants.items()
            if record["selected_variant"] != parsed_policy.signal_sample_profile
        ]
        if disallowed:
            raise production_sample_profile_error(
                "production_sample_contract contains variants incompatible with the selected profile: "
                + ", ".join(disallowed)
            )


def require_data_driven_profile_certification(sidecar: Mapping[str, Any]) -> None:
    """Authorize new transformation only from current certified profile metadata."""

    provenance = sidecar.get("sumw2_storage_provenance")
    if not isinstance(provenance, Mapping):
        raise production_sample_profile_error(
            "Data-driven transformation requires sumw2 provenance."
        )
    if provenance.get("schema_version") != SUMW2_PROVENANCE_SCHEMA_VERSION:
        raise production_sample_profile_error(
            "This processor sidecar predates certified production sample profiles and "
            "may be reopened read-only, but it cannot authorize run_data_driven. "
            "Regenerate the processor output and sidecar with the current run_analysis preflight."
        )
    contract = sidecar.get("production_sample_contract")
    if contract is None:
        raise production_sample_profile_error(
            "Current data-driven transformation requires production_sample_contract "
            "certification. Regenerate the processor output with run_analysis."
        )
    validate_production_sample_contract(contract, provenance)
