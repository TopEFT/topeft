from __future__ import annotations

import copy

import pytest

from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.data_driven_products import (
    PRECANONICAL_RESOLVED_DATA_DRIVEN_CONTRACT_VERSION,
    certify_data_driven_preflight,
    data_driven_product_error,
    reresolve_nonprompt_policy_from_sidecar,
    resolve_requested_product_input,
    resolve_data_driven_products,
)
from topeft.modules.nonprompt_policy import (
    DATA_OR_NON_MC,
    DEFAULT_NONPROMPT_ALIAS_DEFINITIONS,
    certify_active_nonprompt_policy,
)
from topeft.modules.sumw2_policy import resolve_sumw2_storage_policy

RUN2_YEARS = ("UL16APV", "UL16", "UL17", "UL18")
RUN2_ACTIVE_BASES = (
    "TTGJets_central", "TTGamma_central", "TTTo2L2Nu_central",
    "TTToSemiLeptonic_central", "TTZToLL_M1to10_central", "TWZToLL_central",
    "WWTo2L2Nu_central", "WWW_4F_central", "WWZ_4F_central",
    "WZTo3LNu_central", "WZZ_central", "ZZTo4L_central", "ZZZ_central",
    "tHq_private", "tllq_private", "ttHJet_private", "ttllJet_private",
    "ttlnuJet_private", "tttt_private",
)
RUN3_ACTIVE_BASES = tuple(
    definition.raw_process_base
    for definition in DEFAULT_NONPROMPT_ALIAS_DEFINITIONS
    if "run3" in definition.run_eras and definition.policy_role != DATA_OR_NON_MC
)
RUN3_NEW_PROMPT_BASES = (
    "TTto2L2Nu_central", "ttH_private", "ggToZZTo2e2mu_central",
    "ggToZZTo2e2nu_central", "ggToZZTo2e2tau_central",
    "ggToZZTo2mu2tau_central", "ggToZZTo4e_central",
    "ggToZZTo4mu_central", "ggToZZTo4tau_central",
)


def _samples(bases, years):
    samples = {}
    for year in years:
        samples[f"data_{year}"] = {
            "histAxisName": f"data{year}",
            "isData": True,
            "WCnames": [],
        }
        for index, base in enumerate(bases):
            samples[f"mc_{year}_{index}"] = {
                "histAxisName": f"{base}{year}",
                "isData": False,
                "WCnames": ["ctW"] if base.endswith("_private") else [],
            }
    return samples


def _resolved_contract(samples):
    certificate = certify_active_nonprompt_policy(
        samples,
        configuration_source="fresh_synthetic_source",
    )
    data_processes = sorted(
        sample["histAxisName"]
        for sample in samples.values()
        if sample["isData"]
    )
    block = {
        "nonprompt": {
            "enabled": True,
            "source_contributors": {
                "data": {"process_names": data_processes},
                "prompt_mc": {
                    "process_names": list(certificate.resolved_prompt_process_set)
                },
            },
        },
        "flips": {
            "enabled": True,
            "source_contributors": {
                "data": {"process_names": data_processes},
            },
        },
    }
    products = resolve_data_driven_products(
        block,
        data_driven_products_present=True,
        legacy_do_np=False,
        samples=samples,
        runtime_families=("njets",),
        metadata_path="fresh_synthetic_source",
        nonprompt_policy=certificate,
    )
    selected = sorted(set(data_processes) | set(certificate.resolved_prompt_process_set))
    policy = resolve_sumw2_storage_policy(
        {
            "mode": "full_custom",
            "rules": [
                {"process_names": selected, "variables": ["njets"]},
            ],
        },
        samples=samples,
        runtime_families=("njets",),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    requested, contract = certify_data_driven_preflight(products, policy)
    return certificate, policy, requested, contract


def _manifest(samples, sumw2_processes):
    scalar = sorted(
        sample["histAxisName"]
        for sample in samples.values()
        if not sample["WCnames"]
    )
    eft = sorted(
        sample["histAxisName"]
        for sample in samples.values()
        if sample["WCnames"]
    )
    return {
        "families": {
            "njets": {
                "scalar_nominal_processes": scalar,
                "eft_nominal_processes": eft,
                "sumw2_processes": sorted(sumw2_processes),
            }
        }
    }


def _precanonical_contract(contract, prompt_processes):
    stale = copy.deepcopy(contract)
    stale["contract_version"] = PRECANONICAL_RESOLVED_DATA_DRIVEN_CONTRACT_VERSION
    for field in (
        "nonprompt_policy",
        "resolved_prompt_process_set",
        "policy_migration",
    ):
        stale.pop(field)
    prompt_processes = set(prompt_processes)
    stale["required_prompt_signal_processes"] = sorted(prompt_processes)
    for output in stale["products"]["nonprompt"]["generated_outputs"].values():
        data = output["source_contributors"]["data"]
        prompt = sorted(
            set(output["source_contributors"]["prompt_mc"]) & prompt_processes
        )
        output["source_contributors"]["prompt_mc"] = prompt
        output["required_source_sumw2_processes"] = sorted(set(data) | set(prompt))
    return stale


def test_fresh_contract_uses_one_prompt_set_and_omits_representation_hints():
    samples = _samples(RUN3_ACTIVE_BASES, ("2022",))
    certificate, _policy, _requested, contract = _resolved_contract(samples)
    output = contract["products"]["nonprompt"]["generated_outputs"][
        "nonprompt2022"
    ]
    nominal_prompt = output["source_contributors"]["prompt_mc"]
    sumw2_prompt = sorted(
        set(output["required_source_sumw2_processes"])
        - set(output["source_contributors"]["data"])
    )
    assert nominal_prompt == contract["resolved_prompt_process_set"]
    assert sumw2_prompt == contract["resolved_prompt_process_set"]
    assert "ttH_private2022" in certificate.resolved_prompt_process_set
    assert "required_prompt_signal_processes" not in contract
    assert "eft_prompt_processes" not in contract["nonprompt_policy"]
    assert all(
        "eft_sm_point" not in resolution
        for resolution in contract["nonprompt_policy"]["resolutions"]
    )


def test_stale_run3_contract_is_reresolved_with_provenance_and_missing_sumw2():
    samples = _samples(RUN3_ACTIVE_BASES, ("2022",))
    certificate, _current_policy, requested, contract = _resolved_contract(samples)
    missing_new = {f"{base}2022" for base in RUN3_NEW_PROMPT_BASES}
    stale_prompt = set(certificate.resolved_prompt_process_set) - missing_new
    stale_contract = _precanonical_contract(contract, stale_prompt)

    data_processes = {
        sample["histAxisName"] for sample in samples.values() if sample["isData"]
    }
    stale_selected = sorted(data_processes | stale_prompt)
    stale_policy = resolve_sumw2_storage_policy(
        {
            "mode": "full_custom",
            "rules": [
                {"process_names": stale_selected, "variables": ["njets"]},
            ],
        },
        samples=samples,
        runtime_families=("njets",),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    sidecar = {
        "requested_data_driven_products": requested,
        "resolved_data_driven_contract": stale_contract,
        "sumw2_storage_provenance": stale_policy.to_provenance(),
        "sumw2_content_manifest": _manifest(samples, stale_selected),
    }
    migration = reresolve_nonprompt_policy_from_sidecar(sidecar)
    assert migration["status"] == "reresolved_changed"
    assert set(migration["added_prompt_processes"]) == missing_new
    assert migration["removed_prompt_processes"] == []
    assert migration["serialized_contract_provenance"] == stale_contract
    assert migration["resolved_prompt_process_set"] == list(
        certificate.resolved_prompt_process_set
    )
    assert set(migration["missing_process_resolved_sumw2"]["njets"]) == missing_new
    assert set(migration["missing_sumw2_policy_selection"]["njets"]) == missing_new
    assert migration["statistically_complete"] is False
    assert migration["effective_sidecar"] is None


def test_valid_run2_precanonical_policy_reresolves_without_membership_change():
    samples = _samples(RUN2_ACTIVE_BASES, RUN2_YEARS)
    certificate, policy, requested, contract = _resolved_contract(samples)
    stale_contract = _precanonical_contract(
        contract,
        certificate.resolved_prompt_process_set,
    )
    selected = set(certificate.resolved_prompt_process_set) | {
        f"data{year}" for year in RUN2_YEARS
    }
    sidecar = {
        "requested_data_driven_products": requested,
        "resolved_data_driven_contract": stale_contract,
        "sumw2_storage_provenance": policy.to_provenance(),
        "sumw2_content_manifest": _manifest(samples, selected),
    }
    migration = reresolve_nonprompt_policy_from_sidecar(sidecar)
    assert migration["status"] == "reresolved_unchanged"
    assert migration["added_prompt_processes"] == []
    assert migration["removed_prompt_processes"] == []
    assert len(migration["resolved_prompt_process_set"]) == 72
    assert migration["statistically_complete"] is True
    assert migration["effective_sidecar"] is not None


def test_incomplete_run3_source_is_strict_for_nonprompt_but_not_flips_or_reference():
    samples = _samples(RUN3_ACTIVE_BASES, ("2022",))
    certificate, _current_policy, requested, contract = _resolved_contract(samples)
    missing_new = {f"{base}2022" for base in RUN3_NEW_PROMPT_BASES}
    stale_prompt = set(certificate.resolved_prompt_process_set) - missing_new
    stale_contract = _precanonical_contract(contract, stale_prompt)
    data_processes = {
        sample["histAxisName"] for sample in samples.values() if sample["isData"]
    }
    stale_selected = sorted(data_processes | stale_prompt)
    stale_policy = resolve_sumw2_storage_policy(
        {"mode": "full_custom", "rules": [{"process_names": stale_selected, "variables": ["njets"]}]},
        samples=samples,
        runtime_families=("njets",),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    sidecar = {
        "requested_data_driven_products": requested,
        "resolved_data_driven_contract": stale_contract,
        "sumw2_storage_provenance": stale_policy.to_provenance(),
        "sumw2_content_manifest": _manifest(samples, stale_selected),
    }

    with pytest.raises(data_driven_product_error, match="missing process-resolved"):
        resolve_requested_product_input(sidecar, artifact_kind="nonprompt_output")

    reference = resolve_requested_product_input(
        sidecar,
        artifact_kind="nonprompt_nominal_reference_output",
    )
    reference_contract = reference["effective_sidecar"]["nominal_reference_contract"]
    assert reference_contract["statistically_complete"] is False
    assert reference_contract["card_ready"] is False
    assert set(reference_contract["missing_process_resolved_sumw2"]["njets"]) == missing_new
    assert reference["resolved_data_driven_contract"]["resolved_prompt_process_set"] == list(
        certificate.resolved_prompt_process_set
    )

    flips = resolve_requested_product_input(sidecar, artifact_kind="flips_output")
    assert flips["effective_sidecar"]["resolved_data_driven_contract"] == stale_contract


def test_complete_run2_source_remains_a_normal_nonprompt_input():
    samples = _samples(RUN2_ACTIVE_BASES, RUN2_YEARS)
    certificate, policy, requested, contract = _resolved_contract(samples)
    selected = set(certificate.resolved_prompt_process_set) | {
        f"data{year}" for year in RUN2_YEARS
    }
    sidecar = {
        "requested_data_driven_products": requested,
        "resolved_data_driven_contract": _precanonical_contract(
            contract,
            certificate.resolved_prompt_process_set,
        ),
        "sumw2_storage_provenance": policy.to_provenance(),
        "sumw2_content_manifest": _manifest(samples, selected),
    }
    normal = resolve_requested_product_input(sidecar, artifact_kind="nonprompt_output")
    assert normal["migration"]["statistically_complete"] is True
    assert "nominal_reference_contract" not in normal["effective_sidecar"]
