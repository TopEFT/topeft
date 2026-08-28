from __future__ import annotations

import copy

import pytest

from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.data_driven_products import (
    CANONICAL_DATA_DRIVEN_YEARS,
    certify_data_driven_preflight,
    data_driven_product_error,
    parse_process_name,
    resolve_data_driven_products,
    validate_serialized_data_driven_contract,
)
from topeft.modules.sumw2_policy import resolve_sumw2_storage_policy


@pytest.fixture
def samples():
    return {
        "data_a": {
            "histAxisName": "dataUL18",
            "isData": True,
            "WCnames": [],
        },
        "prompt_a": {
            "histAxisName": "TTTo2L2Nu_centralUL18",
            "isData": False,
            "WCnames": [],
        },
        "unselected_a": {
            "histAxisName": "WWTo2L2Nu_centralUL18",
            "isData": False,
            "WCnames": [],
        },
        "eft_a": {
            "histAxisName": "ttHJet_privateUL18",
            "isData": False,
            "WCnames": ["ctG"],
        },
    }


def _explicit_block(*, nonprompt=True, flips=True):
    return {
        "nonprompt": {
            "enabled": nonprompt,
            **(
                {
                    "source_contributors": {
                        "data": {"process_names": ["dataUL18"]},
                        "prompt_mc": {
                            "process_names": [
                                "TTTo2L2Nu_centralUL18",
                                "ttHJet_privateUL18",
                            ]
                        },
                    }
                }
                if nonprompt
                else {}
            ),
        },
        "flips": {
            "enabled": flips,
            **(
                {
                    "source_contributors": {
                        "data": {"process_prefixes": ["data"]},
                    }
                }
                if flips
                else {}
            ),
        },
    }


def _resolve_products(
    block,
    samples,
    *,
    present=True,
    legacy_do_np=False,
):
    return resolve_data_driven_products(
        block,
        data_driven_products_present=present,
        legacy_do_np=legacy_do_np,
        samples=samples,
        runtime_families=("njets", "met"),
        metadata_path="run_options.yml",
    )


def _resolve_policy(block, samples, *, implicit_requirements=()):
    return resolve_sumw2_storage_policy(
        block,
        samples=samples,
        runtime_families=("njets", "met"),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=block is not None,
        implicit_production_requirements=implicit_requirements,
    )


def test_explicit_products_resolve_editable_exact_and_prefix_contributors(samples):
    resolved = _resolve_products(_explicit_block(), samples)
    assert resolved.source == "explicit"
    assert resolved.enabled_products() == ("nonprompt", "flips")
    assert resolved.product("nonprompt").contributors_for("data") == ("dataUL18",)
    assert resolved.product("nonprompt").contributors_for("prompt_mc") == (
        "TTTo2L2Nu_centralUL18",
        "ttHJet_privateUL18",
    )
    assert resolved.product("flips").output_processes == ("flipsUL18",)
    assert resolved.product("nonprompt").output_processes == ("nonpromptUL18",)


def test_absent_block_derives_exact_legacy_products_and_warns(samples):
    with pytest.warns(UserWarning, match="implicit sibling data_driven_products|data_driven_products is absent"):
        resolved = _resolve_products(
            None,
            samples,
            present=False,
            legacy_do_np=True,
        )
    assert resolved.source == "implicit_legacy_data_driven_default"
    assert resolved.enabled_products() == ("nonprompt", "flips")
    assert resolved.product("nonprompt").contributors_for("prompt_mc") == (
        "TTTo2L2Nu_centralUL18",
        "ttHJet_privateUL18",
    )
    assert "ttHJet_privateUL18" in resolved.product(
        "nonprompt"
    ).contributors_for("prompt_mc")


def test_absent_block_without_legacy_do_np_disables_both_products(samples):
    with pytest.warns(UserWarning, match=r"enabled_products=\[\]"):
        resolved = _resolve_products(
            None,
            samples,
            present=False,
            legacy_do_np=False,
        )
    assert resolved.enabled_products() == ()


@pytest.mark.parametrize(
    "mutator,error_match",
    [
        (
            lambda block: block["nonprompt"].__setitem__("variables", ["njets"]),
            "unknown.*variables",
        ),
        (
            lambda block: block.__setitem__("unknown", {"enabled": False}),
            "unknown data_driven_products",
        ),
        (
            lambda block: block["nonprompt"]["source_contributors"].__setitem__(
                "signal", {"process_names": ["WWTo2L2Nu_centralUL18"]}
            ),
            r"unknown=\['signal'\]",
        ),
        (
            lambda block: block["flips"]["source_contributors"]["data"].__setitem__(
                "process_names", ["missingUL18"]
            ),
            "matched nothing",
        ),
    ],
)
def test_unknown_variable_product_role_and_unmatched_selectors_fail(
    samples, mutator, error_match
):
    block = _explicit_block()
    mutator(block)
    with pytest.raises(data_driven_product_error, match=error_match):
        _resolve_products(block, samples)


def test_overlapping_roles_and_ambiguous_duplicate_selector_resolution_fail(samples):
    overlapping = _explicit_block()
    overlapping["nonprompt"]["source_contributors"]["prompt_mc"] = {
        "process_names": ["dataUL18"]
    }
    with pytest.raises(data_driven_product_error, match="must be non-data MC"):
        _resolve_products(overlapping, samples)

    ambiguous = _explicit_block()
    ambiguous["flips"]["source_contributors"]["data"] = {
        "process_names": ["dataUL18"],
        "process_prefixes": ["data"],
    }
    with pytest.raises(data_driven_product_error, match="ambiguous duplicate"):
        _resolve_products(ambiguous, samples)


def test_both_disabled_is_valid_and_requires_no_targets(samples):
    resolved = _resolve_products(_explicit_block(nonprompt=False, flips=False), samples)
    assert resolved.enabled_products() == ()
    assert resolved.required_targets() == ()
    disabled = _resolve_policy({"mode": "disabled"}, samples)
    certify_data_driven_preflight(resolved, disabled)


def test_implicit_production_selects_only_requested_source_targets(samples):
    block = _explicit_block()
    block["nonprompt"]["source_contributors"]["prompt_mc"] = {
        "process_names": [
            "TTTo2L2Nu_centralUL18",
            "ttHJet_privateUL18",
        ]
    }
    products = _resolve_products(block, samples)
    policy = resolve_sumw2_storage_policy(
        None,
        samples=samples,
        runtime_families=("njets", "met"),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=False,
        implicit_production_requirements=products.required_targets(),
    )
    requested, contract = certify_data_driven_preflight(products, policy)
    assert policy.requested_mode == "production"
    assert policy.source == "implicit_production_default"
    assert set(policy.selected_processes("njets")) == {
        "dataUL18",
        "TTTo2L2Nu_centralUL18",
        "ttHJet_privateUL18",
    }
    assert "WWTo2L2Nu_centralUL18" not in policy.selected_processes("njets")
    assert requested["products"]["nonprompt"]["enabled"] is True
    assert set(contract) == {
        "contract_version",
        "nonprompt_policy",
        "resolved_prompt_process_set",
        "policy_migration",
        "products",
    }
    assert contract["contract_version"] == 4
    assert "required_prompt_signal_processes" not in contract
    assert "eft_prompt_processes" not in contract["nonprompt_policy"]
    assert contract["products"]["flips"]["generated_outputs"]["flipsUL18"][
        "required_source_sumw2_processes"
    ] == ["dataUL18"]


def test_full_diagnostics_and_complete_full_custom_pass_all_families(samples):
    products = _resolve_products(_explicit_block(), samples)
    full_diagnostics = _resolve_policy({"mode": "full_diagnostics"}, samples)
    certify_data_driven_preflight(products, full_diagnostics)

    complete_custom = _resolve_policy(
        {
            "mode": "full_custom",
            "rules": [
                {
                    "process_names": [
                        "dataUL18",
                        "TTTo2L2Nu_centralUL18",
                        "ttHJet_privateUL18",
                    ]
                }
            ],
        },
        samples,
    )
    certify_data_driven_preflight(products, complete_custom)


def test_explicit_production_and_taufitter_require_complete_product_sources(samples):
    block = _explicit_block()
    block["nonprompt"]["source_contributors"]["prompt_mc"] = {
        "process_names": [
            "TTTo2L2Nu_centralUL18",
            "ttHJet_privateUL18",
        ]
    }
    products = _resolve_products(block, samples)
    source_rule = {
        "process_names": [
            "dataUL18",
            "TTTo2L2Nu_centralUL18",
            "ttHJet_privateUL18",
        ],
    }
    explicit_production = _resolve_policy(
        {"mode": "production", "rules": [source_rule]},
        samples,
    )
    certify_data_driven_preflight(products, explicit_production)

    taufitter = resolve_sumw2_storage_policy(
        {"mode": "taufitter", "rules": [source_rule]},
        samples=samples,
        runtime_families=("njets", "met"),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        analysis_mode="taufitter",
        sumw2_storage_present=True,
    )
    certify_data_driven_preflight(products, taufitter)


def test_incomplete_full_custom_and_disabled_requested_products_fail_actionably(samples):
    products = _resolve_products(_explicit_block(), samples)
    incomplete = _resolve_policy(
        {
            "mode": "full_custom",
            "rules": [
                {
                    "process_names": ["dataUL18"],
                    "variables": ["njets"],
                }
            ],
        },
        samples,
    )
    with pytest.raises(
        data_driven_product_error,
        match="metadata_path=.*resolved_sumw2_mode=.*missing_contributors=.*Correct one of",
    ):
        certify_data_driven_preflight(products, incomplete)

    disabled = _resolve_policy({"mode": "disabled"}, samples)
    with pytest.raises(data_driven_product_error, match="resolved_sumw2_mode='disabled'"):
        certify_data_driven_preflight(products, disabled)


def test_full_custom_missing_one_applicable_family_fails(samples):
    products = _resolve_products(_explicit_block(), samples)
    subset = _resolve_policy(
        {
            "mode": "full_custom",
            "rules": [
                {
                    "process_names": [
                        "dataUL18",
                        "TTTo2L2Nu_centralUL18",
                    ],
                    "variables": ["njets"],
                }
            ],
        },
        samples,
    )
    with pytest.raises(
        data_driven_product_error,
        match="family 'met'.*missing_contributors",
    ):
        certify_data_driven_preflight(products, subset)


def test_serialized_contract_validation_rejects_tampering(samples):
    products = _resolve_products(_explicit_block(), samples)
    policy = _resolve_policy({"mode": "full_diagnostics"}, samples)
    requested, contract = certify_data_driven_preflight(products, policy)
    assert validate_serialized_data_driven_contract(
        requested,
        contract,
        policy=policy,
    ) == (requested, contract)

    historical = copy.deepcopy(contract)
    historical["required_prompt_signal_processes"] = ["stale_processUL18"]
    historical["nonprompt_policy"]["eft_prompt_processes"] = [
        "stale_processUL18"
    ]
    historical["nonprompt_policy"]["eft_sm_point"] = True
    historical["nonprompt_policy"]["resolutions"][0]["eft_sm_point"] = True
    normalized_requested, normalized_contract = (
        validate_serialized_data_driven_contract(
            requested,
            historical,
            policy=policy,
        )
    )
    assert normalized_requested == requested
    assert normalized_contract == contract

    tampered = copy.deepcopy(contract)
    tampered["products"]["nonprompt"]["generated_outputs"]["nonpromptUL18"][
        "required_source_sumw2_processes"
    ].remove("TTTo2L2Nu_centralUL18")
    with pytest.raises(data_driven_product_error, match="disagree with contributor roles"):
        validate_serialized_data_driven_contract(requested, tampered, policy=policy)

    family_tampered = copy.deepcopy(contract)
    family_tampered["families"] = ["njets"]
    with pytest.raises(data_driven_product_error, match="Invalid resolved.*fields"):
        validate_serialized_data_driven_contract(
            requested, family_tampered, policy=policy
        )


@pytest.mark.parametrize(
    "prompt_processes",
    [
        ["TTTo2L2Nu_centralUL17"],
        ["TTTo2L2Nu_centralUL18", "TTTo2L2Nu_centralUL17"],
    ],
)
def test_orphan_prompt_years_fail_during_resolution(prompt_processes):
    local_samples = {
        "data_18": {
            "histAxisName": "dataUL18",
            "isData": True,
            "WCnames": [],
        },
        **{
            f"prompt_{index}": {
                "histAxisName": process,
                "isData": False,
                "WCnames": [],
            }
            for index, process in enumerate(prompt_processes)
        },
    }
    block = {
        "nonprompt": {
            "enabled": True,
            "source_contributors": {
                "data": {"process_names": ["dataUL18"]},
                "prompt_mc": {"process_names": prompt_processes},
            },
        },
        "flips": {"enabled": False},
    }
    with pytest.raises(
        data_driven_product_error,
        match=(
            r"metadata_path='run_options.yml'.*metadata_source='explicit'.*"
            r"product='nonprompt'.*orphan_years=\['UL17'\].*"
            r"orphan_prompt_processes=.*configured_data_processes_and_years=.*"
            r"configured_prompt_processes_and_years=.*Recommended correction"
        ),
    ):
        _resolve_products(block, local_samples)


def test_implicit_legacy_orphan_prompt_year_fails_during_resolution():
    local_samples = {
        "data_18": {
            "histAxisName": "dataUL18",
            "isData": True,
            "WCnames": [],
        },
        "prompt_17": {
            "histAxisName": "TTTo2L2Nu_centralUL17",
            "isData": False,
            "WCnames": [],
        },
    }
    with pytest.warns(UserWarning, match="data_driven_products is absent"):
        with pytest.raises(
            data_driven_product_error,
            match=r"metadata_source='implicit_legacy_data_driven_default'.*orphan_years=\['UL17'\]",
        ):
            _resolve_products(
                None,
                local_samples,
                present=False,
                legacy_do_np=True,
            )


def test_data_only_year_and_complete_years_resolve_exact_output_maps():
    local_samples = {
        "data_17_a": {"histAxisName": "dataUL17", "isData": True, "WCnames": []},
        "data_17_b": {"histAxisName": "dataUL17", "isData": True, "WCnames": []},
        "data_18": {"histAxisName": "dataUL18", "isData": True, "WCnames": []},
        "prompt_17_a": {
            "histAxisName": "TTTo2L2Nu_centralUL17",
            "isData": False,
            "WCnames": [],
        },
        "prompt_17_b": {
            "histAxisName": "WZTo3LNu_centralUL17",
            "isData": False,
            "WCnames": [],
        },
    }
    block = {
        "nonprompt": {
            "enabled": True,
            "source_contributors": {
                "data": {"process_prefixes": ["data"]},
                "prompt_mc": {
                    "process_names": [
                        "TTTo2L2Nu_centralUL17",
                        "WZTo3LNu_centralUL17",
                    ]
                },
            },
        },
        "flips": {
            "enabled": True,
            "source_contributors": {
                "data": {"process_names": ["dataUL18"]},
            },
        },
    }
    resolved = _resolve_products(block, local_samples)
    nonprompt_outputs = dict(resolved.product("nonprompt").generated_outputs)
    assert tuple(nonprompt_outputs) == ("nonpromptUL17", "nonpromptUL18")
    assert nonprompt_outputs["nonpromptUL17"].contributors_for("data") == (
        "dataUL17",
    )
    assert nonprompt_outputs["nonpromptUL17"].contributors_for("prompt_mc") == (
        "TTTo2L2Nu_centralUL17",
        "WZTo3LNu_centralUL17",
    )
    assert nonprompt_outputs["nonpromptUL18"].contributors_for("prompt_mc") == ()
    assert resolved.product("flips").output_processes == ("flipsUL18",)

    complete_block = copy.deepcopy(block)
    complete_block["nonprompt"]["source_contributors"]["prompt_mc"][
        "process_names"
    ].append("TTTo2L2Nu_centralUL18")
    complete_samples = dict(local_samples)
    complete_samples["prompt_18"] = {
        "histAxisName": "TTTo2L2Nu_centralUL18",
        "isData": False,
        "WCnames": [],
    }
    complete = _resolve_products(complete_block, complete_samples)
    complete_outputs = dict(complete.product("nonprompt").generated_outputs)
    assert complete_outputs["nonpromptUL18"].contributors_for("prompt_mc") == (
        "TTTo2L2Nu_centralUL18",
    )


def test_all_canonical_years_and_overlapping_suffixes_group_exactly():
    local_samples = {}
    data_processes = []
    prompt_processes = []
    for index, year in enumerate(CANONICAL_DATA_DRIVEN_YEARS):
        data_process = f"data{year}"
        prompt_base = (
            "TTTo2L2Nu_central"
            if year.startswith("UL")
            else "TTto2L2Nu_central"
        )
        prompt_process = f"{prompt_base}{year}"
        data_processes.append(data_process)
        prompt_processes.append(prompt_process)
        local_samples[f"data_{index}"] = {
            "histAxisName": data_process,
            "isData": True,
            "WCnames": [],
        }
        local_samples[f"prompt_{index}"] = {
            "histAxisName": prompt_process,
            "isData": False,
            "WCnames": [],
        }
    block = {
        "nonprompt": {
            "enabled": True,
            "source_contributors": {
                "data": {"process_names": data_processes},
                "prompt_mc": {"process_names": prompt_processes},
            },
        },
        "flips": {
            "enabled": True,
            "source_contributors": {
                "data": {"process_prefixes": ["data"]},
            },
        },
    }
    resolved = _resolve_products(block, local_samples)
    assert resolved.product("nonprompt").output_processes == tuple(
        f"nonprompt{year}" for year in CANONICAL_DATA_DRIVEN_YEARS
    )
    assert resolved.product("flips").output_processes == tuple(
        f"flips{year}" for year in CANONICAL_DATA_DRIVEN_YEARS
    )
    assert parse_process_name("dataUL16APV")[1] == "UL16APV"
    assert parse_process_name("data2022EE")[1] == "2022EE"
    assert parse_process_name("data2023BPix")[1] == "2023BPix"


@pytest.mark.parametrize(
    "process",
    [
        "data16APV",
        "data16",
        "data17",
        "data18",
        "data22",
        "data22EE",
        "data23",
        "data23BPix",
    ],
)
def test_shortened_or_unknown_year_aliases_are_rejected(process):
    with pytest.raises(data_driven_product_error, match="year suffix"):
        parse_process_name(process)


def test_serialized_contract_rejects_cross_year_and_orphan_output_tampering(samples):
    products = _resolve_products(_explicit_block(), samples)
    policy = _resolve_policy({"mode": "full_diagnostics"}, samples)
    requested, contract = certify_data_driven_preflight(products, policy)

    wrong_year = copy.deepcopy(contract)
    wrong_year["products"]["nonprompt"]["generated_outputs"]["nonpromptUL18"][
        "year"
    ] = "UL17"
    with pytest.raises(data_driven_product_error, match="label/year mismatch"):
        validate_serialized_data_driven_contract(
            requested, wrong_year, policy=policy
        )

    moved_contributor = copy.deepcopy(contract)
    moved_contributor["products"]["nonprompt"]["generated_outputs"][
        "nonpromptUL18"
    ]["source_contributors"]["prompt_mc"] = ["TTTo2L2Nu_centralUL17"]
    moved_contributor["products"]["nonprompt"]["generated_outputs"][
        "nonpromptUL18"
    ]["required_source_sumw2_processes"] = [
        "TTTo2L2Nu_centralUL17",
        "dataUL18",
    ]
    with pytest.raises(data_driven_product_error, match="has year 'UL17'.*year 'UL18'|certified resolved prompt"):
        validate_serialized_data_driven_contract(
            requested, moved_contributor, policy=policy
        )

    orphan_output = copy.deepcopy(contract)
    orphan_output["products"]["nonprompt"]["generated_outputs"][
        "nonpromptUL17"
    ] = {
        "year": "UL17",
        "source_contributors": {
            "data": [],
            "prompt_mc": ["TTTo2L2Nu_centralUL17"],
        },
        "required_source_sumw2_processes": ["TTTo2L2Nu_centralUL17"],
    }
    orphan_output["products"]["nonprompt"]["output_processes"] = [
        "nonpromptUL17",
        "nonpromptUL18",
    ]
    reordered = orphan_output["products"]["nonprompt"]["generated_outputs"]
    orphan_output["products"]["nonprompt"]["generated_outputs"] = {
        "nonpromptUL17": reordered["nonpromptUL17"],
        "nonpromptUL18": reordered["nonpromptUL18"],
    }
    with pytest.raises(data_driven_product_error, match="at least one same-year data"):
        validate_serialized_data_driven_contract(
            requested, orphan_output, policy=policy
        )
