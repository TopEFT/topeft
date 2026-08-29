from __future__ import annotations

import copy

import pytest

from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.data_driven_products import (
    certify_data_driven_preflight,
    data_driven_product_error,
    resolve_data_driven_products,
    validate_serialized_data_driven_contract,
)
from topeft.modules.production_sample_profile import (
    build_active_sample_universe,
    certify_production_sample_contract,
    production_sample_profile_error,
    validate_active_sample_profile,
)
from topeft.modules.sumw2_policy import (
    resolve_sumw2_storage_mode,
    resolve_sumw2_storage_policy,
)


def _sample(process, *, is_data=False, wc_names=()):
    return {
        "histAxisName": process,
        "isData": is_data,
        "WCnames": list(wc_names),
    }


def _samples(signal_process, *, signal_wc_names=()):
    return {
        "data_dataset": _sample("dataUL18", is_data=True),
        "ordinary_prompt_dataset": _sample("TTTo2L2Nu_centralUL18"),
        "signal_dataset": _sample(
            signal_process,
            wc_names=signal_wc_names,
        ),
    }


def _storage(mode, process_names, *, families=("met",)):
    return {
        "mode": mode,
        "rules": [
            {
                "process_names": list(process_names),
                "variables": list(families),
            }
        ],
    }


def _explicit_products(*, prompt_processes, nonprompt=True, flips=False):
    return {
        "nonprompt": {
            "enabled": nonprompt,
            **(
                {
                    "source_contributors": {
                        "data": {"process_names": ["dataUL18"]},
                        "prompt_mc": {
                            "process_names": list(prompt_processes)
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
                        "data": {"process_names": ["dataUL18"]}
                    }
                }
                if flips
                else {}
            ),
        },
    }


def _resolve_case(
    mode,
    signal_process,
    *,
    signal_wc_names=(),
    product_block=None,
    product_present=True,
    legacy_do_np=False,
    selected_processes=None,
    families=("met",),
):
    samples = _samples(signal_process, signal_wc_names=signal_wc_names)
    storage = _storage(
        mode,
        selected_processes
        or ("dataUL18", "TTTo2L2Nu_centralUL18", signal_process),
        families=families,
    )
    mode_resolution = resolve_sumw2_storage_mode(
        storage,
        sumw2_storage_present=True,
    )
    universe = build_active_sample_universe(
        samples,
        wrapper_identity="required-signal-test",
    )
    validate_active_sample_profile(universe, mode_resolution)
    products = resolve_data_driven_products(
        product_block,
        data_driven_products_present=product_present,
        legacy_do_np=legacy_do_np,
        samples=samples,
        runtime_families=families,
        metadata_path=f"{mode}.yml",
    )
    policy = resolve_sumw2_storage_policy(
        storage,
        samples=samples,
        runtime_families=families,
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
        mode_resolution=mode_resolution,
    )
    return universe, products, policy


@pytest.mark.parametrize(
    "mode,signal_process,signal_wc_names",
    [
        ("production", "tllq_privateUL18", ("ctW",)),
        ("production_central", "tZq_centralUL18", ()),
    ],
)
def test_implicit_metadata_includes_canonical_prompt_process(
    mode, signal_process, signal_wc_names
):
    universe, products, policy = _resolve_case(
        mode,
        signal_process,
        signal_wc_names=signal_wc_names,
        product_block=None,
        product_present=False,
        legacy_do_np=True,
    )
    assert signal_process in products.product("nonprompt").contributors_for(
        "prompt_mc"
    )
    certify_production_sample_contract(universe, policy, products)
    _requested, contract = certify_data_driven_preflight(products, policy)
    assert signal_process in contract["resolved_prompt_process_set"]
    assert "required_prompt_signal_processes" not in contract


@pytest.mark.parametrize(
    "mode,signal_process,signal_wc_names",
    [
        ("production", "tllq_privateUL18", ("ctW",)),
        ("production_central", "tZq_centralUL18", ()),
    ],
)
def test_explicit_signal_omission_fails_at_canonical_policy_gate(
    mode, signal_process, signal_wc_names
):
    with pytest.raises(data_driven_product_error) as captured:
        _resolve_case(
            mode,
            signal_process,
            signal_wc_names=signal_wc_names,
            product_block=_explicit_products(
                prompt_processes=("TTTo2L2Nu_centralUL18",)
            ),
        )
    message = str(captured.value)
    for expected in (
        "NONPROMPT-POLICY-E009",
        "cannot override the canonical resolved prompt process set",
        "missing=",
        signal_process,
    ):
        assert expected in message


@pytest.mark.parametrize(
    "mode,wrong_signal,error_id",
    [
        ("production", "tZq_centralUL18", "SUMW2-PROFILE-E001"),
        ("production_central", "tllq_privateUL18", "SUMW2-PROFILE-E002"),
    ],
)
def test_active_counterpart_still_uses_profile_error(mode, wrong_signal, error_id):
    storage = _storage(mode, (wrong_signal,))
    mode_resolution = resolve_sumw2_storage_mode(
        storage,
        sumw2_storage_present=True,
    )
    universe = build_active_sample_universe(
        {"signal": _sample(wrong_signal)},
    )
    with pytest.raises(production_sample_profile_error, match=error_id):
        validate_active_sample_profile(universe, mode_resolution)


@pytest.mark.parametrize(
    "nonprompt,flips",
    [(False, False), (False, True)],
)
def test_nonprompt_disabled_and_flips_only_require_no_signal(nonprompt, flips):
    signal_process = "tllq_privateUL18"
    block = _explicit_products(
        prompt_processes=(),
        nonprompt=nonprompt,
        flips=flips,
    )
    universe, products, policy = _resolve_case(
        "production",
        signal_process,
        signal_wc_names=("ctW",),
        product_block=block,
        selected_processes=("dataUL18",),
    )
    certify_production_sample_contract(universe, policy, products)
    _requested, contract = certify_data_driven_preflight(products, policy)
    assert "required_prompt_signal_processes" not in contract


def test_nonprompt_plus_flips_requirement_comes_only_from_nonprompt():
    signal_process = "tllq_privateUL18"
    universe, products, policy = _resolve_case(
        "production",
        signal_process,
        signal_wc_names=("ctW",),
        product_block=_explicit_products(
            prompt_processes=(
                "TTTo2L2Nu_centralUL18",
                signal_process,
            ),
            nonprompt=True,
            flips=True,
        ),
    )
    certify_production_sample_contract(universe, policy, products)
    _requested, contract = certify_data_driven_preflight(products, policy)
    assert signal_process in contract["resolved_prompt_process_set"]
    assert "required_prompt_signal_processes" not in contract
    flips_sources = contract["products"]["flips"]["generated_outputs"][
        "flipsUL18"
    ]["source_contributors"]
    assert flips_sources == {"data": ["dataUL18"]}


def test_required_signal_needs_every_dataset_process_family_target():
    signal_process = "tllq_privateUL18"
    samples = _samples(signal_process, signal_wc_names=("ctW",))
    mode_resolution = resolve_sumw2_storage_mode(
        _storage("production", ("dataUL18",), families=("njets", "met")),
        sumw2_storage_present=True,
    )
    universe = build_active_sample_universe(samples)
    products = resolve_data_driven_products(
        _explicit_products(
            prompt_processes=("TTTo2L2Nu_centralUL18", signal_process)
        ),
        data_driven_products_present=True,
        legacy_do_np=False,
        samples=samples,
        runtime_families=("njets", "met"),
        metadata_path="targets.yml",
    )
    incomplete_storage = {
        "mode": "production",
        "rules": [
            {
                "process_names": [
                    "dataUL18",
                    "TTTo2L2Nu_centralUL18",
                ]
            },
            {
                "process_names": [signal_process],
                "variables": ["njets"],
            },
        ],
    }
    policy = resolve_sumw2_storage_policy(
        incomplete_storage,
        samples=samples,
        runtime_families=("njets", "met"),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    with pytest.raises(
        production_sample_profile_error,
        match=r"SUMW2-PROFILE-E005.*signal_dataset/tllq_privateUL18/met",
    ):
        certify_production_sample_contract(universe, policy, products)

    complete_policy = resolve_sumw2_storage_policy(
        _storage(
            "production",
            ("dataUL18", "TTTo2L2Nu_centralUL18", signal_process),
            families=("njets", "met"),
        ),
        samples=samples,
        runtime_families=("njets", "met"),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    certify_production_sample_contract(universe, complete_policy, products)


def test_historical_required_signal_key_is_ignored_but_contributor_tampering_fails():
    signal_process = "tllq_privateUL18"
    universe, products, policy = _resolve_case(
        "production",
        signal_process,
        signal_wc_names=("ctW",),
        product_block=_explicit_products(
            prompt_processes=(
                "TTTo2L2Nu_centralUL18",
                signal_process,
            )
        ),
    )
    certify_production_sample_contract(universe, policy, products)
    requested, contract = certify_data_driven_preflight(products, policy)

    historical = copy.deepcopy(contract)
    historical["required_prompt_signal_processes"] = ["staleUL18"]
    _normalized_requested, normalized_contract = (
        validate_serialized_data_driven_contract(
            requested,
            historical,
            policy=policy,
        )
    )
    assert normalized_contract == contract

    missing_contributor = copy.deepcopy(contract)
    output = missing_contributor["products"]["nonprompt"]["generated_outputs"][
        "nonpromptUL18"
    ]
    output["source_contributors"]["prompt_mc"].remove(signal_process)
    output["required_source_sumw2_processes"].remove(signal_process)
    with pytest.raises(
        data_driven_product_error,
        match="Nominal prompt contributors.*certified resolved prompt process set differ",
    ):
        validate_serialized_data_driven_contract(
            requested,
            missing_contributor,
            policy=policy,
        )
