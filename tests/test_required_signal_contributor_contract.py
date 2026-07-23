from __future__ import annotations

import copy
import json
from pathlib import Path

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
    derive_required_prompt_signal_processes,
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
    required_candidates = derive_required_prompt_signal_processes(
        universe.processes,
        signal_sample_profile=mode_resolution.signal_sample_profile,
        nonprompt_enabled=True,
    )
    products = resolve_data_driven_products(
        product_block,
        data_driven_products_present=product_present,
        legacy_do_np=legacy_do_np,
        samples=samples,
        runtime_families=families,
        metadata_path=f"{mode}.yml",
        required_prompt_signal_processes=required_candidates,
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
def test_implicit_metadata_includes_active_profile_signal(
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
    assert contract["required_prompt_signal_processes"] == [signal_process]


@pytest.mark.parametrize(
    "mode,signal_process,signal_wc_names",
    [
        ("production", "tllq_privateUL18", ("ctW",)),
        ("production_central", "tZq_centralUL18", ()),
    ],
)
def test_explicit_signal_omission_reports_e007_before_processing(
    mode, signal_process, signal_wc_names
):
    universe, products, policy = _resolve_case(
        mode,
        signal_process,
        signal_wc_names=signal_wc_names,
        product_block=_explicit_products(
            prompt_processes=("TTTo2L2Nu_centralUL18",)
        ),
    )
    with pytest.raises(production_sample_profile_error) as captured:
        certify_production_sample_contract(universe, policy, products)
    message = str(captured.value)
    for expected in (
        "SUMW2-PROFILE-E007",
        f"resolved_mode='{mode}'",
        "expected_signal_profile=",
        "sr_cfg_identities=",
        "metadata_source='explicit'",
        "active_required_prompt_signals=",
        "resolved_contributor_processes=",
        signal_process,
        "Recommended correction:",
    ):
        assert expected in message


def test_absent_signal_creates_no_profile_requirement():
    processes = ("dataUL18", "TTTo2L2Nu_centralUL18")
    assert derive_required_prompt_signal_processes(
        processes,
        signal_sample_profile="private",
        nonprompt_enabled=True,
    ) == ()
    assert derive_required_prompt_signal_processes(
        processes,
        signal_sample_profile="central",
        nonprompt_enabled=True,
    ) == ()


def _cfg_processes(cfg_name):
    cfg_path = Path("input_samples/cfgs") / cfg_name
    processes = []
    for raw_line in cfg_path.read_text(encoding="utf-8").splitlines():
        token = raw_line.split("#", 1)[0].strip()
        if not token.endswith(".json"):
            continue
        json_path = cfg_path.parent / token
        processes.append(json.loads(json_path.read_text())["histAxisName"])
    return tuple(processes)


def test_actual_2022_cfgs_derive_only_evidenced_required_signals():
    private_processes = _cfg_processes("NDSkim_2022_mc_signal_samples_sr.cfg")
    central_processes = _cfg_processes("NDSkim_2022_central_signal_samples.cfg")
    assert derive_required_prompt_signal_processes(
        private_processes,
        signal_sample_profile="private",
        nonprompt_enabled=True,
    ) == (
        "tHq_private2022",
        "tllq_private2022",
        "ttlnu_private2022",
        "tttt_private2022",
    )
    assert derive_required_prompt_signal_processes(
        central_processes,
        signal_sample_profile="central",
        nonprompt_enabled=True,
    ) == (
        "TTTT_central2022",
        "tZq_central2022",
        "ttLNu_cental2022",
    )


def test_evidenced_unpaired_prompt_signal_is_required_in_both_profiles():
    for profile in ("private", "central"):
        assert derive_required_prompt_signal_processes(
            ("tHq_private2022",),
            signal_sample_profile=profile,
            nonprompt_enabled=True,
        ) == ("tHq_private2022",)


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
    assert contract["required_prompt_signal_processes"] == []


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
    assert contract["required_prompt_signal_processes"] == [signal_process]
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
    required = derive_required_prompt_signal_processes(
        universe.processes,
        signal_sample_profile="private",
        nonprompt_enabled=True,
    )
    products = resolve_data_driven_products(
        _explicit_products(
            prompt_processes=("TTTo2L2Nu_centralUL18", signal_process)
        ),
        data_driven_products_present=True,
        legacy_do_np=False,
        samples=samples,
        runtime_families=("njets", "met"),
        metadata_path="targets.yml",
        required_prompt_signal_processes=required,
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


def test_required_signal_provenance_rejects_list_and_contributor_tampering():
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

    tampered_list = copy.deepcopy(contract)
    tampered_list["required_prompt_signal_processes"] = []
    with pytest.raises(data_driven_product_error, match="contradict"):
        validate_serialized_data_driven_contract(
            requested,
            tampered_list,
            policy=policy,
        )

    missing_contributor = copy.deepcopy(contract)
    output = missing_contributor["products"]["nonprompt"]["generated_outputs"][
        "nonpromptUL18"
    ]
    output["source_contributors"]["prompt_mc"].remove(signal_process)
    output["required_source_sumw2_processes"].remove(signal_process)
    with pytest.raises(data_driven_product_error, match="omit.*required"):
        validate_serialized_data_driven_contract(
            requested,
            missing_contributor,
            policy=policy,
        )
