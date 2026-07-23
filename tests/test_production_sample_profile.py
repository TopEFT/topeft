from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.data_driven_products import resolve_data_driven_products
from topeft.modules.production_sample_profile import (
    VALIDATED_SIGNAL_VARIANT_GROUPS,
    build_active_sample_universe,
    certify_production_sample_contract,
    production_sample_profile_error,
    signal_variant_group,
    validate_active_sample_profile,
    validate_production_sample_contract,
    _group_active_variants,
)
from topeft.modules.sumw2_policy import (
    resolve_sumw2_storage_mode,
    resolve_sumw2_storage_policy,
)


def _sample(process, *, is_data=False, wc_names=None):
    return {
        "histAxisName": process,
        "isData": is_data,
        "WCnames": list(wc_names or []),
    }


def _mode(mode):
    return resolve_sumw2_storage_mode(
        {"mode": mode, "rules": [{"process_prefixes": ["unused"]}]}
        if mode in {"production", "production_central", "full_custom", "taufitter"}
        else {"mode": mode},
        sumw2_storage_present=True,
    )


@pytest.mark.parametrize(
    "group,year,private_process,central_process",
    [
        ("tllq", "UL16APV", "tllq_privateUL16APV", "tZq_centralUL16APV"),
        ("tttt_run2", "UL18", "tttt_privateUL18", "tttt_centralUL18"),
        ("tttt_run3", "2022EE", "tttt_private2022EE", "TTTT_central2022EE"),
        ("tth_run2", "UL17", "ttHJet_privateUL17", "ttH_centralUL17"),
        ("ttlnu_run2", "UL16", "ttlnuJet_privateUL16", "ttW_centralUL16"),
        ("ttll_run2", "UL18", "ttllJet_privateUL18", "ttZ_centralUL18"),
        ("ttlnu_run3", "2023BPix", "ttlnu_private2023BPix", "ttLNu_cental2023BPix"),
    ],
)
def test_audited_groups_are_mutually_exclusive_by_production_profile(
    group, year, private_process, central_process
):
    private_universe = build_active_sample_universe(
        {"private": _sample(private_process)}
    )
    central_universe = build_active_sample_universe(
        {"central": _sample(central_process)}
    )
    validate_active_sample_profile(private_universe, _mode("production"))
    validate_active_sample_profile(
        central_universe,
        _mode("production_central"),
    )
    with pytest.raises(production_sample_profile_error, match="SUMW2-PROFILE-E001"):
        validate_active_sample_profile(central_universe, _mode("production"))
    with pytest.raises(production_sample_profile_error, match="SUMW2-PROFILE-E002"):
        validate_active_sample_profile(
            private_universe,
            _mode("production_central"),
        )
    records = _group_active_variants((private_process, central_process))
    assert records[0]["signal_group"] == group
    assert records[0]["year"] == year


@pytest.mark.parametrize(
    "mode", ["production", "production_central", "full_diagnostics", "full_custom"]
)
def test_both_variants_reject_for_every_allocating_profile(mode):
    universe = build_active_sample_universe(
        {
            "private": _sample("tllq_private2022"),
            "central": _sample("tZq_central2022"),
        }
    )
    with pytest.raises(production_sample_profile_error, match="SUMW2-PROFILE-E003"):
        validate_active_sample_profile(universe, _mode(mode))


@pytest.mark.parametrize("mode", ["production", "production_central"])
def test_absent_group_and_unpaired_signals_create_no_false_requirement(mode):
    universe = build_active_sample_universe(
        {
            "background": _sample("TTTo2L2Nu_central2022"),
            "unpaired_signal": _sample("tHq_private2022", wc_names=["ctW"]),
        }
    )
    validate_active_sample_profile(universe, _mode(mode))


def test_explicit_contributor_and_prefix_must_exist_in_active_cfg():
    universe = build_active_sample_universe(
        {"data": _sample("dataUL18", is_data=True)}
    )
    for selector in (
        {"process_names": ["missing_centralUL18"]},
        {"process_prefixes": ["missing"]},
    ):
        block = {
            "flips": {
                "enabled": True,
                "source_contributors": {"data": selector},
            }
        }
        with pytest.raises(
            production_sample_profile_error,
            match=r"SUMW2-PROFILE-E004.*wrapper=.*sr_cfg_identities=.*Recommended correction",
        ):
            validate_active_sample_profile(
                universe,
                _mode("full_custom"),
                data_driven_products=block,
                data_driven_products_present=True,
                metadata_path="profile.yml",
            )


def test_missing_contributor_family_target_reports_e005():
    samples = {
        "data_dataset": _sample("dataUL18", is_data=True),
        "prompt_dataset": _sample("TTTo2L2Nu_centralUL18"),
    }
    products = resolve_data_driven_products(
        {
            "nonprompt": {
                "enabled": True,
                "source_contributors": {
                    "data": {"process_names": ["dataUL18"]},
                    "prompt_mc": {
                        "process_names": ["TTTo2L2Nu_centralUL18"]
                    },
                },
            },
            "flips": {"enabled": False},
        },
        data_driven_products_present=True,
        legacy_do_np=False,
        samples=samples,
        runtime_families=("met",),
        metadata_path="profile.yml",
    )
    policy = resolve_sumw2_storage_policy(
        {
            "mode": "full_custom",
            "rules": [{"process_names": ["dataUL18"], "variables": ["met"]}],
        },
        samples=samples,
        runtime_families=("met",),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    universe = build_active_sample_universe(samples)
    with pytest.raises(
        production_sample_profile_error,
        match=r"SUMW2-PROFILE-E005.*affected_data_driven_product='nonprompt'.*affected_family='met'",
    ):
        certify_production_sample_contract(universe, policy, products)


def test_catalog_ambiguity_reports_e006():
    ambiguous = VALIDATED_SIGNAL_VARIANT_GROUPS + (
        signal_variant_group(
            "duplicate_tllq",
            ("2022",),
            ("tllq_private",),
            ("other_central",),
        ),
    )
    with pytest.raises(production_sample_profile_error, match="SUMW2-PROFILE-E006"):
        _group_active_variants(("tllq_private2022",), groups=ambiguous)


def test_recognized_variant_with_shortened_year_alias_reports_e006():
    universe = build_active_sample_universe(
        {"bad_signal": _sample("tllq_private22", wc_names=["ctW"])}
    )
    with pytest.raises(
        production_sample_profile_error,
        match=r"SUMW2-PROFILE-E006.*active_cfg_processes=.*tllq_private22.*Recommended correction",
    ):
        validate_active_sample_profile(universe, _mode("production"))


def test_profile_contract_is_deterministic_and_tamper_rejected():
    samples = {
        "private_dataset": _sample("tllq_private2022", wc_names=["ctW"]),
        "data_dataset": _sample("data2022", is_data=True),
    }
    products = resolve_data_driven_products(
        {
            "nonprompt": {"enabled": False},
            "flips": {"enabled": False},
        },
        data_driven_products_present=True,
        legacy_do_np=False,
        samples=samples,
        runtime_families=("met",),
        metadata_path="profile.yml",
    )
    policy = resolve_sumw2_storage_policy(
        {
            "mode": "production",
            "rules": [{"process_names": ["data2022"], "variables": ["met"]}],
        },
        samples=samples,
        runtime_families=("met",),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    universe = build_active_sample_universe(
        samples,
        wrapper_identity="fullR3_run.sh",
    )
    contract = certify_production_sample_contract(universe, policy, products)
    assert contract["resolved_mode"] == "production"
    assert contract["signal_sample_profile"] == "private"
    assert contract["active_signal_variants"] == {
        "tllq:2022": {
            "signal_group": "tllq",
            "year": "2022",
            "selected_variant": "private",
            "processes": ["tllq_private2022"],
        }
    }
    validate_production_sample_contract(contract, policy)
    tampered = copy.deepcopy(contract)
    tampered["signal_sample_profile"] = "central"
    with pytest.raises(production_sample_profile_error, match="identity does not match"):
        validate_production_sample_contract(tampered, policy)


def _load_cfg_samples(cfg_paths):
    samples = {}
    for cfg_path in cfg_paths:
        cfg_path = Path(cfg_path)
        for raw_line in cfg_path.read_text(encoding="utf-8").splitlines():
            token = raw_line.split("#", 1)[0].strip()
            if not token or not token.endswith(".json"):
                continue
            json_path = Path(token)
            if not json_path.is_file():
                json_path = cfg_path.parent / token
            with json_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
            samples[f"{cfg_path.name}:{json_path.name}"] = payload
    return samples


def test_actual_maintained_2022_private_and_central_cfg_bundles_match_modes():
    cfg_root = Path("input_samples/cfgs")
    common = (
        cfg_root / "NDSkim_2022_background_samples.cfg",
        cfg_root / "NDSkim_2022_data_samples.cfg",
    )
    private_cfgs = (*common, cfg_root / "NDSkim_2022_mc_signal_samples_sr.cfg")
    central_cfgs = (*common, cfg_root / "NDSkim_2022_central_signal_samples.cfg")
    private_universe = build_active_sample_universe(
        _load_cfg_samples(private_cfgs),
        input_paths=private_cfgs,
        wrapper_identity="fullR3_run.sh",
    )
    central_universe = build_active_sample_universe(
        _load_cfg_samples(central_cfgs),
        input_paths=central_cfgs,
        wrapper_identity="synthetic-central-cfg",
    )
    validate_active_sample_profile(private_universe, _mode("production"))
    validate_active_sample_profile(
        central_universe,
        _mode("production_central"),
    )
    with pytest.raises(production_sample_profile_error, match="SUMW2-PROFILE-E001"):
        validate_active_sample_profile(central_universe, _mode("production"))
    with pytest.raises(production_sample_profile_error, match="SUMW2-PROFILE-E002"):
        validate_active_sample_profile(
            private_universe,
            _mode("production_central"),
        )


def test_cfg_identity_is_portable_and_bounded(tmp_path):
    cfg = tmp_path / "samples.cfg"
    cfg.write_text("sample.json\n", encoding="utf-8")
    universe = build_active_sample_universe(
        {"dataset": _sample("background2022")},
        input_paths=(str(cfg),),
        wrapper_identity="wrapper.sh",
    )
    assert universe.serialized_cfg_identities() == [
        {
            "basename": "samples.cfg",
            "content_sha256": universe.serialized_cfg_identities()[0][
                "content_sha256"
            ],
            "input_kind": "cfg",
        }
    ]
    assert str(tmp_path) not in json.dumps(universe.serialized_cfg_identities())
