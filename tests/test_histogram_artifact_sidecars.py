from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import pickle

import hist
import numpy as np
import pytest

from analysis.topeft_run2 import make_cards, make_cr_and_sr_plots, run_data_driven
from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from topcoffea.modules.utils import get_hist_from_pkl
from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.datacard_tools import load_and_merge_histogram_pkls
from topeft.modules.dataDrivenEstimation import (
    DataDrivenProducer,
    _producer_transformation_context,
)
from topeft.modules.data_driven_products import (
    certify_data_driven_preflight,
    data_driven_product_error,
    resolve_data_driven_products,
)
from topeft.modules.histogram_artifact import (
    derive_transformed_required_sumw2_processes,
    histogram_artifact_error,
    lineage_input_from_sidecar,
    merge_histogram_sidecars,
    metadata_sidecar_path,
    read_histogram_sidecar,
    validate_histogram_artifact,
    validate_nonprompt_output,
    write_histogram_artifact,
)
from topeft.modules import histogram_artifact
from topeft.modules.nominal_schema import (
    eft_nominal_key,
    evaluate_nominal_at_wc,
    materialize_legacy_histogram_dict,
    scalar_nominal_key,
    is_split_nominal_mapping,
    sumw2_key,
)
from topeft.modules.sumw2_policy import resolve_sumw2_storage_policy
from topeft.modules.paths import topeft_path
from topeft.modules.production_sample_profile import (
    build_active_sample_universe,
    certify_production_sample_contract,
)


_POLICY_SAMPLES = {
    "data_dataset": {
        "histAxisName": "dataUL18",
        "isData": True,
        "WCnames": [],
    },
    "prompt_dataset": {
        "histAxisName": "TTTo2L2Nu_centralUL18",
        "isData": False,
        "WCnames": [],
    },
    "signal_dataset": {
        "histAxisName": "WWTo2L2Nu_centralUL18",
        "isData": False,
        "WCnames": ["ctG"],
    },
}
_REGISTRY_NAMES = (
    "TOP22_006_CH_LST_SR",
    "TAU_CH_LST_SR",
    "OFFZ_SPLIT_CH_LST_SR",
    "FWD_CH_LST_SR",
    "CH_LST_CR",
    "TAU_CH_LST_CR",
    "ALL_CH_LST_SR",
)
_LEGACY_NONPROMPT_APPLICATION_REGIONS = {
    "isAR_1l",
    "isAR_2lSS",
    "isAR_2lOS",
    "isAR_3l",
}
_LEGACY_FLIPS_APPLICATION_REGION = "isAR_2lSS_OS"
with open(topeft_path("channels/ch_lst.json"), encoding="utf-8") as _registry_stream:
    _CHANNEL_REGISTRY = json.load(_registry_stream)
_REGISTRY_CASES = tuple(
    pytest.param(
        registry_name,
        category_name,
        category,
        id=f"{registry_name}/{category_name}",
    )
    for registry_name in _REGISTRY_NAMES
    for category_name, category in _CHANNEL_REGISTRY[registry_name].items()
)


def _axes(dense_name):
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name=dense_name),
    )


def _fill_sparse(dense_name, entries, *, channel="3l"):
    output = SparseHist(*_axes(dense_name), storage="Double")
    for process, appl, weight in entries:
        output.fill(
            process=process,
            channel=channel,
            systematic="nominal",
            appl=appl,
            **{dense_name: np.asarray([0.5])},
            weight=np.asarray([weight]),
        )
    return output


def _fill_eft(entries, *, channel="3l"):
    output = HistEFT(*_axes("njets"), wc_names=["ctG"], label="Events")
    for process, appl, weight in entries:
        output.fill(
            process=process,
            channel=channel,
            systematic="nominal",
            appl=appl,
            njets=np.asarray([0.5]),
            weight=np.asarray([weight]),
            eft_coeff=np.asarray([[1.25, 2.0, 3.0]]),
        )
    return output


@pytest.fixture
def policy():
    return resolve_sumw2_storage_policy(
        {"mode": "full_diagnostics"},
        samples=_POLICY_SAMPLES,
        runtime_families=("njets",),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )


def _certify_profile(policy, samples, products=None):
    if products is None:
        products = resolve_data_driven_products(
            {
                "nonprompt": {"enabled": False},
                "flips": {"enabled": False},
            },
            data_driven_products_present=True,
            legacy_do_np=False,
            samples=samples,
            runtime_families=policy.runtime_histogram_families,
            metadata_path="test_options.yml",
        )
    return certify_production_sample_contract(
        build_active_sample_universe(samples, wrapper_identity="pytest"),
        policy,
        products,
    )


def _processor_payload(*, channel="3l"):
    scalar_entries = (
        ("dataUL18", "isAR_3l", 10.0),
        ("TTTo2L2Nu_centralUL18", "isAR_3l", 3.0),
        ("dataUL18", "isAR_2lSS_OS", 4.0),
        ("TTTo2L2Nu_centralUL18", "isSR_3l", 2.0),
    )
    companion_entries = (
        ("dataUL18", "isAR_3l", 100.0),
        ("TTTo2L2Nu_centralUL18", "isAR_3l", 9.0),
        ("dataUL18", "isAR_2lSS_OS", 16.0),
        ("TTTo2L2Nu_centralUL18", "isSR_3l", 4.0),
        ("WWTo2L2Nu_centralUL18", "isAR_3l", 81.0),
        ("WWTo2L2Nu_centralUL18", "isSR_3l", 25.0),
    )
    return {
        scalar_nominal_key("njets"): _fill_sparse(
            "njets", scalar_entries, channel=channel
        ),
        eft_nominal_key("njets"): _fill_eft(
            (
                ("WWTo2L2Nu_centralUL18", "isAR_3l", 9.0),
                ("WWTo2L2Nu_centralUL18", "isSR_3l", 5.0),
            ),
            channel=channel,
        ),
        "njets_sumw2": _fill_sparse(
            "njets_sumw2", companion_entries, channel=channel
        ),
    }


def _processor_payload_for_regions(
    source_application_regions,
    *,
    family="njets",
):
    scalar_entries = []
    companion_entries = []
    for index, region in enumerate(sorted(set(source_application_regions)), start=1):
        entries = (
            ("dataUL18", 10.0 + index),
            ("TTTo2L2Nu_centralUL18", 3.0 + index),
        )
        for process, weight in entries:
            scalar_entries.append((process, region, weight))
            companion_entries.append((process, region, weight**2))
    return {
        scalar_nominal_key(family): _fill_sparse(family, scalar_entries),
        f"{family}_sumw2": _fill_sparse(
            f"{family}_sumw2",
            companion_entries,
        ),
    }


def _write_processor(path, policy, products_block=None, source_payload=None):
    samples = {
        "data_dataset": {
            "histAxisName": "dataUL18",
            "isData": True,
            "WCnames": [],
        },
        "prompt_dataset": {
            "histAxisName": "TTTo2L2Nu_centralUL18",
            "isData": False,
            "WCnames": [],
        },
        "signal_dataset": {
            "histAxisName": "WWTo2L2Nu_centralUL18",
            "isData": False,
            "WCnames": ["ctG"],
        },
    }
    products = resolve_data_driven_products(
        products_block
        or {
            "nonprompt": {
                "enabled": True,
                "source_contributors": {
                    "data": {"process_names": ["dataUL18"]},
                    "prompt_mc": {
                        "process_names": ["TTTo2L2Nu_centralUL18"]
                    },
                },
            },
            "flips": {
                "enabled": True,
                "source_contributors": {
                    "data": {"process_names": ["dataUL18"]},
                },
            },
        },
        data_driven_products_present=True,
        legacy_do_np=False,
        samples=samples,
        runtime_families=("njets",),
        metadata_path="test_options.yml",
    )
    requested, contract = certify_data_driven_preflight(products, policy)
    return write_histogram_artifact(
        path,
        histograms=source_payload or _processor_payload(),
        artifact_kind="processor_output",
        sumw2_storage_provenance=policy.to_provenance(),
        production_sample_contract=_certify_profile(policy, samples, products),
        requested_data_driven_products=requested,
        resolved_data_driven_contract=contract,
    )


def _legacy_v3_contract_from_current(contract, *, signal_processes):
    return {
        "contract_version": 3,
        "required_prompt_signal_processes": sorted(set(signal_processes)),
        "products": copy.deepcopy(contract["products"]),
    }


def _write_disjoint_source_processor(
    path,
    *,
    process_prefix,
    year,
    runtime_families,
    contribution,
    prompt_contribution=None,
    signal_process=None,
):
    if prompt_contribution is None:
        prompt_contribution = contribution
    data_process = f"data{year}"
    prompt_process = (
        f"TTTo2L2Nu_central{year}"
        if year.startswith("UL")
        else f"TTto2L2Nu_central{year}"
    )
    samples = {
        f"{process_prefix}_data_dataset": {
            "histAxisName": data_process,
            "isData": True,
            "WCnames": [],
        },
        f"{process_prefix}_prompt_dataset": {
            "histAxisName": prompt_process,
            "isData": False,
            "WCnames": [],
        },
    }
    if signal_process is not None:
        samples[f"{process_prefix}_signal_dataset"] = {
            "histAxisName": signal_process,
            "isData": False,
            "WCnames": [],
        }
    policy = resolve_sumw2_storage_policy(
        {"mode": "full_diagnostics"},
        samples=samples,
        runtime_families=runtime_families,
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    products = resolve_data_driven_products(
        {
            "nonprompt": {
                "enabled": True,
                "source_contributors": {
                    "data": {"process_names": [data_process]},
                    "prompt_mc": {"process_names": [prompt_process]},
                },
            },
            "flips": {
                "enabled": True,
                "source_contributors": {
                    "data": {"process_names": [data_process]},
                },
            },
        },
        data_driven_products_present=True,
        legacy_do_np=False,
        samples=samples,
        runtime_families=runtime_families,
        metadata_path=f"{process_prefix}_{year}.yml",
    )
    requested, contract = certify_data_driven_preflight(products, policy)
    requested["warnings"] = [f"{process_prefix}-warning"]
    payload = {}
    for family in runtime_families:
        payload[scalar_nominal_key(family)] = _fill_sparse(
            family,
            (
                (data_process, "isAR_3l", contribution),
                (prompt_process, "isAR_3l", prompt_contribution),
            ),
        )
        payload[sumw2_key(family)] = _fill_sparse(
            sumw2_key(family),
            (
                (data_process, "isAR_3l", contribution**2),
                (prompt_process, "isAR_3l", prompt_contribution**2),
            ),
        )
    sidecar = write_histogram_artifact(
        path,
        histograms=payload,
        artifact_kind="processor_output",
        sumw2_storage_provenance=policy.to_provenance(),
        production_sample_contract=_certify_profile(policy, samples, products),
        requested_data_driven_products=requested,
        resolved_data_driven_contract=contract,
    )
    return sidecar


def _independent_expected_applicability(source_application_regions):
    regions = set(source_application_regions)
    return {
        "nonprompt": bool(regions & _LEGACY_NONPROMPT_APPLICATION_REGIONS),
        "flips": _LEGACY_FLIPS_APPLICATION_REGION in regions,
    }


def _exercise_applicability_case(tmp_path, policy, source_application_regions):
    source_path = tmp_path / "processor.pkl.gz"
    source_sidecar = _write_processor(
        source_path,
        policy,
        source_payload=_processor_payload_for_regions(
            source_application_regions
        ),
    )
    producer = DataDrivenProducer(
        str(source_path),
        "",
        artifact_kind="nonprompt_output",
    )
    transformed = producer.getDataDrivenHistogram()
    context = producer.get_transformation_context("nonprompt_output")
    contract, required = derive_transformed_required_sumw2_processes(
        input_sidecar=source_sidecar,
        transformation_context=context,
        artifact_kind="nonprompt_output",
        transformed_histograms=transformed,
    )
    generated_labels = {"nonpromptUL18", "flipsUL18"}
    scalar_generated = generated_labels & {
        str(process)
        for process in transformed[scalar_nominal_key("njets")].axes["process"]
    }
    sumw2_generated = generated_labels & {
        str(process) for process in transformed["njets_sumw2"].axes["process"]
    }
    required_generated = generated_labels & set(required["njets"])
    return {
        "contract": contract,
        "required": required,
        "scalar_generated": scalar_generated,
        "sumw2_generated": sumw2_generated,
        "required_generated": required_generated,
    }


def test_nominal_reference_artifact_is_distinct_non_card_ready_and_has_no_nonprompt_sumw2(
    tmp_path,
    policy,
):
    source_path = tmp_path / "processor.pkl.gz"
    _write_processor(source_path, policy)
    reference_path = tmp_path / "reference.pkl.gz"
    producer = DataDrivenProducer(
        str(source_path),
        str(reference_path),
        artifact_kind="nonprompt_nominal_reference_output",
    )
    transformed = producer.getDataDrivenHistogram()
    producer.dumpToPickle()

    metadata = read_histogram_sidecar(reference_path)
    assert metadata["artifact"]["artifact_kind"] == "nonprompt_nominal_reference_output"
    assert metadata["nominal_reference_contract"]["reference_only"] is True
    assert metadata["nominal_reference_contract"]["card_ready"] is False
    assert metadata["nominal_reference_contract"]["statistically_complete"] is False
    scalar_processes = {
        str(process)
        for process in transformed[scalar_nominal_key("njets")].axes["process"]
    }
    sumw2_processes = {
        str(process) for process in transformed["njets_sumw2"].axes["process"]
    }
    assert "nonpromptUL18" in scalar_processes
    assert "nonpromptUL18" not in sumw2_processes
    validate_histogram_artifact(reference_path)
    with pytest.raises(histogram_artifact_error, match="Expected artifact_kind=nonprompt_output"):
        validate_nonprompt_output(reference_path, transformed, metadata)
    with pytest.raises(histogram_artifact_error, match="cannot be merged"):
        load_and_merge_histogram_pkls([str(reference_path)])


def test_nominal_reference_cli_interface_is_explicit_and_non_default():
    parser = run_data_driven._build_argument_parser()
    parsed = parser.parse_args(
        ["--input-pkl", "processor.pkl.gz", "--nominal-only-reference"]
    )
    assert parsed.nominal_only_reference is True
    assert run_data_driven._default_output_path(
        "processor.pkl.gz", nominal_only_reference=True
    ) == "processor_np_nominal_reference.pkl.gz"


@pytest.mark.parametrize(
    "registry_name,category_name,category",
    _REGISTRY_CASES,
)
def test_all_cr_sr_registry_entries_preserve_applicability_semantics(
    tmp_path,
    policy,
    registry_name,
    category_name,
    category,
):
    source_application_regions = sorted(
        set(category["appl_lst"]) | set(category.get("appl_lst_data", []))
    )
    expected_applicability = _independent_expected_applicability(
        source_application_regions
    )
    observed = _exercise_applicability_case(
        tmp_path,
        policy,
        source_application_regions,
    )
    expected_generated = {
        process
        for product, process in (
            ("nonprompt", "nonpromptUL18"),
            ("flips", "flipsUL18"),
        )
        if expected_applicability[product]
    }
    family = observed["contract"]["families"]["njets"]
    assert observed["contract"]["contract_version"] == 3
    assert family["source_application_regions"] == source_application_regions
    assert family["applicable_products"] == expected_applicability
    assert observed["scalar_generated"] == expected_generated
    assert observed["sumw2_generated"] == expected_generated
    assert observed["required_generated"] == expected_generated


def test_cr_sr_registry_schema_and_applicability_counts_are_guarded():
    classifications = {
        "nonprompt_flips": 0,
        "nonprompt_only": 0,
        "neither": 0,
        "flips_only": 0,
    }
    observed_ar_labels = set()
    optional_appl_lst_data = False
    entries = 0
    for registry_name in _REGISTRY_NAMES:
        registry = _CHANNEL_REGISTRY[registry_name]
        entries += len(registry)
        for category in registry.values():
            optional_appl_lst_data |= "appl_lst_data" not in category
            regions = set(category["appl_lst"]) | set(
                category.get("appl_lst_data", [])
            )
            observed_ar_labels.update(
                region for region in regions if region.startswith("isAR")
            )
            applicability = _independent_expected_applicability(regions)
            if applicability == {"nonprompt": True, "flips": True}:
                classifications["nonprompt_flips"] += 1
            elif applicability == {"nonprompt": True, "flips": False}:
                classifications["nonprompt_only"] += 1
            elif applicability == {"nonprompt": False, "flips": True}:
                classifications["flips_only"] += 1
            else:
                classifications["neither"] += 1

    assert len(_REGISTRY_NAMES) == 7
    assert entries == 37
    assert classifications == {
        "nonprompt_flips": 11,
        "nonprompt_only": 18,
        "neither": 8,
        "flips_only": 0,
    }
    assert observed_ar_labels == {
        "isAR_1l",
        "isAR_2lSS",
        "isAR_2lSS_OS",
        "isAR_2lOS",
        "isAR_3l",
    }
    assert optional_appl_lst_data is True
    assert histogram_artifact.derive_data_driven_applicability(
        ["isAR_future"]
    ) == {"nonprompt": False, "flips": False}


@pytest.mark.parametrize(
    "left_class,right_class,expected",
    [
        ("neither", "neither", {"nonprompt": False, "flips": False}),
        ("neither", "nonprompt", {"nonprompt": True, "flips": False}),
        ("nonprompt", "nonprompt", {"nonprompt": True, "flips": False}),
        ("nonprompt", "flips", {"nonprompt": True, "flips": True}),
        ("flips", "neither", {"nonprompt": True, "flips": True}),
    ],
    ids=[
        "neither_neither",
        "neither_nonprompt",
        "nonprompt_nonprompt",
        "nonprompt_flips",
        "flips_neither",
    ],
)
def test_registry_category_union_applicability(
    tmp_path,
    policy,
    left_class,
    right_class,
    expected,
):
    regions_by_class = {"neither": [], "nonprompt": [], "flips": []}
    for registry_name in _REGISTRY_NAMES:
        for category in _CHANNEL_REGISTRY[registry_name].values():
            regions = set(category["appl_lst"]) | set(
                category.get("appl_lst_data", [])
            )
            applicability = _independent_expected_applicability(regions)
            classification = (
                "flips"
                if applicability["flips"]
                else "nonprompt"
                if applicability["nonprompt"]
                else "neither"
            )
            regions_by_class[classification].append(regions)
    left_index = 0
    right_index = 1 if left_class == right_class else 0
    union_regions = (
        regions_by_class[left_class][left_index]
        | regions_by_class[right_class][right_index]
    )
    observed = _exercise_applicability_case(
        tmp_path,
        policy,
        union_regions,
    )
    assert observed["contract"]["families"]["njets"][
        "applicable_products"
    ] == expected


@pytest.mark.parametrize(
    "source_application_regions,expected_generated",
    [
        (["isAR_1l", "isSR_1l"], {"nonpromptUL18"}),
        (["isAR_2lSS_OS"], {"flipsUL18"}),
        (["isSR_1l"], set()),
    ],
    ids=["nonprompt_without_flips", "flips", "no_application_region"],
)
def test_current_transformer_matches_exact_legacy_applicability_cases(
    tmp_path,
    policy,
    source_application_regions,
    expected_generated,
):
    observed = _exercise_applicability_case(
        tmp_path,
        policy,
        source_application_regions,
    )
    assert observed["scalar_generated"] == expected_generated
    assert observed["sumw2_generated"] == expected_generated
    assert observed["required_generated"] == expected_generated


def test_enabled_but_inapplicable_flips_write_truthful_contract(
    tmp_path,
    policy,
):
    source_path = tmp_path / "processor_nonprompt_only.pkl.gz"
    output_path = tmp_path / "nonprompt_only.pkl.gz"
    source_regions = ["isAR_1l", "isSR_1l"]
    source_sidecar = _write_processor(
        source_path,
        policy,
        source_payload=_processor_payload_for_regions(source_regions),
    )
    source_identity = (
        source_path.read_bytes(),
        metadata_sidecar_path(source_path).read_bytes(),
    )
    run_data_driven.main(
        [
            "--input-pkl",
            str(source_path),
            "--output-pkl",
            str(output_path),
            "--quiet",
        ]
    )
    sidecar = read_histogram_sidecar(output_path)
    family_contract = sidecar["transformation_contract"]["families"]["njets"]
    family_manifest = sidecar["sumw2_content_manifest"]["families"]["njets"]
    output = get_hist_from_pkl(str(output_path))

    assert sidecar["transformation_contract"]["contract_version"] == 3
    assert family_contract["source_application_regions"] == source_regions
    assert family_contract["applicable_products"] == {
        "nonprompt": True,
        "flips": False,
    }
    assert family_contract["generated_nonprompt_processes"] == [
        "nonpromptUL18"
    ]
    assert family_contract["generated_flips_processes"] == []
    assert "nonpromptUL18" in output[scalar_nominal_key("njets")].axes[
        "process"
    ]
    assert "nonpromptUL18" in output["njets_sumw2"].axes["process"]
    assert "flipsUL18" not in output[scalar_nominal_key("njets")].axes[
        "process"
    ]
    assert "flipsUL18" not in output["njets_sumw2"].axes["process"]
    assert "nonpromptUL18" in family_manifest["required_sumw2_processes"]
    assert "flipsUL18" not in family_manifest["required_sumw2_processes"]
    assert source_identity == (
        source_path.read_bytes(),
        metadata_sidecar_path(source_path).read_bytes(),
    )
    assert source_sidecar["artifact"]["artifact_kind"] == "processor_output"


@pytest.mark.parametrize(
    "missing_component",
    ["nominal", "sumw2"],
)
def test_applicable_flips_remain_strict_before_publication(
    tmp_path,
    policy,
    missing_component,
):
    source_path = tmp_path / "processor_with_flips.pkl.gz"
    rejected_path = tmp_path / f"missing_flips_{missing_component}.pkl.gz"
    source_sidecar = _write_processor(source_path, policy)
    producer = DataDrivenProducer(
        str(source_path),
        "",
        artifact_kind="nonprompt_output",
    )
    transformed = copy.deepcopy(producer.getDataDrivenHistogram())
    context = producer.get_transformation_context("nonprompt_output")
    assert context["families"]["njets"]["applicable_products"]["flips"] is True
    key = (
        scalar_nominal_key("njets")
        if missing_component == "nominal"
        else "njets_sumw2"
    )
    transformed[key] = transformed[key].remove("process", ["flipsUL18"])

    with pytest.raises(
        histogram_artifact_error,
        match=(
            "scalar nominal roles differ.*flipsUL18"
            if missing_component == "nominal"
            else "requires sumw2 processes absent.*flipsUL18"
        ),
    ):
        write_histogram_artifact(
            rejected_path,
            histograms=transformed,
            artifact_kind="nonprompt_output",
            sumw2_storage_provenance=policy.to_provenance(),
            lineage_inputs=[lineage_input_from_sidecar(source_sidecar)],
            input_sidecar=source_sidecar,
            transformation_context=context,
        )
    assert not rejected_path.exists()
    assert not metadata_sidecar_path(rejected_path).exists()


def test_mixed_families_keep_independent_flips_applicability(tmp_path):
    runtime_families = ("njets", "met")
    local_policy = resolve_sumw2_storage_policy(
        {"mode": "full_diagnostics"},
        samples=_POLICY_SAMPLES,
        runtime_families=runtime_families,
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    payload = {}
    payload.update(
        _processor_payload_for_regions(
            ["isAR_1l", "isSR_1l"],
            family="njets",
        )
    )
    payload.update(
        _processor_payload_for_regions(
            ["isAR_2lSS", "isAR_2lSS_OS", "isSR_2lSS"],
            family="met",
        )
    )
    source_path = tmp_path / "mixed_processor.pkl.gz"
    source_sidecar = _write_processor(
        source_path,
        local_policy,
        source_payload=payload,
    )
    producer = DataDrivenProducer(str(source_path), "")
    transformed = producer.getDataDrivenHistogram()
    contract, required = derive_transformed_required_sumw2_processes(
        input_sidecar=source_sidecar,
        transformation_context=producer.get_transformation_context(
            "nonprompt_output"
        ),
        artifact_kind="nonprompt_output",
        transformed_histograms=transformed,
    )

    assert contract["families"]["njets"]["applicable_products"] == {
        "nonprompt": True,
        "flips": False,
    }
    assert contract["families"]["met"]["applicable_products"] == {
        "nonprompt": True,
        "flips": True,
    }
    assert "flipsUL18" not in required["njets"]
    assert "flipsUL18" in required["met"]
    assert "flipsUL18" not in transformed[
        scalar_nominal_key("njets")
    ].axes["process"]
    assert "flipsUL18" in transformed[
        scalar_nominal_key("met")
    ].axes["process"]


@pytest.mark.parametrize(
    "mutation,error_match",
    [
        (
            "contradict_flips_false",
            "applicable_products.*contradicts authoritative",
        ),
        (
            "contradict_flips_true",
            "applicable_products.*contradicts authoritative",
        ),
        (
            "contradict_nonprompt",
            "applicable_products.*contradicts authoritative",
        ),
        ("non_boolean", "applicable_products.*values must be booleans"),
        (
            "missing_applicability",
            "Invalid transformation context.*missing=.*applicable_products",
        ),
        ("missing_family", "cover the input runtime families"),
        ("unknown_product", "unknown=.*future_product"),
        (
            "generated_processes",
            "Transformed flips processes disagree",
        ),
    ],
)
def test_applicability_context_tampering_is_rejected(
    tmp_path,
    policy,
    mutation,
    error_match,
):
    include_flips = mutation not in {"contradict_flips_true"}
    regions = (
        ["isAR_3l", "isAR_2lSS_OS"]
        if include_flips
        else ["isAR_3l"]
    )
    source_path = tmp_path / f"tamper_{mutation}.pkl.gz"
    source_sidecar = _write_processor(
        source_path,
        policy,
        source_payload=_processor_payload_for_regions(regions),
    )
    producer = DataDrivenProducer(str(source_path), "")
    transformed = producer.getDataDrivenHistogram()
    context = copy.deepcopy(
        producer.get_transformation_context("nonprompt_output")
    )
    family = context["families"]["njets"]
    if mutation == "contradict_flips_false":
        family["applicable_products"]["flips"] = False
    elif mutation == "contradict_flips_true":
        family["applicable_products"]["flips"] = True
    elif mutation == "contradict_nonprompt":
        family["applicable_products"]["nonprompt"] = False
    elif mutation == "non_boolean":
        family["applicable_products"]["flips"] = 1
    elif mutation == "missing_applicability":
        family.pop("applicable_products")
    elif mutation == "missing_family":
        context["families"].pop("njets")
    elif mutation == "unknown_product":
        family["applicable_products"]["future_product"] = False
    else:
        family["generated_flips_processes"] = []

    with pytest.raises(histogram_artifact_error, match=error_match):
        derive_transformed_required_sumw2_processes(
            input_sidecar=source_sidecar,
            transformation_context=context,
            artifact_kind="nonprompt_output",
            transformed_histograms=transformed,
        )


def test_caller_authored_transformation_context_is_rejected(tmp_path, policy):
    source_path = tmp_path / "caller_context.pkl.gz"
    source_sidecar = _write_processor(source_path, policy)
    producer = DataDrivenProducer(str(source_path), "")
    producer_context = producer.get_transformation_context("nonprompt_output")
    caller_context = {
        "eft_prompt_projection": copy.deepcopy(
            producer_context["eft_prompt_projection"]
        ),
        "families": copy.deepcopy(producer_context["families"]),
    }
    with pytest.raises(
        histogram_artifact_error,
        match="caller-authored transformation contexts are not accepted",
    ):
        derive_transformed_required_sumw2_processes(
            input_sidecar=source_sidecar,
            transformation_context=caller_context,
            artifact_kind="nonprompt_output",
            transformed_histograms=producer.getDataDrivenHistogram(),
        )


def _write_transformed_case(
    tmp_path,
    policy,
    name,
    source_application_regions,
):
    source_path = tmp_path / f"{name}_processor.pkl.gz"
    output_path = tmp_path / f"{name}_nonprompt.pkl.gz"
    source_sidecar = _write_processor(
        source_path,
        policy,
        source_payload=_processor_payload_for_regions(
            source_application_regions
        ),
    )
    producer = DataDrivenProducer(str(source_path), "")
    write_histogram_artifact(
        output_path,
        histograms=producer.getDataDrivenHistogram(),
        artifact_kind="nonprompt_output",
        sumw2_storage_provenance=policy.to_provenance(),
        lineage_inputs=[lineage_input_from_sidecar(source_sidecar)],
        input_sidecar=source_sidecar,
        transformation_context=producer.get_transformation_context(
            "nonprompt_output"
        ),
    )
    return output_path


def _downgrade_transformation_contract_fixture(output_path):
    sidecar_path = metadata_sidecar_path(output_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["transformation_contract"][
        "contract_version"
    ] = histogram_artifact.PRIOR_TRANSFORMATION_CONTRACT_VERSION
    for family in sidecar["transformation_contract"]["families"].values():
        family.pop("source_application_regions")
        family.pop("applicable_products")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return read_histogram_sidecar(output_path)


def test_prior_transformation_contract_remains_readable_and_mergeable(
    tmp_path,
    policy,
):
    old_sidecars = []
    for index in range(2):
        output_path = _write_transformed_case(
            tmp_path,
            policy,
            f"old_{index}",
            ["isAR_3l", "isAR_2lSS_OS", "isSR_3l"],
        )
        old_sidecars.append(
            _downgrade_transformation_contract_fixture(output_path)
        )

    merged = merge_histogram_sidecars(old_sidecars)
    assert {
        sidecar["transformation_contract"]["contract_version"]
        for sidecar in old_sidecars
    } == {histogram_artifact.PRIOR_TRANSFORMATION_CONTRACT_VERSION}
    assert merged["transformation_contract"]["contract_version"] == 2
    assert merged["required_sumw2_processes"]["njets"] == [
        "TTTo2L2Nu_centralUL18",
        "dataUL18",
        "flipsUL18",
        "nonpromptUL18",
    ]


def test_new_contract_merge_recomputes_applicability_from_source_evidence(
    tmp_path,
    policy,
):
    nonprompt_path = _write_transformed_case(
        tmp_path,
        policy,
        "merge_nonprompt",
        ["isAR_1l", "isSR_1l"],
    )
    flips_path = _write_transformed_case(
        tmp_path,
        policy,
        "merge_flips",
        ["isAR_2lSS", "isAR_2lSS_OS", "isSR_2lSS"],
    )
    nonprompt_sidecar = read_histogram_sidecar(nonprompt_path)
    flips_sidecar = read_histogram_sidecar(flips_path)
    merged = merge_histogram_sidecars(
        [nonprompt_sidecar, flips_sidecar]
    )
    family = merged["transformation_contract"]["families"]["njets"]

    assert merged["transformation_contract"]["contract_version"] == 3
    assert family["source_application_regions"] == [
        "isAR_1l",
        "isAR_2lSS",
        "isAR_2lSS_OS",
        "isSR_1l",
        "isSR_2lSS",
    ]
    assert family["applicable_products"] == {
        "nonprompt": True,
        "flips": True,
    }
    assert "flipsUL18" in merged["required_sumw2_processes"]["njets"]

    tampered_flips = copy.deepcopy(flips_sidecar)
    tampered_flips["transformation_contract"]["families"]["njets"][
        "generated_flips_processes"
    ] = []
    with pytest.raises(
        histogram_artifact_error,
        match="Transformed flips processes disagree",
    ):
        merge_histogram_sidecars([nonprompt_sidecar, tampered_flips])

    old_sidecar = _downgrade_transformation_contract_fixture(
        _write_transformed_case(
            tmp_path,
            policy,
            "mixed_old",
            ["isAR_3l", "isAR_2lSS_OS", "isSR_3l"],
        )
    )
    with pytest.raises(
        histogram_artifact_error,
        match="prior and applicability-aware",
    ):
        merge_histogram_sidecars([old_sidecar, flips_sidecar])


def test_processor_sidecar_uses_family_free_generated_output_contract(
    tmp_path, policy
):
    path = tmp_path / "processor.pkl.gz"
    sidecar = _write_processor(path, policy)
    contract = sidecar["resolved_data_driven_contract"]
    assert contract["contract_version"] == 4
    assert set(contract) == {
        "contract_version",
        "nonprompt_policy",
        "resolved_prompt_process_set",
        "policy_migration",
        "products",
    }
    assert "required_prompt_signal_processes" not in contract
    assert "eft_prompt_processes" not in contract["nonprompt_policy"]
    assert "families" not in contract
    assert contract["products"]["nonprompt"] == {
        "enabled": True,
        "generated_outputs": {
            "nonpromptUL18": {
                "year": "UL18",
                "source_contributors": {
                    "data": ["dataUL18"],
                    "prompt_mc": ["TTTo2L2Nu_centralUL18"],
                },
                "required_source_sumw2_processes": [
                    "TTTo2L2Nu_centralUL18",
                    "dataUL18",
                ],
            }
        },
        "output_processes": ["nonpromptUL18"],
    }


def test_certified_precanonical_contract_normalizes_from_prompt_contributors(
    tmp_path, policy
):
    path = tmp_path / "legacy_v3_processor.pkl.gz"
    _write_processor(path, policy)
    sidecar_path = metadata_sidecar_path(path)
    serialized = json.loads(sidecar_path.read_text(encoding="utf-8"))
    current = serialized["resolved_data_driven_contract"]
    serialized["resolved_data_driven_contract"] = _legacy_v3_contract_from_current(
        current,
        signal_processes=[],
    )
    sidecar_path.write_text(json.dumps(serialized), encoding="utf-8")

    normalized = read_histogram_sidecar(path)["resolved_data_driven_contract"]
    assert normalized["contract_version"] == 4
    assert normalized["resolved_prompt_process_set"] == [
        "TTTo2L2Nu_centralUL18"
    ]
    assert "required_prompt_signal_processes" not in normalized
    assert normalized["policy_migration"] == {
        "status": "normalized_certified_legacy_contract",
        "previous_contract_version": 3,
        "serialized_prompt_process_set": ["TTTo2L2Nu_centralUL18"],
        "added_prompt_processes": [],
        "removed_prompt_processes": [],
    }


def test_modern_contract_readback_is_unchanged_by_legacy_normalization(
    tmp_path, policy
):
    path = tmp_path / "modern_processor.pkl.gz"
    _write_processor(path, policy)
    serialized = json.loads(metadata_sidecar_path(path).read_text(encoding="utf-8"))

    normalized = read_histogram_sidecar(path)["resolved_data_driven_contract"]
    assert normalized == serialized["resolved_data_driven_contract"]


@pytest.mark.parametrize("tamper", ["missing_contributors", "wrong_year", "extra_field"])
def test_precanonical_contract_normalization_fails_closed(tmp_path, policy, tamper):
    path = tmp_path / f"legacy_v3_{tamper}.pkl.gz"
    _write_processor(path, policy)
    sidecar_path = metadata_sidecar_path(path)
    serialized = json.loads(sidecar_path.read_text(encoding="utf-8"))
    legacy = _legacy_v3_contract_from_current(
        serialized["resolved_data_driven_contract"],
        signal_processes=[],
    )
    output = legacy["products"]["nonprompt"]["generated_outputs"][
        "nonpromptUL18"
    ]
    if tamper == "missing_contributors":
        output.pop("source_contributors")
    elif tamper == "wrong_year":
        output["source_contributors"]["prompt_mc"] = [
            "TTTo2L2Nu_centralUL17"
        ]
        output["required_source_sumw2_processes"] = [
            "TTTo2L2Nu_centralUL17",
            "dataUL18",
        ]
    else:
        legacy["resolved_prompt_process_set"] = ["TTTo2L2Nu_centralUL18"]
    serialized["resolved_data_driven_contract"] = legacy
    sidecar_path.write_text(json.dumps(serialized), encoding="utf-8")

    with pytest.raises(histogram_artifact_error):
        read_histogram_sidecar(path)


def test_normalized_precanonical_contract_uses_current_merge_path(tmp_path, policy):
    sidecars = []
    for index in range(2):
        path = tmp_path / f"legacy_v3_merge_{index}.pkl.gz"
        _write_processor(path, policy)
        sidecar_path = metadata_sidecar_path(path)
        serialized = json.loads(sidecar_path.read_text(encoding="utf-8"))
        serialized["resolved_data_driven_contract"] = (
            _legacy_v3_contract_from_current(
                serialized["resolved_data_driven_contract"],
                signal_processes=[],
            )
        )
        sidecar_path.write_text(json.dumps(serialized), encoding="utf-8")
        sidecars.append(read_histogram_sidecar(path))

    merged = merge_histogram_sidecars(sidecars)
    assert merged["resolved_data_driven_contract"]["contract_version"] == 4
    assert merged["resolved_data_driven_contract"]["resolved_prompt_process_set"] == [
        "TTTo2L2Nu_centralUL18"
    ]


def test_legacy_sumw2_profile_sidecar_reopens_but_cannot_transform(
    tmp_path,
    policy,
):
    path = tmp_path / "legacy_profile_processor.pkl.gz"
    _write_processor(path, policy)
    sidecar_path = metadata_sidecar_path(path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    provenance = sidecar["sumw2_storage_provenance"]
    provenance["schema_version"] = 1
    provenance.pop("resolved_mode")
    provenance.pop("signal_sample_profile")
    sidecar.pop("production_sample_contract")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    reopened = validate_histogram_artifact(path)
    assert reopened["metadata"]["sumw2_storage_provenance"]["schema_version"] == 1
    with pytest.raises(
        data_driven_product_error,
        match=r"predates certified production sample profiles.*read-only.*Regenerate",
    ):
        DataDrivenProducer(str(path), "")


@pytest.mark.parametrize("tamper", ["delete_contributor", "move_contributor", "output_year", "orphan_output"])
def test_processor_sidecar_rejects_generated_output_contract_tampering(
    tmp_path, policy, tamper
):
    path = tmp_path / f"processor_{tamper}.pkl.gz"
    _write_processor(path, policy)
    sidecar_path = metadata_sidecar_path(path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    product = sidecar["resolved_data_driven_contract"]["products"]["nonprompt"]
    output = product["generated_outputs"]["nonpromptUL18"]
    if tamper == "delete_contributor":
        output["source_contributors"]["prompt_mc"].clear()
    elif tamper == "move_contributor":
        output["source_contributors"]["prompt_mc"] = [
            "TTTo2L2Nu_centralUL17"
        ]
        output["required_source_sumw2_processes"] = [
            "TTTo2L2Nu_centralUL17",
            "dataUL18",
        ]
    elif tamper == "output_year":
        output["year"] = "UL17"
    else:
        product["generated_outputs"] = {
            "nonpromptUL17": {
                "year": "UL17",
                "source_contributors": {
                    "data": [],
                    "prompt_mc": ["TTTo2L2Nu_centralUL17"],
                },
                "required_source_sumw2_processes": [
                    "TTTo2L2Nu_centralUL17"
                ],
            },
            **product["generated_outputs"],
        }
        product["output_processes"] = ["nonpromptUL17", "nonpromptUL18"]
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(histogram_artifact_error):
        validate_histogram_artifact(path)


def test_contract_version_one_processor_reresolves_before_transform(
    tmp_path, policy
):
    input_path = tmp_path / "processor_v1.pkl.gz"
    output_path = tmp_path / "nonprompt_from_v1.pkl.gz"
    sidecar = _write_processor(input_path, policy)
    targets_by_process = {}
    for target in policy.resolved_targets:
        targets_by_process.setdefault(target.process, []).append(target.to_dict())
    sidecar["resolved_data_driven_contract"] = {
        "contract_version": 1,
        "families": {
            "njets": {
                "nonprompt": {
                    "enabled": True,
                    "output_processes": ["nonpromptUL18"],
                    "source_contributors": {
                        "data": ["dataUL18"],
                        "prompt_mc": ["TTTo2L2Nu_centralUL18"],
                    },
                    "required_source_sumw2_processes": [
                        "TTTo2L2Nu_centralUL18",
                        "dataUL18",
                    ],
                    "required_source_sumw2_targets": sorted(
                        targets_by_process["TTTo2L2Nu_centralUL18"]
                        + targets_by_process["dataUL18"],
                        key=lambda target: (
                            target["dataset"],
                            target["process"],
                            target["family"],
                        ),
                    ),
                    "requirements_satisfied": True,
                },
                "flips": {
                    "enabled": True,
                    "output_processes": ["flipsUL18"],
                    "source_contributors": {"data": ["dataUL18"]},
                    "required_source_sumw2_processes": ["dataUL18"],
                    "required_source_sumw2_targets": targets_by_process[
                        "dataUL18"
                    ],
                    "requirements_satisfied": True,
                },
            }
        },
    }
    metadata_sidecar_path(input_path).write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    reopened = validate_histogram_artifact(input_path)
    assert reopened["metadata"]["resolved_data_driven_contract"][
        "contract_version"
    ] == 1
    run_data_driven.main(
        [
            "--input-pkl",
            str(input_path),
            "--output-pkl",
            str(output_path),
            "--quiet",
        ]
    )
    migrated = read_histogram_sidecar(output_path)[
        "resolved_data_driven_contract"
    ]
    assert migrated["contract_version"] == 4
    assert migrated["policy_migration"]["previous_contract_version"] == 1


def test_nonprompt_transformation_uses_certified_multi_year_output_map(
    tmp_path,
):
    samples = {
        "data_17": {"histAxisName": "dataUL17", "isData": True, "WCnames": []},
        "data_18": {"histAxisName": "dataUL18", "isData": True, "WCnames": []},
        "prompt_17": {
            "histAxisName": "TTTo2L2Nu_centralUL17",
            "isData": False,
            "WCnames": [],
        },
        "ignored_17": {
            "histAxisName": "WWTo2L2Nu_centralUL17",
            "isData": False,
            "WCnames": [],
        },
    }
    local_policy = resolve_sumw2_storage_policy(
        {"mode": "full_diagnostics"},
        samples=samples,
        runtime_families=("njets",),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    products = resolve_data_driven_products(
        {
            "nonprompt": {
                "enabled": True,
                "source_contributors": {
                    "data": {"process_prefixes": ["data"]},
                    "prompt_mc": {
                        "process_names": ["TTTo2L2Nu_centralUL17"]
                    },
                },
            },
            "flips": {"enabled": False},
        },
        data_driven_products_present=True,
        legacy_do_np=False,
        samples=samples,
        runtime_families=("njets",),
        metadata_path="test_options.yml",
    )
    requested, contract = certify_data_driven_preflight(products, local_policy)
    scalar_entries = (
        ("dataUL17", "isAR_3l", 10.0),
        ("TTTo2L2Nu_centralUL17", "isAR_3l", 3.0),
        ("WWTo2L2Nu_centralUL17", "isAR_3l", 50.0),
        ("dataUL18", "isAR_3l", 4.0),
    )
    companion_entries = (
        ("dataUL17", "isAR_3l", 100.0),
        ("TTTo2L2Nu_centralUL17", "isAR_3l", 9.0),
        ("WWTo2L2Nu_centralUL17", "isAR_3l", 2500.0),
        ("dataUL18", "isAR_3l", 16.0),
    )
    input_path = tmp_path / "multi_year_processor.pkl.gz"
    output_path = tmp_path / "multi_year_nonprompt.pkl.gz"
    write_histogram_artifact(
        input_path,
        histograms={
            scalar_nominal_key("njets"): _fill_sparse("njets", scalar_entries),
            "njets_sumw2": _fill_sparse("njets_sumw2", companion_entries),
        },
        artifact_kind="processor_output",
        sumw2_storage_provenance=local_policy.to_provenance(),
        production_sample_contract=_certify_profile(local_policy, samples, products),
        requested_data_driven_products=requested,
        resolved_data_driven_contract=contract,
    )
    run_data_driven.main(
        [
            "--input-pkl",
            str(input_path),
            "--output-pkl",
            str(output_path),
            "--quiet",
        ]
    )
    output = get_hist_from_pkl(str(output_path))
    assert _processes(output[scalar_nominal_key("njets")]) == [
        "nonpromptUL17",
        "nonpromptUL18",
    ]
    assert _total_for_process(
        output[scalar_nominal_key("njets")], "nonpromptUL17"
    ) == pytest.approx(7.0)
    assert _total_for_process(
        output[scalar_nominal_key("njets")], "nonpromptUL18"
    ) == pytest.approx(4.0)
    assert _total_for_process(output["njets_sumw2"], "nonpromptUL17") == pytest.approx(
        109.0
    )
    assert _total_for_process(output["njets_sumw2"], "nonpromptUL18") == pytest.approx(
        16.0
    )
    sidecar = read_histogram_sidecar(output_path)
    assert sidecar["transformation_contract"]["families"]["njets"][
        "generated_nonprompt_processes"
    ] == ["nonpromptUL17", "nonpromptUL18"]
    migrated_contract = sidecar["resolved_data_driven_contract"]
    assert migrated_contract["resolved_prompt_process_set"] == contract[
        "resolved_prompt_process_set"
    ]
    assert migrated_contract["policy_migration"] == {
        "status": "current_contract_revalidated",
        "previous_contract_version": 4,
        "serialized_prompt_process_set": ["TTTo2L2Nu_centralUL17"],
        "added_prompt_processes": [],
        "removed_prompt_processes": [],
    }


@pytest.mark.parametrize("mutation", ["missing_generated_nominal", "extra_generated_year"])
def test_transformed_nominal_labels_must_match_generated_output_map(
    tmp_path, policy, mutation
):
    source_path = tmp_path / f"processor_{mutation}.pkl.gz"
    output_path = tmp_path / f"nonprompt_{mutation}.pkl.gz"
    source_sidecar = _write_processor(source_path, policy)
    producer = DataDrivenProducer(
        str(source_path),
        "",
        artifact_kind="nonprompt_output",
    )
    transformed = copy.deepcopy(producer.getDataDrivenHistogram())
    if mutation == "missing_generated_nominal":
        transformed[scalar_nominal_key("njets")] = transformed[
            scalar_nominal_key("njets")
        ].remove("process", ["nonpromptUL18"])
    else:
        transformed[scalar_nominal_key("njets")].fill(
            process="nonpromptUL17",
            channel="3l",
            systematic="nominal",
            njets=np.asarray([0.5]),
            weight=np.asarray([1.0]),
        )
    with pytest.raises(
        histogram_artifact_error,
        match="scalar nominal roles differ.*nonpromptUL",
    ):
        write_histogram_artifact(
            output_path,
            histograms=transformed,
            artifact_kind="nonprompt_output",
            sumw2_storage_provenance=policy.to_provenance(),
            production_sample_contract=_certify_profile(policy, _POLICY_SAMPLES),
            lineage_inputs=[lineage_input_from_sidecar(source_sidecar)],
            input_sidecar=source_sidecar,
            transformation_context=producer.get_transformation_context(
                "nonprompt_output"
            ),
        )
    assert not output_path.exists()
    assert not metadata_sidecar_path(output_path).exists()


def test_transformed_sidecar_roles_cannot_disagree_with_output_map(tmp_path, policy):
    source_path = tmp_path / "processor_role_mismatch.pkl.gz"
    output_path = tmp_path / "nonprompt_role_mismatch.pkl.gz"
    _write_processor(source_path, policy)
    run_data_driven.main(
        [
            "--input-pkl",
            str(source_path),
            "--output-pkl",
            str(output_path),
            "--quiet",
        ]
    )
    sidecar_path = metadata_sidecar_path(output_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["transformation_contract"]["families"]["njets"][
        "generated_nonprompt_processes"
    ] = []
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(
        histogram_artifact_error,
        match="Transformed nonprompt processes disagree.*expected=.*nonpromptUL18",
    ):
        validate_histogram_artifact(output_path)


def test_merge_rejects_incompatible_generated_output_maps(tmp_path, policy):
    first_path = tmp_path / "processor_first.pkl.gz"
    second_path = tmp_path / "processor_second.pkl.gz"
    first = _write_processor(first_path, policy)
    second = _write_processor(second_path, policy)
    second = copy.deepcopy(second)
    output = second["resolved_data_driven_contract"]["products"]["nonprompt"][
        "generated_outputs"
    ]["nonpromptUL18"]
    output["year"] = "UL17"
    with pytest.raises(
        histogram_artifact_error,
        match="disagree on resolved data-driven output year for nonprompt/nonpromptUL18",
    ):
        merge_histogram_sidecars([first, second])


def test_merge_composes_requested_warnings_without_weakening_policy_checks(tmp_path):
    first = _write_disjoint_source_processor(
        tmp_path / "first.pkl.gz",
        process_prefix="run3_2022",
        year="2022",
        runtime_families=("njets",),
        contribution=1.0,
    )
    second = _write_disjoint_source_processor(
        tmp_path / "second.pkl.gz",
        process_prefix="run3_2022EE",
        year="2022EE",
        runtime_families=("njets",),
        contribution=2.0,
    )
    first["requested_data_driven_products"]["warnings"] = [
        "2022 contributors",
        "shared warning",
    ]
    second["requested_data_driven_products"]["warnings"] = [
        "2022EE contributors",
        "shared warning",
    ]
    for sidecar in (first, second):
        sidecar["requested_data_driven_products"]["source"] = (
            "implicit_legacy_data_driven_default"
        )

    merged = merge_histogram_sidecars([first, second])
    requested = merged["requested_data_driven_products"]
    assert requested["schema_version"] == 1
    assert requested["source"] == "implicit_legacy_data_driven_default"
    assert requested["products"] == first["requested_data_driven_products"]["products"]
    assert requested["warnings"] == [
        "2022 contributors",
        "2022EE contributors",
        "shared warning",
    ]
    generated_outputs = merged["resolved_data_driven_contract"]["products"][
        "nonprompt"
    ]["generated_outputs"]
    assert {
        output["year"]: output["source_contributors"]
        for output in generated_outputs.values()
    } == {
        "2022": {
            "data": ["data2022"],
            "prompt_mc": ["TTto2L2Nu_central2022"],
        },
        "2022EE": {
            "data": ["data2022EE"],
            "prompt_mc": ["TTto2L2Nu_central2022EE"],
        },
    }

    product_mismatch = copy.deepcopy(second)
    product_mismatch["requested_data_driven_products"]["products"]["nonprompt"][
        "enabled"
    ] = False
    with pytest.raises(
        histogram_artifact_error,
        match="identical requested data-driven product policy",
    ):
        merge_histogram_sidecars([first, product_mismatch])

    source_mismatch = copy.deepcopy(second)
    source_mismatch["requested_data_driven_products"]["source"] = "explicit"
    with pytest.raises(
        histogram_artifact_error,
        match="identical requested data-driven product policy",
    ):
        merge_histogram_sidecars([first, source_mismatch])


def test_explicit_nonprompt_only_contract_suppresses_unrequested_flips(
    tmp_path, policy
):
    input_path = tmp_path / "nonprompt_only_processor.pkl.gz"
    output_path = tmp_path / "nonprompt_only.pkl.gz"
    _write_processor(
        input_path,
        policy,
        products_block={
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
    )
    run_data_driven.main(
        [
            "--input-pkl",
            str(input_path),
            "--output-pkl",
            str(output_path),
            "--quiet",
        ]
    )
    output = get_hist_from_pkl(str(output_path))
    assert _processes(output[scalar_nominal_key("njets")]) == [
        "TTTo2L2Nu_centralUL18",
        "nonpromptUL18",
    ]
    assert "flipsUL18" not in _processes(output["njets_sumw2"])
    sidecar = read_histogram_sidecar(output_path)
    assert sidecar["transformation_contract"]["families"]["njets"][
        "generated_flips_processes"
    ] == []


def test_unrequested_flips_product_is_rejected_before_transformation(tmp_path, policy):
    input_path = tmp_path / "nonprompt_only_processor.pkl.gz"
    _write_processor(
        input_path,
        policy,
        products_block={
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
    )
    output_path = tmp_path / "unrequested_flips.pkl.gz"
    with pytest.raises(
        ValueError,
        match="product 'flips' was not requested.*Regenerate the processor PKL",
    ):
        run_data_driven.main(
            [
                "--input-pkl",
                str(input_path),
                "--output-pkl",
                str(output_path),
                "--only-flips",
                "--quiet",
            ]
        )
    assert not output_path.exists()
    assert not metadata_sidecar_path(output_path).exists()


def _write_raw(path, payload):
    with gzip.open(path, "wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _processes(histogram):
    return sorted(str(process) for process in histogram.axes["process"])


def _total_for_process(histogram, process):
    selected = histogram.integrate("process", process)
    values = selected.view(flow=True, as_dict=True)
    return sum(float(np.asarray(value).sum()) for value in values.values())


def test_metadata_sidecar_path_preserves_full_suffixes(tmp_path, monkeypatch):
    assert metadata_sidecar_path("output.pkl") == Path("output.pkl.metadata.json")
    assert metadata_sidecar_path("output.pkl.gz") == Path(
        "output.pkl.gz.metadata.json"
    )
    absolute = tmp_path / "nested" / "output.pkl.gz"
    assert metadata_sidecar_path(absolute) == Path(f"{absolute}.metadata.json")
    monkeypatch.chdir(tmp_path)
    assert metadata_sidecar_path(Path("relative.pkl")) == Path(
        "relative.pkl.metadata.json"
    )


def test_processor_artifact_is_automatic_self_describing_and_identity_bound(
    tmp_path, policy
):
    path = tmp_path / "processor.pkl.gz"
    sidecar = _write_processor(path, policy)
    assert path.is_file()
    assert metadata_sidecar_path(path).is_file()
    reopened = read_histogram_sidecar(path)
    assert reopened == sidecar
    assert sidecar["metadata_schema_version"] == 2
    assert sidecar["artifact"]["artifact_kind"] == "processor_output"
    assert sidecar["artifact"]["merged"] is False
    assert sidecar["artifact"]["pkl_basename"] == path.name
    assert sidecar["artifact"]["pkl_size_bytes"] == path.stat().st_size
    assert sidecar["lineage"] == {"inputs": []}
    family = sidecar["sumw2_content_manifest"]["families"]["njets"]
    assert family["scalar_nominal_processes"] == [
        "TTTo2L2Nu_centralUL18",
        "dataUL18",
    ]
    assert family["eft_nominal_processes"] == ["WWTo2L2Nu_centralUL18"]
    assert family["sumw2_processes"] == [
        "TTTo2L2Nu_centralUL18",
        "WWTo2L2Nu_centralUL18",
        "dataUL18",
    ]
    assert family["required_sumw2_processes"] == family["sumw2_processes"]
    assert validate_histogram_artifact(path)["metadata"] == sidecar


def test_atomic_pair_does_not_publish_pkl_when_sidecar_write_fails(
    tmp_path, policy, monkeypatch
):
    path = tmp_path / "failed.pkl.gz"

    def fail_sidecar_write(*_args, **_kwargs):
        raise OSError("synthetic metadata write failure")

    monkeypatch.setattr(histogram_artifact, "_write_json", fail_sidecar_write)
    with pytest.raises(OSError, match="metadata write failure"):
        write_histogram_artifact(
            path,
            histograms=_processor_payload(),
            artifact_kind="processor_output",
            sumw2_storage_provenance=policy.to_provenance(),
            production_sample_contract=_certify_profile(
                policy,
                _POLICY_SAMPLES,
            ),
        )
    assert not path.exists()
    assert not metadata_sidecar_path(path).exists()


def test_schema_v2_missing_sidecar_error_is_actionable(tmp_path):
    path = tmp_path / "split.pkl.gz"
    _write_raw(path, _processor_payload())
    with pytest.raises(histogram_artifact_error) as error_info:
        validate_histogram_artifact(path)
    message = str(error_info.value)
    assert str(path) in message
    assert str(metadata_sidecar_path(path)) in message
    assert "detected_split_sibling_keys" in message
    assert "run_analysis" in message
    assert "run_data_driven" in message
    assert "merged-cache writer" in message
    assert "do not supply a sidecar path manually" in message


def test_nonprompt_persisted_flow_reopens_and_preserves_lineage(tmp_path, policy):
    input_path = tmp_path / "processor.pkl.gz"
    output_path = tmp_path / "nonprompt.pkl.gz"
    input_sidecar = _write_processor(input_path, policy)
    run_data_driven.main(
        [
            "--input-pkl",
            str(input_path),
            "--output-pkl",
            str(output_path),
            "--quiet",
        ]
    )
    output_sidecar = read_histogram_sidecar(output_path)
    assert output_sidecar["artifact"]["artifact_kind"] == "nonprompt_output"
    assert output_sidecar["sumw2_storage_provenance"] == input_sidecar[
        "sumw2_storage_provenance"
    ]
    assert output_sidecar["lineage"]["inputs"] == [
        lineage_input_from_sidecar(input_sidecar)
    ]
    merged, report = load_and_merge_histogram_pkls([str(output_path)])
    assert report["artifact_kind"] == "nonprompt_output"
    scalar_processes = _processes(merged[scalar_nominal_key("njets")])
    assert scalar_processes == [
        "TTTo2L2Nu_centralUL18",
        "flipsUL18",
        "nonpromptUL18",
    ]
    assert _processes(merged[eft_nominal_key("njets")]) == [
        "WWTo2L2Nu_centralUL18"
    ]
    assert _processes(merged["njets_sumw2"]) == [
        "TTTo2L2Nu_centralUL18",
        "WWTo2L2Nu_centralUL18",
        "flipsUL18",
        "nonpromptUL18",
    ]
    scalar_view = evaluate_nominal_at_wc(merged, "njets", {})
    assert "nonpromptUL18" in scalar_view.axes["process"]
    datacard_view = materialize_legacy_histogram_dict(
        merged,
        runtime_families=("njets",),
        require_companions=("njets",),
    )
    assert tuple(datacard_view) == ("njets", "njets_sumw2")


def test_ptll_nonprompt_flow_preserves_family_companion_and_provenance(tmp_path):
    input_path = tmp_path / "ptll_processor.pkl.gz"
    output_path = tmp_path / "ptll_nonprompt.pkl.gz"
    input_sidecar = _write_disjoint_source_processor(
        input_path,
        process_prefix="",
        year="UL18",
        runtime_families=("ptll",),
        contribution=2.0,
        prompt_contribution=1.0,
    )

    run_data_driven.main(
        [
            "--input-pkl",
            str(input_path),
            "--output-pkl",
            str(output_path),
            "--quiet",
        ]
    )

    output_sidecar = read_histogram_sidecar(output_path)
    merged, report = load_and_merge_histogram_pkls([str(output_path)])
    assert input_sidecar["sumw2_storage_provenance"][
        "runtime_histogram_families"
    ] == ["ptll"]
    assert output_sidecar["sumw2_storage_provenance"][
        "runtime_histogram_families"
    ] == ["ptll"]
    assert set(output_sidecar["sumw2_content_manifest"]["families"]) == {"ptll"}
    assert report["runtime_histogram_families"] == ["ptll"]
    assert scalar_nominal_key("ptll") in merged
    assert sumw2_key("ptll") in merged
    assert "nonpromptUL18" in _processes(merged[scalar_nominal_key("ptll")])
    assert "nonpromptUL18" in _processes(merged[sumw2_key("ptll")])


def test_direct_data_driven_writer_discovers_and_writes_sidecars(tmp_path, policy):
    input_path = tmp_path / "processor.pkl.gz"
    output_path = tmp_path / "direct_nonprompt.pkl.gz"
    _write_processor(input_path, policy)
    producer = DataDrivenProducer(
        str(input_path),
        str(output_path),
        iterator_mode=True,
    )
    producer.dumpToPickle()
    sidecar = read_histogram_sidecar(output_path)
    assert sidecar["artifact"]["artifact_kind"] == "nonprompt_output"
    merged, report = load_and_merge_histogram_pkls([str(output_path)])
    assert report["artifact_kind"] == "nonprompt_output"
    assert "nonpromptUL18" in merged[scalar_nominal_key("njets")].axes["process"]


def test_flips_persisted_flow_has_separate_stage_contract(tmp_path, policy):
    input_path = tmp_path / "processor.pkl.gz"
    output_path = tmp_path / "flips.pkl.gz"
    _write_processor(input_path, policy)
    run_data_driven.main(
        [
            "--input-pkl",
            str(input_path),
            "--output-pkl",
            str(output_path),
            "--only-flips",
            "--quiet",
        ]
    )
    sidecar = read_histogram_sidecar(output_path)
    assert sidecar["artifact"]["artifact_kind"] == "flips_output"
    merged, report = load_and_merge_histogram_pkls([str(output_path)])
    assert report["artifact_kind"] == "flips_output"
    assert _processes(merged[scalar_nominal_key("njets")]) == ["flipsUL18"]
    assert _processes(merged["njets_sumw2"]) == [
        "WWTo2L2Nu_centralUL18",
        "flipsUL18",
    ]
    assert _processes(merged[eft_nominal_key("njets")]) == [
        "WWTo2L2Nu_centralUL18"
    ]
    assert sum(
        float(np.asarray(values).sum())
        for values in merged[eft_nominal_key("njets")].eval({}).values()
    ) == pytest.approx(6.25)
    family = sidecar["sumw2_content_manifest"]["families"]["njets"]
    assert family["sumw2_processes"] == ["WWTo2L2Nu_centralUL18", "flipsUL18"]
    assert family["required_sumw2_processes"] == [
        "WWTo2L2Nu_centralUL18",
        "flipsUL18",
    ]
    scalar_view = evaluate_nominal_at_wc(merged, "njets", {})
    assert sorted(str(process) for process in scalar_view.axes["process"]) == [
        "WWTo2L2Nu_centralUL18",
        "flipsUL18",
    ]
    datacard_view = materialize_legacy_histogram_dict(
        merged,
        runtime_families=("njets",),
        require_companions=("njets",),
    )
    assert tuple(datacard_view) == ("njets", "njets_sumw2")


def _transformed_payload(artifact_kind, *, channel="3l"):
    processes = (
        ("flipsUL18",)
        if artifact_kind == "flips_output"
        else ("nonpromptUL18", "flipsUL18")
    )
    return {
        scalar_nominal_key("njets"): _fill_sparse(
            "njets",
            tuple((process, "isSR_3l", 2.0) for process in processes),
            channel=channel,
        ),
        "njets_sumw2": _fill_sparse(
            "njets_sumw2",
            tuple((process, "isSR_3l", 4.0) for process in processes),
            channel=channel,
        ),
    }


def _transformed_context(source_sidecar, artifact_kind):
    source = source_sidecar["sumw2_content_manifest"]["families"]["njets"]
    return _producer_transformation_context(
        {
            "eft_prompt_projection": {
                "mode": "sm_point",
                "required_processes": [],
                "generated_nonprompt_eft_dependence": False,
            },
            "families": {
                "njets": {
                    "source_scalar_processes": source[
                        "scalar_nominal_processes"
                    ],
                    "source_eft_processes": source[
                        "eft_nominal_processes"
                    ],
                    "retained_scalar_processes": [],
                    "retained_eft_processes": [],
                    "generated_nonprompt_processes": (
                        ["nonpromptUL18"]
                        if artifact_kind == "nonprompt_output"
                        else []
                    ),
                    "generated_flips_processes": ["flipsUL18"],
                    "source_application_regions": [
                        "isAR_2lSS_OS",
                        "isAR_3l",
                        "isSR_3l",
                    ],
                    "applicable_products": {
                        "nonprompt": True,
                        "flips": True,
                    },
                }
            },
        }
    )


@pytest.mark.parametrize(
    "artifact_kind",
    ["processor_output", "nonprompt_output", "flips_output"],
)
def test_compatible_stage_merges_regenerate_deterministic_sidecar(
    tmp_path, policy, artifact_kind
):
    source_path = tmp_path / f"{artifact_kind}_source.pkl.gz"
    source_sidecar = _write_processor(source_path, policy)
    paths = []
    for index in range(2):
        path = tmp_path / f"{artifact_kind}_{index}.pkl.gz"
        channel = ("3l_onZ_2b_4j", "2lss_m_4j")[index]
        if artifact_kind == "processor_output":
            _write_processor(
                path,
                policy,
                source_payload=_processor_payload(channel=channel),
            )
        else:
            write_histogram_artifact(
                path,
                histograms=_transformed_payload(
                    artifact_kind, channel=channel
                ),
                artifact_kind=artifact_kind,
                sumw2_storage_provenance=policy.to_provenance(),
                lineage_inputs=[lineage_input_from_sidecar(source_sidecar)],
                input_sidecar=source_sidecar,
                transformation_context=_transformed_context(
                    source_sidecar, artifact_kind
                ),
            )
        paths.append(str(path))
    merged, report = load_and_merge_histogram_pkls(paths)
    cached_path = make_cards._cache_merged_histograms(
        merged,
        f"merged_{artifact_kind}",
        str(tmp_path),
        report,
    )
    merged_sidecar = read_histogram_sidecar(cached_path)
    assert merged_sidecar["artifact"]["artifact_kind"] == artifact_kind
    assert merged_sidecar["artifact"]["merged"] is True
    assert len(merged_sidecar["lineage"]["inputs"]) == 2
    reopened, reopened_report = load_and_merge_histogram_pkls([cached_path])
    assert tuple(reopened) == tuple(merged)
    assert reopened_report["artifact_kind"] == artifact_kind


@pytest.mark.parametrize(
    "layout_name,runtime_families",
    (
        (
            "2l_2lss_1tau_2los_1tau_4l",
            ("njets", "lj0pt", "ptz", "ptz_wtau", "lt"),
        ),
        ("3l_fwd", ("njets", "lj0pt", "ptz", "lt")),
        ("3l_m_offz_3l_p_offz", ("njets", "lj0pt", "ptz", "lt")),
        ("3l_onz_tau", ("njets", "lj0pt", "ptz", "lt")),
    ),
)
def test_cross_run_source_universes_are_rejected_by_loader(
    tmp_path,
    layout_name,
    runtime_families,
):
    run2_path = tmp_path / f"{layout_name}_run2.pkl.gz"
    run3_path = tmp_path / f"{layout_name}_run3.pkl.gz"
    _write_disjoint_source_processor(
        run2_path,
        process_prefix="run2",
        year="UL18",
        runtime_families=runtime_families,
        contribution=1.0,
    )
    _write_disjoint_source_processor(
        run3_path,
        process_prefix="run3",
        year="2022",
        runtime_families=runtime_families,
        contribution=2.0,
    )

    with pytest.raises(
        RuntimeError,
        match=r"Histogram-level Run 2 \+ Run 3 composition is unsupported",
    ):
        load_and_merge_histogram_pkls([str(run2_path), str(run3_path)])


def test_plotting_merged_cache_writer_preserves_artifact_stage(tmp_path, policy):
    paths = []
    for index in range(2):
        path = tmp_path / f"plot_input_{index}.pkl.gz"
        channel = ("3l_onZ_2b_4j", "2lss_m_4j")[index]
        _write_processor(
            path,
            policy,
            source_payload=_processor_payload(channel=channel),
        )
        paths.append(str(path))
    merged, report = load_and_merge_histogram_pkls(paths)
    cached_path = make_cr_and_sr_plots._cache_merged_histograms(
        merged,
        "plot_cache",
        str(tmp_path),
        report,
    )
    sidecar = read_histogram_sidecar(cached_path)
    assert sidecar["artifact"]["artifact_kind"] == "processor_output"
    assert sidecar["artifact"]["merged"] is True
    assert len(sidecar["lineage"]["inputs"]) == 2


def test_private_and_central_profile_transformed_artifacts_do_not_merge(tmp_path):
    transformed_sidecars = []
    for profile, signal_process in (
        ("production", "tllq_privateUL18"),
        ("production_central", "tZq_centralUL18"),
    ):
        samples = {
            "data_dataset": {
                "histAxisName": "dataUL18",
                "isData": True,
                "WCnames": [],
            },
            "prompt_dataset": {
                "histAxisName": "TTTo2L2Nu_centralUL18",
                "isData": False,
                "WCnames": [],
            },
            "signal_dataset": {
                "histAxisName": signal_process,
                "isData": False,
                "WCnames": [],
            },
        }
        policy = resolve_sumw2_storage_policy(
            {
                "mode": profile,
                "rules": [
                    {
                        "process_names": [
                            "dataUL18",
                            "TTTo2L2Nu_centralUL18",
                            signal_process,
                        ],
                        "variables": ["njets"],
                    }
                ],
            },
            samples=samples,
            runtime_families=("njets",),
            axes_info=axes_info,
            axes_info_2d=axes_info_2d,
            sumw2_storage_present=True,
        )
        products = resolve_data_driven_products(
            {
                "nonprompt": {
                    "enabled": True,
                    "source_contributors": {
                        "data": {"process_names": ["dataUL18"]},
                        "prompt_mc": {
                            "process_names": [
                                "TTTo2L2Nu_centralUL18",
                                signal_process,
                            ]
                        },
                    },
                },
                "flips": {
                    "enabled": True,
                    "source_contributors": {
                        "data": {"process_names": ["dataUL18"]}
                    },
                },
            },
            data_driven_products_present=True,
            legacy_do_np=False,
            samples=samples,
            runtime_families=("njets",),
            metadata_path=f"{profile}.yml",
        )
        requested, contract = certify_data_driven_preflight(products, policy)
        processor_path = tmp_path / f"{profile}_processor.pkl.gz"
        source_sidecar = write_histogram_artifact(
            processor_path,
            histograms={
                scalar_nominal_key("njets"): _fill_sparse(
                    "njets",
                        (
                            ("dataUL18", "isAR_3l", 10.0),
                            ("TTTo2L2Nu_centralUL18", "isAR_3l", 3.0),
                            (signal_process, "isAR_3l", 1.0),
                            ("dataUL18", "isAR_2lSS_OS", 4.0),
                        ),
                ),
                "njets_sumw2": _fill_sparse(
                    "njets_sumw2",
                        (
                            ("dataUL18", "isAR_3l", 100.0),
                            ("TTTo2L2Nu_centralUL18", "isAR_3l", 9.0),
                            (signal_process, "isAR_3l", 1.0),
                            ("dataUL18", "isAR_2lSS_OS", 16.0),
                        ),
                ),
            },
            artifact_kind="processor_output",
            sumw2_storage_provenance=policy.to_provenance(),
            production_sample_contract=_certify_profile(policy, samples, products),
            requested_data_driven_products=requested,
            resolved_data_driven_contract=contract,
        )
        producer = DataDrivenProducer(str(processor_path), "")
        transformed_path = tmp_path / f"{profile}_nonprompt.pkl.gz"
        write_histogram_artifact(
            transformed_path,
            histograms=producer.getDataDrivenHistogram(),
            artifact_kind="nonprompt_output",
            sumw2_storage_provenance=policy.to_provenance(),
            lineage_inputs=[lineage_input_from_sidecar(source_sidecar)],
            input_sidecar=source_sidecar,
            transformation_context=producer.get_transformation_context(
                "nonprompt_output"
            ),
        )
        transformed_sidecar = read_histogram_sidecar(transformed_path)
        assert transformed_sidecar["sumw2_storage_provenance"][
            "signal_sample_profile"
        ] == ("private" if profile == "production" else "central")
        assert transformed_sidecar["production_sample_contract"] == source_sidecar[
            "production_sample_contract"
        ]
        assert transformed_sidecar["sumw2_content_manifest"]["families"][
            "njets"
        ]["required_sumw2_processes"] == ["flipsUL18", "nonpromptUL18"]
        transformed_sidecars.append(transformed_sidecar)

    with pytest.raises(
        RuntimeError,
        match="source-allocation provenance|sample profile contracts",
    ):
        merge_histogram_sidecars(transformed_sidecars)


def test_incompatible_artifact_stage_merges_are_rejected(tmp_path, policy):
    processor_path = tmp_path / "processor.pkl.gz"
    processor_sidecar = _write_processor(processor_path, policy)
    nonprompt_path = tmp_path / "nonprompt.pkl.gz"
    flips_path = tmp_path / "flips.pkl.gz"
    for path, artifact_kind in (
        (nonprompt_path, "nonprompt_output"),
        (flips_path, "flips_output"),
    ):
        write_histogram_artifact(
            path,
            histograms=_transformed_payload(artifact_kind),
            artifact_kind=artifact_kind,
            sumw2_storage_provenance=policy.to_provenance(),
            lineage_inputs=[lineage_input_from_sidecar(processor_sidecar)],
            input_sidecar=processor_sidecar,
            transformation_context=_transformed_context(
                processor_sidecar, artifact_kind
            ),
        )
    with pytest.raises(RuntimeError, match="incompatible histogram artifact kinds"):
        load_and_merge_histogram_pkls(
            [str(processor_path), str(nonprompt_path)],
        )
    with pytest.raises(RuntimeError, match="incompatible histogram artifact kinds"):
        load_and_merge_histogram_pkls(
            [str(nonprompt_path), str(flips_path)],
        )


@pytest.mark.parametrize("artifact_kind", ["nonprompt_output", "flips_output"])
def test_transformed_missing_and_unexpected_companions_are_actionable(
    tmp_path, policy, artifact_kind
):
    source_path = tmp_path / "processor.pkl.gz"
    source_sidecar = _write_processor(source_path, policy)
    path = tmp_path / f"{artifact_kind}.pkl.gz"
    payload = _transformed_payload(artifact_kind)
    expected_process = (
        "flipsUL18" if artifact_kind == "flips_output" else "nonpromptUL18"
    )
    write_histogram_artifact(
        path,
        histograms=payload,
        artifact_kind=artifact_kind,
        sumw2_storage_provenance=policy.to_provenance(),
        lineage_inputs=[lineage_input_from_sidecar(source_sidecar)],
        input_sidecar=source_sidecar,
        transformation_context=_transformed_context(source_sidecar, artifact_kind),
    )

    missing_payload = {scalar_nominal_key("njets"): payload[scalar_nominal_key("njets")]}
    _write_raw(path, missing_payload)
    with pytest.raises(histogram_artifact_error) as missing_error:
        validate_histogram_artifact(path)
    missing_message = str(missing_error.value)
    assert "pkl_path=" in missing_message
    assert "sidecar_path=" in missing_message
    assert f"artifact_kind={artifact_kind}" in missing_message
    assert "family=njets" in missing_message
    assert "missing_required_companions=" in missing_message
    assert expected_process in missing_message
    assert "run_data_driven" in missing_message

    write_histogram_artifact(
        path,
        histograms=payload,
        artifact_kind=artifact_kind,
        sumw2_storage_provenance=policy.to_provenance(),
        lineage_inputs=[lineage_input_from_sidecar(source_sidecar)],
        input_sidecar=source_sidecar,
        transformation_context=_transformed_context(source_sidecar, artifact_kind),
    )
    unexpected_payload = copy.deepcopy(payload)
    unexpected_payload["njets_sumw2"].fill(
        process="unexpectedUL18",
        channel="3l",
        systematic="nominal",
        appl="isSR_3l",
        njets_sumw2=np.asarray([0.5]),
        weight=np.asarray([1.0]),
    )
    _write_raw(path, unexpected_payload)
    with pytest.raises(histogram_artifact_error) as unexpected_error:
        validate_histogram_artifact(path)
    unexpected_message = str(unexpected_error.value)
    assert "unexpected_companions=['unexpectedUL18']" in unexpected_message
    assert "family=njets" in unexpected_message

    producer_path = tmp_path / f"producer_rejects_{artifact_kind}.pkl.gz"
    with pytest.raises(
        histogram_artifact_error,
        match="unexpected_companions=.*unexpectedUL18",
    ):
        write_histogram_artifact(
            producer_path,
            histograms=unexpected_payload,
            artifact_kind=artifact_kind,
            sumw2_storage_provenance=policy.to_provenance(),
            lineage_inputs=[lineage_input_from_sidecar(source_sidecar)],
            input_sidecar=source_sidecar,
            transformation_context=_transformed_context(
                source_sidecar, artifact_kind
            ),
        )
    assert not producer_path.exists()
    assert not metadata_sidecar_path(producer_path).exists()


@pytest.mark.parametrize(
    "missing_process",
    [
        "WWTo2L2Nu_centralUL18",
        "TTTo2L2Nu_centralUL18",
        "nonpromptUL18",
        "flipsUL18",
    ],
)
def test_independent_nonprompt_contract_rejects_partial_companion_loss_before_publish(
    tmp_path, policy, missing_process
):
    source_path = tmp_path / "processor.pkl.gz"
    output_path = tmp_path / f"missing_{missing_process}.pkl.gz"
    source_sidecar = _write_processor(source_path, policy)
    producer = DataDrivenProducer(
        str(source_path),
        "",
        artifact_kind="nonprompt_output",
    )
    transformed = copy.deepcopy(producer.getDataDrivenHistogram())
    context = producer.get_transformation_context("nonprompt_output")
    contract, required = derive_transformed_required_sumw2_processes(
        input_sidecar=source_sidecar,
        transformation_context=context,
        artifact_kind="nonprompt_output",
        transformed_histograms=transformed,
    )
    assert required["njets"] == [
        "TTTo2L2Nu_centralUL18",
        "WWTo2L2Nu_centralUL18",
        "flipsUL18",
        "nonpromptUL18",
    ]
    assert contract["families"]["njets"]["consumed_source_processes"] == [
        "dataUL18"
    ]

    transformed["njets_sumw2"] = transformed["njets_sumw2"].remove(
        "process", [missing_process]
    )
    _, after_loss = derive_transformed_required_sumw2_processes(
        input_sidecar=source_sidecar,
        transformation_context=context,
        artifact_kind="nonprompt_output",
    )
    assert after_loss == required
    with pytest.raises(
        histogram_artifact_error,
        match=f"requires sumw2 processes absent.*{missing_process}",
    ):
        write_histogram_artifact(
            output_path,
            histograms=transformed,
            artifact_kind="nonprompt_output",
            sumw2_storage_provenance=policy.to_provenance(),
            lineage_inputs=[lineage_input_from_sidecar(source_sidecar)],
            input_sidecar=source_sidecar,
            transformation_context=context,
        )
    assert not output_path.exists()
    assert not metadata_sidecar_path(output_path).exists()


def test_pre_product_contract_cannot_authorize_new_transformed_requirements(tmp_path):
    selective_policy = resolve_sumw2_storage_policy(
        {
            "mode": "full_custom",
            "rules": [
                {
                    "process_names": ["WWTo2L2Nu_centralUL18"],
                    "variables": ["njets"],
                }
            ],
        },
        samples={
            "data_dataset": {
                "histAxisName": "dataUL18",
                "isData": True,
                "WCnames": [],
            },
            "prompt_dataset": {
                "histAxisName": "TTTo2L2Nu_centralUL18",
                "isData": False,
                "WCnames": [],
            },
            "signal_dataset": {
                "histAxisName": "WWTo2L2Nu_centralUL18",
                "isData": False,
                "WCnames": ["ctG"],
            },
        },
        runtime_families=("njets",),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    source_payload = _processor_payload()
    source_payload["njets_sumw2"] = _fill_sparse(
        "njets_sumw2",
        (("WWTo2L2Nu_centralUL18", "isSR_3l", 25.0),),
    )
    source_path = tmp_path / "selective_processor.pkl.gz"
    source_sidecar = write_histogram_artifact(
        source_path,
        histograms=source_payload,
        artifact_kind="processor_output",
        sumw2_storage_provenance=selective_policy.to_provenance(),
        production_sample_contract=_certify_profile(
            selective_policy,
            _POLICY_SAMPLES,
        ),
    )
    source_family = source_sidecar["sumw2_content_manifest"]["families"]["njets"]
    context = {
        "families": {
            "njets": {
                "source_scalar_processes": source_family[
                    "scalar_nominal_processes"
                ],
                "source_eft_processes": source_family["eft_nominal_processes"],
                "retained_scalar_processes": source_family[
                    "scalar_nominal_processes"
                ],
                "retained_eft_processes": source_family[
                    "eft_nominal_processes"
                ],
                "generated_nonprompt_processes": [],
                "generated_flips_processes": [],
            }
        }
    }
    with pytest.raises(
        histogram_artifact_error,
        match="lacks the requested data-driven product contract.*Regenerate.*run_analysis",
    ):
        derive_transformed_required_sumw2_processes(
            input_sidecar=source_sidecar,
            transformation_context=context,
            artifact_kind="nonprompt_output",
        )


def test_flips_contract_requires_every_generated_year_label(tmp_path):
    multi_year_samples = {
        "data_ul18": {
            "histAxisName": "dataUL18",
            "isData": True,
            "WCnames": [],
        },
        "data_2022": {
            "histAxisName": "data2022",
            "isData": True,
            "WCnames": [],
        },
    }
    multi_year_policy = resolve_sumw2_storage_policy(
        {"mode": "full_diagnostics"},
        samples=multi_year_samples,
        runtime_families=("njets",),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    entries = (
        ("dataUL18", "isAR_2lSS_OS", 4.0),
        ("data2022", "isAR_2lSS_OS", 5.0),
    )
    source_payload = {
        scalar_nominal_key("njets"): _fill_sparse("njets", entries),
        "njets_sumw2": _fill_sparse("njets_sumw2", entries),
    }
    source_path = tmp_path / "multi_year_processor.pkl.gz"
    output_path = tmp_path / "multi_year_flips.pkl.gz"
    products = resolve_data_driven_products(
        {
            "flips": {
                "enabled": True,
                "source_contributors": {
                    "data": {"process_prefixes": ["data"]},
                },
            }
        },
        data_driven_products_present=True,
        legacy_do_np=False,
        samples=multi_year_samples,
        runtime_families=("njets",),
        metadata_path="test_options.yml",
    )
    requested, contract = certify_data_driven_preflight(
        products,
        multi_year_policy,
    )
    source_sidecar = write_histogram_artifact(
        source_path,
        histograms=source_payload,
        artifact_kind="processor_output",
        sumw2_storage_provenance=multi_year_policy.to_provenance(),
        production_sample_contract=_certify_profile(
            multi_year_policy,
            multi_year_samples,
            products,
        ),
        requested_data_driven_products=requested,
        resolved_data_driven_contract=contract,
    )
    run_data_driven.main(
        [
            "--input-pkl",
            str(source_path),
            "--output-pkl",
            str(output_path),
            "--only-flips",
            "--quiet",
        ]
    )
    sidecar = read_histogram_sidecar(output_path)
    family = sidecar["sumw2_content_manifest"]["families"]["njets"]
    assert family["required_sumw2_processes"] == ["flips2022", "flipsUL18"]
    output = get_hist_from_pkl(str(output_path))
    producer = DataDrivenProducer(
        str(source_path),
        "",
        artifact_kind="flips_output",
    )
    context = producer.get_transformation_context("flips_output")
    output["njets_sumw2"] = output["njets_sumw2"].remove(
        "process", ["flipsUL18"]
    )
    rejected_path = tmp_path / "partial_flips.pkl.gz"
    with pytest.raises(
        histogram_artifact_error,
        match="requires sumw2 processes absent.*flipsUL18",
    ):
        write_histogram_artifact(
            rejected_path,
            histograms=output,
            artifact_kind="flips_output",
            sumw2_storage_provenance=multi_year_policy.to_provenance(),
            lineage_inputs=[lineage_input_from_sidecar(source_sidecar)],
            input_sidecar=source_sidecar,
            transformation_context=context,
        )


def test_merged_flips_contract_unions_independently_validated_requirements(
    tmp_path, policy
):
    source_path = tmp_path / "processor.pkl.gz"
    source_sidecar = _write_processor(source_path, policy)
    source_family = source_sidecar["sumw2_content_manifest"]["families"]["njets"]
    sidecars = []
    for index in range(2):
        process = "flipsUL18"
        path = tmp_path / f"flips_{index}.pkl.gz"
        context = _transformed_context(source_sidecar, "flips_output")
        payload = {
            scalar_nominal_key("njets"): _fill_sparse(
                "njets", ((process, "isSR_3l", 2.0),)
            ),
            "njets_sumw2": _fill_sparse(
                "njets_sumw2", ((process, "isSR_3l", 4.0),)
            ),
        }
        write_histogram_artifact(
            path,
            histograms=payload,
            artifact_kind="flips_output",
            sumw2_storage_provenance=policy.to_provenance(),
            lineage_inputs=[lineage_input_from_sidecar(source_sidecar)],
            input_sidecar=source_sidecar,
            transformation_context=context,
        )
        sidecars.append(read_histogram_sidecar(path))
    report = merge_histogram_sidecars(sidecars)
    assert report["required_sumw2_processes"] == {"njets": ["flipsUL18"]}
    assert report["transformation_contract"]["families"]["njets"][
        "generated_flips_processes"
    ] == ["flipsUL18"]
    assert report["resolved_data_driven_contract"] == source_sidecar[
        "resolved_data_driven_contract"
    ]


def test_transformed_required_tampering_cannot_authorize_partial_payload(tmp_path, policy):
    source_path = tmp_path / "processor.pkl.gz"
    output_path = tmp_path / "nonprompt.pkl.gz"
    _write_processor(source_path, policy)
    run_data_driven.main(
        ["--input-pkl", str(source_path), "--output-pkl", str(output_path), "--quiet"]
    )
    payload = get_hist_from_pkl(str(output_path))
    payload["njets_sumw2"] = payload["njets_sumw2"].remove(
        "process", ["WWTo2L2Nu_centralUL18"]
    )
    _write_raw(output_path, payload)
    sidecar_path = metadata_sidecar_path(output_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    family = sidecar["sumw2_content_manifest"]["families"]["njets"]
    family["sumw2_processes"].remove("WWTo2L2Nu_centralUL18")
    family["required_sumw2_processes"].remove("WWTo2L2Nu_centralUL18")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(
        histogram_artifact_error,
        match="required_sumw2_processes disagree.*WWTo2L2Nu_centralUL18",
    ):
        validate_histogram_artifact(output_path)


def test_pre_contract_transformed_metadata_requires_regeneration(tmp_path, policy):
    source_path = tmp_path / "processor.pkl.gz"
    output_path = tmp_path / "nonprompt.pkl.gz"
    _write_processor(source_path, policy)
    run_data_driven.main(
        ["--input-pkl", str(source_path), "--output-pkl", str(output_path), "--quiet"]
    )
    sidecar_path = metadata_sidecar_path(output_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar.pop("transformation_contract")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(
        histogram_artifact_error,
        match="predates independent transformed-companion validation.*run_data_driven",
    ):
        validate_histogram_artifact(output_path)


def test_original_transformed_publication_preserves_source_provenance(tmp_path, policy):
    source_path = tmp_path / "processor.pkl.gz"
    source_sidecar = _write_processor(source_path, policy)
    producer = DataDrivenProducer(str(source_path), "")
    altered_provenance = copy.deepcopy(policy.to_provenance())
    altered_provenance["warnings"] = ["not immutable"]
    with pytest.raises(
        histogram_artifact_error,
        match="preserve.*sumw2_storage_provenance unchanged",
    ):
        write_histogram_artifact(
            tmp_path / "altered.pkl.gz",
            histograms=producer.getDataDrivenHistogram(),
            artifact_kind="nonprompt_output",
            sumw2_storage_provenance=altered_provenance,
            lineage_inputs=[lineage_input_from_sidecar(source_sidecar)],
            input_sidecar=source_sidecar,
            transformation_context=producer.get_transformation_context(
                "nonprompt_output"
            ),
        )


def test_sidecar_free_two_dimensional_only_payload_is_structurally_ambiguous(tmp_path):
    family = "lepton_pt_vs_eta"
    dense_axes = tuple(
        hist.axis.Regular(
            axis_spec["processing"]["bins"],
            axis_spec["processing"]["start"],
            axis_spec["processing"]["stop"],
            name=axis_spec["name"],
        )
        for axis_spec in axes_info_2d[family]["axes"]
    )
    payload = {
        family: SparseHist(
            hist.axis.StrCategory([], name="process", growth=True),
            hist.axis.StrCategory([], name="channel", growth=True),
            hist.axis.StrCategory([], name="systematic", growth=True),
            hist.axis.StrCategory([], name="appl", growth=True),
            *dense_axes,
            storage="Double",
        )
    }
    path = tmp_path / "two_dimensional_only.pkl.gz"
    _write_raw(path, payload)
    assert not is_split_nominal_mapping(payload)
    assert validate_histogram_artifact(path) == {
        "schema": "legacy_uniform",
        "metadata": None,
        "legacy_metadata_present": False,
    }


@pytest.mark.parametrize(
    "tamper",
    [
        "basename",
        "size",
        "checksum",
        "unknown_kind",
        "nominal_schema",
        "nominal_layout",
        "missing_artifact_field",
        "partial_manifest",
        "wrong_processes",
        "malformed_lineage",
    ],
)
def test_metadata_tampering_is_rejected(tmp_path, policy, tamper):
    path = tmp_path / f"tamper_{tamper}.pkl.gz"
    _write_processor(path, policy)
    sidecar_path = metadata_sidecar_path(path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if tamper == "basename":
        sidecar["artifact"]["pkl_basename"] = "wrong.pkl.gz"
    elif tamper == "size":
        sidecar["artifact"]["pkl_size_bytes"] += 1
    elif tamper == "checksum":
        sidecar["artifact"]["pkl_sha256"] = "0" * 64
    elif tamper == "unknown_kind":
        sidecar["artifact"]["artifact_kind"] = "unknown"
    elif tamper == "nominal_schema":
        sidecar["artifact"]["nominal_container_schema_version"] = 999
    elif tamper == "nominal_layout":
        sidecar["artifact"]["nominal_container_layout"] = "wrong_layout"
    elif tamper == "missing_artifact_field":
        sidecar["artifact"].pop("merged")
    elif tamper == "partial_manifest":
        sidecar["sumw2_content_manifest"]["families"]["njets"].pop(
            "sumw2_processes"
        )
    elif tamper == "wrong_processes":
        sidecar["sumw2_content_manifest"]["families"]["njets"][
            "scalar_nominal_processes"
        ].append("wrongUL18")
    else:
        sidecar["lineage"] = {"inputs": [{"pkl_basename": "missing-fields"}]}
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(histogram_artifact_error):
        validate_histogram_artifact(path)


def test_legacy_uniform_without_sidecar_remains_usable(tmp_path):
    path = tmp_path / "legacy.pkl.gz"
    nominal = HistEFT(*_axes("njets"), wc_names=["ctG"], label="Events")
    nominal.fill(
        process="signalUL18",
        channel="3l",
        systematic="nominal",
        appl="isSR_3l",
        njets=np.asarray([0.5]),
        weight=np.asarray([2.0]),
        eft_coeff=np.asarray([[1.5, 2.0, 3.0]]),
    )
    companion = HistEFT(*_axes("njets_sumw2"), wc_names=["ctG"], label="Events")
    companion.fill(
        process="signalUL18",
        channel="3l",
        systematic="nominal",
        appl="isSR_3l",
        njets_sumw2=np.asarray([0.5]),
        weight=np.asarray([3.0]),
        eft_coeff=np.asarray([[1.0, 0.0, 0.0]]),
    )
    _write_raw(path, {"njets": nominal, "njets_sumw2": companion})
    with pytest.warns(UserWarning, match="legacy uniform") as warning_records:
        merged, report = load_and_merge_histogram_pkls([str(path)])
    assert len(warning_records) == 1
    assert report["schema"] == "legacy_uniform"
    scalar = evaluate_nominal_at_wc(merged, "njets", {}, schema_version=None)
    assert _processes(scalar) == ["signalUL18"]
    datacard_view = materialize_legacy_histogram_dict(
        merged,
        schema_version=None,
        require_companions=("njets",),
    )
    assert tuple(datacard_view) == ("njets", "njets_sumw2")
    assert not metadata_sidecar_path(path).exists()


def test_recognized_legacy_metadata_does_not_create_schema_v2_sidecar(
    tmp_path, policy
):
    path = tmp_path / "legacy_with_metadata.pkl.gz"
    nominal = _fill_eft((("signalUL18", "isSR_3l", 2.0),))
    companion = _fill_eft((("signalUL18", "isSR_3l", 3.0),))
    _write_raw(path, {"njets": nominal, "njets_sumw2": companion})
    legacy_metadata = {
        "metadata_version": 1,
        "input_histogram": str(path),
        "sumw2_storage_provenance": policy.to_provenance(),
        "nominal_container_schema_version": 2,
        "nominal_container_layout": "split_sibling_v1",
    }
    metadata_sidecar_path(path).write_text(
        json.dumps(legacy_metadata), encoding="utf-8"
    )
    with pytest.warns(UserWarning, match="legacy uniform"):
        merged, report = load_and_merge_histogram_pkls([str(path)])
    assert report["schema"] == "legacy_uniform"
    assert tuple(merged) == ("njets", "njets_sumw2")
    assert json.loads(metadata_sidecar_path(path).read_text(encoding="utf-8")) == (
        legacy_metadata
    )


def test_no_user_facing_sidecar_cli_option_and_shared_discovery_source():
    parser_help = run_data_driven._build_argument_parser().format_help()
    assert "--metadata-json" not in parser_help
    assert "--metadata-sidecar" not in parser_help
    assert "--sidecar" not in parser_help
    repository_root = Path(__file__).resolve().parents[1]
    consumer_sources = [
        repository_root / "analysis/topeft_run2/run_analysis.py",
        repository_root / "analysis/topeft_run2/run_data_driven.py",
        repository_root / "analysis/topeft_run2/make_cards.py",
        repository_root / "analysis/topeft_run2/make_cr_and_sr_plots.py",
        repository_root / "analysis/topeft_run2/faketau_sf_fitter.py",
        repository_root / "analysis/topeft_run2/tauFitter.py",
        repository_root / "topeft/modules/datacard_tools.py",
    ]
    for source_path in consumer_sources:
        source = source_path.read_text(encoding="utf-8")
        assert "--metadata-sidecar" not in source
        assert "--sidecar" not in source
        assert "--metadata-json" not in source
    assert "metadata_sidecar_path" in (
        repository_root / "topeft/modules/histogram_artifact.py"
    ).read_text(encoding="utf-8")
