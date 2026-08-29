from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path

import cloudpickle
import hist
import numpy as np
import pytest

from analysis.topeft_run2 import repair_histogram_process_labels as repair_module
from analysis.topeft_run2.repair_histogram_process_labels import (
    RUN2_PROCESS_LABEL_REPAIRS,
    _repair_histograms,
    _repair_typed_metadata,
    repair_artifacts,
    repair_error,
)
from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from topcoffea.modules.utils import get_hist_from_pkl
from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.dataDrivenEstimation import DataDrivenProducer
from topeft.modules.data_driven_products import (
    certify_data_driven_preflight,
    resolve_data_driven_products,
)
from topeft.modules.histogram_artifact import (
    histogram_artifact_error,
    histogram_sidecar_error,
    lineage_input_from_sidecar,
    metadata_sidecar_path,
    read_histogram_sidecar,
    validate_histogram_artifact,
    write_histogram_artifact,
)
from topeft.modules.nominal_schema import scalar_nominal_key, sumw2_key
from topeft.modules.production_sample_profile import (
    build_active_sample_universe,
    certify_production_sample_contract,
)
from topeft.modules.sumw2_policy import resolve_sumw2_storage_policy


OLD_LABELS = tuple(RUN2_PROCESS_LABEL_REPAIRS)
CANONICAL_LABELS = tuple(RUN2_PROCESS_LABEL_REPAIRS.values())
UNCHANGED_LABELS = (
    "WWW_4F_centralUL16",
    "WWZ_4F_centralUL18",
    "unrelated_centralUL17",
)
DATA_LABELS = ("dataUL16APV", "dataUL17")
_legacy_labels_by_canonical = {
    target: source for source, target in RUN2_PROCESS_LABEL_REPAIRS.items()
}


def _axes(dense_name):
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.Regular(2, 0.0, 2.0, name=dense_name),
    )


def _fill_sparse(dense_name, entries):
    output = SparseHist(*_axes(dense_name), storage="Double")
    for process, channel, weight in entries:
        output.fill(
            process=process,
            channel=channel,
            systematic="nominal",
            appl="isSR_3l",
            **{dense_name: np.asarray([0.5, 1.5])},
            weight=np.asarray([weight, weight + 0.25]),
        )
    return output


def _fill_eft(process):
    output = HistEFT(*_axes("njets"), wc_names=["ctG"], label="Events")
    output.fill(
        process=process,
        channel="3l",
        systematic="nominal",
        appl="isSR_3l",
        njets=np.asarray([0.5]),
        weight=np.asarray([2.0]),
        eft_coeff=np.asarray([[1.0, 2.0, 3.0]]),
    )
    return output


def _samples(include_collision=False):
    labels = (*OLD_LABELS, *UNCHANGED_LABELS)
    if include_collision:
        labels = (*labels, CANONICAL_LABELS[0])
    samples = {
        f"sample_{index}": {
            "histAxisName": label,
            "isData": False,
            "WCnames": [],
        }
        for index, label in enumerate(labels)
    }
    for index, label in enumerate(DATA_LABELS):
        samples[f"data_{index}"] = {
            "histAxisName": label,
            "isData": True,
            "WCnames": [],
        }
    return samples


def _write_synthetic_artifact(path: Path, *, include_collision=False):
    samples = _samples(include_collision=include_collision)
    policy = resolve_sumw2_storage_policy(
        {"mode": "full_diagnostics"},
        samples=samples,
        runtime_families=("njets",),
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )
    products = resolve_data_driven_products(
        {
            # These fixtures intentionally contain process labels that predate
            # the canonical policy.  Keep data-driven production disabled so
            # the repair tool, rather than fresh-production certification,
            # remains the component under test.
            "nonprompt": {"enabled": False},
            "flips": {"enabled": False},
        },
        data_driven_products_present=True,
        legacy_do_np=False,
        samples=samples,
        runtime_families=("njets",),
        metadata_path="synthetic_options.yml",
    )
    requested, resolved = certify_data_driven_preflight(products, policy)
    entries = [
        (label, "3l", float(index + 1))
        for index, label in enumerate((*OLD_LABELS, *UNCHANGED_LABELS, *DATA_LABELS))
    ]
    if include_collision:
        entries.append((CANONICAL_LABELS[0], "3l", 100.0))
    payload = {
        scalar_nominal_key("njets"): _fill_sparse("njets", entries),
        sumw2_key("njets"): _fill_sparse(
            sumw2_key("njets"),
            [(process, channel, weight**2) for process, channel, weight in entries],
        ),
    }
    production_contract = certify_production_sample_contract(
        build_active_sample_universe(samples, wrapper_identity="pytest"),
        policy,
        products,
    )
    return write_histogram_artifact(
        path,
        histograms=payload,
        artifact_kind="processor_output",
        sumw2_storage_provenance=policy.to_provenance(),
        requested_data_driven_products=requested,
        resolved_data_driven_contract=resolved,
        production_sample_contract=production_contract,
    )


def _dense_content(histogram):
    return {
        tuple(key): np.asarray(value).copy()
        for key, value in histogram.view(flow=True).items()
    }


def _sidecar_contains_exact_label(value, labels):
    if isinstance(value, dict):
        return any(_sidecar_contains_exact_label(child, labels) for child in value.values())
    if isinstance(value, list):
        return any(_sidecar_contains_exact_label(child, labels) for child in value)
    return isinstance(value, str) and value in labels


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _map_typed_process_metadata(value, mapping, path=()):
    if isinstance(value, dict):
        mapped = {}
        for key, child in value.items():
            child_path = (*path, key)
            if key == "production_sample_contract":
                mapped[key] = copy.deepcopy(child)
            elif key in repair_module._PROCESS_SCALAR_FIELDS:
                mapped[key] = mapping.get(child, child)
            elif key in repair_module._PROCESS_LIST_FIELDS:
                mapped_values = [mapping.get(item, item) for item in child]
                mapped[key] = sorted(set(mapped_values))
            else:
                mapped[key] = _map_typed_process_metadata(
                    child,
                    mapping,
                    child_path,
                )
        return mapped
    if isinstance(value, list):
        return [
            _map_typed_process_metadata(child, mapping, (*path, index))
            for index, child in enumerate(value)
        ]
    return copy.deepcopy(value)


def _map_histogram_processes(histograms, mapping):
    mapped_histograms = {}
    for histogram_name, histogram in histograms.items():
        labels = repair_module._process_labels(histogram)
        if not labels & set(mapping):
            mapped_histograms[histogram_name] = histogram
            continue
        groups = {}
        for label in sorted(labels):
            groups.setdefault(mapping.get(label, label), []).append(label)
        mapped_histograms[histogram_name] = histogram.group("process", groups)
    return mapped_histograms


def _write_raw_sidecar(input_path, sidecar):
    metadata_sidecar_path(input_path).write_text(
        json.dumps(sidecar, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_enabled_precanonical_v3_artifact(input_path, *, warning_texts=()):
    source_path = input_path.parent / f"{input_path.stem}_source.pkl.gz"
    samples = {
        **{
            f"prompt_{index}": {
                "histAxisName": label,
                "isData": False,
                "WCnames": [],
            }
            for index, label in enumerate(CANONICAL_LABELS)
        },
        **{
            f"data_{index}": {
                "histAxisName": label,
                "isData": True,
                "WCnames": [],
            }
            for index, label in enumerate(DATA_LABELS)
        },
    }
    policy = resolve_sumw2_storage_policy(
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
                    "data": {"process_names": list(DATA_LABELS)},
                    "prompt_mc": {"process_names": list(CANONICAL_LABELS)},
                },
            },
            "flips": {"enabled": False},
        },
        data_driven_products_present=True,
        legacy_do_np=False,
        samples=samples,
        runtime_families=("njets",),
        metadata_path="precanonical_v3_fixture.yml",
    )
    requested, resolved = certify_data_driven_preflight(products, policy)
    entries = [
        (label, "3l", float(index + 1))
        for index, label in enumerate((*CANONICAL_LABELS, *DATA_LABELS))
    ]
    payload = {
        scalar_nominal_key("njets"): _fill_sparse("njets", entries),
        sumw2_key("njets"): _fill_sparse(
            sumw2_key("njets"),
            [(process, channel, weight**2) for process, channel, weight in entries],
        ),
    }
    production_contract = certify_production_sample_contract(
        build_active_sample_universe(samples, wrapper_identity="pytest"),
        policy,
        products,
    )
    source_sidecar = write_histogram_artifact(
        source_path,
        histograms=payload,
        artifact_kind="processor_output",
        sumw2_storage_provenance=policy.to_provenance(),
        requested_data_driven_products=requested,
        resolved_data_driven_contract=resolved,
        production_sample_contract=production_contract,
    )
    producer = DataDrivenProducer(
        str(source_path),
        "",
        artifact_kind="nonprompt_output",
    )
    transformed = producer.getDataDrivenHistogram()
    output_sidecar = write_histogram_artifact(
        input_path,
        histograms=transformed,
        artifact_kind="nonprompt_output",
        sumw2_storage_provenance=policy.to_provenance(),
        lineage_inputs=[lineage_input_from_sidecar(source_sidecar)],
        input_sidecar=source_sidecar,
        transformation_context=producer.get_transformation_context(
            "nonprompt_output"
        ),
    )

    legacy_histograms = _map_histogram_processes(
        transformed,
        _legacy_labels_by_canonical,
    )
    with gzip.open(input_path, "wb") as stream:
        cloudpickle.dump(legacy_histograms, stream)
    legacy_sidecar = copy.deepcopy(output_sidecar)
    current_contract = legacy_sidecar["resolved_data_driven_contract"]
    # Version-3 contracts stored only profile-required prompt signals here.
    # This fixture has WWW/WWZ and data contributors, none of which belonged to
    # the maintained historical signal-variant or unpaired-signal sets.
    legacy_required_prompt_signal_processes = []
    legacy_sidecar["resolved_data_driven_contract"] = {
        "contract_version": 3,
        "required_prompt_signal_processes": legacy_required_prompt_signal_processes,
        "products": copy.deepcopy(current_contract["products"]),
    }
    legacy_sidecar = _map_typed_process_metadata(
        legacy_sidecar,
        _legacy_labels_by_canonical,
    )
    legacy_sidecar["requested_data_driven_products"]["warnings"] = list(
        warning_texts
    )
    legacy_sidecar["artifact"]["pkl_size_bytes"] = input_path.stat().st_size
    legacy_sidecar["artifact"]["pkl_sha256"] = _sha256(input_path)
    _write_raw_sidecar(input_path, legacy_sidecar)
    return legacy_histograms, legacy_sidecar


def test_exact_repair_is_dry_run_first_and_write_is_copy_only(tmp_path):
    input_path = tmp_path / "input.pkl.gz"
    original_sidecar = _write_synthetic_artifact(input_path)
    input_identity = (input_path.read_bytes(), metadata_sidecar_path(input_path).read_bytes())
    output_dir = tmp_path / "corrected"

    dry_run = repair_artifacts([input_path], output_dir=output_dir)

    assert dry_run[0]["input_validation_mode"] == "already_canonical"
    assert (
        dry_run[0]["repaired_representation_validation"]
        == "passed_unchanged_validate_histogram_artifact"
    )
    assert dry_run[0]["mapping_entries_found"] == RUN2_PROCESS_LABEL_REPAIRS
    assert dry_run[0]["mapping_entries_absent"] == []
    assert set(dry_run[0]["payload_histograms_affected"]) == {
        scalar_nominal_key("njets"),
        sumw2_key("njets"),
    }
    assert {
        "sumw2_storage_provenance",
        "sumw2_content_manifest",
    } <= set(dry_run[0]["sidecar_surfaces_affected"])
    assert dry_run[0]["write_performed"] is False
    assert not output_dir.exists()
    assert input_identity == (
        input_path.read_bytes(),
        metadata_sidecar_path(input_path).read_bytes(),
    )

    written = repair_artifacts([input_path], output_dir=output_dir, write=True)
    output_path = output_dir / input_path.name
    output_sidecar_path = metadata_sidecar_path(output_path)

    assert written[0]["write_performed"] is True
    assert output_path.is_file()
    assert output_sidecar_path.is_file()
    assert input_identity == (
        input_path.read_bytes(),
        metadata_sidecar_path(input_path).read_bytes(),
    )
    output = get_hist_from_pkl(str(output_path))
    validated = validate_histogram_artifact(output_path, histograms=output)
    repaired_sidecar = read_histogram_sidecar(output_path)
    assert validated["metadata"] == repaired_sidecar
    assert not _sidecar_contains_exact_label(repaired_sidecar, set(OLD_LABELS))
    assert repaired_sidecar["production_sample_contract"] == original_sidecar[
        "production_sample_contract"
    ]
    for histogram in output.values():
        labels = {str(label) for label in histogram.axes["process"]}
        assert not labels & set(OLD_LABELS)
        assert set(CANONICAL_LABELS) <= labels
        assert set(UNCHANGED_LABELS) <= labels

    original = get_hist_from_pkl(str(input_path))
    for key in original:
        repaired_content = _dense_content(output[key])
        for categorical_key, dense_values in _dense_content(original[key]).items():
            categories = list(categorical_key)
            categories[0] = RUN2_PROCESS_LABEL_REPAIRS.get(categories[0], categories[0])
            np.testing.assert_array_equal(repaired_content[tuple(categories)], dense_values)

    with pytest.raises(repair_error, match="Refusing to overwrite"):
        repair_artifacts([input_path], output_dir=output_dir, write=True)


def test_enabled_precanonical_v3_fixture_dry_run_is_canonically_validated(tmp_path):
    input_path = tmp_path / "precanonical.pkl.gz"
    legacy_histograms, _legacy_sidecar = _write_enabled_precanonical_v3_artifact(
        input_path
    )
    input_identity = (
        input_path.read_bytes(),
        metadata_sidecar_path(input_path).read_bytes(),
    )
    output_dir = tmp_path / "corrected"

    with pytest.raises(histogram_artifact_error) as baseline_error:
        validate_histogram_artifact(input_path, histograms=legacy_histograms)
    assert "active alias is not valid for its exact run era" in str(
        baseline_error.value
    )
    assert OLD_LABELS[0] in str(baseline_error.value)

    dry_run = repair_artifacts([input_path], output_dir=output_dir)

    assert dry_run[0]["input_validation_mode"] == "known_repairable_legacy"
    assert dry_run[0]["mapping_entries_found"] == RUN2_PROCESS_LABEL_REPAIRS
    assert (
        dry_run[0]["repaired_representation_validation"]
        == "passed_unchanged_validate_histogram_artifact"
    )
    assert dry_run[0]["write_performed"] is False
    assert not output_dir.exists()
    assert input_identity == (
        input_path.read_bytes(),
        metadata_sidecar_path(input_path).read_bytes(),
    )


def test_enabled_precanonical_v3_warning_text_is_non_authoritative(tmp_path):
    input_path = tmp_path / "warning_text.pkl.gz"
    warning_texts = (
        "DATA-DRIVEN-W001: historical request mentions " + ", ".join(OLD_LABELS),
    )
    legacy_histograms, legacy_sidecar = _write_enabled_precanonical_v3_artifact(
        input_path,
        warning_texts=warning_texts,
    )
    output_dir = tmp_path / "corrected"

    with pytest.raises(histogram_artifact_error) as baseline_error:
        validate_histogram_artifact(input_path, histograms=legacy_histograms)
    assert "active alias is not valid for its exact run era" in str(
        baseline_error.value
    )

    dry_run = repair_artifacts([input_path], output_dir=output_dir)
    prepared = repair_module._load_and_prepare(input_path, output_dir)

    assert dry_run[0]["input_validation_mode"] == "known_repairable_legacy"
    assert (
        dry_run[0]["repaired_representation_validation"]
        == "passed_unchanged_validate_histogram_artifact"
    )
    assert legacy_sidecar["requested_data_driven_products"]["warnings"] == list(
        warning_texts
    )
    assert prepared["sidecar"]["requested_data_driven_products"]["warnings"] == list(
        warning_texts
    )
    assert not output_dir.exists()


def test_enabled_precanonical_v3_fixture_write_is_copy_only_and_canonical(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "precanonical.pkl.gz"
    _legacy_histograms, legacy_sidecar = _write_enabled_precanonical_v3_artifact(
        input_path
    )
    input_identity = (
        input_path.read_bytes(),
        metadata_sidecar_path(input_path).read_bytes(),
    )
    original = get_hist_from_pkl(str(input_path))
    output_dir = tmp_path / "corrected"
    validation_paths = []
    real_validate = repair_module.validate_histogram_artifact

    def tracking_validate(pkl_path, histograms=None):
        validation_paths.append(Path(pkl_path))
        return real_validate(pkl_path, histograms=histograms)

    monkeypatch.setattr(
        repair_module,
        "validate_histogram_artifact",
        tracking_validate,
    )
    written = repair_artifacts([input_path], output_dir=output_dir, write=True)
    output_path = output_dir / input_path.name
    output = get_hist_from_pkl(str(output_path))
    repaired_sidecar = read_histogram_sidecar(output_path)

    assert written[0]["write_performed"] is True
    assert len(validation_paths) == 4
    assert validate_histogram_artifact(output_path, histograms=output)["metadata"]
    assert input_identity == (
        input_path.read_bytes(),
        metadata_sidecar_path(input_path).read_bytes(),
    )
    assert repaired_sidecar["production_sample_contract"] == legacy_sidecar[
        "production_sample_contract"
    ]
    for key in original:
        repaired_content = _dense_content(output[key])
        for categorical_key, dense_values in _dense_content(original[key]).items():
            categories = list(categorical_key)
            categories[0] = RUN2_PROCESS_LABEL_REPAIRS.get(
                categories[0], categories[0]
            )
            np.testing.assert_array_equal(
                repaired_content[tuple(categories)],
                dense_values,
            )


def test_histogram_repair_handles_eft_and_preserves_coefficients():
    original = _fill_eft(OLD_LABELS[1])
    repaired, affected = _repair_histograms({"njets__eft_nominal": original})

    assert affected == {"njets__eft_nominal": [OLD_LABELS[1]]}
    assert {str(label) for label in repaired["njets__eft_nominal"].axes["process"]} == {
        CANONICAL_LABELS[1]
    }
    np.testing.assert_array_equal(
        next(iter(original.view(flow=True).values())),
        next(iter(repaired["njets__eft_nominal"].view(flow=True).values())),
    )


def test_histogram_repair_preserves_weight_storage_values_and_variances():
    original = SparseHist(*_axes("njets"), storage="Weight")
    original.fill(
        process=OLD_LABELS[2],
        channel="3l",
        systematic="nominal",
        appl="isSR_3l",
        njets=np.asarray([0.5, 1.5]),
        weight=np.asarray([2.0, 3.0]),
    )

    repaired, _ = _repair_histograms({"weighted": original})
    original_values = next(iter(original.view(flow=True).values()))
    repaired_values = next(iter(repaired["weighted"].view(flow=True).values()))

    np.testing.assert_array_equal(repaired_values.value, original_values.value)
    np.testing.assert_array_equal(repaired_values.variance, original_values.variance)


def test_process_collision_is_refused(tmp_path):
    input_path = tmp_path / "collision.pkl.gz"
    _write_synthetic_artifact(input_path, include_collision=True)

    with pytest.raises(repair_error, match="merge existing categorical support"):
        repair_artifacts([input_path])


def test_transformation_contract_process_fields_are_repaired_by_type():
    metadata = {
        "transformation_contract": {
            "families": {
                "njets": {
                    "source_scalar_processes": [OLD_LABELS[0], "other"],
                    "retained_scalar_processes": [OLD_LABELS[2]],
                    "generated_nonprompt_processes": ["nonpromptUL17"],
                }
            }
        }
    }

    repaired = _repair_typed_metadata(metadata)

    family = repaired["transformation_contract"]["families"]["njets"]
    assert family["source_scalar_processes"] == [CANONICAL_LABELS[0], "other"]
    assert family["retained_scalar_processes"] == [CANONICAL_LABELS[2]]
    assert family["generated_nonprompt_processes"] == ["nonpromptUL17"]


def test_unsupported_old_label_surface_is_refused(tmp_path):
    input_path = tmp_path / "unsupported.pkl.gz"
    _write_enabled_precanonical_v3_artifact(input_path)
    sidecar_path = metadata_sidecar_path(input_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["lineage"]["unsupported_identity"] = OLD_LABELS[0]
    _write_raw_sidecar(input_path, sidecar)

    with pytest.raises(repair_error, match="unsupported metadata field"):
        repair_artifacts([input_path])


def test_top_level_free_text_with_legacy_label_is_refused(tmp_path):
    input_path = tmp_path / "top_level_free_text.pkl.gz"
    _write_enabled_precanonical_v3_artifact(input_path)
    sidecar_path = metadata_sidecar_path(input_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["free_text"] = f"Historical note: {OLD_LABELS[0]}"
    _write_raw_sidecar(input_path, sidecar)

    with pytest.raises(repair_error, match="exact supported sidecar field shape"):
        repair_artifacts([input_path])


def test_unrelated_nested_warnings_with_legacy_label_are_refused(tmp_path):
    input_path = tmp_path / "unrelated_nested_warnings.pkl.gz"
    _write_enabled_precanonical_v3_artifact(input_path)
    sidecar_path = metadata_sidecar_path(input_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["lineage"]["warnings"] = [OLD_LABELS[0]]
    _write_raw_sidecar(input_path, sidecar)

    with pytest.raises(repair_error, match="unsupported metadata field"):
        repair_artifacts([input_path])


def test_malformed_non_authoritative_warning_text_is_refused(tmp_path):
    input_path = tmp_path / "malformed_warning_text.pkl.gz"
    _write_enabled_precanonical_v3_artifact(input_path)
    sidecar_path = metadata_sidecar_path(input_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["requested_data_driven_products"]["warnings"] = {
        "warning": OLD_LABELS[0]
    }
    _write_raw_sidecar(input_path, sidecar)

    with pytest.raises(repair_error, match="Non-authoritative warning text"):
        repair_artifacts([input_path])


def test_production_sample_contract_old_label_is_refused(tmp_path):
    input_path = tmp_path / "production_contract.pkl.gz"
    _write_enabled_precanonical_v3_artifact(input_path)
    sidecar_path = metadata_sidecar_path(input_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["production_sample_contract"]["unsupported_identity"] = OLD_LABELS[0]
    _write_raw_sidecar(input_path, sidecar)

    with pytest.raises(repair_error, match="production_sample_contract"):
        repair_artifacts([input_path])


def test_unknown_or_fuzzy_legacy_label_is_refused(tmp_path):
    input_path = tmp_path / "fuzzy.pkl.gz"
    _write_enabled_precanonical_v3_artifact(input_path)
    sidecar_path = metadata_sidecar_path(input_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["sumw2_storage_provenance"]["resolved_processes"].append(
        "WWW_centralUL18"
    )
    _write_raw_sidecar(input_path, sidecar)

    with pytest.raises(repair_error, match="Unknown or fuzzy legacy process label"):
        repair_artifacts([input_path])


def test_payload_unknown_or_fuzzy_legacy_label_is_refused(tmp_path):
    input_path = tmp_path / "payload_fuzzy.pkl.gz"
    _write_enabled_precanonical_v3_artifact(input_path)
    sidecar_path = metadata_sidecar_path(input_path)
    histograms = dict(get_hist_from_pkl(str(input_path)))
    payload = _map_histogram_processes(
        histograms,
        {OLD_LABELS[0]: "WWW_centralUL18"},
    )
    with gzip.open(input_path, "wb") as stream:
        cloudpickle.dump(payload, stream)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["artifact"]["pkl_size_bytes"] = input_path.stat().st_size
    sidecar["artifact"]["pkl_sha256"] = _sha256(input_path)
    _write_raw_sidecar(input_path, sidecar)

    with pytest.raises(repair_error, match="Payload contains unknown or fuzzy"):
        repair_artifacts([input_path])


def test_artifact_identity_mismatch_is_refused(tmp_path):
    input_path = tmp_path / "identity.pkl.gz"
    _write_enabled_precanonical_v3_artifact(input_path)
    sidecar_path = metadata_sidecar_path(input_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["artifact"]["pkl_sha256"] = "0" * 64
    _write_raw_sidecar(input_path, sidecar)

    with pytest.raises(repair_error, match="artifact identity"):
        repair_artifacts([input_path])


def test_unrelated_invalidity_surviving_mapping_is_refused(tmp_path):
    input_path = tmp_path / "unrelated.pkl.gz"
    _write_enabled_precanonical_v3_artifact(input_path)
    sidecar_path = metadata_sidecar_path(input_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["lineage"] = {"inputs": "invalid"}
    _write_raw_sidecar(input_path, sidecar)

    with pytest.raises(histogram_artifact_error):
        repair_artifacts([input_path])


def test_post_mapping_canonical_validation_is_required(tmp_path, monkeypatch):
    input_path = tmp_path / "post_mapping_validation.pkl.gz"
    _write_enabled_precanonical_v3_artifact(input_path)
    validation_calls = []
    real_validate = repair_module.validate_histogram_artifact

    def reject_after_mapping(pkl_path, histograms=None):
        validation_calls.append(Path(pkl_path))
        result = real_validate(pkl_path, histograms=histograms)
        if len(validation_calls) == 2:
            raise histogram_sidecar_error("post-mapping validation sentinel")
        return result

    monkeypatch.setattr(
        repair_module,
        "validate_histogram_artifact",
        reject_after_mapping,
    )
    with pytest.raises(histogram_sidecar_error, match="post-mapping validation sentinel"):
        repair_artifacts([input_path])
    assert len(validation_calls) == 2


def test_write_mode_requires_explicit_separate_output_directory(tmp_path):
    input_path = tmp_path / "input.pkl.gz"
    _write_synthetic_artifact(input_path)

    with pytest.raises(repair_error, match="explicit --output-dir"):
        repair_artifacts([input_path], write=True)
    with pytest.raises(repair_error, match="in-place repair is forbidden"):
        repair_artifacts([input_path], output_dir=tmp_path)
