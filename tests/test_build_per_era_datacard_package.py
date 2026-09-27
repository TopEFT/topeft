"""Synthetic contracts for the current per-era package builder."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from analysis.topeft_run2 import build_per_era_datacard_package as builder
from analysis.topeft_run2 import datacard_matrix_runner as runner
from topeft.modules import datacard_packaging


@pytest.fixture(autouse=True)
def _synthetic_builder_identity(monkeypatch):
    monkeypatch.setattr(builder, "_builder_identity", lambda: ("b" * 40, "c" * 64, True))


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _record(path):
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": datacard_packaging.sha256_file(path)}


def _unit(tmp_path, unit_id, name, wc):
    source = tmp_path / unit_id
    txt = _write(source / f"ttx_multileptons-{name}.txt", f"card {name}\n".encode())
    root = _write(source / f"ttx_multileptons-{name}.root", b"ROOT\x00" + name.encode())
    selected = _write(source / "selectedWCs.txt", json.dumps({"ttH": [wc]}).encode())
    scaling = _write(source / "scalings-preselect.json", json.dumps([{
        "channel": name, "process": "ttH", "parameters": [wc], "scaling": [[1.0, 0.5]],
    }]).encode())
    return {
        "era": "run2", "unit_id": unit_id, "physical_targets": [name],
        "primary_outputs": [{"physical_name": name, "txt": _record(txt), "root": _record(root)}],
        "selected_wcs_source": _record(selected), "scalings_source": _record(scaling),
    }


def _manifest_receipt(tmp_path, unit_id="unit_01", channel="alpha", era="run2", attempt="attempt_01"):
    control = tmp_path / f"control_{unit_id}"
    output_root = tmp_path / f"payload_{unit_id}"
    name = f"{channel}_ptz"
    txt = _write(output_root / f"ttx_multileptons-{name}.txt", b"card\n")
    root = _write(output_root / f"ttx_multileptons-{name}.root", b"root\x00")
    snapshot = control / "snapshots" / unit_id / attempt
    selected = _write(snapshot / "selectedWCs.txt", b'{"ttH":["cpt"]}')
    scaling = _write(snapshot / "scalings-preselect.json", b"[]")
    merge_snapshot = _write(snapshot / "merge_report.json", b"{}")
    merge = _write(control / "merge.json", b"{}")
    log = _write(control / "runner.log", b"ok\n")
    fingerprint = _write(tmp_path / "producer.py", b"# synthetic\n")
    args = ["--out-dir", str(output_root), "--var-lst", "ptz", "--ch-lst", channel,
            "--year", "UL18", "--miss-parton-file", str(tmp_path / "missing.root"),
            "--merge-report", str(merge)]
    row = {
        "row_id": unit_id, "logical_row_id": unit_id,
        "attempt_id": attempt, "era": era,
        "working_directory": str(tmp_path), "input_pkl": str(tmp_path / "input.pkl"),
        "output_root": str(output_root), "distribution": "ptz", "physical_channels": [channel],
        "years": ["UL18"], "missing_parton_path": str(tmp_path / "missing.root"),
        "merge_report_path": str(merge),
        "snapshot_directory": str(snapshot), "log_path": str(log),
        "expected_output_paths": [str(root), str(txt)], "producer_args": args,
    }
    runtime = {
        "contract_id": "synthetic-v3", "python_executable": str(tmp_path / "python"),
        "make_cards_path": str(fingerprint), "fingerprints": [_record(fingerprint)],
    }
    runtime["fingerprints"][0].pop("size_bytes")
    manifest = {
        "schema": runner.MANIFEST_SCHEMA, "control_root": str(control),
        "lock_path": str(control / "runner.lock"), "runtime_contract": runtime, "rows": [row],
    }
    manifest_path = tmp_path / f"{unit_id}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipt = {
        "schema": runner.RECEIPT_SCHEMA, "manifest_sha256": manifest_hash,
        "row_id": unit_id, "attempt_id": attempt,
        "resolved_argv": runner.resolved_argv(manifest, row),
        "start_timestamp": "start", "end_timestamp": "end", "command_return_code": 0,
        "runtime_contract_id": runtime["contract_id"],
        "python_executable": runtime["python_executable"],
        "make_cards_path": runtime["make_cards_path"],
        "runtime_fingerprints": runtime["fingerprints"],
        "runtime_contract_digest": runner.runtime_digest(runtime),
        "primary_outputs": [_record(root), _record(txt)],
        "artifacts": {
            "merge_report": _record(merge), "selected_wcs_snapshot": _record(selected),
            "scalings_snapshot": _record(scaling), "merge_report_snapshot": _record(merge_snapshot),
            "log": _record(log),
        },
    }
    receipt_path = runner.receipt_path(manifest, row)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return manifest_path, receipt_path, manifest, receipt


def _replace_receipt(path, receipt):
    path.write_text(json.dumps(receipt), encoding="utf-8")


def test_historical_v2_manifest_rejected_for_new_package_build(tmp_path):
    path, _, manifest, _ = _manifest_receipt(tmp_path)
    manifest["schema"] = "topeft_datacard_matrix_v2"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(runner.RunnerError) as exc_info:
        builder._resolve_v3_manifest_units([path], "run2")
    assert exc_info.value.code == "manifest_schema_error"
    message = str(exc_info.value)
    assert all(anchor in message for anchor in (
        "Historical", "topeft_datacard_matrix_v2", "current datacard workflow",
        "topeft_datacard_matrix_v3", "make_datacard_matrix_manifest.py",
    ))
    assert "new runtime execution" not in message


def _partial_manifest(tmp_path):
    first, _, first_manifest, first_receipt = _manifest_receipt(tmp_path, "unit_01", "alpha")
    second, _, second_manifest, _ = _manifest_receipt(tmp_path, "unit_02", "beta")
    plan = copy.deepcopy(first_manifest)
    plan["control_root"] = str(tmp_path)
    plan["rows"].append(second_manifest["rows"][0])
    plan_path = tmp_path / "partial_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    first_receipt["manifest_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan_receipt = runner.receipt_path(plan, plan["rows"][0])
    plan_receipt.parent.mkdir(parents=True, exist_ok=True)
    _replace_receipt(plan_receipt, first_receipt)
    assert not runner.receipt_path(plan, plan["rows"][1]).exists()
    return plan_path, second


def test_current_v3_resolver_preserves_receipt_identities(tmp_path):
    first, _, _, first_receipt = _manifest_receipt(tmp_path, "unit_01", "alpha")
    second, _, _, second_receipt = _manifest_receipt(tmp_path, "unit_02", "beta")
    units, hashes = builder._resolve_v3_manifest_units([second, first], "run2")
    assert [unit["unit_id"] for unit in units] == ["unit_02", "unit_01"]
    assert hashes == [hashlib.sha256(path.read_bytes()).hexdigest() for path in (second, first)]
    assert units[0]["primary_outputs"][0]["txt"] == second_receipt["primary_outputs"][1]
    assert units[1]["primary_outputs"][0]["root"] == first_receipt["primary_outputs"][0]
    assert units[0]["selected_wcs_source"] == second_receipt["artifacts"]["selected_wcs_snapshot"]
    assert units[1]["scalings_source"] == first_receipt["artifacts"]["scalings_snapshot"]
    with pytest.raises(ValueError, match="duplicate manifest identity"):
        builder._resolve_v3_manifest_units([first, first], "run2")
    with pytest.raises(ValueError, match="no selected"):
        builder._resolve_v3_manifest_units([first], "run3")


def test_partial_manifest_and_later_success(tmp_path):
    plan, later = _partial_manifest(tmp_path)
    units, hashes = builder._resolve_v3_manifest_units([plan], "run2")
    assert [unit["unit_id"] for unit in units] == ["unit_01"]
    assert hashes == [hashlib.sha256(plan.read_bytes()).hexdigest()]
    units, hashes = builder._resolve_v3_manifest_units([plan, later], "run2")
    assert [unit["unit_id"] for unit in units] == ["unit_01", "unit_02"]
    assert hashes == [hashlib.sha256(path.read_bytes()).hexdigest() for path in (plan, later)]


def test_cli_missing_target_reports_matrix_row_and_runner_commands(tmp_path):
    plan, _ = _partial_manifest(tmp_path)
    output = tmp_path / "package"
    with pytest.raises(ValueError) as exc:
        builder.main(["build", "--era", "run2", "--matrix-manifest", str(plan),
                      "--output", str(output), "--analysis", "TOP-26-006"])
    message = str(exc.value)
    runner_command = "analysis/topeft_run2/run_datacard_matrix_resumable.sh"
    assert "missing=['beta_ptz']" in message
    assert f"beta_ptz: manifest={plan}, row_id=unit_02" in message
    assert f"{runner_command} --status {plan}" in message
    assert f"{runner_command} {plan}" in message
    assert not output.exists()


def test_cli_uses_only_manifest_surface_for_restricted_package(tmp_path):
    first, _, _, _ = _manifest_receipt(tmp_path / "first", "unit_01", "alpha")
    output = tmp_path / "package"
    assert builder.main(["build", "--era", "run2", "--matrix-manifest", str(first),
                         "--output", str(output), "--analysis", "TOP-26-006"]) == 0
    assert json.loads((output / "physical_to_chN.json").read_text()) == [
        {"physical_name": "alpha_ptz", "per_era_chN": "ch1"}]
    with pytest.raises(SystemExit):
        builder.main(["build", "--era", "run2", "--matrix-manifest", str(first),
                      "--output", str(output), "--analysis", "TOP-26-006",
                      "--physical-target", "alpha_ptz"])
    assert output.is_dir()


def test_duplicate_manifest_target_blocks_before_packaging(tmp_path):
    first, _, _, _ = _manifest_receipt(tmp_path / "first", "unit_01", "alpha")
    second, _, _, _ = _manifest_receipt(tmp_path / "second", "unit_02", "alpha")
    with pytest.raises(ValueError, match="duplicate manifest physical target"):
        builder._declared_manifest_surface([first, second], "run2")


def test_existing_malformed_receipt_fails_closed(tmp_path):
    plan, _ = _partial_manifest(tmp_path)
    manifest, _ = runner.load_manifest(plan)
    receipt_path = runner.receipt_path(manifest, manifest["rows"][1])
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text("{invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable current receipt"):
        builder._resolve_v3_manifest_units([plan], "run2")


def test_duplicate_successful_unit_identity(tmp_path):
    first, _, _, _ = _manifest_receipt(tmp_path / "first", "unit_01", "alpha")
    second, _, _, _ = _manifest_receipt(tmp_path / "second", "unit_01", "beta")
    with pytest.raises(ValueError, match="duplicate unit identity"):
        builder._resolve_v3_manifest_units([first, second], "run2")


@pytest.mark.parametrize("mutation", ["stale", "failed", "incomplete", "identity", "snapshot"])
def test_current_v3_rejects_unbound_or_incomplete_receipt(tmp_path, mutation):
    manifest_path, receipt_path, _, receipt = _manifest_receipt(tmp_path)
    if mutation == "stale":
        receipt["attempt_id"] = "attempt_old"
    elif mutation == "failed":
        receipt["command_return_code"] = 1
    elif mutation == "incomplete":
        receipt["primary_outputs"].pop()
    elif mutation == "identity":
        receipt["primary_outputs"][0]["sha256"] = "0" * 64
    else:
        receipt["artifacts"]["selected_wcs_snapshot"]["sha256"] = "0" * 64
    _replace_receipt(receipt_path, receipt)
    with pytest.raises(ValueError):
        builder._resolve_v3_manifest_units([manifest_path], "run2")


def test_current_v3_rejects_overlap_and_era_mismatch(tmp_path):
    first, _, _, _ = _manifest_receipt(tmp_path, "unit_01", "alpha")
    second, _, _, _ = _manifest_receipt(tmp_path, "unit_02", "alpha")
    with pytest.raises(ValueError, match="duplicate physical"):
        builder._resolve_v3_manifest_units([first, second], "run2")
    third, _, _, _ = _manifest_receipt(tmp_path, "unit_03", "gamma", era="run3")
    with pytest.raises(ValueError, match="no selected"):
        builder._resolve_v3_manifest_units([third], "run2")


def test_builder_core_publishes_canonical_cards_and_minimal_provenance(tmp_path):
    beta = _unit(tmp_path, "unit_beta", "beta_ptz", "ctW")
    alpha = _unit(tmp_path, "unit_alpha", "alpha_ptz", "cpt")
    output = tmp_path / "package"
    provenance = builder.build_per_era_package_from_units(
        "run2", "TOP-26-006", ["alpha_ptz", "beta_ptz"], [beta, alpha], output,
        ["b" * 64, "a" * 64],
    )
    assert set(path.name for path in output.iterdir()) == {
        "cards", "selectedWCs.txt", "scalings.json", "physical_to_chN.json", "package-provenance.json",
    }
    assert json.loads((output / "physical_to_chN.json").read_text()) == [
        {"physical_name": "alpha_ptz", "per_era_chN": "ch1"},
        {"physical_name": "beta_ptz", "per_era_chN": "ch2"},
    ]
    assert set(provenance) == builder._provenance_keys
    assert provenance["source_manifest_sha256s"] == ["b" * 64, "a" * 64]
    assert not (tmp_path / ".package.staging").exists()
    for unit in (alpha, beta):
        for suffix in ("txt", "root"):
            source = unit["primary_outputs"][0][suffix]
            copied = output / "cards" / Path(source["path"]).name
            assert copied.read_bytes() == Path(source["path"]).read_bytes()
            assert datacard_packaging.sha256_file(copied) == source["sha256"]


@pytest.mark.parametrize("fault", ["mapping", "selected", "scalings", "copy"])
def test_faults_preserve_staging_without_publishing(tmp_path, monkeypatch, fault):
    unit = _unit(tmp_path, "unit", "alpha_ptz", "cpt")
    output = tmp_path / "package"
    if fault == "mapping":
        monkeypatch.setattr(datacard_packaging, "build_per_era_mapping",
                            lambda names: [{"physical_name": "alpha_ptz", "per_era_chN": "ch2"}])
    elif fault == "selected":
        monkeypatch.setattr(datacard_packaging, "consolidate_selected_wcs", lambda sources: {"ttH": []})
    elif fault == "scalings":
        monkeypatch.setattr(datacard_packaging, "consolidate_scaling_records", lambda sources, mapping: [])
    else:
        original_copy = builder.shutil.copyfile

        def corrupt_copy(source, destination):
            original_copy(source, destination)
            Path(destination).write_bytes(b"corrupt")

        monkeypatch.setattr(builder.shutil, "copyfile", corrupt_copy)
    with pytest.raises((ValueError, OSError)):
        builder.build_per_era_package_from_units(
            "run2", "TOP-26-006", ["alpha_ptz"], [unit], output,
            ["a" * 64],
        )
    assert not output.exists()
    assert (tmp_path / ".package.staging").is_dir()


def test_missing_extra_and_existing_destinations_block(tmp_path):
    unit = _unit(tmp_path, "unit", "alpha_ptz", "cpt")
    for names in (["alpha_ptz", "beta_ptz"], ["beta_ptz"]):
        with pytest.raises(ValueError, match="target set differs"):
            builder.build_per_era_package_from_units(
                "run2", "TOP-26-006", names, [unit], tmp_path / "package",
                [],
            )
    output = tmp_path / "package"
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        builder.build_per_era_package_from_units("run2", "x", ["alpha_ptz"], [unit], output, [])
    output.rmdir()
    (tmp_path / ".package.staging").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        builder.build_per_era_package_from_units("run2", "x", ["alpha_ptz"], [unit], output, [])


@pytest.mark.parametrize("argument", ["plan", "dry-run", "resume", "--accepted-registry",
                                         "--legacy-certificate", "--historical-registry", "--migration-mode"])
def test_no_historical_or_plan_cli_surface(argument):
    with pytest.raises(SystemExit):
        builder.main([argument])
