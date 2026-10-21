"""Exact frozen Tau runtime overlay and self-contained receipt provenance."""

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

import pytest

from analysis.topeft_run2 import datacard_matrix_runner as runner


MANIFEST = Path("/groups/klannon/apiccine/top26006_tau_filter_perrun_cards_261014_v1/manifests/run3_full.json")
REPOSITORY_ROOT = Path(runner.__file__).resolve().parents[2]
ALLOWED_ROWS = ["run3_07_exec_01", "run3_11_exec_01", "run3_05_exec_01"]
NEW_HASHES = {
    "analysis/topeft_run2/make_cards.py": "3aa9bf681b50cfd441e5ee335a8f2f568ffa82b140089ea2610efb042a7e8ec6",
    "topeft/modules/datacard_tools.py": "d2b879f9e3a1f51ab6f9542e1d271581eaf32608569e9085f389ad37378dd496",
}


def overlay_value(manifest_hash):
    return {
        "schema": runner.RUNTIME_OVERLAY_SCHEMA,
        "base_manifest_sha256": manifest_hash,
        "base_runtime_contract_digest": runner.RUNTIME_OVERLAY_BASE_DIGEST,
        "allowed_row_ids": ALLOWED_ROWS,
        "replacements": [
            {"path": path, "from_sha256": old_hash,
             "to_sha256": NEW_HASHES.get(path, runner.hash_file(REPOSITORY_ROOT / path))}
            for path, old_hash in runner.RUNTIME_OVERLAY_FROM.items()
        ],
    }


def frozen_context():
    manifest, manifest_hash = runner.load_manifest(MANIFEST)
    return manifest, manifest_hash, runner.select_row(manifest, ALLOWED_ROWS[0])


def test_legacy_runtime_validation_remains_strict(tmp_path):
    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("matching\n")
    contract = {
        "contract_id": "legacy", "python_executable": runner.sys.executable,
        "make_cards_path": str(runtime_file),
        "fingerprints": [{"path": str(runtime_file), "sha256": runner.hash_file(runtime_file)}],
    }
    assert runner.validate_runtime(contract) == contract
    runtime_file.write_text("changed\n")
    with pytest.raises(runner.RunnerError, match="fingerprint mismatch"):
        runner.validate_runtime(contract)


def test_exact_overlay_validates_runner_and_keeps_other_fingerprints_strict(monkeypatch):
    manifest, manifest_hash, row = frozen_context()
    overlay = overlay_value(manifest_hash)
    effective = runner.effective_runtime(manifest, manifest_hash, row, overlay)
    base = manifest["runtime_contract"]
    assert runner.runtime_digest(base) == runner.RUNTIME_OVERLAY_BASE_DIGEST
    assert runner.runtime_digest(effective) != runner.runtime_digest(base)
    pinned = {item["path"]: item["sha256"] for item in base["fingerprints"]}
    observed = {item["path"]: item["sha256"] for item in effective["fingerprints"]}
    for relative, old_hash in runner.RUNTIME_OVERLAY_FROM.items():
        path = str(REPOSITORY_ROOT / relative)
        assert pinned[path] == old_hash
        assert observed[path] == runner.hash_file(Path(path)) == NEW_HASHES.get(relative, observed[path])
    assert all(observed[path] == digest for path, digest in pinned.items()
               if path not in {str(REPOSITORY_ROOT / relative) for relative in runner.RUNTIME_OVERLAY_FROM})
    strict_path = next(path for path in pinned if pinned[path] == observed[path])
    original_hash_file = runner.hash_file

    def drifted_hash_file(path):
        return "0" * 64 if str(path) == strict_path else original_hash_file(path)

    monkeypatch.setattr(runner, "hash_file", drifted_hash_file)
    with pytest.raises(runner.RunnerError, match="fingerprint mismatch"):
        runner.effective_runtime(manifest, manifest_hash, row, overlay)


@pytest.mark.parametrize("change", [
    "manifest_sha", "base_digest", "wrong_row", "missing_runner", "runner_from",
    "runner_to", "make_cards_to", "datacard_tools_to", "fourth_path", "unknown_path",
    "duplicate_path", "malformed_schema",
])
def test_invalid_overlay_fails_before_lease_or_producer(monkeypatch, change):
    manifest, manifest_hash, row = frozen_context()
    overlay = overlay_value(manifest_hash)
    replacements = overlay["replacements"]
    by_path = {item["path"]: item for item in replacements}
    runner_path = "analysis/topeft_run2/datacard_matrix_runner.py"
    if change == "manifest_sha":
        overlay["base_manifest_sha256"] = "0" * 64
    elif change == "base_digest":
        overlay["base_runtime_contract_digest"] = "0" * 64
    elif change == "wrong_row":
        row = runner.select_row(manifest, "run3_01_exec_01")
    elif change == "missing_runner":
        replacements.remove(by_path[runner_path])
    elif change == "runner_from":
        by_path[runner_path]["from_sha256"] = "0" * 64
    elif change == "runner_to":
        by_path[runner_path]["to_sha256"] = "0" * 64
    elif change == "make_cards_to":
        by_path["analysis/topeft_run2/make_cards.py"]["to_sha256"] = "0" * 64
    elif change == "datacard_tools_to":
        by_path["topeft/modules/datacard_tools.py"]["to_sha256"] = "0" * 64
    elif change == "fourth_path":
        replacements.append({"path": "analysis/topeft_run2/datacard_matrix_profiles.yml",
                             "from_sha256": "0" * 64, "to_sha256": "0" * 64})
    elif change == "unknown_path":
        by_path[runner_path]["path"] = "unknown.py"
    elif change == "duplicate_path":
        by_path[runner_path]["path"] = "analysis/topeft_run2/make_cards.py"
    elif change == "malformed_schema":
        overlay["schema"] = "unknown"

    @contextmanager
    def forbidden_lease(*_args, **_kwargs):
        pytest.fail("execution reached lease before overlay validation")
        yield

    monkeypatch.setattr(runner, "unit_lease", forbidden_lease)
    with pytest.raises(runner.RunnerError):
        runner.execute_unit(manifest, manifest_hash, row, runtime_overlay=overlay)


def test_non_overridden_mismatch_fails_before_lease(monkeypatch):
    manifest, manifest_hash, row = frozen_context()
    overlay = overlay_value(manifest_hash)
    strict_path = str(REPOSITORY_ROOT / "analysis/topeft_run2/datacard_matrix_profiles.yml")
    original_hash_file = runner.hash_file

    def drifted_hash_file(path):
        return "0" * 64 if str(path) == strict_path else original_hash_file(path)

    @contextmanager
    def forbidden_lease(*_args, **_kwargs):
        pytest.fail("execution reached lease before strict fingerprint validation")
        yield

    monkeypatch.setattr(runner, "hash_file", drifted_hash_file)
    monkeypatch.setattr(runner, "unit_lease", forbidden_lease)
    with pytest.raises(runner.RunnerError, match="fingerprint mismatch"):
        runner.execute_unit(manifest, manifest_hash, row, runtime_overlay=overlay)


def receipt_fixture(tmp_path, manifest, manifest_hash, row, overlay=None):
    row = deepcopy(row)
    row["output_root"] = str(tmp_path / "output" / row["row_id"])
    row["expected_output_paths"] = [str(Path(row["output_root"]) / "card.txt")]
    row["merge_report_path"] = str(tmp_path / "control" / row["row_id"] / "merge.json")
    row["snapshot_directory"] = str(tmp_path / "control" / row["row_id"] / "snapshots")
    row["log_path"] = str(tmp_path / "control" / row["row_id"] / "producer.log")
    artifacts = {"merge_report": Path(row["merge_report_path"]),
                 "selected_wcs_snapshot": runner.snapshots(row)["selected_wcs"],
                 "scalings_snapshot": runner.snapshots(row)["scalings"],
                 "merge_report_snapshot": runner.snapshots(row)["merge_report"],
                 "log": Path(row["log_path"])}
    for path in [Path(row["expected_output_paths"][0]), *artifacts.values()]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic receipt artifact\n")
    runtime = manifest["runtime_contract"]
    receipt = {
        "schema": runner.RECEIPT_SCHEMA, "manifest_sha256": manifest_hash,
        "row_id": row["row_id"], "attempt_id": row["attempt_id"],
        "resolved_argv": runner.resolved_argv(manifest, row), "command_return_code": 0,
        "runtime_contract_id": runtime["contract_id"],
        "python_executable": runtime["python_executable"],
        "make_cards_path": runtime["make_cards_path"],
        "runtime_fingerprints": runtime["fingerprints"],
        "runtime_contract_digest": runner.runtime_digest(runtime),
        "start_timestamp": "synthetic-start", "end_timestamp": "synthetic-end",
        "primary_outputs": [runner.record(Path(path)) for path in row["expected_output_paths"]],
        "artifacts": {key: runner.record(path) for key, path in artifacts.items()},
    }
    if overlay is not None:
        receipt.update(runner.overlay_provenance(manifest, manifest_hash, row, overlay))
    return row, receipt


def test_legacy_and_overlay_receipts_validate_in_same_status(tmp_path):
    manifest, manifest_hash, row = frozen_context()
    overlay = overlay_value(manifest_hash)
    legacy_row = runner.select_row(manifest, "run3_01_exec_01")
    legacy_row, legacy_receipt = receipt_fixture(tmp_path, manifest, manifest_hash, legacy_row)
    overlay_row, overlay_receipt = receipt_fixture(tmp_path, manifest, manifest_hash, row, overlay)
    assert runner.validate_receipt(legacy_receipt, manifest, manifest_hash, legacy_row)[0]
    assert runner.validate_receipt(overlay_receipt, manifest, manifest_hash, overlay_row)[0]
    assert overlay_receipt["runtime_contract_digest"] == runner.RUNTIME_OVERLAY_BASE_DIGEST
    assert overlay_receipt["effective_runtime_contract_digest"] != runner.RUNTIME_OVERLAY_BASE_DIGEST
    for candidate, receipt in [(legacy_row, legacy_receipt), (overlay_row, overlay_receipt)]:
        path = tmp_path / "control" / "receipts" / f"{candidate['row_id']}__{candidate['attempt_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt))
    local_manifest = {**manifest, "control_root": str(tmp_path / "control"),
                      "lock_path": str(tmp_path / "control" / "runner.lock"),
                      "rows": [legacy_row, overlay_row]}
    status = runner.read_only("status", local_manifest, manifest_hash)
    assert [item["status"] for item in status["rows"]] == ["complete", "complete"]
    assert "runtime_contract_overlay_sha256" not in status["rows"][0]
    assert status["rows"][1]["runtime_contract_overlay_sha256"] == runner.runtime_overlay_digest(overlay)
    assert status["rows"][1]["effective_runtime_fingerprints"] == overlay_receipt["effective_runtime_fingerprints"]
    assert status["manifest_sha256"] == manifest_hash

    for key, value in [
        ("runtime_contract_overlay", {**overlay, "allowed_row_ids": ["run3_11_exec_01", "run3_05_exec_01", "run3_01_exec_01"]}),
        ("effective_runtime_contract_digest", "0" * 64),
    ]:
        tampered = deepcopy(overlay_receipt)
        tampered[key] = value
        assert not runner.validate_receipt(tampered, manifest, manifest_hash, overlay_row)[0]


def test_overlay_intent_provenance_is_validated_without_external_file(tmp_path):
    manifest, manifest_hash, row = frozen_context()
    overlay = overlay_value(manifest_hash)
    attempt = tmp_path / "attempt_000001"
    attempt.mkdir()
    staged_row = deepcopy(row)
    staged_row["output_root"] = str(tmp_path / "cards")
    staged_row["expected_output_paths"] = [str(tmp_path / "cards" / "card.txt")]
    staged_row["merge_report_path"] = str(tmp_path / "merge.json")
    staged_row["snapshot_directory"] = str(tmp_path / "snapshots")
    staged_row["log_path"] = str(tmp_path / "log")
    outputs, artifacts = runner.staged_paths(staged_row, attempt)
    for path in [*outputs, *artifacts.values()]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("staged artifact\n")
    intent = {
        "schema": runner.INTENT_SCHEMA, "manifest_sha256": manifest_hash,
        "row_id": staged_row["row_id"], "attempt_id": staged_row["attempt_id"],
        "physical_attempt_id": attempt.name,
        "resolved_argv": runner.resolved_argv(manifest, staged_row),
        "execution_argv": runner.staged_argv(manifest, staged_row, attempt),
        "runtime_contract_digest": runner.RUNTIME_OVERLAY_BASE_DIGEST,
        "command_return_code": 0, "start_timestamp": "synthetic-start",
        "end_timestamp": "synthetic-end",
        "primary_outputs": [runner.record(path) for path in outputs],
        "artifacts": {key: runner.record(path) for key, path in artifacts.items()},
        **runner.overlay_provenance(manifest, manifest_hash, staged_row, overlay),
    }
    (attempt / "intent.json").write_text(json.dumps(intent))
    assert runner.valid_intent(manifest, manifest_hash, staged_row, attempt)[0]
    intent["effective_runtime_contract_digest"] = "0" * 64
    (attempt / "intent.json").write_text(json.dumps(intent))
    assert not runner.valid_intent(manifest, manifest_hash, staged_row, attempt)[0]
