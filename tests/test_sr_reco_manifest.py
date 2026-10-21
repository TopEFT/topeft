import copy
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from analysis.topeft_run2 import sr_reco_manifest as manifest
from test_run2_srplot009_manual_handoff import (
    RUN_CR,
    RUN_DIRECTORY,
    _campaign_state,
    _clean_environment,
    _run,
    _write_backend,
    _write_test_environment,
)


def _backend_environment(validation_root):
    environment = _clean_environment()
    environment.update({
        "SRPLOT009_VALIDATION_BACKEND": str(validation_root / "backend.sh"),
        "SRPLOT009_VALIDATION_ROOT": str(validation_root),
        "SRPLOT009_VALIDATION_SCENARIO": "success",
    })
    return environment


def _replacement_units():
    initial = manifest.load_plan(manifest.DEFAULT_PLAN_PATH)
    pending = copy.deepcopy(initial[2:])
    b_1 = next(unit for unit in pending if unit["id"] == "b_1")
    b_2 = next(unit for unit in pending if unit["id"] == "b_2")
    merged = {
        "id": "v2_b_all",
        "category_groups": b_1["category_groups"],
        "histograms": b_1["histograms"] + b_2["histograms"],
    }
    return [
        {**unit, "id": f"v2_{unit['id']}"}
        for unit in pending if unit["id"] not in {"b_1", "b_2"}
    ] + [merged]


def _guard_accepted_payloads(tmp_path, environment, paths):
    """Make native child Python fail if it opens an accepted PKL for reading."""
    guard_root = tmp_path / "payload_read_guard"
    guard_root.mkdir()
    (guard_root / "sitecustomize.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "original_open = Path.open\n"
        "blocked = set(os.environ['SR_ACCEPTED_PKLS'].split(os.pathsep))\n"
        "Path(os.environ['SR_PAYLOAD_GUARD_MARKER']).write_text('loaded')\n"
        "def guarded_open(self, *args, **kwargs):\n"
        "    mode = args[0] if args else kwargs.get('mode', 'r')\n"
        "    if str(self) in blocked and 'r' in mode and 'b' in mode:\n"
        "        raise RuntimeError('accepted PKL content read: ' + str(self))\n"
        "    return original_open(self, *args, **kwargs)\n"
        "Path.open = guarded_open\n",
        encoding="utf-8",
    )
    guarded = environment.copy()
    guarded["SR_ACCEPTED_PKLS"] = os.pathsep.join(paths)
    guarded["SR_PAYLOAD_GUARD_MARKER"] = str(guard_root / "loaded.txt")
    guarded["PYTHONPATH"] = f"{guard_root}{os.pathsep}{environment.get('PYTHONPATH', '')}"
    return guarded


def test_first_capture_hashes_source_and_nonprompt_payloads(tmp_path, monkeypatch):
    source = tmp_path / "source.pkl.gz"
    nonprompt = tmp_path / "nonprompt.pkl.gz"
    state = {
        "schema_version": 7,
        "sr_reco_plan_revisions": [{"id": "r1"}],
        "sr_reco_scientific_contract_sha256": "a" * 64,
    }
    block = {
        "expected_nominal_path": str(source), "expected_np_path": str(nonprompt),
        "plan_revision": "r1",
    }
    actual_digest_file = manifest.digest_file
    payload_reads = []

    def record_payload_read(path):
        if Path(path) in {source, nonprompt}:
            payload_reads.append(str(path))
        return actual_digest_file(path)

    monkeypatch.setattr(manifest, "digest_file", record_payload_read)
    for stage, path in (("source", source), ("nonprompt", nonprompt)):
        path.write_bytes(f"tiny {stage} payload".encode())
        sidecar = path.with_name(f"{path.name}.metadata.json")
        artifact = {
            "pkl_basename": path.name,
            "pkl_size_bytes": path.stat().st_size,
            "pkl_sha256": "0" * 64,
        }
        sidecar.write_text(json.dumps({"artifact": artifact, "synthetic": True}), encoding="utf-8")
        with pytest.raises(ValueError, match="artifact sidecar identity mismatch"):
            manifest.capture_receipt(state, block, stage)
        assert f"{stage}_receipt" not in block
        artifact["pkl_sha256"] = actual_digest_file(path)
        sidecar.write_text(json.dumps({"artifact": artifact, "synthetic": True}), encoding="utf-8")
        manifest.capture_receipt(state, block, stage)
        assert block[f"{stage}_receipt"]["sha256"] == artifact["pkl_sha256"]
    assert payload_reads == [str(source), str(source), str(nonprompt), str(nonprompt)]


def _legacy_inline_adoption_fixture(tmp_path):
    output_root = tmp_path / "campaign"
    output_root.mkdir()
    env_file = _write_test_environment(tmp_path)
    units = manifest.load_plan(manifest.DEFAULT_PLAN_PATH)
    revision = f"r1_{manifest.canonical_digest(units)[:12]}"
    specs = manifest.plan_blocks(units, "run2_sr_ext", "fixture-campaign", str(output_root), revision)
    state = {
        "schema_version": 6, "production_profile": "run2_sr_ext",
        "campaign_tag": "fixture-campaign", "output_dir": str(output_root),
        "topeft_git_commit": "not_observed_no_git", "env_file": str(env_file),
        "env_file_sha256": manifest.digest_file(env_file), "environment_fingerprint": f"{9:064d}",
        "environment_mode": "explicit_snapshot", "topcoffea_git_commit": f"{8:040d}",
        "topcoffea_relevant_source_fingerprint": f"{7:064d}",
        "ttgamma_sample_role_policy": "split", "do_systs": True, "do_np": True,
        "region": "SR", "nonprompt_mode": "inline",
        "blocks": [manifest.new_pending_block(spec) for spec in specs],
    }
    manifest.decorate_initial_state(state, "25000")
    files = manifest.scientific_files(state)
    state["schema_version"] = 7
    state["sr_reco_scientific_files"] = files
    state["sr_reco_scientific_contract_sha256"] = manifest.scientific_contract(state, files)
    block = state["blocks"][0]
    command = [
        "./fullR3_run.sh", "-y", "run2", "-t", block["output_tag"], "-s", "25000",
        "--sr", "--hist-vars", *block["histograms"],
        "--category-groups", *block["category_groups"],
        "--ttgamma-sample-role-policy", "split", "--do-systs", "--do-np",
        "--np-postprocess=inline", "-p", str(output_root), "--all-analysis",
        "--env-file", state["env_file"], "--snapshot", "-x", "work_queue",
    ]
    runtime = manifest.source_runtime(state, block, command)
    block.update({
        "status": "source_failed", "source_status": "failed", "source_exit_code": 137,
        "source_signal": 9, "source_command_argv": command, "source_runtime": runtime,
        "source_runtime_attempts": [copy.deepcopy(runtime)],
        "nonprompt_status": "blocked", "source_started_at_utc": "2026-10-15T21:42:06Z",
        "source_ended_at_utc": "2026-10-16T18:25:22Z",
    })
    block["transitions"].extend([
        {"stage": "source", "status": "running", "exit_code": None},
        {"stage": "source", "status": "failed", "exit_code": 137, "signal": 9},
    ])
    source = Path(block["expected_nominal_path"])
    source.write_bytes(b"small published processor output")
    cfg_identities = [
        {"basename": Path(name).name, "content_sha256": digest, "input_kind": "cfg"}
        for name, digest in state["sr_reco_contract_files"].items()
        if name.startswith("input_samples/cfgs/")
    ]
    sidecar = {
        "metadata_schema_version": 2,
        "artifact": {
            "pkl_basename": source.name, "pkl_size_bytes": source.stat().st_size,
            "pkl_sha256": manifest.digest_file(source), "artifact_kind": "processor_output",
            "merged": False,
            "nominal_container_schema_version": 2,
            "nominal_container_layout": "split_sibling_v1",
        },
        "lineage": {"inputs": []}, "histogram_applicability": {
            "contract_version": 1, "analysis_mode": "all",
            "producer_query": "AnalysisProcessor.histogram_fill_is_applicable",
            "families": {family: {} for family in block["histograms"]},
        },
        "production_sample_contract": {
            "contract_version": 1, "resolved_mode": "production",
            "wrapper_identity": "run_cr.sh -> fullR3_run.sh", "cfg_identities": cfg_identities,
        },
    }
    sidecar_path = Path(f"{source}.metadata.json")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    state_path = output_root / ".run2_sr_ext_campaign_state.json"
    lock_path = output_root / ".run2_sr_ext_campaign.lock"
    lock_path.touch()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    assert manifest.check_state(manifest.load_state(state_path))[0]["status"] == "source_failed"
    manifest.activate_stage_policy(state_path, revision)
    activated = state_path.read_bytes()
    manifest.activate_stage_policy(state_path, revision)
    assert state_path.read_bytes() == activated
    state = manifest.load_state(state_path)
    block = state["blocks"][0]
    evidence = {
        "version": 1, "accepted_by": "fixture-owner", "evidence_ref": "fixture-validation-001",
        "validation_basis": "accepted_prior_validation", "campaign_tag": state["campaign_tag"],
        "production_profile": state["production_profile"], "output_dir": state["output_dir"],
        "block_id": block["id"], "plan_revision": revision,
        "source_attempt": block["source_runtime"]["attempt"],
        "source_command_sha256": manifest.canonical_digest(block["source_command_argv"]),
        "source_runtime_sha256": manifest.canonical_digest(block["source_runtime"]),
        "scientific_contract_sha256": block["source_runtime"]["scientific_contract_sha256"],
        "environment_file_sha256": state["env_file_sha256"],
        "source_receipt": manifest.artifact_receipt(source),
        "historical_composite_outcome": {
            "exit_code": 137, "signal": 9, "cause": "OOM_CONFIRMED",
        },
        "source_publication_complete": True,
    }
    evidence_path = tmp_path / "accepted_evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return state_path, evidence_path, evidence, source, sidecar_path, revision


def test_guarded_adoption_preserves_historical_failure_and_source_identity(tmp_path):
    state_path, evidence_path, evidence, source, _, revision = _legacy_inline_adoption_fixture(tmp_path)
    before_source = source.read_bytes()
    before = manifest.load_state(state_path)
    assert before["nonprompt_mode"] == "inline"
    assert before["sr_stage_policy"]["effective_nonprompt_mode"] == "separate"
    assert before["sr_reco_contract_sha256"] == before["sr_stage_policy"]["historical_contract_sha256"]
    assert manifest.adopt_source(state_path, evidence["block_id"], evidence_path, revision) == evidence["source_receipt"]["sha256"]
    adopted_bytes = state_path.read_bytes()
    assert manifest.adopt_source(state_path, evidence["block_id"], evidence_path, revision) == evidence["source_receipt"]["sha256"]
    assert state_path.read_bytes() == adopted_bytes
    reopened = manifest.load_state(state_path)
    block = reopened["blocks"][0]
    assert block["status"] == "source_ready" and block["source_status"] == "ready"
    assert block["source_exit_code"] == 137 and block["source_signal"] == 9
    assert block["source_adoption"]["historical_composite_outcome"] == evidence["historical_composite_outcome"]
    assert block["source_adoption"]["source_only_exit_code"] is None
    assert any(item.get("exit_code") == 137 and item.get("signal") == 9 for item in block["transitions"])
    assert block["source_runtime"] == before["blocks"][0]["source_runtime"]
    assert block["source_receipt"] == evidence["source_receipt"]
    assert source.read_bytes() == before_source
    assert all(item["status"] == "planned" for item in reopened["blocks"][1:])
    assert manifest.check_state(reopened)[0]["status"] == "source_ready"


def test_stage_policy_keeps_historical_hash_and_scientific_inputs_distinct(tmp_path):
    state_path, _, _, _, _, _ = _legacy_inline_adoption_fixture(tmp_path)
    state = manifest.load_state(state_path)
    historical_hash = state["sr_reco_scientific_contract_sha256"]
    current_hash = state["sr_reco_current_scientific_contract_sha256"]
    technical_only = copy.deepcopy(state)
    technical_only["nonprompt_mode"] = "separate"
    assert manifest.scientific_contract(
        technical_only, state["sr_reco_scientific_files"], include_execution_mode=False
    ) == current_hash
    assert manifest.scientific_contract(
        technical_only, state["sr_reco_scientific_files"]
    ) != historical_hash
    with pytest.raises(ValueError, match="historical campaign contract|stage policy"):
        manifest.check_state(technical_only)
    changed_science = copy.deepcopy(state)
    changed_science["do_systs"] = False
    assert manifest.scientific_contract(
        changed_science, state["sr_reco_scientific_files"], include_execution_mode=False
    ) != current_hash
    with pytest.raises(ValueError, match="scientific contract|historical campaign contract"):
        manifest.check_state(changed_science)
    unknown_policy = copy.deepcopy(state)
    unknown_policy["sr_stage_policy"]["version"] = 2
    with pytest.raises(ValueError, match="stage policy"):
        manifest.check_state(unknown_policy)
    legacy_with_policy = copy.deepcopy(state)
    legacy_with_policy["schema_version"] = 7
    with pytest.raises(ValueError, match="stage policy requires SR schema 8"):
        manifest.check_state(legacy_with_policy)


def test_schema_six_stage_activation_is_explicit_and_idempotent(tmp_path):
    output_root = tmp_path / "campaign"
    output_root.mkdir()
    units = manifest.load_plan(manifest.DEFAULT_PLAN_PATH)
    revision = f"r1_{manifest.canonical_digest(units)[:12]}"
    state = manifest.decorate_initial_state({
        "schema_version": 6, "production_profile": "run2_sr_ext",
        "campaign_tag": "legacy-inline", "output_dir": str(output_root),
        "topeft_git_commit": "historical", "env_file": str(tmp_path / "environment.tar.gz"),
        "env_file_sha256": "a" * 64, "environment_fingerprint": "b" * 64,
        "environment_mode": "explicit_snapshot", "topcoffea_git_commit": "c" * 40,
        "topcoffea_relevant_source_fingerprint": "d" * 64,
        "ttgamma_sample_role_policy": "split", "do_systs": True, "do_np": True,
        "region": "SR", "nonprompt_mode": "inline",
        "blocks": [manifest.new_pending_block(spec) for spec in manifest.plan_blocks(
            units, "run2_sr_ext", "legacy-inline", str(output_root), revision
        )],
    }, "25000")
    state_path = output_root / ".run2_sr_ext_campaign_state.json"
    (output_root / ".run2_sr_ext_campaign.lock").touch()
    original = json.dumps(state).encode()
    state_path.write_bytes(original)
    migrated_hash = manifest.activate_stage_policy(state_path, revision)
    activated_bytes = state_path.read_bytes()
    migrated = manifest.load_state(state_path)
    assert migrated["schema_version"] == 8
    assert migrated["nonprompt_mode"] == "inline"
    assert migrated["sr_stage_policy"]["activated_from_schema"] == 6
    assert migrated["sr_stage_policy"]["effective_nonprompt_mode"] == "separate"
    assert migrated["sr_reco_contract_sha256"] == state["sr_reco_contract_sha256"]
    assert migrated_hash == migrated["sr_reco_current_scientific_contract_sha256"]
    assert manifest.check_state(migrated) == migrated["blocks"]
    assert manifest.activate_stage_policy(state_path, revision) == migrated_hash
    assert state_path.read_bytes() == activated_bytes
    state_path.write_bytes(original)
    changed = copy.deepcopy(state)
    changed["do_systs"] = False
    state_path.write_text(json.dumps(changed), encoding="utf-8")
    rejected_bytes = state_path.read_bytes()
    with pytest.raises(ValueError, match="historical campaign contract"):
        manifest.activate_stage_policy(state_path, revision)
    assert state_path.read_bytes() == rejected_bytes


def test_adoption_atomic_replace_failure_preserves_state_bytes(tmp_path, monkeypatch):
    state_path, evidence_path, evidence, _, _, revision = _legacy_inline_adoption_fixture(tmp_path)
    before = state_path.read_bytes()
    def interrupted_replace(*_args):
        raise OSError("simulated atomic replacement interruption")
    monkeypatch.setattr(manifest.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="interruption"):
        manifest.adopt_source(state_path, evidence["block_id"], evidence_path, revision)
    assert state_path.read_bytes() == before
    assert not list(state_path.parent.glob(f".{state_path.name}.tmp.*"))


def test_adopted_legacy_source_resumes_nonprompt_only_for_selected_unit(tmp_path):
    state_path, evidence_path, evidence, source, _, revision = _legacy_inline_adoption_fixture(tmp_path)
    _write_backend(tmp_path / "backend.sh")
    environment = _backend_environment(tmp_path)
    source_bytes = source.read_bytes()
    manifest.adopt_source(state_path, evidence["block_id"], evidence_path, revision)
    before = manifest.load_state(state_path)
    source_runtime = copy.deepcopy(before["blocks"][0]["source_runtime"])
    resumed = _run(
        "run2_sr_ext", state_path.parent, before["campaign_tag"], dry_run=False,
        resume=True, environment=environment,
        extra_args=("--selected-fragment", evidence["block_id"]),
    )
    assert resumed.returncode == 0, resumed.stdout
    assert not (tmp_path / "block_calls.tsv").exists()
    assert (tmp_path / "nonprompt_calls.tsv").read_text().splitlines() == [evidence["block_id"]]
    after = manifest.load_state(state_path)
    block = after["blocks"][0]
    assert block["status"] == "success"
    assert block["source_runtime"] == source_runtime
    assert block["source_receipt"] == before["blocks"][0]["source_receipt"]
    assert block["source_exit_code"] == 137 and block["source_signal"] == 9
    assert block["source_adoption"]["source_only_exit_code"] is None
    assert source.read_bytes() == source_bytes
    assert all(item["status"] == "planned" for item in after["blocks"][1:])
    assert after["campaign_status"] == "blocked_incomplete"
    assert manifest.check_state(after)[0]["status"] == "success"


def test_guarded_adoption_rejects_mismatched_evidence_without_mutation(tmp_path):
    state_path, evidence_path, evidence, source, sidecar_path, revision = _legacy_inline_adoption_fixture(tmp_path)
    original_state = state_path.read_bytes()
    wrong_values = {
        "version": 2, "accepted_by": "", "evidence_ref": "",
        "validation_basis": "unknown", "campaign_tag": "wrong", "production_profile": "wrong",
        "output_dir": str(tmp_path), "block_id": "run2_sr_ext_e_2", "plan_revision": "wrong",
        "source_attempt": 2, "source_command_sha256": "0" * 64,
        "source_runtime_sha256": "0" * 64, "scientific_contract_sha256": "0" * 64,
        "environment_file_sha256": "0" * 64, "source_publication_complete": False,
    }
    for key, value in wrong_values.items():
        altered = copy.deepcopy(evidence)
        altered[key] = value
        evidence_path.write_text(json.dumps(altered), encoding="utf-8")
        with pytest.raises(ValueError):
            manifest.adopt_source(state_path, evidence["block_id"], evidence_path, revision)
        assert state_path.read_bytes() == original_state
    for change in (
        {"source_receipt": {**evidence["source_receipt"], "sha256": "0" * 64}},
        {"historical_composite_outcome": {"exit_code": 0, "signal": None, "cause": "wrong"}},
    ):
        evidence_path.write_text(json.dumps({**evidence, **change}), encoding="utf-8")
        with pytest.raises(ValueError):
            manifest.adopt_source(state_path, evidence["block_id"], evidence_path, revision)
        assert state_path.read_bytes() == original_state
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    original_sidecar = sidecar_path.read_bytes()
    sidecar = json.loads(original_sidecar)
    sidecar["production_sample_contract"]["cfg_identities"][0]["content_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError):
        manifest.adopt_source(state_path, evidence["block_id"], evidence_path, revision)
    assert state_path.read_bytes() == original_state
    sidecar_path.write_bytes(original_sidecar)
    np_path = Path(manifest.load_state(state_path)["blocks"][0]["expected_np_path"])
    np_path.write_bytes(b"collision")
    with pytest.raises(ValueError, match="collision"):
        manifest.adopt_source(state_path, evidence["block_id"], evidence_path, revision)
    assert state_path.read_bytes() == original_state
    np_path.unlink()
    parked = source.with_name("parked_source.pkl.gz")
    source.rename(parked)
    source.symlink_to(parked)
    with pytest.raises(ValueError):
        manifest.adopt_source(state_path, evidence["block_id"], evidence_path, revision)
    assert state_path.read_bytes() == original_state
    source.unlink()
    parked.rename(source)
    in_flight = manifest.load_state(state_path)
    in_flight["blocks"][0]["status"] = "source_running"
    in_flight["blocks"][0]["source_status"] = "running"
    state_path.write_text(json.dumps(in_flight), encoding="utf-8")
    in_flight_bytes = state_path.read_bytes()
    with pytest.raises(ValueError, match="failed active composite source"):
        manifest.adopt_source(state_path, evidence["block_id"], evidence_path, revision)
    assert state_path.read_bytes() == in_flight_bytes
    state_path.write_bytes(original_state)
    evidence_path.unlink()
    with pytest.raises(ValueError, match="evidence path"):
        manifest.adopt_source(state_path, evidence["block_id"], evidence_path, revision)
    assert state_path.read_bytes() == original_state
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    foreign_root = tmp_path / "foreign"
    foreign_root.mkdir()
    foreign_state = foreign_root / state_path.name
    foreign_state.write_bytes(original_state)
    (foreign_root / ".run2_sr_ext_campaign.lock").touch()
    with pytest.raises(ValueError, match="foreign or symbolic"):
        manifest.adopt_source(foreign_state, evidence["block_id"], evidence_path, revision)
    assert foreign_state.read_bytes() == original_state


@pytest.mark.parametrize("profile", ("run2_sr_ext", "run3_sr_ext"))
def test_nonprompt_failure_retries_without_source_replay(tmp_path, profile):
    validation_root = tmp_path / profile
    validation_root.mkdir()
    _write_backend(validation_root / "backend.sh")
    environment = _backend_environment(validation_root)
    environment["SRPLOT009_VALIDATION_SCENARIO"] = "sr_np_failure_once"
    output_root = validation_root / "output"
    tag = f"separate-{profile}"
    selected = f"{profile}_a_1"
    extra = ("--selected-fragment", selected)
    first = _run(profile, output_root, tag, dry_run=False, environment=environment, extra_args=extra)
    assert first.returncode == 1, first.stdout
    failed = _campaign_state(output_root, profile)
    block = failed["blocks"][0]
    source = Path(block["expected_nominal_path"])
    source_bytes = source.read_bytes()
    source_identity = (copy.deepcopy(block["source_receipt"]), copy.deepcopy(block["source_runtime"]),
                       copy.deepcopy(block["source_runtime_attempts"]), block["source_command_argv"])
    assert failed["schema_version"] == 8
    assert failed["sr_stage_policy"]["effective_nonprompt_mode"] == "separate"
    assert block["status"] == "nonprompt_failed" and block["source_status"] == "ready"
    assert block["nonprompt_exit_code"] == 137
    assert manifest.check_state(failed)[0]["status"] == "nonprompt_failed"
    assert all(item["status"] == "planned" for item in failed["blocks"][1:])
    assert len((validation_root / "block_calls.tsv").read_text().splitlines()) == 1
    assert (validation_root / "nonprompt_calls.tsv").read_text().splitlines() == [selected]
    resumed = _run(profile, output_root, tag, dry_run=False, resume=True,
                   environment=environment, extra_args=extra)
    assert resumed.returncode == 0, resumed.stdout
    completed = _campaign_state(output_root, profile)
    block = completed["blocks"][0]
    assert block["status"] == "success" and block["nonprompt_exit_code"] == 0
    assert (block["source_receipt"], block["source_runtime"], block["source_runtime_attempts"],
            block["source_command_argv"]) == source_identity
    assert source.read_bytes() == source_bytes
    assert len((validation_root / "block_calls.tsv").read_text().splitlines()) == 1
    assert (validation_root / "nonprompt_calls.tsv").read_text().splitlines() == [selected, selected]
    assert all(item["status"] == "planned" for item in completed["blocks"][1:])
    assert completed["campaign_status"] == "blocked_incomplete"
    assert manifest.check_state(completed)[0]["status"] == "success"
    skipped = _run(profile, output_root, tag, dry_run=False, resume=True,
                   environment=environment, extra_args=extra)
    assert skipped.returncode == 0, skipped.stdout
    assert len((validation_root / "block_calls.tsv").read_text().splitlines()) == 1
    assert (validation_root / "nonprompt_calls.tsv").read_text().splitlines() == [selected, selected]


@pytest.mark.parametrize("profile", ("run2_sr_ext", "run3_sr_ext"))
def test_legacy_receipt_and_pending_runtime_transition(tmp_path, profile, monkeypatch, capsys):
    git_bin = tmp_path / "reject_git"
    git_bin.mkdir()
    git_command = git_bin / "git"
    git_command.write_text("#!/bin/sh\nprintf 'git attempted\\n' >> \"$GIT_ATTEMPTS\"\nexit 88\n", encoding="utf-8")
    git_command.chmod(0o755)
    git_attempts = tmp_path / "git_attempts.txt"
    monkeypatch.setenv("GIT_ATTEMPTS", str(git_attempts))
    monkeypatch.setenv("PATH", f"{git_bin}{os.pathsep}{os.environ['PATH']}")
    validation_root = tmp_path / profile
    validation_root.mkdir()
    _write_backend(validation_root / "backend.sh")
    environment = _backend_environment(validation_root)
    output_root = validation_root / "output"
    tag = f"mixed-runtime-{profile}"
    first_id = f"{profile}_a_1"
    second_id = f"{profile}_a_2"
    first = _run(
        profile, output_root, tag, dry_run=False, environment=environment,
        extra_args=("--selected-fragment", first_id, "--chunksize", "50000"),
    )
    assert first.returncode == 0, first.stdout
    state_path = output_root / f".{profile}_campaign_state.json"
    legacy = _campaign_state(output_root, profile)
    # Recreate the supported pre-runtime schema from a tiny new-format fixture.
    # The completed artifact and its historical receipt remain unchanged.
    legacy["schema_version"] = 6
    for key in (
        "sr_stage_policy", "sr_reco_current_scientific_contract_sha256",
        "sr_reco_scientific_contract_sha256", "sr_reco_scientific_files",
    ):
        legacy.pop(key)
    completed = legacy["blocks"][0]
    for key in ("source_runtime", "source_runtime_attempts", "receipt_runtime_sha256"):
        completed.pop(key)
    completed["receipt_contract_sha256"] = legacy["sr_reco_contract_sha256"]
    state_path.write_text(json.dumps(legacy), encoding="utf-8")
    legacy_bytes = state_path.read_bytes()
    first_block = copy.deepcopy(legacy["blocks"][0])
    first_artifacts = {
        name: Path(name).read_bytes()
        for output in first_block["expected_outputs"]
        for name in (output, f"{output}.metadata.json")
    }
    head = legacy["topeft_git_commit"]
    monkeypatch.setenv("SRPLOT009_VALIDATION_BACKEND", str(validation_root / "backend.sh"))
    interrupted = copy.deepcopy(legacy)
    interrupted["blocks"][1].update(status="source_running", source_status="running")
    state_path.write_text(json.dumps(interrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="in-flight|ambiguous"):
        manifest.upgrade_runtime_policy(state_path, head)
    state_path.write_bytes(legacy_bytes)
    pending_path = Path(legacy["blocks"][1]["expected_nominal_path"])
    pending_path.write_bytes(b"collision")
    with pytest.raises(ValueError, match="collision"):
        manifest.upgrade_runtime_policy(state_path, head)
    assert state_path.read_bytes() == legacy_bytes
    pending_path.unlink()
    contract = manifest.upgrade_runtime_policy(state_path, head)
    upgraded = _campaign_state(output_root, profile)
    assert upgraded["schema_version"] == 7
    assert upgraded["sr_reco_contract_sha256"] == legacy["sr_reco_contract_sha256"]
    assert upgraded["blocks"][0] == first_block
    assert upgraded["output_dir"] == legacy["output_dir"]
    assert upgraded["campaign_tag"] == legacy["campaign_tag"]
    assert upgraded["active_sr_reco_revision"] == legacy["active_sr_reco_revision"]
    assert len(upgraded["blocks"]) == 12
    assert [block["status"] for block in upgraded["blocks"]] == ["success"] + ["planned"] * 11
    assert upgraded["sr_reco_scientific_contract_sha256"] == contract
    guarded_environment = _guard_accepted_payloads(
        tmp_path, environment, upgraded["blocks"][0]["expected_outputs"]
    )
    pending = _run(
        profile, output_root, tag, dry_run=False, resume=True, environment=guarded_environment,
        extra_args=(
            "--selected-fragment", second_id, "--chunksize", "25000",
            "--wq-staging-base", str(validation_root / "unused_staging"),
            "--wq-quota-root", "/groups/klannon",
            "--wq-free-warning-bytes", "20000000000",
            "--wq-min-free-bytes", "10000000000",
            "--wq-poll-seconds", "5", "--wq-treereduction", "10",
            "--wq-processing-cores", "1", "--wq-accumulating-memory-mb", "80000",
            "--wq-accumulating-cores", "2",
        ),
    )
    assert pending.returncode == 0, pending.stdout
    assert Path(guarded_environment["SR_PAYLOAD_GUARD_MARKER"]).read_text() == "loaded"
    mixed = _campaign_state(output_root, profile)
    assert mixed["blocks"][0] == first_block
    assert all(Path(name).read_bytes() == content for name, content in first_artifacts.items())
    runtime = mixed["blocks"][1]["source_runtime"]
    assert runtime["chunksize"] == 25000
    assert runtime["treereduction"] == 10
    assert runtime["processing_cores_max"] == 1
    assert runtime["accumulating_memory_min_mb"] == 80000
    assert runtime["accumulating_cores_min"] == 2
    assert runtime["plan_revision"] == mixed["active_sr_reco_revision"]
    assert mixed["blocks"][1]["receipt_runtime_sha256"] == manifest.canonical_digest(runtime)
    valid_command = list(mixed["blocks"][1]["source_command_argv"])
    selectable = copy.deepcopy(mixed["blocks"][1])
    selectable["status"] = "planned"
    cases = {
        "do_systs missing": (lambda argv: argv.remove("--do-systs"), "--do-systs"),
        "do_systs duplicate": (lambda argv: argv.append("--do-systs"), "--do-systs"),
        "do_systs malformed": (lambda argv: argv.append("--do-systs=false"), "malformed.*--do-systs"),
        "do_np missing": (lambda argv: argv.remove("--do-np"), "--do-np"),
        "do_np duplicate": (lambda argv: argv.append("--do-np"), "--do-np"),
        "sr missing": (lambda argv: argv.remove("--sr"), "--sr"),
        "all-analysis missing": (lambda argv: argv.remove("--all-analysis"), "--all-analysis"),
        "executor mismatch": (lambda argv: argv.__setitem__(argv.index("work_queue"), "local"), "-x"),
        "executor duplicate": (lambda argv: argv.extend(("-x", "local")), "-x"),
        "snapshot missing": (lambda argv: argv.remove("--snapshot"), "--snapshot"),
        "snapshot duplicate": (lambda argv: argv.append("--snapshot"), "--snapshot"),
        "conflicting analysis selector": (lambda argv: argv.append("--tau-h-analysis"), "conflicting SR"),
    }
    for label, (mutate, expected_error) in cases.items():
        invalid_command = list(valid_command)
        mutate(invalid_command)
        with pytest.raises(ValueError, match=expected_error):
            manifest.source_runtime(mixed, selectable, invalid_command)
    no_np_state = copy.deepcopy(mixed)
    no_np_state["do_np"] = False
    no_np_command = [
        item for item in valid_command
        if item not in {"--do-np", "--defer-np", "--np-postprocess=inline"}
    ]
    assert manifest.source_runtime(no_np_state, selectable, no_np_command)["chunksize"] == 25000
    current_resolved = copy.deepcopy(mixed)
    current_resolved["environment_mode"] = "current_resolved"
    current_command = [item for item in valid_command if item != "--snapshot"]
    manifest.validate_source_command_contract(current_resolved, current_command)
    with pytest.raises(ValueError, match="--snapshot"):
        manifest.validate_source_command_contract(
            current_resolved, [*current_command, "--snapshot"]
        )

    redigested = copy.deepcopy(mixed)
    redigested_block = redigested["blocks"][1]
    redigested_block["source_command_argv"].remove("--do-systs")
    redigested_runtime = copy.deepcopy(redigested_block["source_runtime"])
    redigested_runtime["source_command_sha256"] = manifest.canonical_digest(
        redigested_block["source_command_argv"]
    )
    redigested_block["source_runtime"] = redigested_runtime
    redigested_block["source_runtime_attempts"][-1] = copy.deepcopy(redigested_runtime)
    redigested_block["receipt_runtime_sha256"] = manifest.canonical_digest(
        redigested_runtime
    )
    with pytest.raises(ValueError, match="switch contradicts campaign contract: --do-systs"):
        manifest.check_state(redigested)
    assert manifest.check_state(mixed)[1]["status"] == "success"
    assert manifest.check_state(mixed)[0] == first_block
    assert "source_runtime" not in mixed["blocks"][0]
    assert mixed["sr_reco_chunk_size"] == "50000"
    assert [block["status"] for block in mixed["blocks"]] == ["success", "success"] + ["planned"] * 10
    assert len(manifest.check_state(mixed)) == len(mixed["blocks"])
    accepted_paths = {path for block in mixed["blocks"][:2] for path in block["expected_outputs"]}
    real_digest_file = manifest.digest_file

    def reject_accepted_payload_read(path):
        if str(path) in accepted_paths:
            raise AssertionError(f"accepted PKL content read: {path}")
        return real_digest_file(path)

    with monkeypatch.context() as patcher:
        patcher.setattr(manifest, "digest_file", reject_accepted_payload_read)
        assert manifest.check_state(mixed)[1]["status"] == "success"
        rendered = tmp_path / f"{profile}_active.tsv"
        patcher.setattr(sys, "argv", [
            "sr_reco_manifest.py", "render-active", str(state_path), str(rendered), "25000",
        ])
        manifest.main()
        assert len(rendered.read_text(encoding="utf-8").splitlines()) == 12
        patcher.setattr(sys, "argv", ["sr_reco_manifest.py", "consumer-summary", str(state_path)])
        manifest.main()
        summary = json.loads(capsys.readouterr().out.splitlines()[-1])
        assert [unit["status"] for unit in summary["units"][:2]] == ["success", "success"]
        assert summary["target_coverage_count"] == 849
    guarded_environment["SR_ACCEPTED_PKLS"] = os.pathsep.join(sorted(accepted_paths))
    repeated = _run(profile, output_root, tag, dry_run=True, resume=True,
                    environment=guarded_environment)
    assert repeated.returncode == 0, repeated.stdout
    assert f"Skipping validated {profile} block: {first_id}" in repeated.stdout
    assert f"Skipping validated {profile} block: {second_id}" in repeated.stdout
    assert _campaign_state(output_root, profile)["blocks"][0] == first_block
    assert not git_attempts.exists()

    wrong_selection = list(mixed["blocks"][1]["source_command_argv"])
    wrong_selection[wrong_selection.index("--category-groups") + 1] = "3l_fwd"
    with pytest.raises(ValueError, match="source selection"):
        manifest.source_runtime(upgraded, upgraded["blocks"][1], wrong_selection)
    changed_command = copy.deepcopy(mixed)
    changed_command["blocks"][1]["source_command_argv"] = wrong_selection
    with pytest.raises(ValueError, match="runtime receipt"):
        manifest.check_state(changed_command)
    with pytest.raises(ValueError, match="scientific/input"):
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(manifest, "scientific_files", lambda state: {"changed": "input"})
            manifest.check_state(mixed)
    bound_input = next(name for name in upgraded["sr_reco_scientific_files"]
                       if name.startswith("input_samples/"))
    real_digest_file = manifest.digest_file
    with pytest.raises(ValueError, match="scientific/input"):
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(
                manifest, "digest_file",
                lambda path: "0" * 64 if str(path).endswith(bound_input) else real_digest_file(path),
            )
            manifest.check_state(mixed)
    changed_receipt = copy.deepcopy(mixed)
    changed_receipt["blocks"][1]["receipt_runtime_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runtime receipt"):
        manifest.check_state(changed_receipt)
    second_sidecar = Path(f"{mixed['blocks'][1]['expected_nominal_path']}.metadata.json")
    sidecar_bytes = second_sidecar.read_bytes()
    second_sidecar.write_bytes(b"{}")
    with pytest.raises(ValueError, match="sidecar"):
        manifest.check_state(mixed)
    second_sidecar.write_bytes(sidecar_bytes)
    assert len(manifest.check_state(manifest.load_state(state_path))) == 12

    source_path = Path(mixed["blocks"][1]["expected_nominal_path"])
    source_bytes = source_path.read_bytes()
    source_receipt = mixed["blocks"][1]["source_receipt"]
    for key, value, error in (
        ("sha256", "0" * 64, "receipt|identity"),
        ("size_bytes", source_receipt["size_bytes"] + 1, "receipt"),
    ):
        stale = copy.deepcopy(mixed)
        stale["blocks"][1]["source_receipt"][key] = value
        with pytest.raises(ValueError, match=error):
            manifest.check_state(stale, require_contract=False)
    for key, value, error in (
        ("receipt_contract_sha256", "0" * 64, "contract/revision"),
        ("receipt_plan_revision", "wrong_revision", "contract/revision"),
    ):
        stale = copy.deepcopy(mixed)
        stale["blocks"][1][key] = value
        with pytest.raises(ValueError, match=error):
            manifest.check_state(stale, require_contract=False)

    source_path.unlink()
    try:
        with pytest.raises(ValueError, match="required artifact"):
            manifest.check_state(mixed, require_contract=False)
        source_path.symlink_to(second_sidecar)
        with pytest.raises(ValueError, match="required artifact"):
            manifest.check_state(mixed, require_contract=False)
    finally:
        source_path.unlink(missing_ok=True)
        source_path.write_bytes(source_bytes)
    try:
        source_path.write_bytes(source_bytes + b"x")
        with pytest.raises(ValueError, match="receipt|identity"):
            manifest.check_state(mixed, require_contract=False)
    finally:
        source_path.write_bytes(source_bytes)

    second_sidecar.unlink()
    try:
        with pytest.raises(ValueError, match="required artifact"):
            manifest.check_state(mixed, require_contract=False)
    finally:
        second_sidecar.write_bytes(sidecar_bytes)
    try:
        second_sidecar.write_text("not JSON", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            manifest.check_state(mixed, require_contract=False)
    finally:
        second_sidecar.write_bytes(sidecar_bytes)

    nonprompt_sidecar = Path(f"{mixed['blocks'][1]['expected_np_path']}.metadata.json")
    nonprompt_sidecar_bytes = nonprompt_sidecar.read_bytes()
    broken_lineage = json.loads(nonprompt_sidecar_bytes)
    broken_lineage["synthetic"] = False
    broken_lineage["lineage"] = {"inputs": [{"pkl_sha256": "0" * 64}]}
    try:
        nonprompt_sidecar.write_text(json.dumps(broken_lineage), encoding="utf-8")
        stale = copy.deepcopy(mixed)
        stale["blocks"][1]["nonprompt_receipt"]["sidecar_sha256"] = manifest.digest_file(nonprompt_sidecar)
        with pytest.raises(ValueError, match="lineage does not bind source"):
            manifest.check_state(stale, require_contract=False)
    finally:
        nonprompt_sidecar.write_bytes(nonprompt_sidecar_bytes)

    mutated = bytes([source_bytes[0] ^ 1]) + source_bytes[1:]
    try:
        source_path.write_bytes(mutated)
        assert source_path.stat().st_size == len(source_bytes)
        assert manifest.check_state(mixed, require_contract=False)[1]["status"] == "success"
        with pytest.raises(ValueError, match="identity mismatch"):
            manifest.check_state(mixed, require_contract=False, verify_payload_sha256=True)
        with monkeypatch.context() as patcher:
            patcher.setattr(sys, "argv", ["sr_reco_manifest.py", "audit-integrity", str(state_path)])
            with pytest.raises(SystemExit) as audit_error:
                manifest.main()
            assert audit_error.value.code == 1
    finally:
        source_path.write_bytes(source_bytes)
    assert manifest.check_state(mixed, require_contract=False, verify_payload_sha256=True)[1]["status"] == "success"
    with monkeypatch.context() as patcher:
        patcher.setattr(sys, "argv", ["sr_reco_manifest.py", "audit-integrity", str(state_path)])
        manifest.main()
    assert "Full SR payload integrity audit passed" in capsys.readouterr().out


def test_historical_run3_source_ready_finishes_separate_np_after_upgrade(tmp_path, monkeypatch, capsys):
    git_bin = tmp_path / "reject_git"
    git_bin.mkdir()
    git_command = git_bin / "git"
    git_command.write_text("#!/bin/sh\nprintf 'git attempted\\n' >> \"$GIT_ATTEMPTS\"\nexit 88\n", encoding="utf-8")
    git_command.chmod(0o755)
    git_attempts = tmp_path / "git_attempts.txt"
    monkeypatch.setenv("GIT_ATTEMPTS", str(git_attempts))
    monkeypatch.setenv("PATH", f"{git_bin}{os.pathsep}{os.environ['PATH']}")
    profile = "run3_sr_ext"
    validation_root = tmp_path / profile
    validation_root.mkdir()
    _write_backend(validation_root / "backend.sh")
    environment = _backend_environment(validation_root)
    output_root = validation_root / "output"
    tag = "historical-run3-source-ready"
    block_id = f"{profile}_a_1"
    first = _run(profile, output_root, tag, dry_run=False, environment=environment,
                 extra_args=("--selected-fragment", block_id, "--chunksize", "50000"))
    assert first.returncode == 0, first.stdout
    state_path = output_root / f".{profile}_campaign_state.json"
    state = manifest.load_state(state_path)
    block = state["blocks"][0]
    source_receipt = copy.deepcopy(block["source_receipt"])
    source_bytes = Path(block["expected_nominal_path"]).read_bytes()
    source_sidecar = Path(f"{block['expected_nominal_path']}.metadata.json").read_bytes()
    Path(block["expected_np_path"]).unlink()
    Path(f"{block['expected_np_path']}.metadata.json").unlink()
    block["status"] = "source_ready"
    block["nonprompt_status"] = "planned"
    block["nonprompt_exit_code"] = None
    block.pop("nonprompt_receipt")
    for key in ("nonprompt_command_argv", "nonprompt_started_at_utc", "nonprompt_ended_at_utc",
                "nonprompt_duration_seconds", "nonprompt_signal"):
        block[key] = None
    block["transitions"] = block["transitions"][:3]
    block["last_transition_utc"] = block["transitions"][-1]["timestamp_utc"]
    block["last_transition_detail"] = block["transitions"][-1]["detail"]
    block["expected_output_readback"][1].update(
        exists=False, regular_file=False, size_bytes=None, nonempty=False
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    assert manifest.check_state(state)[0]["status"] == "source_ready"
    old_digest_file = manifest.digest_file
    with monkeypatch.context() as patcher:
        patcher.setattr(
            manifest, "digest_file",
            lambda path: (_ for _ in ()).throw(AssertionError("accepted source PKL reread"))
            if str(path) == block["expected_nominal_path"] else old_digest_file(path),
        )
        assert manifest.check_state(state)[0]["status"] == "source_ready"
    manifest.upgrade_runtime_policy(state_path, state["topeft_git_commit"])
    upgraded = manifest.load_state(state_path)
    assert upgraded["blocks"][0] == block
    assert upgraded["blocks"][0]["receipt_contract_sha256"] == state["sr_reco_contract_sha256"]
    with monkeypatch.context() as patcher:
        patcher.setattr(
            manifest, "digest_file",
            lambda path: (_ for _ in ()).throw(AssertionError("accepted source PKL reread"))
            if str(path) == block["expected_nominal_path"] else old_digest_file(path),
        )
        rendered = tmp_path / "run3_source_ready_active.tsv"
        patcher.setattr(sys, "argv", [
            "sr_reco_manifest.py", "render-active", str(state_path), str(rendered), "25000",
        ])
        manifest.main()
        assert len(rendered.read_text(encoding="utf-8").splitlines()) == 12
        patcher.setattr(sys, "argv", ["sr_reco_manifest.py", "consumer-summary", str(state_path)])
        manifest.main()
        summary = json.loads(capsys.readouterr().out.splitlines()[-1])
        assert summary["units"][0]["status"] == "source_ready"
    nonprompt_path = Path(block["expected_np_path"])
    nonprompt_path.write_bytes(b"premature nonprompt")
    try:
        with pytest.raises(ValueError, match="ambiguous deferred nonprompt output"):
            manifest.check_state(upgraded)
    finally:
        nonprompt_path.unlink()
    guarded_environment = _guard_accepted_payloads(
        tmp_path, environment, [block["expected_nominal_path"]]
    )
    resumed = _run(profile, output_root, tag, dry_run=False, resume=True,
                   environment=guarded_environment, extra_args=("--selected-fragment", block_id))
    assert resumed.returncode == 0, resumed.stdout
    assert Path(guarded_environment["SR_PAYLOAD_GUARD_MARKER"]).read_text() == "loaded"
    completed = manifest.load_state(state_path)
    completed_block = completed["blocks"][0]
    assert completed_block["status"] == "success"
    assert completed_block["source_receipt"] == source_receipt
    assert completed_block["receipt_contract_sha256"] == state["sr_reco_contract_sha256"]
    assert "source_runtime" not in completed_block
    assert completed_block["nonprompt_command_argv"]
    assert Path(block["expected_nominal_path"]).read_bytes() == source_bytes
    assert Path(f"{block['expected_nominal_path']}.metadata.json").read_bytes() == source_sidecar
    assert manifest.check_state(completed)[0]["status"] == "success"
    guarded_environment["SR_ACCEPTED_PKLS"] = os.pathsep.join(completed_block["expected_outputs"])
    repeated = _run(profile, output_root, tag, dry_run=False, resume=True,
                    environment=guarded_environment, extra_args=("--selected-fragment", block_id))
    assert repeated.returncode == 0, repeated.stdout
    assert f"Skipping validated {profile} block: {block_id}" in repeated.stdout
    assert not git_attempts.exists()


@pytest.mark.parametrize("profile", ("run2_sr_ext", "run3_sr_ext"))
def test_two_completed_units_replan_in_same_native_campaign(tmp_path, profile):
    validation_root = tmp_path / profile
    validation_root.mkdir()
    _write_backend(validation_root / "backend.sh")
    environment = _backend_environment(validation_root)
    output_root = validation_root / "output"
    tag = f"synthetic-{profile}"
    old_ids = [f"{profile}_a_1", f"{profile}_a_2"]
    for index, block_id in enumerate(old_ids):
        result = _run(
            profile, output_root, tag, dry_run=False, resume=index > 0,
            environment=environment, extra_args=("--selected-fragment", block_id),
        )
        assert result.returncode == 0, result.stdout
    state = _campaign_state(output_root, profile)
    assert len(state["sr_reco_plan_revisions"][0]["blocks"]) == 12
    first_revision = state["active_sr_reco_revision"]
    assert [block["status"] for block in state["blocks"][:2]] == ["success", "success"]
    preserved = {
        path: Path(path).read_bytes()
        for block in state["blocks"][:2]
        for output in block["expected_outputs"]
        for path in (output, f"{output}.metadata.json")
    }
    preserved_receipts = copy.deepcopy([
        (block["source_receipt"], block["nonprompt_receipt"])
        for block in state["blocks"][:2]
    ])
    calls_before = (validation_root / "block_calls.tsv").read_text()

    replacement = {"units": _replacement_units()}
    plan_path = validation_root / "pending.json"
    plan_path.write_text(json.dumps(replacement), encoding="utf-8")
    activation = _run(
        profile, output_root, tag, dry_run=False, resume=True,
        environment=environment, supply_env_file=False,
        extra_args=(
            "--activate-pending-plan", str(plan_path),
            "--expected-plan-revision", first_revision,
        ),
    )
    assert activation.returncode == 0, activation.stdout
    revised = _campaign_state(output_root, profile)
    assert revised["output_dir"] == str(output_root)
    assert len(revised["sr_reco_plan_revisions"]) == 2
    assert len(revised["blocks"]) == 11
    assert revised["active_sr_reco_revision"] != first_revision
    assert [block["plan_revision"] for block in revised["blocks"][:2]] == [first_revision] * 2
    assert [(block["source_receipt"], block["nonprompt_receipt"]) for block in revised["blocks"][:2]] == preserved_receipts
    assert all(Path(path).read_bytes() == data for path, data in preserved.items())
    assert (validation_root / "block_calls.tsv").read_text() == calls_before
    assert len(manifest.check_state(revised)) == 11

    resumed = _run(
        profile, output_root, tag, dry_run=False, resume=True,
        environment=environment,
    )
    assert resumed.returncode == 0, resumed.stdout
    final = _campaign_state(output_root, profile)
    summary = subprocess.run(
        ["python", str(RUN_DIRECTORY / "sr_reco_manifest.py"), "consumer-summary", str(output_root / f".{profile}_campaign_state.json")],
        cwd=RUN_DIRECTORY, text=True, capture_output=True, check=True,
    )
    consumed = json.loads(summary.stdout)
    with (output_root / "campaign_summary.tsv").open(encoding="utf-8", newline="") as handle:
        report_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert consumed["complete"] is True
    assert consumed["target_coverage_count"] == len(manifest.target_coverage())
    assert Counter(item["revision"] for item in consumed["units"]) == {
        first_revision: 2, final["active_sr_reco_revision"]: 9,
    }
    assert len({block["expected_nominal_path"] for block in final["blocks"]}) == 11
    assert {row["block_id"]: row["plan_revision"] for row in report_rows} == {
        block["id"]: block["plan_revision"] for block in final["blocks"]
    }
    assert all(Path(path).read_bytes() == data for path, data in preserved.items())
    calls_after = (validation_root / "block_calls.tsv").read_text()
    for block_id in old_ids:
        assert calls_after.count(f"{block_id}\t") == calls_before.count(f"{block_id}\t")
    for block in final["blocks"][2:]:
        assert calls_after.count(f"{block['id']}\t") == 1
    if profile == "run3_sr_ext":
        nonprompt_calls = (validation_root / "nonprompt_calls.tsv").read_text().splitlines()
        assert Counter(nonprompt_calls) == {block["id"]: 1 for block in final["blocks"]}


def test_rejected_replans_preserve_authoritative_state(tmp_path, monkeypatch):
    profile = "run2_sr_ext"
    validation_root = tmp_path / "rejected"
    validation_root.mkdir()
    _write_backend(validation_root / "backend.sh")
    output_root = validation_root / "output"
    tag = "synthetic-rejected"
    result = _run(
        profile, output_root, tag, dry_run=False,
        environment=_backend_environment(validation_root),
        extra_args=("--selected-fragment", f"{profile}_a_1"),
    )
    assert result.returncode == 0, result.stdout
    state_path = output_root / f".{profile}_campaign_state.json"
    original = state_path.read_bytes()
    state = json.loads(original)
    overlapping_state = copy.deepcopy(state)
    overlapping_state["blocks"][1]["expected_outputs"][0] = state["blocks"][0]["expected_np_path"]
    with pytest.raises(ValueError, match="output path collision"):
        manifest.check_state(overlapping_state, require_contract=False)
    valid = _replacement_units()
    # The second original unit is still pending here, so this candidate has a gap.
    with pytest.raises(ValueError, match="missing"):
        manifest.activate(state_path, _write_plan(validation_root, valid), state["active_sr_reco_revision"])
    assert state_path.read_bytes() == original
    with pytest.raises(ValueError, match="stale"):
        manifest.activate(state_path, _write_plan(validation_root, manifest.load_plan(manifest.DEFAULT_PLAN_PATH)[1:]), "r0_stale")
    assert state_path.read_bytes() == original
    complete_pending = [
        {**unit, "id": f"v2_{unit['id']}"}
        for unit in manifest.load_plan(manifest.DEFAULT_PLAN_PATH)[1:]
    ]
    complete_plan = _write_plan(validation_root, complete_pending)
    with monkeypatch.context() as patcher:
        patcher.setattr(manifest, "contract_files", lambda profile: {"changed": "contract"})
        with pytest.raises(ValueError, match="contract"):
            manifest.activate(state_path, complete_plan, state["active_sr_reco_revision"])
    assert state_path.read_bytes() == original
    with monkeypatch.context() as patcher:
        patcher.setattr(manifest, "atomic_json", lambda path, value: (_ for _ in ()).throw(OSError("injected before replace")))
        with pytest.raises(OSError, match="injected"):
            manifest.activate(state_path, complete_plan, state["active_sr_reco_revision"])
    assert state_path.read_bytes() == original
    next_revision = f"r2_{manifest.canonical_digest(complete_pending)[:12]}"
    collision = manifest.plan_blocks(
        complete_pending, profile, tag, output_root, next_revision,
    )[0]["expected_nominal_path"]
    Path(collision).write_bytes(b"collision")
    with pytest.raises(ValueError, match="replacement output collides"):
        manifest.activate(state_path, complete_plan, state["active_sr_reco_revision"])
    assert state_path.read_bytes() == original
    Path(collision).unlink()
    for occupied_path in (collision, f"{collision}.metadata.json"):
        Path(occupied_path).symlink_to(validation_root / "missing-target")
        with pytest.raises(ValueError, match="replacement output collides"):
            manifest.activate(state_path, complete_plan, state["active_sr_reco_revision"])
        assert state_path.read_bytes() == original
        Path(occupied_path).unlink()
    lock_path = output_root / f".{profile}_campaign.lock"
    lock_bytes = b"existing lock contents must survive\n"
    lock_path.write_bytes(lock_bytes)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        blocked = _run(
            profile, output_root, tag, dry_run=False, resume=True,
            environment=_backend_environment(validation_root), supply_env_file=False,
            extra_args=(
                "--activate-pending-plan", str(complete_plan),
                "--expected-plan-revision", state["active_sr_reco_revision"],
            ),
        )
        assert blocked.returncode != 0
        assert "writer lock is held" in blocked.stdout
    assert lock_path.read_bytes() == lock_bytes
    lock_path.unlink()
    lock_target = validation_root / "lock-target"
    lock_target.write_bytes(b"lock target remains unchanged\n")
    lock_path.symlink_to(lock_target)
    blocked_symlink = _run(
        profile, output_root, tag, dry_run=False, resume=True,
        environment=_backend_environment(validation_root), supply_env_file=False,
        extra_args=(
            "--activate-pending-plan", str(complete_plan),
            "--expected-plan-revision", state["active_sr_reco_revision"],
        ),
    )
    assert blocked_symlink.returncode != 0
    assert "lock path must not be a symbolic link" in blocked_symlink.stdout
    assert lock_path.is_symlink()
    assert lock_target.read_bytes() == b"lock target remains unchanged\n"
    assert state_path.read_bytes() == original
    overlap = copy.deepcopy(manifest.load_plan(manifest.DEFAULT_PLAN_PATH))
    overlap[1]["histograms"].append("invmass")
    with pytest.raises(ValueError, match="coverage partition"):
        manifest.verify_partition(overlap, manifest.target_coverage())
    corrupted = state["blocks"][0]["expected_nominal_path"]
    Path(corrupted).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity mismatch"):
        manifest.activate(state_path, complete_plan, state["active_sr_reco_revision"])
    assert state_path.read_bytes() == original
    Path(corrupted).write_bytes(b"synthetic source\n")
    original_atomic_json = manifest.atomic_json

    def committed_then_interrupted(path, value):
        original_atomic_json(path, value)
        raise OSError("injected after replace")

    with monkeypatch.context() as patcher:
        patcher.setattr(manifest, "atomic_json", committed_then_interrupted)
        with pytest.raises(OSError, match="after replace"):
            manifest.activate(state_path, complete_plan, state["active_sr_reco_revision"])
    published = manifest.load_state(state_path)
    assert published["active_sr_reco_revision"] != state["active_sr_reco_revision"]
    assert manifest.check_state(published)


def _synthetic_sidecar(path):
    path = Path(path)
    path.write_bytes(f"synthetic {path.name}".encode())
    sidecar = {
        "artifact": {
            "pkl_basename": path.name,
            "pkl_size_bytes": path.stat().st_size,
            "pkl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "synthetic": True,
    }
    Path(f"{path}.metadata.json").write_text(json.dumps(sidecar), encoding="utf-8")


def test_run3_verified_source_can_remain_deferred_through_replan(tmp_path):
    profile = "run3_sr_ext"
    plan = manifest.load_plan(manifest.DEFAULT_PLAN_PATH)
    first_revision = f"r1_{manifest.canonical_digest(plan)[:12]}"
    specs = manifest.plan_blocks(plan, profile, "deferred", tmp_path, first_revision)
    state = {
        "schema_version": 6, "production_profile": profile,
        "campaign_tag": "deferred", "output_dir": str(tmp_path),
        "topeft_git_commit": "a" * 40, "env_file": str(tmp_path / "archive"),
        "env_file_sha256": "b" * 64, "environment_fingerprint": "c" * 64,
        "environment_mode": "explicit_snapshot",
        "topcoffea_git_commit": "d" * 40,
        "topcoffea_relevant_source_fingerprint": "e" * 64,
        "ttgamma_sample_role_policy": "split", "do_systs": True,
        "do_np": True, "region": "SR", "nonprompt_mode": "separate",
        "blocks": [manifest.new_pending_block(spec) for spec in specs],
    }
    state_path = tmp_path / ".run3_sr_ext_campaign_state.json"
    manifest.atomic_json(state_path, state)
    manifest.initialize_state(state_path, "100000")
    state = manifest.load_state(state_path)
    for block in state["blocks"][:2]:
        _synthetic_sidecar(block["expected_nominal_path"])
        block["status"] = "source_ready"
        block["source_status"] = "ready"
        block["source_exit_code"] = 0
        block["nonprompt_status"] = "planned"
        block["transitions"].append({
            "stage": "source", "status": "ready", "exit_code": 0,
            "detail": "source_child_exit_zero_expected_source_present",
        })
        manifest.capture_receipt(state, block, "source")
    manifest.atomic_json(state_path, state)
    pending = _replacement_units()
    revision = manifest.activate(
        state_path, _write_plan(tmp_path, pending), first_revision,
    )
    revised = manifest.load_state(state_path)
    assert revised["active_sr_reco_revision"] == revision
    assert [block["status"] for block in revised["blocks"][:2]] == ["source_ready", "source_ready"]
    assert not any(Path(block["expected_np_path"]).exists() for block in revised["blocks"][:2])
    assert manifest.check_state(revised)
    old = copy.deepcopy(revised)
    old.pop("sr_reco_plan_revisions")
    with pytest.raises(ValueError, match="revision history"):
        manifest.check_state(old)
    changed_target = copy.deepcopy(revised)
    changed_target["sr_reco_target_coverage"].pop()
    with pytest.raises(ValueError, match="desired physical coverage"):
        manifest.check_state(changed_target)
    for status, source_status, nonprompt_status in (
        ("source_running", "running", "blocked"),
        ("source_failed", "failed", "blocked"),
        ("nonprompt_running", "ready", "running"),
    ):
        ambiguous = copy.deepcopy(revised)
        pending_block = ambiguous["blocks"][-1]
        pending_block["status"] = status
        pending_block["source_status"] = source_status
        pending_block["nonprompt_status"] = nonprompt_status
        blocked_path = tmp_path / f"{status}.json"
        manifest.atomic_json(blocked_path, ambiguous)
        old_bytes = blocked_path.read_bytes()
        with pytest.raises(ValueError, match="in-flight, failed or ambiguous"):
            manifest.activate(blocked_path, _write_plan(tmp_path, pending), revision)
        assert blocked_path.read_bytes() == old_bytes


def test_plan_rejects_invalid_units_and_pair_sets(tmp_path):
    plan = manifest.load_plan(manifest.DEFAULT_PLAN_PATH)
    duplicate_id = copy.deepcopy(plan)
    duplicate_id[1]["id"] = duplicate_id[0]["id"]
    with pytest.raises(ValueError, match="unique"):
        manifest.load_plan(_write_plan(tmp_path, duplicate_id))
    unknown_axis = copy.deepcopy(plan)
    unknown_axis[0]["histograms"] = ["bjetpt0"]
    with pytest.raises(ValueError, match="unknown"):
        manifest.load_plan(_write_plan(tmp_path, unknown_axis))
    unknown_group = copy.deepcopy(plan)
    unknown_group[0]["category_groups"] = ["invented_sr"]
    with pytest.raises(ValueError, match="unknown"):
        manifest.load_plan(_write_plan(tmp_path, unknown_group))
    inapplicable = copy.deepcopy(plan)
    inapplicable[4]["histograms"] = ["fwd0pt"]
    with pytest.raises(ValueError, match="inapplicable"):
        manifest.load_plan(_write_plan(tmp_path, inapplicable))
    missing = copy.deepcopy(plan[:-1])
    with pytest.raises(ValueError, match="missing"):
        manifest.verify_partition(missing, manifest.target_coverage())
    overlapping = copy.deepcopy(plan)
    overlapping.append({**copy.deepcopy(plan[0]), "id": "another_a_1"})
    with pytest.raises(ValueError, match="duplicate"):
        manifest.verify_partition(overlapping, manifest.target_coverage())


def test_plan_blocks_rejects_overlapping_source_and_nonprompt_paths(tmp_path):
    units = copy.deepcopy(manifest.load_plan(manifest.DEFAULT_PLAN_PATH)[:2])
    units[0]["id"] = "x"
    units[1]["id"] = "x_np"
    with pytest.raises(ValueError, match="output path collision"):
        manifest.plan_blocks(units, "run2_sr_ext", "collision", tmp_path, "r1_test")


def test_run_cr_preoutput_rejects_dangling_output_symlink(tmp_path):
    profile = "run2_sr_ext"
    validation_root = tmp_path / "preoutput-symlink"
    validation_root.mkdir()
    _write_backend(validation_root / "backend.sh")
    output_root = validation_root / "output"
    tag = "synthetic-preoutput-symlink"
    initial = _run(
        profile, output_root, tag, dry_run=False,
        environment=_backend_environment(validation_root),
        extra_args=("--selected-fragment", f"{profile}_a_1"),
    )
    assert initial.returncode == 0, initial.stdout
    state_path = output_root / f".{profile}_campaign_state.json"
    state = manifest.load_state(state_path)
    pending = next(block for block in state["blocks"] if block["status"] == "planned")
    Path(pending["expected_nominal_path"]).symlink_to(validation_root / "missing-pkl")
    lock_path = output_root / f".{profile}_campaign.lock"
    lock_bytes = b"second lock acquisition must preserve bytes\n"
    lock_path.write_bytes(lock_bytes)
    result = _run(
        profile, output_root, tag, dry_run=False, resume=True,
        environment=_backend_environment(validation_root), supply_env_file=False,
        extra_args=("--selected-fragment", pending["id"]),
    )
    assert result.returncode != 0
    assert "expected output path already exists" in result.stdout
    assert lock_path.read_bytes() == lock_bytes


def _write_plan(directory, units):
    path = directory / "candidate.json"
    path.write_text(json.dumps({"units": units}), encoding="utf-8")
    return path


def test_axis_and_locality_contract():
    from topeft.modules.axes import info

    target = manifest.target_coverage()
    assert len(target) == len({manifest.coverage_key(item) for item in target})
    assert len(target) == 849  # observed current channel expansion, not a runtime constant
    assert all(manifest.applicable(item[:3], item[3]) for item in target)
    assert set(item[3] for item in target) == manifest.VARIABLES
    assert manifest.verify_partition(manifest.load_plan(manifest.DEFAULT_PLAN_PATH), target)
    assert info["j0pt"]["processing"] == info["b0pt"]["processing"]
    assert info["j0pt"]["label"] != info["b0pt"]["label"]
    default = manifest.load_plan(manifest.DEFAULT_PLAN_PATH)
    eleven = copy.deepcopy(default)
    eleven[1]["histograms"] += eleven[2]["histograms"]
    eleven.pop(2)
    assert manifest.verify_partition(eleven, target)

    def bin_proxy(unit):
        total = 0
        for pair in manifest.unit_coverage(unit):
            processing = info[pair[3]]["processing"]
            total += processing.get("bins", len(processing.get("edges", [])) - 1) + 2
        return total

    assert max(map(bin_proxy, default)) < max(map(bin_proxy, eleven))


def test_user_supplied_full_plan_selects_one_unit_without_output(tmp_path):
    profile = "run2_sr_ext"
    units = manifest.load_plan(manifest.DEFAULT_PLAN_PATH)
    units[1]["histograms"] += units[2]["histograms"]
    units.pop(2)
    plan_path = _write_plan(tmp_path, units)
    validation_root = tmp_path / "custom"
    validation_root.mkdir()
    _write_backend(validation_root / "backend.sh")
    output_root = validation_root / "output"
    result = _run(
        profile, output_root, "custom", dry_run=True,
        environment=_backend_environment(validation_root),
        extra_args=(
            "--sr-plan", str(plan_path),
            "--selected-fragment", f"{profile}_a_2",
        ),
    )
    assert result.returncode == 0, result.stdout
    assert result.stdout.count("production_block_id:") == 1
    assert "Variables: l0conept j0pt b0pt" in result.stdout
    assert not output_root.exists()
