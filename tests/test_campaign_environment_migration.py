"""Attempt-bound migrations using the native engine and maintained inert backend."""

import copy
import fcntl
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from analysis.topeft_run2 import campaign_manifest as manifest
from topcoffea.modules import remote_environment
from test_campaign_profile_handoff import backend, launch, state_path


def archive_fixture(path, content):
    with tarfile.open(path, "w:gz") as handle:
        member = tarfile.TarInfo("synthetic_module.py")
        member.size = len(content)
        handle.addfile(member, io.BytesIO(content))
    remote_environment.write_archive_manifest(str(path), {
        "environment_fingerprint": "0" * 63 + "9", "python_version": "3.9",
        "resolved_environment_spec": {}, "resolved_environment_spec_fingerprint": "a" * 64,
        "editable_packages": [],
    })
    return path


@pytest.fixture
def new_archive(tmp_path):
    return archive_fixture(tmp_path / "new_environment.tar.gz", b"new worker bytes\n")


def pending_campaign(tmp_path, profile="run2_prod", root=None):
    """Native plan and scientific identity; no alternate lifecycle implementation."""
    scope = profile.split("_")[0]
    years = ["run2"] if scope == "run2" else list(manifest.RUN3_YEARS)
    units = [{"id": name, "years": years, "category_groups": [group], "histograms": ["ptll"]}
             for name, group in [("a", "3l_m_offZ"), ("b", "3l_p_offZ")]]
    root = root or tmp_path / "output"
    root.mkdir(parents=True)
    specs = manifest.make_specs(units, profile, "synthetic", str(root), manifest.make_revision(1, units))
    old_archive = archive_fixture(tmp_path / (profile + "_old.tar.gz"), b"old worker bytes\n")
    provenance = {"environment_file": str(old_archive), "environment_file_sha256": manifest.digest_file(old_archive),
                  "environment_fingerprint": "0" * 63 + "9", "environment_mode": "explicit_snapshot",
                  "topeft_git_commit": "c" * 40, "topcoffea_git_commit": "d" * 40,
                  "topcoffea_relevant_source_fingerprint": "e" * 64}
    return manifest.new_state(specs, manifest.partition_coverage(units, profile),
                              {"production_profile": profile, "campaign_tag": "synthetic", "output_dir": str(root)},
                              manifest.profile_contract(profile), provenance)


def requested(state, archive):
    return dict(state["provenance"]["last_resolved"], environment_file=str(archive),
                environment_file_sha256=manifest.digest_file(archive),
                environment_fingerprint="0" * 63 + "9", environment_mode="explicit_snapshot")


def source_command(state, unit):
    command = ["run_analysis.py", "-y", *unit["years"], "--category-groups", *unit["category_groups"],
            "--hist-vars", *unit["histograms"], "-t", unit["output_tag"],
            "-p", state["campaign_identity"]["output_dir"],
            "--env-file", state["provenance"]["last_resolved"]["environment_file"],
            "--ttgamma-sample-role-policy", "split", "-s", "25000", "--do-systs", "--do-np",
            "--defer-np", "--all-analysis", "-x", "work_queue", "--sr"]
    if state["provenance"]["last_resolved"]["environment_mode"] == "explicit_snapshot":
        command.append("--snapshot")
    return command


def run_stage(state, unit, stage, backend, *, scenario="success"):
    """Execute the maintained backend's artifact/sidecar and the real transitions."""
    if stage == "source":
        argv = source_command(state, unit)
        args = ["run_block", scenario, unit["id"], state["provenance"]["last_resolved"]["environment_file"],
                stage, unit["outputs"]["source"], unit["outputs"]["data_driven"],
                str(Path(state["campaign_identity"]["output_dir"]) / "logs")]
    else:
        argv = ["python", "./run_data_driven.py", "--input-pkl", unit["outputs"]["source"],
                "--output-pkl", unit["outputs"]["data_driven"]]
        args = ["run_nonprompt", scenario, unit["id"], unit["outputs"]["source"], unit["outputs"]["data_driven"]]
    manifest.transition(state, unit["id"], stage, "running", "none", "synthetic", tuple(argv))
    result = subprocess.run([backend["CAMPAIGN_VALIDATION_BACKEND"], *args], env=backend,
                            capture_output=True, text=True, check=False)
    manifest.transition(state, unit["id"], stage, "success" if result.returncode == 0 else "failed",
                        str(result.returncode), "synthetic")
    return result.returncode


def test_default_mismatch_and_same_sha_relocation(tmp_path, new_archive):
    state = pending_campaign(tmp_path)
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match="frozen environment tarball differs"):
        manifest.record_attempt_provenance(state, requested(state, new_archive))
    assert state == before
    moved = tmp_path / "relocated.tar.gz"
    moved.write_bytes(Path(state["provenance"]["initial"]["environment_file"]).read_bytes())
    assert manifest.record_attempt_provenance(state, requested(state, moved))
    assert state["provenance"]["initial"] == before["provenance"]["initial"]
    assert "environment_migrations" not in state


def test_historical_receipts_and_new_attempt_bindings(tmp_path, backend, new_archive):
    state = pending_campaign(tmp_path)
    first, second = state["units"]
    for stage in ("source", "data_driven"):
        assert run_stage(state, first, stage, backend) == 0
    frozen = copy.deepcopy(first)
    initial = copy.deepcopy(state["provenance"]["initial"])
    science = copy.deepcopy(state["scientific_identity"])
    manifest.record_attempt_provenance(state, requested(state, new_archive), allow_environment_migration=True)
    for stage in ("source", "data_driven"):
        assert run_stage(state, second, stage, backend) == 0
    assert first == frozen
    assert state["scientific_identity"] == science
    assert state["provenance"]["initial"] == initial
    for unit in state["units"]:
        for stage in ("source", "data_driven"):
            manifest.validate_receipt(state, unit, stage, verify_payload_sha256=True)
    source = second["stages"]["source"]
    assert source["attempts"][0]["provenance"]["environment_file_sha256"] == manifest.digest_file(new_archive)
    assert source["receipt"]["binding"]["stage_scientific_contract_sha256"] == manifest.canonical_digest({
        "frozen_environment_tarball_sha256": manifest.digest_file(new_archive)})
    with pytest.raises(ValueError, match="source is not eligible"):
        manifest.transition(state, first["id"], "source", "running", "none", "synthetic", tuple(source_command(state, first)))
    path = state_path(tmp_path)
    manifest.atomic_json(path, state)
    assert manifest.load_state(path) == state
    migration, = state["environment_migrations"]
    assert migration["from"] == initial
    assert migration["to"] == state["provenance"]["last_resolved"]
    assert migration["effective_after_attempts"][first["id"]]["source"]["count"] == 1
    damaged = copy.deepcopy(state)
    damaged["units"][0]["stages"]["source"]["attempts"][0]["provenance"]["environment_file"] = "/other"
    with pytest.raises(ValueError, match="historical attempts changed"):
        manifest.validate_state(damaged)


def test_old_source_future_dd(tmp_path, backend, new_archive):
    state = pending_campaign(tmp_path)
    unit = state["units"][0]
    assert run_stage(state, unit, "source", backend) == 0
    assert run_stage(state, unit, "data_driven", backend, scenario="downstream_fail_once") != 0
    source = copy.deepcopy(unit["stages"]["source"])
    manifest.record_attempt_provenance(state, requested(state, new_archive), allow_environment_migration=True)
    assert run_stage(state, unit, "data_driven", backend) == 0
    assert unit["stages"]["source"] == source
    dd = unit["stages"]["data_driven"]
    assert len(dd["attempts"]) == 2
    assert dd["attempts"][-1]["provenance"]["environment_file_sha256"] == manifest.digest_file(new_archive)
    assert dd["receipt"]["binding"]["source_artifact_sha256"] == source["receipt"]["artifact"]["sha256"]
    assert (tmp_path / "source_order.log").read_text().splitlines() == [unit["id"]]
    assert manifest.validate_state(json.loads(json.dumps(state))) == state


def test_current_resolved_to_explicit_snapshot_keeps_old_receipt(tmp_path, backend, new_archive):
    state = pending_campaign(tmp_path)
    for provenance in state["provenance"].values():
        provenance["environment_mode"] = "current_resolved"
    first, second = state["units"]
    assert run_stage(state, first, "source", backend) == 0
    frozen = copy.deepcopy(first)
    initial = copy.deepcopy(state["provenance"]["initial"])
    manifest.record_attempt_provenance(state, requested(state, new_archive), allow_environment_migration=True)
    assert run_stage(state, second, "source", backend) == 0
    assert first == frozen
    assert state["provenance"]["initial"] == initial
    assert second["stages"]["source"]["attempts"][0]["provenance"]["environment_mode"] == "explicit_snapshot"
    manifest.validate_receipt(state, first, "source", verify_payload_sha256=True)
    manifest.validate_state(state)


@pytest.mark.parametrize("damage", ["archive", "manifest", "running", "science", "implicit"])
def test_reject_inconsistent_migration(tmp_path, new_archive, damage, monkeypatch):
    state = pending_campaign(tmp_path)
    provenance = requested(state, new_archive)
    if damage == "archive":
        new_archive.write_bytes(b"invalid archive")
    elif damage == "manifest":
        path = Path(str(new_archive) + ".manifest.json")
        value = json.loads(path.read_text())
        value["archive_sha256"] = "f" * 64
        path.write_text(json.dumps(value))
    elif damage == "running":
        unit = state["units"][0]
        manifest.transition(state, unit["id"], "source", "running", "none", "synthetic", tuple(source_command(state, unit)))
    elif damage == "implicit":
        provenance["environment_mode"] = "current_resolved"
    else:
        monkeypatch.setattr(manifest, "data_driven_science_matches", lambda *args: False)
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match="archive identity|unambiguous attempt boundary|data_driven_scientific_contract|explicit snapshot"):
        manifest.record_attempt_provenance(state, provenance, allow_environment_migration=True)
    assert state == before


def test_second_migration_after_plan_replacement(tmp_path, new_archive):
    state = pending_campaign(tmp_path)
    manifest.record_attempt_provenance(state, requested(state, new_archive), allow_environment_migration=True)
    previous = copy.deepcopy(state["environment_migrations"][0])
    plan = tmp_path / "replacement.json"
    units = [{"id": "replacement_" + u["id"], "years": u["years"],
              "category_groups": u["category_groups"], "histograms": u["histograms"]} for u in state["units"]]
    plan.write_text(json.dumps({"units": units}))
    manifest.activate_plan(state, str(plan), state["plan"]["active_revision"])
    second_archive = archive_fixture(tmp_path / "second.tar.gz", b"next worker bytes\n")
    manifest.record_attempt_provenance(state, requested(state, second_archive), allow_environment_migration=True)
    assert state["environment_migrations"][0] == previous
    assert set(state["environment_migrations"][1]["effective_after_attempts"]) == {
        "run2_prod_a", "run2_prod_b", "run2_prod_replacement_run2_prod_a", "run2_prod_replacement_run2_prod_b"}
    assert manifest.validate_state(json.loads(json.dumps(state))) == state


def test_record_and_new_attempt_identity_checked(tmp_path, new_archive):
    state = pending_campaign(tmp_path)
    manifest.record_attempt_provenance(state, requested(state, new_archive), allow_environment_migration=True)
    damaged = copy.deepcopy(state)
    damaged["environment_migrations"][0]["to"]["environment_file_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="migration record changed"):
        manifest.validate_state(damaged)
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "synthetic", tuple(source_command(state, unit)))
    unit["stages"]["source"]["attempts"][0]["provenance"]["environment_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="identity contradicts migration frontier"):
        manifest.validate_state(state)


def test_writer_contention(tmp_path, new_archive, monkeypatch):
    state = pending_campaign(tmp_path)
    path = state_path(tmp_path)
    manifest.atomic_json(path, state)
    lock = path.with_name(".run2_prod_campaign.lock")
    with lock.open("a") as held, lock.open("a") as contender:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setenv("CAMPAIGN_LOCK_FD", str(contender.fileno()))
        with pytest.raises(ValueError, match="held by another process"):
            manifest.assert_writer_lock(path)
    assert manifest.load_state(path) == state


@pytest.mark.parametrize("combined", [False, True])
def test_launcher_migration_dry_run(tmp_path, backend, new_archive, combined):
    scopes = ("run2", "run3") if combined else ("run2",)
    paths = []
    for scope in scopes:
        root = tmp_path / "output" / scope if combined else tmp_path / "output"
        state = pending_campaign(tmp_path, scope + "_prod", root)
        if combined:
            state["campaign_identity"]["campaign_tag"] = "synthetic_" + scope
        path = root / ("." + scope + "_prod_campaign_state_v9.json")
        manifest.atomic_json(path, state)
        paths.append(path)
    before = {p: p.read_bytes() for p in paths}
    result = launch(tmp_path, backend, "--resume", "--dry-run", "--selected-fragment",
                    "run3_prod_b" if combined else "run2_prod_b", "--allow-environment-migration",
                    "--phase-telemetry", "--chunksize", "25000",
                    profile="run2_run3_prod" if combined else "run2_prod", archive=new_archive)
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(new_archive) in result.stdout
    assert "--phase-telemetry" in result.stdout
    assert {p: p.read_bytes() for p in paths} == before
    assert not (tmp_path / "source_order.log").exists()


@pytest.mark.parametrize("args", [[], ["--resume"], ["--env-file", "/missing"]])
def test_opt_in_requires_resume_and_explicit_archive(tmp_path, backend, args):
    result = launch(tmp_path, backend, "--allow-environment-migration", *args, archive=None)
    assert result.returncode != 0
    assert "requires --resume and --env-file" in result.stderr
