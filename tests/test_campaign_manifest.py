"""Focused schema-9 campaign state and recovery contracts."""

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import hist
import numpy as np
import pytest

from analysis.topeft_run2 import campaign_manifest as manifest
from analysis.topeft_run2 import wq_staging_guard as staging_guard
from analysis.topeft_run2 import analysis_processor
from analysis.topeft_run2 import run_data_driven
from topcoffea.modules.sparseHist import SparseHist
from topcoffea.modules.utils import get_hist_from_pkl
from topeft.modules.dataDrivenEstimation import DataDrivenProducer
from topeft.modules.axes import info as axes_info, info_2d as axes_info_2d
from topeft.modules.data_driven_products import (
    certify_data_driven_preflight, resolve_data_driven_products,
)
from topeft.modules.histogram_artifact import write_histogram_artifact, write_histogram_sidecar
from topeft.modules import histogram_artifact
from topeft.modules.nominal_schema import scalar_nominal_key
from topeft.modules.production_sample_profile import (
    build_active_sample_universe, certify_production_sample_contract,
)
from topeft.modules.sumw2_policy import resolve_sumw2_storage_policy


@pytest.fixture
def make_campaign(tmp_path):
    def build(profile="run2_prod", histograms=("ptll", "lt"), root_name=None):
        units = [
            {"id": "a", "years": ["run2"], "category_groups": ["3l_m_offZ"], "histograms": list(histograms)},
            {"id": "b", "years": ["run2"], "category_groups": ["3l_p_offZ"], "histograms": list(histograms)},
        ]
        root = tmp_path / (root_name or profile)
        root.mkdir(exist_ok=True)
        revision = manifest.make_revision(1, units)
        specs = manifest.make_specs(units, profile, "synthetic", str(root), revision)
        coverage = manifest.partition_coverage(units, profile)
        identity = {"production_profile": profile, "campaign_tag": "synthetic", "output_dir": str(root)}
        provenance = {
            "topeft_git_commit": "old-commit", "environment_file": str(tmp_path / "env.tar.gz"),
            "environment_file_sha256": "a" * 64, "environment_fingerprint": "b" * 64,
            "environment_mode": "explicit_snapshot", "topcoffea_git_commit": "c" * 40,
            "topcoffea_relevant_source_fingerprint": "d" * 64,
        }
        return manifest.new_state(specs, coverage, identity, manifest.profile_contract(profile), provenance)
    return build


def source_command(state, unit, chunksize="12345"):
    policy = state["profile_contract"]
    command = ["./fullR3_run.sh", "-y", *unit["years"], "-t", unit["output_tag"],
               "-s", chunksize, "--sr" if policy["region"] == "SR" else "--cr",
               "--hist-vars", *unit["histograms"], "--category-groups", *unit["category_groups"],
               "--sample-universe-wrapper", "run_cr.sh -> fullR3_run.sh",
               "--ttgamma-sample-role-policy", "split", "--do-np", "--defer-np",
               "-p", state["campaign_identity"]["output_dir"], "--all-analysis",
               "--env-file", state["provenance"]["initial"]["environment_file"],
               "--snapshot", "-x", "work_queue"]
    if policy["systematics"]:
        command.append("--do-systs")
    if policy["raw_counts"]:
        command.append("--record-raw-count")
    return command


@pytest.fixture
def stale_guarded_attempt(make_campaign, tmp_path, monkeypatch):
    state = make_campaign()
    unit = state["units"][0]
    base = tmp_path / "staging"
    base.mkdir()
    invocation = f"synthetic_{unit['id']}_source_attempt_1"
    root = base / invocation
    root.mkdir(mode=0o700)
    command = source_command(state, unit) + [
        "--wq-guard", "--wq-filepath", str(base), "--wq-invocation-id", invocation,
        "--wq-manager-name", "synthetic-manager-a1",
    ]
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(command))
    marker = {"status": "running", "path": str(root), "invocation": invocation,
              "owner_uid": os.getuid(), "owner_inode": root.stat().st_ino,
              "owner_pid": 1_000_000_000, "owner_start": "1"}
    (root / "guard.json").write_text(json.dumps(marker))
    payload = root / "tmpordinarywq-tmp-" / "nested"
    payload.mkdir(parents=True)
    for name in ("accum_large.p", "transfer", "cache", "partial.pkl.gz", "catalog"):
        (payload / name).write_bytes(b"transient")
    path = tmp_path / ".run2_prod_campaign_state_v9.json"
    manifest.atomic_json(path, state)
    monkeypatch.setattr(staging_guard, "_validate_base", lambda _: base)
    monkeypatch.setattr(staging_guard, "_local_readers", lambda *_, **__: False)
    lock = path.with_name(".run2_prod_campaign.lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    monkeypatch.setenv("CAMPAIGN_LOCK_FD", str(fd))
    manifest.assert_writer_lock(path)
    yield SimpleNamespace(state=state, unit=unit, root=root, base=base, path=path, marker=marker)
    os.close(fd)


def reconcile_stale(fixture):
    staging_guard.recover_interrupted_staging(
        fixture.path, fixture.unit["id"], fixture.base, check_only=True)
    assert manifest.reconcile_running(fixture.state, fixture.unit["id"])
    manifest.atomic_json(fixture.path, fixture.state)


def test_stale_guard_recovery_removes_whole_tree_before_distinct_attempt(stale_guarded_attempt):
    f = stale_guarded_attempt
    sibling = copy.deepcopy(f.state["units"][1])
    sibling_root = f.base / "unrelated_source_attempt_1"
    sibling_root.mkdir()
    (sibling_root / "foreign.p").write_text("preserve")
    reconcile_stale(f)
    staging_guard.recover_interrupted_staging(f.path, f.unit["id"], f.base)
    assert not f.root.exists()
    assert (sibling_root / "foreign.p").read_text() == "preserve"
    assert f.state["units"][1] == sibling
    record = manifest.load_state(f.path)["units"][0]["stages"]["source"]
    assert record["status"] == "failed"
    assert len(record["attempts"]) == 1 and record["attempts"][0]["result"] == "interrupted"
    assert f.unit["stages"]["data_driven"]["status"] == "blocked"
    assert manifest.next_source_attempt(f.state, f.unit["id"]) == 2
    # An already-absent root is safe, but the next namespace must still be absent.
    staging_guard.recover_interrupted_staging(f.path, f.unit["id"], f.base)
    next_root = f.base / f"synthetic_{f.unit['id']}_source_attempt_2"
    next_root.mkdir()
    with pytest.raises(staging_guard.staging_guard_error, match="next attempt"):
        staging_guard.recover_interrupted_staging(f.path, f.unit["id"], f.base)


@pytest.mark.parametrize("blocker", ("owner", "reader", "owner_unknown", "symlink", "fifo",
                                    "hardlink", "marker", "lineage", "next_root", "lock"))
def test_stale_guard_material_blockers_preserve_tree(stale_guarded_attempt, monkeypatch, blocker):
    f = stale_guarded_attempt
    reconcile_stale(f)
    if blocker == "owner":
        monkeypatch.setattr(staging_guard, "_process_start", lambda _: "1")
    elif blocker == "owner_unknown":
        monkeypatch.setattr(staging_guard, "_process_start", lambda _: (_ for _ in ()).throw(
            staging_guard.staging_guard_error("identity cannot be read")))
    elif blocker == "reader":
        monkeypatch.setattr(staging_guard, "_local_readers", lambda *_, **__: True)
    elif blocker == "symlink":
        (f.root / "escape").symlink_to(f.base)
    elif blocker == "fifo":
        os.mkfifo(f.root / "special")
    elif blocker == "hardlink":
        os.link(f.root / "guard.json", f.root / "linked")
    elif blocker == "marker":
        (f.root / "guard.json").write_text(json.dumps({**f.marker, "owner_inode": -1}))
    elif blocker == "lineage":
        command = f.unit["stages"]["source"]["attempts"][0]["command_argv"]
        command[command.index("--wq-invocation-id") + 1] = "different"
        manifest.atomic_json(f.path, f.state)
    elif blocker == "next_root":
        (f.base / f"synthetic_{f.unit['id']}_source_attempt_2").mkdir()
    else:
        monkeypatch.delenv("CAMPAIGN_LOCK_FD")
    with pytest.raises((staging_guard.staging_guard_error, ValueError)):
        staging_guard.recover_interrupted_staging(f.path, f.unit["id"], f.base)
    assert (f.root / "tmpordinarywq-tmp-/nested/accum_large.p").exists()
    assert len(f.unit["stages"]["source"]["attempts"]) == 1


@pytest.mark.parametrize("collision", ("final", "sidecar", "pending", "late_final", "late_pending"))
def test_stale_guard_rechecks_publication_after_reader_scan(stale_guarded_attempt, monkeypatch, collision):
    f = stale_guarded_attempt
    reconcile_stale(f)
    def insert():
        if collision in ("pending", "late_pending"):
            manifest.pending_artifact_path(f.state, f.unit, "source").parent.mkdir(exist_ok=True)
        else:
            name = f.unit["outputs"]["source"] + (".metadata.json" if collision == "sidecar" else "")
            Path(name).write_text("collision")
    if collision.startswith("late_"):
        calls = []
        def reader(_, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                insert()
            return False
        monkeypatch.setattr(staging_guard, "_local_readers", reader)
    else:
        insert()
    with pytest.raises(staging_guard.staging_guard_error, match="output|pending"):
        staging_guard.recover_interrupted_staging(f.path, f.unit["id"], f.base)
    assert (f.root / "tmpordinarywq-tmp-/nested/accum_large.p").exists()


def test_selected_reconcile_does_not_touch_other_running_unit(make_campaign):
    state = make_campaign()
    for unit in state["units"]:
        manifest.transition(state, unit["id"], "source", "running", "none", "test",
                            tuple(source_command(state, unit)))
    sibling = copy.deepcopy(state["units"][1])
    assert manifest.reconcile_running(state, state["units"][0]["id"])
    assert state["units"][1] == sibling


@pytest.mark.parametrize("field", ("st_uid", "st_dev"))
def test_stale_guard_rejects_foreign_entry_before_any_deletion(stale_guarded_attempt, monkeypatch, field):
    f = stale_guarded_attempt
    reconcile_stale(f)
    original = staging_guard.os.scandir
    class entry:
        def __init__(self, value):
            self.value = value
            self.name = value.name
        def stat(self, **kwargs):
            info = self.value.stat(**kwargs)
            if self.name != "accum_large.p":
                return info
            values = {name: getattr(info, name) for name in (
                "st_uid", "st_dev", "st_ino", "st_mode", "st_nlink")}
            values[field] += 1
            return SimpleNamespace(**values)
    class scan:
        def __init__(self, path):
            self.iterator = original(path)
        def __enter__(self):
            return (entry(value) for value in self.iterator)
        def __exit__(self, *_):
            self.iterator.close()
    monkeypatch.setattr(staging_guard.os, "scandir", scan)
    with pytest.raises(staging_guard.staging_guard_error, match="foreign"):
        staging_guard.recover_interrupted_staging(f.path, f.unit["id"], f.base)
    assert (f.root / "tmpordinarywq-tmp-/nested/accum_large.p").exists()


def test_stale_guard_pid_reuse_is_not_old_owner(stale_guarded_attempt, monkeypatch):
    f = stale_guarded_attempt
    reconcile_stale(f)
    monkeypatch.setattr(staging_guard, "_process_start", lambda _: "999999")
    staging_guard.recover_interrupted_staging(f.path, f.unit["id"], f.base)
    assert not f.root.exists()


@pytest.mark.parametrize("change", ("root_swap", "symlink_swap", "insert", "state"))
def test_stale_guard_revalidates_after_scan(stale_guarded_attempt, monkeypatch, change):
    f = stale_guarded_attempt
    reconcile_stale(f)
    calls = []
    def reader(_, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            if change in ("root_swap", "symlink_swap"):
                f.root.rename(f.base / "preserved_old")
                if change == "symlink_swap":
                    f.root.symlink_to(f.base / "preserved_old")
                else:
                    f.root.mkdir(mode=0o700)
            elif change == "insert":
                (f.root / "inserted").symlink_to(f.base)
            else:
                f.path.write_text(f.path.read_text() + "\n")
        return False
    monkeypatch.setattr(staging_guard, "_local_readers", reader)
    with pytest.raises((staging_guard.staging_guard_error, OSError)):
        staging_guard.recover_interrupted_staging(f.path, f.unit["id"], f.base)
    preserved = f.base / "preserved_old" if change in ("root_swap", "symlink_swap") else f.root
    assert (preserved / "tmpordinarywq-tmp-/nested/accum_large.p").exists()


def test_live_stale_guard_blocks_before_native_reconcile(stale_guarded_attempt, monkeypatch):
    f = stale_guarded_attempt
    original = f.path.read_bytes()
    monkeypatch.setattr(staging_guard, "_process_start", lambda _: "1")
    with pytest.raises(staging_guard.staging_guard_error, match="owner is still live"):
        staging_guard.recover_interrupted_staging(f.path, f.unit["id"], f.base, check_only=True)
    assert f.path.read_bytes() == original
    assert f.unit["stages"]["source"]["attempts"][0]["result"] == "running"


def test_selected_reconcile_keeps_successful_sibling_pending_alias(make_campaign, tmp_path, monkeypatch):
    state = make_campaign()
    selected, sibling = state["units"]
    for unit in (selected, sibling):
        manifest.transition(state, unit["id"], "source", "running", "none", "test",
                            tuple(source_command(state, unit)))
    write_artifact(sibling["outputs"]["source"], state=state, unit=sibling)
    manifest.transition(state, sibling["id"], "source", "ready", "0", "test")
    alias = manifest.pending_artifact_path(state, sibling, "source").parent
    alias.mkdir()
    (alias / "preserve").write_text("sibling pending alias")
    preserved = copy.deepcopy(sibling)
    path = tmp_path / "selected_reconcile_state.json"
    manifest.atomic_json(path, state)
    monkeypatch.setattr(manifest, "assert_writer_lock", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["campaign_manifest.py", "reconcile", str(path), selected["id"]])
    manifest.main()
    assert manifest.load_state(path)["units"][1] == preserved
    assert (alias / "preserve").read_text() == "sibling pending alias"


def test_recovery_accepts_massive_invocation_shape(tmp_path, monkeypatch):
    invocation = "sr-reco-run3-18-261019-001_run3_sr_ext_size25_18_09_source_attempt_1"
    root = tmp_path / invocation
    root.mkdir(mode=0o700)
    (root / "guard.json").write_text(json.dumps({
        "status": "running", "path": str(root), "invocation": invocation,
        "owner_inode": root.stat().st_ino, "owner_uid": os.getuid(),
        "owner_pid": 1_000_000_000, "owner_start": "1",
    }))
    nested = root / "tmpaccumulationwq-tmp-" / "merge" / "cache"
    nested.mkdir(parents=True)
    for name in ("accum_huge.p", "args_transfer.p", "partial.pkl.gz", "catalog", "temporary"):
        (nested / name).write_text("transient")
    staging_guard._recover_guarded_tree(root, invocation, lambda: None)
    assert not root.exists()


def test_recovery_has_no_incident_specific_branch_or_public_bypass():
    guard_source = Path(staging_guard.__file__).read_text()
    shell_source = (Path(manifest.__file__).parent / "run_cr.sh").read_text()
    assert "18_09" not in guard_source
    for option in ("--force-clean", "--ignore-readers", "--cleanup-attempt", "--recover-1809"):
        assert option not in shell_source


def test_manager_name_changes_attempt_provenance_only(make_campaign):
    baseline = make_campaign()
    state_a = copy.deepcopy(baseline)
    state_b = copy.deepcopy(baseline)
    unit_a = state_a["units"][0]
    unit_b = state_b["units"][0]
    command_a = source_command(state_a, unit_a) + ["--wq-manager-name", "manager-a"]
    command_b = source_command(state_b, unit_b) + ["--wq-manager-name", "manager-b"]

    manifest.transition(state_a, unit_a["id"], "source", "running", "none", "test", tuple(command_a))
    manifest.transition(state_b, unit_b["id"], "source", "running", "none", "test", tuple(command_b))

    assert state_a["campaign_identity"] == state_b["campaign_identity"]
    assert state_a["scientific_identity"] == state_b["scientific_identity"]
    assert state_a["scientific_identity"]["source_stage_scientific_contract_sha256"] == (
        state_b["scientific_identity"]["source_stage_scientific_contract_sha256"]
    )
    assert state_a["scientific_identity"]["data_driven_stage_scientific_contract_sha256"] == (
        state_b["scientific_identity"]["data_driven_stage_scientific_contract_sha256"]
    )
    attempt_a = unit_a["stages"]["source"]["attempts"][0]
    attempt_b = unit_b["stages"]["source"]["attempts"][0]
    assert attempt_a["command_argv"] != attempt_b["command_argv"]
    assert manifest.attempt_identity(attempt_a) != manifest.attempt_identity(attempt_b)


def test_next_source_attempt_uses_validated_persisted_history(make_campaign):
    state = make_campaign()
    unit = state["units"][0]
    unit_id = unit["id"]
    assert manifest.next_source_attempt(state, unit_id) == 1
    command = source_command(state, unit)
    manifest.transition(state, unit_id, "source", "running", "none", "test", tuple(command))
    with pytest.raises(ValueError, match="not eligible"):
        manifest.next_source_attempt(state, unit_id)
    manifest.transition(state, unit_id, "source", "failed", "23", "test")
    assert manifest.next_source_attempt(state, unit_id) == 2
    with pytest.raises(ValueError, match="unknown source"):
        manifest.next_source_attempt(state, "missing")
    with pytest.raises(ValueError, match="inconsistent"):
        manifest.next_source_attempt({**state, "units": [{**unit, "stages": {
            **unit["stages"], "source": {**unit["stages"]["source"], "attempts": "bad"},
        }}]}, unit_id)


def test_canonical_output_capacity_refusal_and_dd_lineage(make_campaign, monkeypatch):
    state = make_campaign()
    unit = state["units"][0]
    outputs = copy.deepcopy(unit["outputs"])
    command = source_command(state, unit) + [
        "--pkl-min-free-bytes", str(10**30), "--pkl-min-free-inodes", "4",
        "--pkl-safety-reserve-bytes", "1", "--pkl-safety-reserve-inodes", "1",
    ]
    with pytest.raises(ValueError, match="capacity admission refused"):
        manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(command))
    assert unit["stages"]["source"]["attempts"] == []
    command[command.index("--pkl-min-free-bytes") + 1] = "2"
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(command))
    assert manifest.effective_output_path(unit, "source") == outputs["source"]
    assert unit["outputs"] == outputs
    write_artifact(outputs["source"])
    manifest.transition(state, unit["id"], "source", "ready", "0", "test")
    receipt_path = unit["stages"]["source"]["receipt"]["artifact"]["path"]
    assert receipt_path == outputs["source"]
    dd_command = ("python", "./run_data_driven.py", "--input-pkl", receipt_path,
                  "--output-pkl", outputs["data_driven"])
    manifest.transition(state, unit["id"], "data_driven", "running", "none", "test", dd_command)
    assert manifest.effective_output_path(unit, "data_driven") == outputs["data_driven"]
    assert unit["outputs"] == outputs


def test_canonical_inode_admission_refuses_without_attempt(make_campaign, monkeypatch):
    state = make_campaign()
    unit = state["units"][0]
    command = source_command(state, unit) + [
        "--pkl-min-free-bytes", "2", "--pkl-min-free-inodes", "4",
        "--pkl-safety-reserve-bytes", "1", "--pkl-safety-reserve-inodes", "1",
    ]
    with monkeypatch.context() as capacity_patch:
        capacity_patch.setattr(manifest.os, "statvfs", lambda _: SimpleNamespace(
            f_bavail=100, f_frsize=1, f_favail=3,
        ))
        with pytest.raises(ValueError, match="capacity admission refused"):
            manifest.transition(state, unit["id"], "source", "running", "none", "test",
                                tuple(command))
    assert unit["stages"]["source"] == {"status": "pending", "attempts": [], "receipt": None}


@pytest.mark.parametrize("collision", (False, True))
def test_canonical_pending_source_republishes_same_attempt(
    make_campaign, tmp_path, monkeypatch, collision
):
    state = make_campaign("run2_stat_diagnostic", histograms=("ptll",))
    unit = state["units"][0]
    command = source_command(state, unit)
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(command))
    final = Path(manifest.effective_output_path(unit, "source"))
    pending = manifest.pending_artifact_path(state, unit, "source")
    original_link = histogram_artifact.os.link

    def interrupt_sidecar(source, destination, *args, **kwargs):
        if Path(destination) == Path(f"{final}.metadata.json"):
            raise OSError("simulated publication interruption")
        return original_link(source, destination, *args, **kwargs)

    with monkeypatch.context() as publication_patch:
        publication_patch.setattr(histogram_artifact.os, "link", interrupt_sidecar)
        with pytest.raises(OSError, match="publication interruption"):
            write_artifact(final, state=state, unit=unit, real_payload=True,
                           pending_id=manifest.pending_artifact_id(state, unit, "source"))
    assert pending.is_file()
    assert Path(f"{pending}.metadata.json").is_file()
    assert final.is_file()
    assert unit["stages"]["source"]["status"] == "running"
    assert len(unit["stages"]["source"]["attempts"]) == 1
    with pytest.raises(ValueError, match="unresolved output evidence"):
        manifest.transition(state, unit["id"], "source", "failed", "1", "test")
    final_sidecar = Path(f"{final}.metadata.json")
    if collision:
        final_sidecar.write_bytes(b"foreign collision")
        with pytest.raises(ValueError, match="collision preserves pending evidence"):
            manifest.reconcile_running(state)
        assert pending.is_file() and final_sidecar.read_bytes() == b"foreign collision"
        assert unit["stages"]["source"]["status"] == "running"
    else:
        state_path = tmp_path / "recovery_state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        monkeypatch.setattr(manifest, "assert_writer_lock", lambda _: None)
        monkeypatch.setattr(sys, "argv", ["campaign_manifest.py", "reconcile", str(state_path)])
        manifest.main()
        durable = json.loads(state_path.read_text(encoding="utf-8"))
        record = durable["units"][0]["stages"]["source"]
        assert record["status"] == "success"
        assert len(record["attempts"]) == 1
        assert record["receipt"]["artifact"]["path"] == str(final)
        assert not pending.parent.exists()


def test_canonical_malformed_pending_locator_cannot_be_reclassified(
    make_campaign, tmp_path, monkeypatch
):
    state = make_campaign()
    unit = state["units"][0]
    command = source_command(state, unit)
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(command))
    pending = manifest.pending_artifact_path(state, unit, "source")
    pending.parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="pending locator is not an owned directory"):
        manifest.reconcile_running(state)
    with pytest.raises(ValueError, match="unresolved output evidence"):
        manifest.transition(state, unit["id"], "source", "failed", "1", "test")
    assert unit["stages"]["source"]["status"] == "running"
    record = unit["stages"]["source"]
    record["status"] = "failed"
    record["attempts"][-1]["result"] = "failed"
    record["attempts"][-1]["exit_code"] = 1
    record["attempts"][-1]["ended_at_utc"] = manifest.now_utc()
    with pytest.raises(ValueError, match="unresolved pending evidence prevents a new attempt"):
        manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(command))
    assert len(record["attempts"]) == 1


def test_canonical_pending_identity_binds_control_root_and_attempt(
    make_campaign, tmp_path, monkeypatch
):
    states = (
        make_campaign(root_name="campaign_a"),
        make_campaign(root_name="campaign_b"),
    )
    pending = []
    for state in states:
        unit = state["units"][0]
        command = source_command(state, unit)
        manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(command))
        pending.append(manifest.pending_artifact_path(state, unit, "source"))
    assert pending[0].name == pending[1].name
    assert pending[0].parent != pending[1].parent
    pending[0].parent.mkdir(mode=0o700)
    assert manifest.reconcile_running(states[1])
    assert states[1]["units"][0]["stages"]["source"]["attempts"][0]["result"] == "interrupted"
    assert pending[0].parent.is_dir()


@pytest.mark.parametrize("telemetry", (False, True))
def test_canonical_pending_data_driven_republishes_with_durable_receipt(
    make_campaign, tmp_path, monkeypatch, telemetry
):
    state = make_campaign("run2_stat_diagnostic", histograms=("ptll",))
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test",
                        tuple(source_command(state, unit)))
    write_artifact(unit["outputs"]["source"], state=state, unit=unit, real_payload=True)
    manifest.transition(state, unit["id"], "source", "ready", "0", "test")
    final = Path(unit["outputs"]["data_driven"])
    command = ("python", "./run_data_driven.py", "--input-pkl",
               unit["stages"]["source"]["receipt"]["artifact"]["path"],
               "--output-pkl", str(final))
    command += ("--phase-telemetry",) if telemetry else ()
    command += ("--pkl-no-clobber",)
    manifest.transition(state, unit["id"], "data_driven", "running", "none", "test", command)
    pending = manifest.pending_artifact_path(state, unit, "data_driven")
    original_link = histogram_artifact.os.link

    def interrupt_sidecar(source, destination, *args, **kwargs):
        if Path(destination) == Path(f"{final}.metadata.json"):
            raise OSError("simulated DD publication interruption")
        return original_link(source, destination, *args, **kwargs)

    with monkeypatch.context() as publication_patch:
        publication_patch.setattr(histogram_artifact.os, "link", interrupt_sidecar)
        with pytest.raises(OSError, match="DD publication interruption"):
            run_data_driven.main(list(command[2:]) + [
                "--pkl-pending-id", manifest.pending_artifact_id(state, unit, "data_driven"),
            ])
    assert pending.is_file() and final.is_file()
    state_path = tmp_path / "recovery_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(manifest, "assert_writer_lock", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["campaign_manifest.py", "reconcile", str(state_path)])
    manifest.main()
    durable = json.loads(state_path.read_text(encoding="utf-8"))
    record = durable["units"][0]["stages"]["data_driven"]
    assert record["status"] == "success"
    assert len(record["attempts"]) == 1
    assert record["receipt"]["artifact"]["path"] == str(final)
    assert not pending.parent.exists()


def write_artifact(path, source_sha256=None, *, state=None, unit=None, real_payload=False,
                   wrapper_identity="run_cr.sh -> fullR3_run.sh", pending_id=None):
    path = Path(path)
    if not real_payload:
        path.write_bytes(b"synthetic processor output")
    if state is not None and source_sha256 is None:
        samples = {
            "data_dataset": {"histAxisName": "dataUL18", "isData": True, "WCnames": []},
            "prompt_dataset": {"histAxisName": "TTTo2L2Nu_centralUL18", "isData": False, "WCnames": []},
            "signal_dataset": {"histAxisName": "WWTo2L2Nu_centralUL18", "isData": False, "WCnames": ["ctG"]},
        }
        mode = "production" if state["profile_contract"]["systematics"] else "full_diagnostics"
        storage = {"mode": mode}
        if mode == "production":
            storage["rules"] = [{"process_prefixes": ["data", "TT", "WW"]}]
        policy = resolve_sumw2_storage_policy(
            storage, samples=samples, runtime_families=tuple(unit["histograms"]),
            axes_info=axes_info, axes_info_2d=axes_info_2d,
            sumw2_storage_present=True,
        )
        products = resolve_data_driven_products(
            {
                "nonprompt": {"enabled": True, "source_contributors": {
                    "data": {"process_names": ["dataUL18"]},
                    "prompt_mc": {"process_names": ["TTTo2L2Nu_centralUL18"]},
                }},
                "flips": {"enabled": True, "source_contributors": {
                    "data": {"process_names": ["dataUL18"]},
                }},
            },
            data_driven_products_present=True, legacy_do_np=False,
            samples=samples, runtime_families=tuple(unit["histograms"]),
            metadata_path="test_options.yml",
        )
        requested, resolved = certify_data_driven_preflight(products, policy)
        contract = certify_production_sample_contract(
            build_active_sample_universe(samples, wrapper_identity=wrapper_identity),
            policy, products,
        )
        files = manifest.scientific_files(state["profile_contract"], "source")
        contract["cfg_identities"] = sorted((
            {"basename": Path(name).name, "content_sha256": digest, "input_kind": "cfg"}
            for name, digest in files.items() if name.startswith("input_samples/cfgs/")
        ), key=lambda item: tuple(sorted(item.items())))
        unsigned = {key: value for key, value in contract.items() if key != "contract_identity_sha256"}
        contract["contract_identity_sha256"] = manifest.canonical_digest(unsigned)
        applicability = {
            "contract_version": 1,
            "producer_query": "AnalysisProcessor.histogram_fill_is_applicable",
            "producer_semantics_sha256": analysis_processor.histogram_applicability_semantics_sha256(),
            "analysis_mode": "all",
            "families": {family: {"channels": {}} for family in unit["histograms"]},
        }
        histograms = {}
        if real_payload:
            for family in unit["histograms"]:
                axes = (
                    hist.axis.StrCategory([], name="process", growth=True),
                    hist.axis.StrCategory([], name="channel", growth=True),
                    hist.axis.StrCategory([], name="systematic", growth=True),
                    hist.axis.StrCategory([], name="appl", growth=True),
                    hist.axis.Regular(1, 0, 1, name=family),
                )
                nominal = SparseHist(*axes, storage="Double")
                sumw2 = SparseHist(*axes, storage="Double")
                for process, region, weight in (
                    ("dataUL18", "isAR_3l", 10.0),
                    ("TTTo2L2Nu_centralUL18", "isAR_3l", 3.0),
                    ("dataUL18", "isAR_2lSS_OS", 4.0),
                ):
                    values = {"process": process, "channel": "3l", "systematic": "nominal",
                              "appl": region, family: np.asarray([0.5])}
                    nominal.fill(**values, weight=np.asarray([weight]))
                    sumw2.fill(**values, weight=np.asarray([weight * weight]))
                histograms[scalar_nominal_key(family)] = nominal
                histograms[f"{family}_sumw2"] = sumw2
            applicability["families"] = {
                family: {"channels": {"3l": "applicable"}} for family in unit["histograms"]
            }
        writer = write_histogram_artifact if real_payload else write_histogram_sidecar
        writer(
            path, histograms=histograms, artifact_kind="processor_output",
            sumw2_storage_provenance=policy.to_provenance(),
            production_sample_contract=contract,
            requested_data_driven_products=requested,
            resolved_data_driven_contract=resolved,
            histogram_applicability=applicability,
            **({"no_clobber": True, "pending_id": pending_id} if pending_id else {}),
        )
        return
    sidecar = {
        "artifact": {"pkl_basename": path.name, "pkl_size_bytes": path.stat().st_size,
                     "pkl_sha256": manifest.digest_file(path)},
        "lineage": {"inputs": [] if source_sha256 is None else [{"pkl_sha256": source_sha256}]},
    }
    if state is not None:
        source_sidecar = json.loads(Path(f"{unit['outputs']['source']}.metadata.json").read_text())
        sidecar.update({
            "metadata_schema_version": 2,
            "production_sample_contract": source_sidecar["production_sample_contract"],
            "sumw2_storage_provenance": source_sidecar["sumw2_storage_provenance"],
            "lineage": {"inputs": [{
                "pkl_basename": Path(unit["outputs"]["source"]).name,
                "artifact_kind": "processor_output", "pkl_sha256": source_sha256,
            }]},
        })
        sidecar["artifact"]["artifact_kind"] = "nonprompt_output"
    Path(f"{path}.metadata.json").write_text(json.dumps(sidecar), encoding="utf-8")


@pytest.mark.parametrize("profile", ["run2_prod", "run2_prod_CR", "run2_stat_diagnostic", "run2_stat_diagnostic_CR"])
def test_canonical_shape_for_component_modes(make_campaign, profile):
    state = make_campaign(profile)
    assert state["schema_version"] == 9
    assert state["profile_contract"]["data_driven_policy"] == "nonprompt_charge_flip"
    assert set(state["units"][0]["stages"]) == {"source", "data_driven"}
    assert manifest.validate_state(json.loads(json.dumps(state)), verify_artifacts=False)
    assert state["scientific_identity"]["scientific_coverage_sha256"] != state["plan"]["plan_partition_sha256"]


def test_sr_and_cr_share_complete_data_driven_policy():
    sr = manifest.profile_contract("run2_prod")
    cr = manifest.profile_contract("run2_prod_CR")
    assert sr["data_driven_policy"] == cr["data_driven_policy"] == "nonprompt_charge_flip"
    assert {key for key in sr if sr[key] != cr[key]} == {"region"}


def test_operational_delta_does_not_change_science(make_campaign):
    state = make_campaign()
    changed = dict(state["provenance"]["initial"], topeft_git_commit="new-commit",
                   environment_file="/relocated/env.tar.gz")
    manifest.validate_resume(state, state["campaign_identity"], state["profile_contract"], changed,
                             state["scientific_identity"]["scientific_coverage"])
    assert manifest.record_attempt_provenance(state, changed)
    assert state["provenance"]["last_resolved"]["environment_file"] == "/relocated/env.tar.gz"


@pytest.mark.parametrize("profile", ["run2_prod", "run2_stat_diagnostic", "run2_prod_CR"])
def test_source_attempt_accepts_arbitrary_positive_chunksize(make_campaign, profile):
    state = make_campaign(profile)
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(source_command(state, unit)))
    assert unit["stages"]["source"]["attempts"][0]["chunksize"] == 12345


def test_environment_provenance_and_coverage_changes(make_campaign):
    state = make_campaign()
    provenance = state["provenance"]["initial"]
    changed = dict(provenance, environment_file_sha256="f" * 64)
    with pytest.raises(ValueError, match="NEW_CAMPAIGN_REQUIRED"):
        manifest.validate_resume(state, state["campaign_identity"], state["profile_contract"], changed,
                                 state["scientific_identity"]["scientific_coverage"])
    relocated = dict(provenance, environment_file="/relocated/same.tar.gz",
                     environment_fingerprint="f" * 64)
    manifest.validate_resume(state, state["campaign_identity"], state["profile_contract"],
                             relocated, state["scientific_identity"]["scientific_coverage"])
    assert manifest.record_attempt_provenance(state, relocated)
    with pytest.raises(ValueError, match="scientific_coverage_incompatible"):
        manifest.validate_resume(state, state["campaign_identity"], state["profile_contract"], provenance, [])


def test_interrupted_attempt_becomes_retry_eligible(make_campaign):
    state = make_campaign()
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(source_command(state, unit)))
    assert manifest.reconcile_running(state)
    assert unit["stages"]["source"]["attempts"][0]["result"] == "interrupted"
    assert unit["stages"]["source"]["status"] == "failed"
    manifest.transition(state, unit["id"], "source", "running", "none", "retry",
                        tuple(source_command(state, unit, "98765")))
    assert unit["stages"]["source"]["attempts"][1]["chunksize"] == 98765


def test_source_receipt_and_downstream_retry(make_campaign):
    state = make_campaign()
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(source_command(state, unit)))
    write_artifact(unit["outputs"]["source"])
    manifest.transition(state, unit["id"], "source", "ready", "0", "test")
    assert unit["stages"]["source"]["receipt"]["binding"]["scientific_contract_sha256"] == state["scientific_identity"]["scientific_contract_sha256"]
    command = ("python", "./run_data_driven.py", "--input-pkl", unit["outputs"]["source"],
               "--output-pkl", unit["outputs"]["data_driven"])
    manifest.transition(state, unit["id"], "data_driven", "running", "none", "test", command)
    manifest.transition(state, unit["id"], "data_driven", "failed", "137", "test")
    assert unit["stages"]["source"]["status"] == "success"
    manifest.transition(state, unit["id"], "data_driven", "running", "none", "retry", command)


def test_receipt_rejects_wrong_artifact_identity(make_campaign):
    state = make_campaign()
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(source_command(state, unit)))
    write_artifact(unit["outputs"]["source"])
    sidecar = Path(f"{unit['outputs']['source']}.metadata.json")
    data = json.loads(sidecar.read_text())
    data["artifact"]["pkl_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact sidecar identity mismatch"):
        manifest.transition(state, unit["id"], "source", "ready", "0", "test")


def test_plan_rejects_overlap_and_missing_coverage(make_campaign):
    state = make_campaign()
    units = [{key: unit[key] for key in ("id", "years", "category_groups", "histograms")} for unit in state["units"]]
    with pytest.raises(ValueError, match="overlaps"):
        manifest.verify_partition([units[0], units[0]], "run2_prod", state["scientific_identity"]["scientific_coverage"])
    with pytest.raises(ValueError, match="changes scientific coverage"):
        manifest.verify_partition(units[:1], "run2_prod", state["scientific_identity"]["scientific_coverage"])


def test_stale_revision_rejected_before_mutation(make_campaign, tmp_path):
    state = make_campaign()
    before = copy.deepcopy(state)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"units": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="stale expected plan revision"):
        manifest.activate_plan(state, plan, "r0_stale")
    assert state == before


def test_old_schemas_have_no_normal_resume(make_campaign):
    state = make_campaign()
    for version in (5, 6, 7, 8):
        with pytest.raises(ValueError, match="schema 9"):
            manifest.validate_state(dict(state, schema_version=version))


@pytest.mark.parametrize("profile", ["run2_sr_ext", "run3_sr_ext"])
def test_sr_extension_default_coverage_is_stable(tmp_path, profile):
    units = manifest.default_units(profile, "-")
    assert len(units) == 12
    coverage = manifest.partition_coverage(units, profile)
    root = tmp_path / profile
    root.mkdir()
    identity = {"production_profile": profile, "campaign_tag": "synthetic", "output_dir": str(root)}
    provenance = {
        "topeft_git_commit": "old", "environment_file": str(tmp_path / "env.tar.gz"),
        "environment_file_sha256": "a" * 64, "environment_fingerprint": "b" * 64,
        "environment_mode": "explicit_snapshot", "topcoffea_git_commit": "c" * 40,
        "topcoffea_relevant_source_fingerprint": "d" * 64,
    }
    specs = manifest.make_specs(units, profile, "synthetic", str(root), manifest.make_revision(1, units))
    state = manifest.new_state(specs, coverage, identity, manifest.profile_contract(profile), provenance)
    assert state["schema_version"] == 9
    assert len(state["units"]) == 12
    assert manifest.validate_state(state, verify_artifacts=False)


def test_historical_sr_resume_binds_recorded_coverage_after_registry_change(tmp_path, monkeypatch):
    profile = "run3_sr_ext"
    units = [{"id": "historical", "years": ["2022"],
              "category_groups": ["2los_1tau"], "histograms": ["b0pt"]}]
    root = tmp_path / "campaign"
    root.mkdir()
    revision = manifest.make_revision(1, units)
    specs = manifest.make_specs(units, profile, "synthetic", str(root), revision)
    historical_coverage = manifest.partition_coverage(units, profile)
    identity = {"production_profile": profile, "campaign_tag": "synthetic", "output_dir": str(root)}
    provenance = {
        "topeft_git_commit": "old", "environment_file": str(tmp_path / "env.tar.gz"),
        "environment_file_sha256": "a" * 64, "environment_fingerprint": "b" * 64,
        "environment_mode": "explicit_snapshot", "topcoffea_git_commit": "c" * 40,
        "topcoffea_relevant_source_fingerprint": "d" * 64,
    }
    state = manifest.new_state(specs, historical_coverage, identity,
                               manifest.profile_contract(profile), provenance)
    state_path = root / f".{profile}_campaign_state_v9.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    channels = json.loads(manifest.CHANNELS_PATH.read_text(encoding="utf-8"))
    historical_jet = channels["ALL_CH_LST_SR"]["2los_1tau"]["jet_lst"][0]
    current_jet = ">42" if historical_jet != ">42" else ">43"
    channels["ALL_CH_LST_SR"]["2los_1tau"]["jet_lst"] = [current_jet]
    current_registry = tmp_path / "current_channels.json"
    current_registry.write_text(json.dumps(channels), encoding="utf-8")
    monkeypatch.setattr(manifest, "CHANNELS_PATH", current_registry)
    assert manifest.partition_coverage(units, profile) != historical_coverage

    # New plans still use the current registry; normal resume uses recorded state.
    current_inputs = ["-", profile, "-", "synthetic", str(root), "new-commit",
                      provenance["environment_file"], provenance["environment_file_sha256"],
                      provenance["environment_fingerprint"], provenance["environment_mode"],
                      provenance["topcoffea_git_commit"],
                      provenance["topcoffea_relevant_source_fingerprint"],
                      "split", "true", "true", "SR", "separate"]
    assert manifest.desired_state_inputs(current_inputs)[-1] != historical_coverage
    restored = manifest.load_state(state_path)
    assert manifest.validate_resume(restored, identity, restored["profile_contract"],
                                    provenance, historical_coverage)
    monkeypatch.setattr(sys, "argv", ["campaign_manifest.py", "validate", str(state_path),
                                      *current_inputs])
    manifest.main()
    active_plan = tmp_path / "active.tsv"
    monkeypatch.setattr(sys, "argv", ["campaign_manifest.py", "render-active",
                                      str(state_path), str(active_plan)])
    manifest.main()
    assert active_plan.read_text(encoding="utf-8") == manifest.plan_tsv(state["units"])
    assert manifest.load_state(state_path)["units"][0]["coverage"] == specs[0]["coverage"]
    assert f"|{historical_jet}|" in historical_coverage[0]
    assert all(f"|{current_jet}|" not in row for row in historical_coverage)

    with pytest.raises(ValueError, match="NEW_CAMPAIGN_REQUIRED"):
        manifest.validate_resume(restored, identity, restored["profile_contract"],
                                 dict(provenance, environment_file_sha256="f" * 64),
                                 historical_coverage)
    # An explicit qualified snapshot binds source science; the mutable manager
    # fingerprint remains operational provenance for this resume.
    assert manifest.validate_resume(restored, identity, restored["profile_contract"],
                                    dict(provenance, topcoffea_relevant_source_fingerprint="f" * 64),
                                    historical_coverage)
    with pytest.raises(ValueError, match="campaign identity or profile contract differs"):
        manifest.validate_resume(restored, dict(identity, campaign_tag="different"),
                                 restored["profile_contract"], provenance, historical_coverage)
    with pytest.raises(ValueError, match="campaign identity or profile contract differs"):
        manifest.validate_resume(restored, dict(identity, output_dir="/different"),
                                 restored["profile_contract"], provenance, historical_coverage)
    with pytest.raises(ValueError, match="campaign identity or profile contract differs"):
        manifest.validate_resume(restored, identity,
                                 dict(restored["profile_contract"], systematics=False),
                                 provenance, historical_coverage)
    supplied_plan = tmp_path / "replacement.json"
    supplied_plan.write_text(json.dumps({"units": units}), encoding="utf-8")
    with pytest.raises(ValueError, match="plan changes scientific coverage"):
        manifest.activate_plan(restored, supplied_plan, revision)


@pytest.mark.parametrize(("field", "expected"), (
    ("unit_coverage", "plan changes scientific coverage"),
    ("snapshot_coverage", "immutable plan revision snapshot changed"),
    ("science_coverage", "scientific coverage digest changed"),
    ("science_contract_coverage", "scientific contract coverage differs"),
))
def test_recorded_coverage_corruption_is_rejected(make_campaign, field, expected):
    state = json.loads(json.dumps(make_campaign()))
    if field == "unit_coverage":
        state["units"][0]["coverage"].pop()
        revision = state["plan"]["revisions"][0]
        revision["units"][0]["coverage"].pop()
        revision["snapshot_sha256"] = manifest.canonical_digest(revision["units"])
    elif field == "snapshot_coverage":
        state["plan"]["revisions"][0]["units"][0]["coverage"].pop()
    elif field == "science_coverage":
        state["scientific_identity"]["scientific_coverage"].pop()
    else:
        state["scientific_identity"]["scientific_contract_payload"][
            "approved_scientific_coverage_sha256"] = "f" * 64
        state["scientific_identity"]["scientific_contract_sha256"] = manifest.canonical_digest(
            state["scientific_identity"]["scientific_contract_payload"])
    with pytest.raises(ValueError, match=expected):
        manifest.validate_state(state, verify_artifacts=False)


def test_plan_revision_repartitions_only_pending_science(make_campaign, tmp_path):
    state = make_campaign()
    original_coverage = state["scientific_identity"]["scientific_coverage_sha256"]
    frozen_sha = state["scientific_identity"]["source_stage_scientific_contract"]["frozen_environment_tarball_sha256"]
    original_revision = state["plan"]["active_revision"]
    units = [
        {"id": "split_left", "years": ["run2"], "category_groups": ["3l_m_offZ"], "histograms": ["ptll"]},
        {"id": "split_right", "years": ["run2"], "category_groups": ["3l_m_offZ"], "histograms": ["lt"]},
        {"id": "b", "years": ["run2"], "category_groups": ["3l_p_offZ"], "histograms": ["ptll", "lt"]},
    ]
    plan = tmp_path / "repartition.json"
    plan.write_text(json.dumps({"units": units}), encoding="utf-8")
    new_revision = manifest.activate_plan(state, plan, original_revision)
    assert new_revision != original_revision
    assert len(state["units"]) == 3
    assert state["scientific_identity"]["scientific_coverage_sha256"] == original_coverage
    assert state["scientific_identity"]["source_stage_scientific_contract"]["frozen_environment_tarball_sha256"] == frozen_sha
    assert manifest.validate_state(state, verify_artifacts=False)


def test_source_ready_coverage_is_frozen_during_repartition(make_campaign, tmp_path):
    state = make_campaign()
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(source_command(state, unit)))
    write_artifact(unit["outputs"]["source"])
    manifest.transition(state, unit["id"], "source", "ready", "0", "test")
    revised = [
        {"id": "split_left", "years": ["run2"], "category_groups": ["3l_m_offZ"], "histograms": ["ptll"]},
        {"id": "split_right", "years": ["run2"], "category_groups": ["3l_m_offZ"], "histograms": ["lt"]},
        {"id": "b", "years": ["run2"], "category_groups": ["3l_p_offZ"], "histograms": ["ptll", "lt"]},
    ]
    plan = tmp_path / "repartition.json"
    plan.write_text(json.dumps({"units": revised}), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage is frozen"):
        manifest.activate_plan(state, plan, state["plan"]["active_revision"])


def test_explicit_schema8_import_preserves_original_and_provenance(tmp_path):
    profile = "run3_sr_ext"
    root = tmp_path / "campaign"
    root.mkdir()
    units = manifest.default_units(profile, "-")
    units[1] = {**units[1], "id": "size25_18_04"}
    coverage = manifest.partition_coverage(units, profile)
    revision = "r1_085b6be02eb3"
    specs = manifest.make_specs(units, profile, "synthetic", str(root), revision)
    identity = {"production_profile": profile, "campaign_tag": "synthetic", "output_dir": str(root)}
    contract = manifest.profile_contract(profile)
    provenance = {
        "topeft_git_commit": "historical", "environment_file": str(tmp_path / "env.tar.gz"),
        "environment_file_sha256": "a" * 64, "environment_fingerprint": "b" * 64,
        "environment_mode": "explicit_snapshot", "topcoffea_git_commit": "c" * 40,
        "topcoffea_relevant_source_fingerprint": "d" * 64,
    }
    current_files = manifest.scientific_files(contract)
    historical_files = {
        name: digest for name, digest in current_files.items()
        if name.startswith("input_samples/") or name in {
            "topeft/modules/axes.py", "topeft/channels/ch_lst.json",
        }
    }
    historical_blocks = []
    for spec in specs:
        historical_blocks.append({
            "id": spec["id"], "years": spec["years"],
            "category_groups": spec["category_groups"], "histograms": spec["histograms"],
            "output_tag": spec["output_tag"], "output_name": spec["output_name"],
            "expected_outputs": [spec["outputs"]["source"], spec["outputs"]["data_driven"]],
            "expected_nominal_path": spec["outputs"]["source"],
            "expected_np_path": spec["outputs"]["data_driven"],
            "plan_revision": revision, "source_status": "planned",
            "nonprompt_status": "blocked", "transitions": [],
        })
    target = sorted({key.split("|", 1)[1] for key in coverage})
    legacy = {
        "schema_version": 8,
        **identity,
        "topeft_git_commit": "historical",
        "env_file_sha256": provenance["environment_file_sha256"],
        "environment_fingerprint": provenance["environment_fingerprint"],
        "topcoffea_git_commit": provenance["topcoffea_git_commit"],
        "topcoffea_relevant_source_fingerprint": provenance["topcoffea_relevant_source_fingerprint"],
        "ttgamma_sample_role_policy": "split", "do_systs": True, "do_np": True,
        "region": "SR", "nonprompt_mode": "separate",
        "sr_reco_scientific_files": historical_files,
        "sr_reco_contract_files": historical_files,
        "sr_reco_target_coverage": target,
        "sr_reco_chunk_size": "25000",
        "active_sr_reco_revision": revision,
        "sr_reco_plan_revisions": [{"id": revision, "blocks": historical_blocks}],
        "blocks": historical_blocks,
        "created_at_utc": "2026-10-19T00:00:00Z",
    }
    legacy["sr_reco_scientific_contract_sha256"] = manifest.old_schema8_contract_digest(
        legacy, current_policy=False
    )
    legacy["sr_reco_current_scientific_contract_sha256"] = manifest.old_schema8_contract_digest(
        legacy, current_policy=True
    )
    legacy["sr_stage_policy"] = {
        "scientific_contract_sha256": legacy["sr_reco_current_scientific_contract_sha256"],
        "effective_nonprompt_mode": "separate",
    }
    old_fields = (
        "production_profile", "campaign_tag", "output_dir", "topeft_git_commit",
        "env_file_sha256", "environment_fingerprint", "topcoffea_git_commit",
        "topcoffea_relevant_source_fingerprint", "ttgamma_sample_role_policy",
        "do_systs", "do_np", "region", "nonprompt_mode",
    )
    old_identity = {key: legacy[key] for key in old_fields}
    old_identity["files"] = historical_files
    old_identity["chunk_size"] = "25000"
    legacy["sr_reco_contract_sha256"] = manifest.canonical_digest(old_identity)
    completed = legacy["blocks"][0]
    source_path = completed["expected_nominal_path"]
    downstream_path = completed["expected_np_path"]
    write_artifact(source_path)
    source_receipt = manifest.artifact_receipt(source_path)
    write_artifact(downstream_path, source_receipt["sha256"])
    completed["source_status"] = "ready"
    completed["nonprompt_status"] = "success"
    completed["source_receipt"] = source_receipt
    completed["nonprompt_receipt"] = manifest.artifact_receipt(downstream_path)
    completed["receipt_contract_sha256"] = legacy["sr_reco_current_scientific_contract_sha256"]
    def historical_source_command(block):
        return ["./fullR3_run.sh", "-y", *block["years"],
                "--category-groups", *block["category_groups"],
                "--hist-vars", *block["histograms"], "-t", block["output_tag"],
                "-p", str(root), "--all-analysis",
                "--sample-universe-wrapper", "run_cr.sh -> fullR3_run.sh",
                "--env-file", provenance["environment_file"]]

    completed["source_command_argv"] = historical_source_command(completed)
    completed["source_runtime"] = {
        "scientific_contract_sha256": legacy["sr_reco_current_scientific_contract_sha256"],
        "plan_revision": revision,
        "environment_fingerprint": provenance["environment_fingerprint"],
        "environment_file_sha256": provenance["environment_file_sha256"],
        "environment_file": provenance["environment_file"],
        "source_command_sha256": manifest.canonical_digest(completed["source_command_argv"]),
    }
    completed["source_runtime_attempts"] = [completed["source_runtime"]]
    completed["receipt_plan_revision"] = revision
    completed["receipt_runtime_sha256"] = manifest.canonical_digest(completed["source_runtime"])
    running = legacy["blocks"][1]
    running["source_status"] = "running"
    running["source_command_argv"] = historical_source_command(running)
    running["source_runtime"] = {
        **completed["source_runtime"],
        "source_command_sha256": manifest.canonical_digest(running["source_command_argv"]),
    }
    running["source_runtime_attempts"] = [running["source_runtime"]]
    real_style_state = manifest.new_state(specs, coverage, identity, contract, provenance)
    write_artifact(running["expected_nominal_path"], state=real_style_state, unit=real_style_state["units"][1])
    legacy_path = root / ".run3_sr_ext_campaign_state.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    original_sha256 = manifest.digest_file(legacy_path)
    imported = manifest.import_schema8_state(
        legacy, legacy_path, (None, identity, contract, provenance, coverage)
    )
    assert manifest.digest_file(legacy_path) == original_sha256
    assert imported["schema_version"] == 9
    assert imported["provenance"]["schema8_import"]["original_state_sha256"] == original_sha256
    assert len(imported["units"]) == 12
    assert imported["units"][0]["stages"]["source"]["receipt"]["artifact"] == source_receipt
    assert imported["units"][0]["stages"]["data_driven"]["receipt"]["artifact"] == completed["nonprompt_receipt"]
    assert imported["units"][1]["stages"]["source"]["status"] == "running"
    assert imported["units"][1]["id"] == "run3_sr_ext_size25_18_04"
    assert not Path(running["expected_np_path"]).exists()
    assert manifest.reconcile_running(imported)
    assert imported["units"][1]["stages"]["source"]["status"] == "success"
    assert imported["units"][1]["stages"]["data_driven"]["status"] == "pending"
    assert imported["units"][2]["stages"]["source"]["attempts"] == []
    assert "analysis/topeft_run2/run_data_driven.py" not in historical_files
    assert manifest.digest_file(legacy_path) == original_sha256
    assert manifest.validate_state(imported, verify_artifacts=False)
    with pytest.raises(ValueError, match="NEW_CAMPAIGN_REQUIRED"):
        manifest.import_schema8_state(
            legacy, legacy_path,
            (None, identity, contract, dict(provenance, environment_file_sha256="f" * 64), coverage),
        )
    assert manifest.digest_file(legacy_path) == original_sha256


def test_running_source_with_verified_artifact_is_reconciled(make_campaign):
    state = make_campaign()
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(source_command(state, unit)))
    write_artifact(unit["outputs"]["source"], state=state, unit=unit)
    prepared = json.loads(Path(f"{unit['outputs']['source']}.metadata.json").read_text())
    assert {item["input_kind"] for item in prepared["production_sample_contract"]["cfg_identities"]} == {"cfg"}
    for product in ("nonprompt", "flips"):
        assert prepared["requested_data_driven_products"]["products"][product]["enabled"] is True
        assert prepared["resolved_data_driven_contract"]["products"][product]["enabled"] is True
    assert manifest.reconcile_running(state)
    assert unit["stages"]["source"]["status"] == "success"
    assert unit["stages"]["source"]["attempts"][0]["result"] == "reconciled_success"
    assert unit["stages"]["data_driven"]["status"] == "pending"
    assert unit["stages"]["source"]["receipt"]["binding"]["attempt_sha256"] == manifest.attempt_identity(
        unit["stages"]["source"]["attempts"][0]
    )


@pytest.mark.parametrize("damage", (
    "missing_cfg", "extra_cfg", "duplicate_cfg", "wrong_cfg", "wrong_basename",
    "stale_input_kind", "wrong_input_kind", "contract_version", "contract_mode",
    "applicability_version", "applicability_mode", "applicability_query",
    "families", "nominal_layout", "nominal_schema", "artifact_kind", "lineage",
    "artifact_sha", "artifact_size", "synthetic", "sumw2_mode", "sumw2_schema",
    "flips_disabled",
))
def test_running_source_rejects_incomplete_or_contradictory_sidecar(make_campaign, damage):
    state = make_campaign()
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(source_command(state, unit)))
    path = unit["outputs"]["source"]
    write_artifact(path, state=state, unit=unit)
    metadata = Path(f"{path}.metadata.json")
    sidecar = json.loads(metadata.read_text(encoding="utf-8"))
    artifact = sidecar["artifact"]
    contract = sidecar["production_sample_contract"]
    applicability = sidecar["histogram_applicability"]
    sumw2 = sidecar["sumw2_storage_provenance"]
    if damage == "missing_cfg":
        contract["cfg_identities"].pop()
    elif damage == "extra_cfg":
        contract["cfg_identities"].append({
            "basename": "extra.cfg", "content_sha256": "0" * 64, "input_kind": "cfg",
        })
        contract["cfg_identities"].sort(key=lambda item: tuple(sorted(item.items())))
    elif damage == "duplicate_cfg":
        contract["cfg_identities"].append(dict(contract["cfg_identities"][0]))
    elif damage == "wrong_cfg":
        contract["cfg_identities"][0]["content_sha256"] = "0" * 64
    elif damage == "wrong_basename":
        contract["cfg_identities"][0]["basename"] = "wrong.cfg"
        contract["cfg_identities"].sort(key=lambda item: tuple(sorted(item.items())))
    elif damage == "stale_input_kind":
        contract["cfg_identities"][0]["input_kind"] = "sample_cfg"
    elif damage == "wrong_input_kind":
        contract["cfg_identities"][0]["input_kind"] = "other"
    elif damage == "contract_version":
        contract["contract_version"] = 999
    elif damage == "contract_mode":
        contract["resolved_mode"] = "full_diagnostics"
    elif damage == "applicability_version":
        applicability["contract_version"] = 999
    elif damage == "applicability_mode":
        applicability["analysis_mode"] = "offz"
    elif damage == "applicability_query":
        applicability["producer_query"] = "other_query"
    elif damage == "families":
        applicability["families"].pop(unit["histograms"][-1])
    elif damage == "nominal_layout":
        artifact["nominal_container_layout"] = "other_layout"
    elif damage == "nominal_schema":
        artifact["nominal_container_schema_version"] = 999
    elif damage == "artifact_kind":
        artifact["artifact_kind"] = "flips_output"
    elif damage == "lineage":
        sidecar["lineage"]["inputs"] = [{"pkl_basename": "other.pkl.gz", "artifact_kind": "processor_output", "pkl_sha256": "0" * 64}]
    elif damage == "artifact_sha":
        artifact["pkl_sha256"] = "0" * 64
    elif damage == "artifact_size":
        artifact["pkl_size_bytes"] += 1
    elif damage == "synthetic":
        sidecar["synthetic"] = True
    elif damage == "sumw2_mode":
        sumw2["requested_mode"] = "full_diagnostics"
    elif damage == "sumw2_schema":
        sumw2["schema_version"] = 999
    elif damage == "flips_disabled":
        sidecar["requested_data_driven_products"]["products"]["flips"]["enabled"] = False
        sidecar["resolved_data_driven_contract"]["products"]["flips"]["enabled"] = False
    if damage in {"missing_cfg", "extra_cfg", "duplicate_cfg", "wrong_cfg", "wrong_basename",
                  "stale_input_kind", "wrong_input_kind", "contract_version", "contract_mode"}:
        contract["contract_identity_sha256"] = manifest.canonical_digest({
            key: value for key, value in contract.items() if key != "contract_identity_sha256"
        })
    metadata.write_text(json.dumps(sidecar), encoding="utf-8")
    failure = ("sample cfg differs from recorded science"
               if damage in {"wrong_cfg", "wrong_basename", "stale_input_kind", "wrong_input_kind"}
               else "recovery_evidence_insufficient|artifact sidecar identity mismatch")
    with pytest.raises(ValueError, match=failure):
        manifest.reconcile_running(state)
    assert unit["stages"]["source"]["status"] == "running"


def test_source_wrapper_spelling_is_provenance_only(make_campaign):
    state = make_campaign()
    unit = state["units"][0]
    command = source_command(state, unit)
    command[command.index("--sample-universe-wrapper") + 1] = "same-science-renamed-wrapper"
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(command))
    write_artifact(unit["outputs"]["source"], state=state, unit=unit,
                   wrapper_identity="same-science-renamed-wrapper")
    assert manifest.reconcile_running(state)
    assert unit["stages"]["source"]["status"] == "success"


def test_source_scientific_command_change_rejects_recovery(make_campaign):
    state = make_campaign()
    unit = state["units"][0]
    command = source_command(state, unit)
    command[command.index("--hist-vars") + 1] = "different_family"
    with pytest.raises(ValueError, match="source command --hist-vars contradicts unit or science"):
        manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(command))


@pytest.mark.parametrize("telemetry", (False, True))
def test_running_data_driven_with_real_style_lineage_is_reconciled(make_campaign, telemetry):
    state = make_campaign("run2_stat_diagnostic")
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(source_command(state, unit)))
    write_artifact(unit["outputs"]["source"], state=state, unit=unit, real_payload=True)
    manifest.transition(state, unit["id"], "source", "ready", "0", "test")
    command = ("python", "./run_data_driven.py", "--input-pkl", unit["outputs"]["source"],
               "--output-pkl", unit["outputs"]["data_driven"])
    command += ("--phase-telemetry",) if telemetry else ()
    manifest.transition(state, unit["id"], "data_driven", "running", "none", "test", command)
    run_data_driven.main(list(command[2:]))
    assert manifest.reconcile_running(state)
    assert unit["stages"]["data_driven"]["status"] == "success"
    assert unit["stages"]["data_driven"]["attempts"][0]["result"] == "reconciled_success"
    output = unit["outputs"]["data_driven"]
    metadata = json.loads(Path(output + ".metadata.json").read_text())
    assert metadata["artifact"]["artifact_kind"] == "nonprompt_output"
    observed = get_hist_from_pkl(output)[scalar_nominal_key("ptll")].view(flow=True, as_dict=True)
    totals = {process: sum(float(np.asarray(values).sum()) for key, values in observed.items()
                           if key[0] == process) for process in ("nonpromptUL18", "flipsUL18")}
    assert totals == {"nonpromptUL18": 7.0, "flipsUL18": 4.0}
    telemetry_path = Path(output + ".phase_telemetry.jsonl")
    assert telemetry_path.exists() == telemetry
    if telemetry:
        records = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
        assert records[-1]["phase"] == "dd_published"


@pytest.mark.parametrize("telemetry", (False, True))
@pytest.mark.parametrize("damage", (None, "input", "output", "only_flips", "nominal_reference",
                                    "unknown", "duplicate_telemetry", "telemetry_value"))
def test_deferred_command_observability_keeps_scientific_contract(make_campaign, telemetry, damage):
    state = make_campaign()
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test",
                        tuple(source_command(state, unit)))
    write_artifact(unit["outputs"]["source"], state=state, unit=unit)
    manifest.transition(state, unit["id"], "source", "ready", "0", "test")
    command = ["python", "./run_data_driven.py", "--input-pkl", unit["outputs"]["source"],
               "--output-pkl", unit["outputs"]["data_driven"]]
    if telemetry:
        command.append("--phase-telemetry")
    if damage in ("input", "output"):
        command[command.index("--input-pkl" if damage == "input" else "--output-pkl") + 1] += ".wrong"
    elif damage:
        command.extend({"only_flips": ["--only-flips"], "nominal_reference": ["--nominal-only-reference"],
                        "unknown": ["--unknown"], "duplicate_telemetry": ["--phase-telemetry", "--phase-telemetry"],
                        "telemetry_value": ["--phase-telemetry", "true"]}[damage])
    command += ["--pkl-no-clobber", "--pkl-min-free-bytes", "2", "--pkl-min-free-inodes", "4",
                "--pkl-safety-reserve-bytes", "1", "--pkl-safety-reserve-inodes", "1"]
    before = json.loads(json.dumps(state))
    if damage:
        with pytest.raises(ValueError):
            manifest.transition(state, unit["id"], "data_driven", "running", "none", "test", tuple(command))
        assert state == before
    else:
        manifest.transition(state, unit["id"], "data_driven", "running", "none", "test", tuple(command))
        assert unit["stages"]["data_driven"]["attempts"][0]["command_argv"] == command
        parsed = run_data_driven._build_argument_parser().parse_args(command[2:])
        assert parsed.phase_telemetry == telemetry
        assert not parsed.only_flips and not parsed.nominal_only_reference


@pytest.mark.parametrize("damage", (
    "lineage", "merged", "nominal_schema", "nominal_layout", "requested_products",
    "resolved_contract", "transformation_contract", "sumw2_manifest",
    "sumw2_provenance", "sample_contract", "applicability", "metadata_schema",
    "artifact_sha", "artifact_size",
))
def test_running_downstream_rejects_damaged_real_sidecar(make_campaign, damage):
    state = make_campaign("run2_stat_diagnostic", histograms=("ptll",))
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(source_command(state, unit)))
    write_artifact(unit["outputs"]["source"], state=state, unit=unit, real_payload=True)
    manifest.transition(state, unit["id"], "source", "ready", "0", "test")
    command = ("python", "./run_data_driven.py", "--input-pkl", unit["outputs"]["source"],
               "--output-pkl", unit["outputs"]["data_driven"])
    manifest.transition(state, unit["id"], "data_driven", "running", "none", "test", command)
    run_data_driven.main(["--input-pkl", unit["outputs"]["source"],
                          "--output-pkl", unit["outputs"]["data_driven"], "--quiet"])
    metadata = Path(f"{unit['outputs']['data_driven']}.metadata.json")
    sidecar = json.loads(metadata.read_text(encoding="utf-8"))
    if damage == "lineage":
        sidecar["lineage"]["inputs"][0]["pkl_sha256"] = "0" * 64
    elif damage == "merged":
        sidecar["artifact"]["merged"] = True
    elif damage == "nominal_schema":
        sidecar["artifact"]["nominal_container_schema_version"] = 999
    elif damage == "nominal_layout":
        sidecar["artifact"]["nominal_container_layout"] = "other"
    elif damage == "requested_products":
        sidecar["requested_data_driven_products"]["products"]["flips"]["enabled"] = False
    elif damage == "resolved_contract":
        sidecar["resolved_data_driven_contract"]["products"]["flips"]["enabled"] = False
    elif damage == "transformation_contract":
        del sidecar["transformation_contract"]
    elif damage == "sumw2_manifest":
        del sidecar["sumw2_content_manifest"]
    elif damage == "sumw2_provenance":
        sidecar["sumw2_storage_provenance"]["requested_mode"] = "production"
    elif damage == "sample_contract":
        sidecar["production_sample_contract"]["cfg_identities"].pop()
    elif damage == "applicability":
        sidecar["histogram_applicability"]["analysis_mode"] = "offz"
    elif damage == "metadata_schema":
        sidecar["metadata_schema_version"] = 999
    elif damage == "artifact_sha":
        sidecar["artifact"]["pkl_sha256"] = "0" * 64
    elif damage == "artifact_size":
        sidecar["artifact"]["pkl_size_bytes"] += 1
    metadata.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="recovery_evidence_insufficient|artifact sidecar identity mismatch"):
        manifest.reconcile_running(state)
    assert unit["stages"]["data_driven"]["status"] == "running"


def test_tiny_run2_inline_and_separate_data_driven_parity(make_campaign):
    state = make_campaign("run2_stat_diagnostic", histograms=("ptll",))
    unit = state["units"][0]
    source_path = unit["outputs"]["source"]
    write_artifact(source_path, state=state, unit=unit, real_payload=True)
    source_histograms = get_hist_from_pkl(source_path)
    inline = DataDrivenProducer(source_histograms, "").getDataDrivenHistogram()
    run_data_driven.main(["--input-pkl", source_path,
                          "--output-pkl", unit["outputs"]["data_driven"], "--quiet"])
    separate = get_hist_from_pkl(unit["outputs"]["data_driven"])
    assert set(inline) == set(separate)
    for key in inline:
        expected = inline[key].view(flow=True, as_dict=True)
        observed = separate[key].view(flow=True, as_dict=True)
        assert set(expected) == set(observed)
        for sparse_key in expected:
            np.testing.assert_array_equal(np.asarray(expected[sparse_key]), np.asarray(observed[sparse_key]))
    nominal = separate[scalar_nominal_key("ptll")].view(flow=True, as_dict=True)
    totals = {
        process: sum(float(np.asarray(values).sum()) for key, values in nominal.items()
                     if key[0] == process)
        for process in ("nonpromptUL18", "flipsUL18")
    }
    assert totals == {"nonpromptUL18": 7.0, "flipsUL18": 4.0}
    assert sum(totals.values()) == 11.0


@pytest.mark.parametrize("wrong_kind", ("flips_output", "processor_output"))
def test_running_downstream_rejects_wrong_kind(make_campaign, wrong_kind):
    state = make_campaign("run2_stat_diagnostic")
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(source_command(state, unit)))
    write_artifact(unit["outputs"]["source"], state=state, unit=unit, real_payload=True)
    manifest.transition(state, unit["id"], "source", "ready", "0", "test")
    command = ("python", "./run_data_driven.py", "--input-pkl", unit["outputs"]["source"],
               "--output-pkl", unit["outputs"]["data_driven"])
    manifest.transition(state, unit["id"], "data_driven", "running", "none", "test", command)
    path = unit["outputs"]["data_driven"]
    run_data_driven.main(["--input-pkl", unit["outputs"]["source"],
                          "--output-pkl", path, "--quiet"])
    metadata = Path(f"{path}.metadata.json")
    sidecar = json.loads(metadata.read_text(encoding="utf-8"))
    sidecar["artifact"]["artifact_kind"] = wrong_kind
    metadata.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="recovery_evidence_insufficient"):
        manifest.reconcile_running(state)


@pytest.mark.parametrize("other_product_option", ("--only-flips", "--nominal-only-reference"))
def test_ordinary_deferred_command_rejects_other_product_options(make_campaign, other_product_option):
    state = make_campaign()
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(source_command(state, unit)))
    write_artifact(unit["outputs"]["source"], state=state, unit=unit)
    manifest.transition(state, unit["id"], "source", "ready", "0", "test")
    command = ("python", "./run_data_driven.py", other_product_option,
               "--input-pkl", unit["outputs"]["source"],
               "--output-pkl", unit["outputs"]["data_driven"])
    with pytest.raises(ValueError, match="ordinary deferred nonprompt artifact contract"):
        manifest.transition(state, unit["id"], "data_driven", "running", "none", "test", command)
    assert unit["stages"]["data_driven"]["attempts"] == []


def test_stage_science_is_local_and_topology_is_operational(make_campaign, monkeypatch):
    state = make_campaign()
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(source_command(state, unit)))
    write_artifact(unit["outputs"]["source"], state=state, unit=unit)
    manifest.transition(state, unit["id"], "source", "ready", "0", "test")
    provenance = state["provenance"]["initial"]
    original_files = manifest.scientific_files

    def downstream_change(contract, stage="source"):
        files = original_files(contract, stage)
        if stage == "data_driven":
            files["topeft/modules/nonprompt_policy.py"] = "f" * 64
        return files

    monkeypatch.setattr(manifest, "scientific_files", downstream_change)
    manifest.validate_resume(state, state["campaign_identity"], state["profile_contract"],
                             provenance, state["scientific_identity"]["scientific_coverage"])
    source_binding = copy.deepcopy(unit["stages"]["source"]["receipt"]["binding"])
    assert not manifest.record_attempt_provenance(state, provenance)
    assert manifest.validate_receipt(state, unit, "source")
    assert unit["stages"]["source"]["receipt"]["binding"] == source_binding
    command = ("python", "./run_data_driven.py", "--input-pkl", unit["outputs"]["source"],
               "--output-pkl", unit["outputs"]["data_driven"])
    with pytest.raises(ValueError, match="data_driven_scientific_contract_incompatible"):
        manifest.transition(state, unit["id"], "data_driven", "running", "none", "test", command)
    assert unit["stages"]["data_driven"]["attempts"] == []
    changed_topology = dict(state["profile_contract"], stages=["producer", "postprocess"])
    assert manifest.scientific_identity(changed_topology, provenance["environment_file_sha256"],
                                        provenance["topcoffea_relevant_source_fingerprint"],
                                        state["scientific_identity"]["scientific_coverage"])["scientific_contract_sha256"] == state["scientific_identity"]["scientific_contract_sha256"]

    def source_change(contract, stage="source"):
        files = original_files(contract, stage)
        if stage == "source":
            name = next(name for name in files if name.startswith("input_samples/"))
            files[name] = "f" * 64
        return files

    monkeypatch.setattr(manifest, "scientific_files", source_change)
    with pytest.raises(ValueError, match="scientific_contract_incompatible"):
        manifest.validate_resume(state, state["campaign_identity"], state["profile_contract"],
                                 provenance, state["scientific_identity"]["scientific_coverage"])


def test_historical_dd_wrapper_binding_survives_technical_change(make_campaign, monkeypatch):
    state = make_campaign()
    accepted, pending = state["units"]
    wrapper = "analysis/topeft_run2/run_data_driven.py"
    historical_sha = "724a46b07e1d9d7b17888fa79acd2fddb1a8cd057f9989a12029c516c4538746"
    assert manifest.digest_file(manifest.REPOSITORY_ROOT / wrapper) != historical_sha
    files = state["scientific_identity"]["data_driven_stage_scientific_contract"]["scientific_files"]
    files[wrapper] = historical_sha
    state["scientific_identity"]["data_driven_stage_scientific_contract_sha256"] = manifest.canonical_digest(
        state["scientific_identity"]["data_driven_stage_scientific_contract"]
    )
    manifest.validate_state(state, verify_artifacts=False)

    manifest.transition(state, accepted["id"], "source", "running", "none", "test",
                        tuple(source_command(state, accepted)))
    write_artifact(accepted["outputs"]["source"], state=state, unit=accepted)
    manifest.transition(state, accepted["id"], "source", "ready", "0", "test")
    dd_command = ("python", "./run_data_driven.py", "--input-pkl", accepted["outputs"]["source"],
                  "--output-pkl", accepted["outputs"]["data_driven"])
    manifest.transition(state, accepted["id"], "data_driven", "running", "none", "test", dd_command)
    write_artifact(accepted["outputs"]["data_driven"],
                   accepted["stages"]["source"]["receipt"]["artifact"]["sha256"],
                   state=state, unit=accepted)
    manifest.transition(state, accepted["id"], "data_driven", "success", "0", "test")
    historical_science = copy.deepcopy(state["scientific_identity"])
    historical_attempt = copy.deepcopy(accepted["stages"]["data_driven"]["attempts"])
    historical_receipt = copy.deepcopy(accepted["stages"]["data_driven"]["receipt"])
    artifact_sha = manifest.digest_file(accepted["outputs"]["data_driven"])

    original_read_text = Path.read_text

    def technical_only_wrapper_change(path, *args, **kwargs):
        source = original_read_text(path, *args, **kwargs)
        return source + "\n# technical output-path handling\n" if str(path).endswith(wrapper) else source

    monkeypatch.setattr(Path, "read_text", technical_only_wrapper_change)
    provenance = dict(state["provenance"]["last_resolved"], topeft_git_commit="technical-change")
    manifest.validate_resume(state, state["campaign_identity"], state["profile_contract"],
                             provenance, historical_science["scientific_coverage"])
    assert manifest.record_attempt_provenance(state, provenance)
    manifest.transition(state, pending["id"], "source", "running", "none", "test",
                        tuple(source_command(state, pending)))
    write_artifact(pending["outputs"]["source"], state=state, unit=pending)
    manifest.transition(state, pending["id"], "source", "ready", "0", "test")
    pending_dd_command = ("python", "./run_data_driven.py", "--input-pkl", pending["outputs"]["source"],
                          "--output-pkl", pending["outputs"]["data_driven"])
    manifest.transition(state, pending["id"], "data_driven", "running", "none", "test", pending_dd_command)
    assert state["scientific_identity"] == historical_science
    assert accepted["stages"]["data_driven"]["attempts"] == historical_attempt
    assert accepted["stages"]["data_driven"]["receipt"] == historical_receipt
    assert manifest.digest_file(accepted["outputs"]["data_driven"]) == artifact_sha
    assert manifest.validate_receipt(state, accepted, "data_driven")


def test_wrapper_filter_science_change_fails_at_dd_boundary(make_campaign, monkeypatch):
    state = make_campaign()
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test",
                        tuple(source_command(state, unit)))
    write_artifact(unit["outputs"]["source"], state=state, unit=unit)
    manifest.transition(state, unit["id"], "source", "ready", "0", "test")
    original_read_text = Path.read_text

    def changed_filter(path, *args, **kwargs):
        source = original_read_text(path, *args, **kwargs)
        if str(path).endswith("analysis/topeft_run2/run_data_driven.py"):
            return source.replace("if not flips:\n", "if flips:\n", 1)
        return source

    monkeypatch.setattr(Path, "read_text", changed_filter)
    manifest.validate_resume(state, state["campaign_identity"], state["profile_contract"],
                             state["provenance"]["initial"], state["scientific_identity"]["scientific_coverage"])
    command = ("python", "./run_data_driven.py", "--input-pkl", unit["outputs"]["source"],
               "--output-pkl", unit["outputs"]["data_driven"])
    with pytest.raises(ValueError, match="data_driven_scientific_contract_incompatible"):
        manifest.transition(state, unit["id"], "data_driven", "running", "none", "test", command)
    assert unit["stages"]["data_driven"]["attempts"] == []


def test_frozen_tarball_cannot_be_replaced_by_provenance_or_plan(make_campaign, tmp_path):
    state = make_campaign()
    initial = copy.deepcopy(state)
    frozen_sha = state["scientific_identity"]["source_stage_scientific_contract"]["frozen_environment_tarball_sha256"]
    changed = dict(state["provenance"]["initial"], environment_file_sha256="f" * 64)
    with pytest.raises(ValueError, match="NEW_CAMPAIGN_REQUIRED"):
        manifest.record_attempt_provenance(state, changed)
    assert state == initial
    state["provenance"]["last_resolved"] = changed
    with pytest.raises(ValueError, match="NEW_CAMPAIGN_REQUIRED"):
        manifest.validate_state(state, verify_artifacts=False)
    with pytest.raises(ValueError, match="NEW_CAMPAIGN_REQUIRED"):
        manifest.transition(state, state["units"][0]["id"], "source", "running", "none", "retry",
                            tuple(source_command(state, state["units"][0])))
    state["provenance"]["last_resolved"] = state["provenance"]["initial"]
    units = [{key: unit[key] for key in ("id", "years", "category_groups", "histograms")} for unit in state["units"]]
    plan = tmp_path / "same_science.json"
    plan.write_text(json.dumps({"units": units}), encoding="utf-8")
    manifest.activate_plan(state, plan, state["plan"]["active_revision"])
    assert state["scientific_identity"]["source_stage_scientific_contract"]["frozen_environment_tarball_sha256"] == frozen_sha


def test_downstream_calculation_dependency_and_provenance_helper_boundaries(make_campaign, monkeypatch):
    state = make_campaign()
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(source_command(state, unit)))
    write_artifact(unit["outputs"]["source"], state=state, unit=unit)
    manifest.transition(state, unit["id"], "source", "ready", "0", "test")
    command = ("python", "./run_data_driven.py", "--input-pkl", unit["outputs"]["source"],
               "--output-pkl", unit["outputs"]["data_driven"])
    manifest.transition(state, unit["id"], "data_driven", "running", "none", "test", command)
    write_artifact(unit["outputs"]["data_driven"],
                   unit["stages"]["source"]["receipt"]["artifact"]["sha256"], state=state, unit=unit)
    manifest.transition(state, unit["id"], "data_driven", "success", "0", "test")
    baseline_source = state["scientific_identity"]["source_stage_scientific_contract_sha256"]
    baseline_downstream = state["scientific_identity"]["data_driven_stage_scientific_contract_sha256"]
    original_digest = manifest.digest_file

    def helper_only_change(path):
        if str(path).endswith("production_sample_profile.py"):
            return "f" * 64
        return original_digest(path)

    monkeypatch.setattr(manifest, "digest_file", helper_only_change)
    current = manifest.scientific_identity(
        state["profile_contract"], state["provenance"]["initial"]["environment_file_sha256"],
        state["provenance"]["initial"]["topcoffea_relevant_source_fingerprint"],
        state["scientific_identity"]["scientific_coverage"],
    )
    assert current["data_driven_stage_scientific_contract_sha256"] == baseline_downstream

    def calculation_change(path):
        if str(path).endswith("topeft/modules/nonprompt_policy.py"):
            return "f" * 64
        return original_digest(path)

    monkeypatch.setattr(manifest, "digest_file", calculation_change)
    changed = manifest.scientific_identity(
        state["profile_contract"], state["provenance"]["initial"]["environment_file_sha256"],
        state["provenance"]["initial"]["topcoffea_relevant_source_fingerprint"],
        state["scientific_identity"]["scientific_coverage"],
    )
    assert changed["source_stage_scientific_contract_sha256"] == baseline_source
    assert changed["data_driven_stage_scientific_contract_sha256"] != baseline_downstream
    manifest.validate_resume(
        state, state["campaign_identity"], state["profile_contract"],
        state["provenance"]["initial"], state["scientific_identity"]["scientific_coverage"],
    )
    next_unit = state["units"][1]
    manifest.transition(state, next_unit["id"], "source", "running", "none", "test",
                        tuple(source_command(state, next_unit)))
    write_artifact(next_unit["outputs"]["source"], state=state, unit=next_unit)
    manifest.transition(state, next_unit["id"], "source", "ready", "0", "test")
    command = ("python", "./run_data_driven.py", "--input-pkl", next_unit["outputs"]["source"],
               "--output-pkl", next_unit["outputs"]["data_driven"])
    with pytest.raises(ValueError, match="data_driven_scientific_contract_incompatible"):
        manifest.transition(state, next_unit["id"], "data_driven", "running", "none", "test", command)
    assert manifest.validate_receipt(state, unit, "source")


def test_downstream_science_uses_resolved_values_and_source_artifact(make_campaign, monkeypatch):
    state = make_campaign()
    contract = state["profile_contract"]
    provenance = state["provenance"]["initial"]
    coverage = state["scientific_identity"]["scientific_coverage"]
    baseline = manifest.scientific_identity(contract, provenance["environment_file_sha256"],
                                            provenance["topcoffea_relevant_source_fingerprint"], coverage)
    original_digest = manifest.digest_file

    def technical_change(path):
        if str(path).endswith("topcoffea/modules/histEFT.py"):
            return "f" * 64
        return original_digest(path)

    monkeypatch.setattr(manifest, "digest_file", technical_change)
    assert manifest.scientific_identity(contract, provenance["environment_file_sha256"],
                                        provenance["topcoffea_relevant_source_fingerprint"], coverage)[
        "data_driven_stage_scientific_contract_sha256"] == baseline[
        "data_driven_stage_scientific_contract_sha256"]
    original_parameters = json.loads((manifest.REPOSITORY_ROOT / "topeft/params/params.json").read_text())
    changed_parameters = dict(original_parameters, prompt_subtraction_samples=["changed"])
    original_read_text = Path.read_text

    def changed_value(path, *args, **kwargs):
        if str(path).endswith("topeft/params/params.json"):
            return json.dumps(changed_parameters)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", changed_value)
    assert manifest.scientific_identity(contract, provenance["environment_file_sha256"],
                                        provenance["topcoffea_relevant_source_fingerprint"], coverage)[
        "data_driven_stage_scientific_contract_sha256"] != baseline[
        "data_driven_stage_scientific_contract_sha256"]
    monkeypatch.setattr(Path, "read_text", original_read_text)
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(source_command(state, unit)))
    write_artifact(unit["outputs"]["source"], state=state, unit=unit)
    manifest.transition(state, unit["id"], "source", "ready", "0", "test")
    command = ("python", "./run_data_driven.py", "--input-pkl", unit["outputs"]["source"],
               "--output-pkl", unit["outputs"]["data_driven"])
    manifest.transition(state, unit["id"], "data_driven", "running", "none", "test", command)
    first = manifest.canonical_digest(manifest.binding(state, unit, "data_driven"))
    unit["stages"]["source"]["receipt"]["artifact"]["sha256"] = "0" * 64
    assert manifest.canonical_digest(manifest.binding(state, unit, "data_driven")) != first


def test_running_artifact_without_attempt_binding_is_not_adopted(make_campaign):
    state = make_campaign()
    unit = state["units"][0]
    manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(source_command(state, unit)))
    write_artifact(unit["outputs"]["source"])
    with pytest.raises(ValueError, match="recovery_evidence_insufficient"):
        manifest.reconcile_running(state)
    assert unit["stages"]["source"]["status"] == "running"


def test_unidentified_existing_artifact_blocks_retry(make_campaign):
    state = make_campaign()
    unit = state["units"][0]
    Path(unit["outputs"]["source"]).write_bytes(b"unverified payload")
    with pytest.raises(ValueError, match="unverified output collision"):
        manifest.transition(state, unit["id"], "source", "running", "none", "test", tuple(source_command(state, unit)))
    assert not unit["stages"]["source"]["attempts"]


def test_corrupt_stage_relationship_is_rejected(make_campaign):
    state = make_campaign()
    state["units"][0]["stages"]["data_driven"]["status"] = "success"
    with pytest.raises(ValueError, match="contradictory stage statuses"):
        manifest.validate_state(state)


def test_changed_scientific_policy_is_rejected(make_campaign):
    state = make_campaign()
    wrong = dict(state["profile_contract"], systematics=False)
    with pytest.raises(ValueError, match="profile contract differs"):
        manifest.validate_resume(state, state["campaign_identity"], wrong,
                                 state["provenance"]["initial"],
                                 state["scientific_identity"]["scientific_coverage"])
    with pytest.raises(ValueError, match="systematics policy contradict"):
        manifest.profile_contract("run2_prod", do_systs=False)


def test_writer_lock_refuses_symlink_and_contending_process(tmp_path):
    import subprocess
    lock_state = tmp_path / ".run2_prod_campaign_state_v9.json"
    lock_path = tmp_path / ".run2_prod_campaign.lock"
    lock_path.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symbolic link"):
        with manifest.campaign_lock(lock_state):
            pass
    lock_path.unlink()
    with manifest.campaign_lock(lock_state):
        child = subprocess.run(
            [
                "/users/apiccine/work/correction-lib/codex-run.sh",
                "/bin/bash", "--noprofile", "--norc", "-c",
                'cd /users/apiccine/work/correction-lib/topeft && "$1" -c "$2" "$3"',
                "writer_lock_test",
                "/users/apiccine/work/miniconda3/envs/clib-env/bin/python",
                "from analysis.topeft_run2 import campaign_manifest as m\n"
                "import sys\n"
                "with m.campaign_lock(sys.argv[1]):\n"
                "    pass\n",
                str(lock_state),
            ],
            capture_output=True, text=True, check=False,
        )
        assert child.returncode != 0
        assert "campaign writer lock is held" in child.stderr


@pytest.mark.parametrize("shape", ("empty", "payload", "zero_payload", "metadata", "absent"))
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("final_present", (False, True))
def test_incomplete_pending_dd_does_not_block_independent_source(make_campaign, shape, reverse, final_present):
    state = make_campaign("run2_stat_diagnostic", histograms=("ptll",))
    # Preserve the immutable plan order; swap which unit carries each role.
    incomplete, independent = state["units"][::-1] if reverse else state["units"]
    for unit in (incomplete, independent):
        manifest.transition(state, unit["id"], "source", "running", "none", "test",
                            tuple(source_command(state, unit)))
    write_artifact(incomplete["outputs"]["source"], state=state, unit=incomplete, real_payload=True)
    manifest.transition(state, incomplete["id"], "source", "ready", "0", "test")
    dd_command = ("python", "./run_data_driven.py", "--input-pkl", incomplete["outputs"]["source"],
                  "--output-pkl", incomplete["outputs"]["data_driven"])
    manifest.transition(state, incomplete["id"], "data_driven", "running", "none", "test", dd_command)
    pending = manifest.pending_artifact_path(state, incomplete, "data_driven")
    if shape != "absent":
        pending.parent.mkdir(mode=0o700)
    if shape in {"payload", "zero_payload"}:
        pending.write_bytes(b"unfinished" if shape == "payload" else b"")
    if shape == "metadata":
        Path(f"{pending}.metadata.json").write_bytes(b"unfinished metadata")
    before = {p.name: p.read_bytes() for p in pending.parent.iterdir()} if shape != "absent" else None
    final = Path(independent["outputs"]["source"])
    write_artifact(final, state=state, unit=independent, real_payload=True,
                   pending_id=manifest.pending_artifact_id(state, independent, "source"))
    valid_pending = manifest.pending_artifact_path(state, independent, "source")
    if not final_present:
        final.unlink()
        Path(f"{final}.metadata.json").unlink()
    assert manifest.reconcile_running(state)
    failed = incomplete["stages"]["data_driven"]
    assert failed["status"] == "failed" and failed["receipt"] is None
    assert failed["attempts"][-1]["result"] == "interrupted"
    assert not Path(incomplete["outputs"]["data_driven"]).exists()
    assert independent["stages"]["source"]["status"] == "success"
    assert independent["stages"]["source"]["attempts"][-1]["result"] == "reconciled_success"
    assert independent["stages"]["source"]["receipt"]["artifact"]["path"] == str(final)
    assert os.path.samefile(valid_pending, final)
    if before is not None:
        assert pending.parent.is_dir()
        assert {p.name: p.read_bytes() for p in pending.parent.iterdir()} == before
    else:
        assert not pending.parent.exists()


@pytest.mark.parametrize("damage", ("digest", "kind", "binding", "collision"))
def test_complete_invalid_pending_blocks_later_independent_record(make_campaign, damage):
    state = make_campaign("run2_stat_diagnostic", histograms=("ptll",))
    invalid, independent = state["units"]
    for unit in (invalid, independent):
        manifest.transition(state, unit["id"], "source", "running", "none", "test",
                            tuple(source_command(state, unit)))
        write_artifact(unit["outputs"]["source"], state=state, unit=unit, real_payload=True,
                       pending_id=manifest.pending_artifact_id(state, unit, "source"))
    pending = manifest.pending_artifact_path(state, invalid, "source")
    sidecar = Path(f"{pending}.metadata.json")
    if damage in {"digest", "kind", "binding"}:
        metadata = json.loads(sidecar.read_text())
        if damage == "digest":
            metadata["artifact"]["pkl_sha256"] = "f" * 64
        elif damage == "kind":
            metadata["artifact"]["artifact_kind"] = "nonprompt_output"
        else:
            metadata["production_sample_contract"]["cfg_identities"][0]["content_sha256"] = "f" * 64
        sidecar.write_text(json.dumps(metadata))
    else:
        final = Path(invalid["outputs"]["source"])
        final.unlink()
        final.write_bytes(b"foreign final collision")
    before = {p.name: p.read_bytes() for p in pending.parent.iterdir()}
    with pytest.raises(ValueError):
        manifest.reconcile_running(state)
    assert all(unit["stages"]["source"]["status"] == "running" for unit in state["units"])
    assert all(unit["stages"]["source"]["receipt"] is None for unit in state["units"])
    assert {p.name: p.read_bytes() for p in pending.parent.iterdir()} == before
