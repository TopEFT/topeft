"""Inert transaction fixtures for the persistent SR plotting controller."""

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "analysis/topeft_run2/plot_campaign_manifest.py"
SPEC = importlib.util.spec_from_file_location("plot_campaign_manifest", MODULE_PATH)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def fixture_plan(tmp_path, monkeypatch, *, profile="sr_crlike_run2", unit_count=1):
    repo = tmp_path / "repo"
    (repo / "histos").mkdir(parents=True)
    monkeypatch.setattr(campaign, "REPO_ROOT", repo)
    root = repo / "histos" / "fixture_campaign"
    units = []
    for index in range(unit_count):
        component = "run3" if profile == "sr_crlike_run2_run3" and index else "run2"
        input_path = tmp_path / f"unit_{index}_SR.pkl.gz"
        input_path.write_bytes(b"inert fixture")
        sidecar_path = Path(str(input_path) + ".metadata.json")
        sidecar_path.write_text(json.dumps({
            "metadata_schema_version": 2,
            "artifact": {"pkl_basename": input_path.name,
                         "pkl_size_bytes": input_path.stat().st_size,
                         "pkl_sha256": "a" * 64},
            "sumw2_storage_provenance": {
                "runtime_histogram_families": ["invmass", "fwd0pt"]},
        }), encoding="utf-8")
        units.append({
            "unit_id": f"unit_{index}", "component": component,
            "producer_unit_id": f"producer_{index}",
            "input_path": str(input_path), "sidecar_path": str(sidecar_path),
            "qualification_source": {
                "status": "accepted", "locator": f"accepted_fixture_{index}",
                "input_sha256": "a" * 64,
                "sidecar_sha256": campaign.sha256_file(sidecar_path),
            },
            "period_view": "combined",
        })
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"schema_version": 1,
                                     "campaign_profile": profile,
                                     "campaign_tag": "fixture_campaign",
                                     "plan_revision": 1, "units": units}), encoding="utf-8")
    return root, plan_path


def fake_success(argv, attempt_root):
    chosen_period = argv[argv.index("--canonical-period") + 1]
    unit_id = argv[argv.index("--negative-report-suffix") + 1]
    output = attempt_root / chosen_period / "merged" / unit_id
    output.mkdir(parents=True)
    (output / f"{unit_id}_invmass_fitting.png").write_bytes(b"fixture-png")
    metadata = attempt_root / "_metadata"
    metadata.mkdir()
    (metadata / "renderer.log").write_text("fwd0pt absent in this region\n", encoding="utf-8")
    (metadata / "guard.jsonl").write_text(
        '{"event":"result","termination_kind":"normal","returncode":0}\n',
        encoding="utf-8",
    )
    write_lifecycle(argv, attempt_root, 0, 0, 0)
    return 0, 0, 0


def write_lifecycle(argv, attempt_root, guard_result, child_result, tee_result):
    guard_path = attempt_root / "_metadata/guard.jsonl"
    campaign.atomic_json(campaign.lifecycle_receipt_path(attempt_root), {
        "schema_version": 1, "attempt_id": attempt_root.name,
        "command_identity": campaign.identity(argv),
        "private_unit_root": str(attempt_root),
        "terminal_guard_evidence": campaign.guard_terminal(guard_path),
        "guard_evidence_sha256": campaign.sha256_file(guard_path),
        "guard_result": guard_result, "child_result": child_result,
        "tee_result": tee_result, "ended_at_utc": "2026-10-27T00:01:00Z",
    })


def fake_failure(argv, attempt_root):
    metadata = attempt_root / "_metadata"
    metadata.mkdir()
    (metadata / "renderer.log").write_text("fixture child failed\n", encoding="utf-8")
    (metadata / "guard.jsonl").write_text(
        '{"event":"result","termination_kind":"normal","returncode":7}\n',
        encoding="utf-8",
    )
    write_lifecycle(argv, attempt_root, 7, 7, 0)
    return 7, 7, 0


def seed_running(tmp_path, monkeypatch, *, profile="sr_crlike_run2", terminal="success"):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, profile=profile,
                                   unit_count=2 if profile.endswith("run2_run3") else 1)
    plan = campaign.normalize_plan(plan_path, profile, "fixture_campaign", root)
    runtime = campaign.runtime_identity()
    root.mkdir()
    states = {}
    for component in campaign.PROFILES[profile]:
        component_path = campaign.component_root(root, profile, component)
        component_path.mkdir(parents=True, exist_ok=True)
        states[component] = campaign.new_component_state(plan, component, runtime)
    unit = plan["units"][0]
    attempt_root = Path(unit["private_unit_root"])
    attempt_root.mkdir(parents=True)
    argv = campaign.child_argv(unit, attempt_root)
    if terminal == "failure":
        fake_failure(argv, attempt_root)
    else:
        fake_success(argv, attempt_root)
    attempt = {
        "attempt_id": "attempt_0001", "plan_revision": 1,
        "started_at_utc": "2026-10-27T00:00:00Z", "ended_at_utc": None,
        "command_identity": campaign.identity(argv), "private_unit_root": str(attempt_root),
        "completion_receipt_identity": None, "materialization_status": "pending",
        "guard_result": None, "child_result": None, "tee_result": None,
        "result": "running", "failure_classification": None,
    }
    states["run2"]["units"]["unit_0"]["status"] = "running"
    states["run2"]["units"]["unit_0"]["attempts"].append(attempt)
    for component, state in states.items():
        campaign.atomic_json(campaign.state_path(root, profile, component), state)
    campaign.atomic_json(campaign.coordinator_path(root), {
        "schema_version": 1, "campaign_profile": profile,
        "campaign_tag": "fixture_campaign", "runtime_contract_identity": runtime,
        "plan": plan,
    })
    return root, plan, attempt_root


def resume_args(root, profile="sr_crlike_run2", selected=None):
    args = ["--campaign-profile", profile, "--campaign-tag", "fixture_campaign",
            "-o", str(root), "--resume"]
    return args + (["--selected-unit", selected] if selected else [])


def test_dry_run_uses_sidecar_variables_without_root_or_output_prediction(tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)
    plan = campaign.normalize_plan(plan_path, "sr_crlike_run2", "fixture_campaign", root)
    campaign.dry_run(plan, root, None)
    output = capsys.readouterr().out
    assert '"variables": [' in output and '"fwd0pt"' in output
    assert '"private_unit_root"' in output and '"child_argv"' in output
    assert "observed_png" not in output and "category_invmass" not in output
    assert not root.exists()
    argv = campaign.child_argv(plan["units"][0], Path(plan["units"][0]["private_unit_root"]))
    for token in ("--sr", "--unblind", "--variables", "--suppress-merge-report"):
        assert token in argv
    for option, value in (("--binning", "fitting"), ("--channel-output", "merged"),
                          ("--workers", "1"), ("--output-formats", "png")):
        assert argv[argv.index(option) + 1] == value
    assert "--debug-canvas-annotations" not in argv


def test_selected_unit_keeps_full_plan_and_sparse_output_is_success(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    monkeypatch.setattr(campaign, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute", selected="unit_0")
    coordinator = campaign.read_json(campaign.coordinator_path(root))
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert len(coordinator["plan"]["units"]) == 2
    assert state["units"]["unit_0"]["status"] == "success"
    assert state["units"]["unit_1"]["status"] == "planned"
    attempt = state["units"]["unit_0"]["attempts"][0]
    receipt = campaign.read_json(campaign.completion_receipt_path(Path(attempt["private_unit_root"])))
    assert receipt["observed_png_count_by_variable"] == {"invmass": 1, "fwd0pt": 0}
    relative = receipt["observed_png_relative_paths"][0]
    assert (root / relative).read_bytes() == b"fixture-png"
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign", action="resume")
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert [len(state["units"][key]["attempts"]) for key in ("unit_0", "unit_1")] == [1, 1]


def test_receipt_interruption_materializes_without_child_rerun(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)
    plan = campaign.normalize_plan(plan_path, "sr_crlike_run2", "fixture_campaign", root)
    runtime = campaign.runtime_identity()
    root.mkdir()
    state = campaign.new_component_state(plan, "run2", runtime)
    campaign.atomic_json(campaign.state_path(root, "sr_crlike_run2", "run2"), state)
    campaign.atomic_json(campaign.coordinator_path(root), {
        "schema_version": 1, "campaign_profile": "sr_crlike_run2",
        "campaign_tag": "fixture_campaign", "runtime_contract_identity": runtime,
        "plan": plan,
    })
    unit = plan["units"][0]
    attempt_root = Path(unit["private_unit_root"])
    attempt_root.mkdir(parents=True)
    fake_success(campaign.child_argv(unit, attempt_root), attempt_root)
    attempt = {"attempt_id": "attempt_0001", "plan_revision": 1,
               "started_at_utc": "2026-10-27T00:00:00Z", "ended_at_utc": "2026-10-27T00:01:00Z",
               "private_unit_root": str(attempt_root), "guard_result": 0,
               "child_result": 0, "tee_result": 0, "materialization_status": "pending",
               "completion_receipt_identity": None, "result": "running"}
    unit_state = state["units"]["unit_0"]
    unit_state["status"] = "running"
    unit_state["attempts"].append(attempt)
    campaign.atomic_json(campaign.state_path(root, "sr_crlike_run2", "run2"), state)
    assert campaign.lifecycle_receipt_path(attempt_root).is_file()
    campaign.create_receipt(plan, unit, attempt, runtime)
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign", action="resume")
    recovered = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert recovered["units"]["unit_0"]["status"] == "success"
    assert recovered["units"]["unit_0"]["attempts"][0]["materialization_status"] == "complete"


def test_unbound_collision_fails_without_overwrite(tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    relative = Path("run2/merged/unit_0/unit_0_invmass_fitting.png")

    def collide(argv, attempt_root):
        fake_success(argv, attempt_root)
        if attempt_root.parent.name == "unit_0":
            destination = root / relative
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"foreign")
        return 0, 0, 0

    monkeypatch.setattr(campaign, "run_guarded_child", collide)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    assert (root / relative).read_bytes() == b"foreign"
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert state["units"]["unit_0"]["status"] == "failed"
    assert state["units"]["unit_1"]["status"] == "success"
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["campaign_status"] == "partial"
    assert (summary["planned_count"], summary["success_count"], summary["failed_count"]) == (2, 1, 1)


def test_failed_and_interrupted_are_not_retried(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    monkeypatch.setattr(campaign, "run_guarded_child", fake_failure)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute", selected="unit_0")
    state_path = campaign.state_path(root, "sr_crlike_run2", "run2")
    state = campaign.read_json(state_path)
    state["units"]["unit_1"]["status"] = "interrupted"
    state["units"]["unit_1"]["attempts"] = [dict(
        state["units"]["unit_0"]["attempts"][0], result="interrupted")]
    campaign.atomic_json(state_path, state)
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("silent retry"))
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign", action="resume")
    state = campaign.read_json(state_path)
    assert state["units"]["unit_0"]["status"] == "failed"
    assert state["units"]["unit_1"]["status"] == "interrupted"


def test_revision_appends_and_preserves_history(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    monkeypatch.setattr(campaign, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute", selected="unit_0")
    before = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))["units"]["unit_0"]
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    raw["plan_revision"] = 2
    original = raw["units"][1]
    third = dict(original, unit_id="unit_2", producer_unit_id="producer_2")
    raw["units"].append(third)
    revision = tmp_path / "revision.json"
    revision.write_text(json.dumps(raw), encoding="utf-8")
    campaign.activate_revision(root, "sr_crlike_run2", "fixture_campaign", revision,
                               1, campaign.runtime_identity())
    after = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert after["units"]["unit_0"] == before
    assert after["units"]["unit_2"]["status"] == "planned"
    with pytest.raises(campaign.CampaignError, match="revision mismatch"):
        campaign.activate_revision(root, "sr_crlike_run2", "fixture_campaign", revision,
                                   1, campaign.runtime_identity())
    raw["plan_revision"] = 3
    raw["units"][0]["qualification_source"]["locator"] = "rewritten"
    raw["units"].append(dict(original, unit_id="unit_3", producer_unit_id="producer_3"))
    revision.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="rewrite"):
        campaign.activate_revision(root, "sr_crlike_run2", "fixture_campaign", revision,
                                   2, campaign.runtime_identity())


def test_combined_component_isolation(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, profile="sr_crlike_run2_run3", unit_count=2)

    def selective(argv, attempt_root):
        if "run2" in attempt_root.parts:
            return fake_failure(argv, attempt_root)
        return fake_success(argv, attempt_root)

    monkeypatch.setattr(campaign, "run_guarded_child", selective)
    campaign.run_campaign(root, "sr_crlike_run2_run3", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    states = {component: campaign.read_json(campaign.state_path(root, "sr_crlike_run2_run3", component))
              for component in ("run2", "run3")}
    assert states["run2"]["units"]["unit_0"]["status"] == "failed"
    assert states["run3"]["units"]["unit_1"]["status"] == "success"


def test_guard_terminal_and_renderer_message_evidence(tmp_path, monkeypatch):
    guard = tmp_path / "guard.jsonl"
    guard.write_text('{"event":"start"}\n{"event":"result","termination_kind":"normal","returncode":0}\n', encoding="utf-8")
    assert campaign.guard_terminal(guard)["returncode"] == 0
    guard.write_text('{"event":"result","returncode":0}\n{"event":"result","returncode":0}\n', encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="ambiguous"):
        campaign.guard_terminal(guard)
    root, plan_path = fixture_plan(tmp_path / "other", monkeypatch)
    plan = campaign.normalize_plan(plan_path, "sr_crlike_run2", "fixture_campaign", root)
    unit = plan["units"][0]
    attempt_root = Path(unit["private_unit_root"])
    attempt_root.mkdir(parents=True)
    fake_success(campaign.child_argv(unit, attempt_root), attempt_root)
    (attempt_root / "_metadata/renderer.log").write_text(
        "Error: required histogram family absent from bound input\n", encoding="utf-8")
    attempt = {"private_unit_root": str(attempt_root), "guard_result": 0,
               "child_result": 0, "tee_result": 0,
               "attempt_id": "attempt_0001", "plan_revision": 1,
               "started_at_utc": "2026-10-27T00:00:00Z",
               "ended_at_utc": "2026-10-27T00:01:00Z"}
    receipt = campaign.create_receipt(plan, unit, attempt, campaign.runtime_identity())
    assert receipt["renderer_log_status"] == "present"
    assert receipt["renderer_message_count"] == 1
    assert "Error: required histogram" in receipt["renderer_messages"][0]


def test_late_renderer_message_is_counted_without_gate(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)
    plan = campaign.normalize_plan(plan_path, "sr_crlike_run2", "fixture_campaign", root)
    unit = plan["units"][0]
    attempt_root = Path(unit["private_unit_root"])
    attempt_root.mkdir(parents=True)
    fake_success(campaign.child_argv(unit, attempt_root), attempt_root)
    log_path = attempt_root / "_metadata/renderer.log"
    log_path.write_text("warning: inert\n" * 201
                        + "Error: required histogram family absent from bound input\n",
                        encoding="utf-8")
    attempt = {"private_unit_root": str(attempt_root), "guard_result": 0,
               "child_result": 0, "tee_result": 0,
               "attempt_id": "attempt_0001", "plan_revision": 1,
               "started_at_utc": "2026-10-27T00:00:00Z",
               "ended_at_utc": "2026-10-27T00:01:00Z"}
    receipt = campaign.create_receipt(plan, unit, attempt, campaign.runtime_identity())
    assert len(receipt["renderer_messages"]) == 200
    assert receipt["renderer_log_line_count"] == 202
    assert receipt["renderer_message_count"] == 202
    assert receipt["renderer_log_status"] == "present"


def test_combined_unresolved_running_blocks_all_children(tmp_path, monkeypatch, capsys):
    profile = "sr_crlike_run2_run3"
    root, _, attempt_root = seed_running(tmp_path, monkeypatch, profile=profile)
    campaign.lifecycle_receipt_path(attempt_root).unlink()
    (attempt_root / "_metadata/guard.jsonl").unlink()
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("overlap"))
    assert campaign.main(resume_args(root, profile, selected="unit_1")) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["campaign_status"] == "waiting"
    assert summary["waiting_on_active_child"] is True
    assert summary["running_count"] == 1 and summary["pending_count"] == 1
    states = {component: campaign.read_json(campaign.state_path(root, profile, component))
              for component in ("run2", "run3")}
    assert states["run2"]["units"]["unit_0"]["status"] == "running"
    assert states["run3"]["units"]["unit_1"]["status"] == "planned"


def test_lifecycle_crash_window_recovers_without_rerun(tmp_path, monkeypatch):
    root, _, attempt_root = seed_running(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    assert campaign.main(resume_args(root)) == 0
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert state["units"]["unit_0"]["status"] == "success"
    assert campaign.completion_receipt_path(attempt_root).is_file()


def test_failed_lifecycle_and_missing_lifecycle_are_terminal_without_rerun(tmp_path, monkeypatch):
    root, _, _ = seed_running(tmp_path / "failed", monkeypatch, terminal="failure")
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    assert campaign.main(resume_args(root)) == 0
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert state["units"]["unit_0"]["status"] == "failed"
    root2, _, attempt_root2 = seed_running(tmp_path / "interrupted", monkeypatch)
    campaign.lifecycle_receipt_path(attempt_root2).unlink()
    assert campaign.main(resume_args(root2)) == 0
    state2 = campaign.read_json(campaign.state_path(root2, "sr_crlike_run2", "run2"))
    assert state2["units"]["unit_0"]["status"] == "interrupted"
    assert not campaign.completion_receipt_path(attempt_root2).exists()


def test_full_and_selected_resume_status_with_terminal_failure(tmp_path, monkeypatch, capsys):
    profile = "sr_crlike_run2_run3"
    root, _, _ = seed_running(tmp_path, monkeypatch, profile=profile, terminal="failure")
    monkeypatch.setattr(campaign, "run_guarded_child", fake_success)
    assert campaign.main(resume_args(root, profile)) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["campaign_status"] == "partial"
    assert (summary["planned_count"], summary["success_count"], summary["failed_count"]) == (2, 1, 1)
    run2 = campaign.read_json(campaign.state_path(root, profile, "run2"))
    run3 = campaign.read_json(campaign.state_path(root, profile, "run3"))
    assert run2["units"]["unit_0"]["status"] == "failed"
    assert run3["units"]["unit_1"]["status"] == "success"
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("silent retry"))
    assert campaign.main(resume_args(root, profile, selected="unit_1")) == 0
    assert campaign.main(resume_args(root, profile, selected="unit_0")) == 0
    assert campaign.main(resume_args(root, profile)) == 0


def test_malformed_component_state_remains_invocation_error(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    path = campaign.state_path(root, "sr_crlike_run2", "run2")
    state = campaign.read_json(path)
    state["units"]["unit_0"]["status"] = "uninterpretable"
    campaign.atomic_json(path, state)
    assert campaign.main(resume_args(root)) == 2


def test_private_root_setup_failure_is_terminal_and_other_unit_continues(tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    original_mkdir = Path.mkdir

    def fail_first_attempt_root(path, *args, **kwargs):
        if path.name == "attempt_0001" and path.parent.name == "unit_0":
            raise OSError("inert prelaunch setup failure")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_first_attempt_root)
    launched = []

    def record_child(argv, attempt_root):
        launched.append(attempt_root.parent.name)
        return fake_success(argv, attempt_root)

    monkeypatch.setattr(campaign, "run_guarded_child", record_child)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert summary["campaign_status"] == "partial"
    assert summary["waiting_on_active_child"] is False
    assert launched == ["unit_1"]
    assert state["units"]["unit_0"]["status"] == "failed"
    assert state["units"]["unit_0"]["attempts"][0]["result"] == "failed"
    assert state["units"]["unit_1"]["status"] == "success"


def test_first_popen_failure_is_terminal_without_child(tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)
    calls = []

    def no_process(*args, **kwargs):
        calls.append(args[0])
        raise OSError("inert first process creation failure")

    monkeypatch.setattr(campaign.subprocess, "Popen", no_process)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert len(calls) == 1
    assert state["units"]["unit_0"]["status"] == "failed"
    assert summary["waiting_on_active_child"] is False


def test_ambiguous_launch_stays_running_and_blocks_other_child(tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    calls = []

    def ambiguous(argv, attempt_root):
        calls.append(attempt_root.parent.name)
        raise OSError("possibly started")

    monkeypatch.setattr(campaign, "run_guarded_child", ambiguous)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert calls == ["unit_0"]
    assert summary["campaign_status"] == "waiting"
    assert summary["waiting_on_active_child"] is True
    assert state["units"]["unit_0"]["status"] == "running"
    assert state["units"]["unit_1"]["status"] == "planned"


@pytest.mark.parametrize("status", ["planned", "running", "failed", "interrupted", "success"])
def test_valid_status_attempt_history(status, tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)
    plan = campaign.normalize_plan(plan_path, "sr_crlike_run2", "fixture_campaign", root)
    runtime = campaign.runtime_identity()
    state = campaign.new_component_state(plan, "run2", runtime)
    unit_state = state["units"]["unit_0"]
    unit_state["status"] = status
    if status != "planned":
        unit_state["attempts"] = [{
            "result": status,
            "completion_receipt_identity": "a" * 64 if status == "success" else None,
            "materialization_status": "complete" if status == "success" else "pending",
        }]
    campaign.validate_state(state, plan, "run2", runtime)


@pytest.mark.parametrize("corruption", ["planned_with_attempt", "success_without_attempt",
                                        "success_without_receipt", "success_incomplete_materialization"])
def test_corrupt_status_attempt_history_rejected_before_child(corruption, tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    monkeypatch.setattr(campaign, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute", selected="unit_0")
    state_path = campaign.state_path(root, "sr_crlike_run2", "run2")
    state = campaign.read_json(state_path)
    if corruption == "planned_with_attempt":
        unit_state = state["units"]["unit_1"]
        unit_state["attempts"] = [dict(state["units"]["unit_0"]["attempts"][0])]
    else:
        unit_state = state["units"]["unit_0"]
        if corruption == "success_without_attempt":
            unit_state["attempts"] = []
        elif corruption == "success_without_receipt":
            unit_state["attempts"][0]["completion_receipt_identity"] = None
        else:
            unit_state["attempts"][0]["materialization_status"] = "pending"
    campaign.atomic_json(state_path, state)
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("child must not start"))
    assert campaign.main(resume_args(root)) == 2
    after = campaign.read_json(state_path)
    assert after == state
    assert not (root / ".campaign_units/unit_1/attempt_0002").exists()


def test_git_and_source_drift_do_not_gate_resume(tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    source = tmp_path / "runtime_source.py"
    source.write_text("initial", encoding="utf-8")
    historical = {"git_head": "a" * 40, "source_manifest": {str(source): "b" * 64}}
    monkeypatch.setattr(campaign, "runtime_identity", lambda: historical)
    monkeypatch.setattr(campaign, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute", selected="unit_0")
    source.write_text("changed", encoding="utf-8")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "other_git"))
    monkeypatch.setenv("GIT_BRANCH", "other_branch")
    monkeypatch.setenv("GIT_HEAD", "f" * 40)
    monkeypatch.setattr(campaign, "runtime_identity", lambda: pytest.fail("runtime provenance queried"))
    assert campaign.main(resume_args(root)) == 0
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert state["units"]["unit_1"]["status"] == "success"
    assert state["runtime_contract_identity"] == historical
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["campaign_status"] == "complete"


def test_missing_renderer_log_is_receipt_evidence(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)

    def success_without_log(argv, attempt_root):
        result = fake_success(argv, attempt_root)
        (attempt_root / "_metadata/renderer.log").unlink()
        return result

    monkeypatch.setattr(campaign, "run_guarded_child", success_without_log)
    assert campaign.main(["--campaign-profile", "sr_crlike_run2", "--campaign-tag",
                          "fixture_campaign", "-o", str(root), "--plan", str(plan_path),
                          "--execute"]) == 0
    receipt = campaign.read_json(campaign.completion_receipt_path(
        root / ".campaign_units/unit_0/attempt_0001"))
    assert receipt["renderer_log_status"] == "missing"
    assert receipt["renderer_log_line_count"] == 0
    assert receipt["renderer_messages"] == []


def test_inert_guard_and_visible_tee_lifecycle(tmp_path):
    attempt_root = tmp_path / "attempt_0001"
    attempt_root.mkdir()
    assert campaign.run_guarded_child(["/bin/echo", "inert-renderer-fixture"], attempt_root) == (0, 0, 0)
    assert "inert-renderer-fixture" in (attempt_root / "_metadata/renderer.log").read_text(encoding="utf-8")
    assert campaign.guard_terminal(attempt_root / "_metadata/guard.jsonl")["returncode"] == 0


def test_qualification_identity_mismatch_fails_before_root(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    raw["units"][0]["qualification_source"]["sidecar_sha256"] = "b" * 64
    plan_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="sidecar contradicts"):
        campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                              plan_path=plan_path, action="execute")
    assert not root.exists()


def test_bound_destination_recovery_and_different_bytes(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    state_path = campaign.state_path(root, "sr_crlike_run2", "run2")
    state = campaign.read_json(state_path)
    unit_state = state["units"]["unit_0"]
    receipt = campaign.read_json(campaign.completion_receipt_path(
        Path(unit_state["attempts"][0]["private_unit_root"])))
    destination = root / receipt["observed_png_relative_paths"][0]
    unit_state["status"] = "running"
    unit_state["attempts"][0]["materialization_status"] = "pending"
    unit_state["attempts"][0]["result"] = "running"
    campaign.atomic_json(state_path, state)
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign", action="resume")
    assert destination.read_bytes() == b"fixture-png"
    state = campaign.read_json(state_path)
    state["units"]["unit_0"]["status"] = "running"
    state["units"]["unit_0"]["attempts"][0]["result"] = "running"
    campaign.atomic_json(state_path, state)
    destination.write_bytes(b"changed")
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign", action="resume")
    assert destination.read_bytes() == b"changed"
    assert campaign.read_json(state_path)["units"]["unit_0"]["status"] == "failed"


def test_pending_combined_revision_reconciles_after_interruption(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch,
                                   profile="sr_crlike_run2_run3", unit_count=2)
    monkeypatch.setattr(campaign, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2_run3", "fixture_campaign",
                          plan_path=plan_path, action="execute", selected="unit_0")
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    raw["plan_revision"] = 2
    raw["units"].append(dict(raw["units"][1], unit_id="unit_2",
                             producer_unit_id="producer_2"))
    revision = tmp_path / "revision.json"
    revision.write_text(json.dumps(raw), encoding="utf-8")
    original_atomic = campaign.atomic_json
    failed = {"value": False}

    def interrupt_once(path, value):
        if (not failed["value"] and Path(path) == campaign.state_path(
                root, "sr_crlike_run2_run3", "run3")
                and value.get("plan_revision") == 2):
            failed["value"] = True
            raise OSError("inert interruption")
        return original_atomic(path, value)

    monkeypatch.setattr(campaign, "atomic_json", interrupt_once)
    with pytest.raises(OSError, match="interruption"):
        campaign.activate_revision(root, "sr_crlike_run2_run3", "fixture_campaign",
                                   revision, 1, campaign.runtime_identity())
    monkeypatch.setattr(campaign, "atomic_json", original_atomic)
    assert "pending_plan" in campaign.read_json(campaign.coordinator_path(root))
    campaign.run_campaign(root, "sr_crlike_run2_run3", "fixture_campaign",
                          action="resume", selected="unit_2")
    assert "pending_plan" not in campaign.read_json(campaign.coordinator_path(root))
    states = {component: campaign.read_json(campaign.state_path(root, "sr_crlike_run2_run3", component))
              for component in ("run2", "run3")}
    assert all(state["plan_revision"] == 2 for state in states.values())
    assert states["run2"]["units"]["unit_0"]["status"] == "success"
    assert states["run3"]["units"]["unit_2"]["status"] == "success"
