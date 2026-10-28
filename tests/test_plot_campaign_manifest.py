"""Inert transaction fixtures for the persistent SR plotting controller."""

import importlib.util
import io
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "analysis/topeft_run2/plot_campaign_manifest.py"
SPEC = importlib.util.spec_from_file_location("plot_campaign_manifest", MODULE_PATH)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def guard_result_line(returncode=0):
    return json.dumps({
        "event": "result", "termination_kind": "normal", "returncode": returncode,
        "signal": None, "peak_pressure_used_bytes": None,
        "minimum_mem_available_kb": None, "pid": 1, "pgid": 1, "start_ticks": 1,
    }) + "\n"


def fixture_plan(tmp_path, monkeypatch, *, profile="sr_crlike_run2", unit_count=1):
    repo = tmp_path / "repo"
    (repo / "histos").mkdir(parents=True)
    monkeypatch.setattr(campaign, "REPO_ROOT", repo)
    root = repo / "histos" / "fixture_campaign"
    units = []
    for index in range(unit_count):
        component = "run3" if profile == "sr_crlike_run2_run3" and index == 1 else "run2"
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
    (metadata / "guard.jsonl").write_text(guard_result_line(), encoding="utf-8")
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
    (metadata / "guard.jsonl").write_text(guard_result_line(7), encoding="utf-8")
    write_lifecycle(argv, attempt_root, 7, 7, 0)
    return 7, 7, 0


def seed_running(tmp_path, monkeypatch, *, profile="sr_crlike_run2", terminal="success",
                 unit_count=None):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, profile=profile,
                                   unit_count=(unit_count if unit_count is not None else
                                               (2 if profile.endswith("run2_run3") else 1)))
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


def seed_missing_lifecycle(tmp_path, monkeypatch, profile="sr_crlike_run2_run3"):
    root, plan, attempt_root = seed_running(tmp_path, monkeypatch, profile=profile,
                                            unit_count=3)
    state_path = campaign.state_path(root, profile, "run2")
    state = campaign.read_json(state_path)
    unit_state = state["units"]["unit_0"]
    unit_state["status"] = unit_state["attempts"][0]["result"] = "interrupted"
    unit_state["attempts"][0]["failure_classification"] = (
        "guard terminal but complete lifecycle unavailable")
    campaign.atomic_json(state_path, state)
    campaign.lifecycle_receipt_path(attempt_root).unlink()
    (attempt_root / "_metadata/guard.jsonl").write_text(
        json.dumps({"event": "start", "identity": {"pid": 99999999,
                    "pgid": 99999999, "start_ticks": 1}}) + "\n" +
        guard_result_line().replace('"pid": 1', '"pid": 99999999').replace(
            '"pgid": 1', '"pgid": 99999999'), encoding="utf-8")
    (attempt_root / "_metadata/renderer.log").write_text(
        f"[SR] Produced 1 plots with stat⊕syst uncertainties and 0 plots with "
        f"stat-only bands after completing 1 rendering tasks in "
        f"{attempt_root / 'run2' / 'merged'}\n", encoding="utf-8")
    return root, plan, attempt_root


def recovery_args(root, profile="sr_crlike_run2_run3"):
    return ["--campaign-profile", profile, "--campaign-tag", "fixture_campaign",
            "-o", str(root), "--recover-terminal-unit", "--selected-unit", "unit_0"]


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
    guard.write_text('{"event":"start"}\n' + guard_result_line(), encoding="utf-8")
    assert campaign.guard_terminal(guard)["returncode"] == 0
    guard.write_text('{"event":"result"}\n', encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="malformed"):
        campaign.guard_terminal(guard)
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


def test_post_tuple_terminal_state_recovers_after_interruption_without_child_rerun(
        tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)
    original_verify = campaign.verify_lifecycle_receipt
    monkeypatch.setattr(campaign, "run_guarded_child", fake_success)

    def interrupt_after_terminal_persistence(*_args):
        state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
        attempt = state["units"]["unit_0"]["attempts"][0]
        assert state["units"]["unit_0"]["status"] == attempt["result"] == "interrupted"
        assert (attempt["guard_result"], attempt["child_result"], attempt["tee_result"]) == (0, 0, 0)
        raise KeyboardInterrupt("inert interruption after terminal tuple")

    monkeypatch.setattr(campaign, "verify_lifecycle_receipt", interrupt_after_terminal_persistence)
    with pytest.raises(KeyboardInterrupt, match="inert interruption"):
        campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                              plan_path=plan_path, action="execute")
    monkeypatch.setattr(campaign, "verify_lifecycle_receipt", original_verify)
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    assert campaign.main(resume_args(root)) == 0
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert state["units"]["unit_0"]["status"] == "success"
    assert state["units"]["unit_0"]["attempts"][0]["failure_classification"] is None


def test_post_guard_terminal_state_recovers_after_interruption_without_child_rerun(
        tmp_path, monkeypatch):
    root, _, _ = seed_running(tmp_path, monkeypatch)
    original_path = campaign.lifecycle_receipt_path

    def interrupt_after_terminal_persistence(attempt_root):
        state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
        attempt = state["units"]["unit_0"]["attempts"][0]
        assert state["units"]["unit_0"]["status"] == attempt["result"] == "interrupted"
        raise KeyboardInterrupt("inert interruption after guard evidence")

    monkeypatch.setattr(campaign, "lifecycle_receipt_path", interrupt_after_terminal_persistence)
    with pytest.raises(KeyboardInterrupt, match="inert interruption"):
        campaign.main(resume_args(root))
    monkeypatch.setattr(campaign, "lifecycle_receipt_path", original_path)
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    assert campaign.main(resume_args(root)) == 0
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert state["units"]["unit_0"]["status"] == "success"
    assert state["units"]["unit_0"]["attempts"][0]["failure_classification"] is None


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


def test_second_popen_failure_with_terminal_guard_allows_other_unit(tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    original_run = campaign.run_guarded_child
    launches = []

    class TerminalGuard:
        def __init__(self, guard_path):
            self.guard_path = guard_path
            self.stdout = io.BytesIO()
            self.guard_path.write_text(guard_result_line(), encoding="utf-8")

        def wait(self):
            pytest.fail("tee creation failure must not wait for an undrained guard")

    def run_child(argv, attempt_root):
        launches.append(attempt_root.parent.name)
        if attempt_root.parent.name != "unit_0":
            return fake_success(argv, attempt_root)
        calls = []

        def popen(*args, **kwargs):
            calls.append(args[0])
            if len(calls) == 1:
                return TerminalGuard(attempt_root / "_metadata" / "guard.jsonl")
            raise OSError("inert tee creation failure")

        with monkeypatch.context() as process_patch:
            process_patch.setattr(campaign.subprocess, "Popen", popen)
            result = original_run(argv, attempt_root)
        return result

    monkeypatch.setattr(campaign, "run_guarded_child", run_child)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert launches == ["unit_0", "unit_1"]
    assert summary["campaign_status"] == "partial"
    assert summary["waiting_on_active_child"] is False
    assert summary["running_count"] == 0
    assert state["units"]["unit_0"]["status"] == "failed"
    assert state["units"]["unit_0"]["attempts"][0]["result"] == "failed"
    assert state["units"]["unit_1"]["status"] == "success"


@pytest.mark.parametrize("failure", ["stdout_close", "tee_wait", "guard_wait"])
def test_terminal_guard_survives_supervisor_cleanup_failure(
        failure, tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    original_run = campaign.run_guarded_child
    launches = []

    class GuardStdout(io.BytesIO):
        def close(self):
            if failure == "stdout_close":
                raise OSError("inert stdout close failure")
            super().close()

    class TerminalGuard:
        def __init__(self, guard_path):
            self.stdout = GuardStdout()
            guard_path.write_text(guard_result_line(), encoding="utf-8")

        def wait(self):
            if failure == "guard_wait":
                raise OSError("inert guard wait failure")
            return 0

    class TerminalTee:
        def wait(self):
            if failure == "tee_wait":
                raise OSError("inert tee wait failure")
            return 0

    def run_child(argv, attempt_root):
        launches.append(attempt_root.parent.name)
        if attempt_root.parent.name != "unit_0":
            return fake_success(argv, attempt_root)
        calls = []

        def popen(*_args, **_kwargs):
            calls.append(1)
            return (TerminalGuard(attempt_root / "_metadata/guard.jsonl")
                    if len(calls) == 1 else TerminalTee())

        with monkeypatch.context() as process_patch:
            process_patch.setattr(campaign.subprocess, "Popen", popen)
            return original_run(argv, attempt_root)

    monkeypatch.setattr(campaign, "run_guarded_child", run_child)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert launches == ["unit_0", "unit_1"]
    assert summary["waiting_on_active_child"] is False
    assert summary["running_count"] == 0
    assert state["units"]["unit_0"]["status"] == "failed"
    assert state["units"]["unit_1"]["status"] == "success"


@pytest.mark.parametrize("guard_record", [None, '{"event":"result"}\n'])
def test_tee_creation_without_terminal_guard_does_not_wait_or_launch_next(
        guard_record, tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    original_run = campaign.run_guarded_child
    launches = []

    class UnresolvedGuard:
        def __init__(self, guard_path):
            self.stdout = io.BytesIO()
            if guard_record is not None:
                guard_path.write_text(guard_record, encoding="utf-8")

        def wait(self):
            pytest.fail("undrained guard wait would block")

    def run_child(argv, attempt_root):
        launches.append(attempt_root.parent.name)
        calls = []

        def popen(*_args, **_kwargs):
            calls.append(1)
            if len(calls) == 1:
                return UnresolvedGuard(attempt_root / "_metadata/guard.jsonl")
            raise OSError("inert tee creation failure")

        with monkeypatch.context() as process_patch:
            process_patch.setattr(campaign.subprocess, "Popen", popen)
            return original_run(argv, attempt_root)

    monkeypatch.setattr(campaign, "run_guarded_child", run_child)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert launches == ["unit_0"]
    assert summary["campaign_status"] == "waiting"
    assert summary["running_count"] == 1
    assert state["units"]["unit_0"]["status"] == "running"
    assert state["units"]["unit_1"]["status"] == "planned"


def test_supervisor_lifecycle_write_failure_after_terminal_guard_is_failed(
        tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    original_run = campaign.run_guarded_child
    original_atomic = campaign.atomic_json
    launches = []

    class TerminalGuard:
        def __init__(self, guard_path):
            self.guard_path = guard_path
            self.stdout = io.BytesIO()

        def wait(self):
            self.guard_path.write_text(guard_result_line(), encoding="utf-8")
            return 0

    class TerminalTee:
        def wait(self):
            return 0

    def run_child(argv, attempt_root):
        launches.append(attempt_root.parent.name)
        if attempt_root.parent.name != "unit_0":
            return fake_success(argv, attempt_root)
        calls = []

        def popen(*args, **kwargs):
            calls.append(args[0])
            if len(calls) == 1:
                return TerminalGuard(attempt_root / "_metadata" / "guard.jsonl")
            return TerminalTee()

        def failed_lifecycle_write(path, value):
            if path == campaign.lifecycle_receipt_path(attempt_root):
                raise TypeError("inert lifecycle receipt write failure")
            return original_atomic(path, value)

        with monkeypatch.context() as supervisor_patch:
            supervisor_patch.setattr(campaign.subprocess, "Popen", popen)
            supervisor_patch.setattr(campaign, "atomic_json", failed_lifecycle_write)
            return original_run(argv, attempt_root)

    monkeypatch.setattr(campaign, "run_guarded_child", run_child)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert launches == ["unit_0", "unit_1"]
    assert summary["campaign_status"] == "partial"
    assert summary["waiting_on_active_child"] is False
    assert state["units"]["unit_0"]["status"] == "failed"
    assert "inert lifecycle receipt write failure" in (
        state["units"]["unit_0"]["attempts"][0]["failure_classification"])
    assert state["units"]["unit_1"]["status"] == "success"


@pytest.mark.parametrize("source", ["supervisor_postproof", "terminal_exception"])
def test_unprintable_terminal_error_cannot_leave_running(
        source, tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    original_run = campaign.run_guarded_child
    original_atomic = campaign.atomic_json
    launches = []

    class UnprintableError(OSError):
        def __str__(self):
            raise ValueError("inert error formatting failure")

    class UnprintableTerminal(campaign.ChildDefinitelyTerminal):
        def __str__(self):
            raise ValueError("inert error formatting failure")

    class TerminalGuard:
        stdout = io.BytesIO()

        def __init__(self, guard_path):
            guard_path.write_text(guard_result_line(), encoding="utf-8")

        def wait(self):
            return 0

    class TerminalTee:
        def wait(self):
            return 0

    def run_child(argv, attempt_root):
        launches.append(attempt_root.parent.name)
        if attempt_root.parent.name != "unit_0":
            return fake_success(argv, attempt_root)
        if source == "terminal_exception":
            raise UnprintableTerminal()
        calls = []

        def popen(*_args, **_kwargs):
            calls.append(1)
            return (TerminalGuard(attempt_root / "_metadata/guard.jsonl")
                    if len(calls) == 1 else TerminalTee())

        def failed_lifecycle_write(path, value):
            if path == campaign.lifecycle_receipt_path(attempt_root):
                raise UnprintableError()
            return original_atomic(path, value)

        with monkeypatch.context() as process_patch:
            process_patch.setattr(campaign.subprocess, "Popen", popen)
            process_patch.setattr(campaign, "atomic_json", failed_lifecycle_write)
            return original_run(argv, attempt_root)

    monkeypatch.setattr(campaign, "run_guarded_child", run_child)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert launches == ["unit_0", "unit_1"]
    assert summary["running_count"] == 0
    assert state["units"]["unit_0"]["status"] == "failed"
    assert state["units"]["unit_0"]["attempts"][0]["failure_classification"]
    assert state["units"]["unit_1"]["status"] == "success"


@pytest.mark.parametrize("corruption, expected_error", [
    ("binding", "lifecycle receipt binding"),
    ("wrong_type", "lifecycle receipt is not a JSON object"),
    ("postprocess_type_error", "inert postprocess type error"),
])
def test_terminal_tuple_survives_lifecycle_verification_failure(
        corruption, expected_error, tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    launches = []
    observed_before_verify = []
    original_verify = campaign.verify_lifecycle_receipt

    def invalid_lifecycle(argv, attempt_root):
        launches.append(attempt_root.parent.name)
        result = fake_success(argv, attempt_root)
        if attempt_root.parent.name == "unit_0":
            path = campaign.lifecycle_receipt_path(attempt_root)
            if corruption == "wrong_type":
                campaign.atomic_json(path, [])
            else:
                lifecycle = campaign.read_json(path)
                lifecycle["command_identity"] = "0" * 64
                campaign.atomic_json(path, lifecycle)
        return result

    def verify_after_persistence(receipt, attempt, argv):
        if Path(attempt["private_unit_root"]).parent.name == "unit_0":
            state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
            persisted = state["units"]["unit_0"]["attempts"][0]
            observed_before_verify.append((state["units"]["unit_0"]["status"],
                                           persisted["result"], persisted["guard_result"],
                                           persisted["child_result"], persisted["tee_result"]))
            if corruption == "postprocess_type_error":
                raise TypeError("inert postprocess type error")
        return original_verify(receipt, attempt, argv)

    monkeypatch.setattr(campaign, "run_guarded_child", invalid_lifecycle)
    monkeypatch.setattr(campaign, "verify_lifecycle_receipt", verify_after_persistence)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    attempt = state["units"]["unit_0"]["attempts"][0]
    assert launches == ["unit_0", "unit_1"]
    assert observed_before_verify == [("interrupted", "interrupted", 0, 0, 0)]
    assert summary["campaign_status"] == "partial"
    assert summary["waiting_on_active_child"] is False
    assert summary["running_count"] == 0
    assert state["units"]["unit_0"]["status"] == attempt["result"] == "failed"
    assert (attempt["guard_result"], attempt["child_result"], attempt["tee_result"]) == (0, 0, 0)
    assert expected_error in attempt["failure_classification"]
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    assert campaign.main(resume_args(root)) == 0
    assert len(campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
               ["units"]["unit_0"]["attempts"]) == 1


def test_terminal_tuple_survives_clock_failure(tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)
    original_now = campaign.utc_now
    calls = []

    def fail_late_clock():
        calls.append(1)
        if len(calls) == 2:
            raise OSError("inert timestamp failure")
        return original_now()

    monkeypatch.setattr(campaign, "utc_now", fail_late_clock)
    monkeypatch.setattr(campaign, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    attempt = state["units"]["unit_0"]["attempts"][0]
    assert len(calls) >= 2
    assert summary["running_count"] == 0
    assert state["units"]["unit_0"]["status"] == attempt["result"] == "success"
    assert attempt["failure_classification"] is None
    assert (attempt["guard_result"], attempt["child_result"], attempt["tee_result"]) == (0, 0, 0)


def test_terminal_supervisor_error_survives_exception_clock_failure(
        tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    original_now = campaign.utc_now
    calls = []
    launches = []

    def fail_late_clock():
        calls.append(1)
        if len(calls) == 2:
            raise OSError("inert timestamp failure")
        return original_now()

    def terminal_error(argv, attempt_root):
        launches.append(attempt_root.parent.name)
        if attempt_root.parent.name == "unit_0":
            raise campaign.ChildDefinitelyTerminal("inert terminal supervisor failure")
        return fake_success(argv, attempt_root)

    monkeypatch.setattr(campaign, "utc_now", fail_late_clock)
    monkeypatch.setattr(campaign, "run_guarded_child", terminal_error)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert launches == ["unit_0", "unit_1"]
    assert summary["running_count"] == 0
    assert state["units"]["unit_0"]["status"] == "failed"
    assert state["units"]["unit_1"]["status"] == "success"


def test_resume_terminal_guard_with_bad_lifecycle_is_interrupted(tmp_path, monkeypatch, capsys):
    root, _, attempt_root = seed_running(tmp_path, monkeypatch)
    campaign.atomic_json(campaign.lifecycle_receipt_path(attempt_root), [])
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    assert campaign.main(resume_args(root)) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    attempt = state["units"]["unit_0"]["attempts"][0]
    assert summary["campaign_status"] == "partial"
    assert summary["waiting_on_active_child"] is False
    assert summary["running_count"] == 0
    assert state["units"]["unit_0"]["status"] == attempt["result"] == "interrupted"
    assert "lifecycle receipt is not a JSON object" in attempt["failure_classification"]


def test_resume_terminal_guard_with_receipt_creation_failure_is_interrupted(
        tmp_path, monkeypatch, capsys):
    root, _, _ = seed_running(tmp_path, monkeypatch)

    def failed_receipt(*_):
        raise OSError("inert receipt creation failure")

    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    monkeypatch.setattr(campaign, "create_receipt", failed_receipt)
    assert campaign.main(resume_args(root)) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    attempt = state["units"]["unit_0"]["attempts"][0]
    assert summary["campaign_status"] == "partial"
    assert summary["waiting_on_active_child"] is False
    assert summary["running_count"] == 0
    assert state["units"]["unit_0"]["status"] == attempt["result"] == "interrupted"
    assert "inert receipt creation failure" in summary["unit_errors"][0]


@pytest.mark.parametrize("guard_record", ['[]\n', '{"event":"result"}\n'])
def test_resume_malformed_guard_evidence_waits_without_launch(
        guard_record, tmp_path, monkeypatch, capsys):
    profile = "sr_crlike_run2_run3"
    root, _, attempt_root = seed_running(tmp_path, monkeypatch, profile=profile)
    campaign.lifecycle_receipt_path(attempt_root).unlink()
    (attempt_root / "_metadata" / "guard.jsonl").write_text(guard_record, encoding="utf-8")
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("overlap"))
    assert campaign.main(resume_args(root, profile, selected="unit_1")) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, profile, "run2"))
    other_state = campaign.read_json(campaign.state_path(root, profile, "run3"))
    assert summary["campaign_status"] == "waiting"
    assert summary["waiting_on_active_child"] is True
    assert summary["running_count"] == 1
    assert state["units"]["unit_0"]["status"] == "running"
    assert other_state["units"]["unit_1"]["status"] == "planned"


@pytest.mark.parametrize("corruption", ["root", "command", "symlink", "metadata_symlink"])
def test_resume_unbound_guard_evidence_waits_without_launch(
        corruption, tmp_path, monkeypatch, capsys):
    profile = "sr_crlike_run2_run3"
    root, _, _ = seed_running(tmp_path, monkeypatch, profile=profile)
    path = campaign.state_path(root, profile, "run2")
    state = campaign.read_json(path)
    attempt = state["units"]["unit_0"]["attempts"][0]
    if corruption in {"root", "symlink", "metadata_symlink"}:
        foreign_root = tmp_path / "unbound_terminal_guard"
        (foreign_root / "_metadata").mkdir(parents=True)
        (foreign_root / "_metadata" / "guard.jsonl").write_text(
            guard_result_line(), encoding="utf-8")
        if corruption == "root":
            attempt["private_unit_root"] = str(foreign_root)
        elif corruption == "symlink":
            bound_root = Path(attempt["private_unit_root"])
            bound_root.rename(tmp_path / "detached_attempt")
            bound_root.symlink_to(foreign_root, target_is_directory=True)
        else:
            bound_metadata = Path(attempt["private_unit_root"]) / "_metadata"
            bound_metadata.rename(tmp_path / "detached_metadata")
            bound_metadata.symlink_to(foreign_root / "_metadata", target_is_directory=True)
    else:
        attempt["command_identity"] = "0" * 64
    campaign.atomic_json(path, state)
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("overlap"))
    assert campaign.main(resume_args(root, profile, selected="unit_1")) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    run2 = campaign.read_json(path)
    run3 = campaign.read_json(campaign.state_path(root, profile, "run3"))
    assert summary["campaign_status"] == "waiting"
    assert summary["waiting_on_active_child"] is True
    assert summary["running_count"] == 1
    assert run2["units"]["unit_0"]["status"] == "running"
    assert run3["units"]["unit_1"]["status"] == "planned"


@pytest.mark.parametrize("field", ["guard_result", "ended_at_utc", "completion_receipt_identity"])
def test_resume_malformed_attempt_field_after_terminal_guard_is_interrupted(
        field, tmp_path, monkeypatch, capsys):
    root, _, _ = seed_running(tmp_path, monkeypatch)
    path = campaign.state_path(root, "sr_crlike_run2", "run2")
    state = campaign.read_json(path)
    state["units"]["unit_0"]["attempts"][0][field] = []
    campaign.atomic_json(path, state)
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    assert campaign.main(resume_args(root)) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(path)
    attempt = state["units"]["unit_0"]["attempts"][0]
    assert summary["campaign_status"] == "partial"
    assert summary["waiting_on_active_child"] is False
    assert state["units"]["unit_0"]["status"] == attempt["result"] == "interrupted"
    assert attempt["failure_classification"]


def test_resume_lifecycle_path_inspection_failure_after_terminal_guard_is_interrupted(
        tmp_path, monkeypatch, capsys):
    root, _, attempt_root = seed_running(tmp_path, monkeypatch)
    lifecycle_path = campaign.lifecycle_receipt_path(attempt_root)
    original_exists = Path.exists

    def failed_exists(path):
        if path == lifecycle_path:
            raise OSError("inert lifecycle path inspection failure")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", failed_exists)
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    assert campaign.main(resume_args(root)) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    attempt = state["units"]["unit_0"]["attempts"][0]
    assert summary["campaign_status"] == "partial"
    assert summary["waiting_on_active_child"] is False
    assert state["units"]["unit_0"]["status"] == attempt["result"] == "interrupted"
    assert "inert lifecycle path inspection failure" in attempt["failure_classification"]


def test_resume_terminal_guard_survives_clock_failure(tmp_path, monkeypatch, capsys):
    root, _, attempt_root = seed_running(tmp_path, monkeypatch)
    campaign.lifecycle_receipt_path(attempt_root).unlink()

    def failed_clock():
        raise OSError("inert timestamp failure")

    monkeypatch.setattr(campaign, "utc_now", failed_clock)
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    assert campaign.main(resume_args(root)) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    attempt = state["units"]["unit_0"]["attempts"][0]
    assert summary["campaign_status"] == "partial"
    assert summary["running_count"] == 0
    assert state["units"]["unit_0"]["status"] == attempt["result"] == "interrupted"


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


def test_missing_lifecycle_recovery_is_no_launch_and_truthful(tmp_path, monkeypatch):
    root, _, attempt_root = seed_missing_lifecycle(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("child launch"))
    monkeypatch.setattr(campaign, "normalize_unit", lambda *_: pytest.fail("input read"))
    assert campaign.main(recovery_args(root)) == 0
    run2 = campaign.read_json(campaign.state_path(root, "sr_crlike_run2_run3", "run2"))
    run3 = campaign.read_json(campaign.state_path(root, "sr_crlike_run2_run3", "run3"))
    unit = run2["units"]["unit_0"]
    attempt = unit["attempts"][0]
    assert unit["status"] == "success" and len(unit["attempts"]) == 1
    assert all(attempt[key] is None for key in ("guard_result", "child_result", "tee_result"))
    assert not campaign.lifecycle_receipt_path(attempt_root).exists()
    receipt = campaign.read_json(campaign.completion_receipt_path(attempt_root))
    provenance = campaign.read_json(campaign.recovery_provenance_path(attempt_root))
    assert receipt["recovery_provenance_identity"] == campaign.identity(provenance)
    assert all(receipt[key] is None for key in ("guard_result", "child_result", "tee_result"))
    relative = receipt["observed_png_relative_paths"][0]
    assert (root / relative).read_bytes() == b"fixture-png"
    assert not (root / "components/run2" / relative).exists()
    assert run3["units"]["unit_1"]["status"] == "planned"
    assert run3["units"]["unit_1"]["attempts"] == []
    assert run2["units"]["unit_2"]["status"] == "planned"
    assert run2["units"]["unit_2"]["attempts"] == []
    assert campaign.main(recovery_args(root)) == 0


def test_recovery_rejects_incomplete_output_and_different_byte_collision(tmp_path, monkeypatch):
    root, _, attempt_root = seed_missing_lifecycle(tmp_path, monkeypatch)
    log_path = attempt_root / "_metadata/renderer.log"
    log_path.write_text("renderer stopped before completion\n", encoding="utf-8")
    assert campaign.main(recovery_args(root)) == 2
    assert not campaign.completion_receipt_path(attempt_root).exists()
    assert campaign.read_json(campaign.state_path(root, "sr_crlike_run2_run3", "run2"))[
        "units"]["unit_0"]["status"] == "interrupted"
    log_path.write_text(
        f"[SR] Produced 1 plots with stat⊕syst uncertainties and 0 plots with "
        f"stat-only bands after completing 1 rendering tasks in "
        f"{attempt_root / 'run2' / 'merged'}\n", encoding="utf-8")
    canonical = root / "run2/merged/unit_0/unit_0_invmass_fitting.png"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"foreign")
    assert campaign.main(recovery_args(root)) == 2
    assert canonical.read_bytes() == b"foreign"
    assert campaign.read_json(campaign.state_path(root, "sr_crlike_run2_run3", "run2"))[
        "units"]["unit_0"]["status"] == "interrupted"
    canonical.write_bytes(b"fixture-png")
    assert campaign.main(recovery_args(root)) == 0
    assert canonical.read_bytes() == b"fixture-png"


def test_legacy_human_png_migrates_with_identity_proof(tmp_path, monkeypatch):
    root, _, attempt_root = seed_missing_lifecycle(tmp_path, monkeypatch)
    assert campaign.main(recovery_args(root)) == 0
    relative = "run2/merged/unit_0/unit_0_invmass_fitting.png"
    canonical = root / relative
    legacy = root / "components/run2" / relative
    legacy.parent.mkdir(parents=True)
    canonical.rename(legacy)
    state_path = campaign.state_path(root, "sr_crlike_run2_run3", "run2")
    state = campaign.read_json(state_path)
    del state["units"]["unit_0"]["human_output_root"]
    campaign.atomic_json(state_path, state)
    assert campaign.main(recovery_args(root)) == 0
    assert canonical.read_bytes() == b"fixture-png"
    assert not legacy.exists()
    assert campaign.read_json(state_path)["units"]["unit_0"]["human_output_root"] == str(root)
    assert campaign.completion_receipt_path(attempt_root).is_file()
    legacy.write_bytes(b"fixture-png")
    assert campaign.main(recovery_args(root)) == 0
    assert legacy.read_bytes() == b"fixture-png"


def test_existing_normal_success_migrates_without_receipt_rewrite(tmp_path, monkeypatch):
    profile = "sr_crlike_run2_run3"
    root, plan_path = fixture_plan(tmp_path, monkeypatch, profile=profile, unit_count=2)
    monkeypatch.setattr(campaign, "run_guarded_child", fake_success)
    campaign.run_campaign(root, profile, "fixture_campaign", plan_path=plan_path,
                          action="execute", selected="unit_0")
    canonical = root / "run2/merged/unit_0/unit_0_invmass_fitting.png"
    legacy = root / "components/run2/run2/merged/unit_0/unit_0_invmass_fitting.png"
    legacy.parent.mkdir(parents=True)
    canonical.rename(legacy)
    runtime = campaign.read_json(campaign.coordinator_path(root))["runtime_contract_identity"]
    _, plan, states = campaign.load_campaign(root, profile, "fixture_campaign", runtime)
    unit_state = states["run2"]["units"]["unit_0"]
    del unit_state["human_output_root"]
    state_path = campaign.state_path(root, profile, "run2")
    campaign.atomic_json(state_path, states["run2"])
    receipt_path = campaign.completion_receipt_path(
        Path(unit_state["attempts"][0]["private_unit_root"]))
    receipt_before = receipt_path.read_bytes()
    monkeypatch.setattr(campaign, "run_guarded_child", lambda *_: pytest.fail("child launch"))
    campaign.migrate_successful_unit(plan, "run2", states["run2"], unit_state, root, runtime)
    assert canonical.read_bytes() == b"fixture-png" and not legacy.exists()
    assert receipt_path.read_bytes() == receipt_before
    assert campaign.read_json(campaign.state_path(root, profile, "run3"))[
        "units"]["unit_1"]["attempts"] == []


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
