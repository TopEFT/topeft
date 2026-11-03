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
    monkeypatch.setattr(campaign.backend.node_memory_guard, "read_cgroup_memory",
                        lambda: {"cgroup_current_bytes": 0, "cgroup_limit_bytes": 10**15})
    repo = tmp_path / "repo"
    (repo / "histos").mkdir(parents=True)
    monkeypatch.setattr(campaign, "REPO_ROOT", repo)
    root = repo / "histos" / "fixture_campaign"
    units = []
    for index in range(unit_count):
        components = campaign.PROFILES[profile]
        component = components[index % len(components)]
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


def fake_success(argv, attempt_root, **_options):
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


def fake_failure(argv, attempt_root, **_options):
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
        "execution_contract": {"version": 1, "backend": "local", "argv": argv,
                               "observer": "node_pressure", "hard_limit_bytes": 245000000000},
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


def resume_args(root, profile="sr_crlike_run2", selected=None, backend=None):
    args = ["--campaign-profile", profile, "--campaign-tag", "fixture_campaign",
            "-o", str(root), "--resume", "--format", "json"]
    return (args + (["--selected-unit", selected] if selected else [])
            + (["--backend", backend] if backend else []))


def finalize_args(root, selected=None):
    args = ["--finalize", "-o", str(root), "--format", "json"]
    return args + (["--selected-unit", selected] if selected else [])


def finalize_ready(root, profile="sr_crlike_run2", selected=None):
    return campaign.finalize_campaign(root, profile, "fixture_campaign", selected,
                                      emit=False)


def test_lifecycle_unification_resume_defers_and_finalize_publishes(
        tmp_path, monkeypatch, capsys):
    root, _, attempt_root = seed_running(tmp_path, monkeypatch)
    assert campaign.main(resume_args(root, selected="unit_0")) == 0
    state_path = campaign.state_path(root, "sr_crlike_run2", "run2")
    unit_state = campaign.read_json(state_path)["units"]["unit_0"]
    assert unit_state["status"] == "interrupted"
    assert unit_state["attempts"][0]["failure_classification"] == (
        campaign.terminal_postprocessing_pending)
    assert not campaign.completion_receipt_path(attempt_root).exists()
    assert not (root / "run2/merged/unit_0/unit_0_invmass_fitting.png").exists()
    assert campaign.main(finalize_args(root, "unit_0")) == 0
    unit_state = campaign.read_json(state_path)["units"]["unit_0"]
    assert unit_state["status"] == "success"
    assert campaign.completion_receipt_path(attempt_root).is_file()
    canonical = root / "run2/merged/unit_0/unit_0_invmass_fitting.png"
    assert canonical.read_bytes() == b"fixture-png"
    state_bytes = state_path.read_bytes()
    assert campaign.main(finalize_args(root, "unit_0")) == 0
    assert state_path.read_bytes() == state_bytes
    assert canonical.read_bytes() == b"fixture-png"
    capsys.readouterr()


def test_lifecycle_unification_finalize_reconciles_only_selected_running_unit(
        tmp_path, monkeypatch, capsys):
    root, _, attempt_root = seed_running(tmp_path, monkeypatch, unit_count=2)
    assert campaign.main(finalize_args(root, "unit_1")) == 2
    assert not campaign.completion_receipt_path(attempt_root).exists()
    assert campaign.main(finalize_args(root, "unit_0")) == 0
    assert campaign.completion_receipt_path(attempt_root).is_file()
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert state["units"]["unit_0"]["status"] == "success"
    assert state["units"]["unit_1"]["status"] == "planned"
    capsys.readouterr()


@pytest.mark.parametrize("selected", [None, "unit_0"])
def test_finalize_stale_running_condor_success_publishes_without_dispatch(
        tmp_path, monkeypatch, capsys, selected):
    root, _, attempt_root = seed_running(tmp_path, monkeypatch)
    state_path = campaign.state_path(root, "sr_crlike_run2", "run2")
    state = campaign.read_json(state_path)
    attempt = state["units"]["unit_0"]["attempts"][0]
    attempt["execution_contract"]["backend"] = "condor"
    campaign.atomic_json(state_path, state)
    observed = []

    def terminal_observation(argv, observed_root, contract):
        observed.append((argv, observed_root, contract))
        assert observed_root == attempt_root
        assert campaign.identity(argv) == attempt["command_identity"]
        return {"state": "terminal_success"}

    monkeypatch.setattr(campaign.backend, "observe", terminal_observation)
    monkeypatch.setattr(campaign.backend, "execute",
                        lambda *_: pytest.fail("finalize submitted a worker"))
    monkeypatch.setattr(campaign, "run_campaign",
                        lambda *_args, **_kwargs: pytest.fail("finalize entered resume/dispatch"))
    assert campaign.main(finalize_args(root, selected)) == 0
    state = campaign.read_json(state_path)
    unit = state["units"]["unit_0"]
    assert len(observed) == 1
    assert unit["status"] == "success"
    assert len(unit["attempts"]) == 1
    assert unit["attempts"][0]["attempt_id"] == "attempt_0001"
    assert unit["attempts"][0]["guard_result"] == 0
    assert unit["attempts"][0]["child_result"] == 0
    assert unit["attempts"][0]["tee_result"] == 0
    assert campaign.completion_receipt_path(attempt_root).is_file()
    assert (root / "run2/merged/unit_0/unit_0_invmass_fitting.png").read_bytes() == b"fixture-png"
    capsys.readouterr()


def test_finalize_all_skips_active_running_without_dispatch(
        tmp_path, monkeypatch, capsys):
    root, _, attempt_root = seed_running(tmp_path, monkeypatch, unit_count=2)
    state_path = campaign.state_path(root, "sr_crlike_run2", "run2")
    state = campaign.read_json(state_path)
    state["units"]["unit_0"]["attempts"][0]["execution_contract"]["backend"] = "condor"
    campaign.atomic_json(state_path, state)
    monkeypatch.setattr(campaign.backend, "observe", lambda *_: {"state": "pending"})
    monkeypatch.setattr(campaign.backend, "execute",
                        lambda *_: pytest.fail("finalize submitted a worker"))
    assert campaign.main(finalize_args(root)) == 0
    state = campaign.read_json(state_path)
    assert state["units"]["unit_0"]["status"] == "running"
    assert state["units"]["unit_1"]["status"] == "planned"
    assert not campaign.completion_receipt_path(attempt_root).exists()
    capsys.readouterr()


@pytest.mark.parametrize("case", ["active", "unresolved", "missing_lifecycle",
                                  "missing_guard", "bad_command", "bad_root",
                                  "duplicate_attempt", "nonzero_guard_child",
                                  "nonzero_tee", "bad_token"])
def test_finalize_stale_running_fails_closed(
        tmp_path, monkeypatch, capsys, case):
    root, _, attempt_root = seed_running(
        tmp_path, monkeypatch, terminal="failure" if case == "nonzero_guard_child" else "success")
    state_path = campaign.state_path(root, "sr_crlike_run2", "run2")
    state = campaign.read_json(state_path)
    attempt = state["units"]["unit_0"]["attempts"][0]
    attempt["execution_contract"]["backend"] = "condor"
    if case == "missing_lifecycle":
        campaign.lifecycle_receipt_path(attempt_root).unlink()
    elif case == "missing_guard":
        (attempt_root / "_metadata/guard.jsonl").unlink()
    elif case == "bad_command":
        attempt["command_identity"] = "0" * 64
    elif case == "bad_root":
        lifecycle_path = campaign.lifecycle_receipt_path(attempt_root)
        lifecycle = campaign.read_json(lifecycle_path)
        lifecycle["private_unit_root"] = str(root / "wrong_root")
        campaign.atomic_json(lifecycle_path, lifecycle)
    elif case == "duplicate_attempt":
        state["units"]["unit_0"]["attempts"].append(dict(attempt))
    elif case == "nonzero_tee":
        argv = campaign.attempt_argv(state["units"]["unit_0"]["binding"], attempt)
        write_lifecycle(argv, attempt_root, 0, 0, 9)
    campaign.atomic_json(state_path, state)

    def observation(*_args):
        if case == "bad_token":
            raise campaign.CampaignError("scheduler token contradiction")
        return {"state": "pending" if case == "active" else
                "unresolved" if case == "unresolved" else "terminal_success"}

    monkeypatch.setattr(campaign.backend, "observe", observation)
    monkeypatch.setattr(campaign.backend, "execute",
                        lambda *_: pytest.fail("finalize submitted a worker"))
    monkeypatch.setattr(campaign, "run_campaign",
                        lambda *_args, **_kwargs: pytest.fail("finalize entered resume/dispatch"))
    assert campaign.main(finalize_args(root, "unit_0")) == 2
    assert not campaign.completion_receipt_path(attempt_root).exists()
    assert not (root / "run2/merged/unit_0/unit_0_invmass_fitting.png").exists()
    assert len(campaign.read_json(state_path)["units"]["unit_0"]["attempts"]) == (
        2 if case == "duplicate_attempt" else 1)
    capsys.readouterr()


def test_lifecycle_unification_execute_defers_and_finalize_all(
        tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute", emit=False)
    state_path = campaign.state_path(root, "sr_crlike_run2", "run2")
    state = campaign.read_json(state_path)
    assert all(item["status"] == "interrupted" for item in state["units"].values())
    assert not list(root.rglob("completion_receipt_v1.json"))
    assert not list(root.glob("run2/merged/**/*.png"))
    monkeypatch.setattr(campaign.backend, "execute", lambda *_: pytest.fail("finalize launched worker"))
    summary = finalize_ready(root)
    assert summary["published_count"] == 2
    assert all(item["status"] == "success" for item in
               campaign.read_json(state_path)["units"].values())


def test_lifecycle_unification_finalize_blocks_failed_and_ambiguous(
        tmp_path, monkeypatch, capsys):
    failed_root, _, failed_attempt = seed_running(tmp_path / "failed", monkeypatch,
                                                   terminal="failure")
    assert campaign.main(resume_args(failed_root)) == 0
    assert campaign.main(finalize_args(failed_root, "unit_0")) == 2
    assert not campaign.completion_receipt_path(failed_attempt).exists()
    ambiguous_root, _, ambiguous_attempt = seed_running(tmp_path / "ambiguous", monkeypatch)
    campaign.lifecycle_receipt_path(ambiguous_attempt).unlink()
    (ambiguous_attempt / "_metadata/guard.jsonl").unlink()
    assert campaign.main(resume_args(ambiguous_root)) == 0
    assert campaign.main(finalize_args(ambiguous_root, "unit_0")) == 2
    assert not campaign.completion_receipt_path(ambiguous_attempt).exists()
    capsys.readouterr()


def test_lifecycle_unification_help_action_inventory(capsys):
    with pytest.raises(SystemExit) as outcome:
        campaign.main(["--campaign-profile", "sr_crlike_run2", "--help"])
    assert outcome.value.code == 0
    help_text = capsys.readouterr().out
    for option in ("--execute", "--resume", "--finalize", "--status", "--dry-run",
                   "--activate-pending-plan", "--selected-unit", "--format",
                   "--follow", "--verify-published", "--observe-scheduler",
                   "--backend", "--resource-observer", "--hard-limit-bytes"):
        assert option in help_text
    for option in ("--recover-terminal-unit", "--recover-rejected-condor-submit",
                   "--recover-failed-unit", "--one-pass", "--publish", "--reconcile"):
        assert option not in help_text


@pytest.mark.parametrize("removed", ["--recover-terminal-unit",
                                      "--recover-rejected-condor-submit",
                                      "--recover-failed-unit", "--one-pass"])
def test_lifecycle_unification_removed_options_are_unsupported(
        tmp_path, monkeypatch, removed):
    root, _, _ = seed_running(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as outcome:
        campaign.main(["--resume", "-o", str(root), removed])
    assert outcome.value.code == 2


def test_action_scope_limits_legacy_policy_binding(tmp_path, monkeypatch, capsys):
    root, plan, _ = seed_running(tmp_path, monkeypatch, unit_count=2)
    coordinator_path = campaign.coordinator_path(root)
    before = coordinator_path.read_bytes()
    assert campaign.main(resume_args(root, selected="unit_0")) == 0
    capsys.readouterr()
    assert coordinator_path.read_bytes() == before
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert state["units"]["unit_0"]["status"] == "interrupted"
    assert state["units"]["unit_1"]["status"] == "planned"
    assert campaign.main(resume_args(root, selected="unit_0")) == 0
    assert coordinator_path.read_bytes() == before


def test_legacy_condor_policy_binds_once_before_selected_submit(tmp_path, monkeypatch, capsys):
    root, _, _ = seed_running(tmp_path, monkeypatch, unit_count=3)
    submitted = []

    def fake_execute(argv, attempt_root, contract):
        bound = campaign.read_json(campaign.coordinator_path(root))["execution_policy_v1"]
        assert bound["backend"] == "condor"
        assert bound["resource_observer"] == "node_pressure"
        assert bound["hard_limit_bytes"] == 123456789
        submitted.append(attempt_root.parent.name)
        raise campaign.ActiveChildUnresolved("synthetic submitted child")

    monkeypatch.setattr(campaign.backend, "execute", fake_execute)
    base = resume_args(root, selected="unit_1") + ["--backend", "condor"]
    assert campaign.main(base) == 2
    assert submitted == []
    assert "execution_policy_v1" not in campaign.read_json(campaign.coordinator_path(root))
    assert campaign.main(base + ["--resource-observer", "node_pressure",
                                "--hard-limit-bytes", "123456789"]) == 0
    assert submitted == ["unit_1"]
    assert campaign.read_json(campaign.coordinator_path(root))["execution_policy_v1"][
        "source"] == "explicitly_bound"
    assert campaign.main(resume_args(root, selected="unit_2")) == 0
    assert submitted == ["unit_1", "unit_2"]
    assert campaign.main(resume_args(root, selected="unit_2") + ["--backend", "local"]) == 2
    assert submitted == ["unit_1", "unit_2"]
    capsys.readouterr()


def test_unselected_legacy_condor_requires_complete_policy(tmp_path, monkeypatch, capsys):
    root, _, _ = seed_running(tmp_path, monkeypatch, unit_count=2)
    assert campaign.main(resume_args(root) + ["--backend", "condor"]) == 2
    assert "execution_policy_v1" not in campaign.read_json(campaign.coordinator_path(root))
    capsys.readouterr()


def test_status_axes_and_dispatch_blocking_are_orthogonal(tmp_path, monkeypatch):
    root, plan, attempt_root = seed_running(tmp_path, monkeypatch, unit_count=2)
    campaign.lifecycle_receipt_path(attempt_root).unlink()
    state_path = campaign.state_path(root, "sr_crlike_run2", "run2")
    state = campaign.read_json(state_path)
    states = {"run2": state}
    scope = campaign.derive_action_scope(plan, states)
    local = campaign.campaign_summary(root, plan, states, action_scope=scope,
                                      execution_policy={"backend": "local"})
    assert local["summary_schema_version"] == 2
    assert local["active_child_present"] is True
    assert local["dispatch_blocked"] is True
    assert local["waiting_on_active_child"] is True
    assert local["campaign_status"] == "waiting"
    state["units"]["unit_0"]["attempts"][0]["execution_contract"]["backend"] = "condor"
    condor = campaign.campaign_summary(root, plan, states, action_scope=scope,
                                       execution_policy={"backend": "condor"})
    assert condor["active_child_present"] is True
    assert condor["dispatch_blocked"] is False
    assert condor["waiting_on_active_child"] is False
    assert condor["waiting_reasons"] == []
    assert condor["campaign_status"] == "in_progress"
    state["units"]["unit_0"]["status"] = "interrupted"
    state["units"]["unit_0"]["attempts"][0]["result"] = "interrupted"
    ordinary = campaign.campaign_summary(root, plan, states)
    assert ordinary["units"][0]["publish"] == "not published"
    state["units"]["unit_0"]["attempts"][0]["failure_classification"] = (
        campaign.terminal_postprocessing_pending)
    recovery = campaign.campaign_summary(root, plan, states)
    assert recovery["units"][0]["publish"] == "publish pending"


def test_published_content_verification_is_read_only_and_detects_same_size_change(
        tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute", emit=False)
    finalize_ready(root)
    state_path = campaign.state_path(root, "sr_crlike_run2", "run2")
    coordinator_path = campaign.coordinator_path(root)
    before = (state_path.read_bytes(), coordinator_path.read_bytes())
    original_sha = campaign.sha256_file
    monkeypatch.setattr(campaign, "sha256_file",
                        lambda *_: pytest.fail("ordinary status hashed content"))
    assert campaign.main(["--status", "-o", str(root), "--format", "json"]) == 0
    ordinary = json.loads(capsys.readouterr().out)
    assert ordinary["units"][0]["integrity"] == "metadata_consistent"
    assert ordinary["units"][0]["publish"] == "published"
    monkeypatch.setattr(campaign, "sha256_file", original_sha)
    assert campaign.main(["--status", "-o", str(root), "--verify-published",
                          "--format", "json"]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["units"][0]["integrity"] == "content_verified"
    canonical = Path(verified["units"][0]["canonical_paths"][0])
    original_bytes = canonical.read_bytes()
    canonical.write_bytes(original_bytes[:-1] + bytes([original_bytes[-1] ^ 1]))
    assert canonical.stat().st_size == len(original_bytes)
    assert campaign.main(["--status", "-o", str(root), "--verify-published",
                          "--format", "json"]) == 0
    mismatched = json.loads(capsys.readouterr().out)
    assert mismatched["units"][0]["publish"] == "published"
    assert mismatched["units"][0]["integrity"] == "content_mismatch"
    assert mismatched["content_mismatch_count"] == 1
    assert (state_path.read_bytes(), coordinator_path.read_bytes()) == before
    canonical.unlink()
    canonical.symlink_to(Path(campaign.read_json(state_path)["units"]["unit_0"][
        "attempts"][0]["private_unit_root"]) / canonical.relative_to(root))
    monkeypatch.setattr(campaign, "sha256_canonical_file",
                        lambda *_: pytest.fail("verification followed a canonical symlink"))
    assert campaign.main(["--status", "-o", str(root), "--verify-published",
                          "--format", "json"]) == 0
    unsafe = json.loads(capsys.readouterr().out)
    assert unsafe["units"][0]["integrity"] == "unavailable"
    assert (state_path.read_bytes(), coordinator_path.read_bytes()) == before


@pytest.mark.parametrize("active_backend,future_backend,starts,blocked", [
    ("local", "local", 0, True),
    ("condor", "local", 1, False),
    ("local", "condor", 1, False),
    ("condor", "condor", 1, False),
])
def test_dispatch_decision_controls_status_and_launch(
        tmp_path, monkeypatch, capsys, active_backend, future_backend, starts, blocked):
    root, _, _ = seed_running(tmp_path, monkeypatch, unit_count=3)
    state_path = campaign.state_path(root, "sr_crlike_run2", "run2")
    state = campaign.read_json(state_path)
    state["units"]["unit_0"]["attempts"][-1]["execution_contract"]["backend"] = active_backend
    campaign.atomic_json(state_path, state)
    coordinator_path = campaign.coordinator_path(root)
    coordinator = campaign.read_json(coordinator_path)
    policy = {"backend": future_backend, "resource_observer": "cgroup_v2",
              "hard_limit_bytes": 192000000000, "source": "explicitly_bound"}
    coordinator["execution_policy_v1"] = policy
    campaign.atomic_json(coordinator_path, coordinator)
    monkeypatch.setattr(campaign, "reconcile_running",
                        lambda *args: (_ for _ in ()).throw(
                            campaign.ActiveChildUnresolved("synthetic active child")))
    calls = []

    def fake_execute(argv, attempt_root, contract):
        calls.append((attempt_root.parent.name, attempt_root.name, contract["backend"]))
        raise campaign.ActiveChildUnresolved("synthetic new child")

    monkeypatch.setattr(campaign.backend, "execute", fake_execute)
    original_summary = campaign.campaign_summary
    decisions = []
    original_decision = campaign.dispatch_decision

    def record_decision(*args, **kwargs):
        result = original_decision(*args, **kwargs)
        decisions.append(result)
        return result

    def inspect_summary(*args, **kwargs):
        assert kwargs["dispatch"] is decisions[-1]
        return original_summary(*args, **kwargs)

    monkeypatch.setattr(campaign, "dispatch_decision", record_decision)
    monkeypatch.setattr(campaign, "campaign_summary", inspect_summary)
    assert campaign.main(resume_args(root, selected="unit_1")) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["active_child_present"] is True
    assert summary["dispatch_blocked"] is blocked
    assert summary["waiting_on_active_child"] is blocked
    assert summary["dispatch_blocked"] is decisions[-1]["dispatch_blocked"]
    assert calls == ([('unit_1', 'attempt_0001', future_backend)] if starts else [])
    final_state = campaign.read_json(state_path)
    assert len(final_state["units"]["unit_0"]["attempts"]) == 1
    assert len(final_state["units"]["unit_1"]["attempts"]) == starts
    assert final_state["units"]["unit_2"]["status"] == "planned"


def test_dispatch_rejects_invalid_active_backend_before_launch(tmp_path, monkeypatch):
    root, _, _ = seed_running(tmp_path, monkeypatch, unit_count=2)
    state_path = campaign.state_path(root, "sr_crlike_run2", "run2")
    state = campaign.read_json(state_path)
    state["units"]["unit_0"]["attempts"][-1]["execution_contract"]["backend"] = "unknown"
    campaign.atomic_json(state_path, state)
    monkeypatch.setattr(campaign, "reconcile_running",
                        lambda *args: (_ for _ in ()).throw(
                            campaign.ActiveChildUnresolved("synthetic active child")))
    monkeypatch.setattr(campaign.backend, "execute", lambda *args: pytest.fail("launch"))
    assert campaign.main(resume_args(root, selected="unit_1", backend="local")) == 2
    state = campaign.read_json(state_path)
    assert state["units"]["unit_1"]["status"] == "planned"


def test_publication_history_survives_current_file_damage(tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute", emit=False)
    finalize_ready(root)
    state_path = campaign.state_path(root, "sr_crlike_run2", "run2")
    coordinator_path = campaign.coordinator_path(root)
    before = (state_path.read_bytes(), coordinator_path.read_bytes())

    def status(verify=False):
        args = ["--status", "-o", str(root), "--format", "json"]
        if verify:
            args.append("--verify-published")
        assert campaign.main(args) == 0
        result = json.loads(capsys.readouterr().out)["units"][0]
        assert result["publication"] == "published"
        assert result["canonical_paths"] == [str(canonical)]
        assert (state_path.read_bytes(), coordinator_path.read_bytes()) == before
        return result

    state = campaign.read_json(state_path)
    relative = next(iter(state["units"]["unit_0"]["materializations"]))
    canonical = root / relative
    healthy = canonical.read_bytes()
    assert status()["integrity"] == "metadata_consistent"
    assert status(True)["integrity"] == "content_verified"
    canonical.write_bytes(healthy[:-1] + bytes([healthy[-1] ^ 1]))
    assert status(True)["integrity"] == "content_mismatch"
    canonical.write_bytes(healthy[:2])
    assert status()["integrity"] == "content_mismatch"
    canonical.unlink()
    assert status()["integrity"] == "unavailable"
    canonical.symlink_to(root / "missing-target")
    final_link = status(True)
    assert final_link["integrity"] == "unavailable"
    assert "symlink" in final_link["warning"]
    canonical.unlink()
    parent = canonical.parent
    saved_parent = parent.with_name(parent.name + "_saved")
    parent.rename(saved_parent)
    parent.symlink_to(saved_parent)
    ancestor_link = status(True)
    assert ancestor_link["integrity"] == "unavailable"
    assert "symbolic-link" in ancestor_link["warning"]
    parent.unlink()
    saved_parent.rename(parent)
    canonical.write_bytes(healthy)
    damaged = campaign.read_json(state_path)
    damaged["units"]["unit_0"]["materializations"][relative]["receipt_identity"] = "0" * 64
    campaign.atomic_json(state_path, damaged)
    assert campaign.main(["--status", "-o", str(root), "--format", "json"]) == 0
    result = json.loads(capsys.readouterr().out)["units"][0]
    assert result["publication"] != "published"
    assert result["integrity"] == "unavailable"


def test_legacy_unselected_four_condor_units_bind_before_submit_once(
        tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=7)
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute", selected="unit_0", emit=False)
    for index in (1, 2):
        campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                              action="resume", selected=f"unit_{index}", emit=False)
    coordinator_path = campaign.coordinator_path(root)
    coordinator = campaign.read_json(coordinator_path)
    del coordinator["execution_policy_v1"]
    campaign.atomic_json(coordinator_path, coordinator)
    submitted = []

    def fake_submit(argv, attempt_root, contract):
        bound = campaign.read_json(coordinator_path)["execution_policy_v1"]
        assert bound == {"backend": "condor", "resource_observer": "cgroup_v2",
                         "hard_limit_bytes": 192000000000, "source": "explicitly_bound"}
        submitted.append((attempt_root.parent.name, attempt_root.name))
        raise campaign.ActiveChildUnresolved("synthetic submitted child")

    monkeypatch.setattr(campaign.backend, "execute", fake_submit)
    observed = []

    def fake_observe(argv, attempt_root, contract):
        observed.append((attempt_root.parent.name, contract["backend"]))
        return {"state": "pending"}

    monkeypatch.setattr(campaign.backend, "observe", fake_observe)
    base = resume_args(root)
    assert campaign.main(base + ["--backend", "condor"]) == 2
    assert "execution_policy_v1" not in campaign.read_json(coordinator_path)
    assert submitted == []
    assert campaign.main(base + ["--backend", "condor", "--resource-observer", "cgroup_v2",
                                 "--hard-limit-bytes", "192000000000"]) == 0
    capsys.readouterr()
    assert submitted == [(f"unit_{index}", "attempt_0001") for index in range(3, 7)]
    assert observed == [(f"unit_{index}", "condor") for index in range(3, 7)]
    state_path = campaign.state_path(root, "sr_crlike_run2", "run2")
    state = campaign.read_json(state_path)
    assert [state["units"][f"unit_{index}"]["status"] for index in range(3)] == [
        "interrupted"] * 3
    assert campaign.main(base) == 0
    assert submitted == [(f"unit_{index}", "attempt_0001") for index in range(3, 7)]
    assert observed == [(f"unit_{index}", "condor") for index in range(3, 7)] * 2
    state = campaign.read_json(state_path)
    assert all(len(state["units"][f"unit_{index}"]["attempts"]) == 1
               for index in range(7))


def test_fresh_policy_defaults_and_help_docs_agree(tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute", emit=False)
    bound = campaign.read_json(campaign.coordinator_path(root))["execution_policy_v1"]
    assert {key: bound[key] for key in campaign.DEFAULT_EXECUTION_POLICY} == (
        campaign.DEFAULT_EXECUTION_POLICY)
    assert bound["source"] == "fresh_defaults"
    with pytest.raises(SystemExit) as exit_info:
        campaign.main(["--campaign-profile", "sr_crlike_run2", "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert 'opt in to foreground Condor' in help_text
    assert '--finalize' in help_text and '--one-pass' not in help_text
    for value in ("local", "cgroup_v2", "245000000000"):
        assert value in help_text
        assert value in (MODULE_PATH.parents[2] / "docs/how_to/plotting.md").read_text()


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
            "-o", str(root), "--resume", "--selected-unit", "unit_0"]


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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute", selected="unit_0")
    finalize_ready(root, selected="unit_0")
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
    fake_success(campaign.child_argv(unit, attempt_root, historical=True), attempt_root)
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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign", action="resume")
    finalize_ready(root)
    recovered = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert recovered["units"]["unit_0"]["status"] == "success"
    assert recovered["units"]["unit_0"]["attempts"][0]["materialization_status"] == "complete"


def test_unbound_collision_fails_without_overwrite(tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    relative = Path("run2/merged/unit_0/unit_0_invmass_fitting.png")

    def collide(argv, attempt_root, **_options):
        fake_success(argv, attempt_root)
        if attempt_root.parent.name == "unit_0":
            destination = root / relative
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"foreign")
        return 0, 0, 0

    monkeypatch.setattr(campaign.backend, "run_guarded_child", collide)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    assert (root / relative).read_bytes() == b"foreign"
    with pytest.raises(campaign.OutputCollision):
        finalize_ready(root, selected="unit_0")
    finalize_ready(root, selected="unit_1")
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert state["units"]["unit_0"]["status"] == "interrupted"
    assert state["units"]["unit_1"]["status"] == "success"
    summary = campaign.campaign_summary(root, campaign.read_json(campaign.coordinator_path(root))["plan"],
                                        {"run2": state})
    assert summary["campaign_status"] == "partial"
    assert (summary["planned_count"], summary["success_count"], summary["interrupted_count"]) == (2, 1, 1)


def test_failed_and_interrupted_are_not_retried(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_failure)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute", selected="unit_0")
    state_path = campaign.state_path(root, "sr_crlike_run2", "run2")
    state = campaign.read_json(state_path)
    state["units"]["unit_1"]["status"] = "interrupted"
    state["units"]["unit_1"]["attempts"] = [dict(
        state["units"]["unit_0"]["attempts"][0], result="interrupted")]
    campaign.atomic_json(state_path, state)
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("silent retry"))
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign", action="resume")
    state = campaign.read_json(state_path)
    assert state["units"]["unit_0"]["status"] == "failed"
    assert state["units"]["unit_1"]["status"] == "interrupted"


def test_revision_appends_and_preserves_history(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_success)
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

    def selective(argv, attempt_root, **_options):
        if "run2" in attempt_root.parts:
            return fake_failure(argv, attempt_root)
        return fake_success(argv, attempt_root)

    monkeypatch.setattr(campaign.backend, "run_guarded_child", selective)
    campaign.run_campaign(root, "sr_crlike_run2_run3", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    finalize_ready(root, profile="sr_crlike_run2_run3", selected="unit_1")
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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("overlap"))
    assert campaign.main(resume_args(root, profile, selected="unit_1", backend="local")) == 0
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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    assert campaign.main(resume_args(root)) == 0
    finalize_ready(root)
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert state["units"]["unit_0"]["status"] == "success"
    assert campaign.completion_receipt_path(attempt_root).is_file()


def test_post_tuple_terminal_state_recovers_after_interruption_without_child_rerun(
        tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)
    original_verify = campaign.verify_lifecycle_receipt
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_success)

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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    assert campaign.main(resume_args(root)) == 0
    finalize_ready(root)
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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    assert campaign.main(resume_args(root)) == 0
    finalize_ready(root)
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert state["units"]["unit_0"]["status"] == "success"
    assert state["units"]["unit_0"]["attempts"][0]["failure_classification"] is None


def test_failed_lifecycle_and_missing_lifecycle_are_terminal_without_rerun(tmp_path, monkeypatch):
    root, _, _ = seed_running(tmp_path / "failed", monkeypatch, terminal="failure")
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_success)
    assert campaign.main(resume_args(root, profile, backend="local")) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["campaign_status"] == "partial"
    assert (summary["planned_count"], summary["success_count"], summary["failed_count"]) == (2, 0, 1)
    run2 = campaign.read_json(campaign.state_path(root, profile, "run2"))
    run3 = campaign.read_json(campaign.state_path(root, profile, "run3"))
    assert run2["units"]["unit_0"]["status"] == "failed"
    assert run3["units"]["unit_1"]["status"] == "interrupted"
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("silent retry"))
    assert campaign.main(resume_args(root, profile, selected="unit_1")) == 0
    assert campaign.main(resume_args(root, profile, selected="unit_0")) == 0
    assert campaign.main(resume_args(root, profile)) == 0


def test_malformed_component_state_remains_invocation_error(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_success)
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

    def record_child(argv, attempt_root, **_options):
        launched.append(attempt_root.parent.name)
        return fake_success(argv, attempt_root)

    monkeypatch.setattr(campaign.backend, "run_guarded_child", record_child)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert summary["campaign_status"] == "partial"
    assert summary["waiting_on_active_child"] is False
    assert launched == ["unit_1"]
    assert state["units"]["unit_0"]["status"] == "failed"
    assert state["units"]["unit_0"]["attempts"][0]["result"] == "failed"
    assert state["units"]["unit_1"]["status"] == "interrupted"


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
    original_run = campaign.backend.run_guarded_child
    launches = []

    class TerminalGuard:
        def __init__(self, guard_path):
            self.guard_path = guard_path
            self.stdout = io.BytesIO()
            self.guard_path.write_text(guard_result_line(), encoding="utf-8")

        def wait(self):
            pytest.fail("tee creation failure must not wait for an undrained guard")

    def run_child(argv, attempt_root, **_options):
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

    monkeypatch.setattr(campaign.backend, "run_guarded_child", run_child)
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
    assert state["units"]["unit_1"]["status"] == "interrupted"


@pytest.mark.parametrize("failure", ["stdout_close", "tee_wait", "guard_wait"])
def test_terminal_guard_survives_supervisor_cleanup_failure(
        failure, tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    original_run = campaign.backend.run_guarded_child
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

    def run_child(argv, attempt_root, **_options):
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

    monkeypatch.setattr(campaign.backend, "run_guarded_child", run_child)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert launches == ["unit_0", "unit_1"]
    assert summary["waiting_on_active_child"] is False
    assert summary["running_count"] == 0
    assert state["units"]["unit_0"]["status"] == "failed"
    assert state["units"]["unit_1"]["status"] == "interrupted"


@pytest.mark.parametrize("guard_record", [None, '{"event":"result"}\n'])
def test_tee_creation_without_terminal_guard_does_not_wait_or_launch_next(
        guard_record, tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    original_run = campaign.backend.run_guarded_child
    launches = []

    class UnresolvedGuard:
        def __init__(self, guard_path):
            self.stdout = io.BytesIO()
            if guard_record is not None:
                guard_path.write_text(guard_record, encoding="utf-8")

        def wait(self):
            pytest.fail("undrained guard wait would block")

    def run_child(argv, attempt_root, **_options):
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

    monkeypatch.setattr(campaign.backend, "run_guarded_child", run_child)
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
    original_run = campaign.backend.run_guarded_child
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

    def run_child(argv, attempt_root, **_options):
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
            supervisor_patch.setattr(campaign.backend, "atomic_json", failed_lifecycle_write)
            return original_run(argv, attempt_root)

    monkeypatch.setattr(campaign.backend, "run_guarded_child", run_child)
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
    assert state["units"]["unit_1"]["status"] == "interrupted"


@pytest.mark.parametrize("source", ["supervisor_postproof", "terminal_exception"])
def test_unprintable_terminal_error_cannot_leave_running(
        source, tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    original_run = campaign.backend.run_guarded_child
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

    def run_child(argv, attempt_root, **_options):
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
            process_patch.setattr(campaign.backend, "atomic_json", failed_lifecycle_write)
            return original_run(argv, attempt_root)

    monkeypatch.setattr(campaign.backend, "run_guarded_child", run_child)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert launches == ["unit_0", "unit_1"]
    assert summary["running_count"] == 0
    assert state["units"]["unit_0"]["status"] == "failed"
    assert state["units"]["unit_0"]["attempts"][0]["failure_classification"]
    assert state["units"]["unit_1"]["status"] == "interrupted"


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

    def invalid_lifecycle(argv, attempt_root, **_options):
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

    monkeypatch.setattr(campaign.backend, "run_guarded_child", invalid_lifecycle)
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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    attempt = state["units"]["unit_0"]["attempts"][0]
    assert len(calls) >= 2
    assert summary["running_count"] == 0
    assert state["units"]["unit_0"]["status"] == attempt["result"] == "interrupted"
    assert attempt["failure_classification"] == campaign.terminal_postprocessing_pending
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

    def terminal_error(argv, attempt_root, **_options):
        launches.append(attempt_root.parent.name)
        if attempt_root.parent.name == "unit_0":
            raise campaign.ChildDefinitelyTerminal("inert terminal supervisor failure")
        return fake_success(argv, attempt_root)

    monkeypatch.setattr(campaign, "utc_now", fail_late_clock)
    monkeypatch.setattr(campaign.backend, "run_guarded_child", terminal_error)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert launches == ["unit_0", "unit_1"]
    assert summary["running_count"] == 0
    assert state["units"]["unit_0"]["status"] == "failed"
    assert state["units"]["unit_1"]["status"] == "interrupted"


def test_resume_terminal_guard_with_bad_lifecycle_is_interrupted(tmp_path, monkeypatch, capsys):
    root, _, attempt_root = seed_running(tmp_path, monkeypatch)
    campaign.atomic_json(campaign.lifecycle_receipt_path(attempt_root), [])
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
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

    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    monkeypatch.setattr(campaign, "create_receipt", failed_receipt)
    assert campaign.main(resume_args(root)) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    attempt = state["units"]["unit_0"]["attempts"][0]
    assert summary["campaign_status"] == "partial"
    assert summary["waiting_on_active_child"] is False
    assert summary["running_count"] == 0
    assert state["units"]["unit_0"]["status"] == attempt["result"] == "interrupted"
    assert summary["unit_errors"] == []
    assert campaign.main(finalize_args(root, "unit_0")) == 2
    assert "inert receipt creation failure" in capsys.readouterr().err


@pytest.mark.parametrize("guard_record", ['[]\n', '{"event":"result"}\n'])
def test_resume_malformed_guard_evidence_waits_without_launch(
        guard_record, tmp_path, monkeypatch, capsys):
    profile = "sr_crlike_run2_run3"
    root, _, attempt_root = seed_running(tmp_path, monkeypatch, profile=profile)
    campaign.lifecycle_receipt_path(attempt_root).unlink()
    (attempt_root / "_metadata" / "guard.jsonl").write_text(guard_record, encoding="utf-8")
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("overlap"))
    assert campaign.main(resume_args(root, profile, selected="unit_1", backend="local")) == 0
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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("overlap"))
    assert campaign.main(resume_args(root, profile, selected="unit_1", backend="local")) == 0
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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
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

    def ambiguous(argv, attempt_root, **_options):
        calls.append(attempt_root.parent.name)
        raise OSError("possibly started")

    monkeypatch.setattr(campaign.backend, "run_guarded_child", ambiguous)
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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute", selected="unit_0")
    finalize_ready(root, selected="unit_0")
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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("child must not start"))
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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute", selected="unit_0")
    source.write_text("changed", encoding="utf-8")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "other_git"))
    monkeypatch.setenv("GIT_BRANCH", "other_branch")
    monkeypatch.setenv("GIT_HEAD", "f" * 40)
    monkeypatch.setattr(campaign, "runtime_identity", lambda: pytest.fail("runtime provenance queried"))
    assert campaign.main(resume_args(root)) == 0
    state = campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))
    assert state["units"]["unit_1"]["status"] == "interrupted"
    assert state["runtime_contract_identity"] == historical
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["campaign_status"] == "partial"


def test_missing_renderer_log_is_receipt_evidence(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)

    def success_without_log(argv, attempt_root, **_options):
        result = fake_success(argv, attempt_root)
        (attempt_root / "_metadata/renderer.log").unlink()
        return result

    monkeypatch.setattr(campaign.backend, "run_guarded_child", success_without_log)
    assert campaign.main(["--campaign-profile", "sr_crlike_run2", "--campaign-tag",
                          "fixture_campaign", "-o", str(root), "--plan", str(plan_path),
                          "--execute"]) == 0
    finalize_ready(root)
    receipt = campaign.read_json(campaign.completion_receipt_path(
        root / ".campaign_units/unit_0/attempt_0001"))
    assert receipt["renderer_log_status"] == "missing"
    assert receipt["renderer_log_line_count"] == 0
    assert receipt["renderer_messages"] == []


def test_inert_guard_and_visible_tee_lifecycle(tmp_path):
    attempt_root = tmp_path / "attempt_0001"
    attempt_root.mkdir()
    assert campaign.backend.run_guarded_child(["/bin/echo", "inert-renderer-fixture"], attempt_root) == (0, 0, 0)
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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2", "fixture_campaign",
                          plan_path=plan_path, action="execute")
    finalize_ready(root)
    state_path = campaign.state_path(root, "sr_crlike_run2", "run2")
    state = campaign.read_json(state_path)
    unit_state = state["units"]["unit_0"]
    receipt = campaign.read_json(campaign.completion_receipt_path(
        Path(unit_state["attempts"][0]["private_unit_root"])))
    destination = root / receipt["observed_png_relative_paths"][0]
    unit_state["status"] = "interrupted"
    unit_state["attempts"][0]["materialization_status"] = "pending"
    unit_state["attempts"][0]["result"] = "interrupted"
    unit_state["attempts"][0]["failure_classification"] = campaign.terminal_postprocessing_pending
    campaign.atomic_json(state_path, state)
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("child rerun"))
    finalize_ready(root)
    assert destination.read_bytes() == b"fixture-png"
    state = campaign.read_json(state_path)
    state["units"]["unit_0"]["status"] = "interrupted"
    state["units"]["unit_0"]["attempts"][0]["result"] = "interrupted"
    state["units"]["unit_0"]["attempts"][0]["failure_classification"] = campaign.terminal_postprocessing_pending
    state["units"]["unit_0"]["attempts"][0]["materialization_status"] = "pending"
    campaign.atomic_json(state_path, state)
    destination.write_bytes(b"changed")
    with pytest.raises(campaign.OutputCollision):
        finalize_ready(root)
    assert destination.read_bytes() == b"changed"
    assert campaign.read_json(state_path)["units"]["unit_0"]["status"] == "interrupted"


def test_missing_lifecycle_recovery_is_no_launch_and_truthful(tmp_path, monkeypatch):
    root, _, attempt_root = seed_missing_lifecycle(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("child launch"))
    monkeypatch.setattr(campaign, "normalize_unit", lambda *_: pytest.fail("input read"))
    assert campaign.main(recovery_args(root)) == 0
    run2 = campaign.read_json(campaign.state_path(root, "sr_crlike_run2_run3", "run2"))
    run3 = campaign.read_json(campaign.state_path(root, "sr_crlike_run2_run3", "run3"))
    unit = run2["units"]["unit_0"]
    attempt = unit["attempts"][0]
    assert unit["status"] == "interrupted" and len(unit["attempts"]) == 1
    assert all(attempt[key] is None for key in ("guard_result", "child_result", "tee_result"))
    assert not campaign.lifecycle_receipt_path(attempt_root).exists()
    assert not campaign.completion_receipt_path(attempt_root).exists()
    finalize_ready(root, profile="sr_crlike_run2_run3", selected="unit_0")
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
    assert campaign.main(recovery_args(root)) == 0
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
    assert campaign.main(recovery_args(root)) == 0
    assert campaign.main(finalize_args(root, "unit_0")) == 2
    assert canonical.read_bytes() == b"foreign"
    assert campaign.read_json(campaign.state_path(root, "sr_crlike_run2_run3", "run2"))[
        "units"]["unit_0"]["status"] == "interrupted"
    canonical.write_bytes(b"fixture-png")
    assert campaign.main(finalize_args(root, "unit_0")) == 0
    assert canonical.read_bytes() == b"fixture-png"


def test_legacy_human_png_migrates_with_identity_proof(tmp_path, monkeypatch):
    root, _, attempt_root = seed_missing_lifecycle(tmp_path, monkeypatch)
    assert campaign.main(recovery_args(root)) == 0
    finalize_ready(root, profile="sr_crlike_run2_run3", selected="unit_0")
    relative = "run2/merged/unit_0/unit_0_invmass_fitting.png"
    canonical = root / relative
    legacy = root / "components/run2" / relative
    legacy.parent.mkdir(parents=True)
    canonical.rename(legacy)
    state_path = campaign.state_path(root, "sr_crlike_run2_run3", "run2")
    state = campaign.read_json(state_path)
    del state["units"]["unit_0"]["human_output_root"]
    campaign.atomic_json(state_path, state)
    finalize_ready(root, profile="sr_crlike_run2_run3", selected="unit_0")
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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_success)
    campaign.run_campaign(root, profile, "fixture_campaign", plan_path=plan_path,
                          action="execute", selected="unit_0")
    finalize_ready(root, profile=profile, selected="unit_0")
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
    monkeypatch.setattr(campaign.backend, "run_guarded_child", lambda *_: pytest.fail("child launch"))
    finalize_ready(root, profile=profile, selected="unit_0")
    assert canonical.read_bytes() == b"fixture-png" and not legacy.exists()
    assert receipt_path.read_bytes() == receipt_before
    assert campaign.read_json(campaign.state_path(root, profile, "run3"))[
        "units"]["unit_1"]["attempts"] == []


def test_pending_combined_revision_reconciles_after_interruption(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch,
                                   profile="sr_crlike_run2_run3", unit_count=2)
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_success)
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
    assert states["run2"]["units"]["unit_0"]["status"] == "interrupted"
    assert states["run3"]["units"]["unit_2"]["status"] == "interrupted"


@pytest.mark.parametrize('submission', ['123.0\n', '', '123.0 - 123.1\n'])
def test_condor_waits_without_local_fallback_or_second_attempt(tmp_path, monkeypatch, capsys, submission):
    from types import SimpleNamespace
    from analysis.topeft_run2 import plot_campaign_condor as condor
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    pool = 'collector.fixture.invalid'
    schedd = 'submit.fixture.invalid'
    monkeypatch.setattr(condor, 'resolve_target', lambda: {'pool': pool, 'schedd': schedd})
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        assert command[1:5] == ['-pool', pool, '-name', schedd]
        return SimpleNamespace(stdout=submission if command[0] == 'condor_submit' else '[]',
                               stderr='', returncode=0)
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    monkeypatch.setattr(campaign.backend, 'run_guarded_child', lambda *a, **k: pytest.fail('local fallback'))
    campaign.run_campaign(root, 'sr_crlike_run2', 'fixture_campaign', plan_path=plan_path,
                          action='execute', backend_name='condor')
    with pytest.raises(campaign.CampaignError, match="contradicts bound"):
        campaign.run_campaign(root, 'sr_crlike_run2', 'fixture_campaign',
                              action='resume', backend_name='local')
    state=campaign.read_json(campaign.state_path(root,'sr_crlike_run2','run2'))
    assert state['units']['unit_0']['status']=='running'
    assert len(state['units']['unit_0']['attempts'])==1
    assert state['units']['unit_1']['status']=='running'
    assert len(state['units']['unit_1']['attempts'])==1
    assert sum(c[0]=='condor_submit' for c in calls)==2
    assert campaign.read_json(campaign.coordinator_path(root))["execution_policy_v1"]["backend"] == "condor"
    capsys.readouterr()


def test_condor_dispatches_all_planned_units_and_resume_does_not_retry(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=4)
    submitted = []
    def fake_execute(argv, attempt_root, contract):
        submitted.append((attempt_root.parent.name, attempt_root.name, contract['backend']))
        raise campaign.ActiveChildUnresolved('synthetic independent Condor job')
    monkeypatch.setattr(campaign.backend, 'execute', fake_execute)
    monkeypatch.setattr(campaign.backend, 'observe', lambda *a: {'state': 'pending'})
    campaign.run_campaign(root, 'sr_crlike_run2', 'fixture_campaign',
                          plan_path=plan_path, action='execute', backend_name='condor')
    assert submitted == [(f'unit_{i}', 'attempt_0001', 'condor') for i in range(4)]
    campaign.run_campaign(root, 'sr_crlike_run2', 'fixture_campaign',
                          action='resume', backend_name='condor')
    assert len(submitted) == 4
    state = campaign.read_json(campaign.state_path(root, 'sr_crlike_run2', 'run2'))
    assert all(item['status'] == 'running' and len(item['attempts']) == 1
               for item in state['units'].values())


def test_selected_condor_resume_dispatches_only_selected_unit(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=4)
    submitted = []
    def fake_execute(argv, attempt_root, contract):
        submitted.append(attempt_root.parent.name)
        raise campaign.ActiveChildUnresolved('synthetic submission')
    monkeypatch.setattr(campaign.backend, 'execute', fake_execute)
    campaign.run_campaign(root, 'sr_crlike_run2', 'fixture_campaign',
                          plan_path=plan_path, action='execute', selected='unit_0',
                          backend_name='condor')
    campaign.run_campaign(root, 'sr_crlike_run2', 'fixture_campaign',
                          action='resume', selected='unit_2', backend_name='condor')
    assert submitted == ['unit_0', 'unit_2']


def test_one_running_condor_unit_does_not_block_three_planned(tmp_path, monkeypatch):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=4)
    submitted = []
    def fake_execute(argv, attempt_root, contract):
        submitted.append(attempt_root.parent.name)
        raise campaign.ActiveChildUnresolved('synthetic submission')
    monkeypatch.setattr(campaign.backend, 'execute', fake_execute)
    monkeypatch.setattr(campaign.backend, 'observe', lambda *a: {'state': 'pending'})
    campaign.run_campaign(root, 'sr_crlike_run2', 'fixture_campaign',
                          plan_path=plan_path, action='execute', selected='unit_0',
                          backend_name='condor')
    campaign.run_campaign(root, 'sr_crlike_run2', 'fixture_campaign',
                          action='resume', backend_name='condor')
    assert submitted == [f'unit_{i}' for i in range(4)]


def test_read_only_status_presentation_and_receipt_boundary(tmp_path, monkeypatch, capsys):
    root, plan, attempt_root = seed_running(tmp_path, monkeypatch)
    assert campaign.main(['--status', '-o', str(root), '--format', 'json']) == 0
    pending = json.loads(capsys.readouterr().out)
    assert pending['units'][0]['execution'] == 'completed'
    assert pending['units'][0]['publish'] == 'publish pending'
    assert pending['published_count'] == 0
    before = campaign.state_path(root, 'sr_crlike_run2', 'run2').read_bytes()
    assert campaign.main(['--status', '-o', str(root)]) == 0
    human = capsys.readouterr().out
    assert 'fixture_campaign' in human and str(root) in human
    assert 'completed' in human and 'publish pending' in human
    assert '.campaign_units' not in human
    assert campaign.state_path(root, 'sr_crlike_run2', 'run2').read_bytes() == before
    assert campaign.main(['--resume', '-o', str(root), '--format', 'json']) == 0
    capsys.readouterr()
    assert campaign.main(['--finalize', '-o', str(root), '--format', 'json']) == 0
    capsys.readouterr()
    assert campaign.main(['--status', '-o', str(root), '--format', 'yaml']) == 0
    import yaml
    published = yaml.safe_load(capsys.readouterr().out)
    assert published['published_count'] == 1
    assert published['units'][0]['publish'] == 'published'
    assert all(path.startswith(str(root) + '/') for path in published['units'][0]['canonical_paths'])
    assert campaign.main(['--status', '-o', str(root), '--format', 'human']) == 0
    assert published['units'][0]['canonical_paths'][0] in capsys.readouterr().out
    receipt = campaign.completion_receipt_path(attempt_root)
    receipt.write_text('{}')
    assert campaign.main(['--status', '-o', str(root), '--format', 'json']) == 0
    invalid = json.loads(capsys.readouterr().out)
    assert invalid['published_count'] == 0
    assert invalid['units'][0]['publish'] != 'published'


@pytest.mark.parametrize('ad,state,attention', [
    ({'JobStatus': 2}, 'pending', False),
    ({'JobStatus': 4, 'ExitCode': 0, 'ExitBySignal': False}, 'terminal_success', False),
    (None, 'unresolved', True),
    ({'JobStatus': 5, 'HoldReason': 'fixture held'}, 'pending', True),
    ({'JobStatus': 4, 'ExitCode': 7}, 'terminal_failure', True),
    ({'JobStatus': 4, 'ExitBySignal': True, 'ExitSignal': 9}, 'terminal_failure', True),
])
def test_scheduler_status_is_observation_not_publication(ad, state, attention):
    observation = {'state': state, 'ad': ad, 'source': 'history'}
    result = campaign.scheduler_status(observation)
    assert result['attention'] is attention
    assert result['raw_classad'] == ad
    assert result['state'] == ('UNKNOWN' if ad is None else {
        2: 'RUNNING', 4: 'COMPLETED', 5: 'HELD'}[ad['JobStatus']])
    assert 'publication' not in result and 'dispatch' not in result


@pytest.mark.parametrize('output_format', ['human', 'json', 'yaml'])
def test_monitor_command_uses_active_environment_and_relative_launcher(
        tmp_path, monkeypatch, capsys, output_format):
    import shlex

    root = tmp_path / "campaign output 'quoted' $literal"
    monkeypatch.setattr(campaign.sys, 'executable', '/foreign/env/bin/python')
    monkeypatch.setattr(campaign, 'RUN_DIRECTORY', Path('/foreign/checkout/analysis/topeft_run2'))
    monkeypatch.setattr(campaign, 'REPO_ROOT', Path('/foreign/checkout'))
    monkeypatch.setenv('PYTHON_BIN', '/foreign/override/bin/python')
    monkeypatch.setenv('CONDA_PREFIX', '/foreign/env')
    campaign.show_monitor_command(root, output_format)
    captured = capsys.readouterr()
    handoff = captured.out if output_format == 'human' else captured.err
    assert (captured.err if output_format == 'human' else captured.out) == ''
    assert 'From the TOPEFT checkout with the intended Conda environment already active:' in handoff
    assert 'PYTHON_BIN' not in handoff and '/foreign/' not in handoff
    assert shlex.split(handoff.split('Monitor with:\n  ')[1].strip()) == [
        './analysis/topeft_run2/run_plotter.sh', '--status', '-o', str(root.resolve()),
        '--observe-scheduler']
    assert not root.exists()


@pytest.mark.parametrize('memory,rss,expected', [
    (0, 75000000, '0'), (123, 75000000, '123'), (12.5, None, '12.5'),
    ('/Expr(((ResidentSetSize + 1023) / 1024))/', 75000000, '73243'),
    ('expression', 0, '0'), (None, 1023, '1'), (None, 1024, '1'),
    (None, 1025, '2'), (None, 0.5, '1'), (False, 1025, '2'),
    (float('inf'), 1025, '2'), (float('nan'), 1025, '2'), (-1, 1025, '2'),
    (None, None, '-'), ('expression', 'expression', '-'),
    ('123', None, '-'), (None, False, '-'), (None, -1, '-'),
    (None, float('inf'), '-'), (None, float('nan'), '-'),
])
def test_human_scheduler_memory_is_numeric_without_changing_machine_values(
        tmp_path, monkeypatch, capsys, memory, rss, expected):
    ad = {'JobStatus': 2, 'MemoryUsage': memory, 'ResidentSetSize': rss}
    scheduler = campaign.scheduler_status({'state': 'pending', 'ad': ad})
    plan = {'campaign_profile': 'sr_crlike_run2', 'campaign_tag': 'memory_fixture',
            'plan_revision': 1, 'units': [{'unit_id': 'unit_0', 'component': 'run2'}]}
    states = {'run2': {'units': {'unit_0': {'status': 'planned', 'attempts': []}}}}
    summary = campaign.campaign_summary(tmp_path, plan, states)
    summary['units'][0]['scheduler'] = scheduler
    before = json.dumps(summary, sort_keys=True)
    monkeypatch.setattr(campaign, 'eval', lambda *_: pytest.fail('expression evaluated'), raising=False)
    campaign.show_summary(summary, 'human')
    output = capsys.readouterr().out
    assert f'MEM_MB={expected} RSS_KIB=' in output
    assert json.dumps(summary, sort_keys=True) == before
    assert scheduler['mem_mb'] is memory and scheduler['rss_kib'] is rss
    assert scheduler['raw_classad'] is ad
    campaign.show_summary(summary, 'json')
    assert capsys.readouterr().out.strip() == before


def test_scheduler_status_opt_in_is_read_only_and_shared(tmp_path, monkeypatch, capsys):
    from analysis.topeft_run2 import plot_campaign_condor as condor
    from types import SimpleNamespace
    import pwd
    import os
    import yaml
    root, plan_path = fixture_plan(tmp_path, monkeypatch)
    def fake_execute(argv, attempt_root, contract):
        descriptor = campaign.backend.execution_descriptor(argv, attempt_root, contract)
        metadata = attempt_root / '_metadata'
        metadata.mkdir(exist_ok=True)
        (metadata / 'execution.json').write_text(json.dumps(descriptor))
        intent = dict(token=descriptor['token'], pool='fixture.invalid', schedd='fixture.invalid')
        (metadata / 'submission_intent.json').write_text(json.dumps(intent))
        (metadata / 'scheduler_identity.json').write_text(json.dumps({**intent, 'job': '123.0'}))
        raise campaign.ActiveChildUnresolved('fixture submitted')
    monkeypatch.setattr(campaign.backend, 'execute', fake_execute)
    calls = []
    def fake(command, **kwargs):
        calls.append(command[0])
        assert command[0] in ('condor_q', 'condor_history')
        descriptor_path = next(root.rglob('execution.json'))
        token = json.loads(descriptor_path.read_text())['token']
        ad = dict(ClusterId=123, ProcId=0, Owner=pwd.getpwuid(os.getuid()).pw_name,
                  PlotExecutionToken=token, JobStatus=5, HoldReason='fixture held',
                  MemoryUsage=0, ResidentSetSize=123456)
        return SimpleNamespace(returncode=0, stdout=json.dumps([ad]), stderr='')
    monkeypatch.setattr(condor, '_run', fake)
    campaign.run_campaign(root, 'sr_crlike_run2', 'fixture_campaign', plan_path=plan_path,
                          action='execute', backend_name='condor', emit=False)
    assert calls == ['condor_q']
    calls.clear()
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert campaign.main(['--status', '-o', str(root), '--format', 'json']) == 0
    assert json.loads(capsys.readouterr().out)['units'][0]['scheduler'] is None
    assert calls == []
    for output_format in ('json', 'yaml', 'human'):
        assert campaign.main(['--status', '-o', str(root), '--observe-scheduler',
                              '--format', output_format]) == 0
        output = capsys.readouterr().out
        if output_format == 'human':
            assert 'HELD' in output and 'fixture held' in output
            assert 'MEM_MB=0 RSS_KIB=123456' in output
        else:
            summary = yaml.safe_load(output)
            assert summary['published_count'] == 0
            assert summary['units'][0]['execution'] == 'running'
            scheduler = summary['units'][0]['scheduler']
            assert scheduler['attention'] and scheduler['mem_mb'] == 0
            assert scheduler['rss_kib'] == 123456
    assert calls == ['condor_q'] * 3
    def mismatched(command, **kwargs):
        result = fake(command, **kwargs)
        ads = json.loads(result.stdout)
        ads[0]['PlotExecutionToken'] = 'wrong'
        result.stdout = json.dumps(ads)
        return result
    monkeypatch.setattr(condor, '_run', mismatched)
    assert campaign.main(['--status', '-o', str(root), '--observe-scheduler',
                          '--format', 'json']) == 0
    unresolved = json.loads(capsys.readouterr().out)['units'][0]['scheduler']
    assert unresolved['state'] == 'UNKNOWN' and unresolved['attention']
    assert unresolved['raw_classad'] is None
    assert 'identity/token/owner contradiction' in unresolved['reason']
    assert {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()} == before


def seed_condor_status_units(tmp_path, monkeypatch, *, unit_count=2):
    root, plan_path = fixture_plan(tmp_path, monkeypatch,
                                   profile="sr_crlike_run2_run3", unit_count=unit_count)

    def fake_execute(argv, attempt_root, contract):
        descriptor = campaign.backend.execution_descriptor(argv, attempt_root, contract)
        metadata = attempt_root / "_metadata"
        metadata.mkdir(exist_ok=True)
        intent = dict(token=descriptor["token"], pool="fixture.invalid", schedd="fixture.invalid")
        (metadata / "execution.json").write_text(json.dumps(descriptor))
        (metadata / "submission_intent.json").write_text(json.dumps(intent))
        job = f"{123 + int(attempt_root.parent.name.split('_')[-1])}.0"
        (metadata / "scheduler_identity.json").write_text(json.dumps({**intent, "job": job}))
        raise campaign.ActiveChildUnresolved("fixture submitted")

    monkeypatch.setattr(campaign.backend, "execute", fake_execute)
    monkeypatch.setattr(campaign.backend, "observe", lambda *a: {"state": "pending"})
    campaign.run_campaign(root, "sr_crlike_run2_run3", "fixture_campaign",
                          plan_path=plan_path, action="execute", backend_name="condor",
                          emit=False)
    return root


@pytest.mark.parametrize('statuses,expected', [
    ([4] * 9 + [2], {'COMPLETED': 9, 'RUNNING': 1, 'IDLE': 0, 'HELD': 0, 'OTHER/UNKNOWN': 0}),
    ([1, 2, 4, 5, None, 3, 6, 7, 99, 2],
     {'COMPLETED': 1, 'RUNNING': 2, 'IDLE': 1, 'HELD': 1, 'OTHER/UNKNOWN': 5}),
])
def test_scheduler_aggregate_matches_rows_and_preserves_persisted_counts(
        tmp_path, monkeypatch, capsys, statuses, expected):
    from collections import Counter
    from analysis.topeft_run2 import plot_campaign_condor as condor
    import yaml

    root = seed_condor_status_units(tmp_path, monkeypatch, unit_count=len(statuses))
    before = {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
              for p in root.rglob('*')}
    calls = []

    def unexpected(*args, **kwargs):
        pytest.fail('status attempted mutation or direct scheduler access')

    def observe(argv, attempt_root, contract, *, read_only):
        assert read_only is True
        unit_id = attempt_root.parent.name
        calls.append(unit_id)
        status = statuses[int(unit_id.split('_')[-1])]
        if status is None:
            raise campaign.CampaignError('synthetic scheduler unavailable')
        return {'state': 'terminal_success' if status == 4 else 'pending',
                'source': 'synthetic', 'ad': {'JobStatus': status,
                    'MemoryUsage': '/Expr(((ResidentSetSize + 1023) / 1024))/',
                    'ResidentSetSize': 75000000}}

    for module, name in [(campaign, 'atomic_json'), (campaign, 'reconcile_running'),
                         (campaign, 'finalize_campaign'), (campaign.backend, 'execute'),
                         (campaign.backend, 'atomic_json'), (condor, '_persist_identity'),
                         (condor, '_run'), (condor, 'submit')]:
        monkeypatch.setattr(module, name, unexpected)
    monkeypatch.setattr(campaign.backend, 'observe', unexpected)
    assert campaign.main(['--status', '-o', str(root), '--format', 'json']) == 0
    offline = json.loads(capsys.readouterr().out)
    assert all(unit['scheduler'] is None for unit in offline['units'])
    assert calls == []
    monkeypatch.setattr(campaign.backend, 'observe', observe)
    for output_format in ('json', 'yaml', 'human'):
        assert campaign.main(['--status', '-o', str(root), '--observe-scheduler',
                              '--format', output_format]) == 0
        output = capsys.readouterr().out
        if output_format == 'human':
            assert (f'Campaign state (persisted): 0 published / {len(statuses)} planned'
                    f' | Running: {len(statuses)} | Pending: 0 | Failed: 0 | Interrupted: 0') in output
            assert 'CAMPAIGN VIEW' in output and 'EXECUTION' not in output
            aggregate = next(line for line in output.splitlines()
                             if line.startswith('Scheduler observed: '))
            shown = dict(part.rsplit(' ', 1) for part in
                         aggregate.removeprefix('Scheduler observed: ').split(' | '))
            assert {key: int(value) for key, value in shown.items()} == expected
            rows = Counter(line.strip().split()[1] for line in output.splitlines()
                           if line.startswith('  Scheduler: '))
            assert sum(rows.values()) == len(statuses) == sum(expected.values())
            for label in ('COMPLETED', 'RUNNING', 'IDLE', 'HELD'):
                assert rows[label] == expected[label]
            assert sum(count for label, count in rows.items()
                       if label not in {'COMPLETED', 'RUNNING', 'IDLE', 'HELD'}) == expected['OTHER/UNKNOWN']
            assert 'MEM_MB=/Expr(' not in output
            assert 'MEM_MB=73243 RSS_KIB=75000000' in output
        else:
            summary = yaml.safe_load(output)
            assert summary['summary_schema_version'] == 2
            assert summary.keys() == offline.keys()
            assert {k: v for k, v in summary.items() if k.endswith('_count')} == {
                k: v for k, v in offline.items() if k.endswith('_count')}
            assert summary['planned_count'] == summary['running_count'] == len(statuses)
            assert summary['success_count'] == summary['published_count'] == 0
            assert summary['campaign_status'] == offline['campaign_status'] == 'in_progress'
            assert all(unit['execution'] == 'running' for unit in summary['units'])
    assert calls == [f'unit_{i}' for i in range(len(statuses))] * 3
    assert {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
            for p in root.rglob('*')} == before


@pytest.mark.parametrize("selected,expected_ids", [
    ("unit_0", ["unit_0"]), ("unit_1", ["unit_1"]), (None, ["unit_0", "unit_1"]),
])
@pytest.mark.parametrize("output_format", ["json", "yaml", "human"])
def test_status_selection_scopes_observer_and_projection_without_writes(
        tmp_path, monkeypatch, capsys, selected, expected_ids, output_format):
    from analysis.topeft_run2 import plot_campaign_condor as condor
    import yaml

    root = seed_condor_status_units(tmp_path, monkeypatch)
    before = {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
              for p in root.rglob("*")}
    calls = []

    def observe(argv, attempt_root, contract, *, read_only):
        descriptor = json.loads((attempt_root / "_metadata/execution.json").read_text())
        assert descriptor == campaign.backend.execution_descriptor(argv, attempt_root, contract)
        assert read_only is True
        unit_id = attempt_root.parent.name
        calls.append(unit_id)
        job = "123.0" if unit_id == "unit_0" else "124.0"
        return {"state": "pending", "job": job, "source": "queue",
                "ad": {"JobStatus": 2, "PlotExecutionToken": descriptor["token"]}}

    def unexpected(*args, **kwargs):
        pytest.fail("read-only status reached a mutation or direct scheduler call")

    monkeypatch.setattr(campaign.backend, "observe", observe)
    for module, name in [(campaign, "atomic_json"), (campaign, "reconcile_running"),
                         (campaign, "finalize_campaign"), (campaign.backend, "execute"),
                         (campaign.backend, "atomic_json"), (condor, "_persist_identity"),
                         (condor, "_run"), (campaign, "sha256_canonical_file")]:
        monkeypatch.setattr(module, name, unexpected)
    args = ["--status", "-o", str(root), "--observe-scheduler", "--format", output_format]
    if selected is not None:
        args += ["--selected-unit", selected]
    assert campaign.main(args) == 0
    output = capsys.readouterr().out
    assert calls == expected_ids
    if output_format == "human":
        assert f"Campaign state (persisted): 0 published / {len(expected_ids)} planned" in output
        assert f"Running: {len(expected_ids)} | Pending: 0" in output
        assert "fixture_campaign" in output and str(root) in output
        for unit_id in ["unit_0", "unit_1"]:
            assert (unit_id in output) == (unit_id in expected_ids)
    else:
        summary = yaml.safe_load(output)
        assert [unit["unit_id"] for unit in summary["units"]] == expected_ids
        assert [unit["job"] for unit in summary["units"]] == [
            "123.0" if unit_id == "unit_0" else "124.0" for unit_id in expected_ids]
        assert all(unit["scheduler"]["state"] == "RUNNING" for unit in summary["units"])
        assert summary["planned_count"] == summary["running_count"] == len(expected_ids)
        assert summary["pending_count"] == summary["published_count"] == 0
        assert summary["campaign_status"] == "in_progress"
        assert summary["campaign_profile"] == "sr_crlike_run2_run3"
        assert summary["campaign_tag"] == "fixture_campaign" and summary["plan_revision"] == 1
        assert summary["canonical_output_root"] == str(root)
        assert summary["execution_policy"]["backend"] == "condor"
        assert len(summary["active_child_reasons"]) == len(expected_ids)
        if selected is not None:
            assert summary["action_scope"]["selected_unit_ids"] == expected_ids
        else:
            assert summary["action_scope"] is None  # Existing no-selector schema.
    calls.clear()
    assert campaign.main(["--status", "-o", str(root), "--selected-unit", "unknown",
                          "--observe-scheduler", "--verify-published"]) == 2
    assert "unknown selected unit" in capsys.readouterr().err
    assert calls == []
    assert {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
            for p in root.rglob("*")} == before


@pytest.mark.parametrize("selected,expected_ids", [
    ("unit_0", ["unit_0"]), ("unit_1", ["unit_1"]), (None, ["unit_0", "unit_1"]),
])
@pytest.mark.parametrize("verify", [False, True])
def test_status_selection_scopes_canonical_hashes(
        tmp_path, monkeypatch, capsys, selected, expected_ids, verify):
    root, plan_path = fixture_plan(tmp_path, monkeypatch,
                                   profile="sr_crlike_run2_run3", unit_count=2)
    monkeypatch.setattr(campaign.backend, "run_guarded_child", fake_success)
    campaign.run_campaign(root, "sr_crlike_run2_run3", "fixture_campaign",
                          plan_path=plan_path, action="execute", emit=False)
    finalize_ready(root, "sr_crlike_run2_run3")
    before = {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
              for p in root.rglob("*")}
    original_hash = campaign.sha256_canonical_file
    calls = []

    def hash_content(path, canonical_root):
        calls.append(path)
        return original_hash(path, canonical_root)

    monkeypatch.setattr(campaign, "sha256_canonical_file", hash_content)
    monkeypatch.setattr(campaign.backend, "observe", lambda *a, **kw: pytest.fail("observer"))
    monkeypatch.setattr(campaign, "atomic_json", lambda *a, **kw: pytest.fail("state write"))
    args = ["--status", "-o", str(root), "--format", "json"]
    if selected is not None:
        args += ["--selected-unit", selected]
    if verify:
        args += ["--verify-published"]
    assert campaign.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert [unit["unit_id"] for unit in summary["units"]] == expected_ids
    assert summary["campaign_status"] == "complete"
    assert summary["planned_count"] == summary["success_count"] == len(expected_ids)
    assert summary["published_count"] == len(expected_ids)
    assert summary["content_verified_count"] == (len(expected_ids) if verify else 0)
    expected_paths = [Path(path) for unit in summary["units"] for path in unit["canonical_paths"]]
    assert len(expected_paths) == len(expected_ids)
    assert calls == (expected_paths if verify else [])
    assert [path.parent.name for path in expected_paths] == expected_ids
    calls.clear()
    assert campaign.main(["--status", "-o", str(root), "--selected-unit", "unknown",
                          "--observe-scheduler", "--verify-published"]) == 2
    assert "unknown selected unit" in capsys.readouterr().err
    assert calls == []
    assert {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
            for p in root.rglob("*")} == before


def test_selected_status_rejects_nonunique_unit_before_deep_checks(tmp_path, monkeypatch):
    root, plan, _ = seed_running(tmp_path, monkeypatch)
    states = {"run2": campaign.read_json(campaign.state_path(root, "sr_crlike_run2", "run2"))}
    plan["units"].append(dict(plan["units"][0]))
    monkeypatch.setattr(campaign.backend, "observe", lambda *a, **kw: pytest.fail("observer"))
    monkeypatch.setattr(campaign, "sha256_canonical_file", lambda *a: pytest.fail("hash"))
    with pytest.raises(campaign.CampaignError, match="selected unit is not unique"):
        campaign.campaign_summary(root, plan, states, selected="unit_0",
                                   observe_scheduler=True, verify_published=True)


def test_follow_reports_changes_only_and_interrupt_is_nonmutating(tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    submitted = []
    def fake_execute(argv, attempt_root, contract):
        submitted.append(attempt_root.parent.name)
        raise campaign.ActiveChildUnresolved(
            'Condor execution submitted; resume observes this execution only')
    monkeypatch.setattr(campaign.backend, 'execute', fake_execute)
    observations = {'unit_0': 0, 'unit_1': 0}
    def fake_observe(argv, attempt_root, contract):
        unit_id = attempt_root.parent.name
        observations[unit_id] += 1
        if observations[unit_id] >= (2 if unit_id == 'unit_0' else 3):
            fake_success(argv, attempt_root)
            return {'state': 'terminal_success'}
        return {'state': 'pending'}
    monkeypatch.setattr(campaign.backend, 'observe', fake_observe)
    monkeypatch.setattr(campaign.time, 'sleep', lambda _: None)
    summary = campaign.run_campaign(root, 'sr_crlike_run2', 'fixture_campaign',
                                    plan_path=plan_path, action='execute', backend_name='condor',
                                    emit=False)
    assert summary['running_count'] == 2
    final = campaign.follow_campaign(summary, root, 'sr_crlike_run2', 'fixture_campaign',
                                     None, 'condor', 'cgroup_v2', 245000000000, 'human', 1)
    assert final['published_count'] == 0
    assert final['interrupted_count'] == 2
    assert submitted == ['unit_0', 'unit_1']
    # Routine pending observations do not repeat the unchanged human summary.
    assert capsys.readouterr().out.count('Campaign:') == 2
    state_path = campaign.state_path(root, 'sr_crlike_run2', 'run2')
    before = state_path.read_bytes()
    monkeypatch.setattr(campaign.time, 'sleep', lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        campaign.follow_campaign(summary, root, 'sr_crlike_run2', 'fixture_campaign',
                                 None, 'condor', 'cgroup_v2', 245000000000, 'human', 1)
    assert state_path.read_bytes() == before


def seed_rejected_condor_campaign(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from analysis.topeft_run2 import plot_campaign_condor as condor
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=1)
    monkeypatch.setattr(condor, 'resolve_target', lambda: {
        'pool': 'collector.fixture.invalid', 'schedd': 'submit.fixture.invalid'})
    calls = []
    submit_root = condor.backend.REPO_ROOT
    rejected = f'ERROR: No such directory: {submit_root}/"{submit_root}"\n'
    def fake(command, **kwargs):
        calls.append(command[0])
        if command[0] == 'condor_submit':
            return SimpleNamespace(returncode=1, stdout='', stderr=rejected)
        return SimpleNamespace(returncode=0, stdout='[]', stderr='')
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    campaign.run_campaign(root, 'sr_crlike_run2', 'fixture_campaign', plan_path=plan_path,
                          action='execute', backend_name='condor')
    state_path = campaign.state_path(root, 'sr_crlike_run2', 'run2')
    state = campaign.read_json(state_path)
    attempt = state['units']['unit_0']['attempts'][0]
    metadata = Path(attempt['private_unit_root']) / '_metadata'
    original = (metadata / 'execution.sub').read_text()
    for key in ('executable', 'initialdir', 'output', 'error', 'log'):
        line = next(line for line in original.splitlines() if line.startswith(key + ' = '))
        value = line.split(' = ', 1)[1]
        original = original.replace(line + '\n', f'{key} = "{value}"\n')
    (metadata / 'execution.sub').write_text(original)
    return root, state_path, metadata, calls


def test_explicit_condor_recovery_selected_same_attempt_only(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from analysis.topeft_run2 import plot_campaign_condor as condor
    root, state_path, metadata, calls = seed_rejected_condor_campaign(tmp_path, monkeypatch)
    assert calls.count('condor_submit') == 1
    args = ['--campaign-profile', 'sr_crlike_run2', '--campaign-tag', 'fixture_campaign',
            '-o', str(root), '--resume', '--selected-unit', 'unit_0']
    def recovered(command, **kwargs):
        calls.append(command[0])
        return SimpleNamespace(returncode=0,
                               stdout='123.0\n' if command[0] == 'condor_submit' else '[]',
                               stderr='')
    monkeypatch.setattr(condor.subprocess, 'run', recovered)
    assert campaign.main(args) == 0
    state = campaign.read_json(state_path)['units']['unit_0']
    assert state['status'] == 'running'
    assert [item['attempt_id'] for item in state['attempts']] == ['attempt_0001']
    assert (metadata / 'rejected_execution.sub').read_text() == (metadata / 'execution.sub').read_text()
    assert (metadata / 'rejected_submission_response.json').read_bytes() == (metadata / 'submission_response.json').read_bytes()
    assert calls.count('condor_submit') == 2
    assert campaign.main(args) == 0
    assert calls.count('condor_submit') == 2
    assert not (metadata.parent.parent / 'attempt_0002').exists()
    capsys.readouterr()


@pytest.mark.parametrize('blocker', ['worker_started.json', 'renderer.log', 'worker.out',
                                    'scheduler_identity.json', 'state'])
def test_explicit_condor_recovery_blocks_execution_or_state_evidence(
        tmp_path, monkeypatch, blocker):
    from analysis.topeft_run2 import plot_campaign_condor as condor
    root, state_path, metadata, calls = seed_rejected_condor_campaign(tmp_path, monkeypatch)
    if blocker == 'state':
        state = campaign.read_json(state_path)
        state['units']['unit_0']['attempts'][0]['failure_classification'] = 'other'
        campaign.atomic_json(state_path, state)
    else:
        (metadata / blocker).write_text('{}')
    monkeypatch.setattr(condor.subprocess, 'run',
                        lambda *a, **k: pytest.fail('scheduler or submit after blocker'))
    with pytest.raises(campaign.CampaignError):
        campaign.recover_rejected_condor_unit(root, 'sr_crlike_run2', 'fixture_campaign', 'unit_0')
    assert calls.count('condor_submit') == 1


@pytest.mark.parametrize('scheduler_state', ['terminal_success', 'terminal_failure', 'unresolved'])
def test_condor_terminal_evidence_needs_controller_receipt_and_materialization(tmp_path, monkeypatch, scheduler_state):
    root,plan,attempt_root=seed_running(tmp_path,monkeypatch)
    path=campaign.state_path(root,'sr_crlike_run2','run2')
    state=campaign.read_json(path); attempt=state['units']['unit_0']['attempts'][0]
    attempt['execution_contract']['backend']='condor'; campaign.atomic_json(path,state)
    monkeypatch.setattr(campaign.backend,'observe',lambda *a: {'state':scheduler_state})
    monkeypatch.setattr(campaign.backend,'execute',lambda *a: pytest.fail('retry'))
    campaign.run_campaign(root,'sr_crlike_run2','fixture_campaign',action='resume')
    result=campaign.read_json(path)['units']['unit_0']
    assert result['status']=={'terminal_success':'interrupted','terminal_failure':'failed','unresolved':'running'}[scheduler_state]
    assert len(result['attempts'])==1
    if scheduler_state=='terminal_success':
        assert not campaign.completion_receipt_path(attempt_root).exists()
        finalize_ready(root)
        assert (root/'run2/merged/unit_0/unit_0_invmass_fitting.png').read_bytes()==b'fixture-png'
        assert campaign.read_json(path)['units']['unit_0']['attempts'][0]['materialization_status']=='complete'


def test_historical_command_identity_is_verification_only(tmp_path, monkeypatch):
    root,plan,attempt_root=seed_missing_lifecycle(tmp_path,monkeypatch)
    path=campaign.state_path(root,'sr_crlike_run2_run3','run2')
    state=campaign.read_json(path); attempt=state['units']['unit_0']['attempts'][0]
    del attempt['execution_contract']
    historical=campaign.child_argv(plan['units'][0],attempt_root,historical=True)
    attempt['command_identity']=campaign.identity(historical)
    campaign.atomic_json(path,state)
    assert campaign.attempt_argv(plan['units'][0],attempt)==historical
    assert historical[0].endswith('run_plotter.sh')
    monkeypatch.setattr(campaign.backend,'execute',lambda *a: pytest.fail('historical command executed'))
    assert campaign.main(recovery_args(root))==0
    result=campaign.read_json(path)['units']['unit_0']
    assert result['status']=='interrupted'
    finalize_ready(root, profile='sr_crlike_run2_run3', selected='unit_0')
    assert result['attempts'][0]['command_identity']==campaign.identity(historical)
    assert 'execution_contract' not in result['attempts'][0]


@pytest.mark.parametrize('outcomes', [
    ('idle', 'running'), ('success', 'running'), ('success', 'success'),
    ('failure', 'running'), ('failure', 'failure'), ('unknown', 'unknown'),
    ('held_failure', 'held_failure'), ('signal', 'running'),
    ('success_without_receipt', 'running'), ('invalid_receipt', 'unknown'),
])
@pytest.mark.parametrize('output_format', ['human', 'json', 'yaml'])
def test_condor_execute_one_shot_handoff_and_terminal_states(
        tmp_path, monkeypatch, capsys, outcomes, output_format):
    import shlex
    import yaml

    profile = 'sr_crlike_run2_run3'
    root, plan_path = fixture_plan(tmp_path, monkeypatch, profile=profile, unit_count=2)
    # Exercise shell quoting without executing the printed scheduler command.
    root = root.with_name("campaign output 'quoted' $literal")
    submitted, observed, reconciled = [], [], []
    original_reconcile = campaign.reconcile_running

    def record_reconcile(*args):
        reconciled.append(args[3]['binding']['unit_id'])
        return original_reconcile(*args)

    def fake_execute(argv, attempt_root, contract):
        submitted.append((attempt_root.parent.name, attempt_root.name))
        raise campaign.ActiveChildUnresolved(
            'Condor execution submitted; resume observes this execution only')

    def fake_observe(argv, attempt_root, contract):
        assert len(submitted) == 2  # Both submissions precede the initial readback.
        unit_id = attempt_root.parent.name
        observed.append(unit_id)
        outcome = outcomes[int(unit_id[-1])]
        if outcome == 'success':
            fake_success(argv, attempt_root)
            return {'state': 'terminal_success'}
        if outcome in {'failure', 'signal'}:
            # Backend terminality is sufficient for failure, even without guard files.
            return {'state': 'terminal_failure', 'ad': {
                'JobStatus': 4, 'ExitCode': 1, 'ExitBySignal': outcome == 'signal'}}
        if outcome in {'held_failure', 'invalid_receipt'}:
            metadata = attempt_root / '_metadata'
            metadata.mkdir()
            (metadata / 'guard.jsonl').write_text(guard_result_line(1))
            write_lifecycle(argv, attempt_root, 1, 1, 0)
            if outcome == 'invalid_receipt':
                receipt_path = campaign.lifecycle_receipt_path(attempt_root)
                receipt = campaign.read_json(receipt_path)
                receipt['command_identity'] = 'wrong'
                campaign.atomic_json(receipt_path, receipt)
            return {'state': 'pending', 'ad': {
                'JobStatus': 5, 'ExitCode': 1, 'ExitBySignal': False,
                'NumJobStarts': 1, 'HoldReason': 'OnExitHold expression is true'}}
        if outcome == 'success_without_receipt':
            return {'state': 'terminal_success'}
        return {'state': 'unresolved' if outcome == 'unknown' else 'pending',
                'ad': {'JobStatus': 1 if outcome == 'idle' else 2}}

    monkeypatch.setattr(campaign, 'reconcile_running', record_reconcile)
    monkeypatch.setattr(campaign.backend, 'execute', fake_execute)
    monkeypatch.setattr(campaign.backend, 'observe', fake_observe)
    monkeypatch.setattr(campaign.time, 'sleep', lambda *_: pytest.fail('implicit polling'))
    monkeypatch.setattr(campaign, 'follow_campaign', lambda *_: pytest.fail('implicit follow'))
    assert campaign.main([
        '--campaign-profile', profile, '--campaign-tag', 'fixture_campaign',
        '--plan', str(plan_path), '-o', str(root), '--backend', 'condor',
        '--execute', '--format', output_format]) == 0
    captured = capsys.readouterr()
    assert submitted == [('unit_0', 'attempt_0001'), ('unit_1', 'attempt_0001')]
    assert observed == reconciled == ['unit_0', 'unit_1']
    handoff = captured.out if output_format == 'human' else captured.err
    assert handoff.count('Monitor with:') == 1
    command = handoff.split('Monitor with:\n  ')[1].strip()
    assert shlex.split(command) == [
        './analysis/topeft_run2/run_plotter.sh',
        '--status', '-o', str(root.resolve()), '--observe-scheduler']
    if output_format == 'human':
        assert 'Campaign state (persisted):' in captured.out and 'Running:' in captured.out
        assert 'Failed:' in captured.out and 'Pending:' in captured.out
    else:
        summary = yaml.safe_load(captured.out)
        assert summary['summary_schema_version'] == campaign.SUMMARY_SCHEMA_VERSION
        assert summary['planned_count'] == 2
        assert 'Monitor with:' not in captured.out

    terminal = {'failure', 'signal', 'held_failure', 'success'}
    _, plan, states = campaign.load_campaign(root, profile, 'fixture_campaign',
                                            campaign.runtime_identity())
    final_summary = campaign.campaign_summary(root, plan, states)
    assert final_summary['published_count'] == 0
    assert final_summary['failed_count'] == sum(outcome in terminal - {'success'} for outcome in outcomes)
    assert final_summary['running_count'] == sum(
        outcome not in terminal | {'success_without_receipt'} for outcome in outcomes)
    assert final_summary['active_child_present'] == bool(final_summary['running_count'])
    for index, outcome in enumerate(outcomes):
        component = 'run2' if index == 0 else 'run3'
        unit = campaign.read_json(campaign.state_path(root, profile, component))['units'][f'unit_{index}']
        expected = ('failed' if outcome in terminal - {'success'}
                    else 'interrupted' if outcome in {'success', 'success_without_receipt'}
                    else 'running')
        assert unit['status'] == expected
        assert len(unit['attempts']) == 1
        attempt = unit['attempts'][0]
        if outcome == 'held_failure':
            assert attempt['child_result'] == 1
            assert attempt['failure_classification'] == 'terminal guard/child/tee failure'
        if outcome == 'success_without_receipt':
            assert attempt['failure_classification'] == campaign.terminal_postprocessing_pending
            assert final_summary['units'][index]['publish'] == 'publish pending'
        if outcome == 'success':
            assert attempt['materialization_status'] == 'pending'
            assert attempt['completion_receipt_identity'] is None
            assert final_summary['units'][index]['publish'] == 'publish pending'
        else:
            assert not unit['materializations']
            assert attempt['completion_receipt_identity'] is None
        if expected == 'running':
            if output_format == 'human':
                assert 'terminal guard evidence unavailable' in captured.out
            else:
                expected_warning = ('terminal evidence unavailable' if outcome == 'invalid_receipt'
                                    else 'terminal guard evidence unavailable')
                assert expected_warning in summary['units'][index]['warning']

    # An ordinary resume does not follow or retry terminal units. Unknown jobs
    # remain explicit; no synthetic output is written on this second observation.
    monkeypatch.setattr(campaign.backend, 'execute', lambda *_: pytest.fail('retry'))
    monkeypatch.setattr(campaign.backend, 'observe', lambda *a: {'state': 'unresolved'})
    assert campaign.main(['--resume', '-o', str(root), '--format', output_format]) == 0
    capsys.readouterr()
    for index, component in enumerate(('run2', 'run3')):
        unit = campaign.read_json(campaign.state_path(root, profile, component))['units'][f'unit_{index}']
        assert len(unit['attempts']) == 1
        assert unit['attempts'][0]['attempt_id'] == 'attempt_0001'
        if outcomes[index] in terminal:
            assert unit['status'] == ('interrupted' if outcomes[index] == 'success' else 'failed')
        elif outcomes[index] == 'success_without_receipt':
            assert unit['status'] == 'interrupted'
    assert not list(root.rglob('attempt_0002'))


@pytest.mark.parametrize('scheduler_failure', [False, True])
def test_condor_resume_closes_held_failure_without_retry(tmp_path, monkeypatch, capsys, scheduler_failure):
    root, _, attempt_root = seed_running(tmp_path, monkeypatch, terminal='failure')
    path = campaign.state_path(root, 'sr_crlike_run2', 'run2')
    state = campaign.read_json(path)
    state['units']['unit_0']['attempts'][0]['execution_contract']['backend'] = 'condor'
    campaign.atomic_json(path, state)

    def observe(*args):
        if scheduler_failure:
            raise campaign.CampaignError('synthetic scheduler unavailable')
        return {'state': 'pending', 'ad': {'JobStatus': 5, 'ExitCode': 7}}

    monkeypatch.setattr(campaign.backend, 'observe', observe)
    monkeypatch.setattr(campaign.backend, 'execute', lambda *_: pytest.fail('retry'))
    monkeypatch.setattr(campaign.time, 'sleep', lambda *_: pytest.fail('implicit polling'))
    assert campaign.main(['--resume', '-o', str(root)]) == 0
    unit = campaign.read_json(path)['units']['unit_0']
    assert unit['status'] == unit['attempts'][0]['result'] == 'failed'
    assert unit['attempts'][0]['child_result'] == 7
    assert len(unit['attempts']) == 1
    assert not (attempt_root.parent / 'attempt_0002').exists()
    assert 'reconciliation needed' not in capsys.readouterr().out


def seed_failed_condor_attempts(tmp_path, monkeypatch):
    profile = 'sr_crlike_run2_run3'
    root, plan_path = fixture_plan(tmp_path, monkeypatch, profile=profile, unit_count=2)
    raw = campaign.read_json(plan_path)
    for unit in raw['units']:
        sidecar_path = Path(unit['sidecar_path'])
        sidecar = campaign.read_json(sidecar_path)
        digest = campaign.sha256_file(Path(unit['input_path']))
        sidecar['artifact']['pkl_sha256'] = digest
        campaign.atomic_json(sidecar_path, sidecar)
        unit['qualification_source'].update(
            input_sha256=digest, sidecar_sha256=campaign.sha256_file(sidecar_path))
    campaign.atomic_json(plan_path, raw)

    def first_failure(argv, attempt_root, contract):
        result = fake_failure(argv, attempt_root)
        descriptor = campaign.backend.execution_descriptor(argv, attempt_root, contract)
        metadata = attempt_root / '_metadata'
        campaign.atomic_json(metadata / 'execution.json', descriptor)
        campaign.atomic_json(metadata / 'scheduler_identity.json', {
            'job': '123.0', 'pool': 'fixture_pool', 'schedd': 'fixture_schedd',
            'token': descriptor['token'],
        })
        return result

    monkeypatch.setattr(campaign.backend, 'execute', first_failure)
    monkeypatch.setattr(campaign.backend, 'observe',
                        lambda *a, **k: pytest.fail('unexpected scheduler query'))
    campaign.run_campaign(root, profile, 'fixture_campaign', action='execute',
                          plan_path=plan_path, backend_name='condor',
                          observer='cgroup_v2', hard_limit_bytes=192000000000, emit=False)
    return root, raw, profile


def retired_failed_observation(*args, **kwargs):
    return {'state': 'terminal_failure', 'job': '123.0', 'source': 'history',
            'ad': {'JobStatus': 3, 'NumJobStarts': 1, 'ExitCode': 7, 'ExitBySignal': False}}


def queued_recovery(monkeypatch):
    calls = []

    def submit(argv, attempt_root, contract):
        calls.append((argv, attempt_root, contract))
        metadata = attempt_root / '_metadata'
        metadata.mkdir()
        descriptor = campaign.backend.execution_descriptor(argv, attempt_root, contract)
        campaign.atomic_json(metadata / 'execution.json', descriptor)
        campaign.atomic_json(metadata / 'scheduler_identity.json', {
            'job': '124.0', 'pool': 'fixture_pool', 'schedd': 'fixture_schedd',
            'token': descriptor['token'],
        })
        raise campaign.ActiveChildUnresolved('synthetic ambiguous submit response')

    monkeypatch.setattr(campaign.backend, 'execute', submit)
    monkeypatch.setattr(campaign.backend, 'observe', retired_failed_observation)
    return calls


@pytest.mark.parametrize('selected,component', [('unit_0', 'run2'), ('unit_1', 'run3')])
@pytest.mark.parametrize('old_absent', [False, True])
def test_failed_recovery_is_selected_one_shot_and_preserves_first_attempt(
        tmp_path, monkeypatch, capsys, selected, component, old_absent):
    root, _, profile = seed_failed_condor_attempts(tmp_path, monkeypatch)
    state_path = campaign.state_path(root, profile, component)
    before = campaign.read_json(state_path)['units'][selected]['attempts'][0]
    old_root = Path(before['private_unit_root'])
    old_bytes = {str(p.relative_to(old_root)): p.read_bytes()
                 for p in old_root.rglob('*') if p.is_file()}
    coordinator_bytes = campaign.coordinator_path(root).read_bytes()
    calls = queued_recovery(monkeypatch)
    if old_absent:
        monkeypatch.setattr(campaign.backend, 'observe', lambda *a, **k: {
            'state': 'unresolved', 'job': '123.0', 'source': 'unresolved', 'ad': None})
    args = ['--resume', '-o', str(root), '--selected-unit', selected]
    assert campaign.main(args) == 0
    capsys.readouterr()
    state = campaign.read_json(state_path)['units'][selected]
    assert state['status'] == 'running'
    assert state['attempts'][0] == before
    assert [a['attempt_id'] for a in state['attempts']] == ['attempt_0001', 'attempt_0002']
    old_descriptor = campaign.read_json(old_root / '_metadata/execution.json')
    new_root = Path(state['attempts'][1]['private_unit_root'])
    new_descriptor = campaign.read_json(new_root / '_metadata/execution.json')
    assert old_descriptor['token'] != new_descriptor['token']
    assert coordinator_bytes == campaign.coordinator_path(root).read_bytes()
    assert old_bytes == {str(p.relative_to(old_root)): p.read_bytes()
                         for p in old_root.rglob('*') if p.is_file()}
    other = 'run3' if component == 'run2' else 'run2'
    assert len(next(iter(campaign.read_json(campaign.state_path(root, profile, other))
                         ['units'].values()))['attempts']) == 1
    monkeypatch.setattr(campaign.backend, 'observe', lambda *a, **k: {'state': 'pending'})
    assert campaign.main(args) == 0
    other_unit = 'unit_1' if selected == 'unit_0' else 'unit_0'
    assert campaign.main(['--resume', '-o', str(root),
                          '--selected-unit', other_unit]) == 0
    assert campaign.main(['--resume', '-o', str(root), '--selected-unit', selected]) == 0
    assert len(calls) == 1
    assert not list(root.rglob('attempt_0003'))


@pytest.mark.parametrize('blocker', [
    'missing_receipt', 'bad_guard_hash', 'occupied_second', 'input_bytes', 'sidecar_bytes',
    'held', 'running', 'restart', 'unknown', 'descriptor', 'command', 'success_lifecycle',
])
def test_failed_recovery_rejects_unbound_or_active_evidence_without_new_attempt(
        tmp_path, monkeypatch, capsys, blocker):
    root, _, profile = seed_failed_condor_attempts(tmp_path, monkeypatch)
    state_path = campaign.state_path(root, profile, 'run2')
    state = campaign.read_json(state_path)
    unit = state['units']['unit_0']
    attempt = unit['attempts'][0]
    old_root = Path(attempt['private_unit_root'])
    receipt_path = campaign.lifecycle_receipt_path(old_root)
    if blocker == 'missing_receipt':
        receipt_path.unlink()
    elif blocker == 'bad_guard_hash':
        receipt = campaign.read_json(receipt_path)
        receipt['guard_evidence_sha256'] = '0' * 64
        campaign.atomic_json(receipt_path, receipt)
    elif blocker == 'occupied_second':
        (old_root.parent / 'attempt_0002').mkdir()
    elif blocker == 'input_bytes':
        path = Path(unit['binding']['input_path'])
        path.write_bytes(b'x' * path.stat().st_size)
    elif blocker == 'sidecar_bytes':
        Path(unit['binding']['sidecar_path']).write_text('{}')
    elif blocker == 'descriptor':
        campaign.atomic_json(old_root / '_metadata/execution.json', {})
    elif blocker == 'command':
        attempt['command_identity'] = '0' * 64
        campaign.atomic_json(state_path, state)
    elif blocker == 'success_lifecycle':
        (old_root / '_metadata/guard.jsonl').write_text(guard_result_line())
        write_lifecycle(attempt['execution_contract']['argv'], old_root, 0, 0, 0)
        attempt.update(guard_result=0, child_result=0, tee_result=0)
        campaign.atomic_json(state_path, state)
    observation = retired_failed_observation()
    if blocker in {'held', 'running'}:
        observation['state'] = 'pending'
        observation['ad']['JobStatus'] = 5 if blocker == 'held' else 2
    elif blocker == 'restart':
        observation['ad']['NumJobStarts'] = 2
    elif blocker == 'unknown':
        observation = {'state': 'unresolved', 'reason': 'query unavailable'}
    monkeypatch.setattr(campaign.backend, 'observe', lambda *a, **k: observation)
    monkeypatch.setattr(campaign.backend, 'execute', lambda *a: pytest.fail('unsafe launch'))
    before = state_path.read_bytes()
    assert campaign.main(['--resume', '-o', str(root),
                          '--selected-unit', 'unit_0']) == 0
    assert 'Action needed:' in capsys.readouterr().out
    assert state_path.read_bytes() == before
    assert not (old_root.parent / 'attempt_0002').exists() or blocker == 'occupied_second'


@pytest.mark.parametrize('returncode', [0, 7])
def test_failed_recovery_second_attempt_reconciles_without_retry(
        tmp_path, monkeypatch, capsys, returncode):
    root, _, profile = seed_failed_condor_attempts(tmp_path, monkeypatch)
    calls = queued_recovery(monkeypatch)
    args = ['--resume', '-o', str(root), '--selected-unit', 'unit_0']
    assert campaign.main(args) == 0
    argv, attempt_root, _ = calls[0]
    metadata = attempt_root / '_metadata'
    (metadata / 'guard.jsonl').write_text(guard_result_line(returncode))
    (metadata / 'renderer.log').write_text('fwd0pt absent in this region\n')
    if returncode == 0:
        output = attempt_root / 'run2/merged/unit_0'
        output.mkdir(parents=True)
        (output / 'unit_0_invmass_fitting.png').write_bytes(b'fixture-png')
    write_lifecycle(argv, attempt_root, returncode, returncode, 0)
    monkeypatch.setattr(campaign.backend, 'observe', lambda *a, **k: {
        'state': 'terminal_success' if returncode == 0 else 'pending',
        'ad': {'JobStatus': 4 if returncode == 0 else 5, 'ExitCode': returncode}})
    assert campaign.main(['--resume', '-o', str(root), '--selected-unit', 'unit_0']) == 0
    unit = campaign.read_json(campaign.state_path(root, profile, 'run2'))['units']['unit_0']
    assert unit['status'] == ('interrupted' if returncode == 0 else 'failed')
    assert unit['attempts'][0]['result'] == 'failed'
    assert unit['attempts'][1]['child_result'] == returncode
    assert len(unit['attempts']) == 2
    if returncode == 0:
        assert not campaign.completion_receipt_path(attempt_root).exists()
        assert campaign.main(finalize_args(root, 'unit_0')) == 0
        assert campaign.read_json(campaign.state_path(root, profile, 'run2'))[
            'units']['unit_0']['status'] == 'success'
    assert campaign.main(args) == 0
    assert len(calls) == 1
    assert not list(root.rglob('attempt_0003'))
    capsys.readouterr()


@pytest.mark.parametrize('action', ['execute', 'resume'])
def test_condor_follow_requires_explicit_opt_in(tmp_path, monkeypatch, action):
    root, plan_path = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign.backend, 'execute', lambda *_: (_ for _ in ()).throw(
        campaign.ActiveChildUnresolved('synthetic submission')))
    monkeypatch.setattr(campaign.backend, 'observe', lambda *_: {'state': 'pending'})
    followed = []
    monkeypatch.setattr(campaign, 'follow_campaign', lambda *args: followed.append(args))
    if action == 'resume':
        campaign.run_campaign(root, 'sr_crlike_run2', 'fixture_campaign',
                              plan_path=plan_path, action='execute', backend_name='condor', emit=False)
        args = ['--resume', '-o', str(root)]
    else:
        args = ['--campaign-profile', 'sr_crlike_run2', '--campaign-tag', 'fixture_campaign',
                '--execute', '--plan', str(plan_path), '-o', str(root), '--backend', 'condor']
    assert campaign.main(args + ['--follow']) == 0
    assert len(followed) == 1


def test_local_execute_keeps_foreground_sequential_semantics(tmp_path, monkeypatch, capsys):
    root, plan_path = fixture_plan(tmp_path, monkeypatch, unit_count=2)
    calls = []

    def execute(argv, attempt_root, contract):
        if calls:
            state = campaign.read_json(campaign.state_path(root, 'sr_crlike_run2', 'run2'))
            assert state['units']['unit_0']['status'] == 'interrupted'
        calls.append(attempt_root.parent.name)
        return fake_success(argv, attempt_root)

    monkeypatch.setattr(campaign.backend, 'execute', execute)
    monkeypatch.setattr(campaign.backend, 'observe', lambda *_: pytest.fail('local scheduler observation'))
    assert campaign.main(['--campaign-profile', 'sr_crlike_run2', '--campaign-tag', 'fixture_campaign',
                          '--execute', '--plan', str(plan_path), '-o', str(root)]) == 0
    assert calls == ['unit_0', 'unit_1']
    output = capsys.readouterr().out
    assert 'Campaign state (persisted): 0 published / 2 planned' in output
    assert 'Monitor with:' not in output


@pytest.mark.parametrize('family', ['sr_crlike', 'sr_datacard'])
@pytest.mark.parametrize('view', ['run2', 'run3', 'run2_run3'])
def test_profile_component_binding_and_native_variables(tmp_path, monkeypatch, family, view):
    profile = f'{family}_{view}'
    components = ('run2', 'run3') if view == 'run2_run3' else (view,)
    root, plan_path = fixture_plan(tmp_path, monkeypatch, profile=profile,
                                  unit_count=5 * len(components))
    plan = campaign.normalize_plan(plan_path, profile, 'fixture_campaign', root)
    assert campaign.PROFILES[profile] == components
    for component in components:
        owned = [u for u in plan['units'] if u['component'] == component]
        assert len(owned) == 5
        assert all(u['variables'] == ['invmass', 'fwd0pt'] for u in owned)
        state = campaign.new_component_state(plan, component, campaign.runtime_identity())
        assert set(state['units']) == {u['unit_id'] for u in owned}
        expected_parent = root / 'components' / component if len(components) > 1 else root
        assert campaign.state_path(root, profile, component) == (
            expected_parent / f'.{family}_{component}_plot_campaign_state_v1.json')
        wrong = 'run3' if component == 'run2' else 'run2'
        assert all(u['component'] != wrong for u in owned)
    assert campaign.main(['--campaign-profile', profile, '--campaign-tag', 'fixture_campaign',
                          '-o', str(root), '--plan', str(plan_path), '--dry-run']) == 0
    assert not root.exists()


def test_datacard_combined_component_failure_isolation(tmp_path, monkeypatch):
    root, path = fixture_plan(tmp_path, monkeypatch, profile='sr_datacard_run2_run3', unit_count=10)
    plan = campaign.normalize_plan(path, 'sr_datacard_run2_run3', 'fixture_campaign', root)
    states = {c: campaign.new_component_state(plan, c, campaign.runtime_identity())
              for c in ('run2', 'run3')}
    for unit in states['run2']['units'].values():
        unit['status'] = 'success'
    for unit in states['run3']['units'].values():
        unit['status'] = 'failed'
    assert all(u['status'] == 'success' for u in states['run2']['units'].values())
    assert campaign.state_path(root, plan['campaign_profile'], 'run2') != campaign.state_path(
        root, plan['campaign_profile'], 'run3')
