"""Opt-in exhaustion policy, evaluated by the installed native ClassAd tools."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from analysis.topeft_run2 import datacard_matrix_condor as condor
from analysis.topeft_run2 import datacard_matrix_runner as runner
from analysis.topeft_run2 import make_cards
from test_datacard_matrix_condor import fake_query, write_worker_ad
from test_datacard_matrix_resumable_runner import fake_script, make_row, manifest, sha


resources = dict(request_cpus=1, request_memory="1GB", request_disk="1GB",
                 max_retries=4, max_starts=5)


@pytest.mark.parametrize("policy", [None, "remove", "hold"])
def test_frontend_policy_forwarding(tmp_path, fake_script, fake_query, monkeypatch,
                                    capsys, policy):
    row = make_row(tmp_path, fake_script)
    path, _ = manifest(tmp_path, fake_script, [row])
    destination = tmp_path / "dispatch"
    monkeypatch.setattr(condor.shutil, "which", lambda name: str(fake_query[0]))
    argv = ["make_cards.py", "--condor", "--prepare-only", "--matrix-manifest", str(path),
            "--condor-output", str(destination), "--request-cpus", "1",
            "--request-memory", "1GB", "--request-disk", "1GB", "--max-retries", "4",
            "--max-starts", "5"]
    if policy is not None:
        argv.extend(["--exhaustion-policy", policy])
    monkeypatch.setattr(sys, "argv", argv)
    make_cards.main()
    assert json.loads(capsys.readouterr().out)["submitted"] is False
    submit = (destination / "datacards.sub").read_text()
    assert ("on_exit_hold =" in submit) == (policy == "hold")
    assert ("periodic_remove = false" in submit) == (policy == "hold")
    assert runner.read_reservation(row) is None


def test_non_condor_default_and_explicit_policy_guard(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(make_cards, "_resolve_pkl_paths", lambda *args: ["input"])
    monkeypatch.setattr(make_cards, "_run_producer", lambda *args: calls.append(args))
    base = ["make_cards.py", "--out-dir", str(tmp_path / "cards")]
    monkeypatch.setattr(sys, "argv", base)
    make_cards.main()
    assert len(calls) == 1
    for policy in ("remove", "hold", "invalid"):
        monkeypatch.setattr(sys, "argv", [*base, "--exhaustion-policy", policy])
        with pytest.raises(SystemExit) as exc:
            make_cards.main()
        assert exc.value.code == 2
        assert "--condor" in capsys.readouterr().err or policy == "invalid"
    assert len(calls) == 1


@pytest.mark.parametrize("policy", ["invalid", "HOLD", None, 1])
def test_invalid_policy_is_pre_effect(tmp_path, fake_script, fake_query, policy):
    path, _ = manifest(tmp_path, fake_script, [make_row(tmp_path, fake_script)])
    destination = tmp_path / "dispatch"
    with pytest.raises(runner.RunnerError, match="exhaustion_policy"):
        condor.prepare_material(path, destination, **resources, exhaustion_policy=policy,
                                query_executable=str(fake_query[0]))
    assert not destination.exists()


def test_remove_default_material_is_unchanged(tmp_path, fake_script, fake_query):
    path, _ = manifest(tmp_path, fake_script, [make_row(tmp_path, fake_script)])
    texts = []
    for policy in (None, "remove"):
        destination = tmp_path / ("default" if policy is None else policy)
        condor.prepare_material(path, destination, **resources,
                                **({} if policy is None else {"exhaustion_policy": policy}),
                                query_executable=str(fake_query[0]))
        texts.append((destination / "datacards.sub").read_text().replace(str(destination), "DEST"))
    assert texts[0] == texts[1]
    assert "on_exit_hold" not in texts[0]
    assert "periodic_remove = (JobStatus == 5) || (JobStatus == 1 && NumJobStarts >= 5)" in texts[0]


def test_same_token_restart_preserves_partial_attempt_and_skips_receipt(
        tmp_path, fake_script, fake_query, monkeypatch, record_property):
    counter, marker = tmp_path / "counter", tmp_path / "interrupted_once"
    row = make_row(tmp_path, fake_script, extra=["--counter", str(counter),
                                               "--partial-once", str(marker)])
    path, data = manifest(tmp_path, fake_script, [row])
    destination = tmp_path / "dispatch"
    material = condor.prepare_material(path, destination, **resources,
                                      exhaustion_policy="hold", query_executable=str(fake_query[0]))
    with condor.submission_transaction(path, destination, material["scheduler"]) as transaction:
        token = transaction.reservations[0]["owner_id"]
        transaction.mark_submission_started()  # Synthetic transaction; no scheduler call.
    ad = tmp_path / "worker.ad"
    worker_env = write_worker_ad(ad, row["row_id"], token)
    monkeypatch.setenv("_CONDOR_JOB_AD", worker_env["_CONDOR_JOB_AD"])
    with pytest.raises(runner.RunnerError, match="returned 7"):
        runner.execute_unit(data, sha(path), row, backend="condor", reservation_token=token)
    first, = runner.attempts(sha(path), row)
    assert first.name == "attempt_000001"
    first_bytes = {str(p.relative_to(first)): p.read_bytes() for p in first.rglob("*") if p.is_file()}
    assert first_bytes and not runner.receipt_path(data, row).exists()
    assert not Path(row["output_root"]).exists()
    assert runner.classify(data, sha(path), row)["status"] == "partial_staging_attempt"
    # A still-held physical lease must prevent same-token overlap.
    with runner.unit_lease(data, row):
        with pytest.raises(runner.RunnerError, match="execution ownership unavailable"):
            runner.execute_unit(data, sha(path), row, backend="condor", reservation_token=token)
    assert counter.read_text() == "1"
    result = runner.execute_unit(data, sha(path), row, backend="condor", reservation_token=token)
    assert result["action"] == "executed"
    receipt = json.loads(runner.receipt_path(data, row).read_text())
    assert runner.validate_receipt(receipt, data, sha(path), row)[0]
    assert receipt["physical_attempt_id"] == "attempt_000002"
    assert [p.name for p in runner.attempts(sha(path), row)] == ["attempt_000001", "attempt_000002"]
    assert first_bytes == {str(p.relative_to(first)): p.read_bytes() for p in first.rglob("*") if p.is_file()}
    assert runner.read_reservation(row)["owner_id"] == token
    assert runner.read_reservation(row)["scheduler_id"] == "123.7"
    assert runner.execute_unit(data, sha(path), row, backend="condor", reservation_token=token)["action"] == "skipped_valid_receipt"
    assert counter.read_text() == "2"
    record_property("same_token_restart", json.dumps({"token": token, "job": "123.7",
                    "physical_attempts": ["attempt_000001", "attempt_000002"],
                    "producer_invocations_after_receipt_reentry": int(counter.read_text())}))


@pytest.mark.parametrize("max_starts,max_retries", [(5, 4), (7, 1), (2, 8)])
def test_installed_hold_policy_truth_table(tmp_path, fake_script, fake_query,
                                         record_property, max_starts, max_retries):
    submit_executable = shutil.which("condor_submit")
    assert submit_executable, "installed native submit parser required for hold qualification"
    path, _ = manifest(tmp_path, fake_script,
                       [make_row(tmp_path, fake_script, name) for name in ("z_last", "a_first")])
    destination = tmp_path / "dispatch"
    condor.prepare_material(path, destination,
                            **{**resources, "max_starts": max_starts, "max_retries": max_retries},
                            exhaustion_policy="hold", query_executable=str(fake_query[0]))
    env = {"PATH": "/usr/bin:/bin", "LANG": "C", "_CONDOR_NETWORK_INTERFACE": "127.0.0.1"}
    config = Path(submit_executable).resolve().parents[2] / "etc" / "condor_config"
    if config.is_file():
        env["CONDOR_CONFIG"] = str(config)
    elif os.environ.get("CONDOR_CONFIG"):
        env["CONDOR_CONFIG"] = os.environ["CONDOR_CONFIG"]
    probes = []

    def native(argv):
        result = subprocess.run(argv, env=env, text=True, capture_output=True, timeout=30)
        probes.append(dict(argv=argv, returncode=result.returncode,
                           stdout=result.stdout, stderr=result.stderr))
        record_property("native_probes", json.dumps(probes))
        return result

    parsed_path = tmp_path / "parsed.ads"
    parsed = native([submit_executable, "-dry-run", str(parsed_path), str(destination / "datacards.sub")])
    assert parsed.returncode == 0, parsed.stderr
    common, ads = {}, []
    for block in re.split(r"\n\s*\n", parsed_path.read_text().strip()):
        fields = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        if fields:
            if not ads:
                common = fields
            ads.append({**common, **fields})
    record_property("effective_ads", json.dumps(ads))
    assert len(ads) == 2
    for ad in ads:
        assert ad["JobMaxRetries"] == str(max_retries)
        assert ad["JobSuccessExitCode"] == "0"
        assert "NumJobCompletions > JobMaxRetries" in ad["OnExitRemove"]
        assert ad["PeriodicRemove"] == ad["PeriodicRelease"] == "false"
        assert "exhausted" in ad["OnExitHoldReason"] and "exhausted" in ad["PeriodicHoldReason"]
    base = ads[0]
    expressions = ["OnExitHold", "OnExitRemove", "PeriodicHold", "PeriodicRemove",
                   "PeriodicRelease", "Requirements", "StrictSuccess", "RunningPolicySafe"]
    target = dict(Arch='"X86_64"', OpSys='"LINUX"', OpSysAndVer='"RedHat9"',
                  Disk="2000000000", Memory="1000000", Cpus="8",
                  FileSystemDomain=base["FileSystemDomain"])
    target_context = "[" + ";".join(f"{k}={v}" for k, v in target.items()) + "]"
    # This installed condor_status does not bind -target during -ads output
    # evaluation. Bind the same machine attributes explicitly in native ClassAd
    # scope; Requirements itself remains exactly the installed parser output.
    record_property("matching_context", "native nested TARGET ClassAd and site has_groups attribute; external -target binding unavailable")
    cases = []

    def case(name, updates, hold=False, remove=False, periodic=False, match=True, success=False):
        values = {"NumJobStarts": "0", "NumJobCompletions": "0", "JobStatus": "1",
                  "ExitBySignal": "false", "ExitCode": "1", "TARGET": target_context,
                  "has_groups": "true", **updates}
        cases.append((name, {**base, **values, "Scenario": json.dumps(name),
                      "StrictSuccess": "(ExitBySignal =?= false) && (ExitCode =?= 0)",
                      "RunningPolicySafe": "(JobStatus =!= 2) || !(PeriodicHold || PeriodicRemove)"},
                      [hold, remove, periodic, False, False, match, success, True]))

    case("before_first_start", {})
    case("failed_before_exhaustion", {"JobStatus": "2", "NumJobStarts": "1", "NumJobCompletions": "1"})
    case("signal_before_exhaustion", {"NumJobStarts": "1", "ExitBySignal": "true", "ExitCode": "undefined"})
    case("last_idle_before_start", {"NumJobStarts": str(max_starts - 1)})
    # OnExitHold is consulted only on exit. The running activation is governed
    # by periodic expressions; qualify that distinction in RunningPolicySafe.
    case("last_running", {"JobStatus": "2", "NumJobStarts": str(max_starts)}, hold=True, match=False)
    case("last_success", {"JobStatus": "2", "NumJobStarts": str(max_starts), "ExitCode": "0"}, remove=True, match=False, success=True)
    case("early_success", {"NumJobStarts": "1", "ExitCode": "0"}, remove=True, success=True)
    case("last_failed_exit", {"JobStatus": "2", "NumJobStarts": str(max_starts), "NumJobCompletions": str(max_retries + 1)}, hold=True, remove=True, match=False)
    case("last_signal", {"JobStatus": "2", "NumJobStarts": str(max_starts), "ExitBySignal": "true", "ExitCode": "undefined"}, hold=True, match=False)
    case("last_eviction", {"NumJobStarts": str(max_starts)}, hold=True, periodic=True, match=False)
    case("retry_budget", {"NumJobStarts": "1", "NumJobCompletions": str(max_retries + 1)}, hold=True, remove=True)
    case("held_exhaustion", {"JobStatus": "5", "NumJobStarts": str(max_starts)}, hold=True, match=False)
    case("held_other_reason", {"JobStatus": "5", "HoldReasonCode": "12"})
    case("missing_signal_and_code", {"ExitBySignal": "undefined", "ExitCode": "undefined"})
    case("missing_code", {"ExitCode": "undefined"})
    # Native max_retries sees ExitCode=0 without guarding ExitBySignal. Hold
    # conservatively rather than let that generated OR claim a success.
    case("missing_signal_with_zero", {"ExitBySignal": "undefined", "ExitCode": "0"}, hold=True, remove=True)
    case("signal_with_stale_zero", {"ExitBySignal": "true", "ExitCode": "0"}, hold=True, remove=True)
    case("last_unknown_exit", {"JobStatus": "2", "NumJobStarts": str(max_starts), "ExitBySignal": "undefined", "ExitCode": "undefined"}, hold=True, match=False)
    case("insufficient_disk", {"TARGET": target_context.replace("Disk=2000000000", "Disk=1")}, match=False)
    case("wrong_arch", {"TARGET": target_context.replace('Arch="X86_64"', 'Arch="OTHER"')}, match=False)
    case("missing_site_group", {"has_groups": "false"}, match=False)
    cases_path, target_path = tmp_path / "cases.ads", tmp_path / "target.ad"
    cases_path.write_text("\n\n".join("\n".join(f"{k} = {v}" for k, v in ad.items()) for _, ad, _ in cases) + "\n")
    target_path.write_text('Arch = "X86_64"\nOpSys = "LINUX"\nOpSysAndVer = "RedHat9"\nhas_groups = true\nDisk = 2000000000\nMemory = 1000000\nCpus = 8\nHasFileTransfer = true\nFileSystemDomain = ' + base["FileSystemDomain"] + '\n')
    evaluator = shutil.which("classad_eval")
    probe = native([evaluator, "-quiet", "[X=1]", "X"]) if evaluator else None
    if probe is not None and probe.returncode == 0:
        results = []
        for name, ad, expected in cases:
            context = "[" + ";".join(f"{k}={v}" for k, v in ad.items()) + "]"
            actual = []
            for expression in expressions:
                # Evaluate full matching with a bound target below; the other
                # expressions need only the job's own attributes.
                if expression == "Requirements":
                    actual.append(None)
                    continue
                evaluated = native([evaluator, "-quiet", context, ad[expression]])
                assert evaluated.returncode == 0, evaluated.stderr
                actual.append(evaluated.stdout.strip())
            results.append(actual)
    else:
        record_property("classad_eval_unavailable", "absent" if probe is None else probe.stderr)
        results = None
    status = shutil.which("condor_status")
    assert status, "native offline matching evaluator required"
    evaluated = native([status, "-ads", str(cases_path), "-target", str(target_path),
                        "-autoformat:t", "Scenario", *expressions])
    assert evaluated.returncode == 0, evaluated.stderr
    lines = [line.split() for line in evaluated.stdout.splitlines() if line.strip()]
    actual = {line[0]: line[1:] for line in lines}
    assert len(actual) == len(cases), evaluated.stdout
    for index, (name, _, expected) in enumerate(cases):
        want = [str(value).lower() for value in expected]
        assert actual[name] == want, (name, actual[name], want)
        if results is not None:
            assert all(value is None or value == want[i] for i, value in enumerate(results[index]))
    record_property("reached_native_truth_table", json.dumps({"cases": actual,
                    "max_starts": max_starts, "max_retries": max_retries,
                    "precedence": "on_exit_hold before on_exit_remove; held periodic removal/release false"}))
