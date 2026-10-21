"""Inert control-flow checks for the dedicated two-unit excluded-MC launcher."""

import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time

import pytest

from topeft.modules.histogram_category_compatibility import (
    PTZ_WTAU_CHANNEL_FILL,
    find_incompatible_histograms,
)


repo_root = Path(__file__).resolve().parents[1]
launcher = repo_root / "analysis/topeft_run2/run_excluded_mc.sh"
campaign_tag = "top26006-excludedmc-cross-era-njets-20261018-001"
manager_a = f"{campaign_tag}-run3-block-a"
manager_d = f"{campaign_tag}-run3-block-d"


def plan_args(output_root, env_file):
    return [
        "--run", "run3", "--block", "all", "--campaign-tag", campaign_tag,
        "--output-root", str(output_root), "--port-a", "31001", "--port-d", "31002",
        "--wq-staging-base", "/synthetic/staging", "--wq-quota-root", "/synthetic/quota",
        "--wq-free-warning-bytes", "100", "--wq-min-free-bytes", "50",
        "--wq-poll-seconds", "1", "--env-file", str(env_file),
    ]


def test_real_chain_dry_run_resolves_both_units_without_outputs(tmp_path):
    output_root = tmp_path / "never_created"
    command = [str(launcher), *plan_args(output_root, tmp_path / "synthetic_env.tar.gz"), "--dry-run"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    output = result.stdout
    assert output.index("EXCLUDED_MC_UNIT run3/a") < output.index("EXCLUDED_MC_UNIT run3/d")
    assert f"EXCLUDED_MC_MANAGER {manager_a}" in output
    assert f"EXCLUDED_MC_MANAGER {manager_d}" in output
    assert f"^{campaign_tag}-run3-block-(a|d)$" in output
    assert f"EXCLUDED_MC_OUTPUT {output_root}/run3/excluded_mc_run3_block_a_njets.pkl.gz" in output
    assert f"EXCLUDED_MC_OUTPUT {output_root}/run3/excluded_mc_run3_block_d_njets.pkl.gz" in output
    assert "EXCLUDED_MC_PORT 31001" in output and "EXCLUDED_MC_PORT 31002" in output
    assert f"EXCLUDED_MC_ENV_FILE {tmp_path}/synthetic_env.tar.gz" in output
    assert "EXCLUDED_MC_ENV_MODE current_source" in output
    assert "--snapshot" not in output
    assert output.count("python run_analysis_excluded_mc.py") == 2
    assert output.count("--sample-universe-wrapper fullR3_run_excluded_mc.sh") >= 2
    assert output.count("--ttgamma-sample-role-policy split") >= 2
    assert output.count("--years 2022 2022EE 2023 2023BPix") == 2
    assert output.count("--skip-cr") == 2
    python_commands = [
        shlex.split(line) for line in output.splitlines()
        if line.startswith("python run_analysis_excluded_mc.py ")
    ]
    assert len(python_commands) == 2
    for argv, categories, histograms in zip(
        python_commands,
        (("2l", "2los_1tau"), ("3l_onZ_tau",)),
        (("njets", "lj0pt", "ptz", "lt"),
         ("njets", "lj0pt", "ptz", "lt")),
    ):
        assert argv[argv.index("--hist-list") + 1:argv.index("--skip-cr")] == list(histograms)
        assert argv[argv.index("--category-groups") + 1:argv.index("--wq-manager-name")] == list(categories)
        assert argv[argv.index("--years") + 1:argv.index("--hist-list")] == ["2022", "2022EE", "2023", "2023BPix"]
        assert "--do-systs" in argv
        assert "--do-np" not in argv
        assert "--np-postprocess=skip" in argv
        assert "--defer-np" not in argv
        assert "--record-raw-count" not in argv
        assert "--all-analysis" in argv
        assert [argv[index + 1] for index, flag in enumerate(argv[:-1]) if flag == "--sumw2-family"] == list(histograms)
        assert "--cfg-override" not in argv  # The wrapper forwards the dedicated CFG as a positional input.
        assert argv[2].endswith("/input_samples/cfgs/excluded_mc_run3_sr.cfg")
        assert "--sample-universe-wrapper" in argv
        assert "fullR3_run_excluded_mc.sh" in argv
        assert "--ttgamma-sample-role-policy" in argv
        assert argv[argv.index("--ttgamma-sample-role-policy") + 1] == "split"
        assert "--snapshot" not in argv
        assert "--options" not in argv
        assert "--no-sumw2" not in argv
    assert "excluded_mc_run3_sr.cfg" in output
    assert "--hist-list njets --skip-cr" not in output
    assert output.count("--wq-processing-cores 1") >= 2
    assert output.count("--wq-accumulating-memory-mb 32000") >= 2
    assert output.count("--wq-accumulating-cores 2") >= 2
    assert "EXCLUDED_MC_CAMPAIGN_STATUS planned_only units=2" in output
    assert not output_root.exists()


def test_run3_a_histograms_match_live_channel_fill_rule(tmp_path):
    result = subprocess.run(
        [str(launcher), *plan_args(tmp_path / "never_created", tmp_path / "synthetic_env.tar.gz"), "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    argv = shlex.split(next(
        line for line in result.stdout.splitlines()
        if line.startswith("python run_analysis_excluded_mc.py ")
    ))
    categories = argv[argv.index("--category-groups") + 1:argv.index("--wq-manager-name")]
    histograms = argv[argv.index("--hist-list") + 1:argv.index("--skip-cr")]
    channel_config = json.loads(
        (repo_root / "topeft/channels/ch_lst.json").read_text(encoding="utf-8")
    )["ALL_CH_LST_SR"]
    selected = {category: channel_config[category] for category in categories}

    assert categories == ["2l", "2los_1tau"]
    assert find_incompatible_histograms(histograms, selected_category_dicts=(selected,)) == ()
    incompatible = find_incompatible_histograms(
        ["ptz_wtau"], selected_category_dicts=(selected,)
    )
    assert [(item.histogram_family, item.required_capability) for item in incompatible] == [
        ("ptz_wtau", PTZ_WTAU_CHANNEL_FILL)
    ]
    full_a = {category: channel_config[category] for category in (*categories, "2lss_1tau", "4l")}
    assert find_incompatible_histograms(
        ["ptz_wtau"], selected_category_dicts=(full_a,)
    ) == ()


def test_snapshot_dry_run_forwards_to_dedicated_entrypoint(tmp_path):
    output_root = tmp_path / "never_created"
    result = subprocess.run(
        [str(launcher), *plan_args(output_root, tmp_path / "synthetic_env.tar.gz"),
         "--snapshot", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    output = result.stdout
    assert "EXCLUDED_MC_ENV_MODE explicit_snapshot" in output
    assert output.index("EXCLUDED_MC_UNIT run3/a") < output.index("EXCLUDED_MC_UNIT run3/d")
    assert output.count("--snapshot") == 4  # launcher argv and wrapper Python command, twice each
    assert output.count("python run_analysis_excluded_mc.py") == 2
    assert "EXCLUDED_MC_CAMPAIGN_STATUS planned_only units=2" in output
    assert not output_root.exists()


def test_snapshot_requires_explicit_env_file(tmp_path):
    args = plan_args(tmp_path / "never_created", tmp_path / "synthetic_env.tar.gz")
    env_index = args.index("--env-file")
    del args[env_index:env_index + 2]
    result = subprocess.run(
        [str(launcher), *args, "--snapshot", "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "--snapshot requires an explicit --env-file" in result.stderr


stub_runner = r'''#!/usr/bin/env bash
set -euo pipefail
outname="" outdir="" manager="" cfg=""
while (($#)); do
  case "$1" in
    --output-name|-p|--wq-manager-name|--cfg-override)
      case "$1" in
        --output-name) outname="$2" ;; -p) outdir="$2" ;;
        --wq-manager-name) manager="$2" ;; --cfg-override) cfg="$2" ;;
      esac
      shift 2 ;;
    *) shift ;;
  esac
done
printf 'start:%s:%s\n' "$outname" "$manager" >> "$EXCLUDED_MOCK_TRACE"
printf 'producer-stdout:%s\n' "$outname"
printf 'producer-stderr:%s\n' "$outname" >&2
if [[ "${EXCLUDED_MOCK_BEHAVIOR:-success}" == fail_a && "$outname" == *block_a* ]]; then exit 7; fi
if [[ "${EXCLUDED_MOCK_BEHAVIOR:-success}" == missing_a && "$outname" == *block_a* ]]; then exit 0; fi
if [[ "${EXCLUDED_MOCK_BEHAVIOR:-success}" == sleep_a && "$outname" == *block_a* ]]; then
  sleep 30
fi
printf 'end:%s:%s\n' "$outname" "$manager" >> "$EXCLUDED_MOCK_TRACE"
'''


def inert_checkout(tmp_path):
    repo = tmp_path / "inert_checkout"
    analysis = repo / "analysis/topeft_run2"
    cfg_dir = repo / "input_samples/cfgs"
    analysis.mkdir(parents=True)
    cfg_dir.mkdir(parents=True)
    copied_launcher = analysis / launcher.name
    shutil.copy2(launcher, copied_launcher)
    runner = analysis / "fullR3_run_excluded_mc.sh"
    runner.write_text(stub_runner)
    runner.chmod(0o755)
    (analysis / "run_analysis_excluded_mc.py").write_text("# inert test marker\n")
    (cfg_dir / "excluded_mc_run3_sr.cfg").write_text("# inert synthetic CFG; no production samples\n")
    output_root = tmp_path / "synthetic_output"
    output_root.mkdir()
    env_file = tmp_path / "synthetic_env.tar.gz"
    env_file.write_text("inert archive marker")
    trace = tmp_path / "mock_trace.txt"
    return copied_launcher, output_root, env_file, trace


@pytest.mark.parametrize(
    ("behavior", "expected_success", "expected_starts"),
    [
        ("success", True, 2),
        ("fail_a", False, 1),
        ("missing_a", True, 2),
    ],
)
@pytest.mark.parametrize("snapshot", [False, True])
def test_inert_execute_waits_and_gates_next_unit(tmp_path, behavior, expected_success, expected_starts, snapshot):
    copied_launcher, output_root, env_file, trace = inert_checkout(tmp_path)
    command = [str(copied_launcher), *plan_args(output_root, env_file), "--execute"]
    if snapshot:
        command.append("--snapshot")
    result = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "EXCLUDED_MOCK_TRACE": str(trace), "EXCLUDED_MOCK_BEHAVIOR": behavior})
    assert (result.returncode == 0) is expected_success, result.stderr
    starts = [line for line in trace.read_text().splitlines() if line.startswith("start:")]
    assert len(starts) == expected_starts
    assert starts[0].endswith(manager_a)
    assert ("EXCLUDED_MC_ENV_MODE explicit_snapshot" if snapshot else "EXCLUDED_MC_ENV_MODE current_source") in result.stdout
    assert "producer-stdout:excluded_mc_run3_block_a_njets" in result.stdout
    assert "producer-stderr:excluded_mc_run3_block_a_njets" in result.stderr
    assert not list(output_root.rglob("*.runner.log"))
    if expected_success:
        assert starts[1].endswith(manager_d)
        lines = trace.read_text().splitlines()
        if behavior == "success":
            assert lines.index(f"end:excluded_mc_run3_block_a_njets:{manager_a}") < lines.index(f"start:excluded_mc_run3_block_d_njets:{manager_d}")
        else:
            assert not any(line.startswith("end:excluded_mc_run3_block_a_njets") for line in lines)
        assert "EXCLUDED_MC_CAMPAIGN_STATUS complete units=2" in result.stdout
        assert result.stdout.index("producer-stdout:excluded_mc_run3_block_a_njets") < result.stdout.index("producer-stdout:excluded_mc_run3_block_d_njets")
        assert "producer-stderr:excluded_mc_run3_block_d_njets" in result.stderr
    else:
        assert not any(manager_d in line for line in starts)
        assert "EXCLUDED_MC_CAMPAIGN_STATUS complete" not in result.stdout
        if behavior == "fail_a":
            assert result.returncode == 7


def test_inert_execute_refuses_namespace_output_collision(tmp_path):
    copied_launcher, output_root, env_file, trace = inert_checkout(tmp_path)
    existing = output_root / "run3/excluded_mc_run3_block_d_njets.pkl.gz"
    existing.parent.mkdir()
    existing.write_text("foreign prior output")
    result = subprocess.run([str(copied_launcher), *plan_args(output_root, env_file), "--execute"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Refusing collision" in result.stderr
    assert not trace.exists()
    assert existing.read_text() == "foreign prior output"


def test_inert_interruption_stops_child_and_never_starts_d(tmp_path):
    copied_launcher, output_root, env_file, trace = inert_checkout(tmp_path)
    command = [str(copied_launcher), *plan_args(output_root, env_file), "--execute"]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "EXCLUDED_MOCK_TRACE": str(trace), "EXCLUDED_MOCK_BEHAVIOR": "sleep_a"},
        start_new_session=True,
    )
    try:
        for _ in range(100):
            if trace.exists() and "start:excluded_mc_run3_block_a_njets" in trace.read_text():
                break
            time.sleep(0.02)
        else:
            pytest.fail("Inert first child did not start")
        os.killpg(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode != 0
    assert "EXCLUDED_MC_INTERRUPTED" not in stderr
    assert manager_d not in trace.read_text()
    assert "EXCLUDED_MC_CAMPAIGN_STATUS complete" not in stdout


def test_launcher_has_no_extra_execution_guards_or_obsolete_flag():
    source = launcher.read_text()
    for obsolete in ("--expected-source-head", "coproc", "tee -a", "setsid", "flock", "ss -H", "verify_completed_unit", ".runner.log"):
        assert obsolete not in source


def test_all_block_plan_rejects_run2(tmp_path):
    args = plan_args(tmp_path / "output", tmp_path / "env")
    args[args.index("run3")] = "run2"
    result = subprocess.run([str(launcher), *args, "--dry-run"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Run3 only" in result.stderr


def test_run2_single_unit_still_resolves_without_launch(tmp_path):
    args = plan_args(tmp_path / "never_created", tmp_path / "synthetic_env.tar.gz")
    args[args.index("run3")] = "run2"
    args[args.index("all")] = "a"
    port_a = args.index("--port-a")
    args[port_a:port_a + 4] = ["--port", "31001"]
    result = subprocess.run([str(launcher), *args, "--dry-run"], capture_output=True, text=True, check=True)
    assert "EXCLUDED_MC_UNIT run2/a" in result.stdout
    assert "Resolved years: UL16 UL16APV UL17 UL18" in result.stdout
    assert "excluded_mc_run2_sr.cfg" in result.stdout
    assert "EXCLUDED_MC_UNIT run2/d" not in result.stdout
