"""Inert control-flow checks for the dedicated two-unit excluded-MC launcher."""

import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import pytest


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
    assert output.count("--sumw2-family njets") >= 2
    assert output.count("--np-postprocess=skip") >= 2
    assert output.count("--wq-processing-cores 1") >= 2
    assert output.count("--wq-accumulating-memory-mb 32000") >= 2
    assert output.count("--wq-accumulating-cores 2") >= 2
    assert "EXCLUDED_MC_CAMPAIGN_STATUS planned_only units=2" in output
    assert not output_root.exists()


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
  sleep 30 &
  printf 'sleep_pid:%s\n' "$!" >> "$EXCLUDED_MOCK_TRACE"
  wait
fi
mkdir -p "$outdir"
pkl="$outdir/$outname.pkl.gz"
printf 'inert-pkl:%s\n' "$outname" > "$pkl"
size=$(stat -c '%s' "$pkl")
sha=$(sha256sum "$pkl"); sha="${sha%% *}"
cfg_sha=$(sha256sum "$cfg"); cfg_sha="${cfg_sha%% *}"
if [[ "${EXCLUDED_MOCK_BEHAVIOR:-success}" == wrong_a && "$outname" == *block_a* ]]; then sha=0000000000000000000000000000000000000000000000000000000000000000; fi
jq -n --arg basename "${pkl##*/}" --arg sha "$sha" --argjson size "$size" \
  --arg cfg_basename "${cfg##*/}" --arg cfg_sha "$cfg_sha" '
  {metadata_schema_version:2,
   artifact:{pkl_basename:$basename,pkl_sha256:$sha,pkl_size_bytes:$size,artifact_kind:"processor_output",merged:false,nominal_container_schema_version:2,nominal_container_layout:"split_sibling_v1"},
   lineage:{inputs:[]},sumw2_content_manifest:{synthetic:true},histogram_applicability:{synthetic:true},
   production_sample_contract:{contract_version:1,compatibility_validated:true,contract_identity_sha256:"synthetic",wrapper_identity:"fullR3_run_excluded_mc.sh",cfg_identities:[{basename:$cfg_basename,content_sha256:$cfg_sha,input_kind:"cfg"}]},
   sumw2_storage_provenance:{requested_mode:"full_custom",resolved_mode:"full_custom",runtime_histogram_families:["njets"],resolved_datasets:["inert"],resolved_targets:["inert"]},
   requested_data_driven_products:{products:{nonprompt:{enabled:false},flips:{enabled:false}}},
   resolved_data_driven_contract:{products:{nonprompt:{enabled:false},flips:{enabled:false}}}}
' > "$pkl.metadata.json"
printf 'end:%s:%s\n' "$outname" "$manager" >> "$EXCLUDED_MOCK_TRACE"
'''


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


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
    git(repo, "init", "-q")
    git(repo, "add", "analysis/topeft_run2/run_excluded_mc.sh", "analysis/topeft_run2/fullR3_run_excluded_mc.sh", "analysis/topeft_run2/run_analysis_excluded_mc.py", "input_samples/cfgs/excluded_mc_run3_sr.cfg")
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Inert Test", "-c", "user.email=inert@example.invalid", "commit", "-qm", "Inert fixture"], check=True)
    output_root = tmp_path / "synthetic_output"
    output_root.mkdir()
    env_file = tmp_path / "synthetic_env.tar.gz"
    env_file.write_text("inert archive marker")
    trace = tmp_path / "mock_trace.txt"
    return copied_launcher, output_root, env_file, trace, git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize(
    ("behavior", "expected_success", "expected_starts"),
    [
        ("success", True, 2),
        ("fail_a", False, 1),
        ("missing_a", False, 1),
        ("wrong_a", False, 1),
    ],
)
@pytest.mark.parametrize("snapshot", [False, True])
def test_inert_execute_waits_and_gates_next_unit(tmp_path, behavior, expected_success, expected_starts, snapshot):
    copied_launcher, output_root, env_file, trace, head = inert_checkout(tmp_path)
    command = [str(copied_launcher), *plan_args(output_root, env_file), "--expected-source-head", head, "--execute"]
    if snapshot:
        command.append("--snapshot")
    result = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "EXCLUDED_MOCK_TRACE": str(trace), "EXCLUDED_MOCK_BEHAVIOR": behavior})
    assert (result.returncode == 0) is expected_success, result.stderr
    starts = [line for line in trace.read_text().splitlines() if line.startswith("start:")]
    assert len(starts) == expected_starts
    assert starts[0].endswith(manager_a)
    assert ("EXCLUDED_MC_ENV_MODE explicit_snapshot" if snapshot else "EXCLUDED_MC_ENV_MODE current_source") in result.stdout
    log_a = (output_root / "run3/excluded_mc_run3_block_a_njets.runner.log").read_text()
    assert "producer-stdout:excluded_mc_run3_block_a_njets" in result.stdout
    assert "producer-stderr:excluded_mc_run3_block_a_njets" in result.stdout
    assert "producer-stdout:excluded_mc_run3_block_a_njets" in log_a
    assert "producer-stderr:excluded_mc_run3_block_a_njets" in log_a
    if expected_success:
        assert starts[1].endswith(manager_d)
        lines = trace.read_text().splitlines()
        assert lines.index(f"end:excluded_mc_run3_block_a_njets:{manager_a}") < lines.index(f"start:excluded_mc_run3_block_d_njets:{manager_d}")
        assert "EXCLUDED_MC_CAMPAIGN_STATUS complete units=2" in result.stdout
        log_d = (output_root / "run3/excluded_mc_run3_block_d_njets.runner.log").read_text()
        assert result.stdout.index("producer-stdout:excluded_mc_run3_block_a_njets") < result.stdout.index("producer-stdout:excluded_mc_run3_block_d_njets")
        assert "producer-stdout:excluded_mc_run3_block_d_njets" in log_d
        assert "producer-stderr:excluded_mc_run3_block_d_njets" in log_d
    else:
        assert not any(manager_d in line for line in starts)
        assert "EXCLUDED_MC_CAMPAIGN_STATUS complete" not in result.stdout
        if behavior == "fail_a":
            assert result.returncode == 7
            assert "rc=7" in log_a


def test_inert_display_failure_blocks_d_after_successful_child(tmp_path):
    copied_launcher, output_root, env_file, trace, head = inert_checkout(tmp_path)
    bin_dir = tmp_path / "mock_bin"
    bin_dir.mkdir()
    tee = bin_dir / "tee"
    tee.write_text("#!/usr/bin/env bash\ncat >/dev/null\nexit 9\n")
    tee.chmod(0o755)
    result = subprocess.run(
        [str(copied_launcher), *plan_args(output_root, env_file), "--expected-source-head", head, "--execute"],
        capture_output=True, text=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "EXCLUDED_MOCK_TRACE": str(trace)},
    )
    assert result.returncode == 9
    assert "display/logging failed with rc 9" in result.stderr
    assert "rc=0" in (output_root / "run3/excluded_mc_run3_block_a_njets.runner.log").read_text()
    assert manager_d not in trace.read_text()
    assert "EXCLUDED_MC_CAMPAIGN_STATUS complete" not in result.stdout


def test_inert_execute_refuses_namespace_output_collision(tmp_path):
    copied_launcher, output_root, env_file, trace, head = inert_checkout(tmp_path)
    existing = output_root / "run3/excluded_mc_run3_block_d_njets.pkl.gz"
    existing.parent.mkdir()
    existing.write_text("foreign prior output")
    result = subprocess.run([str(copied_launcher), *plan_args(output_root, env_file), "--expected-source-head", head, "--execute"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Refusing collision" in result.stderr
    assert not trace.exists()
    assert existing.read_text() == "foreign prior output"


def test_inert_execute_refuses_held_namespace_lock(tmp_path):
    copied_launcher, output_root, env_file, trace, head = inert_checkout(tmp_path)
    lock_fd = os.open(output_root, os.O_RDONLY)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [str(copied_launcher), *plan_args(output_root, env_file), "--expected-source-head", head, "--execute"],
            capture_output=True, text=True,
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    assert result.returncode != 0
    assert "namespace is in use" in result.stderr
    assert not trace.exists()


def test_inert_interruption_stops_child_and_never_starts_d(tmp_path):
    copied_launcher, output_root, env_file, trace, head = inert_checkout(tmp_path)
    command = [str(copied_launcher), *plan_args(output_root, env_file), "--expected-source-head", head, "--execute"]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "EXCLUDED_MOCK_TRACE": str(trace), "EXCLUDED_MOCK_BEHAVIOR": "sleep_a"},
    )
    try:
        for _ in range(100):
            if trace.exists() and "start:excluded_mc_run3_block_a_njets" in trace.read_text():
                break
            time.sleep(0.02)
        else:
            pytest.fail("Inert first child did not start")
        child_pid = int(next(line.split(":", 1)[1] for line in trace.read_text().splitlines() if line.startswith("sleep_pid:")))
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode != 0
    assert "EXCLUDED_MC_INTERRUPTED" in stderr
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    assert manager_d not in trace.read_text()
    assert "EXCLUDED_MC_CAMPAIGN_STATUS complete" not in stdout


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
