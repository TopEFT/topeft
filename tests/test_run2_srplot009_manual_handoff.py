import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_DIRECTORY = REPOSITORY_ROOT / "analysis" / "topeft_run2"
RUN_CR = RUN_DIRECTORY / "run_cr.sh"
RUN_ANALYSIS = RUN_DIRECTORY / "run_analysis.py"
FROZEN_ENV = RUN_DIRECTORY / "topeft-envs" / "env_spec_9d72aad444117c28.tar.gz"
FROZEN_SHA256 = "8245afe4b3c28f4948039d383ad2176f1ee3ebb5e61bcdf1b49289452b025332"
PUBLIC_PROFILES = {
    "run2_full", "run3_full", "run2_run3_full",
    "run2_full_CR", "run3_full_CR", "run2_run3_full_CR",
}
MATRIX_EARLY_PROFILES = (
    "run2_full",
    "run2_full_CR",
    "run3_full_CR",
    "run2_run3_full",
    "run2_run3_full_CR",
)
RUN2_SR_BLOCKS = [
    (("UL16", "UL16APV", "UL17", "UL18"), ("2l", "2lss_1tau", "2los_1tau", "4l"), ("njets", "lj0pt", "ptz", "ptz_wtau", "lt")),
    (("UL16", "UL16APV", "UL17", "UL18"), ("3l_m_offZ",), ("njets", "lj0pt", "ptll", "lt")),
    (("UL16", "UL16APV", "UL17", "UL18"), ("3l_p_offZ",), ("njets", "lj0pt", "ptll", "lt")),
    (("UL16", "UL16APV", "UL17", "UL18"), ("3l_onZ_tau",), ("njets", "lj0pt", "ptz", "lt")),
    (("UL16", "UL16APV", "UL17", "UL18"), ("3l_fwd",), ("njets", "lj0pt", "ptz", "lt")),
]
RUN3_SR_BLOCKS = [
    (("2022", "2022EE", "2023", "2023BPix"), categories, histograms)
    for _, categories, histograms in RUN2_SR_BLOCKS
]


def _option_values(argv, option, count=None):
    start = argv.index(option) + 1
    if count is not None:
        return argv[start : start + count]
    values = []
    for item in argv[start:]:
        if item.startswith("-"):
            break
        values.append(item)
    return values


def _commands(stdout):
    return [
        shlex.split(text)
        for text in re.findall(r"Running the following command:\n([^\n]+)", stdout)
    ]


def _scientific_signature(argv):
    return (
        tuple(_option_values(argv, "--years")),
        tuple(_option_values(argv, "--category-groups")),
        tuple(_option_values(argv, "--hist-list")),
        "--skip-cr" in argv,
        "--skip-sr" in argv,
        "--do-systs" in argv,
        "--do-np" in argv,
        "--np-postprocess=inline" in argv,
        "--np-postprocess=defer" in argv,
        tuple(_option_values(argv, "-x", 1)),
        tuple(_option_values(argv, "-s", 1)),
        tuple(_option_values(argv, "--env-file", 1)),
        "--snapshot" in argv,
    )


def _clean_environment():
    environment = os.environ.copy()
    for name in (
        "SRPLOT009_VALIDATION_BACKEND",
        "SRPLOT009_VALIDATION_ROOT",
        "SRPLOT009_VALIDATION_SCENARIO",
    ):
        environment.pop(name, None)
    return environment


def _run(profile, output_dir, campaign_tag, *, dry_run=True, environment=None):
    command = [
        str(RUN_CR), "--production-profile", profile,
        "--output-dir", str(output_dir),
        "--campaign-tag", campaign_tag,
        "--env-file", str(FROZEN_ENV),
    ]
    if dry_run:
        command.append("--dry-run")
    return subprocess.run(
        command,
        cwd=RUN_DIRECTORY,
        env=environment or _clean_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _assert_common(argv):
    assert argv[:2] == ["python", "run_analysis.py"]
    assert "--snapshot" in argv
    assert _option_values(argv, "--env-file", 1) == [str(FROZEN_ENV)]
    assert _option_values(argv, "-s", 1) == ["100000"]
    assert _option_values(argv, "-x", 1) == ["work_queue"]
    assert "--workers" not in argv
    assert "--nworkers" not in argv
    assert "--do-systs" in argv
    assert "--do-np" in argv
    assert "--options" in argv


def _write_backend(path):
    path.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
action="$1"
scenario="$2"
shift 2
printf '%s\\t%s\\n' "$action" "$scenario" >> "${{SRPLOT009_VALIDATION_ROOT}}/backend_actions.tsv"
case "$action" in
  validate_environment)
    archive="$1"
    if [[ "$scenario" == run2_validation_failure ]]; then
      if [[ ! -e "${{SRPLOT009_VALIDATION_ROOT}}/.run2_validation_seen" ]]; then
        : > "${{SRPLOT009_VALIDATION_ROOT}}/.run2_validation_seen"
        exit 18
      fi
    elif [[ "$scenario" == run3_validation_failure ]]; then
      if [[ ! -e "${{SRPLOT009_VALIDATION_ROOT}}/.run2_validation_seen" ]]; then
        : > "${{SRPLOT009_VALIDATION_ROOT}}/.run2_validation_seen"
      else
        exit 18
      fi
    elif [[ "$scenario" == run2_output_collision ]]; then
      if [[ ! -e "${{SRPLOT009_VALIDATION_ROOT}}/.run2_validation_seen" ]]; then
        : > "${{SRPLOT009_VALIDATION_ROOT}}/.run2_validation_seen"
        mkdir -p -- "${{SRPLOT009_VALIDATION_ROOT}}/output/run2" \
          "${{SRPLOT009_VALIDATION_ROOT}}/output/run2_CR"
      fi
    fi
    printf 'env_file: %s\\n' "$archive"
    printf 'env_file_sha256: {FROZEN_SHA256}\\n'
    printf 'env_manifest: %s.manifest.json\\n' "$archive"
    printf 'environment_fingerprint: %064d\\n' 9
    printf 'environment_validation_status: valid\\n'
    printf 'topcoffea_git_commit: %040d\\n' 8
    printf 'topcoffea_relevant_source_fingerprint: %064d\\n' 7
    ;;
  run_block)
    block_id="$1"
    env_file="$2"
    stage="$3"
    source_path="$4"
    nonprompt_path="$5"
    native_log_dir="$6"
    shift 6
    printf '%s\\t%s\\t%s\\n' "$block_id" "$env_file" "$*" >> "${{SRPLOT009_VALIDATION_ROOT}}/block_calls.tsv"
    : > "${{SRPLOT009_VALIDATION_ROOT}}/${{block_id}}.started"
    mkdir -p -- "$native_log_dir"
    for native_log in debug.log tr.log stats.log tasks.log; do
      printf '%s\\t%s\\n' "$block_id" "$native_log" > "$native_log_dir/$native_log"
    done
    if [[ "$scenario" == block_failure_3 && ( "$block_id" == run2_full_c || "$block_id" == run2_full_CR_block3 ) ]]; then
      exit 23
    fi
    if [[ "$scenario" == block_failure_2 && ( "$block_id" == *_b || "$block_id" == *_block2 ) ]]; then
      exit 22
    fi
    if [[ "$scenario" == two_failures && ( "$block_id" == *_b || "$block_id" == *_d ) ]]; then
      exit 24
    fi
    mkdir -p -- "$(dirname -- "$source_path")"
    printf 'synthetic source\\n' > "$source_path"
    if [[ " $* " != *" --defer-np "* ]]; then
      printf 'synthetic nonprompt\\n' > "$nonprompt_path"
    fi
    if [[ "$scenario" == run2_state_contradiction && "$block_id" == run2_full_a ]]; then
      rm -f -- "$native_log_dir/debug.log" "$native_log_dir/tr.log" \
        "$native_log_dir/stats.log" "$native_log_dir/tasks.log"
      printf '{{malformed\\n' > "${{SRPLOT009_VALIDATION_ROOT}}/output/run2/.run2_full_campaign_state.json"
      exit 0
    fi
    : > "${{SRPLOT009_VALIDATION_ROOT}}/${{block_id}}.finished"
    ;;
  run_nonprompt)
    block_id="$1"
    source_path="$2"
    nonprompt_path="$3"
    shift 3
    printf '%s\\n' "$block_id" >> "${{SRPLOT009_VALIDATION_ROOT}}/nonprompt_calls.tsv"
    [[ "$scenario" != nonprompt_failure_2 || "$block_id" != *_b ]] || exit 25
    [[ -s "$source_path" ]] || exit 26
    printf 'synthetic nonprompt\\n' > "$nonprompt_path"
    ;;
  *) exit 90 ;;
esac
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _stubbed_run(tmp_path, profile, scenario, *, name=None):
    validation_root = tmp_path / (name or f"{profile}-{scenario}")
    validation_root.mkdir()
    backend = validation_root / "backend.sh"
    _write_backend(backend)
    environment = _clean_environment()
    environment.update(
        {
            "SRPLOT009_VALIDATION_BACKEND": str(backend),
            "SRPLOT009_VALIDATION_ROOT": str(validation_root),
            "SRPLOT009_VALIDATION_SCENARIO": scenario,
        }
    )
    output_root = validation_root / "output"
    result = _run(
        profile,
        output_root,
        f"stub-{profile}",
        dry_run=False,
        environment=environment,
    )
    return result, output_root, validation_root


def _campaign_state(output_root, profile):
    return json.loads(
        (output_root / f".{profile}_campaign_state.json").read_text(
            encoding="utf-8"
        )
    )


def _combined_summary_tool(tmp_path):
    source = RUN_CR.read_text(encoding="utf-8")
    match = re.search(
        r"srplot009_write_combined_summary\(\) \{.*?<<'PY'\n(.*?)\nPY\n\}",
        source,
        re.DOTALL,
    )
    assert match
    tool = tmp_path / "combined_summary_tool.py"
    tool.write_text(match.group(1), encoding="utf-8")
    return tool


def _derive_combined_summary(
    tmp_path,
    profile,
    run2_state,
    run3_state,
    run2_classification,
    run3_classification,
    *,
    shared_component="none",
):
    output_root = tmp_path / "derived-summary"
    output_root.mkdir()
    tool = _combined_summary_tool(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(tool),
            profile,
            "synthetic-combined",
            str(output_root),
            str(run2_state),
            str(run3_state),
            run2_classification,
            run3_classification,
            "1",
            shared_component,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    summary = (output_root / "campaign_summary.md").read_text(encoding="utf-8")
    lines = (output_root / "campaign_summary.tsv").read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    rows = [dict(zip(header, line.split("\t"))) for line in lines[1:]]
    return summary, rows


def test_public_profile_inventory_and_default_alias():
    source = RUN_CR.read_text(encoding="utf-8")
    for profile in PUBLIC_PROFILES:
        assert profile in source
    assert "rebin_fine" in source
    result = subprocess.run(
        [str(RUN_CR), "--dry-run"],
        cwd=RUN_DIRECTORY,
        env=_clean_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert [signature[:3] for signature in map(_scientific_signature, _commands(result.stdout))] == RUN2_SR_BLOCKS


def test_four_component_profiles_resolve_authoritative_contracts(tmp_path):
    observed = {}
    for profile in ("run2_full", "run3_full", "run2_full_CR", "run3_full_CR"):
        result = _run(profile, tmp_path / profile, f"test-{profile}")
        assert result.returncode == 0, result.stdout
        commands = _commands(result.stdout)
        observed[profile] = commands
        assert commands
        for argv in commands:
            _assert_common(argv)
        assert result.stdout.count("sumw2_storage_mode: full_diagnostics") == 1
        assert not (tmp_path / profile).exists()

    assert [signature[:3] for signature in map(_scientific_signature, observed["run2_full"])] == RUN2_SR_BLOCKS
    assert [signature[:3] for signature in map(_scientific_signature, observed["run3_full"])] == RUN3_SR_BLOCKS
    assert len(observed["run2_full_CR"]) == 6
    assert len(observed["run3_full_CR"]) == 12
    assert {tuple(_option_values(a, "--years")) for a in observed["run2_full_CR"]} == {("2016APV", "2016", "2017", "2018")}
    assert {tuple(_option_values(a, "--years")) for a in observed["run3_full_CR"]} == {("2022", "2022EE"), ("2023", "2023BPix")}
    assert all("--skip-cr" in a for a in observed["run2_full"] + observed["run3_full"])
    assert all("--skip-sr" in a for a in observed["run2_full_CR"] + observed["run3_full_CR"])


def test_combined_profiles_reuse_components_and_separate_namespaces(tmp_path):
    for combined, first, second, suffix in (
        ("run2_run3_full", "run2_full", "run3_full", ""),
        ("run2_run3_full_CR", "run2_full_CR", "run3_full_CR", "_CR"),
    ):
        run2 = _run(first, tmp_path / f"single2{suffix}", f"matrix_run2{suffix}")
        run3 = _run(second, tmp_path / f"single3{suffix}", f"matrix_run3{suffix}")
        combined_root = tmp_path / combined
        result = _run(combined, combined_root, "matrix")
        assert run2.returncode == run3.returncode == result.returncode == 0
        expected = list(map(_scientific_signature, _commands(run2.stdout) + _commands(run3.stdout)))
        assert list(map(_scientific_signature, _commands(result.stdout))) == expected
        assert f"{combined_root}/run2{suffix}" in result.stdout
        assert f"{combined_root}/run3{suffix}" in result.stdout
        assert not combined_root.exists()


def test_stubbed_known_run2_failure_continues_later_blocks_and_combined_run3(tmp_path):
    backend = tmp_path / "backend.sh"
    _write_backend(backend)
    environment = _clean_environment()
    environment.update(
        {
            "SRPLOT009_VALIDATION_BACKEND": str(backend),
            "SRPLOT009_VALIDATION_ROOT": str(tmp_path),
            "SRPLOT009_VALIDATION_SCENARIO": "block_failure_3",
        }
    )
    output_root = tmp_path / "combined"
    result = _run("run2_run3_full", output_root, "failure-test", dry_run=False, environment=environment)
    assert result.returncode == 1, result.stdout
    assert (output_root / "run2").is_dir()
    assert (output_root / "run3").is_dir()
    assert (tmp_path / "run2_full_c.started").exists()
    assert (tmp_path / "run2_full_d.started").exists()
    assert (tmp_path / "run2_full_e.started").exists()
    assert (tmp_path / "run3_full_a.started").exists()
    run2_state = json.loads(
        (output_root / "run2" / ".run2_full_campaign_state.json").read_text()
    )
    assert run2_state["blocks"][2]["status"] == "source_failed"
    assert run2_state["blocks"][2]["source_exit_code"] == 23
    assert run2_state["campaign_status"] == "complete_with_known_failures"
    assert (output_root / "campaign_summary.tsv").is_file()
    summary = (output_root / "campaign_summary.md").read_text()
    assert "configured_count: 10" in summary
    assert "attempted_count: 10" in summary
    assert "successful_count: 9" in summary
    assert "known_failed_count: 1" in summary
    assert "final_classification: `complete_with_known_failures`" in summary
    calls = [line.split("\t", 1)[0] for line in (tmp_path / "block_calls.tsv").read_text().splitlines()]
    assert calls.count("run2_full_c") == 1
    assert calls.count("run3_full_a") == 1


@pytest.mark.parametrize(
    ("profile", "configured_count"),
    (("run2_run3_full", 10), ("run2_run3_full_CR", 18)),
)
def test_stubbed_combined_all_success_has_exact_configured_denominator(
    tmp_path, profile, configured_count
):
    result, output_root, _ = _stubbed_run(tmp_path, profile, "success")
    assert result.returncode == 0, result.stdout
    summary = (output_root / "campaign_summary.md").read_text()
    assert f"configured_count: {configured_count}" in summary
    assert f"attempted_count: {configured_count}" in summary
    assert f"successful_count: {configured_count}" in summary
    assert "known_failed_count: 0" in summary
    assert "attempted_incomplete_count: 0" in summary
    assert "not_attempted_count: 0" in summary
    assert "final_classification: `success`" in summary
    lines = (output_root / "campaign_summary.tsv").read_text().splitlines()
    assert len(lines) == configured_count + 1


def test_stubbed_run2_all_success_attempts_every_block_and_returns_zero(tmp_path):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, "run2_full", "success"
    )
    assert result.returncode == 0, result.stdout
    state = _campaign_state(output_root, "run2_full")
    assert [block["status"] for block in state["blocks"]] == ["success"] * 5
    assert state["attempted_block_count"] == 5
    assert state["final_process_exit_code"] == 0
    assert len(list(validation_root.glob("run2_full_*.started"))) == 5
    summary_lines = (output_root / "campaign_summary.tsv").read_text().splitlines()
    header = summary_lines[0].split("\t")
    rows = [dict(zip(header, line.split("\t"))) for line in summary_lines[1:]]
    assert [row["block_id"] for row in rows] == [
        block["id"] for block in state["blocks"]
    ]
    assert [row["final_block_status"] for row in rows] == [
        block["status"] for block in state["blocks"]
    ]


def test_stubbed_run2_block_two_failure_continues_blocks_three_through_five(tmp_path):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, "run2_full", "block_failure_2"
    )
    assert result.returncode == 1, result.stdout
    state = _campaign_state(output_root, "run2_full")
    assert state["blocks"][1]["status"] == "source_failed"
    assert state["blocks"][1]["source_exit_code"] == 22
    failed = state["blocks"][1]
    assert failed["source_command_argv"][0] == "./fullR3_run.sh"
    assert failed["source_started_at_utc"]
    assert failed["source_ended_at_utc"]
    assert failed["source_duration_seconds"] is not None
    snapshot = output_root / "failure_diagnostics" / "run2_full_b_source.tsv"
    snapshot_text = snapshot.read_text()
    assert "exit_code\t22" in snapshot_text
    assert "expected_output_1_size_bytes\t" in snapshot_text
    assert "wq_debug_log\t" in snapshot_text
    assert all(
        (validation_root / f"run2_full_{suffix}.started").exists()
        for suffix in "cde"
    )
    assert state["known_failed_block_count"] == 1


@pytest.mark.parametrize(
    ("profile", "block_count"),
    (("run2_full_CR", 6), ("run3_full_CR", 12)),
)
def test_stubbed_cr_component_known_failure_continues_remaining_blocks(
    tmp_path, profile, block_count
):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, profile, "block_failure_2"
    )
    assert result.returncode == 1, result.stdout
    state = _campaign_state(output_root, profile)
    assert len(state["blocks"]) == block_count
    assert state["blocks"][1]["status"] == "source_failed"
    assert state["blocks"][1]["source_exit_code"] == 22
    assert all(block["status"] == "success" for block in state["blocks"][2:])
    assert (validation_root / f"{profile}_block{block_count}.started").exists()


def test_stubbed_two_known_failures_are_retained_in_exact_aggregate(tmp_path):
    result, output_root, _ = _stubbed_run(
        tmp_path, "run2_full", "two_failures"
    )
    assert result.returncode == 1, result.stdout
    state = _campaign_state(output_root, "run2_full")
    failed = [block for block in state["blocks"] if block["status"] == "source_failed"]
    assert [block["id"] for block in failed] == ["run2_full_b", "run2_full_d"]
    assert state["known_failed_block_count"] == 2
    calls = [line.split("\t", 1)[0] for line in (output_root.parent / "block_calls.tsv").read_text().splitlines()]
    assert calls == [f"run2_full_{suffix}" for suffix in "abcde"]
    summary_rows = (output_root / "campaign_summary.tsv").read_text().splitlines()
    assert len(summary_rows) == 6
    assert sum("source_failed" in row for row in summary_rows[1:]) == 2


@pytest.mark.parametrize(
    ("profile", "suffix", "configured_count", "run3_block_count"),
    (
        ("run2_run3_full", "", 10, 5),
        ("run2_run3_full_CR", "_CR", 18, 12),
    ),
)
def test_stubbed_run2_local_preflight_failure_still_enters_combined_run3(
    tmp_path, profile, suffix, configured_count, run3_block_count
):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, profile, "run2_validation_failure"
    )
    assert result.returncode != 0
    assert not (output_root / f"run2{suffix}" / f".run2_full{suffix}_campaign_state.json").exists()
    assert (output_root / f"run3{suffix}").is_dir()
    assert len(list(validation_root.glob(f"run3_full{suffix}_*.started"))) == run3_block_count
    summary = (output_root / "campaign_summary.md").read_text()
    assert f"configured_count: {configured_count}" in summary
    assert f"attempted_count: {run3_block_count}" in summary
    assert f"successful_count: {run3_block_count}" in summary
    assert f"not_attempted_count: {configured_count - run3_block_count}" in summary
    assert "final_classification: `completed_with_component_blockers`" in summary
    lines = (output_root / "campaign_summary.tsv").read_text().splitlines()
    assert len(lines) == configured_count + 1
    assert sum("not_attempted_component_blocked" in line for line in lines[1:]) == configured_count - run3_block_count


def test_stubbed_run2_local_output_collision_still_enters_combined_run3(tmp_path):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, "run2_run3_full", "run2_output_collision"
    )
    assert result.returncode != 0
    assert (output_root / "run3").is_dir()
    assert len(list(validation_root.glob("run3_full_*.started"))) == 5
    summary = (output_root / "campaign_summary.md").read_text()
    assert "configured_count: 10" in summary
    assert "attempted_count: 5" in summary
    assert "not_attempted_count: 5" in summary


def test_stubbed_run2_local_state_contradiction_still_enters_combined_run3(tmp_path):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, "run2_run3_full", "run2_state_contradiction"
    )
    assert result.returncode != 0
    assert (output_root / "run3").is_dir()
    assert len(list(validation_root.glob("run3_full_*.started"))) == 5
    assert not any(
        (validation_root / "native_logs" / name).exists()
        for name in ("debug.log", "tr.log", "stats.log", "tasks.log")
    )
    summary = (output_root / "campaign_summary.md").read_text()
    assert "run2_component_status: `blocked_state_contradiction`" in summary
    assert "configured_count: 10" in summary
    assert "successful_count: 5" in summary
    assert "not_attempted_count: 5" in summary
    assert "final_classification: `completed_with_component_blockers`" in summary


def test_stubbed_run2_native_log_archive_failure_blocks_combined_run3(tmp_path):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, "run2_run3_full", "archive_copy_failure"
    )
    assert result.returncode != 0
    assert not (output_root / "run3").exists()
    assert not list(validation_root.glob("run3_full_*.started"))
    assert (validation_root / "native_logs" / "debug.log").is_file()
    summary = (output_root / "campaign_summary.md").read_text()
    assert "configured_count: 10" in summary
    assert "attempted_count: 1" in summary
    assert "attempted_incomplete_count: 1" in summary
    assert "not_attempted_count: 9" in summary
    assert "final_classification: `shared_campaign_blocked`" in summary
    lines = (output_root / "campaign_summary.tsv").read_text().splitlines()
    assert sum("not_attempted_shared_campaign_blocked" in line for line in lines[1:]) == 9


def test_stubbed_shared_blocker_after_first_run3_block_keeps_full_roster(tmp_path):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, "run2_run3_full", "run3_archive_copy_failure"
    )
    assert result.returncode != 0
    assert len(list(validation_root.glob("run2_full_*.started"))) == 5
    assert (validation_root / "run3_full_a.started").is_file()
    assert not (validation_root / "run3_full_b.started").exists()
    assert (validation_root / "native_logs" / "debug.log").is_file()
    summary = (output_root / "campaign_summary.md").read_text()
    assert "configured_count: 10" in summary
    assert "attempted_count: 6" in summary
    assert "successful_count: 5" in summary
    assert "attempted_incomplete_count: 1" in summary
    assert "not_attempted_count: 4" in summary
    assert "final_classification: `shared_campaign_blocked`" in summary
    lines = (output_root / "campaign_summary.tsv").read_text().splitlines()
    assert len(lines) == 11
    assert sum("not_attempted_shared_campaign_blocked" in line for line in lines[1:]) == 4


def test_stubbed_run3_local_preflight_failure_is_fully_accounted(tmp_path):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, "run2_run3_full", "run3_validation_failure"
    )
    assert result.returncode != 0
    assert len(list(validation_root.glob("run2_full_*.started"))) == 5
    assert not list(validation_root.glob("run3_full_*.started"))
    summary = (output_root / "campaign_summary.md").read_text()
    assert "configured_count: 10" in summary
    assert "attempted_count: 5" in summary
    assert "successful_count: 5" in summary
    assert "not_attempted_count: 5" in summary
    assert "final_classification: `completed_with_component_blockers`" in summary


def test_combined_summary_uses_plan_roster_and_stage_aware_attempted_semantics(tmp_path):
    run2_state = tmp_path / "run2_state.json"
    run2_state.write_text(
        json.dumps(
            {
                "production_profile": "run2_full",
                "campaign_tag": "synthetic-run2",
                "blocks": [
                    {"id": "run2_full_a", "status": "source_ready"},
                    *(
                        {"id": f"run2_full_{suffix}", "status": "planned"}
                        for suffix in "bcde"
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    summary, rows = _derive_combined_summary(
        tmp_path,
        "run2_run3_full",
        run2_state,
        tmp_path / "absent_run3_state.json",
        "blocked_incomplete",
        "not_attempted",
    )
    assert "configured_count: 10" in summary
    assert "attempted_count: 1" in summary
    assert "successful_count: 0" in summary
    assert "known_failed_count: 0" in summary
    assert "attempted_incomplete_count: 1" in summary
    assert "not_attempted_count: 9" in summary
    assert len(rows) == 10
    assert len({(row["component"], row["block_id"]) for row in rows}) == 10
    source_ready = next(row for row in rows if row["block_id"] == "run2_full_a")
    assert source_ready["aggregate_classification"] == "attempted_incomplete"
    assert source_ready["aggregate_reason"] == "attempted_incomplete"


def test_combined_cr_summary_keeps_all_eighteen_configured_rows_without_state(tmp_path):
    summary, rows = _derive_combined_summary(
        tmp_path,
        "run2_run3_full_CR",
        tmp_path / "absent_run2_state.json",
        tmp_path / "absent_run3_state.json",
        "blocked_global",
        "not_attempted",
    )
    assert "configured_count: 18" in summary
    assert "attempted_count: 0" in summary
    assert "not_attempted_count: 18" in summary
    assert len(rows) == 18
    assert len({(row["component"], row["block_id"]) for row in rows}) == 18


def test_combined_running_state_is_shared_blocked_and_never_reclassified_failed(tmp_path):
    run2_state = tmp_path / "run2_running_state.json"
    run2_state.write_text(
        json.dumps(
            {
                "production_profile": "run2_full",
                "campaign_tag": "synthetic-running",
                "blocks": [
                    {"id": "run2_full_a", "status": "source_running"},
                    *(
                        {"id": f"run2_full_{suffix}", "status": "planned"}
                        for suffix in "bcde"
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    summary, rows = _derive_combined_summary(
        tmp_path,
        "run2_run3_full",
        run2_state,
        tmp_path / "absent_run3_state.json",
        "blocked_ambiguous",
        "not_attempted",
        shared_component="run2",
    )
    assert "attempted_count: 1" in summary
    assert "attempted_incomplete_count: 1" in summary
    assert "known_failed_count: 0" in summary
    assert "not_attempted_count: 9" in summary
    assert "final_classification: `shared_campaign_blocked`" in summary
    assert sum(row["aggregate_reason"] == "not_attempted_shared_campaign_blocked" for row in rows) == 9
    source = RUN_CR.read_text(encoding="utf-8")
    assert 'if [[ "${component_classification}" == "blocked_ambiguous" ]]' in source


def test_stubbed_run3_source_failure_skips_its_nonprompt_and_continues(tmp_path):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, "run3_full", "block_failure_2"
    )
    assert result.returncode == 1, result.stdout
    state = _campaign_state(output_root, "run3_full")
    assert state["blocks"][1]["status"] == "source_failed"
    nonprompt_calls = (validation_root / "nonprompt_calls.tsv").read_text().splitlines()
    assert "run3_full_b" not in nonprompt_calls
    assert "run3_full_c" in nonprompt_calls
    assert (validation_root / "run3_full_e.started").exists()


def test_stubbed_nonprompt_failure_preserves_source_and_continues(tmp_path):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, "run3_full", "nonprompt_failure_2"
    )
    assert result.returncode == 1, result.stdout
    state = _campaign_state(output_root, "run3_full")
    failed = state["blocks"][1]
    assert failed["status"] == "nonprompt_failed"
    assert failed["source_status"] == "ready"
    assert failed["nonprompt_exit_code"] == 25
    assert (validation_root / "run3_full_c.started").exists()


def test_native_wq_logs_are_copy_verified_attributed_and_generic_sources_removed(tmp_path):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, "run2_full", "success"
    )
    assert result.returncode == 0, result.stdout
    state = _campaign_state(output_root, "run2_full")
    for block in state["blocks"]:
        metadata = block["native_work_queue_logs"]["source"]
        for key in (
            "wq_debug_log",
            "wq_transactions_log",
            "wq_stats_log",
            "wq_tasks_accum_log",
        ):
            item = metadata[key]
            assert item["exists"] is True
            assert item["regular_file"] is True
            assert item["nonempty"] is True
            archive = Path(item["path"])
            assert archive.is_file()
            assert block["id"] in archive.read_text()
    assert not any(
        (validation_root / "native_logs" / name).exists()
        for name in ("debug.log", "tr.log", "stats.log", "tasks.log")
    )


def test_native_wq_copy_verification_failure_preserves_originals_and_stops(tmp_path):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, "run2_full", "archive_copy_failure"
    )
    assert result.returncode != 0
    assert (validation_root / "native_logs" / "debug.log").is_file()
    assert not (validation_root / "run2_full_b.started").exists()
    state = _campaign_state(output_root, "run2_full")
    assert state["blocks"][0]["status"] == "source_ready"
    assert state["blocks"][1]["status"] == "planned"


def test_unexpected_generic_wq_log_fails_closed_before_child_launch(tmp_path):
    validation_root = tmp_path / "preexisting-native-log"
    validation_root.mkdir()
    backend = validation_root / "backend.sh"
    _write_backend(backend)
    native_dir = validation_root / "native_logs"
    native_dir.mkdir()
    stale_log = native_dir / "debug.log"
    stale_log.write_text("unowned prior invocation\n")
    environment = _clean_environment()
    environment.update(
        {
            "SRPLOT009_VALIDATION_BACKEND": str(backend),
            "SRPLOT009_VALIDATION_ROOT": str(validation_root),
            "SRPLOT009_VALIDATION_SCENARIO": "success",
        }
    )
    output_root = validation_root / "output"
    result = _run(
        "run2_full", output_root, "preexisting", dry_run=False, environment=environment
    )
    assert result.returncode != 0
    assert stale_log.read_text() == "unowned prior invocation\n"
    assert not (validation_root / "run2_full_a.started").exists()
    state = _campaign_state(output_root, "run2_full")
    assert all(block["status"] == "planned" for block in state["blocks"])


def test_production_child_stream_path_has_no_output_interposition():
    source = RUN_CR.read_text(encoding="utf-8")
    assert "run_production_source_child" in source
    assert "2>&1 |" not in source
    assert "| tee" not in source
    assert "exec > >(tee" not in source
    assert 'if run_production_source_child ' in source
    assert '"${command[@]}"' in source


def test_native_work_queue_executor_policy_is_unchanged():
    source = RUN_ANALYSIS.read_text(encoding="utf-8")
    for expected in (
        '"debug_log": "debug.log"',
        '"transactions_log": "tr.log"',
        '"stats_log": "stats.log"',
        '"tasks_accum_log": "tasks.log"',
        '"retries": 15',
        '"resources_mode": "auto"',
        '"resource_monitor": "measure"',
        '"split_on_exhaustion": True',
        '"verbose": True',
        '"print_stdout": False',
    ):
        assert expected in source


def test_collision_and_frozen_archive_override_fail_closed(tmp_path):
    collision = tmp_path / "collision"
    collision.mkdir()
    result = _run("run2_full_CR", collision, "collision")
    assert result.returncode != 0
    assert "refusing overwrite" in result.stdout

    result = subprocess.run(
        [
            str(RUN_CR), "--production-profile", "run2_full", "--dry-run",
            "--output-dir", str(tmp_path / "other"), "--campaign-tag", "other",
            "--env-file", str(tmp_path / "wrong.tar.gz"),
        ],
        cwd=RUN_DIRECTORY,
        env=_clean_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode != 0
    assert "pinned to the required frozen snapshot archive" in result.stdout


@pytest.mark.parametrize("profile", MATRIX_EARLY_PROFILES)
def test_every_matrix_owned_profile_rejects_unknown_tokens_before_side_effects(
    tmp_path, profile
):
    output_root = tmp_path / profile
    result = subprocess.run(
        [
            str(RUN_CR),
            "--production-profile",
            profile,
            "--dry-rnu",
            "--output-dir",
            str(output_root),
            "--campaign-tag",
            f"unknown-{profile}",
        ],
        cwd=RUN_DIRECTORY,
        env=_clean_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode != 0
    assert "unsupported run_cr.sh option '--dry-rnu'" in result.stdout
    assert not output_root.exists()


def test_matrix_owned_missing_env_value_fails_before_side_effects(tmp_path):
    output_root = tmp_path / "missing-env-value"
    result = subprocess.run(
        [
            str(RUN_CR),
            "--production-profile",
            "run2_full",
            "--dry-run",
            "--output-dir",
            str(output_root),
            "--campaign-tag",
            "missing-env-value",
            "--env-file",
        ],
        cwd=RUN_DIRECTORY,
        env=_clean_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode != 0
    assert "--env-file requires a value" in result.stdout
    assert not output_root.exists()


@pytest.mark.parametrize(
    "value_option",
    ("--production-profile", "--output-dir", "--campaign-tag", "--env-file"),
)
def test_matrix_value_options_reject_another_option_as_value_before_side_effects(
    tmp_path, value_option
):
    output_root = tmp_path / value_option.removeprefix("--")
    command = [
        str(RUN_CR),
        "--production-profile",
        "run2_full",
        "--output-dir",
        str(output_root),
        "--campaign-tag",
        "malformed-value",
        "--env-file",
        str(FROZEN_ENV),
    ]
    if value_option == "--campaign-tag":
        del command[5:7]
    elif value_option == "--env-file":
        del command[7:9]
    command.extend((value_option, "--dry-run"))

    result = subprocess.run(
        command,
        cwd=RUN_DIRECTORY,
        env=_clean_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode != 0
    assert f"{value_option} requires a value" in result.stdout
    assert not output_root.exists()
