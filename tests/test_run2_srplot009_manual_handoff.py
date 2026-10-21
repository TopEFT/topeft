import copy
import hashlib
import importlib.util
import io
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from pathlib import Path

import pytest
from topcoffea.modules import remote_environment
from analysis.topeft_run2 import wq_staging_guard as staging_guard
from analysis.topeft_run2 import sr_reco_manifest as sr_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_DIRECTORY = REPOSITORY_ROOT / "analysis" / "topeft_run2"
RUN_CR = RUN_DIRECTORY / "run_cr.sh"
RUN_ANALYSIS = RUN_DIRECTORY / "run_analysis.py"
TEST_ENV_NAME = "run_cr_test_environment.tar.gz"
T0_HISTORICAL_LAUNCHER_COMMIT = "5e7a7b4cdfa5babddab922650e5c35bb0a2c2ea2"
PUBLIC_PROFILES = {
    "run2_full", "run3_full", "run2_run3_full",
    "run2_full_CR", "run3_full_CR", "run2_run3_full_CR",
    "t0_sr_statonly", "t0_cr_statonly",
}
PROFILE_BLOCK_IDS = {
    "run2_full": [f"run2_full_{suffix}" for suffix in "abcde"],
    "run3_full": [f"run3_full_{suffix}" for suffix in "abcde"],
    "t0_sr_statonly": [
        f"t0_sr_statonly_{era}_{suffix}"
        for era in ("run2", "run3")
        for suffix in "abcde"
    ],
    "t0_cr_statonly": [
        f"t0_cr_statonly_block{index}" for index in range(1, 19)
    ],
    "run2_full_CR": [f"run2_full_CR_block{index}" for index in range(1, 7)],
    "run3_full_CR": [f"run3_full_CR_block{index}" for index in range(1, 13)],
}
MATRIX_EARLY_PROFILES = (
    "run2_full",
    "run2_full_CR",
    "run3_full_CR",
    "run2_run3_full",
    "run2_run3_full_CR",
    "t0_sr_statonly",
    "t0_cr_statonly",
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
T0_RUN2_SR_BLOCKS = [
    (("2016APV", "2016", "2017", "2018"), categories, histograms)
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


def _write_test_environment(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TEST_ENV_NAME
    if path.is_file() and path.with_name(f"{path.name}.manifest.json").is_file():
        return path
    content = b"synthetic snapshot environment"
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("environment.txt")
        entry.size = len(content)
        archive.addfile(entry, io.BytesIO(content))
    request = {
        "environment_fingerprint": "synthetic-environment-fingerprint",
        "python_version": "3.9.23",
        "resolved_environment_spec": {"conda": {"packages": []}, "pip": []},
        "resolved_environment_spec_fingerprint": "synthetic-spec-fingerprint",
        "editable_packages": [
            {
                "package_name": "topcoffea",
                "git_commit": "8" * 40,
                "watched_source_fingerprint": "7" * 64,
            }
        ],
    }
    remote_environment.write_archive_manifest(str(path), request)
    return path


def _run(
    profile,
    output_dir,
    campaign_tag,
    *,
    dry_run=True,
    resume=False,
    environment=None,
    supply_env_file=True,
    extra_args=(),
):
    env_file = _write_test_environment(Path(output_dir).parent)
    command = [
        str(RUN_CR), "--production-profile", profile,
        "--output-dir", str(output_dir),
        "--campaign-tag", campaign_tag,
    ]
    if supply_env_file:
        command.extend(("--env-file", str(env_file)))
    if dry_run:
        command.append("--dry-run")
    if resume:
        command.append("--resume")
    command.extend(extra_args)
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
    env_file = Path(_option_values(argv, "--env-file", 1)[0])
    assert env_file.is_absolute()
    assert env_file.name == TEST_ENV_NAME
    assert _option_values(argv, "-s", 1) == ["100000"]
    assert _option_values(argv, "-x", 1) == ["work_queue"]
    assert "--workers" not in argv
    assert "--nworkers" not in argv
    assert "--do-systs" in argv
    assert "--do-np" in argv
    assert "--options" in argv


def _write_backend(path):
    archive = _write_test_environment(path.parent)
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    path.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
action="$1"
scenario="$2"
shift 2
write_sidecar() {{
  local artifact_path="$1"
  local artifact_sha
  artifact_sha=$(sha256sum "$artifact_path")
  artifact_sha="${{artifact_sha%% *}}"
  printf '{{"artifact":{{"pkl_basename":"%s","pkl_size_bytes":%s,"pkl_sha256":"%s"}},"synthetic":true}}\n' \
    "$(basename -- "$artifact_path")" "$(stat -c %s -- "$artifact_path")" "$artifact_sha" \
    > "$artifact_path.metadata.json"
}}
printf '%s\\t%s\\n' "$action" "$scenario" >> "${{SRPLOT009_VALIDATION_ROOT}}/backend_actions.tsv"
case "$action" in
  prepare_environment)
    archive="{archive}"
    archive_sha256="{archive_sha256}"
    printf 'env_file: %s\n' "$archive"
    printf 'env_file_sha256: %s\n' "$archive_sha256"
    printf 'env_manifest: %s.manifest.json\n' "$archive"
    printf 'environment_fingerprint: %064d\n' 9
    printf 'environment_validation_status: valid\n'
    printf 'topcoffea_git_commit: %040d\n' 8
    printf 'topcoffea_relevant_source_fingerprint: %064d\n' 7
    ;;
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
    archive_sha256=$(sha256sum "$archive")
    archive_sha256="${{archive_sha256%% *}}"
    printf 'env_file: %s\\n' "$archive"
    printf 'env_file_sha256: %s\\n' "$archive_sha256"
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
    if [[ "$scenario" == guard_failure_second && "$block_id" == *_a_2 ]]; then
      exit 143
    fi
    if [[ "$scenario" == two_failures && ( "$block_id" == *_b || "$block_id" == *_d ) ]]; then
      exit 24
    fi
    mkdir -p -- "$(dirname -- "$source_path")"
    printf 'synthetic source\\n' > "$source_path"
    write_sidecar "$source_path"
    if [[ " $* " != *" --defer-np "* ]]; then
      printf 'synthetic nonprompt\\n' > "$nonprompt_path"
      write_sidecar "$nonprompt_path"
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
    if [[ "$scenario" == sr_np_failure_once ]]; then
      if [[ "$block_id" == run2_sr_ext_* ]]; then
        state_profile=run2_sr_ext
      else
        state_profile=run3_sr_ext
      fi
      state_path="$(dirname -- "$source_path")/.$state_profile"_campaign_state.json
      grep -Fq '"source_status": "ready"' "$state_path" || exit 96
      grep -Fq '"source_receipt": {{' "$state_path" || exit 97
      if [[ ! -e "${{SRPLOT009_VALIDATION_ROOT}}/.np_failed_once" ]]; then
        : > "${{SRPLOT009_VALIDATION_ROOT}}/.np_failed_once"
        exit 137
      fi
    fi
    if [[ "$scenario" == sr_np_failure_e1 && "$block_id" == *_e_1 ]]; then
      exit 137
    fi
    if [[ "$scenario" == sr_np_partial_failure_e1 && "$block_id" == *_e_1 ]]; then
      printf 'partial synthetic nonprompt\\n' > "$nonprompt_path"
      exit 137
    fi
    [[ "$scenario" != nonprompt_failure_2 || "$block_id" != *_b ]] || exit 25
    [[ -s "$source_path" ]] || exit 26
    printf 'synthetic nonprompt\\n' > "$nonprompt_path"
    write_sidecar "$nonprompt_path"
    ;;
  *) exit 90 ;;
esac
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _stubbed_run(tmp_path, profile, scenario, *, name=None, supply_env_file=True,
                 dry_run=False, extra_args=()):
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
        dry_run=dry_run,
        environment=environment,
        supply_env_file=supply_env_file,
        extra_args=extra_args,
    )
    return result, output_root, validation_root


def _campaign_state(output_root, profile):
    return json.loads(
        (output_root / f".{profile}_campaign_state.json").read_text(
            encoding="utf-8"
        )
    )


def _prepare_t0_recovery_fixture(tmp_path, retry_indices=range(10)):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, "t0_sr_statonly", "success"
    )
    assert result.returncode == 0, result.stdout
    state_path = output_root / ".t0_sr_statonly_campaign_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["topeft_git_commit"] = T0_HISTORICAL_LAUNCHER_COMMIT
    preserved_transitions = {}
    retry_indices = set(retry_indices)
    for index, block in enumerate(state["blocks"]):
        if index not in retry_indices:
            continue
        Path(block["expected_np_path"]).unlink()
        Path(f'{block["expected_np_path"]}.metadata.json').unlink()
        block["status"] = "nonprompt_failed"
        block["source_status"] = "ready"
        block["source_exit_code"] = 0
        block["nonprompt_status"] = "failed"
        block["nonprompt_exit_code"] = 1
        block["exit_code"] = 1
        block.pop("nonprompt_postprocessor_provenance", None)
        for transition in block["transitions"]:
            transition.pop("postprocessor_provenance", None)
        block["transitions"].append(
            {
                "timestamp_utc": "2026-01-02T00:00:00Z",
                "stage": "nonprompt",
                "status": "failed",
                "exit_code": 1,
                "signal": None,
                "duration_seconds": 1,
                "detail": "synthetic_historical_nonprompt_failure",
            }
        )
        block["expected_output_readback"] = [
            {
                "path": block["expected_nominal_path"],
                "exists": True,
                "regular_file": True,
                "size_bytes": Path(block["expected_nominal_path"]).stat().st_size,
                "nonempty": True,
            },
            {
                "path": block["expected_np_path"],
                "exists": False,
                "regular_file": False,
                "size_bytes": None,
                "nonempty": False,
            },
        ]
        preserved_transitions[block["id"]] = copy.deepcopy(block["transitions"])
    state["campaign_status"] = "complete_with_known_failures"
    state["successful_block_count"] = 10 - len(retry_indices)
    state["known_failed_block_count"] = len(retry_indices)
    state["attempted_block_count"] = 10
    state["not_attempted_block_count"] = 0
    state["final_process_exit_code"] = 1
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (validation_root / "block_calls.tsv").write_text("", encoding="utf-8")
    (validation_root / "nonprompt_calls.tsv").write_text("", encoding="utf-8")
    return state_path, output_root, validation_root, preserved_transitions


def _resume_components(profile):
    if profile == "run2_run3_full":
        return ("run2_full", "run3_full")
    if profile == "run2_run3_full_CR":
        return ("run2_full_CR", "run3_full_CR")
    return (profile,)


def _resume_environment(validation_root, scenario="success"):
    environment = _clean_environment()
    environment.update(
        {
            "SRPLOT009_VALIDATION_BACKEND": str(validation_root / "backend.sh"),
            "SRPLOT009_VALIDATION_ROOT": str(validation_root),
            "SRPLOT009_VALIDATION_SCENARIO": scenario,
        }
    )
    return environment


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


def test_t0_sr_statonly_resolves_exact_nominal_raw_count_contract(tmp_path):
    output_dir = tmp_path / "t0_sr_statonly"
    validation_root = tmp_path / "t0_sr_statonly_validation"
    validation_root.mkdir()
    backend = validation_root / "backend.sh"
    _write_backend(backend)
    env_file = validation_root / TEST_ENV_NAME
    env_sha256 = hashlib.sha256(env_file.read_bytes()).hexdigest()
    environment = _clean_environment()
    environment.update(
        {
            "SRPLOT009_VALIDATION_BACKEND": str(backend),
            "SRPLOT009_VALIDATION_ROOT": str(validation_root),
            "SRPLOT009_VALIDATION_SCENARIO": "success",
        }
    )
    result = subprocess.run(
        [
            str(RUN_CR),
            "--production-profile", "t0_sr_statonly",
            "--dry-run",
            "--output-dir", str(output_dir),
            "--campaign-tag", "t0_sr_statonly_test",
        ],
        cwd=RUN_DIRECTORY,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    commands = _commands(result.stdout)
    assert [signature[:3] for signature in map(_scientific_signature, commands)] == (
        T0_RUN2_SR_BLOCKS + RUN3_SR_BLOCKS
    )
    assert len(commands) == 10
    for argv in commands:
        assert argv[:2] == ["python", "run_analysis.py"]
        assert "--snapshot" not in argv
        assert _option_values(argv, "--env-file", 1) == [str(env_file)]
        assert _option_values(argv, "-s", 1) == ["100000"]
        assert _option_values(argv, "-x", 1) == ["work_queue"]
        assert "--workers" not in argv
        assert "--nworkers" not in argv
        assert "--do-systs" not in argv
        assert "--record-raw-count" in argv
        assert "--do-np" in argv
        assert "--np-postprocess=defer" in argv
        assert "--options" in argv
        assert "--rebuild-env" not in argv
        assert "--prepare-env-only" not in argv
    assert result.stdout.count("sumw2_storage_mode: full_diagnostics") == 1
    assert f"env_file_sha256: {env_sha256}" in result.stdout
    assert "environment_policy: current_resolved" in result.stdout
    assert "dry_run_complete: ten commands resolved" in result.stdout
    assert "--legacy-campaign-state" not in result.stdout
    assert "--legacy-campaign-block" not in result.stdout
    assert "2024" not in result.stdout
    assert not output_dir.exists()

    explicit_snapshot = subprocess.run(
        [
            str(RUN_CR),
            "--production-profile", "t0_sr_statonly",
            "--dry-run",
            "--output-dir", str(tmp_path / "wrong_archive"),
            "--campaign-tag", "t0_sr_statonly_wrong_archive",
            "--env-file", str(env_file),
        ],
        cwd=RUN_DIRECTORY,
        env=_clean_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert explicit_snapshot.returncode == 0, explicit_snapshot.stdout
    assert "environment_policy: explicit_snapshot" in explicit_snapshot.stdout
    assert all("--snapshot" in argv for argv in _commands(explicit_snapshot.stdout))
    assert not (tmp_path / "wrong_archive").exists()


def test_t0_cr_statonly_composes_full_cr_with_native_data_driven_contract(tmp_path):
    run2 = _run("run2_full_CR", tmp_path / "run2", "t0-cr-run2-authority")
    run3 = _run("run3_full_CR", tmp_path / "run3", "t0-cr-run3-authority")
    output_dir = tmp_path / "t0_cr_statonly"
    validation_root = tmp_path / "t0_cr_current_resolved"
    validation_root.mkdir()
    backend = validation_root / "backend.sh"
    _write_backend(backend)
    environment = _clean_environment()
    environment.update(
        {
            "SRPLOT009_VALIDATION_BACKEND": str(backend),
            "SRPLOT009_VALIDATION_ROOT": str(validation_root),
            "SRPLOT009_VALIDATION_SCENARIO": "success",
        }
    )
    result = _run(
        "t0_cr_statonly",
        output_dir,
        "t0-cr-statonly-test",
        environment=environment,
        supply_env_file=False,
    )

    assert run2.returncode == run3.returncode == result.returncode == 0
    authority_commands = _commands(run2.stdout) + _commands(run3.stdout)
    commands = _commands(result.stdout)
    current_env_file = validation_root / TEST_ENV_NAME
    current_env_sha256 = hashlib.sha256(current_env_file.read_bytes()).hexdigest()
    assert len(commands) == 18
    assert [
        _scientific_signature(command)[:5] for command in commands
    ] == [
        _scientific_signature(command)[:5] for command in authority_commands
    ]
    for argv in commands:
        assert argv[:2] == ["python", "run_analysis.py"]
        assert "--snapshot" not in argv
        assert _option_values(argv, "--env-file", 1) == [str(current_env_file)]
        assert _option_values(argv, "-s", 1) == ["100000"]
        assert _option_values(argv, "-x", 1) == ["work_queue"]
        assert "--workers" not in argv
        assert "--nworkers" not in argv
        assert "--do-systs" not in argv
        assert "--record-raw-count" in argv
        assert "--do-np" in argv
        assert "--np-postprocess=defer" in argv
        assert "--options" in argv
        assert "--skip-sr" in argv
        assert "--skip-cr" not in argv
        assert _option_values(argv, "--sample-universe-wrapper", 3) == [
            "run_cr.sh", "->", "fullR3_run.sh"
        ]

    data_driven_commands = [
        shlex.split(text)
        for text in re.findall(
            r"Separate nonprompt/charge-flip command "
            r"\(not executed by dry-run\):\n([^\n]+)",
            result.stdout,
        )
    ]
    assert len(data_driven_commands) == 18
    for source_argv, data_driven_argv in zip(commands, data_driven_commands):
        assert data_driven_argv[:2] == ["python", "./run_data_driven.py"]
        assert "--only-flips" not in data_driven_argv
        source_path = Path(_option_values(source_argv, "-p", 1)[0]) / (
            f'{_option_values(source_argv, "-o", 1)[0]}.pkl.gz'
        )
        assert _option_values(data_driven_argv, "--input-pkl", 1) == [
            str(source_path)
        ]
        assert _option_values(data_driven_argv, "--output-pkl", 1) == [
            str(source_path).removesuffix(".pkl.gz") + "_np.pkl.gz"
        ]

    assert result.stdout.count("sumw2_storage_mode: full_diagnostics") == 1
    assert f"env_file_sha256: {current_env_sha256}" in result.stdout
    assert "environment_policy: current_resolved" in result.stdout
    assert "do_systs: false" in result.stdout
    assert "do_np: true" in result.stdout
    assert "run_cr: true" in result.stdout
    assert "run_sr: false" in result.stdout
    assert "eighteen source commands and eighteen separate data-driven commands" in result.stdout
    assert not output_dir.exists()

    explicit_snapshot = subprocess.run(
        [
            str(RUN_CR),
            "--production-profile", "t0_cr_statonly",
            "--dry-run",
            "--output-dir", str(tmp_path / "rejected_archive"),
            "--campaign-tag", "t0-cr-statonly-rejected-archive",
            "--env-file", str(current_env_file),
        ],
        cwd=RUN_DIRECTORY,
        env=_clean_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert explicit_snapshot.returncode == 0, explicit_snapshot.stdout
    assert "environment_policy: explicit_snapshot" in explicit_snapshot.stdout
    assert all("--snapshot" in argv for argv in _commands(explicit_snapshot.stdout))
    assert not (tmp_path / "rejected_archive").exists()


def test_t0_cr_statonly_state_tracks_source_and_data_driven_children(tmp_path):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, "t0_cr_statonly", "success"
    )

    assert result.returncode == 0, result.stdout
    state = _campaign_state(output_root, "t0_cr_statonly")
    assert state["region"] == "CR"
    assert state["nonprompt_mode"] == "separate"
    assert state["do_systs"] is False
    assert state["successful_block_count"] == 18
    assert state["known_failed_block_count"] == 0
    assert [block["id"] for block in state["blocks"]] == (
        PROFILE_BLOCK_IDS["t0_cr_statonly"]
    )
    assert all(block["source_status"] == "ready" for block in state["blocks"])
    assert all(block["nonprompt_status"] == "success" for block in state["blocks"])
    assert all(
        block["nonprompt_postprocessor_provenance"]
        for block in state["blocks"]
    )
    block_calls = (validation_root / "block_calls.tsv").read_text().splitlines()
    assert len(block_calls) == 18
    assert all("--defer-np" in line for line in block_calls)
    assert (
        validation_root / "nonprompt_calls.tsv"
    ).read_text().splitlines() == PROFILE_BLOCK_IDS["t0_cr_statonly"]
    for block in state["blocks"]:
        assert Path(block["expected_nominal_path"]).is_file()
        assert Path(f'{block["expected_nominal_path"]}.metadata.json').is_file()
        assert Path(block["expected_np_path"]).is_file()
        assert Path(f'{block["expected_np_path"]}.metadata.json').is_file()


def test_t0_resume_reuses_all_sources_and_appends_postprocessor_provenance(tmp_path):
    state_path, output_root, validation_root, preserved_transitions = (
        _prepare_t0_recovery_fixture(tmp_path)
    )

    result = _run(
        "t0_sr_statonly",
        output_root,
        "stub-t0_sr_statonly",
        dry_run=False,
        resume=True,
        environment=_resume_environment(validation_root),
    )

    assert result.returncode == 0, result.stdout
    assert result.stdout.count("Reusing validated completed source") == 10
    assert (validation_root / "block_calls.tsv").read_text() == ""
    nonprompt_calls = (
        validation_root / "nonprompt_calls.tsv"
    ).read_text().splitlines()
    assert nonprompt_calls == PROFILE_BLOCK_IDS["t0_sr_statonly"]
    recovered = json.loads(state_path.read_text(encoding="utf-8"))
    assert recovered["topeft_git_commit"] == T0_HISTORICAL_LAUNCHER_COMMIT
    assert recovered["campaign_status"] == "success"
    assert recovered["successful_block_count"] == 10
    assert recovered["known_failed_block_count"] == 0
    assert recovered["final_process_exit_code"] == 0
    topcoffea_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT.parent / "topcoffea",
        text=True,
    ).strip()
    expected_provenance = {
        "topeft_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "topcoffea_git_commit": topcoffea_commit,
    }
    for block in recovered["blocks"]:
        historical = preserved_transitions[block["id"]]
        assert block["transitions"][: len(historical)] == historical
        recovery = block["transitions"][len(historical) :]
        assert [item["status"] for item in recovery] == ["running", "success"]
        assert all(
            item["postprocessor_provenance"] == expected_provenance
            for item in recovery
        )
        assert block["nonprompt_postprocessor_provenance"] == expected_provenance
        assert Path(block["expected_np_path"]).is_file()
        assert Path(block["expected_np_path"]).stat().st_size > 0
    summary = (output_root / "campaign_summary.md").read_text(encoding="utf-8")
    assert "successful: 10" in summary
    assert "known_failed: 0" in summary
    assert "final_classification: `success`" in summary


def test_t0_resume_retries_only_failed_nonprompt_subset(tmp_path):
    _, output_root, validation_root, _ = _prepare_t0_recovery_fixture(
        tmp_path, retry_indices=range(4, 10)
    )

    result = _run(
        "t0_sr_statonly",
        output_root,
        "stub-t0_sr_statonly",
        dry_run=False,
        resume=True,
        environment=_resume_environment(validation_root),
    )

    assert result.returncode == 0, result.stdout
    assert (validation_root / "block_calls.tsv").read_text() == ""
    assert (
        validation_root / "nonprompt_calls.tsv"
    ).read_text().splitlines() == PROFILE_BLOCK_IDS["t0_sr_statonly"][4:]
    assert result.stdout.count("Skipping validated t0_sr_statonly block") == 4
    assert result.stdout.count("Reusing validated completed source") == 6


def test_t0_resume_dry_run_transports_legacy_context_without_verdict(tmp_path):
    _, output_root, validation_root, _ = _prepare_t0_recovery_fixture(
        tmp_path,
        retry_indices=range(4, 10),
    )

    result = _run(
        "t0_sr_statonly",
        output_root,
        "stub-t0_sr_statonly",
        dry_run=True,
        resume=True,
        environment=_resume_environment(validation_root),
    )

    assert result.returncode == 0, result.stdout
    assert result.stdout.count("Legacy context transported (no applicability verdict)") == 6
    assert result.stdout.count("--legacy-campaign-state") == 6
    assert result.stdout.count("--legacy-campaign-block") == 6
    assert result.stdout.count('"analysis_mode":"all"') == 6
    assert result.stdout.count('"producer_topeft_commit"') == 6
    assert result.stdout.count('"source_pkl_sha256"') == 6
    assert result.stdout.count('"category_groups"') == 6
    assert result.stdout.count('"histogram_families"') == 6
    assert "not_applicable" not in result.stdout
    assert "applicable\"" not in result.stdout
    assert (validation_root / "block_calls.tsv").read_text() == ""
    assert (validation_root / "nonprompt_calls.tsv").read_text() == ""


def test_t0_resume_fails_closed_when_ready_source_is_missing(tmp_path):
    _, output_root, validation_root, _ = _prepare_t0_recovery_fixture(tmp_path)
    state = _campaign_state(output_root, "t0_sr_statonly")
    Path(state["blocks"][0]["expected_nominal_path"]).unlink()

    result = _run(
        "t0_sr_statonly",
        output_root,
        "stub-t0_sr_statonly",
        dry_run=False,
        resume=True,
        environment=_resume_environment(validation_root),
    )

    assert result.returncode != 0
    assert "records a reusable source, but it is missing or empty" in result.stdout
    assert (validation_root / "block_calls.tsv").read_text() == ""
    assert (validation_root / "nonprompt_calls.tsv").read_text() == ""


def test_t0_resume_fails_closed_on_np_collision_or_running_state(tmp_path):
    _, output_root, validation_root, _ = _prepare_t0_recovery_fixture(tmp_path)
    state = _campaign_state(output_root, "t0_sr_statonly")
    Path(f'{state["blocks"][0]["expected_np_path"]}.metadata.json').write_text(
        '{"unexpected": "sidecar collision"}\n', encoding="utf-8"
    )
    collision = _run(
        "t0_sr_statonly",
        output_root,
        "stub-t0_sr_statonly",
        dry_run=False,
        resume=True,
        environment=_resume_environment(validation_root),
    )
    assert collision.returncode != 0
    assert "expected _np path already exists" in collision.stdout
    assert (validation_root / "block_calls.tsv").read_text() == ""
    assert (validation_root / "nonprompt_calls.tsv").read_text() == ""

    running_root = tmp_path / "running"
    running_root.mkdir()
    state_path, running_output, running_validation, _ = _prepare_t0_recovery_fixture(
        running_root
    )
    running_state = json.loads(state_path.read_text(encoding="utf-8"))
    running_state["blocks"][0]["status"] = "nonprompt_running"
    running_state["blocks"][0]["nonprompt_status"] = "running"
    state_path.write_text(json.dumps(running_state), encoding="utf-8")
    running = _run(
        "t0_sr_statonly",
        running_output,
        "stub-t0_sr_statonly",
        dry_run=False,
        resume=True,
        environment=_resume_environment(running_validation),
    )
    assert running.returncode != 0
    assert "ambiguous interrupted nonprompt stage" in running.stdout
    assert (running_validation / "block_calls.tsv").read_text() == ""
    assert (running_validation / "nonprompt_calls.tsv").read_text() == ""


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


@pytest.mark.parametrize("profile", ("run2_full", "run3_full"))
def test_stubbed_nonprompt_failure_preserves_source_and_continues(tmp_path, profile):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, profile, "nonprompt_failure_2"
    )
    assert result.returncode == 1, result.stdout
    state = _campaign_state(output_root, profile)
    failed = state["blocks"][1]
    assert state["schema_version"] == 6 and state["nonprompt_mode"] == "separate"
    assert failed["status"] == "nonprompt_failed"
    assert failed["source_status"] == "ready"
    assert failed["nonprompt_exit_code"] == 25
    assert failed["source_receipt"]["path"] == failed["expected_nominal_path"]
    assert (validation_root / f"{profile}_c.started").exists()


@pytest.mark.parametrize(
    ("profile", "component"),
    (("run2_sr_ext", "run2"), ("run3_sr_ext", "run3")),
)
def test_native_wq_logs_are_campaign_owned_copy_verified_and_attributed(
    tmp_path, profile, component
):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, profile, "success"
    )
    assert result.returncode == 0, result.stdout
    state = _campaign_state(output_root, profile)
    resolved_log_dirs = set()
    for block in state["blocks"]:
        command = block["source_command_argv"]
        log_dir = Path(command[command.index("--wq-log-dir") + 1])
        assert "--defer-np" in command
        assert "--np-postprocess=inline" not in command
        expected_log_dir = (
            output_root
            / "work_queue_logs"
            / component
            / block["id"]
            / "source_native"
        )
        assert log_dir == expected_log_dir
        assert log_dir.is_dir()
        resolved_log_dirs.add(log_dir)
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
            (log_dir / name).exists()
            for name in ("debug.log", "tr.log", "stats.log", "tasks.log")
        )
    assert len(resolved_log_dirs) == len(state["blocks"])
    if profile == "run3_sr_ext":
        assert len((validation_root / "nonprompt_calls.tsv").read_text().splitlines()) == len(
            state["blocks"]
        )


def test_native_wq_copy_verification_failure_preserves_originals_and_stops(tmp_path):
    result, output_root, validation_root = _stubbed_run(
        tmp_path, "run2_full", "archive_copy_failure"
    )
    assert result.returncode != 0
    active_log = (
        output_root
        / "work_queue_logs"
        / "run2"
        / "run2_full_a"
        / "source_native"
        / "debug.log"
    )
    assert active_log.is_file()
    assert not (validation_root / "run2_full_b.started").exists()
    state = _campaign_state(output_root, "run2_full")
    assert state["blocks"][0]["status"] == "source_ready"
    assert state["blocks"][1]["status"] == "planned"


def test_stale_checkout_global_wq_log_does_not_block_campaign_invocation(tmp_path):
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
    assert result.returncode == 0, result.stdout
    assert stale_log.read_text() == "unowned prior invocation\n"
    assert (validation_root / "run2_full_a.started").exists()
    state = _campaign_state(output_root, "run2_full")
    assert state["blocks"][0]["status"] == "success"


def test_exact_owned_wq_log_fails_closed_before_executor_construction(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(RUN_DIRECTORY))
    spec = importlib.util.spec_from_file_location("run_analysis_log_path_test", RUN_ANALYSIS)
    run_analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_analysis)

    output_root = tmp_path / "campaign"
    unrelated_legacy_log = tmp_path / "checkout" / "debug.log"
    unrelated_legacy_log.parent.mkdir()
    unrelated_legacy_log.write_text("preserve unrelated generic log\n")
    first_paths = run_analysis._prepare_work_queue_log_paths(
        str(output_root), "fragment_a"
    )
    second_paths = run_analysis._prepare_work_queue_log_paths(
        str(output_root), "fragment_b"
    )
    assert Path(first_paths["debug_log"]).parent == (
        output_root / "work_queue_logs" / "fragment_a" / "source_native"
    )
    assert Path(second_paths["debug_log"]).parent == (
        output_root / "work_queue_logs" / "fragment_b" / "source_native"
    )
    assert set(first_paths.values()).isdisjoint(second_paths.values())
    assert unrelated_legacy_log.read_text() == "preserve unrelated generic log\n"

    owned_log = Path(first_paths["debug_log"])
    owned_log.write_text("preserve exact owned log\n")
    with pytest.raises(SystemExit, match="Owned Work Queue log already exists"):
        run_analysis._prepare_work_queue_log_paths(str(output_root), "fragment_a")
    assert owned_log.read_text() == "preserve exact owned log\n"


def test_production_child_stream_path_has_no_output_interposition():
    source = RUN_CR.read_text(encoding="utf-8")
    assert "run_production_source_child" in source
    assert "2>&1 |" not in source
    assert "| tee" not in source
    assert "exec > >(tee" not in source
    assert 'if run_production_source_child ' in source
    assert '"${command[@]}"' in source


def test_native_work_queue_executor_policy_preserves_runtime_settings():
    source = RUN_ANALYSIS.read_text(encoding="utf-8")
    for expected in (
        '"debug_log": os.path.join(log_dir, "debug.log")',
        '"transactions_log": os.path.join(log_dir, "tr.log")',
        '"stats_log": os.path.join(log_dir, "stats.log")',
        '"tasks_accum_log": os.path.join(log_dir, "tasks.log")',
        '"--wq-log-dir"',
        "wq_log_paths = _prepare_work_queue_log_paths(outpath, outname, wq_log_dir)",
        "**wq_log_paths",
        '"retries": 15',
        '"resources_mode": "auto"',
        '"resource_monitor": "measure"',
        '"split_on_exhaustion": True',
        '"verbose": True',
        '"print_stdout": False',
    ):
        assert expected in source


def test_collision_and_invalid_explicit_archive_fail_closed(tmp_path):
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
    assert "could not validate its environment archive" in result.stdout
    assert "archive must be a readable non-empty regular file" in result.stdout


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
        "/tmp/parser-only-environment.tar.gz",
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


@pytest.mark.parametrize("profile", sorted(PUBLIC_PROFILES))
def test_all_public_profiles_resume_validated_campaign_state(tmp_path, profile):
    initial, output_root, validation_root = _stubbed_run(tmp_path, profile, "success")
    assert initial.returncode == 0, initial.stdout
    block_calls = validation_root / "block_calls.tsv"
    calls_before_resume = block_calls.read_text(encoding="utf-8")

    resumed = _run(
        profile,
        output_root,
        f"stub-{profile}",
        dry_run=True,
        resume=True,
        environment=_resume_environment(validation_root),
    )

    assert resumed.returncode == 0, resumed.stdout
    assert "has no automatic resume" not in resumed.stdout
    if profile.startswith("run2_run3"):
        assert "combined output namespace already exists" not in resumed.stdout
    for component in _resume_components(profile):
        for block_id in PROFILE_BLOCK_IDS[component]:
            assert f"Skipping validated {component} block: {block_id}" in resumed.stdout
    assert "Running the following command:" not in resumed.stdout
    assert block_calls.read_text(encoding="utf-8") == calls_before_resume


@pytest.mark.parametrize(
    "profile",
    ("run2_full_CR", "run3_full_CR", "run2_run3_full_CR"),
)
def test_cr_non_resume_still_rejects_existing_namespace(tmp_path, profile):
    _, output_root, _ = _stubbed_run(tmp_path, profile, "success")

    result = _run(profile, output_root, f"stub-{profile}")

    assert result.returncode != 0
    if profile == "run2_run3_full_CR":
        assert "combined output namespace already exists" in result.stdout
    else:
        assert "output directory already exists" in result.stdout


def test_cr_resume_recovers_only_the_existing_inline_completion_transition(tmp_path):
    _, output_root, validation_root = _stubbed_run(tmp_path, "run2_full_CR", "success")
    state_path = output_root / ".run2_full_CR_campaign_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    block = state["blocks"][0]
    block["status"] = "source_ready"
    block["source_status"] = "ready"
    block["nonprompt_status"] = "planned"
    block["nonprompt_exit_code"] = None
    state_path.write_text(json.dumps(state), encoding="utf-8")
    calls_before_resume = (validation_root / "block_calls.tsv").read_text(encoding="utf-8")

    resumed = _run(
        "run2_full_CR", output_root, "stub-run2_full_CR", dry_run=True,
        resume=True, environment=_resume_environment(validation_root),
    )

    assert resumed.returncode == 0, resumed.stdout
    assert "Skipping recovered run2_full_CR block: run2_full_CR_block1" in resumed.stdout
    assert "Running the following command:" not in resumed.stdout
    assert (validation_root / "block_calls.tsv").read_text(encoding="utf-8") == calls_before_resume
    assert _campaign_state(output_root, "run2_full_CR")["blocks"][0]["status"] == "success"


def test_cr_resume_fails_closed_for_invalid_or_ambiguous_state(tmp_path):
    _, output_root, validation_root = _stubbed_run(tmp_path, "run2_full_CR", "success")
    state_path = output_root / ".run2_full_CR_campaign_state.json"
    original_state = state_path.read_text(encoding="utf-8")
    original = json.loads(original_state)
    source_path, nonprompt_path = original["blocks"][0]["expected_outputs"]
    calls_before_resume = (validation_root / "block_calls.tsv").read_text(encoding="utf-8")

    Path(nonprompt_path).unlink()
    missing_output = _run(
        "run2_full_CR", output_root, "stub-run2_full_CR", dry_run=True,
        resume=True, environment=_resume_environment(validation_root),
    )
    assert missing_output.returncode != 0
    assert "marks run2_full_CR_block1 successful" in missing_output.stdout
    assert (validation_root / "block_calls.tsv").read_text(encoding="utf-8") == calls_before_resume
    Path(nonprompt_path).write_text("synthetic nonprompt\n", encoding="utf-8")

    for status, source_status, nonprompt_status, expected in (
        ("source_running", "running", "blocked", "ambiguous interrupted source stage"),
        ("nonprompt_running", "ready", "running", "ambiguous interrupted nonprompt stage"),
        ("nonprompt_failed", "ready", "failed", "unrecoverable inline nonprompt state"),
    ):
        state = json.loads(original_state)
        block = state["blocks"][0]
        block["status"] = status
        block["source_status"] = source_status
        block["nonprompt_status"] = nonprompt_status
        state_path.write_text(json.dumps(state), encoding="utf-8")
        result = _run(
            "run2_full_CR", output_root, "stub-run2_full_CR", dry_run=True,
            resume=True, environment=_resume_environment(validation_root),
        )
        assert result.returncode != 0
        assert expected in result.stdout
        assert (validation_root / "block_calls.tsv").read_text(encoding="utf-8") == calls_before_resume

    state = json.loads(original_state)
    state["campaign_tag"] = "incompatible-plan"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    plan_mismatch = _run(
        "run2_full_CR", output_root, "stub-run2_full_CR", dry_run=True,
        resume=True, environment=_resume_environment(validation_root),
    )
    assert plan_mismatch.returncode != 0
    assert "mismatch for campaign_tag" in plan_mismatch.stdout

    state_path.write_text("{malformed\n", encoding="utf-8")
    malformed = _run(
        "run2_full_CR", output_root, "stub-run2_full_CR", dry_run=True,
        resume=True, environment=_resume_environment(validation_root),
    )
    assert malformed.returncode != 0

    state_path.unlink()
    missing_state = _run(
        "run2_full_CR", output_root, "stub-run2_full_CR", dry_run=True,
        resume=True, environment=_resume_environment(validation_root),
    )
    assert missing_state.returncode != 0
    assert "requires campaign state" in missing_state.stdout


@pytest.mark.parametrize("status", ("planned", "source_failed"))
def test_cr_resume_runs_only_a_planned_or_failed_block(tmp_path, status):
    _, output_root, validation_root = _stubbed_run(tmp_path, "run2_full_CR", "success")
    state_path = output_root / ".run2_full_CR_campaign_state.json"
    state = _campaign_state(output_root, "run2_full_CR")
    block = state["blocks"][0]
    block["status"] = status
    block["source_status"] = "planned" if status == "planned" else "failed"
    block["nonprompt_status"] = "blocked"
    block["source_exit_code"] = None
    block["nonprompt_exit_code"] = None
    for output in block["expected_outputs"]:
        Path(output).unlink()
        Path(f"{output}.metadata.json").unlink()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    calls_before_resume = (validation_root / "block_calls.tsv").read_text(encoding="utf-8")

    resumed = _run(
        "run2_full_CR", output_root, "stub-run2_full_CR", dry_run=False,
        resume=True, environment=_resume_environment(validation_root),
    )

    assert resumed.returncode == 0, resumed.stdout
    calls_after_resume = (validation_root / "block_calls.tsv").read_text(encoding="utf-8")
    assert calls_after_resume.count("run2_full_CR_block1\t") == calls_before_resume.count("run2_full_CR_block1\t") + 1
    assert _campaign_state(output_root, "run2_full_CR")["blocks"][0]["status"] == "success"


@pytest.mark.parametrize("profile", ("run2_sr_ext_clike", "run3_sr_ext_clike", "run2_sr_ext20", "run3_sr_ext20", "rebin_fine"))
def test_retired_profiles_reject_before_output(tmp_path, profile):
    output_root = tmp_path / profile
    result = _run(profile, output_root, "retired", dry_run=True)
    assert result.returncode != 0
    assert "unsupported production profile" in result.stdout
    assert not output_root.exists()


@pytest.mark.parametrize("profile", ("run2_sr_ext", "run3_sr_ext"))
def test_selected_canonical_fragment_dry_run_has_no_output_root(tmp_path, profile):
    fragment = f"{profile}_a_3"
    result, output_root, _ = _stubbed_run(
        tmp_path, profile, "success", dry_run=True,
        extra_args=("--selected-fragment", fragment),
    )
    assert result.returncode == 0, result.stdout
    assert re.findall(r"^production_block_id: (\S+)$", result.stdout, re.MULTILINE) == [fragment]
    assert len(_commands(result.stdout)) == 1
    assert "Variables: j0pt b0pt" in result.stdout
    assert not output_root.exists()


@pytest.mark.parametrize(("profile", "np_option"),
                         (("run2_sr_ext", "--np-postprocess=defer"),
                          ("run3_sr_ext", "--np-postprocess=defer")))
def test_processing_core_option_is_opt_in_and_reaches_selected_source(tmp_path, profile, np_option):
    base_options = ("--selected-fragment", f"{profile}_a_3",
                    "--wq-staging-base", str(tmp_path / "unused_staging"),
                    "--wq-quota-root", "/groups/klannon",
                    "--wq-free-warning-bytes", "20000000000",
                    "--wq-min-free-bytes", "10000000000",
                    "--wq-poll-seconds", "5")
    default_result, default_root, _ = _stubbed_run(
        tmp_path, profile, "success", name="default", dry_run=True,
        extra_args=base_options)
    opted_result, opted_root, _ = _stubbed_run(
        tmp_path, profile, "success", name="opted", dry_run=True,
        extra_args=base_options + ("--wq-processing-cores", "1"))
    assert default_result.returncode == opted_result.returncode == 0
    default_source = _commands(default_result.stdout)[0]
    opted_source = _commands(opted_result.stdout)[0]
    assert "--wq-processing-cores" not in default_source
    assert _option_values(opted_source, "--wq-processing-cores", 1) == ["1"]
    for option in ("--wq-guard", "--wq-filepath", "--wq-min-free-bytes"):
        assert option in default_source and option in opted_source
    assert np_option in opted_source
    assert len(_commands(opted_result.stdout)) == 1
    if profile == "run3_sr_ext":
        assert "Separate nonprompt command (not executed by dry-run):" in opted_result.stdout
    assert not default_root.exists() and not opted_root.exists()


def test_processing_core_callback_only_sets_processing_category(monkeypatch):
    from coffea import processor
    from coffea.processor import work_queue_tools

    monkeypatch.syspath_prepend(str(RUN_DIRECTORY))
    spec = importlib.util.spec_from_file_location("run_analysis_processing_core_test", RUN_ANALYSIS)
    run_analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_analysis)
    calls = []
    class fake_queue:
        executor = SimpleNamespace(cores=None, memory=None, disk=None, gpus=None,
                                   resource_monitor="measure", resources_mode="auto",
                                   fast_terminate_workers=0)
        def tune(self, *_):
            pass
        def enable_monitoring(self, *, watchdog):
            assert watchdog is False
        def specify_category_max_resources(self, category, resources):
            calls.append((category, dict(resources)))
        def specify_category_mode(self, *_):
            pass
    queue = fake_queue()
    work_queue_tools.CoffeaWQ._declare_resources(queue)
    assert calls == [(category, {}) for category in
                     ("default", "preprocessing", "processing", "accumulating")]
    assert processor.WorkQueueExecutor.__dataclass_fields__["custom_init"].default is None
    run_analysis._request_one_processing_core(queue)
    assert calls[-1] == ("processing", {"cores": 1})
    assert calls[:4] == [(category, {}) for category in
                         ("default", "preprocessing", "processing", "accumulating")]


def test_accumulating_minimum_composes_after_category_declarations(monkeypatch):
    from coffea.processor import work_queue_tools

    monkeypatch.syspath_prepend(str(RUN_DIRECTORY))
    spec = importlib.util.spec_from_file_location("run_analysis_accumulating_test", RUN_ANALYSIS)
    run_analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_analysis)
    calls = []

    class fake_queue:
        executor = SimpleNamespace(cores=None, memory=None, disk=None, gpus=None,
                                   resource_monitor="measure", resources_mode="auto",
                                   fast_terminate_workers=0)
        def tune(self, *_):
            pass
        def enable_monitoring(self, *, watchdog):
            assert watchdog is False
        def specify_category_max_resources(self, category, resources):
            calls.append(("max", category, dict(resources)))
        def specify_category_min_resources(self, category, resources):
            calls.append(("min", category, dict(resources)))
        def specify_category_mode(self, *_):
            pass

    queue = fake_queue()
    work_queue_tools.CoffeaWQ._declare_resources(queue)
    declaration_count = len(calls)
    run_analysis._configure_wq_categories(
        queue, processing_cores=1, accumulating_memory_mb=80000, accumulating_cores=2,
    )
    assert calls[declaration_count:] == [
        ("max", "processing", {"cores": 1}),
        ("min", "accumulating", {"memory": 80000, "cores": 2}),
    ]


def test_accumulating_cli_forwards_pair_and_rejects_invalid_values(tmp_path):
    profile = "run2_sr_ext"
    base_options = (
        "--selected-fragment", f"{profile}_a_2",
        "--wq-staging-base", str(tmp_path / "unused_staging"),
        "--wq-quota-root", "/groups/klannon",
        "--wq-free-warning-bytes", "20000000000",
        "--wq-min-free-bytes", "10000000000",
        "--wq-poll-seconds", "5",
    )
    selected, output_root, _ = _stubbed_run(
        tmp_path, profile, "success", name="accumulating", dry_run=True,
        extra_args=base_options + (
            "--wq-processing-cores", "1",
            "--wq-accumulating-memory-mb", "80000",
            "--wq-accumulating-cores", "2",
        ),
    )
    assert selected.returncode == 0, selected.stdout
    source = _commands(selected.stdout)[0]
    assert _option_values(source, "--wq-accumulating-memory-mb", 1) == ["80000"]
    assert _option_values(source, "--wq-accumulating-cores", 1) == ["2"]
    assert not output_root.exists()
    for index, invalid in enumerate((
        ("--wq-accumulating-memory-mb", "80000"),
        ("--wq-accumulating-memory-mb", "0", "--wq-accumulating-cores", "2"),
        ("--wq-accumulating-memory-mb", "80000", "--wq-accumulating-cores", "no"),
    )):
        blocked, blocked_root, _ = _stubbed_run(
            tmp_path, profile, "success", name=f"invalid_accumulating_{index}", dry_run=True,
            extra_args=base_options + invalid,
        )
        assert blocked.returncode != 0
        assert not blocked_root.exists()


@pytest.mark.parametrize("profile", ("run2_sr_ext", "run2_full_CR"))
def test_real_work_queue_launcher_requires_guarded_staging(tmp_path, profile):
    output_root = tmp_path / profile
    result = _run(profile, output_root, "fresh", dry_run=False)
    assert result.returncode != 0
    assert "implicit /tmp staging is forbidden" in result.stdout
    assert not output_root.exists()

    staging = tmp_path / "root_backed_staging"
    staging.mkdir(exist_ok=True)
    options = (
        "--wq-staging-base", str(staging),
        "--wq-quota-root", "/groups/klannon",
        "--wq-free-warning-bytes", "20000000000",
        "--wq-min-free-bytes", "10000000000",
        "--wq-poll-seconds", "5",
    )
    root_backed = _run(
        profile, output_root, "fresh", dry_run=False, extra_args=options,
    )
    assert root_backed.returncode != 0
    assert "root-backed manager staging is forbidden" in root_backed.stdout
    assert not output_root.exists()


def test_direct_work_queue_selection_rejects_implicit_staging(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(RUN_ANALYSIS), "missing.json",
            "--skip-topcoffea-data-check", "-x", "work_queue",
            "-p", str(tmp_path / "unused"),
        ],
        cwd=RUN_DIRECTORY, text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "requires explicit guarded --wq-filepath staging" in result.stderr
    assert not (tmp_path / "unused").exists()


def test_retired_work_queue_budget_option_fails_clearly(tmp_path):
    result = _run(
        "run2_sr_ext", tmp_path / "unused", "fresh", dry_run=True,
        extra_args=("--wq-stop-bytes", "250000000000"),
    )
    assert result.returncode != 0
    assert "retired Work Queue staging-budget option" in result.stdout
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("profile", ("run2_run3_full", "run2_run3_full_CR"))
def test_combined_work_queue_requires_staging_before_shared_root(tmp_path, profile):
    output_root = tmp_path / profile
    result = _run(profile, output_root, "fresh", dry_run=False)
    assert result.returncode != 0
    assert "combined Work Queue production requires --wq-staging-base" in result.stdout
    assert not output_root.exists()


def test_guarded_sr_failure_stops_before_next_fragment(tmp_path):
    profile = "run2_sr_ext"
    result, output_root, validation_root = _stubbed_run(
        tmp_path, profile, "guard_failure_second",
        extra_args=(
            "--wq-staging-base", str(tmp_path / "owned_staging"),
            "--wq-quota-root", "/groups/klannon",
            "--wq-free-warning-bytes", "20000000000",
            "--wq-min-free-bytes", "10000000000",
            "--wq-poll-seconds", "5",
        ),
    )
    assert result.returncode != 0
    state = _campaign_state(output_root, profile)
    assert state["blocks"][1]["source_status"] == "failed"
    assert state["blocks"][2]["status"] == "planned"
    assert not (validation_root / f"{profile}_a_3.started").exists()


def test_guarded_run2_resume_continues_from_e1_nonprompt_to_e2(tmp_path):
    profile = "run2_sr_ext"
    validation_root = tmp_path / "validation"
    validation_root.mkdir()
    backend = validation_root / "backend.sh"
    _write_backend(backend)
    output_root = validation_root / "output"
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    guard_options = (
        "--wq-staging-base", str(staging_root),
        "--wq-quota-root", "/groups/klannon",
        "--wq-free-warning-bytes", "20000000000",
        "--wq-min-free-bytes", "10000000000",
        "--wq-poll-seconds", "5",
    )
    environment = _clean_environment()
    environment.update({
        "SRPLOT009_VALIDATION_BACKEND": str(backend),
        "SRPLOT009_VALIDATION_ROOT": str(validation_root),
        "SRPLOT009_VALIDATION_SCENARIO": "success",
    })
    tag = "resume-e1-e2"
    prepared = _run(
        profile, output_root, tag, dry_run=False, environment=environment,
        extra_args=guard_options,
    )
    assert prepared.returncode == 0, prepared.stdout
    state_path = output_root / f".{profile}_campaign_state.json"
    state = _campaign_state(output_root, profile)
    assert [block["id"] for block in state["blocks"][-2:]] == [
        f"{profile}_e_1", f"{profile}_e_2"
    ]
    # Preserve the completed earlier receipts, then model the native source-ready
    # and planned statuses for the two fragments under test.
    e1, e2 = state["blocks"][-2:]
    for artifact in (e1["expected_np_path"], e2["expected_nominal_path"],
                     e2["expected_np_path"]):
        Path(artifact).unlink()
        Path(f"{artifact}.metadata.json").unlink()
    e1["status"] = "source_ready"
    e1["exit_code"] = None
    e1["nonprompt_status"] = "planned"
    e1["nonprompt_exit_code"] = None
    e1["nonprompt_signal"] = None
    e1["nonprompt_command_argv"] = None
    e1["nonprompt_started_at_utc"] = None
    e1["nonprompt_ended_at_utc"] = None
    e1["nonprompt_duration_seconds"] = None
    e1["transitions"] = [
        transition for transition in e1["transitions"]
        if transition["stage"] != "nonprompt"
    ]
    e1["last_transition_detail"] = "synthetic_source_ready_fixture"
    e1.pop("nonprompt_receipt")
    e2_spec = state["sr_reco_plan_revisions"][-1]["blocks"][-1]
    state["blocks"][-1] = sr_manifest.new_pending_block(e2_spec)
    state["campaign_status"] = "blocked_incomplete"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    assert sr_manifest.check_state(_campaign_state(output_root, profile))
    before = _campaign_state(output_root, profile)
    assert all(block["status"] == "success" for block in before["blocks"][:-2])
    e1_before = before["blocks"][-2]
    assert e1_before["source_status"] == "ready"
    assert before["blocks"][-1]["status"] == "planned"
    source_identity = (
        copy.deepcopy(e1_before["source_receipt"]),
        copy.deepcopy(e1_before["source_runtime"]),
        copy.deepcopy(e1_before["source_runtime_attempts"]),
        e1_before["source_command_argv"],
    )
    source_path = Path(e1_before["expected_nominal_path"])
    source_bytes = source_path.read_bytes()
    source_sidecar_bytes = Path(f"{source_path}.metadata.json").read_bytes()
    prepared_state = state_path.read_bytes()
    for resume_scenario, expected_e1_status in (
        ("success", "success"),
        ("sr_np_failure_e1", "nonprompt_failed"),
        ("sr_np_partial_failure_e1", "nonprompt_failed"),
    ):
        state_path.write_bytes(prepared_state)
        for artifact in (e1_before["expected_np_path"],
                         before["blocks"][-1]["expected_nominal_path"],
                         before["blocks"][-1]["expected_np_path"]):
            for path in (Path(artifact), Path(f"{artifact}.metadata.json")):
                if path.exists():
                    path.unlink()
        assert sr_manifest.check_state(_campaign_state(output_root, profile))
        source_calls_before = (validation_root / "block_calls.tsv").read_text().splitlines()
        nonprompt_calls_before = (validation_root / "nonprompt_calls.tsv").read_text().splitlines()
        environment["SRPLOT009_VALIDATION_SCENARIO"] = resume_scenario
        resumed = _run(
            profile, output_root, tag, dry_run=False, resume=True,
            environment=environment, extra_args=guard_options,
        )
        after = _campaign_state(output_root, profile)
        e1_after, e2_after = after["blocks"][-2:]
        source_calls_after = (validation_root / "block_calls.tsv").read_text().splitlines()
        nonprompt_calls_after = (validation_root / "nonprompt_calls.tsv").read_text().splitlines()
        assert source_calls_after[len(source_calls_before):][0].startswith(f"{profile}_e_2\t")
        assert len(source_calls_after) == len(source_calls_before) + 1
        assert nonprompt_calls_after[len(nonprompt_calls_before):] == [
            f"{profile}_e_1", f"{profile}_e_2"
        ]
        assert (validation_root / f"{profile}_e_2.started").exists()
        assert (e1_after["source_receipt"], e1_after["source_runtime"],
                e1_after["source_runtime_attempts"], e1_after["source_command_argv"]) == source_identity
        assert source_path.read_bytes() == source_bytes
        assert Path(f"{source_path}.metadata.json").read_bytes() == source_sidecar_bytes
        assert all(block["status"] == "success" for block in after["blocks"][:-2])
        assert e1_after["status"] == expected_e1_status
        assert e2_after["status"] == "success"
        if expected_e1_status == "success":
            assert resumed.returncode == 0, resumed.stdout
            assert after["campaign_status"] == "success"
        else:
            assert resumed.returncode == 1, resumed.stdout
            assert e1_after["nonprompt_exit_code"] == 137
            assert e1_after["source_status"] == "ready"
            assert after["campaign_status"] == "complete_with_known_failures"
            assert after["known_failed_block_count"] == 1
        if resume_scenario == "sr_np_partial_failure_e1":
            assert Path(e1_after["expected_np_path"]).read_bytes() == b"partial synthetic nonprompt\n"
            assert e1_after.get("nonprompt_receipt") is None


def test_guarded_run3_nonprompt_failure_still_starts_later_source(tmp_path):
    profile = "run3_sr_ext"
    result, output_root, validation_root = _stubbed_run(
        tmp_path, profile, "sr_np_failure_e1",
        extra_args=(
            "--wq-staging-base", str(tmp_path / "staging"),
            "--wq-quota-root", "/groups/klannon",
            "--wq-free-warning-bytes", "20000000000",
            "--wq-min-free-bytes", "10000000000",
            "--wq-poll-seconds", "5",
        ),
    )
    assert result.returncode == 1, result.stdout
    state = _campaign_state(output_root, profile)
    assert state["blocks"][-2]["status"] == "nonprompt_failed"
    assert state["blocks"][-2]["source_status"] == "ready"
    assert state["blocks"][-1]["status"] == "success"
    assert (validation_root / f"{profile}_e_2.started").exists()
    assert state["campaign_status"] == "complete_with_known_failures"


def test_staging_guard_rejects_root_and_symlink_paths(tmp_path):
    with pytest.raises(staging_guard.staging_guard_error, match="root temporary"):
        staging_guard._validate_base(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(staging_guard.staging_guard_error, match="symlink"):
        staging_guard._validate_base(link)


@pytest.mark.parametrize(("measurement_error", "error_text"),
                         ((OSError("quota unavailable"), "OSError: quota unavailable"),
                          (subprocess.TimeoutExpired(["quota"], 15), "TimeoutExpired: Command")))
def test_staging_guard_warns_and_keeps_running_after_telemetry_failures(
    tmp_path, monkeypatch, capsys, measurement_error, error_text
):
    path = tmp_path / "owned"
    path.mkdir()
    pid = os.getpid()
    monkeypatch.setattr(staging_guard.os, "getppid", lambda: pid)
    start = staging_guard._process_start(pid)
    marker = path / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(path),
                                  "owner_uid": os.getuid(), "owner_pid": pid,
                                  "owner_start": start, "owner_inode": path.stat().st_ino}))
    identity = (path.stat().st_ino, marker.stat().st_ino)
    signalled = []
    monkeypatch.setattr(staging_guard.os, "kill", lambda target, sig: signalled.append((target, sig)))
    def no_tree_scan(*_):
        raise AssertionError("runtime monitoring must not enumerate staging")
    monkeypatch.setattr(staging_guard, "_charged_tree", no_tree_scan, raising=False)
    real_scandir = staging_guard.os.scandir
    monkeypatch.setattr(staging_guard.os, "scandir", lambda current: no_tree_scan()
                        if os.fspath(current) == str(path) else real_scandir(current))
    measurements = iter((measurement_error, (20000000000, 0), (20000000000, 2000)))
    def measure(*_):
        value = next(measurements)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(staging_guard, "_space", measure)
    class stop_event:
        polls = 0
        def is_set(self):
            return self.polls >= 3
        def wait(self, _):
            self.polls += 1
            return self.is_set()
    degraded = SimpleNamespace(value=False)
    degraded.set = lambda: setattr(degraded, "value", True)
    staging_guard._watch(path, tmp_path, "/groups/klannon", pid, start,
                         30000000000, 10000000000, 1, stop_event(), *identity, degraded)
    assert signalled == []
    assert degraded.value
    output = capsys.readouterr().out
    assert error_text in output
    assert "available_bytes=20000000000 free_inodes=0" in output
    assert "available_bytes=20000000000" in output
    assert "measurement_failed" not in output
    assert json.loads(marker.read_text())["status"] == "running"


@pytest.mark.parametrize(("available_bytes", "should_signal"), (
    (10000000000, True),
    (0, True),
    (10000000001, False),
))
def test_staging_guard_signals_only_at_or_below_critical_free_space(
    tmp_path, monkeypatch, capsys, available_bytes, should_signal
):
    path = tmp_path / "owned"
    path.mkdir()
    pid = os.getpid()
    monkeypatch.setattr(staging_guard.os, "getppid", lambda: pid)
    start = staging_guard._process_start(pid)
    marker = path / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(path),
                                  "owner_uid": os.getuid(), "owner_pid": pid,
                                  "owner_start": start, "owner_inode": path.stat().st_ino}))
    signals = []
    monkeypatch.setattr(staging_guard.os, "kill",
                        lambda target, sig: signals.append((target, sig)))
    monkeypatch.setattr(staging_guard, "_space", lambda *_: (available_bytes, 2000))

    class stop_event:
        polls = 0
        def is_set(self):
            return self.polls >= 2
        def wait(self, _):
            self.polls += 1
            return self.is_set()

    staging_guard._watch(path, tmp_path, "/groups/klannon", pid, start,
                         30000000000, 10000000000, 1, stop_event(),
                         path.stat().st_ino, marker.stat().st_ino)
    assert signals == ([(pid, staging_guard.signal.SIGTERM)] if should_signal else [])
    metadata = json.loads(marker.read_text())
    assert metadata["status"] == "running"
    if should_signal:
        assert metadata["reason"] == "critical_free_space"
        assert metadata["critical_free_space"]["available_bytes"] == available_bytes
        assert metadata["critical_free_space"]["min_free_bytes"] == 10000000000
        assert metadata["critical_free_space"]["observed_at"]
        assert metadata["critical_free_space"]["observed_at_unix_ns"] > 0
        assert "requesting SIGTERM for verified manager pid" in capsys.readouterr().out
    else:
        assert "reason" not in metadata
        capsys.readouterr()


def test_staging_guard_does_not_signal_when_critical_marker_logging_fails(
    tmp_path, monkeypatch
):
    path = tmp_path / "owned"
    path.mkdir()
    pid = os.getpid()
    monkeypatch.setattr(staging_guard.os, "getppid", lambda: pid)
    start = staging_guard._process_start(pid)
    marker = path / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(path),
                                  "owner_uid": os.getuid(), "owner_pid": pid,
                                  "owner_start": start, "owner_inode": path.stat().st_ino}))
    signals = []
    monkeypatch.setattr(staging_guard.os, "kill",
                        lambda target, sig: signals.append((target, sig)))
    monkeypatch.setattr(staging_guard, "_space", lambda *_: (0, 2000))
    monkeypatch.setattr(staging_guard.os, "fsync",
                        lambda _: (_ for _ in ()).throw(OSError("marker sync failed")))

    class stop_event:
        def is_set(self):
            return False

    with pytest.raises(staging_guard.staging_guard_error,
                       match="critical free-space signal was not safely completed"):
        staging_guard._watch(path, tmp_path, "/groups/klannon", pid, start,
                             30000000000, 10000000000, 1, stop_event(),
                             path.stat().st_ino, marker.stat().st_ino)
    assert signals == []
    assert json.loads(marker.read_text())["reason"] == "critical_free_space"


def test_staging_guard_missing_telemetry_file_does_not_signal(tmp_path, monkeypatch, capsys):
    path = tmp_path / "owned"
    path.mkdir()
    pid = os.getpid()
    monkeypatch.setattr(staging_guard.os, "getppid", lambda: pid)
    start = staging_guard._process_start(pid)
    marker = path / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(path),
                                  "owner_uid": os.getuid(), "owner_pid": pid,
                                  "owner_start": start, "owner_inode": path.stat().st_ino}))
    signalled = []
    monkeypatch.setattr(staging_guard.os, "kill", lambda target, sig: signalled.append((target, sig)))
    monkeypatch.setattr(staging_guard, "_space", lambda *_: (_ for _ in ()).throw(FileNotFoundError("quota output vanished")))

    class stop_event:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, _):
            self.stopped = True
            return True

    staging_guard._watch(path, tmp_path, "/groups/klannon", pid, start,
                         20000000000, 10000000000, 1, stop_event(), path.stat().st_ino,
                         marker.stat().st_ino)
    assert signalled == []
    assert json.loads(marker.read_text())["status"] == "running"
    assert "quota output vanished" in capsys.readouterr().out


@pytest.mark.parametrize("failure_point", ("marker_read", "proc_read"))
def test_staging_guard_transient_identity_read_does_not_signal(
    tmp_path, monkeypatch, capsys, failure_point
):
    path = tmp_path / "owned"
    path.mkdir()
    pid = os.getpid()
    monkeypatch.setattr(staging_guard.os, "getppid", lambda: pid)
    start = staging_guard._process_start(pid)
    marker = path / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(path),
                                  "owner_uid": os.getuid(), "owner_pid": pid,
                                  "owner_start": start, "owner_inode": path.stat().st_ino}))
    original = (staging_guard._verify_owned_tree if failure_point == "marker_read"
                else staging_guard._process_start)
    calls = SimpleNamespace(count=0)
    def transient(*args):
        calls.count += 1
        if calls.count == 1:
            if failure_point == "marker_read":
                raise PermissionError("marker read race")
            raise staging_guard.staging_guard_error("manager process identity cannot be read")
        return original(*args)
    monkeypatch.setattr(staging_guard,
                        "_verify_owned_tree" if failure_point == "marker_read" else "_process_start",
                        transient)
    monkeypatch.setattr(staging_guard, "_space", lambda *_: (0, 2000))
    signalled = []
    monkeypatch.setattr(staging_guard.os, "kill", lambda pid, sig: signalled.append((pid, sig)))
    class stop_event:
        polls = 0
        def is_set(self):
            return self.polls >= 1
        def wait(self, _):
            self.polls += 1
            return self.is_set()
    degraded = SimpleNamespace(value=False)
    degraded.set = lambda: setattr(degraded, "value", True)
    staging_guard._watch(path, tmp_path, "/groups/klannon", pid, start,
                         30000000000, 10000000000, 1, stop_event(), path.stat().st_ino,
                         marker.stat().st_ino, degraded)
    assert degraded.value and signalled == []
    assert json.loads(marker.read_text())["status"] == "running"
    assert ("marker read race" if failure_point == "marker_read" else
            "manager process identity cannot be read") in capsys.readouterr().out


def test_staging_cleanup_is_owned_and_preserves_ambiguous_files(tmp_path, monkeypatch):
    class fake_event:
        def set(self):
            pass
    class fake_watchdog:
        exitcode = 0
        def join(self, timeout):
            pass
        def is_alive(self):
            return False
    monkeypatch.setattr(staging_guard, "_local_readers", lambda _: False)
    owned = tmp_path / "owned"
    owned.mkdir()
    metadata = {"status": "running", "path": str(owned), "owner_uid": os.getuid(),
                "owner_pid": 99999999, "owner_start": "never",
                "owner_inode": owned.stat().st_ino}
    (owned / "guard.json").write_text(json.dumps(metadata))
    (owned / "out_1.p").write_bytes(b"unfinished")
    staging_guard.finish_guard(owned, owned.stat().st_ino, fake_event(), fake_watchdog(), True)
    with pytest.raises(staging_guard.staging_guard_error, match="unexpected"):
        staging_guard.cleanup_owned_staging(owned)
    assert (owned / "out_1.p").exists()
    assert json.loads((owned / "guard.json").read_text())["status"] == "cleanup_needed"
    clean = tmp_path / "clean"
    clean.mkdir()
    metadata.update(path=str(clean), owner_inode=clean.stat().st_ino)
    (clean / "guard.json").write_text(json.dumps(metadata))
    staging_guard.finish_guard(clean, clean.stat().st_ino, fake_event(), fake_watchdog(), True)
    staging_guard.cleanup_owned_staging(clean)
    assert not clean.exists()

    completed = tmp_path / "completed"
    completed.mkdir()
    metadata.update(path=str(completed), owner_inode=completed.stat().st_ino)
    (completed / "guard.json").write_text(json.dumps(metadata))
    temporary = completed / "tmpabcwq-tmp-"
    temporary.mkdir(mode=0o700)
    (temporary / "out_p_1.p").write_bytes(b"intermediate")
    staging_guard.finish_guard(completed, completed.stat().st_ino, fake_event(), fake_watchdog(), True)
    staging_guard.cleanup_owned_staging(completed)
    assert not completed.exists()

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    metadata.update(path=str(foreign), owner_inode=foreign.stat().st_ino)
    (foreign / "guard.json").write_text(json.dumps(metadata))
    temporary = foreign / "tmpabcwq-tmp-"
    temporary.mkdir(mode=0o700)
    (temporary / "out_p_1.p").write_bytes(b"owned")
    (temporary / "unrecognized.txt").write_bytes(b"preserve")
    staging_guard.finish_guard(foreign, foreign.stat().st_ino, fake_event(), fake_watchdog(), True)
    with pytest.raises(staging_guard.staging_guard_error, match="unknown or foreign"):
        staging_guard.cleanup_owned_staging(foreign)
    assert (temporary / "out_p_1.p").exists()
    assert (temporary / "unrecognized.txt").exists()
    assert json.loads((foreign / "guard.json").read_text())["status"] == "cleanup_needed"

    failed = tmp_path / "failed"
    failed.mkdir()
    metadata.update(path=str(failed), owner_inode=failed.stat().st_ino)
    (failed / "guard.json").write_text(json.dumps(metadata))
    (failed / "out_p_1.p").write_bytes(b"unfinished")
    staging_guard.finish_guard(failed, failed.stat().st_ino, fake_event(), fake_watchdog(), False)
    with pytest.raises(staging_guard.staging_guard_error, match="completed invocation"):
        staging_guard.cleanup_owned_staging(failed)
    assert (failed / "out_p_1.p").exists()


def test_staging_preflight_rejects_insufficient_free_space_and_collision(tmp_path, monkeypatch):
    monkeypatch.setattr(staging_guard, "_validate_base", lambda _: tmp_path)
    monkeypatch.setattr(staging_guard.os.path, "commonpath", lambda _: "/groups/klannon")
    for available_bytes in (9999999999, 10000000000):
        monkeypatch.setattr(staging_guard, "_space", lambda *_, value=available_bytes: (value, 2000))
        with pytest.raises(staging_guard.staging_guard_error, match="minimum free space"):
            staging_guard.start_guard(tmp_path, "attempt", "/groups/klannon",
                                      20000000000, 10000000000, 1)
    assert not (tmp_path / "attempt").exists()
    monkeypatch.setattr(staging_guard, "_space", lambda *_: (20000000000, 2000))
    (tmp_path / "attempt").mkdir()
    with pytest.raises(FileExistsError):
        staging_guard.start_guard(tmp_path, "attempt", "/groups/klannon", 20000000000, 10000000000, 1)


def test_staging_preflight_rejects_unknown_space_and_invalid_ownership(tmp_path, monkeypatch):
    monkeypatch.setattr(staging_guard, "_validate_base", lambda _: tmp_path)
    monkeypatch.setattr(staging_guard.os.path, "commonpath", lambda _: "/groups/klannon")
    monkeypatch.setattr(staging_guard, "_space", lambda *_: (_ for _ in ()).throw(OSError("quota unavailable")))
    with pytest.raises(OSError, match="quota unavailable"):
        staging_guard.start_guard(tmp_path, "attempt", "/groups/klannon",
                                  20000000000, 10000000000, 1)
    assert not (tmp_path / "attempt").exists()
    with pytest.raises(staging_guard.staging_guard_error, match="quota root"):
        staging_guard.start_guard(tmp_path, "attempt", "/invalid",
                                  20000000000, 10000000000, 1)
    monkeypatch.setattr(staging_guard, "_validate_base", lambda _: (_ for _ in ()).throw(
        staging_guard.staging_guard_error("staging base ownership or write access is invalid")))
    with pytest.raises(staging_guard.staging_guard_error, match="ownership"):
        staging_guard.start_guard(tmp_path, "attempt", "/groups/klannon",
                                  20000000000, 10000000000, 1)
    assert not (tmp_path / "attempt").exists()


@pytest.mark.parametrize(("watchdog_exitcode", "reason"),
                         ((0, "before requested shutdown"), (1, "watchdog exited")))
def test_lost_watchdog_blocks_completion_without_signalling_manager(
    tmp_path, monkeypatch, watchdog_exitcode, reason
):
    monkeypatch.setattr(staging_guard, "_validate_base", lambda _: tmp_path)
    monkeypatch.setattr(staging_guard.os.path, "commonpath", lambda _: "/groups/klannon")
    monkeypatch.setattr(staging_guard, "_space", lambda *_: (20000000000, 2000))
    signalled = []
    monkeypatch.setattr(staging_guard.os, "kill", lambda pid, sig: signalled.append((pid, sig)))
    class fake_event:
        def set(self):
            pass
        def is_set(self):
            return False
        def wait(self, timeout):
            return False
    class fake_process:
        checks = 0
        exitcode = watchdog_exitcode
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
        def join(self, timeout):
            pass
        def is_alive(self):
            self.checks += 1
            return self.checks == 1
    class fake_context:
        def Event(self):
            return fake_event()
        def Process(self, **kwargs):
            return fake_process(**kwargs)
    monkeypatch.setattr(staging_guard.multiprocessing, "get_context", lambda _: fake_context())
    state = staging_guard.start_guard(tmp_path, "attempt", "/groups/klannon",
                                      20000000000, 10000000000, 1)
    with pytest.raises(staging_guard.staging_guard_error, match=reason):
        staging_guard.finish_guard(*state, succeeded=True)
    assert signalled == []
    assert json.loads((tmp_path / "attempt" / "guard.json").read_text())["status"] == "running"


def test_finish_guard_waits_for_slow_valid_quota_poll_and_acknowledges_shutdown(tmp_path):
    owned = tmp_path / "owned"
    owned.mkdir()
    pid = os.getpid()
    start = staging_guard._process_start(pid)
    marker = owned / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(owned),
                                  "owner_uid": os.getuid(), "owner_pid": pid,
                                  "owner_start": start, "owner_inode": owned.stat().st_ino}))

    class event:
        def __init__(self, value=False):
            self.value = value
        def set(self):
            self.value = True
        def is_set(self):
            return self.value

    class slow_watchdog:
        exitcode = 0
        owner_pid = pid
        owner_start = start
        marker_inode = marker.stat().st_ino
        shutdown_ack = event()
        telemetry_degraded = event()
        def join(self, timeout):
            self.join_timeout = timeout
            # A bounded in-flight quota call may consume 15 seconds; model its
            # successful completion without sleeping in the test.
            assert timeout > 15
            self.shutdown_ack.set()
        def is_alive(self):
            return False

    watcher = slow_watchdog()
    staging_guard.finish_guard(owned, owned.stat().st_ino, event(), watcher, True)
    assert watcher.join_timeout == 30
    completed = json.loads(marker.read_text())
    assert completed["status"] == "completed_pending_receipt"
    assert completed["telemetry_degraded"] is False


@pytest.mark.parametrize("watcher_state", ("missing_ack", "still_alive"))
def test_finish_guard_rejects_missing_ack_or_stuck_watchdog(tmp_path, watcher_state):
    owned = tmp_path / "owned"
    owned.mkdir()
    pid = os.getpid()
    start = staging_guard._process_start(pid)
    marker = owned / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(owned),
                                  "owner_uid": os.getuid(), "owner_pid": pid,
                                  "owner_start": start, "owner_inode": owned.stat().st_ino}))

    class event:
        def set(self):
            pass
        def is_set(self):
            return watcher_state != "missing_ack"

    class watcher:
        exitcode = 0
        owner_pid = pid
        owner_start = start
        marker_inode = marker.stat().st_ino
        shutdown_ack = event()
        telemetry_degraded = event()
        def join(self, timeout):
            assert timeout == 30
        def is_alive(self):
            return watcher_state == "still_alive"

    reason = "did not stop cleanly" if watcher_state == "still_alive" else "shutdown"
    with pytest.raises(staging_guard.staging_guard_error, match=reason):
        staging_guard.finish_guard(owned, owned.stat().st_ino, event(), watcher(), True)
    assert json.loads(marker.read_text())["status"] == "running"


def test_guard_marker_mutation_blocks_completion_and_cleanup(tmp_path, monkeypatch):
    owned = tmp_path / "owned"
    owned.mkdir()
    marker = owned / "guard.json"
    pid = os.getpid()
    monkeypatch.setattr(staging_guard.os, "getppid", lambda: pid)
    start = staging_guard._process_start(pid)
    metadata = {"status": "running", "path": str(owned), "owner_uid": os.getuid(),
                "owner_pid": pid, "owner_start": start, "owner_inode": owned.stat().st_ino}
    marker.write_text(json.dumps(metadata))
    metadata["owner_inode"] += 1
    marker.write_text(json.dumps(metadata))
    signalled = []
    monkeypatch.setattr(staging_guard.os, "kill", lambda pid, sig: signalled.append((pid, sig)))
    monkeypatch.setattr(staging_guard, "_space", lambda *_: (20000000000, 2000))
    class stop_event:
        def is_set(self):
            return False
    with pytest.raises(staging_guard.staging_guard_error, match="owner marker changed"):
        staging_guard._watch(owned, tmp_path, "/groups/klannon", pid, start,
                         30000000000, 10000000000, 1, stop_event(), owned.stat().st_ino,
                             marker.stat().st_ino)
    class fake_watchdog:
        exitcode = 1
        owner_pid = pid
        owner_start = start
        marker_inode = marker.stat().st_ino
        def join(self, timeout):
            pass
        def is_alive(self):
            return False
    with pytest.raises(staging_guard.staging_guard_error, match="owner marker changed"):
        staging_guard.finish_guard(owned, owned.stat().st_ino,
                                   SimpleNamespace(set=lambda: None), fake_watchdog(), True)
    with pytest.raises(staging_guard.staging_guard_error, match="completed invocation"):
        staging_guard.cleanup_owned_staging(owned)
    assert owned.exists() and signalled == []


def test_transient_telemetry_degradation_keeps_completed_marker_distinct(tmp_path):
    owned = tmp_path / "owned"
    owned.mkdir()
    marker = owned / "guard.json"
    metadata = {"status": "running", "path": str(owned), "owner_uid": os.getuid(),
                "owner_pid": os.getpid(),
                "owner_start": staging_guard._process_start(os.getpid()),
                "owner_inode": owned.stat().st_ino}
    marker.write_text(json.dumps(metadata))
    class fake_event:
        def set(self):
            pass
        def is_set(self):
            return True
    class fake_watchdog:
        exitcode = 0
        telemetry_degraded = fake_event()
        def join(self, timeout):
            pass
        def is_alive(self):
            return False
    staging_guard.finish_guard(owned, owned.stat().st_ino, fake_event(), fake_watchdog(), True)
    completed = json.loads(marker.read_text())
    assert completed["status"] == "completed_pending_receipt"
    assert completed["telemetry_degraded"] is True
    assert "reason" not in completed


def test_cleanup_refusal_does_not_relabel_source_receipt_as_failed():
    launcher = RUN_CR.read_text(encoding="utf-8")
    assert "source failed none staging_cleanup_needed" not in launcher
    assert '"source" "failed" "none" "staging_cleanup_needed"' not in launcher
    assert launcher.count("source receipt remains valid and staging requires review") == 2


def test_cleanup_accepts_more_than_100000_recognized_files_without_materializing_them(tmp_path, monkeypatch):
    owned = tmp_path / "owned"
    owned.mkdir()
    temporary = owned / "tmpabcwq-tmp-"
    temporary.mkdir(mode=0o700)
    metadata = {"status": "completed_pending_receipt", "path": str(owned),
                "owner_uid": os.getuid(), "owner_pid": 99999999,
                "owner_start": "never", "owner_inode": owned.stat().st_ino}
    (owned / "guard.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(staging_guard, "_local_readers", lambda _: False)
    real_scandir = staging_guard.os.scandir
    real_unlink = Path.unlink
    removed = SimpleNamespace(count=0)
    class fake_entry:
        def __init__(self, index):
            self.name = f"out_p_{index}.p"
            self.path = str(temporary / self.name)
        def stat(self, follow_symlinks=False):
            return SimpleNamespace(st_mode=stat.S_IFREG | 0o600,
                                   st_uid=os.getuid(), st_nlink=1)
    class fake_scan:
        def __enter__(self):
            return (fake_entry(index) for index in range(100001))
        def __exit__(self, *_):
            return False
    monkeypatch.setattr(staging_guard.os, "scandir", lambda path: fake_scan()
                        if os.fspath(path) == str(temporary) else real_scandir(path))
    def unlink(path, *args, **kwargs):
        if path.parent == temporary:
            removed.count += 1
        else:
            real_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", unlink)
    staging_guard.cleanup_owned_staging(owned)
    assert removed.count == 100001
    assert not owned.exists()
