import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from functools import lru_cache
from pathlib import Path

import pytest
from topcoffea.modules import remote_environment


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPOSITORY_ROOT / "analysis" / "topeft_run2"
RUN_CR = ANALYSIS_DIR / "run_cr.sh"
STATE_FILENAME = ".run3_full_campaign_state.json"
YEARS = "2022 2022EE 2023 2023BPix"
EXPECTED_BLOCKS = [
    ("run3_full_a", "2l 2lss_1tau 2los_1tau 4l", "njets lj0pt ptz ptz_wtau lt"),
    ("run3_full_b", "3l_m_offZ", "njets lj0pt ptll lt"),
    ("run3_full_c", "3l_p_offZ", "njets lj0pt ptll lt"),
    ("run3_full_d", "3l_onZ_tau", "njets lj0pt ptz lt"),
    ("run3_full_e", "3l_fwd", "njets lj0pt ptz lt"),
]
REBIN_FINE_BLOCKS = [
    ("run2_a", "2016APV 2016 2017 2018", "2lss_1tau 3l_m_offZ", "lj0pt ptll ptz_wtau"),
    ("run2_b", "2016APV 2016 2017 2018", "3l_p_offZ 3l_onZ_tau", "lj0pt ptz ptll"),
    ("run2_c", "2016APV 2016 2017 2018", "3l_fwd", "lt"),
    ("run3_a", YEARS, "2lss_1tau 3l_m_offZ", "lj0pt ptll ptz_wtau"),
    ("run3_b", YEARS, "3l_p_offZ 3l_onZ_tau", "lj0pt ptz ptll"),
    ("run3_c", YEARS, "3l_fwd", "lt"),
]


def _run(*args):
    return subprocess.run(
        [str(RUN_CR), *args],
        cwd=ANALYSIS_DIR,
        check=False,
        capture_output=True,
        text=True,
    )


def _commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()


@lru_cache(maxsize=1)
def _current_environment_request():
    return remote_environment.resolve_environment_request(
        extra_pip_local={"topeft": ["topeft", "setup.py"]},
        unstaged="fail",
    )


def _write_env(tmp_path, content=b"synthetic current environment", *, current=True):
    path = tmp_path / "verified_env.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("environment.txt")
        entry.size = len(content)
        archive.addfile(entry, io.BytesIO(content))
    archive_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    request = _current_environment_request() if current else {
        "environment_fingerprint": "stale-environment-fingerprint",
        "python_version": "3.9.23",
        "resolved_environment_spec": {"conda": {"packages": []}, "pip": []},
        "resolved_environment_spec_fingerprint": "stale-spec-fingerprint",
        "editable_packages": [
            {
                "package_name": "topcoffea",
                "git_commit": "stale-topcoffea-commit",
                "watched_source_fingerprint": "stale-topcoffea-source",
            }
        ],
    }
    remote_environment.write_archive_manifest(str(path), request)
    return path


def _planned_blocks(output_dir, campaign_tag):
    blocks = []
    for block_id, categories, histograms in EXPECTED_BLOCKS:
        output_tag = "{}_{}_{}".format(
            campaign_tag,
            categories.replace(" ", "-"),
            histograms.replace(" ", "-"),
        )
        output_name = "{}SRs_{}".format(YEARS.replace(" ", "-"), output_tag)
        nominal = str(output_dir / f"{output_name}.pkl.gz")
        nonprompt = str(output_dir / f"{output_name}_np.pkl.gz")
        blocks.append(
            {
                "id": block_id,
                "years": YEARS.split(),
                "category_groups": categories.split(),
                "histograms": histograms.split(),
                "output_tag": output_tag,
                "output_name": output_name,
                "expected_outputs": [nominal, nonprompt],
                "expected_nominal_path": nominal,
                "expected_np_path": nonprompt,
                "status": "planned",
                "exit_code": None,
                "last_transition_utc": "2026-01-01T00:00:00Z",
                "last_transition_detail": "campaign_initialized",
                "transitions": [
                    {
                        "timestamp_utc": "2026-01-01T00:00:00Z",
                        "status": "planned",
                        "exit_code": None,
                        "detail": "campaign_initialized",
                    }
                ],
            }
        )
    return blocks


def _write_state(output_dir, campaign_tag, env_file, *, status="planned"):
    blocks = _planned_blocks(output_dir, campaign_tag)
    source_status = "ready" if status == "success" else "planned"
    nonprompt_status = "success" if status == "success" else "blocked"
    for block in blocks:
        block["status"] = status
        block["source_status"] = source_status
        block["source_exit_code"] = 0 if status == "success" else None
        block["nonprompt_status"] = nonprompt_status
        block["nonprompt_exit_code"] = 0 if status == "success" else None
        block["transitions"][-1]["status"] = status
    manifest = json.loads(
        env_file.with_name(f"{env_file.name}.manifest.json").read_text(encoding="utf-8")
    )
    topcoffea = next(
        item for item in manifest["editable_packages"] if item["package_name"] == "topcoffea"
    )
    state = {
        "schema_version": 3,
        "production_profile": "run3_full",
        "campaign_tag": campaign_tag,
        "output_dir": str(output_dir),
        "topeft_git_commit": _commit(),
        "env_file": str(env_file.resolve()),
        "env_file_sha256": hashlib.sha256(env_file.read_bytes()).hexdigest(),
        "environment_fingerprint": manifest["environment_fingerprint"],
        "topcoffea_git_commit": topcoffea["git_commit"],
        "topcoffea_relevant_source_fingerprint": topcoffea["watched_source_fingerprint"],
        "ttgamma_sample_role_policy": "split",
        "do_systs": True,
        "do_np": True,
        "created_at_utc": "2026-01-01T00:00:00Z",
        "updated_at_utc": "2026-01-01T00:00:00Z",
        "blocks": blocks,
    }
    state_path = output_dir / STATE_FILENAME
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return state_path, state


def _resume_args(output_dir, env_file, campaign_tag="run3_complete"):
    return (
        "--production-profile",
        "run3_full",
        "--resume",
        "--dry-run",
        "--output-dir",
        str(output_dir),
        "--campaign-tag",
        campaign_tag,
        "--env-file",
        str(env_file),
    )


def _state_tool_source():
    source = RUN_CR.read_text(encoding="utf-8")
    start_marker = "production_state_tool() {\n  python - \"$@\" <<'PY'\n"
    start = source.index(start_marker) + len(start_marker)
    end = source.index("\nPY\n}\n", start)
    return source[start:end]


def _run_state_tool(tmp_path, *arguments):
    state_tool = tmp_path / "production_state_tool.py"
    state_tool.write_text(_state_tool_source(), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(state_tool), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _write_production_plan(tmp_path, output_dir, campaign_tag, profile, blocks):
    plan_path = tmp_path / f"{profile}_plan.tsv"
    rows = []
    for block_id, years, categories, histograms in blocks:
        output_tag = "{}_{}_{}".format(
            campaign_tag,
            categories.replace(" ", "-"),
            histograms.replace(" ", "-"),
        )
        output_name = "{}SRs_{}".format(years.replace(" ", "-"), output_tag)
        rows.append(
            "\t".join(
                (
                    block_id,
                    years,
                    categories,
                    histograms,
                    output_tag,
                    output_name,
                    str(output_dir / f"{output_name}.pkl.gz"),
                    str(output_dir / f"{output_name}_np.pkl.gz"),
                )
            )
        )
    plan_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return plan_path


def _write_run3_full_plan(tmp_path, output_dir, campaign_tag):
    blocks = [
        (block_id, YEARS, categories, histograms)
        for block_id, categories, histograms in EXPECTED_BLOCKS
    ]
    return _write_production_plan(
        tmp_path, output_dir, campaign_tag, "run3_full", blocks
    )


def test_fresh_run3_full_state_materializes_plan_before_first_block_update(tmp_path):
    output_dir = tmp_path / "fresh_run3_full"
    state_path = output_dir / STATE_FILENAME
    plan_path = _write_run3_full_plan(tmp_path, output_dir, "run3_complete")
    output_dir.mkdir()

    initialize = _run_state_tool(
        tmp_path,
        "initialize",
        str(state_path),
        str(plan_path),
        "run3_full",
        "3",
        "run3_complete",
        str(output_dir),
        "topeft-test-commit",
        "/tmp/current_environment.tar.gz",
        "environment-sha256",
        "environment-fingerprint",
        "topcoffea-test-commit",
        "topcoffea-source-fingerprint",
        "split",
        "true",
        "true",
    )
    assert initialize.returncode == 0, initialize.stderr

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert [block["id"] for block in state["blocks"]] == [
        block_id for block_id, _categories, _histograms in EXPECTED_BLOCKS
    ]
    assert [block["years"] for block in state["blocks"]] == [
        YEARS.split()
    ] * len(EXPECTED_BLOCKS)
    assert [block["category_groups"] for block in state["blocks"]] == [
        categories.split() for _block_id, categories, _histograms in EXPECTED_BLOCKS
    ]
    assert [block["histograms"] for block in state["blocks"]] == [
        histograms.split() for _block_id, _categories, histograms in EXPECTED_BLOCKS
    ]
    assert all("2016" not in block["years"] for block in state["blocks"])
    assert all("CR" not in category for block in state["blocks"] for category in block["category_groups"])
    assert all(block["status"] == "planned" for block in state["blocks"])
    assert all(block["source_status"] == "planned" for block in state["blocks"])
    assert all(block["nonprompt_status"] == "blocked" for block in state["blocks"])

    first_status = _run_state_tool(tmp_path, "status", str(state_path), "run3_full_a")
    assert first_status.returncode == 0, first_status.stderr
    assert first_status.stdout.strip() == "planned\tplanned\tblocked"

    source_ready = _run_state_tool(
        tmp_path,
        "mark",
        str(state_path),
        "run3_full_a",
        "source",
        "ready",
        "0",
        "source_child_exit_zero_expected_source_present",
    )
    assert source_ready.returncode == 0, source_ready.stderr
    nonprompt_status = _run_state_tool(
        tmp_path, "status", str(state_path), "run3_full_a"
    )
    assert nonprompt_status.returncode == 0, nonprompt_status.stderr
    assert nonprompt_status.stdout.strip() == "source_ready\tready\tplanned"


def test_fresh_rebin_fine_state_preserves_six_block_stage_model(tmp_path):
    output_dir = tmp_path / "fresh_rebin_fine"
    state_path = output_dir / ".rebin_fine_campaign_state.json"
    plan_path = _write_production_plan(
        tmp_path, output_dir, "fine_complete", "rebin_fine", REBIN_FINE_BLOCKS
    )
    output_dir.mkdir()

    initialize = _run_state_tool(
        tmp_path,
        "initialize",
        str(state_path),
        str(plan_path),
        "rebin_fine",
        "3",
        "fine_complete",
        str(output_dir),
        "topeft-test-commit",
        "/tmp/current_environment.tar.gz",
        "environment-sha256",
        "environment-fingerprint",
        "topcoffea-test-commit",
        "topcoffea-source-fingerprint",
        "split",
        "true",
        "true",
    )
    assert initialize.returncode == 0, initialize.stderr

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert [block["id"] for block in state["blocks"]] == [
        block_id for block_id, _years, _categories, _histograms in REBIN_FINE_BLOCKS
    ]
    assert all(block["source_status"] == "planned" for block in state["blocks"])
    assert all(block["nonprompt_status"] == "blocked" for block in state["blocks"])

    source_ready = _run_state_tool(
        tmp_path,
        "mark",
        str(state_path),
        "run2_a",
        "source",
        "ready",
        "0",
        "source_child_exit_zero_expected_source_present",
    )
    assert source_ready.returncode == 0, source_ready.stderr
    nonprompt_failed = _run_state_tool(
        tmp_path,
        "mark",
        str(state_path),
        "run2_a",
        "nonprompt",
        "failed",
        "1",
        "separate_nonprompt_failed_or_missing_output",
    )
    assert nonprompt_failed.returncode == 0, nonprompt_failed.stderr
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["blocks"][0]["status"] == "nonprompt_failed"
    assert state["blocks"][0]["source_status"] == "ready"
    assert state["blocks"][0]["nonprompt_status"] == "failed"


def test_baseline_is_retired_and_no_argument_invocation_fails_closed():
    no_arguments = _run()
    assert no_arguments.returncode != 0
    assert "--production-profile is required" in no_arguments.stderr
    assert "Executing:" not in no_arguments.stdout

    baseline = _run("--production-profile", "baseline")
    assert baseline.returncode != 0
    assert "unsupported production profile 'baseline'" in baseline.stderr
    assert "Executing:" not in baseline.stdout


def test_run3_full_dry_run_resolves_exact_complete_five_block_plan(tmp_path):
    env_file = _write_env(tmp_path)
    output_dir = tmp_path / "fresh_run3_complete"
    result = _run(
        "--production-profile",
        "run3_full",
        "--dry-run",
        "--output-dir",
        str(output_dir),
        "--campaign-tag",
        "run3_complete",
        "--env-file",
        str(env_file),
    )

    assert result.returncode == 0, result.stderr
    resolved = re.findall(
        r"Mode: SR\nYears: ([^\n]+)\nCategories: ([^\n]+)\nVariables: ([^\n]+)",
        result.stdout,
    )
    assert resolved == [
        (YEARS, categories, histograms)
        for _block_id, categories, histograms in EXPECTED_BLOCKS
    ]
    assert result.stdout.count("run3_full two-stage dry-run resolved") == 5
    assert result.stdout.count("Separate nonprompt command (not executed by dry-run)") == 5
    assert "Mode: CR" not in result.stdout
    assert "2016APV" not in result.stdout
    assert "--ttgamma-sample-role-policy split" in result.stdout
    assert result.stdout.count("--do-systs") >= 5
    assert result.stdout.count("--do-np") >= 5
    assert result.stdout.count("--np-postprocess=defer") >= 5
    assert result.stdout.count("run_data_driven.py") >= 5
    assert "--split-lep-flavor" not in result.stdout
    assert not output_dir.exists()


def test_run_cr_derives_checkout_paths_and_runs_from_unrelated_cwd(tmp_path):
    source = RUN_CR.read_text(encoding="utf-8")
    assert "/users/apiccine/work/correction-lib/topeft" not in source
    assert 'dirname -- "${BASH_SOURCE[0]}"' in source
    assert 'git -C "${script_dir}" rev-parse --show-toplevel' in source

    env_file = _write_env(tmp_path)
    output_dir = tmp_path / "portable_dry_run"
    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()
    result = subprocess.run(
        [
            str(RUN_CR),
            "--production-profile",
            "run3_full",
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--campaign-tag",
            "portable_dry_run",
            "--env-file",
            str(env_file),
        ],
        cwd=unrelated_cwd,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("run3_full two-stage dry-run resolved") == 5
    assert not output_dir.exists()


def test_run3_full_requires_explicit_output_identity_and_pins_explicit_archives(tmp_path):
    env_file = _write_env(tmp_path)
    output_dir = tmp_path / "fresh"
    missing_output_identity = _run(
        "--production-profile", "run3_full", "--dry-run", "--env-file", str(env_file)
    )
    assert missing_output_identity.returncode != 0
    assert "requires explicit --output-dir and --campaign-tag" in missing_output_identity.stderr

    relative_env = _run(
        "--production-profile",
        "run3_full",
        "--dry-run",
        "--output-dir",
        str(output_dir),
        "--campaign-tag",
        "run3_complete",
        "--env-file",
        "stale.tar.gz",
    )
    assert relative_env.returncode != 0
    assert "must be an absolute path" in relative_env.stderr

    source = RUN_CR.read_text(encoding="utf-8")
    assert "validation_args=(--prepare-env-only)" in source
    assert "run_analysis.py did not return a complete valid environment identity" in source


def test_run3_full_rejects_stale_environment_before_state_mutation(tmp_path):
    env_file = _write_env(tmp_path, current=False)
    output_dir = tmp_path / "must_not_be_created"
    result = _run(
        "--production-profile",
        "run3_full",
        "--output-dir",
        str(output_dir),
        "--campaign-tag",
        "run3_complete",
        "--env-file",
        str(env_file),
    )

    assert result.returncode != 0
    assert "could not resolve a strict current environment archive" in result.stderr
    assert not output_dir.exists()
    assert not (output_dir / STATE_FILENAME).exists()


def test_run3_full_fresh_namespace_and_historical_v3_are_rejected(tmp_path):
    env_file = _write_env(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    collision = _run(
        "--production-profile",
        "run3_full",
        "--dry-run",
        "--output-dir",
        str(existing),
        "--campaign-tag",
        "run3_complete",
        "--env-file",
        str(env_file),
    )
    assert collision.returncode != 0
    assert "output directory already exists" in collision.stderr

    historical = _run(
        "--production-profile",
        "run3_full",
        "--dry-run",
        "--output-dir",
        str(tmp_path / "new"),
        "--campaign-tag",
        "rebin-fine-260818-v3",
        "--env-file",
        str(env_file),
    )
    assert historical.returncode != 0
    assert "historical baseline or v3 campaign" in historical.stderr


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("production_profile", "rebin_fine"), "mismatch for production_profile"),
        (("topeft_git_commit", "different_commit"), "mismatch for topeft_git_commit"),
        (("env_file_sha256", "different_sha"), "mismatch for env_file_sha256"),
        (("blocks.0.histograms", ["njets"]), "field histograms"),
    ],
)
def test_run3_full_resume_requires_exact_profile_source_env_and_plan(
    tmp_path, mutation, message
):
    env_file = _write_env(tmp_path)
    output_dir = tmp_path / "resume"
    output_dir.mkdir()
    state_path, state = _write_state(output_dir, "run3_complete", env_file)
    key, value = mutation
    if key == "blocks.0.histograms":
        state["blocks"][0]["histograms"] = value
    else:
        state[key] = value
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = _run(*_resume_args(output_dir, env_file))
    assert result.returncode != 0
    assert message in result.stderr


def test_run3_full_success_requires_nominal_and_np_and_refuses_partial_output(tmp_path):
    env_file = _write_env(tmp_path)
    output_dir = tmp_path / "resume_success"
    output_dir.mkdir()
    state_path, state = _write_state(
        output_dir, "run3_complete", env_file, status="success"
    )
    for block in state["blocks"]:
        for output in block["expected_outputs"]:
            Path(output).write_bytes(b"small synthetic output")
    Path(state["blocks"][0]["expected_np_path"]).unlink()

    missing_np = _run(*_resume_args(output_dir, env_file))
    assert missing_np.returncode != 0
    assert "marks run3_full_a successful" in missing_np.stderr
    updated = json.loads(state_path.read_text())
    assert updated["blocks"][0]["status"] == "nonprompt_failed"
    assert updated["blocks"][0]["transitions"][-1]["detail"] == (
        "success_state_missing_expected_nonprompt"
    )

    resume_nonprompt_only = _run(*_resume_args(output_dir, env_file))
    assert resume_nonprompt_only.returncode == 0, resume_nonprompt_only.stderr
    assert "Reusing validated completed source for run3_full_a" in resume_nonprompt_only.stdout
    assert resume_nonprompt_only.stdout.count("run3_full two-stage dry-run resolved") == 1
    assert resume_nonprompt_only.stdout.count("Skipping validated run3_full block") == 4

    state_path, state = _write_state(output_dir, "run3_complete", env_file)
    Path(state["blocks"][0]["expected_nominal_path"]).write_bytes(b"ambiguous partial")
    partial = _run(*_resume_args(output_dir, env_file))
    assert partial.returncode != 0
    assert "Refusing ambiguous overwrite" in partial.stderr
