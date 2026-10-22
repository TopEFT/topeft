"""Maintained public profiles preserve the resolved scientific launch contract."""

import hashlib
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from analysis.topeft_run2 import campaign_manifest as manifest


RUN_DIRECTORY = Path(__file__).resolve().parents[1] / "analysis" / "topeft_run2"
RUN_CR = RUN_DIRECTORY / "run_cr.sh"
ARCHIVE = RUN_DIRECTORY / "topeft-envs" / "env_spec_1707ade2a411bb14.tar.gz"
PUBLIC_COMPONENTS = (
    "run2_prod", "run3_prod", "run2_prod_CR", "run3_prod_CR",
    "run2_stat_diagnostic", "run3_stat_diagnostic",
    "run2_stat_diagnostic_CR", "run3_stat_diagnostic_CR",
    "run2_sr_ext", "run3_sr_ext",
)
PUBLIC_COMBINED = (
    "run2_run3_prod", "run2_run3_prod_CR",
    "run2_run3_stat_diagnostic", "run2_run3_stat_diagnostic_CR",
)
SR_BLOCKS = (
    (("2l", "2lss_1tau", "2los_1tau", "4l"), ("njets", "lj0pt", "ptz", "ptz_wtau", "lt")),
    (("3l_m_offZ",), ("njets", "lj0pt", "ptll", "lt")),
    (("3l_p_offZ",), ("njets", "lj0pt", "ptll", "lt")),
    (("3l_onZ_tau",), ("njets", "lj0pt", "ptz", "lt")),
    (("3l_fwd",), ("njets", "lj0pt", "ptz", "lt")),
)


@pytest.fixture
def validation_environment(tmp_path):
    backend = tmp_path / "backend.sh"
    backend.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "case \"$1\" in validate_environment|prepare_environment) ;; *) exit 64 ;; esac\n"
        f"printf 'env_file: %s\\n' '{ARCHIVE}'\n"
        f"printf 'env_file_sha256: %s\\n' '{hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()}'\n"
        "printf 'environment_fingerprint: %064d\\n' 9\n"
        "printf 'environment_validation_status: valid\\n'\n"
        "printf 'topcoffea_git_commit: %040d\\n' 8\n"
        "printf 'topcoffea_relevant_source_fingerprint: %064d\\n' 7\n",
        encoding="utf-8",
    )
    backend.chmod(0o700)
    environment = os.environ.copy()
    environment.update({
        "CAMPAIGN_VALIDATION_BACKEND": str(backend),
        "CAMPAIGN_VALIDATION_ROOT": str(tmp_path),
        "TMPDIR": str(tmp_path),
    })
    return environment


def run_profile(profile, tmp_path, environment, *extra):
    return subprocess.run(
        [str(RUN_CR), "--production-profile", profile,
         "--output-dir", str(tmp_path / "output"), "--campaign-tag", "synthetic",
         "--env-file", str(ARCHIVE), "--dry-run", *extra],
        cwd=RUN_DIRECTORY, env=environment, text=True, capture_output=True, check=False,
    )


def source_commands(stdout):
    return [
        shlex.split(line)
        for line in re.findall(r"Running the following command:\n([^\n]+)", stdout)
    ]


def option_values(command, option):
    values = []
    for value in command[command.index(option) + 1:]:
        if value.startswith("-"):
            break
        values.append(value)
    return tuple(values)


@pytest.mark.parametrize("profile", PUBLIC_COMPONENTS)
def test_component_science_and_default_coverage(profile, tmp_path, validation_environment):
    result = run_profile(profile, tmp_path, validation_environment)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = source_commands(result.stdout)
    expected_count = 12 if profile.endswith("_sr_ext") else (
        6 if profile.startswith("run2_") and profile.endswith("_CR") else
        12 if profile.startswith("run3_") and profile.endswith("_CR") else
        5
    )
    assert len(commands) == expected_count
    policy = manifest.profile_contract(profile)
    for command in commands:
        assert ("--do-systs" in command) is policy["systematics"]
        assert ("--record-raw-count" in command) is policy["raw_counts"]
        assert ("--skip-cr" in command) is (policy["region"] == "SR")
        assert ("--skip-sr" in command) is (policy["region"] == "CR")
        assert "--do-np" in command
        assert "--np-postprocess=defer" in command
        assert option_values(command, "--ttgamma-sample-role-policy") == ("split",)
    if policy["region"] == "SR" and policy["mode"] != "sr_ext":
        expected_years = ("UL16", "UL16APV", "UL17", "UL18") if profile.startswith("run2_") else (
            "2022", "2022EE", "2023", "2023BPix"
        )
        assert [
            (option_values(command, "--category-groups"), option_values(command, "--hist-list"))
            for command in commands
        ] == list(SR_BLOCKS)
        assert all(option_values(command, "--years") == expected_years for command in commands)
    if policy["region"] == "CR":
        expected_year_sets = {
            ("2016APV", "2016", "2017", "2018")
        } if profile.startswith("run2_") else {
            ("2022", "2022EE"), ("2023", "2023BPix")
        }
        assert {option_values(command, "--years") for command in commands} == expected_year_sets
    if profile.endswith("_sr_ext"):
        assert manifest.digest_file(manifest.DEFAULT_SR_PLAN) == "bee12084cccf221ec9109998a5e5d383e256b57383bcdea1450701e8afe87306"
    assert not list((tmp_path / "output").glob("*campaign_state*"))


@pytest.mark.parametrize("profile", PUBLIC_COMPONENTS)
def test_selection_keeps_full_plan_and_positive_chunksize(profile, tmp_path, validation_environment):
    first = "a_1" if profile.endswith("_sr_ext") else (
        f"{profile.lower()}_block1" if profile.endswith("_CR") else f"{profile}_a"
    )
    selected = f"{profile}_{first}" if profile.endswith("_sr_ext") else first
    result = run_profile(profile, tmp_path, validation_environment,
                         "--selected-fragment", selected, "--chunksize", "12345")
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(source_commands(result.stdout)) == 1
    assert "-s 12345" in result.stdout
    assert "dry_run_complete: 1 unit" in result.stdout


def test_exact_manager_name_reaches_nominal_runner_for_one_source(tmp_path, validation_environment):
    result = run_profile("run3_prod", tmp_path, validation_environment,
                         "--selected-fragment", "run3_prod_a",
                         "--wq-manager-name", "top26006-test-manager")
    assert result.returncode == 0, result.stdout + result.stderr
    commands = source_commands(result.stdout)
    assert len(commands) == 1
    assert commands[0].count("--wq-manager-name") == 1
    assert option_values(commands[0], "--wq-manager-name") == ("top26006-test-manager",)
    assert not (tmp_path / "output").exists()


def test_exact_manager_name_rejects_multiple_sources_before_output(tmp_path, validation_environment):
    result = run_profile("run3_prod", tmp_path, validation_environment,
                         "--wq-manager-name", "top26006-test-manager")
    assert result.returncode != 0
    assert "resolved multiple" in result.stderr
    assert source_commands(result.stdout) == []
    assert not (tmp_path / "output").exists()


def test_exact_manager_name_rejects_combined_profile_before_children(tmp_path, validation_environment):
    result = run_profile("run2_run3_prod", tmp_path, validation_environment,
                         "--wq-manager-name", "top26006-test-manager")
    assert result.returncode != 0
    assert "requires one component campaign" in result.stderr
    assert source_commands(result.stdout) == []
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("profile", (
    "run2_full", "run3_full", "run2_run3_full", "run2_full_CR",
    "run3_full_CR", "run2_run3_full_CR", "t0_sr_statonly", "t0_cr_statonly",
))
def test_retired_public_profiles_are_rejected(profile, tmp_path, validation_environment):
    result = run_profile(profile, tmp_path, validation_environment)
    assert result.returncode != 0
    assert "unsupported production profile" in result.stderr


def test_retired_plan_option_is_rejected(tmp_path, validation_environment):
    result = run_profile("run2_sr_ext", tmp_path, validation_environment, "--sr-plan", "anything")
    assert result.returncode != 0
    assert "unsupported run_cr.sh option" in result.stderr


def test_unknown_selected_fragment_is_rejected(tmp_path, validation_environment):
    result = run_profile("run2_prod", tmp_path, validation_environment,
                         "--selected-fragment", "missing_unit")
    assert result.returncode != 0
    assert "unknown selected fragment" in result.stderr


def test_combined_profiles_are_public_and_child_owned():
    help_result = subprocess.run([str(RUN_CR), "--help"], cwd=RUN_DIRECTORY,
                                 text=True, capture_output=True, check=False)
    assert help_result.returncode == 0
    for profile in PUBLIC_COMBINED:
        assert profile in help_result.stdout


@pytest.mark.parametrize("profile", PUBLIC_COMBINED)
def test_combined_dry_run_dispatches_exact_child_profiles(profile, tmp_path, validation_environment):
    result = run_profile(profile, tmp_path, validation_environment)
    assert result.returncode == 0, result.stdout + result.stderr
    suffix = "_CR" if profile.endswith("_CR") else ""
    mode = "stat_diagnostic" if "stat_diagnostic" in profile else "prod"
    assert f"production_profile: run2_{mode}{suffix}" in result.stdout
    assert f"production_profile: run3_{mode}{suffix}" in result.stdout
    assert not (tmp_path / "output").exists()
