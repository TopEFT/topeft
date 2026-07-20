from __future__ import annotations

import os
import pathlib
import re
import subprocess


root = pathlib.Path(__file__).resolve().parents[1]
driver = root / "analysis/topeft_run2/run_run3_missing_parton_pkls_overnight.sh"


def _run_driver(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "ALL_ANALYSIS_CHUNKS": "2",
            "MISSING_PARTON_EXECUTOR": "futures",
        }
    )
    return subprocess.run(
        [str(driver), *arguments],
        cwd=driver.parent,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_help_describes_fresh_resume_and_non_submitting_plan_modes():
    result = _run_driver("--help")

    assert result.returncode == 0
    assert "[output_root]" in result.stdout
    assert "--resume <existing_output_root>" in result.stdout
    assert "--print-plan <prospective_output_root>" in result.stdout
    assert "without creating the campaign root or invoking a backend" in result.stdout


def test_print_plan_is_complete_and_has_no_campaign_storage_effect(tmp_path):
    prospective_root = tmp_path / "prospective_campaign"
    result = _run_driver("--print-plan", str(prospective_root))

    assert result.returncode == 0, result.stdout
    assert not prospective_root.exists()
    assert "plan_mode=true" in result.stdout
    assert re.search(r"source_branch=\S+", result.stdout)
    assert re.search(r"source_commit=[0-9a-f]{40}", result.stdout)
    assert re.search(r"processor_sha256=[0-9a-f]{64}", result.stdout)
    assert "sumw2_contract=sm_only_complete_event_contribution_squared" in result.stdout
    assert result.stdout.count("raw_chunk\tall_analysis\t") == 4
    assert result.stdout.count("canonical\tall_analysis\t") == 2
    assert result.stdout.count("diagnostic\ttop22006\t") == 2
    execution_lines = result.stdout.split("--- execution_commands ---\n", 1)[1].splitlines()
    central_lines = [
        line for line in execution_lines if line.startswith("all_analysis_central_tzq_")
    ]
    private_lines = [
        line for line in execution_lines if line.startswith("all_analysis_private_tllq_")
    ]
    assert len(central_lines) == 2
    assert len(private_lines) == 2
    assert sum(line.startswith("canonicalize_central_tzq\t") for line in execution_lines) == 1
    assert sum(line.startswith("canonicalize_private_tllq\t") for line in execution_lines) == 1
    assert sum(line.startswith("top22006_central_tzq\t") for line in execution_lines) == 1
    assert sum(line.startswith("top22006_private_tllq\t") for line in execution_lines) == 1

    assert central_lines
    assert private_lines
    assert all("--do-systs" not in line for line in central_lines)
    assert any("--do-systs" in line for line in private_lines)


def test_invalid_arguments_and_relative_plan_root_are_rejected():
    unknown = _run_driver("--unknown")
    relative = _run_driver("--print-plan", "relative/campaign")

    assert unknown.returncode != 0
    assert "unknown option" in unknown.stdout
    assert relative.returncode != 0
    assert "output root must be an absolute path" in relative.stdout


def test_fresh_mode_rejects_a_populated_root_before_backend_execution(tmp_path):
    populated_root = tmp_path / "populated"
    populated_root.mkdir()
    marker = populated_root / "user_owned.txt"
    marker.write_text("preserve\n", encoding="utf-8")

    result = _run_driver(str(populated_root))

    assert result.returncode != 0
    assert "refusing to reuse a populated campaign root" in result.stdout
    assert marker.read_text(encoding="utf-8") == "preserve\n"


def test_resume_rejects_an_already_successful_campaign(tmp_path):
    campaign_root = tmp_path / "successful_campaign"
    campaign_root.mkdir()
    (campaign_root / "campaign_metadata.txt").write_text(
        "source_commit=placeholder\n", encoding="utf-8"
    )
    (campaign_root / "status.txt").write_text(
        "state=success\ntimestamp=placeholder\ndetail=complete\n",
        encoding="utf-8",
    )
    (campaign_root / "output_contract.tsv").write_text(
        "classification\tscope\trole\tchunk\tpath\n", encoding="utf-8"
    )
    (campaign_root / "validation.tsv").write_text("header\n", encoding="utf-8")
    (campaign_root / "execution_commands.tsv").write_text(
        "header\n", encoding="utf-8"
    )

    result = _run_driver("--resume", str(campaign_root))

    assert result.returncode != 0
    assert "a successful campaign is immutable and cannot be resumed" in result.stdout
    assert not (campaign_root / ".campaign_lock").exists()
