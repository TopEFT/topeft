"""Smoke-test the unified run wrapper in dry-run mode."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_full_run_sh_dry_run(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    analysis_dir = repo_root / "analysis" / "topeft_run2"
    script_path = analysis_dir / "full_run.sh"
    outdir = tmp_path / "histos"
    outdir.mkdir(parents=True, exist_ok=True)

    sample_path = "../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json"

    cmd = [
        str(script_path),
        "--sr",
        "-y",
        "UL17",
        "--tag",
        "drytest",
        "--outdir",
        str(outdir),
        "--executor",
        "futures",
        "--samples",
        sample_path,
        "--dry-run",
    ]

    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        cwd=analysis_dir,
    )

    stdout = completed.stdout
    assert "Resolved years: UL17" in stdout
    assert "Executor: futures" in stdout
    assert "UL17_SRs_drytest" in stdout
    assert str(outdir) in stdout
    assert "--skip-cr --do-systs" in stdout


def test_full_run_sh_dry_run_without_conda(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    analysis_dir = repo_root / "analysis" / "topeft_run2"
    script_path = analysis_dir / "full_run.sh"
    outdir = tmp_path / "histos"
    outdir.mkdir(parents=True, exist_ok=True)

    sample_path = "../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json"

    env = {
        "PATH": "/usr/bin",
        "PYTHONPATH": "",
    }

    cmd = [
        str(script_path),
        "--sr",
        "-y",
        "UL17",
        "--tag",
        "drytest",
        "--outdir",
        str(outdir),
        "--executor",
        "futures",
        "--samples",
        sample_path,
        "--dry-run",
    ]

    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        cwd=analysis_dir,
        env=env,
    )

    assert "Note: no active conda environment detected" in completed.stderr
    assert "Resolved years: UL17" in completed.stdout
    assert "Executor: futures" in completed.stdout
    assert "UL17_SRs_drytest" in completed.stdout
