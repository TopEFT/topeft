"""Smoke-test the local futures shell wrapper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="topcoffea histEFT type hints require Python 3.10+",
)
def test_local_futures_run_sh(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    analysis_dir = repo_root / "analysis" / "topeft_run2"
    script_path = analysis_dir / "local_futures_run.sh"
    outdir = tmp_path / "histos"
    outdir.mkdir(parents=True, exist_ok=True)

    sample_path = "../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json"

    cmd = [
        str(script_path),
        "--sr",
        "-y",
        "UL17",
        "--tag",
        "localtest",
        "--outdir",
        str(outdir),
        "--workers",
        "1",
        "--futures-prefetch",
        "0",
        "--samples",
        sample_path,
        "--nchunks",
        "1",
        "--chunksize",
        "4000",
    ]

    subprocess.run(cmd, check=True, cwd=analysis_dir)

    expected_pickle = outdir / "UL17_SRs_localtest.pkl.gz"
    assert expected_pickle.exists()
