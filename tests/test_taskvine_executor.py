import subprocess
import sys
from pathlib import Path

import pytest


def test_run_analysis_with_taskvine(tmp_path):
    """Launch run_analysis.py with the TaskVine executor and verify the output."""

    taskvine = pytest.importorskip(
        "ndcctools.taskvine", reason="TaskVine Python bindings are unavailable."
    )
    processor = pytest.importorskip(
        "coffea.processor", reason="Coffea processor module missing."
    )

    if getattr(processor, "TaskVineExecutor", None) is None:
        pytest.skip("Coffea build does not provide TaskVineExecutor.")

    port = 9135
    manager_host_port = f"localhost:{port}"

    repo_root = Path(__file__).resolve().parents[1]
    analysis_dir = repo_root / "analysis" / "topeft_run2"

    scratch_dir = tmp_path / "taskvine_staging"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "histos"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_manifest = (
        repo_root
        / "input_samples"
        / "sample_jsons"
        / "test_samples"
        / "UL17_private_ttH_for_CI.json"
    )

    args = [
        sys.executable,
        "run_analysis.py",
        str(json_manifest),
        "--executor",
        "taskvine",
        "--port",
        f"{port}-{port}",
        "--no-port-negotiation",
        "--nworkers",
        "1",
        "--chunksize",
        "500",
        "--nchunks",
        "1",
        "--outname",
        "output_taskvine",
        "--outpath",
        str(output_dir),
        "--scratch-dir",
        str(scratch_dir),
        "--no-environment-file",
        "--prefix",
        "http://www.crc.nd.edu/~kmohrman/files/root_files/for_ci/",
        "--summary-verbosity",
        "none",
    ]

    factory = taskvine.Factory("local", manager_host_port=manager_host_port)
    factory.max_workers = 1
    factory.min_workers = 1
    factory.cores = 1
    factory.memory = 2000
    factory.disk = 2000
    factory.scratch_dir = str(tmp_path / "factory")

    with factory:
        subprocess.run(
            args,
            cwd=analysis_dir,
            check=True,
            timeout=420,
        )

    artifact = output_dir / "output_taskvine.pkl.gz"
    assert artifact.exists()
