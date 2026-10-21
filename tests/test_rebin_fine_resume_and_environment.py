import hashlib
import importlib
import io
import json
import subprocess
import tarfile
import sys
from functools import lru_cache
from pathlib import Path

from topcoffea.modules import remote_environment


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPOSITORY_ROOT / "analysis" / "topeft_run2"
RUN_CR = ANALYSIS_DIR / "run_cr.sh"
FULL_R3_RUN = ANALYSIS_DIR / "fullR3_run.sh"
STATE_FILENAME = ".rebin_fine_campaign_state.json"
CHANNEL_REGISTRY = REPOSITORY_ROOT / "topeft" / "channels" / "ch_lst.json"


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
        unstaged="rebuild",
    )


def _write_env(tmp_path, content=b"synthetic poncho archive"):
    path = tmp_path / "verified_env.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("environment.txt")
        entry.size = len(content)
        archive.addfile(entry, io.BytesIO(content))
    remote_environment.write_archive_manifest(
        str(path), _current_environment_request()
    )
    return path


def test_rebin_fine_is_retired_before_output_creation(tmp_path):
    output_dir = tmp_path / "retired"
    result = _run("--production-profile", "rebin_fine", "--output-dir", str(output_dir), "--campaign-tag", "retired")
    assert result.returncode != 0
    assert "unsupported production profile" in result.stderr
    assert not output_dir.exists()


def test_fullr3_forwards_env_file_to_run_analysis_dry_run(tmp_path):
    env_file = _write_env(tmp_path)
    result = subprocess.run(
        [
            str(FULL_R3_RUN),
            "-y",
            "2022",
            "-t",
            "env_forward",
            "--sr",
            "--dry-run",
            "--env-file",
            str(env_file.resolve()),
        ],
        cwd=ANALYSIS_DIR,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"--env-file {env_file.resolve()}" in result.stdout


def test_env_override_bypasses_automatic_poncho_packaging(tmp_path, monkeypatch):
    env_file = _write_env(tmp_path)
    monkeypatch.syspath_prepend(str(ANALYSIS_DIR))
    run_analysis = importlib.import_module("run_analysis")

    def unexpected_packaging(**_kwargs):
        raise AssertionError("automatic environment packaging was called")

    monkeypatch.setattr(run_analysis.remote_environment, "get_environment", unexpected_packaging)
    monkeypatch.setattr(
        run_analysis.remote_environment,
        "resolve_environment_request",
        lambda **_kwargs: {"environment_fingerprint": "current"},
    )
    monkeypatch.setattr(
        run_analysis.remote_environment,
        "validate_environment_archive",
        lambda path, *_args, **_kwargs: {
            "archive_path": str(Path(path).resolve()),
            "archive_sha256": "synthetic",
            "manifest_path": f"{path}.manifest.json",
            "environment_fingerprint": "current",
            "status": "valid",
            "mismatches": [],
            "editable_packages": [],
            "usable": True,
        },
    )
    assert run_analysis._resolve_environment_file(str(env_file), True) == str(
        env_file.resolve()
    )


def test_environment_cli_rejects_invalid_snapshot_combinations(tmp_path):
    snapshot_without_archive = subprocess.run(
        [sys.executable, str(ANALYSIS_DIR / "run_analysis.py"), "--snapshot"],
        cwd=ANALYSIS_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    assert snapshot_without_archive.returncode != 0
    assert "--snapshot requires --env-file" in snapshot_without_archive.stderr

    archive = _write_env(tmp_path)
    rebuild_with_archive = subprocess.run(
        [
            sys.executable,
            str(ANALYSIS_DIR / "run_analysis.py"),
            "--env-file",
            str(archive),
            "--rebuild-env",
        ],
        cwd=ANALYSIS_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rebuild_with_archive.returncode != 0
    assert "cannot be combined with --env-file" in rebuild_with_archive.stderr
