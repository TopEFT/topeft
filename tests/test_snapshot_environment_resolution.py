import importlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from topcoffea.modules import remote_environment


analysis_dir = Path(__file__).resolve().parents[1] / "analysis" / "topeft_run2"


@pytest.fixture(params=["run_analysis", "run_analysis_excluded_mc"])
def producer(request, monkeypatch):
    monkeypatch.syspath_prepend(str(analysis_dir))
    return importlib.import_module(request.param)


@pytest.fixture
def archive(tmp_path):
    archive_path = tmp_path / "environment.tar.gz"
    with tarfile.open(archive_path, "w:gz") as handle:
        content = b"small environment fixture"
        member = tarfile.TarInfo("environment.txt")
        member.size = len(content)
        handle.addfile(member, io.BytesIO(content))
    manifest_path = remote_environment.write_archive_manifest(
        str(archive_path),
        {
            "environment_fingerprint": "a" * 64,
            "python_version": "fixture",
            "resolved_environment_spec": {"fixture": True},
            "resolved_environment_spec_fingerprint": "b" * 64,
            "editable_packages": [
                {
                    "package_name": "topcoffea",
                    "git_commit": "c" * 40,
                    "watched_source_fingerprint": "d" * 64,
                }
            ],
        },
    )
    return archive_path, Path(manifest_path)


def test_snapshot_uses_verified_archive_without_resolving_checkout(
    producer, archive, monkeypatch, capsys
):
    archive_path, _ = archive

    def dirty_checkout(**_kwargs):
        raise remote_environment.UnstagedChanges(["topeft"])

    monkeypatch.setattr(remote_environment, "resolve_environment_request", dirty_checkout)
    resolved = producer._resolve_environment_file(str(archive_path), True, snapshot=True)
    output = capsys.readouterr().out

    assert resolved == str(archive_path.resolve())
    assert "environment_validation_status: valid" in output
    assert f"env_file: {archive_path.resolve()}" in output
    assert "env_file_sha256: " in output
    assert "environment_fingerprint: " + "a" * 64 in output
    assert "topcoffea_git_commit: " + "c" * 40 in output
    assert "topcoffea_relevant_source_fingerprint: " + "d" * 64 in output


@pytest.mark.parametrize(
    "damage",
    ["missing_archive", "invalid_tar", "missing_manifest", "invalid_manifest", "missing_field", "wrong_sha"],
)
def test_snapshot_rejects_invalid_archive_or_manifest(
    producer, archive, monkeypatch, damage
):
    archive_path, manifest_path = archive
    monkeypatch.setattr(
        remote_environment,
        "resolve_environment_request",
        lambda **_kwargs: pytest.fail("snapshot resolved the current checkout"),
    )
    if damage == "missing_archive":
        archive_path.unlink()
    elif damage == "invalid_tar":
        archive_path.write_bytes(b"not a tarball")
    elif damage == "missing_manifest":
        manifest_path.unlink()
    elif damage == "invalid_manifest":
        manifest_path.write_text("{", encoding="utf-8")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if damage == "missing_field":
            del manifest["environment_fingerprint"]
        else:
            manifest["archive_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit, match="environment_status:"):
        producer._resolve_environment_file(str(archive_path), True, snapshot=True)


def test_non_snapshot_still_resolves_current_request(producer, archive, monkeypatch):
    archive_path, _ = archive
    calls = []

    def current_request(**kwargs):
        calls.append(kwargs)
        return {"environment_fingerprint": "a" * 64}

    monkeypatch.setattr(remote_environment, "resolve_environment_request", current_request)
    assert producer._resolve_environment_file(str(archive_path), True) == str(archive_path)
    assert calls == [{"extra_pip_local": {"topeft": ["topeft", "setup.py"]}, "unstaged": "fail"}]


def test_non_snapshot_preserves_dirty_and_mismatch_errors(
    producer, archive, monkeypatch
):
    archive_path, _ = archive

    def dirty_checkout(**_kwargs):
        raise remote_environment.UnstagedChanges(["topeft"])

    monkeypatch.setattr(remote_environment, "resolve_environment_request", dirty_checkout)
    with pytest.raises(remote_environment.UnstagedChanges):
        producer._resolve_environment_file(str(archive_path), True)

    monkeypatch.setattr(
        remote_environment,
        "resolve_environment_request",
        lambda **_kwargs: {"environment_fingerprint": "different"},
    )
    with pytest.raises(SystemExit, match="environment_status: stale"):
        producer._resolve_environment_file(str(archive_path), True)


def test_integrity_only_keeps_archive_identity_without_current_request(
    producer, archive, monkeypatch, capsys
):
    archive_path, _ = archive
    monkeypatch.setattr(
        remote_environment,
        "resolve_environment_request",
        lambda **_kwargs: pytest.fail("integrity-only resolved the current checkout"),
    )
    assert producer._resolve_environment_file(
        str(archive_path), True, integrity_only=True
    ) == str(archive_path)
    assert "environment_validation_status: valid" in capsys.readouterr().out
