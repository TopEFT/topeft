import os
import time
from pathlib import Path

import pytest

from topeft.modules.executor import resolve_environment_file


class _RemoteEnvironmentStub:
    def __init__(self, cache_dir: Path):
        self.env_dir_cache = cache_dir
        self.calls = []

    def get_environment(self, *, extra_pip_local, extra_conda):  # pragma: no cover - not used
        self.calls.append((extra_pip_local, extra_conda))
        return str(self.env_dir_cache / "built.tar.gz")


def test_resolve_environment_cached_uses_latest(tmp_path):
    cache_dir = tmp_path / "envs"
    cache_dir.mkdir()

    oldest = cache_dir / "env_spec_old.tar.gz"
    oldest.write_text("old")
    os.utime(oldest, (time.time() - 120, time.time() - 120))

    newest = cache_dir / "env_spec_new.tar.gz"
    newest.write_text("new")

    remote_env = _RemoteEnvironmentStub(cache_dir)

    path = resolve_environment_file("cached", remote_env)
    assert path == str(newest)
    assert remote_env.calls == []


def test_resolve_environment_cached_missing(tmp_path):
    cache_dir = tmp_path / "envs"
    cache_dir.mkdir()
    remote_env = _RemoteEnvironmentStub(cache_dir)

    with pytest.raises(FileNotFoundError) as exc:
        resolve_environment_file("cached", remote_env)

    message = str(exc.value)
    assert "cached remote environment" in message
    assert "--environment-file auto" in message
