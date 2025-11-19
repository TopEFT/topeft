import sys
import types

import pytest

from topeft._dependency_checks import ensure_topcoffea_branch


def _install_topcoffea_stub(monkeypatch, tmp_path, *, head_contents=None):
    repo_root = tmp_path / "topcoffea-src"
    pkg_dir = repo_root / "topcoffea"
    pkg_dir.mkdir(parents=True)
    init_file = pkg_dir / "__init__.py"
    init_file.write_text("# stub")

    module = types.ModuleType("topcoffea")
    module.__file__ = str(init_file)
    module.modules = types.SimpleNamespace()

    monkeypatch.setitem(sys.modules, "topcoffea", module)

    if head_contents is not None:
        git_dir = repo_root / ".git"
        git_dir.mkdir(parents=True)
        (git_dir / "HEAD").write_text(head_contents)


def test_branch_guard_accepts_expected_branch(monkeypatch, tmp_path):
    _install_topcoffea_stub(
        monkeypatch,
        tmp_path,
        head_contents="ref: refs/heads/ch_update_calcoffea\n",
    )

    ensure_topcoffea_branch()


def test_branch_guard_rejects_mismatched_branch(monkeypatch, tmp_path):
    _install_topcoffea_stub(
        monkeypatch,
        tmp_path,
        head_contents="ref: refs/heads/main\n",
    )

    with pytest.raises(RuntimeError) as excinfo:
        ensure_topcoffea_branch()

    assert "ch_update_calcoffea" in str(excinfo.value)


def test_branch_guard_accepts_env_override(monkeypatch, tmp_path):
    _install_topcoffea_stub(monkeypatch, tmp_path, head_contents=None)
    monkeypatch.setenv("TOPCOFFEA_BRANCH", "ch_update_calcoffea")

    ensure_topcoffea_branch()
