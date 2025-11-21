"""Exercise the workflow import shims for topcoffea helpers."""

from __future__ import annotations

import importlib
import runpy
import sys
import warnings

import pytest


def _clear_modules(monkeypatch: pytest.MonkeyPatch, *module_names: str) -> None:
    for name in module_names:
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_workflow_imports_topcoffea_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_modules(
        monkeypatch,
        "analysis.topeft_run2",
        "analysis.topeft_run2.workflow",
        "topcoffea.modules.paths",
        "topcoffea.modules.utils",
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        module_globals = runpy.run_module(
            "analysis.topeft_run2.workflow", run_name="analysis.topeft_run2.workflow"
        )

    assert callable(module_globals["topcoffea_path"])
    assert module_globals["topcoffea_utils"] is not None


def test_workflow_imports_missing_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_modules(
        monkeypatch, "analysis.topeft_run2", "analysis.topeft_run2.workflow", "topcoffea.modules.paths"
    )

    real_import_module = importlib.import_module

    def _raise_for_paths(name: str, *args, **kwargs):
        if name.endswith(".modules.paths"):
            raise ModuleNotFoundError("paths module unavailable")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _raise_for_paths)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with pytest.raises(ImportError, match="ch_update_calcoffea"):
            runpy.run_module(
                "analysis.topeft_run2.workflow", run_name="analysis.topeft_run2.workflow"
            )
