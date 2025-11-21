from __future__ import annotations

import importlib
import sys
import types

import pytest


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = f"<mocked {name}>"
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _reload_runner_output(monkeypatch, module_map: dict[str, types.ModuleType]):
    for key, value in module_map.items():
        monkeypatch.setitem(sys.modules, key, value)
    sys.modules.pop("topeft.modules.runner_output", None)
    return importlib.import_module("topeft.modules.runner_output")


def test_runner_output_imports_lowercase_hist_eft(monkeypatch):
    topcoffea_pkg = _module("topcoffea")
    modules_pkg = _module("topcoffea.modules")
    hist_eft_module = _module("topcoffea.modules.histEFT")

    class _DummyHistEFT:
        pass

    hist_eft_module.HistEFT = _DummyHistEFT
    modules_pkg.histEFT = hist_eft_module  # type: ignore[attr-defined]

    runner_output = _reload_runner_output(
        monkeypatch,
        {
            "topcoffea": topcoffea_pkg,
            "topcoffea.modules": modules_pkg,
            "topcoffea.modules.histEFT": hist_eft_module,
        },
    )

    assert runner_output.HistEFT is _DummyHistEFT


def test_runner_output_imports_raise_helpful_error(monkeypatch):
    topcoffea_pkg = _module("topcoffea")
    modules_pkg = _module("topcoffea.modules")

    runner_output = _reload_runner_output(
        monkeypatch,
        {
            "topcoffea": topcoffea_pkg,
            "topcoffea.modules": modules_pkg,
        },
    )

    assert runner_output.HistEFT is None
    assert runner_output._HISTEFT_IMPORT_ERROR is not None
    with pytest.raises(ImportError, match="does not provide a HistEFT class"):
        runner_output._summarise_histogram(object())
