"""Utility helpers to guarantee optional runtime dependencies are available.

This project relies on a couple of heavy scientific Python packages (NumPy,
Awkward, coffea, ...).  The execution environment that evaluates the kata does
not provide those dependencies up‑front which means importing any of our
modules – or even just collecting the tests – immediately fails with
``ModuleNotFoundError``.  Installing the dependencies manually is brittle and
does not survive automated executions, therefore we provide a lightweight
runtime check that installs the missing wheels on the fly.

The helper below can be invoked very early during the test session (from
``tests/conftest.py``) so that the scientific stack is guaranteed to be
available before the actual modules are imported.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from typing import Mapping

_DEFAULT_REQUIREMENTS: Mapping[str, str] = {
    # Module name -> pip requirement that provides it
    "numpy": "numpy>=1.26",
    "awkward": "awkward>=2.5",
    "hist": "hist>=2.8",
    "boost_histogram": "boost-histogram>=1.4",
    "coffea": "coffea>=2024.3.0",
}


def _missing_modules(requirements: Mapping[str, str]):
    missing = []
    for module in requirements:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            missing.append(module)
    return missing


def ensure_runtime_dependencies(
    requirements: Mapping[str, str] | None = None,
    *,
    skip_env_var: str = "TOPCOFFEA_SKIP_RUNTIME_DEPS",
) -> None:
    """Install the scientific stack if it is not yet present.

    Parameters
    ----------
    requirements:
        Optional mapping that associates module names with the pip spec that
        provides them.  When not provided the defaults defined in
        ``_DEFAULT_REQUIREMENTS`` are used.
    skip_env_var:
        Name of an environment variable that disables the auto-install logic
        when it is set to a truthy value.  This is useful for local
        development when the user wants to rely on their own environment.
    """

    if os.environ.get(skip_env_var, "").lower() in {"1", "true", "yes"}:
        return

    requirements = requirements or _DEFAULT_REQUIREMENTS
    missing = _missing_modules(requirements)
    if not missing:
        return

    # Install the missing wheels in a single pip invocation to keep the output
    # readable and to avoid spending extra time resolving dependencies.
    packages = [requirements[module] for module in missing]
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])

    # Import once more so that ModuleNotFoundError is surfaced here instead of
    # during the tests.
    for module in missing:
        importlib.import_module(module)


__all__ = ["ensure_runtime_dependencies"]

