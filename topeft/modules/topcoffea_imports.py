"""Helpers for resolving ``topcoffea`` namespace modules."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from typing import Callable

import topcoffea

__all__ = ["require_module", "require_script"]


def _resolve_importer() -> Callable[[str], ModuleType]:
    importer = getattr(topcoffea, "import_module", None)
    if importer is not None:
        return importer
    return import_module


def _load_namespace(namespace: str) -> ModuleType:
    try:
        return getattr(topcoffea, namespace)
    except AttributeError:
        importer = _resolve_importer()
        importer(f"topcoffea.{namespace}")
        return getattr(topcoffea, namespace)


def _resolve_attr(namespace: str, dotted_name: str) -> ModuleType:
    importer = _resolve_importer()
    module_path = f"topcoffea.{namespace}.{dotted_name}"
    module = importer(module_path)
    namespace_module = _load_namespace(namespace)
    current = namespace_module
    parts = dotted_name.split(".")
    for index, part in enumerate(parts, 1):
        if not hasattr(current, part):
            subpath = ".".join(parts[:index])
            child = sys.modules.get(f"topcoffea.{namespace}.{subpath}")
            if child is None:
                child = importer(f"topcoffea.{namespace}.{subpath}")
            setattr(current, part, child)
        current = getattr(current, part)
    return current


def require_module(name: str) -> ModuleType:
    """Return ``topcoffea.modules.<name>`` after ensuring it is imported."""

    return _resolve_attr("modules", name)


def require_script(name: str) -> ModuleType:
    """Return ``topcoffea.scripts.<name>`` after ensuring it is imported."""

    return _resolve_attr("scripts", name)
