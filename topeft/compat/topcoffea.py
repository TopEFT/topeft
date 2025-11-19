"""Runtime helpers that patch ``topcoffea`` for older Python versions."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Optional

_PATCH_TARGET = "ArrayLike | Mapping | None"
_PATCH_REPLACEMENT = "Union[ArrayLike, Mapping, None]"


def _patched_histEFT_source(source: str) -> Optional[str]:
    if _PATCH_TARGET not in source:
        return None
    return source.replace(_PATCH_TARGET, _PATCH_REPLACEMENT)


def ensure_histEFT_py39_compat() -> None:
    """Load ``topcoffea.modules.histEFT`` with Python 3.9 friendly annotations."""

    if sys.version_info >= (3, 10):
        return

    fullname = "topcoffea.modules.histEFT"
    if fullname in sys.modules:
        return

    spec = importlib.util.find_spec(fullname)
    if spec is None or spec.loader is None or not hasattr(spec.loader, "get_source"):
        return

    source = spec.loader.get_source(fullname)
    if source is None:
        return

    patched_source = _patched_histEFT_source(source)
    if patched_source is None:
        # Nothing to patch, so fall back to the regular import path.
        importlib.import_module(fullname)
        return

    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[fullname] = module
        exec(compile(patched_source, spec.origin or fullname, "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(fullname, None)
        raise

    package_name, _, attr = fullname.rpartition(".")
    package = importlib.import_module(package_name)
    setattr(package, attr, module)
