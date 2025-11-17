"""Compatibility helpers for the deprecated :mod:`coffea.hist` API.

Historically ``topcoffea`` relied on the ``coffea.hist`` module to build
histograms.  Newer releases of ``coffea`` have dropped that module altogether,
but our legacy tests – and therefore downstream users – still expect the old
symbols to be available.  Rather than pinning the project to an outdated
``coffea`` release (which would in turn downgrade ``numpy``/``awkward`` and
break other tooling) we provide a tiny shim that re-implements the handful of
helpers we still need.

The shim simply forwards the public constructors to the modern ``hist``
package, keeping the rest of the library untouched.
"""

from __future__ import annotations

import sys
import types
from typing import Iterable, Sequence


def _as_sequence(categories: Iterable | None) -> Sequence:
    if categories is None:
        return []
    if isinstance(categories, str):
        return [categories]
    return list(categories)


def _build_module():
    from hist import axis as _axis

    shim = types.ModuleType("coffea.hist")

    def Cat(name, label=None, categories=None, *, growth=True):
        return _axis.StrCategory(
            _as_sequence(categories),
            name=name,
            label=label or name,
            growth=growth,
        )

    def Bin(name, label, bins, start, stop, *, flow=False):
        return _axis.Regular(
            bins,
            start,
            stop,
            name=name,
            label=label,
            flow=flow,
        )

    shim.Cat = Cat  # type: ignore[attr-defined]
    shim.Bin = Bin  # type: ignore[attr-defined]
    shim.axis = _axis
    shim.__all__ = ["Cat", "Bin", "axis"]
    return shim


def ensure_coffea_hist_module() -> None:
    """Expose ``coffea.hist`` when the upstream package no longer ships it."""

    try:
        import coffea.hist  # type: ignore[attr-defined]

        return
    except ModuleNotFoundError:
        import coffea

        shim = _build_module()
        sys.modules["coffea.hist"] = shim
        coffea.hist = shim  # type: ignore[attr-defined]


__all__ = ["ensure_coffea_hist_module"]

