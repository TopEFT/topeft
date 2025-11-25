"""Helpers for constructing NanoEvents factories with explicit modes.

The helpers here centralize the logic required by coffea >=0.7, where
``NanoEvents`` objects expect their originating factory to advertise an
explicit ``_mode`` (for example ``"numpy"`` for in-memory processing).
By funnelling all factory creation through these helpers we ensure a
consistent mode is attached regardless of whether coffea sets it
internally.
"""

from __future__ import annotations

from coffea.nanoevents import NanoEventsFactory


def ensure_factory_mode(factory: NanoEventsFactory, *, mode: str = "numpy") -> NanoEventsFactory:
    """Guarantee ``factory._mode`` is populated.

    Parameters
    ----------
    factory:
        The factory instance produced by ``NanoEventsFactory.from_root`` or
        similar helpers.
    mode:
        The mode value to enforce when the factory does not already define
        ``_mode``.
    """

    try:
        current_mode = getattr(factory, "_mode", None)
    except Exception:  # pragma: no cover - defensive fallback
        current_mode = None

    if current_mode != mode:
        try:
            factory._mode = mode
        except Exception:  # pragma: no cover - best effort
            pass

    return factory


def nanoevents_factory_from_root(*args, mode: str = "numpy", **kwargs) -> NanoEventsFactory:
    """Wrap ``NanoEventsFactory.from_root`` and enforce an explicit mode."""

    factory = NanoEventsFactory.from_root(*args, **kwargs)
    return ensure_factory_mode(factory, mode=mode)

