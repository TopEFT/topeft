"""Lightweight helpers for configuring Python logging in run_analysis."""

from __future__ import annotations

import logging
from typing import Optional

from .run_analysis_helpers import VALID_LOG_LEVELS

_configured = False


def _level_name_to_numeric(level_name: str) -> int:
    resolved = getattr(logging, level_name.upper(), None)
    if not isinstance(resolved, int):
        raise ValueError(f"Unknown logging level '{level_name}'.")
    return resolved


def configure_logging(level_name: str, *, formatter: Optional[str] = None) -> None:
    """Configure root logging handlers with a consistent format.

    The helper intentionally keeps the configuration minimal: a single stream
    handler with timestamps and module names. In multi-process futures runs
    the configuration only applies to the main process for now—workers inherit
    coffea's defaults until we plumb per-process hooks.
    """

    global _configured

    if level_name.upper() not in VALID_LOG_LEVELS:
        raise ValueError(
            f"log level '{level_name}' is not in {', '.join(sorted(VALID_LOG_LEVELS))}"
        )

    numeric_level = _level_name_to_numeric(level_name)
    root = logging.getLogger()

    if not _configured:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                formatter or "%(asctime)s %(levelname)s %(name)s: %(message)s"
            )
        )
        handler.setLevel(numeric_level)
        root.addHandler(handler)
        _configured = True
    else:
        for handler in root.handlers:
            handler.setLevel(numeric_level)

    root.setLevel(numeric_level)
