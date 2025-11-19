"""Smoke tests for external dependencies that we vendor locally."""

import importlib


def test_topcoffea_is_importable():
    module = importlib.import_module("topcoffea")
    assert module.__name__ == "topcoffea"
