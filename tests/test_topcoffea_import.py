"""Smoke tests for external dependencies that we vendor locally."""

import importlib
import importlib.util


def test_topcoffea_is_importable():
    module = importlib.import_module("topcoffea")
    assert module.__name__ == "topcoffea"


def test_local_shims_are_removed_and_delegated():
    assert importlib.util.find_spec("topeft.compat") is None
    assert importlib.util.find_spec("topeft.modules.utils") is None

    topcoffea_utils = importlib.import_module("topcoffea.modules.utils")
    topcoffea_hist_utils = importlib.import_module("topcoffea.modules.hist_utils")

    yield_tools = importlib.import_module("topeft.modules.yield_tools")
    data_driven_estimation = importlib.import_module("topeft.modules.dataDrivenEstimation")

    assert yield_tools.utils is topcoffea_utils
    assert yield_tools.canonicalize_process_name is topcoffea_utils.canonicalize_process_name
    assert data_driven_estimation.iterate_hist_from_pkl is topcoffea_hist_utils.iterate_hist_from_pkl
