import importlib.util
from pathlib import Path

import numpy as np
import pytest
from matplotlib.colors import TwoSlopeNorm


script_path = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "topeft_run2"
    / "make_jvm_reviewer_panels.py"
)
specification = importlib.util.spec_from_file_location("jvm_reviewer_panels", script_path)
jvm_reviewer_panels = importlib.util.module_from_spec(specification)
specification.loader.exec_module(jvm_reviewer_panels)


def test_period_processes_keeps_2022_and_2022ee_separate():
    processes = ("data2022", "ttbar2022", "data2022EE", "ttbar2022EE")

    assert jvm_reviewer_panels.period_processes(processes, "2022") == (
        "data2022",
        ("ttbar2022",),
    )
    assert jvm_reviewer_panels.period_processes(processes, "2022EE") == (
        "data2022EE",
        ("ttbar2022EE",),
    )


def test_pair_normalization_uses_union_and_sign_aware_mc_mode():
    data_norm, data_minimum, data_maximum, data_mode = jvm_reviewer_panels.build_normalization(
        np.asarray([[0.0, 2.0]]), np.asarray([[3.0, 1.0]]), "data"
    )
    assert (data_minimum, data_maximum, data_mode) == (0.0, 3.0, "linear_nonnegative")
    assert data_norm.vmin == 0.0 and data_norm.vmax == 3.0

    mc_norm, mc_minimum, mc_maximum, mc_mode = jvm_reviewer_panels.build_normalization(
        np.asarray([[-2.0, 1.0]]), np.asarray([[4.0, -0.5]]), "mc"
    )
    assert (mc_minimum, mc_maximum, mc_mode) == (-2.0, 4.0, "two_slope_sign_aware")
    assert isinstance(mc_norm, TwoSlopeNorm)
    assert mc_norm.vcenter == 0.0


def test_output_directory_refuses_collisions(tmp_path):
    output_dir = tmp_path / "panels"
    assert jvm_reviewer_panels.ensure_empty_output_directory(output_dir) == output_dir
    (output_dir / "existing.png").write_text("do not overwrite")

    with pytest.raises(FileExistsError, match="not empty"):
        jvm_reviewer_panels.ensure_empty_output_directory(output_dir)


def test_boundary_segments_draws_only_exposed_edges():
    segments = jvm_reviewer_panels.boundary_segments(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([[True, True], [False, False]]),
    )

    assert len(segments) == 6
