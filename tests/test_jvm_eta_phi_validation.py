import importlib.util
from pathlib import Path

import numpy as np
import pytest


script_path = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "topeft_run2"
    / "validate_jvm_eta_phi.py"
)
specification = importlib.util.spec_from_file_location("jvm_eta_phi_validation", script_path)
jvm_eta_phi_validation = importlib.util.module_from_spec(specification)
specification.loader.exec_module(jvm_eta_phi_validation)


class SyntheticJetVetoMap:
    def evaluate(self, category, eta, phi):
        assert category in {"jetvetomap", "jetvetomap_all"}
        return 100.0 if 0.5 <= eta < 1.5 else 0.0


def test_run3_period_process_token_is_unchanged():
    processes = ("data2022", "ttbar2022", "data2022EE", "ttbar2022EE")

    assert jvm_eta_phi_validation.process_period_token("2022") == "2022"
    assert jvm_eta_phi_validation.period_processes(processes, "2022") == (
        "data2022",
        ("ttbar2022",),
    )


@pytest.mark.parametrize(
    ("display_period", "serialized_token"),
    (
        ("2016APV", "UL16APV"),
        ("2016", "UL16"),
        ("2017", "UL17"),
        ("2018", "UL18"),
    ),
)
def test_run2_period_processes_use_serialized_ul_tokens(display_period, serialized_token):
    processes = (
        f"data{serialized_token}",
        f"ttbar{serialized_token}",
        f"diboson{serialized_token}",
    )

    assert jvm_eta_phi_validation.process_period_token(display_period) == serialized_token
    assert jvm_eta_phi_validation.period_processes(processes, display_period) == (
        f"data{serialized_token}",
        (f"ttbar{serialized_token}", f"diboson{serialized_token}"),
    )


def test_classification_identifies_fully_vetoed_and_nonvetoed_bins():
    labels, fractions = jvm_eta_phi_validation.classify_analysis_bins(
        SyntheticJetVetoMap(),
        np.asarray([0.0, 0.5, 1.5, 2.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 0.5, 1.5, 2.0]),
        np.asarray([0.0, 1.0]),
    )

    assert labels[:, 0].tolist() == ["fully_nonvetoed", "fully_vetoed", "fully_nonvetoed"]
    assert fractions[:, 0].tolist() == [0.0, 1.0, 0.0]


def test_classification_preserves_boundary_mixed_bins():
    labels, fractions = jvm_eta_phi_validation.classify_analysis_bins(
        SyntheticJetVetoMap(),
        np.asarray([0.0, 0.5, 1.5, 2.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([0.0, 1.0]),
    )

    assert labels[:, 0].tolist() == ["boundary_mixed", "boundary_mixed"]
    assert fractions[:, 0].tolist() == [0.5, 0.5]


def test_summary_uses_absolute_residuals_for_signed_mc():
    summary = jvm_eta_phi_validation.summarize(
        np.asarray([[-2.0, 0.0], [0.25, 0.0]]),
        np.asarray([[True, True], [True, False]]),
        tolerance=1e-9,
    )

    assert summary == {
        "sum": -1.75,
        "absolute_sum": 2.25,
        "nonzero_bins": 2,
        "max_absolute_bin": 2.0,
    }


def test_run2_classification_uses_all_map_category():
    labels, _ = jvm_eta_phi_validation.classify_analysis_bins(
        SyntheticJetVetoMap(),
        np.asarray([0.0, 0.5, 1.5, 2.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 0.5, 1.5, 2.0]),
        np.asarray([0.0, 1.0]),
        "jetvetomap_all",
    )

    assert labels[:, 0].tolist() == [
        "fully_nonvetoed",
        "fully_vetoed",
        "fully_nonvetoed",
    ]
