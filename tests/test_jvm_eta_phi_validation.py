import importlib.util
from pathlib import Path

import numpy as np


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
        assert category == "jetvetomap"
        return 100.0 if 0.5 <= eta < 1.5 else 0.0


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
