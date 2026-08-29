from __future__ import annotations

import pickle

import pytest

hist = pytest.importorskip("hist")
np = pytest.importorskip("numpy")

from analysis.topeft_run2.analysis_processor import (
    calculate_sm_sumw2_weights,
    evaluate_eft_coefficients_at_sm,
)
from topcoffea.modules import eft_helper
from topcoffea.modules.sparseHist import SparseHist


def _make_companion() -> SparseHist:
    return SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.Regular(2, 0.0, 2.0, name="njets_sumw2"),
        storage="Double",
    )


def _eval_array(histogram: SparseHist) -> np.ndarray:
    evaluated = histogram.view(flow=True, as_dict=True)
    return np.sum(
        [np.asarray(values, dtype=float) for values in evaluated.values()],
        axis=0,
    )


def test_sm_evaluation_uses_polynomial_constant_not_a_named_c0_coordinate():
    coefficients = np.asarray(
        [
            [1.5, 100.0, -20.0],
            [-2.0, 4.0, 9.0],
            [0.0, -7.0, 3.0],
        ]
    )

    np.testing.assert_array_equal(
        evaluate_eft_coefficients_at_sm(coefficients),
        np.asarray([1.5, -2.0, 0.0]),
    )


def test_eft_sumw2_squares_the_complete_sm_event_contribution():
    scalar_weights = np.asarray([2.0, -3.0, 4.0])
    coefficients = np.asarray(
        [
            [1.5, 10.0, 2.0],
            [-2.0, -5.0, 1.0],
            [0.0, 8.0, -4.0],
        ]
    )

    expected = np.asarray([9.0, 36.0, 0.0])
    corrected = calculate_sm_sumw2_weights(scalar_weights, coefficients)
    w2_coefficients = eft_helper.calc_w2_coeffs(coefficients)
    calc_w2_sm = eft_helper.calc_eft_w2(
        w2_coefficients,
        np.zeros(1, dtype=coefficients.dtype),
    ) * np.square(scalar_weights)

    np.testing.assert_allclose(corrected, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(calc_w2_sm, expected, rtol=0.0, atol=0.0)
    assert np.all(corrected >= 0.0)


def test_nominal_cancellation_does_not_cancel_sumw2():
    scalar_weights = np.asarray([2.0, 2.0])
    coefficients = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    nominal_contributions = scalar_weights * evaluate_eft_coefficients_at_sm(
        coefficients
    )
    assert np.sum(nominal_contributions) == 0.0
    assert np.sum(calculate_sm_sumw2_weights(scalar_weights, coefficients)) == 8.0


def test_selection_and_axis_masks_apply_before_sumw2_calculation():
    scalar_weights = np.asarray([1.0, 2.0, 3.0, 4.0])
    coefficients = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    selection_mask = np.asarray([True, True, False, True])
    axis_validity_mask = np.asarray([False, True, True])

    selected_weights = scalar_weights[selection_mask][axis_validity_mask]
    selected_coefficients = coefficients[selection_mask][axis_validity_mask]

    np.testing.assert_array_equal(
        calculate_sm_sumw2_weights(selected_weights, selected_coefficients),
        np.asarray([16.0, 256.0]),
    )


def test_non_eft_sumw2_retains_exact_operation_and_signed_weight_behavior():
    scalar_weights = np.asarray([3.0, -4.0, 0.0, 1.25])
    legacy = np.square(scalar_weights)
    corrected = calculate_sm_sumw2_weights(scalar_weights)

    np.testing.assert_array_equal(corrected, legacy)


def test_invalid_quadratic_coefficient_count_and_shape_mismatch_fail_clearly():
    with pytest.raises(ValueError, match="quadratic polynomial"):
        evaluate_eft_coefficients_at_sm(np.ones((2, 5)))

    with pytest.raises(ValueError, match="matching shapes"):
        calculate_sm_sumw2_weights(np.ones(3), np.ones((2, 3)))


def test_scalar_sparse_companion_preserves_consumer_operations():
    companion = _make_companion()
    corrected_weights = np.asarray([9.0, 36.0, 4.0])
    companion.fill(
        process="tllq",
        channel="3l_onZ_1b",
        systematic="nominal",
        appl="isSR_3l_onZ_1b",
        njets_sumw2=np.asarray([0.5, 1.5]),
        weight=corrected_weights[:2],
    )
    companion.fill(
        process="control",
        channel="3l_onZ_1b",
        systematic="nominal",
        appl="isSR_3l_onZ_1b",
        njets_sumw2=np.asarray([1.5]),
        weight=corrected_weights[2:],
    )

    assert isinstance(companion, SparseHist)
    assert [axis.name for axis in companion.axes] == [
        "process",
        "channel",
        "systematic",
        "appl",
        "njets_sumw2",
    ]
    np.testing.assert_allclose(_eval_array(companion), [0.0, 9.0, 40.0, 0.0])

    selected = companion.integrate("systematic", "nominal")
    grouped = selected.group("process", {"signal": ["tllq"]})
    grouped_values = grouped.view(flow=True, as_dict=True)
    np.testing.assert_allclose(
        np.asarray(next(iter(grouped_values.values()))).reshape(-1),
        np.asarray([0.0, 9.0, 36.0, 0.0]),
    )

    scaled = companion.copy()
    scaled.scale(2.0)
    np.testing.assert_allclose(_eval_array(scaled), 2.0 * _eval_array(companion))

    merged = companion.copy()
    merged += companion
    np.testing.assert_allclose(_eval_array(merged), 2.0 * _eval_array(companion))

    restored = pickle.loads(pickle.dumps(companion))
    assert isinstance(restored, SparseHist)
    np.testing.assert_allclose(_eval_array(restored), _eval_array(companion))

    from analysis.topeft_run2 import make_cr_and_sr_plots

    plot_values = make_cr_and_sr_plots._values_with_flow_or_overflow(grouped)
    np.testing.assert_allclose(np.asarray(plot_values).reshape(-1), [0.0, 9.0, 36.0, 0.0])
    assert make_cr_and_sr_plots.effective_entries(45.0, 45.0) == 45.0
