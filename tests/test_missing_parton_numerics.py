from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from topeft.modules.missing_parton_contract import (
    build_registry_payload_layout,
    validate_legacy_missing_parton_values,
)


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "topeft_run2"
    / "missing_parton.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "missing_parton_numerics_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def errors(size, *, down=0.0, up=0.0):
    return np.full(size, down), np.full(size, up)


def calculate(module, private, central, *, down=0.0, up=0.0):
    private = np.asarray(private, dtype=float)
    central = np.asarray(central, dtype=float)
    down_values, up_values = errors(len(private), down=down, up=up)
    return module.calculate_missing_parton_per_bin(
        private,
        central,
        down_values,
        up_values,
        base_channel="test_channel",
    )


def test_ordinary_positive_ratio_preserves_public_formula():
    module = load_module()

    parton, fraction = calculate(module, [10.0], [4.0], down=2.0)

    assert parton.tolist() == pytest.approx([math.sqrt(32.0)])
    assert fraction.tolist() == pytest.approx([math.sqrt(32.0) / 10.0])


def test_upward_private_error_is_selected_when_central_is_larger():
    module = load_module()

    parton, fraction = calculate(module, [10.0], [14.0], down=1.0, up=3.0)

    assert parton.tolist() == pytest.approx([math.sqrt(7.0)])
    assert fraction.tolist() == pytest.approx([math.sqrt(7.0) / 10.0])


def test_equal_private_and_central_rates_are_neutral():
    module = load_module()

    parton, fraction = calculate(module, [7.0], [7.0])

    assert parton.tolist() == pytest.approx([0.0])
    assert fraction.tolist() == pytest.approx([0.0])


def test_both_zero_or_effectively_zero_rates_are_neutral():
    module = load_module()

    parton, fraction = calculate(
        module,
        [0.0, 0.5e-5],
        [0.0, -0.5e-5],
    )

    assert parton.tolist() == pytest.approx([0.0, 0.0])
    assert fraction.tolist() == pytest.approx([0.0, 0.0])


def test_near_zero_positive_private_with_nonzero_central_is_neutral():
    module = load_module()

    parton, fraction = calculate(module, [0.5e-5], [1.0])

    assert parton.tolist() == [0.0]
    assert fraction.tolist() == [0.0]
    assert 1.0 + fraction[0] == 1.0


def test_near_zero_negative_private_with_nonzero_central_is_neutral():
    module = load_module()

    parton, fraction = calculate(module, [-0.5e-5], [1.0])

    assert parton.tolist() == [0.0]
    assert fraction.tolist() == [0.0]
    assert 1.0 + fraction[0] == 1.0


def test_effectively_zero_private_never_uses_threshold_as_denominator():
    module = load_module()

    _, fraction = calculate(module, [0.25e-5], [1.0e6])

    assert fraction.tolist() == [0.0]


def test_private_value_at_threshold_uses_ordinary_formula():
    module = load_module()

    parton, fraction = calculate(module, [1.0e-5], [0.0])

    assert parton.tolist() == pytest.approx([1.0e-5])
    assert fraction.tolist() == pytest.approx([1.0])


def test_zero_or_near_zero_central_with_positive_private_is_supported():
    module = load_module()

    _, fraction = calculate(module, [10.0, 10.0], [0.0, 0.5e-5])

    assert fraction.tolist() == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("bad_value", (np.nan, np.inf, -np.inf))
def test_nonfinite_numerical_inputs_fail(bad_value):
    module = load_module()

    with pytest.raises(ValueError, match="Non-finite"):
        calculate(module, [10.0], [bad_value])


def test_materially_negative_private_denominator_fails_without_absolute_value():
    module = load_module()

    with pytest.raises(
        ValueError, match="Materially negative private denominator"
    ) as exc_info:
        calculate(module, [-0.5], [0.0])

    assert "no clipping or absolute-value fallback" in str(exc_info.value)


def test_uncertainty_larger_than_rate_difference_yields_neutral_fraction():
    module = load_module()

    parton, fraction = calculate(module, [10.0], [9.0], down=2.0)

    assert parton.tolist() == pytest.approx([0.0])
    assert fraction.tolist() == pytest.approx([0.0])


def _synthetic_layout(base_category, jet_lst):
    return build_registry_payload_layout(
        "ALL_CH_LST_SR",
        {
            "synthetic": {
                "lep_chan_lst": [[base_category]],
                "jet_lst": list(jet_lst),
            }
        },
    ).categories[0]


def _card(module, nominal, *, shapes=(), rate_systematics=()):
    nominal = np.asarray(nominal, dtype=float)
    return module.base_category_card_data(
        nominal_values=nominal,
        shape_values=tuple(np.asarray(values, dtype=float) for values in shapes),
        bin_edges=np.arange(len(nominal) + 1, dtype=float),
        parsed_txt=module.parsed_card(
            process_names=("tllq_sm",),
            rates=(float(np.sum(nominal)),),
            rate_systematics=tuple(rate_systematics),
        ),
    )


@pytest.mark.parametrize("terminal_threshold", (3, 4, 5, 6, 7))
def test_terminal_tail_aggregates_source_inputs_before_formula(
    terminal_threshold,
):
    module = load_module()
    base_category = "synthetic"
    private_values = np.asarray([11.0, 9.0, 13.0, 8.0, 7.0, 6.0, 5.0, 4.0])
    central_values = np.asarray([2.0, 3.0, 4.0, 2.0, 1.0, 3.0, 2.0, -2.0])
    shape_down = private_values - 0.5
    shape_up = private_values + 0.75
    rate_systematics = (("rate", ("0.9/1.2",)),)
    private_card = _card(
        module,
        private_values,
        shapes=(shape_down, shape_up),
        rate_systematics=rate_systematics,
    )
    central_card = _card(module, central_values)
    layout = _synthetic_layout(
        base_category,
        (f">{terminal_threshold}",),
    )

    stored = module.build_category_payload(
        base_channel=base_category,
        private_card=private_card,
        central_card=central_card,
        layout=layout,
    )

    down_error, up_error = module.private_rate_errors(private_card)
    _, direct_fractions = module.calculate_missing_parton_per_bin(
        private_values[:terminal_threshold],
        central_values[:terminal_threshold],
        down_error[:terminal_threshold],
        up_error[:terminal_threshold],
        base_channel=base_category,
    )
    tail = slice(terminal_threshold, None)
    private_total = float(np.sum(private_values[tail]))
    central_total = float(np.sum(central_values[tail]))
    shape_down_shift = float(np.sum(shape_down[tail] - private_values[tail]))
    aggregate_down_error = math.hypot(
        abs(shape_down_shift),
        0.1 * private_total,
    )
    aggregate_delta = private_total - central_total
    expected_tail = math.sqrt(
        max(aggregate_delta**2 - aggregate_down_error**2, 0.0)
    ) / private_total

    assert stored.shape == (terminal_threshold + 1,)
    assert stored[:terminal_threshold] == pytest.approx(direct_fractions)
    assert stored[terminal_threshold] == pytest.approx(expected_tail)
    assert len(stored) - 1 == terminal_threshold

    per_bin_missing, _ = module.calculate_missing_parton_per_bin(
        private_values[tail],
        central_values[tail],
        down_error[tail],
        up_error[tail],
        base_channel=base_category,
    )
    combined_derived_fraction = float(np.sum(per_bin_missing)) / private_total
    if terminal_threshold < 7:
        assert stored[terminal_threshold] != pytest.approx(
            combined_derived_fraction
        )
    else:
        assert stored[terminal_threshold] == pytest.approx(
            combined_derived_fraction
        )


def test_exact_terminal_keeps_every_public_index_direct():
    module = load_module()
    private_values = np.arange(10.0, 18.0)
    central_values = np.arange(2.0, 10.0)
    layout = _synthetic_layout("synthetic", ("=2", "=3", "=4"))

    stored = module.build_category_payload(
        base_channel="synthetic",
        private_card=_card(module, private_values),
        central_card=_card(module, central_values),
        layout=layout,
    )
    _, expected = calculate(
        module,
        private_values[:5],
        central_values[:5],
    )

    assert stored.shape == (5,)
    assert stored == pytest.approx(expected)


def test_numerical_array_shape_mismatch_fails():
    module = load_module()

    with pytest.raises(ValueError, match="array mismatch"):
        module.calculate_missing_parton_per_bin(
            np.ones(8),
            np.ones(7),
            np.zeros(8),
            np.zeros(8),
            base_channel="test_channel",
        )


def test_invalid_stored_fraction_and_kappa_are_rejected():
    with pytest.raises(ValueError, match="Negative stored"):
        validate_legacy_missing_parton_values(
            [0.0, -0.1],
            base_channel="test_channel",
            expected_length=2,
        )


def test_nonfinite_stored_fraction_is_rejected():
    with pytest.raises(ValueError, match="Non-finite"):
        validate_legacy_missing_parton_values(
            [0.0, np.inf],
            base_channel="test_channel",
            expected_length=2,
        )
