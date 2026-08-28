from collections import namedtuple

import numpy as np
import pytest

from topeft.modules.datacard_tools import (
    _sanitize_negative_template_bins,
    _validate_ff_template_support,
)


systematic_key = namedtuple("systematic_key", ["systematic"])


def _templates(nominal, variation, variation_name="FFDown"):
    nominal_key = systematic_key("nominal")
    variation_key = systematic_key(variation_name)
    templates = {
        nominal_key: [np.asarray(nominal, dtype=float), np.ones(len(nominal))],
        variation_key: [
            np.asarray(variation, dtype=float),
            np.ones(len(variation)),
        ],
    }
    return templates, nominal_key, variation_key


@pytest.mark.parametrize(
    (
        "raw_nominal",
        "raw_variation",
        "expected_nominal",
        "expected_variation",
        "expected_nominal_sumw2",
        "expected_variation_sumw2",
    ),
    [
        (
            [-1.0, 2.0],
            [0.5, 3.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ),
        (
            [0.0, 2.0],
            [0.5, 3.0],
            [0.0, 2.0],
            [0.5, 3.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ),
        (
            [1.0, 2.0],
            [-0.5, 3.0],
            [1.0, 2.0],
            [0.0, 3.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ),
    ],
)
def test_negative_template_sanitation_distinguishes_raw_negative_and_zero_nominal(
    raw_nominal,
    raw_variation,
    expected_nominal,
    expected_variation,
    expected_nominal_sumw2,
    expected_variation_sumw2,
):
    templates, nominal_key, variation_key = _templates(raw_nominal, raw_variation)
    original_templates = {
        sp_key: [component.copy() for component in arr]
        for sp_key, arr in templates.items()
    }

    sanitized = _sanitize_negative_template_bins(templates)

    np.testing.assert_array_equal(sanitized[nominal_key][0], expected_nominal)
    np.testing.assert_array_equal(sanitized[variation_key][0], expected_variation)
    np.testing.assert_array_equal(sanitized[nominal_key][1], expected_nominal_sumw2)
    np.testing.assert_array_equal(
        sanitized[variation_key][1],
        expected_variation_sumw2,
    )
    for sp_key, arr in templates.items():
        for component, original_component in zip(arr, original_templates[sp_key]):
            np.testing.assert_array_equal(component, original_component)


def test_all_negative_nominal_cannot_leave_ff_support_after_sanitation():
    templates, nominal_key, variation_key = _templates(
        nominal=[-1.0, -2.0],
        variation=[0.5, -0.25],
    )

    sanitized = _sanitize_negative_template_bins(templates)

    np.testing.assert_array_equal(sanitized[nominal_key][0], [0.0, 0.0])
    np.testing.assert_array_equal(sanitized[variation_key][0], [0.0, 0.0])
    np.testing.assert_array_equal(sanitized[variation_key][1], [0.0, 0.0])
    _validate_ff_template_support(
        variation_key.systematic,
        sanitized[variation_key],
        nominal_content_is_zero=True,
        nominal_sumw2_is_zero=True,
    )


def test_true_zero_nominal_with_unsupported_ff_variation_still_raises():
    templates, _, variation_key = _templates(
        nominal=[0.0, 0.0],
        variation=[0.5, 0.0],
    )

    sanitized = _sanitize_negative_template_bins(templates)

    np.testing.assert_array_equal(sanitized[variation_key][0], [0.5, 0.0])
    with pytest.raises(
        Warning,
        match="Zero values in 'nominal' but non-zero in 'FFDown'",
    ):
        _validate_ff_template_support(
            variation_key.systematic,
            sanitized[variation_key],
            nominal_content_is_zero=True,
            nominal_sumw2_is_zero=True,
        )
