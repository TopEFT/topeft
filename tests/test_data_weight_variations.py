import types

import numpy as np
from coffea.analysis_tools import Weights
import pytest

from topeft.modules.systematics import (
    add_fake_factor_weights,
    validate_data_weight_variations,
)


def test_same_sign_data_weights_include_fliprate_and_requested_variations():
    weights = Weights(3)
    events = types.SimpleNamespace(
        fakefactor_2l=np.array([0.9, 1.1, 1.0]),
        fakefactor_2l_up=np.array([1.0, 1.2, 1.1]),
        fakefactor_2l_down=np.array([0.8, 1.0, 0.9]),
        nom=np.ones(3),
        fakefactor_2l_pt1=np.full(3, 1.05),
        fakefactor_2l_pt2=np.full(3, 0.95),
        fakefactor_2l_be1=np.full(3, 1.02),
        fakefactor_2l_be2=np.full(3, 0.98),
        fakefactor_2l_elclosureup=np.full(3, 1.01),
        fakefactor_2l_elclosuredown=np.full(3, 0.99),
        fakefactor_2l_muclosureup=np.full(3, 1.03),
        fakefactor_2l_muclosuredown=np.full(3, 0.97),
        flipfactor_2l=np.array([1.2, 0.8, 1.0]),
    )

    add_fake_factor_weights(
        weights,
        events,
        "2lss",
        "UL18",
        requested_data_weight_label="FF",
    )

    weights.add("fliprate", events.flipfactor_2l)

    central_modifiers = getattr(weights, "weight_modifiers", None)
    if central_modifiers is None:
        central_modifiers = getattr(weights, "_names", None)

    assert central_modifiers is not None
    assert "fliprate" in set(central_modifiers)

    assert set(weights.variations) == {"FFUp", "FFDown"}


def test_data_weight_whitelist_accepts_requested_fake_factor_variation():
    weights = Weights(2)
    events = types.SimpleNamespace(
        fakefactor_2l=np.full(2, 0.95),
        fakefactor_2l_up=np.full(2, 1.05),
        fakefactor_2l_down=np.full(2, 0.85),
        nom=np.ones(2),
        fakefactor_2l_pt1=np.full(2, 1.0),
        fakefactor_2l_pt2=np.full(2, 1.0),
        fakefactor_2l_be1=np.full(2, 1.0),
        fakefactor_2l_be2=np.full(2, 1.0),
        fakefactor_2l_elclosureup=np.full(2, 1.0),
        fakefactor_2l_elclosuredown=np.full(2, 1.0),
        fakefactor_2l_muclosureup=np.full(2, 1.0),
        fakefactor_2l_muclosuredown=np.full(2, 1.0),
    )

    add_fake_factor_weights(
        weights,
        events,
        "2lss",
        "UL18",
        requested_data_weight_label="FF",
    )

    validate_data_weight_variations(
        weights,
        {"FFUp", "FFDown", "FFptUp", "FFptDown"},
        "FF",
        "FFUp",
    )


def test_data_weight_whitelist_rejects_missing_requested_variations():
    weights = Weights(1)
    weights.add("FF", np.ones(1))

    with pytest.raises(Exception) as excinfo:
        validate_data_weight_variations(
            weights,
            {"FFUp", "FFDown"},
            "FF",
            "FFUp",
        )

    assert "Missing expected fake-factor variations" in str(excinfo.value)


def test_data_weight_whitelist_rejects_unexpected_variations():
    weights = Weights(1)
    weights.add("Bad", np.ones(1), np.ones(1), np.ones(1))

    with pytest.raises(Exception) as excinfo:
        validate_data_weight_variations(
            weights,
            {"FFUp", "FFDown"},
            "FF",
            "FFUp",
        )

    assert "Unexpected wgt variations" in str(excinfo.value)
