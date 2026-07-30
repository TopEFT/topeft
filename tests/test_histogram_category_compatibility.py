import pytest

from topeft.modules.histogram_category_compatibility import (
    FAKE_TAU_OBJECT,
    PTZ_WTAU_CHANNEL_FILL,
    TIGHT_TAU_OBJECT,
    category_capabilities,
    find_incompatible_histograms,
    histogram_category_compatibility_error,
    validate_histogram_category_compatibility,
)


ZERO_TAU_CATEGORY = {
    "lep_chan_lst": [
        ["misleading_1tau_onZ_name", "2lss", "0tau"],
    ]
}
TIGHT_TAU_CATEGORY = {
    "lep_chan_lst": [
        ["ordinary_tight_tau_channel", "1l", "1tau"],
    ]
}
FAKE_TAU_CATEGORY = {
    "lep_chan_lst": [
        ["ordinary_fake_tau_channel", "2los", "1Ftau"],
    ]
}
PTZ_WTAU_CATEGORY = {
    "lep_chan_lst": [
        ["2lss_m_1tau_onZ", "2lss", "1tau", "onZ_tau"],
    ]
}


@pytest.mark.parametrize(
    ("histogram_family", "required_capability"),
    [
        ("ptz_wtau", PTZ_WTAU_CHANNEL_FILL),
        ("tau0Fpt", FAKE_TAU_OBJECT),
        ("tau0Tpt", TIGHT_TAU_OBJECT),
    ],
)
def test_zero_tau_plus_tau_histogram_fails(
    histogram_family,
    required_capability,
):
    incompatible = find_incompatible_histograms(
        [histogram_family],
        selected_category_dicts=({"zero_tau": ZERO_TAU_CATEGORY},),
    )

    assert len(incompatible) == 1
    assert incompatible[0].histogram_family == histogram_family
    assert incompatible[0].required_capability == required_capability


@pytest.mark.parametrize(
    ("histogram_family", "category_definition"),
    [
        ("ptz_wtau", PTZ_WTAU_CATEGORY),
        ("tau0Fpt", FAKE_TAU_CATEGORY),
        ("tau0Tpt", TIGHT_TAU_CATEGORY),
    ],
)
def test_tau_capable_plus_each_tau_histogram_passes(
    histogram_family,
    category_definition,
):
    validate_histogram_category_compatibility(
        [histogram_family],
        selected_category_dicts=({"tau": category_definition},),
        histogram_selection_explicit=True,
    )


def test_mixed_tau_and_zero_tau_passes():
    validate_histogram_category_compatibility(
        ["ptz_wtau"],
        selected_category_dicts=(
            {"zero_tau": ZERO_TAU_CATEGORY},
            {"tau": PTZ_WTAU_CATEGORY},
        ),
        histogram_selection_explicit=True,
    )


def test_ordinary_histogram_plus_zero_tau_passes():
    validate_histogram_category_compatibility(
        ["met"],
        selected_category_dicts=({"zero_tau": ZERO_TAU_CATEGORY},),
        histogram_selection_explicit=True,
    )


def test_unknown_histogram_without_requirement_preserves_existing_behavior():
    validate_histogram_category_compatibility(
        ["future_histogram"],
        selected_category_dicts=({"zero_tau": ZERO_TAU_CATEGORY},),
        histogram_selection_explicit=True,
    )


def test_no_name_substring_heuristic():
    assert category_capabilities(ZERO_TAU_CATEGORY) == frozenset()
    assert (
        category_capabilities(
            {
                "lep_chan_lst": [
                    ["new_2lss_1tau_onZ_name", "2lss", "1tau", "onZ_tau"],
                ]
            }
        )
        == frozenset({FAKE_TAU_OBJECT, TIGHT_TAU_OBJECT})
    )


def test_diagnostic_contains_histogram_category_and_requirement():
    with pytest.raises(histogram_category_compatibility_error) as excinfo:
        validate_histogram_category_compatibility(
            ["ptz_wtau"],
            selected_category_dicts=({"zero_tau": ZERO_TAU_CATEGORY},),
            histogram_selection_explicit=True,
        )

    message = str(excinfo.value)
    assert "ptz_wtau" in message
    assert "zero_tau" in message
    assert PTZ_WTAU_CHANNEL_FILL in message
    assert "Processing was not started" in message
    assert "never silently removed" in message


def test_implicit_non_product_empty_family_preserves_historical_behavior():
    incompatible = validate_histogram_category_compatibility(
        ["ptz_wtau"],
        selected_category_dicts=({"zero_tau": ZERO_TAU_CATEGORY},),
        histogram_selection_explicit=False,
    )

    assert [item.histogram_family for item in incompatible] == ["ptz_wtau"]


def test_implicit_requested_product_empty_family_fails():
    with pytest.raises(histogram_category_compatibility_error) as excinfo:
        validate_histogram_category_compatibility(
            ["ptz_wtau"],
            selected_category_dicts=({"zero_tau": ZERO_TAU_CATEGORY},),
            histogram_selection_explicit=False,
            requested_data_driven_products=("nonprompt",),
        )

    assert "histogram_selection=implicit/default" in str(excinfo.value)
    assert "product_required_empty_family=yes" in str(excinfo.value)
