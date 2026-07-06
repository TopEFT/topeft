import pytest

from analysis.topeft_run2 import faketau_sf_fitter as fitter


FTAU_SPLIT_BINS = (
    "2los_ee_1tau_Ftau_2j",
    "2los_em_1tau_Ftau_2j",
    "2los_mm_1tau_Ftau_2j",
)
TTAU_SPLIT_BINS = (
    "2los_ee_1tau_Ttau_2j",
    "2los_em_1tau_Ttau_2j",
    "2los_mm_1tau_Ttau_2j",
)
FTAU_AGGREGATE_BIN = "2los_1tau_Ftau_2j"
TTAU_AGGREGATE_BIN = "2los_1tau_Ttau_2j"


@pytest.mark.parametrize(
    "tau_family, configured_bins, aggregate_bin",
    [
        ("Ftau", FTAU_SPLIT_BINS, FTAU_AGGREGATE_BIN),
        ("Ttau", TTAU_SPLIT_BINS, TTAU_AGGREGATE_BIN),
    ],
)
def test_aggregate_only_tau_bins_are_accepted(
    tau_family, configured_bins, aggregate_bin
):
    resolution = fitter.resolve_tau_cr_channel_bins(
        {FTAU_AGGREGATE_BIN, TTAU_AGGREGATE_BIN, "2los_CRZ_2j"},
        configured_bins,
        tau_family=tau_family,
        hist_name="tau0Fpt" if tau_family == "Ftau" else "tau0Tpt",
    )

    assert resolution["resolution_mode"] == "aggregate"
    assert resolution["selected_bins"] == (aggregate_bin,)
    assert resolution["missing_flavor_split_bins"] == configured_bins


def test_complete_flavor_split_bins_are_preferred_over_aggregate():
    resolution = fitter.resolve_tau_cr_channel_bins(
        {*FTAU_SPLIT_BINS, FTAU_AGGREGATE_BIN},
        FTAU_SPLIT_BINS,
        tau_family="Ftau",
        hist_name="tau0Fpt",
    )

    assert resolution["resolution_mode"] == "flavor_split"
    assert resolution["selected_bins"] == FTAU_SPLIT_BINS
    assert FTAU_AGGREGATE_BIN not in resolution["selected_bins"]


def test_incomplete_flavor_split_bins_use_aggregate_without_double_counting():
    resolution = fitter.resolve_tau_cr_channel_bins(
        {FTAU_SPLIT_BINS[0], FTAU_AGGREGATE_BIN},
        FTAU_SPLIT_BINS,
        tau_family="Ftau",
        hist_name="tau0Fpt",
    )

    assert resolution["resolution_mode"] == "aggregate"
    assert resolution["selected_bins"] == (FTAU_AGGREGATE_BIN,)
    assert FTAU_SPLIT_BINS[0] not in resolution["selected_bins"]
    assert resolution["missing_flavor_split_bins"] == FTAU_SPLIT_BINS[1:]


def test_missing_split_and_aggregate_bins_raise_actionable_error():
    with pytest.raises(RuntimeError) as exc_info:
        fitter.resolve_tau_cr_channel_bins(
            {"2los_CRZ_2j"},
            FTAU_SPLIT_BINS,
            tau_family="Ftau",
            hist_name="tau0Fpt",
        )

    message = str(exc_info.value)
    assert "The 'tau0Fpt' histogram" in message
    assert "resolution attempted: complete flavor-split bins, then aggregate fallback" in message
    assert "missing flavor-split bins" in message
    assert "missing aggregate fallback bins" in message
    assert FTAU_AGGREGATE_BIN in message
    assert "aggregate fallback available: no" in message


def test_observed_aggregate_channel_set_resolves_both_tau_families():
    available_bins = {
        FTAU_AGGREGATE_BIN,
        TTAU_AGGREGATE_BIN,
        "2los_CRZ_2j",
    }

    ftau_resolution = fitter.resolve_tau_cr_channel_bins(
        available_bins,
        FTAU_SPLIT_BINS,
        tau_family="Ftau",
        hist_name="tau0Fpt",
    )
    ttau_resolution = fitter.resolve_tau_cr_channel_bins(
        available_bins,
        TTAU_SPLIT_BINS,
        tau_family="Ttau",
        hist_name="tau0Tpt",
    )

    assert ftau_resolution["selected_bins"] == (FTAU_AGGREGATE_BIN,)
    assert ttau_resolution["selected_bins"] == (TTAU_AGGREGATE_BIN,)
