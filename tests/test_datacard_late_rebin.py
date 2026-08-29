import hist
import numpy as np
import pytest

from topcoffea.modules.histEFT import HistEFT
from topeft.modules.axis_binning import (
    histogram_dense_edges,
    validate_matching_histogram_edges,
)
from topeft.modules.datacard_tools import DatacardMaker


CHANNEL = "3l_1tau_1b_2j"


def _signal_histogram(*, dense_axis=None):
    if dense_axis is None:
        dense_axis = hist.axis.Regular(12, 0, 600, name="lj0pt")
    return HistEFT(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        dense_axis,
        wc_names=["ctG"],
        label="Events",
    )


def _maker_for_mode(mode, source):
    maker = DatacardMaker.__new__(DatacardMaker)
    maker.binning_mode = mode
    maker.hists = {"lj0pt": source}
    maker.coeffs = []
    maker.tolerance = 1.0e-4
    maker.verbose = False
    return maker


def test_datacard_fitting_view_preserves_eft_scaling_bin_correspondence():
    source = _signal_histogram()
    source.fill(
        process="ttH",
        channel=CHANNEL,
        systematic="nominal",
        lj0pt=np.array([25.0, 175.0, 275.0, 375.0]),
        eft_coeff=np.array(
            [
                [2.0, 1.0, 0.5],
                [3.0, -1.0, 1.0],
                [4.0, 2.0, 1.5],
                [5.0, -2.0, 2.0],
            ]
        ),
    )

    fitting = _maker_for_mode("fitting", source).binning_view(
        source.integrate("channel", [CHANNEL]), "lj0pt", CHANNEL
    )
    template = fitting.integrate("process", ["ttH"])
    scaling_hist = template.integrate("systematic", ["nominal"])

    validate_matching_histogram_edges(
        template, scaling_hist, context="synthetic datacard template/scaling"
    )
    assert np.array_equal(histogram_dense_edges(template), [0, 150, 250, 350])
    scalings = scaling_hist.make_scaling()
    # Four fitting bins are serialized after the underflow is removed: three
    # finite intervals plus the physical overflow bin.
    assert scalings.shape[-2] - 1 == 4


def test_physical_edges_reject_same_length_scaling_mismatch():
    template = _signal_histogram(
        dense_axis=hist.axis.Variable([0, 150, 250, 350], name="lj0pt")
    )
    scaling = _signal_histogram(
        dense_axis=hist.axis.Variable([0, 150, 300, 350], name="lj0pt")
    )
    with pytest.raises(ValueError, match="Physical dense-axis mismatch"):
        validate_matching_histogram_edges(
            template, scaling, context="same-length different-edge regression"
        )


@pytest.mark.parametrize(
    ("mode", "expected_selected"),
    (("processing", {"ctG"}), ("fitting", set())),
)
def test_wc_selection_uses_the_selected_card_view(mode, expected_selected):
    source = _signal_histogram()
    source.fill(
        process="ttH",
        channel=CHANNEL,
        systematic="nominal",
        lj0pt=np.array([25.0, 75.0]),
        eft_coeff=np.array([[1.0, 10.0, 0.0], [1.0, -10.0, 0.0]]),
    )
    maker = _maker_for_mode(mode, source)

    selected = maker.get_selected_wcs("lj0pt", [CHANNEL])

    # Either 50-GeV source bin alone selects ctG, while their fitting-bin sum
    # cancels. The decision must track the card-facing view in both modes.
    assert selected["ttH"] == expected_selected


def test_processing_view_uses_old_coarse_stored_axis_without_resolving_fitting():
    source = _signal_histogram(
        dense_axis=hist.axis.Variable([0, 150, 250, 500], name="lj0pt")
    )
    source.fill(
        process="ttH",
        channel=CHANNEL,
        systematic="nominal",
        lj0pt=np.array([25.0]),
        eft_coeff=np.array([[1.0, 0.0, 0.0]]),
    )
    channel_hist = source.integrate("channel", [CHANNEL])
    processing = _maker_for_mode("processing", source).binning_view(
        channel_hist, "lj0pt", CHANNEL
    )

    assert processing is channel_hist
    assert np.array_equal(histogram_dense_edges(processing), [0, 150, 250, 500])
    with pytest.raises(ValueError, match="not exactly representable"):
        _maker_for_mode("fitting", source).binning_view(
            channel_hist, "lj0pt", CHANNEL
        )


def test_selected_views_keep_sumw2_eft_and_scaling_payloads_aligned():
    source = _signal_histogram()
    sumw2 = _signal_histogram()
    source.fill(
        process="ttH",
        channel=CHANNEL,
        systematic="nominal",
        lj0pt=np.array([25.0, 75.0, 175.0]),
        eft_coeff=np.array([[2.0, 1.0, 0.5], [3.0, -1.0, 1.0], [4.0, 2.0, 1.5]]),
    )
    sumw2.fill(
        process="ttH",
        channel=CHANNEL,
        systematic="nominal",
        lj0pt=np.array([25.0, 75.0, 175.0]),
        eft_coeff=np.array([[5.0, 0.0, 0.0], [7.0, 0.0, 0.0], [11.0, 0.0, 0.0]]),
    )

    for mode, expected_edges, expected_scaling_bins in (
        ("processing", np.arange(0, 650, 50), 13),
        ("fitting", np.array([0, 150, 250, 350]), 4),
    ):
        maker = _maker_for_mode(mode, source)
        selected = maker.binning_view(
            source.integrate("channel", [CHANNEL]), "lj0pt", CHANNEL
        )
        selected_sumw2 = maker.binning_view(
            sumw2.integrate("channel", [CHANNEL]), "lj0pt", CHANNEL
        )
        validate_matching_histogram_edges(
            selected,
            selected_sumw2,
            context=f"synthetic {mode} nominal/sumw2",
        )
        assert np.array_equal(histogram_dense_edges(selected), expected_edges)
        payload = next(iter(selected.view(flow=True).values()))
        assert np.allclose(np.sum(payload, axis=0)[1:-1], [9.0, 2.0, 3.0])
        scalings = selected.integrate("process", ["ttH"]).integrate(
            "systematic", ["nominal"]
        ).make_scaling()
        assert scalings.shape[-2] - 1 == expected_scaling_bins


@pytest.mark.parametrize(
    ("family", "source_axis", "expected_edges"),
    (
        ("njets", hist.axis.Regular(2, 0, 2, name="njets"), [0, 1, 2]),
        ("lj0pt", hist.axis.Regular(12, 0, 600, name="lj0pt"), [0, 150, 250, 350]),
    ),
)
def test_scaling_json_projects_categories_before_underflow_removal(
    family, source_axis, expected_edges
):
    source = _signal_histogram(dense_axis=source_axis)
    coordinate = family
    source.fill(
        process="ttH",
        channel=CHANNEL,
        systematic="nominal",
        **{
            coordinate: np.array([0.25, 0.75]) if family == "njets" else np.array([25.0, 75.0]),
            "eft_coeff": np.array([[2.0, 1.0, 0.5], [3.0, 2.0, 1.0]]),
        },
    )
    maker = _maker_for_mode("fitting", source)
    maker.wc_ranges = {"ctG": (-1.0, 1.0)}
    channel_hist = maker.binning_view(
        source.integrate("channel", [CHANNEL]), family, CHANNEL
    )
    retained = channel_hist.integrate("process", ["ttH"]).integrate(
        "systematic", ["nominal"]
    )
    stale_scaling = np.asarray(retained.make_scaling())
    assert stale_scaling.shape[:3] == (1, 1, 1)
    assert stale_scaling.tolist()[1:] == []

    retained_before = next(iter(retained.view(flow=True).values())).copy()
    scaling_hist = maker._scaling_histogram_for_json(channel_hist, CHANNEL, "ttH")
    retained_after = next(iter(retained.view(flow=True).values()))
    expected_scaling = np.asarray(scaling_hist.make_scaling())
    records = maker.make_scalings_json(
        [], CHANNEL, family, "ttH", ["ctG"], expected_scaling
    )
    serialized_scaling = np.asarray(records[0]["scaling"])

    assert tuple(scaling_hist.categorical_axes.name) == ()
    assert np.array_equal(histogram_dense_edges(scaling_hist), expected_edges)
    np.testing.assert_allclose(retained_before, retained_after)
    np.testing.assert_allclose(serialized_scaling, expected_scaling[1:])
    assert serialized_scaling.shape[0] == expected_scaling.shape[0] - 1
    assert np.all(np.isfinite(serialized_scaling))

    historical_projection = channel_hist[
        {"channel": CHANNEL, "process": "ttH", "systematic": "nominal"}
    ]
    np.testing.assert_allclose(
        serialized_scaling, np.asarray(historical_projection.make_scaling())[1:]
    )


def test_scaling_json_rejects_retained_categorical_axes():
    source = HistEFT(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="source", growth=True),
        hist.axis.Regular(12, 0, 600, name="lj0pt"),
        wc_names=["ctG"],
        label="Events",
    )
    source.fill(
        process="ttH",
        channel=CHANNEL,
        systematic="nominal",
        source="unexpected",
        lj0pt=np.array([25.0]),
        eft_coeff=np.array([[2.0, 1.0, 0.5]]),
    )
    maker = _maker_for_mode("fitting", source)
    channel_hist = maker.binning_view(
        source.integrate("channel", [CHANNEL]), "lj0pt", CHANNEL
    )

    with pytest.raises(ValueError, match="category-projected HistEFT input"):
        maker._scaling_histogram_for_json(channel_hist, CHANNEL, "ttH")
