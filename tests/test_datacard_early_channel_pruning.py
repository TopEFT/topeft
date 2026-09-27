from __future__ import annotations

import copy

import hist
import numpy as np

from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules import datacard_tools
from topeft.modules.datacard_tools import DatacardMaker
from topeft.modules.nominal_schema import (
    eft_nominal_key,
    materialize_legacy_histogram_dict,
    scalar_nominal_key,
)


FAMILY = "lj0pt"
REQUESTED_CHANNEL = "2lss_4t_m_4j"
UNREQUESTED_CHANNEL = "2lss_4t_p_4j"


def _axes(dense_name):
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.Regular(12, 0.0, 600.0, name=dense_name),
    )


def _fill_scalar(histogram, *, process, channel, systematic, values, weights):
    histogram.fill(
        process=process,
        channel=channel,
        systematic=systematic,
        appl="isSR_2lSS",
        **{histogram.dense_axes[0].name: np.asarray(values)},
        weight=np.asarray(weights),
        record_raw_count=True,
    )


def _fill_eft(histogram, *, channel, systematic, coefficients):
    values = np.asarray([25.0, 175.0, 275.0])
    histogram.fill(
        process="ttH_2022",
        channel=channel,
        systematic=systematic,
        appl="isSR_2lSS",
        lj0pt=values,
        weight=np.ones(len(values)),
        eft_coeff=np.asarray(coefficients),
        record_raw_count=True,
    )


def _split_payload():
    scalar = SparseHist(*_axes(FAMILY), storage="Double", track_raw_counts=True)
    eft = HistEFT(
        *_axes(FAMILY),
        wc_names=["ctG"],
        label="Events",
        track_raw_counts=True,
    )
    sumw2 = SparseHist(
        *_axes(f"{FAMILY}_sumw2"),
        storage="Double",
        track_raw_counts=True,
    )

    for channel, offset in (
        (REQUESTED_CHANNEL, 0.0),
        (UNREQUESTED_CHANNEL, 100.0),
    ):
        for systematic, scale in (
            ("nominal", 1.0),
            ("JES_2022Up", 1.1),
            ("JES_2022Down", 0.9),
        ):
            _fill_scalar(
                scalar,
                process="background_2022",
                channel=channel,
                systematic=systematic,
                values=[25.0, 175.0, 275.0],
                weights=[scale * (2.0 + offset), scale * 3.0, scale * 4.0],
            )
            _fill_eft(
                eft,
                channel=channel,
                systematic=systematic,
                coefficients=[
                    [2.0 + offset, 1.0, 0.5],
                    [3.0, -1.0, 1.0],
                    [4.0, 2.0, 1.5],
                ],
            )

        for process, weights in (
            ("background_2022", [5.0 + offset, 7.0, 11.0]),
            ("ttH_2022", [13.0 + offset, 17.0, 19.0]),
        ):
            _fill_scalar(
                sumw2,
                process=process,
                channel=channel,
                systematic="nominal",
                values=[25.0, 175.0, 275.0],
                weights=weights,
            )

    return {
        scalar_nominal_key(FAMILY): scalar,
        eft_nominal_key(FAMILY): eft,
        f"{FAMILY}_sumw2": sumw2,
    }


def _assert_histograms_equal(actual, expected):
    assert type(actual) is type(expected)
    assert tuple(actual.categorical_axes.name) == tuple(expected.categorical_axes.name)
    assert [list(axis) for axis in actual.categorical_axes] == [
        list(axis) for axis in expected.categorical_axes
    ]
    assert np.array_equal(actual.dense_axes[0].edges, expected.dense_axes[0].edges)
    if isinstance(actual, HistEFT):
        assert actual.wc_names == expected.wc_names
    actual_values = actual.view(flow=True, as_dict=True)
    expected_values = expected.view(flow=True, as_dict=True)
    assert set(actual_values) == set(expected_values)
    for coordinate in actual_values:
        assert np.array_equal(actual_values[coordinate], expected_values[coordinate])


def test_early_projection_preserves_selected_split_payload_semantics():
    source = _split_payload()
    baseline_split = datacard_tools._card_numerical_split_view(source)
    early_split = datacard_tools._card_numerical_split_view(
        source,
        channel_patterns=[rf"^{REQUESTED_CHANNEL}$"],
    )

    for source_histogram in source.values():
        assert source_histogram.track_raw_counts
        assert set(source_histogram.axes["channel"]) == {
            REQUESTED_CHANNEL,
            UNREQUESTED_CHANNEL,
        }
    for early_histogram in early_split.values():
        assert not early_histogram.track_raw_counts
        assert list(early_histogram.axes["channel"]) == [REQUESTED_CHANNEL]

    baseline = materialize_legacy_histogram_dict(baseline_split)
    early = materialize_legacy_histogram_dict(early_split)
    assert set(early) == {FAMILY, f"{FAMILY}_sumw2"}
    for key in early:
        expected = baseline[key].prune("channel", [REQUESTED_CHANNEL])
        _assert_histograms_equal(early[key], expected)


def test_datacard_maker_projects_before_legacy_materialization(monkeypatch, tmp_path):
    observed_channels = []
    real_materialize = datacard_tools.materialize_legacy_histogram_dict

    def inspect_materialization_input(histograms, **kwargs):
        observed_channels.append(
            {
                key: list(histogram.axes["channel"])
                for key, histogram in histograms.items()
            }
        )
        return real_materialize(histograms, **kwargs)

    monkeypatch.setattr(
        datacard_tools,
        "materialize_legacy_histogram_dict",
        inspect_materialization_input,
    )
    DatacardMaker(
        hists=_split_payload(),
        out_dir=str(tmp_path),
        var_lst=[FAMILY],
        channel_patterns=[rf"^{REQUESTED_CHANNEL}$"],
        year_lst=["2022"],
        do_nuisance=True,
        skip_missing_parton_rate_syst=True,
        verbose=False,
    )

    assert observed_channels
    assert all(
        channels == [REQUESTED_CHANNEL]
        for channels in observed_channels[0].values()
    )


def test_early_and_late_selected_datacard_views_are_equivalent(tmp_path):
    common = {
        "out_dir": str(tmp_path),
        "var_lst": [FAMILY],
        "year_lst": ["2022"],
        "do_nuisance": True,
        "skip_missing_parton_rate_syst": True,
        "verbose": False,
    }
    baseline = DatacardMaker(hists=copy.deepcopy(_split_payload()), **common)
    early = DatacardMaker(
        hists=copy.deepcopy(_split_payload()),
        channel_patterns=[rf"^{REQUESTED_CHANNEL}$"],
        **common,
    )

    for key in (FAMILY, f"{FAMILY}_sumw2"):
        expected = baseline.hists[key].prune("channel", [REQUESTED_CHANNEL])
        _assert_histograms_equal(early.hists[key], expected)

    baseline_wcs = baseline.get_selected_wcs(FAMILY, [REQUESTED_CHANNEL])
    early_wcs = early.get_selected_wcs(FAMILY, [REQUESTED_CHANNEL])
    assert early_wcs == baseline_wcs

    baseline_channel = baseline.binning_view(
        baseline.hists[FAMILY].integrate("channel", [REQUESTED_CHANNEL]),
        FAMILY,
        REQUESTED_CHANNEL,
    )
    early_channel = early.binning_view(
        early.hists[FAMILY].integrate("channel", [REQUESTED_CHANNEL]),
        FAMILY,
        REQUESTED_CHANNEL,
    )
    _assert_histograms_equal(early_channel, baseline_channel)

    baseline_scaling = baseline._scaling_histogram_for_json(
        baseline_channel, REQUESTED_CHANNEL, "ttH"
    ).make_scaling()
    early_scaling = early._scaling_histogram_for_json(
        early_channel, REQUESTED_CHANNEL, "ttH"
    ).make_scaling()
    assert np.array_equal(early_scaling, baseline_scaling)


def test_unfiltered_split_view_retains_all_channels():
    source = _split_payload()
    unfiltered = datacard_tools._card_numerical_split_view(source)

    for histogram in unfiltered.values():
        assert list(histogram.axes["channel"]) == [
            REQUESTED_CHANNEL,
            UNREQUESTED_CHANNEL,
        ]
