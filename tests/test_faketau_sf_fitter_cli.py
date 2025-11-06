import pathlib
import logging
import re
import sys

import pytest

hist = pytest.importorskip("hist")
np = pytest.importorskip("numpy")

from topcoffea.modules.histEFT import HistEFT

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import analysis.topeft_run2.faketau_sf_fitter as fitter


def _build_tau_histograms(process_weights=None):
    tau_edges = [20, 30, 40, 50, 60, 80, 100, 200]
    proc_axis = hist.axis.StrCategory([], name="process", growth=True)
    channel_axis = hist.axis.StrCategory([], name="channel", growth=True)
    syst_axis = hist.axis.StrCategory([], name="systematic", growth=True)
    appl_axis = hist.axis.StrCategory([], name="appl", growth=True)
    tau_axis = hist.axis.Variable(tau_edges, name="tau0pt")
    tau_sumw2_axis = hist.axis.Variable(tau_edges, name="tau0pt_sumw2")

    tau_hist = HistEFT(
        proc_axis,
        channel_axis,
        syst_axis,
        appl_axis,
        tau_axis,
        wc_names=[],
        label="Events",
    )
    tau_hist.metadata = {}

    tau_sumw2_hist = HistEFT(
        proc_axis,
        channel_axis,
        syst_axis,
        appl_axis,
        tau_sumw2_axis,
        wc_names=[],
        label="Events",
    )
    tau_sumw2_hist.metadata = {"_hist_sumw2_axis_mapping": {"tau0pt_sumw2": "tau0pt"}}

    values = [25.0, 35.0, 45.0, 55.0, 70.0, 90.0, 150.0]
    channel_scales = {
        "2los_ee_1tau_Ftau_2j": 1.0,
        "2los_ee_1tau_Ttau_2j": 0.6,
    }
    if process_weights is None:
        base_weights = {
            "ttbar": [12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0],
            "data2018": [18.0, 16.0, 15.0, 13.0, 11.0, 9.0, 8.0],
        }
    else:
        base_weights = dict(process_weights)

    def _fill(target, axis_name, process, channel, entries):
        for value, weight in entries:
            target.fill(
                process=process,
                channel=channel,
                systematic="nominal",
                appl="isSR_2lOS",
                **{axis_name: np.array([value])},
                weight=np.array([weight]),
            )

    expected_errors = {
        ("MC", "fake"): [],
        ("MC", "tight"): [],
        ("Data", "fake"): [],
        ("Data", "tight"): [],
    }

    for process, weights in base_weights.items():
        is_data = process.startswith("data")
        label_prefix = "Data" if is_data else "MC"
        for channel, scale in channel_scales.items():
            scaled = [weight * scale for weight in weights]
            entries = list(zip(values, scaled))
            sumw2_entries = [(val, w ** 2) for val, w in zip(values, scaled)]
            _fill(tau_hist, "tau0pt", process, channel, entries)
            _fill(tau_sumw2_hist, "tau0pt_sumw2", process, channel, sumw2_entries)

            key = (label_prefix, "fake" if "Ftau" in channel else "tight")
            expected_errors[key] = scaled

    return {"tau0pt": tau_hist, "tau0pt_sumw2": tau_sumw2_hist}, expected_errors


def test_tau_sumw2_histogram_passes_validation(caplog):
    histograms, expected_errors = _build_tau_histograms()
    ftau_channels = ["2los_ee_1tau_Ftau_2j"]
    ttau_channels = ["2los_ee_1tau_Ttau_2j"]

    with caplog.at_level(logging.WARNING):
        _mc_y, mc_e, _data_x, _data_y, data_e, stage_details = fitter.getPoints(
            histograms,
            ftau_channels,
            ttau_channels,
        )

    assert not any(
        "Poisson counting uncertainties" in record.getMessage()
        for record in caplog.records
    )

    mc_fake_errors = stage_details["native_yields"]["MC fake"][1]
    data_fake_errors = stage_details["native_yields"]["Data fake"][1]

    assert mc_fake_errors == pytest.approx(expected_errors[("MC", "fake")])
    assert data_fake_errors == pytest.approx(expected_errors[("Data", "fake")])
    assert mc_e.size == len(mc_fake_errors)
    assert data_e.size == len(data_fake_errors)
    assert stage_details["year_filter"]["selected_years"] is None


def test_year_filter_limits_samples():
    custom_weights = {
        "ttbarUL16": [12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0],
        "ttbarUL18": [11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
        "data2017": [17.0, 15.0, 14.0, 12.0, 10.0, 8.0, 7.0],
        "data2018": [18.0, 16.0, 15.0, 13.0, 11.0, 9.0, 8.0],
    }
    histograms, _ = _build_tau_histograms(custom_weights)
    ftau_channels = ["2los_ee_1tau_Ftau_2j"]
    ttau_channels = ["2los_ee_1tau_Ttau_2j"]

    sample_filters = fitter._resolve_year_filters(["2018"])
    _mc_y, _mc_e, _data_x, _data_y, _data_e, stage_details = fitter.getPoints(
        histograms,
        ftau_channels,
        ttau_channels,
        sample_filters=sample_filters,
    )

    summary = stage_details["year_filter"]
    assert summary["selected_years"] == ["2018"]
    assert summary["mc_samples"] == ["ttbarUL18"]
    assert summary["data_samples"] == ["data2018"]
    assert "ttbarUL16" in summary["mc_removed"]
    assert "data2017" in summary["data_removed"]


@pytest.mark.parametrize(
    "sample_filter_updates, expected_message",
    [
        (
            {"mc_whitelist": ["data"], "mc_blacklist": []},
            "No MC processes remain after applying sample filters. "
            "Filtered MC samples: ['data2018']",
        ),
        (
            {"data_whitelist": ["ttbar"], "data_blacklist": []},
            "No data processes remain after applying sample filters. "
            "Filtered data samples: ['ttbar']",
        ),
    ],
)
def test_get_points_raises_when_filters_remove_all_processes(
    sample_filter_updates, expected_message
):
    histograms, _ = _build_tau_histograms()
    ftau_channels = ["2los_ee_1tau_Ftau_2j"]
    ttau_channels = ["2los_ee_1tau_Ttau_2j"]

    sample_filters = fitter._resolve_year_filters(None)
    sample_filters.update(sample_filter_updates)

    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        fitter.getPoints(
            histograms,
            ftau_channels,
            ttau_channels,
            sample_filters=sample_filters,
        )
