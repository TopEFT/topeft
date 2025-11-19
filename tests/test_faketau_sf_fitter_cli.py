import gzip
import json
import logging
import math
import pathlib
import re
import subprocess
import sys

import pytest

hist = pytest.importorskip("hist")
np = pytest.importorskip("numpy")
cloudpickle = pytest.importorskip("cloudpickle")

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


def _compute_expected_tau_sf_payload(histograms, channels_json_path=None):
    ftau_channels, ttau_channels = fitter.load_tau_control_channels(
        channels_json_path
    )
    sample_filters = fitter._resolve_year_filters(None)

    y_mc, yerr_mc, x_data, y_data, yerr_data, stage_details = fitter.getPoints(
        histograms,
        ftau_channels,
        ttau_channels,
        sample_filters=sample_filters,
    )

    def _as_flat_float(array):
        return np.asarray(array, dtype=float).reshape(-1)

    y_mc = _as_flat_float(y_mc)
    yerr_mc = _as_flat_float(yerr_mc)
    x_data = _as_flat_float(x_data)
    y_data = _as_flat_float(y_data)
    yerr_data = _as_flat_float(yerr_data)
    full_x_data = x_data.copy()

    regroup_labels = np.array(stage_details["regroup_labels"], dtype=object)

    sf_values = np.divide(
        y_data,
        y_mc,
        out=np.full_like(y_data, np.nan, dtype=float),
        where=(y_mc != 0),
    )
    sf_errors = fitter._combine_ratio_uncertainty_array(
        y_data,
        yerr_data,
        y_mc,
        yerr_mc,
    )

    sf_errors = np.where(sf_errors <= 0, 1e-3, sf_errors)

    valid = (
        np.isfinite(sf_values)
        & np.isfinite(sf_errors)
        & (sf_errors > 0)
        & np.isfinite(full_x_data)
        & np.isfinite(y_mc)
        & (y_mc > 0)
        & np.isfinite(y_data)
    )

    tau_pt_edges = stage_details.get("tau_pt_edges")
    if tau_pt_edges is None:
        tau_pt_edges = np.asarray(fitter.TAU_PT_BIN_EDGES, dtype=float)
    else:
        tau_pt_edges = np.asarray(tau_pt_edges, dtype=float)

    (
        filtered_regroup_labels,
        filtered_mc_summary,
        filtered_data_summary,
    ) = fitter._filter_and_validate_regroup_summaries(
        valid,
        regroup_labels,
        tau_pt_edges,
        full_x_data,
        stage_details.get("mc_regroup_summary"),
        stage_details.get("data_regroup_summary"),
    )

    pt_bin_starts = stage_details.get("tau_pt_bin_starts")
    if pt_bin_starts is None:
        pt_bin_starts = np.asarray(tau_pt_edges[:-1], dtype=float)
    else:
        pt_bin_starts = np.asarray(pt_bin_starts, dtype=float)

    report_pt_values = full_x_data.copy()
    if report_pt_values.size:
        report_pt_values = report_pt_values[valid]
    else:
        report_pt_values = pt_bin_starts.copy()

    sf_values = sf_values[valid]
    sf_errors = sf_errors[valid]
    x_data = x_data[valid]

    if sf_values.size < 2:
        raise RuntimeError(
            "Insufficient valid tau fake-rate points for fitting: "
            f"only {sf_values.size} bin(s) remain after filtering."
        )

    c0, c1, cov = fitter.SF_fit(sf_values, sf_errors, x_data)

    eigenvalues, eigenvectors = np.linalg.eig(cov)
    lv0 = np.sqrt(abs(eigenvalues.dot(eigenvectors[0])))
    lv1 = np.sqrt(abs(eigenvalues.dot(eigenvectors[1])))

    nominal_sf = c1 * report_pt_values + c0
    sf_up = (1 + lv0) * c0 + (1 + lv1) * c1 * report_pt_values
    sf_down = (1 - lv0) * c0 + (1 - lv1) * c1 * report_pt_values

    regroup_summary = (
        filtered_mc_summary
        or filtered_data_summary
        or stage_details.get("mc_regroup_summary")
        or stage_details.get("data_regroup_summary")
    )
    if regroup_summary is None:
        raise RuntimeError("Unable to determine regroup summary for tau fake SF payload.")

    def _format_json_edge(edge):
        edge = float(edge)
        if math.isinf(edge):
            return "inf"
        return format(edge, ".15g")

    tau_sf_entries = {}
    for entry, value, up_val, down_val in zip(
        regroup_summary,
        nominal_sf,
        sf_up,
        sf_down,
    ):
        start, stop = entry["slice"]
        low_edge = float(tau_pt_edges[start])
        high_edge = float(tau_pt_edges[stop])
        bin_key = f"pt:[{_format_json_edge(low_edge)},{_format_json_edge(high_edge)}]"
        tau_sf_entries[bin_key] = {
            "value": float(value),
            "up": float(up_val),
            "down": float(down_val),
        }

    return {"TauSF": {"pt": tau_sf_entries}}


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


def test_get_points_raises_when_tau_histogram_missing():
    histograms, _ = _build_tau_histograms()
    histograms = dict(histograms)
    histograms.pop("tau0pt")
    ftau_channels = ["2los_ee_1tau_Ftau_2j"]
    ttau_channels = ["2los_ee_1tau_Ttau_2j"]

    with pytest.raises(RuntimeError) as excinfo:
        fitter.getPoints(
            histograms,
            ftau_channels,
            ttau_channels,
        )

    message = str(excinfo.value)
    assert "missing the required 'tau0pt' histogram" in message
    assert "Available histograms: tau0pt_sumw2" in message


def test_cli_outputs_tau_fake_sf_json(tmp_path):
    process_weights = {
        "ttbar": [12.0, 11.0, 9.0, 8.0, 7.0, 6.0, 5.0],
        "data2018": [19.0, 16.0, 14.0, 12.0, 10.0, 8.0, 7.0],
    }
    histograms, _ = _build_tau_histograms(process_weights)

    tau_bin_centers = np.asarray([25.0, 35.0, 45.0, 55.0, 70.0, 90.0, 150.0])
    ttau_adjustments = np.asarray([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02])
    histograms["tau0pt"].fill(
        process="data2018",
        channel="2los_ee_1tau_Ttau_2j",
        systematic="nominal",
        appl="isSR_2lOS",
        tau0pt=tau_bin_centers,
        weight=ttau_adjustments,
    )
    histograms["tau0pt_sumw2"].fill(
        process="data2018",
        channel="2los_ee_1tau_Ttau_2j",
        systematic="nominal",
        appl="isSR_2lOS",
        tau0pt_sumw2=tau_bin_centers,
        weight=ttau_adjustments ** 2,
    )
    pkl_path = tmp_path / "toy.pkl.gz"
    with gzip.open(pkl_path, "wb") as handle:
        cloudpickle.dump(histograms, handle)

    channel_config = {
        "TAU_CH_LST_CR": {
            "2los_1tau": {
                "lep_chan_lst": [
                    [
                        "2los_1tau_Ftau",
                        "2los",
                        "2l_nozeeveto",
                        "bmask_atleast1m2l",
                        "1Ftau",
                    ],
                    [
                        "2los_1tau_Ttau",
                        "2los",
                        "2l_nozeeveto",
                        "bmask_atleast1m2l",
                        "1tau",
                    ],
                ],
                "lep_flav_lst": ["ee"],
                "appl_lst": ["isSR_2lOS"],
                "jet_lst": ["=2"],
            }
        }
    }
    channels_json = tmp_path / "channels.json"
    with channels_json.open("w", encoding="utf-8") as handle:
        json.dump(channel_config, handle)

    expected_payload = _compute_expected_tau_sf_payload(
        histograms, channels_json
    )

    json_path = tmp_path / "tau_sf.json"
    script_path = ROOT / "analysis" / "topeft_run2" / "faketau_sf_fitter.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--pkl-file-path",
            str(pkl_path),
            "--channels-json",
            str(channels_json),
            "--output-json",
            str(json_path),
        ],
        check=True,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert json_path.exists(), "CLI did not produce the expected JSON output file"

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload.keys() == expected_payload.keys() == {"TauSF"}

    actual_entries = payload["TauSF"].get("pt", {})
    expected_entries = expected_payload["TauSF"].get("pt", {})
    assert actual_entries.keys() == expected_entries.keys()

    for bin_key, expected_values in expected_entries.items():
        actual_values = actual_entries[bin_key]
        for field in ("value", "up", "down"):
            assert field in actual_values
            assert actual_values[field] == pytest.approx(expected_values[field])
