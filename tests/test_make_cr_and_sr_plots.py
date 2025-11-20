import warnings

import hist
import numpy as np
import pytest
from collections import defaultdict

from analysis.topeft_run2 import make_cr_and_sr_plots


class _DummyHist:
    def __init__(self):
        self.scale_factors = []

    def eval(self, _):
        return {"sample": np.zeros(4)}

    def scale(self, factor):
        self.scale_factors.append(factor)


def test_unit_normalization_skips_empty_histograms(monkeypatch):
    dummy_mc = _DummyHist()
    dummy_data = _DummyHist()

    def _stop_after_normalization(*args, **kwargs):
        raise RuntimeError("stop-after-normalization")

    monkeypatch.setattr(make_cr_and_sr_plots.plt, "subplots", _stop_after_normalization)

    logged_messages = []

    def _capture_warning(msg, *args, **kwargs):
        if args:
            msg = msg % args
        logged_messages.append(msg)

    monkeypatch.setattr(make_cr_and_sr_plots.logger, "warning", _capture_warning)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeError, match="stop-after-normalization"):
            make_cr_and_sr_plots.make_region_stacked_ratio_fig(
                h_mc=dummy_mc,
                h_data=dummy_data,
                unit_norm_bool=True,
                bins=np.array([0.0, 1.0]),
            )

    assert not dummy_mc.scale_factors
    assert not dummy_data.scale_factors

    assert any("Skipping MC unit normalization" in msg for msg in logged_messages)
    assert any("Skipping data unit normalization" in msg for msg in logged_messages)


def test_unmatched_sample_gets_fallback_group(monkeypatch):
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    value_axis = hist.axis.Regular(2, 0.0, 2.0, name="lj0pt")
    h_mc = hist.Hist(process_axis, value_axis, storage=hist.storage.Double())
    h_data = hist.Hist(process_axis, value_axis, storage=hist.storage.Double())

    mc_inputs = {
        "ttbarSL": np.array([10.0, 5.0]),
        "mysteryProcess": np.array([3.0, 2.0]),
    }

    for proc, values in mc_inputs.items():
        for bin_idx, weight in enumerate(values):
            h_mc.fill(process=proc, lj0pt=[bin_idx + 0.25], weight=[weight])

    data_yields = np.sum(list(mc_inputs.values()), axis=0) + np.array([1.0, 0.0])
    for bin_idx, weight in enumerate(data_yields):
        h_data.fill(process="data", lj0pt=[bin_idx + 0.25], weight=[weight])

    pattern_map = {"Top": ["ttbar"]}
    samples = list(mc_inputs.keys())

    with monkeypatch.context() as m:
        captured = []

        def _log_warning(msg, *args, **kwargs):
            if args:
                msg = msg % args
            captured.append(msg)

        m.setattr(make_cr_and_sr_plots.logger, "warning", _log_warning)
        group_map = make_cr_and_sr_plots.populate_group_map(samples, pattern_map)

    assert "Top" in group_map
    assert "mysteryProcess" in group_map
    assert group_map["mysteryProcess"] == ["mysteryProcess"]
    assert any("mysteryProcess" in msg for msg in captured)

    plotted_calls = []

    def _fake_histplot(*args, **kwargs):
        plotted_calls.append({"args": args, "kwargs": kwargs})
        return None

    monkeypatch.setattr(make_cr_and_sr_plots.hep, "histplot", _fake_histplot)
    monkeypatch.setattr(
        make_cr_and_sr_plots.hist.Hist,
        "as_hist",
        lambda self, mapping=None: self,
        raising=False,
    )

    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        unit_norm_bool=False,
        bins=None,
        group=group_map,
        var="lj0pt",
    )

    try:
        assert plotted_calls, "Expected histplot to be called at least once"
        mc_call = plotted_calls[0]
        mc_stack_inputs = mc_call["args"][0]
        stacked_total = np.sum(np.stack(mc_stack_inputs), axis=0)
        mc_totals = h_mc[{"process": sum}].as_hist({}).values(flow=True)[1:]
        np.testing.assert_allclose(stacked_total, mc_totals)

        colors = mc_call["kwargs"].get("color", [])
        assert len(colors) == len(mc_stack_inputs)
    finally:
        plt = make_cr_and_sr_plots.plt
        plt.close(fig)


def test_both_njets_preserves_variables_for_merged_output(tmp_path):
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    channel_axis = hist.axis.StrCategory([], name="channel", growth=True)
    syst_axis = hist.axis.StrCategory([], name="systematic", growth=True)
    njets_axis = hist.axis.Regular(1, 0.0, 1.0, name="njets")
    met_axis = hist.axis.Regular(1, 0.0, 1.0, name="met")

    h_njets = make_cr_and_sr_plots.SparseHist(
        process_axis, channel_axis, syst_axis, njets_axis
    )
    h_met = make_cr_and_sr_plots.SparseHist(
        process_axis, channel_axis, syst_axis, met_axis
    )

    for hist_obj in (h_njets, h_met):
        setattr(hist_obj, "_sumw2", defaultdict(lambda: None))

    for channel in ("2lss_ee_CR_1j", "2lss_em_CR_1j"):
        h_njets.fill(
            process="dataUL18",
            channel=channel,
            systematic="nominal",
            njets=0.5,
            weight=1.0,
        )
        h_njets.fill(
            process="ttH_centralUL18",
            channel=channel,
            systematic="nominal",
            njets=0.5,
            weight=2.0,
        )
        h_met.fill(
            process="dataUL18",
            channel=channel,
            systematic="nominal",
            met=0.5,
            weight=3.0,
        )
        h_met.fill(
            process="ttH_centralUL18",
            channel=channel,
            systematic="nominal",
            met=0.5,
            weight=4.0,
        )

    make_cr_and_sr_plots.run_plots_for_region(
        "CR",
        {"njets": h_njets, "met": h_met},
        years=["2018"],
        save_dir_path=str(tmp_path),
        channel_output="both-njets",
        skip_syst_errs=True,
        workers=1,
        verbose=False,
    )

    merged_dir = tmp_path / "cr_2lss_1j"
    assert merged_dir.exists()

    plot_names = sorted(path.name for path in merged_dir.glob("*.png"))
    assert {
        "cr_2lss_1j_met.png",
        "cr_2lss_1j_njets.png",
    }.issubset(set(plot_names)), plot_names
