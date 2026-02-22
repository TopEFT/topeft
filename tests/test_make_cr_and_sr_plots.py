import copy
import re
import warnings
from types import SimpleNamespace

import hist
import numpy as np
import pytest
from collections import defaultdict
from matplotlib.transforms import Bbox

from analysis.topeft_run2 import make_cr_and_sr_plots


class _DummyHist:
    def __init__(self):
        self.scale_factors = []

    def eval(self, _):
        return {"sample": np.zeros(4)}

    def scale(self, factor):
        self.scale_factors.append(factor)


def _find_zero_yield_entry(summary, *, label, variable):
    for entry in summary.get("channel_entries", []):
        if entry.get("label") == label and entry.get("variable") == variable:
            return entry
    return None


def _make_met_histogram_for_channels(channel_names):
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    channel_axis = hist.axis.StrCategory([], name="channel", growth=True)
    syst_axis = hist.axis.StrCategory([], name="systematic", growth=True)
    met_axis = hist.axis.Regular(1, 0.0, 1.0, name="met")

    hist_obj = make_cr_and_sr_plots.tc_sparseHist.SparseHist(
        process_axis, channel_axis, syst_axis, met_axis
    )
    setattr(hist_obj, "_sumw2", defaultdict(lambda: None))

    for channel_name in channel_names:
        hist_obj.fill(
            process="ttH_central2022",
            channel=channel_name,
            systematic="nominal",
            met=0.5,
            weight=1.0,
        )
        hist_obj.fill(
            process="data2022",
            channel=channel_name,
            systematic="nominal",
            met=0.5,
            weight=1.0,
        )

    return hist_obj


def _make_simple_stacked_inputs():
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    value_axis = hist.axis.Regular(2, 0.0, 2.0, name="lj0pt")
    h_mc = hist.Hist(process_axis, value_axis, storage=hist.storage.Double())
    h_data = hist.Hist(process_axis, value_axis, storage=hist.storage.Double())

    for bin_idx, weight in enumerate((10.0, 5.0)):
        h_mc.fill(process="ttbarSL", lj0pt=[bin_idx + 0.25], weight=[weight])
    for bin_idx, weight in enumerate((8.0, 6.0)):
        h_data.fill(process="data", lj0pt=[bin_idx + 0.25], weight=[weight])

    group_map = {"Top": ["ttbarSL"]}
    return h_mc, h_data, group_map


def _make_multigroup_stacked_inputs(num_groups=8):
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    value_axis = hist.axis.Regular(2, 0.0, 2.0, name="lj0pt")
    h_mc = hist.Hist(process_axis, value_axis, storage=hist.storage.Double())
    h_data = hist.Hist(process_axis, value_axis, storage=hist.storage.Double())

    group_map = {}
    for proc_idx in range(num_groups):
        proc_name = f"mc_proc_{proc_idx}"
        group_map[f"Group {proc_idx}"] = [proc_name]
        for bin_idx, base_weight in enumerate((1.0, 2.0)):
            h_mc.fill(
                process=proc_name,
                lj0pt=[bin_idx + 0.25],
                weight=[base_weight + proc_idx],
            )

    for bin_idx in range(2):
        h_data.fill(process="data", lj0pt=[bin_idx + 0.25], weight=[1.0])

    return h_mc, h_data, group_map


def _get_cms_text_union_bbox(fig, ax, renderer):
    def _cms_matches(text_artist):
        text = text_artist.get_text() or ""
        return ("CMS" in text) or ("Simulation" in text)

    cms_texts = [text for text in fig.texts if _cms_matches(text)]
    if not cms_texts:
        cms_texts = [text for text in ax.texts if _cms_matches(text)]
    assert cms_texts, (
        "Could not find CMS label text in fig.texts or ax.texts; "
        "expected at least one text containing 'CMS' or 'Simulation'."
    )

    cms_bboxes = [
        text.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        for text in cms_texts
    ]
    return Bbox.union(cms_bboxes)


def test_blind_mode_does_not_draw_data_or_ratio_markers_and_omits_ratio_panel(monkeypatch):
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

    h_mc, h_data, group_map = _make_simple_stacked_inputs()
    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        unblind=False,
    )

    try:
        errorbar_calls = [
            call for call in plotted_calls if call["kwargs"].get("histtype") == "errorbar"
        ]
        assert not errorbar_calls
        assert len(fig.axes) == 1
    finally:
        make_cr_and_sr_plots.plt.close(fig)


def test_blind_mode_figure_legend_stays_above_axes():
    h_mc, h_data, group_map = _make_multigroup_stacked_inputs(num_groups=8)
    mc_totals = h_mc[{"process": sum}].values(flow=True)[1:]

    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        err_p_syst=mc_totals + 0.5,
        err_m_syst=np.clip(mc_totals - 0.5, a_min=0.0, a_max=None),
        syst_err="syst",
        unblind=False,
    )

    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        assert len(fig.axes) == 1
        assert len(fig.legends) == 1
        legend = fig.legends[0]
        renderer = fig.canvas.get_renderer()
        legend_box = legend.get_window_extent(renderer).transformed(
            fig.transFigure.inverted()
        )
        cms_box = _get_cms_text_union_bbox(fig, ax, renderer)
        ax_box = ax.get_position()
        overlaps = (
            cms_box.x0 < legend_box.x1
            and cms_box.x1 > legend_box.x0
            and cms_box.y0 < legend_box.y1
            and cms_box.y1 > legend_box.y0
        )

        assert legend_box.y0 >= ax_box.y1 - 1e-3
        assert not overlaps, (
            f"CMS label overlaps legend in blind mode: cms={cms_box.bounds}, "
            f"legend={legend_box.bounds}"
        )
        assert ax.get_ylabel() == "Events"
        assert ax.yaxis.label.get_visible()
        assert not any((text.get_text() or "") == "Events" for text in fig.texts)
        assert ax.get_legend() is None or ax.get_legend().get_visible() is False
    finally:
        make_cr_and_sr_plots.plt.close(fig)


def test_blind_and_unblind_share_events_axis_ylabel_geometry():
    h_mc, h_data, group_map = _make_multigroup_stacked_inputs(num_groups=8)
    mc_totals = h_mc[{"process": sum}].values(flow=True)[1:]

    blind_fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        err_p_syst=mc_totals + 0.5,
        err_m_syst=np.clip(mc_totals - 0.5, a_min=0.0, a_max=None),
        syst_err="syst",
        unblind=False,
    )
    unblind_fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        err_p_syst=mc_totals + 0.5,
        err_m_syst=np.clip(mc_totals - 0.5, a_min=0.0, a_max=None),
        syst_err="syst",
        unblind=True,
    )

    try:
        blind_fig.canvas.draw()
        unblind_fig.canvas.draw()

        blind_ax = blind_fig.axes[0]
        unblind_ax = unblind_fig.axes[0]
        assert blind_ax.get_ylabel() == "Events"
        assert unblind_ax.get_ylabel() == "Events"

        blind_renderer = blind_fig.canvas.get_renderer()
        unblind_renderer = unblind_fig.canvas.get_renderer()

        blind_box = blind_ax.yaxis.label.get_window_extent(blind_renderer).transformed(
            blind_fig.transFigure.inverted()
        )
        unblind_box = unblind_ax.yaxis.label.get_window_extent(unblind_renderer).transformed(
            unblind_fig.transFigure.inverted()
        )

        assert abs(blind_box.x0 - unblind_box.x0) <= 1e-3
        assert abs(blind_box.x1 - unblind_box.x1) <= 1e-3
    finally:
        make_cr_and_sr_plots.plt.close(blind_fig)
        make_cr_and_sr_plots.plt.close(unblind_fig)


def test_unblind_mode_still_draws_data_and_ratio_panels(monkeypatch):
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

    h_mc, h_data, group_map = _make_simple_stacked_inputs()
    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        unblind=True,
    )

    try:
        assert len(fig.axes) == 2
        assert any(
            call["kwargs"].get("histtype") == "errorbar"
            and call["kwargs"].get("label") == "Data"
            for call in plotted_calls
        )
        ratio_ax = fig.axes[1]
        assert any(
            call["kwargs"].get("histtype") == "errorbar"
            and call["kwargs"].get("ax") is ratio_ax
            for call in plotted_calls
        )
    finally:
        make_cr_and_sr_plots.plt.close(fig)


def test_region_context_no_longer_exposes_use_mc_as_data_when_blinded():
    hist_inputs = {"met": _make_met_histogram_for_channels(["2lss_ee_CR_1j"])}
    region_ctx = make_cr_and_sr_plots.build_region_context(
        "CR", hist_inputs, years=["2022"], unblind=True
    )

    assert not hasattr(region_ctx, "use_mc_as_data_when_blinded")


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


def test_unmatched_sample_is_skipped_from_group_map(monkeypatch):
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
    assert "mysteryProcess" not in group_map
    assert any(
        "mysteryProcess" in msg and "skipping" in msg.lower() for msg in captured
    )

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
        mc_totals = (
            h_mc[{"process": "ttbarSL"}].as_hist({}).values(flow=True)[1:]
        )
        np.testing.assert_allclose(stacked_total, mc_totals)

        colors = mc_call["kwargs"].get("color", [])
        assert len(colors) == len(mc_stack_inputs)
    finally:
        plt = make_cr_and_sr_plots.plt
        plt.close(fig)


def test_hist_is_empty_handles_missing_and_raising_empty():
    class _RaisingEmpty:
        def empty(self):
            raise RuntimeError("empty failed")

    class _EmptyReturnsFalse:
        def empty(self):
            return False

    class _NoEmpty:
        pass

    assert make_cr_and_sr_plots._hist_is_empty(None) is True
    assert make_cr_and_sr_plots._hist_is_empty(_RaisingEmpty()) is True
    assert make_cr_and_sr_plots._hist_is_empty(_EmptyReturnsFalse()) is False
    assert make_cr_and_sr_plots._hist_is_empty(_NoEmpty()) is False


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


def test_sr_njets_channel_transform_matches_cr_behavior():
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    channel_axis = hist.axis.StrCategory([], name="channel", growth=True)
    syst_axis = hist.axis.StrCategory([], name="systematic", growth=True)
    njets_axis = hist.axis.Regular(1, 0.0, 1.0, name="njets")

    hist_obj = make_cr_and_sr_plots.tc_sparseHist.SparseHist(
        process_axis, channel_axis, syst_axis, njets_axis
    )

    for channel in ("3l_m_offZ_1b", "3l_p_offZ_2b"):
        hist_obj.fill(
            process="ttH_central2022",
            channel=channel,
            systematic="nominal",
            njets=0.5,
            weight=1.0,
        )

    hist_inputs = {"njets": hist_obj}

    region_ctx = make_cr_and_sr_plots.build_region_context(
        "SR", hist_inputs, years=["2022"], unblind=True
    )
    payload = make_cr_and_sr_plots._prepare_variable_payload(
        "njets", region_ctx, metadata_only=True
    )

    assert "njets" in payload["channel_transformations"]

    channel_bins = payload["channel_dict"].get("3l_offZ_SR")
    assert channel_bins, "Expected SR channel bins for 3l_offZ_SR"

    make_cr_and_sr_plots.validate_channel_group(
        [hist_obj],
        channel_bins,
        payload["channel_transformations"],
        region=region_ctx.name,
        subgroup="3l_offZ_SR",
        variable="njets",
    )


def test_sr_zero_yield_summary_respects_njets_aggregation(monkeypatch):
    process_axis = hist.axis.StrCategory(
        ["ttH_central2022", "data2022"], name="process"
    )
    channel_axis = hist.axis.StrCategory(
        ["2lss_4t_m", "2lss_4t_p", "2lss_m", "2lss_p"], name="channel"
    )
    syst_axis = hist.axis.StrCategory(["nominal"], name="systematic")
    njets_axis = hist.axis.Regular(1, 0.0, 1.0, name="njets")

    hist_obj = make_cr_and_sr_plots.tc_sparseHist.SparseHist(
        process_axis, channel_axis, syst_axis, njets_axis
    )

    hist_obj.fill(
        process="ttH_central2022",
        channel="2lss_4t_m",
        systematic="nominal",
        njets=0.5,
        weight=1.0,
    )
    hist_obj.fill(
        process="data2022",
        channel="2lss_4t_m",
        systematic="nominal",
        njets=0.5,
        weight=0.0,
    )
    for channel in ("2lss_4t_p", "2lss_m", "2lss_p"):
        hist_obj.fill(
            process="data2022",
            channel=channel,
            systematic="nominal",
            njets=0.5,
            weight=0.0,
        )

    hist_inputs = {"njets": hist_obj}

    patched_cfg = copy.deepcopy(
        make_cr_and_sr_plots.REGION_PLOTTING.get("SR", {})
    )
    patched_cfg.update({"skip_variables": ["ptz"]})

    with monkeypatch.context() as m:
        m.setitem(make_cr_and_sr_plots.REGION_PLOTTING, "SR", patched_cfg)
        region_ctx = make_cr_and_sr_plots.build_region_context(
            "SR", hist_inputs, years=["2022"], unblind=True
        )
        summary = make_cr_and_sr_plots._summarize_zero_yield_processes(
            hist_inputs,
            region_name="SR",
            region_ctx=region_ctx,
            variables=["njets"],
        )

    entry = _find_zero_yield_entry(summary, label="2lss_SR", variable="njets")
    assert entry is not None
    assert not entry["missing_bins"]


def test_sr_zero_yield_summary_flags_missing_channels_for_lj0pt():
    process_axis = hist.axis.StrCategory(
        ["ttH_central2022", "data2022"], name="process"
    )
    channel_axis = hist.axis.StrCategory(
        ["2lss_4t_m_4j", "2lss_4t_m_5j"], name="channel"
    )
    syst_axis = hist.axis.StrCategory(["nominal"], name="systematic")
    lj0pt_axis = hist.axis.Regular(1, 0.0, 1.0, name="lj0pt")

    hist_obj = make_cr_and_sr_plots.tc_sparseHist.SparseHist(
        process_axis, channel_axis, syst_axis, lj0pt_axis
    )

    hist_obj.fill(
        process="ttH_central2022",
        channel="2lss_4t_m_4j",
        systematic="nominal",
        lj0pt=0.5,
        weight=1.0,
    )
    hist_obj.fill(
        process="data2022",
        channel="2lss_4t_m_4j",
        systematic="nominal",
        lj0pt=0.5,
        weight=0.0,
    )
    hist_obj.fill(
        process="data2022",
        channel="2lss_4t_m_5j",
        systematic="nominal",
        lj0pt=0.5,
        weight=0.0,
    )

    hist_inputs = {"lj0pt": hist_obj}
    region_ctx = make_cr_and_sr_plots.build_region_context(
        "SR", hist_inputs, years=["2022"], unblind=True
    )

    summary = make_cr_and_sr_plots._summarize_zero_yield_processes(
        hist_inputs,
        region_name="SR",
        region_ctx=region_ctx,
        variables=["lj0pt"],
    )

    entry = _find_zero_yield_entry(summary, label="2lss_SR", variable="lj0pt")
    assert entry is not None
    assert "2lss_4t_m_6j" in entry["missing_bins"]
    assert "2lss_4t_m_5j" not in entry["missing_bins"]

    zero_processes = {proc for proc, _ in entry["zero_processes"]}
    assert "data2022" in zero_processes


def test_sr_zero_yield_summary_respects_skip_variables():
    process_axis = hist.axis.StrCategory(["ttH_central2022"], name="process")
    channel_axis = hist.axis.StrCategory(["3l_onZ_1b_2j"], name="channel")
    syst_axis = hist.axis.StrCategory(["nominal"], name="systematic")
    ptz_axis = hist.axis.Regular(1, 0.0, 1.0, name="ptz")

    hist_obj = make_cr_and_sr_plots.tc_sparseHist.SparseHist(
        process_axis, channel_axis, syst_axis, ptz_axis
    )

    hist_obj.fill(
        process="ttH_central2022",
        channel="3l_onZ_1b_2j",
        systematic="nominal",
        ptz=0.5,
        weight=1.0,
    )

    hist_inputs = {"ptz": hist_obj}
    region_ctx = make_cr_and_sr_plots.build_region_context(
        "SR", hist_inputs, years=["2022"], unblind=True
    )

    summary = make_cr_and_sr_plots._summarize_zero_yield_processes(
        hist_inputs,
        region_name="SR",
        region_ctx=region_ctx,
        variables=["ptz"],
    )

    assert not summary["channel_entries"]


def test_cr_zero_yield_summary_respects_skip_variables(monkeypatch):
    process_axis = hist.axis.StrCategory(["ttH_central2022"], name="process")
    channel_axis = hist.axis.StrCategory(["2lss_ee_CR_1j"], name="channel")
    syst_axis = hist.axis.StrCategory(["nominal"], name="systematic")
    ptz_axis = hist.axis.Regular(1, 0.0, 1.0, name="ptz")

    hist_obj = make_cr_and_sr_plots.tc_sparseHist.SparseHist(
        process_axis, channel_axis, syst_axis, ptz_axis
    )

    hist_obj.fill(
        process="ttH_central2022",
        channel="2lss_ee_CR_1j",
        systematic="nominal",
        ptz=0.5,
        weight=1.0,
    )

    hist_inputs = {"ptz": hist_obj}

    patched_cfg = copy.deepcopy(
        make_cr_and_sr_plots.REGION_PLOTTING.get("CR", {})
    )
    patched_cfg.update({"skip_variables": ["ptz"]})

    with monkeypatch.context() as m:
        m.setitem(make_cr_and_sr_plots.REGION_PLOTTING, "CR", patched_cfg)
        region_ctx = make_cr_and_sr_plots.build_region_context(
            "CR", hist_inputs, years=["2022"], unblind=True
        )
        summary = make_cr_and_sr_plots._summarize_zero_yield_processes(
            hist_inputs,
            region_name="CR",
            region_ctx=region_ctx,
            variables=["ptz"],
        )

    assert not summary["channel_entries"]


def test_cr_zero_yield_summary_reports_variable_and_missing_bins():
    process_axis = hist.axis.StrCategory(
        ["ttH_central2022", "data2022"], name="process"
    )
    channel_axis = hist.axis.StrCategory(
        ["2lss_ee_CR_1j", "2lss_mm_CR_1j"], name="channel"
    )
    syst_axis = hist.axis.StrCategory(["nominal"], name="systematic")
    met_axis = hist.axis.Regular(1, 0.0, 1.0, name="met")

    hist_obj = make_cr_and_sr_plots.tc_sparseHist.SparseHist(
        process_axis, channel_axis, syst_axis, met_axis
    )

    for channel in ("2lss_ee_CR_1j", "2lss_mm_CR_1j"):
        hist_obj.fill(
            process="ttH_central2022",
            channel=channel,
            systematic="nominal",
            met=0.5,
            weight=1.0,
        )
        hist_obj.fill(
            process="data2022",
            channel=channel,
            systematic="nominal",
            met=0.5,
            weight=0.0,
        )

    hist_inputs = {"met": hist_obj}
    region_ctx = make_cr_and_sr_plots.build_region_context(
        "CR", hist_inputs, years=["2022"], unblind=True
    )

    summary = make_cr_and_sr_plots._summarize_zero_yield_processes(
        hist_inputs,
        region_name="CR",
        region_ctx=region_ctx,
        variables=["met"],
    )

    entry = _find_zero_yield_entry(summary, label="cr_2lss", variable="met")
    assert entry is not None
    assert "2lss_mm_CR_2j" in entry["missing_bins"]

    zero_processes = {proc for proc, _ in entry["zero_processes"]}
    assert "data2022" in zero_processes


def test_sr_zero_yield_skips_unmatched_processes(monkeypatch):
    process_axis = hist.axis.StrCategory(
        ["ttH_central2022", "ZG_MLL-50_PTG-600_central2022"], name="process"
    )
    channel_axis = hist.axis.StrCategory(["2lss_4t_m_4j"], name="channel")
    syst_axis = hist.axis.StrCategory(["nominal"], name="systematic")
    lj0pt_axis = hist.axis.Regular(1, 0.0, 1.0, name="lj0pt")

    hist_obj = make_cr_and_sr_plots.tc_sparseHist.SparseHist(
        process_axis, channel_axis, syst_axis, lj0pt_axis
    )

    hist_obj.fill(
        process="ttH_central2022",
        channel="2lss_4t_m_4j",
        systematic="nominal",
        lj0pt=0.5,
        weight=1.0,
    )
    hist_obj.fill(
        process="ZG_MLL-50_PTG-600_central2022",
        channel="2lss_4t_m_4j",
        systematic="nominal",
        lj0pt=0.5,
        weight=1.0,
    )

    hist_inputs = {"lj0pt": hist_obj}

    with monkeypatch.context() as m:
        warnings = []

        def _capture_warning(msg, *args, **kwargs):
            if args:
                msg = msg % args
            warnings.append(msg)

        m.setattr(make_cr_and_sr_plots.logger, "warning", _capture_warning)
        region_ctx = make_cr_and_sr_plots.build_region_context(
            "SR", hist_inputs, years=["2022"], unblind=True
        )
        summary = make_cr_and_sr_plots._summarize_zero_yield_processes(
            hist_inputs,
            region_name="SR",
            region_ctx=region_ctx,
            variables=["lj0pt"],
        )

    assert any("ZG_MLL-50_PTG-600_central2022" in msg for msg in warnings)
    unmatched_present = any(
        proc == "ZG_MLL-50_PTG-600_central2022"
        for entry in summary["channel_entries"]
        for proc, _ in entry["zero_processes"]
    )
    assert not unmatched_present


def test_cr_zero_yield_skips_unmatched_processes(monkeypatch):
    process_axis = hist.axis.StrCategory(
        ["TTTo2L2Nu_central2022", "mysteryProc2022"], name="process"
    )
    channel_axis = hist.axis.StrCategory(["2lss_ee_CR_1j"], name="channel")
    syst_axis = hist.axis.StrCategory(["nominal"], name="systematic")
    met_axis = hist.axis.Regular(1, 0.0, 1.0, name="met")

    hist_obj = make_cr_and_sr_plots.tc_sparseHist.SparseHist(
        process_axis, channel_axis, syst_axis, met_axis
    )

    hist_obj.fill(
        process="TTTo2L2Nu_central2022",
        channel="2lss_ee_CR_1j",
        systematic="nominal",
        met=0.5,
        weight=1.0,
    )
    hist_obj.fill(
        process="mysteryProc2022",
        channel="2lss_ee_CR_1j",
        systematic="nominal",
        met=0.5,
        weight=1.0,
    )

    hist_inputs = {"met": hist_obj}

    with monkeypatch.context() as m:
        warnings = []

        def _capture_warning(msg, *args, **kwargs):
            if args:
                msg = msg % args
            warnings.append(msg)

        m.setattr(make_cr_and_sr_plots.logger, "warning", _capture_warning)
        region_ctx = make_cr_and_sr_plots.build_region_context(
            "CR", hist_inputs, years=["2022"], unblind=True
        )
        summary = make_cr_and_sr_plots._summarize_zero_yield_processes(
            hist_inputs,
            region_name="CR",
            region_ctx=region_ctx,
            variables=["met"],
        )

    assert any("mysteryProc2022" in msg for msg in warnings)
    unmatched_present = any(
        proc == "mysteryProc2022"
        for entry in summary["channel_entries"]
        for proc, _ in entry["zero_processes"]
    )
    assert not unmatched_present


def test_sr_aggregate_blinded_renders_when_data_empty(tmp_path):
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    channel_axis = hist.axis.StrCategory([], name="channel", growth=True)
    syst_axis = hist.axis.StrCategory([], name="systematic", growth=True)
    njets_axis = hist.axis.Regular(1, 0.0, 1.0, name="njets")

    hist_obj = make_cr_and_sr_plots.tc_sparseHist.SparseHist(
        process_axis, channel_axis, syst_axis, njets_axis
    )

    hist_obj.fill(
        process="ttH_central2022",
        channel="2lss_p",
        systematic="nominal",
        njets=0.5,
        weight=5.0,
    )
    hist_obj.fill(
        process="data2022",
        channel="2lss_p",
        systematic="nominal",
        njets=0.5,
        weight=0.0,
    )

    make_cr_and_sr_plots.run_plots_for_region(
        "SR",
        {"njets": hist_obj},
        years=["2022"],
        save_dir_path=str(tmp_path),
        channel_output="merged",
        skip_syst_errs=True,
        workers=1,
        verbose=False,
        unblind=False,
    )

    plot_paths = list(tmp_path.rglob("*_njets.png"))
    assert plot_paths, "Expected SR blinded plot when MC is non-zero and data is empty"


def test_data_driven_samples_preserved_for_1tau_cr():
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    channel_axis = hist.axis.StrCategory([], name="channel", growth=True)
    syst_axis = hist.axis.StrCategory([], name="systematic", growth=True)
    value_axis = hist.axis.Regular(1, 0.0, 1.0, name="met")

    hist_obj = make_cr_and_sr_plots.tc_sparseHist.SparseHist(
        process_axis, channel_axis, syst_axis, value_axis
    )

    channel_name = "1l_e_1tau_CR_2j"
    sample_names = ["nonpromptUL18", "flipsUL18", "ttbarUL18", "dataUL18"]

    for sample in sample_names:
        hist_obj.fill(
            process=sample,
            channel=channel_name,
            systematic="nominal",
            met=0.5,
            weight=1.0,
        )

    region_ctx = make_cr_and_sr_plots.build_region_context(
        "CR", {"met": hist_obj}, years=["2018"], unblind=True
    )

    assert "nonpromptUL18" in region_ctx.mc_samples
    assert "flipsUL18" in region_ctx.mc_samples
    assert any(
        sample.startswith("nonprompt")
        for sample in region_ctx.group_map.get("Nonprompt", [])
    )
    assert any(
        sample.startswith("flips") for sample in region_ctx.group_map.get("Flips", [])
    )


def test_both_includes_split_channels_when_available(tmp_path):
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    channel_axis = hist.axis.StrCategory([], name="channel", growth=True)
    syst_axis = hist.axis.StrCategory([], name="systematic", growth=True)
    met_axis = hist.axis.Regular(1, 0.0, 1.0, name="met")

    h_met = make_cr_and_sr_plots.SparseHist(
        process_axis, channel_axis, syst_axis, met_axis
    )

    setattr(h_met, "_sumw2", defaultdict(lambda: None))

    for channel, weight in ("2los_ee_CRZ_0j", 1.0), ("2los_mm_CRZ_0j", 2.0):
        h_met.fill(
            process="dataUL18",
            channel=channel,
            systematic="nominal",
            met=0.25,
            weight=weight,
        )
        h_met.fill(
            process="ttH_centralUL18",
            channel=channel,
            systematic="nominal",
            met=0.75,
            weight=weight,
        )

    make_cr_and_sr_plots.run_plots_for_region(
        "CR",
        {"met": h_met},
        years=["2018"],
        save_dir_path=str(tmp_path),
        channel_output="both",
        skip_syst_errs=True,
        workers=1,
        verbose=False,
    )

    merged_dir = tmp_path / "cr_2los_Z"
    assert merged_dir.exists()

    split_dirs = [
        tmp_path / "cr_2los_Z_ee",
        tmp_path / "cr_2los_Z_mm",
    ]
    for split_dir in split_dirs:
        assert split_dir.exists()


def test_split_warning_uses_cr_reference_bins_for_cr_region(monkeypatch, tmp_path):
    class _AxisOnlyHist:
        def __init__(self, channel_labels):
            self.axes = {"channel": channel_labels}

    warning_messages = []

    def _capture_warning(msg, *args, **kwargs):
        if args:
            msg = msg % args
        warning_messages.append(msg)

    hist_inputs = {"met": _AxisOnlyHist(["2lss_mm_CR_1j"])}

    with monkeypatch.context() as m:
        m.setattr(make_cr_and_sr_plots._logger, "warning", _capture_warning)
        m.setattr(
            make_cr_and_sr_plots,
            "CR_CHAN_DICT",
            {"cr_ref": ["cr_only_mm_2j"]},
        )
        m.setattr(
            make_cr_and_sr_plots,
            "SR_CHAN_DICT",
            {"sr_ref": ["sr_only_ee_2j"]},
        )
        m.setattr(
            make_cr_and_sr_plots.yt,
            "is_split_by_lepflav",
            lambda *args, **kwargs: False,
        )
        m.setattr(
            make_cr_and_sr_plots.yt,
            "restore_split_channel_labels",
            lambda *args, **kwargs: False,
        )
        m.setattr(
            make_cr_and_sr_plots,
            "build_region_context",
            lambda *args, **kwargs: SimpleNamespace(
                name="CR", channel_mode="per-channel"
            ),
        )
        m.setattr(
            make_cr_and_sr_plots,
            "_summarize_zero_yield_processes",
            lambda *args, **kwargs: {
                "region": "CR",
                "channels_scanned": 0,
                "channel_entries": [],
                "zero_process_total": 0,
                "data_driven_zero_total": 0,
                "missing_data_driven_prefixes": set(),
                "errors": [],
            },
        )
        m.setattr(
            make_cr_and_sr_plots,
            "_emit_zero_yield_summary",
            lambda *args, **kwargs: None,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            make_cr_and_sr_plots.run_plots_for_region(
                "CR",
                hist_inputs,
                years=["2022"],
                save_dir_path=str(tmp_path),
                channel_output="split-njets",
                skip_syst_errs=True,
                workers=1,
                verbose=False,
            )

    assert warning_messages
    warning_text = warning_messages[-1]
    assert "region=CR" in warning_text
    assert "Expected flavour-split bins (from configuration): 1 total; showing first 1:" in warning_text
    assert "cr_only_mm_2j" in warning_text
    assert "sr_only_ee_2j" not in warning_text


@pytest.mark.parametrize(
    "region_name,channel_dict_attr,base_key,channel_bins",
    [
        (
            "CR",
            "CR_CHAN_DICT",
            "cr_dy_tautau_m",
            (
                "1l_m_dy_tautau_CR_2j",
                "1l_m_dy_tautau_CR_3j",
                "1l_m_dy_tautau_CR_4j",
            ),
        ),
        (
            "SR",
            "SR_CHAN_DICT",
            "2lss_m",
            (
                "2lss_m_4j",
                "2lss_m_5j",
                "2lss_m_6j",
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "channel_output,expected_modes",
    [
        ("merged-njets", ("aggregate",)),
        ("split-njets", ("per-channel",)),
        ("both-njets", ("aggregate", "per-channel")),
    ],
)
def test_njets_channel_output_modes_accept_base_channel_keys(
    tmp_path,
    monkeypatch,
    region_name,
    channel_dict_attr,
    base_key,
    channel_bins,
    channel_output,
    expected_modes,
):
    hist_inputs = {"met": _make_met_histogram_for_channels(channel_bins)}
    captured_channel_maps = {}
    base_channel_map = {base_key: list(channel_bins)}

    assert list(base_channel_map.keys()) == [base_key]

    def _capture_payload(region_ctx, *args, **kwargs):
        payload = make_cr_and_sr_plots._prepare_variable_payload(
            "met", region_ctx, metadata_only=True
        )
        captured_channel_maps[region_ctx.channel_mode] = payload["channel_dict"]

    minimal_summary = {
        "region": region_name,
        "channels_scanned": 0,
        "channel_entries": [],
        "zero_process_total": 0,
        "data_driven_zero_total": 0,
        "missing_data_driven_prefixes": set(),
        "errors": [],
    }

    with monkeypatch.context() as m:
        if channel_dict_attr == "CR_CHAN_DICT":
            m.setattr(make_cr_and_sr_plots, "CR_CHAN_DICT", base_channel_map)
            m.setattr(
                make_cr_and_sr_plots,
                "CHANNEL_REFERENCE_MAP",
                {**base_channel_map, **make_cr_and_sr_plots.SR_CHAN_DICT},
            )
        else:
            m.setattr(make_cr_and_sr_plots, "SR_CHAN_DICT", base_channel_map)
            m.setattr(
                make_cr_and_sr_plots,
                "CHANNEL_REFERENCE_MAP",
                {**make_cr_and_sr_plots.CR_CHAN_DICT, **base_channel_map},
            )

        m.setattr(make_cr_and_sr_plots, "produce_region_plots", _capture_payload)
        m.setattr(
            make_cr_and_sr_plots,
            "_summarize_zero_yield_processes",
            lambda *args, **kwargs: minimal_summary,
        )
        m.setattr(
            make_cr_and_sr_plots, "_emit_zero_yield_summary", lambda *args, **kwargs: None
        )

        make_cr_and_sr_plots.run_plots_for_region(
            region_name,
            hist_inputs,
            years=["2022"],
            save_dir_path=str(tmp_path),
            channel_output=channel_output,
            skip_syst_errs=True,
            workers=1,
            verbose=False,
        )

    assert set(captured_channel_maps.keys()) == set(expected_modes)

    expected_keys = {}
    for bin_name in channel_bins:
        suffix = make_cr_and_sr_plots._extract_njet_suffix(bin_name)
        assert suffix is not None
        expected_keys[f"{base_key}_{suffix}"] = [bin_name]

    for mode_name in expected_modes:
        transformed_map = captured_channel_maps[mode_name]
        assert base_key not in transformed_map
        for expected_key, expected_bins in expected_keys.items():
            assert expected_key in transformed_map
            assert transformed_map[expected_key] == expected_bins


@pytest.mark.parametrize("channel_output", ["both", "both-njets"])
def test_all_variables_render_for_merged_and_split_categories(
    tmp_path, channel_output, monkeypatch
):
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    channel_axis = hist.axis.StrCategory([], name="channel", growth=True)
    syst_axis = hist.axis.StrCategory([], name="systematic", growth=True)
    j0pt_axis = hist.axis.Regular(1, 0.0, 1.0, name="j0pt")
    met_axis = hist.axis.Regular(1, 0.0, 1.0, name="met")

    h_j0pt = make_cr_and_sr_plots.SparseHist(
        process_axis, channel_axis, syst_axis, j0pt_axis
    )
    h_met = make_cr_and_sr_plots.SparseHist(
        process_axis, channel_axis, syst_axis, met_axis
    )

    for hist_obj in (h_j0pt, h_met):
        setattr(hist_obj, "_sumw2", defaultdict(lambda: None))

    for channel, weight in ("2los_ee_CRZ_0j", 1.0), ("2los_mm_CRZ_0j", 2.0):
        h_j0pt.fill(
            process="dataUL18",
            channel=channel,
            systematic="nominal",
            j0pt=0.5,
            weight=weight,
        )
        h_j0pt.fill(
            process="ttH_centralUL18",
            channel=channel,
            systematic="nominal",
            j0pt=0.5,
            weight=weight,
        )
        h_met.fill(
            process="dataUL18",
            channel=channel,
            systematic="nominal",
            met=0.5,
            weight=weight,
        )
        h_met.fill(
            process="ttH_centralUL18",
            channel=channel,
            systematic="nominal",
            met=0.5,
            weight=weight,
        )

    with monkeypatch.context() as m:
        patched_cfg = copy.deepcopy(
            make_cr_and_sr_plots.REGION_PLOTTING.get("CR", {})
        )
        patched_cfg.update(
            {
                "skip_variables": ["met"],
                "category_skips": [
                    {
                        "categories": {"contains": ["mm"]},
                        "variable_includes": ["j0pt", "met"],
                    }
                ],
                "skip_sparse_2d": True,
            }
        )
        m.setitem(make_cr_and_sr_plots.REGION_PLOTTING, "CR", patched_cfg)

        make_cr_and_sr_plots.run_plots_for_region(
            "CR",
            {"j0pt": h_j0pt, "met": h_met},
            years=["2018"],
            save_dir_path=str(tmp_path),
            channel_output=channel_output,
            skip_syst_errs=True,
            workers=1,
            verbose=False,
        )

    merged_dir_name = "cr_2los_Z_0j" if channel_output.endswith("njets") else "cr_2los_Z"
    merged_dir = tmp_path / merged_dir_name
    assert merged_dir.exists()

    merged_plots = {path.name for path in merged_dir.glob("*.png")}
    expected_merged = {f"{merged_dir_name}_j0pt.png", f"{merged_dir_name}_met.png"}
    assert expected_merged.issubset(merged_plots)

    if channel_output.endswith("njets"):
        split_dirs = [
            tmp_path / "cr_2los_Z_ee_0j_ee",
            tmp_path / "cr_2los_Z_mm_0j_mm",
        ]
    else:
        split_dirs = [tmp_path / "cr_2los_Z_ee", tmp_path / "cr_2los_Z_mm"]
    for split_dir in split_dirs:
        assert split_dir.exists()
        split_plots = {path.name for path in split_dir.glob("*.png")}
        base_split_name = (
            split_dir.name
            if channel_output.endswith("njets")
            else re.sub(r"_0j(?=_)", "", split_dir.name)
        )
        expected_plots = {
            f"{base_split_name}_j0pt.png",
            f"{base_split_name}_met.png",
        }
        assert expected_plots.issubset(split_plots)


def test_data_driven_reinsertion_respects_year_tokens():
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    channel_axis = hist.axis.StrCategory([], name="channel", growth=True)
    syst_axis = hist.axis.StrCategory([], name="systematic", growth=True)
    value_axis = hist.axis.Regular(1, 0.0, 1.0, name="met")

    hist_obj = make_cr_and_sr_plots.tc_sparseHist.SparseHist(
        process_axis, channel_axis, syst_axis, value_axis
    )

    channel_name = "2lss_em_CR_1j"
    sample_names = [
        "ttbar2022",
        "data2022",
        "ttbarUL16",
        "dataUL16",
        "nonprompt2022",
        "nonprompt2022EE",
        "nonprompt2023",
        "nonprompt2023BPix",
        "flips2022",
        "flips2022EE",
        "flips2023",
        "flips2023BPix",
        "nonpromptUL16",
        "nonpromptUL17",
        "flipsUL16",
        "flipsUL17",
    ]

    for sample in sample_names:
        hist_obj.fill(
            process=sample,
            channel=channel_name,
            systematic="nominal",
            met=0.5,
            weight=1.0,
        )

    hist_inputs = {"met": hist_obj}

    ctx_2022 = make_cr_and_sr_plots.build_region_context(
        "CR", hist_inputs, years=["2022"], unblind=True
    )
    assert "nonprompt2022" in ctx_2022.mc_samples
    assert "flips2022" in ctx_2022.mc_samples
    assert "nonprompt2023" not in ctx_2022.mc_samples
    assert "flips2023BPix" not in ctx_2022.mc_samples
    assert "nonpromptUL16" not in ctx_2022.mc_samples

    ctx_pair = make_cr_and_sr_plots.build_region_context(
        "CR", hist_inputs, years=["2022", "2022EE"], unblind=True
    )
    assert "nonprompt2022" in ctx_pair.mc_samples
    assert "nonprompt2022EE" in ctx_pair.mc_samples
    assert "flips2022EE" in ctx_pair.mc_samples
    assert "nonprompt2023" not in ctx_pair.mc_samples

    ctx_run3 = make_cr_and_sr_plots.build_region_context(
        "CR", hist_inputs, years=["run3"], unblind=True
    )
    for label in [
        "nonprompt2022",
        "nonprompt2022EE",
        "nonprompt2023",
        "nonprompt2023BPix",
    ]:
        assert label in ctx_run3.mc_samples

    ctx_run2 = make_cr_and_sr_plots.build_region_context(
        "CR", hist_inputs, years=["run2"], unblind=True
    )
    for label in ["nonpromptUL16", "nonpromptUL17", "flipsUL16", "flipsUL17"]:
        assert label in ctx_run2.mc_samples
    assert "nonprompt2022" not in ctx_run2.mc_samples
    assert "flips2022" not in ctx_run2.mc_samples
