import copy
import re
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


def _find_zero_yield_entry(summary, *, label, variable):
    for entry in summary.get("channel_entries", []):
        if entry.get("label") == label and entry.get("variable") == variable:
            return entry
    return None


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


def test_sr_aggregate_blinded_uses_mc_when_data_empty(tmp_path):
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

    plot_dir = tmp_path / "2lss_SR"
    assert plot_dir.exists()
    plot_paths = list(plot_dir.glob("*_njets.png"))
    assert plot_paths, "Expected SR aggregate plot when MC is non-zero and data is empty"


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
