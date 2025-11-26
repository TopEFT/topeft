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


def test_data_driven_samples_preserved_for_1tau_cr():
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    channel_axis = hist.axis.StrCategory([], name="channel", growth=True)
    syst_axis = hist.axis.StrCategory([], name="systematic", growth=True)
    value_axis = hist.axis.Regular(1, 0.0, 1.0, name="met")

    hist_obj = make_cr_and_sr_plots.tc_sparseHist.SparseHist(
        process_axis, channel_axis, syst_axis, value_axis
    )

    channel_name = "1l_e_1tau_CR_2j"
    sample_names = ["nonpromptUL16", "flipsUL16APV", "ttbarUL18", "dataUL18"]

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

    assert "nonpromptUL16" in region_ctx.mc_samples
    assert "flipsUL16APV" in region_ctx.mc_samples
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
