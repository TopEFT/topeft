import numpy as np
import hist
import pytest

from collections import OrderedDict
from pathlib import Path

from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist

from analysis.topeft_run2 import make_cr_and_sr_plots as plots


_PROCESS_NAME = "background"


def _build_histogram(variable_name, channel_bins, *, hist_type="HistEFT"):
    process_axis = hist.axis.StrCategory([_PROCESS_NAME], name="process", growth=True)
    channel_axis = hist.axis.StrCategory(channel_bins, name="channel", growth=True)
    systematic_axis = hist.axis.StrCategory(["nominal"], name="systematic", growth=True)
    dense_axis = hist.axis.Regular(1, 0.0, 1.0, name=variable_name)
    axes = (process_axis, channel_axis, systematic_axis, dense_axis)

    if hist_type == "SparseHist":
        histogram = SparseHist(*axes)
    else:
        histogram = HistEFT(*axes, wc_names=[], label="Events")

    for channel in channel_bins:
        histogram.fill(
            process=_PROCESS_NAME,
            channel=channel,
            systematic="nominal",
            **{variable_name: np.array([0.5], dtype=float)},
        )

    return histogram


def _build_sumw2_histogram(variable_name, channel_bins):
    process_axis = hist.axis.StrCategory([_PROCESS_NAME], name="process", growth=True)
    channel_axis = hist.axis.StrCategory(channel_bins, name="channel", growth=True)
    systematic_axis = hist.axis.StrCategory(["nominal"], name="systematic", growth=True)
    dense_axis = hist.axis.Regular(1, 0.0, 1.0, name=f"{variable_name}_sumw2")

    histogram = HistEFT(
        process_axis,
        channel_axis,
        systematic_axis,
        dense_axis,
        wc_names=[],
        label="Events",
    )

    for channel in channel_bins:
        histogram.fill(
            process=_PROCESS_NAME,
            channel=channel,
            systematic="nominal",
            **{f"{variable_name}_sumw2": np.array([0.5], dtype=float)},
        )

    return histogram


def _default_channel_rules():
    return {
        "default": [],
        "variables": {},
        "conditional": [
            {"when": "not_split_by_lepflav", "apply": ["lepflav"]},
        ],
    }


def _make_region_context(dict_of_hists, *, channel_map, channel_mode):
    sumw2_suffix = "_sumw2"
    sumw2_hists = {}
    for hist_name, hist_obj in dict_of_hists.items():
        if hist_name.endswith(sumw2_suffix) and hist_name.count("sumw2") == 1:
            base_name = hist_name[: -len(sumw2_suffix)]
            sumw2_hists[base_name] = hist_obj

    return plots.RegionContext(
        name="CR",
        dict_of_hists=dict_of_hists,
        years=("2022",),
        channel_map=channel_map,
        group_patterns={},
        group_map={"Background": [_PROCESS_NAME]},
        all_samples=(_PROCESS_NAME,),
        mc_samples=(_PROCESS_NAME,),
        data_samples=(),
        samples_to_remove={"mc": [], "data": []},
        sumw2_hists=sumw2_hists,
        signal_samples=(),
        unblind_default=True,
        lumi_pair=("1", "13TeV"),
        skip_variables=None,
        analysis_bins=None,
        stacked_ratio_style=None,
        channel_rules=_default_channel_rules(),
        sample_removal_rules=[],
        category_skip_rules=[],
        skip_sparse_2d=False,
        channel_mode=channel_mode,
        variable_label="Observable",
        debug_channel_lists=False,
        sumw2_remove_signal=False,
        sumw2_remove_signal_when_blinded=False,
        use_mc_as_data_when_blinded=False,
        rate_syst_by_sample=None,
    )


def _test_channel_map():
    return OrderedDict(
        [
            ("cr_all", ["category_em", "category_mm"]),
            ("cr_all_em", ["category_em"]),
            ("cr_all_mm", ["category_mm"]),
        ]
    )


def test_unsplit_channel_output_prunes_flavour_categories():
    variable = "observable"
    histograms = {
        variable: _build_histogram(variable, ["category"], hist_type="HistEFT"),
        f"{variable}_sumw2": _build_sumw2_histogram(variable, ["category"]),
    }
    region_ctx = _make_region_context(
        histograms,
        channel_map=_test_channel_map(),
        channel_mode="aggregate",
    )

    assert region_ctx.channels_split_by_lepflav is False

    payload = plots._prepare_variable_payload(variable, region_ctx)
    assert list(payload["channel_dict"].keys()) == ["cr_all"]
    assert payload["channel_dict"]["cr_all"] == ["category"]


def test_split_mode_skips_when_hist_not_flavour_split(monkeypatch, tmp_path):
    variable = "observable"
    histograms = {
        variable: _build_histogram(variable, ["category"], hist_type="HistEFT"),
        f"{variable}_sumw2": _build_sumw2_histogram(variable, ["category"]),
    }

    channel_map = _test_channel_map()

    def fake_build_region_context(region_name, dict_of_hists, years, *, unblind=None, channel_mode_override=None):
        channel_mode = channel_mode_override or "aggregate"
        return _make_region_context(
            dict_of_hists,
            channel_map=channel_map,
            channel_mode=channel_mode,
        )

    monkeypatch.setattr(plots, "build_region_context", fake_build_region_context)

    calls = []

    def fake_produce(region_ctx, *args, **kwargs):
        calls.append(region_ctx.channel_mode)

    monkeypatch.setattr(plots, "produce_region_plots", fake_produce)

    with pytest.warns(RuntimeWarning, match="Skipping split channel output"):
        plots.run_plots_for_region(
            "CR",
            histograms,
            years=("2022",),
            save_dir_path=str(tmp_path),
            variables=[variable],
            channel_output="split",
        )

    assert calls == []


def test_sumw2_histogram_passed_to_stacked_plot(monkeypatch, tmp_path):
    variable = "observable"
    histograms = {
        variable: _build_histogram(variable, ["category"], hist_type="HistEFT"),
        f"{variable}_sumw2": _build_sumw2_histogram(variable, ["category"]),
    }
    region_ctx = _make_region_context(
        histograms,
        channel_map=_test_channel_map(),
        channel_mode="aggregate",
    )

    payload = plots._prepare_variable_payload(variable, region_ctx)

    captured = {}

    def fake_make_region_stacked_ratio_fig(
        hist_mc_integrated,
        hist_data_to_plot,
        unit_norm_bool,
        *,
        var,
        **kwargs,
    ):
        captured["h_mc_sumw2"] = kwargs.get("h_mc_sumw2")

        class _Figure:
            def savefig(self, *args, **kwargs):
                pass

        return _Figure()

    monkeypatch.setattr(plots, "make_region_stacked_ratio_fig", fake_make_region_stacked_ratio_fig)

    channel_bins = payload["channel_dict"]["cr_all"]

    plots._render_variable_category(
        variable,
        "cr_all",
        channel_bins,
        region_ctx=region_ctx,
        channel_transformations=payload["channel_transformations"],
        hist_mc=payload["hist_mc"],
        hist_data=payload["hist_data"],
        hist_mc_sumw2_orig=payload["hist_mc_sumw2_orig"],
        is_sparse2d=payload["is_sparse2d"],
        save_dir_path=str(tmp_path),
        skip_syst_errs=True,
        unit_norm_bool=False,
        stacked_log_y=False,
        unblind_flag=True,
    )

    assert captured["h_mc_sumw2"] is not None


def test_flavour_split_hist_preserves_per_channel_entries():
    variable = "observable"
    histograms = {
        variable: _build_histogram(variable, ["category_em", "category_mm"], hist_type="SparseHist"),
    }
    region_ctx = _make_region_context(
        histograms,
        channel_map=_test_channel_map(),
        channel_mode="per-channel",
    )

    assert region_ctx.channels_split_by_lepflav is True

    payload = plots._prepare_variable_payload(variable, region_ctx)
    assert set(payload["channel_dict"].keys()) == {"cr_all_em", "cr_all_mm"}


def test_channel_output_both_runs_all_modes_and_uses_sumw2(monkeypatch, tmp_path):
    # Use an observable that only declares regular binning metadata.
    variable = "npvs"
    channel_bins = ["category_em", "category_mm"]
    histograms = {
        variable: _build_histogram(variable, channel_bins, hist_type="HistEFT"),
        f"{variable}_sumw2": _build_sumw2_histogram(variable, channel_bins),
    }

    invocations = []

    original_render = plots._render_variable_category

    def tracked_render(*args, **kwargs):
        region_ctx = kwargs["region_ctx"]
        invocations.append(
            (region_ctx.channel_mode, kwargs.get("hist_mc_sumw2_orig"))
        )
        return original_render(*args, **kwargs)

    monkeypatch.setattr(plots, "_render_variable_category", tracked_render)

    plots.run_plots_for_region(
        "CR",
        histograms,
        years=("2022",),
        save_dir_path=str(tmp_path),
        variables=[variable],
        channel_output="both",
        skip_syst_errs=True,
    )

    modes_seen = [mode for mode, _ in invocations]
    assert set(modes_seen) == {"aggregate", "per-channel"}
    assert all(payload is not None for _, payload in invocations)


def test_split_mode_groups_year_suffixed_channels(monkeypatch, tmp_path):
    variable = "observable"
    year_bins = ["category_em_2016", "category_em_2017"]
    histograms = {
        variable: _build_histogram(variable, year_bins, hist_type="HistEFT"),
        f"{variable}_sumw2": _build_sumw2_histogram(variable, year_bins),
    }

    channel_map = OrderedDict(
        [
            ("combined", list(year_bins)),
            ("category_em_2016", ["category_em_2016"]),
            ("category_em_2017", ["category_em_2017"]),
        ]
    )

    aggregate_ctx = _make_region_context(
        histograms,
        channel_map=channel_map,
        channel_mode="aggregate",
    )
    split_ctx = _make_region_context(
        histograms,
        channel_map=channel_map,
        channel_mode="per-channel",
    )

    payload = plots._prepare_variable_payload(
        variable,
        split_ctx,
        unblind_flag=True,
    )
    assert list(payload["channel_dict"].keys()) == ["category_em"]
    assert payload["channel_dict"]["category_em"] == year_bins

    rate_payload = (np.array([0.2]), np.array([0.3]))
    shape_payload = (np.array([0.1]), np.array([0.05]))

    monkeypatch.setattr(
        plots,
        "get_rate_syst_arrs",
        lambda *args, **kwargs: rate_payload,
    )
    monkeypatch.setattr(
        plots,
        "get_shape_syst_arrs",
        lambda *args, **kwargs: shape_payload,
    )
    monkeypatch.setattr(plots, "_close_figure_payload", lambda fig: None)

    render_calls = []
    current_mode = {"value": None}

    def fake_make_region_stacked_ratio_fig(
        hist_mc_integrated,
        hist_data_to_plot,
        unit_norm_bool,
        *,
        var,
        **kwargs,
    ):
        call = {"mode": current_mode["value"], "kwargs": kwargs, "paths": []}
        render_calls.append(call)

        class _Figure:
            def savefig(self, path, *args, **kwargs):
                call["paths"].append(path)

        return _Figure()

    monkeypatch.setattr(
        plots,
        "make_region_stacked_ratio_fig",
        fake_make_region_stacked_ratio_fig,
    )

    current_mode["value"] = "aggregate"
    plots.produce_region_plots(
        aggregate_ctx,
        str(tmp_path / "agg"),
        [variable],
        skip_syst_errs=False,
        unit_norm_bool=False,
        stacked_log_y=False,
        unblind=True,
        workers=1,
    )

    aggregate_calls = list(render_calls)

    current_mode["value"] = "per-channel"
    plots.produce_region_plots(
        split_ctx,
        str(tmp_path / "split"),
        [variable],
        skip_syst_errs=False,
        unit_norm_bool=False,
        stacked_log_y=False,
        unblind=True,
        workers=1,
    )

    per_channel_calls = render_calls[len(aggregate_calls) :]
    assert len(per_channel_calls) == 1

    per_call = per_channel_calls[0]
    assert per_call["paths"]
    for saved in per_call["paths"]:
        filename = Path(saved)
        assert "2016" not in filename.name
        assert "2017" not in filename.name
        assert filename.parent.name == "category_em"
        assert filename.name == f"category_em_{variable}.png"

    aggregate_kwargs = aggregate_calls[0]["kwargs"]
    per_kwargs = per_call["kwargs"]
    for key in ("err_p_syst", "err_m_syst", "err_ratio_p_syst", "err_ratio_m_syst"):
        assert key in aggregate_kwargs and key in per_kwargs
        assert np.allclose(aggregate_kwargs[key], per_kwargs[key])
    assert aggregate_kwargs.get("syst_err") == per_kwargs.get("syst_err") == "total"
