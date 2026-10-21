import copy
import concurrent.futures
import csv
import os
import re
import warnings
from types import SimpleNamespace

import hist
import numpy as np
import pytest
from collections import Counter, defaultdict
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.transforms import Bbox

from analysis.topeft_run2 import make_cr_and_sr_plots


def _mixed_sr_histogram(process, *, companion=False, auxiliary=False,
                        sr_weight=1.0, ar_weight=0.0, shape=False):
    from topcoffea.modules.sparseHist import SparseHist

    dense_name = "njets_sumw2" if companion else "njets"
    axes = [
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
    ]
    if auxiliary:
        axes.append(hist.axis.StrCategory([], name="appl", growth=True))
    axes.append(hist.axis.Regular(2, 0, 2, name=dense_name))
    output = SparseHist(*axes, storage="Double")
    for channel, sr_label, ar_label in (
        ("2lss_p", "isSR_2lSS", "isAR_2lSS"),
        ("3l_onZ_2b", "isSR_3l", "isAR_3l"),
    ):
        variations = [("nominal", sr_weight)]
        if shape:
            variations += [("shapeUp", sr_weight * 1.2),
                           ("shapeDown", sr_weight * 0.8)]
        for systematic, weight in variations:
            values = {
                "process": process, "channel": channel,
                "systematic": systematic, dense_name: np.asarray([0.5]),
                "weight": np.asarray([weight]),
            }
            if auxiliary:
                values["appl"] = sr_label
            output.fill(**values)
        if auxiliary and ar_weight:
            output.fill(
                process=process, channel=channel, systematic="nominal",
                appl=ar_label, **{dense_name: np.asarray([0.5])},
                weight=np.asarray([ar_weight]),
            )
    return output


def _mixed_sr_fixture(monkeypatch, *, auxiliary_process="DY50_centralUL18",
                      auxiliary_histogram=None):
    from topeft.modules.nominal_schema import scalar_nominal_key

    scalar_key = scalar_nominal_key("njets")
    primary = {
        scalar_key: _mixed_sr_histogram("dataUL18", sr_weight=7),
        "njets_sumw2": _mixed_sr_histogram("baselineUL18", companion=True,
                                           sr_weight=16),
    }
    primary[scalar_key] += _mixed_sr_histogram(
        "baselineUL18", sr_weight=4, shape=True
    )
    auxiliary = {
        scalar_key: auxiliary_histogram or _mixed_sr_histogram(
            auxiliary_process, auxiliary=True, sr_weight=2, ar_weight=30
        ),
        "njets_sumw2": _mixed_sr_histogram(
            auxiliary_process, auxiliary=True, companion=True,
            sr_weight=5, ar_weight=900,
        ),
    }
    sidecars = [
        {"artifact": {"artifact_kind": "nonprompt_output",
                      "pkl_sha256": "a" * 64},
         "sumw2_storage_provenance": {"resolved_processes": [
             "dataUL18", "baselineUL18"]}},
        {"artifact": {"artifact_kind": "processor_output",
                      "pkl_sha256": "b" * 64},
         "sumw2_storage_provenance": {"resolved_processes": [
             auxiliary_process]},
         "requested_data_driven_products": {"products": {
             "nonprompt": {"enabled": False}, "flips": {"enabled": False}}},
         "sumw2_content_manifest": {"families": {"njets": {
             "required_sumw2_processes": [auxiliary_process]}}}},
    ]
    def load_one(paths, **_kwargs):
        return (
            primary if paths == ["primary.pkl.gz"] else auxiliary,
            {"schema": "split_sibling_v1",
             "runtime_histogram_families": ["njets"]},
        )
    monkeypatch.setattr(make_cr_and_sr_plots, "load_and_merge_histogram_pkls",
                        load_one)
    return sidecars


def _mixed_sr_total(histogram, process, systematic="nominal"):
    selected = histogram[{"process": process, "systematic": systematic}]
    return float(np.sum(selected.values(flow=True)))


def test_mixed_stage_sr_view_projects_sr_and_preserves_data_sumw2_and_provenance(
    monkeypatch,
):
    sidecars = _mixed_sr_fixture(monkeypatch)
    grouping_before = dict(make_cr_and_sr_plots.SR_GRP_PATTERNS)
    view, report = make_cr_and_sr_plots._load_mixed_stage_sr_plot_view(
        ["primary.pkl.gz", "auxiliary.pkl.gz"], sidecars,
        selected_load={"selected_families": ["njets"]},
        year_coverage_policy="off",
    )
    assert _mixed_sr_total(view["njets__scalar_nominal"], "dataUL18") == 14
    assert _mixed_sr_total(view["njets__scalar_nominal"], "baselineUL18") == 8
    assert _mixed_sr_total(view["njets__scalar_nominal"], "DY50_centralUL18") == 4
    assert _mixed_sr_total(view["njets_sumw2"], "DY50_centralUL18") == 10
    assert "appl" not in view["njets_sumw2"].axes.name
    assert report["auxiliaries"][0]["sr_labels_by_component_and_channel"][
        "njets_sumw2"
    ] == {"2lss_p": "isSR_2lSS", "3l_onZ_2b": "isSR_3l"}
    assert report["primary"]["artifact_kind"] == "nonprompt_output"
    assert report["auxiliaries"][0]["artifact_kind"] == "processor_output"
    assert report["generic_sidecar_merge_bypassed"]
    assert report["cacheable"] is False and report["persistent_artifact"] is False
    assert make_cr_and_sr_plots.SR_GRP_PATTERNS == grouping_before
    shape_histo = view["njets__scalar_nominal"].prune("channel", ["2lss_p"])
    (_, shape_up), details = make_cr_and_sr_plots.get_shape_syst_arrs(
        shape_histo, group_type="SR", return_details=True
    )
    assert details["evaluated_bases"] == ("shape",)
    assert np.max(np.asarray(shape_up)) == pytest.approx(0.8**2)


@pytest.mark.parametrize("process, error", [
    ("baselineUL18", "process identity collision"),
    ("dataUL18", "Data or data-driven"),
])
def test_mixed_stage_rejects_collision_and_auxiliary_data(monkeypatch, process, error):
    sidecars = _mixed_sr_fixture(monkeypatch, auxiliary_process=process)
    with pytest.raises(ValueError, match=error):
        make_cr_and_sr_plots._load_mixed_stage_sr_plot_view(
            ["primary.pkl.gz", "auxiliary.pkl.gz"], sidecars,
            selected_load={"selected_families": ["njets"]},
            year_coverage_policy="off",
        )


def test_mixed_stage_rejects_missing_sr_appl(monkeypatch):
    auxiliary = _mixed_sr_histogram(
        "DY50_centralUL18", auxiliary=True, sr_weight=2, ar_weight=30
    )
    auxiliary = auxiliary.remove("appl", ["isSR_2lSS", "isSR_3l"])
    sidecars = _mixed_sr_fixture(monkeypatch, auxiliary_histogram=auxiliary)
    with pytest.raises(ValueError, match="missing source-defined SR appl"):
        make_cr_and_sr_plots._load_mixed_stage_sr_plot_view(
            ["primary.pkl.gz", "auxiliary.pkl.gz"], sidecars,
            selected_load={"selected_families": ["njets"]},
            year_coverage_policy="off",
        )


def test_mixed_stage_automatic_sr_dispatch_uses_single_input_validation(
    monkeypatch, tmp_path,
):
    sidecars = _mixed_sr_fixture(monkeypatch)
    monkeypatch.setattr(make_cr_and_sr_plots, "_mixed_stage_sidecars",
                        lambda _paths: sidecars)
    calls = []
    loader = make_cr_and_sr_plots.load_and_merge_histogram_pkls

    def record_loader(paths, **kwargs):
        calls.append(paths)
        return loader(paths, **kwargs)

    monkeypatch.setattr(make_cr_and_sr_plots, "load_and_merge_histogram_pkls",
                        record_loader)
    parser = make_cr_and_sr_plots.build_arg_parser()
    args = parser.parse_args([
        "-f", "primary.pkl.gz", "-f", "auxiliary.pkl.gz",
        "-o", str(tmp_path), "-n", "mixed", "--sr",
        "--variables", "njets", "--merge-only",
    ])
    assert make_cr_and_sr_plots.run_with_args(args, parser) == 0
    assert calls == [["primary.pkl.gz"], ["auxiliary.pkl.gz"]]


def test_mixed_stage_kind_classifier_requires_one_final_primary(
    monkeypatch, tmp_path,
):
    paths = [str(tmp_path / "primary.pkl.gz"),
             str(tmp_path / "auxiliary.pkl.gz")]
    for path in paths:
        make_cr_and_sr_plots.metadata_sidecar_path(path).touch()
    kinds = ["nonprompt_output", "processor_output"]
    monkeypatch.setattr(
        make_cr_and_sr_plots, "read_histogram_sidecar",
        lambda path: {"artifact": {"artifact_kind": kinds[paths.index(path)]}},
    )
    assert make_cr_and_sr_plots._mixed_stage_sidecars(paths) is not None
    kinds[0] = "flips_output"
    with pytest.raises(ValueError, match="exactly one nonprompt_output"):
        make_cr_and_sr_plots._mixed_stage_sidecars(paths)


def test_mixed_stage_rejects_cross_run_and_unproven_prompt_only(monkeypatch):
    sidecars = _mixed_sr_fixture(monkeypatch)
    sidecars[1]["sumw2_storage_provenance"]["resolved_processes"] = [
        "DY50_central2022"
    ]
    with pytest.raises(RuntimeError, match=r"Run 2 \+ Run 3"):
        make_cr_and_sr_plots._load_mixed_stage_sr_plot_view(
            ["primary.pkl.gz", "auxiliary.pkl.gz"], sidecars,
            selected_load={"selected_families": ["njets"]},
            year_coverage_policy="off",
        )
    sidecars[1]["sumw2_storage_provenance"]["resolved_processes"] = [
        "DY50_centralUL18"
    ]
    del sidecars[1]["requested_data_driven_products"]
    with pytest.raises(ValueError, match="prompt-only provenance is unproven"):
        make_cr_and_sr_plots._load_mixed_stage_sr_plot_view(
            ["primary.pkl.gz", "auxiliary.pkl.gz"], sidecars,
            selected_load={"selected_families": ["njets"]},
            year_coverage_policy="off",
        )


@pytest.mark.parametrize("extra_args, error", [
    (["--cr"], "only for the SR plot view"),
    (["--sr", "--cache-merged-pkl", "cached"], "not cacheable"),
])
def test_mixed_stage_wrong_region_and_cache_fail_before_load(
    monkeypatch, tmp_path, extra_args, error,
):
    sidecars = _mixed_sr_fixture(monkeypatch)
    monkeypatch.setattr(make_cr_and_sr_plots, "_mixed_stage_sidecars",
                        lambda _paths: sidecars)
    parser = make_cr_and_sr_plots.build_arg_parser()
    args = parser.parse_args([
        "-f", "primary.pkl.gz", "-f", "auxiliary.pkl.gz",
        "-o", str(tmp_path), "-n", "mixed", "--variables", "njets",
        *extra_args,
    ])
    with pytest.raises(ValueError, match=error):
        make_cr_and_sr_plots.run_with_args(args, parser)


def test_generic_sidecar_merge_rejects_mixed_kinds():
    from topeft.modules.histogram_artifact import merge_histogram_sidecars

    with pytest.raises(Exception, match="incompatible histogram artifact kinds"):
        merge_histogram_sidecars([
            {"artifact": {"artifact_kind": "nonprompt_output"}},
            {"artifact": {"artifact_kind": "processor_output"}},
        ])


def test_sr_njets_selectors_reach_native_loader_before_materialization(monkeypatch, tmp_path):
    parser = make_cr_and_sr_plots.build_arg_parser()
    args = parser.parse_args([
        "-f", "synthetic.pkl.gz", "-o", str(tmp_path), "-n", "bounded",
        "--sr", "--variables", "njets", "--merge-only",
    ])
    captured = {}

    def capture_load(_paths, **kwargs):
        captured.update(kwargs)
        return {}, {"schema": "split_sibling_v1", "runtime_histogram_families": ["njets"]}

    monkeypatch.setattr(make_cr_and_sr_plots, "load_and_merge_histogram_pkls", capture_load)
    assert make_cr_and_sr_plots.run_with_args(args, parser) == 0
    assert captured["selected_families"] == ["njets"]
    assert any(re.search(pattern, "3l_onZ_2b_4j") for pattern in captured["channel_patterns"])
    assert any(re.search(pattern, "3l_onZ_2b") for pattern in captured["channel_patterns"])
    assert not any(re.search(pattern, "2lss_ee_CR_1j") for pattern in captured["channel_patterns"])


def test_sr_njets_selected_consumer_matches_unselected_synthetic_multi_pkl(tmp_path, monkeypatch):
    from tests.test_make_cards_selective_loading import _payload, _assert_histograms_equal
    from tests.test_split_nominal_loader import _policy_for_families, _write_versioned, _scalar
    from topeft.modules import datacard_tools, nominal_schema
    from topeft.modules.nominal_schema import materialize_scalar_histogram_dict

    channels = ("3l_onZ_2b_4j", "2lss_ee_CR_1j")
    first_payload = _payload(channels=channels, signal=False)
    second_payload = _payload(channels=channels, background=False)
    data_samples = {
        "background_dataset": {"histAxisName": "background", "isData": False, "WCnames": []},
        "signal_dataset": {"histAxisName": "signal", "isData": False, "WCnames": ["ctG"]},
        "data_dataset": {"histAxisName": "data", "isData": True, "WCnames": []},
    }
    first_payload[nominal_schema.scalar_nominal_key("njets")] += _scalar(
        "data", 7.0, channel=channels[0]
    )
    first_payload[nominal_schema.sumw2_key("njets")] += _scalar(
        "data", 7.0, companion=True, channel=channels[0]
    )
    first, second = tmp_path / "first.pkl.gz", tmp_path / "second.pkl.gz"
    policy = _policy_for_families(("njets", "ptz_wtau"), samples=data_samples)
    _write_versioned(first, first_payload, policy, samples=data_samples)
    _write_versioned(second, second_payload, policy, samples=data_samples)
    paths = [str(first), str(second)]
    eager, eager_report = datacard_tools.load_and_merge_histogram_pkls(paths)
    expected = {
        key: histogram.prune("channel", [channels[0]])
        for key, histogram in eager.items() if key.startswith("njets")
    }
    expected_view = materialize_scalar_histogram_dict(
        expected, runtime_families=["njets"], wc_values={},
    )
    captured = {}
    original_loader = make_cr_and_sr_plots.load_and_merge_histogram_pkls

    def capture_loader(*args, **kwargs):
        result = original_loader(*args, **kwargs)
        captured["report"] = result[1]
        return result

    monkeypatch.setattr(make_cr_and_sr_plots, "load_and_merge_histogram_pkls", capture_loader)
    monkeypatch.setattr(
        make_cr_and_sr_plots, "run_plots_for_region",
        lambda _region, histograms, *_args, **_kwargs: captured.update(view=histograms),
    )
    parser = make_cr_and_sr_plots.build_arg_parser()
    args = parser.parse_args([
        "-f", str(first), "-f", str(second), "-o", str(tmp_path),
        "-n", "selected", "--sr", "--variables", "njets",
    ])
    assert make_cr_and_sr_plots.run_with_args(args, parser) == 0
    assert set(captured["view"]) == {"njets", "njets_sumw2"}
    assert "ptz_wtau__scalar_nominal" in captured["report"]["files"][0]["discarded_keys"]
    assert channels[1] not in captured["report"]["files"][0]["resolved_channels"]
    _assert_histograms_equal(captured["view"], expected_view)
    assert "data" in list(captured["view"]["njets"].axes["process"])
    assert "data" in list(eager[nominal_schema.scalar_nominal_key("njets")].axes["process"])
    assert eager_report["schema"] == captured["report"]["schema"]


def test_sr_njets_selected_loader_preserves_collision_rejection(tmp_path):
    from tests.test_make_cards_selective_loading import _write_fixture
    from topeft.modules.datacard_tools import load_and_merge_histogram_pkls

    first, second = tmp_path / "first.pkl.gz", tmp_path / "second.pkl.gz"
    _write_fixture(first, signal=False)
    _write_fixture(second, signal=False)
    with pytest.raises(RuntimeError, match="Duplicate histogram contribution support"):
        load_and_merge_histogram_pkls(
            [str(first), str(second)], selected_families=["njets"],
            channel_patterns=make_cr_and_sr_plots._sr_njets_channel_patterns(),
        )


class _DummyHist:
    def __init__(self):
        self.scale_factors = []

    def view(self, *, flow, as_dict):
        assert flow is True
        assert as_dict is True
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


def _make_simple_stacked_inputs(axis_label=None):
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    value_axis = hist.axis.Regular(
        2, 0.0, 2.0, name="lj0pt", label=axis_label
    )
    h_mc = hist.Hist(process_axis, value_axis, storage=hist.storage.Double())
    h_data = hist.Hist(process_axis, value_axis, storage=hist.storage.Double())

    for bin_idx, weight in enumerate((10.0, 5.0)):
        h_mc.fill(process="ttbarSL", lj0pt=[bin_idx + 0.25], weight=[weight])
    for bin_idx, weight in enumerate((8.0, 6.0)):
        h_data.fill(process="data", lj0pt=[bin_idx + 0.25], weight=[weight])

    group_map = {"Top": ["ttbarSL"]}
    return h_mc, h_data, group_map


def _make_negative_stacked_inputs():
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    value_axis = hist.axis.Regular(4, 0.0, 4.0, name="lj0pt")
    h_mc = hist.Hist(process_axis, value_axis, storage=hist.storage.Double())
    h_sumw2 = hist.Hist(process_axis, value_axis, storage=hist.storage.Double())
    h_data = hist.Hist(process_axis, value_axis, storage=hist.storage.Double())

    central_values = {
        "source_a": np.array([10.0, -5.0, 0.0, 4.0]),
        "source_b": np.array([3.0, 2.0, 0.0, 1.0]),
    }
    variances = {
        "source_a": np.array([4.0, 25.0, 9.0, 16.0]),
        "source_b": np.array([1.0, 4.0, 1.0, 1.0]),
    }
    for process, values in central_values.items():
        for bin_index, value in enumerate(values):
            h_mc.fill(process=process, lj0pt=bin_index + 0.5, weight=value)
    for process, values in variances.items():
        for bin_index, value in enumerate(values):
            h_sumw2.fill(process=process, lj0pt=bin_index + 0.5, weight=value)
    for bin_index, value in enumerate((13.0, 4.0, 0.0, 10.0)):
        h_data.fill(process="data", lj0pt=bin_index + 0.5, weight=value)

    return h_mc, h_sumw2, h_data, {
        "Process A": ["source_a"],
        "Process B": ["source_b"],
    }


def _make_data_interval_stacked_inputs():
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    value_axis = hist.axis.Regular(3, 0.0, 3.0, name="lj0pt")
    h_mc = hist.Hist(process_axis, value_axis, storage=hist.storage.Double())
    h_data = hist.Hist(process_axis, value_axis, storage=hist.storage.Double())

    for bin_index, value in enumerate((10.0, 5.0, 0.0)):
        if value:
            h_mc.fill(process="source", lj0pt=bin_index + 0.5, weight=value)
    for bin_index, value in enumerate((0.0, 1.0, 2.0)):
        if value:
            h_data.fill(process="data", lj0pt=bin_index + 0.5, weight=value)

    return h_mc, h_data, {"Process": ["source"]}


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


def _make_mixed_raw_count_plotting_input(weights):
    histogram = make_cr_and_sr_plots.SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(2, 0.0, 2.0, name="lj0pt"),
        storage="Double",
        track_raw_counts=True,
    )
    histogram.fill(
        process="prompt",
        systematic="nominal",
        lj0pt=np.asarray([0.25, 1.25]),
        weight=np.asarray(weights[:2]),
        record_raw_count=True,
    )
    histogram.fill(
        process="nonprompt",
        systematic="nominal",
        lj0pt=np.asarray([0.25, 1.25]),
        weight=np.asarray(weights[2:]),
        record_raw_count=False,
    )
    return histogram


def test_plotting_numeric_view_preserves_payload_and_unrecords_raw_counts():
    nominal = _make_mixed_raw_count_plotting_input([2.5, -1.25, 7.0, 3.5])
    sumw2 = _make_mixed_raw_count_plotting_input([6.25, 1.5625, 49.0, 12.25])

    source_values = {
        key: np.array(values, copy=True)
        for key, values in nominal.view(flow=True, as_dict=True).items()
    }
    source_sumw2 = {
        key: np.array(values, copy=True)
        for key, values in sumw2.view(flow=True, as_dict=True).items()
    }
    source_raw_states = {
        key: None if state is None else np.array(state, copy=True)
        for key, state in nominal._validated_raw_count_states().items()
    }

    with pytest.raises(RuntimeError, match="recorded and explicitly unrecorded"):
        nominal[{"process": sum}]

    nominal_view = make_cr_and_sr_plots._plotting_numeric_view(nominal)
    sumw2_view = make_cr_and_sr_plots._plotting_numeric_view(sumw2)

    assert nominal_view is not nominal
    assert sumw2_view is not sumw2
    assert nominal_view.axes.name == nominal.axes.name
    assert tuple(type(axis) for axis in nominal_view.axes) == tuple(
        type(axis) for axis in nominal.axes
    )
    assert [list(axis) for axis in nominal_view.categorical_axes] == [
        list(axis) for axis in nominal.categorical_axes
    ]
    assert list(nominal_view.axes["process"]) == ["prompt", "nonprompt"]
    np.testing.assert_array_equal(
        nominal_view.axes["lj0pt"].edges,
        nominal.axes["lj0pt"].edges,
    )
    for key, values in nominal_view.view(flow=True, as_dict=True).items():
        np.testing.assert_array_equal(values, source_values[key])
    for key, values in sumw2_view.view(flow=True, as_dict=True).items():
        np.testing.assert_array_equal(values, source_sumw2[key])
    for key, state in nominal._validated_raw_count_states().items():
        expected = source_raw_states[key]
        if expected is None:
            assert state is None
        else:
            np.testing.assert_array_equal(state, expected)
    assert all(
        state is None
        for state in nominal_view._validated_raw_count_states().values()
    )
    assert all(
        state is None
        for state in sumw2_view._validated_raw_count_states().values()
    )
    np.testing.assert_array_equal(
        next(
            iter(
                nominal_view[
                    {"process": sum, "systematic": "nominal"}
                ].view(flow=True, as_dict=True).values()
            )
        ),
        np.asarray([0.0, 9.5, 2.25, 0.0]),
    )
    np.testing.assert_array_equal(
        next(
            iter(
                sumw2_view[
                    {"process": sum, "systematic": "nominal"}
                ].view(flow=True, as_dict=True).values()
            )
        ),
        np.asarray([0.0, 55.25, 13.8125, 0.0]),
    )

    dense_histogram = hist.Hist(hist.axis.Regular(1, 0.0, 1.0, name="x"))
    assert make_cr_and_sr_plots._plotting_numeric_view(dense_histogram) is dense_histogram


def test_ratio_axis_range_is_fixed_and_ratio_legend_is_opt_in():
    h_mc, h_data, group_map = _make_simple_stacked_inputs()

    default_fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        unblind=True,
    )
    legend_fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        unblind=True,
        style={"ratio_band_legend": {"enabled": True}},
    )

    try:
        default_ratio_axis = default_fig.axes[1]
        legend_ratio_axis = legend_fig.axes[1]
        assert default_ratio_axis.get_ylim() == pytest.approx((0.0, 2.0))
        assert legend_ratio_axis.get_ylim() == pytest.approx((0.0, 2.0))
        assert default_ratio_axis.get_legend() is None
        assert legend_ratio_axis.get_legend() is not None
    finally:
        make_cr_and_sr_plots.plt.close(default_fig)
        make_cr_and_sr_plots.plt.close(legend_fig)


def test_nominal_sm_plot_view_clips_grouped_centrals_without_redistribution():
    grouped_centrals = np.array(
        [[10.0, -5.0, 0.0, 4.0], [3.0, 2.0, 0.0, 1.0]]
    )
    grouped_variances = np.array(
        [[4.0, 25.0, 9.0, 16.0], [1.0, 4.0, 1.0, 1.0]]
    )
    source_centrals = grouped_centrals.copy()
    source_variances = grouped_variances.copy()

    plot_view = make_cr_and_sr_plots._prepare_nominal_sm_plot_view(
        grouped_centrals,
        grouped_variances=grouped_variances,
        process_names=("Process A", "Process B"),
        bin_edges=np.arange(5, dtype=float),
        diagnostic_context={
            "region": "CR",
            "year_or_run": "run2",
            "presentation_mode": "merged",
            "binning_mode": "fitting",
            "category": "2los_CRZ",
            "variable": "lj0pt",
        },
    )

    np.testing.assert_array_equal(
        plot_view["plot_grouped_centrals"]["Process A"],
        [10.0, 0.0, 0.0, 4.0],
    )
    np.testing.assert_array_equal(
        plot_view["plot_grouped_centrals"]["Process B"],
        [3.0, 2.0, 0.0, 1.0],
    )
    np.testing.assert_array_equal(plot_view["displayed_total"], [13.0, 2.0, 0.0, 5.0])
    np.testing.assert_array_equal(plot_view["grouped_variances"]["Process A"], grouped_variances[0])
    np.testing.assert_array_equal(plot_view["total_variance"], [5.0, 29.0, 10.0, 17.0])
    np.testing.assert_array_equal(grouped_centrals, source_centrals)
    np.testing.assert_array_equal(grouped_variances, source_variances)

    assert plot_view["clipped_negative_diagnostics"] == [
        {
            "region": "CR",
            "year_or_run": "run2",
            "presentation_mode": "merged",
            "binning_mode": "fitting",
            "category": "2los_CRZ",
            "variable": "lj0pt",
            "bin_index": 1,
            "bin_low": 1.0,
            "bin_high": 2.0,
            "display_process": "Process A",
            "raw_signed_yield": -5.0,
            "plotted_yield": 0.0,
            "variance": 25.0,
            "displayed_total_after_clip": 2.0,
            "ratio_denominator": 2.0,
        }
    ]


def test_garwood_data_intervals_match_exact_numeric_anchors():
    intervals = make_cr_and_sr_plots._garwood_data_intervals(
        np.array([0.0, 1.0, 2.0])
    )

    np.testing.assert_allclose(
        intervals["lower_endpoints"],
        [0.0, 0.17274753436, 0.70817037722],
        rtol=0.0,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        intervals["upper_endpoints"],
        [1.84105476095, 3.29956971157, 4.63791009915],
        rtol=0.0,
        atol=1e-11,
    )
    np.testing.assert_array_equal(intervals["central_counts"], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(intervals["zero_count_mask"], [True, False, False])
    assert intervals["confidence_level"] == pytest.approx(0.6827)


@pytest.mark.parametrize("value", [0, 1, 2, 3.0])
def test_garwood_data_intervals_accept_integer_valued_counts(value):
    intervals = make_cr_and_sr_plots._garwood_data_intervals([value])
    assert intervals["central_counts"][0] == value


def test_garwood_data_intervals_reject_noninteger_data():
    with pytest.raises(ValueError, match="must be integer-valued"):
        make_cr_and_sr_plots._garwood_data_intervals([1.25])


def test_linear_data_uses_asymmetric_garwood_errors_without_epsilon():
    h_mc, h_data, group_map = _make_data_interval_stacked_inputs()
    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        unblind=True,
        uncertainty_mode="stat",
    )

    try:
        intervals = fig._topeft_data_intervals
        assert intervals["central_counts"][0] == 0.0
        assert intervals["lower_errors"][0] == 0.0
        assert intervals["upper_errors"][0] == pytest.approx(1.84105476095)
        assert intervals["lower_errors"][1] != pytest.approx(
            intervals["upper_errors"][1]
        )
        assert fig._topeft_plot_summary["renderer_epsilon"] is None
        assert fig._topeft_plot_summary["Garwood_upper_limit_count"] == 0
    finally:
        make_cr_and_sr_plots.plt.close(fig)


def test_log_zero_data_uses_garwood_upper_limit_without_fake_marker(monkeypatch):
    h_mc, h_data, group_map = _make_data_interval_stacked_inputs()
    histplot_calls = []
    original_histplot = make_cr_and_sr_plots.hep.histplot

    def _capture_histplot(*args, **kwargs):
        histplot_calls.append((args, kwargs))
        return original_histplot(*args, **kwargs)

    monkeypatch.setattr(make_cr_and_sr_plots.hep, "histplot", _capture_histplot)
    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        unblind=True,
        uncertainty_mode="stat",
        log_scale=True,
    )

    try:
        data_call = next(
            call for call in histplot_calls if call[1].get("label") == "Data"
        )
        rendered_data_centrals = np.asarray(data_call[0][0], dtype=float)
        assert np.isnan(rendered_data_centrals[0])
        assert np.isnan(rendered_data_centrals[-1])
        assert rendered_data_centrals[1] == 1.0
        rendered_data_errors = np.asarray(data_call[1]["yerr"], dtype=float)
        assert rendered_data_errors[0, 1] != pytest.approx(
            rendered_data_errors[1, 1]
        )
        assert fig._topeft_plot_summary["Garwood_upper_limit_count"] == 2
        np.testing.assert_allclose(
            fig._topeft_plot_summary["Garwood_upper_limit_anchors"],
            [1.84105476095, 1.84105476095],
            rtol=0.0,
            atol=1e-11,
        )
        assert len(fig._topeft_data_upper_limit_artists) == 1
        assert fig._topeft_plot_summary["renderer_epsilon"] != pytest.approx(
            fig._topeft_plot_summary["Garwood_upper_limit_anchors"][0]
        )
        assert fig._topeft_data_intervals["lower_endpoints"][0] == 0.0
    finally:
        make_cr_and_sr_plots.plt.close(fig)


def test_ratio_uses_garwood_endpoints_and_masks_zero_mc():
    h_mc, h_data, group_map = _make_data_interval_stacked_inputs()
    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        unblind=True,
        uncertainty_mode="stat",
    )

    try:
        intervals = fig._topeft_data_intervals
        denominator = fig._topeft_plot_summary["ratio_denominator"]
        assert fig._topeft_ratio_values[0] == 0.0
        assert fig._topeft_ratio_errors[0, 0] == 0.0
        assert fig._topeft_ratio_errors[1, 0] == pytest.approx(
            intervals["upper_endpoints"][0] / denominator[0]
        )
        assert fig._topeft_ratio_values[1] == pytest.approx(1.0 / 5.0)
        assert fig._topeft_ratio_interval_endpoints["lower"][1] == pytest.approx(
            intervals["lower_endpoints"][1] / denominator[1]
        )
        assert fig._topeft_ratio_interval_endpoints["upper"][1] == pytest.approx(
            intervals["upper_endpoints"][1] / denominator[1]
        )
        assert np.isnan(fig._topeft_ratio_values[2])
        assert np.all(np.isnan(fig._topeft_ratio_errors[:, 2]))
        assert np.isnan(fig._topeft_ratio_interval_endpoints["lower"][2])
        assert np.isnan(fig._topeft_ratio_interval_endpoints["upper"][2])
    finally:
        make_cr_and_sr_plots.plt.close(fig)


def test_negative_diagnostics_keep_presentation_and_binning_modes_separate():
    plot_view = make_cr_and_sr_plots._prepare_nominal_sm_plot_view(
        [[-1.0], [2.0]],
        process_names=("Negative", "Positive"),
        diagnostic_context={
            "presentation_mode": "merged-njets",
            "binning_mode": "processing",
        },
    )

    diagnostic = plot_view["clipped_negative_diagnostics"][0]
    assert diagnostic["presentation_mode"] == "merged-njets"
    assert diagnostic["binning_mode"] == "processing"
    assert diagnostic["presentation_mode"] != diagnostic["binning_mode"]


def test_stacked_plot_uses_clipped_total_for_ratio_and_preserved_sumw2():
    h_mc, h_sumw2, h_data, group_map = _make_negative_stacked_inputs()
    source_mc = h_mc.values(flow=True).copy()
    source_sumw2 = h_sumw2.values(flow=True).copy()

    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        h_mc_sumw2=h_sumw2,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        unblind=True,
        uncertainty_mode="stat",
    )

    try:
        np.testing.assert_array_equal(
            fig._topeft_plot_view["displayed_total"], [13.0, 2.0, 0.0, 5.0, 0.0]
        )
        np.testing.assert_allclose(
            fig._topeft_ratio_values,
            [1.0, 2.0, np.nan, 2.0, np.nan],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            fig._topeft_uncertainty_bands["mc_stat_uncertainty"],
            np.sqrt([5.0, 29.0, 10.0, 17.0, 0.0]),
        )
        assert np.isnan(fig._topeft_uncertainty_bands["ratio_stat_band_up"][2])
        assert np.isnan(fig._topeft_uncertainty_bands["ratio_stat_band_down"][2])
        assert fig._topeft_plot_diagnostics[0]["variance"] == 25.0
        np.testing.assert_array_equal(h_mc.values(flow=True), source_mc)
        np.testing.assert_array_equal(h_sumw2.values(flow=True), source_sumw2)
    finally:
        make_cr_and_sr_plots.plt.close(fig)


def test_log_plot_keeps_zero_centrals_and_uses_renderer_only_epsilon():
    h_mc, h_sumw2, h_data, group_map = _make_negative_stacked_inputs()
    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        h_mc_sumw2=h_sumw2,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        unblind=True,
        uncertainty_mode="stat",
        log_scale=True,
    )

    try:
        assert fig.axes[0].get_yscale() == "log"
        np.testing.assert_array_equal(
            fig._topeft_plot_view["displayed_total"], [13.0, 2.0, 0.0, 5.0, 0.0]
        )
        assert fig._topeft_plot_summary["renderer_epsilon"] == pytest.approx(0.02)
        assert fig._topeft_plot_summary["zero_total_bin_count"] == 2
        assert np.isnan(fig._topeft_ratio_values[2])
        assert isinstance(
            fig.axes[0].yaxis.get_major_formatter(),
            make_cr_and_sr_plots.ticker.LogFormatterMathtext,
        )
        assert not fig._topeft_plot_summary["scientific_formatter_used"]
    finally:
        make_cr_and_sr_plots.plt.close(fig)


def test_all_zero_displayed_total_falls_back_from_log(caplog):
    plot_arrays = [np.zeros(3), np.zeros(3)]
    resolved = make_cr_and_sr_plots._prepare_log_scaled_stacks(
        plot_arrays, "lj0pt", True
    )

    resolved_arrays, requested, enabled, epsilon, displayed_total = resolved
    assert requested is False
    assert enabled is False
    assert epsilon is None
    np.testing.assert_array_equal(displayed_total, np.zeros(3))
    for resolved_array, source_array in zip(resolved_arrays, plot_arrays):
        np.testing.assert_array_equal(resolved_array, source_array)
    assert "displayed clipped MC total is zero in every bin" in caplog.text


@pytest.mark.parametrize(
    ("ylim", "expected_scientific"),
    [((0.0, 99.9), False), ((0.0, 100.0), True), ((-1.0, 10000.0), True)],
)
def test_main_y_formatter_uses_inclusive_final_visible_threshold(
    ylim, expected_scientific
):
    fig, (ax, rax) = make_cr_and_sr_plots.plt.subplots(2, 1)
    try:
        ax.set_ylim(*ylim)
        result = make_cr_and_sr_plots._configure_main_y_formatter(
            ax, log_axis_enabled=False
        )
        make_cr_and_sr_plots._configure_ratio_y_formatter(rax)
        fig.canvas.draw()
        assert result["scientific_formatter_used"] is expected_scientific
        assert ax.yaxis.offsetText.get_text() if expected_scientific else not ax.yaxis.offsetText.get_text()
        assert not rax.yaxis.offsetText.get_text()
        assert not rax.yaxis.get_major_formatter()._scientific
    finally:
        make_cr_and_sr_plots.plt.close(fig)


def test_show_ratio_legend_cli_is_disabled_by_default_and_explicitly_enabled():
    parser = make_cr_and_sr_plots.build_arg_parser()
    assert parser.parse_args(["-f", "input.pkl.gz"]).show_ratio_legend is False
    assert (
        parser.parse_args(
            ["-f", "input.pkl.gz", "--show-ratio-legend"]
        ).show_ratio_legend
        is True
    )


def test_binning_cli_defaults_to_processing_and_accepts_fitting():
    parser = make_cr_and_sr_plots.build_arg_parser()
    assert parser.parse_args(["-f", "input.pkl.gz"]).binning == "processing"
    assert (
        parser.parse_args(["-f", "input.pkl.gz", "--binning", "fitting"]).binning
        == "fitting"
    )


def test_output_formats_cli_defaults_to_png_and_accepts_vector_formats():
    parser = make_cr_and_sr_plots.build_arg_parser()
    assert parser.parse_args(["-f", "input.pkl.gz"]).output_formats == ["png"]
    assert parser.parse_args(
        ["-f", "input.pkl.gz", "--output-formats", "png", "pdf", "svg"]
    ).output_formats == ["png", "pdf", "svg"]


def test_plot_binning_view_uses_stored_processing_and_shared_fitting_resolution():
    channel = "3l_1tau_1b_2j"
    histogram = make_cr_and_sr_plots.SparseHist(
        hist.axis.StrCategory(["background"], name="process", growth=True),
        hist.axis.Regular(12, 0, 600, name="lj0pt"),
    )
    histogram.fill(process="background", lj0pt=np.array([25.0, 175.0, 375.0]))

    processing_view = make_cr_and_sr_plots._apply_plot_binning_view(
        histogram, "lj0pt", [channel], "processing"
    )
    fitting_view = make_cr_and_sr_plots._apply_plot_binning_view(
        histogram, "lj0pt", [channel], "fitting"
    )

    assert processing_view is histogram
    assert np.array_equal(processing_view.axes["lj0pt"].edges, np.arange(0, 601, 50))
    assert np.array_equal(fitting_view.axes["lj0pt"].edges, [0, 150, 250, 350])
    assert sum(next(iter(fitting_view.view(flow=True).values()))) == pytest.approx(3.0)


def test_plot_binning_view_preserves_no_fitting_family_identity(monkeypatch):
    channel = "2los_CRZ_2j"
    histogram = make_cr_and_sr_plots.SparseHist(
        hist.axis.StrCategory(["background"], name="process", growth=True),
        hist.axis.Regular(15, 0, 300, name="j0pt"),
    )
    histogram.fill(process="background", j0pt=np.array([25.0, 175.0, 275.0]))
    original_edges = histogram.axes["j0pt"].edges.copy()
    original_view = {
        key: value.copy() for key, value in histogram.view(flow=True).items()
    }

    def _unexpected_rebin(*args, **kwargs):
        pytest.fail("no-fitting families must not call rebin_histogram")

    monkeypatch.setattr(
        make_cr_and_sr_plots, "rebin_histogram", _unexpected_rebin
    )
    fitting_view = make_cr_and_sr_plots._apply_plot_binning_view(
        histogram, "j0pt", [channel], "fitting"
    )

    assert fitting_view is histogram
    assert np.array_equal(fitting_view.axes["j0pt"].edges, original_edges)
    assert set(fitting_view.view(flow=True)) == set(original_view)
    for key, value in fitting_view.view(flow=True).items():
        assert np.array_equal(value, original_view[key])


def test_aggregate_fitting_plot_rejects_incompatible_exact_channel_axes():
    histogram = make_cr_and_sr_plots.SparseHist(
        hist.axis.StrCategory(["background"], name="process", growth=True),
        hist.axis.Regular(12, 0, 600, name="lt"),
    )
    histogram.fill(process="background", lt=25.0)
    with pytest.raises(ValueError, match="incompatible fitting axes"):
        make_cr_and_sr_plots._apply_plot_binning_view(
            histogram,
            "lt",
            ["3l_m_offZ_2b_fwd_1j", "3l_m_offZ_2b_fwd_2j"],
            "fitting",
        )


def test_uncertainty_mode_parser_and_resolution_contract():
    parser = make_cr_and_sr_plots.build_arg_parser()

    assert make_cr_and_sr_plots._resolve_uncertainty_mode() == "total"
    assert make_cr_and_sr_plots._resolve_uncertainty_mode(skip_syst=True) == "stat"
    assert (
        make_cr_and_sr_plots._resolve_uncertainty_mode(no_uncertainties=True)
        == "none"
    )
    assert parser.parse_args(["-f", "input.pkl.gz"]).no_uncertainties is False
    assert parser.parse_args(["-f", "input.pkl.gz", "--skip-syst"]).skip_syst
    assert parser.parse_args(["-f", "input.pkl.gz", "--no-uncertainties"]).no_uncertainties
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["-f", "input.pkl.gz", "--skip-syst", "--no-uncertainties"]
        )
    assert (
        make_cr_and_sr_plots._resolve_plot_axis_label("missing_plot_axis")
        == "missing_plot_axis"
    )


def test_none_mode_preserves_unblinded_markers_without_errorbars_or_bands(monkeypatch):
    h_mc, h_data, group_map = _make_simple_stacked_inputs()
    plotted_calls = []
    fill_between_calls = []

    original_histplot = make_cr_and_sr_plots.hep.histplot

    def _capture_histplot(*args, **kwargs):
        plotted_calls.append((args, kwargs))
        return original_histplot(*args, **kwargs)

    def _capture_fill_between(self, *args, **kwargs):
        fill_between_calls.append((args, kwargs))
        return None

    monkeypatch.setattr(make_cr_and_sr_plots.hep, "histplot", _capture_histplot)
    monkeypatch.setattr(
        make_cr_and_sr_plots.mpl.axes.Axes,
        "fill_between",
        _capture_fill_between,
    )

    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        unblind=True,
        uncertainty_mode="none",
    )

    try:
        errorbar_calls = [
            (args, kwargs)
            for args, kwargs in plotted_calls
            if kwargs.get("histtype") == "errorbar"
        ]
        assert len(errorbar_calls) == 2
        assert errorbar_calls[0][1]["label"] == "Data"
        assert errorbar_calls[0][1]["yerr"] is False
        assert errorbar_calls[1][1]["yerr"] is False
        np.testing.assert_allclose(errorbar_calls[1][0][0][:2], np.array([0.8, 1.2]))
        assert not fill_between_calls
        assert not any(
            "unc." in text.get_text().lower() for text in fig.texts
        )
    finally:
        make_cr_and_sr_plots.plt.close(fig)


def test_none_mode_keeps_blinded_mc_stack_without_band(monkeypatch):
    h_mc, h_data, group_map = _make_simple_stacked_inputs()
    fill_between_calls = []

    def _capture_fill_between(self, *args, **kwargs):
        fill_between_calls.append((args, kwargs))
        return None

    monkeypatch.setattr(
        make_cr_and_sr_plots.mpl.axes.Axes,
        "fill_between",
        _capture_fill_between,
    )
    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        unblind=False,
        uncertainty_mode="none",
    )

    try:
        assert len(fig.axes) == 1
        assert not fill_between_calls
    finally:
        make_cr_and_sr_plots.plt.close(fig)


def test_none_mode_propagates_to_aggregate_and_per_channel_paths(monkeypatch, tmp_path):
    captured_modes = []
    negative_report_calls = []

    def _build_context(*_args, channel_mode_override, **_kwargs):
        return SimpleNamespace(
            name="CR",
            channel_mode=channel_mode_override,
            stacked_ratio_style=None,
        )

    def _capture_production(region_ctx, _save_dir, _variables, uncertainty_mode, *_args, **_kwargs):
        captured_modes.append((region_ctx.channel_mode, uncertainty_mode))
        return []

    minimal_summary = {
        "region": "CR",
        "channels_scanned": 0,
        "channel_entries": [],
        "zero_process_total": 0,
        "data_driven_zero_total": 0,
        "missing_data_driven_prefixes": set(),
        "errors": [],
    }
    monkeypatch.setattr(make_cr_and_sr_plots, "build_region_context", _build_context)
    monkeypatch.setattr(
        make_cr_and_sr_plots.yt, "is_split_by_lepflav", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(make_cr_and_sr_plots, "produce_region_plots", _capture_production)
    monkeypatch.setattr(
        make_cr_and_sr_plots,
        "_summarize_zero_yield_processes",
        lambda *_args, **_kwargs: minimal_summary,
    )
    monkeypatch.setattr(
        make_cr_and_sr_plots, "_emit_zero_yield_summary", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        make_cr_and_sr_plots,
        "write_negative_weight_report",
        lambda *_args, **_kwargs: negative_report_calls.append(True) or {"csv": "x", "markdown": "y"},
    )

    make_cr_and_sr_plots.run_plots_for_region(
        "CR",
        {},
        years=["2022"],
        save_dir_path=str(tmp_path),
        channel_output="both",
        uncertainty_mode="none",
        negative_weight_report=True,
    )

    assert captured_modes == [("aggregate", "none"), ("per-channel", "none")]
    assert negative_report_calls == [True]


def test_none_mode_does_not_calculate_rate_or_shape_systematics(monkeypatch, tmp_path):
    hist_inputs = {
        "met": _make_met_histogram_for_channels(["2lss_ee_CR_1j"])
    }

    def _unexpected_systematics(*_args, **_kwargs):
        raise AssertionError("none mode must not calculate drawing-only systematics")

    monkeypatch.setattr(
        make_cr_and_sr_plots, "get_rate_syst_arrs", _unexpected_systematics
    )
    monkeypatch.setattr(
        make_cr_and_sr_plots, "get_shape_syst_arrs", _unexpected_systematics
    )
    monkeypatch.setattr(
        make_cr_and_sr_plots.yt, "is_split_by_lepflav", lambda *_args, **_kwargs: True
    )

    make_cr_and_sr_plots.run_plots_for_region(
        "CR",
        hist_inputs,
        years=["2022"],
        save_dir_path=str(tmp_path),
        variables=["met"],
        channel_output="both",
        uncertainty_mode="none",
        workers=1,
    )


@pytest.mark.parametrize(
    "render_results, expected_summary, expected_tasks",
    [
        (
            [(1, 0, set(), []), (0, 0, set(), [])],
            "1 plot without uncertainty bars or bands",
            "after completing 2 rendering tasks",
        ),
        (
            [(1, 0, set(), []), (1, 0, set(), []), (0, 0, set(), [])],
            "2 plots without uncertainty bars or bands",
            "after completing 3 rendering tasks",
        ),
    ],
)
def test_none_summary_reports_successful_plot_count_not_task_count(
    monkeypatch,
    capsys,
    render_results,
    expected_summary,
    expected_tasks,
):
    variable_names = [f"var_{idx}" for idx in range(len(render_results))]
    region_ctx = SimpleNamespace(
        dict_of_hists={name: object() for name in variable_names},
        name="CR",
        apply_category_skips=False,
        skip_variables=set(),
        unblind_default=True,
    )
    payload = {
        "channel_dict": {},
        "channel_transformations": {},
        "is_sparse2d": False,
    }
    results = iter(render_results)

    monkeypatch.setattr(
        make_cr_and_sr_plots,
        "_resolve_requested_variables",
        lambda *_args, **_kwargs: variable_names,
    )
    monkeypatch.setattr(
        make_cr_and_sr_plots,
        "_prepare_variable_payload",
        lambda *_args, **_kwargs: payload,
    )
    monkeypatch.setattr(
        make_cr_and_sr_plots,
        "_render_variable",
        lambda *_args, **_kwargs: next(results),
    )

    make_cr_and_sr_plots.produce_region_plots(
        region_ctx,
        None,
        None,
        "none",
        False,
        False,
        workers=1,
    )

    output = capsys.readouterr().out
    assert expected_summary in output
    assert expected_tasks in output


def _parallel_test_region_context(variable_names):
    return SimpleNamespace(
        dict_of_hists={name: object() for name in variable_names},
        name="SR",
        apply_category_skips=False,
        skip_variables=set(),
        unblind_default=False,
    )


def _parallel_test_payload():
    return {
        "channel_dict": {},
        "channel_transformations": {},
        "is_sparse2d": False,
    }


def test_parallel_rendering_caps_worker_count_at_task_count(monkeypatch):
    variable_names = ["met", "lt"]
    captured = {"max_workers": None, "submitted": []}

    class _FakeExecutor:
        def __init__(self, *, max_workers, **_kwargs):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, _function, task_id, payload):
            captured["submitted"].append((task_id, payload))
            future = concurrent.futures.Future()
            future.set_result((task_id, 0, 0, set(), []))
            return future

    monkeypatch.setattr(
        make_cr_and_sr_plots,
        "_resolve_requested_variables",
        lambda *_args, **_kwargs: variable_names,
    )
    monkeypatch.setattr(
        make_cr_and_sr_plots,
        "_prepare_variable_payload",
        lambda *_args, **_kwargs: _parallel_test_payload(),
    )
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _FakeExecutor)

    make_cr_and_sr_plots.produce_region_plots(
        _parallel_test_region_context(variable_names),
        None,
        None,
        "none",
        False,
        False,
        workers=9,
    )

    assert captured["max_workers"] == 2
    assert captured["submitted"] == [(1, "met"), (2, "lt")]


def test_parallel_rendering_propagates_worker_failure(monkeypatch):
    variable_names = ["met", "lt"]

    class _FakeExecutor:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, _function, task_id, _payload):
            future = concurrent.futures.Future()
            if task_id == 1:
                future.set_exception(RuntimeError("synthetic render failure"))
            else:
                future.set_result((task_id, 0, 0, set(), []))
            return future

    monkeypatch.setattr(
        make_cr_and_sr_plots,
        "_resolve_requested_variables",
        lambda *_args, **_kwargs: variable_names,
    )
    monkeypatch.setattr(
        make_cr_and_sr_plots,
        "_prepare_variable_payload",
        lambda *_args, **_kwargs: _parallel_test_payload(),
    )
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _FakeExecutor)

    with pytest.raises(RuntimeError, match="synthetic render failure"):
        make_cr_and_sr_plots.produce_region_plots(
            _parallel_test_region_context(variable_names),
            None,
            None,
            "none",
            False,
            False,
            workers=2,
        )


@pytest.mark.parametrize("unblind", [False, True])
def test_authoritative_axis_label_replaces_histogram_label(monkeypatch, unblind):
    h_mc, h_data, group_map = _make_simple_stacked_inputs(
        axis_label="description stored in pkl"
    )
    authoritative_label = "authoritative axes.py label"
    monkeypatch.setitem(
        make_cr_and_sr_plots.te_axes_info,
        "lj0pt",
        {"label": authoritative_label},
    )

    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        unblind=unblind,
    )

    try:
        visible_axis_labels = [
            axis.get_xlabel()
            for axis in fig.axes
            if axis.xaxis.label.get_visible() and axis.get_xlabel()
        ]
        visible_texts = [
            text.get_text()
            for text in fig.texts
            if text.get_visible() and text.get_text()
        ]
        assert visible_axis_labels == []
        assert visible_texts.count(authoritative_label) == 1
        assert "description stored in pkl" not in visible_texts
    finally:
        make_cr_and_sr_plots.plt.close(fig)


@pytest.mark.parametrize("uncertainty_mode", ["stat", "total"])
def test_stat_and_total_modes_keep_mc_uncertainty_bands(monkeypatch, uncertainty_mode):
    h_mc, h_data, group_map = _make_simple_stacked_inputs()
    mc_totals = h_mc[{"process": sum}].values(flow=True)[1:]
    fill_between_calls = []
    original_fill_between = make_cr_and_sr_plots.mpl.axes.Axes.fill_between

    def _capture_fill_between(self, *args, **kwargs):
        fill_between_calls.append((args, kwargs))
        return original_fill_between(self, *args, **kwargs)

    monkeypatch.setattr(
        make_cr_and_sr_plots.mpl.axes.Axes,
        "fill_between",
        _capture_fill_between,
    )

    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        unblind=True,
        uncertainty_mode=uncertainty_mode,
        h_mc_sumw2=h_mc if uncertainty_mode == "total" else None,
        syst_err="total",
        err_p_syst=mc_totals + 1.0,
        err_m_syst=mc_totals - 1.0,
        err_ratio_p_syst=np.array([1.1, 1.2]),
        err_ratio_m_syst=np.array([0.9, 0.8]),
    )

    try:
        assert fill_between_calls
        if uncertainty_mode == "stat":
            assert len(fill_between_calls) == 2
        else:
            assert len(fill_between_calls) == 4
    finally:
        make_cr_and_sr_plots.plt.close(fig)


def test_total_band_recenters_signed_systematic_excursions_on_displayed_total():
    h_mc, h_sumw2, h_data, group_map = _make_negative_stacked_inputs()
    raw_mc_totals = h_mc[{"process": sum}].values(flow=True)[1:]

    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        h_mc_sumw2=h_sumw2,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        unblind=True,
        uncertainty_mode="total",
        syst_err="total",
        err_p_syst=raw_mc_totals + 1.0,
        err_m_syst=raw_mc_totals - 1.0,
        err_ratio_p_syst=np.ones_like(raw_mc_totals),
        err_ratio_m_syst=np.ones_like(raw_mc_totals),
    )

    try:
        bands = fig._topeft_uncertainty_bands
        assert bands["systematic_reference_total"][1] == pytest.approx(-3.0)
        expected_uncertainty = np.sqrt(29.0 + 1.0)
        assert bands["mc_total_band_up_physical"][1] == pytest.approx(
            2.0 + expected_uncertainty
        )
        assert bands["mc_total_band_down_physical"][1] == pytest.approx(0.0)
        assert bands["ratio_total_band_up"][1] == pytest.approx(
            1.0 + expected_uncertainty / 2.0
        )
    finally:
        make_cr_and_sr_plots.plt.close(fig)


def test_uncertainty_band_shape_mismatch_fails_closed():
    with pytest.raises(ValueError) as exc_info:
        make_cr_and_sr_plots._compute_uncertainty_bands(
            ax=None,
            rax=None,
            bins=np.array([0.0, 1.0, 2.0]),
            mc_totals=np.array([4.0, 5.0]),
            mc_sumw2_vals={"synthetic": np.array([1.0, 1.0])},
            h_mc_sumw2=object(),
            unit_norm_bool=False,
            mc_scaled=False,
            mc_norm_factor=1.0,
            err_p_syst=np.array([5.0, 6.0]),
            err_m_syst=np.array([3.0, 4.0]),
            err_ratio_p_syst=None,
            err_ratio_m_syst=None,
            syst_err="total",
            raw_mc_totals=np.array([4.0, 5.0, 6.0, 7.0]),
        )

    message = str(exc_info.value)
    assert "systematic reference and displayed MC total shapes differ" in message
    assert "systematic_reference_total.shape=(4,)" in message
    assert "mc_totals.shape=(2,)" in message
    assert "Silent truncation/padding is forbidden" in message


def test_total_band_requires_nominal_sumw2_moments():
    with pytest.raises(ValueError, match="requires nominal MC sumw2 moments"):
        make_cr_and_sr_plots._compute_uncertainty_bands(
            ax=None,
            rax=None,
            bins=np.array([0.0, 1.0, 2.0]),
            mc_totals=np.array([4.0, 5.0]),
            mc_sumw2_vals={},
            h_mc_sumw2=None,
            unit_norm_bool=False,
            mc_scaled=False,
            mc_norm_factor=1.0,
            err_p_syst=np.array([5.0, 6.0]),
            err_m_syst=np.array([3.0, 4.0]),
            err_ratio_p_syst=None,
            err_ratio_m_syst=None,
            syst_err="total",
        )


def test_total_band_rejects_missing_statistical_bins():
    with pytest.raises(ValueError, match="MC sumw2 bin shapes differ"):
        make_cr_and_sr_plots._compute_uncertainty_bands(
            ax=None,
            rax=None,
            bins=np.array([0.0, 1.0, 2.0]),
            mc_totals=np.array([4.0, 5.0]),
            mc_sumw2_vals={"synthetic": np.array([1.0])},
            h_mc_sumw2=object(),
            unit_norm_bool=False,
            mc_scaled=False,
            mc_norm_factor=1.0,
            err_p_syst=np.array([5.0, 6.0]),
            err_m_syst=np.array([3.0, 4.0]),
            err_ratio_p_syst=None,
            err_ratio_m_syst=None,
            syst_err="total",
        )


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


def test_cms_label_always_routes_preliminary_status(monkeypatch):
    captured = {}

    def _capture_label(**kwargs):
        captured.update(kwargs)
        return ()

    monkeypatch.setattr(make_cr_and_sr_plots.hep.cms, "label", _capture_label)
    make_cr_and_sr_plots._draw_cms_label(
        object(),
        "138",
        "13",
        (("138", "13"),),
        fontsize=18.0,
    )

    status_keys = {
        key: captured[key] for key in ("data", "label") if key in captured
    }
    assert status_keys == {"data": True, "label": "Preliminary"}


@pytest.mark.parametrize("unblind", (False, True))
def test_stacked_plot_routes_cms_preliminary_status(unblind):
    h_mc, h_data, group_map = _make_simple_stacked_inputs()
    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        unblind=unblind,
    )

    try:
        label_text = " ".join(
            text.get_text() or ""
            for text in (*fig.texts, *fig.axes[0].texts)
        )
        assert "Preliminary" in label_text
        assert "Simulation" not in label_text
    finally:
        make_cr_and_sr_plots.plt.close(fig)


def test_scientific_multiplier_does_not_overlap_cms_status():
    h_mc, h_data, group_map = _make_simple_stacked_inputs()
    h_mc *= 1.0e7
    h_data *= 1.0e7

    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc,
        h_data=h_data,
        unit_norm_bool=False,
        var="lj0pt",
        group=group_map,
        unblind=True,
    )

    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        renderer = fig.canvas.get_renderer()
        cms_box = _get_cms_text_union_bbox(fig, ax, renderer)
        offset_box = ax.yaxis.offsetText.get_window_extent(renderer).transformed(
            fig.transFigure.inverted()
        )

        assert ax.yaxis.offsetText.get_text()
        assert offset_box.x1 < cms_box.x0
    finally:
        make_cr_and_sr_plots.plt.close(fig)


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


@pytest.mark.parametrize(
    ("process_label", "family"),
    [
        ("DY10to50_centralUL17", "DY"),
        ("DY50_centralUL17", "DY"),
        ("DYJetsToLL_MLL-10to50_central2022", "DY"),
        ("DYJetsToLL_MLL-50_central2023BPix", "DY"),
        ("ZGToLLG_centralUL18", "ZGamma"),
        ("ZG_MLL-4to50_PTG-100to200_central2022EE", "ZGamma"),
        ("ZG_MLL-50_PTG-600_central2023", "ZGamma"),
        ("ST_top_t-channel_centralUL16", "Single top"),
        ("tW_centralUL17", "Single top"),
        ("tbarW_centralUL18", "Single top"),
        ("ST_tW_Leptonic_central2022", "Single top"),
        ("ST_tbarW_Semileptonic_central2023BPix", "Single top"),
        ("WJetsToLNu_centralUL16APV", "Singleboson"),
        ("WJetsToLNu_central2023", "Singleboson"),
    ],
)
def test_excluded_mc_labels_use_native_cr_sr_metadata_once(process_label, family):
    for pattern_map in (
        make_cr_and_sr_plots.CR_GRP_PATTERNS,
        make_cr_and_sr_plots.SR_GRP_PATTERNS,
    ):
        group_map = make_cr_and_sr_plots.populate_group_map(
            [process_label], pattern_map
        )
        assert [
            group for group, members in group_map.items() if process_label in members
        ] == [family]

    canonical_name = make_cr_and_sr_plots.tc_utils.canonicalize_process_name(
        process_label
    )
    matching_patterns = [
        (group, pattern)
        for group, patterns in make_cr_and_sr_plots.SR_GRP_PATTERNS.items()
        for pattern in patterns
        if isinstance(pattern, str) and (pattern in canonical_name or pattern in process_label)
    ]
    assert len(matching_patterns) == 1
    assert matching_patterns[0][0] == family


@pytest.mark.parametrize(
    ("process_label", "group"),
    [
        ("TTTo2L2Nu_centralUL17", "ttll"),
        ("TTGJets_centralUL18", "Conv"),
        ("WWW_central2022", "Triboson"),
        ("TWZ_central2023", "tWZ"),
        ("WZTo3LNu_centralUL16", "Diboson"),
        ("ttH_central2022EE", "ttH"),
    ],
)
def test_excluded_mc_sr_metadata_preserves_baseline_groups(process_label, group):
    group_map = make_cr_and_sr_plots.populate_group_map(
        [process_label], make_cr_and_sr_plots.SR_GRP_PATTERNS
    )
    assert [
        name for name, members in group_map.items() if process_label in members
    ] == [group]


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
        "2lss_CR_1j_met_processing.png",
        "2lss_CR_1j_njets_processing.png",
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

    for subgroup in ("3l_m_offZ_1b", "3l_p_offZ_2b"):
        channel_bins = payload["channel_dict"].get(subgroup)
        assert channel_bins == [subgroup]

        make_cr_and_sr_plots.validate_channel_group(
            [hist_obj],
            channel_bins,
            payload["channel_transformations"],
            region=region_ctx.name,
            subgroup=subgroup,
            variable="njets",
            available_channels=channel_bins,
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

    entry = _find_zero_yield_entry(summary, label="2lss_4t_m", variable="njets")
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

    entry = _find_zero_yield_entry(summary, label="2lss_4t_m", variable="lj0pt")
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

    entry = _find_zero_yield_entry(summary, label="2lss_CR", variable="met")
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

    plot_paths = list(tmp_path.rglob("*_njets_processing.png"))
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

    for channel, weight in ("2los_ee_CRZ_2j", 1.0), ("2los_mm_CRZ_2j", 2.0):
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
    merged_plots = {path.name for path in merged_dir.glob("*.png")}
    assert {"2los_CRZ_met_processing.png"}.issubset(merged_plots)
    split_ee = tmp_path / "cr_2los_Z_ee"
    split_mm = tmp_path / "cr_2los_Z_mm"
    assert split_ee.exists()
    assert split_mm.exists()
    assert {"2los_CRZ_ee_met_processing.png"}.issubset(
        {path.name for path in split_ee.glob("*.png")}
    )
    assert {"2los_CRZ_mm_met_processing.png"}.issubset(
        {path.name for path in split_mm.glob("*.png")}
    )


@pytest.mark.parametrize(
    "channel_output,required_dirs,forbidden_dirs",
    [
        (
            "merged",
            {"cr_2lss"},
            {"cr_2lss_ee", "cr_2lss_mm"},
        ),
        (
            "split",
            {"cr_2lss_ee", "cr_2lss_mm"},
            {"cr_2lss"},
        ),
        (
            "both",
            {"cr_2lss", "cr_2lss_ee", "cr_2lss_mm"},
            set(),
        ),
        (
            "merged-njets",
            {"cr_2lss_1j", "cr_2lss_2j"},
            {
                "cr_2lss_ee_1j",
                "cr_2lss_mm_1j",
                "cr_2lss_ee_2j",
                "cr_2lss_mm_2j",
            },
        ),
        (
            "both-njets",
            {
                "cr_2lss_1j",
                "cr_2lss_2j",
                "cr_2lss_ee_1j",
                "cr_2lss_mm_1j",
                "cr_2lss_ee_2j",
                "cr_2lss_mm_2j",
            },
            set(),
        ),
    ],
)
def test_cr_channel_output_folder_routing_semantics(
    tmp_path, channel_output, required_dirs, forbidden_dirs
):
    h_met = _make_met_histogram_for_channels(
        (
            "2lss_ee_CR_1j",
            "2lss_mm_CR_1j",
            "2lss_ee_CR_2j",
            "2lss_mm_CR_2j",
        )
    )

    make_cr_and_sr_plots.run_plots_for_region(
        "CR",
        {"met": h_met},
        years=["2022"],
        save_dir_path=str(tmp_path),
        channel_output=channel_output,
        skip_syst_errs=True,
        workers=1,
        verbose=False,
    )

    produced_dirs = {path.name for path in tmp_path.iterdir() if path.is_dir()}
    assert required_dirs.issubset(produced_dirs)
    assert produced_dirs.isdisjoint(forbidden_dirs)
    assert all("_Nj" not in path for path in produced_dirs)


@pytest.mark.parametrize(
    "channel_output,folder_name,expected_filename",
    [
        ("merged", "cr_2lss", "2lss_CR_met.png"),
        ("split", "cr_2lss_ee", "2lss_CR_ee_met.png"),
        ("both", "cr_2lss", "2lss_CR_met.png"),
        ("merged-njets", "cr_2lss_1j", "2lss_CR_1j_met.png"),
        ("both-njets", "cr_2lss_ee_1j", "2lss_CR_ee_1j_met.png"),
    ],
)
@pytest.mark.parametrize("binning_mode", ("processing", "fitting"))
def test_png_stems_use_category_labels_not_folder_aliases(
    tmp_path,
    monkeypatch,
    channel_output,
    folder_name,
    expected_filename,
    binning_mode,
):
    h_met = _make_met_histogram_for_channels(
        (
            "2lss_ee_CR_1j",
            "2lss_mm_CR_1j",
            "2lss_ee_CR_2j",
            "2lss_mm_CR_2j",
        )
    )
    if binning_mode == "fitting":
        monkeypatch.setattr(
            make_cr_and_sr_plots,
            "_apply_plot_binning_view",
            lambda histogram, *_args, **_kwargs: histogram,
        )

    make_cr_and_sr_plots.run_plots_for_region(
        "CR",
        {"met": h_met},
        years=["2022"],
        save_dir_path=str(tmp_path),
        channel_output=channel_output,
        skip_syst_errs=True,
        workers=1,
        verbose=False,
        binning_mode=binning_mode,
    )

    folder = tmp_path / folder_name
    assert folder.exists()
    plot_names = {path.name for path in folder.glob("*.png")}
    expected_mode_filename = expected_filename.replace(
        ".png", f"_{binning_mode}.png"
    )
    assert expected_mode_filename in plot_names
    assert f"{folder_name}_met_{binning_mode}.png" not in plot_names


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
            {"cr_ref": ["cr_mm_only_2j"]},
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
    assert "cr_mm_only_2j" in warning_text
    assert "sr_only_ee_2j" not in warning_text


@pytest.mark.parametrize(
    "region_name,channel_dict_attr,base_key,channel_bins",
    [
        (
            "CR",
            "CR_CHAN_DICT",
            "1l_dy_tautau_CR",
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

    expected_keys_by_mode = {}
    for mode_name in expected_modes:
        expected_keys = {}
        for bin_name in channel_bins:
            suffix = make_cr_and_sr_plots._extract_njet_suffix(bin_name)
            assert suffix is not None
            expected_key = f"{base_key}_{suffix}"
            if region_name == "CR" and mode_name == "per-channel":
                token = make_cr_and_sr_plots._parse_lepflav_token_for_region(
                    bin_name,
                    region_name="CR",
                    is_lepton_flavor_in_pkl=True,
                )
                expected_key = make_cr_and_sr_plots._build_split_channel_key(
                    expected_key, token
                )
            expected_keys[expected_key] = [bin_name]
        expected_keys_by_mode[mode_name] = expected_keys

    for mode_name in expected_modes:
        transformed_map = captured_channel_maps[mode_name]
        assert base_key not in transformed_map
        expected_keys = expected_keys_by_mode[mode_name]
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

    for channel, weight in ("2los_ee_CRZ_2j", 1.0), ("2los_mm_CRZ_2j", 2.0):
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

    merged_dir_name = (
        "cr_2los_Z_2j" if channel_output.endswith("njets") else "cr_2los_Z"
    )
    merged_dir = tmp_path / merged_dir_name
    assert merged_dir.exists()

    merged_plots = {path.name for path in merged_dir.glob("*.png")}
    if channel_output.endswith("njets"):
        expected_merged = {
            "2los_CRZ_2j_j0pt_processing.png",
            "2los_CRZ_2j_met_processing.png",
        }
    else:
        expected_merged = {
            "2los_CRZ_j0pt_processing.png",
            "2los_CRZ_met_processing.png",
        }
    assert expected_merged.issubset(merged_plots)

    if channel_output.endswith("njets"):
        split_expectations = {
            "cr_2los_Z_ee_2j": {
                "2los_CRZ_ee_2j_j0pt_processing.png",
                "2los_CRZ_ee_2j_met_processing.png",
            },
            "cr_2los_Z_mm_2j": {
                "2los_CRZ_mm_2j_j0pt_processing.png",
                "2los_CRZ_mm_2j_met_processing.png",
            },
        }
        for dir_name, expected_plots in split_expectations.items():
            split_dir = tmp_path / dir_name
            assert split_dir.exists()
            split_plots = {path.name for path in split_dir.glob("*.png")}
            assert expected_plots.issubset(split_plots)
    else:
        split_expectations = {
            "cr_2los_Z_ee": {
                "2los_CRZ_ee_j0pt_processing.png",
                "2los_CRZ_ee_met_processing.png",
            },
            "cr_2los_Z_mm": {
                "2los_CRZ_mm_j0pt_processing.png",
                "2los_CRZ_mm_met_processing.png",
            },
        }
        for dir_name, expected_plots in split_expectations.items():
            split_dir = tmp_path / dir_name
            assert split_dir.exists()
            split_plots = {path.name for path in split_dir.glob("*.png")}
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


def test_combined_run2_run3_year_context_is_rejected():
    parser = make_cr_and_sr_plots.build_arg_parser()
    parsed = parser.parse_args(["-f", "input.pkl.gz", "--year", "run2", "run3"])
    assert parsed.year == ["run2", "run3"]

    hist_inputs = {"met": _make_met_histogram_for_channels(["2lss_ee_CR_1j"])}
    with pytest.raises(ValueError, match=r"Combined Run 2 \+ Run 3 plotting"):
        make_cr_and_sr_plots.build_region_context(
            "CR", hist_inputs, years=["run2", "run3"], unblind=True
        )
    run2_ctx = make_cr_and_sr_plots.build_region_context(
        "CR", hist_inputs, years=["run2"], unblind=True
    )
    run3_ctx = make_cr_and_sr_plots.build_region_context(
        "CR", hist_inputs, years=["run3"], unblind=True
    )
    physical_year_ctx = make_cr_and_sr_plots.build_region_context(
        "CR", hist_inputs, years=["2022"], unblind=True
    )

    assert run2_ctx.lumi_pair == ("138", "13")
    assert run2_ctx.lumi_components == (("138", "13"),)
    assert run2_ctx.scope_label == "Run 2"
    assert run3_ctx.lumi_pair == ("61.9", "13.6")
    assert run3_ctx.lumi_components == (("61.9", "13.6"),)
    assert run3_ctx.scope_label == "Run 3"
    assert physical_year_ctx.lumi_pair == ("7.98", "13.6")
    assert physical_year_ctx.scope_label is None
def test_parse_rebin_plot_vars_accepts_colon_and_equals():
    parsed = make_cr_and_sr_plots.parse_rebin_plot_vars("j0pt:2,l1conept=3")

    assert parsed == {"j0pt": 2, "l1conept": 3}


@pytest.mark.parametrize(
    "raw_value",
    ["j0pt", "j0pt:abc", "j0pt:1", ":2"],
)
def test_parse_rebin_plot_vars_rejects_malformed_entries(raw_value):
    with pytest.raises(ValueError):
        make_cr_and_sr_plots.parse_rebin_plot_vars(raw_value)


def test_rebin_1d_helpers_preserve_yields_and_sum_variances_with_leftovers():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    variances = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    edges = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])

    rebinned_values = make_cr_and_sr_plots.rebin_1d_values(values, 2)
    rebinned_variances = make_cr_and_sr_plots.rebin_1d_variances(variances, 2)
    rebinned_edges = make_cr_and_sr_plots.rebin_1d_edges(edges, 2)

    assert np.sum(rebinned_values) == np.sum(values)
    assert np.allclose(rebinned_values, [3.0, 7.0, 5.0])
    assert np.allclose(rebinned_variances, [5.0, 25.0, 25.0])
    assert np.allclose(rebinned_edges, [0.0, 20.0, 40.0, 50.0])


def test_resolve_rebin_plot_edges_skips_unlisted_variables():
    h_mc, _, _ = _make_simple_stacked_inputs()

    target_edges, factor, had_leftover = make_cr_and_sr_plots._resolve_rebin_plot_edges(
        "lj0pt",
        h_mc,
        {"j0pt": 2},
    )

    assert target_edges is None
    assert factor is None
    assert had_leftover is False


def test_clone_sumw2_with_rebinned_axis_accepts_suffix_axis():
    _, h_sumw2, _ = _make_negative_report_hists(
        variable="l1conept",
        sumw2_axis_name="l1conept_sumw2",
    )
    target_edges = make_cr_and_sr_plots.rebin_1d_edges([0.0, 1.0, 2.0, 3.0, 4.0], 2)

    cloned = make_cr_and_sr_plots._clone_sumw2_with_rebinned_axis(
        h_sumw2,
        "l1conept",
        target_edges,
    )

    assert np.allclose(cloned.axes["l1conept_sumw2"].edges, [0.0, 2.0, 4.0])
    values = make_cr_and_sr_plots._process_values_from_hist(
        cloned,
        ["neg_proc"],
        "l1conept_sumw2",
        np.zeros(2),
    )
    assert np.allclose(values, [34.0, 8.0])


def test_clone_sumw2_with_rebinned_axis_accepts_nominal_axis():
    _, h_sumw2, _ = _make_negative_report_hists(variable="j0pt")
    target_edges = make_cr_and_sr_plots.rebin_1d_edges([0.0, 1.0, 2.0, 3.0, 4.0], 2)

    cloned = make_cr_and_sr_plots._clone_sumw2_with_rebinned_axis(
        h_sumw2,
        "j0pt",
        target_edges,
    )

    assert np.allclose(cloned.axes["j0pt"].edges, [0.0, 2.0, 4.0])
    values = make_cr_and_sr_plots._process_values_from_hist(
        cloned,
        ["neg_proc"],
        "j0pt",
        np.zeros(2),
    )
    assert np.allclose(values, [34.0, 8.0])


def test_clone_sumw2_with_rebinned_axis_returns_none_for_none_input():
    assert (
        make_cr_and_sr_plots._clone_sumw2_with_rebinned_axis(
            None,
            "j0pt",
            [0.0, 2.0, 4.0],
        )
        is None
    )


def _make_negative_report_hists(variable="j0pt", sumw2_axis_name=None):
    sumw2_axis_name = sumw2_axis_name or variable
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    value_axis = hist.axis.Variable([0.0, 1.0, 2.0, 3.0, 4.0], name=variable)
    sumw2_axis = hist.axis.Variable(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        name=sumw2_axis_name,
    )
    h_mc = hist.Hist(process_axis, value_axis, storage=hist.storage.Double())
    h_sumw2 = hist.Hist(process_axis, sumw2_axis, storage=hist.storage.Double())
    h_data = hist.Hist(process_axis, value_axis, storage=hist.storage.Double())

    mc_values = {
        "neg_proc": [-5.0, 3.0, 2.0, 2.0],
        "pos_proc": [1.0, 1.0, 1.0, 1.0],
    }
    sumw2_values = {
        "neg_proc": [25.0, 9.0, 4.0, 4.0],
        "pos_proc": [1.0, 1.0, 1.0, 1.0],
    }
    for process_name, values in mc_values.items():
        for bin_index, value in enumerate(values):
            h_mc.fill(
                process=process_name,
                **{variable: bin_index + 0.5},
                weight=value,
            )
            h_sumw2.fill(
                process=process_name,
                **{sumw2_axis_name: bin_index + 0.5},
                weight=sumw2_values[process_name][bin_index],
            )
    for bin_index in range(4):
        h_data.fill(
            process="data",
            **{variable: bin_index + 0.5},
            weight=10.0,
        )

    return h_mc, h_sumw2, h_data


def test_negative_contribution_rows_include_requested_metrics_and_flags():
    h_mc, h_sumw2, h_data = _make_negative_report_hists()

    rows = make_cr_and_sr_plots.collect_negative_contribution_rows(
        variable="j0pt",
        channel_or_region="CR",
        category_if_available="2los_CRZ",
        stage="nominal_no_rebin",
        hist_mc=h_mc,
        hist_mc_sumw2=h_sumw2,
        hist_data=h_data,
        group_map={"Singleboson": ["neg_proc"], "Other": ["pos_proc"]},
    )

    process_rows = [row for row in rows if row["level"] == "process"]
    group_rows = [row for row in rows if row["level"] == "group"]

    assert len(process_rows) == 1
    assert len(group_rows) == 1
    process_row = process_rows[0]
    assert process_row["yield"] == -5.0
    assert process_row["sumw2"] == 25.0
    assert process_row["error"] == 5.0
    assert process_row["effective_entries"] == 1.0
    assert process_row["is_compatible_with_zero_1sigma"] is True
    assert process_row["is_single_effective_entry_like"] is True
    assert process_row["is_low_effective_entries"] is True
    assert process_row["total_mc_yield"] == -4.0
    assert process_row["data_yield"] == 10.0
    assert process_row["yield_over_total_mc"] == 1.25
    assert process_row["abs_yield_over_total_mc"] == 1.25
    assert process_row["process"] == "neg_proc"
    assert process_row["group"] == "Singleboson"
    assert group_rows[0]["process"] == ""
    assert group_rows[0]["group"] == "Singleboson"


def test_negative_report_omits_positive_bins_and_reports_post_rebin_rows():
    h_mc, h_sumw2, h_data = _make_negative_report_hists()
    target_edges = make_cr_and_sr_plots.rebin_1d_edges([0.0, 1.0, 2.0, 3.0, 4.0], 2)

    rows = make_cr_and_sr_plots.collect_negative_rows_for_plot_stage(
        variable="j0pt",
        channel_or_region="CR",
        category_if_available="2los_CRZ",
        stage="post_rebin",
        hist_mc=h_mc,
        hist_mc_sumw2=h_sumw2,
        hist_data=h_data,
        group_map={"Singleboson": ["neg_proc"], "Other": ["pos_proc"]},
        target_edges=target_edges,
    )

    process_rows = [row for row in rows if row["level"] == "process"]
    assert len(process_rows) == 1
    assert process_rows[0]["stage"] == "post_rebin"
    assert process_rows[0]["bin_index"] == 0
    assert process_rows[0]["bin_low"] == 0.0
    assert process_rows[0]["bin_high"] == 2.0
    assert process_rows[0]["yield"] == -2.0
    assert process_rows[0]["sumw2"] == 34.0
    assert all(row["yield"] < 0 for row in rows)


def test_prepare_plot_rebin_and_negative_rows_collects_pre_and_post_rows():
    h_mc, h_sumw2, h_data = _make_negative_report_hists()

    result = make_cr_and_sr_plots._prepare_plot_rebin_and_negative_rows(
        variable="j0pt",
        channel_or_region="CR",
        category_if_available="2los_CRZ",
        hist_mc=h_mc,
        hist_mc_sumw2=h_sumw2,
        hist_data=h_data,
        group_map={"Singleboson": ["neg_proc"], "Other": ["pos_proc"]},
        rebin_plot_vars={"j0pt": 2},
        negative_weight_report=True,
    )

    assert np.allclose(result["bins"], [0.0, 2.0, 4.0])
    stages = {row["stage"] for row in result["negative_rows"]}
    assert stages == {"pre_rebin", "post_rebin"}
    post_process_row = next(
        row
        for row in result["negative_rows"]
        if row["stage"] == "post_rebin" and row["level"] == "process"
    )
    assert post_process_row["yield"] == -2.0
    assert post_process_row["sumw2"] == 34.0


def test_prepare_plot_rebin_and_negative_rows_collects_nominal_when_not_rebinned():
    h_mc, h_sumw2, h_data = _make_negative_report_hists()
    base_edges = [0.0, 1.0, 2.0, 3.0, 4.0]

    result = make_cr_and_sr_plots._prepare_plot_rebin_and_negative_rows(
        variable="j0pt",
        channel_or_region="CR",
        category_if_available="2los_CRZ",
        hist_mc=h_mc,
        hist_mc_sumw2=h_sumw2,
        hist_data=h_data,
        group_map={"Singleboson": ["neg_proc"], "Other": ["pos_proc"]},
        rebin_plot_vars={},
        base_edges=base_edges,
        negative_weight_report=True,
    )

    assert result["target_edges"] is None
    assert result["bins"] is base_edges
    assert {row["stage"] for row in result["negative_rows"]} == {"nominal_no_rebin"}


def test_prepare_plot_rebin_and_negative_rows_skips_rows_when_report_disabled():
    h_mc, h_sumw2, h_data = _make_negative_report_hists()

    result = make_cr_and_sr_plots._prepare_plot_rebin_and_negative_rows(
        variable="j0pt",
        channel_or_region="CR",
        category_if_available="2los_CRZ",
        hist_mc=h_mc,
        hist_mc_sumw2=h_sumw2,
        hist_data=h_data,
        group_map={"Singleboson": ["neg_proc"], "Other": ["pos_proc"]},
        rebin_plot_vars={"j0pt": 2},
        negative_weight_report=False,
    )

    assert np.allclose(result["bins"], [0.0, 2.0, 4.0])
    assert result["negative_rows"] == []


def test_negative_report_post_rebin_resolves_sumw2_suffix_axis():
    h_mc, h_sumw2, h_data = _make_negative_report_hists(
        variable="l1conept",
        sumw2_axis_name="l1conept_sumw2",
    )
    target_edges = make_cr_and_sr_plots.rebin_1d_edges([0.0, 1.0, 2.0, 3.0, 4.0], 2)

    rows = make_cr_and_sr_plots.collect_negative_rows_for_plot_stage(
        variable="l1conept",
        channel_or_region="CR",
        category_if_available="2los_CRZ",
        stage="post_rebin",
        hist_mc=h_mc,
        hist_mc_sumw2=h_sumw2,
        hist_data=h_data,
        group_map={"Singleboson": ["neg_proc"], "Other": ["pos_proc"]},
        target_edges=target_edges,
    )

    process_row = next(row for row in rows if row["level"] == "process")
    assert process_row["variable"] == "l1conept"
    assert process_row["stage"] == "post_rebin"
    assert process_row["bin_low"] == 0.0
    assert process_row["bin_high"] == 2.0
    assert process_row["yield"] == -2.0
    assert process_row["sumw2"] == 34.0
    assert process_row["effective_entries"] == pytest.approx(4.0 / 34.0)


def test_negative_report_post_rebin_without_sumw2_does_not_crash():
    h_mc, _, h_data = _make_negative_report_hists(variable="l1conept")
    target_edges = make_cr_and_sr_plots.rebin_1d_edges([0.0, 1.0, 2.0, 3.0, 4.0], 2)

    rows = make_cr_and_sr_plots.collect_negative_rows_for_plot_stage(
        variable="l1conept",
        channel_or_region="CR",
        category_if_available="2los_CRZ",
        stage="post_rebin",
        hist_mc=h_mc,
        hist_mc_sumw2=None,
        hist_data=h_data,
        group_map={"Singleboson": ["neg_proc"], "Other": ["pos_proc"]},
        target_edges=target_edges,
    )

    process_row = next(row for row in rows if row["level"] == "process")
    assert process_row["yield"] == -2.0
    assert np.isnan(process_row["sumw2"])
    assert np.isnan(process_row["effective_entries"])


def test_negative_report_ambiguous_sumw2_dense_axis_lists_available_axes():
    h_mc, _, h_data = _make_negative_report_hists(variable="l1conept")
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    h_sumw2 = hist.Hist(
        process_axis,
        hist.axis.Variable([0.0, 1.0, 2.0, 3.0, 4.0], name="alt_a"),
        hist.axis.Variable([0.0, 1.0, 2.0, 3.0, 4.0], name="alt_b"),
        storage=hist.storage.Double(),
    )
    target_edges = make_cr_and_sr_plots.rebin_1d_edges([0.0, 1.0, 2.0, 3.0, 4.0], 2)

    with pytest.raises(
        ValueError,
        match=(
            "Cannot resolve sumw2 dense axis for variable 'l1conept'.*"
            "Available dense axes: alt_a, alt_b"
        ),
    ):
        make_cr_and_sr_plots.collect_negative_rows_for_plot_stage(
            variable="l1conept",
            channel_or_region="CR",
            category_if_available="2los_CRZ",
            stage="post_rebin",
            hist_mc=h_mc,
            hist_mc_sumw2=h_sumw2,
            hist_data=h_data,
            group_map={"Singleboson": ["neg_proc"], "Other": ["pos_proc"]},
            target_edges=target_edges,
        )


def test_negative_report_missing_sumw2_dense_axis_lists_available_axes():
    h_mc, _, h_data = _make_negative_report_hists(variable="l1conept")
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    h_sumw2 = hist.Hist(process_axis, storage=hist.storage.Double())
    target_edges = make_cr_and_sr_plots.rebin_1d_edges([0.0, 1.0, 2.0, 3.0, 4.0], 2)

    with pytest.raises(
        ValueError,
        match=(
            "Cannot resolve sumw2 dense axis for variable 'l1conept'.*"
            "Available dense axes: <none>"
        ),
    ):
        make_cr_and_sr_plots.collect_negative_rows_for_plot_stage(
            variable="l1conept",
            channel_or_region="CR",
            category_if_available="2los_CRZ",
            stage="post_rebin",
            hist_mc=h_mc,
            hist_mc_sumw2=h_sumw2,
            hist_data=h_data,
            group_map={"Singleboson": ["neg_proc"], "Other": ["pos_proc"]},
            target_edges=target_edges,
        )


def test_write_negative_weight_report_creates_csv_and_markdown(tmp_path):
    h_mc, h_sumw2, h_data = _make_negative_report_hists()
    rows = make_cr_and_sr_plots.collect_negative_contribution_rows(
        variable="j0pt",
        channel_or_region="CR",
        category_if_available="2los_CRZ",
        stage="nominal_no_rebin",
        hist_mc=h_mc,
        hist_mc_sumw2=h_sumw2,
        hist_data=h_data,
        group_map={"Singleboson": ["neg_proc"], "Other": ["pos_proc"]},
    )

    paths = make_cr_and_sr_plots.write_negative_weight_report(
        rows,
        tmp_path,
        binning_mode="processing",
    )

    csv_path = paths["csv"]
    md_path = paths["markdown"]
    assert os.path.basename(csv_path) == "negative_weight_contribution_report_processing.csv"
    assert os.path.basename(md_path) == "negative_weight_contribution_summary_processing.md"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        written_rows = list(reader)
    assert reader.fieldnames == make_cr_and_sr_plots.NEGATIVE_WEIGHT_REPORT_COLUMNS
    assert len(written_rows) == len(rows)
    assert written_rows[0]["effective_entries"] == "1"

    summary_text = open(md_path).read()
    assert "total negative process bins: 1" in summary_text
    assert "total negative group bins: 1" in summary_text
    assert "Top Negative Process Contributions" in summary_text


def test_binning_mode_output_paths_are_collision_free(tmp_path):
    logical_png = tmp_path / "category" / "stem.png"
    logical_extensionless = tmp_path / "category" / "stem"

    fitting_png = make_cr_and_sr_plots._mode_bearing_output_path(
        logical_png,
        "fitting",
    )
    processing_png = make_cr_and_sr_plots._mode_bearing_output_path(
        logical_png,
        "processing",
    )
    fitting_extensionless = make_cr_and_sr_plots._mode_bearing_output_path(
        logical_extensionless,
        "fitting",
    )

    assert fitting_png == os.fspath(tmp_path / "category" / "stem_fitting.png")
    assert processing_png == os.fspath(
        tmp_path / "category" / "stem_processing.png"
    )
    assert fitting_extensionless == os.fspath(
        tmp_path / "category" / "stem_fitting"
    )
    assert os.path.dirname(fitting_png) == os.path.dirname(processing_png)
    assert fitting_png != processing_png
    assert os.fspath(logical_png) not in {fitting_png, processing_png}

    with pytest.raises(ValueError, match="Unsupported binning mode"):
        make_cr_and_sr_plots._mode_bearing_output_path(logical_png, "unknown")


def test_save_figure_formats_uses_one_mode_bearing_root(tmp_path):
    class RecordingFigure:
        def __init__(self):
            self.calls = []

        def savefig(self, path, **kwargs):
            self.calls.append((path, kwargs))

    figure = RecordingFigure()
    output_paths = make_cr_and_sr_plots._save_figure_formats(
        figure,
        tmp_path / "category" / "stem",
        "fitting",
        ("png", "pdf", "svg", "png"),
        bbox_inches="tight",
    )

    expected_paths = tuple(
        str(tmp_path / "category" / f"stem_fitting.{suffix}")
        for suffix in ("png", "pdf", "svg")
    )
    assert output_paths == expected_paths
    assert [call[0] for call in figure.calls] == list(expected_paths)
    assert [call[1]["format"] for call in figure.calls] == ["png", "pdf", "svg"]
    assert all(call[1]["bbox_inches"] == "tight" for call in figure.calls)


def test_negative_weight_reports_for_binning_modes_coexist(tmp_path):
    fitting_paths = make_cr_and_sr_plots.write_negative_weight_report(
        [],
        tmp_path,
        binning_mode="fitting",
    )
    processing_paths = make_cr_and_sr_plots.write_negative_weight_report(
        [],
        tmp_path,
        binning_mode="processing",
    )

    assert set(fitting_paths.values()).isdisjoint(processing_paths.values())
    assert {path.name for path in tmp_path.iterdir()} == {
        "negative_weight_contribution_report_fitting.csv",
        "negative_weight_contribution_report_processing.csv",
        "negative_weight_contribution_summary_fitting.md",
        "negative_weight_contribution_summary_processing.md",
    }


def test_plotter_parser_exposes_rebin_and_negative_report_switches():
    parser = make_cr_and_sr_plots.build_arg_parser()

    args = parser.parse_args(
        [
            "-f",
            "input.pkl.gz",
            "--rebin-plot-vars",
            "j0pt:2",
            "--no-negative-weight-report",
        ]
    )

    assert args.rebin_plot_vars == "j0pt:2"
    assert args.negative_weight_report is False


def _make_sr_ttbar_display_inputs(*, negative=False):
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    jet_axis = hist.axis.Regular(2, 3.0, 5.0, name="njets")
    h_mc = hist.Hist(process_axis, jet_axis, storage=hist.storage.Double())
    h_sumw2 = hist.Hist(process_axis, jet_axis, storage=hist.storage.Double())
    h_data = hist.Hist(process_axis, jet_axis, storage=hist.storage.Double())
    samples = {
        "TTTo2L2Nu_centralUL16": [4.0, -4.0 if negative else 1.0, 3.0],
        "TTToSemiLeptonic_centralUL17": [2.0, 1.0, 2.0],
        "TTZToLL_M1to10_centralUL18": [1.0, 2.0, 0.0],
        "ttllJet_privateUL16APV": [3.0, 2.0, 1.0],
        "other": [5.0, 1.0, 1.0],
    }
    for process, values in samples.items():
        for jet, value in zip((3.5, 4.5, 5.5), values):
            h_mc.fill(process=process, njets=jet, weight=value)
            h_sumw2.fill(process=process, njets=jet, weight=abs(value) + 1.0)
    for jet, value in zip((3.5, 4.5, 5.5), (14.0, 6.0, 7.0)):
        h_data.fill(process="data", njets=jet, weight=value)
    group_map = {
        "ttll": [name for name in samples if name != "other"],
        "Other": ["other"],
    }
    return h_mc, h_sumw2, h_data, group_map


def test_run3_fully_hadronic_display_preserves_inputs_and_resolves_legend_colors():
    eras = ("2022", "2022EE", "2023", "2023BPix")
    fully_hadronic = {f"TTto4Q_central{era}" for era in eras}
    samples = {
        **{name: [2.0, -0.25 if name.endswith("2022") else 1.0, 3.0]
           for name in fully_hadronic},
        "TTto2L2Nu_central2022": [5.0, 2.0, 1.0],
        "TTtoLNu2Q_central2023": [4.0, 1.0, 2.0],
        "ttll_private2022": [3.0, 1.0, 2.0],
        "DYJetsToLL_MLL-50_central2022": [3.0, 2.0, 1.0],
        "TTGJets_central2023": [1.0, 1.0, 1.0],
        "ZGToLLG_central2022": [1.0, 2.0, 1.0],
        "ttH_central2022": [1.0, 1.0, 2.0],
        "ST_top_t-channel_central2022": [1.0, 1.0, 1.0],
        "tllq_central2023": [1.0, 1.0, 1.0],
        "tttt_central2022": [1.0, 1.0, 1.0],
        "WJetsToLNu_central2022": [1.0, 1.0, 1.0],
    }
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    jet_axis = hist.axis.Regular(3, 3.0, 6.0, name="njets")
    h_mc = hist.Hist(process_axis, jet_axis, storage=hist.storage.Double())
    h_sumw2 = hist.Hist(process_axis, jet_axis, storage=hist.storage.Double())
    h_data = hist.Hist(process_axis, jet_axis, storage=hist.storage.Double())
    for process, values in samples.items():
        for jet, value in zip((3.5, 4.5, 5.5), values):
            h_mc.fill(process=process, njets=jet, weight=value)
            h_sumw2.fill(process=process, njets=jet, weight=abs(value) + 1.0)
    for jet, value in zip((3.5, 4.5, 5.5), (25.0, 15.0, 18.0)):
        h_data.fill(process="data", njets=jet, weight=value)

    group_map = make_cr_and_sr_plots.populate_group_map(
        samples, make_cr_and_sr_plots.SR_GRP_PATTERNS
    )
    ownership = Counter(member for members in group_map.values() for member in members)
    assert ownership == Counter(samples.keys())
    assert set(group_map["ttll"]) == fully_hadronic | {
        "TTto2L2Nu_central2022", "TTtoLNu2Q_central2023", "ttll_private2022"
    }
    assert all(
        make_cr_and_sr_plots.populate_group_map(
            [name], make_cr_and_sr_plots.SR_GRP_PATTERNS
        )["ttll"] == [name]
        for name in fully_hadronic
    )
    assert not make_cr_and_sr_plots.populate_group_map(
        ["TTto4Q_central2022_extra"], make_cr_and_sr_plots.SR_GRP_PATTERNS
    )["ttll"]

    before_mc = h_mc.values(flow=True).copy()
    before_sumw2 = h_sumw2.values(flow=True).copy()
    before_data = h_data.values(flow=True).copy()
    total = h_mc[{"process": sum}].values(flow=True)[1:]
    render_kwargs = dict(
        h_mc=h_mc, h_data=h_data, h_mc_sumw2=h_sumw2,
        unit_norm_bool=False, var="njets", group=group_map,
        unblind=True, uncertainty_mode="total", syst_err="total",
        err_p_syst=total + 2.0, err_m_syst=total - 2.0,
        err_ratio_p_syst=np.ones_like(total),
        err_ratio_m_syst=np.ones_like(total),
    )
    original = make_cr_and_sr_plots.make_region_stacked_ratio_fig(**render_kwargs)
    split = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        **render_kwargs, sr_ttbar_display_years=eras
    )
    try:
        view = split._topeft_sr_ttbar_display_view
        assert not view["fallback"]
        assert set(view["fully_hadronic_members"]) == fully_hadronic
        assert set(view["ttbar_members"]) == {
            "TTto2L2Nu_central2022", "TTtoLNu2Q_central2023"
        }
        assert view["remainder_members"] == ("ttll_private2022",)
        assert Counter(
            member for members in view["draw_group"].values() for member in members
        ) == Counter(samples.keys())
        np.testing.assert_allclose(
            view["ttbar_values"] + view["fully_hadronic_values"]
            + view["remainder_values"],
            original._topeft_plot_view["raw_grouped_centrals"]["ttll"],
        )
        for key in ("displayed_total", "total_variance"):
            np.testing.assert_allclose(
                split._topeft_plot_view[key], original._topeft_plot_view[key]
            )
        for key in original._topeft_uncertainty_bands:
            if key != "main_band_handles":
                np.testing.assert_allclose(
                    split._topeft_uncertainty_bands[key],
                    original._topeft_uncertainty_bands[key], equal_nan=True
                )
        np.testing.assert_array_equal(h_mc.values(flow=True), before_mc)
        np.testing.assert_array_equal(h_sumw2.values(flow=True), before_sumw2)
        np.testing.assert_array_equal(h_data.values(flow=True), before_data)

        legend = split.legends[0]
        legend_colors = {
            label.get_text(): handle.get_facecolor()
            for handle, label in zip(legend.legend_handles, legend.get_texts())
            if hasattr(handle, "get_facecolor")
        }
        cr = make_cr_and_sr_plots.CR_GROUP_INFO
        sr = make_cr_and_sr_plots.SR_GROUP_INFO
        assert to_rgba(cr["DY"]["color"]) == to_rgba(sr["DY"]["color"])
        assert to_rgba(cr["Ttbar"]["color"]) != to_rgba(cr["DY"]["color"])
        assert to_rgba(legend_colors["ttbar"]) == to_rgba(cr["Ttbar"]["color"])
        assert to_rgba(legend_colors["DY"]) == to_rgba(cr["DY"]["color"])
        split_config = sr["ttll"]["display_split"]
        assert to_rgba(legend_colors[split_config["fully_hadronic_label"]]) == to_rgba(
            split_config["fully_hadronic_color"]
        )
        assert to_rgba(legend_colors[split_config["remainder_label"]]) == to_rgba(
            sr["ttll"]["color"]
        )
        for first, second in (("DY", split_config["remainder_label"]),
                              ("ZGamma", "ttH"), ("Single top", "tllq"),
                              ("tttt", "Singleboson")):
            assert to_rgba(legend_colors[first]) != to_rgba(legend_colors[second])
        assert all(
            to_rgba(legend_colors[split_config["fully_hadronic_label"]]) != to_rgba(color)
            for label, color in legend_colors.items()
            if view["draw_group"].get(label)
            and label != split_config["fully_hadronic_label"]
        )
        displayed_mc_colors = [
            to_rgba(color) for label, color in legend_colors.items()
            if view["draw_group"].get(label)
        ]
        assert len(displayed_mc_colors) == len(set(displayed_mc_colors))
        dark_green = to_rgb(legend_colors["ttbar"])
        light_green = to_rgb(legend_colors[split_config["fully_hadronic_label"]])
        assert light_green[1] > dark_green[1]
        assert light_green[1] >= light_green[0] and light_green[1] >= light_green[2]
    finally:
        make_cr_and_sr_plots.plt.close(original)
        make_cr_and_sr_plots.plt.close(split)


def test_run3_fully_hadronic_negative_bucket_keeps_original_stack():
    samples = {
        "TTto4Q_central2022": -3.0,
        "TTto2L2Nu_central2022": 4.0,
        "ttll_private2022": 2.0,
    }
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    jet_axis = hist.axis.Regular(1, 3.0, 4.0, name="njets")
    h_mc = hist.Hist(process_axis, jet_axis, storage=hist.storage.Double())
    h_data = hist.Hist(process_axis, jet_axis, storage=hist.storage.Double())
    for process, value in samples.items():
        h_mc.fill(process=process, njets=3.5, weight=value)
    h_data.fill(process="data", njets=3.5, weight=3.0)
    group_map = make_cr_and_sr_plots.populate_group_map(
        samples, make_cr_and_sr_plots.SR_GRP_PATTERNS
    )
    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc=h_mc, h_data=h_data, unit_norm_bool=False, var="njets",
        group=group_map, unblind=False, sr_ttbar_display_years=["2022"],
    )
    try:
        view = fig._topeft_sr_ttbar_display_view
        assert view["fallback"]
        assert list(view["draw_group"]["ttll"]) == list(samples)
        np.testing.assert_allclose(view["original_visible_ttll"], [3.0, 0.0])
        np.testing.assert_allclose(view["split_visible_ttll"], [6.0, 0.0])
        np.testing.assert_allclose(fig._topeft_plot_view["displayed_total"], [3.0, 0.0])
        assert "ttbar (fully hadronic)" in {
            label.get_text() for legend in fig.legends for label in legend.get_texts()
        }
    finally:
        make_cr_and_sr_plots.plt.close(fig)


def test_sr_ttbar_metadata_partition_covers_census_identities():
    run2_eras = ("UL16", "UL16APV", "UL17", "UL18")
    run3_eras = ("2022", "2022EE", "2023", "2023BPix")
    run2 = [f"{stem}{era}" for era in run2_eras for stem in (
        "TTTo2L2Nu_central", "TTToSemiLeptonic_central",
        "TTZToLL_M1to10_central", "ttllJet_private",
    )]
    run3 = [f"{stem}{era}" for era in run3_eras for stem in (
        "TTto2L2Nu_central", "TTtoLNu2Q_central", "ttll_private",
    )]
    assert len(run2 + run3) == 28
    for members, years, expected_ttbar in (
        (run2, ["2016", "2016APV", "2017", "2018"], 8),
        (run3, ["2022", "2022EE", "2023", "2023BPix"], 8),
    ):
        process_axis = hist.axis.StrCategory(members, name="process")
        jet_axis = hist.axis.Regular(1, 3, 4, name="njets")
        source = hist.Hist(process_axis, jet_axis)
        for member in members:
            source.fill(process=member, njets=3.5)
        view = make_cr_and_sr_plots._resolve_sr_ttbar_display_view(
            source, {"ttll": members}, years,
        )
        assert len(view["ttbar_members"]) == expected_ttbar
        assert set(view["ttbar_members"]).isdisjoint(view["remainder_members"])
        assert set(view["ttbar_members"]) | set(view["remainder_members"]) == set(members)


def test_sr_ttbar_metadata_partition_is_exact_and_fail_closed():
    group_map = {
        "ttll": [
            "TTTo2L2Nu_centralUL16APV",
            "TTToSemiLeptonic_centralUL18",
            "TTZToLL_M1to10_centralUL17",
            "ttllJet_privateUL16",
        ]
    }
    source = hist.Hist(
        hist.axis.StrCategory(group_map["ttll"], name="process"),
        hist.axis.Regular(1, 3, 4, name="njets"),
    )
    for member in group_map["ttll"]:
        source.fill(process=member, njets=3.5)
    view = make_cr_and_sr_plots._resolve_sr_ttbar_display_view(
        source, group_map, ["2016", "2016APV", "2017", "2018"]
    )
    ttbar, remainder = view["ttbar_members"], view["remainder_members"]
    assert len(ttbar) == 2
    assert remainder == (
        "TTZToLL_M1to10_centralUL17", "ttllJet_privateUL16"
    )
    assert set(ttbar).isdisjoint(remainder)
    assert set(ttbar) | set(remainder) == set(group_map["ttll"])

    run3_map = {
        "ttll": ["TTto2L2Nu_central2022EE", "TTtoLNu2Q_central2023BPix",
                 "ttll_private2022", "ttll_private2023"]
    }
    run3_source = hist.Hist(
        hist.axis.StrCategory(run3_map["ttll"], name="process"),
        hist.axis.Regular(1, 3, 4, name="njets"),
    )
    for member in run3_map["ttll"]:
        run3_source.fill(process=member, njets=3.5)
    run3_view = make_cr_and_sr_plots._resolve_sr_ttbar_display_view(
        run3_source, run3_map, ["2022", "2022EE", "2023", "2023BPix"]
    )
    run3_ttbar, run3_remainder = run3_view["ttbar_members"], run3_view["remainder_members"]
    assert len(run3_ttbar) == len(run3_remainder) == 2

    for invalid in (
        ["ttllJet_privateUL16"],
        ["TTTo2L2Nu_centralUL16", "TTTo2L2Nu_centralUL16"],
        ["TTTo2L2Nu_centralUL16", "TTTo2L2Nu_centralUL16_extra",
         "ttllJet_privateUL16"],
    ):
        with pytest.raises(ValueError):
            make_cr_and_sr_plots._resolve_sr_ttbar_display_view(
                source, {"ttll": invalid}, ["2016"]
            )
    with pytest.raises(ValueError, match="selected-era"):
        make_cr_and_sr_plots._resolve_sr_ttbar_display_view(
            run3_source, run3_map, ["2016"]
        )


def test_sr_ttbar_yaml_absence_and_empty_category_leave_legacy_draw(monkeypatch):
    h_mc, h_sumw2, h_data, group_map = _make_sr_ttbar_display_inputs()
    original_info = make_cr_and_sr_plots.SR_GROUP_INFO
    without_split = dict(original_info)
    without_split["ttll"] = {
        key: value for key, value in original_info["ttll"].items()
        if key != "display_split"
    }
    monkeypatch.setattr(make_cr_and_sr_plots, "SR_GROUP_INFO", without_split)
    fig = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        h_mc, h_data, False, var="njets", group=group_map,
        h_mc_sumw2=h_sumw2, unblind=True, uncertainty_mode="stat",
    )
    try:
        assert not hasattr(fig, "_topeft_sr_ttbar_display_view")
        assert "ttll" in fig._topeft_plot_view["raw_grouped_centrals"]
    finally:
        make_cr_and_sr_plots.plt.close(fig)

    monkeypatch.setattr(make_cr_and_sr_plots, "SR_GROUP_INFO", original_info)
    assert make_cr_and_sr_plots._resolve_sr_ttbar_display_view(
        h_mc, {"Other": ["other"]}, ["2016", "2016APV", "2017", "2018"]
    ) is None
    empty = hist.Hist(
        hist.axis.StrCategory(group_map["ttll"], name="process"),
        hist.axis.Regular(2, 3, 5, name="njets"),
    )
    assert make_cr_and_sr_plots._resolve_sr_ttbar_display_view(
        empty, {"ttll": group_map["ttll"]},
        ["2016", "2016APV", "2017", "2018"],
    ) is None
    absent = hist.Hist(
        hist.axis.StrCategory(["other"], name="process"),
        hist.axis.Regular(2, 3, 5, name="njets"),
    )
    absent.fill(process="other", njets=3.5)
    assert make_cr_and_sr_plots._resolve_sr_ttbar_display_view(
        absent, group_map, ["2016", "2016APV", "2017", "2018"]
    ) is None
    partial = hist.Hist(
        hist.axis.StrCategory([group_map["ttll"][0]], name="process"),
        hist.axis.Regular(2, 3, 5, name="njets"),
    )
    partial.fill(process=group_map["ttll"][0], njets=3.5)
    with pytest.raises(ValueError, match="missing original members"):
        make_cr_and_sr_plots._resolve_sr_ttbar_display_view(
            partial, group_map, ["2016", "2016APV", "2017", "2018"]
        )
    with pytest.raises(ValueError, match="overlaps"):
        make_cr_and_sr_plots._resolve_sr_ttbar_display_view(
            h_mc, {**group_map, "Other": ["other", group_map["ttll"][0]]},
            ["2016", "2016APV", "2017", "2018"],
        )


@pytest.mark.parametrize("invalid", [None, {}, {"ttbar_group": "Unknown", "ttbar_label": "t", "remainder_label": "r"},
                                    {"ttbar_group": "Ttbar", "ttbar_label": "same", "remainder_label": "same"}])
def test_sr_ttbar_malformed_present_yaml_fails_explicitly(monkeypatch, invalid):
    h_mc, _, _, group_map = _make_sr_ttbar_display_inputs()
    info = dict(make_cr_and_sr_plots.SR_GROUP_INFO)
    info["ttll"] = dict(info["ttll"], display_split=invalid)
    monkeypatch.setattr(make_cr_and_sr_plots, "SR_GROUP_INFO", info)
    with pytest.raises(ValueError, match="display_split"):
        make_cr_and_sr_plots._resolve_sr_ttbar_display_view(
            h_mc, group_map, ["2016", "2016APV", "2017", "2018"]
        )


@pytest.mark.parametrize("negative", [False, True])
def test_sr_ttbar_draw_view_preserves_signed_total_variance_and_full_band(negative):
    h_mc, h_sumw2, h_data, group_map = _make_sr_ttbar_display_inputs(
        negative=negative
    )
    before_mc = h_mc.values(flow=True).copy()
    before_variance = h_sumw2.values(flow=True).copy()
    signed_total = h_mc[{"process": sum}].values(flow=True)[1:]
    render_kwargs = dict(
        h_mc=h_mc, h_data=h_data, h_mc_sumw2=h_sumw2,
        unit_norm_bool=False, var="njets", group=group_map,
        unblind=True, uncertainty_mode="total", syst_err="total",
        err_p_syst=signed_total + 2.0,
        err_m_syst=signed_total - 3.0,
        err_ratio_p_syst=np.ones_like(signed_total),
        err_ratio_m_syst=np.ones_like(signed_total),
    )
    original = make_cr_and_sr_plots.make_region_stacked_ratio_fig(**render_kwargs)
    split = make_cr_and_sr_plots.make_region_stacked_ratio_fig(
        **render_kwargs,
        sr_ttbar_display_years=["2016", "2016APV", "2017", "2018"],
    )
    try:
        view = split._topeft_sr_ttbar_display_view
        np.testing.assert_allclose(
            view["ttbar_values"] + view["remainder_values"],
            original._topeft_plot_view["raw_grouped_centrals"]["ttll"],
        )
        np.testing.assert_allclose(
            split._topeft_plot_view["displayed_total"],
            original._topeft_plot_view["displayed_total"],
        )
        np.testing.assert_allclose(
            split._topeft_plot_summary["ratio_denominator"],
            original._topeft_plot_summary["ratio_denominator"],
        )
        for key in original._topeft_uncertainty_bands:
            if key == "main_band_handles":
                continue
            np.testing.assert_allclose(
                split._topeft_uncertainty_bands[key],
                original._topeft_uncertainty_bands[key], equal_nan=True,
            )
        np.testing.assert_allclose(
            split._topeft_plot_view["total_variance"],
            original._topeft_plot_view["total_variance"],
        )
        np.testing.assert_allclose(
            split._topeft_data_intervals["central_counts"],
            original._topeft_data_intervals["central_counts"],
        )
        np.testing.assert_allclose(
            split._topeft_ratio_values, original._topeft_ratio_values,
        )
        np.testing.assert_allclose(
            sum(split._topeft_plot_view["raw_grouped_centrals"].values()),
            sum(original._topeft_plot_view["raw_grouped_centrals"].values()),
        )
        assert view["fallback"] is negative
        assert len(split.axes) == len(original.axes) == 2
        assert [axis.get_title() for axis in split.axes] == ["", ""]
        assert "sr_ttbar_companion_panel" not in split._topeft_plot_summary
        labels = [text.get_text() for legend in split.legends for text in legend.get_texts()]
        assert "ttbar" in labels
        assert "ttll (ttbar excluded)" in labels
        assert "ttll" not in labels
        assert "ttbar (existing ttll)" not in labels
        assert "other former-ttll members" not in labels
        if negative:
            assert list(split._topeft_plot_view["raw_grouped_centrals"]) == list(
                original._topeft_plot_view["raw_grouped_centrals"]
            )
            assert view["split_visible_ttll"][1] > view["original_visible_ttll"][1]
            assert {"ttbar", "ttll (ttbar excluded)"}.issubset(
                {artist.get_label() for artist in split.axes[0].patches}
            )
            assert any(text.get_text() == "-3" for text in split.axes[0].texts)
            np.testing.assert_allclose(split.axes[0].get_ylim(), original.axes[0].get_ylim())
        else:
            assert "ttbar" in split._topeft_plot_view[
                "raw_grouped_centrals"
            ]
        np.testing.assert_array_equal(h_mc.values(flow=True), before_mc)
        np.testing.assert_array_equal(h_sumw2.values(flow=True), before_variance)
    finally:
        make_cr_and_sr_plots.plt.close(original)
        make_cr_and_sr_plots.plt.close(split)


def test_sr_ttbar_retired_cli_option_is_not_recognized():
    parser = make_cr_and_sr_plots.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--sr-ttbar-display-split"])


def test_sr_ttbar_yaml_does_not_limit_sr_plot_categories(monkeypatch):
    category_calls = []
    region_ctx = SimpleNamespace(
        name="SR", dict_of_hists={"njets": object()},
        apply_category_skips=False, unblind_default=True,
        variable_label="Variable",
    )
    payload = {
        "channel_dict": {
            "2los_onZ_1tau": ["2los_onZ_1tau"],
            "unrelated_sr_category": ["unrelated_sr_category"],
        },
        "channel_transformations": [],
        "is_sparse2d": False,
        "hist_mc": object(), "hist_data": object(),
        "hist_mc_sumw2_orig": object(),
        "_global_channel_coverage_validated": True,
    }
    monkeypatch.setattr(
        make_cr_and_sr_plots, "_prepare_variable_payload",
        lambda *args, **kwargs: payload,
    )

    def record_category(var_name, hist_cat, *args, **kwargs):
        category_calls.append((var_name, hist_cat))
        return 0, 1, set(), []

    monkeypatch.setattr(
        make_cr_and_sr_plots, "_render_variable_category", record_category
    )
    make_cr_and_sr_plots.produce_region_plots(
        region_ctx, None, ["njets"], "total", False, False,
        unblind=True, workers=1,
    )
    assert category_calls == [
        ("njets", "2los_onZ_1tau"),
        ("njets", "unrelated_sr_category"),
    ]


def test_sr_ttbar_yaml_activates_merged_and_per_channel_1d(tmp_path, monkeypatch):
    source = make_cr_and_sr_plots.SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name="njets"),
    )
    for process, weight in (
        ("TTto2L2Nu_central2022", 3.0),
        ("ttll_private2022", 2.0),
        ("data2022", 4.0),
    ):
        source.fill(
            process=process, channel="2lss_p", systematic="nominal",
            njets=0.5, weight=weight,
        )
    captured = []
    original_render = make_cr_and_sr_plots.make_region_stacked_ratio_fig

    def capture_render(*args, **kwargs):
        fig = original_render(*args, **kwargs)
        if fig is not None:
            captured.append((
                kwargs.get("sr_ttbar_display_years"),
                getattr(fig, "_topeft_sr_ttbar_display_view", None),
            ))
        return fig

    monkeypatch.setattr(
        make_cr_and_sr_plots, "make_region_stacked_ratio_fig", capture_render
    )
    make_cr_and_sr_plots.run_plots_for_region(
        "SR", {"njets": source}, years=["2022"],
        save_dir_path=str(tmp_path), channel_output="both",
        skip_syst_errs=True, workers=1, unblind=True,
        variables=["njets"],
    )
    assert len(captured) >= 2
    assert all(years == ("2022",) and view is not None for years, view in captured)
    assert all(view["ttbar_members"] == ("TTto2L2Nu_central2022",) for _, view in captured)
