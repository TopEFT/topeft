from __future__ import annotations

import ast
from pathlib import Path
import pickle
import shlex

import cloudpickle
import hist
import numpy as np
import pytest

from analysis.topeft_run2.analysis_processor import AnalysisProcessor
from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules.axes import info as axes_info
from topeft.modules.axis_binning import (
    processing_edges,
    rebin_histogram,
    resolve_axis_edges,
)
from topeft.modules.nominal_schema import scalar_nominal_key


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CARD_MATRIX_PATH = REPOSITORY_ROOT / "run_make_cards_run3_yawen_matrix.sh"


def _maintained_fitting_matrix():
    mapping = {}
    for raw_line in CARD_MATRIX_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("run_job "):
            continue
        fields = shlex.split(line)
        family = fields[5]
        category_patterns = fields[7:]
        assert len(category_patterns) == int(fields[6])
        for pattern in category_patterns:
            category = pattern.removeprefix("^")
            category = (
                category[:-2]
                if category.endswith("\\$")
                else category.removesuffix("$")
            )
            assert category not in mapping
            mapping[category] = family
    return mapping


def test_run_analysis_flag_reaches_processor_constructor_and_options_owner():
    source = (REPOSITORY_ROOT / "analysis/topeft_run2/run_analysis.py").read_text()
    tree = ast.parse(source)

    flag_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--record-raw-count"
    ]
    assert len(flag_calls) == 1
    assert any(
        keyword.arg == "action"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "store_true"
        for keyword in flag_calls[0].keywords
    )

    processor_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "AnalysisProcessor"
    ]
    assert len(processor_calls) == 1
    propagated = [
        keyword
        for keyword in processor_calls[0].keywords
        if keyword.arg == "record_raw_count"
    ]
    assert len(propagated) == 1
    assert isinstance(propagated[0].value, ast.Name)
    assert propagated[0].value.id == "record_raw_count"
    assert 'ops.pop("record_raw_count", record_raw_count)' in source


def test_processor_enables_only_authoritative_fitting_family_nominal_objects():
    fitting_families = {
        family for family, axis_config in axes_info.items() if "fitting" in axis_config
    }
    assert fitting_families == {"ptz", "ptll", "lj0pt", "ptz_wtau", "lt"}

    disabled = AnalysisProcessor(
        samples={},
        hist_lst=["ptz", "njets"],
        fill_sumw2_hist=True,
        record_raw_count=False,
    )
    enabled = AnalysisProcessor(
        samples={},
        hist_lst=["ptz", "njets"],
        fill_sumw2_hist=True,
        record_raw_count=True,
    )

    disabled_ptz = disabled.accumulator[scalar_nominal_key("ptz")]
    enabled_ptz = enabled.accumulator[scalar_nominal_key("ptz")]
    enabled_njets = enabled.accumulator[scalar_nominal_key("njets")]
    enabled_sumw2 = enabled.accumulator["ptz_sumw2"]
    assert disabled_ptz.track_raw_counts is False
    assert not hasattr(disabled_ptz, "_raw_counts")
    assert enabled_ptz.track_raw_counts is True
    assert enabled_njets.track_raw_counts is False
    assert enabled_sumw2.track_raw_counts is False


def test_central_fill_classification_truth_table_and_real_fill_contract():
    tracked = SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(2, 0.0, 2.0, name="x"),
        track_raw_counts=True,
    )
    untracked = SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(2, 0.0, 2.0, name="x"),
    )

    classify = AnalysisProcessor._raw_count_fill_classification
    assert classify(tracked, is_data=False, wgt_fluct="nominal") is True
    assert classify(tracked, is_data=True, wgt_fluct="nominal") is False
    assert classify(tracked, is_data=False, wgt_fluct="JESUp") is False
    assert classify(untracked, is_data=False, wgt_fluct="nominal") is None

    for process, systematic, is_data, values in (
        ("mc", "nominal", False, [0.25, 1.25]),
        ("data", "nominal", True, [0.25]),
        ("mc", "JESUp", False, [1.25]),
    ):
        classification = classify(
            tracked,
            is_data=is_data,
            wgt_fluct=systematic,
        )
        tracked.fill(
            process=process,
            systematic=systematic,
            x=np.asarray(values),
            weight=-3.0 * np.ones(len(values)),
            record_raw_count=classification,
        )

    assert set(tracked.raw_counts(flow=True)) == {("mc", "nominal")}
    np.testing.assert_array_equal(
        tracked.raw_counts(flow=True)[("mc", "nominal")],
        [0, 1, 1, 0],
    )


@pytest.mark.parametrize("histogram_type", [SparseHist, HistEFT])
def test_semantic_rebin_preserves_checked_integer_counts_and_flow(histogram_type):
    kwargs = {}
    if histogram_type is HistEFT:
        kwargs["wc_names"] = ["ctG"]
    histogram = histogram_type(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(4, 0.0, 4.0, name="x"),
        track_raw_counts=True,
        **kwargs,
    )
    fill_kwargs = {}
    if histogram_type is HistEFT:
        fill_kwargs["eft_coeff"] = np.asarray(
            [[1.0, 2.0, 3.0]] * 6,
        )
    histogram.fill(
        process="mc",
        systematic="nominal",
        x=np.asarray([-1.0, 0.25, 1.25, 2.25, 3.25, 5.0]),
        weight=np.asarray([2.0, -3.0, 0.5, 7.0, -11.0, 13.0]),
        record_raw_count=True,
        **fill_kwargs,
    )

    rebinned = rebin_histogram(histogram, [0.0, 2.0, 4.0])

    np.testing.assert_array_equal(
        rebinned.raw_counts(flow=True)[("mc", "nominal")],
        [1, 2, 2, 1],
    )
    assert rebinned.raw_counts(flow=True)[("mc", "nominal")].dtype == np.uint64
    source_weighted_sum = sum(
        np.asarray(value).sum()
        for value in histogram.view(flow=True, as_dict=True).values()
    )
    rebinned_weighted_sum = sum(
        np.asarray(value).sum()
        for value in rebinned.view(flow=True, as_dict=True).values()
    )
    assert rebinned_weighted_sum == source_weighted_sum


def test_semantic_rebin_detects_uint64_overflow():
    histogram = SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.Regular(2, 0.0, 2.0, name="x"),
        track_raw_counts=True,
    )
    histogram.fill(
        process="mc",
        x=np.asarray([0.25]),
        record_raw_count=True,
    )
    key = next(iter(histogram._raw_counts))
    histogram._raw_counts[key][1:3] = [np.iinfo(np.uint64).max, 1]

    with pytest.raises(OverflowError, match="overflow"):
        rebin_histogram(histogram, [0.0, 2.0])


@pytest.mark.parametrize("serializer", [pickle, cloudpickle])
def test_maintained_plotting_consumer_preserves_histeft_serialization(serializer):
    from analysis.topeft_run2 import make_cr_and_sr_plots  # noqa: F401

    untracked = HistEFT(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.Regular(2, 0.0, 2.0, name="x"),
        wc_names=["ctG"],
    )
    untracked.fill(
        process="mc",
        x=np.asarray([0.25]),
        weight=np.asarray([2.0]),
        eft_coeff=np.asarray([[1.0, 3.0, 5.0]]),
    )
    tracked = HistEFT(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.Regular(2, 0.0, 2.0, name="x"),
        wc_names=["ctG"],
        track_raw_counts=True,
    )
    tracked.fill(
        process="mc",
        x=np.asarray([1.25]),
        weight=np.asarray([-7.0]),
        eft_coeff=np.asarray([[1.0, 2.0, 3.0]]),
        record_raw_count=True,
    )

    restored = serializer.loads(
        serializer.dumps({"untracked": untracked, "tracked": tracked})
    )

    assert restored["untracked"].track_raw_counts is False
    assert not hasattr(restored["untracked"], "_raw_counts")
    np.testing.assert_array_equal(
        restored["tracked"].raw_counts(flow=True)[("mc",)],
        [0, 0, 1, 0],
    )
    np.testing.assert_allclose(
        restored["tracked"].eval({})[("mc",)],
        [0.0, 0.0, -7.0, 0.0],
    )


def test_raw_counts_are_semantically_addressable_across_129_555_topology():
    fitting_matrix = _maintained_fitting_matrix()
    assert len(fitting_matrix) == 129

    addressed_cells = 0
    observed_identities = set()
    for category, family in fitting_matrix.items():
        source_edges = processing_edges(axes_info[family])
        histogram = SparseHist(
            hist.axis.StrCategory([], name="process", growth=True),
            hist.axis.StrCategory([], name="channel", growth=True),
            hist.axis.StrCategory([], name="appl", growth=True),
            hist.axis.StrCategory([], name="systematic", growth=True),
            hist.axis.Variable(source_edges, name=family),
            track_raw_counts=True,
        )
        histogram.fill(
            process="mc",
            channel=category,
            appl="isSR",
            systematic="nominal",
            **{
                family: np.asarray(
                    [(source_edges[0] + source_edges[1]) / 2.0, source_edges[-1] + 1.0]
                ),
                "record_raw_count": True,
            },
        )

        target_edges = resolve_axis_edges(
            family,
            mode="fitting",
            channel=category,
        )
        rebinned = rebin_histogram(histogram, target_edges)
        identity, raw = next(iter(rebinned.raw_counts(flow=True).items()))
        observed_identities.add((family, *identity))

        # Final-bin authority retains the overflow cell and excludes underflow.
        final_cells = raw[1:]
        assert final_cells.dtype == np.uint64
        assert final_cells.shape == (len(target_edges),)
        assert final_cells[0] == 1
        assert final_cells[-1] == 1
        addressed_cells += final_cells.size

    assert len(observed_identities) == 129
    assert addressed_cells == 555
