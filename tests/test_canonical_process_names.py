import hist
import numpy as np
import pytest

from analysis.topeft_run2 import make_cr_and_sr_plots
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules.dataDrivenEstimation import DataDrivenProducer
from topeft.modules.yield_tools import YieldTools


@pytest.fixture
def sparse_hist_axes():
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name="pt"),
    )


def test_data_driven_producer_canonicalizes_data_driven_outputs(sparse_hist_axes):
    legacy_met_axes = (
        *sparse_hist_axes[:-1],
        hist.axis.Regular(1, 0.0, 1.0, name="met"),
    )
    histogram = SparseHist(*legacy_met_axes)

    histogram.fill(
        process="data2023BPix",
        appl="isAR_2lSS_OS",
        systematic="nominal",
        met=0.5,
        weight=7.0,
    )
    histogram.fill(
        process="dataUL16",
        appl="isAR_3l",
        systematic="nominal",
        met=0.5,
        weight=11.0,
    )
    histogram.fill(
        process="TTTo2L2Nu_centralUL16",
        appl="isAR_3l",
        systematic="nominal",
        met=0.5,
        weight=3.0,
    )
    histogram.fill(
        process="TTTo2L2Nu_centralUL16",
        appl="isSR_3l",
        systematic="nominal",
        met=0.5,
        weight=1.5,
    )

    producer = DataDrivenProducer({"met": histogram}, "")
    output_hist = producer.getDataDrivenHistogram()["met"]

    processes = list(output_hist.axes["process"])
    assert "nonpromptUL16" in processes
    assert "flips2023BPix" in processes

    nonprompt_yields = output_hist[
        {"process": "nonpromptUL16", "systematic": "nominal"}
    ].values()
    flips_yields = output_hist[
        {"process": "flips2023BPix", "systematic": "nominal"}
    ].values()

    np.testing.assert_allclose(nonprompt_yields, np.array([8.0]))
    np.testing.assert_allclose(flips_yields, np.array([7.0]))


def test_generated_processes_are_unrecorded_while_retained_mc_stays_tracked(
    sparse_hist_axes,
):
    histogram = SparseHist(
        *sparse_hist_axes,
        storage="Double",
        track_raw_counts=True,
    )
    for process, appl, weight, record_raw_count in (
        ("data2023BPix", "isAR_2lSS_OS", 7.0, False),
        ("dataUL16", "isAR_3l", 11.0, False),
        ("TTTo2L2Nu_centralUL16", "isAR_3l", 3.0, True),
        ("TTTo2L2Nu_centralUL16", "isSR_3l", 1.5, True),
    ):
        histogram.fill(
            process=process,
            appl=appl,
            systematic="nominal",
            pt=np.asarray([0.5]),
            weight=np.asarray([weight]),
            record_raw_count=record_raw_count,
        )

    output = DataDrivenProducer({"met": histogram}, "").getDataDrivenHistogram()[
        "met"
    ]
    raw_counts = output.raw_counts(flow=True)
    recorded_by_process = {
        str(categories.process): state for categories, state in raw_counts.items()
    }

    assert "nonpromptUL16" not in recorded_by_process
    assert "flips2023BPix" not in recorded_by_process
    np.testing.assert_array_equal(
        recorded_by_process["TTTo2L2Nu_centralUL16"],
        np.asarray([0, 1, 0], dtype=np.uint64),
    )


@pytest.mark.parametrize(
    "process_name,expected",
    [
        ("NonPromptUL16", "fakes"),
        ("Flips2023BPix", "flips"),
    ],
)
def test_yield_tools_handles_legacy_process_casing(process_name, expected):
    yt = YieldTools()
    assert yt.get_short_name(process_name) == expected


def test_populate_group_map_handles_canonical_and_legacy_names():
    samples = ["nonpromptUL16", "NonPromptUL16", "ttH_centralUL16"]
    pattern_map = {
        "Nonprompt": ["nonprompt"],
        "Signal": ["ttH"],
    }

    group_map = make_cr_and_sr_plots.populate_group_map(samples, pattern_map)

    assert group_map["Nonprompt"] == ["nonpromptUL16", "NonPromptUL16"]
    assert group_map["Signal"] == ["ttH_centralUL16"]
    assert list(group_map.keys()) == ["Nonprompt", "Signal"]
