from collections import defaultdict
from types import MethodType

import hist
import numpy as np
import pytest

from analysis.topeft_run2 import make_cr_and_sr_plots
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules.dataDrivenEstimation import DataDrivenProducer
from topeft.modules.get_renormfact_envelope import (
    RENORMFACT_VAR_LST,
    get_renormfact_envelope,
)
from topeft.modules.yield_tools import YieldTools


@pytest.fixture
def sparse_hist_axes():
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name="pt"),
    )


def _assign_default_sumw2(histogram):
    setattr(histogram, "_sumw2", defaultdict(lambda: None))
    return histogram


def test_data_driven_producer_canonicalizes_data_driven_outputs(sparse_hist_axes):
    histogram = _assign_default_sumw2(SparseHist(*sparse_hist_axes))

    histogram.fill(
        process="data2023BPix",
        appl="isAR_2lSS_OS",
        systematic="nominal",
        pt=0.5,
        weight=7.0,
    )
    histogram.fill(
        process="dataUL16",
        appl="isAR_3l",
        systematic="nominal",
        pt=0.5,
        weight=11.0,
    )
    histogram.fill(
        process="TTTo2L2Nu_centralUL16",
        appl="isAR_3l",
        systematic="nominal",
        pt=0.5,
        weight=3.0,
    )
    histogram.fill(
        process="TTTo2L2Nu_centralUL16",
        appl="isSR_3l",
        systematic="nominal",
        pt=0.5,
        weight=1.5,
    )

    producer = DataDrivenProducer({"nominal": histogram}, "")
    output_hist = producer.getDataDrivenHistogram()["nominal"]

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


@pytest.fixture
def renorm_envelope_hist():
    axes = (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name="pt"),
    )
    histogram = _assign_default_sumw2(SparseHist(*axes))

    def _remove(self, bins, axis_name=None):
        if axis_name is None:
            raise TypeError("axis_name is required")
        return SparseHist.remove(self, axis_name, bins)

    histogram.remove = MethodType(_remove, histogram)

    for channel in ("3l", "2lss"):
        for idx, systematic in enumerate(["nominal", *RENORMFACT_VAR_LST]):
            histogram.fill(
                process="ttH_centralUL16",
                channel=channel,
                systematic=systematic,
                pt=0.5,
                weight=1.0 + idx,
            )

    histogram.fill(
        process="nonpromptUL16",
        channel="3l",
        systematic="nominal",
        pt=0.5,
        weight=5.0,
    )
    histogram.fill(
        process="flipsUL16",
        channel="2lss",
        systematic="nominal",
        pt=0.5,
        weight=4.0,
    )
    return histogram


def test_get_renormfact_envelope_skips_lowercase_prefixed_processes(renorm_envelope_hist):
    outputs = get_renormfact_envelope({"hist": renorm_envelope_hist})
    out_hist = outputs["hist"]

    processes = list(out_hist.axes["process"])
    assert "nonpromptUL16" in processes
    assert "flipsUL16" in processes

    nominal_slice = out_hist[
        {"process": "nonpromptUL16", "channel": "3l", "systematic": "nominal"}
    ].values()
    np.testing.assert_allclose(nominal_slice, np.array([5.0]))


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
