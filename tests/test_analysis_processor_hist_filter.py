import pytest

cloudpickle = pytest.importorskip("cloudpickle")

from analysis.topeft_run2 import analysis_processor
from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d


def _make_processor(hist_lst=None, fill_sumw2_hist=True):
    # The processor only needs a samples dictionary for instantiation; the
    # individual entries are accessed during processing, so the tests can use an
    # empty mapping here.
    return analysis_processor.AnalysisProcessor(
        samples={},
        wc_names_lst=[],
        hist_lst=hist_lst,
        fill_sumw2_hist=fill_sumw2_hist,
    )


def test_accumulator_keys_without_hist_filter():
    processor = _make_processor()
    base_names = set(axes_info.keys()) | set(axes_info_2d.keys())
    expected_keys = set(base_names)
    expected_keys.update(f"{name}_sumw2" for name in base_names)

    assert set(processor.accumulator.keys()) == expected_keys
    assert set(processor._hist_lst) == expected_keys
    assert set(processor._hist_axis_map.keys()) == expected_keys
    assert set(processor._hist_requires_eft.keys()) == expected_keys


@pytest.mark.parametrize(
    "requested_hists",
    [
        ["njets"],
        ["njets", "ptz_sumw2"],
        ["njets_sumw2"],
    ],
)
def test_filtered_hist_construction(requested_hists):
    processor = _make_processor(hist_lst=requested_hists)
    expected_keys = set(requested_hists)

    assert set(processor.accumulator.keys()) == expected_keys
    assert set(processor._hist_lst) == expected_keys
    assert set(processor._hist_axis_map.keys()) == expected_keys
    assert set(processor._hist_requires_eft.keys()) == expected_keys

    serialized = cloudpickle.dumps(processor.accumulator)
    restored = cloudpickle.loads(serialized)
    assert set(restored.keys()) == expected_keys

    if any(name.endswith("_sumw2") for name in requested_hists):
        # The mapping is stored with the base histogram name so that the
        # filling logic can look up the dense axis associated with the sumw2
        # histogram.
        base_names = {
            name[: -len("_sumw2")] for name in requested_hists if name.endswith("_sumw2")
        }
        assert set(processor._hist_sumw2_axis_mapping.keys()) == base_names
    else:
        assert not processor._hist_sumw2_axis_mapping

