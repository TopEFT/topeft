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
    one_dimensional = set(axes_info)
    two_dimensional = set(axes_info_2d)
    base_names = one_dimensional | two_dimensional
    expected_keys = {f"{name}__scalar_nominal" for name in one_dimensional}
    expected_keys.update(two_dimensional)
    expected_keys.update(f"{name}_sumw2" for name in base_names)

    assert set(processor.accumulator.keys()) == expected_keys
    assert set(processor._hist_lst) == expected_keys
    assert set(processor._hist_axis_map.keys()) == expected_keys | one_dimensional
    assert set(processor._hist_requires_eft.keys()) == expected_keys


@pytest.mark.parametrize(
    "requested_hists",
    [
        ["njets"],
        ["fwd0eta"],
        ["fwd0pt"],
        ["njets", "ptz_sumw2"],
        ["njets_sumw2"],
    ],
)
def test_filtered_hist_construction(requested_hists):
    processor = _make_processor(hist_lst=requested_hists)
    sumw2_suffix = "_sumw2"
    fill_sumw2_hist = processor._fill_sumw2_hist

    base_names = {
        name[: -len(sumw2_suffix)] if name.endswith(sumw2_suffix) else name
        for name in requested_hists
    }
    expected_accumulator_keys = set()
    for base_name in base_names:
        if base_name in axes_info_2d:
            expected_accumulator_keys.add(base_name)
        else:
            expected_accumulator_keys.add(f"{base_name}__scalar_nominal")
        if fill_sumw2_hist:
            expected_accumulator_keys.add(f"{base_name}{sumw2_suffix}")

    assert set(processor.accumulator.keys()) == expected_accumulator_keys
    assert set(processor._hist_lst) == expected_accumulator_keys
    assert set(processor._hist_axis_map.keys()) == expected_accumulator_keys | {
        name for name in base_names if name in axes_info
    }
    assert set(processor._hist_requires_eft.keys()) == expected_accumulator_keys

    serialized = cloudpickle.dumps(processor.accumulator)
    restored = cloudpickle.loads(serialized)
    assert set(restored.keys()) == expected_accumulator_keys

    # The mapping is stored with the base histogram name so that the filling
    # logic can look up the dense axis associated with the sumw2 histogram.
    if fill_sumw2_hist:
        assert set(processor._hist_sumw2_axis_mapping.keys()) == base_names
    else:
        assert not processor._hist_sumw2_axis_mapping


def test_sample_metadata_preallocates_both_siblings_and_preserves_two_dimensional_sparse():
    samples = {
        "data": {"histAxisName": "data", "isData": True, "WCnames": []},
        "background": {
            "histAxisName": "background",
            "isData": False,
            "WCnames": [],
        },
        "signal": {"histAxisName": "signal", "isData": False, "WCnames": ["ctG"]},
    }
    processor = analysis_processor.AnalysisProcessor(
        samples=samples,
        wc_names_lst=["ctG"],
        hist_lst=["njets", "lepton_pt_vs_eta"],
        fill_sumw2_hist=False,
    )
    assert tuple(processor.accumulator) == (
        "njets__scalar_nominal",
        "njets__eft_nominal",
        "lepton_pt_vs_eta",
    )
    assert processor._nominal_component_availability == {"scalar": True, "eft": True}
    assert [axis.name for axis in processor.accumulator["lepton_pt_vs_eta"].dense_axes] == [
        "lepton_pt_vs_eta_pt",
        "lepton_pt_vs_eta_abseta",
    ]


def test_variable_multi_axis_reuse_is_unchanged():
    processor = analysis_processor.AnalysisProcessor(
        samples={},
        wc_names_lst=[],
        hist_lst=["njets"],
        fill_sumw2_hist=True,
        rebin=False,
    )
    nominal_edges = processor.accumulator["njets__scalar_nominal"].dense_axes[0].edges
    companion_edges = processor.accumulator["njets_sumw2"].dense_axes[0].edges
    assert list(nominal_edges) == list(companion_edges)


@pytest.mark.parametrize(
    "fill_sumw2_hist,wgt_fluct,expected",
    [
        (True, "nominal", True),
        (True, "triggerSF_2022Up", False),
        (True, "JERUp", False),
        (False, "nominal", False),
    ],
)
def test_sumw2_fill_gate_only_allows_nominal_path(
    fill_sumw2_hist,
    wgt_fluct,
    expected,
):
    assert (
        analysis_processor.AnalysisProcessor._should_fill_sumw2_histogram(
            fill_sumw2_hist,
            wgt_fluct=wgt_fluct,
        )
        is expected
    )
