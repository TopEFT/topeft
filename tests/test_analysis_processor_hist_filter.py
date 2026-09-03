import gzip
import inspect

import cloudpickle
import hist
import numpy as np
import pytest
from coffea import processor as coffea_processor

from analysis.topeft_run2 import analysis_processor
from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist


ONE_DIMENSIONAL_HIST = "njets"
TWO_DIMENSIONAL_HIST = "lepton_pt_vs_eta"
WC_NAMES = ["ctG", "ctW"]


def _make_processor(hist_lst=None, fill_sumw2_hist=True):
    return analysis_processor.AnalysisProcessor(
        samples={},
        wc_names_lst=WC_NAMES,
        hist_lst=hist_lst,
        fill_sumw2_hist=fill_sumw2_hist,
    )


def _normalized(mapping):
    return {tuple(key): np.asarray(value).copy() for key, value in mapping.items()}


def _fill_toy_output(output):
    output[ONE_DIMENSIONAL_HIST].fill(
        process="ttH",
        channel="2lss",
        systematic="nominal",
        appl="isSR",
        njets=np.array([1.0, 1.0]),
        weight=np.array([3.0, -2.0]),
        eft_coeff=np.array(
            [
                [2.0, 0.5, 0.25, 0.0, 0.0, 0.0],
                [1.0, -0.5, 0.25, 0.0, 0.0, 0.0],
            ]
        ),
        fill_sumw2=True,
    )
    output[ONE_DIMENSIONAL_HIST].fill(
        process="ttH",
        channel="2lss",
        systematic="JERUp",
        appl="isSR",
        njets=np.array([1.0, 1.0]),
        weight=np.array([3.0, -2.0]),
        eft_coeff=None,
        fill_sumw2=False,
    )

    two_dimensional = output[TWO_DIMENSIONAL_HIST]
    axis_names = two_dimensional.dense_axes.name
    two_dimensional.fill(
        process="ttH",
        channel="2lss",
        systematic="nominal",
        appl="isSR",
        **{
            axis_names[0]: np.array([10.0, 10.0]),
            axis_names[1]: np.array([0.1, 0.1]),
        },
        weight=np.array([2.0, -3.0]),
    )
    return output


def _only_dense_view(histogram):
    views = histogram.view(flow=False, as_dict=True)
    assert len(views) == 1
    return next(iter(views.values()))


def test_analysis_processor_has_no_backend_parameter():
    signature = inspect.signature(analysis_processor.AnalysisProcessor)
    assert "use_multicell" not in signature.parameters

    with pytest.raises(TypeError, match="use_multicell"):
        analysis_processor.AnalysisProcessor(
            samples={},
            use_multicell=True,
        )


def test_selected_1d_and_2d_histograms_use_multicell_storage():
    output = _make_processor(
        hist_lst=[ONE_DIMENSIONAL_HIST, TWO_DIMENSIONAL_HIST]
    ).accumulator

    assert set(output) == {ONE_DIMENSIONAL_HIST, TWO_DIMENSIONAL_HIST}
    assert not any(name.endswith("_sumw2") for name in output)

    one_dimensional = output[ONE_DIMENSIONAL_HIST]
    assert isinstance(one_dimensional, HistEFT)
    assert one_dimensional._use_multicell is True
    assert one_dimensional.store_sumw2 is True
    assert "quadratic_term" not in one_dimensional.axes.name

    two_dimensional = output[TWO_DIMENSIONAL_HIST]
    assert isinstance(two_dimensional, SparseHist)
    assert not isinstance(two_dimensional, HistEFT)
    assert isinstance(two_dimensional._init_args["storage"], hist.storage.Weight)


def test_sumw2_input_aliases_resolve_to_logical_histogram_names():
    output = _make_processor(
        hist_lst=[
            f"{ONE_DIMENSIONAL_HIST}_sumw2",
            f"{TWO_DIMENSIONAL_HIST}_sumw2",
        ]
    ).accumulator

    assert set(output) == {ONE_DIMENSIONAL_HIST, TWO_DIMENSIONAL_HIST}
    assert not any(name.endswith("_sumw2") for name in output)


@pytest.mark.parametrize(
    "enabled,variation,expected",
    [
        (True, "nominal", True),
        (True, "JERUp", False),
        (False, "nominal", False),
    ],
)
def test_embedded_sumw2_fill_gate_only_allows_enabled_nominal(
    enabled,
    variation,
    expected,
):
    assert (
        analysis_processor.AnalysisProcessor._should_fill_nominal_sumw2(
            enabled,
            wgt_fluct=variation,
        )
        is expected
    )


def test_nominal_1d_variance_and_2d_weighted_variance_are_correct():
    output = _fill_toy_output(
        _make_processor(
            hist_lst=[ONE_DIMENSIONAL_HIST, TWO_DIMENSIONAL_HIST]
        ).accumulator
    )

    one_dimensional_sumw2 = _normalized(
        output[ONE_DIMENSIONAL_HIST].nominal_sumw2(flow=False)
    )
    nominal_key = ("ttH", "2lss", "nominal", "isSR")
    systematic_key = ("ttH", "2lss", "JERUp", "isSR")
    assert np.sum(one_dimensional_sumw2[nominal_key]) == pytest.approx(40.0)
    assert np.sum(one_dimensional_sumw2[systematic_key]) == pytest.approx(0.0)

    two_dimensional = _only_dense_view(output[TWO_DIMENSIONAL_HIST])
    assert np.sum(two_dimensional.value) == pytest.approx(-1.0)
    assert np.sum(two_dimensional.variance) == pytest.approx(13.0)


def test_accumulation_adds_values_and_variances():
    outputs = [
        _fill_toy_output(
            _make_processor(
                hist_lst=[ONE_DIMENSIONAL_HIST, TWO_DIMENSIONAL_HIST]
            ).accumulator
        )
        for _ in range(2)
    ]

    single_1d_yields = _normalized(
        outputs[0][ONE_DIMENSIONAL_HIST].yield_coefficients(flow=False)
    )
    single_1d_sumw2 = _normalized(
        outputs[0][ONE_DIMENSIONAL_HIST].nominal_sumw2(flow=False)
    )
    single_2d = _only_dense_view(outputs[0][TWO_DIMENSIONAL_HIST])
    single_2d_values = np.asarray(single_2d.value).copy()
    single_2d_variances = np.asarray(single_2d.variance).copy()

    combined = coffea_processor.accumulate(outputs)

    for key, expected in single_1d_yields.items():
        np.testing.assert_allclose(
            _normalized(
                combined[ONE_DIMENSIONAL_HIST].yield_coefficients(flow=False)
            )[key],
            2.0 * expected,
        )
    for key, expected in single_1d_sumw2.items():
        np.testing.assert_allclose(
            _normalized(
                combined[ONE_DIMENSIONAL_HIST].nominal_sumw2(flow=False)
            )[key],
            2.0 * expected,
        )

    combined_2d = _only_dense_view(combined[TWO_DIMENSIONAL_HIST])
    np.testing.assert_allclose(combined_2d.value, 2.0 * single_2d_values)
    np.testing.assert_allclose(
        combined_2d.variance,
        2.0 * single_2d_variances,
    )


def test_cloudpickle_gzip_round_trip_preserves_histogram_data(tmp_path):
    output = _fill_toy_output(
        _make_processor(
            hist_lst=[ONE_DIMENSIONAL_HIST, TWO_DIMENSIONAL_HIST]
        ).accumulator
    )
    output_path = tmp_path / "multicell-output.pkl.gz"

    with gzip.open(output_path, "wb") as stream:
        cloudpickle.dump(output, stream)
    with gzip.open(output_path, "rb") as stream:
        restored = cloudpickle.load(stream)

    assert set(restored) == {ONE_DIMENSIONAL_HIST, TWO_DIMENSIONAL_HIST}

    expected_yields = _normalized(
        output[ONE_DIMENSIONAL_HIST].yield_coefficients(flow=True)
    )
    restored_yields = _normalized(
        restored[ONE_DIMENSIONAL_HIST].yield_coefficients(flow=True)
    )
    assert set(restored_yields) == set(expected_yields)
    for key, expected in expected_yields.items():
        np.testing.assert_array_equal(restored_yields[key], expected)

    expected_sumw2 = _normalized(
        output[ONE_DIMENSIONAL_HIST].nominal_sumw2(flow=True)
    )
    restored_sumw2 = _normalized(
        restored[ONE_DIMENSIONAL_HIST].nominal_sumw2(flow=True)
    )
    assert set(restored_sumw2) == set(expected_sumw2)
    for key, expected in expected_sumw2.items():
        np.testing.assert_array_equal(restored_sumw2[key], expected)

    expected_2d = _only_dense_view(output[TWO_DIMENSIONAL_HIST])
    restored_2d = _only_dense_view(restored[TWO_DIMENSIONAL_HIST])
    np.testing.assert_array_equal(restored_2d.value, expected_2d.value)
    np.testing.assert_array_equal(restored_2d.variance, expected_2d.variance)
