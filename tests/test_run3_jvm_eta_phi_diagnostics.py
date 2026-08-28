import awkward as ak
import numpy as np

from analysis.topeft_run2.analysis_processor import (
    AnalysisProcessor,
    flatten_jagged_jet_eta_phi_weights,
    get_jvm_eta_phi_event_mask,
    should_include_jet_veto_in_histogram_selection,
    should_fill_jvm_eta_phi_diagnostic,
)


BEFORE = "jet_eta_phi_before_veto"
AFTER = "jet_eta_phi_after_veto"


def _processor(hist_lst):
    return AnalysisProcessor(
        samples={},
        wc_names_lst=[],
        hist_lst=hist_lst,
        fill_sumw2_hist=True,
    )


def _fill_diagnostic(histogram, histogram_name, jets, base_mask, jet_veto, weights):
    event_mask = get_jvm_eta_phi_event_mask(base_mask, jet_veto, histogram_name)
    eta, phi, per_jet_weights = flatten_jagged_jet_eta_phi_weights(
        jets,
        event_mask,
        weights,
    )
    eta_axis, phi_axis = [axis.name for axis in histogram.dense_axes]
    histogram.fill(
        **{
            eta_axis: eta,
            phi_axis: phi,
            "process": "synthetic",
            "channel": "inclusive",
            "systematic": "nominal",
            "appl": "isSR",
            "weight": per_jet_weights,
        }
    )
    return eta, phi, per_jet_weights


def _dense_slice(histogram):
    return histogram[
        {
            "process": "synthetic",
            "channel": "inclusive",
            "systematic": "nominal",
            "appl": "isSR",
        }
    ]


def test_jvm_eta_phi_histograms_are_registered_and_explicitly_selectable():
    processor = _processor([BEFORE, AFTER])

    assert set(processor.accumulator) == {BEFORE, AFTER}
    for histogram_name in (BEFORE, AFTER):
        assert [axis.size for axis in processor.accumulator[histogram_name].dense_axes] == [
            104,
            72,
        ]

    all_histograms = _processor(None)
    assert BEFORE in all_histograms.accumulator
    assert AFTER in all_histograms.accumulator


def test_jvm_eta_phi_before_after_event_semantics_and_per_jet_weights():
    processor = _processor([BEFORE, AFTER])
    jets = ak.Array(
        [
            [
                {"eta": 3.1, "phi": 0.2},
                {"eta": 3.1, "phi": 0.2},
            ],
            [
                {"eta": -4.7, "phi": -0.4},
                {"eta": 0.4, "phi": 1.0},
            ],
        ]
    )
    base_mask = np.array([True, True])
    # This fixture independently specifies the event-level map decision: event
    # zero passes; event one fails because one of its JVM-input jets vetoes it.
    jet_veto = np.array([True, False])
    weights = np.array([2.5, 4.0])

    before_rows = _fill_diagnostic(
        processor.accumulator[BEFORE], BEFORE, jets, base_mask, jet_veto, weights
    )
    after_rows = _fill_diagnostic(
        processor.accumulator[AFTER], AFTER, jets, base_mask, jet_veto, weights
    )

    assert ak.to_list(before_rows[0]) == [3.1, 3.1, -4.7, 0.4]
    assert ak.to_list(after_rows[0]) == [3.1, 3.1]
    assert ak.to_list(before_rows[2]) == [2.5, 2.5, 4.0, 4.0]
    assert ak.to_list(after_rows[2]) == [2.5, 2.5]

    before_dense = _dense_slice(processor.accumulator[BEFORE])
    after_dense = _dense_slice(processor.accumulator[AFTER])
    before_values = before_dense.values(flow=False)
    after_values = after_dense.values(flow=False)
    eta_axis, phi_axis = before_dense.axes
    same_bin = (eta_axis.index(3.1), phi_axis.index(0.2))
    forward_bin = (eta_axis.index(-4.7), phi_axis.index(-0.4))

    assert before_values[same_bin] == 5.0
    assert after_values[same_bin] == 5.0
    assert before_values[forward_bin] == 4.0
    assert after_values[forward_bin] == 0.0
    assert before_values.sum() == 13.0
    assert after_values.sum() == 5.0


def test_jvm_eta_phi_unit_weights_and_nominal_guard():
    jets = ak.Array([[{"eta": 4.8, "phi": 0.1}, {"eta": 4.8, "phi": 0.1}]])
    eta, phi, weights = flatten_jagged_jet_eta_phi_weights(
        jets,
        np.array([True]),
        np.array([1.0]),
    )

    assert ak.to_list(eta) == [4.8, 4.8]
    assert ak.to_list(phi) == [0.1, 0.1]
    assert ak.to_list(weights) == [1.0, 1.0]
    assert should_fill_jvm_eta_phi_diagnostic(True, "nominal", "nominal")
    assert not should_fill_jvm_eta_phi_diagnostic(True, "jes_up", "nominal")
    assert not should_fill_jvm_eta_phi_diagnostic(True, "nominal", "pu_up")
    assert not should_fill_jvm_eta_phi_diagnostic(False, "nominal", "nominal")
    assert should_include_jet_veto_in_histogram_selection("njets")
    assert not should_include_jet_veto_in_histogram_selection(BEFORE)
    assert not should_include_jet_veto_in_histogram_selection(AFTER)
