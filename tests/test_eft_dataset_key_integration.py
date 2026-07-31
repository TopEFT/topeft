from __future__ import annotations

import inspect
from unittest import mock

import numpy as np
import pytest

from analysis.topeft_run2 import analysis_processor
from analysis.topeft_run2.analysis_processor import (
    calculate_sm_sumw2_weights,
    prepare_event_eft_coefficients,
)
from topcoffea.modules import eft_helper
from topeft.modules.sumw2_policy import resolve_nominal_component_availability


class synthetic_events:
    def __init__(self, *, eft_coefficients=None, gen_weight=None):
        if eft_coefficients is not None:
            self.EFTfitCoefficients = np.asarray(eft_coefficients)
        if gen_weight is not None:
            self.genWeight = np.asarray(gen_weight)

    def __getitem__(self, name):
        return getattr(self, name)


class wc_lookup_forbidden(dict):
    def __getitem__(self, name):
        if name == "WCnames":
            raise AssertionError("WCnames lookup was not bypassed")
        return super().__getitem__(name)


def _shorten_stream(dataset_key):
    dataset = dataset_key
    for stream in (
        "Muon",
        "SingleMuon",
        "SingleElectron",
        "EGamma",
        "MuonEG",
        "DoubleMuon",
        "DoubleElectron",
        "DoubleEG",
    ):
        if dataset.startswith(stream):
            dataset = dataset.split("_")[0]
    return dataset


@pytest.mark.parametrize(
    "dataset_key,stream",
    [
        ("Muon_Run2022F-22Sep2023_NDSkim", "Muon"),
        ("EGamma_Run2022F-22Sep2023_NDSkim", "EGamma"),
    ],
)
def test_data_stream_key_bypasses_eft_setup_and_wc_lookup(dataset_key, stream):
    samples = {
        dataset_key: wc_lookup_forbidden(
            isData=True,
            histAxisName="data2022",
        )
    }
    sample_metadata = samples[dataset_key]
    assert _shorten_stream(dataset_key) == stream

    with mock.patch.object(
        analysis_processor,
        "prepare_eft_coefficients",
        side_effect=AssertionError("EFT helper was called"),
    ) as prepare:
        observed = prepare_event_eft_coefficients(
            synthetic_events(),
            sample_metadata,
            [],
            None,
            sample_name=dataset_key,
        )

    assert observed is None
    prepare.assert_not_called()
    assert stream not in samples


def test_scalar_mc_bypasses_eft_setup_and_preserves_genweight_and_sibling():
    sample_metadata = wc_lookup_forbidden(
        isData=False,
        histAxisName="DY_central2022",
    )
    events = synthetic_events(gen_weight=[-1.0, 2.0])
    with mock.patch.object(
        analysis_processor,
        "prepare_eft_coefficients",
        side_effect=AssertionError("EFT helper was called"),
    ) as prepare:
        eft_coefficients = prepare_event_eft_coefficients(
            events,
            sample_metadata,
            [],
            None,
            sample_name="DYJetsToLL_2022",
        )

    prepare.assert_not_called()
    assert eft_coefficients is None
    np.testing.assert_array_equal(events["genWeight"], [-1.0, 2.0])
    np.testing.assert_array_equal(
        calculate_sm_sumw2_weights(events["genWeight"], eft_coefficients),
        [1.0, 4.0],
    )
    assert resolve_nominal_component_availability(
        {"dy": {**sample_metadata, "WCnames": []}}
    ) == {"scalar": True, "eft": False}


@pytest.mark.parametrize("sample_name", ["ordinary_ttll", "other_eft_signal"])
@pytest.mark.parametrize("systematic", ["nominal", "renormUp", "JESUp"])
def test_absent_role_full_eft_vectors_use_immutable_metadata(
    sample_name,
    systematic,
):
    native_wc_names = ["ctW", "ctZ"]
    global_wc_names = ["ctZ", "ctW"]
    source_vectors = {
        "nominal": [[2.0, 3.0, 5.0, 7.0, 11.0, 13.0]],
        "renormUp": [[17.0, 19.0, 23.0, 29.0, 31.0, 37.0]],
        "JESUp": [[41.0, 43.0, 47.0, 53.0, 59.0, 61.0]],
    }
    source = np.asarray(source_vectors[systematic])
    sample_metadata = {
        "isData": False,
        "histAxisName": "ttll_private2022",
        "WCnames": native_wc_names,
    }
    expected = eft_helper.remap_coeffs(native_wc_names, global_wc_names, source)

    observed = prepare_event_eft_coefficients(
        synthetic_events(eft_coefficients=source),
        sample_metadata,
        global_wc_names,
        None,
        sample_name=sample_name,
    )

    np.testing.assert_array_equal(observed, expected)
    assert np.count_nonzero(observed[..., 1:]) > 0
    assert resolve_nominal_component_availability({sample_name: sample_metadata}) == {
        "scalar": False,
        "eft": True,
    }


def test_full_eft_helper_receives_wc_names_from_immutable_sample_metadata():
    dataset_key = "ttll_private_2022_source_key"
    sample_metadata = {
        "isData": False,
        "histAxisName": "ttll_private2022",
        "WCnames": ["ctW"],
    }
    source = np.asarray([[2.0, 3.0, 5.0]])
    with mock.patch.object(
        analysis_processor,
        "prepare_eft_coefficients",
        wraps=analysis_processor.prepare_eft_coefficients,
    ) as prepare:
        prepare_event_eft_coefficients(
            synthetic_events(eft_coefficients=source),
            sample_metadata,
            ["ctW"],
            None,
            sample_name=dataset_key,
        )

    assert prepare.call_args.args[1] is sample_metadata["WCnames"]
    assert prepare.call_args.kwargs["sample_name"] == dataset_key


def test_sm_only_projection_sibling_and_sumw2_are_unchanged():
    sample_metadata = {
        "isData": False,
        "histAxisName": "ttll_private2022",
        "WCnames": ["ctW"],
        "eft_treatment": "sm_only",
    }
    source = np.asarray([[7.0, 11.0, 13.0], [2.0, -5.0, 4.0]])
    observed = prepare_event_eft_coefficients(
        synthetic_events(eft_coefficients=source),
        sample_metadata,
        ["ctW"],
        "sm_only",
        sample_name="ttll_mll4to10_2022",
    )

    np.testing.assert_array_equal(observed, [[7.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    np.testing.assert_array_equal(
        calculate_sm_sumw2_weights(np.asarray([1.5, 3.0]), observed),
        [110.25, 36.0],
    )
    assert resolve_nominal_component_availability(
        {"low_mass": sample_metadata}
    ) == {"scalar": False, "eft": True}


def test_sm_only_missing_branch_preserves_stable_runtime_error():
    sample_metadata = {
        "isData": False,
        "histAxisName": "ttll_private2022",
        "WCnames": ["ctW"],
        "eft_treatment": "sm_only",
    }
    with pytest.raises(RuntimeError, match="EFT-TREATMENT-E004"):
        prepare_event_eft_coefficients(
            synthetic_events(),
            sample_metadata,
            ["ctW"],
            "sm_only",
            sample_name="ttll_mll4to10_missing_branch_2022",
        )


def test_process_callsite_uses_sample_metadata_not_mutable_stream_label():
    source = inspect.getsource(analysis_processor.AnalysisProcessor.process)
    call_start = source.index("eft_coeffs = prepare_event_eft_coefficients(")
    call_end = source.index("\n        # Initialize the out object", call_start)
    call = source[call_start:call_end]

    assert "sample_metadata" in call
    assert "sample_name=dataset_key" in call
    assert "self._samples[dataset]" not in call
    assert "WCnames" not in call
