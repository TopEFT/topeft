import importlib
import sys
import types

import awkward as ak
import numpy as np

_hist_eft_stub = types.ModuleType("topcoffea.modules.HistEFT")


class _DummyHistEFT:
    def __init__(self, *_, **__):
        pass

    def fill(self, **__):
        pass


_hist_eft_stub.HistEFT = _DummyHistEFT
importlib.import_module("topcoffea")
modules_pkg = importlib.import_module("topcoffea.modules")
modules_pkg.HistEFT = _hist_eft_stub  # type: ignore[attr-defined]
sys.modules["topcoffea.modules.HistEFT"] = _hist_eft_stub
sys.modules["topcoffea.modules.histEFT"] = _hist_eft_stub

import analysis.topeft_run2.analysis_processor as ap


class _DummyMapping(dict):
    """Mapping wrapper that mimics NanoEventsFactory metadata."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metadata = {
            "uuid": "jets-layout-test",
            "num_rows": len(next(iter(kwargs.values()))) if kwargs else 0,
            "object_path": "Events",
        }


def _build_processor(monkeypatch):
    processor = ap.AnalysisProcessor.__new__(ap.AnalysisProcessor)
    processor._debug_logging = True
    processor._debug = lambda *_, **__: None
    processor._var_def = "ht"
    processor.tau_h_analysis = False

    class _NoopJetCorrections:
        def __init__(self, *_, **__):
            pass

    monkeypatch.setattr(ap, "build_corrected_jets", lambda *_args, **_kwargs: _args[1])
    monkeypatch.setattr(ap, "build_corrected_met", lambda *_args, **_kwargs: _args[1])
    monkeypatch.setattr(ap, "ApplyJetSystematics", lambda *_args, **_kwargs: _args[1])
    monkeypatch.setattr(ap, "ApplyJetCorrections", _NoopJetCorrections)

    monkeypatch.setattr(
        ap.tc_os,
        "is_tight_jet",
        lambda pt, eta, jet_id, **__: (pt > 30) & (np.abs(eta) < 2.5) & (jet_id >= 2),
    )
    monkeypatch.setattr(
        ap.te_os,
        "isFwdJet",
        lambda pt, eta, jet_id, jetPtCut=40.0: (pt > jetPtCut) & (np.abs(eta) > 2.4) & (jet_id >= 2),
    )

    def _noop_event_selection(*_args, **_kwargs):
        return None

    monkeypatch.setattr(ap.te_es, "add1lMaskAndSFs", _noop_event_selection)
    monkeypatch.setattr(ap.te_es, "add2lMaskAndSFs", _noop_event_selection)
    monkeypatch.setattr(ap.te_es, "add3lMaskAndSFs", _noop_event_selection)
    monkeypatch.setattr(ap.te_es, "add4lMaskAndSFs", _noop_event_selection)
    monkeypatch.setattr(ap.te_es, "addLepCatMasks", _noop_event_selection)

    return processor


def test_jets_and_masks_have_flat_counts(monkeypatch):
    processor = _build_processor(monkeypatch)

    jets_per_event = [
        {
            "pt": np.float32(55.0),
            "eta": np.float32(0.5),
            "phi": np.float32(0.1),
            "mass": np.float32(10.0),
            "rawFactor": np.float32(0.1),
            "jetId": 6,
        },
        {
            "pt": np.float32(75.0),
            "eta": np.float32(2.6),
            "phi": np.float32(-0.4),
            "mass": np.float32(12.0),
            "rawFactor": np.float32(0.05),
            "jetId": 6,
        },
    ]
    jets = ak.Array([jets_per_event] * 5)

    variation_objects = ap.VariationObjects(
        met=ak.Array([[{"pt": np.float32(50.0), "phi": np.float32(0.0), "sumEt": np.float32(55.0)}]] * 5),
        electrons=ak.Array([[]] * 5),
        muons=ak.Array([[]] * 5),
        taus=ak.Array([[]] * 5),
        jets=jets,
        loose_leptons=ak.Array([[]] * 5),
        fakeable_leptons=ak.Array([[{"jetIdx": -1}]] * 5),
        fakeable_sorted=ak.Array([[{"jetIdx": -1}]] * 5),
        cleaning_taus=None,
        n_loose_taus=None,
        tau0=None,
    )

    variation_state = ap.VariationState(
        request=ap.VariationRequest(variation=None, histogram_label="nominal"),
        name="nominal",
        base=None,
        variation_type=None,
        metadata={},
        object_variation="nominal",
        weight_variations=[],
        requested_data_weight_label=None,
        sow_variation_key_map={},
        sow_variations={},
        objects=variation_objects,
        lepton_selection=object(),
        jets_rho=ak.Array([np.float32(0.5)] * 5),
    )

    dataset = ap.DatasetContext(
        dataset="sample",
        trigger_dataset="sample",
        hist_axis_name="sample",
        is_data=True,
        is_eft=False,
        year="2018",
        xsec=1.0,
        sow=1.0,
        run_era=None,
        is_run2=True,
        is_run3=False,
        sample_type="data",
        is_lo_sample=False,
        lumi_mask=ak.Array([True] * 5),
        lumi=1.0,
        eft_coeffs=None,
        eft_w2_coeffs=None,
    )

    events = _DummyMapping(event=ak.Array(np.arange(5)))

    processed_state = processor._apply_object_variations(events, dataset, variation_state)

    cleaned_counts = ak.num(processed_state.cleaned_jets.pt, axis=-1)
    good_counts = ak.num(processed_state.good_jets.pt, axis=-1)
    fwd_counts = ak.num(processed_state.fwd_jets.pt, axis=-1)
    jet_counts = ak.num(processed_state.good_jets.pt, axis=-1)
    good_mask_counts = ak.sum(processed_state.cleaned_jets.isGood, axis=-1)
    fwd_mask_counts = ak.sum(processed_state.cleaned_jets.isFwd, axis=-1)

    for counts in (
        cleaned_counts,
        good_counts,
        fwd_counts,
        good_mask_counts,
        fwd_mask_counts,
        processed_state.njets,
        processed_state.nfwdj,
    ):
        assert ak.to_layout(counts).purelist_depth == 1

    assert ak.all(good_counts == processed_state.njets)
    assert ak.all(fwd_counts == processed_state.nfwdj)

    assert ak.all(good_counts == good_mask_counts)
    assert ak.all(fwd_counts == fwd_mask_counts)
