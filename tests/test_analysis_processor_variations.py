import awkward as ak
import numpy as np

from analysis.topeft_run2 import analysis_processor as ap


class _NoopJetCorrections:
    def __init__(self, *args, **kwargs):
        pass

    def build(self, *args, **kwargs):
        return args[0]


def _patch_common(monkeypatch):
    monkeypatch.setattr(ap, "ApplyJetCorrections", lambda *args, **kwargs: _NoopJetCorrections())

    def _noop_systematics(year, jets, variation):
        return jets

    monkeypatch.setattr(ap, "ApplyJetSystematics", _noop_systematics)

    def _noop_event_selection(*args, **kwargs):
        return None

    monkeypatch.setattr(ap.te_es, "add1lMaskAndSFs", _noop_event_selection)
    monkeypatch.setattr(ap.te_es, "add2lMaskAndSFs", _noop_event_selection)
    monkeypatch.setattr(ap.te_es, "add3lMaskAndSFs", _noop_event_selection)
    monkeypatch.setattr(ap.te_es, "add4lMaskAndSFs", _noop_event_selection)
    monkeypatch.setattr(ap.te_es, "addLepCatMasks", _noop_event_selection)


def _make_dataset_context(**overrides):
    defaults = dict(
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
        sample_type="mc",
        is_lo_sample=False,
        lumi_mask=ak.Array([True]),
        lumi=1.0,
        eft_coeffs=None,
        eft_w2_coeffs=None,
    )
    defaults.update(overrides)
    return ap.DatasetContext(**defaults)


def _make_variation_state(*, jets, fakeable_leptons, fakeable_sorted, taus=None, cleaning_taus=None):
    if taus is None:
        taus = ak.Array([[]])
    variation_objects = ap.VariationObjects(
        met=ak.Array([{"pt": 80.0, "phi": 0.0, "sumEt": 100.0}]),
        electrons=ak.Array([[]]),
        muons=ak.Array([[]]),
        taus=taus,
        jets=jets,
        loose_leptons=ak.Array([[]]),
        fakeable_leptons=fakeable_leptons,
        fakeable_sorted=fakeable_sorted,
        cleaning_taus=cleaning_taus,
        n_loose_taus=None,
        tau0=None,
    )
    jets_rho = ak.ones_like(jets.pt) * np.float32(0.5)
    return ap.VariationState(
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
        jets_rho=jets_rho,
    )


def _make_processor():
    processor = ap.AnalysisProcessor.__new__(ap.AnalysisProcessor)
    processor._var_def = {"ht", "j0"}
    processor.tau_h_analysis = False
    processor._debug_logging = False
    processor._debug = lambda *args, **kwargs: None
    return processor


def test_apply_object_variations_preserves_good_jet_count(monkeypatch):
    _patch_common(monkeypatch)
    processor = _make_processor()
    jets = ak.Array([
        [
            {
                "pt": 55.0,
                "eta": 0.2,
                "phi": 0.1,
                "mass": 10.0,
                "rawFactor": 0.1,
                "jetId": 6,
                "matched_gen": {"pt": 50.0},
            },
            {
                "pt": 25.0,
                "eta": 2.5,
                "phi": -0.3,
                "mass": 5.0,
                "rawFactor": 0.05,
                "jetId": 6,
                "matched_gen": {"pt": 20.0},
            },
        ]
    ])
    fakeable = ak.Array([[{
        "pt": 40.0,
        "eta": 0.1,
        "phi": 0.2,
        "mass": 0.1,
        "jetIdx": -1,
    }]])
    fakeable_sorted = ak.Array([[
        {"pt": 45.0, "conept": 45.0, "jetIdx": -1},
        {"pt": 35.0, "conept": 35.0, "jetIdx": -1},
        {"pt": 25.0, "conept": 25.0, "jetIdx": -1},
    ]])
    variation_state = _make_variation_state(
        jets=jets, fakeable_leptons=fakeable, fakeable_sorted=fakeable_sorted
    )
    events = ak.Array({"event": [1]})
    dataset = _make_dataset_context(is_data=True)

    updated_state = processor._apply_object_variations(events, dataset, variation_state, None)

    assert ak.to_list(updated_state.njets) == [1]
    assert ak.to_list(updated_state.ht) == [55.0]
    assert ak.to_list(events["njets"]) == [1]


def test_apply_tau_variations_updates_tau_properties(monkeypatch):
    _patch_common(monkeypatch)

    def _tes(year, tau, is_data, variation):
        return tau.pt + 5.0, tau.mass + 0.1

    def _fes(year, tau, is_data, variation):
        return tau.pt + 2.0, tau.mass + 0.2

    monkeypatch.setattr(ap, "ApplyTESSystematic", _tes)
    monkeypatch.setattr(ap, "ApplyFESSystematic", _fes)

    processor = _make_processor()
    processor.tau_h_analysis = True

    jets = ak.Array([
        [
            {
                "pt": 50.0,
                "eta": 0.1,
                "phi": 0.2,
                "mass": 8.0,
                "rawFactor": 0.1,
                "jetId": 6,
                "matched_gen": {"pt": 45.0},
            }
        ]
    ])
    fakeable = ak.Array([[{
        "pt": 30.0,
        "eta": 0.1,
        "phi": 0.1,
        "mass": 0.1,
        "jetIdx": -1,
    }]])
    fakeable_sorted = ak.Array([[{ "pt": 30.0, "conept": 30.0, "jetIdx": -1 }]])
    taus = ak.Array([[{
        "pt": 25.0,
        "mass": 1.0,
        "isLoose": 1,
        "eta": 0.1,
        "phi": 0.2,
        "jetIdx": -1,
    }]])
    variation_state = _make_variation_state(
        jets=jets,
        fakeable_leptons=fakeable,
        fakeable_sorted=fakeable_sorted,
        taus=taus,
        cleaning_taus=None,
    )
    events = ak.Array({"event": [1]})
    dataset = _make_dataset_context(is_data=False)

    updated_state = processor._apply_object_variations(events, dataset, variation_state, None)

    updated_tau_pt = ak.to_list(updated_state.objects.taus.pt)
    assert updated_tau_pt == [[32.0]]
    assert ak.to_list(updated_state.objects.n_loose_taus) == [1]
    assert ak.to_list(updated_state.objects.tau0.pt) == [32.0]


def test_lepton_ordering_is_preserved(monkeypatch):
    _patch_common(monkeypatch)
    processor = _make_processor()
    jets = ak.Array([
        [
            {
                "pt": 60.0,
                "eta": 0.1,
                "phi": 0.1,
                "mass": 9.0,
                "rawFactor": 0.1,
                "jetId": 6,
                "matched_gen": {"pt": 55.0},
            },
            {
                "pt": 45.0,
                "eta": 2.8,
                "phi": -0.2,
                "mass": 6.0,
                "rawFactor": 0.1,
                "jetId": 6,
                "matched_gen": {"pt": 40.0},
            },
            {
                "pt": 25.0,
                "eta": 1.5,
                "phi": 0.3,
                "mass": 5.0,
                "rawFactor": 0.1,
                "jetId": 6,
                "matched_gen": {"pt": 20.0},
            },
        ]
    ])
    fakeable = ak.Array([[{
        "pt": 35.0,
        "eta": 0.1,
        "phi": 0.2,
        "mass": 0.1,
        "jetIdx": -1,
    }]])
    fakeable_sorted = ak.Array([[
        {"pt": 40.0, "conept": 40.0, "jetIdx": -1},
        {"pt": 30.0, "conept": 30.0, "jetIdx": -1},
        {"pt": 20.0, "conept": 20.0, "jetIdx": -1},
    ]])
    variation_state = _make_variation_state(
        jets=jets, fakeable_leptons=fakeable, fakeable_sorted=fakeable_sorted
    )
    events = ak.Array({"event": [1]})
    dataset = _make_dataset_context(is_data=True)

    updated_state = processor._apply_object_variations(events, dataset, variation_state, None)

    assert ak.to_list(updated_state.l0.pt) == [40.0]
    assert ak.to_list(updated_state.l1.pt) == [30.0]
    assert ak.to_list(updated_state.l2.pt) == [20.0]
