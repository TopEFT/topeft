import importlib
import sys
import types

import awkward as ak
import numpy as np

from coffea.nanoevents import BaseSchema, NanoEventsFactory

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
            "uuid": "cacheless-jetmet-test",
            "num_rows": len(next(iter(kwargs.values()))) if kwargs else 0,
            "object_path": "Events",
        }


def _build_processor():
    processor = ap.AnalysisProcessor.__new__(ap.AnalysisProcessor)
    processor._debug_logging = False
    processor._debug = lambda *_, **__: None
    processor._available_systematics = {
        "object": (),
        "weight": (),
        "theory": (),
        "data_weight": (),
    }
    processor._available_systematics_sets = {
        key: set(values) for key, values in processor._available_systematics.items()
    }
    processor._accumulator = {"status": "ok"}
    processor._var_def = set()
    processor.tau_h_analysis = False
    processor.fwd_analysis = False
    processor.offZ_3l_split = False
    processor._channel_features = frozenset()

    def _noop_event_selection(*args, **kwargs):
        return None

    ap.te_es.add1lMaskAndSFs = _noop_event_selection
    ap.te_es.add2lMaskAndSFs = _noop_event_selection
    ap.te_es.add3lMaskAndSFs = _noop_event_selection
    ap.te_es.add4lMaskAndSFs = _noop_event_selection
    ap.te_es.addLepCatMasks = _noop_event_selection

    base_objects = ap.BaseObjectState(
        met=ak.Array(
            [
                [
                    {
                        "pt": np.float32(50.0),
                        "phi": np.float32(0.0),
                        "sumEt": np.float32(55.0),
                        "MetUnclustEnUpDeltaX": np.float32(0.0),
                        "MetUnclustEnUpDeltaY": np.float32(0.0),
                    }
                ]
            ]
        ),
        electrons=ak.Array([[[]]]),
        muons=ak.Array([[[]]]),
        taus=ak.Array([[[]]]),
        jets=ak.Array(
            [
                [
                    {
                        "pt": np.float32(45.0),
                        "eta": np.float32(0.2),
                        "phi": np.float32(0.1),
                        "mass": np.float32(9.0),
                        "rawFactor": np.float32(0.1),
                        "jetId": 6,
                        "area": np.float32(0.5),
                    }
                ]
            ]
        ),
        loose_leptons=ak.Array([[[]]]),
        fakeable_leptons=ak.Array(
            [
                [
                    {
                        "pt": np.float32(0.0),
                        "eta": np.float32(0.0),
                        "phi": np.float32(0.0),
                        "mass": np.float32(0.0),
                        "jetIdx": -1,
                    }
                ]
            ]
        ),
        fakeable_sorted=ak.Array(
            [
                [
                    {
                        "pt": np.float32(0.0),
                        "conept": np.float32(0.0),
                        "jetIdx": -1,
                    }
                ]
            ]
        ),
        jets_rho=ak.Array([[np.float32(0.5)]]),
        lepton_selection=object(),
        cleaning_taus=None,
        n_loose_taus=None,
        tau0=None,
    )

    dataset = ap.DatasetContext(
        dataset="sample",
        trigger_dataset="sample",
        hist_axis_name="sample",
        is_data=False,
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

    def _build_dataset_context(self, events):
        return dataset

    def _select_base_objects(self, events, _dataset):
        return base_objects

    def _build_variation_requests(self):
        return [ap.VariationRequest(variation=None, histogram_label="nominal")]

    def _initialize_variation_state(
        self, request, _base_objects, _dataset, *_args, **_kwargs
    ):
        return ap.VariationState(
            request=request,
            name="nominal",
            base=None,
            variation_type=None,
            metadata={},
            object_variation="nominal",
            weight_variations=[],
            requested_data_weight_label=None,
            sow_variation_key_map={},
            sow_variations={},
            objects=ap.VariationObjects.from_base(base_objects),
            lepton_selection=base_objects.lepton_selection,
            jets_rho=base_objects.jets_rho,
        )

    processed_variations = []

    def _register_weights_for_variation(
        self, events, _dataset, variation_state, *_args, **_kwargs
    ):
        processed_variations.append(variation_state.name)
        return {"weight": ak.ones_like(events.event, dtype=np.float32)}

    def _fill_histograms_for_variation(
        self, _events, _dataset, _variation_state, weights, *_args, **_kwargs
    ):
        weights["filled"] = True

    processor._build_dataset_context = _build_dataset_context.__get__(
        processor, ap.AnalysisProcessor
    )
    processor._select_base_objects = _select_base_objects.__get__(
        processor, ap.AnalysisProcessor
    )
    processor._build_variation_requests = _build_variation_requests.__get__(
        processor, ap.AnalysisProcessor
    )
    processor._initialize_variation_state = _initialize_variation_state.__get__(
        processor, ap.AnalysisProcessor
    )
    processor._register_weights_for_variation = _register_weights_for_variation.__get__(
        processor, ap.AnalysisProcessor
    )
    processor._fill_histograms_for_variation = _fill_histograms_for_variation.__get__(
        processor, ap.AnalysisProcessor
    )

    return processor, processed_variations


def test_analysis_processor_runs_without_caches_for_jetmet():
    jets = ak.Array(
        [
            [
                {
                    "pt": np.float32(45.0),
                    "eta": np.float32(0.2),
                    "phi": np.float32(0.1),
                    "mass": np.float32(9.0),
                    "rawFactor": np.float32(0.1),
                    "jetId": 6,
                    "area": np.float32(0.5),
                }
            ]
        ]
    )
    met = ak.Array(
        [
            [
                {
                    "pt": np.float32(50.0),
                    "phi": np.float32(0.0),
                    "sumEt": np.float32(55.0),
                    "MetUnclustEnUpDeltaX": np.float32(0.0),
                    "MetUnclustEnUpDeltaY": np.float32(0.0),
                }
            ]
        ]
    )

    events = NanoEventsFactory.from_preloaded(
        _DummyMapping(event=ak.Array([1]), Jet=jets, MET=met), schemaclass=BaseSchema
    ).events()

    assert not hasattr(events, "caches")

    processor, processed_variations = _build_processor()
    result = processor.process(events)

    assert result == {"status": "ok"}
    assert processed_variations == ["nominal"]
    assert ak.to_list(events["njets"]) == [1]


def test_cache_free_corrections_stackable_with_processor():
    jets = ak.Array(
        [
            [
                {
                    "pt": np.float32(45.0),
                    "eta": np.float32(0.2),
                    "phi": np.float32(0.1),
                    "mass": np.float32(9.0),
                    "rawFactor": np.float32(0.1),
                    "jetId": 6,
                    "area": np.float32(0.5),
                }
            ]
        ]
    )
    met = ak.Array(
        [
            [
                {
                    "pt": np.float32(50.0),
                    "phi": np.float32(0.0),
                    "sumEt": np.float32(55.0),
                    "MetUnclustEnUpDeltaX": np.float32(0.0),
                    "MetUnclustEnUpDeltaY": np.float32(0.0),
                }
            ]
        ]
    )

    events = NanoEventsFactory.from_preloaded(
        _DummyMapping(event=ak.Array([1]), Jet=jets, MET=met), schemaclass=BaseSchema
    ).events()

    captures = {}

    processor, _ = _build_processor()

    def _capture_histograms(self, _events, _dataset, variation_state, weights, *_args, **_kwargs):
        captures["jets"] = variation_state.cleaned_jets
        captures["met"] = variation_state.objects.met
        weights["filled"] = True

    processor._fill_histograms_for_variation = _capture_histograms.__get__(
        processor, ap.AnalysisProcessor
    )

    result = processor.process(events)

    assert result == {"status": "ok"}
    assert "jets" in captures and "met" in captures

    stack_fn = getattr(ap.ak, "stack", None)
    if stack_fn is None:
        stack_fn = lambda arrays, axis=0: ak.concatenate(
            [ak.Array(arr)[None, ...] for arr in arrays], axis=axis
        )

    jet_pts = [captures["jets"].pt]
    if "JES_jes" in ak.fields(captures["jets"]):
        jet_pts.extend(
            [captures["jets"].JES_jes.up.pt, captures["jets"].JES_jes.down.pt]
        )
    stacked_jets = stack_fn(jet_pts, axis=0)

    met_pts = [captures["met"].pt]
    if "MET_UnclusteredEnergy" in ak.fields(captures["met"]):
        met_pts.extend(
            [
                captures["met"].MET_UnclusteredEnergy.up.pt,
                captures["met"].MET_UnclusteredEnergy.down.pt,
            ]
        )
    stacked_met = stack_fn(met_pts, axis=0)

    assert np.isfinite(ak.to_numpy(stacked_jets)).all()
    assert np.isfinite(ak.to_numpy(stacked_met)).all()
