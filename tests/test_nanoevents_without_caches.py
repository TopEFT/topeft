import importlib
import sys
import types
import uuid

import awkward as ak
from coffea.nanoevents import BaseSchema, NanoEventsFactory
from analysis.topeft_run2.nanoevents_helpers import ensure_factory_mode


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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metadata = {
            "uuid": uuid.uuid4(),
            "num_rows": len(next(iter(kwargs.values()))) if kwargs else 0,
            "object_path": "Events",
        }


def _build_minimal_processor(monkeypatch):
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

    base_objects = ap.BaseObjectState(
        met=ak.Array([[{"pt": 35.0, "phi": 0.0, "sumEt": 40.0}]]),
        electrons=ak.Array([[[]]]),
        muons=ak.Array([[[]]]),
        taus=ak.Array([[[]]]),
        jets=ak.Array([[
            {
                "pt": 45.0,
                "eta": 0.1,
                "phi": 0.2,
                "mass": 6.0,
                "rawFactor": 0.0,
                "jetId": 6,
            }
        ]]),
        loose_leptons=ak.Array([[[]]]),
        fakeable_leptons=ak.Array([[[]]]),
        fakeable_sorted=ak.Array([[[]]]),
        jets_rho=ak.ones_like(ak.Array([[0.5]])),
        lepton_selection=object(),
        cleaning_taus=None,
        n_loose_taus=None,
        tau0=None,
    )

    variation_objects = ap.VariationObjects.from_base(base_objects)
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
        lumi_mask=ak.Array([True]),
        lumi=1.0,
        eft_coeffs=None,
        eft_w2_coeffs=None,
    )

    def _build_dataset_context(self, events):
        return dataset

    def _select_base_objects(self, events, dataset):
        return base_objects

    def _build_variation_requests(self):
        return [ap.VariationRequest(variation=None, histogram_label="nominal")]

    def _initialize_variation_state(
        self, request, base_objects, dataset, *_args, **_kwargs
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
            objects=variation_objects,
            lepton_selection=base_objects.lepton_selection,
            jets_rho=base_objects.jets_rho,
        )

    processed = []

    def _apply_object_variations(self, events, dataset, variation_state):
        processed.append(variation_state.name)
        return variation_state

    def _register_weights_for_variation(
        self, events, dataset, variation_state, *_args, **_kwargs
    ):
        return {"weight": ak.Array([1.0])}

    def _fill_histograms_for_variation(
        self, events, dataset, variation_state, weights, *_args, **_kwargs
    ):
        weights["filled"] = True

    processor._build_dataset_context = types.MethodType(_build_dataset_context, processor)
    processor._select_base_objects = types.MethodType(_select_base_objects, processor)
    processor._build_variation_requests = types.MethodType(
        _build_variation_requests, processor
    )
    processor._initialize_variation_state = types.MethodType(
        _initialize_variation_state, processor
    )
    processor._apply_object_variations = types.MethodType(
        _apply_object_variations, processor
    )
    processor._register_weights_for_variation = types.MethodType(
        _register_weights_for_variation, processor
    )
    processor._fill_histograms_for_variation = types.MethodType(
        _fill_histograms_for_variation, processor
    )

    return processor, processed


def test_process_handles_nanoevents_without_cache(monkeypatch):
    events = NanoEventsFactory.from_preloaded(
        _DummyMapping(event=ak.Array([1, 2])), schemaclass=BaseSchema
    ).events()

    assert not hasattr(events, "caches")

    processor, processed_variations = _build_minimal_processor(monkeypatch)

    result = processor.process(events)

    assert result == {"status": "ok"}
    assert processed_variations == ["nominal"]


def test_factory_mode_is_set_for_numpy(monkeypatch):
    factory = ensure_factory_mode(
        NanoEventsFactory.from_preloaded(
            _DummyMapping(event=ak.Array([1.0])), schemaclass=BaseSchema
        ),
        mode="numpy",
    )

    events = factory.events()

    processor, processed_variations = _build_minimal_processor(monkeypatch)

    result = processor.process(events)

    assert result == {"status": "ok"}
    assert processed_variations == ["nominal"]
    assert getattr(factory, "_mode", None) == "numpy"
