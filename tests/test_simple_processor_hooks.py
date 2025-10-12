import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_test_stubs():
    if "numpy" not in sys.modules:
        np_module = types.ModuleType("numpy")
        np_module.seterr = lambda **_: None
        np_module.float32 = float
        np_module.ndarray = object
        sys.modules["numpy"] = np_module

    if "awkward" not in sys.modules:
        ak_module = types.ModuleType("awkward")
        ak_module.Array = list
        ak_module.num = lambda array: array  # type: ignore[attr-defined]
        ak_module.concatenate = lambda arrays, axis=0: sum(arrays, [])  # type: ignore[attr-defined]
        ak_module.argmax = lambda array, axis=None, keepdims=False: 0  # type: ignore[attr-defined]
        sys.modules["awkward"] = ak_module

    coffea_pkg = sys.modules.setdefault("coffea", types.ModuleType("coffea"))

    hist_module = types.ModuleType("coffea.hist")

    class _DummyAxis:
        def __init__(self, *_, **__):
            pass

    class _DummyHist:
        def __init__(self, *_, **__):
            pass

        def fill(self, **__):
            pass

    hist_module.Hist = _DummyHist
    hist_module.Cat = _DummyAxis
    hist_module.Bin = _DummyAxis
    sys.modules["coffea.hist"] = hist_module
    coffea_pkg.hist = hist_module  # type: ignore[attr-defined]

    processor_module = types.ModuleType("coffea.processor")

    class _DummyProcessorABC:
        pass

    def _dict_accumulator(mapping):
        class _DummyAccumulator(dict):
            def identity(self):
                return dict(self)

        return _DummyAccumulator(mapping)

    processor_module.ProcessorABC = _DummyProcessorABC
    processor_module.dict_accumulator = _dict_accumulator
    sys.modules["coffea.processor"] = processor_module
    coffea_pkg.processor = processor_module  # type: ignore[attr-defined]

    analysis_tools_module = types.ModuleType("coffea.analysis_tools")

    class _DummyPackedSelection:
        def add(self, *_, **__):
            pass

        def all(self, *_):
            return []

    analysis_tools_module.PackedSelection = _DummyPackedSelection
    sys.modules["coffea.analysis_tools"] = analysis_tools_module

    topcoffea_pkg = sys.modules.setdefault("topcoffea", types.ModuleType("topcoffea"))
    modules_pkg = sys.modules.setdefault("topcoffea.modules", types.ModuleType("topcoffea.modules"))
    topcoffea_pkg.modules = modules_pkg  # type: ignore[attr-defined]

    objects_module = types.ModuleType("topcoffea.modules.objects")
    objects_module.isClean = lambda *_, **__: True  # type: ignore[attr-defined]
    sys.modules["topcoffea.modules.objects"] = objects_module

    selection_module = types.ModuleType("topcoffea.modules.selection")
    sys.modules["topcoffea.modules.selection"] = selection_module

    hist_eft_module = types.ModuleType("topcoffea.modules.HistEFT")

    class _DummyHistEFT:
        def __init__(self, *_, **__):
            pass

        def fill(self, **__):
            pass

    hist_eft_module.HistEFT = _DummyHistEFT
    sys.modules["topcoffea.modules.HistEFT"] = hist_eft_module

    eft_helper_module = types.ModuleType("topcoffea.modules.eft_helper")
    eft_helper_module.calc_w2_coeffs = lambda *_, **__: None  # type: ignore[attr-defined]
    eft_helper_module.remap_coeffs = lambda *args, **__: args[2] if len(args) > 2 else None  # type: ignore[attr-defined]
    sys.modules["topcoffea.modules.eft_helper"] = eft_helper_module


def _build_processor(monkeypatch, is_data):
    _install_test_stubs()
    from analysis.training.simple_processor import AnalysisProcessor, ProcessingContext

    sample_key = "SampleData" if is_data else "SampleMC"
    samples = {
        sample_key: {
            "year": "2018",
            "xsec": 1.0,
            "nSumOfWeights": 1.0,
            "isData": is_data,
            "WCnames": [],
        }
    }

    proc = AnalysisProcessor(samples=samples)

    context = ProcessingContext(
        events=types.SimpleNamespace(),
        dataset=sample_key,
        year="2018",
        xsec=1.0,
        sow=1.0,
        is_data=is_data,
        is_eft=False,
        hist_output={},
    )

    call_order = []

    def fake_prepare_context(self, events):
        call_order.append("prepare_context")
        return context

    def fake_prepare_collections(self, ctx):
        call_order.append("prepare_collections")
        assert ctx is context
        assert ctx.is_data is is_data

    def fake_build_selections(self, ctx):
        call_order.append("build_selections")

    def fake_compute_weights(self, ctx):
        call_order.append("compute_weights")

    def fake_fill_histograms(self, ctx):
        call_order.append("fill_histograms")
        ctx.hist_output = {"completed": True, "is_data": ctx.is_data}

    monkeypatch.setattr(AnalysisProcessor, "_prepare_context", fake_prepare_context)
    monkeypatch.setattr(AnalysisProcessor, "_prepare_collections", fake_prepare_collections)
    monkeypatch.setattr(AnalysisProcessor, "_build_selections", fake_build_selections)
    monkeypatch.setattr(AnalysisProcessor, "_compute_weights", fake_compute_weights)
    monkeypatch.setattr(AnalysisProcessor, "_fill_histograms", fake_fill_histograms)

    return proc, context, call_order


@pytest.mark.parametrize("is_data", [True, False])
def test_process_invokes_pipeline_stages(monkeypatch, is_data):
    proc, context, call_order = _build_processor(monkeypatch, is_data)

    result = proc.process(events=types.SimpleNamespace(metadata={"dataset": context.dataset}))

    assert result == {"completed": True, "is_data": is_data}
    assert call_order == [
        "prepare_context",
        "prepare_collections",
        "build_selections",
        "compute_weights",
        "fill_histograms",
    ]
