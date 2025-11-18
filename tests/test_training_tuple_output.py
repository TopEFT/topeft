"""Integration tests for the training processor histogram structure."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Tuple

import awkward as ak
import numpy as np
import pytest
from hist import Hist, axis, storage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_training_stubs() -> None:
    """Provide lightweight shims for the topcoffea modules used by the processor."""

    topcoffea_pkg = sys.modules.setdefault("topcoffea", types.ModuleType("topcoffea"))
    modules_pkg = sys.modules.setdefault("topcoffea.modules", types.ModuleType("topcoffea.modules"))
    topcoffea_pkg.modules = modules_pkg  # type: ignore[attr-defined]

    objects_module = sys.modules.setdefault("topcoffea.modules.objects", types.ModuleType("topcoffea.modules.objects"))

    def _always_clean(jets, leptons, drmin=0.4):  # pragma: no cover - exercised indirectly
        return ak.ones_like(jets.pt, dtype=bool)

    objects_module.isClean = _always_clean  # type: ignore[attr-defined]

    sys.modules.setdefault("topcoffea.modules.selection", types.ModuleType("topcoffea.modules.selection"))

    eft_helper_module = sys.modules.setdefault(
        "topcoffea.modules.eft_helper", types.ModuleType("topcoffea.modules.eft_helper")
    )
    eft_helper_module.remap_coeffs = lambda *_args, **_kwargs: _args[2] if len(_args) > 2 else None  # type: ignore[attr-defined]
    eft_helper_module.calc_w2_coeffs = lambda coeffs, *_: coeffs  # type: ignore[attr-defined]

    class _DummyHistEFT(Hist):
        def __init__(self, dense_axis, wc_names=None, label="Events"):
            super().__init__(
                dense_axis,
                axis.Integer(0, 1, name="quadratic_term"),
                storage=storage.Double(),
                label=label,
            )

        def fill(self, eft_coeff=None, eft_err_coeff=None, **kwargs):  # pragma: no cover - delegates to hist
            converted = {}
            length = None
            for name, value in kwargs.items():
                data = value
                if isinstance(data, ak.Array):
                    while data.ndim > 1:
                        data = ak.flatten(data, axis=-1)
                    data = ak.to_numpy(data)
                data = np.asarray(data)
                if data.ndim == 0:
                    if length is None:
                        length = 1
                    data = np.full(length, data)
                else:
                    length = len(data)
                converted[name] = data
            if length is None:
                length = 1
            converted.setdefault("quadratic_term", np.zeros(length, dtype=int))
            return super().fill(**converted)

    hist_eft_module = types.ModuleType("topcoffea.modules.HistEFT")
    hist_eft_module.HistEFT = _DummyHistEFT
    sys.modules["topcoffea.modules.HistEFT"] = hist_eft_module
    sys.modules["topcoffea.modules.histEFT"] = hist_eft_module


@pytest.fixture()
def training_processor(monkeypatch):
    _install_training_stubs()
    module = importlib.import_module("analysis.training.simple_processor")

    original_num = module.ak.num

    def patched_num(array, *args, **kwargs):  # pragma: no cover - exercised indirectly
        if not args and not kwargs:
            try:
                fields = module.ak.fields(array)
            except Exception:  # pragma: no cover - defensive
                fields = []
            if "pt" in fields:
                return original_num(array["pt"], *args, **kwargs)
        return original_num(array, *args, **kwargs)

    monkeypatch.setattr(module.ak, "num", patched_num)

    samples = {
        "SampleMC": {
            "year": "2018",
            "xsec": 1.0,
            "nSumOfWeights": 1.0,
            "isData": False,
            "WCnames": [],
        }
    }
    processor = module.AnalysisProcessor(samples=samples)
    return processor, module


@pytest.fixture()
def synthetic_events() -> ak.Array:
    events = ak.Array(
        {
            "GenPart": {
                "pdgId": [[11, -11, 13, -13]] * 3,
                "pt": [[35.0, 32.0, 40.0, 25.0], [30.0, 28.0, 33.0, 31.0], [26.0, 24.0, 30.0, 28.0]],
                "eta": [[0.2, -0.3, 0.1, -0.2], [0.4, -0.1, 0.2, -0.2], [0.3, -0.25, 0.15, -0.15]],
                "phi": [[0.1, -0.1, 0.2, -0.2], [0.3, -0.3, 0.25, -0.25], [0.4, -0.4, 0.35, -0.35]],
                "mass": [[0.0, 0.0, 0.0, 0.0]] * 3,
            },
            "GenJet": {
                "pt": [[45.0, 42.0, 38.0], [44.0, 41.0, 36.0], [48.0, 46.0, 40.0]],
                "eta": [[0.2, -0.3, 0.1], [0.4, -0.1, 0.2], [0.3, -0.2, 0.15]],
                "phi": [[0.5, -0.5, 0.4], [0.6, -0.6, 0.5], [0.7, -0.7, 0.55]],
                "mass": [[10.0, 10.0, 10.0]] * 3,
            },
            "MET": {"pt": [100.0, 80.0, 90.0]},
            "run": [1, 2, 3],
            "luminosityBlock": [10, 11, 12],
            "event": [1000, 1001, 1002],
        }
    )
    events.metadata = {"dataset": "SampleMC"}
    return events


@pytest.fixture()
def training_result(training_processor, synthetic_events):
    processor, module = training_processor
    accumulator = processor.process(synthetic_events)
    return accumulator, module, processor


def _assert_tuple_key(key: Tuple[str, str, str, str, str]) -> None:
    assert isinstance(key, tuple)
    assert len(key) == 5
    variable, channel, application, sample, systematic = key
    assert variable in {"counts", "njets", "j0pt", "j0eta", "l0pt"}
    assert isinstance(channel, str) and channel
    assert application == "inclusive"
    assert sample == "SampleMC"
    assert systematic == "nominal"


def test_training_accumulator_uses_tuple_keys(training_result):
    accumulator, _module, _processor = training_result
    assert accumulator
    for key in accumulator.keys():
        _assert_tuple_key(key)


def test_training_histograms_have_no_categorical_axes(training_result):
    accumulator, _module, processor = training_result
    for histogram in accumulator.values():
        for axis_obj in histogram.axes:
            assert not isinstance(axis_obj, axis.StrCategory)

    counts_key = processor._build_histogram_key("counts", "SampleMC")
    counts_hist = accumulator[counts_key]
    assert pytest.approx(counts_hist.values(flow=False).sum()) == 3.0
