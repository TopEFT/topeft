import pytest

np = pytest.importorskip("numpy")
ak = pytest.importorskip("awkward")

import sys
import types

if "topcoffea.modules.corrections" not in sys.modules:
    corrections_stub = types.ModuleType("topcoffea.modules.corrections")
    corrections_stub.AttachPSWeights = lambda *args, **kwargs: None  # type: ignore[assignment]
    corrections_stub.AttachScaleWeights = lambda *args, **kwargs: None  # type: ignore[assignment]
    sys.modules["topcoffea.modules.corrections"] = corrections_stub

from analysis.topeft_run2 import sow_processor


class DummyEvents:
    def __init__(self, data, metadata):
        self._data = dict(data)
        self.metadata = dict(metadata)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name) from None

    def __len__(self):
        return len(self._data["genWeight"])


@pytest.fixture()
def sample_processor(monkeypatch):
    samples = {
        "dummy": {
            "isData": False,
            "WCnames": [],
            "nSumOfWeights": 6.0,
            "nEvents": 3,
            "nGenEvents": 3,
            "xsec": 6.0,
        }
    }

    def fake_ps_weights(events):
        weights = {
            "ISRUp": ak.Array([1.1, 1.1, 1.1]),
            "ISRDown": ak.Array([0.9, 0.9, 0.9]),
            "FSRUp": ak.Array([1.05, 1.05, 1.05]),
            "FSRDown": ak.Array([0.95, 0.95, 0.95]),
        }
        for key, value in weights.items():
            events[key] = value

    def fake_scale_weights(events):
        weights = {
            "renormUp": ak.Array([1.2, 1.2, 1.2]),
            "renormDown": ak.Array([0.8, 0.8, 0.8]),
            "factUp": ak.Array([1.15, 1.15, 1.15]),
            "factDown": ak.Array([0.85, 0.85, 0.85]),
            "renormfactUp": ak.Array([1.25, 1.25, 1.25]),
            "renormfactDown": ak.Array([0.75, 0.75, 0.75]),
        }
        for key, value in weights.items():
            events[key] = value

    monkeypatch.setattr(
        sow_processor.corrections,
        "AttachPSWeights",
        fake_ps_weights,
    )
    monkeypatch.setattr(
        sow_processor.corrections,
        "AttachScaleWeights",
        fake_scale_weights,
    )

    return sow_processor.AnalysisProcessor(
        samples=samples,
        wc_names_lst=[],
        do_errors=False,
        dtype=np.float64,
        debug=True,
    )


def test_sow_processor_builds_histograms_and_metadata(sample_processor):
    gen_weight = ak.Array([1.0, 2.0, 3.0])
    events = DummyEvents(
        {"genWeight": gen_weight},
        metadata={"dataset": "dummy"},
    )

    output = sample_processor.process(events)

    sow_hist = output["sow"]["SumOfWeights"]
    assert np.isclose(np.sum(sow_hist.values()), np.sum(gen_weight))

    sow_isr_up = output["sow"]["SumOfWeights_ISRUp"]
    expected_isr_up = np.sum(gen_weight * np.array([1.1, 1.1, 1.1]))
    assert np.isclose(np.sum(sow_isr_up.values()), expected_isr_up)

    metadata = output["metadata"]
    assert tuple(metadata["weight_variations"]) == tuple(sow_processor._WEIGHT_VARIATIONS.keys())

    dataset_meta = metadata["datasets"]["dummy"]
    assert dataset_meta["processed_events"] == 3
    assert dataset_meta["cross_section"] == pytest.approx(6.0)
    assert dataset_meta["normalization_factor"] == pytest.approx(1.0)

    totals = dataset_meta["totals"]
    assert totals["nom"] == pytest.approx(np.sum(gen_weight))
    assert totals["ISRUp"] == pytest.approx(expected_isr_up)

    metadata_keys = dataset_meta["metadata_keys"]
    assert metadata_keys["nom"] == "nSumOfWeights"
    assert metadata_keys["ISRUp"] == "nSumOfWeights_ISRUp"

    sum_map = dataset_meta["sum_of_weights"]
    assert sum_map["nom"]["histogram"] == "SumOfWeights"
    assert sum_map["ISRUp"]["histogram"] == "SumOfWeights_ISRUp"

    norm_totals = dataset_meta["normalized_totals"]
    assert norm_totals["nom"] == pytest.approx(np.sum(gen_weight))
    assert norm_totals["ISRUp"] == pytest.approx(expected_isr_up)

