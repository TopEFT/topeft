from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from topeft.modules.systematics import register_trigger_sf_weight


class DummyWeights:
    def __init__(self):
        self.add_calls = []

    def add(self, label, *values):
        self.add_calls.append((label, values))


def _patch_trigger_sf(monkeypatch, central=0.95, up=1.05, down=0.85):
    def fake_get_trigger_sf(year, events, lep0, lep1):
        events.trigger_sf = [central]
        events.trigger_sfUp = [up]
        events.trigger_sfDown = [down]

    monkeypatch.setattr("topeft.modules.corrections.GetTriggerSF", fake_get_trigger_sf)


def test_register_trigger_sf_weight_nominal(monkeypatch):
    _patch_trigger_sf(monkeypatch)

    events = SimpleNamespace()
    weights = DummyWeights()

    register_trigger_sf_weight(
        weights,
        year="2018",
        events=events,
        lepton0=SimpleNamespace(pt=[30.0]),
        lepton1=SimpleNamespace(pt=[25.0]),
        label="trigger",
        variation_descriptor={"has_systematics": False, "variation_base": "trigger_sf"},
    )

    assert len(weights.add_calls) == 1
    label, values = weights.add_calls[0]
    assert label == "trigger"
    assert len(values) == 1
    assert values[0] == [0.95]


def test_register_trigger_sf_weight_with_variations(monkeypatch):
    _patch_trigger_sf(monkeypatch, central=1.0, up=1.1, down=0.9)

    events = SimpleNamespace()
    weights = DummyWeights()

    register_trigger_sf_weight(
        weights,
        year="2018",
        events=events,
        lepton0=SimpleNamespace(pt=[35.0]),
        lepton1=SimpleNamespace(pt=[28.0]),
        label="trigger",
        variation_descriptor={
            "has_systematics": True,
            "variation_base": "trigger_sf",
            "variation_name": "trigger_sf_2018",
        },
    )

    assert len(weights.add_calls) == 1
    label, values = weights.add_calls[0]
    assert label == "trigger"
    assert len(values) == 3
    central, up, down = values
    assert central == [1.0]
    assert up == [1.1]
    assert down == [0.9]
    assert up is not events.trigger_sfUp
    assert down is not events.trigger_sfDown
