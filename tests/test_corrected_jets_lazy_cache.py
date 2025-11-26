import types

import awkward as ak

from topeft.modules.corrections import ApplyJetSystematics, build_corrected_jets


class _DummyJetFactory:
    def __init__(self):
        self.last_lazy_cache = None

    def build(self, jets, lazy_cache=None):
        self.last_lazy_cache = lazy_cache

        corrected = ak.with_field(jets, jets.pt * 1.1, "pt")
        corrected = ak.with_field(corrected, jets.mass * 1.1, "mass")

        return corrected


def test_corrected_jets_use_lazy_cache_and_feed_systematics():
    jets = ak.Array([[{"pt": 50.0, "mass": 5.0}]])

    dummy_cache = object()
    dummy_events = types.SimpleNamespace(caches=[dummy_cache])
    jets._events = dummy_events

    jet_factory = _DummyJetFactory()

    corrected = build_corrected_jets(jet_factory, jets, lazy_cache=dummy_cache)

    assert jet_factory.last_lazy_cache is dummy_cache
    assert ak.all(corrected.pt != jets.pt)
    assert ak.all(ak.isclose(corrected.pt, jets.pt * 1.1))
    assert ak.all(ak.isclose(corrected.mass, jets.mass * 1.1))

    nominal = ApplyJetSystematics("2018", corrected, "nominal")
    assert ak.all(ak.isclose(nominal.pt, corrected.pt))
