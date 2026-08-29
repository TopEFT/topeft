import awkward as ak
import numpy as np

from topeft.modules import corrections


class _RecordingCorrection:
    def __init__(self, correction=None):
        self.correction = correction
        self.calls = []

    def evaluate(self, *args):
        assert not any(isinstance(arg, ak.highlevel.Array) for arg in args)
        self.calls.append(args)
        if self.correction is None:
            numeric = next(arg for arg in args if isinstance(arg, np.ndarray))
            return numeric + 1.0
        return self.correction.evaluate(*args)


class _RecordingCorrectionSet:
    def __init__(self, correction_set, calls):
        self.correction_set = correction_set
        self.calls = calls

    def __getitem__(self, key):
        correction = _RecordingCorrection(self.correction_set[key])
        self.calls.append((key, correction))
        return correction

    @property
    def compound(self):
        return _RecordingCompoundSet(self.correction_set.compound, self.calls)


class _RecordingCompoundSet:
    def __init__(self, compound_set, calls):
        self.compound_set = compound_set
        self.calls = calls

    def __getitem__(self, key):
        correction = _RecordingCorrection(self.compound_set[key])
        self.calls.append((key, correction))
        return correction


def _recording_factory(monkeypatch):
    original_correction_set = corrections.correctionlib.CorrectionSet
    recorded = []

    class RecordingFactory:
        @staticmethod
        def from_file(path):
            return _RecordingCorrectionSet(
                original_correction_set.from_file(path), recorded
            )

    monkeypatch.setattr(corrections.correctionlib, "CorrectionSet", RecordingFactory)
    return recorded


def _assert_recorded_numpy_only(recorded):
    calls = [call for _, correction in recorded for call in correction.calls]
    assert calls
    for call in calls:
        assert not any(isinstance(arg, ak.highlevel.Array) for arg in call)
    return calls


def test_numpy_boundary_preserves_flat_values_and_scalar_categories():
    jagged = ak.Array([[1.5], [], [2.5, 3.5]])
    flat = ak.flatten(jagged)
    correction = _RecordingCorrection()

    result = corrections._evaluate_correctionlib(
        correction, "nominal", flat, np.int64(7)
    )

    category, actual, integer = correction.calls[0]
    assert category == "nominal"
    assert integer == np.int64(7)
    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, ak.to_numpy(flat))
    np.testing.assert_array_equal(result, np.array([2.5, 3.5, 4.5]))
    assert ak.to_list(ak.unflatten(result, ak.num(jagged))) == [
        [2.5],
        [],
        [3.5, 4.5],
    ]


def test_numpy_boundary_preserves_empty_numeric_vectors():
    correction = _RecordingCorrection()

    result = corrections._evaluate_correctionlib(
        correction, ak.Array([]), "up"
    )

    assert isinstance(correction.calls[0][0], np.ndarray)
    assert correction.calls[0][0].size == 0
    assert correction.calls[0][1] == "up"
    assert result.size == 0


def test_attach_electron_sf_run2_uses_numpy_and_preserves_structure(monkeypatch):
    recorded = _recording_factory(monkeypatch)

    electrons = ak.Array(
        [
            [{"pt": 12.0, "eta": 0.1, "phi": 0.2, "pdgId": 11}],
            [],
            [
                {"pt": 25.0, "eta": -1.2, "phi": -0.4, "pdgId": -11},
                {"pt": 80.0, "eta": 2.0, "phi": 2.1, "pdgId": 11},
            ],
        ]
    )
    original_counts = ak.to_list(ak.num(electrons.pt))

    corrections.AttachElectronSF(
        electrons, "2018", looseWP="none", useRun3MVA=False
    )

    calls = _assert_recorded_numpy_only(recorded)
    assert len(calls) == 6
    assert {call[1] for call in calls} == {"sf", "sfup", "sfdown"}
    for call in calls:
        assert call[0] == "2018"
        assert isinstance(call[2], str)
        assert isinstance(call[3], np.ndarray)
        assert isinstance(call[4], np.ndarray)
        np.testing.assert_array_equal(call[3], np.array([0.1, -1.2, 2.0]))

    for field in (
        "sf_nom_2l_elec",
        "sf_hi_2l_elec",
        "sf_lo_2l_elec",
        "sf_nom_3l_elec",
        "sf_hi_3l_elec",
        "sf_lo_3l_elec",
    ):
        assert ak.to_list(ak.num(electrons[field])) == original_counts
        assert np.all(np.isfinite(ak.to_numpy(ak.flatten(electrons[field]))))


def test_attach_electron_sf_data_and_mc_share_the_same_function_contract():
    assert "isData" not in corrections.AttachElectronSF.__code__.co_varnames


def test_run3_veto_map_uses_numpy_and_preserves_irregular_counts(monkeypatch):
    recorded = _recording_factory(monkeypatch)
    jets = ak.Array(
        [
            [{"eta": 0.1, "phi": 0.2}],
            [],
            [{"eta": -2.0, "phi": -1.0}, {"eta": 6.0, "phi": 4.0}],
        ]
    )

    result = corrections.ApplyJetVetoMaps(jets, "2022")

    calls = _assert_recorded_numpy_only(recorded)
    assert len(calls) == 1
    assert calls[0][0] == "jetvetomap"
    np.testing.assert_array_equal(calls[0][1], np.array([0.1, -2.0, 5.19]))
    np.testing.assert_array_equal(calls[0][2], np.array([0.2, -1.0, 3.14159]))
    assert len(result) == 3


def test_muon_sf_run2_and_run3_use_numpy_and_preserve_structure(monkeypatch):
    recorded = _recording_factory(monkeypatch)
    muons = ak.Array(
        [
            [{"pt": 12.0, "eta": 0.1, "pdgId": 13}],
            [],
            [
                {"pt": 25.0, "eta": -1.2, "pdgId": -13},
                {"pt": 80.0, "eta": 2.0, "pdgId": 13},
            ],
        ]
    )
    counts = ak.to_list(ak.num(muons.pt))

    corrections.AttachMuonSF(muons, "2018", useRun3MVA=False)
    _assert_recorded_numpy_only(recorded)
    assert ak.to_list(ak.num(muons.sf_nom_2l_muon)) == counts

    recorded.clear()
    corrections.AttachMuonSF(muons, "2022", useRun3MVA=False)
    _assert_recorded_numpy_only(recorded)
    assert ak.to_list(ak.num(muons.sf_hi_3l_muon)) == counts


def test_run3_electron_sf_and_energy_corrections_use_numpy(monkeypatch):
    recorded = _recording_factory(monkeypatch)
    electrons = ak.Array(
        [
            [
                {
                    "pt": 25.0,
                    "eta": 0.1,
                    "phi": 0.2,
                    "pdgId": 11,
                    "deltaEtaSC": 0.1,
                    "r9": 0.95,
                    "seedGain": 12,
                }
            ],
            [],
            [
                {
                    "pt": 80.0,
                    "eta": -1.2,
                    "phi": -0.4,
                    "pdgId": -11,
                    "deltaEtaSC": -1.2,
                    "r9": 0.9,
                    "seedGain": 12,
                }
            ],
        ]
    )
    counts = ak.to_list(ak.num(electrons.pt))

    corrections.AttachElectronSF(
        electrons, "2022", looseWP="none", useRun3MVA=False
    )
    _assert_recorded_numpy_only(recorded)
    assert ak.to_list(ak.num(electrons.sf_nom_2l_elec)) == counts

    recorded.clear()
    corrections.AttachElectronCorrections(
        electrons, ak.Array([355200, 355201, 355202]), "2022", isData=True
    )
    _assert_recorded_numpy_only(recorded)
    assert ak.to_list(ak.num(electrons.pt)) == counts

    electrons["pt"] = electrons.pt_raw
    recorded.clear()
    corrections.AttachElectronCorrections(
        electrons, ak.Array([1, 2, 3]), "2022", isData=False
    )
    calls = _assert_recorded_numpy_only(recorded)
    assert {call[0] for call in calls} == {"smear", "esmear", "escale"}
    assert ak.to_list(ak.num(electrons.pt_scale_up)) == counts


def test_run3_tau_sf_uses_numpy_and_preserves_structure(monkeypatch):
    recorded = _recording_factory(monkeypatch)
    events = ak.Array([{"seed": 0}, {"seed": 1}, {"seed": 2}])
    taus = ak.Array(
        [
            [
                {
                    "pt": 40.0,
                    "mass": 1.7,
                    "eta": 0.2,
                    "decayMode": 0,
                    "genPartFlav": 5,
                    "isMedium": 1,
                    "iseTight": 1,
                    "ismTight": 1,
                    "idDeepTau2018v2p5VSmu": 4,
                }
            ],
            [],
            [
                {
                    "pt": 35.0,
                    "mass": 1.5,
                    "eta": -1.1,
                    "decayMode": 1,
                    "genPartFlav": 1,
                    "isMedium": 1,
                    "iseTight": 1,
                    "ismTight": 1,
                    "idDeepTau2018v2p5VSmu": 4,
                }
            ],
        ]
    )

    corrections.AttachTauSF(events, taus, "2022", vsJetWP="Medium")

    _assert_recorded_numpy_only(recorded)
    assert ak.to_list(ak.num(taus.sf_tau_real)) == [1, 0, 1]
    assert len(events.sf_2l_taus_real) == 3


def test_run3_fake_rate_uses_numpy_and_preserves_structure(monkeypatch):
    recorded = _recording_factory(monkeypatch)
    leptons = ak.Array(
        [
            [{"pt": 25.0, "conept": 25.0, "eta": 0.2, "pdgId": 13}],
            [],
            [
                {"pt": 45.0, "conept": 45.0, "eta": -1.1, "pdgId": -13}
            ],
        ]
    )

    corrections.AttachPerLeptonFR(leptons, "Muon", "2022")

    calls = _assert_recorded_numpy_only(recorded)
    assert len(calls) == len(corrections.ffSysts)
    assert {call[2] for call in calls} == set(corrections.ffSysts)
    assert ak.to_list(ak.num(leptons.fakefactor)) == [1, 0, 1]
