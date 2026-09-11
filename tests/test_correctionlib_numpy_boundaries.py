import awkward as ak
import numpy as np

from topeft.modules import corrections, event_selection


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


class _ConstantCorrection:
    def __init__(self, values):
        self.values = values

    def evaluate(self, *args):
        variation = next(
            (arg for arg in args if isinstance(arg, str) and arg in self.values),
            None,
        )
        if variation is None:
            raise AssertionError(f"No configured variation found in {args!r}")
        numeric = next(arg for arg in args if isinstance(arg, np.ndarray))
        return np.full(numeric.shape, self.values[variation], dtype=float)


def _install_constant_correction_set(monkeypatch, values):
    correction_set = {"constant": _ConstantCorrection(values)}

    class ConstantCorrectionSet:
        @staticmethod
        def from_file(path):
            return correction_set

    monkeypatch.setattr(
        corrections.correctionlib, "CorrectionSet", ConstantCorrectionSet
    )
    return correction_set


def _constant_lookup(value):
    def lookup(*args):
        return ak.ones_like(args[-1], dtype=np.float64) * value

    return lookup


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
        "sf_nom_2l_elec_mva",
        "sf_hi_2l_elec_mva",
        "sf_lo_2l_elec_mva",
        "sf_nom_3l_elec_mva",
        "sf_hi_3l_elec_mva",
        "sf_lo_3l_elec_mva",
        "sf_nom_2l_elec_non_mva",
        "sf_hi_2l_elec_non_mva",
        "sf_lo_2l_elec_non_mva",
        "sf_nom_3l_elec_non_mva",
        "sf_hi_3l_elec_non_mva",
        "sf_lo_3l_elec_non_mva",
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


def test_run2_electron_3l_sf_uses_same_components_for_nominal_and_variations(
    monkeypatch,
):
    correction_set = _install_constant_correction_set(
        monkeypatch,
        {"sf": 1.10, "sfup": 1.11, "sfdown": 1.09},
    )
    correction_set["UL-Electron-ID-SF"] = correction_set.pop("constant")
    monkeypatch.setattr(
        corrections,
        "SFevaluator",
        {
            "ElecLooseSF_2018": _constant_lookup(1.20),
            "ElecLooseSF_2018_er": _constant_lookup(0.02),
            "ElecSF_2018_2lss": _constant_lookup(1.30),
            "ElecSF_2018_2lss_er": _constant_lookup(0.03),
            "ElecSF_2018_3l": _constant_lookup(1.40),
            "ElecSF_2018_3l_er": _constant_lookup(0.04),
            "ElecIsoSF_2018": _constant_lookup(1.50),
            "ElecIsoSF_2018_er": _constant_lookup(0.05),
        },
    )
    electrons = ak.Array(
        [[{"pt": 25.0, "eta": 0.2, "phi": 0.1, "pdgId": 11}]]
    )

    corrections.AttachElectronSF(
        electrons, "2018", looseWP="none", useRun3MVA=False
    )

    expected_3l = {
        "sf_nom_3l_elec": 1.10 * 1.40 * 1.20 * 1.50,
        "sf_hi_3l_elec": 1.11 * 1.44 * 1.22 * 1.55,
        "sf_lo_3l_elec": 1.09 * 1.36 * 1.18 * 1.45,
    }
    for field, expected in expected_3l.items():
        np.testing.assert_allclose(
            ak.to_numpy(ak.flatten(electrons[field])), [expected]
        )
    assert not np.isclose(expected_3l["sf_nom_3l_elec"], 1.10 * 1.40 * 1.20)

    np.testing.assert_allclose(
        ak.to_numpy(ak.flatten(electrons.sf_nom_2l_elec)),
        [1.10 * 1.30 * 1.20 * 1.50],
    )
    np.testing.assert_allclose(
        ak.to_numpy(ak.flatten(electrons.sf_hi_2l_elec)),
        [1.11 * 1.33 * 1.22 * 1.55],
    )
    np.testing.assert_allclose(
        ak.to_numpy(ak.flatten(electrons.sf_lo_2l_elec)),
        [1.09 * 1.27 * 1.18 * 1.45],
    )
    expected_components = {
        "sf_nom_2l_elec_mva": 1.30,
        "sf_hi_2l_elec_mva": 1.33,
        "sf_lo_2l_elec_mva": 1.27,
        "sf_nom_3l_elec_mva": 1.40,
        "sf_hi_3l_elec_mva": 1.44,
        "sf_lo_3l_elec_mva": 1.36,
        "sf_nom_2l_elec_non_mva": 1.10 * 1.20 * 1.50,
        "sf_hi_2l_elec_non_mva": 1.11 * 1.22 * 1.55,
        "sf_lo_2l_elec_non_mva": 1.09 * 1.18 * 1.45,
        "sf_nom_3l_elec_non_mva": 1.10 * 1.20 * 1.50,
        "sf_hi_3l_elec_non_mva": 1.11 * 1.22 * 1.55,
        "sf_lo_3l_elec_non_mva": 1.09 * 1.18 * 1.45,
    }
    for field, expected in expected_components.items():
        np.testing.assert_allclose(
            ak.to_numpy(ak.flatten(electrons[field])), [expected]
        )
    np.testing.assert_allclose(
        ak.to_numpy(ak.flatten(electrons.sf_nom_2l_elec)),
        ak.to_numpy(ak.flatten(electrons.sf_nom_2l_elec_mva))
        * ak.to_numpy(ak.flatten(electrons.sf_nom_2l_elec_non_mva)),
    )
    np.testing.assert_allclose(
        ak.to_numpy(ak.flatten(electrons.sf_nom_3l_elec)),
        ak.to_numpy(ak.flatten(electrons.sf_nom_3l_elec_mva))
        * ak.to_numpy(ak.flatten(electrons.sf_nom_3l_elec_non_mva)),
    )


def test_run2_muon_3l_sf_uses_same_components_for_nominal_and_variations(
    monkeypatch,
):
    correction_set = _install_constant_correction_set(
        monkeypatch,
        {"nominal": 1.40, "syst": 0.04, "stat": 0.03},
    )
    correction_set["NUM_LooseID_DEN_TrackerMuons"] = correction_set.pop(
        "constant"
    )
    monkeypatch.setattr(
        corrections,
        "SFevaluator",
        {
            "MuonRecoSF_2018": _constant_lookup(1.10),
            "MuonRecoSF_2018_er": _constant_lookup(0.01),
            "MuonIsoSF_2018": _constant_lookup(1.20),
            "MuonIsoSF_2018_er": _constant_lookup(0.02),
            "MuonSF_2018": _constant_lookup(1.30),
            "MuonSF_2018_er": _constant_lookup(0.03),
        },
    )
    muons = ak.Array([[{"pt": 18.0, "eta": 0.2, "pdgId": 13}]])

    corrections.AttachMuonSF(muons, "2018", useRun3MVA=False)

    expected_3l = {
        "sf_nom_3l_muon": 1.30 * 1.10 * 1.40 * 1.20,
        "sf_hi_3l_muon": 1.33 * 1.11 * 1.45 * 1.22,
        "sf_lo_3l_muon": 1.27 * 1.09 * 1.35 * 1.18,
    }
    for field, expected in expected_3l.items():
        np.testing.assert_allclose(
            ak.to_numpy(ak.flatten(muons[field])), [expected]
        )
    assert not np.isclose(expected_3l["sf_nom_3l_muon"], 1.30 * 1.10 * 1.40)

    np.testing.assert_allclose(
        ak.to_numpy(ak.flatten(muons.sf_nom_2l_muon)),
        [1.30 * 1.10 * 1.40 * 1.20],
    )
    np.testing.assert_allclose(
        ak.to_numpy(ak.flatten(muons.sf_hi_2l_muon)),
        [1.33 * 1.11 * 1.45 * 1.22],
    )
    np.testing.assert_allclose(
        ak.to_numpy(ak.flatten(muons.sf_lo_2l_muon)),
        [1.27 * 1.09 * 1.35 * 1.18],
    )
    expected_components = {
        "sf_nom_2l_muon_mva": 1.30,
        "sf_hi_2l_muon_mva": 1.33,
        "sf_lo_2l_muon_mva": 1.27,
        "sf_nom_3l_muon_mva": 1.30,
        "sf_hi_3l_muon_mva": 1.33,
        "sf_lo_3l_muon_mva": 1.27,
        "sf_nom_2l_muon_non_mva": 1.10 * 1.40 * 1.20,
        "sf_hi_2l_muon_non_mva": 1.11 * 1.45 * 1.22,
        "sf_lo_2l_muon_non_mva": 1.09 * 1.35 * 1.18,
        "sf_nom_3l_muon_non_mva": 1.10 * 1.40 * 1.20,
        "sf_hi_3l_muon_non_mva": 1.11 * 1.45 * 1.22,
        "sf_lo_3l_muon_non_mva": 1.09 * 1.35 * 1.18,
    }
    for field, expected in expected_components.items():
        np.testing.assert_allclose(
            ak.to_numpy(ak.flatten(muons[field])), [expected]
        )
    np.testing.assert_allclose(
        ak.to_numpy(ak.flatten(muons.sf_nom_2l_muon)),
        ak.to_numpy(ak.flatten(muons.sf_nom_2l_muon_mva))
        * ak.to_numpy(ak.flatten(muons.sf_nom_2l_muon_non_mva)),
    )
    np.testing.assert_allclose(
        ak.to_numpy(ak.flatten(muons.sf_nom_3l_muon)),
        ak.to_numpy(ak.flatten(muons.sf_nom_3l_muon_mva))
        * ak.to_numpy(ak.flatten(muons.sf_nom_3l_muon_non_mva)),
    )


def test_run3_sf_paths_keep_iso_at_unity(monkeypatch):
    correction_set = _install_constant_correction_set(
        monkeypatch,
        {
            "sf": 1.10,
            "sfup": 1.11,
            "sfdown": 1.09,
            "nominal": 1.40,
            "syst": 0.04,
            "stat": 0.03,
        },
    )
    constant = correction_set.pop("constant")
    correction_set["Electron-ID-SF"] = constant
    correction_set["NUM_LooseID_DEN_TrackerMuons"] = constant

    class FailingEvaluator:
        def __getitem__(self, key):
            raise AssertionError(f"Run 3 unexpectedly requested legacy SF {key}")

    monkeypatch.setattr(corrections, "SFevaluator", FailingEvaluator())
    electrons = ak.Array(
        [[{"pt": 25.0, "eta": 0.2, "phi": 0.1, "pdgId": 11}]]
    )
    muons = ak.Array([[{"pt": 25.0, "eta": 0.2, "pdgId": 13}]])

    corrections.AttachElectronSF(
        electrons, "2022", looseWP="none", useRun3MVA=False
    )
    corrections.AttachMuonSF(muons, "2022", useRun3MVA=False)

    for field, expected in (
        ("sf_nom_2l_elec", 1.10),
        ("sf_hi_2l_elec", 1.11),
        ("sf_lo_2l_elec", 1.09),
        ("sf_nom_3l_elec", 1.10),
        ("sf_hi_3l_elec", 1.11),
        ("sf_lo_3l_elec", 1.09),
    ):
        np.testing.assert_allclose(
            ak.to_numpy(ak.flatten(electrons[field])), [expected]
        )
    for field in (
        "sf_nom_2l_muon",
        "sf_hi_2l_muon",
        "sf_lo_2l_muon",
        "sf_nom_3l_muon",
        "sf_hi_3l_muon",
        "sf_lo_3l_muon",
    ):
        np.testing.assert_allclose(ak.to_numpy(ak.flatten(muons[field])), [1.0])


def test_run3_mva_and_non_mva_components_remain_independently_available(
    monkeypatch,
):
    correction_sets = {
        "electron.json.gz": {
            "Electron-ID-SF": _ConstantCorrection(
                {"sf": 1.10, "sfup": 1.11, "sfdown": 1.09}
            )
        },
        "muon_Z.json.gz": {
            "NUM_LooseID_DEN_TrackerMuons": _ConstantCorrection(
                {"nominal": 1.40, "syst": 0.04, "stat": 0.03}
            )
        },
        "leptonSF_2022.json.gz": {
            "el_allflavor": _ConstantCorrection(
                {"": 1.30, "_elup": 1.33, "_eldn": 1.27}
            ),
            "mu_allflavor": _ConstantCorrection(
                {"": 1.50, "_muup": 1.55, "_mudn": 1.45}
            ),
        },
    }

    class ConstantCorrectionSet:
        @staticmethod
        def from_file(path):
            return next(
                correction_set
                for suffix, correction_set in correction_sets.items()
                if str(path).endswith(suffix)
            )

    monkeypatch.setattr(
        corrections.correctionlib, "CorrectionSet", ConstantCorrectionSet
    )
    electrons = ak.Array(
        [
            [
                {
                    "pt": 25.0,
                    "eta": 0.2,
                    "phi": 0.1,
                    "pdgId": 11,
                    "mvaTTHrun3": 0.9,
                }
            ]
        ]
    )
    muons = ak.Array(
        [[{"pt": 25.0, "eta": 0.2, "pdgId": 13, "mvaTTHrun3": 0.9}]]
    )

    corrections.AttachElectronSF(
        electrons, "2022", looseWP="none", useRun3MVA=True
    )
    corrections.AttachMuonSF(muons, "2022", useRun3MVA=True)

    for category in ("2l", "3l"):
        electron_expected = {
            f"sf_nom_{category}_elec_mva": 1.30,
            f"sf_hi_{category}_elec_mva": 1.33,
            f"sf_lo_{category}_elec_mva": 1.27,
            f"sf_nom_{category}_elec_non_mva": 1.10,
            f"sf_hi_{category}_elec_non_mva": 1.11,
            f"sf_lo_{category}_elec_non_mva": 1.09,
        }
        for field, expected in electron_expected.items():
            np.testing.assert_allclose(
                ak.to_numpy(ak.flatten(electrons[field])), [expected]
            )
        muon_expected = {
            f"sf_nom_{category}_muon_mva": 1.50,
            f"sf_hi_{category}_muon_mva": 1.55,
            f"sf_lo_{category}_muon_mva": 1.45,
            f"sf_nom_{category}_muon_non_mva": 1.0,
            f"sf_hi_{category}_muon_non_mva": 1.0,
            f"sf_lo_{category}_muon_non_mva": 1.0,
        }
        for field, expected in muon_expected.items():
            np.testing.assert_allclose(
                ak.to_numpy(ak.flatten(muons[field])), [expected]
            )
        np.testing.assert_allclose(
            ak.to_numpy(ak.flatten(electrons[f"sf_nom_{category}_elec"])),
            ak.to_numpy(
                ak.flatten(electrons[f"sf_nom_{category}_elec_mva"])
            )
            * ak.to_numpy(
                ak.flatten(electrons[f"sf_nom_{category}_elec_non_mva"])
            ),
        )
        np.testing.assert_allclose(
            ak.to_numpy(ak.flatten(muons[f"sf_nom_{category}_muon"])),
            ak.to_numpy(ak.flatten(muons[f"sf_nom_{category}_muon_mva"]))
            * ak.to_numpy(
                ak.flatten(muons[f"sf_nom_{category}_muon_non_mva"])
            ),
        )


def test_four_lepton_weights_reuse_three_lepton_per_lepton_fields():
    leptons = []
    for index, pdg_id in enumerate((11, -11, 13, -13)):
        leptons.append(
            {
                "conept": (30.0, 25.0, 20.0, 16.0)[index],
                "isTightLep": True,
                "pdgId": pdg_id,
                "convVeto": 1,
                "lostHits": 0,
                "sf_nom_3l_elec": (1.10, 1.20, 1.0, 1.0)[index],
                "sf_hi_3l_elec": (1.11, 1.21, 1.0, 1.0)[index],
                "sf_lo_3l_elec": (1.09, 1.19, 1.0, 1.0)[index],
                "sf_nom_3l_muon": (1.0, 1.0, 1.30, 1.40)[index],
                "sf_hi_3l_muon": (1.0, 1.0, 1.31, 1.41)[index],
                "sf_lo_3l_muon": (1.0, 1.0, 1.29, 1.39)[index],
                "sf_nom_3l_elec_mva": (1.10, 1.20, 1.0, 1.0)[index],
                "sf_hi_3l_elec_mva": (1.11, 1.21, 1.0, 1.0)[index],
                "sf_lo_3l_elec_mva": (1.09, 1.19, 1.0, 1.0)[index],
                "sf_nom_3l_elec_non_mva": 1.0,
                "sf_hi_3l_elec_non_mva": 1.0,
                "sf_lo_3l_elec_non_mva": 1.0,
                "sf_nom_3l_muon_mva": (1.0, 1.0, 1.30, 1.40)[index],
                "sf_hi_3l_muon_mva": (1.0, 1.0, 1.31, 1.41)[index],
                "sf_lo_3l_muon_mva": (1.0, 1.0, 1.29, 1.39)[index],
                "sf_nom_3l_muon_non_mva": 1.0,
                "sf_hi_3l_muon_non_mva": 1.0,
                "sf_lo_3l_muon_non_mva": 1.0,
            }
        )
    events = ak.Array(
        [
            {
                "l_fo_conept_sorted": leptons,
                "minMllAFAS": 20.0,
                "Flag": {
                    "goodVertices": True,
                    "globalSuperTightHalo2016Filter": True,
                    "HBHENoiseFilter": True,
                    "HBHENoiseIsoFilter": True,
                    "EcalDeadCellTriggerPrimitiveFilter": True,
                    "BadPFMuonFilter": True,
                    "ecalBadCalibFilter": True,
                    "eeBadScFilter": True,
                },
            }
        ]
    )

    event_selection.add4lMaskAndSFs(events, "2018", isData=True)

    expected = {
        "sf_4l_elec": 1.10 * 1.20,
        "sf_4l_hi_elec": 1.11 * 1.21,
        "sf_4l_lo_elec": 1.09 * 1.19,
        "sf_4l_muon": 1.30 * 1.40,
        "sf_4l_hi_muon": 1.31 * 1.41,
        "sf_4l_lo_muon": 1.29 * 1.39,
        "sf_4l_elec_mva": 1.10 * 1.20,
        "sf_4l_hi_elec_mva": 1.11 * 1.21,
        "sf_4l_lo_elec_mva": 1.09 * 1.19,
        "sf_4l_elec_non_mva": 1.0,
        "sf_4l_hi_elec_non_mva": 1.0,
        "sf_4l_lo_elec_non_mva": 1.0,
        "sf_4l_muon_mva": 1.30 * 1.40,
        "sf_4l_hi_muon_mva": 1.31 * 1.41,
        "sf_4l_lo_muon_mva": 1.29 * 1.39,
        "sf_4l_muon_non_mva": 1.0,
        "sf_4l_hi_muon_non_mva": 1.0,
        "sf_4l_lo_muon_non_mva": 1.0,
    }
    for field, value in expected.items():
        np.testing.assert_allclose(ak.to_numpy(events[field]), [value])
    np.testing.assert_allclose(
        ak.to_numpy(events.sf_4l_elec),
        ak.to_numpy(events.sf_4l_elec_mva)
        * ak.to_numpy(events.sf_4l_elec_non_mva),
    )
    np.testing.assert_allclose(
        ak.to_numpy(events.sf_4l_muon),
        ak.to_numpy(events.sf_4l_muon_mva)
        * ak.to_numpy(events.sf_4l_muon_non_mva),
    )


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
