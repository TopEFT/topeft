import importlib
import json
import sys
import types

import awkward as ak
import numpy as np
import pytest


class _EventNamespace(dict):
    """Simple container supporting attribute and key access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive fallback
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _install_stubs(monkeypatch):
    """Install lightweight module stubs required by the analysis processor."""

    def _module(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    # HistEFT stub
    hist_module = _module("topcoffea.modules.histEFT")

    class _DummyHistEFT:
        def __init__(self, *args, **kwargs):
            self._sumw = 0.0

        def fill(self, weight=None, **kwargs):
            if weight is None:
                return
            array = np.asarray(weight)
            self._sumw += float(np.sum(array))

    hist_module.HistEFT = _DummyHistEFT

    # ``hist`` axis helpers
    hist_pkg = _module("hist")
    hist_pkg.axis = types.SimpleNamespace(Regular=lambda *_, **__: None)

    # Coffea interfaces
    coffea_pkg = _module("coffea")
    processor_module = _module("coffea.processor")

    class _ProcessorABC:  # pragma: no cover - interface shim
        pass

    processor_module.ProcessorABC = _ProcessorABC
    coffea_pkg.processor = processor_module  # type: ignore[attr-defined]

    analysis_tools_module = _module("coffea.analysis_tools")

    class _PackedSelection:
        def __init__(self):
            self._masks = {}

        def add(self, name, mask):
            self._masks[name] = mask

        def all(self, *names):
            if not names:
                return True
            result = None
            for name in names:
                mask = self._masks.get(name, True)
                result = mask if result is None else (result & mask)
            return result if result is not None else True

    class _Weights:
        def __init__(self, size, storeIndividual=False):
            self._weights = {"nominal": np.ones(size)}
            self.variations = ()

        def add(self, name, weight, *args, **kwargs):
            arr = np.asarray(weight)
            self._weights[name] = arr
            if name != "nominal" and name not in self.variations:
                self.variations = tuple(self.variations) + (name,)

        def weight(self, name=None):
            return self._weights.get(name or "nominal", self._weights["nominal"])

    analysis_tools_module.PackedSelection = _PackedSelection
    analysis_tools_module.Weights = _Weights

    lumi_tools_module = _module("coffea.lumi_tools")

    class _DummyLumiMask:
        def __init__(self, *_, **__):
            pass

        def __call__(self, run, lumi):
            return ak.ones_like(run, dtype=bool)

    lumi_tools_module.LumiMask = _DummyLumiMask

    # Parameter helpers
    paths_module = _module("topcoffea.modules.paths")
    paths_module.topcoffea_path = lambda path: str(path)

    te_paths_module = _module("topeft.modules.paths")
    te_paths_module.topeft_path = lambda path: str(path)

    def _dummy_get_param(mapping):
        defaults = {
            "lo_xsec_samples": [],
            "conv_samples": [],
            "prompt_and_conv_samples": [],
            "eta_j_cut": 2.5,
            "jet_id_cut": 2,
            "lumi_2018": 1.0,
            "btag_wp_loose_UL18": 0.0,
            "btag_wp_medium_UL18": 0.0,
        }
        return defaults.get(mapping, 0.0)

    get_param_module = _module("topcoffea.modules.get_param_from_jsons")
    get_param_module.GetParam = lambda *args, **kwargs: _dummy_get_param

    # Corrections stubs
    corrections_module = _module("topcoffea.modules.corrections")

    class _JetCorrections:
        def __init__(self, *args, **kwargs):
            pass

        def build(self, collection, *_, **__):
            return collection

    corrections_module.ApplyJetCorrections = lambda *args, **kwargs: _JetCorrections()
    corrections_module.AttachScaleWeights = lambda *args, **kwargs: None
    corrections_module.GetPUSF = lambda *args, **kwargs: np.ones(1)
    corrections_module.btag_sf_eval = lambda *args, **kwargs: np.ones(1)
    corrections_module.get_method1a_wgt_doublewp = (
        lambda *args, **kwargs: (np.ones(1), np.ones(1))
    )

    topeft_corr_module = _module("topeft.modules.corrections")
    topeft_corr_module.ApplyJetCorrections = corrections_module.ApplyJetCorrections
    topeft_corr_module.ApplyJetSystematics = lambda *args, **kwargs: args[1]
    topeft_corr_module.GetBtagEff = lambda *args, **kwargs: np.ones(1)
    topeft_corr_module.AttachMuonSF = lambda *args, **kwargs: None
    topeft_corr_module.AttachElectronSF = lambda *args, **kwargs: None
    topeft_corr_module.AttachTauSF = lambda *args, **kwargs: None
    topeft_corr_module.ApplyTES = lambda *args, **kwargs: (args[1], args[1])
    topeft_corr_module.ApplyTESSystematic = lambda *args, **kwargs: (args[1], args[1])
    topeft_corr_module.ApplyFESSystematic = lambda *args, **kwargs: (args[1], args[1])
    topeft_corr_module.AttachPerLeptonFR = lambda *args, **kwargs: None
    topeft_corr_module.ApplyRochesterCorrections = (
        lambda year, mu, isData: mu
    )
    topeft_corr_module.GetTriggerSF = lambda *args, **kwargs: np.ones(1)

    btag_module = _module("topeft.modules.btag_weights")
    btag_module.register_btag_sf_weights = lambda *args, **kwargs: None

    # Selection helpers
    te_obj_module = _module("topeft.modules.object_selection")

    class _DummyLeptonSelection:
        def coneptElec(self, ele):
            return ak.fill_none(ele.pt, 0)

        def coneptMuon(self, mu):
            return ak.fill_none(mu.pt, 0)

        def isPresElec(self, ele):
            return True

        def isLooseElec(self, ele):
            return True

        def isFOElec(self, ele, year):
            return True

        def tightSelElec(self, ele):
            return True

        def isPresMuon(self, mu):
            return True

        def isLooseMuon(self, mu):
            return True

        def isFOMuon(self, mu, year):
            return True

        def tightSelMuon(self, mu):
            return True

    te_obj_module.run2leptonselection = lambda: _DummyLeptonSelection()
    te_obj_module.run3leptonselection = lambda: _DummyLeptonSelection()
    te_obj_module.ttH_idEmu_cuts_E3 = lambda *args, **kwargs: ak.Array([0])
    te_obj_module.isPresTau = lambda *args, **kwargs: ak.Array([True])
    te_obj_module.isClean = lambda *args, **kwargs: ak.Array([True])
    te_obj_module.isGood = lambda *args, **kwargs: ak.Array([True])
    te_obj_module.isVLooseTau = lambda *args, **kwargs: ak.Array([True])
    te_obj_module.isLooseTau = lambda *args, **kwargs: ak.Array([True])
    te_obj_module.iseTightTau = lambda *args, **kwargs: ak.Array([True])
    te_obj_module.ismTightTau = lambda *args, **kwargs: ak.Array([True])
    te_obj_module.isFwdJet = lambda *args, **kwargs: ak.Array([False])

    tc_obj_module = _module("topcoffea.modules.object_selection")
    tc_obj_module.is_tight_jet = (
        lambda pt, eta, jet_id, pt_cut=30.0, eta_cut=2.5, id_cut=2: ak.ones_like(pt, dtype=bool)
    )

    te_evt_module = _module("topeft.modules.event_selection")
    te_evt_module.add1lMaskAndSFs = lambda events, *args, **kwargs: events.__setitem__("is1l", ak.ones_like(events.nom, dtype=bool))
    te_evt_module.add2lMaskAndSFs = lambda events, *args, **kwargs: events.__setitem__("is2l", ak.ones_like(events.nom, dtype=bool))
    te_evt_module.add3lMaskAndSFs = lambda events, *args, **kwargs: events.__setitem__("is3l", ak.ones_like(events.nom, dtype=bool))
    te_evt_module.add4lMaskAndSFs = lambda events, *args, **kwargs: events.__setitem__("is4l", ak.ones_like(events.nom, dtype=bool))
    te_evt_module.addLepCatMasks = lambda events: events.__setitem__("is_e", ak.ones_like(events.nom, dtype=bool))

    tc_evt_module = _module("topcoffea.modules.event_selection")
    tc_evt_module.get_Z_peak_mask = lambda *args, **kwargs: ak.Array([True])
    tc_evt_module.get_off_Z_mask_low = lambda *args, **kwargs: ak.Array([True])
    tc_evt_module.get_any_sfos_pair = lambda *args, **kwargs: ak.Array([True])
    tc_evt_module.trg_pass_no_overlap = lambda *args, **kwargs: ak.Array([True])

    systematics_module = _module("topeft.modules.systematics")
    systematics_module.add_fake_factor_weights = lambda *args, **kwargs: None
    systematics_module.apply_theory_weight_variations = lambda **kwargs: {}
    systematics_module.register_lepton_sf_weight = lambda *args, **kwargs: None
    systematics_module.register_trigger_sf_weight = lambda *args, **kwargs: None
    systematics_module.register_weight_variation = lambda *args, **kwargs: None
    systematics_module.validate_data_weight_variations = lambda *args, **kwargs: None


@pytest.fixture
def processor(monkeypatch, tmp_path):
    # Ensure we import the processor with the test stubs.
    for module_name in list(sys.modules):
        if module_name.startswith("analysis.topeft_run2.analysis_processor"):
            sys.modules.pop(module_name)

    _install_stubs(monkeypatch)

    # Create minimal golden JSON
    golden_json = tmp_path / "golden.json"
    golden_json.write_text(json.dumps({}), encoding="utf-8")

    analysis_processor = importlib.import_module("analysis.topeft_run2.analysis_processor")

    def _fake_combinations(array, *args, **kwargs):
        return ak.Array([[{"l0": 0.0, "l1": 0.0}]])[:, :0]

    monkeypatch.setattr(analysis_processor.ak, "combinations", _fake_combinations)
    monkeypatch.setattr(analysis_processor.ak, "with_name", lambda array, name: array)

    sample = {
        "isData": True,
        "histAxisName": "DummyData",
        "year": "2018",
        "xsec": 1.0,
        "nSumOfWeights": 1.0,
        "WCnames": [],
        "path": "store/data/Run2018A-UL/RAW",
    }

    hist_keys = {
        "nominal": (
            (
                "observable",
                "3l_p_offZ_1b_2j",
                "isSR_3l",
                "DummyDataset",
                "nominal",
            ),
        )
    }

    var_info = {
        "label": "Dummy",
        "regular": (1, 0.0, 1.0),
        "definition": 'np.ones_like(events["event"])',
    }

    channel_dict = {
        "jet_selection": "exactly_2j",
        "chan_def_lst": ["3l_p_offZ_1b"],
        "lep_flav_lst": ["eee"],
        "appl_region": "isSR_3l",
        "features": (),
    }

    processor = analysis_processor.AnalysisProcessor(
        sample=sample,
        wc_names_lst=[],
        hist_keys=hist_keys,
        var_info=var_info,
        ecut_threshold=None,
        do_errors=False,
        split_by_lepton_flavor=False,
        channel_dict=channel_dict,
        golden_json_path=str(golden_json),
        available_systematics={
            "object": (),
            "weight": (),
            "theory": (),
            "data_weight": (),
        },
    )

    return processor


def _build_minimal_events():
    events = _EventNamespace()
    events.metadata = {"dataset": "DummyDataset"}
    events.caches = [dict()]
    events.run = ak.Array([1])
    events.luminosityBlock = ak.Array([1])
    events.event = ak.Array([1])

    events.MET = ak.Array([{"pt": 40.0, "phi": 0.0}])
    electron_template = ak.Array(
        [
            [
                {
                    "pt": 30.0,
                    "eta": 0.1,
                    "phi": 0.0,
                    "mass": 0.0005,
                    "hoe": 0.1,
                    "deltaEtaSC": 0.0,
                    "eInvMinusPInv": 0.0,
                    "sieie": 0.0,
                    "matched_jet": {"btagDeepFlavB": 0.1},
                    "matched_gen": {"pdgId": 11},
                    "jetIdx": 0,
                    "charge": 1,
                }
            ]
        ]
    )
    muon_template = ak.Array(
        [
            [
                {
                    "pt": 25.0,
                    "eta": 0.2,
                    "phi": 0.1,
                    "mass": 0.105,
                    "matched_jet": {"btagDeepFlavB": 0.05},
                    "matched_gen": {"pdgId": 13},
                    "jetIdx": 1,
                    "charge": -1,
                }
            ]
        ]
    )
    events.Electron = electron_template[:, :0]
    events.Muon = muon_template[:, :0]
    events.Tau = ak.Array(
        [
            [
                {
                    "pt": 20.0,
                    "eta": 0.3,
                    "phi": 0.2,
                    "mass": 1.77,
                    "dxy": 0.0,
                    "dz": 0.0,
                    "idDeepTau2017v2p1VSjet": 1,
                    "idDeepTau2017v2p1VSe": 1,
                    "idDeepTau2017v2p1VSmu": 1,
                    "decayMode": 0,
                }
            ]
        ]
    )
    events.Jet = ak.Array(
        [
            [
                {
                    "pt": 50.0,
                    "eta": 0.5,
                    "phi": 0.3,
                    "mass": 5.0,
                    "rawFactor": 0.0,
                    "jetId": 6,
                    "btagDeepFlavB": 0.2,
                    "hadronFlavour": 0,
                },
                {
                    "pt": 45.0,
                    "eta": -0.4,
                    "phi": -0.2,
                    "mass": 4.0,
                    "rawFactor": 0.0,
                    "jetId": 6,
                    "btagDeepFlavB": 0.05,
                    "hadronFlavour": 1,
                },
            ]
        ]
    )
    events.fixedGridRhoFastjetAll = ak.Array([10.0])
    return events


def test_process_nominal_run_is_quiet(processor, capsys, monkeypatch):
    events = _build_minimal_events()

    def _fake_process(self, events):
        dataset = events.metadata["dataset"]
        self._debug(
            "Processing variation '%s' (type: %s, base: %s)",
            "nominal",
            None,
            None,
        )
        self._debug(
            "Variation group mapping for '%s': mapping=%s key=%s info=%s",
            "nominal",
            {},
            (),
            {},
        )
        self._debug(
            "Filling histograms for channel '%s' (base '%s') with cuts %s",
            self.channel,
            self.channel,
            {},
        )
        self._debug(
            "Filled histkey %s with %d selected events",
            (self.var, self.channel, self.appregion, dataset, self.syst),
            0,
        )
        return {dataset: 0}

    monkeypatch.setattr(processor, "process", _fake_process.__get__(processor, type(processor)))

    result = processor.process(events)

    captured = capsys.readouterr()

    assert captured.out == ""
    assert result == {"DummyDataset": 0}
