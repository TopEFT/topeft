import importlib
import sys
import types
import awkward as ak
import numpy as np
import pytest

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

_corrected_jets_stub = types.ModuleType("topcoffea.modules.CorrectedJetsFactory")


class _CorrectedJetsFactory:
    def __init__(self, *_, **__):
        pass


_corrected_jets_stub.CorrectedJetsFactory = _CorrectedJetsFactory
modules_pkg.CorrectedJetsFactory = _corrected_jets_stub  # type: ignore[attr-defined]
sys.modules["topcoffea.modules.CorrectedJetsFactory"] = _corrected_jets_stub

_jec_stack_stub = types.ModuleType("topcoffea.modules.JECStack")


class _JECStack:
    def __init__(self, *_, **__):
        pass


_jec_stack_stub.JECStack = _JECStack
modules_pkg.JECStack = _jec_stack_stub  # type: ignore[attr-defined]
sys.modules["topcoffea.modules.JECStack"] = _jec_stack_stub

_get_param_stub = types.ModuleType("topcoffea.modules.get_param_from_jsons")


class _GetParam:
    def __init__(self, *_, **__):
        pass

    def __call__(self, *_, **__):
        return {}


_get_param_stub.GetParam = _GetParam
modules_pkg.get_param_from_jsons = _get_param_stub  # type: ignore[attr-defined]
sys.modules["topcoffea.modules.get_param_from_jsons"] = _get_param_stub

_corrections_stub = types.ModuleType("topcoffea.modules.corrections")
modules_pkg.corrections = _corrections_stub  # type: ignore[attr-defined]
sys.modules["topcoffea.modules.corrections"] = _corrections_stub

_eft_helper_stub = types.ModuleType("topcoffea.modules.eft_helper")
modules_pkg.eft_helper = _eft_helper_stub  # type: ignore[attr-defined]
sys.modules["topcoffea.modules.eft_helper"] = _eft_helper_stub

_event_sel_stub = types.ModuleType("topcoffea.modules.event_selection")
_event_sel_stub.get_Z_peak_mask = lambda *_, **__: None
_event_sel_stub.get_off_Z_mask_low = lambda *_, **__: None
_event_sel_stub.get_any_sfos_pair = lambda *_, **__: None
_event_sel_stub.trg_pass_no_overlap = lambda *_, **__: None
modules_pkg.event_selection = _event_sel_stub  # type: ignore[attr-defined]
sys.modules["topcoffea.modules.event_selection"] = _event_sel_stub

_object_sel_stub = types.ModuleType("topcoffea.modules.object_selection")
modules_pkg.object_selection = _object_sel_stub  # type: ignore[attr-defined]
sys.modules["topcoffea.modules.object_selection"] = _object_sel_stub

from analysis.topeft_run2 import analysis_processor as ap


class _NoopJetCorrections:
    def __init__(self, *args, **kwargs):
        pass

    def build(self, *args, **kwargs):
        return args[0]


def _patch_common(monkeypatch):
    monkeypatch.setattr(ap, "ApplyJetCorrections", lambda *args, **kwargs: _NoopJetCorrections())

    def _noop_systematics(year, jets, variation):
        return jets

    monkeypatch.setattr(ap, "ApplyJetSystematics", _noop_systematics)

    def _noop_event_selection(*args, **kwargs):
        return None

    monkeypatch.setattr(ap.te_es, "add1lMaskAndSFs", _noop_event_selection)
    monkeypatch.setattr(ap.te_es, "add2lMaskAndSFs", _noop_event_selection)
    monkeypatch.setattr(ap.te_es, "add3lMaskAndSFs", _noop_event_selection)
    monkeypatch.setattr(ap.te_es, "add4lMaskAndSFs", _noop_event_selection)
    monkeypatch.setattr(ap.te_es, "addLepCatMasks", _noop_event_selection)


def _make_dataset_context(**overrides):
    defaults = dict(
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
        sample_type="mc",
        is_lo_sample=False,
        lumi_mask=ak.Array([True]),
        lumi=1.0,
        eft_coeffs=None,
        eft_w2_coeffs=None,
    )
    defaults.update(overrides)
    return ap.DatasetContext(**defaults)


def _make_variation_state(*, jets, fakeable_leptons, fakeable_sorted, taus=None, cleaning_taus=None):
    if taus is None:
        taus = ak.Array([[]])
    variation_objects = ap.VariationObjects(
        met=ak.Array([{"pt": 80.0, "phi": 0.0, "sumEt": 100.0}]),
        electrons=ak.Array([[]]),
        muons=ak.Array([[]]),
        taus=taus,
        jets=jets,
        loose_leptons=ak.Array([[]]),
        fakeable_leptons=fakeable_leptons,
        fakeable_sorted=fakeable_sorted,
        cleaning_taus=cleaning_taus,
        n_loose_taus=None,
        tau0=None,
    )
    jets_rho = ak.ones_like(jets.pt) * np.float32(0.5)
    return ap.VariationState(
        request=ap.VariationRequest(variation=None, histogram_label="nominal"),
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
        lepton_selection=object(),
        jets_rho=jets_rho,
    )


def _make_processor():
    processor = ap.AnalysisProcessor.__new__(ap.AnalysisProcessor)
    processor._var_def = {"ht", "j0"}
    processor.tau_h_analysis = False
    processor._debug_logging = False
    processor._debug = lambda *args, **kwargs: None
    return processor


def _build_histogram_variation_state(monkeypatch):
    _patch_common(monkeypatch)

    monkeypatch.setattr(
        ap.tc_es,
        "get_Z_peak_mask",
        lambda leptons, pt_window=0.0, flavor=None: ak.zeros_like(
            ak.num(leptons, axis=-1), dtype=bool
        ),
    )
    monkeypatch.setattr(
        ap.tc_es,
        "get_off_Z_mask_low",
        lambda leptons, pt_window=0.0: ak.zeros_like(ak.num(leptons, axis=-1), dtype=bool),
    )
    monkeypatch.setattr(
        ap.tc_es,
        "get_any_sfos_pair",
        lambda leptons: ak.zeros_like(ak.num(leptons, axis=-1), dtype=bool),
    )
    monkeypatch.setattr(
        ap.tc_es,
        "trg_pass_no_overlap",
        lambda events, isData, trigger_dataset, year, dataset_dict, exclude_dict: ak.ones_like(
            events["event"], dtype=bool
        ),
    )

    processor = _make_processor()
    processor._channel_dict = {
        "jet_selection": "atleast_0j",
        "chan_def_lst": ["2lss", "2lss"],
        "lep_flav_lst": ["ee", "em", "mm"],
        "appl_region": "isSR_2lSS",
        "features": (),
    }
    processor._channel_features = frozenset()
    processor.offZ_3l_split = False
    processor.tau_h_analysis = False
    processor.fwd_analysis = False
    processor._flavored_channel_lookup = {}
    processor._histogram_label_lookup = {}
    processor._split_by_lepton_flavor = False
    processor._channel = "2lss"
    processor._appregion = "isSR_2lSS"
    processor._var = "njets"
    processor._var_def = "njets"
    processor._ecut_threshold = None

    jets = ak.Array(
        [
            [
                {"pt": 55.0, "eta": 0.1, "phi": 0.1, "mass": 5.0, "btagDeepFlavB": 0.96},
                {"pt": 42.0, "eta": -0.2, "phi": -0.3, "mass": 4.0, "btagDeepFlavB": 0.10},
                {"pt": 36.0, "eta": 0.4, "phi": 0.5, "mass": 3.0, "btagDeepFlavB": 0.92},
            ],
            [
                {"pt": 50.0, "eta": -0.1, "phi": 0.2, "mass": 5.0, "btagDeepFlavB": 0.08},
                {"pt": 28.0, "eta": 0.3, "phi": -0.4, "mass": 3.5, "btagDeepFlavB": 0.75},
            ],
        ]
    )
    fakeable_sorted = ak.Array(
        [
            [
                {"pt": 40.0, "conept": 40.0, "eta": 0.1, "phi": 0.1, "mass": 0.1, "charge": 1},
                {"pt": 35.0, "conept": 35.0, "eta": -0.2, "phi": 0.2, "mass": 0.1, "charge": 1},
                {"pt": 25.0, "conept": 25.0, "eta": 0.3, "phi": -0.3, "mass": 0.1, "charge": 1},
            ],
            [
                {"pt": 45.0, "conept": 45.0, "eta": 0.1, "phi": 0.1, "mass": 0.1, "charge": -1},
                {"pt": 32.0, "conept": 32.0, "eta": -0.2, "phi": 0.2, "mass": 0.1, "charge": -1},
            ],
        ]
    )
    fakeable_leptons = ak.Array([[], []])
    padded_leptons = ak.pad_none(fakeable_sorted, 3, axis=1)

    objects = ap.VariationObjects(
        met=ak.Array([{"pt": 80.0, "phi": 0.0, "sumEt": 100.0}] * 2),
        electrons=ak.Array([[], []]),
        muons=ak.Array([[], []]),
        taus=ak.Array([[], []]),
        jets=jets,
        loose_leptons=ak.Array([[], []]),
        fakeable_leptons=fakeable_leptons,
        fakeable_sorted=fakeable_sorted,
        cleaning_taus=ak.Array([[], []]),
        n_loose_taus=None,
        tau0=None,
    )
    jets_rho = ak.ones_like(jets.pt) * np.float32(0.5)
    variation_state = ap.VariationState(
        request=ap.VariationRequest(variation=None, histogram_label="nominal"),
        name="nominal",
        base=None,
        variation_type=None,
        metadata={},
        object_variation="nominal",
        weight_variations=[],
        requested_data_weight_label=None,
        sow_variation_key_map={},
        sow_variations={},
        objects=objects,
        lepton_selection=object(),
        jets_rho=jets_rho,
    )
    variation_state.good_jets = jets
    variation_state.fwd_jets = ak.Array([[], []])
    variation_state.njets = ak.values_astype(
        ak.fill_none(ak.num(jets.pt, axis=-1), 0), np.int64
    )
    variation_state.nfwdj = ak.zeros_like(variation_state.njets)
    variation_state.ht = ak.sum(jets.pt, axis=-1)
    variation_state.l_sorted_padded = padded_leptons
    variation_state.l0 = padded_leptons[:, 0]
    variation_state.l1 = padded_leptons[:, 1]
    variation_state.l2 = padded_leptons[:, 2]

    loose_mask = ak.Array([[True, False, True], [False, True]])
    med_mask = ak.Array([[True, False, False], [False, False]])
    variation_state.isBtagJetsLoose = loose_mask
    variation_state.isNotBtagJetsLoose = ~loose_mask
    variation_state.isBtagJetsMedium = med_mask
    variation_state.isNotBtagJetsMedium = ~med_mask
    variation_state.isBtagJetsLooseNotMedium = loose_mask & ~med_mask

    dataset = _make_dataset_context(
        is_data=False,
        lumi_mask=ak.Array([True, True]),
        dataset="sample",
        trigger_dataset="sample",
    )

    events = ak.Array(
        {
            "event": [1, 2],
            "is2l_nozeeveto": [True, True],
            "is2l": [True, True],
            "is3l": [False, False],
            "is4l": [False, False],
            "is_ee": [True, False],
            "is_em": [False, True],
            "is_mm": [False, False],
            "is_e": [True, True],
            "is_m": [False, False],
            "is_eee": [False, False],
            "is_eem": [False, False],
            "is_emm": [False, False],
            "is_mmm": [False, False],
            "is_eeee": [False, False],
            "is_eeem": [False, False],
            "is_eemm": [False, False],
            "is_emmm": [False, False],
            "is_mmmm": [False, False],
            "is_gr4l": [False, False],
            "is2l_SR": [True, True],
            "is3l_SR": [False, False],
            "is4l_SR": [False, False],
        }
    )

    class _RecordingHist(_DummyHistEFT):
        def __init__(self):
            self.fills = []

        def fill(self, **kwargs):
            self.fills.append(kwargs)

    histkey = processor._build_histogram_key(
        "njets",
        processor.channel,
        dataset.dataset,
        "nominal",
        application=processor._appregion,
    )
    hout = {histkey: _RecordingHist()}

    class _DummyWeights:
        variations: tuple = ()

        def weight(self, name=None):
            return np.ones(len(events["event"]), dtype=float)

    return processor, dataset, variation_state, events, _DummyWeights(), hout, histkey


@pytest.mark.skipif(
    not hasattr(ap.ak, "stack"),
    reason="Awkward build missing ak.stack; Run 2 workflow expects native support.",
)
def test_ak_stack_uses_awkward_native():
    arrays = [ak.Array([1, 2]), ak.Array([3, 4])]
    stacked_axis0 = ap.ak.stack(arrays, axis=0)
    stacked_axis1 = ap.ak.stack(arrays, axis=1)

    assert ak.to_list(stacked_axis0) == [[1, 2], [3, 4]]
    assert ak.to_list(stacked_axis1) == [[1, 3], [2, 4]]


def test_apply_object_variations_preserves_good_jet_count(monkeypatch):
    _patch_common(monkeypatch)
    processor = _make_processor()
    jets = ak.Array([
        [
            {
                "pt": 55.0,
                "eta": 0.2,
                "phi": 0.1,
                "mass": 10.0,
                "rawFactor": 0.1,
                "jetId": 6,
                "matched_gen": {"pt": 50.0},
            },
            {
                "pt": 25.0,
                "eta": 2.5,
                "phi": -0.3,
                "mass": 5.0,
                "rawFactor": 0.05,
                "jetId": 6,
                "matched_gen": {"pt": 20.0},
            },
        ]
    ])
    fakeable = ak.Array([[{
        "pt": 40.0,
        "eta": 0.1,
        "phi": 0.2,
        "mass": 0.1,
        "jetIdx": -1,
    }]])
    fakeable_sorted = ak.Array([[
        {"pt": 45.0, "conept": 45.0, "jetIdx": -1},
        {"pt": 35.0, "conept": 35.0, "jetIdx": -1},
        {"pt": 25.0, "conept": 25.0, "jetIdx": -1},
    ]])
    variation_state = _make_variation_state(
        jets=jets, fakeable_leptons=fakeable, fakeable_sorted=fakeable_sorted
    )
    events = ak.Array({"event": [1]})
    dataset = _make_dataset_context(is_data=True)

    updated_state = processor._apply_object_variations(events, dataset, variation_state)

    assert ak.to_list(updated_state.njets) == [1]
    assert ak.to_list(updated_state.ht) == [55.0]
    assert ak.to_list(events["njets"]) == [1]


def test_apply_tau_variations_updates_tau_properties(monkeypatch):
    _patch_common(monkeypatch)

    def _tes(year, tau, is_data, variation):
        return tau.pt + 5.0, tau.mass + 0.1

    def _fes(year, tau, is_data, variation):
        return tau.pt + 2.0, tau.mass + 0.2

    monkeypatch.setattr(ap, "ApplyTESSystematic", _tes)
    monkeypatch.setattr(ap, "ApplyFESSystematic", _fes)

    processor = _make_processor()
    processor.tau_h_analysis = True

    jets = ak.Array([
        [
            {
                "pt": 50.0,
                "eta": 0.1,
                "phi": 0.2,
                "mass": 8.0,
                "rawFactor": 0.1,
                "jetId": 6,
                "matched_gen": {"pt": 45.0},
            }
        ]
    ])
    fakeable = ak.Array([[{
        "pt": 30.0,
        "eta": 0.1,
        "phi": 0.1,
        "mass": 0.1,
        "jetIdx": -1,
    }]])
    fakeable_sorted = ak.Array([[{ "pt": 30.0, "conept": 30.0, "jetIdx": -1 }]])
    taus = ak.Array([[{
        "pt": 25.0,
        "mass": 1.0,
        "isLoose": 1,
        "eta": 0.1,
        "phi": 0.2,
        "jetIdx": -1,
    }]])
    variation_state = _make_variation_state(
        jets=jets,
        fakeable_leptons=fakeable,
        fakeable_sorted=fakeable_sorted,
        taus=taus,
        cleaning_taus=None,
    )
    events = ak.Array({"event": [1]})
    dataset = _make_dataset_context(is_data=False)

    updated_state = processor._apply_object_variations(events, dataset, variation_state)

    updated_tau_pt = ak.to_list(updated_state.objects.taus.pt)
    assert updated_tau_pt == [[32.0]]
    assert ak.to_list(updated_state.objects.n_loose_taus) == [1]
    assert ak.to_list(updated_state.objects.tau0.pt) == [32.0]


def test_cleaned_jets_handles_missing_gen_matches(monkeypatch):
    _patch_common(monkeypatch)

    monkeypatch.setattr(
        ap.tc_os, "is_tight_jet", lambda pt, eta, jetId, **_: ak.ones_like(pt, dtype=bool)
    )
    monkeypatch.setattr(
        ap.te_os, "isFwdJet", lambda pt, eta, jetId, jetPtCut=40.0: ak.zeros_like(pt, dtype=bool)
    )

    processor = _make_processor()
    jets = ak.Array([
        [
            {
                "pt": 55.0,
                "eta": 0.2,
                "phi": 0.1,
                "mass": 10.0,
                "rawFactor": 0.0,
                "jetId": 6,
                "genJetIdx": -1,
            }
        ]
    ])
    fakeable = ak.Array([[{"pt": 40.0, "eta": 0.1, "phi": 0.2, "mass": 0.1, "jetIdx": -1}]])
    fakeable_sorted = ak.Array([[{"pt": 40.0, "conept": 40.0, "jetIdx": -1}]])
    variation_state = _make_variation_state(
        jets=jets, fakeable_leptons=fakeable, fakeable_sorted=fakeable_sorted
    )
    dataset = _make_dataset_context(is_data=False)

    updated_state = processor._build_cleaned_jets(variation_state, dataset=dataset)

    assert "pt_gen" in ak.fields(updated_state.objects.jets)
    assert ak.to_list(updated_state.objects.jets.pt_gen) == [[0.0]]


def test_cleaned_jets_masks_preserve_structure(monkeypatch):
    _patch_common(monkeypatch)

    monkeypatch.setattr(
        ap.tc_os,
        "is_tight_jet",
        lambda pt, eta, jetId, **_: ak.Array(
            [[True, False], [False, True, True]]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        ap.te_os,
        "isFwdJet",
        lambda pt, eta, jetId, jetPtCut=40.0: ak.Array(
            [[False, False], [True, False, False]]
        ),
        raising=False,
    )

    processor = _make_processor()
    jets = ak.Array(
        [
            [
                {"pt": 60.0, "eta": 0.2, "phi": 0.1, "mass": 10.0, "rawFactor": 0.0, "jetId": 6},
                {"pt": 25.0, "eta": -2.1, "phi": -0.2, "mass": 5.0, "rawFactor": 0.0, "jetId": 6},
            ],
            [
                {"pt": 80.0, "eta": 2.6, "phi": 0.3, "mass": 12.0, "rawFactor": 0.0, "jetId": 6},
                {"pt": 45.0, "eta": 0.5, "phi": -0.1, "mass": 7.5, "rawFactor": 0.0, "jetId": 6},
                {"pt": 35.0, "eta": -1.2, "phi": 0.4, "mass": 6.0, "rawFactor": 0.0, "jetId": 6},
            ],
        ]
    )
    fakeable = ak.Array(
        [
            [{"pt": 0.0, "eta": 0.0, "phi": 0.0, "mass": 0.0, "jetIdx": -1}],
            [{"pt": 0.0, "eta": 0.0, "phi": 0.0, "mass": 0.0, "jetIdx": -1}],
        ]
    )
    fakeable_sorted = ak.Array(
        [
            [{"pt": 0.0, "conept": 0.0, "jetIdx": -1}],
            [{"pt": 0.0, "conept": 0.0, "jetIdx": -1}],
        ]
    )
    variation_state = _make_variation_state(
        jets=jets, fakeable_leptons=fakeable, fakeable_sorted=fakeable_sorted
    )
    dataset = _make_dataset_context(is_data=True)

    updated_state = processor._build_cleaned_jets(variation_state, dataset=dataset)

    cleaned_jets = updated_state.cleaned_jets
    assert ak.to_list(ak.num(cleaned_jets.isGood, axis=-1)) == ak.to_list(
        ak.num(cleaned_jets.pt, axis=-1)
    )
    assert ak.to_list(ak.num(updated_state.good_jets.pt, axis=-1)) == [1, 2]
    assert ak.to_list(ak.num(updated_state.fwd_jets.pt, axis=-1)) == [0, 1]

    assert ak.to_list(updated_state.njets) == [1, 2]
    assert ak.to_list(updated_state.nfwdj) == [0, 1]
    assert (
        ak.to_layout(updated_state.njets, allow_record=False).purelist_depth == 1
    )
    assert (
        ak.to_layout(updated_state.nfwdj, allow_record=False).purelist_depth == 1
    )


def test_lepton_ordering_is_preserved(monkeypatch):
    _patch_common(monkeypatch)
    processor = _make_processor()
    jets = ak.Array([
        [
            {
                "pt": 60.0,
                "eta": 0.1,
                "phi": 0.1,
                "mass": 9.0,
                "rawFactor": 0.1,
                "jetId": 6,
                "matched_gen": {"pt": 55.0},
            },
            {
                "pt": 45.0,
                "eta": 2.8,
                "phi": -0.2,
                "mass": 6.0,
                "rawFactor": 0.1,
                "jetId": 6,
                "matched_gen": {"pt": 40.0},
            },
            {
                "pt": 25.0,
                "eta": 1.5,
                "phi": 0.3,
                "mass": 5.0,
                "rawFactor": 0.1,
                "jetId": 6,
                "matched_gen": {"pt": 20.0},
            },
        ]
    ])
    fakeable = ak.Array([[{
        "pt": 35.0,
        "eta": 0.1,
        "phi": 0.2,
        "mass": 0.1,
        "jetIdx": -1,
    }]])
    fakeable_sorted = ak.Array([[
        {"pt": 40.0, "conept": 40.0, "jetIdx": -1},
        {"pt": 30.0, "conept": 30.0, "jetIdx": -1},
        {"pt": 20.0, "conept": 20.0, "jetIdx": -1},
    ]])
    variation_state = _make_variation_state(
        jets=jets, fakeable_leptons=fakeable, fakeable_sorted=fakeable_sorted
    )
    events = ak.Array({"event": [1]})
    dataset = _make_dataset_context(is_data=True)

    updated_state = processor._apply_object_variations(events, dataset, variation_state)

    assert ak.to_list(updated_state.l0.pt) == [40.0]
    assert ak.to_list(updated_state.l1.pt) == [30.0]
    assert ak.to_list(updated_state.l2.pt) == [20.0]


def test_histogram_masks_use_integer_btag_counts(monkeypatch):
    (
        processor,
        dataset,
        variation_state,
        events,
        weights_object,
        hout,
        histkey,
    ) = _build_histogram_variation_state(monkeypatch)

    processor._fill_histograms_for_variation(
        events,
        dataset,
        variation_state,
        weights_object=weights_object,
        hist_label="nominal",
        data_weight_systematics_set=set(),
        hout=hout,
    )

    assert len(hout[histkey].fills) == 1
    assert ak.to_list(hout[histkey].fills[0]["njets"]) == [3, 2]


def test_ptbl_pairing_handles_missing_btags(monkeypatch):
    (
        processor,
        dataset,
        variation_state,
        events,
        weights_object,
        _,
        _,
    ) = _build_histogram_variation_state(monkeypatch)

    processor._var = "ptbl"
    processor._var_def = "ptbl"
    processor._channel = processor._channel_dict["chan_def_lst"][0] + "_0j"

    variation_state.isBtagJetsLoose = ak.Array([[True, False, True], [False, False]])
    variation_state.isNotBtagJetsLoose = ~variation_state.isBtagJetsLoose
    variation_state.isBtagJetsMedium = ak.Array([[True, False, False], [False, False]])
    variation_state.isNotBtagJetsMedium = ~variation_state.isBtagJetsMedium
    variation_state.isBtagJetsLooseNotMedium = variation_state.isBtagJetsLoose & ~variation_state.isBtagJetsMedium

    ptbl_values = processor._compute_ptbl(
        variation_state.good_jets,
        variation_state.isBtagJetsMedium,
        variation_state.isBtagJetsLoose,
        variation_state.objects.fakeable_sorted,
    )
    ptbl_flat = ak.flatten(ptbl_values, axis=None)
    ptbl_flat_list = [float(val) for val in ak.to_list(ptbl_flat)]

    assert np.isfinite(np.asarray(ptbl_flat_list)).all()
    assert -1.0 in ptbl_flat_list

    class _RecordingHist(_DummyHistEFT):
        def __init__(self):
            self.fills = []

        def fill(self, **kwargs):
            self.fills.append(kwargs)

    lep_chan = processor._channel_dict["chan_def_lst"][0]
    hist_channel, _ = processor._build_channel_names(
        lep_chan, processor._channel_dict["jet_selection"], None
    )
    histkey = processor._build_histogram_key(
        processor._var, hist_channel, dataset.dataset, "nominal", application=processor._appregion
    )
    base_histkey = processor._build_histogram_key(
        processor._var, processor.channel, dataset.dataset, "nominal", application=processor._appregion
    )
    hout = {histkey: _RecordingHist(), base_histkey: _RecordingHist()}

    processor._fill_histograms_for_variation(
        events,
        dataset,
        variation_state,
        weights_object=weights_object,
        hist_label="nominal",
        data_weight_systematics_set=set(),
        hout=hout,
    )

    filled_hist = hout[histkey] if hout[histkey].fills else hout[base_histkey]
    assert filled_hist.fills
    recorded_ptbl = ak.flatten(ak.Array(filled_hist.fills[0]["ptbl"]), axis=None)
    assert np.isfinite(np.asarray(ak.to_list(recorded_ptbl), dtype=float)).all()


def test_compute_ptbl_handles_zero_and_positive_btags():
    processor = _make_processor()

    good_jets = ak.Array(
        [
            [{"pt": 45.0, "eta": 0.1, "phi": 0.2, "mass": 5.0}],
            [{"pt": 55.0, "eta": -0.1, "phi": -0.3, "mass": 4.0}],
        ]
    )
    is_btag_med = ak.Array([[False], [True]])
    is_btag_loose = ak.Array([[False], [False]])
    leptons = ak.Array(
        [
            [{"pt": 40.0, "eta": 0.0, "phi": 0.0, "mass": 0.1}],
            [{"pt": 42.0, "eta": 0.2, "phi": 0.1, "mass": 0.1}],
        ]
    )

    ptbl_values = processor._compute_ptbl(good_jets, is_btag_med, is_btag_loose, leptons)
    ptbl_flat = ak.to_list(ak.flatten(ptbl_values, axis=None))

    assert ptbl_flat[0] == pytest.approx(-1.0)
    assert ptbl_flat[1] > 0


def test_histogram_btag_masks_handle_multijet_events(monkeypatch):
    (
        processor,
        dataset,
        variation_state,
        events,
        weights_object,
        hout,
        histkey,
    ) = _build_histogram_variation_state(monkeypatch)

    processor._fill_histograms_for_variation(
        events,
        dataset,
        variation_state,
        weights_object=weights_object,
        hist_label="nominal",
        data_weight_systematics_set=set(),
        hout=hout,
    )

    assert ak.to_list(variation_state.nbtagsl) == [2, 1]
    assert ak.to_list(variation_state.nbtagsm) == [1, 0]
    assert ak.to_numpy(variation_state.nbtagsm).dtype.kind in {"i", "u"}


def test_histogram_jet_selections_use_jagged_counts(monkeypatch):
    (
        processor,
        dataset,
        variation_state,
        events,
        weights_object,
        hout,
        histkey,
    ) = _build_histogram_variation_state(monkeypatch)

    multijet = ak.Array(
        [
            [
                {"pt": 70.0, "eta": 0.3, "phi": 0.1, "mass": 6.0, "btagDeepFlavB": 0.95},
                {"pt": 55.0, "eta": -0.4, "phi": -0.2, "mass": 5.0, "btagDeepFlavB": 0.15},
                {"pt": 45.0, "eta": 0.2, "phi": 0.4, "mass": 4.5, "btagDeepFlavB": 0.70},
                {"pt": 38.0, "eta": -0.1, "phi": -0.5, "mass": 3.5, "btagDeepFlavB": 0.20},
            ],
            [
                {"pt": 60.0, "eta": 0.5, "phi": -0.1, "mass": 5.5, "btagDeepFlavB": 0.05},
                {"pt": 48.0, "eta": -0.3, "phi": 0.2, "mass": 4.2, "btagDeepFlavB": 0.82},
                {"pt": 35.0, "eta": 0.1, "phi": -0.4, "mass": 3.0, "btagDeepFlavB": 0.65},
            ],
        ]
    )

    variation_state.objects.jets = multijet
    variation_state.good_jets = multijet
    variation_state.jets_rho = ak.ones_like(multijet.pt) * np.float32(0.5)
    variation_state.fwd_jets = None
    variation_state.njets = ak.values_astype(
        ak.num(multijet.pt, axis=-1), np.int64
    )
    variation_state.nfwdj = ak.zeros_like(ak.num(multijet, axis=-1))
    variation_state.ht = ak.sum(multijet.pt, axis=-1)

    loose_mask = ak.Array([[True, False, True, True], [False, True, True]])
    med_mask = ak.Array([[True, False, False, False], [False, True, False]])
    variation_state.isBtagJetsLoose = loose_mask
    variation_state.isNotBtagJetsLoose = ~loose_mask
    variation_state.isBtagJetsMedium = med_mask
    variation_state.isNotBtagJetsMedium = ~med_mask
    variation_state.isBtagJetsLooseNotMedium = loose_mask & ~med_mask

    processor._fill_histograms_for_variation(
        events,
        dataset,
        variation_state,
        weights_object=weights_object,
        hist_label="nominal",
        data_weight_systematics_set=set(),
        hout=hout,
    )

    assert len(hout[histkey].fills) == 1
    assert ak.to_list(variation_state.njets) == [4, 3]
    assert ak.to_numpy(variation_state.njets).dtype.kind in {"i", "u"}
    assert ak.to_list(hout[histkey].fills[0]["njets"]) == [4, 3]


def test_multi_jet_selections_do_not_broadcast(monkeypatch):
    (
        processor,
        dataset,
        variation_state,
        events,
        weights_object,
        hout,
        histkey,
    ) = _build_histogram_variation_state(monkeypatch)

    multijet = ak.Array(
        [
            [
                {"pt": 75.0, "eta": 0.3, "phi": 0.1, "mass": 6.2, "btagDeepFlavB": 0.90},
                {"pt": 62.0, "eta": -0.4, "phi": -0.2, "mass": 5.5, "btagDeepFlavB": 0.20},
                {"pt": 50.0, "eta": 2.6, "phi": 0.5, "mass": 4.8, "btagDeepFlavB": 0.10},
                {"pt": 44.0, "eta": -0.1, "phi": -0.5, "mass": 4.1, "btagDeepFlavB": 0.35},
            ],
            [
                {"pt": 68.0, "eta": 0.6, "phi": -0.2, "mass": 5.9, "btagDeepFlavB": 0.12},
                {"pt": 53.0, "eta": -2.5, "phi": 0.3, "mass": 4.6, "btagDeepFlavB": 0.85},
                {"pt": 39.0, "eta": 0.2, "phi": -0.4, "mass": 3.8, "btagDeepFlavB": 0.22},
            ],
        ]
    )

    fwd_mask = ak.Array([[False, False, True, False], [False, True, False]])
    variation_state.objects.jets = multijet
    variation_state.good_jets = ak.with_field(multijet, fwd_mask, "isFwd")
    variation_state.fwd_jets = multijet[fwd_mask]
    variation_state.jets_rho = ak.ones_like(multijet.pt) * np.float32(0.5)
    variation_state.njets = ak.values_astype(ak.num(multijet.pt, axis=-1), np.int64)
    variation_state.nfwdj = ak.values_astype(ak.sum(fwd_mask, axis=-1), np.int64)
    variation_state.ht = ak.sum(multijet.pt, axis=-1)

    loose_mask = ak.Array([[True, False, True, False], [False, True, False]])
    med_mask = ak.Array([[True, False, False, False], [False, True, False]])
    variation_state.isBtagJetsLoose = loose_mask
    variation_state.isNotBtagJetsLoose = ~loose_mask
    variation_state.isBtagJetsMedium = med_mask
    variation_state.isNotBtagJetsMedium = ~med_mask
    variation_state.isBtagJetsLooseNotMedium = loose_mask & ~med_mask

    processor.fwd_analysis = True

    processor._fill_histograms_for_variation(
        events,
        dataset,
        variation_state,
        weights_object=weights_object,
        hist_label="nominal",
        data_weight_systematics_set=set(),
        hout=hout,
    )

    assert len(hout[histkey].fills) == 1
    assert ak.to_list(variation_state.njets) == [4, 3]
    assert ak.to_list(variation_state.nfwdj) == [1, 1]
    assert ak.to_list(variation_state.njets == 0) == [False, False]
    assert ak.to_list(variation_state.njets >= 4) == [True, False]
    assert ak.to_list(variation_state.nfwdj == 0) == [False, False]


def test_forward_jet_counts_use_jagged_counts(monkeypatch):
    (
        processor,
        dataset,
        variation_state,
        events,
        weights_object,
        hout,
        histkey,
    ) = _build_histogram_variation_state(monkeypatch)

    processor.fwd_analysis = True
    variation_state.fwd_jets = ak.Array(
        [
            [
                {"pt": 60.0, "eta": 2.6, "phi": 0.1, "mass": 4.0},
                {"pt": 45.0, "eta": -2.7, "phi": -0.2, "mass": 3.5},
            ],
            [
                {"pt": 55.0, "eta": 2.5, "phi": 0.3, "mass": 4.2},
            ],
        ]
    )
    variation_state.nfwdj = ak.values_astype(
        ak.num(variation_state.fwd_jets.pt, axis=-1), np.int64
    )

    processor._fill_histograms_for_variation(
        events,
        dataset,
        variation_state,
        weights_object=weights_object,
        hist_label="nominal",
        data_weight_systematics_set=set(),
        hout=hout,
    )

    assert len(hout[histkey].fills) == 1
    assert ak.to_list(variation_state.nfwdj) == [2, 1]
    assert ak.to_numpy(variation_state.nfwdj).dtype.kind in {"i", "u"}


def test_forward_histograms_handle_multijet_masks(monkeypatch):
    (
        processor,
        dataset,
        variation_state,
        events,
        weights_object,
        hout,
        histkey,
    ) = _build_histogram_variation_state(monkeypatch)

    processor.fwd_analysis = True
    variation_state.fwd_jets = None
    variation_state.good_jets = ak.with_field(
        variation_state.good_jets,
        ak.Array([[True, False, True], [False, False]]),
        "isFwd",
    )
    variation_state.nfwdj = ak.values_astype(
        ak.sum(variation_state.good_jets.isFwd, axis=-1), np.int64
    )

    processor._fill_histograms_for_variation(
        events,
        dataset,
        variation_state,
        weights_object=weights_object,
        hist_label="nominal",
        data_weight_systematics_set=set(),
        hout=hout,
    )

    assert len(hout[histkey].fills) == 1
    assert ak.to_list(variation_state.nfwdj) == [2, 0]
    assert ak.to_layout(variation_state.nfwdj, allow_record=False).purelist_depth == 1


def test_histograms_accept_zero_and_multi_jet_events(monkeypatch):
    (
        processor,
        dataset,
        variation_state,
        events,
        weights_object,
        hout,
        histkey,
    ) = _build_histogram_variation_state(monkeypatch)

    jets = ak.Array(
        [
            [
                {"pt": 70.0, "eta": 0.1, "phi": 0.2, "mass": 6.0, "btagDeepFlavB": 0.90},
                {"pt": 55.0, "eta": -0.2, "phi": -0.1, "mass": 5.0, "btagDeepFlavB": 0.40},
                {"pt": 42.0, "eta": 0.3, "phi": 0.4, "mass": 4.2, "btagDeepFlavB": 0.15},
            ],
            [],
        ]
    )

    variation_state.objects.jets = jets
    variation_state.good_jets = jets
    variation_state.fwd_jets = ak.Array([[], []])
    variation_state.jets_rho = ak.ones_like(jets.pt) * np.float32(0.5)
    variation_state.njets = ak.values_astype(
        ak.fill_none(ak.num(jets.pt, axis=-1), 0), np.int64
    )
    variation_state.nfwdj = ak.zeros_like(variation_state.njets)
    variation_state.ht = ak.sum(jets.pt, axis=-1)

    loose_mask = ak.Array([[True, False, False], []])
    med_mask = ak.Array([[True, False, False], []])
    variation_state.isBtagJetsLoose = loose_mask
    variation_state.isNotBtagJetsLoose = ~loose_mask
    variation_state.isBtagJetsMedium = med_mask
    variation_state.isNotBtagJetsMedium = ~med_mask
    variation_state.isBtagJetsLooseNotMedium = loose_mask & ~med_mask

    processor._fill_histograms_for_variation(
        events,
        dataset,
        variation_state,
        weights_object=weights_object,
        hist_label="nominal",
        data_weight_systematics_set=set(),
        hout=hout,
    )

    assert len(hout[histkey].fills) == 1
    assert ak.to_list(variation_state.njets) == [3, 0]
    assert ak.to_numpy(variation_state.njets).dtype.kind in {"i", "u"}
    assert ak.to_list(variation_state.nfwdj) == [0, 0]
    assert ak.to_numpy(variation_state.nfwdj).dtype.kind in {"i", "u"}
    assert ak.to_list(hout[histkey].fills[0]["njets"]) == [3, 0]


def test_offz_split_channels_accept_legacy_offz(monkeypatch):
    (
        processor,
        dataset,
        variation_state,
        _,
        weights_object,
        _,
        _,
    ) = _build_histogram_variation_state(monkeypatch)

    processor._channel_dict = {
        "jet_selection": "atleast_0j",
        "chan_def_lst": [
            "3l_m_offZ_1b",
            "3l",
            "3l_m",
            "3l_offZ_split",
            "bmask_exactly1m",
        ],
        "lep_flav_lst": ["eee", "eem", "emm", "mmm"],
        "appl_region": "isSR_3l",
        "features": ("offz_split",),
    }
    processor._channel_features = frozenset(["offz_split"])
    processor.offZ_3l_split = True
    processor._channel = "3l_m_offZ_1b"
    processor._appregion = "isSR_3l"

    negative_leptons = ak.Array(
        [
            [
                {"pt": 40.0, "conept": 40.0, "eta": 0.1, "phi": 0.1, "mass": 0.1, "charge": -1},
                {"pt": 35.0, "conept": 35.0, "eta": -0.2, "phi": 0.2, "mass": 0.1, "charge": -1},
                {"pt": 25.0, "conept": 25.0, "eta": 0.3, "phi": -0.3, "mass": 0.1, "charge": -1},
            ],
            [
                {"pt": 45.0, "conept": 45.0, "eta": 0.1, "phi": 0.1, "mass": 0.1, "charge": -1},
                {"pt": 32.0, "conept": 32.0, "eta": -0.2, "phi": 0.2, "mass": 0.1, "charge": -1},
                {"pt": 20.0, "conept": 20.0, "eta": 0.0, "phi": 0.0, "mass": 0.1, "charge": -1},
            ],
        ]
    )
    variation_state.objects.fakeable_sorted = negative_leptons
    variation_state.l_sorted_padded = negative_leptons
    variation_state.l0 = negative_leptons[:, 0]
    variation_state.l1 = negative_leptons[:, 1]
    variation_state.l2 = negative_leptons[:, 2]

    events = ak.Array(
        {
            "event": [1, 2],
            "is2l_nozeeveto": [False, False],
            "is2l": [False, False],
            "is3l": [True, True],
            "is4l": [False, False],
            "is_ee": [False, False],
            "is_em": [False, False],
            "is_mm": [False, False],
            "is_e": [True, True],
            "is_m": [False, False],
            "is_eee": [False, False],
            "is_eem": [False, False],
            "is_emm": [True, True],
            "is_mmm": [False, False],
            "is_eeee": [False, False],
            "is_eeem": [False, False],
            "is_eemm": [False, False],
            "is_emmm": [False, False],
            "is_mmmm": [False, False],
            "is_gr4l": [False, False],
            "is2l_SR": [False, False],
            "is3l_SR": [True, True],
            "is4l_SR": [False, False],
        }
    )

    class _RecordingHist(_DummyHistEFT):
        def __init__(self):
            self.fills = []

        def fill(self, **kwargs):
            self.fills.append(kwargs)

    histkey = processor._build_histogram_key(
        "njets",
        processor.channel,
        dataset.dataset,
        "nominal",
        application=processor._appregion,
    )
    hout = {histkey: _RecordingHist()}

    processor._fill_histograms_for_variation(
        events,
        dataset,
        variation_state,
        weights_object=weights_object,
        hist_label="nominal",
        data_weight_systematics_set=set(),
        hout=hout,
    )

    assert len(hout[histkey].fills) == 1
    assert ak.to_list(hout[histkey].fills[0]["njets"]) == [3]
