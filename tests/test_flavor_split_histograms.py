import importlib
import sys
import types
from pathlib import Path
from typing import Callable


_DATASET_NAME = "SingleMuon_A-UL2018"

try:
    import numpy as np
except ModuleNotFoundError:
    np_module = types.ModuleType("numpy")

    def _np_array(values):
        if isinstance(values, (list, tuple)):
            return list(values)
        return [values]

    def _np_sum(values):
        if isinstance(values, list):
            return float(sum(values))
        return float(values)

    def _np_ones_like(values):
        if isinstance(values, (list, tuple)):
            return [1.0 for _ in values]
        return 1.0

    def _np_ones(length):
        if isinstance(length, int):
            return [1.0] * length
        if isinstance(length, (list, tuple)) and length:
            total = 1
            for dim in length:
                total *= int(dim)
            return [1.0] * total
        return [1.0]

    def _np_square(values):
        if isinstance(values, list):
            return [float(v) * float(v) for v in values]
        return float(values) * float(values)

    def _np_invert(values):
        if isinstance(values, list):
            return [not bool(v) for v in values]
        return not bool(values)

    def _np_seterr(**kwargs):
        return None

    np_module.array = _np_array  # type: ignore[attr-defined]
    np_module.sum = _np_sum  # type: ignore[attr-defined]
    np_module.ones_like = _np_ones_like  # type: ignore[attr-defined]
    np_module.ones = _np_ones  # type: ignore[attr-defined]
    np_module.square = _np_square  # type: ignore[attr-defined]
    np_module.invert = _np_invert  # type: ignore[attr-defined]
    np_module.seterr = _np_seterr  # type: ignore[attr-defined]
    np_module.float32 = float  # type: ignore[attr-defined]
    np_module.bool_ = bool  # type: ignore[attr-defined]
    np_module.isscalar = lambda value: not isinstance(value, (list, tuple, dict))  # type: ignore[attr-defined]
    sys.modules["numpy"] = np_module
    np = np_module  # type: ignore[assignment]
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
class _DummyHistEFT:
    def __init__(self, *args, **kwargs):
        self._sumw = 0.0

    def fill(self, weight=None, eft_coeff=None, **axes):
        if weight is None:
            return
        self._sumw += float(np.sum(np.array(weight)))

    def values(self):
        return {(): self._sumw}
class _DummyAxis:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _DummyPackedSelection:
    def __init__(self):
        self._masks = {}

    def add(self, name, mask):
        self._masks[name] = mask

    def all(self, *names):
        return True

    def any(self, name):
        return True


class _DummyWeights:
    def __init__(self, size, storeIndividual=False):
        self._weights = {"nominal": [1.0] * size}
        self.variations = ()

    def add(self, name, weight, *args, **kwargs):
        self._weights[name] = weight
        if name != "nominal" and name not in self.variations:
            self.variations = tuple(self.variations) + (name,)

    def weight(self, name=None):
        key = name or "nominal"
        return self._weights.get(key, self._weights["nominal"])


class _DummyLumiMask:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, run, lumi):
        return True


class _DummyJetCorrections:
    def __init__(self, *args, **kwargs):
        pass

    def build(self, *args, **kwargs):
        return args[0]


class _ProcessorABC:
    pass


def _dummy_get_param(key):
    if key == "lo_xsec_samples" or key == "conv_samples" or key == "prompt_and_conv_samples":
        return ()
    if key.startswith("lumi_"):
        return 1.0
    return 0.0


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, module: types.ModuleType) -> None:
    parts = name.split(".")
    for depth in range(1, len(parts)):
        parent_name = ".".join(parts[:depth])
        parent_module = sys.modules.get(parent_name)
        if parent_module is None:
            parent_module = types.ModuleType(parent_name)
            parent_module.__path__ = []  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, parent_name, parent_module)
        child_name = parts[depth]
        if depth + 1 < len(parts):
            child_full_name = ".".join(parts[: depth + 1])
            child_module = sys.modules.get(child_full_name)
            if child_module is None:
                child_module = types.ModuleType(child_full_name)
                child_module.__path__ = []  # type: ignore[attr-defined]
                monkeypatch.setitem(sys.modules, child_full_name, child_module)
            setattr(parent_module, child_name, child_module)
        else:
            setattr(parent_module, child_name, module)
    monkeypatch.setitem(sys.modules, name, module)


def _maybe_install_stub(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    factory: Callable[[], types.ModuleType],
) -> None:
    try:
        importlib.import_module(name)
        return
    except (ModuleNotFoundError, ImportError):
        pass

    module = factory()
    _install_module(monkeypatch, name, module)


def _install_test_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    def _package(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        return module

    def _hist_factory() -> types.ModuleType:
        module = types.ModuleType("hist")
        module.axis = types.SimpleNamespace(Regular=_DummyAxis, Variable=_DummyAxis)  # type: ignore[attr-defined]
        return module

    _maybe_install_stub(monkeypatch, "hist", _hist_factory)

    def _awkward_factory() -> types.ModuleType:
        awkward_module = types.ModuleType("awkward")

        def _ak_ones_like(values, dtype=None):
            if isinstance(values, (list, tuple)):
                return [1.0] * len(values)
            return 1.0

        awkward_module.ones_like = _ak_ones_like  # type: ignore[attr-defined]
        awkward_module.Array = list  # type: ignore[attr-defined]
        awkward_module.to_numpy = lambda array: array  # type: ignore[attr-defined]
        awkward_module.with_name = lambda array, name: array  # type: ignore[attr-defined]
        awkward_module.fill_none = lambda array, value: array  # type: ignore[attr-defined]
        awkward_module.values_astype = lambda array, dtype: array  # type: ignore[attr-defined]
        awkward_module.argmax = lambda array, axis=-1, keepdims=False: 0  # type: ignore[attr-defined]
        awkward_module.concatenate = lambda arrays, axis=0: sum(arrays, [])  # type: ignore[attr-defined]
        awkward_module.cartesian = lambda mapping: mapping  # type: ignore[attr-defined]
        awkward_module.flatten = lambda array: array  # type: ignore[attr-defined]
        awkward_module.max = lambda array, axis=None: array  # type: ignore[attr-defined]
        awkward_module.sum = lambda array, axis=None: array  # type: ignore[attr-defined]
        awkward_module.ones_like = _ak_ones_like  # type: ignore[attr-defined]
        return awkward_module

    _maybe_install_stub(monkeypatch, "awkward", _awkward_factory)

    def _hist_eft_factory() -> types.ModuleType:
        module = types.ModuleType("topcoffea.modules.histEFT")
        module.HistEFT = _DummyHistEFT
        return module

    _maybe_install_stub(
        monkeypatch,
        "topcoffea.modules.histEFT",
        _hist_eft_factory,
    )

    def _topcoffea_corrections_factory() -> types.ModuleType:
        module = types.ModuleType("topcoffea.modules.corrections")
        module.AttachScaleWeights = lambda *args, **kwargs: None
        module.GetPUSF = lambda *args, **kwargs: np.ones_like(args[0])
        module.btag_sf_eval = lambda *args, **kwargs: np.ones_like(args[0])
        module.get_method1a_wgt_doublewp = (
            lambda *args, **kwargs: (np.ones_like(args[4]), np.ones_like(args[4]))
        )
        return module

    _maybe_install_stub(
        monkeypatch,
        "topcoffea.modules.corrections",
        _topcoffea_corrections_factory,
    )

    _maybe_install_stub(monkeypatch, "coffea", lambda: _package("coffea"))

    def _processor_factory() -> types.ModuleType:
        module = types.ModuleType("coffea.processor")
        module.ProcessorABC = _ProcessorABC
        return module

    _maybe_install_stub(monkeypatch, "coffea.processor", _processor_factory)

    def _analysis_tools_factory() -> types.ModuleType:
        module = types.ModuleType("coffea.analysis_tools")
        module.PackedSelection = _DummyPackedSelection
        module.Weights = _DummyWeights
        return module

    _maybe_install_stub(monkeypatch, "coffea.analysis_tools", _analysis_tools_factory)

    def _lumi_tools_factory() -> types.ModuleType:
        module = types.ModuleType("coffea.lumi_tools")
        module.LumiMask = _DummyLumiMask
        return module

    _maybe_install_stub(monkeypatch, "coffea.lumi_tools", _lumi_tools_factory)

    def _eft_helper_factory() -> types.ModuleType:
        module = types.ModuleType("topcoffea.modules.eft_helper")
        module.remap_coeffs = lambda src, dest, coeffs: coeffs
        module.calc_w2_coeffs = lambda coeffs, dtype: coeffs
        return module

    _maybe_install_stub(monkeypatch, "topcoffea.modules.eft_helper", _eft_helper_factory)

    def _topcoffea_paths_factory() -> types.ModuleType:
        module = types.ModuleType("topcoffea.modules.paths")
        module.topcoffea_path = lambda path: str(path)
        return module

    _maybe_install_stub(monkeypatch, "topcoffea.modules.paths", _topcoffea_paths_factory)

    def _get_param_factory() -> types.ModuleType:
        module = types.ModuleType("topcoffea.modules.get_param_from_jsons")
        module.GetParam = lambda *args, **kwargs: _dummy_get_param
        return module

    _maybe_install_stub(
        monkeypatch,
        "topcoffea.modules.get_param_from_jsons",
        _get_param_factory,
    )

    _maybe_install_stub(
        monkeypatch,
        "topcoffea.modules.event_selection",
        lambda: _package("topcoffea.modules.event_selection"),
    )
    _maybe_install_stub(
        monkeypatch,
        "topcoffea.modules.object_selection",
        lambda: _package("topcoffea.modules.object_selection"),
    )

    def _topeft_paths_factory() -> types.ModuleType:
        module = types.ModuleType("topeft.modules.paths")
        module.topeft_path = lambda path: str(path)
        return module

    _maybe_install_stub(monkeypatch, "topeft.modules.paths", _topeft_paths_factory)

    def _topeft_corrections_factory() -> types.ModuleType:
        module = types.ModuleType("topeft.modules.corrections")
        module.ApplyJetCorrections = _DummyJetCorrections
        module.GetBtagEff = lambda *args, **kwargs: np.ones(1)
        module.AttachMuonSF = lambda *args, **kwargs: None
        module.AttachElectronSF = lambda *args, **kwargs: None
        module.AttachTauSF = lambda *args, **kwargs: None
        module.ApplyTES = lambda *args, **kwargs: (np.ones(1), np.ones(1))
        module.ApplyTESSystematic = lambda *args, **kwargs: (np.ones(1), np.ones(1))
        module.ApplyFESSystematic = lambda *args, **kwargs: (np.ones(1), np.ones(1))
        module.AttachPerLeptonFR = lambda *args, **kwargs: None
        module.ApplyRochesterCorrections = lambda year, mu, isData: np.ones_like(mu)
        module.ApplyJetSystematics = (
            lambda *args, **kwargs: args[1] if len(args) > 1 else None
        )
        module.GetTriggerSF = lambda *args, **kwargs: np.ones(1)
        return module

    _maybe_install_stub(
        monkeypatch,
        "topeft.modules.corrections",
        _topeft_corrections_factory,
    )

    def _topeft_btag_factory() -> types.ModuleType:
        module = types.ModuleType("topeft.modules.btag_weights")
        module.register_btag_sf_weights = lambda *args, **kwargs: None
        return module

    _maybe_install_stub(
        monkeypatch,
        "topeft.modules.btag_weights",
        _topeft_btag_factory,
    )

    _maybe_install_stub(
        monkeypatch,
        "topeft.modules.event_selection",
        lambda: _package("topeft.modules.event_selection"),
    )
    _maybe_install_stub(
        monkeypatch,
        "topeft.modules.object_selection",
        lambda: _package("topeft.modules.object_selection"),
    )

    def _topeft_systematics_factory() -> types.ModuleType:
        module = types.ModuleType("topeft.modules.systematics")
        module.add_fake_factor_weights = lambda *args, **kwargs: None
        module.apply_theory_weight_variations = lambda **kwargs: {}
        module.register_lepton_sf_weight = lambda *args, **kwargs: None
        module.register_weight_variation = lambda *args, **kwargs: None
        module.validate_data_weight_variations = lambda *args, **kwargs: None
        return module

    _maybe_install_stub(
        monkeypatch,
        "topeft.modules.systematics",
        _topeft_systematics_factory,
    )



@pytest.fixture
def processor(tmp_path, monkeypatch):
    _install_test_stubs(monkeypatch)

    from analysis.topeft_run2.analysis_processor import AnalysisProcessor

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
                "dummy",
                "3l_p_offZ_1b_2j",
                "isSR_3l",
                _DATASET_NAME,
                "nominal",
            ),
            (
                "dummy",
                "3l_eee_p_offZ_1b_2j",
                "isSR_3l",
                _DATASET_NAME,
                "nominal",
            ),
            (
                "dummy",
                "3l_emm_p_offZ_1b_2j",
                "isSR_3l",
                _DATASET_NAME,
                "nominal",
            ),
        )
    }

    var_info = {
        "label": "Dummy observable",
        "regular": (1, 0.0, 1.0),
        "definition": 'np.ones_like(events["event"])',
    }

    channel_dict = {
        "jet_selection": "exactly_2j",
        "chan_def_lst": [
            "3l_p_offZ_1b",
            "3l",
            "3l_p",
            "3l_offZ",
            "bmask_exactly1m",
        ],
        "lep_flav_lst": ["eee", "eem", "emm", "mmm"],
        "appl_region": "isSR_3l",
        "features": (),
    }

    golden_json = tmp_path / "golden.json"
    golden_json.write_text("{}", encoding="utf-8")

    return AnalysisProcessor(
        sample=sample,
        wc_names_lst=[],
        hist_keys=hist_keys,
        var_info=var_info,
        ecut_threshold=None,
        do_errors=False,
        split_by_lepton_flavor=True,
        channel_dict=channel_dict,
        golden_json_path=str(golden_json),
        available_systematics={
            "object": (),
            "weight": (),
            "theory": (),
            "data_weight": (),
        },
    )


def test_flavor_split_registers_flavored_histograms(processor):
    lep_chan = "3l_p_offZ_1b"
    njet_ch = "exactly_2j"
    flav_ch = "eee"

    flav_name, base_name = processor._build_channel_names(lep_chan, njet_ch, flav_ch)

    assert base_name == processor.channel
    assert flav_name == "3l_eee_p_offZ_1b_2j"

    base_hist_key = (
        "dummy",
        "3l_p_offZ_1b_2j",
        "isSR_3l",
        _DATASET_NAME,
        "nominal",
    )
    flavored_hist_key = (
        "dummy",
        "3l_eee_p_offZ_1b_2j",
        "isSR_3l",
        _DATASET_NAME,
        "nominal",
    )
    base_sumw2_key = (
        "dummy_sumw2",
        "3l_p_offZ_1b_2j",
        "isSR_3l",
        _DATASET_NAME,
        "nominal",
    )

    assert flavored_hist_key in processor.accumulator
    assert base_hist_key in processor.accumulator
    assert base_sumw2_key in processor.accumulator
    assert (
        "dummy_sumw2",
        "3l_eee_p_offZ_1b_2j",
        "isSR_3l",
        _DATASET_NAME,
        "nominal",
    ) not in processor.accumulator

    assert flavored_hist_key in processor.hist_keys_to_fill
    assert base_sumw2_key in processor.hist_keys_to_fill

    processor.accumulator[flavored_hist_key].fill(
        dummy=np.array([0.5]), weight=np.array([2.0]), eft_coeff=None
    )
    assert processor.accumulator[flavored_hist_key].values()[()] == pytest.approx(2.0)
    assert (
        processor._flavored_channel_lookup["3l_eee_p_offZ_1b_2j"] == processor.channel
    )


def test_dataset_name_resolution_preserves_full_hist_key(processor):
    dataset_hist, dataset_trig = processor._resolve_dataset_names(_DATASET_NAME)

    assert dataset_hist == _DATASET_NAME
    assert dataset_trig == "SingleMuon"


def test_histogram_fallback_finds_base_channel(processor):
    dataset_hist, _ = processor._resolve_dataset_names(_DATASET_NAME)

    dense_axis_name = processor.var
    hist_variation_label = processor.syst
    base_ch_name = processor.channel
    missing_flavor_channel = "3l_eem_p_offZ_1b_2j"

    histkey = (
        dense_axis_name,
        missing_flavor_channel,
        processor.appregion,
        dataset_hist,
        hist_variation_label,
    )
    fallback_histkey = (
        dense_axis_name,
        base_ch_name,
        processor.appregion,
        dataset_hist,
        hist_variation_label,
    )

    hout = processor.accumulator

    assert histkey not in hout
    assert fallback_histkey in hout
