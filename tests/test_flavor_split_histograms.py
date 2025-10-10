import sys
import types
from pathlib import Path

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


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
    return module


def _install_module(name: str, module: types.ModuleType) -> None:
    parts = name.split(".")
    for depth in range(1, len(parts)):
        parent_name = ".".join(parts[:depth])
        child_name = parts[depth]
        parent_module = _ensure_module(parent_name)
        child_full_name = ".".join(parts[: depth + 1])
        child_module = sys.modules.get(child_full_name)
        if child_module is None:
            child_module = types.ModuleType(child_full_name)
            child_module.__path__ = []  # type: ignore[attr-defined]
            sys.modules[child_full_name] = child_module
        setattr(parent_module, child_name, child_module)
    sys.modules[name] = module


class _DummyHistEFT:
    def __init__(self, *args, **kwargs):
        self._sumw = 0.0

    def fill(self, weight=None, eft_coeff=None, **axes):
        if weight is None:
            return
        self._sumw += float(np.sum(np.array(weight)))

    def values(self):
        return {(): self._sumw}


# Install lightweight stubs for the topcoffea and topeft modules used by the processor.
hist_module = types.ModuleType("topcoffea.modules.histEFT")
hist_module.HistEFT = _DummyHistEFT
_install_module("topcoffea.modules.histEFT", hist_module)

hist_pkg = types.ModuleType("hist")


class _DummyAxis:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


hist_pkg.axis = types.SimpleNamespace(Regular=_DummyAxis, Variable=_DummyAxis)  # type: ignore[arg-type]
sys.modules["hist"] = hist_pkg

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
sys.modules["awkward"] = awkward_module

corrections_module = types.ModuleType("topcoffea.modules.corrections")
corrections_module.AttachScaleWeights = lambda *args, **kwargs: None
corrections_module.GetPUSF = lambda *args, **kwargs: np.ones_like(args[0])
corrections_module.btag_sf_eval = lambda *args, **kwargs: np.ones_like(args[0])
corrections_module.get_method1a_wgt_doublewp = (
    lambda *args, **kwargs: (np.ones_like(args[4]), np.ones_like(args[4]))
)
_install_module("topcoffea.modules.corrections", corrections_module)

coffea_module = types.ModuleType("coffea")
_install_module("coffea", coffea_module)

processor_module = types.ModuleType("coffea.processor")


class _ProcessorABC:
    pass


processor_module.ProcessorABC = _ProcessorABC
_install_module("coffea.processor", processor_module)
coffea_module.processor = processor_module


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


analysis_tools_module = types.ModuleType("coffea.analysis_tools")
analysis_tools_module.PackedSelection = _DummyPackedSelection
analysis_tools_module.Weights = _DummyWeights
_install_module("coffea.analysis_tools", analysis_tools_module)
coffea_module.analysis_tools = analysis_tools_module


class _DummyLumiMask:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, run, lumi):
        return True


lumi_module = types.ModuleType("coffea.lumi_tools")
lumi_module.LumiMask = _DummyLumiMask
_install_module("coffea.lumi_tools", lumi_module)
coffea_module.lumi_tools = lumi_module

eft_helper_module = types.ModuleType("topcoffea.modules.eft_helper")
eft_helper_module.remap_coeffs = lambda src, dest, coeffs: coeffs
eft_helper_module.calc_w2_coeffs = lambda coeffs, dtype: coeffs
_install_module("topcoffea.modules.eft_helper", eft_helper_module)

paths_module = types.ModuleType("topcoffea.modules.paths")
paths_module.topcoffea_path = lambda path: str(path)
_install_module("topcoffea.modules.paths", paths_module)

get_param_module = types.ModuleType("topcoffea.modules.get_param_from_jsons")


class _DummyGetParam:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, key):
        return 0.0


def _dummy_get_param(key):
    if key == "lo_xsec_samples" or key == "conv_samples" or key == "prompt_and_conv_samples":
        return ()
    if key.startswith("lumi_"):
        return 1.0
    return 0.0


get_param_module.GetParam = lambda *args, **kwargs: _dummy_get_param
_install_module("topcoffea.modules.get_param_from_jsons", get_param_module)

_install_module("topcoffea.modules.event_selection", types.ModuleType("topcoffea.modules.event_selection"))
_install_module("topcoffea.modules.object_selection", types.ModuleType("topcoffea.modules.object_selection"))

te_paths_module = types.ModuleType("topeft.modules.paths")
te_paths_module.topeft_path = lambda path: str(path)
_install_module("topeft.modules.paths", te_paths_module)

te_corrections_module = types.ModuleType("topeft.modules.corrections")
class _DummyJetCorrections:
    def __init__(self, *args, **kwargs):
        pass

    def build(self, *args, **kwargs):
        return args[0]

te_corrections_module.ApplyJetCorrections = _DummyJetCorrections
te_corrections_module.GetBtagEff = lambda *args, **kwargs: np.ones(1)
te_corrections_module.AttachMuonSF = lambda *args, **kwargs: None
te_corrections_module.AttachElectronSF = lambda *args, **kwargs: None
te_corrections_module.AttachTauSF = lambda *args, **kwargs: None
te_corrections_module.ApplyTES = lambda *args, **kwargs: (np.ones(1), np.ones(1))
te_corrections_module.ApplyTESSystematic = lambda *args, **kwargs: (np.ones(1), np.ones(1))
te_corrections_module.ApplyFESSystematic = lambda *args, **kwargs: (np.ones(1), np.ones(1))
te_corrections_module.AttachPerLeptonFR = lambda *args, **kwargs: None
te_corrections_module.ApplyRochesterCorrections = lambda year, mu, isData: np.ones_like(mu)
te_corrections_module.ApplyJetSystematics = lambda *args, **kwargs: args[1] if len(args) > 1 else None
te_corrections_module.GetTriggerSF = lambda *args, **kwargs: np.ones(1)
_install_module("topeft.modules.corrections", te_corrections_module)

te_btag_module = types.ModuleType("topeft.modules.btag_weights")
te_btag_module.register_btag_sf_weights = lambda *args, **kwargs: None
_install_module("topeft.modules.btag_weights", te_btag_module)

_install_module("topeft.modules.event_selection", types.ModuleType("topeft.modules.event_selection"))
_install_module("topeft.modules.object_selection", types.ModuleType("topeft.modules.object_selection"))

systematics_module = types.ModuleType("topeft.modules.systematics")
systematics_module.add_fake_factor_weights = lambda *args, **kwargs: None
systematics_module.apply_theory_weight_variations = lambda **kwargs: {}
systematics_module.register_lepton_sf_weight = lambda *args, **kwargs: None
systematics_module.register_weight_variation = lambda *args, **kwargs: None
systematics_module.validate_data_weight_variations = lambda *args, **kwargs: None
_install_module("topeft.modules.systematics", systematics_module)


from analysis.topeft_run2.analysis_processor import AnalysisProcessor


@pytest.fixture
def processor(tmp_path):
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
                "DummySample",
                "nominal",
            ),
            (
                "dummy",
                "3l_eee_p_offZ_1b_2j",
                "isSR_3l",
                "DummySample",
                "nominal",
            ),
            (
                "dummy",
                "3l_emm_p_offZ_1b_2j",
                "isSR_3l",
                "DummySample",
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
        "DummySample",
        "nominal",
    )
    flavored_hist_key = (
        "dummy",
        "3l_eee_p_offZ_1b_2j",
        "isSR_3l",
        "DummySample",
        "nominal",
    )
    base_sumw2_key = (
        "dummy_sumw2",
        "3l_p_offZ_1b_2j",
        "isSR_3l",
        "DummySample",
        "nominal",
    )

    assert flavored_hist_key in processor.accumulator
    assert base_hist_key in processor.accumulator
    assert base_sumw2_key in processor.accumulator
    assert (
        "dummy_sumw2",
        "3l_eee_p_offZ_1b_2j",
        "isSR_3l",
        "DummySample",
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
