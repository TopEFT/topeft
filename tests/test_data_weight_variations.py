import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from coffea.analysis_tools import Weights

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_analysis_processor_with_correction_stubs():
    analysis_path = REPO_ROOT / "analysis" / "topeft_run2" / "analysis_processor.py"

    def _make_stub(module_name):
        module = types.ModuleType(module_name)

        def _placeholder(*_args, **_kwargs):
            return None

        module.__getattr__ = lambda _name: _placeholder  # type: ignore[attr-defined]
        return module

    stubbed_modules = {
        "topeft.modules.corrections": _make_stub("topeft.modules.corrections"),
        "topcoffea.modules.corrections": _make_stub("topcoffea.modules.corrections"),
    }

    saved_modules = {}
    module_name = "analysis_processor_for_test"

    try:
        for name, stub in stubbed_modules.items():
            if name in sys.modules:
                saved_modules[name] = sys.modules[name]
            sys.modules[name] = stub

        spec = importlib.util.spec_from_file_location(module_name, analysis_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        for name in stubbed_modules:
            if name in saved_modules:
                sys.modules[name] = saved_modules[name]
            else:
                sys.modules.pop(name, None)


def test_same_sign_data_weights_include_fliprate_and_requested_variations():
    analysis_module = _load_analysis_processor_with_correction_stubs()

    weights = Weights(3)
    events = types.SimpleNamespace(
        fakefactor_2l=np.array([0.9, 1.1, 1.0]),
        fakefactor_2l_up=np.array([1.0, 1.2, 1.1]),
        fakefactor_2l_down=np.array([0.8, 1.0, 0.9]),
        nom=np.ones(3),
        fakefactor_2l_pt1=np.full(3, 1.05),
        fakefactor_2l_pt2=np.full(3, 0.95),
        fakefactor_2l_be1=np.full(3, 1.02),
        fakefactor_2l_be2=np.full(3, 0.98),
        fakefactor_2l_elclosureup=np.full(3, 1.01),
        fakefactor_2l_elclosuredown=np.full(3, 0.99),
        fakefactor_2l_muclosureup=np.full(3, 1.03),
        fakefactor_2l_muclosuredown=np.full(3, 0.97),
        flipfactor_2l=np.array([1.2, 0.8, 1.0]),
    )

    analysis_module._add_fake_factor_weights(
        weights,
        events,
        "2lss",
        "UL18",
        requested_data_weight_label="FF",
    )

    weights.add("fliprate", events.flipfactor_2l)

    central_modifiers = getattr(weights, "weight_modifiers", None)
    if central_modifiers is None:
        central_modifiers = getattr(weights, "_names", None)

    assert central_modifiers is not None
    assert "fliprate" in set(central_modifiers)

    assert set(weights.variations) == {"FFUp", "FFDown"}
