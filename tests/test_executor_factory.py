import importlib
import sys
import types

import pytest

from analysis.topeft_run2.run_analysis_helpers import RunConfig


@pytest.fixture
def stub_coffea_modules(monkeypatch):
    coffea_pkg = types.ModuleType("coffea")
    processor_mod = types.ModuleType("coffea.processor")
    nanoevents_mod = types.ModuleType("coffea.nanoevents")

    class _DummyExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _DummyRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    processor_mod.FuturesExecutor = _DummyExecutor
    processor_mod.IterativeExecutor = _DummyExecutor
    processor_mod.TaskVineExecutor = _DummyExecutor
    processor_mod.Runner = _DummyRunner

    coffea_pkg.processor = processor_mod
    nanoevents_mod.NanoAODSchema = object

    monkeypatch.setitem(sys.modules, "coffea", coffea_pkg)
    monkeypatch.setitem(sys.modules, "coffea.processor", processor_mod)
    monkeypatch.setitem(sys.modules, "coffea.nanoevents", nanoevents_mod)

    topcoffea_pkg = types.ModuleType("topcoffea")
    modules_pkg = types.ModuleType("topcoffea.modules")
    remote_env_mod = types.ModuleType("topcoffea.modules.remote_environment")
    remote_env_mod.get_environment = lambda **_: None

    modules_pkg.remote_environment = remote_env_mod
    topcoffea_pkg.modules = modules_pkg

    monkeypatch.setitem(sys.modules, "topcoffea", topcoffea_pkg)
    monkeypatch.setitem(sys.modules, "topcoffea.modules", modules_pkg)
    monkeypatch.setitem(sys.modules, "topcoffea.modules.remote_environment", remote_env_mod)

    sys.modules.pop("analysis.topeft_run2.workflow", None)

    yield

    for module_name in [
        "coffea",
        "coffea.processor",
        "coffea.nanoevents",
        "topcoffea",
        "topcoffea.modules",
        "topcoffea.modules.remote_environment",
    ]:
        sys.modules.pop(module_name, None)
    sys.modules.pop("analysis.topeft_run2.workflow", None)


def test_work_queue_unavailable_error(stub_coffea_modules):
    workflow = importlib.import_module("analysis.topeft_run2.workflow")
    workflow = importlib.reload(workflow)
    factory = workflow.ExecutorFactory(RunConfig(executor="work_queue"))

    with pytest.raises(RuntimeError) as excinfo:
        factory.create_runner()

    message = str(excinfo.value)
    assert "WorkQueueExecutor is not available" in message
    assert "Coffea 2025.7" in message or "2025.7" in message
