import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import coffea  # ensure the real coffea package is available during tests

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


@pytest.fixture
def stub_remote_environment(monkeypatch):
    import types

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
        "topcoffea.modules.remote_environment",
        "topcoffea.modules",
        "topcoffea",
    ]:
        sys.modules.pop(module_name, None)
    sys.modules.pop("analysis.topeft_run2.workflow", None)


def test_executor_factory_futures_instantiates(monkeypatch, stub_remote_environment):
    workflow = importlib.import_module("analysis.topeft_run2.workflow")
    workflow = importlib.reload(workflow)

    config = RunConfig(
        executor="futures",
        environment_file=None,
        nworkers=3,
        futures_status=False,
        futures_tail_timeout=180,
    )
    factory = workflow.ExecutorFactory(config)
    runner = factory.create_runner()

    import coffea.processor as processor
    from coffea.nanoevents import NanoAODSchema

    assert isinstance(runner.executor, processor.FuturesExecutor)
    assert runner.executor.workers == 3
    assert runner.executor.status is False
    assert runner.executor.tailtimeout == 180
    assert runner.schema is NanoAODSchema


def test_executor_factory_iterative_instantiates(monkeypatch, stub_remote_environment):
    workflow = importlib.import_module("analysis.topeft_run2.workflow")
    workflow = importlib.reload(workflow)

    config = RunConfig(executor="iterative", environment_file=None)
    factory = workflow.ExecutorFactory(config)
    runner = factory.create_runner()

    import coffea.processor as processor

    assert isinstance(runner.executor, processor.IterativeExecutor)


def test_executor_factory_taskvine_instantiates(tmp_path, monkeypatch, stub_remote_environment):
    workflow = importlib.import_module("analysis.topeft_run2.workflow")
    workflow = importlib.reload(workflow)

    scratch_dir = tmp_path / "staging"
    config = RunConfig(
        executor="taskvine",
        environment_file=None,
        scratch_dir=str(scratch_dir),
        port="9125",
        negotiate_manager_port=False,
        manager_name="test-taskvine",
    )

    factory = workflow.ExecutorFactory(config)
    runner = factory.create_runner()

    import coffea.processor as processor

    assert isinstance(runner.executor, processor.TaskVineExecutor)
    assert runner.executor.manager_name == "test-taskvine"
    assert runner.executor.filepath == str(scratch_dir)
    assert runner.skipbadfiles is True
    assert runner.xrootdtimeout == 300
