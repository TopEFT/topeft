import argparse
from types import SimpleNamespace

from topeft.modules.executor_cli import (
    ExecutorCLIHelper,
    FuturesArgumentSpec,
    TaskVineArgumentSpec,
)


class _RecordingRemoteEnvironment:
    def __init__(self, return_path: str):
        self.return_path = return_path
        self.calls: list[tuple[dict[str, list[str]], list[str]]] = []

    def get_environment(self, *, extra_pip_local, extra_conda):
        self.calls.append((dict(extra_pip_local), list(extra_conda)))
        return self.return_path


class _DummyTaskVineExecutor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_executor_cli_taskvine_configuration(tmp_path):
    remote_env = _RecordingRemoteEnvironment(return_path="/tmp/env.tar")
    helper = ExecutorCLIHelper(
        remote_environment=remote_env,
        futures_spec=FuturesArgumentSpec(
            workers_default=8,
            include_status=True,
            include_tail_timeout=True,
            include_memory=True,
            include_retries=True,
            include_retry_wait=True,
        ),
        taskvine_spec=TaskVineArgumentSpec(
            include_manager_name=True,
            include_manager_template=True,
            include_scratch_dir=True,
            include_resource_monitor=True,
            include_resources_mode=True,
            resource_monitor_default="measure",
            resources_mode_default="auto",
        ),
        extra_pip_local={"topeft": ["topeft", "setup.py"]},
        extra_conda=["pyyaml"],
    )

    parser = argparse.ArgumentParser()
    helper.configure_parser(parser)

    scratch_dir = tmp_path / "staging"
    args = parser.parse_args(
        [
            "--executor",
            "taskvine",
            "--port",
            "9100-9101",
            "--environment-file",
            "auto",
            "--manager-name",
            "custom-manager",
            "--scratch-dir",
            str(scratch_dir),
            "--resource-monitor",
            "none",
            "--resources-mode",
            "manual",
            "--no-taskvine-print-stdout",
            "--futures-workers",
            "4",
            "--futures-status",
            "--futures-tail-timeout",
            "60",
            "--futures-memory",
            "2048",
            "--futures-retries",
            "2",
            "--futures-retry-wait",
            "1.5",
        ]
    )

    config = helper.parse_args(args)

    assert remote_env.calls == [({"topeft": ["topeft", "setup.py"]}, ["pyyaml"])]
    assert config.executor == "taskvine"

    futures = config.futures
    assert futures.workers == 4
    assert futures.status is True
    assert futures.tailtimeout == 60
    assert futures.memory == 2048
    assert futures.retries == 2
    assert futures.retry_wait == 1.5

    taskvine = config.taskvine
    assert taskvine.port_range == (9100, 9101)
    assert taskvine.manager_name == "custom-manager"
    assert taskvine.manager_name_template == "custom-manager-{pid}"
    assert taskvine.resource_monitor is None
    assert taskvine.resources_mode == "manual"
    assert taskvine.environment_file == "/tmp/env.tar"
    assert config.taskvine_print_stdout is False
    assert taskvine.print_stdout is False

    staging_dir = taskvine.staging_directory()
    assert staging_dir == scratch_dir
    assert staging_dir.exists()

    logs_dir = taskvine.logs_directory(staging_dir)
    assert logs_dir == scratch_dir / "logs" / "taskvine"
    assert logs_dir.exists()

    kwargs = taskvine.executor_kwargs(extra_input_files=["proc.py"], logs_dir=logs_dir)
    assert kwargs["extra_input_files"] == ["proc.py"]
    assert kwargs["filepath"] == str(staging_dir)
    assert kwargs["environment_file"] == "/tmp/env.tar"
    assert kwargs["print_stdout"] is False

    processor_module = SimpleNamespace(TaskVineExecutor=_DummyTaskVineExecutor)
    instance = taskvine.instantiate(processor_module, kwargs, negotiate_port=False)
    assert isinstance(instance, _DummyTaskVineExecutor)
    assert instance.kwargs["port"] == 9100


def test_executor_cli_defaults(monkeypatch, tmp_path):
    remote_env = _RecordingRemoteEnvironment(return_path=str(tmp_path / "env.tar"))
    helper = ExecutorCLIHelper(
        remote_environment=remote_env,
        futures_spec=FuturesArgumentSpec(
            workers_default=4,
            include_status=False,
            include_tail_timeout=False,
        ),
        taskvine_spec=TaskVineArgumentSpec(
            include_manager_name=False,
            include_manager_template=False,
            include_scratch_dir=False,
            include_resource_monitor=False,
            include_resources_mode=False,
            resource_monitor_default="measure",
            resources_mode_default="auto",
        ),
    )

    parser = argparse.ArgumentParser()
    helper.configure_parser(parser)

    monkeypatch.setenv("USER", "cliuser")
    staging_root = tmp_path / "shared"
    monkeypatch.setenv("TOPEFT_EXECUTOR_STAGING", str(staging_root))

    config = helper.parse_args(parser.parse_args([]))

    assert config.executor == "taskvine"
    futures = config.futures
    assert futures.workers == 4
    assert futures.status is None
    assert futures.tailtimeout is None

    taskvine = config.taskvine
    assert taskvine.manager_name == "cliuser-taskvine-coffea"
    assert taskvine.manager_name_template == "cliuser-taskvine-coffea-{pid}"
    assert taskvine.resource_monitor == "measure"
    assert taskvine.resources_mode == "auto"
    assert taskvine.environment_file == str(tmp_path / "env.tar")
    assert config.taskvine_print_stdout is True
    assert taskvine.print_stdout is True

    staging_dir = taskvine.staging_directory()
    assert staging_dir == staging_root
    assert staging_dir.exists()

    logs_dir = taskvine.logs_directory(staging_dir)
    assert logs_dir.exists()

    kwargs = taskvine.executor_kwargs(extra_input_files=["proc.py"], logs_dir=logs_dir)
    assert kwargs["print_stdout"] is True

    assert remote_env.calls == [({}, [])]


def test_executor_cli_futures_allows_missing_cached(tmp_path):
    helper = ExecutorCLIHelper(
        remote_environment=SimpleNamespace(env_dir_cache=tmp_path / "envs"),
        futures_spec=FuturesArgumentSpec(
            workers_default=2,
            include_status=False,
            include_tail_timeout=False,
        ),
        taskvine_spec=TaskVineArgumentSpec(
            include_manager_name=False,
            include_manager_template=False,
            include_scratch_dir=False,
            include_resource_monitor=False,
            include_resources_mode=False,
        ),
        default_environment="cached",
        default_executor="taskvine",
    )

    parser = argparse.ArgumentParser()
    helper.configure_parser(parser)

    args = parser.parse_args(["--executor", "futures"])
    config = helper.parse_args(args)

    assert config.executor == "futures"
    assert config.environment_file is None
    assert config.taskvine.environment_file is None
