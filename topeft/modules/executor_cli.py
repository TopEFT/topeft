"""Helpers for composing executor-focused command line interfaces."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .executor import (
    build_taskvine_args,
    distributed_logs_dir,
    instantiate_taskvine_executor,
    manager_name_base,
    parse_port_range,
    resolve_environment_file,
)


@dataclass(frozen=True)
class FuturesArgumentSpec:
    """Configuration describing which futures options to expose on the CLI."""

    workers_default: int = 8
    include_status: bool = True
    include_tail_timeout: bool = True
    include_memory: bool = False
    include_prefetch: bool = False
    include_retries: bool = False
    include_retry_wait: bool = False


@dataclass(frozen=True)
class TaskVineArgumentSpec:
    """Configuration describing which TaskVine options to expose on the CLI."""

    include_manager_name: bool = True
    include_manager_template: bool = True
    include_scratch_dir: bool = True
    include_resource_monitor: bool = True
    include_resources_mode: bool = True
    resource_monitor_default: Optional[str] = None
    resources_mode_default: Optional[str] = None


@dataclass
class FuturesConfig:
    """Normalised futures executor configuration."""

    workers: int
    status: Optional[bool]
    tailtimeout: Optional[int]
    memory: Optional[int]
    prefetch: Optional[int]
    retries: int
    retry_wait: float


@dataclass
class TaskVineConfig:
    """Normalised TaskVine executor configuration."""

    executor: str
    port_range: tuple[int, int]
    manager_name: Optional[str]
    manager_name_template: Optional[str]
    resource_monitor: Optional[str]
    resources_mode: Optional[str]
    environment_file: Optional[str]
    scratch_dir: Optional[Path]
    negotiate_port: bool = True
    _staging_dir: Optional[Path] = field(default=None, init=False, repr=False)

    def staging_directory(self) -> Path:
        """Return the staging directory, creating it when necessary."""

        if self._staging_dir is not None:
            return self._staging_dir

        if self.scratch_dir:
            staging = Path(self.scratch_dir).expanduser()
        else:
            base_dir = os.environ.get("TOPEFT_EXECUTOR_STAGING")
            if base_dir:
                staging = Path(base_dir).expanduser()
            else:
                staging = Path(tempfile.gettempdir()) / "topeft" / manager_name_base(
                    self.executor
                )

        staging.mkdir(parents=True, exist_ok=True)
        self._staging_dir = staging
        return staging

    def logs_directory(self, staging_dir: Path | None = None) -> Path:
        """Return the directory where TaskVine logs should be stored."""

        base_dir = staging_dir or self.staging_directory()
        return distributed_logs_dir(base_dir, self.executor)

    def executor_kwargs(
        self,
        *,
        extra_input_files: Sequence[str],
        custom_init: Optional[Any] = None,
        logs_dir: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Return keyword arguments for ``processor.TaskVineExecutor``."""

        staging_dir = self.staging_directory()
        resolved_logs = logs_dir or self.logs_directory(staging_dir)
        return build_taskvine_args(
            staging_dir=staging_dir,
            logs_dir=resolved_logs,
            manager_name=self.manager_name,
            manager_name_template=self.manager_name_template,
            extra_input_files=extra_input_files,
            resource_monitor=self.resource_monitor,
            resources_mode=self.resources_mode,
            environment_file=self.environment_file,
            custom_init=custom_init,
        )

    def instantiate(
        self,
        processor_module: Any,
        executor_kwargs: dict[str, Any],
        *,
        negotiate_port: Optional[bool] = None,
    ) -> Any:
        """Instantiate the TaskVine executor using the stored configuration."""

        return instantiate_taskvine_executor(
            processor_module,
            executor_kwargs,
            port_range=self.port_range,
            negotiate_port=self.negotiate_port if negotiate_port is None else negotiate_port,
        )


@dataclass
class ExecutorConfig:
    """Normalised executor configuration derived from CLI arguments."""

    executor: str
    futures: FuturesConfig
    taskvine: TaskVineConfig
    environment_file: Optional[str]


class ExecutorCLIHelper:
    """Compose executor CLI options and translate them into runtime settings."""

    def __init__(
        self,
        *,
        remote_environment: Any,
        futures_spec: FuturesArgumentSpec | None = None,
        taskvine_spec: TaskVineArgumentSpec | None = None,
        extra_pip_local: Optional[dict[str, Iterable[str]]] = None,
        extra_conda: Optional[Sequence[str]] = None,
        default_executor: str = "taskvine",
        default_port: str = "9123-9130",
        default_environment: str = "auto",
    ) -> None:
        self._remote_environment = remote_environment
        self._futures_spec = futures_spec or FuturesArgumentSpec()
        self._taskvine_spec = taskvine_spec or TaskVineArgumentSpec()
        self._extra_pip_local = extra_pip_local or {}
        self._extra_conda = list(extra_conda or ())
        self._default_executor = default_executor
        self._default_port = default_port
        self._default_environment = default_environment

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add executor-related options to ``parser``."""

        parser.add_argument(
            "--executor",
            "-x",
            default=self._default_executor,
            help="Which executor to use (futures, iterative, or taskvine)",
        )
        parser.add_argument(
            "--port",
            default=self._default_port,
            help="TaskVine manager port (PORT or PORT_MIN-PORT_MAX).",
        )
        parser.add_argument(
            "--environment-file",
            default=self._default_environment,
            help=(
                "Environment tarball for distributed executors ('cached', 'auto', 'none', or path)."
            ),
        )
        parser.add_argument(
            "--no-environment-file",
            dest="environment_file",
            action="store_const",
            const="none",
            help="Disable environment shipping (equivalent to --environment-file=none).",
        )

        if self._taskvine_spec.include_manager_name:
            parser.add_argument(
                "--manager-name",
                default=None,
                help="Override the distributed executor manager identifier.",
            )

        if self._taskvine_spec.include_manager_template:
            parser.add_argument(
                "--manager-name-template",
                default=None,
                help="Template for TaskVine manager names (use '{pid}' for the process id).",
            )

        if self._taskvine_spec.include_scratch_dir:
            parser.add_argument(
                "--scratch-dir",
                default=None,
                help="Shared staging directory for distributed executors.",
            )

        if self._taskvine_spec.include_resource_monitor:
            parser.add_argument(
                "--resource-monitor",
                default=self._taskvine_spec.resource_monitor_default,
                help="TaskVine resource monitor setting ('none' to disable).",
            )

        if self._taskvine_spec.include_resources_mode:
            parser.add_argument(
                "--resources-mode",
                default=self._taskvine_spec.resources_mode_default,
                help="TaskVine resources mode (for example auto).",
            )

        parser.add_argument(
            "--futures-workers",
            type=int,
            default=self._futures_spec.workers_default,
            help="Maximum number of local processes for the futures executor.",
        )

        if self._futures_spec.include_status:
            parser.add_argument(
                "--futures-status",
                action=argparse.BooleanOptionalAction,
                default=None,
                help="Toggle the coffea futures progress bar.",
            )

        if self._futures_spec.include_tail_timeout:
            parser.add_argument(
                "--futures-tail-timeout",
                type=int,
                default=None,
                help="Timeout in seconds for cancelling stalled futures tasks.",
            )

        if self._futures_spec.include_memory:
            parser.add_argument(
                "--futures-memory",
                type=int,
                default=None,
                help="Approximate per-worker memory budget in MB for dynamic chunk sizing.",
            )

        if self._futures_spec.include_prefetch:
            parser.add_argument(
                "--futures-prefetch",
                type=int,
                default=1,
                help="Number of input files to prefetch with the futures executor (0 disables).",
            )

        if self._futures_spec.include_retries:
            parser.add_argument(
                "--futures-retries",
                type=int,
                default=0,
                help="Number of times to retry a futures task after a failure before aborting.",
            )

        if self._futures_spec.include_retry_wait:
            parser.add_argument(
                "--futures-retry-wait",
                type=float,
                default=5.0,
                help="Seconds to wait between futures retry attempts.",
            )

    def parse_args(self, args: argparse.Namespace) -> ExecutorConfig:
        """Normalise CLI arguments into executor configuration objects."""

        executor = (getattr(args, "executor", "") or "").strip().lower() or self._default_executor
        port_spec = getattr(args, "port", self._default_port)
        port_range = parse_port_range(port_spec)
        environment_setting = getattr(args, "environment_file", self._default_environment)
        environment_file = resolve_environment_file(
            environment_setting,
            self._remote_environment,
            extra_pip_local=self._extra_pip_local,
            extra_conda=self._extra_conda,
        )

        futures_cfg = self._parse_futures(args)
        taskvine_cfg = self._parse_taskvine(args, executor, port_range, environment_file)

        return ExecutorConfig(
            executor=executor,
            futures=futures_cfg,
            taskvine=taskvine_cfg,
            environment_file=environment_file,
        )

    def _parse_futures(self, args: argparse.Namespace) -> FuturesConfig:
        workers = max(int(getattr(args, "futures_workers", self._futures_spec.workers_default) or 1), 1)
        status = getattr(args, "futures_status", None) if self._futures_spec.include_status else None

        tailtimeout: Optional[int]
        if self._futures_spec.include_tail_timeout:
            timeout_value = getattr(args, "futures_tail_timeout", None)
            if timeout_value and int(timeout_value) > 0:
                tailtimeout = int(timeout_value)
            else:
                tailtimeout = None
        else:
            tailtimeout = None

        memory: Optional[int]
        if self._futures_spec.include_memory:
            memory_value = getattr(args, "futures_memory", None)
            if memory_value and int(memory_value) > 0:
                memory = int(memory_value)
            else:
                memory = None
        else:
            memory = None

        prefetch: Optional[int]
        if self._futures_spec.include_prefetch:
            prefetch_value = getattr(args, "futures_prefetch", None)
            if prefetch_value is None:
                prefetch = None
            else:
                prefetch_int = int(prefetch_value)
                prefetch = max(prefetch_int, 0)
        else:
            prefetch = None

        retries: int
        if self._futures_spec.include_retries:
            retries = max(int(getattr(args, "futures_retries", 0) or 0), 0)
        else:
            retries = 0

        retry_wait: float
        if self._futures_spec.include_retry_wait:
            retry_wait = float(max(float(getattr(args, "futures_retry_wait", 0.0) or 0.0), 0.0))
        else:
            retry_wait = 0.0

        return FuturesConfig(
            workers=workers,
            status=status,
            tailtimeout=tailtimeout,
            memory=memory,
            prefetch=prefetch,
            retries=retries,
            retry_wait=retry_wait,
        )

    def _parse_taskvine(
        self,
        args: argparse.Namespace,
        executor: str,
        port_range: tuple[int, int],
        environment_file: Optional[str],
    ) -> TaskVineConfig:
        manager_name_arg = getattr(args, "manager_name", None) if self._taskvine_spec.include_manager_name else None
        manager_name = manager_name_arg or manager_name_base(executor)

        manager_template_arg = (
            getattr(args, "manager_name_template", None)
            if self._taskvine_spec.include_manager_template
            else None
        )
        manager_name_template = manager_template_arg or (
            f"{manager_name}-{{pid}}" if manager_name else None
        )

        scratch_dir_value = (
            getattr(args, "scratch_dir", None) if self._taskvine_spec.include_scratch_dir else None
        )
        scratch_dir = Path(scratch_dir_value).expanduser() if scratch_dir_value else None

        if self._taskvine_spec.include_resource_monitor:
            resource_monitor_setting = getattr(
                args,
                "resource_monitor",
                self._taskvine_spec.resource_monitor_default,
            )
        else:
            resource_monitor_setting = self._taskvine_spec.resource_monitor_default
        resource_monitor = _normalise_optional_string(resource_monitor_setting)

        if self._taskvine_spec.include_resources_mode:
            resources_mode_setting = getattr(
                args,
                "resources_mode",
                self._taskvine_spec.resources_mode_default,
            )
        else:
            resources_mode_setting = self._taskvine_spec.resources_mode_default
        resources_mode = _normalise_optional_string(resources_mode_setting)

        return TaskVineConfig(
            executor=executor,
            port_range=port_range,
            manager_name=manager_name,
            manager_name_template=manager_name_template,
            resource_monitor=resource_monitor,
            resources_mode=resources_mode,
            environment_file=environment_file,
            scratch_dir=scratch_dir,
        )


def _normalise_optional_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalised = str(value).strip()
    if not normalised or normalised.lower() in {"none", "off", "false", "0"}:
        return None
    return normalised


__all__ = [
    "ExecutorCLIHelper",
    "ExecutorConfig",
    "FuturesArgumentSpec",
    "FuturesConfig",
    "TaskVineArgumentSpec",
    "TaskVineConfig",
]

