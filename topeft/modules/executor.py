"""Helpers for constructing coffea executors consistently.

This module centralises the boilerplate required to launch coffea processors
across the supported executors (futures and TaskVine).  Several of the
``analysis`` entrypoints previously reimplemented the same dictionaries and
port negotiation logic; consolidating the helpers here keeps their behaviour in
sync and makes it easier to roll out configuration updates across the scripts.
"""

from __future__ import annotations

import errno
import os
import socket
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple
import warnings


def parse_port_range(port: str) -> Tuple[int, int]:
    """Normalise a ``PORT`` or ``PORT_MIN-PORT_MAX`` string to a tuple."""

    try:
        tokens = [int(token) for token in str(port).split("-") if token]
    except ValueError as exc:
        raise ValueError("Port specification must be an integer or range") from exc

    if not tokens:
        raise ValueError("At least one port value should be specified.")
    if len(tokens) > 2:
        raise ValueError("More than one port range was specified.")
    if len(tokens) == 1:
        tokens.append(tokens[0])
    return tokens[0], tokens[1]


def resolve_environment_file(
    setting: Optional[str],
    remote_environment: Any,
    *,
    extra_pip_local: Optional[Dict[str, Iterable[str]]] = None,
    extra_conda: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Normalise the distributed environment setting to an on-disk archive.

    Parameters
    ----------
    setting:
        The user-facing configuration value.  ``"auto"`` triggers environment
        packaging, ``"none"``/``None`` disables shipping, and any other string
        is returned verbatim.
    remote_environment:
        The :mod:`topcoffea.modules.remote_environment` module (or compatible)
        used to materialise the packaged archive.
    extra_pip_local, extra_conda:
        Forwarded to :func:`remote_environment.get_environment` when the
        ``"auto"`` path is requested.
    """

    if setting is None:
        return None

    normalised = str(setting).strip()
    if not normalised or normalised.lower() == "none":
        return None

    if normalised.lower() == "auto":
        return remote_environment.get_environment(
            extra_pip_local=extra_pip_local or {},
            extra_conda=list(extra_conda or ()),
        )

    return normalised


def manager_name_base(executor: str, *, default_user: Optional[str] = None) -> str:
    """Return the default manager name prefix for distributed executors."""

    user = os.environ.get("USER") or default_user
    if not user:
        try:
            import getpass

            user = getpass.getuser()
        except Exception:  # pragma: no cover - best effort fallback
            user = "coffea"
    return f"{user}-{executor}-coffea"


def distributed_logs_dir(staging_dir: Path, executor: str) -> Path:
    """Return the log directory for the requested executor, creating it."""

    if executor == "taskvine":
        logs_dir = staging_dir / "logs" / "taskvine"
    else:
        logs_dir = staging_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def _base_distributed_args(
    *,
    staging_dir: Path,
    extra_input_files: Sequence[str],
    resource_monitor: Optional[str],
    resources_mode: Optional[str],
    environment_file: Optional[str],
) -> Dict[str, Any]:
    args: Dict[str, Any] = {
        "filepath": str(staging_dir),
        "extra_input_files": list(extra_input_files),
        "retries": 15,
        "compression": 8,
        "fast_terminate_workers": 0,
        "verbose": True,
        "print_stdout": False,
    }
    if resource_monitor is not None:
        args["resource_monitor"] = resource_monitor
    if resources_mode is not None:
        args["resources_mode"] = resources_mode
    if environment_file:
        args["environment_file"] = environment_file
    return args


def build_taskvine_args(
    *,
    staging_dir: Path,
    logs_dir: Path,
    manager_name: Optional[str],
    manager_name_template: Optional[str],
    extra_input_files: Sequence[str],
    resource_monitor: Optional[str],
    resources_mode: Optional[str],
    environment_file: Optional[str],
    custom_init: Optional[Callable[[Any], None]] = None,
) -> Dict[str, Any]:
    """Return TaskVine executor keyword arguments with shared defaults."""

    args = _base_distributed_args(
        staging_dir=staging_dir,
        extra_input_files=extra_input_files,
        resource_monitor=resource_monitor,
        resources_mode=resources_mode,
        environment_file=environment_file,
    )
    if manager_name:
        args["manager_name"] = manager_name
    if manager_name_template:
        args["manager_name_template"] = manager_name_template
    if custom_init is not None:
        args["custom_init"] = custom_init

    # TaskVine logs are configured through the manager during initialisation, so
    # we do not inject explicit file paths here beyond the ``custom_init`` hook.
    return args


def taskvine_log_configurator(logs_dir: Path) -> Callable[[Any], None]:
    """Return a ``custom_init`` callback that enables TaskVine manager logs."""

    logs_dir.mkdir(parents=True, exist_ok=True)

    def _configure(manager: Any) -> None:
        log_map = {
            "set_debug_log": logs_dir / "debug.log",
            "set_transactions_log": logs_dir / "transactions.log",
            "set_stats_file": logs_dir / "stats.log",
            "set_tasks_accumulation_file": logs_dir / "tasks.log",
        }
        for method_name, log_path in log_map.items():
            setter = getattr(manager, method_name, None)
            if callable(setter):
                try:
                    setter(str(log_path))
                except Exception:
                    warnings.warn(
                        f"TaskVine manager could not enable {method_name} at {log_path}.",
                        RuntimeWarning,
                    )

    return _configure


def instantiate_taskvine_executor(
    processor_module: Any,
    base_args: Dict[str, Any],
    *,
    port_range: Tuple[int, int],
    negotiate_port: bool = True,
) -> Any:
    """Instantiate ``processor.TaskVineExecutor`` with port negotiation."""

    taskvine_cls = getattr(processor_module, "TaskVineExecutor", None)
    if taskvine_cls is None:
        raise RuntimeError("TaskVineExecutor not available.")

    port_min, port_max = port_range
    if port_min > port_max:
        raise ValueError("Invalid port range: minimum exceeds maximum.")

    def _attempt_instantiate(attempt_args: Dict[str, Any]) -> Any:
        while True:
            try:
                return taskvine_cls(**attempt_args)
            except TypeError as exc:
                if (
                    "manager_name_template" in attempt_args
                    and "manager_name_template" in str(exc)
                ):
                    attempt_args.pop("manager_name_template", None)
                    continue
                raise

    if not negotiate_port:
        attempt_args = dict(base_args)
        attempt_args["port"] = port_min
        try:
            return _attempt_instantiate(attempt_args)
        except Exception as exc:  # pragma: no cover - best effort
            if _is_port_allocation_error(exc):
                raise RuntimeError(
                    f"TaskVineExecutor could not bind manager port {port_min}."
                ) from exc
            raise

    attempted_ports: set[int] = set()
    last_port_error: Optional[BaseException] = None
    for _ in range(port_min, port_max + 1):
        port = _select_manager_port(port_min, port_max, exclude=attempted_ports)
        attempted_ports.add(port)

        attempt_args = dict(base_args)
        attempt_args["port"] = port

        try:
            return _attempt_instantiate(attempt_args)
        except Exception as exc:
            if _is_port_allocation_error(exc):
                last_port_error = exc
                continue
            raise

    range_desc = f"{port_min}-{port_max}" if port_min != port_max else str(port_min)
    message = (
        "TaskVineExecutor could not bind a manager port in range " f"{range_desc}."
    )
    if last_port_error is not None:
        raise RuntimeError(message) from last_port_error
    raise RuntimeError(message)


def _select_manager_port(
    port_min: int,
    port_max: int,
    *,
    exclude: Optional[Iterable[int]] = None,
) -> int:
    if port_min > port_max:
        raise ValueError("Invalid port range: minimum exceeds maximum.")

    excluded = set(exclude or ())
    for port in range(port_min, port_max + 1):
        if port in excluded:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                candidate.bind(("", port))
            except OSError:
                continue
            return port

    raise RuntimeError(
        f"No available port found in the requested range {port_min}-{port_max}."
    )


def _is_port_allocation_error(exc: BaseException) -> bool:
    if isinstance(exc, OSError) and exc.errno == errno.EADDRINUSE:
        return True
    message = str(exc).lower()
    return "address already in use" in message or (
        "port" in message and "in use" in message
    )


def build_futures_executor(
    processor_module: Any,
    *,
    workers: int,
    status: Optional[bool] = None,
    tailtimeout: Optional[int] = None,
) -> Any:
    """Instantiate the coffea futures executor with shared defaults."""

    executor_args: Dict[str, Any] = {"workers": max(int(workers), 1)}
    if status is not None:
        executor_args["status"] = bool(status)
    if tailtimeout is not None:
        executor_args["tailtimeout"] = int(tailtimeout)

    futures_cls = getattr(processor_module, "FuturesExecutor", None)
    if futures_cls is None:  # pragma: no cover - depends on coffea build
        futures_cls = getattr(processor_module, "futures_executor")
    return futures_cls(**executor_args)


def futures_runner_overrides(
    runner_fields: Iterable[str],
    *,
    memory: Optional[int] = None,
    prefetch: Optional[int] = None,
) -> Dict[str, Any]:
    """Return Runner keyword overrides that mirror futures configuration."""

    fields = set(runner_fields)
    overrides: Dict[str, Any] = {}

    if memory is not None and "dynamic_chunksize" in fields:
        overrides["dynamic_chunksize"] = {"memory": int(memory)}

    if (
        prefetch is not None
        and prefetch > 0
        and "prefetch" in fields
    ):
        overrides["prefetch"] = int(prefetch)

    return overrides

