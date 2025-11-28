#!/usr/bin/env python

"""Command-line interface for the Run 2 analysis workflow.

The YAML-first configuration pipeline, helper responsibilities, and common
extension points are documented in ``docs/run_analysis_configuration.md``.  For
a step-by-step walkthrough of environment prerequisites, metadata bundles, and
example invocations that mirror ``fullR2_run.sh``, consult the
``docs/quickstart_top22_006.md`` guide.
"""

from __future__ import annotations

import argparse
import importlib
from typing import Sequence

import topcoffea


def _verify_numpy_pandas_abi() -> None:
    """Ensure pandas and NumPy load with matching binary interfaces.

    When a pandas wheel compiled against an older NumPy ABI sneaks into the
    environment, imports can fail deep inside the Run 2 workflow (for example
    during ``topeft.modules.systematics`` initialization).  Catch the issue
    early with a lightweight import and extension-module check so users see an
    actionable hint instead of an opaque crash.
    """

    try:
        np = importlib.import_module("numpy")
        pd = importlib.import_module("pandas")
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Failed to import numpy/pandas before launching the workflow. "
            "Recreate the coffea2025 environment and rebuild the TaskVine "
            "tarball before rerunning: `conda env update -f environment.yml "
            "--prune` and `python -m topcoffea.modules.remote_environment`."
        ) from exc

    try:  # pragma: no cover - environment guard
        from pandas import _libs as _pd_libs

        # Touching a compiled extension exercises the linked NumPy ABI.
        _ = _pd_libs.hashtable.Int64HashTable
    except Exception as exc:
        raise RuntimeError(
            "Detected a pandas/NumPy ABI mismatch (numpy "
            f"{np.__version__}, pandas {pd.__version__}). Recreate the "
            "coffea2025 environment and rebuild the TaskVine tarball: "
            "`conda env update -f environment.yml --prune` followed by "
            "`python -m topcoffea.modules.remote_environment`. Use the "
            "refreshed environment for both futures and TaskVine runs."
        ) from exc

from run_analysis_helpers import RunConfig, RunConfigBuilder
from topeft.modules.executor_cli import (
    ExecutorCLIHelper,
    FuturesArgumentSpec,
    TaskVineArgumentSpec,
)
from topeft.modules.executor import resolve_environment_file

from analysis.topeft_run2.workflow import run_workflow
from analysis.topeft_run2.logging_utils import configure_logging

remote_environment = topcoffea.modules.remote_environment


EXECUTOR_CLI = ExecutorCLIHelper(
    remote_environment=remote_environment,
    futures_spec=FuturesArgumentSpec(
        workers_default=8,
        include_status=True,
        include_tail_timeout=True,
        include_memory=True,
        include_prefetch=True,
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
    default_environment="cached",
)


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser used by ``run_analysis.py``."""

    parser = argparse.ArgumentParser(
        description="You can customize your run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "TaskVine workers can be launched with:\n"
            "  vine_submit_workers --python-env \"$(python -m topcoffea.modules.remote_environment)\" \\\n"
            "    --cores 4 --memory 16000 --disk 16000 -M <manager-name>\n"
            "run_analysis expects a cached remote environment tarball by default\n"
            "(--environment-file=cached). Use --environment-file auto to rebuild\n"
            "the archive on demand. Adjust the resources and manager name to\n"
            "match your deployment."
        ),
    )
    parser.add_argument(
        "jsonFiles",
        nargs="?",
        default="",
        help="Json file(s) containing files and metadata",
    )
    parser.add_argument(
        "--prefix",
        "-r",
        nargs="?",
        default="",
        help="Prefix or redirector to look for the files",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="To perform a test, run over a few events in a couple of chunks",
    )
    parser.add_argument(
        "--pretend",
        action="store_true",
        help="Read json files but do not execute the analysis",
    )
    EXECUTOR_CLI.configure_parser(parser)
    parser.add_argument(
        "--nworkers",
        "-n",
        type=int,
        default=8,
        help="Number of workers",
    )
    parser.add_argument(
        "--chunksize",
        "-s",
        default=100000,
        help="Number of events per chunk",
    )
    parser.add_argument(
        "--nchunks",
        "-c",
        default=None,
        help="You can choose to run only a number of chunks",
    )
    parser.add_argument(
        "--outname",
        "-o",
        default="plotsTopEFT",
        help="Name of the output file with histograms",
    )
    parser.add_argument(
        "--outpath",
        "-p",
        default="histos",
        help="Name of the output directory",
    )
    parser.add_argument(
        "--treename",
        default="Events",
        help="Name of the tree inside the files",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help=(
            "Path to the metadata YAML describing channels, variables, and"
            " systematics. Defaults to topeft/params/metadata.yml when"
            " omitted."
        ),
    )
    parser.add_argument(
        "--do-errors",
        action="store_true",
        help="Save the w**2 coefficients",
    )
    parser.add_argument(
        "--do-systs",
        action="store_true",
        help="Compute systematic variations",
    )
    parser.add_argument(
        "--split-lep-flavor",
        action="store_true",
        help="Split up categories by lepton flavor",
    )
    parser.add_argument(
        "--summary-verbosity",
        choices=["none", "brief", "full"],
        default="brief",
        help=(
            "Control the histogram summary emitted before task submission. "
            "'none' disables the summary, 'brief' prints bullet lists of the "
            "planned samples, channel/application pairs, variables, and "
            "systematics, and 'full' prepends those lists to the per-combination "
            "table plus the structured dump (including a note when "
            "--split-lep-flavor is active)."
        ),
    )
    parser.add_argument(
        "--debug-logging",
        action="store_true",
        default=False,
        help=(
            "Enable verbose AnalysisProcessor debugging/instrumentation. "
            "Leave unset to suppress debug logs during normal production runs."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help=(
            "Set the Python logging level "
            "(DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to INFO when unset."
        ),
    )
    parser.add_argument(
        "--log-tasks",
        action="store_true",
        help=(
            "Print a single-line futures submission log for each histogram task, "
            "showing the (sample, channel, variable, application, systematic) tuple."
        ),
    )
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        help=(
            "Scenario name defined in metadata to select channel groups."
            " Defaults to 'TOP_22_006' when not provided. Can be supplied"
            " multiple times to combine scenarios."
        ),
    )
    parser.add_argument(
        "--skip-sr",
        action="store_true",
        help="Skip all signal region categories",
    )
    parser.add_argument(
        "--skip-cr",
        action="store_true",
        help="Skip all control region categories",
    )
    parser.add_argument(
        "--do-np",
        action="store_true",
        help=(
            "Perform nonprompt estimation on the output hist, and save a new hist "
            "with the np contribution included. Signal, background and data samples "
            "must all be processed together."
        ),
    )
    parser.add_argument(
        "--do-renormfact-envelope",
        action="store_true",
        help=(
            "Perform renorm/fact envelope calculation on the output hist "
            "(saves the modified with the same name as the original)."
        ),
    )
    parser.add_argument(
        "--wc-list",
        action="extend",
        nargs="+",
        help="Specify a list of Wilson coefficients to use in filling histograms.",
    )
    parser.add_argument(
        "--ecut",
        default=None,
        help="Energy cut threshold i.e. throw out events above this (GeV)",
    )
    parser.add_argument(
        "--no-port-negotiation",
        dest="negotiate_manager_port",
        action="store_false",
        help=(
            "Disable automatic TaskVine port negotiation. When set the first value "
            "from --port is used directly and any allocation failure aborts the run."
        ),
    )
    parser.add_argument(
        "--options",
        default=None,
        help=(
            "YAML file that specifies command-line options. Accepts either"
            " 'path.yml' for the default profile or 'path.yml:profile' to select"
            " a specific profile. When provided, CLI flags are ignored in favour"
            " of the YAML configuration."
        ),
    )
    parser.set_defaults(negotiate_manager_port=True)
    return parser


def _resolve_logging_controls(config: RunConfig) -> tuple[str, bool]:
    """Return the log level and processor-debug flag based on CLI inputs."""

    if config.debug_logging:
        return "DEBUG", True
    if config.log_level:
        normalized = config.log_level.upper()
        return normalized, normalized == "DEBUG"
    return "INFO", False


def main(argv: Sequence[str] | None = None) -> None:
    _verify_numpy_pandas_abi()

    parser = build_parser()
    parser_defaults = parser.parse_args([])
    args = parser.parse_args(argv)
    executor_choice = (getattr(args, "executor", "") or "").strip().lower()
    if not executor_choice:
        executor_choice = "taskvine"
    setattr(args, "executor", executor_choice)

    config_builder = RunConfigBuilder(parser_defaults)
    config = config_builder.build(
        args,
        getattr(args, "options", None),
    )

    effective_log_level, processor_debug = _resolve_logging_controls(config)
    # Currently configures logging for the driver process; futures workers keep
    # their default handlers until we plumb a per-worker hook.
    configure_logging(effective_log_level)
    config.log_level = effective_log_level
    config.debug_logging = processor_debug

    if config.executor == "taskvine":
        config.environment_file = resolve_environment_file(
            config.environment_file,
            remote_environment,
            extra_pip_local={"topeft": ["topeft", "setup.py"]},
            extra_conda=["pyyaml"],
        )

    run_workflow(config)


if __name__ == "__main__":
    main()
