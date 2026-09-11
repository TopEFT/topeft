#!/usr/bin/env python3
"""Standalone helper to build data-driven histograms from a saved PKL.

Quickstart examples:
  - Direct pickle paths: python run_data_driven.py --input-pkl histos/plotsTopEFT.pkl.gz \
      --output-pkl histos/plotsTopEFT_np.pkl.gz
  - Legacy/materialized fallback: add --legacy-dict-mode to restore the
      original fully materialized dict workflow.

By default the helper uses the streaming iterator path, writing output with
``dump_dict_streaming(..., protocol=3, clear_memo_interval=1)`` to cap RSS.
"""

from __future__ import annotations

import argparse
import ast
import copy
import ctypes
import gc
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import threading
import time
import tracemalloc
from typing import Any, Dict, Iterable, List, Optional, Tuple

import topcoffea.modules.utils as utils

from topeft.modules.dataDrivenEstimation import DataDrivenProducer
from topeft.modules.data_driven_products import (
    FLIPS_OUTPUT_ARTIFACT_KIND,
    generated_output_processes_from_contract,
    NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND,
    NONPROMPT_OUTPUT_ARTIFACT_KIND,
    validate_requested_product_input,
)
from topeft.modules.histogram_artifact import (
    lineage_input_from_sidecar,
    read_histogram_sidecar,
    validate_histogram_artifact,
    write_histogram_artifact,
)
from topeft.modules.nominal_schema import EFT_NOMINAL_SUFFIX
from topeft.modules.get_renormfact_envelope import raise_unsupported_renormfact_envelope
from topeft.modules.sumw2_policy import resolved_policy_from_provenance
from analysis.topeft_run2 import analysis_processor

_STREAMING_PICKLE_PROTOCOL = 3
_STREAMING_MEMO_CLEAR_INTERVAL = 1
_LIBC = None
_DD_REPORT_FAMILY_ORDER = {"sr": 0, "nonprompt": 1, "flips": 2}


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize nonprompt/flips histograms from a processor PKL. Its artifact "
            "sidecar is discovered automatically.\n\n"
            "Quickstart:\n"
            "  python run_data_driven.py --input-pkl histos/plotsTopEFT.pkl.gz\\\n"
            "      --output-pkl histos/plotsTopEFT_np.pkl.gz\n"
            "Default mode is streaming iterator mode (lower peak RSS). "
            "Pass --legacy-dict-mode to restore the original materialized-dict behavior.\n"
            f"Streaming serialization defaults are hardcoded to protocol={_STREAMING_PICKLE_PROTOCOL} "
            f"and clear_memo_interval={_STREAMING_MEMO_CLEAR_INTERVAL}."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-pkl",
        required=True,
        help="Path to the histogram pickle emitted by run_analysis.py (pre data-driven step).",
    )
    parser.add_argument(
        "--output-pkl",
        help="Destination for the histogram pickle with data-driven contributions applied.",
    )
    parser.add_argument(
        "--legacy-campaign-state",
        help=(
            "Transient maintained campaign-state context for a declaration-free "
            "schema-v2 processor artifact. Must be paired with --legacy-campaign-block."
        ),
    )
    parser.add_argument(
        "--legacy-campaign-block",
        help=(
            "Exact block id inside --legacy-campaign-state. It transports provenance "
            "and execution scope, never an applicability verdict."
        ),
    )
    parser.add_argument(
        "--apply-renormfact-envelope",
        action="store_true",
        help=(
            "Deprecated unsupported option. It exits before opening the input or creating output."
        ),
    )
    parser.add_argument(
        "--only-flips",
        action="store_true",
        help="Drop nonprompt processes so only flips contributions remain in the output histograms.",
    )
    parser.add_argument(
        "--nominal-only-reference",
        action="store_true",
        help=(
            "Create an explicitly non-card-ready nominal-only nonprompt reference. "
            "It records missing prompt sumw2 and never fabricates second moments."
        ),
    )
    parser.add_argument(
        "--dd-report",
        action="store_true",
        help=(
            "Print a compact text report of the raw data-driven inputs and outputs "
            "before only-flips filtering and renorm/fact-envelope postprocessing."
        ),
    )
    parser.add_argument(
        "--dd-report-md",
        help=(
            "Write a detailed DD report to a Markdown file. Does not print to "
            "stdout unless --dd-report is also passed."
        ),
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help=(
            "Emit a progress heartbeat while histograms are finalized. "
            "Set to 0 to log every histogram; combine with --quiet to suppress the heartbeat."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence progress heartbeats during histogram finalization.",
    )
    parser.add_argument(
        "--mem-report",
        action="store_true",
        help=(
            "Print stage-tagged memory (RSS) usage and periodic memory heartbeats while "
            "processing histograms."
        ),
    )
    parser.add_argument(
        "--mem-tracemalloc",
        action="store_true",
        help=(
            "Also collect and print tracemalloc top allocations at major stages. "
            "Implies --mem-report."
        ),
    )
    parser.add_argument(
        "--mem-top-n",
        type=int,
        default=20,
        help="How many tracemalloc entries to print per stage when --mem-tracemalloc is enabled.",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--legacy-dict-mode",
        action="store_true",
        help=(
            "Restore the original materialized-dict path (higher peak RSS): "
            "build the full histogram dict in memory and write it with dump_to_pkl."
        ),
    )
    # Backward-compatible no-op alias: the default is already iterator mode.
    mode_group.add_argument(
        "--iterator-mode",
        dest="legacy_dict_mode",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(legacy_dict_mode=False)
    return parser


def _default_output_path(input_path: str, *, nominal_only_reference: bool = False) -> str:
    if input_path.endswith(".pkl.gz"):
        base = input_path[:-7]
    elif input_path.endswith(".pkl"):
        base = input_path[:-4]
    else:
        base = input_path
    suffix = "_np_nominal_reference" if nominal_only_reference else "_np"
    return f"{base}{suffix}.pkl.gz"


def _validate_input_path(input_path: str) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Histogram pickle not found: {input_path}")


def _command_option_values(command, option):
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1:
        raise RuntimeError(
            f"Legacy source command must contain exactly one {option}; found {len(positions)}."
        )
    values = []
    for value in command[positions[0] + 1 :]:
        if value.startswith("-"):
            break
        values.append(value)
    if not values:
        raise RuntimeError(f"Legacy source command has no values for {option}.")
    return values


def _command_analysis_mode(command):
    mode_flags = {
        "--all-analysis": "all",
        "--offZ-3l-split": "offz",
        "--tau-h-analysis": "tau",
        "--fwd-analysis": "fwd",
    }
    selected = [mode for flag, mode in mode_flags.items() if flag in command]
    if len(selected) > 1:
        raise RuntimeError("Legacy source command contains conflicting analysis modes.")
    return selected[0] if selected else "default"


def _selected_category_dicts(
    *,
    analysis_mode,
    region,
    category_groups,
    category_config,
):
    mode_flags = {
        "all": (False, False, False, True),
        "offz": (True, False, False, False),
        "tau": (False, True, False, False),
        "fwd": (False, False, True, False),
        "default": (False, False, False, False),
    }
    if analysis_mode not in mode_flags:
        raise RuntimeError(f"Unsupported legacy analysis mode {analysis_mode!r}.")
    sr_name, cr_name = analysis_processor.resolve_category_dict_names(
        *mode_flags[analysis_mode]
    )
    block_name = sr_name if region == "SR" else cr_name if region == "CR" else None
    if block_name is None or block_name not in category_config:
        raise RuntimeError(f"Unsupported legacy region {region!r}.")
    selected = {}
    for group in category_groups:
        if group not in category_config[block_name]:
            raise RuntimeError(
                f"Legacy category group {group!r} is absent from {block_name}."
            )
        selected[group] = category_config[block_name][group]
    return (selected, {}) if region == "SR" else ({}, selected)


def _git_show_text(repository_root, producer_commit, repository_path):
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "show",
            f"{producer_commit}:{repository_path}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Legacy producer commit {producer_commit} is unavailable or lacks "
            f"{repository_path}."
        )
    return result.stdout


def _load_legacy_category_config(*, state, block, repository_root, producer_commit):
    retained_path_text = block.get(
        "retained_category_config_path",
        state.get("retained_category_config_path"),
    )
    if retained_path_text is not None:
        if not isinstance(retained_path_text, str) or not retained_path_text:
            raise RuntimeError(
                "Legacy retained_category_config_path must be a nonempty path."
            )
        retained_path = Path(retained_path_text)
        try:
            config_text = retained_path.read_text(encoding="utf-8")
        except OSError as error:
            raise RuntimeError(
                f"Cannot read retained legacy category configuration {retained_path}: "
                f"{error}"
            ) from error
        scope_source = "retained_execution_config"
        scope_locator = str(retained_path)
    else:
        config_text = _git_show_text(
            repository_root,
            producer_commit,
            "topeft/channels/ch_lst.json",
        )
        scope_source = "producer_commit_config"
        scope_locator = (
            f"{producer_commit}:topeft/channels/ch_lst.json"
        )
    try:
        category_config = json.loads(config_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Legacy category configuration {scope_locator} is not valid JSON: {error}"
        ) from error
    if not isinstance(category_config, dict):
        raise RuntimeError(
            f"Legacy category configuration {scope_locator} must be an object."
        )
    return category_config, scope_source, scope_locator


def _historical_legacy_producer_proxy(source_text, analysis_mode):
    """Build a bounded proxy from only the historical applicability methods."""

    method_names = analysis_processor._HISTOGRAM_APPLICABILITY_LEGACY_METHODS
    try:
        tree = ast.parse(source_text)
    except SyntaxError as error:
        raise RuntimeError("Historical producer source is not parseable.") from error
    analysis_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AnalysisProcessor"
        ),
        None,
    )
    if analysis_class is None:
        raise RuntimeError("Historical producer source lacks AnalysisProcessor.")
    methods = {
        node.name: node
        for node in analysis_class.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    }
    attributes = {}
    for method_name in method_names:
        if method_name not in methods:
            def _missing_historical_method(*_args, _method_name=method_name, **_kwargs):
                raise RuntimeError(
                    "Historical producer source cannot resolve the relevant "
                    f"applicability method {_method_name}."
                )

            attributes[method_name] = (
                staticmethod(_missing_historical_method)
                if method_name != "_should_skip_histogram_fill"
                else _missing_historical_method
            )
            continue
        method_node = copy.deepcopy(methods[method_name])
        method_node.decorator_list = []
        module = ast.fix_missing_locations(
            ast.Module(body=[method_node], type_ignores=[])
        )
        namespace = {}
        exec(
            compile(module, "<historical_analysis_processor>", "exec"),
            namespace,
        )
        method = namespace[method_name]
        attributes[method_name] = (
            staticmethod(method)
            if method_name != "_should_skip_histogram_fill"
            else method
        )

    proxy_class = type(
        "HistoricalLegacyApplicabilityProxy",
        (analysis_processor.AnalysisProcessor,),
        attributes,
    )
    proxy = object.__new__(proxy_class)
    proxy._analysis_mode = analysis_mode
    return proxy


def _applicability_decision_differences(left, right):
    differences = []
    left_families = left["families"]
    right_families = right["families"]
    for family in sorted(set(left_families) | set(right_families)):
        left_channels = left_families.get(family, {}).get("channels", {})
        right_channels = right_families.get(family, {}).get("channels", {})
        for channel in sorted(set(left_channels) | set(right_channels)):
            left_state = left_channels.get(channel)
            right_state = right_channels.get(channel)
            if left_state != right_state:
                differences.append(
                    {
                        "family": family,
                        "channel": channel,
                        "historical": left_state,
                        "current": right_state,
                    }
                )
    return differences


def _resolve_legacy_histogram_applicability(
    *,
    input_pkl,
    input_sidecar,
    campaign_state_path,
    campaign_block_id,
):
    """Qualify transient legacy context and query the producer authority."""

    state_path = Path(campaign_state_path)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot read legacy campaign context {state_path}: {error}"
        ) from error
    matching_blocks = [
        block for block in state.get("blocks", []) if block.get("id") == campaign_block_id
    ]
    if len(matching_blocks) != 1:
        raise RuntimeError(
            f"Legacy campaign context must contain exactly one block {campaign_block_id!r}."
        )
    block = matching_blocks[0]
    if block.get("source_status") != "ready":
        raise RuntimeError(
            f"Legacy campaign block {campaign_block_id!r} is not a reusable ready source."
        )
    expected_source_path = block.get("expected_nominal_path")
    if not isinstance(expected_source_path, str) or not expected_source_path:
        raise RuntimeError("Legacy campaign block lacks expected_nominal_path.")
    observed_identity = input_sidecar["artifact"]
    expected_path = Path(expected_source_path).resolve()
    observed_path = Path(input_pkl).resolve()
    if observed_path == expected_path:
        context_binding_method = "resolved_input_path"
    else:
        expected_sidecar = read_histogram_sidecar(expected_source_path)
        expected_sha256 = expected_sidecar.get("artifact", {}).get("pkl_sha256")
        observed_sha256 = observed_identity.get("pkl_sha256")
        if (
            not isinstance(expected_sha256, str)
            or not isinstance(observed_sha256, str)
            or expected_sha256 != observed_sha256
        ):
            raise RuntimeError(
                "Legacy context/source binding failed: the consumed input is neither "
                "the recorded source path nor a SHA-identified copy of that source."
            )
        context_binding_method = "frozen_source_sha256"

    source_command = block.get("source_command_argv")
    if not isinstance(source_command, list) or any(
        not isinstance(value, str) for value in source_command
    ):
        raise RuntimeError("Legacy campaign block lacks a valid source_command_argv.")
    category_groups = list(
        dict.fromkeys(_command_option_values(source_command, "--category-groups"))
    )
    command_histogram_families = list(
        dict.fromkeys(_command_option_values(source_command, "--hist-vars"))
    )
    years = list(dict.fromkeys(_command_option_values(source_command, "-y")))
    diagnostic_provenance_differences = []
    for field, command_values in (
        ("category_groups", category_groups),
        ("histograms", command_histogram_families),
        ("years", years),
    ):
        recorded_values = block.get(field)
        if not isinstance(recorded_values, list) or {
            str(value) for value in recorded_values
        } != set(command_values):
            diagnostic_provenance_differences.append(field)
    region_flags = [region for flag, region in (("--sr", "SR"), ("--cr", "CR")) if flag in source_command]
    if len(region_flags) != 1:
        raise RuntimeError("Legacy source command does not resolve exactly one analysis region.")
    if state.get("region") != region_flags[0]:
        diagnostic_provenance_differences.append("region")
    analysis_mode = _command_analysis_mode(source_command)
    runtime_families = list(
        resolved_policy_from_provenance(
            input_sidecar["sumw2_storage_provenance"]
        ).runtime_histogram_families
    )
    if set(command_histogram_families) != set(runtime_families):
        raise RuntimeError(
            "Legacy semantic scope resolution failed: the requested family set "
            "disagrees with the source artifact."
        )

    producer_commit = state.get("topeft_git_commit")
    if not isinstance(producer_commit, str) or re.fullmatch(
        r"[0-9a-f]{40}", producer_commit
    ) is None:
        raise RuntimeError(
            "Legacy semantic scope resolution lacks an exact producer commit locator."
        )
    repository_root = Path(__file__).resolve().parents[2]
    category_config, historical_scope_source, historical_scope_locator = (
        _load_legacy_category_config(
            state=state,
            block=block,
            repository_root=repository_root,
            producer_commit=producer_commit,
        )
    )
    historical_source = _git_show_text(
        repository_root,
        producer_commit,
        "analysis/topeft_run2/analysis_processor.py",
    )
    try:
        historical_semantics = (
            analysis_processor.legacy_histogram_applicability_semantics_sha256(
                historical_source
            )
        )
    except (SyntaxError, ValueError):
        historical_semantics = None
    current_legacy_semantics = (
        analysis_processor.legacy_histogram_applicability_semantics_sha256()
    )

    selected_category_dicts = _selected_category_dicts(
        analysis_mode=analysis_mode,
        region=region_flags[0],
        category_groups=category_groups,
        category_config=category_config,
    )
    applicability = analysis_processor.AnalysisProcessor.build_histogram_applicability(
        analysis_mode=analysis_mode,
        runtime_families=runtime_families,
        selected_category_dicts=selected_category_dicts,
        split_by_lepton_flavor="--split-lep-flavor" in source_command,
        is_run3_values=tuple(
            dict.fromkeys(year.startswith("202") for year in years)
        ),
    )
    semantic_comparison = "legacy_method_semantics_identical"
    if historical_semantics is None or historical_semantics != current_legacy_semantics:
        historical_proxy = _historical_legacy_producer_proxy(
            historical_source,
            analysis_mode,
        )
        historical_applicability = (
            analysis_processor.AnalysisProcessor.build_histogram_applicability(
                analysis_mode=analysis_mode,
                runtime_families=runtime_families,
                selected_category_dicts=selected_category_dicts,
                split_by_lepton_flavor="--split-lep-flavor" in source_command,
                is_run3_values=tuple(
                    dict.fromkeys(year.startswith("202") for year in years)
                ),
                producer_proxy=historical_proxy,
            )
        )
        decision_differences = _applicability_decision_differences(
            historical_applicability,
            applicability,
        )
        if decision_differences:
            raise RuntimeError(
                "Legacy applicability decisions differ over the resolved semantic "
                "scope: "
                + json.dumps(decision_differences, sort_keys=True)
            )
        semantic_comparison = "relevant_binary_decisions_equal"
    return applicability, {
        "context_binding_method": context_binding_method,
        "source_pkl_sha256": observed_identity.get("pkl_sha256"),
        "producer_topeft_commit": producer_commit,
        "producer_semantics_sha256": applicability["producer_semantics_sha256"],
        "historical_legacy_semantics_sha256": historical_semantics,
        "current_legacy_semantics_sha256": current_legacy_semantics,
        "producer_semantic_comparison": semantic_comparison,
        "historical_scope_source": historical_scope_source,
        "historical_scope_locator": historical_scope_locator,
        "analysis_mode": analysis_mode,
        "region": region_flags[0],
        "category_groups": category_groups,
        "histogram_families": runtime_families,
        "years": years,
        "split_by_lepton_flavor": "--split-lep-flavor" in source_command,
        "diagnostic_provenance_differences": sorted(
            diagnostic_provenance_differences
        ),
    }


def _current_local_category_config_candidate(repository_root):
    repository_path = "topeft/channels/ch_lst.json"
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            repository_path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    state = result.stdout.strip()
    path = repository_root / repository_path
    if result.returncode != 0 or not state or not path.is_file():
        return None
    return state, path


def _validate_legacy_histogram_artifact(input_pkl, applicability, resolution):
    try:
        return validate_histogram_artifact(
            input_pkl,
            histogram_applicability=applicability,
        )
    except ValueError as historical_error:
        error_text = str(historical_error)
        if not any(
            marker in error_text
            for marker in ("structurally absent", "structurally present")
        ):
            raise

        repository_root = Path(__file__).resolve().parents[2]
        candidate = _current_local_category_config_candidate(repository_root)
        if candidate is None:
            raise
        local_state, local_path = candidate
        try:
            local_config = json.loads(local_path.read_text(encoding="utf-8"))
            selected_category_dicts = _selected_category_dicts(
                analysis_mode=resolution["analysis_mode"],
                region=resolution["region"],
                category_groups=resolution["category_groups"],
                category_config=local_config,
            )
            local_applicability = (
                analysis_processor.AnalysisProcessor.build_histogram_applicability(
                    analysis_mode=resolution["analysis_mode"],
                    runtime_families=resolution["histogram_families"],
                    selected_category_dicts=selected_category_dicts,
                    split_by_lepton_flavor=resolution["split_by_lepton_flavor"],
                    is_run3_values=tuple(
                        dict.fromkeys(
                            year.startswith("202") for year in resolution["years"]
                        )
                    ),
                )
            )
            validate_histogram_artifact(
                input_pkl,
                histogram_applicability=local_applicability,
            )
        except Exception:
            raise historical_error
        raise RuntimeError(
            "historical_config_execution_ambiguity: the recorded-commit category "
            "configuration contradicts artifact structure, while one bounded "
            f"diagnostic with local config {local_path} ({local_state}) is "
            "structurally consistent; the local file was not promoted to semantic "
            "authority."
        ) from historical_error


def _peak_rss_mb() -> float:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports ru_maxrss in KiB; macOS reports bytes.
    if sys.platform == "darwin":
        return peak_rss / (1024.0 * 1024.0)
    return peak_rss / 1024.0


def _current_rss_mb() -> float:
    try:
        with open("/proc/self/status") as status_stream:
            for line in status_stream:
                if line.startswith("VmRSS:"):
                    fields = line.split()
                    if len(fields) >= 2:
                        return float(fields[1]) / 1024.0
    except OSError:
        pass
    # Fallback when /proc is unavailable.
    return _peak_rss_mb()


def _trim_allocator() -> None:
    global _LIBC
    if sys.platform != "linux":
        return
    if _LIBC is False:
        return
    if _LIBC is None:
        try:
            _LIBC = ctypes.CDLL("libc.so.6")
        except OSError:
            _LIBC = False
            return
    _LIBC.malloc_trim(0)


class _MemoryReporter:
    def __init__(
        self,
        *,
        enabled: bool,
        include_tracemalloc: bool,
        heartbeat_seconds: float,
        top_n: int,
    ) -> None:
        self.enabled = enabled
        self.include_tracemalloc = include_tracemalloc
        self.heartbeat_seconds = heartbeat_seconds
        self.top_n = max(1, top_n)
        self._stage = "startup"
        self._stage_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.enabled:
            return
        if self.include_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start(25)

        interval = self.heartbeat_seconds if self.heartbeat_seconds > 0 else 30.0

        def _heartbeat_worker() -> None:
            while not self._stop_event.wait(interval):
                self._emit(f"heartbeat ({self._get_stage()})", include_top=False)

        self._thread = threading.Thread(
            target=_heartbeat_worker,
            name="run-data-driven-mem-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.include_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()

    def _get_stage(self) -> str:
        with self._stage_lock:
            return self._stage

    def _set_stage(self, stage: str) -> None:
        with self._stage_lock:
            self._stage = stage

    def _emit(self, stage: str, *, include_top: bool) -> None:
        rss_mb = _current_rss_mb()
        peak_mb = _peak_rss_mb()
        print(
            f"[run_data_driven][mem] {stage}: "
            f"rss={rss_mb:.1f} MB peak={peak_mb:.1f} MB"
        )

        if not include_top:
            return
        if not self.include_tracemalloc or not tracemalloc.is_tracing():
            return

        stats = tracemalloc.take_snapshot().statistics("lineno")
        count = min(len(stats), self.top_n)
        print(f"[run_data_driven][mem] top {count} allocations ({stage}):")
        for idx, stat in enumerate(stats[:count], start=1):
            frame = stat.traceback[0]
            print(
                "[run_data_driven][mem]   "
                f"{idx:02d}. {frame.filename}:{frame.lineno} "
                f"{stat.size / (1024.0 * 1024.0):.1f} MB in {stat.count} blocks"
            )

    def mark(self, stage: str, *, include_top: bool = False) -> None:
        if not self.enabled:
            return
        self._set_stage(stage)
        self._emit(stage, include_top=include_top)


def _filter_to_flips(histo: Any) -> Any:
    if histo is None:
        return histo
    process_axis: Optional[Iterable[str]] = None
    try:
        process_axis = list(histo.axes["process"])  # type: ignore[index]
    except Exception:
        process_axis = None
    if not process_axis:
        return histo
    flips = [proc for proc in process_axis if "flips" in proc.lower()]
    if not flips:
        return histo
    to_remove = [proc for proc in process_axis if proc not in flips]
    if not to_remove:
        return histo
    if not hasattr(histo, "remove"):
        return histo
    return histo.remove("process", to_remove)


def _filter_to_allowed_processes(histo: Any, allowed_processes: Iterable[str]) -> Any:
    """Retain an exact generated/selected role set in a companion histogram."""

    if histo is None:
        return histo
    try:
        process_axis = [str(process) for process in histo.axes["process"]]
    except Exception:
        return histo
    allowed = set(allowed_processes)
    to_remove = [process for process in process_axis if process not in allowed]
    if not to_remove or not hasattr(histo, "remove"):
        return histo
    return histo.remove("process", to_remove)


def _maybe_emit_heartbeat(
    *,
    count: int,
    start_time: float,
    last_heartbeat: float,
    heartbeat_seconds: float,
    quiet: bool,
) -> Tuple[float, bool]:
    if quiet:
        return last_heartbeat, False
    now = time.monotonic()
    if heartbeat_seconds <= 0 or now - last_heartbeat >= heartbeat_seconds:
        elapsed = now - start_time
        print(f"[run_data_driven] Processed {count} histograms after {elapsed:.1f}s...")
        return now, True
    return last_heartbeat, False


def _envelope_single_histogram(key: str, histo: Any) -> Any:
    raise_unsupported_renormfact_envelope()


def _dd_channel_label(channel_name: Optional[str]) -> str:
    return "<all>" if channel_name is None else str(channel_name)


def _dd_is_zero(value: float) -> bool:
    return abs(value) < 1e-12


def _format_dd_total(value: float) -> str:
    if _dd_is_zero(value):
        value = 0.0
    return format(value, ".12g")


def _format_dd_labels(labels: Optional[Iterable[str]]) -> str:
    label_values = [str(label) for label in (labels or ())]
    return "<none>" if not label_values else ",".join(label_values)


def _format_dd_breakdown(entries: Optional[Iterable[Dict[str, Any]]]) -> str:
    entry_list = list(entries or ())
    if not entry_list:
        return "<none>"
    return ", ".join(
        f"{entry['process']}={_format_dd_total(entry['total'])}" for entry in entry_list
    )


def _dd_row_sort_key(row: Dict[str, Any]) -> Tuple[int, str, str]:
    return (
        _DD_REPORT_FAMILY_ORDER.get(row.get("family"), 99),
        str(row.get("region") or ""),
        str(row.get("output_process") or ""),
    )


def _dd_report_family_label(family_name: Optional[str]) -> str:
    if family_name == "sr":
        return "SR"
    if family_name == "nonprompt":
        return "nonprompt"
    if family_name == "flips":
        return "flips"
    return str(family_name or "unknown")


def _dd_report_absent_note(
    report: Dict[str, Any],
    *,
    region_name: str,
    channel_name: Optional[str],
) -> str:
    return (
        f"expected appl region {region_name} is missing for channel="
        f"{_dd_channel_label(channel_name)}; "
        f"available_regions={_format_dd_labels(report.get('regions'))}"
    )


def _dd_report_row_notes(row: Dict[str, Any]) -> List[str]:
    family_name = row.get("family")
    if family_name == "sr" and _dd_is_zero(row["retained_total"]):
        return ["nominal retained total is zero."]
    if family_name == "nonprompt" and _dd_is_zero(row["result"]):
        return ["nominal result is zero after data minus prompt subtraction."]
    if family_name == "flips" and _dd_is_zero(row["result"]):
        return ["nominal flips result is zero."]
    return []


def _iter_dd_report_channel_entries(
    report: Dict[str, Any],
) -> Iterable[Tuple[Optional[str], List[Dict[str, Any]]]]:
    rows = list(report.get("rows") or [])
    rows_by_channel: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for row in rows:
        rows_by_channel.setdefault(row.get("channel"), []).append(row)

    channels = sorted(
        report.get("channels") or rows_by_channel.keys(),
        key=lambda channel_name: _dd_channel_label(channel_name),
    )

    for channel_name in channels:
        channel_rows = rows_by_channel.get(channel_name, [])
        expected_regions = DataDrivenProducer.dd_report_expected_regions_for_channel(
            channel_name
        )
        covered_row_ids = set()
        entries: List[Dict[str, Any]] = []

        for family_name, region_name in expected_regions:
            matching_rows = [
                row
                for row in channel_rows
                if row.get("family") == family_name and row.get("region") == region_name
            ]
            if not matching_rows:
                entries.append(
                    {
                        "kind": "absent",
                        "family": family_name,
                        "region": region_name,
                    }
                )
                continue

            for row in sorted(matching_rows, key=_dd_row_sort_key):
                covered_row_ids.add(id(row))
                entries.append({"kind": "row", "row": row})

        extra_rows = [
            row for row in channel_rows if id(row) not in covered_row_ids
        ]
        for row in sorted(extra_rows, key=_dd_row_sort_key):
            entries.append({"kind": "row", "row": row})

        yield channel_name, entries


def _dd_report_stdout_lines(report: Optional[Dict[str, Any]]) -> List[str]:
    if not report:
        return []

    key = report.get("key", "<unknown>")
    if report.get("empty"):
        return [f"[dd-report] hist={key} status=empty"]

    lines: List[str] = []
    for channel_name, entries in _iter_dd_report_channel_entries(report):
        lines.append(f"[dd-report] hist={key} channel={_dd_channel_label(channel_name)}")
        for entry in entries:
            if entry["kind"] == "absent":
                family_name = entry["family"]
                region_name = entry["region"]
                lines.append(f"  {family_name} region={region_name} absent")
                continue

            row = entry["row"]
            family_name = row.get("family")
            if family_name == "sr":
                suffix = " zero_used_total" if _dd_is_zero(row["retained_total"]) else ""
                lines.append(
                    "  sr"
                    f" region={row['region']}"
                    f" retained_total={_format_dd_total(row['retained_total'])}{suffix}"
                )
            elif family_name == "nonprompt":
                suffix = " zero_used_total" if _dd_is_zero(row["result"]) else ""
                lines.append(
                    "  nonprompt"
                    f" region={row['region']}"
                    f" out={row['output_process']}"
                    f" data_used={_format_dd_total(row['data_used'])}"
                    f" prompt_sub_used={_format_dd_total(row['prompt_sub_used'])}"
                    f" result={_format_dd_total(row['result'])}{suffix}"
                )
            elif family_name == "flips":
                suffix = " zero_used_total" if _dd_is_zero(row["result"]) else ""
                lines.append(
                    "  flips"
                    f" region={row['region']}"
                    f" out={row['output_process']}"
                    f" data_used={_format_dd_total(row['data_used'])}"
                    f" result={_format_dd_total(row['result'])}{suffix}"
                )

    return lines


def _emit_dd_report(report: Optional[Dict[str, Any]]) -> None:
    for line in _dd_report_stdout_lines(report):
        print(line)


def _dd_report_markdown_lines(report: Optional[Dict[str, Any]]) -> List[str]:
    if not report:
        return []

    key = report.get("key", "<unknown>")
    lines = [f"## Histogram: `{key}`", ""]

    if report.get("empty"):
        lines.append("- Status: empty input histogram before appl integration.")
        lines.append("  - Note: input histogram has no populated bins before appl integration.")
        return lines

    for channel_name, entries in _iter_dd_report_channel_entries(report):
        lines.append(f"### Channel: `{_dd_channel_label(channel_name)}`")
        lines.append("")
        for entry in entries:
            if entry["kind"] == "absent":
                family_name = _dd_report_family_label(entry["family"])
                region_name = entry["region"]
                lines.append(f"- {family_name} region `{region_name}`: absent")
                lines.append(
                    "  - Note: "
                    + _dd_report_absent_note(
                        report,
                        region_name=region_name,
                        channel_name=channel_name,
                    )
                )
                continue

            row = entry["row"]
            family_name = row.get("family")
            if family_name == "sr":
                lines.append(
                    f"- SR region `{row['region']}` retained total: `{_format_dd_total(row['retained_total'])}`"
                )
            elif family_name == "nonprompt":
                suffix = " (zero used total)" if _dd_is_zero(row["result"]) else ""
                lines.append(
                    f"- nonprompt region `{row['region']}` output `{row['output_process']}`{suffix}"
                )
                lines.append(f"  - data used: `{_format_dd_total(row['data_used'])}`")
                lines.append(
                    f"  - prompt subtraction used: `{_format_dd_total(row['prompt_sub_used'])}`"
                )
                lines.append(f"  - result: `{_format_dd_total(row['result'])}`")
                lines.append(
                    f"  - data sources: `{_format_dd_breakdown(row.get('data_sources'))}`"
                )
                lines.append(
                    "  - prompt subtraction sources: "
                    f"`{_format_dd_breakdown(row.get('prompt_sub_sources'))}`"
                )
                prompt_sub_systematics = row.get("prompt_sub_systematics") or {}
                lines.append(
                    "  - prompt subtraction systematics: "
                    f"`kept={_format_dd_labels(prompt_sub_systematics.get('kept'))}`; "
                    f"`dropped={_format_dd_labels(prompt_sub_systematics.get('dropped'))}`"
                )
            elif family_name == "flips":
                suffix = " (zero used total)" if _dd_is_zero(row["result"]) else ""
                lines.append(
                    f"- flips region `{row['region']}` output `{row['output_process']}`{suffix}"
                )
                lines.append(f"  - data used: `{_format_dd_total(row['data_used'])}`")
                lines.append(f"  - result: `{_format_dd_total(row['result'])}`")
                lines.append(
                    f"  - data sources: `{_format_dd_breakdown(row.get('data_sources'))}`"
                )
                systematics = row.get("systematics") or {}
                lines.append(
                    "  - systematics: "
                    f"`kept={_format_dd_labels(systematics.get('kept'))}`; "
                    f"`dropped={_format_dd_labels(systematics.get('dropped'))}`"
                )
            for note in _dd_report_row_notes(row):
                lines.append(f"  - Note: {note}")
        lines.append("")

    if lines and lines[-1] == "":
        lines.pop()
    return lines


class _DDReportMarkdownWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        self._stream = None

    def open(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._stream = open(self.path, "w", encoding="utf-8")
        self._stream.write("# Data-driven report\n\n")

    def write_report(self, report: Optional[Dict[str, Any]]) -> None:
        if self._stream is None:
            raise RuntimeError("Markdown DD report writer is not open.")
        lines = _dd_report_markdown_lines(report)
        if not lines:
            return
        self._stream.write("\n".join(lines))
        self._stream.write("\n\n")

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def _finalize_histograms(
    input_pkl: str,
    output_pkl: str,
    *,
    only_flips: bool,
    nominal_only_reference: bool = False,
    apply_envelope: bool,
    dd_report_stdout: bool = False,
    dd_report_md: Optional[str] = None,
    iterator_mode: bool = True,
    heartbeat_seconds: float = 30.0,
    quiet: bool = False,
    mem_report: bool = False,
    mem_tracemalloc: bool = False,
    mem_top_n: int = 20,
    serialization_path: Optional[str] = None,
    input_sidecar: Optional[Dict[str, Any]] = None,
    input_artifact_validation: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if apply_envelope:
        raise_unsupported_renormfact_envelope()
    serialization_path = serialization_path or output_pkl
    collect_dd_report = dd_report_stdout or bool(dd_report_md)
    memory_reporter = _MemoryReporter(
        enabled=(mem_report or mem_tracemalloc),
        include_tracemalloc=mem_tracemalloc,
        heartbeat_seconds=heartbeat_seconds,
        top_n=mem_top_n,
    )
    markdown_writer = (
        _DDReportMarkdownWriter(dd_report_md)
        if dd_report_md
        else None
    )
    if markdown_writer is not None:
        markdown_writer.open()
    memory_reporter.start()

    try:
        memory_reporter.mark("start")
        memory_reporter.mark("before DataDrivenProducer(...)")
        artifact_kind = (
            FLIPS_OUTPUT_ARTIFACT_KIND
            if only_flips
            else NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND
            if nominal_only_reference
            else NONPROMPT_OUTPUT_ARTIFACT_KIND
        )
        ddp_kwargs: Dict[str, Any] = {"iterator_mode": iterator_mode}
        if input_sidecar is not None:
            ddp_kwargs["artifact_kind"] = artifact_kind
            ddp_kwargs["input_artifact_validation"] = input_artifact_validation
        if collect_dd_report:
            ddp_kwargs["dd_report"] = True
        ddp = DataDrivenProducer(input_pkl, output_pkl, **ddp_kwargs)
        retained_selected_eft_by_family: Dict[str, List[str]] = {}
        certified_flips_outputs = None
        if input_sidecar is not None:
            certified_flips_outputs = set(
                generated_output_processes_from_contract(
                    input_sidecar["resolved_data_driven_contract"],
                    "flips",
                )
            )
        if only_flips and input_sidecar is not None:
            policy = resolved_policy_from_provenance(
                input_sidecar["sumw2_storage_provenance"]
            )
            for family, manifest in input_sidecar["sumw2_content_manifest"][
                "families"
            ].items():
                retained_selected_eft_by_family[family] = sorted(
                    set(manifest["eft_nominal_processes"])
                    & set(policy.selected_processes(family))
                )
        memory_reporter.mark("after DataDrivenProducer(...)", include_top=mem_tracemalloc)
        os.makedirs(os.path.dirname(output_pkl) or ".", exist_ok=True)

        start_time = time.monotonic()
        last_heartbeat = start_time
        processed = 0

        if iterator_mode:
            def _iter_output_items():
                nonlocal processed, last_heartbeat
                for key, histo in ddp.iter_data_driven_histograms():
                    processed += 1
                    last_heartbeat, emitted_heartbeat = _maybe_emit_heartbeat(
                        count=processed,
                        start_time=start_time,
                        last_heartbeat=last_heartbeat,
                        heartbeat_seconds=heartbeat_seconds,
                        quiet=quiet,
                    )

                    report = ddp.get_dd_report(key) if collect_dd_report else None
                    if dd_report_stdout:
                        _emit_dd_report(report)
                    if markdown_writer is not None:
                        markdown_writer.write_report(report)
                    if only_flips and key.endswith("_sumw2"):
                        family = key[: -len("_sumw2")]
                        generated_flips = set(certified_flips_outputs or ())
                        working_histo = _filter_to_allowed_processes(
                            histo,
                            generated_flips
                            | set(retained_selected_eft_by_family.get(family, ())),
                        )
                    elif only_flips:
                        working_histo = (
                            histo
                            if key.endswith(EFT_NOMINAL_SUFFIX)
                            else _filter_to_allowed_processes(
                                histo,
                                certified_flips_outputs,
                            )
                        ) if certified_flips_outputs is not None else _filter_to_flips(histo)
                    else:
                        working_histo = histo
                    if emitted_heartbeat:
                        memory_reporter.mark(f"processed {processed} histograms")

                    yield key, working_histo
                    del working_histo
                    del histo
                    gc.collect()
                    _trim_allocator()

            memory_reporter.mark("before dump_dict_streaming()", include_top=mem_tracemalloc)
            utils.dump_dict_streaming(
                serialization_path,
                _iter_output_items(),
                protocol=_STREAMING_PICKLE_PROTOCOL,
                clear_memo_interval=_STREAMING_MEMO_CLEAR_INTERVAL,
            )
            memory_reporter.mark("after dump_dict_streaming()")
        else:
            histograms = ddp.getDataDrivenHistogram()
            memory_reporter.mark("after getDataDrivenHistogram()")

            filtered: Optional[Dict[str, Any]] = {} if only_flips else None
            for key, histo in histograms.items():
                processed += 1
                last_heartbeat, emitted_heartbeat = _maybe_emit_heartbeat(
                    count=processed,
                    start_time=start_time,
                    last_heartbeat=last_heartbeat,
                    heartbeat_seconds=heartbeat_seconds,
                    quiet=quiet,
                )

                report = ddp.get_dd_report(key) if collect_dd_report else None
                if dd_report_stdout:
                    _emit_dd_report(report)
                if markdown_writer is not None:
                    markdown_writer.write_report(report)
                if only_flips:
                    assert filtered is not None
                    if key.endswith("_sumw2"):
                        family = key[: -len("_sumw2")]
                        generated_flips = set(certified_flips_outputs or ())
                        filtered[key] = _filter_to_allowed_processes(
                            histo,
                            generated_flips
                            | set(retained_selected_eft_by_family.get(family, ())),
                        )
                    elif certified_flips_outputs is not None:
                        filtered[key] = (
                            histo
                            if key.endswith(EFT_NOMINAL_SUFFIX)
                            else _filter_to_allowed_processes(
                                histo,
                                certified_flips_outputs,
                            )
                        )
                    else:
                        filtered[key] = _filter_to_flips(histo)

                if emitted_heartbeat:
                    memory_reporter.mark(f"processed {processed} histograms")

            if only_flips:
                assert filtered is not None
                memory_reporter.mark("before only-flips replacement")
                histograms = filtered
                del filtered
                memory_reporter.mark("after only-flips replacement")

            memory_reporter.mark("before dump_to_pkl()", include_top=mem_tracemalloc)
            utils.dump_to_pkl(serialization_path, histograms)
            memory_reporter.mark("after dump_to_pkl()")

        if not quiet and processed:
            elapsed = time.monotonic() - start_time
            print(f"[run_data_driven] Finalized {processed} histograms in {elapsed:.1f}s.")

        transformation_context = (
            ddp.get_transformation_context(artifact_kind)
            if input_sidecar is not None
            else None
        )
        del ddp
    finally:
        if markdown_writer is not None:
            markdown_writer.close()
        memory_reporter.stop()
    return transformation_context


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    if args.only_flips and args.nominal_only_reference:
        parser.error("--only-flips and --nominal-only-reference are mutually exclusive.")
    if bool(args.legacy_campaign_state) != bool(args.legacy_campaign_block):
        parser.error(
            "--legacy-campaign-state and --legacy-campaign-block must be supplied together."
        )
    if args.apply_renormfact_envelope:
        raise_unsupported_renormfact_envelope()
    dd_report_stdout = args.dd_report
    dd_report_md = args.dd_report_md

    input_pkl = os.path.normpath(args.input_pkl)
    _validate_input_path(input_pkl)

    output_pkl = os.path.normpath(args.output_pkl) if args.output_pkl else None
    if not output_pkl:
        output_pkl = _default_output_path(
            input_pkl,
            nominal_only_reference=args.nominal_only_reference,
        )

    try:
        serialized_sidecar = read_histogram_sidecar(input_pkl)
    except FileNotFoundError:
        serialized_sidecar = None
    serialized_applicability = (
        None
        if serialized_sidecar is None
        else serialized_sidecar.get("histogram_applicability")
    )
    legacy_context_requested = args.legacy_campaign_state is not None
    if serialized_applicability is not None and legacy_context_requested:
        raise RuntimeError(
            "A self-describing source artifact must not receive legacy campaign context."
        )
    if (
        serialized_sidecar is not None
        and serialized_applicability is None
        and not legacy_context_requested
    ):
        raise RuntimeError(
            "A declaration-free schema-v2 source artifact requires qualified legacy "
            "campaign context."
        )
    legacy_context_summary = None
    if legacy_context_requested:
        if serialized_sidecar is None:
            raise RuntimeError(
                "Legacy campaign context applies only to a declaration-free schema-v2 artifact."
            )
        legacy_applicability, legacy_context_summary = (
            _resolve_legacy_histogram_applicability(
                input_pkl=input_pkl,
                input_sidecar=serialized_sidecar,
                campaign_state_path=args.legacy_campaign_state,
                campaign_block_id=args.legacy_campaign_block,
            )
        )
        input_validation = _validate_legacy_histogram_artifact(
            input_pkl,
            legacy_applicability,
            legacy_context_summary,
        )
        print(
            "[run_data_driven] qualified legacy context: "
            + json.dumps(legacy_context_summary, sort_keys=True)
        )
    else:
        input_validation = validate_histogram_artifact(input_pkl)
    input_sidecar = input_validation["metadata"]
    finalize_kwargs = {
        "only_flips": args.only_flips,
        "nominal_only_reference": args.nominal_only_reference,
        "apply_envelope": args.apply_renormfact_envelope,
        "dd_report_stdout": dd_report_stdout,
        "dd_report_md": dd_report_md,
        "iterator_mode": not args.legacy_dict_mode,
        "heartbeat_seconds": args.heartbeat_seconds,
        "quiet": args.quiet,
        "mem_report": args.mem_report,
        "mem_tracemalloc": args.mem_tracemalloc,
        "mem_top_n": args.mem_top_n,
    }
    if input_sidecar is None:
        _finalize_histograms(input_pkl, output_pkl, **finalize_kwargs)
    else:
        if input_sidecar["artifact"]["artifact_kind"] != "processor_output":
            raise RuntimeError(
                "run_data_driven requires a processor_output input artifact; got "
                f"{input_sidecar['artifact']['artifact_kind']!r} for '{input_pkl}'."
            )
        resolution = validate_requested_product_input(
            input_sidecar,
            artifact_kind=(
                FLIPS_OUTPUT_ARTIFACT_KIND
                if args.only_flips
                else NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND
                if args.nominal_only_reference
                else NONPROMPT_OUTPUT_ARTIFACT_KIND
            ),
        )
        effective_input_sidecar = resolution["effective_sidecar"]

        def _write_payload(staged_path: str) -> Dict[str, Any]:
            transformation_context = _finalize_histograms(
                input_pkl,
                output_pkl,
                serialization_path=staged_path,
                input_sidecar=effective_input_sidecar,
                input_artifact_validation={
                    **input_validation,
                    "metadata": effective_input_sidecar,
                },
                **finalize_kwargs,
            )
            assert transformation_context is not None
            return transformation_context

        write_histogram_artifact(
            output_pkl,
            payload_writer=_write_payload,
            artifact_kind=(
                FLIPS_OUTPUT_ARTIFACT_KIND
                if args.only_flips
                else NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND
                if args.nominal_only_reference
                else NONPROMPT_OUTPUT_ARTIFACT_KIND
            ),
            sumw2_storage_provenance=effective_input_sidecar["sumw2_storage_provenance"],
            lineage_inputs=[lineage_input_from_sidecar(input_sidecar)],
            input_sidecar=effective_input_sidecar,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
