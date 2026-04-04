#!/usr/bin/env python3
"""Standalone helper to build data driven histograms from saved metadata.

Quickstart examples:
  - Metadata sidecar: python run_data_driven.py --metadata-json histos/plotsTopEFT_np.pkl.gz.metadata.json
  - Direct pickle paths: python run_data_driven.py --input-pkl histos/plotsTopEFT.pkl.gz \
      --output-pkl histos/plotsTopEFT_np.pkl.gz --apply-renormfact-envelope
  - Legacy/materialized fallback: add --legacy-dict-mode to restore the
      original fully materialized dict workflow.

By default the helper uses the streaming iterator path, writing output with
``dump_dict_streaming(..., protocol=3, clear_memo_interval=1)`` to cap RSS.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import os
import resource
import sys
import threading
import time
import tracemalloc
from typing import Any, Dict, Iterable, List, Optional, Tuple

import topcoffea.modules.utils as utils

from topeft.modules.dataDrivenEstimation import DataDrivenProducer
from topeft.modules.deferred_np_metadata import load_deferred_np_metadata
from topeft.modules.get_renormfact_envelope import (
    apply_renormfact_envelope_to_histogram,
    get_renormfact_envelope,
)

_STREAMING_PICKLE_PROTOCOL = 3
_STREAMING_MEMO_CLEAR_INTERVAL = 1
_LIBC = None
_DD_REPORT_FAMILY_ORDER = {"sr": 0, "nonprompt": 1, "flips": 2}


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize deferred nonprompt/flips histograms using the metadata emitted by run_analysis.py.\n\n"
            "Quickstart:\n"
            "  - Metadata sidecar: python run_data_driven.py --metadata-json histos/plotsTopEFT_np.pkl.gz.metadata.json\n"
            "  - Direct pickle paths: python run_data_driven.py --input-pkl histos/plotsTopEFT.pkl.gz\\\n"
            "      --output-pkl histos/plotsTopEFT_np.pkl.gz --apply-renormfact-envelope\n"
            "Default mode is streaming iterator mode (lower peak RSS). "
            "Pass --legacy-dict-mode to restore the original materialized-dict behavior.\n"
            f"Streaming serialization defaults are hardcoded to protocol={_STREAMING_PICKLE_PROTOCOL} "
            f"and clear_memo_interval={_STREAMING_MEMO_CLEAR_INTERVAL}."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metadata-json",
        help=(
            "Path to the metadata file created by run_analysis.py when using "
            "--np-postprocess=defer."
        ),
    )
    parser.add_argument(
        "--input-pkl",
        help="Path to the histogram pickle emitted by run_analysis.py (pre data-driven step).",
    )
    parser.add_argument(
        "--output-pkl",
        help="Destination for the histogram pickle with data-driven contributions applied.",
    )
    parser.add_argument(
        "--apply-renormfact-envelope",
        action="store_true",
        help=(
            "Also run the renorm/fact envelope step on the output histogram. "
            "Metadata-driven runs also honor the contract-recorded envelope setting automatically."
        ),
    )
    parser.add_argument(
        "--only-flips",
        action="store_true",
        help="Drop nonprompt processes so only flips contributions remain in the output histograms.",
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


def _load_metadata(metadata_path: str) -> Dict[str, Any]:
    return load_deferred_np_metadata(metadata_path)


def _default_output_path(input_path: str) -> str:
    if input_path.endswith(".pkl.gz"):
        base = input_path[:-7]
    elif input_path.endswith(".pkl"):
        base = input_path[:-4]
    else:
        base = input_path
    return f"{base}_np.pkl.gz"


def _resolve_path(
    arg_value: Optional[str],
    metadata_value: Optional[str],
    *,
    metadata_dir: Optional[str] = None,
) -> Optional[str]:
    if arg_value:
        return arg_value
    if not metadata_value:
        return None
    if metadata_dir and not os.path.isabs(metadata_value):
        return os.path.normpath(os.path.join(metadata_dir, metadata_value))
    return os.path.normpath(metadata_value)


def _validate_input_path(input_path: str) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Histogram pickle not found: {input_path}")


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
    return apply_renormfact_envelope_to_histogram(
        histo, verbose=False, hist_name=key
    )


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
    apply_envelope: bool,
    dd_report_stdout: bool = False,
    dd_report_md: Optional[str] = None,
    iterator_mode: bool = True,
    heartbeat_seconds: float = 30.0,
    quiet: bool = False,
    mem_report: bool = False,
    mem_tracemalloc: bool = False,
    mem_top_n: int = 20,
) -> None:
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
        ddp_kwargs: Dict[str, Any] = {"iterator_mode": iterator_mode}
        if collect_dd_report:
            ddp_kwargs["dd_report"] = True
        ddp = DataDrivenProducer(input_pkl, output_pkl, **ddp_kwargs)
        memory_reporter.mark("after DataDrivenProducer(...)", include_top=mem_tracemalloc)
        os.makedirs(os.path.dirname(output_pkl) or ".", exist_ok=True)

        start_time = time.monotonic()
        last_heartbeat = start_time
        processed = 0

        if iterator_mode:
            if apply_envelope:
                memory_reporter.mark(
                    "iterator mode: envelope is applied per histogram",
                    include_top=mem_tracemalloc,
                )

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
                    working_histo = _filter_to_flips(histo) if only_flips else histo
                    if apply_envelope:
                        working_histo = _envelope_single_histogram(key, working_histo)

                    if emitted_heartbeat:
                        memory_reporter.mark(f"processed {processed} histograms")

                    yield key, working_histo
                    del working_histo
                    del histo
                    gc.collect()
                    _trim_allocator()

            memory_reporter.mark("before dump_dict_streaming()", include_top=mem_tracemalloc)
            utils.dump_dict_streaming(
                output_pkl,
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
                    filtered[key] = _filter_to_flips(histo)

                if emitted_heartbeat:
                    memory_reporter.mark(f"processed {processed} histograms")

            if only_flips:
                assert filtered is not None
                memory_reporter.mark("before only-flips replacement")
                histograms = filtered
                del filtered
                memory_reporter.mark("after only-flips replacement")

            if apply_envelope:
                memory_reporter.mark("before get_renormfact_envelope()", include_top=mem_tracemalloc)
                histograms = get_renormfact_envelope(histograms, verbose=False)
                memory_reporter.mark("after get_renormfact_envelope()", include_top=mem_tracemalloc)

            memory_reporter.mark("before dump_to_pkl()", include_top=mem_tracemalloc)
            utils.dump_to_pkl(output_pkl, histograms)
            memory_reporter.mark("after dump_to_pkl()")

        if not quiet and processed:
            elapsed = time.monotonic() - start_time
            print(f"[run_data_driven] Finalized {processed} histograms in {elapsed:.1f}s.")

        del ddp
    finally:
        if markdown_writer is not None:
            markdown_writer.close()
        memory_reporter.stop()


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    dd_report_stdout = args.dd_report
    dd_report_md = args.dd_report_md

    metadata: Dict[str, Any] = {}
    metadata_dir: Optional[str] = None
    if args.metadata_json:
        metadata = _load_metadata(args.metadata_json)
        metadata_dir = os.path.dirname(os.path.abspath(args.metadata_json))

    input_pkl = _resolve_path(
        args.input_pkl, metadata.get("input_histogram"), metadata_dir=metadata_dir
    )
    if not input_pkl:
        raise ValueError("Input histogram path must be provided via --input-pkl or the metadata file.")
    _validate_input_path(input_pkl)

    output_pkl = _resolve_path(
        args.output_pkl, metadata.get("output_histogram"), metadata_dir=metadata_dir
    )
    if not output_pkl:
        output_pkl = _default_output_path(input_pkl)

    apply_envelope = args.apply_renormfact_envelope or metadata.get(
        "apply_renormfact_envelope", False
    )

    _finalize_histograms(
        input_pkl,
        output_pkl,
        only_flips=args.only_flips,
        apply_envelope=apply_envelope,
        dd_report_stdout=dd_report_stdout,
        dd_report_md=dd_report_md,
        iterator_mode=not args.legacy_dict_mode,
        heartbeat_seconds=args.heartbeat_seconds,
        quiet=args.quiet,
        mem_report=args.mem_report,
        mem_tracemalloc=args.mem_tracemalloc,
        mem_top_n=args.mem_top_n,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
