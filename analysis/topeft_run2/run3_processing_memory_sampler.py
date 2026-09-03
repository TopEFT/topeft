#!/usr/bin/env python3
"""Sample aggregate memory for one process group without controlling it."""

import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-group-id", type=int, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--attempt-kind", required=True)
    parser.add_argument("--samples-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--startup-grace-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.process_group_id <= 0:
        parser.error("--process-group-id must be positive")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    if args.startup_grace_seconds < 0:
        parser.error("--startup-grace-seconds must be nonnegative")
    return args


def process_group_id_for_pid(pid):
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        closing_paren = stat_text.rfind(")")
        if closing_paren < 0:
            return None
        fields_after_comm = stat_text[closing_paren + 2 :].split()
        return int(fields_after_comm[2])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        return None


def process_group_members(target_process_group_id):
    members = []
    try:
        proc_entries = os.scandir("/proc")
    except OSError:
        return members
    with proc_entries:
        for entry in proc_entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if process_group_id_for_pid(pid) == target_process_group_id:
                members.append(pid)
    return sorted(members)


def status_value_kb(pid, field_name):
    try:
        with Path(f"/proc/{pid}/status").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(f"{field_name}:"):
                    return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        return None
    return None


def pss_kb(pid):
    try:
        with Path(f"/proc/{pid}/smaps_rollup").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        return None
    return None


def system_mem_available_kb():
    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return None
    return None


def utc_timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def write_summary(summary_path, values):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("metric", "value"))
        for key, value in values.items():
            writer.writerow((key, value))


def main():
    args = parse_args()
    args.samples_path.parent.mkdir(parents=True, exist_ok=True)

    peak_group_rss_kb = 0
    peak_group_pss_kb = None
    peak_process_count = 0
    min_system_mem_available_kb = None
    memory_sample_count = 0
    pss_rank = 0
    seen_group = False
    startup_deadline = time.monotonic() + args.startup_grace_seconds
    next_sample_time = time.monotonic()

    with args.samples_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        while True:
            members = process_group_members(args.process_group_id)
            if not members:
                if seen_group or time.monotonic() >= startup_deadline:
                    break
                time.sleep(min(args.interval_seconds, 0.1))
                continue

            seen_group = True
            rss_values = [status_value_kb(pid, "VmRSS") for pid in members]
            aggregate_rss_kb = sum(value for value in rss_values if value is not None)

            pss_values = [pss_kb(pid) for pid in members]
            readable_pss_values = [value for value in pss_values if value is not None]
            if len(readable_pss_values) == len(members):
                pss_status = "available"
                pss_rank = max(pss_rank, 2)
            elif readable_pss_values:
                pss_status = "partial"
                pss_rank = max(pss_rank, 1)
            else:
                pss_status = "unavailable"
            aggregate_pss_kb = sum(readable_pss_values) if readable_pss_values else ""
            mem_available_kb = system_mem_available_kb()

            writer.writerow(
                (
                    args.task_id,
                    args.attempt_kind,
                    utc_timestamp(),
                    args.process_group_id,
                    len(members),
                    aggregate_rss_kb,
                    aggregate_pss_kb,
                    pss_status,
                    "" if mem_available_kb is None else mem_available_kb,
                )
            )
            handle.flush()

            memory_sample_count += 1
            peak_group_rss_kb = max(peak_group_rss_kb, aggregate_rss_kb)
            peak_process_count = max(peak_process_count, len(members))
            if readable_pss_values:
                measured_pss_kb = sum(readable_pss_values)
                if peak_group_pss_kb is None:
                    peak_group_pss_kb = measured_pss_kb
                else:
                    peak_group_pss_kb = max(peak_group_pss_kb, measured_pss_kb)
            if mem_available_kb is not None:
                if min_system_mem_available_kb is None:
                    min_system_mem_available_kb = mem_available_kb
                else:
                    min_system_mem_available_kb = min(min_system_mem_available_kb, mem_available_kb)

            next_sample_time += args.interval_seconds
            time.sleep(max(0.0, next_sample_time - time.monotonic()))

    summary_pss_status = {0: "unavailable", 1: "partial", 2: "available"}[pss_rank]
    write_summary(
        args.summary_path,
        {
            "task_id": args.task_id,
            "attempt_kind": args.attempt_kind,
            "process_group_id": args.process_group_id,
            "peak_group_rss_kb": peak_group_rss_kb,
            "peak_group_pss_kb": "" if peak_group_pss_kb is None else peak_group_pss_kb,
            "pss_status": summary_pss_status,
            "peak_process_count": peak_process_count,
            "min_system_mem_available_kb": "" if min_system_mem_available_kb is None else min_system_mem_available_kb,
            "memory_sample_count": memory_sample_count,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
