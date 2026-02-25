#!/usr/bin/env python3
"""
Parallel check of skim ROOT files referenced by NDSkim_202*.cfg JSONs for required branches.

Features:
- Scans cfg files in ../../input_samples/cfgs starting with "NDSkim_202"
- For each JSON referenced in each cfg:
  - Opens each ROOT file listed in JSON["files"], using local path /cms/cephfs/data + "/store/..."
  - Checks tree "Events" (or JSON["treeName"]) contains BOTH:
      Electron_mvaTTHrun3, Muon_mvaTTHrun3
  - Prints a readable per-JSON summary (counts + optional missing-file list)
- Uses a process pool for parallel file checks (default: min(8, cpu_count))
- Prints an end-of-run summary listing JSONs with missing-branches failures (separate from open/missing-tree issues)

Requires: uproot
  pip install uproot
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import uproot


REQUIRED_BRANCHES = ("Electron_mvaTTHrun3", "Muon_mvaTTHrun3")
DEFAULT_TREE = "Events"
LOCAL_PREFIX = "/cms/cephfs/data"


# ----------------------------
# Helpers: cfg/json parsing
# ----------------------------

def find_cfgs(cfg_dir: Path, prefix: str = "NDSkim_202") -> List[Path]:
    return sorted([p for p in cfg_dir.glob(f"{prefix}*.cfg") if p.is_file()])


def parse_cfg(cfg_path: Path) -> Tuple[Optional[str], List[str]]:
    """
    Returns: (prefix_line_or_none, json_paths)

    Note: cfgs often start with a redirector like 'root://.../'.
    We ignore it because we open files locally via /cms/cephfs/data.
    """
    base_prefix = None
    json_paths: List[str] = []

    with cfg_path.open() as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue

            if base_prefix is None and (raw.startswith("root://") or raw.startswith("file://")):
                base_prefix = raw
                continue

            json_paths.append(raw)

    return base_prefix, json_paths


def load_json(json_path: Path) -> Dict:
    with json_path.open() as f:
        return json.load(f)


def localize_store_path(path_in_json: str) -> str:
    """
    Convert '/store/...' into '/cms/cephfs/data/store/...'
    If it's already under /cms/cephfs/data, keep it.
    Otherwise keep as-is.
    """
    if path_in_json.startswith(LOCAL_PREFIX + "/"):
        return path_in_json
    if path_in_json.startswith("/store/"):
        return LOCAL_PREFIX + path_in_json
    return path_in_json


# ----------------------------
# Worker: root inspection
# ----------------------------

def _root_has_required_branches(root_path: str, tree_name: str) -> Tuple[bool, Optional[str]]:
    """
    Returns (ok, reason).
    ok=True iff:
      - file opens
      - tree exists
      - both required branches exist

    reason is one of:
      - missing_tree:<tree_name>
      - missing_branches:<comma-separated-branch-names>
      - open_error:<ExceptionType>:<message>
    """
    try:
        with uproot.open(root_path) as f:
            if tree_name not in f:
                return False, f"missing_tree:{tree_name}"
            tree = f[tree_name]
            keys = tree.keys()
            missing = [b for b in REQUIRED_BRANCHES if b not in keys]
            if missing:
                return False, "missing_branches:" + ",".join(missing)
            return True, None
    except Exception as e:
        return False, f"open_error:{type(e).__name__}:{e}"


# ----------------------------
# Reporting structures
# ----------------------------

@dataclass
class FileResult:
    path: str
    ok: bool
    reason: Optional[str] = None


@dataclass
class JsonReport:
    json_path: str
    tree_name: str
    n_total: int
    n_ok: int
    n_fail: int
    n_missing_branches: int
    n_missing_tree: int
    n_open_error: int
    failures: List[FileResult]


def _fmt_ratio(ok: int, tot: int) -> str:
    if tot <= 0:
        return "0/0"
    return f"{ok}/{tot}"


def _fmt_pct(ok: int, tot: int) -> str:
    if tot <= 0:
        return "0.0%"
    return f"{(100.0 * ok / tot):.1f}%"


def _group_fail_reasons(failures: List[FileResult]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for fr in failures:
        r = fr.reason or "unknown"
        counts[r] = counts.get(r, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _classify_failure_counts(failures: List[FileResult]) -> Tuple[int, int, int]:
    """
    Returns (n_missing_branches, n_missing_tree, n_open_error)
    """
    n_mb = 0
    n_mt = 0
    n_oe = 0
    for fr in failures:
        r = fr.reason or ""
        if r.startswith("missing_branches:"):
            n_mb += 1
        elif r.startswith("missing_tree:"):
            n_mt += 1
        elif r.startswith("open_error:"):
            n_oe += 1
        else:
            # unknown category -> count it as open_error-ish to avoid hiding it
            n_oe += 1
    return n_mb, n_mt, n_oe


# ----------------------------
# Main JSON inspection (parallel)
# ----------------------------

def inspect_json_parallel(
    json_path_str: str,
    workers: int,
    print_missing: bool,
    max_missing_print: int,
    show_reason_summary: bool,
    show_progress: bool,
) -> JsonReport:
    json_path = Path(json_path_str)
    payload = load_json(json_path)

    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError(f"{json_path}: missing or invalid 'files' list")

    tree_name = payload.get("treeName") or DEFAULT_TREE

    localized_files = [localize_store_path(str(fp)) for fp in files]
    n_total = len(localized_files)

    failures: List[FileResult] = []
    ok_count = 0

    # Submit checks
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut_map = {
            ex.submit(_root_has_required_branches, fp, tree_name): fp
            for fp in localized_files
        }

        # Simple progress tick (readable, not spammy)
        done = 0
        step = max(1, n_total // 10)  # 10 ticks per JSON
        for fut in as_completed(fut_map):
            fp = fut_map[fut]
            ok, reason = fut.result()
            done += 1

            if ok:
                ok_count += 1
            else:
                failures.append(FileResult(path=fp, ok=False, reason=reason))

            if show_progress and n_total >= 50 and (done % step == 0 or done == n_total):
                print(f"      progress: {done}/{n_total}", flush=True)

    n_fail = len(failures)
    n_missing_branches, n_missing_tree, n_open_error = _classify_failure_counts(failures)

    # Optional missing-file list
    if print_missing and failures:
        to_print = failures if max_missing_print < 0 else failures[:max_missing_print]
        print("      failing files (first {}):".format("ALL" if max_missing_print < 0 else len(to_print)))
        for fr in to_print:
            print(f"        - {fr.path}  ({fr.reason})")
        if max_missing_print >= 0 and len(failures) > max_missing_print:
            print(f"        ... plus {len(failures) - max_missing_print} more")

    # Optional reason summary
    if show_reason_summary and failures:
        by_reason = _group_fail_reasons(failures)
        print("      failure reasons:")
        for reason, cnt in by_reason.items():
            print(f"        - {cnt:5d}  {reason}")

    return JsonReport(
        json_path=str(json_path),
        tree_name=tree_name,
        n_total=n_total,
        n_ok=ok_count,
        n_fail=n_fail,
        n_missing_branches=n_missing_branches,
        n_missing_tree=n_missing_tree,
        n_open_error=n_open_error,
        failures=failures,
    )


# ----------------------------
# CLI entry
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg-dir",
        default="../../input_samples/cfgs",
        help="Directory containing cfg files (default: ../../input_samples/cfgs)",
    )
    ap.add_argument(
        "--cfg-prefix",
        default="NDSkim_202",
        help='Only consider cfg files whose filename starts with this prefix (default: "NDSkim_202")',
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=min(8, (os.cpu_count() or 2)),
        help="Process workers for ROOT file checks (default: min(8, cpu_count))",
    )
    ap.add_argument(
        "--print-missing",
        action="store_true",
        help="Print failing ROOT files (missing branches/tree or that could not be opened)",
    )
    ap.add_argument(
        "--max-missing-print",
        type=int,
        default=-1,
        help="Max failing files to print per JSON when --print-missing is used. Use -1 for unlimited. (default: 20)",
    )
    ap.add_argument(
        "--reason-summary",
        action="store_true",
        help="Print a compact summary of failure reasons per JSON",
    )
    ap.add_argument(
        "--only-failing-jsons",
        action="store_true",
        help="Only print per-JSON blocks for JSONs that have at least one failing ROOT file",
    )
    ap.add_argument(
        "--end-summary-max",
        type=int,
        default=-1,
        help="Max JSONs to list in the end summary (sorted by missing_branches desc). Use -1 for unlimited. (default: -1)",
    )
    args = ap.parse_args()

    cfg_dir = Path(args.cfg_dir)
    cfgs = find_cfgs(cfg_dir, prefix=args.cfg_prefix)

    if not cfgs:
        print(f"[ERROR] No cfg files found in {cfg_dir} starting with '{args.cfg_prefix}'")
        raise SystemExit(2)

    print("=== Branch check ===")
    print(f"Required branches: {REQUIRED_BRANCHES[0]}, {REQUIRED_BRANCHES[1]}")
    print(f"Local prefix for /store paths: {LOCAL_PREFIX}")
    print(f"CFG dir: {cfg_dir}")
    print(f"CFG prefix: {args.cfg_prefix}")
    print(f"Workers: {args.workers}")
    print()

    grand_total = 0
    grand_ok = 0
    grand_fail = 0
    n_json = 0
    n_json_with_fail = 0

    # For end summary: only "missing_branches" category
    jsons_with_missing_branches: List[JsonReport] = []

    for cfg in cfgs:
        base_prefix, json_paths = parse_cfg(cfg)

        # Gather JSON results for this cfg
        cfg_reports: List[JsonReport] = []

        for jp in json_paths:
            n_json += 1

            # We'll print the per-JSON header only if not suppressing, or if error happens.
            header_printed = False

            def _print_header():
                nonlocal header_printed
                if header_printed:
                    return
                print(f"CFG: {cfg.name}")
                if base_prefix:
                    print(f"  (prefix line ignored): {base_prefix}")
                print(f"  JSON: {jp}")
                header_printed = True

            if not args.only_failing_jsons:
                _print_header()

            try:
                rep = inspect_json_parallel(
                    jp,
                    workers=args.workers,
                    print_missing=args.print_missing,
                    max_missing_print=args.max_missing_print,
                    show_reason_summary=args.reason_summary,
                    show_progress=(not args.only_failing_jsons),
                )
                cfg_reports.append(rep)

                if rep.n_fail > 0:
                    n_json_with_fail += 1
                if rep.n_missing_branches > 0:
                    jsons_with_missing_branches.append(rep)

                if args.only_failing_jsons and rep.n_fail == 0:
                    # suppress output entirely for passing JSONs
                    continue

                if args.only_failing_jsons:
                    _print_header()

                print(
                    f"      treeName={rep.tree_name}  "
                    f"OK {_fmt_ratio(rep.n_ok, rep.n_total)} ({_fmt_pct(rep.n_ok, rep.n_total)})  "
                    f"FAIL {_fmt_ratio(rep.n_fail, rep.n_total)}  "
                    f"[missing_branches={rep.n_missing_branches}, missing_tree={rep.n_missing_tree}, open_error={rep.n_open_error}]"
                )
            except Exception as e:
                _print_header()
                print(f"      ERROR inspecting JSON ({type(e).__name__}): {e}")

            print()

        # Update grand totals
        for rep in cfg_reports:
            grand_total += rep.n_total
            grand_ok += rep.n_ok
            grand_fail += rep.n_fail

    print("=== GRAND TOTALS ===")
    print(f"JSONs inspected: {n_json}")
    print(f"JSONs with >=1 failing ROOT file: {n_json_with_fail}")
    print(f"OK:   {grand_ok}/{grand_total} ({_fmt_pct(grand_ok, grand_total)})")
    print(f"FAIL: {grand_fail}/{grand_total}")
    print()

    # End-of-run summary: JSONs with missing branches (sorted by count desc)
    print("=== JSONs with missing branches (>=1) ===")
    if jsons_with_missing_branches:
        jsons_with_missing_branches.sort(key=lambda r: (-r.n_missing_branches, r.json_path))
        to_show = jsons_with_missing_branches if args.end_summary_max < 0 else jsons_with_missing_branches[: args.end_summary_max]
        for rep in to_show:
            print(
                f"  - {rep.json_path}  "
                f"(missing_branches={rep.n_missing_branches} / total={rep.n_total}, fail_total={rep.n_fail})"
            )
        if args.end_summary_max >= 0 and len(jsons_with_missing_branches) > args.end_summary_max:
            print(f"  ... plus {len(jsons_with_missing_branches) - args.end_summary_max} more")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()