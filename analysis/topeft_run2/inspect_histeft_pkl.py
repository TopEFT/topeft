#!/usr/bin/env python3
"""Read-only summary tool for TOP EFT histogram pickle files."""

from __future__ import annotations

import argparse
import gzip
import pickle
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected in the analysis env.
    np = None


def _enable_pickle_compat() -> None:
    try:
        from topcoffea.modules.compat import ensure_histEFT_py39_compat
    except Exception:
        return
    try:
        ensure_histEFT_py39_compat()
    except Exception:
        return


def _load_pickle(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        return pickle.load(handle)


def _is_mapping_like(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        return True
    return hasattr(obj, "items") and callable(getattr(obj, "items"))


def _items(obj: Any):
    if _is_mapping_like(obj):
        return obj.items()
    return ()


def _is_hist_like(obj: Any) -> bool:
    return hasattr(obj, "axes") and (
        hasattr(obj, "values") or hasattr(obj, "view") or hasattr(obj, "eval")
    )


def _axis_name(axis: Any, index: int) -> str:
    for attr in ("name", "label"):
        try:
            value = getattr(axis, attr)
        except Exception:
            value = None
        if value not in (None, ""):
            return str(value)
    try:
        metadata = getattr(axis, "metadata")
    except Exception:
        metadata = None
    if isinstance(metadata, Mapping):
        value = metadata.get("name") or metadata.get("label")
        if value not in (None, ""):
            return str(value)
    return f"axis_{index}"


def _axes(obj: Any) -> list[Any]:
    try:
        return list(obj.axes)
    except Exception:
        return []


def _axis_names(obj: Any) -> list[str]:
    return [_axis_name(axis, idx) for idx, axis in enumerate(_axes(obj))]


def _axis_by_name(obj: Any, name: str) -> Any | None:
    try:
        return obj.axes[name]
    except Exception:
        pass
    for idx, axis in enumerate(_axes(obj)):
        if _axis_name(axis, idx) == name:
            return axis
    return None


def _safe_len(obj: Any) -> int | None:
    try:
        return len(obj)
    except Exception:
        return None


def _format_limited(values: list[str], max_labels: int) -> str:
    if len(values) > max_labels:
        shown = values[:max_labels]
        return ", ".join(shown) + f", ... ({len(values)} total)"
    return ", ".join(values)


def _axis_labels(axis: Any, max_labels: int) -> tuple[list[str], int | None]:
    labels: list[str] = []
    total = _safe_len(axis)
    try:
        iterator = iter(axis)
    except Exception:
        return labels, total
    for idx, value in enumerate(iterator):
        if idx >= max_labels:
            break
        labels.append(str(value))
    return labels, total


def _axis_edges(axis: Any, max_labels: int) -> str | None:
    if "Category" in type(axis).__name__:
        return None
    if np is None:
        return None
    try:
        edges = np.asarray(axis.edges)
    except Exception:
        return None
    if edges.ndim != 1 or edges.size == 0:
        return None
    shown = [f"{float(x):g}" for x in edges[: max_labels + 1]]
    suffix = f", ... ({edges.size} edges)" if edges.size > max_labels + 1 else ""
    return ", ".join(shown) + suffix


def _wc_names(obj: Any) -> list[str]:
    for attr in ("wc_names", "_wc_names"):
        try:
            value = getattr(obj, attr)
        except Exception:
            continue
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if value is not None:
            try:
                return [str(x) for x in value]
            except Exception:
                return [str(value)]
    return []


def _maybe_nominal(obj: Any) -> Any:
    axis = _axis_by_name(obj, "systematic")
    if axis is None:
        return obj
    labels, _ = _axis_labels(axis, max_labels=100000)
    if "nominal" not in labels:
        return obj
    for attempt in (
        lambda: obj.integrate("systematic", "nominal"),
        lambda: obj[{"systematic": "nominal"}],
    ):
        try:
            return attempt()
        except Exception:
            pass
    return obj


def _values(obj: Any) -> Any:
    obj = _maybe_nominal(obj)
    if hasattr(obj, "eval"):
        try:
            return obj.eval({})
        except Exception:
            pass
    for kwargs in ({"flow": True}, {"flow": False}, {}):
        try:
            return obj.values(**kwargs)
        except Exception:
            pass
    return None


def _variances(obj: Any) -> Any:
    obj = _maybe_nominal(obj)
    for kwargs in ({"flow": True}, {"flow": False}, {}):
        try:
            return obj.variances(**kwargs)
        except Exception:
            pass
    return None


def _array_sum(value: Any) -> float | None:
    if np is None or value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except Exception:
        return None
    if array.size == 0:
        return 0.0
    try:
        return float(np.nansum(array))
    except Exception:
        return None


def _sum_payload(payload: Any) -> tuple[float | None, int]:
    if payload is None:
        return None, 0
    if _is_mapping_like(payload):
        total = 0.0
        count = 0
        for value in payload.values():
            subtotal = _array_sum(value)
            if subtotal is None:
                continue
            total += subtotal
            count += 1
        return (total if count else None), count
    subtotal = _array_sum(payload)
    return subtotal, 1 if subtotal is not None else 0


def _print_axis_summary(obj: Any, max_labels: int) -> None:
    axes = _axes(obj)
    if not axes:
        print("  axes: unavailable")
        return
    print(f"  axes ({len(axes)}):")
    for idx, axis in enumerate(axes):
        name = _axis_name(axis, idx)
        axis_type = type(axis).__module__ + "." + type(axis).__name__
        size = _safe_len(axis)
        size_text = "unknown" if size is None else str(size)
        print(f"    - {name}: {axis_type}, size={size_text}")
        edges = _axis_edges(axis, max_labels=max_labels)
        if edges is not None:
            print(f"      edges: {edges}")
            continue
        labels, total = _axis_labels(axis, max_labels=max_labels)
        if labels:
            total_text = "unknown" if total is None else str(total)
            print(f"      labels: {_format_limited(labels, max_labels)}")
            print(f"      label_count: {total_text}")


def _print_known_labels(obj: Any, max_labels: int) -> None:
    for axis_name in ("process", "channel", "systematic", "appl"):
        axis = _axis_by_name(obj, axis_name)
        if axis is None:
            continue
        labels, total = _axis_labels(axis, max_labels=max_labels)
        total_text = "unknown" if total is None else str(total)
        if labels:
            print(f"  {axis_name} labels: {_format_limited(labels, max_labels)}")
            print(f"  {axis_name} label_count: {total_text}")
        else:
            print(f"  {axis_name} labels: unavailable")


def _print_yield_summary(obj: Any) -> None:
    values_total, value_entries = _sum_payload(_values(obj))
    variances_total, variance_entries = _sum_payload(_variances(obj))
    if values_total is None:
        print("  total nominal yield: unavailable")
    else:
        print(
            "  total nominal yield: "
            f"{values_total:.12g} from {value_entries} value block(s)"
        )
    if variances_total is not None:
        print(
            "  total nominal variance: "
            f"{variances_total:.12g} from {variance_entries} variance block(s)"
        )


def _hist_items(root: Any):
    if _is_hist_like(root):
        return [("<top-level>", root)]
    if _is_mapping_like(root):
        return [(str(key), value) for key, value in _items(root) if _is_hist_like(value)]
    return []


def summarize(args: argparse.Namespace) -> int:
    path = Path(args.pkl_path)
    if not path.exists():
        print(f"ERROR: file does not exist: {path}", file=sys.stderr)
        return 2

    _enable_pickle_compat()
    try:
        root = _load_pickle(path)
    except Exception as exc:
        print(f"ERROR: failed to load {path}: {exc!r}", file=sys.stderr)
        return 1

    print(f"file: {path}")
    print(f"top-level type: {type(root).__module__}.{type(root).__name__}")

    if _is_mapping_like(root):
        keys = [str(key) for key, _ in _items(root)]
        print(f"top-level keys: {_format_limited(keys, args.max_labels)}")
        print(f"top-level key_count: {len(keys)}")

    hist_items = _hist_items(root)
    if args.hist:
        selected = [(name, obj) for name, obj in hist_items if name == args.hist]
        if not selected:
            print(f"ERROR: histogram key not found or not histogram-like: {args.hist}", file=sys.stderr)
            return 2
        hist_items = selected
    else:
        hist_items = hist_items[: args.max_hists]

    print(f"histogram-like objects shown: {len(hist_items)}")
    if not hist_items:
        return 0

    for name, obj in hist_items:
        print("")
        print(f"histogram: {name}")
        print(f"  type: {type(obj).__module__}.{type(obj).__name__}")
        dense_hists = getattr(obj, "_dense_hists", None)
        if isinstance(dense_hists, Mapping):
            print(f"  dense block count: {len(dense_hists)}")
        wc_names = _wc_names(obj)
        if wc_names:
            print(f"  WC names: {_format_limited(wc_names, args.max_labels)}")
            print(f"  WC count: {len(wc_names)}")
        _print_axis_summary(obj, max_labels=args.max_labels)
        _print_known_labels(obj, max_labels=args.max_labels)
        if args.yield_summary:
            _print_yield_summary(obj)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print a read-only summary of a TOP EFT histogram pkl or pkl.gz file. "
            "The tool is intentionally introspective and does not modify the input."
        )
    )
    parser.add_argument("pkl_path", help="Path to a .pkl or .pkl.gz file.")
    parser.add_argument(
        "--hist",
        help="Inspect only one top-level histogram key. By default, the first histogram-like keys are shown.",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=20,
        help="Maximum labels or edges to print per axis/key list. Default: 20.",
    )
    parser.add_argument(
        "--max-hists",
        type=int,
        default=5,
        help="Maximum histogram-like objects to summarize when --hist is not given. Default: 5.",
    )
    parser.add_argument(
        "--yield-summary",
        action="store_true",
        help="Also print a simple nominal total yield and variance when discoverable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_labels < 1:
        parser.error("--max-labels must be at least 1")
    if args.max_hists < 1:
        parser.error("--max-hists must be at least 1")
    return summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
