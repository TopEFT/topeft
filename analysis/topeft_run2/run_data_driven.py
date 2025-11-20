#!/usr/bin/env python3
"""Standalone helper to build data driven histograms from saved metadata.

Quickstart examples:
  - Metadata sidecar: python run_data_driven.py --metadata-json histos/plotsTopEFT_np.pkl.gz.metadata.json \
      --apply-renormfact-envelope
  - Direct pickle paths: python run_data_driven.py --input-pkl histos/plotsTopEFT.pkl.gz \
      --output-pkl histos/plotsTopEFT_np.pkl.gz --apply-renormfact-envelope
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional

import topcoffea.modules.utils as utils

from topeft.modules.dataDrivenEstimation import DataDrivenProducer
from topeft.modules.get_renormfact_envelope import get_renormfact_envelope


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize deferred nonprompt/flips histograms using the metadata emitted by run_analysis.py.\n\n"
            "Quickstart:\n"
            "  - Metadata sidecar: python run_data_driven.py --metadata-json histos/plotsTopEFT_np.pkl.gz.metadata.json\\\n"
            "      --apply-renormfact-envelope\n"
            "  - Direct pickle paths: python run_data_driven.py --input-pkl histos/plotsTopEFT.pkl.gz\\\n"
            "      --output-pkl histos/plotsTopEFT_np.pkl.gz --apply-renormfact-envelope"
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
        help="Also run the renorm/fact envelope step on the output histogram.",
    )
    parser.add_argument(
        "--only-flips",
        action="store_true",
        help="Drop nonprompt processes so only flips contributions remain in the output histograms.",
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
    return parser


def _load_metadata(metadata_path: str) -> Dict[str, Any]:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path) as metadata_stream:
        payload = json.load(metadata_stream)
    version = payload.get("metadata_version")
    if version != 1:
        raise ValueError(
            f"Unsupported metadata schema version {version!r}. Expected version 1 metadata."
        )
    resolved_years = payload.get("resolved_years")
    sample_years = payload.get("sample_years")
    if resolved_years and sample_years:
        resolved_set = set(resolved_years)
        sample_set = set(sample_years)
        missing = resolved_set - sample_set
        if missing:
            raise ValueError(
                "Metadata contains requested years that are not present in the samples: "
                f"{sorted(missing)}"
            )
    return payload


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
) -> float:
    if quiet:
        return last_heartbeat
    now = time.monotonic()
    if heartbeat_seconds <= 0 or now - last_heartbeat >= heartbeat_seconds:
        elapsed = now - start_time
        print(f"[run_data_driven] Processed {count} histograms after {elapsed:.1f}s...")
        return now
    return last_heartbeat


def _finalize_histograms(
    input_pkl: str,
    output_pkl: str,
    *,
    only_flips: bool,
    apply_envelope: bool,
    heartbeat_seconds: float,
    quiet: bool,
) -> None:
    ddp = DataDrivenProducer(input_pkl, output_pkl)
    histograms = ddp.getDataDrivenHistogram()

    start_time = time.monotonic()
    last_heartbeat = start_time
    processed = 0
    filtered: Dict[str, Any] = {}
    for key, histo in histograms.items():
        processed += 1
        last_heartbeat = _maybe_emit_heartbeat(
            count=processed,
            start_time=start_time,
            last_heartbeat=last_heartbeat,
            heartbeat_seconds=heartbeat_seconds,
            quiet=quiet,
        )

        working_histo = _filter_to_flips(histo) if only_flips else histo
        filtered[key] = working_histo

    if not quiet and processed:
        elapsed = time.monotonic() - start_time
        print(f"[run_data_driven] Finalized {processed} histograms in {elapsed:.1f}s.")
    histograms = filtered
    if apply_envelope:
        histograms = get_renormfact_envelope(histograms)
    os.makedirs(os.path.dirname(output_pkl) or ".", exist_ok=True)
    utils.dump_to_pkl(output_pkl, histograms)


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    metadata: Dict[str, Any] = {}
    metadata_dir: Optional[str] = None
    if args.metadata_json:
        metadata = _load_metadata(args.metadata_json)
        metadata_dir = os.path.dirname(os.path.abspath(args.metadata_json))
        if not metadata.get("do_np", True):
            raise ValueError(
                "Metadata indicates nonprompt estimation was disabled (do_np=False). Nothing to do."
            )

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

    _finalize_histograms(
        input_pkl,
        output_pkl,
        only_flips=args.only_flips,
        apply_envelope=args.apply_renormfact_envelope,
        heartbeat_seconds=args.heartbeat_seconds,
        quiet=args.quiet,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
