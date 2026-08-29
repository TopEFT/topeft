#!/usr/bin/env python3
"""Check a combined Run-3 JVM PKL against native correctionlib veto geometry.

This validator distinguishes three analysis-bin classes: fully vetoed,
fully nonvetoed, and boundary mixed.  It only requires zero post-veto content
in fully vetoed bins; off-mask depletion is reported rather than treated as a
failure because the Run-3 JVM prescription vetoes the full event.
"""

import argparse
import gzip
import json
import math
import pickle
from pathlib import Path

import correctionlib
import numpy as np


repository_root = Path(__file__).resolve().parents[2]
default_payload_root = (
    repository_root.parent / "topcoffea" / "topcoffea" / "data" / "POG" / "JME"
)
period_config = {
    "2022": ("Summer22_23Sep2023_RunCD_V1", "2022_Summer22"),
    "2022EE": ("Summer22EE_23Sep2023_RunEFG_V1", "2022_Summer22EE"),
    "2023": ("Summer23Prompt23_RunC_V1", "2023_Summer23"),
    "2023BPix": ("Summer23BPixPrompt23_RunD_V1", "2023_Summer23BPix"),
}
histogram_keys = ("jet_eta_phi_before_veto", "jet_eta_phi_after_veto")
selected_coordinates = {
    "channel": "2los_CRtt_2j",
    "systematic": "nominal",
    "appl": "isSR_2lOS",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pkl", required=True, type=Path)
    parser.add_argument("--payload-root", type=Path, default=default_payload_root)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args()


def dense_values(histogram):
    view = histogram.view(flow=False, as_dict=True)
    if isinstance(view, dict):
        if len(view) != 1:
            raise RuntimeError("Histogram selection still has categorical entries")
        view = next(iter(view.values()))
    if getattr(view.dtype, "fields", None) and "value" in view.dtype.fields:
        view = view["value"]
    return np.asarray(view, dtype=float)


def project_histogram(histogram, processes):
    if isinstance(processes, str):
        selected = histogram.integrate("process", processes)
    else:
        selected = histogram.integrate("process", processes)[{"process": sum}]
    for axis_name, value in selected_coordinates.items():
        selected = selected.integrate(axis_name, value)
    return dense_values(selected)


def load_histograms(input_pkl):
    with gzip.open(input_pkl, "rb") as source:
        histogram_dict = pickle.load(source)
    missing = [key for key in histogram_keys if key not in histogram_dict]
    if missing:
        raise KeyError(f"Missing required JVM histograms: {missing}")
    before, after = (histogram_dict[key] for key in histogram_keys)
    eta_edges = np.asarray(before.dense_axes[0].edges, dtype=float)
    phi_edges = np.asarray(before.dense_axes[1].edges, dtype=float)
    if len(eta_edges) != 105 or not np.allclose((eta_edges[0], eta_edges[-1]), (-5.2, 5.2)):
        raise ValueError("Unexpected eta-axis contract")
    if len(phi_edges) != 73 or not np.allclose((phi_edges[0], phi_edges[-1]), (-math.pi, math.pi)):
        raise ValueError("Unexpected phi-axis contract")
    if not np.array_equal(eta_edges, np.asarray(after.dense_axes[0].edges)):
        raise ValueError("Before/after eta axes differ")
    if not np.array_equal(phi_edges, np.asarray(after.dense_axes[1].edges)):
        raise ValueError("Before/after phi axes differ")
    return before, after, eta_edges, phi_edges


def payload_data(payload_root, period):
    correction_name, directory = period_config[period]
    path = Path(payload_root) / directory / "jetvetomaps.json.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with gzip.open(path, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    correction_item = next(
        item for item in payload["corrections"] if item["name"] == correction_name
    )
    map_item = next(
        item for item in correction_item["data"]["content"] if item["key"] == "jetvetomap"
    )
    node = map_item["value"]
    correction = correctionlib.CorrectionSet.from_file(str(path))[correction_name]
    return correction, np.asarray(node["edges"][0], dtype=float), np.asarray(node["edges"][1], dtype=float)


def partition_edges(low, high, payload_edges):
    return np.asarray(sorted({low, high, *[edge for edge in payload_edges if low < edge < high]}))


def classify_analysis_bins(correction, payload_eta_edges, payload_phi_edges, eta_edges, phi_edges):
    """Classify bins using every sub-rectangle induced by payload edges."""
    labels = np.empty((len(eta_edges) - 1, len(phi_edges) - 1), dtype=object)
    fractions = np.zeros(labels.shape, dtype=float)
    for eta_index, (eta_low, eta_high) in enumerate(zip(eta_edges[:-1], eta_edges[1:])):
        eta_parts = partition_edges(eta_low, eta_high, payload_eta_edges)
        for phi_index, (phi_low, phi_high) in enumerate(zip(phi_edges[:-1], phi_edges[1:])):
            phi_parts = partition_edges(phi_low, phi_high, payload_phi_edges)
            total_area = (eta_high - eta_low) * (phi_high - phi_low)
            veto_area = 0.0
            values = []
            for sub_eta_low, sub_eta_high in zip(eta_parts[:-1], eta_parts[1:]):
                for sub_phi_low, sub_phi_high in zip(phi_parts[:-1], phi_parts[1:]):
                    value = float(
                        correction.evaluate(
                            "jetvetomap",
                            (sub_eta_low + sub_eta_high) / 2,
                            (sub_phi_low + sub_phi_high) / 2,
                        )
                    )
                    values.append(value)
                    if value != 0.0:
                        veto_area += (sub_eta_high - sub_eta_low) * (sub_phi_high - sub_phi_low)
            fractions[eta_index, phi_index] = veto_area / total_area
            if all(value != 0.0 for value in values):
                labels[eta_index, phi_index] = "fully_vetoed"
            elif all(value == 0.0 for value in values):
                labels[eta_index, phi_index] = "fully_nonvetoed"
            else:
                labels[eta_index, phi_index] = "boundary_mixed"
    return labels, fractions


def summarize(values, mask, tolerance):
    selected = values[mask]
    return {
        "sum": float(np.sum(selected)),
        "absolute_sum": float(np.sum(np.abs(selected))),
        "nonzero_bins": int(np.count_nonzero(np.abs(selected) > tolerance)),
        "max_absolute_bin": float(np.max(np.abs(selected))) if selected.size else 0.0,
    }


def period_metrics(before_histogram, after_histogram, eta_edges, phi_edges, payload_root, period, tolerance):
    correction, payload_eta_edges, payload_phi_edges = payload_data(payload_root, period)
    labels, fractions = classify_analysis_bins(
        correction, payload_eta_edges, payload_phi_edges, eta_edges, phi_edges
    )
    processes = tuple(str(value) for value in before_histogram.axes["process"])
    data_process = f"data{period}"
    mc_processes = tuple(
        process for process in processes if process.endswith(period) and not process.startswith("data")
    )
    if data_process not in processes or not mc_processes:
        raise KeyError(f"Missing period processes for {period}")
    results = {
        "mask": {
            "fully_vetoed_bins": int(np.count_nonzero(labels == "fully_vetoed")),
            "boundary_mixed_bins": int(np.count_nonzero(labels == "boundary_mixed")),
            "fully_nonvetoed_bins": int(np.count_nonzero(labels == "fully_nonvetoed")),
            "veto_area_fraction": float(np.mean(fractions)),
        },
        "samples": {},
    }
    masks = {label: labels == label for label in ("fully_vetoed", "boundary_mixed", "fully_nonvetoed")}
    for sample_name, processes_to_sum in (("data", data_process), ("mc", mc_processes)):
        before = project_histogram(before_histogram, processes_to_sum)
        after = project_histogram(after_histogram, processes_to_sum)
        sample_metrics = {}
        for label, mask in masks.items():
            before_summary = summarize(before, mask, tolerance)
            after_summary = summarize(after, mask, tolerance)
            sample_metrics[label] = {
                "before": before_summary,
                "after": after_summary,
                "changed_bins": int(np.count_nonzero(np.abs((before - after)[mask]) > tolerance)),
            }
        results["samples"][sample_name] = sample_metrics
    return results


def main():
    arguments = parse_arguments()
    before_histogram, after_histogram, eta_edges, phi_edges = load_histograms(arguments.input_pkl)
    results = {
        period: period_metrics(
            before_histogram,
            after_histogram,
            eta_edges,
            phi_edges,
            arguments.payload_root,
            period,
            arguments.tolerance,
        )
        for period in period_config
    }
    residuals = []
    for period, result in results.items():
        for sample_name, sample_result in result["samples"].items():
            residual = sample_result["fully_vetoed"]["after"]["max_absolute_bin"]
            if residual > arguments.tolerance:
                residuals.append(f"{period} {sample_name}: {residual}")
    print(json.dumps(results, indent=2, sort_keys=True))
    if residuals:
        raise SystemExit("Fully-vetoed post-veto residual(s): " + "; ".join(residuals))


if __name__ == "__main__":
    main()
