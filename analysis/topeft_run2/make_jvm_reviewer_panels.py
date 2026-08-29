#!/usr/bin/env python3
"""Render Run-3 jet-veto-map eta-phi panels from a combined JVM PKL.

The tool is intentionally narrow: it draws data and total-MC before/after
panels for the nominal ``2los_CRtt_2j`` JVM diagnostic histograms.  It does
not run the processor or apply data-driven postprocessing.
"""

import argparse
import gzip
import json
import math
import pickle
from pathlib import Path

import correctionlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import yaml


repository_root = Path(__file__).resolve().parents[2]
default_payload_root = (
    repository_root.parent / "topcoffea" / "topcoffea" / "data" / "POG" / "JME"
)
metadata_path = repository_root / "topeft" / "params" / "cr_sr_plots_metadata.yml"

period_config = {
    "2022": {
        "correction": "Summer22_23Sep2023_RunCD_V1",
        "payload_directory": "2022_Summer22",
    },
    "2022EE": {
        "correction": "Summer22EE_23Sep2023_RunEFG_V1",
        "payload_directory": "2022_Summer22EE",
    },
    "2023": {
        "correction": "Summer23Prompt23_RunC_V1",
        "payload_directory": "2023_Summer23",
    },
    "2023BPix": {
        "correction": "Summer23BPixPrompt23_RunD_V1",
        "payload_directory": "2023_Summer23BPix",
    },
}

histogram_keys = ("jet_eta_phi_before_veto", "jet_eta_phi_after_veto")
selected_coordinates = {
    "channel": "2los_CRtt_2j",
    "systematic": "nominal",
    "appl": "isSR_2lOS",
}
output_names = {
    "data": "run3_jvm_data_before_after_with_payload",
    "mc": "run3_jvm_mc_before_after_with_payload",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pkl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--payload-root",
        type=Path,
        default=default_payload_root,
        help="Root containing the period-specific JME payload directories.",
    )
    return parser.parse_args()


def load_period_luminosities(path=metadata_path):
    """Read Run-3 luminosity labels from the live plotting metadata."""
    with Path(path).open(encoding="utf-8") as source:
        metadata = yaml.safe_load(source)
    pairs = metadata["LUMI_COM_PAIRS"]
    return {period: tuple(pairs[period]) for period in period_config}


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


def period_processes(processes, period):
    """Return exact data and MC process labels for one period token."""
    data_process = f"data{period}"
    mc_processes = tuple(
        process
        for process in processes
        if process.endswith(period) and not process.startswith("data")
    )
    if data_process not in processes:
        raise KeyError(f"Missing data process {data_process}")
    if not mc_processes:
        raise KeyError(f"Missing MC processes for {period}")
    return data_process, mc_processes


def validate_dense_axes(before_histogram, after_histogram):
    """Validate the accepted JVM eta/phi binning and return its edges."""
    eta_edges = np.asarray(before_histogram.dense_axes[0].edges, dtype=float)
    phi_edges = np.asarray(before_histogram.dense_axes[1].edges, dtype=float)
    if len(eta_edges) != 105 or not np.allclose(
        (eta_edges[0], eta_edges[-1]), (-5.2, 5.2)
    ):
        raise ValueError("Unexpected eta-axis contract")
    if len(phi_edges) != 73 or not np.allclose(
        (phi_edges[0], phi_edges[-1]), (-math.pi, math.pi)
    ):
        raise ValueError("Unexpected phi-axis contract")
    if not np.array_equal(eta_edges, np.asarray(after_histogram.dense_axes[0].edges)):
        raise ValueError("Before/after eta axes differ")
    if not np.array_equal(phi_edges, np.asarray(after_histogram.dense_axes[1].edges)):
        raise ValueError("Before/after phi axes differ")
    return eta_edges, phi_edges


def load_histograms(input_pkl):
    if not input_pkl.is_file():
        raise FileNotFoundError(input_pkl)
    with gzip.open(input_pkl, "rb") as source:
        histogram_dict = pickle.load(source)
    missing = [key for key in histogram_keys if key not in histogram_dict]
    if missing:
        raise KeyError(f"Missing required JVM histograms: {missing}")
    before_histogram = histogram_dict[histogram_keys[0]]
    after_histogram = histogram_dict[histogram_keys[1]]
    eta_edges, phi_edges = validate_dense_axes(before_histogram, after_histogram)
    return before_histogram, after_histogram, eta_edges, phi_edges


def select_period_arrays(before_histogram, after_histogram):
    processes = tuple(str(value) for value in before_histogram.axes["process"])
    selected = {}
    for period in period_config:
        data_process, mc_processes = period_processes(processes, period)
        selected[period] = {
            "data_before": project_histogram(before_histogram, data_process),
            "data_after": project_histogram(after_histogram, data_process),
            "mc_before": project_histogram(before_histogram, mc_processes),
            "mc_after": project_histogram(after_histogram, mc_processes),
        }
    return selected


def payload_path(payload_root, period):
    return Path(payload_root) / period_config[period]["payload_directory"] / "jetvetomaps.json.gz"


def load_payload_boundary(payload_root, period):
    """Return exact exposed nonzero-cell boundaries for one JVM payload."""
    path = payload_path(payload_root, period)
    if not path.is_file():
        raise FileNotFoundError(path)
    with gzip.open(path, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    correction_name = period_config[period]["correction"]
    correction_item = next(
        item for item in payload["corrections"] if item["name"] == correction_name
    )
    map_item = next(
        item
        for item in correction_item["data"]["content"]
        if item["key"] == "jetvetomap"
    )
    node = map_item["value"]
    eta_edges = np.asarray(node["edges"][0], dtype=float)
    phi_edges = np.asarray(node["edges"][1], dtype=float)
    correction = correctionlib.CorrectionSet.from_file(str(path))[correction_name]
    active = np.zeros((len(eta_edges) - 1, len(phi_edges) - 1), dtype=bool)
    for eta_index, (eta_low, eta_high) in enumerate(zip(eta_edges[:-1], eta_edges[1:])):
        for phi_index, (phi_low, phi_high) in enumerate(zip(phi_edges[:-1], phi_edges[1:])):
            active[eta_index, phi_index] = (
                correction.evaluate(
                    "jetvetomap",
                    float((eta_low + eta_high) / 2),
                    float((phi_low + phi_high) / 2),
                )
                != 0.0
            )
    return eta_edges, phi_edges, active


def boundary_segments(eta_edges, phi_edges, active):
    """Return line segments at exposed native payload-cell edges."""
    segments = []
    eta_bins, phi_bins = active.shape
    for eta_index in range(eta_bins):
        for phi_index in range(phi_bins):
            if not active[eta_index, phi_index]:
                continue
            eta_low, eta_high = eta_edges[eta_index], eta_edges[eta_index + 1]
            phi_low, phi_high = phi_edges[phi_index], phi_edges[phi_index + 1]
            if eta_index == 0 or not active[eta_index - 1, phi_index]:
                segments.append(((eta_low, phi_low), (eta_low, phi_high)))
            if eta_index == eta_bins - 1 or not active[eta_index + 1, phi_index]:
                segments.append(((eta_high, phi_low), (eta_high, phi_high)))
            if phi_index == 0 or not active[eta_index, phi_index - 1]:
                segments.append(((eta_low, phi_low), (eta_high, phi_low)))
            if phi_index == phi_bins - 1 or not active[eta_index, phi_index + 1]:
                segments.append(((eta_low, phi_high), (eta_high, phi_high)))
    return segments


def build_normalization(before, after, sample_kind):
    """Build one non-clipping normalization shared by a before/after pair."""
    minimum = float(min(np.min(before), np.min(after)))
    maximum = float(max(np.max(before), np.max(after)))
    if sample_kind == "data":
        if minimum < 0.0:
            raise ValueError("Data occupancy has a negative bin")
        upper = max(maximum, 1.0)
        return Normalize(vmin=0.0, vmax=upper), 0.0, upper, "linear_nonnegative"
    if minimum < 0.0 < maximum:
        return (
            TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum),
            minimum,
            maximum,
            "two_slope_sign_aware",
        )
    if maximum <= 0.0:
        return Normalize(vmin=minimum, vmax=0.0), minimum, 0.0, "linear_nonpositive"
    upper = max(maximum, 1.0)
    return Normalize(vmin=0.0, vmax=upper), 0.0, upper, "linear_nonnegative"


def ensure_empty_output_directory(output_dir):
    """Create an output directory or fail rather than overwrite its contents."""
    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    return output_dir


def draw_panel(axis, values, eta_edges, phi_edges, normalization, segments, row_label):
    image = axis.pcolormesh(
        eta_edges,
        phi_edges,
        values.T,
        shading="auto",
        cmap="viridis",
        norm=normalization,
    )
    axis.add_collection(
        LineCollection(segments, colors="crimson", linewidths=0.75, zorder=4)
    )
    axis.plot([], [], color="crimson", linewidth=1.2, label="Jet veto map")
    axis.legend(loc="lower right", frameon=True, fontsize=8)
    axis.set(xlim=(-5.2, 5.2), ylim=(-math.pi, math.pi), xlabel="Jet eta", ylabel="Jet phi")
    axis.set_yticks((-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi))
    axis.set_yticklabels((r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"))
    if row_label:
        axis.text(
            0.015,
            0.975,
            row_label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5},
        )
    return image


def render_figure(
    sample_kind,
    selected_arrays,
    eta_edges,
    phi_edges,
    payload_root,
    luminosities,
    output_dir,
):
    figure, axes = plt.subplots(4, 2, figsize=(14, 20), constrained_layout=True)
    figure.suptitle(
        f"CMS  {'Data' if sample_kind == 'data' else 'Simulation'}\n"
        r"$2\ell_{OS}$ $t\bar{t}$ control region ($e\mu$, exactly 2 jets)",
        fontsize=17,
        fontweight="bold",
    )
    axes[0, 0].set_title("Before jet veto", fontsize=14)
    axes[0, 1].set_title("After jet veto", fontsize=14)
    normalization_records = {}
    for row_index, period in enumerate(period_config):
        before = selected_arrays[period][f"{sample_kind}_before"]
        after = selected_arrays[period][f"{sample_kind}_after"]
        normalization, lower, upper, mode = build_normalization(before, after, sample_kind)
        eta_payload, phi_payload, active = load_payload_boundary(payload_root, period)
        segments = boundary_segments(eta_payload, phi_payload, active)
        lumi, energy = luminosities[period]
        row_label = f"{period}  ({lumi} fb$^{{-1}}$, {energy} TeV)"
        before_image = draw_panel(
            axes[row_index, 0], before, eta_edges, phi_edges, normalization, segments, row_label
        )
        draw_panel(axes[row_index, 1], after, eta_edges, phi_edges, normalization, segments, None)
        colorbar = figure.colorbar(
            before_image, ax=axes[row_index, :], fraction=0.025, pad=0.012
        )
        colorbar.set_label("Jet entries" if sample_kind == "data" else "Weighted jet yield")
        normalization_records[period] = {
            "vmin": lower,
            "vmax": upper,
            "mode": mode,
            "segment_count": len(segments),
        }
    basename = output_names[sample_kind]
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"
    figure.savefig(png_path, dpi=180)
    figure.savefig(pdf_path)
    plt.close(figure)
    return normalization_records, png_path, pdf_path


def main():
    arguments = parse_arguments()
    output_dir = ensure_empty_output_directory(arguments.output_dir)
    luminosities = load_period_luminosities()
    before_histogram, after_histogram, eta_edges, phi_edges = load_histograms(arguments.input_pkl)
    selected_arrays = select_period_arrays(before_histogram, after_histogram)
    results = {}
    for sample_kind in ("data", "mc"):
        records, png_path, pdf_path = render_figure(
            sample_kind,
            selected_arrays,
            eta_edges,
            phi_edges,
            arguments.payload_root,
            luminosities,
            output_dir,
        )
        results[sample_kind] = {
            "normalization": records,
            "png": str(png_path),
            "pdf": str(pdf_path),
        }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
