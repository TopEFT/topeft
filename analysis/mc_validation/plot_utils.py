"""Plotting helpers for MC validation scripts."""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import mplhep as hep


__all__ = ["make_single_fig", "make_single_fig_with_ratio"]


def make_single_fig(
    histo,
    unit_norm_bool: bool,
    axis: Optional[str] = None,
    bins: Optional[Iterable[float]] = None,
    group: Optional[Iterable[str]] = None,
):
    """Plot a histogram with optional overlays from a sparse axis."""

    del group  # Retained for backward compatibility; no grouping performed here.
    bin_edges = list(bins) if bins is not None else None

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    hep.style.use("CMS")
    plt.sca(ax)
    hep.cms.label(lumi="7.9804", com="13.6", fontsize=10.0)

    if axis is None:
        hep.histplot(
            histo.eval({})[()][1:-1],
            ax=ax,
            bins=bin_edges,
            stack=False,
            density=unit_norm_bool,
            histtype="fill",
        )
    else:
        for axis_name in histo.axes[axis]:
            hep.histplot(
                histo[{axis: axis_name}].eval({})[()][1:-1],
                bins=bin_edges,
                stack=True,
                density=unit_norm_bool,
                label=axis_name,
            )

    plt.legend()
    ax.autoscale(axis="y")
    return fig


def make_single_fig_with_ratio(
    histo,
    axis_name: str,
    cat_ref: str,
    var: str = "lj0pt",
    err_p=None,
    err_m=None,
    err_ratio_p=None,
    err_ratio_m=None,
):
    """Plot stacked histogram with ratio panel and optional error bands."""

    del axis_name, cat_ref  # Ratio plotting between categories is not implemented.

    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7, 7),
        gridspec_kw={"height_ratios": (3, 1)},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.07)

    hep.histplot(
        histo,
        ax=ax,
        stack=False,
        clear=False,
    )

    plot_syst_err = all(v is not None for v in (err_p, err_m, err_ratio_p, err_ratio_m))
    if plot_syst_err:
        bin_edges_arr = histo.axes[var].edges
        ax.fill_between(
            bin_edges_arr,
            err_m,
            err_p,
            step="post",
            facecolor="none",
            edgecolor="gray",
            label="Syst err",
            hatch="///",
        )
        ax.set_ylim(0.0, 1.2 * max(err_p))
        rax.fill_between(
            bin_edges_arr,
            err_ratio_m,
            err_ratio_p,
            step="post",
            facecolor="none",
            edgecolor="gray",
            label="Syst err",
            hatch="////",
        )

    ax.set_xlabel("")
    rax.axhline(1.0, linestyle="-", color="k", linewidth=1)
    rax.set_ylabel("Ratio")
    rax.autoscale(axis="y")

    return fig
