#!/usr/bin/env python

"""Command-line interface for the Run 2 analysis workflow."""

from __future__ import annotations

import argparse
from typing import Sequence

from run_analysis_helpers import RunConfigBuilder

from .workflow import run_workflow


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser used by ``run_analysis.py``."""

    parser = argparse.ArgumentParser(description="You can customize your run")
    parser.add_argument(
        "jsonFiles",
        nargs="?",
        default="",
        help="Json file(s) containing files and metadata",
    )
    parser.add_argument(
        "--executor",
        "-x",
        default="work_queue",
        help="Which executor to use",
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
    parser.add_argument(
        "--nworkers",
        "-n",
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
        "--channel-feature",
        dest="channel_features",
        action="append",
        help=(
            "Include channel groups advertising the specified feature tag"
            " (for example 'offz_split' or 'requires_tau')."
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
        "--port",
        default="9123-9130",
        help="Specify the Work Queue port. An integer PORT or an integer range PORT_MIN-PORT_MAX.",
    )
    parser.add_argument(
        "--options",
        default=None,
        help="YAML file that specifies command-line options. Options explicitly set at command-line take precedence",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    parser_defaults = parser.parse_args([])
    args = parser.parse_args(argv)
    config_builder = RunConfigBuilder(parser_defaults)
    config = config_builder.build(args, getattr(args, "options", None))
    run_workflow(config)


if __name__ == "__main__":
    main()

