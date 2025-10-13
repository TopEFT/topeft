"""Command line entry point for the Run 2 quickstart helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from analysis.topeft_run2.quickstart import (
    DEFAULT_SCENARIO_NAME,
    PreparedSamples,
    prepare_samples,
    run_quickstart,
)


def _comma_separated(value: str) -> Iterable[str]:
    if "," in value:
        return [item for item in (token.strip() for token in value.split(",")) if item]
    return [value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute a lightweight Run 2 processing job using the new quickstart "
            "helpers.  The command validates the input samples before running "
            "the Coffea workflow with a limited histogram selection."
        )
    )
    parser.add_argument(
        "samples",
        help="Path to a sample JSON, directory or CFG file describing the inputs.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=Path("quickstart_outputs"),
        help="Directory where the output pickle file will be written.",
    )
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO_NAME,
        help="Metadata scenario to activate (default: %(default)s).",
    )
    parser.add_argument(
        "--variable",
        dest="variables",
        action="append",
        help=(
            "Restrict the histogram list to the specified variables. If omitted "
            "the defaults from the quickstart helper are used."
        ),
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Redirector prefix added to each file path (e.g. root://cmsxrootd.fnal.gov/).",
    )
    parser.add_argument(
        "--executor",
        default="futures",
        help="Executor backend to use (default: %(default)s).",
    )
    parser.add_argument(
        "--nworkers",
        type=int,
        default=1,
        help="Number of local workers to spawn when using the futures executor.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50000,
        help="Number of events per chunk to process per worker.",
    )
    parser.add_argument(
        "--nchunks",
        type=int,
        default=2,
        help="Limit the number of chunks processed from each sample (default: %(default)s).",
    )
    parser.add_argument(
        "--outname",
        default="quickstart",
        help="Base name of the output pickle (default: %(default)s).",
    )
    parser.add_argument(
        "--do-systs",
        action="store_true",
        help="Enable systematic variations (disabled by default to keep the run light).",
    )
    parser.add_argument(
        "--split-lep-flavor",
        action="store_true",
        help="Split histograms by lepton flavor categories.",
    )
    parser.add_argument(
        "--skip-sr",
        action="store_true",
        help="Skip all signal region categories from the metadata definition.",
    )
    parser.add_argument(
        "--skip-cr",
        action="store_true",
        help="Skip all control region categories from the metadata definition.",
    )
    parser.add_argument(
        "--pretend",
        action="store_true",
        help="Validate everything but do not execute the Coffea processor.",
    )
    parser.add_argument(
        "--treename",
        help="Override the tree name advertised by the sample JSON.",
    )
    parser.add_argument(
        "--wc",
        dest="wc_list",
        action="append",
        help="Optional Wilson coefficients to keep when running the processor.",
    )
    parser.add_argument(
        "--metadata",
        help="Path to an alternate metadata YAML file.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    variables = None
    if args.variables:
        variables = []
        for entry in args.variables:
            variables.extend(_comma_separated(entry))

    wc_list = None
    if args.wc_list:
        wc_list = []
        for entry in args.wc_list:
            wc_list.extend(_comma_separated(entry))

    prepared: PreparedSamples = prepare_samples(
        args.samples,
        scenario=args.scenario,
        metadata_path=args.metadata,
        prefix=args.prefix,
        variables=variables,
    )

    print(
        f"Validated {len(prepared.samples)} sample(s) covering {prepared.total_events} events."
    )

    if args.pretend:
        parser.exit(0, "Input samples validated successfully. Pretend mode requested.\n")

    run_quickstart(
        args.output,
        cfg_path=args.samples,
        scenario=args.scenario,
        metadata_path=args.metadata,
        prefix=args.prefix,
        variables=variables,
        executor=args.executor,
        nworkers=args.nworkers,
        chunksize=args.chunksize,
        nchunks=args.nchunks,
        outname=args.outname,
        treename=args.treename,
        split_lep_flavor=args.split_lep_flavor,
        do_systs=args.do_systs,
        skip_sr=args.skip_sr,
        skip_cr=args.skip_cr,
        wc_list=wc_list,
        pretend=args.pretend,
    )


if __name__ == "__main__":
    main()
