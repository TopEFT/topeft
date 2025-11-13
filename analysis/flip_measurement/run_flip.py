from __future__ import annotations

import argparse
import cloudpickle
import gzip
import json
import os
import time
from collections import OrderedDict
from typing import Any, Mapping, MutableMapping, Sequence

import coffea.processor as processor
from coffea.nanoevents import NanoAODSchema
import topcoffea.modules.remote_environment as remote_environment

try:  # pragma: no cover - import resolution depends on execution context
    from . import flip_mr_processor  # type: ignore[import]
    from . import flip_ar_processor  # type: ignore[import]
except ImportError:  # pragma: no cover - fallback for script execution
    import flip_mr_processor  # type: ignore[no-redef]
    import flip_ar_processor  # type: ignore[no-redef]

from topcoffea.modules.utils import load_sample_json_file, read_cfg_file, update_cfg
from topeft.modules.executor import build_futures_executor, taskvine_log_configurator
from topeft.modules.executor_cli import (
    ExecutorCLIHelper,
    FuturesArgumentSpec,
    TaskVineArgumentSpec,
)
from topeft.modules.runner_output import (
    TupleKey,
    materialise_tuple_dict,
    normalise_runner_output,
)

parser = argparse.ArgumentParser(
    description=(
        "You can customize your run. The output pickle stores tuple-keyed "
        "histogram summaries for downstream consumers."
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=(
        "TaskVine workers can be launched with:\n"
        "  vine_submit_workers --python-env \"$(python -m topcoffea.modules.remote_environment)\" \\\n"
        "    --cores 4 --memory 16000 --disk 16000 -M <manager-name>\n"
        "Adjust the resources and manager to match your deployment."
    ),
)
parser.add_argument(
    "inputFiles",
    nargs="?",
    default="",
    help="Json or cfg file(s) containing files and metadata",
)
parser.add_argument(
    "--processor_name",
    "-r",
    default="flip_mr_processor",
    help="Which processor to run",
)
parser.add_argument(
    "--chunksize",
    "-s",
    default=100000,
    type=int,
    help="Number of events per chunk",
)
parser.add_argument(
    "--max-files",
    "-N",
    default=0,
    type=int,
    help="If specified, limit the number of root files per sample. Useful for testing",
)
parser.add_argument(
    "--nchunks",
    "-c",
    default=0,
    type=int,
    help="You can choose to run only a number of chunks",
)
parser.add_argument(
    "--outname",
    "-o",
    default="flipTopEFT",
    help="Name of the output file storing tuple-keyed histogram summaries",
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
    "--xrd",
    default="",
    help="The XRootD redirector to use when reading directly from json files",
)
EXECUTOR_CLI = ExecutorCLIHelper(
    remote_environment=remote_environment,
    futures_spec=FuturesArgumentSpec(
        workers_default=8,
        include_status=True,
        include_tail_timeout=True,
    ),
    taskvine_spec=TaskVineArgumentSpec(
        include_manager_name=True,
        include_manager_template=True,
        include_scratch_dir=True,
        include_resource_monitor=True,
        include_resources_mode=True,
        resource_monitor_default="measure",
        resources_mode_default="auto",
    ),
    extra_pip_local={"topeft": ["topeft", "setup.py"]},
    extra_conda=["pyyaml"],
)

EXECUTOR_CLI.configure_parser(parser)


def _summarise_tuple_entries(
    payload: Mapping[Any, Any]
) -> "OrderedDict[TupleKey, Mapping[str, Any]] | None":
    """Return a tuple-keyed summary mapping extracted from *payload*."""

    if not isinstance(payload, Mapping):
        return None

    tuple_entries: "OrderedDict[TupleKey, Any]" = OrderedDict()
    for key, value in payload.items():
        if isinstance(key, tuple) and len(key) == 4:
            tuple_entries[key] = value

    if not tuple_entries:
        return None

    return materialise_tuple_dict(tuple_entries)


def _print_sample_summary(name: str, payload: Mapping[str, Any]) -> None:
    jsn_txt = json.dumps(payload, indent=2, sort_keys=True)
    jsn_txt = jsn_txt[1:-1].strip("\n")
    summary = ""
    summary += f">> {name}\n"
    summary += f"{jsn_txt}\n"
    print(summary)


def _load_samples(
    input_files: Sequence[str],
    *,
    max_files: int,
    redirector: str,
) -> MutableMapping[str, MutableMapping[str, Any]]:
    samples: MutableMapping[str, MutableMapping[str, Any]] = {}
    for fn in input_files:
        if fn.endswith(".json"):
            sample = os.path.basename(fn).replace(".json", "")
            jsn = load_sample_json_file(fn)
            samples = update_cfg(
                jsn,
                name=sample,
                cfg=samples,
                max_files=max_files,
                redirector=redirector,
            )
        elif fn.endswith(".cfg"):
            samples = read_cfg_file(fn, cfg=samples, max_files=max_files)
        else:
            raise RuntimeError(f"Unknown input file: {fn}")
    return samples


def _build_file_map(samples: Mapping[str, Mapping[str, Any]]) -> MutableMapping[str, Sequence[str]]:
    flist: MutableMapping[str, Sequence[str]] = {}
    for sample_name, jsn in samples.items():
        xrd_src = jsn["redirector"]
        flist[sample_name] = [f"{xrd_src}{fn}" for fn in jsn["files"]]
        _print_sample_summary(sample_name, jsn)
    return flist


def _instantiate_processor(name: str, samples: Mapping[str, Any]):
    if name == "flip_mr_processor":
        processor_instance = flip_mr_processor.AnalysisProcessor(samples)
        extra_input_files_lst = ["flip_mr_processor.py"]
    elif name == "flip_ar_processor":
        processor_instance = flip_ar_processor.AnalysisProcessor(samples)
        extra_input_files_lst = ["flip_ar_processor.py"]
    else:
        raise Exception(f"Error: Unknown processor \"{name}\".")
    return processor_instance, extra_input_files_lst


def _run_executor(
    executor_name: str,
    *,
    executor_config,
    chunksize: int,
    nchunks: int | None,
    treename: str,
    flist: Mapping[str, Sequence[str]],
    processor_instance,
    extra_input_files_lst,
):
    futures_workers = executor_config.futures.workers
    futures_status = executor_config.futures.status
    futures_tail_timeout = executor_config.futures.tailtimeout

    if executor_name == "futures":
        exec_instance = build_futures_executor(
            processor,
            workers=futures_workers,
            status=futures_status,
            tailtimeout=futures_tail_timeout,
        )
    elif executor_name == "iterative":
        try:
            exec_instance = processor.IterativeExecutor()
        except AttributeError:  # pragma: no cover - depends on coffea build
            exec_instance = processor.iterative_executor()
    elif executor_name == "taskvine":
        taskvine_config = executor_config.taskvine
        staging_dir = taskvine_config.staging_directory()
        logs_dir = taskvine_config.logs_directory(staging_dir)

        taskvine_args = taskvine_config.executor_kwargs(
            extra_input_files=extra_input_files_lst,
            custom_init=taskvine_log_configurator(logs_dir),
            logs_dir=logs_dir,
        )
        exec_instance = taskvine_config.instantiate(
            processor,
            taskvine_args,
        )
    else:
        raise Exception(f"Executor \"{executor_name}\" is not known.")

    runner = processor.Runner(
        exec_instance,
        schema=NanoAODSchema,
        chunksize=chunksize,
        maxchunks=nchunks,
        skipbadfiles=False,
        xrootdtimeout=180,
    )
    return runner(flist, treename, processor_instance)


def main(argv: Sequence[str] | None = None) -> int:
    args = parser.parse_args(argv)
    executor_config = EXECUTOR_CLI.parse_args(args)

    input_files = args.inputFiles.replace(" ", "").split(",")
    executor_name = executor_config.executor
    processor_name = args.processor_name
    chunksize = args.chunksize
    nchunks = args.nchunks if args.nchunks else None
    outname = args.outname
    outpath = args.outpath
    treename = args.treename
    xrd = args.xrd
    max_files = args.max_files

    samples_to_process = _load_samples(
        input_files,
        max_files=max_files,
        redirector=xrd,
    )
    flist = _build_file_map(samples_to_process)

    processor_instance, extra_input_files_lst = _instantiate_processor(
        processor_name,
        samples_to_process,
    )

    tstart = time.time()
    output = _run_executor(
        executor_name,
        executor_config=executor_config,
        chunksize=chunksize,
        nchunks=nchunks,
        treename=treename,
        flist=flist,
        processor_instance=processor_instance,
        extra_input_files_lst=extra_input_files_lst,
    )
    serialised_output = normalise_runner_output(output)
    tuple_payload = _summarise_tuple_entries(serialised_output)
    stored_payload = tuple_payload if tuple_payload is not None else serialised_output

    dt = time.time() - tstart
    _ = dt  # retained for potential future logging

    if not os.path.isdir(outpath):
        os.system(f"mkdir -p {outpath}")
    out_pkl_file = os.path.join(outpath, outname + ".pkl.gz")
    print(f"\nSaving output in {out_pkl_file}...")
    with gzip.open(out_pkl_file, "wb") as fout:
        cloudpickle.dump(stored_payload, fout)

    print("Done!")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
