import argparse
import cloudpickle
import gzip
import json
import os
import time
from typing import Any

# import uproot
import coffea.processor as processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import topcoffea

if hasattr(NanoEventsFactory, "warn_missing_crossrefs"):
    NanoEventsFactory.warn_missing_crossrefs = False
elif hasattr(NanoAODSchema, "warn_missing_crossrefs"):
    NanoAODSchema.warn_missing_crossrefs = False

import sow_processor

from topeft.modules.executor_cli import (
    ExecutorCLIHelper,
    FuturesArgumentSpec,
    TaskVineArgumentSpec,
)


remote_environment = topcoffea.modules.remote_environment
tc_executor = topcoffea.modules.executor
build_futures_executor = tc_executor.build_futures_executor
futures_runner_overrides = tc_executor.futures_runner_overrides
taskvine_log_configurator = tc_executor.taskvine_log_configurator
tc_utils = topcoffea.modules.utils
load_sample_json_file = tc_utils.load_sample_json_file
read_cfg_file = tc_utils.read_cfg_file
update_cfg = tc_utils.update_cfg

parser = argparse.ArgumentParser(description='You can customize your run')
parser.add_argument('inputFiles'       , nargs='?', default='', help = 'Json or cfg file(s) containing files and metadata')
parser.add_argument('--chunksize','-s' , default=100000, type=int, help = 'Number of events per chunk')
parser.add_argument('--max-files','-N' , default=0, type=int, help = 'If specified, limit the number of root files per sample. Useful for testing')
parser.add_argument('--nchunks','-c'   , default=0, type=int, help = 'You can choose to run only a number of chunks')
parser.add_argument('--outname','-o'   , default='sowTopEFT', help = 'Name of the output file with histograms')
parser.add_argument('--outpath','-p'   , default='histos', help = 'Name of the output directory')
parser.add_argument('--treename'       , default='Events', help = 'Name of the tree inside the files')
parser.add_argument('--xrd'            , default='', help = 'The XRootD redirector to use when reading directly from json files')
parser.add_argument('--wc-list'        , action='extend', nargs='+', help = 'Specify a list of Wilson coefficients to use in filling histograms.')
EXECUTOR_CLI = ExecutorCLIHelper(
    remote_environment=remote_environment,
    futures_spec=FuturesArgumentSpec(
        workers_default=8,
        include_status=True,
        include_tail_timeout=True,
        include_memory=True,
        include_retries=True,
        include_retry_wait=True,
    ),
    taskvine_spec=TaskVineArgumentSpec(
        include_manager_name=False,
        include_manager_template=False,
        include_scratch_dir=False,
        include_resource_monitor=False,
        include_resources_mode=False,
        resource_monitor_default="measure",
        resources_mode_default="auto",
    ),
    extra_pip_local={"topeft": ["topeft", "setup.py"]},
    extra_conda=["pyyaml"],
)

EXECUTOR_CLI.configure_parser(parser)
parser.add_argument('--debug', action='store_true', help='Enable verbose logging in the sow processor.')

args = parser.parse_args()
executor_config = EXECUTOR_CLI.parse_args(args)

inputFiles = args.inputFiles.replace(' ','').split(',')  # Remove whitespace and split by commas
executor   = executor_config.executor
chunksize  = args.chunksize
nchunks    = args.nchunks if args.nchunks else None
outname    = args.outname
outpath    = args.outpath
treename   = args.treename
xrd        = args.xrd
max_files  = args.max_files
wc_lst     = args.wc_list if args.wc_list is not None else []
debug_mode = args.debug
futures_workers = executor_config.futures.workers
futures_status = executor_config.futures.status
futures_tail_timeout = executor_config.futures.tailtimeout
futures_memory = executor_config.futures.memory
futures_retries = executor_config.futures.retries
futures_retry_wait = executor_config.futures.retry_wait

samples_to_process = {}
for fn in inputFiles:
    if fn.endswith('.json'):
        sample = os.path.basename(fn).replace('.json','')
        jsn = load_sample_json_file(fn)
        samples_to_process = update_cfg(
            jsn,
            name=sample,
            cfg=samples_to_process,
            max_files=max_files,
            redirector=xrd
        )
    elif fn.endswith('.cfg'):
        samples_to_process = read_cfg_file(fn,cfg=samples_to_process,max_files=max_files)
    else:
        raise RuntimeError(f"Unknown input file: {fn}")

flist = {}
#for sample_name,jsn in samples_to_process['jsons'].items():
for sample_name,jsn in samples_to_process.items():
    xrd_src = jsn['redirector']
    flist[sample_name] = [f"{xrd_src}{fn}" for fn in jsn['files']]

    # Basically only try to build up the WC list if we haven't manually specified a list to use
    if len(wc_lst) == 0:
        for wc in jsn['WCnames']:
            if wc not in wc_lst:
                wc_lst.append(wc)

    jsn_txt = json.dumps(jsn,indent=2,sort_keys=True)

    # Strips off the leading and closing curly brackets, along with any associated newlines
    jsn_txt = jsn_txt[1:-1].strip("\n",)

    s = ""
    s += f">> {sample_name}\n"
    s += f"{jsn_txt}\n"
    print(s)

if wc_lst:
    if len(wc_lst) == 1:
        wc_print = ", ".join(wc_lst)
    else:
        wc_print = ", ".join(wc_lst[:-1]) + f", and {wc_lst[-1]}"
    print(f"Wilson Coefficients: {wc_print}.")
else:
    print("No Wilson coefficients specified")

processor_instance = sow_processor.AnalysisProcessor(samples_to_process,wc_lst, debug=debug_mode)

# Run the processor and get the output
tstart = time.time()

if executor == "futures":
    exec_instance = build_futures_executor(
        processor,
        workers=futures_workers,
        status=futures_status,
        tailtimeout=futures_tail_timeout,
    )
elif executor == "iterative":
    try:
        exec_instance = processor.IterativeExecutor()
    except AttributeError:  # pragma: no cover - depends on coffea build
        exec_instance = processor.iterative_executor()
elif executor == "taskvine":
    taskvine_config = executor_config.taskvine
    staging_dir = taskvine_config.staging_directory()
    logs_dir = taskvine_config.logs_directory(staging_dir)

    taskvine_args = taskvine_config.executor_kwargs(
        extra_input_files=["sow_processor.py"],
        custom_init=taskvine_log_configurator(logs_dir),
        logs_dir=logs_dir,
    )
    exec_instance = taskvine_config.instantiate(
        processor,
        taskvine_args,
    )
else:
    raise Exception(f"Executor \"{executor}\" is not known.")

runner_fields = set(getattr(processor.Runner, "__dataclass_fields__", {}))
runner_kwargs: dict[str, Any] = {}
if executor == "futures":
    runner_kwargs.update(
        futures_runner_overrides(
            runner_fields,
            memory=futures_memory,
        )
    )

runner = processor.Runner(
    exec_instance,
    schema=NanoAODSchema,
    chunksize=chunksize,
    maxchunks=nchunks,
    skipbadfiles=False,
    xrootdtimeout=900,
    **runner_kwargs,
)

attempt = 0
while True:
    try:
        output = runner(
            flist,
            processor_instance,
            treename,
            # coffea Runner.__call__ expects (fileset, processor_instance, treename)
        )
    except Exception as exc:
        if executor != "futures" or attempt >= futures_retries:
            raise
        attempt += 1
        print(
            "[futures] sow task failed (attempt {attempt}/{limit}): {error}.".format(
                attempt=attempt,
                limit=futures_retries,
                error=exc,
            )
        )
        if futures_retry_wait > 0:
            time.sleep(futures_retry_wait)
        continue
    else:
        break

expected_keys = {"sow", "sow_norm", "nEvents"}
missing_keys = expected_keys.difference(output.keys())
if missing_keys:
    raise RuntimeError(
        "sow_processor accumulator is missing expected keys: " + ", ".join(sorted(missing_keys))
    )

if debug_mode:
    present_keys = ", ".join(sorted(output.keys()))
    print(f"Accumulator keys present: {present_keys}")

dt = time.time() - tstart

# Save the output
if not os.path.isdir(outpath): os.system(f"mkdir -p {outpath}")
out_pkl_file = os.path.join(outpath,outname+".pkl.gz")
print(f"\nSaving output in {out_pkl_file}...")
with gzip.open(out_pkl_file, "wb") as fout:
    cloudpickle.dump(output, fout)

with gzip.open(out_pkl_file, "rb") as fin:
    saved_output = cloudpickle.load(fin)

missing_keys = expected_keys.difference(saved_output.keys())
if missing_keys:
    raise RuntimeError(
        "Persisted sow_processor accumulator is missing expected keys: "
        + ", ".join(sorted(missing_keys))
    )

print("Done!")
