import argparse
import cloudpickle
import gzip
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

# import uproot
import coffea.processor as processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

if hasattr(NanoEventsFactory, "warn_missing_crossrefs"):
    NanoEventsFactory.warn_missing_crossrefs = False
elif hasattr(NanoAODSchema, "warn_missing_crossrefs"):
    NanoAODSchema.warn_missing_crossrefs = False
import topcoffea.modules.remote_environment as remote_environment

import sow_processor

from topcoffea.modules.executor import (
    build_futures_executor,
    build_taskvine_args,
    build_work_queue_args,
    distributed_logs_dir,
    futures_runner_overrides,
    instantiate_taskvine_executor,
    instantiate_work_queue_executor,
    manager_name_base,
    parse_port_range,
    resolve_environment_file,
    taskvine_log_configurator,
)
from topcoffea.modules.utils import load_sample_json_file, read_cfg_file, update_cfg


def _staging_directory(executor: str) -> Path:
    base_dir = os.environ.get('TOPEFT_EXECUTOR_STAGING')
    if base_dir:
        staging = Path(base_dir).expanduser()
    else:
        staging = Path(tempfile.gettempdir()) / 'topeft' / manager_name_base(executor)
    staging.mkdir(parents=True, exist_ok=True)
    return staging


parser = argparse.ArgumentParser(description='You can customize your run')
parser.add_argument('inputFiles'       , nargs='?', default='', help = 'Json or cfg file(s) containing files and metadata')
parser.add_argument('--executor','-x'  , default='taskvine', help = 'Which executor to use')
parser.add_argument('--chunksize','-s' , default=100000, type=int, help = 'Number of events per chunk')
parser.add_argument('--max-files','-N' , default=0, type=int, help = 'If specified, limit the number of root files per sample. Useful for testing')
parser.add_argument('--nchunks','-c'   , default=0, type=int, help = 'You can choose to run only a number of chunks')
parser.add_argument('--outname','-o'   , default='sowTopEFT', help = 'Name of the output file with histograms')
parser.add_argument('--outpath','-p'   , default='histos', help = 'Name of the output directory')
parser.add_argument('--treename'       , default='Events', help = 'Name of the tree inside the files')
parser.add_argument('--xrd'            , default='', help = 'The XRootD redirector to use when reading directly from json files')
parser.add_argument('--wc-list'        , action='extend', nargs='+', help = 'Specify a list of Wilson coefficients to use in filling histograms.')
parser.add_argument(
    '--futures-workers',
    type=int,
    default=8,
    help='Maximum number of local processes to launch with the futures executor.',
)
parser.add_argument(
    '--futures-status',
    action=argparse.BooleanOptionalAction,
    default=None,
    help='Toggle the coffea futures progress bar.',
)
parser.add_argument(
    '--futures-tail-timeout',
    type=int,
    default=None,
    help='Timeout (in seconds) for cancelling stalled futures tasks.',
)
parser.add_argument(
    '--futures-memory',
    type=int,
    default=None,
    help='Approximate per-worker memory budget in MB for dynamic chunk sizing.',
)
parser.add_argument(
    '--futures-retries',
    type=int,
    default=0,
    help='Number of times to retry a futures execution after a failure.',
)
parser.add_argument(
    '--futures-retry-wait',
    type=float,
    default=5.0,
    help='Seconds to wait between futures retry attempts.',
)
parser.add_argument('--port'           , default='9123-9130', help = 'Specify the Work Queue/TaskVine port range (PORT or PORT_MIN-PORT_MAX).')
parser.add_argument(
    '--environment-file',
    default='auto',
    help=(
        "Environment tarball to ship with distributed executors. Use 'auto' to build via remote_environment, "
        "or 'none' to disable staging when workers already provide a Python environment."
    ),
)
parser.add_argument(
    '--no-environment-file',
    dest='environment_file',
    action='store_const',
    const='none',
    help="Disable environment shipping entirely (equivalent to --environment-file=none).",
)
parser.add_argument('--debug', action='store_true', help='Enable verbose logging in the sow processor.')

args = parser.parse_args()
inputFiles = args.inputFiles.replace(' ','').split(',')  # Remove whitespace and split by commas
executor   = (args.executor or '').strip().lower() or 'taskvine'
chunksize  = args.chunksize
nchunks    = args.nchunks if args.nchunks else None
outname    = args.outname
outpath    = args.outpath
treename   = args.treename
xrd        = args.xrd
max_files  = args.max_files
wc_lst     = args.wc_list if args.wc_list is not None else []
debug_mode = args.debug
port_min, port_max = parse_port_range(args.port)
environment_file = resolve_environment_file(
    args.environment_file,
    remote_environment,
    extra_pip_local={"topeft": ["topeft", "setup.py"]},
    extra_conda=["pyyaml"],
)
futures_workers = max(int(args.futures_workers or 1), 1)
futures_status = args.futures_status
futures_tail_timeout = None
if args.futures_tail_timeout and args.futures_tail_timeout > 0:
    futures_tail_timeout = int(args.futures_tail_timeout)
if args.futures_memory is None:
    futures_memory = None
else:
    futures_memory = int(args.futures_memory)
    if futures_memory <= 0:
        futures_memory = None
futures_retries = max(int(args.futures_retries or 0), 0)
futures_retry_wait = max(float(args.futures_retry_wait or 0.0), 0.0)

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
elif executor in {"work_queue", "taskvine"}:
    staging_dir = _staging_directory(executor)
    logs_dir = distributed_logs_dir(staging_dir, executor)
    manager_base = manager_name_base(executor)

    if executor == 'work_queue':
        executor_args = build_work_queue_args(
            staging_dir=staging_dir,
            logs_dir=logs_dir,
            manager_name=manager_base,
            port_range=(port_min, port_max),
            extra_input_files=["sow_processor.py"],
            resource_monitor='measure',
            resources_mode='auto',
            environment_file=environment_file,
        )
        exec_instance = instantiate_work_queue_executor(processor, executor_args)
    else:
        taskvine_args = build_taskvine_args(
            staging_dir=staging_dir,
            logs_dir=logs_dir,
            manager_name=manager_base,
            manager_name_template=f"{manager_base}-{{pid}}",
            extra_input_files=["sow_processor.py"],
            resource_monitor='measure',
            resources_mode='auto',
            environment_file=environment_file,
            custom_init=taskvine_log_configurator(logs_dir),
        )
        exec_instance = instantiate_taskvine_executor(
            processor,
            taskvine_args,
            port_range=(port_min, port_max),
            negotiate_port=True,
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
        output = runner(flist, treename, processor_instance)
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
