import argparse
import cloudpickle
import getpass
import gzip
import json
import os
import tempfile
import time
from pathlib import Path

# import uproot
from coffea import processor
from coffea.nanoevents import NanoAODSchema
import topcoffea.modules.remote_environment as remote_environment

import sow_processor

from topcoffea.modules.utils import load_sample_json_file, read_cfg_file, update_cfg


def _normalize_environment_file(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    if lowered in {"none", "null", "false", "0", "off", "disable", "disabled"}:
        return None
    if lowered == "auto":
        return "auto"
    return normalized


def _parse_port_range(port: str) -> tuple[int, int]:
    try:
        tokens = [int(token) for token in port.split('-') if token]
    except ValueError as exc:
        raise ValueError('Port specification must be an integer or range') from exc
    if not tokens:
        raise ValueError('At least one port value should be specified.')
    if len(tokens) > 2:
        raise ValueError('More than one port range was specified.')
    if len(tokens) == 1:
        tokens.append(tokens[0])
    return tokens[0], tokens[1]


def _staging_directory(executor: str) -> Path:
    base_dir = os.environ.get('TOPEFT_EXECUTOR_STAGING')
    if base_dir:
        staging = Path(base_dir).expanduser()
    else:
        staging = Path(tempfile.gettempdir()) / 'topeft' / _manager_name_base(executor)
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def _manager_name_base(executor: str) -> str:
    user = os.environ.get('USER')
    if not user:
        try:
            user = getpass.getuser()
        except Exception:  # pragma: no cover - best effort fallback
            user = 'coffea'
    return f"{user}-{executor}-coffea"

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
port_min, port_max = _parse_port_range(args.port)
environment_setting = _normalize_environment_file(args.environment_file)

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

processor_instance = sow_processor.AnalysisProcessor(samples_to_process,wc_lst)

# Run the processor and get the output
tstart = time.time()

if executor == "futures":
    try:
        exec_instance = processor.FuturesExecutor(workers=8)
    except AttributeError:  # pragma: no cover - depends on coffea build
        exec_instance = processor.futures_executor(workers=8)
elif executor == "iterative":
    try:
        exec_instance = processor.IterativeExecutor()
    except AttributeError:  # pragma: no cover - depends on coffea build
        exec_instance = processor.iterative_executor()
elif executor in {"work_queue", "taskvine"}:
    staging_dir = _staging_directory(executor)
    logs_dir = staging_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    manager_base = _manager_name_base(executor)
    executor_args = {
        'port': [port_min, port_max],
        'debug_log': str(logs_dir / 'debug.log'),
        'transactions_log': str(logs_dir / 'transactions.log'),
        'stats_log': str(logs_dir / 'stats.log'),
        'tasks_accum_log': str(logs_dir / 'tasks.log'),
        'extra_input_files': ["sow_processor.py"],
        'retries': 15,
        'compression': 8,
        'resource_monitor': 'measure',
        'resources_mode': 'auto',
        'filepath': str(staging_dir),
        'chunks_per_accum': 25,
        'chunks_accum_in_mem': 2,
        'fast_terminate_workers': 0,
        'verbose': True,
        'print_stdout': False,
    }

    env_payload = None
    if environment_setting == 'auto':
        env_payload = remote_environment.get_environment(
            extra_pip_local={"topeft": ["topeft", "setup.py"]},
            extra_conda=["pyyaml"],
        )
    elif environment_setting:
        env_payload = environment_setting
    if env_payload:
        executor_args['environment_file'] = env_payload

    if executor == 'work_queue':
        executor_args['master_name'] = manager_base
        exec_instance = processor.WorkQueueExecutor(**executor_args)
    else:
        executor_args['manager_name'] = manager_base
        executor_args['manager_name_template'] = f"{manager_base}-{{pid}}"
        try:
            exec_instance = processor.TaskVineExecutor(**executor_args)
        except TypeError as exc:
            if 'manager_name_template' in str(exc):
                executor_args.pop('manager_name_template', None)
                exec_instance = processor.TaskVineExecutor(**executor_args)
            else:
                raise
        except AttributeError as exc:  # pragma: no cover - depends on coffea build
            raise RuntimeError("TaskVineExecutor not available.") from exc
else:
    raise Exception(f"Executor \"{executor}\" is not known.")

runner = processor.Runner(exec_instance, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks, skipbadfiles=False, xrootdtimeout=900)
output = runner(flist, treename, processor_instance)

dt = time.time() - tstart

# Save the output
if not os.path.isdir(outpath): os.system(f"mkdir -p {outpath}")
out_pkl_file = os.path.join(outpath,outname+".pkl.gz")
print(f"\nSaving output in {out_pkl_file}...")
with gzip.open(out_pkl_file, "wb") as fout:
    cloudpickle.dump(output, fout)

print("Done!")
