import argparse
import cloudpickle
import getpass
import errno
import gzip
import json
import os
import socket
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Set

# import uproot
import coffea.processor as processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

if hasattr(NanoEventsFactory, "warn_missing_crossrefs"):
    NanoEventsFactory.warn_missing_crossrefs = False
elif hasattr(NanoAODSchema, "warn_missing_crossrefs"):
    NanoAODSchema.warn_missing_crossrefs = False
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


def _taskvine_custom_init(logs_dir: Path) -> Callable[[Any], None]:
    logs_dir.mkdir(parents=True, exist_ok=True)

    def _configure(manager: Any) -> None:
        log_map = {
            'set_debug_log': logs_dir / 'debug.log',
            'set_transactions_log': logs_dir / 'transactions.log',
            'set_stats_file': logs_dir / 'stats.log',
            'set_tasks_accumulation_file': logs_dir / 'tasks.log',
        }
        for method_name, log_path in log_map.items():
            setter = getattr(manager, method_name, None)
            if callable(setter):
                try:
                    setter(str(log_path))
                except Exception:
                    warnings.warn(
                        f"TaskVine manager could not enable {method_name} at {log_path}.",
                        RuntimeWarning,
                    )

    return _configure


def _select_manager_port(port_min: int, port_max: int, *, exclude: Optional[Set[int]] = None) -> int:
    if port_min > port_max:
        raise ValueError('Invalid port range: minimum exceeds maximum.')
    excluded = exclude or set()
    for port in range(port_min, port_max + 1):
        if port in excluded:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                candidate.bind(('', port))
            except OSError:
                continue
            return port
    raise RuntimeError(f'No available port found in the requested range {port_min}-{port_max}.')


def _is_port_allocation_error(exc: BaseException) -> bool:
    if isinstance(exc, OSError) and exc.errno == errno.EADDRINUSE:
        return True
    message = str(exc).lower()
    return 'address already in use' in message or ('port' in message and 'in use' in message)

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
port_min, port_max = _parse_port_range(args.port)
environment_setting = _normalize_environment_file(args.environment_file)
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
    executor_args = {"workers": futures_workers}
    if futures_status is not None:
        executor_args["status"] = bool(futures_status)
    if futures_tail_timeout is not None:
        executor_args["tailtimeout"] = int(futures_tail_timeout)
    futures_cls = getattr(processor, "FuturesExecutor", None)
    if futures_cls is None:  # pragma: no cover - depends on coffea build
        futures_cls = getattr(processor, "futures_executor")
    exec_instance = futures_cls(**executor_args)
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

    env_payload: Optional[str] = None
    if environment_setting == 'auto':
        env_payload = remote_environment.get_environment(
            extra_pip_local={"topeft": ["topeft", "setup.py"]},
            extra_conda=["pyyaml"],
        )
    elif environment_setting:
        env_payload = environment_setting

    if executor == 'work_queue':
        work_queue_cls = getattr(processor, 'WorkQueueExecutor', None)
        if work_queue_cls is None:
            raise RuntimeError(
                'WorkQueueExecutor is not available in this Coffea installation. '
                'Use the TaskVine executor or install a Coffea build that provides WorkQueueExecutor.'
            )
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
            'master_name': manager_base,
        }
        if env_payload:
            executor_args['environment_file'] = env_payload
        exec_instance = work_queue_cls(**executor_args)
    else:
        taskvine_cls = getattr(processor, 'TaskVineExecutor', None)
        if taskvine_cls is None:
            raise RuntimeError('TaskVineExecutor not available.')

        executor_args = {
            'manager_name': manager_base,
            'manager_name_template': f"{manager_base}-{{pid}}",
            'filepath': str(staging_dir),
            'extra_input_files': ["sow_processor.py"],
            'retries': 15,
            'compression': 8,
            'resource_monitor': 'measure',
            'resources_mode': 'auto',
            'fast_terminate_workers': 0,
            'verbose': True,
            'print_stdout': False,
            'custom_init': _taskvine_custom_init(logs_dir),
        }
        if env_payload:
            executor_args['environment_file'] = env_payload

        attempted_ports: Set[int] = set()
        last_port_error: Optional[BaseException] = None
        exec_instance = None
        for _ in range(port_max - port_min + 1):
            try:
                port = _select_manager_port(port_min, port_max, exclude=attempted_ports)
            except RuntimeError as exc:
                if last_port_error is not None:
                    range_desc = f"{port_min}-{port_max}" if port_min != port_max else str(port_min)
                    raise RuntimeError(
                        f'TaskVineExecutor could not bind a manager port in range {range_desc}.'
                    ) from last_port_error
                raise

            attempted_ports.add(port)
            executor_args['port'] = port

            while True:
                try:
                    exec_instance = taskvine_cls(**executor_args)
                except TypeError as exc:
                    if 'manager_name_template' in str(exc) and 'manager_name_template' in executor_args:
                        executor_args.pop('manager_name_template', None)
                        continue
                    raise
                except Exception as exc:
                    if not _is_port_allocation_error(exc):
                        raise
                    last_port_error = exc
                    break
                else:
                    break

            if exec_instance is not None:
                break
        else:
            range_desc = f"{port_min}-{port_max}" if port_min != port_max else str(port_min)
            message = f'TaskVineExecutor could not bind a manager port in range {range_desc}.'
            if last_port_error is not None:
                raise RuntimeError(message) from last_port_error
            raise RuntimeError(message)
else:
    raise Exception(f"Executor \"{executor}\" is not known.")

runner_fields = set(getattr(processor.Runner, "__dataclass_fields__", {}))
runner_kwargs: dict[str, Any] = {}
if (
    executor == "futures"
    and futures_memory is not None
    and "dynamic_chunksize" in runner_fields
):
    runner_kwargs["dynamic_chunksize"] = {"memory": int(futures_memory)}

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
