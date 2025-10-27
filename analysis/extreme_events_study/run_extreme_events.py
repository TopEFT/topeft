import argparse
import cloudpickle
import gzip
import json
import os
import tempfile
import time
from pathlib import Path

import coffea.processor as processor
from coffea.nanoevents import NanoAODSchema
import topcoffea.modules.remote_environment as remote_environment

import extreme_events

from topcoffea.modules.utils import load_sample_json_file, read_cfg_file, update_cfg
from topeft.modules.executor import (
    build_futures_executor,
    build_taskvine_args,
    build_work_queue_args,
    distributed_logs_dir,
    instantiate_taskvine_executor,
    instantiate_work_queue_executor,
    manager_name_base,
    parse_port_range,
    resolve_environment_file,
    taskvine_log_configurator,
)


def _staging_directory(executor: str, scratch_dir: str | None) -> Path:
    if scratch_dir:
        staging = Path(scratch_dir).expanduser()
    else:
        base_dir = os.environ.get("TOPEFT_EXECUTOR_STAGING")
        if base_dir:
            staging = Path(base_dir).expanduser()
        else:
            staging = Path(tempfile.gettempdir()) / "topeft" / manager_name_base(executor)
    staging.mkdir(parents=True, exist_ok=True)
    return staging


parser = argparse.ArgumentParser(
    description='You can customize your run',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=(
        "TaskVine workers can be launched with:\n"
        "  vine_submit_workers --python-env \"$(python -m topcoffea.modules.remote_environment)\" \\\n"
        "    --cores 4 --memory 16000 --disk 16000 -M <manager-name>\n"
        "Adjust the resources and manager to match your deployment."
    ),
)
parser.add_argument('inputFiles'            , nargs='?', default='', help = 'Json or cfg file(s) containing files and metadata')
parser.add_argument('--executor','-x'       , default='taskvine', help = 'Which executor to use (futures, work_queue, or taskvine)')
parser.add_argument('--chunksize','-s'      , default=100000, type=int, help = 'Number of events per chunk')
parser.add_argument('--max-files','-N'      , default=0, type=int, help = 'If specified, limit the number of root files per sample. Useful for testing')
parser.add_argument('--nchunks','-c'        , default=0, type=int, help = 'You can choose to run only a number of chunks')
parser.add_argument('--outname','-o'        , default='flipTopEFT', help = 'Name of the output file with histograms')
parser.add_argument('--outpath','-p'        , default='histos', help = 'Name of the output directory')
parser.add_argument('--treename'            , default='Events', help = 'Name of the tree inside the files')
parser.add_argument('--xrd'                 , default='', help = 'The XRootD redirector to use when reading directly from json files')
parser.add_argument('--port'                , default='9123-9130', help = 'TaskVine/Work Queue manager port (PORT or PORT_MIN-PORT_MAX).')
parser.add_argument('--environment-file'    , default='auto', help = "Environment tarball for distributed executors ('auto', 'none', or path).")
parser.add_argument('--no-environment-file' , dest='environment_file', action='store_const', const='none', help = 'Disable environment shipping (equivalent to --environment-file=none).')
parser.add_argument('--manager-name'        , default=None, help = "Override the distributed executor manager identifier.")
parser.add_argument('--manager-name-template', default=None, help = "Template for TaskVine manager names (use '{pid}' for the process id).")
parser.add_argument('--scratch-dir'         , default=None, help = 'Shared staging directory for distributed executors.')
parser.add_argument('--resource-monitor'    , default='measure', help = "TaskVine/Work Queue resource monitor setting ('none' to disable).")
parser.add_argument('--resources-mode'      , default='auto', help = 'TaskVine/Work Queue resources mode (for example auto).')
parser.add_argument('--futures-workers'     , default=8, type=int, help = 'Maximum number of local processes for the futures executor.')
parser.add_argument('--futures-status'      , action=argparse.BooleanOptionalAction, default=None, help = 'Toggle the coffea futures progress bar.')
parser.add_argument('--futures-tail-timeout', default=None, type=int, help = 'Timeout in seconds for cancelling stalled futures tasks.')

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
port_min, port_max = parse_port_range(args.port)

environment_file = resolve_environment_file(
    args.environment_file,
    remote_environment,
    extra_pip_local={"topeft": ["topeft", "setup.py"]},
    extra_conda=["pyyaml"],
)

manager_name = args.manager_name or manager_name_base(executor)
manager_template = args.manager_name_template
if manager_template is None and manager_name:
    manager_template = f"{manager_name}-{{pid}}"
scratch_dir = args.scratch_dir

resource_monitor = args.resource_monitor
if resource_monitor:
    rm_normalized = resource_monitor.strip().lower()
    if rm_normalized in {"none", "off", "false", "0"}:
        resource_monitor = None

resources_mode = args.resources_mode
if resources_mode:
    mode_normalized = resources_mode.strip().lower()
    if mode_normalized in {"none", "off", "false", "0"}:
        resources_mode = None

futures_workers = max(int(args.futures_workers or 1), 1)
futures_status = args.futures_status
futures_tail_timeout = None
if args.futures_tail_timeout and args.futures_tail_timeout > 0:
    futures_tail_timeout = int(args.futures_tail_timeout)


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
for sample_name,jsn in samples_to_process.items():
    #if jsn['WCnames'] != []: raise Exception(f"Error: This processor is not set up to handle EFT samples.")
    xrd_src = jsn['redirector']
    flist[sample_name] = [f"{xrd_src}{fn}" for fn in jsn['files']]

    jsn_txt = json.dumps(jsn,indent=2,sort_keys=True)

    # Strips off the leading and closing curly brackets, along with any associated newlines
    jsn_txt = jsn_txt[1:-1].strip("\n",)

    s = ""
    s += f">> {sample_name}\n"
    s += f"{jsn_txt}\n"
    print(s)

processor_instance = extreme_events.AnalysisProcessor(samples_to_process)

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
    staging_dir = _staging_directory(executor, scratch_dir)
    logs_dir = distributed_logs_dir(staging_dir, executor)

    if executor == "work_queue":
        executor_args = build_work_queue_args(
            staging_dir=staging_dir,
            logs_dir=logs_dir,
            manager_name=manager_name,
            port_range=(port_min, port_max),
            extra_input_files=['extreme_events.py'],
            resource_monitor=resource_monitor,
            resources_mode=resources_mode,
            environment_file=environment_file,
        )
        exec_instance = instantiate_work_queue_executor(processor, executor_args)
    else:
        taskvine_args = build_taskvine_args(
            staging_dir=staging_dir,
            logs_dir=logs_dir,
            manager_name=manager_name,
            manager_name_template=manager_template,
            extra_input_files=['extreme_events.py'],
            resource_monitor=resource_monitor,
            resources_mode=resources_mode,
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

runner = processor.Runner(exec_instance, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks, skipbadfiles=False, xrootdtimeout=180)
output = runner(flist, treename, processor_instance)

dt = time.time() - tstart

# Save the output
if not os.path.isdir(outpath): os.system(f"mkdir -p {outpath}")
out_pkl_file = os.path.join(outpath,outname+".pkl.gz")
print(f"\nSaving output in {out_pkl_file}...")
with gzip.open(out_pkl_file, "wb") as fout:
    cloudpickle.dump(output, fout)

print("Done!")
