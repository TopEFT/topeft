import argparse
import cloudpickle
import gzip
import json
import os
import time

import coffea.processor as processor
from coffea.nanoevents import NanoAODSchema
import topcoffea.modules.remote_environment as remote_environment

import flip_mr_processor
import flip_ar_processor

from topcoffea.modules.utils import load_sample_json_file, read_cfg_file, update_cfg
from topeft.modules.executor import build_futures_executor, taskvine_log_configurator
from topeft.modules.executor_cli import (
    ExecutorCLIHelper,
    FuturesArgumentSpec,
    TaskVineArgumentSpec,
)
from topeft.modules.runner_output import normalise_runner_output


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
parser.add_argument('--processor_name','-r' , default='flip_mr_processor', help = 'Which processor to run')
parser.add_argument('--chunksize','-s'      , default=100000, type=int, help = 'Number of events per chunk')
parser.add_argument('--max-files','-N'      , default=0, type=int, help = 'If specified, limit the number of root files per sample. Useful for testing')
parser.add_argument('--nchunks','-c'        , default=0, type=int, help = 'You can choose to run only a number of chunks')
parser.add_argument('--outname','-o'        , default='flipTopEFT', help = 'Name of the output file with histograms')
parser.add_argument('--outpath','-p'        , default='histos', help = 'Name of the output directory')
parser.add_argument('--treename'            , default='Events', help = 'Name of the tree inside the files')
parser.add_argument('--xrd'                 , default='', help = 'The XRootD redirector to use when reading directly from json files')
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

args = parser.parse_args()
executor_config = EXECUTOR_CLI.parse_args(args)

inputFiles = args.inputFiles.replace(' ','').split(',')  # Remove whitespace and split by commas
executor   = executor_config.executor
processor_name  = args.processor_name
chunksize  = args.chunksize
nchunks    = args.nchunks if args.nchunks else None
outname    = args.outname
outpath    = args.outpath
treename   = args.treename
xrd        = args.xrd
max_files  = args.max_files
futures_workers = executor_config.futures.workers
futures_status = executor_config.futures.status
futures_tail_timeout = executor_config.futures.tailtimeout


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

# Which processor are we running
if processor_name == "flip_mr_processor":
    processor_instance = flip_mr_processor.AnalysisProcessor(samples_to_process)
    extra_input_files_lst = ["flip_mr_processor.py"]
elif processor_name == "flip_ar_processor":
    processor_instance = flip_ar_processor.AnalysisProcessor(samples_to_process)
    extra_input_files_lst = ["flip_ar_processor.py"]
else:
    raise Exception(f"Error: Unknown processor \"{processor_name}\".")

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
        extra_input_files=extra_input_files_lst,
        custom_init=taskvine_log_configurator(logs_dir),
        logs_dir=logs_dir,
    )
    exec_instance = taskvine_config.instantiate(
        processor,
        taskvine_args,
    )
else:
    raise Exception(f"Executor \"{executor}\" is not known.")

runner = processor.Runner(exec_instance, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks, skipbadfiles=False, xrootdtimeout=180)
output = runner(flist, treename, processor_instance)
serialised_output = normalise_runner_output(output)

dt = time.time() - tstart

# Save the output
if not os.path.isdir(outpath): os.system(f"mkdir -p {outpath}")
out_pkl_file = os.path.join(outpath,outname+".pkl.gz")
print(f"\nSaving output in {out_pkl_file}...")
with gzip.open(out_pkl_file, "wb") as fout:
    cloudpickle.dump(serialised_output, fout)

print("Done!")
