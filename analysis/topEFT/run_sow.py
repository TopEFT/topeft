import json
import time
import cloudpickle
import gzip
import os
import argparse

# import uproot
import numpy as np
from coffea import processor
from coffea.nanoevents import NanoAODSchema
import topcoffea.modules.remote_environment as remote_environment

import sow_processor

from topcoffea.modules.utils import load_sample_json_file, read_cfg_file, update_cfg

parser = argparse.ArgumentParser(description='You can customize your run')
parser.add_argument('inputFiles'       , nargs='?', default='', help = 'Json or cfg file(s) containing files and metadata')
parser.add_argument('--executor','-x'  , default='work_queue', help = 'Which executor to use')
parser.add_argument('--chunksize','-s' , default=100000, type=int, help = 'Number of events per chunk')
parser.add_argument('--max-files','-N' , default=0, type=int, help = 'If specified, limit the number of root files per sample. Useful for testing')
parser.add_argument('--nchunks','-c'   , default=0, type=int, help = 'You can choose to run only a number of chunks')
parser.add_argument('--outname','-o'   , default='sowTopEFT', help = 'Name of the output file with histograms')
parser.add_argument('--outpath','-p'   , default='histos', help = 'Name of the output directory')
parser.add_argument('--treename'       , default='Events', help = 'Name of the tree inside the files')
parser.add_argument('--xrd'            , default='', help = 'The XRootD redirector to use when reading directly from json files')
parser.add_argument('--wc-list'        , action='extend', nargs='+', help = 'Specify a list of Wilson coefficients to use in filling histograms.')

args = parser.parse_args()
inputFiles = args.inputFiles.replace(' ','').split(',')  # Remove whitespace and split by commas
executor   = args.executor
chunksize  = args.chunksize
nchunks    = args.nchunks if args.nchunks else None
outname    = args.outname
outpath    = args.outpath
treename   = args.treename
xrd        = args.xrd
max_files  = args.max_files
wc_lst     = args.wc_list if args.wc_list is not None else []

samples_to_process = {}
for fn in inputFiles:
    if fn.endswith('.json'):
        sample = os.path.basename(fn).replace('.json','')
        jsn = load_sample_json_file(fn)
        samples_to_process = update_cfg(jsn,
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

if executor == "work_queue":
    executor_args = {
        'master_name': '{}-workqueue-coffea'.format(os.environ['USER']),

        # find a port to run work queue in this range:
        'port': [9123,9130],

        'debug_log': 'debug.log',
        'transactions_log': 'tr.log',
        'stats_log': 'stats.log',
        'tasks_accum_log': 'tasks.log',

        'environment_file': remote_environment.get_environment(),
        'extra_input_files': ["sow_processor.py"],

        # use mid-range compression for chunks results. 9 is the default for work
        # queue in coffea. Valid values are 0 (minimum compression, less memory
        # usage) to 16 (maximum compression, more memory usage).
        'compression': 9,

        # automatically find an adequate resource allocation for tasks.
        # tasks are first tried using the maximum resources seen of previously ran
        # tasks. on resource exhaustion, they are retried with the maximum resource
        # values, if specified below. if a maximum is not specified, the task waits
        # forever until a larger worker connects.
        'resource_monitor': True,
        'resources_mode': 'auto',

        # this resource values may be omitted when using
        # resources_mode: 'auto', but they do make the initial portion
        # of a workflow run a little bit faster.
        # Rather than using whole workers in the exploratory mode of
        # resources_mode: auto, tasks are forever limited to a maximum
        # of 8GB of mem and disk.
        #
        # NOTE: The very first tasks in the exploratory
        # mode will use the values specified here, so workers need to be at least
        # this large. If left unspecified, tasks will use whole workers in the
        # exploratory mode.
        #'cores': 1,
        #'disk': 8000,   #MB
        #'memory': 10000, #MB

        # control the size of accumulation tasks. Results are
        # accumulated in groups of size chunks_per_accum, keeping at
        # most chunks_per_accum at the same time in memory per task.
        'chunks_per_accum': 25,
        'chunks_accum_in_mem': 2,

        # terminate workers on which tasks have been running longer than average.
        # This is useful for temporary conditions on worker nodes where a task will
        # be finish faster is ran in another worker.
        # the time limit is computed by multipliying the average runtime of tasks
        # by the value of 'fast_terminate_workers'.  Since some tasks can be
        # legitimately slow, no task can trigger the termination of workers twice.
        #
        # warning: small values (e.g. close to 1) may cause the workflow to misbehave,
        # as most tasks will be terminated.
        #
        # Less than 1 disables it.
        'fast_terminate_workers': 0,

        # print messages when tasks are submitted, finished, etc.,
        # together with their resource allocation and usage. If a task
        # fails, its standard output is also printed, so we can turn
        # off print_stdout for all tasks.
        'verbose': True,
        'print_stdout': False,
    }

# Run the processor and get the output
tstart = time.time()

if executor == "futures":
    exec_instance = processor.FuturesExecutor(workers=8)
elif executor ==  "work_queue":
    exec_instance = processor.WorkQueueExecutor(**executor_args)
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
