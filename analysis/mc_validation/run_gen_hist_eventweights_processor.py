#!/usr/bin/env python
import argparse
import json
import time
import cloudpickle
import gzip
import os
import importlib

from coffea import processor
from coffea.nanoevents import NanoAODSchema

import topcoffea.modules.remote_environment as remote_environment

LST_OF_KNOWN_EXECUTORS = ["futures","work_queue"]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='You can customize your run')
    parser.add_argument('jsonFiles'        , nargs='?', help = 'Json file(s) containing files and metadata')
    parser.add_argument('--executor','-x'  , default='work_queue', help = 'Which executor to use')
    parser.add_argument('--prefix', '-r'   , nargs='?', default='', help = 'Prefix or redirector to look for the files')
    parser.add_argument('--pretend'        , action='store_true', help = 'Read json files but, not execute the analysis')
    parser.add_argument('--nworkers','-n' , default=8  , help = 'Number of workers')
    parser.add_argument('--chunksize','-s', default=100000  , help = 'Number of events per chunk')
    parser.add_argument('--nchunks','-c'  , default=None  , help = 'You can choose to run only a number of chunks')
    parser.add_argument('--outname','-o'  , default='histos', help = 'Name of the output file with histograms')
    parser.add_argument('--treename'      , default='Events', help = 'Name of the tree inside the files')
    parser.add_argument('--wc-list', action='extend', nargs='+', help = 'Specify a list of Wilson coefficients to use in filling histograms.')
    parser.add_argument('--hist-list', action='extend', nargs='+', help = 'Specify a list of histograms to fill.')
    parser.add_argument('--port', default='9123-9130', help = 'Specify the Work Queue port. An integer PORT or an integer range PORT_MIN-PORT_MAX.')
    parser.add_argument('--processor', '-p', default='gen_hist_eventweights_processor.py', help='Specify processor file name')

    args        = parser.parse_args()
    jsonFiles  = args.jsonFiles
    executor    = args.executor
    prefix      = args.prefix
    pretend     = args.pretend
    nworkers    = int(args.nworkers)
    chunksize   = int(args.chunksize)
    nchunks     = int(args.nchunks) if not args.nchunks is None else args.nchunks
    outname     = args.outname
    treename    = args.treename
    wc_lst = args.wc_list if args.wc_list is not None else []
    proc_file   = args.processor
    proc_name   = args.processor[:-3]

    print("\n running with processor: ", proc_file, '\n')
    print("\n jsonFiles argument: ", jsonFiles, '\n')

    analysis_processor = importlib.import_module(proc_name)

    # Check if we have valid options
    if executor not in LST_OF_KNOWN_EXECUTORS:
        raise Exception(f"The \"{executor}\" executor is not known. Please specify an executor from the known executors ({LST_OF_KNOWN_EXECUTORS}). Exiting.")

    if executor == "work_queue":
        # construct wq port range
        port = list(map(int, args.port.split('-')))
        if len(port) < 1:
            raise ValueError("At least one port value should be specified.")
        if len(port) > 2:
            raise ValueError("More than one port range was specified.")
        if len(port) == 1:
            # convert single values into a range of one element
            port.append(port[0])

    if args.hist_list == ["weights"]:
        hist_lst = ["weights_SM_log" ,
                    "weights_pt0_log",
                    "weights_pt1_log",
                    "weights_pt2_log",
                    "weights_pt3_log",
                    "weights_pt4_log",
                    "weights_pt5_log"]
    elif args.hist_list == ["kinematics"]:
        hist_lst = ["tops_pt", "avg_top_pt", "l0pt", "dr_leps", "ht", "jets_pt", "j0pt", "njets", "mtt", "mll"]
    else:
        hist_lst = args.hist_list

    # Load samples from json and setup the inputs to the processor
    samplesdict = {}
    allInputFiles = []

    nevts_total = 0
    def LoadJsonToSampleName(jsonFile, prefix):
        sampleName = jsonFile if not '/' in jsonFile else jsonFile[jsonFile.rfind('/')+1:]
        if sampleName.endswith('.json'): sampleName = sampleName[:-5]
        with open(jsonFile) as jf:
            samplesdict[sampleName] = json.load(jf)
            samplesdict[sampleName]['redirector'] = prefix
            nevts_total += samplesdict[sname]["nEvents"]

    if isinstance(jsonFiles, str) and ',' in jsonFiles:
        jsonFiles = jsonFiles.replace(' ', '').split(',')
    elif isinstance(jsonFiles, str):
        jsonFiles = [jsonFiles]

    for jsonFile in jsonFiles:
        if os.path.isdir(jsonFile):
            if not jsonFile.endswith('/'): jsonFile+='/'
            for f in os.path.listdir(jsonFile):
                if f.endswith('.json'): allInputFiles.append(jsonFile+f)
        else:
            allInputFiles.append(jsonFile)

    # Read from cfg files
    for f in allInputFiles:
        if not os.path.isfile(f):
            raise Exception(f'[ERROR] Input file {f} not found!')
        # This input file is a json file, not a cfg
        if f.endswith('.json'):
            LoadJsonToSampleName(f, prefix)
        # Open cfg files
        else:
            with open(f) as fin:
                print(' >> Reading json from cfg file...')
                lines = fin.readlines()
                for l in lines:
                    if '#' in l:
                        l=l[:l.find('#')]
                    l = l.replace(' ', '').replace('\n', '')
                    if l == '': continue
                    if ',' in l:
                        l = l.split(',')
                        for nl in l:
                            if not os.path.isfile(l):
                                prefix = nl
                            else:
                                LoadJsonToSampleName(nl, prefix)
                    else:
                        if not os.path.isfile(l):
                            prefix = l
                        else:
                            LoadJsonToSampleName(l, prefix)
                    # else:
                    #     if not l.endswith('.json'):
                    #         prefix = l
                    #     else:
                    #         LoadJsonToSampleName(l, prefix)

    flist = {}
    for sname in samplesdict.keys():
        #flist[sname] = samplesdict[sname]['files']
        redirector = samplesdict[sname]['redirector']
        flist[sname] = [(redirector+f) for f in samplesdict[sname]['files']]

    # Extract the list of all WCs, as long as we haven't already specified one.
    if len(wc_lst) == 0:
        for k in samplesdict.keys():
            for wc in samplesdict[k]['WCnames']:
                if wc not in wc_lst:
                    wc_lst.append(wc)

    if len(wc_lst) > 0:
        # Yes, why not have the output be in correct English?
        if len(wc_lst) == 1:
            wc_print = wc_lst[0]
        elif len(wc_lst) == 2:
            wc_print = wc_lst[0] + ' and ' + wc_lst[1]
        else:
            wc_print = ', '.join(wc_lst[:-1]) + ', and ' + wc_lst[-1]
            print('Wilson Coefficients: {}.'.format(wc_print))

    else:
        print('No Wilson coefficients specified')

    # Run the processor and get the output
    processor_instance = analysis_processor.AnalysisProcessor(samplesdict,wc_lst,hist_lst)

    if executor == "work_queue":
        executor_args = {
            'master_name': '{}-work_queue-coffea'.format(os.environ['USER']),

            # find a port to run work queue in this range:
            'port': port,

            'debug_log': 'debug.log',
            'transactions_log': 'tr.log',
            'stats_log': 'stats.log',
            'tasks_accum_log': 'tasks.log',

            'environment_file': remote_environment.get_environment(
                extra_pip_local = {"topeft": ["topeft", "setup.py"]},
            ),
            #'filepath': f'/project01/ndcms/{os.environ["USER"]}',
            'extra_input_files' : [proc_file],

            'retries': 5,

            # use mid-range compression for chunks results. 9 is the default for work
            # queue in coffea. Valid values are 0 (minimum compression, less memory
            # usage) to 16 (maximum compression, more memory usage).
            'compression': 0,

            "filepath": f'/tmp/{os.environ["USER"]}',
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
            # 'cores': 1,
            # 'disk': 8000,   #MB
            # 'memory': 10000, #MB

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
        exec_instance = processor.FuturesExecutor(workers=nworkers)
        runner = processor.Runner(exec_instance, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks)
    elif executor ==  "work_queue":
        executor = processor.WorkQueueExecutor(**executor_args)
        runner = processor.Runner(executor, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks, skipbadfiles=False, xrootdtimeout=180)

    output = runner(flist, treename, processor_instance)

    dt = time.time() - tstart

    if executor == "work_queue":
        print('Processed {} events in {} seconds ({:.2f} evts/sec).'.format(nevts_total,dt,nevts_total/dt))
    elif executor == "futures":
        print("Processing time: %1.2f s with %i workers (%.2f s cpu overall)" % (dt, nworkers, dt*nworkers, ))


    # Save the output
    outpath = "."
    if not os.path.isdir(outpath): os.system("mkdir -p %s"%outpath)
    out_pkl_file = os.path.join(outpath,outname+".pkl.gz")
    print(f"\nSaving output in {out_pkl_file}...")
    with gzip.open(out_pkl_file, "wb") as fout:
        cloudpickle.dump(output, fout)
    print("Done!")

