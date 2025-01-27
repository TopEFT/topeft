#!/usr/bin/env python
import json
import time
import cloudpickle
import gzip
import os

import numpy as np
from coffea import hist, processor
from coffea.nanoevents import NanoAODSchema

import topcoffea.modules.remote_environment as remote_environment

import btagMCeff

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='You can customize your run')
    parser.add_argument('jsonFiles'        , nargs='?', default='', help = 'Json file(s) containing files and metadata')
    parser.add_argument('--prefix', '-r'   , nargs='?', default='', help = 'Prefix or redirector to look for the files')
    parser.add_argument('--executor','-x'  , default='work_queue', help = 'Which executor to use')
    parser.add_argument('--test','-t'       , action='store_true'  , help = 'To perform a test, run over a few events in a couple of chunks')
    parser.add_argument('--nworkers','-n'   , default=8  , help = 'Number of workers')
    parser.add_argument('--chunksize','-s'   , default=100000  , help = 'Number of events per chunk')
    parser.add_argument('--nchunks','-c'   , default=None  , help = 'You can choose to run only a number of chunks')
    parser.add_argument('--outname','-o'   , default='btagMCeff', help = 'Name of the output file with histograms')
    parser.add_argument('--outpath','-p'   , default='histos', help = 'Name of the output directory')
    parser.add_argument('--treename'   , default='Events', help = 'Name of the tree inside the files')
    parser.add_argument('--port', default='9123-9130', help = 'Specify the Work Queue port. An integer PORT or an integer range PORT_MIN-PORT_MAX.')


    args = parser.parse_args()
    jsonFiles  = args.jsonFiles
    prefix     = args.prefix
    executor   = args.executor
    dotest     = args.test
    nworkers   = int(args.nworkers)
    chunksize  = int(args.chunksize)
    nchunks    = int(args.nchunks) if not args.nchunks is None else args.nchunks
    outname    = args.outname
    outpath    = args.outpath
    treename   = args.treename

    if dotest:
        if executor == "futures":
            nchunks = 2
            chunksize = 10000
            nworkers = 1
            print('Running a fast test with %i workers, %i chunks of %i events'%(nworkers, nchunks, chunksize))
        else:
            raise Exception(f"The \"test\" option is not set up to work with the {executor} executor. Exiting.")


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

    ### Load samples from json
    samplesdict = {}
    allInputFiles = []

    def LoadJsonToSampleName(jsonFile, prefix):
        sampleName = jsonFile if not '/' in jsonFile else jsonFile[jsonFile.rfind('/')+1:]
        if sampleName.endswith('.json'): sampleName = sampleName[:-5]
        with open(jsonFile) as jf:
            samplesdict[sampleName] = json.load(jf)
            samplesdict[sampleName]['redirector'] = prefix

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


    flist = {}
    nevts_total = 0
    for sname in samplesdict.keys():
        redirector = samplesdict[sname]['redirector']
        flist[sname] = [(redirector+f) for f in samplesdict[sname]['files']]
        samplesdict[sname]['year'] = samplesdict[sname]['year']
        samplesdict[sname]['xsec'] = float(samplesdict[sname]['xsec'])
        samplesdict[sname]['nEvents'] = int(samplesdict[sname]['nEvents'])
        nevts_total += samplesdict[sname]['nEvents']
        samplesdict[sname]['nGenEvents'] = int(samplesdict[sname]['nGenEvents'])
        samplesdict[sname]['nSumOfWeights'] = float(samplesdict[sname]['nSumOfWeights'])
        # Print file info
        print('>> '+sname)
        print('   - isData?      : %s'   %('YES' if samplesdict[sname]['isData'] else 'NO'))
        print('   - year         : %s'   %samplesdict[sname]['year'])
        print('   - xsec         : %f'   %samplesdict[sname]['xsec'])
        print('   - histAxisName : %s'   %samplesdict[sname]['histAxisName'])
        print('   - options      : %s'   %samplesdict[sname]['options'])
        print('   - tree         : %s'   %samplesdict[sname]['treeName'])
        print('   - nEvents      : %i'   %samplesdict[sname]['nEvents'])
        print('   - nGenEvents   : %i'   %samplesdict[sname]['nGenEvents'])
        print('   - SumWeights   : %i'   %samplesdict[sname]['nSumOfWeights'])
        print('   - Prefix       : %s'   %samplesdict[sname]['redirector'])
        print('   - nFiles       : %i'   %len(samplesdict[sname]['files']))
        for fname in samplesdict[sname]['files']: print('     %s'%fname)

    processor_instance = btagMCeff.AnalysisProcessor(samplesdict)

    if executor == "work_queue":
        executor_args = {
            'master_name': '{}-workqueue-coffea'.format(os.environ['USER']),

            # find a port to run work queue in this range:
            'port': port,

            'debug_log': 'debug.log',
            'transactions_log': 'tr.log',
            'stats_log': 'stats.log',
            'tasks_accum_log': 'tasks.log',

            'environment_file': remote_environment.get_environment(
                extra_pip_local = {"topeft": ["topeft", "setup.py"]},
            ),
            'extra_input_files': ["analysis_processor.py"],

            'retries': 20,
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
            #'filepath': f'/tmp/{os.environ["USER"]}', ##Placeholder to comment out if you don't want to save wq-factory dirs in afs

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
            # as most tasks will be terminated.            #
            # Less than 1 disables it.
            'fast_terminate_workers': 0,

            # print messages when tasks are submitted, finished, etc.,
            # together with their resource allocation and usage. If a task
            # fails, its standard output is also printed, so we can turn
            # off print_stdout for all tasks.
            'verbose': True,
            'print_stdout': False,
        }


    if executor == "futures":
        executor = processor.futures_executor(workers=nworkers)
        runner = processor.Runner(executor, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks)

    elif executor == "work_queue":
        executor = processor.WorkQueueExecutor(**executor_args)
        runner = processor.Runner(executor, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks, skipbadfiles=False, xrootdtimeout=180)


    tstart = time.time()
    output = runner(flist, treename, processor_instance)
    dt = time.time() - tstart

    nbins = sum(sum(arr.size for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
    nfilled = sum(sum(np.sum(arr > 0) for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
    #print("Filled %.0f bins, nonzero bins: %1.1f %%" % (nbins, 100*nfilled/nbins,))
    print("Processing time: %1.2f s with %i workers (%.2f s cpu overall)" % (dt, nworkers, dt*nworkers, ))

    # This is taken from the DM photon analysis...
    # Pickle is not very fast or memory efficient, will be replaced by something better soon
    #    with lz4f.open("pods/"+options.year+"/"+dataset+".pkl.gz", mode="xb", compression_level=5) as fout:
    if not outpath.endswith('/'): outpath += '/'
    if not os.path.isdir(outpath): os.system("mkdir -p %s"%outpath)
    print('Saving output in %s...'%(outpath + outname + ".pkl.gz"))
    with gzip.open(outpath + outname + ".pkl.gz", "wb") as fout:
        cloudpickle.dump(output, fout)
    print('Done!')




