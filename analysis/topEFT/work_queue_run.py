#!/usr/bin/env python

import lz4.frame as lz4f
import pickle
import json
import time
import cloudpickle
import gzip
import os
from optparse import OptionParser

import uproot
import numpy as np
from coffea import hist, processor
from coffea.util import load, save
from coffea.nanoevents import NanoAODSchema

import topeft
from topcoffea.modules import samples
from topcoffea.modules import fileReader
from topcoffea.modules.dataDrivenEstimation import DataDrivenProducer
import topeftenv

import argparse
parser = argparse.ArgumentParser(description='You can customize your run')
parser.add_argument('jsonFiles'        , nargs='?', default='', help = 'Json file(s) containing files and metadata')
parser.add_argument('--prefix', '-r'   , nargs='?', default='', help = 'Prefix or redirector to look for the files')
parser.add_argument('--pretend'        , action='store_true', help = 'Read json files but, not execute the analysis')
parser.add_argument('--chunksize','-s' , default=100000, help = 'Number of events per chunk')
parser.add_argument('--nchunks','-c'   , default=None, help = 'You can choose to run only a number of chunks')
parser.add_argument('--outname','-o'   , default='plotsTopEFT', help = 'Name of the output file with histograms')
parser.add_argument('--outpath','-p'   , default='histos', help = 'Name of the output directory')
parser.add_argument('--treename'       , default='Events', help = 'Name of the tree inside the files')
parser.add_argument('--do-errors'      , action='store_true', help = 'Save the w**2 coefficients')
parser.add_argument('--do-systs', action='store_true', help = 'Run over systematic samples (takes longer)')
parser.add_argument('--split-lep-flavor', action='store_true', help = 'Split up categories by lepton flavor')
parser.add_argument('--skip-sr', action='store_true', help = 'Skip all signal region categories')
parser.add_argument('--skip-cr', action='store_true', help = 'Skip all control region categories')
parser.add_argument('--do-np'  , action='store_true', help = 'Perform nonprompt estimation on the output hist, and save a new hist with the np contribution included. Note that signal, background and data samples should all be processed together in order for this option to make sense.')
parser.add_argument('--wc-list', action='extend', nargs='+', help = 'Specify a list of Wilson coefficients to use in filling histograms.')
parser.add_argument('--hist-list', action='extend', nargs='+', help = 'Specify a list of histograms to fill.')
parser.add_argument('--port', default='9123-9130', help = 'Specify the Work Queue port. An integer PORT or an integer range PORT_MIN-PORT_MAX.')

args = parser.parse_args()
jsonFiles  = args.jsonFiles
prefix     = args.prefix
chunksize  = int(args.chunksize)
nchunks    = int(args.nchunks) if not args.nchunks is None else args.nchunks
outname    = args.outname
outpath    = args.outpath
pretend    = args.pretend
treename   = args.treename
do_errors  = args.do_errors
do_systs   = args.do_systs
split_lep_flavor = args.split_lep_flavor
skip_sr    = args.skip_sr
skip_cr    = args.skip_cr
do_np      = args.do_np
port       = list(map(int, args.port.split('-')))
wc_lst = args.wc_list if args.wc_list is not None else []

# Figure out which hists to include
if args.hist_list == ["ana"]:
  # Here we hardcode a list of hists used for the analysis
  hist_lst = ["njets","ht","ptbl","ptz"]
else:
  # We want to specify a custom list
  # If we don't specify this argument, it will be None, and the processor will fill all hists 
  hist_lst = args.hist_list

### Load samples from json
samplesdict = {}
allInputFiles = []

def LoadJsonToSampleName(jsonFile, prefix):
 sampleName = jsonFile if not '/' in jsonFile else jsonFile[jsonFile.rfind('/')+1:]
 if sampleName.endswith('.json'): sampleName = sampleName[:-5]
 with open(jsonFile) as jf:
   samplesdict[sampleName] = json.load(jf)
   samplesdict[sampleName]['redirector'] = prefix

if   isinstance(jsonFiles, str) and ',' in jsonFiles: jsonFiles = jsonFiles.replace(' ', '').split(',')
elif isinstance(jsonFiles, str)                     : jsonFiles = [jsonFiles]
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
        if '#' in l: l=l[:l.find('#')]
        l = l.replace(' ', '').replace('\n', '')
        if l == '': continue
        if ',' in l:
          l = l.split(',')
          for nl in l:
            if not os.path.isfile(l): prefix = nl
            else: LoadJsonToSampleName(nl, prefix)
        else:
          if not os.path.isfile(l): prefix = l
          else: LoadJsonToSampleName(l, prefix)

flist = {};
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

if pretend:
  print('pretending...')
  exit()

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

processor_instance = topeft.AnalysisProcessor(samplesdict,wc_lst,hist_lst,do_errors,do_systs,split_lep_flavor,skip_sr,skip_cr)

executor_args = {
    'master_name': '{}-workqueue-coffea'.format(os.environ['USER']),

    'xrootdtimeout': 180,

    # find a port to run work queue in this range:
    'port': port,

    'debug_log': 'debug.log',
    'transactions_log': 'tr.log',
    'stats_log': 'stats.log',

    'environment_file': topeftenv.get_environment(),
    'extra_input_files': ["topeft.py"],

    'schema': NanoAODSchema,
    'skipbadfiles': False,

    'retries': 5,

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
    'cores': 1,
    'disk': 8000,   #MB
    'memory': 10000, #MB

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
output = processor.run_uproot_job(flist, treename=treename, processor_instance=processor_instance, executor=processor.work_queue_executor, executor_args=executor_args, chunksize=chunksize, maxchunks=nchunks)
dt = time.time() - tstart

print('Processed {} events in {} seconds ({:.2f} evts/sec).'.format(nevts_total,dt,nevts_total/dt))

nbins = sum(sum(arr.size for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
nfilled = sum(sum(np.sum(arr > 0) for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
print("Filled %.0f bins, nonzero bins: %1.1f %%" % (nbins, 100*nfilled/nbins,))

# Save the output
if not os.path.isdir(outpath): os.system("mkdir -p %s"%outpath)
out_pkl_file = os.path.join(outpath,outname+".pkl.gz")
print(f"\nSaving output in {out_pkl_file}...")
with gzip.open(out_pkl_file, "wb") as fout:
  cloudpickle.dump(output, fout)
print("Done!")

# Run the data driven estimation, save the output
if do_np:
  print("\nDoing the nonprompt estimation...")
  out_pkl_file_np = os.path.join(outpath,outname+"_np.pkl.gz")
  ddp = DataDrivenProducer(out_pkl_file,out_pkl_file_np)
  print(f"Saving output in {out_pkl_file_np}...")
  ddp.dumpToPickle()
  print("Done!")
