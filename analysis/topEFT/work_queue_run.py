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
import topeftenv

import argparse
parser = argparse.ArgumentParser(description='You can customize your run')
parser.add_argument('cfgfile'          , nargs='?', default=''           , help = 'Config file with dataset names')
parser.add_argument('--test','-t'      , action='store_true'  , help = 'To perform a test, run over a few events in a couple of chunks')
parser.add_argument('--nworkers','-n'  , default=8  , help = 'Number of workers')
parser.add_argument('--chunksize','-s' , default=100000  , help = 'Number of events per chunk')
parser.add_argument('--nchunks','-c'   , default=None  , help = 'You can choose to run only a number of chunks')
parser.add_argument('--outname','-o'   , default='plotsTopEFT', help = 'Name of the output file with histograms')
parser.add_argument('--outpath','-p'   , default='histos', help = 'Name of the output directory')
parser.add_argument('--treename'       , default='Events', help = 'Name of the tree inside the files')
parser.add_argument('--do-errors'      , action='store_true', help = 'Save the w**2 coefficients')

args = parser.parse_args()
cfgfile    = args.cfgfile
dotest     = args.test
nworkers   = int(args.nworkers)
chunksize  = int(args.chunksize)
nchunks    = int(args.nchunks) if not args.nchunks is None else args.nchunks
outname    = args.outname
outpath    = args.outpath
treename   = args.treename
do_errors  = args.do_errors

if dotest:
  nchunks = 2
  chunksize = 10000
  nworkers = 1
  print('Running a fast test with %i workers, %i chunks of %i events'%(nworkers, nchunks, chunksize))

### Load samples                                                                                                                                                                                           
if cfgfile != '':
  samplesdict = samples.main()
elif os.path.isfile('.samples.coffea'):
  print('Using samples form .samples.coffea')
  samplesdict = load('.samples.coffea')
else:
  print('Execute as [path]/run.py [path]/samples.cfg')
  exit()
  
flist = {}; xsec = {}; sow = {}; isData = {}
for k in samplesdict.keys():
  samplesdict[k]['WCnames'] = fileReader.GetListOfWCs(samplesdict[k]['files'][0])
  flist[k] = samplesdict[k]['files']
  xsec[k]  = samplesdict[k]['xsec']
  sow[k]   = samplesdict[k]['nSumOfWeights']
  isData[k]= samplesdict[k]['isData']

# Check that all datasets have the same list of WCs
for i,k in enumerate(samplesdict.keys()):
  if i == 0:
    wc_lst = samplesdict[k]['WCnames']
  if wc_lst != samplesdict[k]['WCnames']:
    raise Exception("Not all of the datasets have the same list of WCs.")
    
processor_instance = topeft.AnalysisProcessor(samplesdict,wc_lst,do_errors)

executor_args = {#'flatten': True, #used for all executors
                 'compression': 0, #used for all executors
                 'cores': 2,
                 'disk': 5000, #MB
                 'memory': 10000, #MB
                 'resource-monitor': True,
                 'debug-log': 'debug.log',
                 'transactions-log': 'tr.log',
                 'stats-log': 'stats.log',
                 'verbose': False,
                 'port': [9123,9130],
                 'environment-file': topeftenv.get_environment(),
                 'master-name': '{}-workqueue-coffea'.format(os.environ['USER']),
                 'print-stdout': True,
                 'skipbadfiles': False,
                 'schema': NanoAODSchema,
                 'extra-input-files': ["topeft.py"]
}

# Run the processor and get the output                                                                                                                                                                     
tstart = time.time()
output = processor.run_uproot_job(flist, treename=treename, processor_instance=processor_instance, executor=processor.work_queue_executor, executor_args=executor_args, chunksize=chunksize, maxchunks=nchunks)
#output = processor.run_uproot_job(flist, treename=treename, processor_instance=processor_instance, executor=processor.work_queue_executor, executor_args=executor_args, chunksize=chunksize, maxchunks=nchunks, extra-input-files=["topeft.py"])
dt = time.time() - tstart

nbins = sum(sum(arr.size for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
nfilled = sum(sum(np.sum(arr > 0) for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
print("Filled %.0f bins, nonzero bins: %1.1f %%" % (nbins, 100*nfilled/nbins,))
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

