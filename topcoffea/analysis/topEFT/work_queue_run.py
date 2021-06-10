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
parser.add_argument('jsonFiles'        , nargs='?', default='', help = 'Json file(s) containing files and metadata')
parser.add_argument('--prefix', '-r'   , nargs='?', default='', help = 'Prefix or redirector to look for the files')
parser.add_argument('--pretend'        , action='store_true', help = 'Read json files but, not execute the analysis')
parser.add_argument('--chunksize','-s' , default=100000, help = 'Number of events per chunk')
parser.add_argument('--nchunks','-c'   , default=None, help = 'You can choose to run only a number of chunks')
parser.add_argument('--outname','-o'   , default='plotsTopEFT', help = 'Name of the output file with histograms')
parser.add_argument('--outpath','-p'   , default='histos', help = 'Name of the output directory')
parser.add_argument('--treename'       , default='Events', help = 'Name of the tree inside the files')
parser.add_argument('--do-errors'      , action='store_true', help = 'Save the w**2 coefficients')

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
    print('[WARNING] Input file "%s% not found!'%f)
    continue
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
for sname in samplesdict.keys():
  redirector = samplesdict[sname]['redirector']
  flist[sname] = [(redirector+f) for f in samplesdict[sname]['files']]
  samplesdict[sname]['year'] = int(samplesdict[sname]['year'])
  samplesdict[sname]['xsec'] = float(samplesdict[sname]['xsec'])
  samplesdict[sname]['nEvents'] = int(samplesdict[sname]['nEvents'])
  samplesdict[sname]['nGenEvents'] = int(samplesdict[sname]['nGenEvents'])
  samplesdict[sname]['nSumOfWeights'] = float(samplesdict[sname]['nSumOfWeights'])

  # Print file info
  print('>> '+sname)
  print('   - isData?      : %s'   %('YES' if samplesdict[sname]['isData'] else 'NO'))
  print('   - year         : %i'   %samplesdict[sname]['year'])
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
dt = time.time() - tstart

nbins = sum(sum(arr.size for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
nfilled = sum(sum(np.sum(arr > 0) for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
print("Filled %.0f bins, nonzero bins: %1.1f %%" % (nbins, 100*nfilled/nbins,))

# This is taken from the DM photon analysis...                                                                                                                                                             
# Pickle is not very fast or memory efficient, will be replaced by something better soon                                                                                                                   
#    with lz4f.open("pods/"+options.year+"/"+dataset+".pkl.gz", mode="xb", compression_level=5) as fout:                                                                                                   
if not outpath.endswith('/'): outpath += '/'
if not os.path.isdir(outpath): os.system("mkdir -p %s"%outpath)
print('Saving output in %s...'%(outpath + outname + ".pkl.gz"))
with gzip.open(outpath + outname + ".pkl.gz", "wb") as fout:
  cloudpickle.dump(output, fout)
print('Done!')

