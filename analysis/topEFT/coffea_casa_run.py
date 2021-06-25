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

if __name__ == '__main__':

  import argparse
  parser = argparse.ArgumentParser(description='You can customize your run')
  parser.add_argument('jsonFiles'           , nargs='?', default=''           , help = 'Json file(s) containing files and metadata')
  parser.add_argument('--prefix', '-r'     , nargs='?', default=''           , help = 'Prefix or redirector to look for the files')
  parser.add_argument('--test','-t'       , action='store_true'  , help = 'To perform a test, run over a few events in a couple of chunks')
  parser.add_argument('--pretend'        , action='store_true'  , help = 'Read json files but, not execute the analysis')
  parser.add_argument('--nworkers','-n'   , default=8  , help = 'Number of workers')
  parser.add_argument('--chunksize','-s'   , default=100000  , help = 'Number of events per chunk')
  parser.add_argument('--nchunks','-c'   , default=None  , help = 'You can choose to run only a number of chunks')
  parser.add_argument('--outname','-o'   , default='plotsTopEFT', help = 'Name of the output file with histograms')
  parser.add_argument('--outpath','-p'   , default='histos', help = 'Name of the output directory')
  parser.add_argument('--treename'   , default='Events', help = 'Name of the tree inside the files')
  parser.add_argument('--do-errors', action='store_true', help = 'Save the w**2 coefficients')
  parser.add_argument('--do-systs', action='store_true', help = 'Run over systematic samples (takes longer)')

  args = parser.parse_args()
  jsonFiles  = args.jsonFiles
  prefix     = args.prefix
  dotest     = args.test
  nworkers   = int(args.nworkers)
  chunksize  = int(args.chunksize)
  nchunks    = int(args.nchunks) if not args.nchunks is None else args.nchunks
  outname    = args.outname
  outpath    = args.outpath
  pretend    = args.pretend
  treename   = args.treename
  do_errors = args.do_errors
  do_systs  = args.do_systs

  if dotest:
    nchunks = 2
    chunksize = 10000
    nworkers = 1
    print('Running a fast test with %i workers, %i chunks of %i events'%(nworkers, nchunks, chunksize))

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
    print('   - SumWeights   : %f'   %samplesdict[sname]['nSumOfWeights'])
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
 
  processor_instance = topeft.AnalysisProcessor(samplesdict,wc_lst,do_errors,do_systs)

  # Run the processor and get the output
  tstart = time.time()
  processor_instance = topcoffea.analysis.topEFT.topeft.AnalysisProcessor(samplesdict,wc_lst,do_errors)

  from dask.distributed import Client, Worker, WorkerPlugin
  import os, shutil
  from typing import List
  class DependencyInstaller(WorkerPlugin):
      def __init__(self, dependencies: List[str]):
          self._depencendies = " ".join(f"'{dep}'" for dep in dependencies)
      def setup(self, worker: Worker):
          os.system(f"pip install {self._depencendies}")
  dependency_installer = DependencyInstaller([
      "git+https://github.com/oshadura/topcoffea.git@coffea-casa-analysis","awkward==1.3.0"
  ])

  client = Client("tls://localhost:8786")
  client.register_worker_plugin(dependency_installer)

  shutil.make_archive("analysis", "zip", base_dir="analysis")
  client.upload_file("analysis.zip")

  executor_args = {
                  'schema': NanoAODSchema,
                  'client': client,
                  'savemetrics': True
  }

  # Run the processor and get the output                                                                                                                                                                     
  tic = time.time()
  output = processor.run_uproot_job(flist,
                                    treename=treename,
                                    processor_instance=processor_instance,
                                    executor=processor.dask_executor,
                                    executor_args=executor_args,
                                    chunksize=chunksize,
                                    maxchunks=nchunks
                                  )
  toc = time.time()

  print("Dask client:", client)
  print("Total time: %.0f" % (toc - tic))
  print("Events / s / thread: {:,.0f}".format(output[1]['entries'] / output[1]['processtime']))
  print("Events / s: {:,.0f}".format(output[1]['entries'] / (toc - tic)))

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
