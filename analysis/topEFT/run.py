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

WGT_VAR_LST = [
    "nSumOfWeights_ISRUp",
    "nSumOfWeights_ISRDown",
    "nSumOfWeights_FSRUp",
    "nSumOfWeights_FSRDown",
    "nSumOfWeights_renormUp",
    "nSumOfWeights_renormDown",
    "nSumOfWeights_factUp",
    "nSumOfWeights_factDown",
    "nSumOfWeights_renormfactUp",
    "nSumOfWeights_renormfactDown",
]

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
  parser.add_argument('--split-lep-flavor', action='store_true', help = 'Split up categories by lepton flavor')
  parser.add_argument('--skip-sr', action='store_true', help = 'Skip all signal region categories')
  parser.add_argument('--skip-cr', action='store_true', help = 'Skip all control region categories')
  parser.add_argument('--do-np', action='store_true', help = 'Perform nonprompt estimation on the output hist, and save a new hist with the np contribution included. Note that signal, background and data samples should all be processed together in order for this option to make sense.')
  parser.add_argument('--wc-list', action='extend', nargs='+', help = 'Specify a list of Wilson coefficients to use in filling histograms.')
  parser.add_argument('--hist-list', action='extend', nargs='+', help = 'Specify a list of histograms to fill.')
  parser.add_argument('--ecut', default=None  , help = 'Energy cut threshold i.e. throw out events above this (GeV)')
  
  args = parser.parse_args()
  jsonFiles        = args.jsonFiles
  prefix           = args.prefix
  dotest           = args.test
  nworkers         = int(args.nworkers)
  chunksize        = int(args.chunksize)
  nchunks          = int(args.nchunks) if not args.nchunks is None else args.nchunks
  outname          = args.outname
  outpath          = args.outpath
  pretend          = args.pretend
  treename         = args.treename
  do_errors        = args.do_errors
  do_systs         = args.do_systs
  split_lep_flavor = args.split_lep_flavor
  skip_sr          = args.skip_sr
  skip_cr          = args.skip_cr
  do_np            = args.do_np
  wc_lst = args.wc_list if args.wc_list is not None else []

  # Set the threshold for the ecut (if not applying a cut, should be None)
  ecut_threshold = args.ecut
  if ecut_threshold is not None: ecut_threshold = float(args.ecut)

  # Figure out which hists to include
  if args.hist_list == ["ana"]:
    # Here we hardcode a list of hists used for the analysis
    hist_lst = ["njets","lj0pt","ptz"]
  elif args.hist_list == ["cr"]:
    # Here we hardcode a list of hists used for the CRs
    hist_lst = ["met", "ljptsum", "l0pt", "l0eta", "l1pt", "l1eta", "j0pt", "j0eta", "njets", "nbtagsl", "invmass"]
  else:
    # We want to specify a custom list
    # If we don't specify this argument, it will be None, and the processor will fill all hists 
    hist_lst = args.hist_list

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
  for sname in samplesdict.keys():
    redirector = samplesdict[sname]['redirector']
    flist[sname] = [(redirector+f) for f in samplesdict[sname]['files']]
    samplesdict[sname]['year'] = samplesdict[sname]['year']
    samplesdict[sname]['xsec'] = float(samplesdict[sname]['xsec'])
    samplesdict[sname]['nEvents'] = int(samplesdict[sname]['nEvents'])
    samplesdict[sname]['nGenEvents'] = int(samplesdict[sname]['nGenEvents'])
    samplesdict[sname]['nSumOfWeights'] = float(samplesdict[sname]['nSumOfWeights'])
    if not samplesdict[sname]["isData"]:
      for wgt_var in WGT_VAR_LST:
          # Check that MC samples have all needed weight sums (only needed if doing systs)
          if do_systs:
              if (wgt_var not in samplesdict[sname]):
                  raise Exception(f"Missing weight variation \"{wgt_var}\".")
              else:
                  samplesdict[sname][wgt_var] = float(samplesdict[sname][wgt_var])

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
    print('   - SumWeights   : %f'   %samplesdict[sname]['nSumOfWeights'])
    if not samplesdict[sname]["isData"]:
      for wgt_var in WGT_VAR_LST:
        if wgt_var in samplesdict[sname]:
            print(f'   - {wgt_var}: {samplesdict[sname][wgt_var]}')
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

  processor_instance = topeft.AnalysisProcessor(samplesdict,wc_lst,hist_lst,ecut_threshold,do_errors,do_systs,split_lep_flavor,skip_sr,skip_cr)

  exec_instance = processor.FuturesExecutor(workers=nworkers)
  runner = processor.Runner(exec_instance, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks)

  # Run the processor and get the output
  tstart = time.time()
  output = runner(flist, treename, processor_instance)
  dt = time.time() - tstart

  nbins = sum(sum(arr.size for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
  nfilled = sum(sum(np.sum(arr > 0) for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
  print("Filled %.0f bins, nonzero bins: %1.1f %%" % (nbins, 100*nfilled/nbins,))
  print("Processing time: %1.2f s with %i workers (%.2f s cpu overall)" % (dt, nworkers, dt*nworkers, ))

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
