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

nameSamples   = 'samples'
nameProcessor = 'topeft'
coffeapath = './coffeaFiles/'
outname = 'plotsTopEFT'

mocapath   = 'moca'
mocaScripts = ['corrections',  'objects', 'samples',  'selection']
#analysis = 
#treeName

nworkers = 8

### (Re)produce inputs...

### Produce/load analysis object
#print("Executing python analysis/topEFT/topeft.py...")
#os.system('python analysis/topEFT/topeft.py')
processor_instance=load(coffeapath+nameProcessor+'.coffea')

### Load samples
samplesdict = load(coffeapath+nameSamples+'.coffea')
flist = {}; xsec = {}; sow = {}; isData = {}
for k in samplesdict.keys():
  flist[k] = samplesdict[k]['files']
  xsec[k]  = samplesdict[k]['xsec']
  sow[k]   = samplesdict[k]['nSumOfWeights']
  isData[k]= samplesdict[k]['isData']


# Run the processor and get the output
tstart = time.time()
#output = processor.run_uproot_job(flist, treename='Events', processor_instance=processor_instance, executor=processor.futures_executor, executor_args={'workers': nworkers, 'pre_workers': 1}, chunksize=500000)
output = processor.run_uproot_job(flist, treename='Events', processor_instance=processor_instance, executor=processor.futures_executor, executor_args={'nano':True,'workers': nworkers, 'pre_workers': 1}, chunksize=500000)
dt = time.time() - tstart

nbins = sum(sum(arr.size for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
nfilled = sum(sum(np.sum(arr > 0) for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
print("Filled %.0f bins" % (nbins, ))
print("Nonzero bins: %.1f%%" % (100*nfilled/nbins, ))

# This is taken from the DM photon analysis...
# Pickle is not very fast or memory efficient, will be replaced by something better soon
#    with lz4f.open("pods/"+options.year+"/"+dataset+".pkl.gz", mode="xb", compression_level=5) as fout:
os.system("mkdir -p histos/")
with gzip.open("histos/" + outname + ".pkl.gz", "wb") as fout:
  cloudpickle.dump(output, fout)

#print("%.2f *cpu overall" % (dt*nworkers, ))



