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

#import modules.topeft as topeft
import topeft

nameSamples   = 'samples'
coffeapath = './coffeaFiles/'
outname = 'plotsTopEFT'

nworkers = 8

### Load samples
samplesdict = load(coffeapath+nameSamples+'.coffea')
flist = {}; xsec = {}; sow = {}; isData = {}
for k in samplesdict.keys():
  flist[k] = samplesdict[k]['files']
  xsec[k]  = samplesdict[k]['xsec']
  sow[k]   = samplesdict[k]['nSumOfWeights']
  isData[k]= samplesdict[k]['isData']

processor_instance = topeft.AnalysisProcessor(samplesdict)

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



