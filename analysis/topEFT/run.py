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

outname = 'plotsTopEFT'

nworkers = 8

### Load samples
samplesdict = samples.main()
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
output = processor.run_uproot_job(flist, treename='Events', processor_instance=processor_instance, executor=processor.futures_executor, executor_args={"schema": NanoAODSchema,'workers': nworkers, 'pre_workers': 1}, chunksize=500000)
dt = time.time() - tstart

nbins = sum(sum(arr.size for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
nfilled = sum(sum(np.sum(arr > 0) for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
print("Filled %.0f bins, nonzero bins: %1.1f %%" % (nbins, 100*nfilled/nbins,))
print("Processing time: %1.2f s with %i workers (%.2f s cpu overall)" % (dt, nworkers, dt*nworkers, ))

# This is taken from the DM photon analysis...
# Pickle is not very fast or memory efficient, will be replaced by something better soon
#    with lz4f.open("pods/"+options.year+"/"+dataset+".pkl.gz", mode="xb", compression_level=5) as fout:
os.system("mkdir -p histos/")
print('Saving output in %s...'%("histos/" + outname + ".pkl.gz"))
with gzip.open("histos/" + outname + ".pkl.gz", "wb") as fout:
  cloudpickle.dump(output, fout)
print('Done!')




