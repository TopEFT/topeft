'''
 This script is used to transform scale factors, which are tipically provided as 2D histograms within root files,
 into coffea format of corrections.
'''

import os, sys
#from coffea.util import save
basepath = os.path.abspath(__file__).rsplit('/topcoffea/',1)[0]+'/topcoffea/'
#sys.path.append(basepath)
import uproot, uproot_methods
#import numpy as np
#from coffea.arrays import Initialize
from coffea import hist, lookup_tools

def GetHistoFun(fname, hname):
  f = uproot.open(fname)
  h = f[hname]
  return lookup_tools.dense_lookup.dense_lookup(h.values, h.edges)

getMuonIso = GetHistoFun(basepath+'data/scaleFactors/MuonISO.root', 'NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta')
getMuonId  = GetHistoFun(basepath+'data/scaleFactors/MuonID.root',  'NUM_TightID_DEN_genTracks_pt_abseta')
