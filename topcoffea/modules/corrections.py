'''
 This script is used to transform scale factors, which are tipically provided as 2D histograms within root files,
 into coffea format of corrections.
'''

#import uproot, uproot_methods
import uproot
from coffea import hist, lookup_tools
import os, sys

# Should probably do this in a different way...
basepath = os.path.abspath(__file__).rsplit('/')
basepath = basepath[:len(basepath)-3]
basepath = "/"+os.path.join(*basepath)

def GetHistoFun(fname, hname):
  f = uproot.open(fname)
  h = f[hname]
  return lookup_tools.dense_lookup.dense_lookup(h.values, h.edges)

#getMuonIso = GetHistoFun(basepath+'data/scaleFactors/MuonISO.root', 'NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta')
#getMuonId  = GetHistoFun(basepath+'data/scaleFactors/MuonID.root',  'NUM_TightID_DEN_genTracks_pt_abseta')
getMuonIso = GetHistoFun(os.path.join(basepath,'data/scaleFactors/MuonISO.root'), 'NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta')
getMuonId  = GetHistoFun(os.path.join(basepath,'data/scaleFactors/MuonID.root'), 'NUM_TightID_DEN_genTracks_pt_abseta')
