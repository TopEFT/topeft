'''
 This script is used to transform scale factors, which are tipically provided as 2D histograms within root files,
 into coffea format of corrections.
'''

#import uproot, uproot_methods
import uproot
from coffea import hist, lookup_tools
import os, sys
from topcoffea.modules.paths import topcoffea_path

def GetHistoFun(fname, hname):
  f = uproot.open(fname)
  h = f[hname]
  return lookup_tools.dense_lookup.dense_lookup(h.values, h.edges)

getMuonIso = GetHistoFun(topcoffea_path('data/scaleFactors/MuonISO.root'), 'NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta')
getMuonId  = GetHistoFun(topcoffea_path('data/scaleFactors/MuonID.root'), 'NUM_TightID_DEN_genTracks_pt_abseta')
