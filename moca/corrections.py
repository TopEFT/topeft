import os, sys
from coffea.util import save
basepath = os.path.abspath(__file__).rsplit('/topcoffea/',1)[0]+'/topcoffea/'
sys.path.append(basepath)
import uproot, uproot_methods
import numpy as np
from coffea.arrays import Initialize
from coffea import hist, lookup_tools

outdir  = basepath+'coffeaFiles/'
outname = 'corrections'

def GetHistoFun(fname, hname):
  f = uproot.open(fname)
  h = f[hname]
  return lookup_tools.dense_lookup.dense_lookup(h.values, h.edges)

getMuonIso = GetHistoFun(basepath+'data/scalefactors/MuonISO.root', 'NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta')
getMuonId  = GetHistoFun(basepath+'data/scalefactors/MuonID.root',  'NUM_TightID_DEN_genTracks_pt_abseta')
#getRecoEB  = GetHistoFun('./inputs/ElecReco_EB_30_100.root',  'g_scalefactors')
#getRecoEE  = GetHistoFun('./inputs/ElecReco_EE_30_100.root',  'g_scalefactors')

corrections = {}
corrections['getMuonIso'] = getMuonIso
corrections['getMuonID' ] = getMuonId
#corrections['getRecoEB']  = getRecoEB
#corrections['getRecoEE']  = getRecoEE
if not os.path.isdir(outdir): os.system('mkdir -r ' + outdir)
save(corrections, outdir+outname+'.coffea')

