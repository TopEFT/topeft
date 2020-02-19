import os, sys
basepath = os.path.abspath(__file__).rsplit('/topcoffea/',1)[0]+'/topcoffea/'
sys.path.append(basepath)
import uproot, uproot_methods
import numpy as np
from coffea.arrays import Initialize
from coffea import hist, lookup_tools
from coffea.util import save

outdir  = basepath+'coffeaFiles/'
outname = 'selection'
seldic = {}

def passNJets(nJets, lim=2):
  return nJets >= lim

def passMETcut(met, metCut=40):
  return met >= metCut

seldic['passNJets' ] = passNJets
seldic['passMETcut'] = passMETcut

if not os.path.isdir(outdir): os.system('mkdir -r ' + outdir)
save(seldic, outdir+outname+'.coffea')
