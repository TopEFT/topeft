import os, sys
basepath = os.path.abspath(__file__).rsplit('/topcoffea/',1)[0]+'/topcoffea/'
sys.path.append(basepath)
import uproot, uproot_methods
import numpy as np
from coffea.arrays import Initialize
from coffea import hist, lookup_tools
from coffea.util import save

outdir  = basepath+'coffeaFiles/'
outname = 'functions'
fundic = {}

pow2 = lambda x : x*x

fundic ['pow2'] = pow2

if not os.path.isdir(outdir): os.system('mkdir -r ' + outdir)
save(fundic, outdir+outname+'.coffea')
