'''
 functions.py
 
 This script contains several functions used in the analyses.
 The functions are stored in a dictionary and saved in a '.coffea' file.

 Usage:
 >> python functions.py

'''
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

def GetGoodTriplets(triplet):
  print('triplet: ', triplet)
  pair  = triplet.i0
  third = triplet.i1
  neq1 = np.not_equal(pair.i0.p4, third.p4)
  neq2 = np.not_equal(pair.i1.p4, third.p4)
  mask = (neq1 & neq2)
  return triplet[mask]

def IsClosestToZ(masses, thr=15):
  delta = np.abs(91.18 - masses)
  closest_masses = delta.min()
  is_closest = (delta == closest_masses)&(delta<(np.ones_like(delta)*thr))
  return is_closest

fundic ['pow2'] = pow2
fundic ['GetGoodTriplets'] = GetGoodTriplets
fundic ['IsClosestToZ'] = IsClosestToZ

if not os.path.isdir(outdir): os.system('mkdir -r ' + outdir)
save(fundic, outdir+outname+'.coffea')
