import os, sys
basepath = os.path.abspath(__file__).rsplit('/topcoffea/',1)[0]+'/topcoffea/'
sys.path.append(basepath)
import awkward
import uproot, uproot_methods
import numpy as np
from coffea.arrays import Initialize
from coffea import hist, lookup_tools
from coffea.util import save

outdir  = basepath+'coffeaFiles/'
outname = 'objects'

def isTightMuonPOG(pt,eta,dxy,dz,iso,tight_id,year):
    #dxy and dz cuts are baked on tight_id; tight isolation is 0.15
    mask = ~(pt==np.nan)#just a complicated way to initialize a jagged array with the needed shape to True
    mask = (pt>10)&(abs(eta)<2.4)&(tight_id)&(iso<0.15)
    return mask

def isTightElectronPOG(pt,eta,dxy,dz,tight_id,year):
    mask = ~(pt==np.nan)#just a complicated way to initialize a jagged array with the needed shape to True
    mask = ((pt>10)&(abs(eta)<2.5)&(tight_id==4)) # Trigger: HLT_Ele27_WPTight_Gsf_v
    return mask

def isGoodJet(pt, eta, jet_id, jetPtCut=30):
    mask = (pt>jetPtCut) & (abs(eta)<2.4) & ((jet_id&2)==2)
    return mask

def isMuonMVA(pt, eta, dxy, dz, miniIso, sip3D, mvaTTH, mediumPrompt, jetDeepB=0, minpt=15):
  mask = (pt>minpt) & (abs(eta)<2.4) & (abs(dxy)<0.05) & (abs(dz)<0.1) &
    (miniIso<0.4) & (sip3D<5) & (mvaTTH>0.55) & (mediumPrompt) & (jetDeepB<0.1522)
  return mask

def isElecMVA(pt, eta, dxy, dz, miniIso, sip3D, mvaTTH, elecMVA, lostHits, convVeto, jetDeepB=0, minpt=15):
  miniIsoCut = 0.085 # Tight
  mask = (pt>minpt) & (abs(eta)<2.4) & (abs(dxy)<0.05) & (abs(dz)<0.1) &
    (miniIso<miniIsoCut) & (sip3D<8) & (mvaTTH>0.125) & (elecMVA) & (jetDeepB<0.1522) &
    (lostHits<1) & (convVeto)
  return mask 

ids = {}
ids['isTightMuonPOG'] = isTightMuonPOG
ids['isTightElectronPOG'] = isTightElectronPOG
ids['isMuonMVA'] = isMuonMVA
ids['isElecMVA'] = isElecMVA
ids['isGoodJet'] = isGoodJet

if not os.path.isdir(outdir): os.system('mkdir -r ' + outdir)
save(ids, outdir+outname+'.coffea')


   

