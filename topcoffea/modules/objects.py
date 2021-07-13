'''
 objects.py
 This script contains several functions that implement the object selection according to different object definitions.
 The functions are called with (jagged)arrays as imputs and return a boolean mask.
'''
import numpy as np
import awkward as ak

### These functions have not been synchronized with ttH, but ARE used in topeft ###

def isPresTau(pt, eta, dxy, dz, idDecayModeNewDMs, idDeepTau2017v2p1VSjet, minpt=20.0):
  return  (pt>minpt)&(abs(eta)<2.3)&(abs(dxy)<1000.)&(abs(dz)<0.2)&(idDecayModeNewDMs)&(idDeepTau2017v2p1VSjet>>1 & 1 ==1)

def isTightTau(idDeepTau2017v2p1VSjet):
  return (idDeepTau2017v2p1VSjet>>2 & 1)

def isTightJet(pt, eta, jet_id, jetPtCut=25.0):
    mask = (pt>jetPtCut) & (abs(eta)<2.4) & (jet_id>0)
    return mask

### These functions have been synchronized with ttH ###

def ttH_idEmu_cuts_E3(hoe, eta, deltaEtaSC, eInvMinusPInv, sieie):
  return (hoe<(0.10-0.00*(abs(eta+deltaEtaSC)>1.479))) & (eInvMinusPInv>-0.04) & (sieie<(0.011+0.019*(abs(eta+deltaEtaSC)>1.479)))

def smoothBFlav(jetpt,ptmin,ptmax,year,scale_loose=1.0):
  wploose = (0.0614, 0.0521, 0.0494)
  wpmedium = (0.3093, 0.3033, 0.2770)
  x = np.minimum(np.maximum(0, jetpt - ptmin)/(ptmax-ptmin), 1.0)
  return x*wploose[year-2016]*scale_loose + (1-x)*wpmedium[year-2016]

def coneptElec(pt, mvaTTH, jetRelIso):
  conePt = (0.90 * pt * (1 + jetRelIso))
  return ak.where((mvaTTH>0.80),pt,conePt)

def coneptMuon(pt, mvaTTH, jetRelIso, mediumId):
  conePt = (0.90 * pt * (1 + jetRelIso))
  return ak.where(((mvaTTH>0.85)&(mediumId>0)),pt,conePt)

def isPresElec(pt, eta, dxy, dz, miniIso, sip3D, eleId):
  mask = (pt>7)&(abs(eta)<2.5)&(abs(dxy)<0.05)&(abs(dz)<0.1)&(miniIso<0.4)&(sip3D<8)&(eleId)
  return mask

def isPresMuon(dxy, dz, sip3D, eta, pt, miniRelIso):
  mask = (abs(dxy)<0.05)&(abs(dz)<0.1)&(sip3D<8)&(abs(eta)<2.4)&(pt>5)&(miniRelIso<0.4)
  return mask

def isLooseElec(miniPFRelIso_all,sip3d,lostHits):
  return (miniPFRelIso_all<0.4) & (sip3d<8) & (lostHits<=1)

def isLooseMuon(miniPFRelIso_all,sip3d,looseId):
  return (miniPFRelIso_all<0.4) & (sip3d<8) & (looseId)

def isFOElec(conept, jetBTagDeepFlav, ttH_idEmu_cuts_E3, convVeto, lostHits, mvaTTH, jetRelIso, mvaFall17V2noIso_WP80, year):
  bTagCut = 0.3093 if year==2016 else 0.3033 if year==2017 else 0.2770
  btabReq    = (jetBTagDeepFlav<bTagCut)
  ptReq      = (conept>10)
  qualityReq = (ttH_idEmu_cuts_E3 & convVeto & (lostHits==0))
  mvaReq     = ((mvaTTH>0.80) | ((mvaFall17V2noIso_WP80) & (jetRelIso<0.70)))
  return ptReq & btabReq & qualityReq & mvaReq

def isFOMuon(pt, conept, jetBTagDeepFlav, mvaTTH, jetRelIso, year):
  bTagCut = 0.3093 if year==2016 else 0.3033 if year==2017 else 0.2770
  btagReq = (jetBTagDeepFlav<bTagCut)
  ptReq   = (conept>10)
  mvaReq  = ((mvaTTH>0.85) | ((jetBTagDeepFlav<smoothBFlav(0.9*pt*(1+jetRelIso),20,45,year)) & (jetRelIso < 0.50)))
  return ptReq & btagReq & mvaReq

def tightSelElec(clean_and_FO_selection_TTH, mvaTTH):
  return (clean_and_FO_selection_TTH) & (mvaTTH > 0.80)

def tightSelMuon(clean_and_FO_selection_TTH, mediumId, mvaTTH):
  return (clean_and_FO_selection_TTH) & (mediumId>0) & (mvaTTH > 0.85)

def isClean(obj_A, obj_B, drmin=0.4):
   objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
   mask = ak.fill_none(objB_DR > drmin, True)
   return (mask)
