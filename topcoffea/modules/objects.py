'''
 objects.py
 This script contains several functions that implement the object selection according to different object definitions.
 The functions are called with (jagged)arrays as imputs and return a boolean mask.
'''
import numpy as np
import awkward as ak

def isTightMuonPOG(pt,eta,dxy,dz,iso,tight_id, tightCharge, year):
    #dxy and dz cuts are baked on tight_id; tight isolation is 0.15
    mask = ~(pt==np.nan)#just a complicated way to initialize a jagged array with the needed shape to True
    mask = (pt>10)&(abs(eta)<2.4)&(tight_id)&(tightCharge)&(iso<0.15)
    return mask

def isTightElectronPOG(pt,eta,dxy,dz,tight_id,tightCharge,year):
    mask = ~(pt==np.nan)#just a complicated way to initialize a jagged array with the needed shape to True
    mask = ((pt>10)&(abs(eta)<2.5)&(tight_id==4)&(tightCharge)) # Trigger: HLT_Ele27_WPTight_Gsf_v
    return mask

#Return if obj_A is close to obj_B, obj can be lepton or jet.
def isClean(obj_A, obj_B, drmin=0.4):
   ABpairs = obj_A.cross(obj_B, nested=True)
   ABgoodPairs = (ABpairs.i0.delta_r(ABpairs.i1) > drmin).all()
   return ABgoodPairs
   
def isTightJet(pt, eta, jet_id, neHEF, neEmEF, chHEF, chEmEF, nConstituents, jetPtCut=30.0):
    mask = (pt>jetPtCut) & (abs(eta)<2.4) & ((jet_id&6)==6)
    loose = (pt>0)&(neHEF<0.99)&(neEmEF<0.99)&(chEmEF<0.99)&(nConstituents>1)
    tight = (neHEF<0.9)&(neEmEF<0.9)&(chHEF>0.0)
    jetMETcorrection = (pt>0)&(neEmEF + chEmEF < 0.9)
    mask = mask & loose & tight & jetMETcorrection
    return mask

def isClean(obj_A, obj_B, drmin=0.4):
   objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
   mask = ak.fill_none(objB_DR > drmin, True)
   return (mask)

def isCleanJet(jets, electrons, muons, taus, drmin=0.4):
  ''' Returns mask to select clean jets '''
  # Think we can do the jet cleaning like this? It's based on L355 of https://github.com/areinsvo/TTGamma_LongExercise/blob/FullAnalysis/ttgamma/processor.py
  # However, not really sure how "nearest" works, so probably should look into that more at some point
  # But this new version of the function does seem to be returning the same mask as the old version
  jetEle, jetEleDR = jets.nearest(electrons,return_metric=True)
  jetMu,  jetMuDR  = jets.nearest(muons,returyyppn_metric=True)
  jetTau, jetTauDR = jets.nearest(taus, returyyppn_metric=True)
  jetEleMask = ak.fill_none(jetEleDR > drmin, True)
  jetMuMask  = ak.fill_none(jetMuDR  > drmin, True)
  jetTauMask = ak.fill_none(jetTauDR > drmin, True)
  return (jetEleMask & jetMuMask & jetTauMask)

def isPresMuon(dxy, dz, sip3D, looseId):
  mask = (abs(dxy)<0.05)&(abs(dz)<0.1)&(sip3D<8)&(looseId)
  return mask
  
def isTightMuon(pt, eta, dxy, dz, miniIso, sip3D, mvaTTH, mediumPrompt, tightCharge, looseId, minpt=10.0):
  mask = (pt>minpt)&(abs(eta)<2.5)&(abs(dxy)<0.05)&(abs(dz)<0.1)&(sip3D<8)&(looseId)&(miniIso<0.25)&(mvaTTH>0.90)&(tightCharge==2)&(mediumPrompt)
  return mask

def isPresElec(pt, eta, dxy, dz, miniIso, sip3D, lostHits, minpt=15.0):
  mask = (pt>minpt)&(abs(eta)<2.5)&(abs(dxy)<0.05)&(abs(dz)<0.1)&(sip3D<8)&(lostHits<=1)
  return mask
 
def isTightElec(pt, eta, dxy, dz, miniIso, sip3D, mvaTTH, elecMVA, lostHits, convVeto, tightCharge, sieie, hoe, eInvMinusPInv, minpt=15.0):
  maskPOGMVA = ((pt<10)&(abs(eta)<0.8)&(elecMVA>-0.13))|((pt<10)&(abs(eta)>0.8)&(abs(eta)<1.44)&(elecMVA>-0.32))|((pt<10)&(abs(eta)>1.44)&(elecMVA>-0.08))|\
               ((pt>10)&(abs(eta)<0.8)&(elecMVA>-0.86))|((pt>10)&(abs(eta)>0.8)&(abs(eta)<1.44)&(elecMVA>-0.81))|((pt>10)&(abs(eta)>1.44)&(elecMVA>-0.72))
  maskSieie  = ((abs(eta)<1.479)&(sieie<0.011))|((abs(eta)>1.479)&(sieie<0.030))
  maskhoe    = ((abs(eta)<1.479)&(hoe<0.10))|((abs(eta)>1.479)&(hoe<0.07))
  mask = (pt>minpt)&(abs(eta)<2.5)&(abs(dxy)<0.05)&(abs(dz)<0.1)&(sip3D<8)&(lostHits<=1)&\
         (convVeto)&(maskSieie)&(maskPOGMVA)&(eInvMinusPInv>-0.04)&(maskhoe)&(miniIso<0.25)&(mvaTTH>0.90)&(tightCharge==2)
  return mask
 
def isPresTau(pt, eta, dxy, dz, leadTkPtOverTauPt, idAntiMu, idAntiEle, rawIso, idDecayModeNewDMs, minpt=20.0):
  kinematics = (pt>minpt)&(abs(eta)<2.3)&(dxy<1000.)&(dz<0.2)&(leadTkPtOverTauPt*pt>0.5)
  medium = (idAntiMu>0.5)&(idAntiEle>0.5)&(rawIso>0.5)&(idDecayModeNewDMs)
  return kinematics & medium


def ttH_idEmu_cuts_E3(hoe, eta, deltaEtaSC, eInvMinusPInv, sieie):
  return (hoe<(0.10-0.00*(abs(eta+deltaEtaSC)>1.479))) & (eInvMinusPInv>-0.04) & (sieie<(0.011+0.019*(abs(eta+deltaEtaSC)>1.479)))

def smoothBFlav(jetpt,ptmin,ptmax,year,scale_loose=1.0):
  wploose = (0.0614, 0.0521, 0.0494)
  wpmedium = (0.3093, 0.3033, 0.2770)
  x = np.minimum(np.maximum(0, jetpt - ptmin)/(ptmax-ptmin), 1)
  return x*wploose[year-2016]*scale_loose + (1-x)*wpmedium[year-2016]

def coneptElec(pt, mvaTTH, jetRelIso):
  cone_pT = pt* (mvaTTH>0.80)
  cone_pT = cone_pT + (0.90 * pt * (1 + jetRelIso))* (mvaTTH <= 0.80)
  return cone_pT

def coneptMuon(pt, mvaTTH, jetRelIso, mediumId):
  cone_pT = pt* (mvaTTH>0.80)
  cone_pT = cone_pT + (0.90 * pt * (1 + jetRelIso))* ((mediumId<=0) | (mvaTTH<=0.85))
  return cone_pT

def isFOElec(conept, jetBTagDeepFlav, ttH_idEmu_cuts_E3, convVeto, lostHits, mvaTTH, jetRelIso, mvaFall17V2noIso_WP80, year=2018):
  bTagCut = 0.3093 if year==2016 else 0.3033 if year==2017 else 0.2770
  return (conept>10) & (jetBTagDeepFlav<bTagCut) & (ttH_idEmu_cuts_E3 & convVeto & lostHits == 0) & (mvaTTH>0.80) | ((mvaFall17V2noIso_WP80) & (jetRelIso < 0.70))

def isFOMuon(pt, conept, jetBTagDeepFlav, ttH_idEmu_cuts_E3, mvaTTH, jetRelIso, year=2018):
  bTagCut = 0.3093 if year==2016 else 0.3033 if year==2017 else 0.2770
  return (conept>10) & (jetBTagDeepFlav<bTagCut) & (mvaTTH>0.85) | ((jetBTagDeepFlav < smoothBFlav(0.9*pt*1+jetRelIso, 20, 45, year)) & (jetRelIso < 0.50))

def tightSelElec(clean_and_FO_selection_TTH, mvaTTH):
  return (clean_and_FO_selection_TTH) & (mvaTTH > 0.80)

def tightSelMuon(clean_and_FO_selection_TTH, mediumId, mvaTTH):
  return (clean_and_FO_selection_TTH) & (mediumId>0) & (mvaTTH > 0.85)
  return kinematics & medium
