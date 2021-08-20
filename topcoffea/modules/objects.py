'''
 objects.py
 This script contains several functions that implement the object selection according to different object definitions.
 The functions are called with (jagged)arrays as imputs and return a boolean mask.
'''

import numpy as np
import awkward as ak
from topcoffea.modules.GetValuesFromJsons import get_cut

### These functions have been synchronized with ttH ###

def isPresTau(pt, eta, dxy, dz, idDeepTau2017v2p1VSjet, minpt=20.0):
    return  (pt>minpt)&(abs(eta)<get_cut("eta_t_cut"))&(abs(dxy)<get_cut("dxy_tau_cut"))&(abs(dz)<0.2)&(idDeepTau2017v2p1VSjet>>1 & 1 ==1)

def isTightTau(idDeepTau2017v2p1VSjet):
    return (idDeepTau2017v2p1VSjet>>2 & 1)

def isTightJet(pt, eta, jet_id, jetPtCut=25.0):
    mask = ((pt>jetPtCut) & (abs(eta)<get_cut("eta_j_cut")) & (jet_id>get_cut("jet_id_cut")))
    return mask

def ttH_idEmu_cuts_E3(hoe, eta, deltaEtaSC, eInvMinusPInv, sieie):
    return (hoe<(0.10-0.00*(abs(eta+deltaEtaSC)>1.479))) & (eInvMinusPInv>-0.04) & (sieie<(0.011+0.019*(abs(eta+deltaEtaSC)>1.479)))

def smoothBFlav(jetpt,ptmin,ptmax,year,scale_loose=1.0):

    # Get the btag wp for the year
    if ((year == "2016") or (year == "2016APV")):
        wploose  = get_cut("btag_wp_loose_forFO_2016")
        wpmedium = get_cut("btag_wp_medium_forFO_2016")
    elif (year == "2017"):
        wploose  = get_cut("btag_wp_loose_forFO_2017")
        wpmedium = get_cut("btag_wp_medium_forFO_2017")
    elif (year == "2018"):
        wploose  = get_cut("btag_wp_loose_forFO_2018")
        wpmedium = get_cut("btag_wp_medium_forFO_2018")
    else:
        raise Exception(f"Error: Unknown year \"{year}\". Exiting...")

    x = np.minimum(np.maximum(0, jetpt - ptmin)/(ptmax-ptmin), 1.0)
    return x*wploose*scale_loose + (1-x)*wpmedium

def coneptElec(pt, mvaTTH, jetRelIso):
    conePt = (0.90 * pt * (1 + jetRelIso))
    return ak.where((mvaTTH>get_cut("mva_TTH_e_cut")),pt,conePt)

def coneptMuon(pt, mvaTTH, jetRelIso, mediumId):
    conePt = (0.90 * pt * (1 + jetRelIso))
    return ak.where(((mvaTTH>get_cut("mva_TTH_m_cut"))&(mediumId>0)),pt,conePt)

def isPresElec(pt, eta, dxy, dz, miniIso, sip3D, eleId):
    pt_mask    = (pt       > get_cut("pres_e_pt_cut"))
    eta_mask   = (abs(eta) < get_cut("eta_e_cut"))
    dxy_mask   = (abs(dxy) < get_cut("dxy_cut"))
    dz_mask    = (abs(dz)  < get_cut("dz_cut"))
    iso_mask   = (miniIso  < get_cut("iso_cut"))
    sip3d_mask = (sip3D    < get_cut("sip3d_cut"))
    return (pt_mask & eta_mask & dxy_mask & dz_mask & iso_mask & sip3d_mask & eleId)

def isPresMuon(dxy, dz, sip3D, eta, pt, miniRelIso):
    pt_mask    = (pt         > get_cut("pres_m_pt_cut"))
    eta_mask   = (abs(eta)   < get_cut("eta_m_cut"))
    dxy_mask   = (abs(dxy)   < get_cut("dxy_cut"))
    dz_mask    = (abs(dz)    < get_cut("dz_cut"))
    iso_mask   = (miniRelIso < get_cut("iso_cut"))
    sip3d_mask = (sip3D      < get_cut("sip3d_cut"))
    return (pt_mask & eta_mask & dxy_mask & dz_mask & iso_mask & sip3d_mask)

def isLooseElec(miniPFRelIso_all,sip3d,lostHits):
    return (miniPFRelIso_all<get_cut("iso_cut")) & (sip3d<get_cut("sip3d_cut")) & (lostHits<=1)

def isLooseMuon(miniPFRelIso_all,sip3d,looseId):
    return (miniPFRelIso_all<get_cut("iso_cut")) & (sip3d<get_cut("sip3d_cut")) & (looseId)

def isFOElec(conept, jetBTagDeepFlav, ttH_idEmu_cuts_E3, convVeto, lostHits, mvaTTH, jetRelIso, mvaFall17V2noIso_WP80, year):

    # Get the btag cut for the year
    if ((year == "2016") or (year == "2016APV")):
        bTagCut = get_cut("btag_wp_medium_forFO_2016")
    elif (year == "2017"):
        bTagCut = get_cut("btag_wp_medium_forFO_2017")
    elif (year == "2018"):
        bTagCut = get_cut("btag_wp_medium_forFO_2018")
    else:
        raise Exception(f"Error: Unknown year \"{year}\". Exiting...")

    btabReq    = (jetBTagDeepFlav<bTagCut)
    ptReq      = (conept>get_cut("fo_pt_cut"))
    qualityReq = (ttH_idEmu_cuts_E3 & convVeto & (lostHits==0))
    mvaReq     = ((mvaTTH>get_cut("mva_TTH_e_cut")) | ((mvaFall17V2noIso_WP80) & (jetRelIso<get_cut("fo_e_jetRelIso_cut"))))

    return ptReq & btabReq & qualityReq & mvaReq

def isFOMuon(pt, conept, jetBTagDeepFlav, mvaTTH, jetRelIso, year):

    # Get the btag cut for the year
    if ((year == "2016") or (year == "2016APV")):
        bTagCut = get_cut("btag_wp_medium_forFO_2016")
    elif (year == "2017"):
        bTagCut = get_cut("btag_wp_medium_forFO_2017")
    elif (year == "2018"):
        bTagCut = get_cut("btag_wp_medium_forFO_2018")
    else:
        raise Exception(f"Error: Unknown year \"{year}\". Exiting...")

    btagReq = (jetBTagDeepFlav<bTagCut)
    ptReq   = (conept>get_cut("fo_pt_cut"))
    mvaReq  = ((mvaTTH>get_cut("mva_TTH_m_cut")) | ((jetBTagDeepFlav<smoothBFlav(0.9*pt*(1+jetRelIso),20,45,year)) & (jetRelIso < get_cut("fo_m_jetRelIso_cut"))))
    return ptReq & btagReq & mvaReq

def tightSelElec(clean_and_FO_selection_TTH, mvaTTH):
    return (clean_and_FO_selection_TTH) & (mvaTTH > get_cut("mva_TTH_e_cut"))

def tightSelMuon(clean_and_FO_selection_TTH, mediumId, mvaTTH):
    return (clean_and_FO_selection_TTH) & (mediumId>0) & (mvaTTH > get_cut("mva_TTH_m_cut"))

def isClean(obj_A, obj_B, drmin=0.4):
    objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_DR > drmin, True)
    return (mask)
