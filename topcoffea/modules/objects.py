'''
 objects.py
 This script contains several functions that implement the object selection according to different object definitions.
 The functions are called with (jagged)arrays as imputs and return a boolean mask.
'''

import numpy as np
import awkward as ak
import xgboost as xgb

from topcoffea.modules.GetValuesFromJsons import get_param
from topcoffea.modules.paths import topcoffea_path

### These functions have been synchronized with ttH ###

def isPresTau(pt, eta, dxy, dz, idDeepTau2017v2p1VSjet, minpt=20.0):
    return  (pt>minpt)&(abs(eta)<get_param("eta_t_cut"))&(abs(dxy)<get_param("dxy_tau_cut"))&(abs(dz)<get_param("dz_tau_cut"))&(idDeepTau2017v2p1VSjet>>1 & 1 ==1)

def isTightTau(idDeepTau2017v2p1VSjet):
    return (idDeepTau2017v2p1VSjet>>2 & 1)

def isTightJet(pt, eta, jet_id, jetPtCut=25.0):
    mask = ((pt>jetPtCut) & (abs(eta)<get_param("eta_j_cut")) & (jet_id>get_param("jet_id_cut")))
    return mask

def ttH_idEmu_cuts_E3(hoe, eta, deltaEtaSC, eInvMinusPInv, sieie):
    return (hoe<(0.10-0.00*(abs(eta+deltaEtaSC)>1.479))) & (eInvMinusPInv>-0.04) & (sieie<(0.011+0.019*(abs(eta+deltaEtaSC)>1.479)))

def smoothBFlav(jetpt,ptmin,ptmax,year,scale_loose=1.0):

    # Get the btag wp for the year
    if (year == "2016"):
        wploose  = get_param("btag_wp_loose_UL16")
        wpmedium = get_param("btag_wp_medium_UL16")
    elif (year == "2016APV"):
        wploose  = get_param("btag_wp_loose_UL16APV")
        wpmedium = get_param("btag_wp_medium_UL16APV")
    elif (year == "2017"):
        wploose  = get_param("btag_wp_loose_UL17")
        wpmedium = get_param("btag_wp_medium_UL17")
    elif (year == "2018"):
        wploose  = get_param("btag_wp_loose_UL18")
        wpmedium = get_param("btag_wp_medium_UL18")
    else:
        raise Exception(f"Error: Unknown year \"{year}\". Exiting...")

    x = np.minimum(np.maximum(0, jetpt - ptmin)/(ptmax-ptmin), 1.0)
    return x*wploose*scale_loose + (1-x)*wpmedium

def coneptElec(pt, mvaTTHUL, jetRelIso):
    conePt = (0.90 * pt * (1 + jetRelIso))
    return ak.where((mvaTTHUL>get_param("mva_TTH_e_cut")),pt,conePt)

def coneptMuon(pt, mvaTTHUL, jetRelIso, mediumId):
    conePt = (0.90 * pt * (1 + jetRelIso))
    return ak.where(((mvaTTHUL>get_param("mva_TTH_m_cut"))&(mediumId>0)),pt,conePt)

def isPresElec(pt, eta, dxy, dz, miniIso, sip3D, eleId):
    pt_mask    = (pt       > get_param("pres_e_pt_cut"))
    eta_mask   = (abs(eta) < get_param("eta_e_cut"))
    dxy_mask   = (abs(dxy) < get_param("dxy_cut"))
    dz_mask    = (abs(dz)  < get_param("dz_cut"))
    iso_mask   = (miniIso  < get_param("iso_cut"))
    sip3d_mask = (sip3D    < get_param("sip3d_cut"))
    return (pt_mask & eta_mask & dxy_mask & dz_mask & iso_mask & sip3d_mask & eleId)

def isPresMuon(dxy, dz, sip3D, eta, pt, miniRelIso):
    pt_mask    = (pt         > get_param("pres_m_pt_cut"))
    eta_mask   = (abs(eta)   < get_param("eta_m_cut"))
    dxy_mask   = (abs(dxy)   < get_param("dxy_cut"))
    dz_mask    = (abs(dz)    < get_param("dz_cut"))
    iso_mask   = (miniRelIso < get_param("iso_cut"))
    sip3d_mask = (sip3D      < get_param("sip3d_cut"))
    return (pt_mask & eta_mask & dxy_mask & dz_mask & iso_mask & sip3d_mask)

def isLooseElec(miniPFRelIso_all,sip3d,lostHits):
    return (miniPFRelIso_all<get_param("iso_cut")) & (sip3d<get_param("sip3d_cut")) & (lostHits<=1)

def isLooseMuon(miniPFRelIso_all,sip3d,looseId):
    return (miniPFRelIso_all<get_param("iso_cut")) & (sip3d<get_param("sip3d_cut")) & (looseId)

def isFOElec(pt, conept, jetBTagDeepFlav, ttH_idEmu_cuts_E3, convVeto, lostHits, mvaTTHUL, jetRelIso, mvaFall17V2noIso_WP90, year):

    # Get the btag cut for the year
    if (year == "2016"):
        bTagCut = get_param("btag_wp_medium_UL16")
    elif (year == "2016APV"):
        bTagCut = get_param("btag_wp_medium_UL16APV")
    elif (year == "2017"):
        bTagCut = get_param("btag_wp_medium_UL17")
    elif (year == "2018"):
        bTagCut = get_param("btag_wp_medium_UL18")
    else:
        raise Exception(f"Error: Unknown year \"{year}\". Exiting...")

    btabReq    = (jetBTagDeepFlav<bTagCut)
    ptReq      = (conept>get_param("fo_pt_cut"))
    qualityReq = (ttH_idEmu_cuts_E3 & convVeto & (lostHits==0))
    mvaReq     = ((mvaTTHUL>get_param("mva_TTH_e_cut")) | ((mvaFall17V2noIso_WP90) & (jetBTagDeepFlav<smoothBFlav(0.9*pt*(1+jetRelIso),20,45,year)) & (jetRelIso < get_param("fo_e_jetRelIso_cut"))))

    return ptReq & btabReq & qualityReq & mvaReq

def isFOMuon(pt, conept, jetBTagDeepFlav, mvaTTHUL, jetRelIso, year):

    # Get the btag cut for the year
    if (year == "2016"):
        bTagCut = get_param("btag_wp_medium_UL16")
    elif (year == "2016APV"):
        bTagCut = get_param("btag_wp_medium_UL16APV")
    elif (year == "2017"):
        bTagCut = get_param("btag_wp_medium_UL17")
    elif (year == "2018"):
        bTagCut = get_param("btag_wp_medium_UL18")
    else:
        raise Exception(f"Error: Unknown year \"{year}\". Exiting...")

    btagReq = (jetBTagDeepFlav<bTagCut)
    ptReq   = (conept>get_param("fo_pt_cut"))
    mvaReq  = ((mvaTTHUL>get_param("mva_TTH_m_cut")) | ((jetBTagDeepFlav<smoothBFlav(0.9*pt*(1+jetRelIso),20,45,year)) & (jetRelIso < get_param("fo_m_jetRelIso_cut"))))
    return ptReq & btagReq & mvaReq

def tightSelElec(clean_and_FO_selection_TTH, mvaTTHUL):
    return (clean_and_FO_selection_TTH) & (mvaTTHUL > get_param("mva_TTH_e_cut"))

def tightSelMuon(clean_and_FO_selection_TTH, mediumId, mvaTTHUL):
    return (clean_and_FO_selection_TTH) & (mediumId>0) & (mvaTTHUL > get_param("mva_TTH_m_cut"))

def isClean(obj_A, obj_B, drmin=0.4):
    objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_DR > drmin, True)
    return (mask)



######### WWZ 4l analysis object selection #########

# WWZ preselection for electrons
def is_presel_wwz_ele(ele):
    mask = (
        (ele.pt               >  get_param("wwz_pres_e_pt")) &
        (abs(ele.eta)         <  get_param("wwz_pres_e_eta")) &
        (abs(ele.dxy)         <  get_param("wwz_pres_e_dxy")) &
        (abs(ele.dz)          <  get_param("wwz_pres_e_dz")) &
        (abs(ele.sip3d)       <  get_param("wwz_pres_e_sip3d")) &
        (ele.miniPFRelIso_all <  get_param("wwz_pres_e_miniPFRelIso_all")) &
        (ele.lostHits         <= get_param("wwz_pres_e_lostHits"))
    )
    return mask


# WWZ preselection for muons
def is_presel_wwz_mu(mu):
    mask = (
        (mu.pt               >  get_param("wwz_pres_m_pt")) &
        (abs(mu.eta)         <  get_param("wwz_pres_m_eta")) &
        (abs(mu.dxy)         <  get_param("wwz_pres_m_dxy")) &
        (abs(mu.dz)          <  get_param("wwz_pres_m_dz")) &
        (abs(mu.sip3d)       <  get_param("wwz_pres_m_sip3d")) &
        (mu.miniPFRelIso_all <  get_param("wwz_pres_m_miniPFRelIso_all")) &
        (mu.mediumId)
    )
    return mask


# Get MVA score from TOP MVA for electrons
def get_topmva_score_ele(events, year):

    ele = events.Electron

    # Get the model path
    if (year == "2016"):      ulbase = "UL16"
    elif (year == "2016APV"): ulbase = "UL16APV"
    elif (year == "2017"):    ulbase = "UL17"
    elif (year == "2018"):    ulbase = "UL18"
    else: raise Exception(f"Error: Unknown year \"{year}\". Exiting...")
    model_fpath = topcoffea_path(f"data/topmva/lepid_weights/el_TOP{ulbase}_XGB.weights.bin")

    # Get the input data
    ele["btagDeepFlavB"] = ak.fill_none(ele.matched_jet.btagDeepFlavB, 0)
    ele["jetPtRatio"] = 1./(ele.jetRelIso+1.)
    ele["miniPFRelIso_diff_all_chg"] = ele.miniPFRelIso_all - ele.miniPFRelIso_chg
    # The order here comes from https://github.com/cmstas/VVVNanoLooper/blob/8a194165cdbbbee3bcf69f932d837e95a0a265e6/src/ElectronIDHelper.cc#L110-L122
    in_vals = np.array([
        ak.flatten(ele.pt),
        ak.flatten(ele.eta),
        ak.flatten(ele.jetNDauCharged),
        ak.flatten(ele.miniPFRelIso_chg),
        ak.flatten(ele.miniPFRelIso_diff_all_chg),
        ak.flatten(ele.jetPtRelv2),
        ak.flatten(ele.jetPtRatio),
        ak.flatten(ele.pfRelIso03_all),
        ak.flatten(ele.btagDeepFlavB),
        ak.flatten(ele.sip3d),
        ak.flatten(np.log(abs(ele.dxy))),
        ak.flatten(np.log(abs(ele.dz))),
        ak.flatten(ele.mvaFall17V2noIso),
    ])
    in_vals = np.transpose(in_vals) # To go from e.g. [ [pt1,pt1] , [eta1,eta2] ] -> [ [pt1,eta1] , [pt2,eta2] ]
    in_vals = xgb.DMatrix(in_vals) # The format xgb expects

    # Load model and evaluate
    bst = xgb.Booster()
    bst.load_model(model_fpath)
    score = bst.predict(in_vals)

    # Restore the shape (i.e. unflatten)
    counts = ak.num(ele.pt)
    score = ak.unflatten(score,counts)
    return score


# Get MVA score from TOP MVA for muons
def get_topmva_score_mu(events, year):

    mu = events.Muon

    # Get the model path
    if (year == "2016"):      ulbase = "UL16"
    elif (year == "2016APV"): ulbase = "UL16APV"
    elif (year == "2017"):    ulbase = "UL17"
    elif (year == "2018"):    ulbase = "UL18"
    else: raise Exception(f"Error: Unknown year \"{year}\". Exiting...")
    model_fpath = topcoffea_path(f"data/topmva/lepid_weights/mu_TOP{ulbase}_XGB.weights.bin")

    # Get the input data
    mu["btagDeepFlavB"] = ak.zeros_like(mu.pt) # TODO: Note sure how to handle this, unclear in the c++ code
    mu["jetPtRatio"] = 1./(mu.jetRelIso+1.)
    mu["miniPFRelIso_diff_all_chg"] = mu.miniPFRelIso_all - mu.miniPFRelIso_chg
    in_vals = np.array([
        ak.flatten(mu.pt),
        ak.flatten(mu.eta),
        ak.flatten(mu.jetNDauCharged),
        ak.flatten(mu.miniPFRelIso_chg),
        ak.flatten(mu.miniPFRelIso_diff_all_chg),
        ak.flatten(mu.jetPtRelv2),
        ak.flatten(mu.jetPtRatio),
        ak.flatten(mu.pfRelIso03_all),
        ak.flatten(mu.btagDeepFlavB),
        ak.flatten(mu.sip3d),
        ak.flatten(np.log(abs(mu.dxy))),
        ak.flatten(np.log(abs(mu.dz))),
        ak.flatten(mu.segmentComp),
    ])
    in_vals = np.transpose(in_vals)
    in_vals = xgb.DMatrix(in_vals)

    # Load model and evaluate
    bst = xgb.Booster()
    bst.load_model(model_fpath)
    score = bst.predict(in_vals)

    # Restore the shape (i.e. unflatten)
    counts = ak.num(mu.pt)
    score = ak.unflatten(score,counts)
    return score




