'''
 objects.py
 This script contains several functions that implement the object selection according to different object definitions.
 The functions are called with (jagged)arrays as imputs and return a boolean mask.
'''

import numpy as np
import awkward as ak

from topcoffea.modules.get_param_from_jsons import GetParam
from topcoffea.modules.paths import topcoffea_path
from topeft.modules.paths import topeft_path
get_tc_param = GetParam(topcoffea_path("params/params.json"))
get_te_param = GetParam(topeft_path("params/params.json"))

### These functions have been synchronized with ttH ###

def isPresTau(pt, eta, dxy, dz, idDeepTauVSjet, idDeepTauVSe, idDeepTauVSmu, minpt=20.0):
    return  (pt>minpt)&(abs(eta)<get_te_param("eta_t_cut"))&(abs(dxy)<get_te_param("dxy_tau_cut"))&(abs(dz)<get_te_param("dz_tau_cut"))&(idDeepTauVSjet>>3 & 1 ==1)&(idDeepTauVSe>>1 & 1 ==1)&(idDeepTauVSmu>>1 & 1 ==1)

def isVLooseTau(idDeepTauVSjet):
    return (idDeepTauVSjet>>2 & 1)

def isLooseTau(idDeepTauVSjet):
    return (idDeepTauVSjet>>3 & 1)

def isMediumTau(idDeepTauVSjet):
    return (idDeepTauVSjet>>4 & 1)

def isTightTau(idDeepTauVSjet):
    return (idDeepTauVSjet>>5 & 1)

def isVTightTau(idDeepTauVSjet):
    return (idDeepTauVSjet>>6 & 1)

def isVVTightTau(idDeepTauVSjet):
    return (idDeepTauVSjet>>7 & 1)

def iseTightTau(idDeepTauVSe):
    return (idDeepTauVSe>>1 & 1)

def ismTightTau(idDeepTauVSmu):
    return (idDeepTauVSmu>>1 & 1)

def lepJetBTagAdder(leptons, jets, btagger="btagDeepFlavB", dummyValue=-99.):
    is_matched = (leptons.jetIdx > -1) & (leptons.jetIdx < ak.num(jets))
    leptons["jetBTag"] = ak.fill_none(leptons.matched_jet.btagDeepFlavB, dummyValue)
    
def ttH_idEmu_cuts_E3(hoe, eta, deltaEtaSC, eInvMinusPInv, sieie):
    return (hoe<(0.10-0.00*(abs(eta+deltaEtaSC)>1.479))) & (eInvMinusPInv>-0.04) & (sieie<(0.011+0.019*(abs(eta+deltaEtaSC)>1.479)))

def isFwdJet(pt, eta, jet_id, jetPtCut=25.0):
    mask = ((pt>jetPtCut) & (abs(eta)>get_te_param("eta_j_cut")) & (jet_id>get_te_param("jet_id_cut")))
    return mask

def smoothBFlav(jetpt,ptmin,ptmax,year,scale_loose=1.0):

    # Get the btag wp for the year
    if (year == "2016"):
        wploose  = get_tc_param("btag_wp_loose_UL16")
        wpmedium = get_tc_param("btag_wp_medium_UL16")
    elif (year == "2016APV"):
        wploose  = get_tc_param("btag_wp_loose_UL16APV")
        wpmedium = get_tc_param("btag_wp_medium_UL16APV")
    elif (year == "2017"):
        wploose  = get_tc_param("btag_wp_loose_UL17")
        wpmedium = get_tc_param("btag_wp_medium_UL17")
    elif (year == "2018"):
        wploose  = get_tc_param("btag_wp_loose_UL18")
        wpmedium = get_tc_param("btag_wp_medium_UL18")
    elif (year == "2022"):
        wploose = get_tc_param("btag_wp_loose_2022")
        wpmedium = get_tc_param("btag_wp_medium_2022")
    elif (year == "2022EE"):
        wploose = get_tc_param("btag_wp_loose_2022EE")
        wpmedium = get_tc_param("btag_wp_medium_2022EE")
    elif (year == "2023"):
        wploose = get_tc_param("btag_wp_loose_2023")
        wpmedium = get_tc_param("btag_wp_medium_2023")
    elif (year == "2023BPix"):
        wploose = get_tc_param("btag_wp_loose_2023BPix")
        wpmedium = get_tc_param("btag_wp_medium_2023BPix")
    else:
        raise Exception(f"Error: Unknown year \"{year}\". Exiting...")

    x = np.minimum(np.maximum(0, jetpt - ptmin)/(ptmax-ptmin), 1.0)
    return x*wploose*scale_loose + (1-x)*wpmedium

def smoothSip3D(jetpt, sipmin, sipmax, ptmin, ptmax):
    x = np.minimum(np.maximum(0, jetpt - ptmin)/(ptmax-ptmin), 1.0)
    return x*sipmax+(1-x)*sipmin

def get_medium_btag_foryear(year):
    # Get the btag cut for the year
    if (year == "2016"):
        return get_tc_param("btag_wp_medium_UL16")
    elif (year == "2016APV"):
        return get_tc_param("btag_wp_medium_UL16APV")
    elif (year == "2017"):
        return get_tc_param("btag_wp_medium_UL17")
    elif (year == "2018"):
        return get_tc_param("btag_wp_medium_UL18")
    elif (year == "2022"):
        return get_tc_param("btag_wp_medium_2022")
    elif (year == "2022EE"):
        return get_tc_param("btag_wp_medium_2022EE")
    elif (year == "2023"):
        return get_tc_param("btag_wp_medium_2023")
    elif (year == "2023BPix"):
        return get_tc_param("btag_wp_medium_2023BPix")
    else:
        raise Exception(f"Error: Unknown year \"{year}\". Exiting...")

class run2leptonselection:

    def __init__(self):
        pass

    def coneptElec(self, ele):
        conePt = (0.90 * ele.pt * (1 + ele.jetRelIso))
        return ak.where((ele.mvaTTHUL>get_te_param("mva_TTH_e_cut")),ele.pt,conePt)

    def coneptMuon(self, muo):
        conePt = (0.90 * muo.pt * (1 + muo.jetRelIso))
        return ak.where(((muo.mvaTTHUL>get_te_param("mva_TTH_m_cut"))&(muo.mediumId>0)),muo.pt,conePt)

    def isPresElec(self, ele):
        pt_mask    = (ele.pt       > get_te_param("pres_e_pt_cut"))
        eta_mask   = (abs(ele.eta) < get_te_param("eta_e_cut"))
        dxy_mask   = (abs(ele.dxy) < get_te_param("dxy_cut"))
        dz_mask    = (abs(ele.dz)  < get_te_param("dz_cut"))
        iso_mask   = (ele.miniPFRelIso_all  < get_te_param("iso_cut"))
        sip3d_mask = (ele.sip3d    < get_te_param("sip3d_cut"))
        return (pt_mask & eta_mask & dxy_mask & dz_mask & iso_mask & sip3d_mask & ele.mvaFall17V2noIso_WPL)

    def isPresMuon(self, muon):
        pt_mask    = (muon.pt         > get_te_param("pres_m_pt_cut"))
        eta_mask   = (abs(muon.eta)   < get_te_param("eta_m_cut"))
        dxy_mask   = (abs(muon.dxy)   < get_te_param("dxy_cut"))
        dz_mask    = (abs(muon.dz)    < get_te_param("dz_cut"))
        iso_mask   = (muon.miniPFRelIso_all < get_te_param("iso_cut"))
        sip3d_mask = (muon.sip3d      < get_te_param("sip3d_cut"))
        return (pt_mask & eta_mask & dxy_mask & dz_mask & iso_mask & sip3d_mask)

    def isLooseElec(self, ele):
        return (ele.miniPFRelIso_all<get_te_param("iso_cut")) & (ele.sip3d<get_te_param("sip3d_cut")) & (ele.lostHits<=1)

    def isLooseMuon(self, muon):
        return (muon.miniPFRelIso_all<get_te_param("iso_cut")) & (muon.sip3d<get_te_param("sip3d_cut")) & (muon.looseId)

    def isFOElec(self, ele, year):
        bTagCut=get_medium_btag_foryear(year)
        btabReq    = (ele.jetBTag<bTagCut)
        ptReq      = (ele.conept>get_te_param("fo_pt_cut"))
        qualityReq = (ele.idEmu & ele.convVeto & (ele.lostHits==0))
        mvaReq     = ((ele.mvaTTHUL>get_te_param("mva_TTH_e_cut")) | ((ele.mvaFall17V2noIso_WP90) & (ele.jetBTag<smoothBFlav(0.9*ele.pt*(1+ele.jetRelIso),20,45,year)) & (ele.jetRelIso < get_te_param("fo_e_jetRelIso_cut"))))

        return ptReq & btabReq & qualityReq & mvaReq

    def isFOMuon(self, muo, year):
        bTagCut=get_medium_btag_foryear(year)
        btagReq = (muo.jetBTag<bTagCut)
        ptReq   = (muo.conept>get_te_param("fo_pt_cut"))
        mvaReq  = ((muo.mvaTTHUL>get_te_param("mva_TTH_m_cut")) | ((muo.jetBTag<smoothBFlav(0.9*muo.pt*(1+muo.jetRelIso),20,45,year)) & (muo.jetRelIso < get_te_param("fo_m_jetRelIso_cut"))))
        return ptReq & btagReq & mvaReq

    def tightSelElec(self, ele):
        return (ele.isFO) & (ele.mvaTTHUL > get_te_param("mva_TTH_e_cut"))

    def tightSelMuon(self, muo):
        return (muo.isFO) & (muo.mediumId>0) & (muo.mvaTTHUL > get_te_param("mva_TTH_m_cut"))

class run3leptonselection:
    def __init__(self, useMVA=True):
        self.useMVA = useMVA

    def coneptElec(self, ele):
        conePt = (0.90 * ele.pt_corrected * (1 + ele.jetRelIso))
        return ak.where( (ele.mvaTTHrun3>get_te_param("mva_TTH_e_cut_run3")), ele.pt_corrected, conePt)

    def coneptMuon(self, muo):
        conePt = (0.90 * muo.pt * (1 + muo.jetRelIso))
        return ak.where( ((muo.mvaTTHrun3>get_te_param("mva_TTH_m_cut_run3"))&(muo.mediumId>0)), muo.pt, conePt)

    def isPresElec(self, ele):
        pt_mask    = (ele.pt       > get_te_param("pres_e_pt_cut"))
        eta_mask   = (abs(ele.eta) < get_te_param("eta_e_cut"))
        dxy_mask   = (abs(ele.dxy) < get_te_param("dxy_cut"))
        dz_mask    = (abs(ele.dz)  < get_te_param("dz_cut"))
        iso_mask   = (ele.miniPFRelIso_all  < get_te_param("iso_cut"))
        sip3d_mask = (ele.sip3d    < get_te_param("sip3d_cut"))
        ecal_crack_mask = (((abs(ele.deltaEtaSC+ele.eta) < 1.4442) | (abs(ele.deltaEtaSC+ele.eta) > 1.566)))
        return (pt_mask & eta_mask & dxy_mask & dz_mask & iso_mask & sip3d_mask & ecal_crack_mask)

    def isPresMuon(self, muon):
        pt_mask    = (muon.pt         > get_te_param("pres_m_pt_cut"))
        eta_mask   = (abs(muon.eta)   < get_te_param("eta_m_cut"))
        dxy_mask   = (abs(muon.dxy)   < get_te_param("dxy_cut"))
        dz_mask    = (abs(muon.dz)    < get_te_param("dz_cut"))
        iso_mask   = (muon.miniPFRelIso_all < get_te_param("iso_cut"))
        sip3d_mask = (muon.sip3d      < get_te_param("sip3d_cut"))
        return (pt_mask & eta_mask & dxy_mask & dz_mask & iso_mask & sip3d_mask)

    def isLooseElec(self, ele):
        return (ele.miniPFRelIso_all<get_te_param("iso_cut")) & (ele.sip3d<get_te_param("sip3d_cut")) & (ele.lostHits<=1)

    def isLooseMuon(self, muon):
        return (muon.miniPFRelIso_all<get_te_param("iso_cut")) & (muon.sip3d<get_te_param("sip3d_cut")) & (muon.looseId)

    def isFOElec(self, ele, year):
        bTagCut    = get_medium_btag_foryear(year)
        btagReq    = (ele.jetBTag<bTagCut)
        ptReq      = (ele.conept>get_te_param("fo_pt_cut"))
        qualityReq = (ele.idEmu & ele.convVeto & (ele.lostHits==0))
        if not self.useMVA:
            mvaReq     = (((ele.mvaIso > get_te_param("fo_e_mvaiso_cut_run3"))  & (ele.jetRelIso < get_te_param("fo_e_jetRelIso_cut"))))
        else:
            mvaReq     = ((ele.mvaTTHrun3>get_te_param("mva_TTH_e_cut_run3")) | ((ele.mvaIso > get_te_param("fo_e_mvaiso_cut_run3")) & (ele.jetRelIso < get_te_param("fo_e_jetRelIso_cut")))) ##original cut from Sergio
        return ptReq & btagReq & qualityReq & mvaReq
        
    def isFOMuon(self, muo, year):
        bTagCut=get_medium_btag_foryear(year)
        btagReq = (muo.jetBTag<bTagCut)
        ptReq   = (muo.conept>get_te_param("fo_pt_cut"))
        if not self.useMVA:
            mvaReq  = (((muo.jetBTag<smoothBFlav(0.9*muo.pt*(1+muo.jetRelIso),20,45,year)) & (muo.jetRelIso < get_te_param("fo_m_jetRelIso_cut")) & (muo.sip3d < smoothSip3D(0.9*muo.pt*(1+muo.jetRelIso),2.5,8.,15,45))))
        else:
            mvaReq  = ((muo.mvaTTHrun3>get_te_param("mva_TTH_m_cut_run3")) | ((muo.jetBTag<smoothBFlav(0.9*muo.pt*(1+muo.jetRelIso),20,45,year)) & (muo.jetRelIso < get_te_param("fo_m_jetRelIso_cut")) & (muo.sip3d < smoothSip3D(0.9*muo.pt*(1+muo.jetRelIso),2.5,8.,15,45)))) #original cut from Sergio
        return ptReq & btagReq & mvaReq

    def tightSelElec(self, ele):
        if not self.useMVA:
            return ((ele.isFO) & (ele.miniPFRelIso_all<0.1))
        else:
            return (ele.isFO) & (ele.mvaTTHrun3 > get_te_param("mva_TTH_e_cut_run3")) #original cut from Sergio
        
    def tightSelMuon(self, muo):
        if not self.useMVA:
            return ((muo.isFO) & (muo.mediumId>0) & (muo.miniPFRelIso_all<0.1))
        else:
            return (muo.isFO) & (muo.mediumId>0) & (muo.mvaTTHrun3 > get_te_param("mva_TTH_m_cut_run3")) #original cut from Sergio

def isClean(obj_A, obj_B, drmin=0.4):
    objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_DR > drmin, True)
    return (mask)
