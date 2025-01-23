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
from topeft.modules.genParentage import maxHistoryPDGID
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
        btabReq    = (ele.jetBTagDeepFlav<bTagCut)
        ptReq      = (ele.conept>get_te_param("fo_pt_cut"))
        qualityReq = (ele.idEmu & ele.convVeto & (ele.lostHits==0))
        mvaReq     = ((ele.mvaTTHUL>get_te_param("mva_TTH_e_cut")) | ((ele.mvaFall17V2noIso_WP90) & (ele.jetBTagDeepFlav<smoothBFlav(0.9*ele.pt*(1+ele.jetRelIso),20,45,year)) & (ele.jetRelIso < get_te_param("fo_e_jetRelIso_cut"))))

        return ptReq & btabReq & qualityReq & mvaReq

    def isFOMuon(self, muo, year):
        bTagCut=get_medium_btag_foryear(year)
        btagReq = (muo.jetBTagDeepFlav<bTagCut)
        ptReq   = (muo.conept>get_te_param("fo_pt_cut"))
        mvaReq  = ((muo.mvaTTHUL>get_te_param("mva_TTH_m_cut")) | ((muo.jetBTagDeepFlav<smoothBFlav(0.9*muo.pt*(1+muo.jetRelIso),20,45,year)) & (muo.jetRelIso < get_te_param("fo_m_jetRelIso_cut"))))
        return ptReq & btagReq & mvaReq

    def tightSelElec(self, ele):
        return (ele.isFO) & (ele.mvaTTHUL > get_te_param("mva_TTH_e_cut"))

    def tightSelMuon(self, muo):
        return (muo.isFO) & (muo.mediumId>0) & (muo.mvaTTHUL > get_te_param("mva_TTH_m_cut"))

class run3leptonselection:

    def __init__(self):
        pass

    def coneptElec(self, ele):
        conePt = (0.90 * ele.pt * (1 + ele.jetRelIso))
        return ak.where((ele.mvaTTH_Run3>get_te_param("mva_TTH_e_cut_run3")),ele.pt,conePt)

    def coneptMuon(self, muo):
        conePt = (0.90 * muo.pt * (1 + muo.jetRelIso))
        return ak.where(((muo.mvaTTH_Run3>get_te_param("mva_TTH_m_cut_run3"))&(muo.mediumId>0)),muo.pt,conePt)

    def isPresElec(self, ele):
        pt_mask    = (ele.pt       > get_te_param("pres_e_pt_cut"))
        eta_mask   = (abs(ele.eta) < get_te_param("eta_e_cut"))
        dxy_mask   = (abs(ele.dxy) < get_te_param("dxy_cut"))
        dz_mask    = (abs(ele.dz)  < get_te_param("dz_cut"))
        iso_mask   = (ele.miniPFRelIso_all  < get_te_param("iso_cut"))
        sip3d_mask = (ele.sip3d    < get_te_param("sip3d_cut"))
        ecal_crack_mask = (((abs(ele.etaSC) < 1.4442) | (abs(ele.etaSC) > 1.566)))
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
        return (muon.miniPFRelIso_all<get_te_param("iso_cut")) & (muon.sip3d<get_te_param("sip3d_cut")) & (muon.mediumId)

    def isFOElec(self, ele, year):
        bTagCut=get_medium_btag_foryear(year)
        btabReq    = (ele.jetBTagDeepFlav<bTagCut)
        ptReq      = (ele.conept>get_te_param("fo_pt_cut"))
        qualityReq = (ele.idEmu & ele.convVeto & (ele.lostHits==0))
        mvaReq     = ((ele.mvaTTH_Run3>get_te_param("mva_TTH_e_cut_run3")) | ((ele.mvaIso > get_te_param("fo_e_mvaiso_cut_run3"))  & (ele.jetRelIso < get_te_param("fo_e_jetRelIso_cut"))))

    def isFOMuon(self, muo, year):
        bTagCut=get_medium_btag_foryear(year)
        btagReq = (muo.jetBTagDeepFlav<bTagCut)
        ptReq   = (muo.conept>get_te_param("fo_pt_cut"))
        mvaReq  = ((muo.mvaTTH_Run3>get_te_param("mva_TTH_m_cut_run3")) | ((muo.jetBTagDeepFlav<smoothBFlav(0.9*muo.pt*(1+muo.jetRelIso),20,45,year)) & (muo.jetRelIso < get_te_param("fo_m_jetRelIso_cut")) & (muo.sip3d < smoothSip3D(0.9*muo.pt*(1+muo.jetRelIso),2.5,8.,15,45))))
        return ptReq & btagReq & mvaReq

    def tightSelElec(self, ele):
        return (ele.isFO) & (ele.mvaTTH_Run3 > get_te_param("mva_TTH_e_cut_run3"))

    def tightSelMuon(self, muo):
        return (muo.isFO) & (muo.mediumId>0) & (muo.mvaTTH_Run3 > get_te_param("mva_TTH_m_cut_run3"))

def pt_eta_cut_genMatched_objects(gen_matched_object,pt_threshold,eta_threshold):
    pt_mask = gen_matched_object.pt > pt_threshold
    eta_mask = abs(gen_matched_object.eta) < eta_threshold
    pt_eta_mask = ak.fill_none(ak.pad_none((pt_mask & eta_mask),1),False)

    return pt_eta_mask

def object_sel_photon(ph_collection, pt_threshold, eta_threshold):
    pt_mask = ph_collection.pt > pt_threshold
    eta_mask = abs(ph_collection.eta) < eta_threshold
    pt_eta_mask = ak.fill_none(ak.pad_none((pt_mask & eta_mask), 1), False)

    return pt_eta_mask

def pt_eta_cut_genMatched_photons(pt,eta):
    pt_mask = pt > 20
    eta_mask = abs(eta) < 1.44
    pt_eta_mask = (pt_mask & eta_mask)
    return pt_eta_mask

def mediumPhoton(ph):
    return (ph.cutBased >= 2)

def mediumEle(ele):
    return (ele.isMedium)

def mediumMu(mu):
    return (mu.mediumId)

def maxParentage(genPart):
    """Filter generated events with overlapping phase space"""
    genMotherIdx = genPart.genPartIdxMother
    genpdgId = genPart.pdgId
    ##calculate maxparent pdgId of the event
    idx = ak.to_numpy(ak.flatten(abs(genPart.pdgId)))
    par = ak.to_numpy(ak.flatten(genPart.genPartIdxMother))
    num = ak.to_numpy(ak.num(genPart.pdgId))
    maxParentFlatten = maxHistoryPDGID(idx,par,num)
    genPart["maxParent"] = ak.unflatten(maxParentFlatten, num)
    genPart["maxParent"] = ak.unflatten(maxParentFlatten, num)

    non_hadronic_parentage = ((abs(genPart.pdgId) == 22) & (genPart.maxParent < 37))

    return non_hadronic_parentage

def selectPhoton(photons):
    photon_pt_eta_mask = (photons.pt > 20) & (abs(photons.eta)<1.44) #this is what we want for our SR

    photon_pixelSeed_electronVeto_mask = (np.invert(photons.pixelSeed) & (photons.electronVeto))  #We invert the pixel seed cause we want to veto events with photons that have pixelSeed cause they are misid electrons
    photon_mediumID = (photons.cutBased >= 2) #At least medium
    #Let's relax two components from medium cutBasedID -- 1. charged isolation and 2. sigmaetaeta
    #split out the ID requirement using the vid (versioned ID) bitmap
    #"(x & 3) >= 2" makes sure each component passes medium threshold
    photon_MinPtCut = (photons.vidNestedWPBitmap >> 0 & 3) >= 2
    photon_PhoSCEtaMultiRangeCut = (photons.vidNestedWPBitmap >> 2 & 3) >= 2
    photon_PhoSingleTowerHadOverEmCut = (photons.vidNestedWPBitmap >> 4 & 3) >= 2
    photon_sieieCut = (photons.vidNestedWPBitmap >> 6 & 3) >= 2
    photon_ChIsoCut = (photons.vidNestedWPBitmap >> 8 & 3) >= 2
    photon_NeuIsoCut = (photons.vidNestedWPBitmap >> 10 & 3) >= 2
    photon_PhoIsoCut = (photons.vidNestedWPBitmap >> 12 & 3) >= 2

    #also define the charged hadron isolation for photons
    photon_chIso = ((photons.pfRelIso03_chg) * (photons.pt))

    # photons passing all ID requirements, without the charged hadron isolation cut applied
    mediumPhoton_noSieie_noChIso = (
        photon_MinPtCut &
        photon_PhoSCEtaMultiRangeCut &
        photon_PhoSingleTowerHadOverEmCut &
        #& photon_sieieCut
        #& (photons.sieie < 0.010)
        #& (photons.pfRelIso03_chg < 1.141)
        #& (photon_chIso < 1.141) &
        photon_NeuIsoCut &
        photon_PhoIsoCut
    )

    mediumPhoton_noChIso = (
        photon_MinPtCut &
        photon_PhoSCEtaMultiRangeCut &
        photon_PhoSingleTowerHadOverEmCut &
        photon_sieieCut &
        #& (photons.sieie < 0.010)
        #& (photons.pfRelIso03_chg < 1.141)
        #& (photon_chIso < 1.141) &
        photon_NeuIsoCut &
        photon_PhoIsoCut
    )

    #mediumPhotons_relaxed = photons[photon_pt_eta_mask & photon_pixelSeed_electronVeto_mask & photonID_relaxed]
    photons['mediumPhoton'] = (photon_pt_eta_mask & photon_pixelSeed_electronVeto_mask & photon_mediumID)
    photons['mediumPhoton_noSieie_noChIso'] = (photon_pt_eta_mask & photon_pixelSeed_electronVeto_mask & mediumPhoton_noSieie_noChIso)
    photons['mediumPhoton_noChIso'] = (photon_pt_eta_mask & photon_pixelSeed_electronVeto_mask & mediumPhoton_noChIso)
    mediumPhotons = photons[photon_pt_eta_mask & photon_pixelSeed_electronVeto_mask & photon_mediumID]

def categorizeGenPhoton(photons):    #currently unused
    """A helper function to categorize MC reconstructed photons

    Returns an integer array to label them as either a generated true photon (1),
    a mis-identified generated electron (2), a photon from a hadron decay (3),
    or a fake (e.g. from pileup) (4).
    Taken from TTGamma processor used in CMSDAS 2023
    """
    #### Photon categories, using pdgID of the matched gen particle for the leading photon in the event
    # reco photons matched to a generated photon
    # if matched_gen is None (i.e. no match), then we set the flag False
    matchedPho = ak.fill_none(photons.matched_gen.pdgId == 22, False)
    # reco photons really generated as electrons
    matchedEle = ak.fill_none(abs(photons.matched_gen.pdgId) == 11, False)
    # if the gen photon has a PDG ID > 25 in its history, it has a hadronic parent
    hadronicParent = ak.fill_none(photons.matched_gen.maxParent > 25, False)

    # define the photon categories for tight photon events
    # a genuine photon is a reconstructed photon which is matched to a generator level photon, and does not have a hadronic parent
    isGenPho = matchedPho & ~hadronicParent
    # a hadronic photon is a reconstructed photon which is matched to a generator level photon, but has a hadronic parent
    isHadPho = matchedPho & hadronicParent
    # a misidentified electron is a reconstructed photon which is matched to a generator level electron
    isMisIDele = matchedEle
    # a hadronic/fake photon is a reconstructed photon that does not fall within any of the above categories
    isHadFake = ~isMisIDele & ~isHadPho & ~isGenPho

    # integer definition for the photon category axis
    # since false = 0 , true = 1, this only leaves the integer value of the category it falls into
    return 1 * isGenPho + 2 * isMisIDele + 3 * isHadPho + 4 * isHadFake

def isClean(obj_A, obj_B, drmin=0.4):
    objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_DR > drmin, True)
    return (mask)

def is_prompt_photon(genpart_collection,photon_obj):

    #let's first define a mask to make sure our reco photon is matched to a true gen level photon
    is_true_photon = ak.fill_none(abs(photon_obj.matched_gen.pdgId)==22,False)

    true_photon = photon_obj[is_true_photon]  #collection of reco photon that is matched to a true gen photon

    #first use genPartFlav
    ph_prompt_match = (true_photon.genPartFlav == 1)

    #Next, let's look at genPartIdx of true_photon collection
    genpartidx_of_true_photon = true_photon.genPartIdx
    genparticles_at_genpartidx = genpart_collection[genpartidx_of_true_photon]
    mother_of_gen_particle = genparticles_at_genpartidx.parent

    #Now let's loop over the parentage history of the genparticles_at_genpartidx collection and see if there is a hadron in the chain. If there is one, it is non-prompt photon
    mother_is_not_hadron = True
    while not ak.all(ak.is_none(mother_of_gen_particle, axis=1)):
        mother_is_not_hadron = (mother_is_not_hadron & ( (ak.fill_none(abs(mother_of_gen_particle.pdgId), 0) < 37) ))
        mother_of_gen_particle = mother_of_gen_particle.parent

    is_prompt_photon = (ph_prompt_match & mother_is_not_hadron)
    has_prompt_photon = ak.any(is_prompt_photon,axis=1) #WARNING: The reason this is probably fine is cause we eventually require that the event has exactly 1 photon

    return has_prompt_photon
