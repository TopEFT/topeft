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

def isPresTau(pt, eta, dxy, dz, idDeepTau2017v2p1VSjet, minpt=20.0):
    return  (pt>minpt)&(abs(eta)<get_te_param("eta_t_cut"))&(abs(dxy)<get_te_param("dxy_tau_cut"))&(abs(dz)<get_te_param("dz_tau_cut"))&(idDeepTau2017v2p1VSjet>>1 & 1 ==1)

def isVLooseTau(idDeepTau2017v2p1VSjet):
    return (idDeepTau2017v2p1VSjet>>2 & 1)

def isLooseTau(idDeepTau2017v2p1VSjet):
    return (idDeepTau2017v2p1VSjet>>3 & 1)

def isMediumTau(idDeepTau2017v2p1VSjet):
    return (idDeepTau2017v2p1VSjet>>4 & 1)

def isTightTau(idDeepTau2017v2p1VSjet):
    return (idDeepTau2017v2p1VSjet>>5 & 1)

def isVTightTau(idDeepTau2017v2p1VSjet):
    return (idDeepTau2017v2p1VSjet>>6 & 1)

def isVVTightTau(idDeepTau2017v2p1VSjet):
    return (idDeepTau2017v2p1VSjet>>7 & 1)

def ttH_idEmu_cuts_E3(hoe, eta, deltaEtaSC, eInvMinusPInv, sieie):
    return (hoe<(0.10-0.00*(abs(eta+deltaEtaSC)>1.479))) & (eInvMinusPInv>-0.04) & (sieie<(0.011+0.019*(abs(eta+deltaEtaSC)>1.479)))

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
    else:
        raise Exception(f"Error: Unknown year \"{year}\". Exiting...")

    x = np.minimum(np.maximum(0, jetpt - ptmin)/(ptmax-ptmin), 1.0)
    return x*wploose*scale_loose + (1-x)*wpmedium

def coneptElec(pt, mvaTTHUL, jetRelIso):
    conePt = (0.90 * pt * (1 + jetRelIso))
    return ak.where((mvaTTHUL>get_tc_param("mva_TTH_e_cut")),pt,conePt)

def coneptMuon(pt, mvaTTHUL, jetRelIso, mediumId):
    conePt = (0.90 * pt * (1 + jetRelIso))
    return ak.where(((mvaTTHUL>get_tc_param("mva_TTH_m_cut"))&(mediumId>0)),pt,conePt)

def isPresElec(pt, eta, dxy, dz, miniIso, sip3D, eleId):
    pt_mask    = (pt       > get_te_param("pres_e_pt_cut"))
    eta_mask   = (abs(eta) < get_te_param("eta_e_cut"))
    dxy_mask   = (abs(dxy) < get_te_param("dxy_cut"))
    dz_mask    = (abs(dz)  < get_te_param("dz_cut"))
    iso_mask   = (miniIso  < get_te_param("iso_cut"))
    sip3d_mask = (sip3D    < get_te_param("sip3d_cut"))
    return (pt_mask & eta_mask & dxy_mask & dz_mask & iso_mask & sip3d_mask & eleId)

def isPresMuon(dxy, dz, sip3D, eta, pt, miniRelIso):
    pt_mask    = (pt         > get_te_param("pres_m_pt_cut"))
    eta_mask   = (abs(eta)   < get_te_param("eta_m_cut"))
    dxy_mask   = (abs(dxy)   < get_te_param("dxy_cut"))
    dz_mask    = (abs(dz)    < get_te_param("dz_cut"))
    iso_mask   = (miniRelIso < get_te_param("iso_cut"))
    sip3d_mask = (sip3D      < get_te_param("sip3d_cut"))
    return (pt_mask & eta_mask & dxy_mask & dz_mask & iso_mask & sip3d_mask)

def isLooseElec(miniPFRelIso_all,sip3d,lostHits):
    return (miniPFRelIso_all<get_te_param("iso_cut")) & (sip3d<get_te_param("sip3d_cut")) & (lostHits<=1)

def isLooseMuon(miniPFRelIso_all,sip3d,looseId):
    return (miniPFRelIso_all<get_te_param("iso_cut")) & (sip3d<get_te_param("sip3d_cut")) & (looseId)

def isFOElec(pt, conept, jetBTagDeepFlav, ttH_idEmu_cuts_E3, convVeto, lostHits, mvaTTHUL, jetRelIso, mvaFall17V2noIso_WP90, year):

    # Get the btag cut for the year
    if (year == "2016"):
        bTagCut = get_tc_param("btag_wp_medium_UL16")
    elif (year == "2016APV"):
        bTagCut = get_tc_param("btag_wp_medium_UL16APV")
    elif (year == "2017"):
        bTagCut = get_tc_param("btag_wp_medium_UL17")
    elif (year == "2018"):
        bTagCut = get_tc_param("btag_wp_medium_UL18")
    else:
        raise Exception(f"Error: Unknown year \"{year}\". Exiting...")

    btabReq    = (jetBTagDeepFlav<bTagCut)
    ptReq      = (conept>get_te_param("fo_pt_cut"))
    qualityReq = (ttH_idEmu_cuts_E3 & convVeto & (lostHits==0))
    mvaReq     = ((mvaTTHUL>get_tc_param("mva_TTH_e_cut")) | ((mvaFall17V2noIso_WP90) & (jetBTagDeepFlav<smoothBFlav(0.9*pt*(1+jetRelIso),20,45,year)) & (jetRelIso < get_te_param("fo_e_jetRelIso_cut"))))

    return ptReq & btabReq & qualityReq & mvaReq

def isFOMuon(pt, conept, jetBTagDeepFlav, mvaTTHUL, jetRelIso, year):

    # Get the btag cut for the year
    if (year == "2016"):
        bTagCut = get_tc_param("btag_wp_medium_UL16")
    elif (year == "2016APV"):
        bTagCut = get_tc_param("btag_wp_medium_UL16APV")
    elif (year == "2017"):
        bTagCut = get_tc_param("btag_wp_medium_UL17")
    elif (year == "2018"):
        bTagCut = get_tc_param("btag_wp_medium_UL18")
    else:
        raise Exception(f"Error: Unknown year \"{year}\". Exiting...")

    btagReq = (jetBTagDeepFlav<bTagCut)
    ptReq   = (conept>get_te_param("fo_pt_cut"))
    mvaReq  = ((mvaTTHUL>get_tc_param("mva_TTH_m_cut")) | ((jetBTagDeepFlav<smoothBFlav(0.9*pt*(1+jetRelIso),20,45,year)) & (jetRelIso < get_te_param("fo_m_jetRelIso_cut"))))
    return ptReq & btagReq & mvaReq

def tightSelElec(clean_and_FO_selection_TTH, mvaTTHUL):
    return (clean_and_FO_selection_TTH) & (mvaTTHUL > get_tc_param("mva_TTH_e_cut"))

def tightSelMuon(clean_and_FO_selection_TTH, mediumId, mvaTTHUL):
    return (clean_and_FO_selection_TTH) & (mediumId>0) & (mvaTTHUL > get_tc_param("mva_TTH_m_cut"))

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
