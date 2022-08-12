#!/usr/bin/env python
import lz4.frame as lz4f
import cloudpickle
import json
import pprint
import copy
import coffea
import numpy as np
from numba import njit
from numba.typed import List
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import hist, processor
from coffea.util import load, save
from optparse import OptionParser
from coffea.analysis_tools import PackedSelection
from coffea.lumi_tools import LumiMask

import topcoffea.modules.utils as utils
from topcoffea.modules.GetValuesFromJsons import get_param
from topcoffea.modules.objects import *
from topcoffea.modules.corrections import SFevaluator, GetBTagSF, ApplyJetCorrections, GetBtagEff, AttachMuonSF, AttachElectronSF, AttachPerLeptonFR, GetPUSF, ApplyRochesterCorrections, ApplyJetSystematics, AttachPSWeights, AttachPdfWeights, AttachScaleWeights, GetTriggerSF
from topcoffea.modules.selection import *
from topcoffea.modules.HistEFT import HistEFT
from topcoffea.modules.paths import topcoffea_path
import topcoffea.modules.eft_helper as efth


# Compares run:lumi:event for this chunk agianst a given reference
# Makes a mask that is True for the events that are present in the ref list
@njit
def construct_mask(runlumievt_test_arr,runlumievt_ref_lst):
    out_mask = List()
    for runlumievt_tup in runlumievt_test_arr:
        if runlumievt_tup in runlumievt_ref_lst:
            out_mask.append(True)
        else:
            out_mask.append(False)
    return out_mask


# Takes a list of run:lumi:event strings, returns as a list of tuples
def get_runlumievent_tup_from_strs(in_lst):
    out_lst = []
    for rle_str in in_lst:
        run,lumi,event = rle_str.split(":")
        out_lst.append((int(run),int(lumi),int(event)))
    return out_lst


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, ecut_threshold=None, do_errors=False, do_systematics=False, split_by_lepton_flavor=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        # Create the histograms
        self._accumulator = processor.dict_accumulator({

            # Counting things
            "ntightleps" : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("ntightleps", "n tight lep", 5, 0, 5)),
            "njets"      : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("njets", "n jets", 10, 0, 10)),

            # Lepton variables
            "l0_pt"  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l0_pt", "l0 pt", 20, 0, 500)),
            "l1_pt"  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l1_pt", "l1 pt", 20, 0, 500)),
            "l2_pt"  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l2_pt", "l2 pt", 20, 0, 500)),
            "l0_eta" : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l0_eta", "l0 eta", 12, -3.0, 3.0)),
            "l1_eta" : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l1_eta", "l1 eta", 12, -3.0, 3.0)),
            "l2_eta" : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l2_eta", "l2 eta", 12, -3.0, 3.0)),
            "l0_mvaTTH"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l0_mvaTTH", "l0 mvaTTH", 20, 0, 1.0)),
            "l1_mvaTTH"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l1_mvaTTH", "l1 mvaTTH", 20, 0, 1.0)),
            "l2_mvaTTH"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l2_mvaTTH", "l2 mvaTTH", 20, 0, 1.0)),
            "l0_mvaTTHUL" : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l0_mvaTTHUL", "l0 mvaTTHUL", 20, 0.7, 1.0)),
            "l1_mvaTTHUL" : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l1_mvaTTHUL", "l1 mvaTTHUL", 20, 0.7, 1.0)),
            "l2_mvaTTHUL" : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l2_mvaTTHUL", "l2 mvaTTHUL", 20, 0.7, 1.0)),

            "l0_btagDeepFlavB"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l0_btagDeepFlavB", "l0 matched j btagDeepFlavB", 10, 0, 0.5)),
            "l1_btagDeepFlavB"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l1_btagDeepFlavB", "l1 matched j btagDeepFlavB", 10, 0, 0.5)),
            "l2_btagDeepFlavB"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l2_btagDeepFlavB", "l1 matched j btagDeepFlavB", 10, 0, 0.5)),

            "l0_lostHits"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l0_lostHits", "l0 lostHits", 5, 0, 5.0)),
            "l1_lostHits"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l1_lostHits", "l1 lostHits", 5, 0, 5.0)),
            "l2_lostHits"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l2_lostHits", "l1 lostHits", 5, 0, 5.0)),

            "l0_jetRelIso"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l0_jetRelIso", "l0 jetRelIso", 20, 0, 1.0)),
            "l1_jetRelIso"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l1_jetRelIso", "l1 jetRelIso", 20, 0, 1.0)),
            "l2_jetRelIso"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("l2_jetRelIso", "l1 jetRelIso", 20, 0, 1.0)),


            # Jet variables

            "j0_pt"  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("j0_pt", "j0 pt", 20, 0, 300)),
            "j1_pt"  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("j1_pt", "j1 pt", 20, 0, 300)),
            "j2_pt"  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("j2_pt", "j2 pt", 20, 0, 300)),
            "j3_pt"  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("j3_pt", "j3 pt", 20, 0, 300)),

            "j0_eta"  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("j0_eta", "j0 eta", 10, -3.0, 3.0)),
            "j1_eta"  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("j1_eta", "j1 eta", 10, -3.0, 3.0)),
            "j2_eta"  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("j2_eta", "j2 eta", 10, -3.0, 3.0)),
            "j3_eta"  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("j3_eta", "j3 eta", 10, -3.0, 3.0)),

            "j0_btagDeepFlavB"  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("j0_btagDeepFlavB", "j0 btagDeepFlavB", 15, 0, 1.0)),
            "j1_btagDeepFlavB"  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("j1_btagDeepFlavB", "j1 btagDeepFlavB", 15, 0, 1.0)),
            "j2_btagDeepFlavB"  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("j2_btagDeepFlavB", "j2 btagDeepFlavB", 15, 0, 1.0)),
            "j3_btagDeepFlavB"  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("j3_btagDeepFlavB", "j3 btagDeepFlavB", 15, 0, 1.0)),

        })

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, events):

        # Dataset parameters
        dataset = events.metadata["dataset"]

        isData             = self._samples[dataset]["isData"]
        histAxisName       = self._samples[dataset]["histAxisName"]
        year               = self._samples[dataset]["year"]

        datasets = ["SingleMuon", "SingleElectron", "EGamma", "MuonEG", "DoubleMuon", "DoubleElectron", "DoubleEG"]
        for d in datasets: 
            if d in dataset: dataset = dataset.split('_')[0]

        # Set the sampleType (used for MC matching requirement)
        sampleType = "prompt"
        if isData:
            sampleType = "data"
        elif histAxisName in get_param("conv_samples"):
            sampleType = "conversions"
        elif histAxisName in get_param("prompt_and_conv_samples"):
            # Just DY (since we care about prompt DY for Z CR, and conv DY for 3l CR)
            sampleType = "prompt_and_conversions"

        # Initialize objects
        met  = events.MET
        e    = events.Electron
        mu   = events.Muon
        tau  = events.Tau
        jets = events.Jet

        e["idEmu"] = ttH_idEmu_cuts_E3(e.hoe, e.eta, e.deltaEtaSC, e.eInvMinusPInv, e.sieie)
        e["conept"] = coneptElec(e.pt, e.mvaTTHUL, e.jetRelIso)
        mu["conept"] = coneptMuon(mu.pt, mu.mvaTTHUL, mu.jetRelIso, mu.mediumId)
        e["btagDeepFlavB"] = ak.fill_none(e.matched_jet.btagDeepFlavB, -99)
        mu["btagDeepFlavB"] = ak.fill_none(mu.matched_jet.btagDeepFlavB, -99)

        if not isData:
            e["gen_pdgId"] = ak.fill_none(e.matched_gen.pdgId, 0)
            mu["gen_pdgId"] = ak.fill_none(mu.matched_gen.pdgId, 0)

        ################### Electron selection ####################

        e["isPres"] = isPresElec(e.pt, e.eta, e.dxy, e.dz, e.miniPFRelIso_all, e.sip3d, getattr(e,"mvaFall17V2noIso_WPL"))
        e["isLooseE"] = isLooseElec(e.miniPFRelIso_all,e.sip3d,e.lostHits)
        e["isFO"] = isFOElec(e.pt, e.conept, e.btagDeepFlavB, e.idEmu, e.convVeto, e.lostHits, e.mvaTTHUL, e.jetRelIso, e.mvaFall17V2noIso_WP90, year)
        e["isTightLep"] = tightSelElec(e.isFO, e.mvaTTHUL)      

        ################### Muon selection ####################

        mu["pt"] = ApplyRochesterCorrections(year, mu, isData) # Need to apply corrections before doing muon selection
        mu["isPres"] = isPresMuon(mu.dxy, mu.dz, mu.sip3d, mu.eta, mu.pt, mu.miniPFRelIso_all)
        mu["isLooseM"] = isLooseMuon(mu.miniPFRelIso_all,mu.sip3d,mu.looseId)
        mu["isFO"] = isFOMuon(mu.pt, mu.conept, mu.btagDeepFlavB, mu.mvaTTHUL, mu.jetRelIso, year)
        mu["isTightLep"]= tightSelMuon(mu.isFO, mu.mediumId, mu.mvaTTHUL)

        ################### Loose selection ####################

        m_loose = mu[mu.isPres & mu.isLooseM]
        e_loose = e[e.isPres & e.isLooseE]
        l_loose = ak.with_name(ak.concatenate([e_loose, m_loose], axis=1), 'PtEtaPhiMCandidate')

        ################### Tau selection ####################

        tau["isPres"]  = isPresTau(tau.pt, tau.eta, tau.dxy, tau.dz, tau.idDeepTau2017v2p1VSjet, minpt=20)
        tau["isClean"] = isClean(tau, l_loose, drmin=0.3)
        tau["isGood"]  =  tau["isClean"] & tau["isPres"]
        tau = tau[tau.isGood] # use these to clean jets
        tau["isTight"] = isTightTau(tau.idDeepTau2017v2p1VSjet) # use these to veto

        ################### FO selection ####################

        # Compute pair invariant masses, for all flavors all signes
        llpairs = ak.combinations(l_loose, 2, fields=["l0","l1"])
        events["minMllAFAS"] = ak.min( (llpairs.l0+llpairs.l1).mass, axis=-1)

        # Build FO collection
        m_fo = mu[mu.isPres & mu.isLooseM & mu.isFO]
        e_fo = e[e.isPres & e.isLooseE & e.isFO]

        # Attach the lepton SFs to the electron and muons collections
        AttachElectronSF(e_fo,year=year)
        AttachMuonSF(m_fo,year=year)

        # Attach per lepton fake rates
        AttachPerLeptonFR(e_fo, flavor = "Elec", year=year)
        AttachPerLeptonFR(m_fo, flavor = "Muon", year=year)
        m_fo['convVeto'] = ak.ones_like(m_fo.charge); 
        m_fo['lostHits'] = ak.zeros_like(m_fo.charge); 
        l_fo = ak.with_name(ak.concatenate([e_fo, m_fo], axis=1), 'PtEtaPhiMCandidate')
        l_fo_conept_sorted = l_fo[ak.argsort(l_fo.conept, axis=-1,ascending=False)]

        #################### Jets ####################

        # Jet cleaning, before any jet selection
        vetos_tocleanjets = ak.with_name( ak.concatenate([tau, l_fo], axis=1), "PtEtaPhiMCandidate")
        #vetos_tocleanjets = ak.with_name( l_fo, "PtEtaPhiMCandidate")
        tmp = ak.cartesian([ak.local_index(jets.pt), vetos_tocleanjets.jetIdx], nested=True)
        cleanedJets = jets[~ak.any(tmp.slot0 == tmp.slot1, axis=-1)] # this line should go before *any selection*, otherwise lep.jetIdx is not aligned with the jet index

        # Selecting jets and cleaning them
        jetptname = "pt_nom" if hasattr(cleanedJets, "pt_nom") else "pt"
        cleanedJets["isGood"] = isTightJet(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, jetPtCut=30.) # temporary at 25 for synch, TODO: Do we want 30 or 25?
        goodJets = cleanedJets[cleanedJets.isGood]

        # Count jets
        njets = ak.num(goodJets)
        ht = ak.sum(goodJets.pt,axis=-1)
        goodJets_ptsorted_4jpadded = ak.pad_none(goodJets[ak.argsort(goodJets.pt, axis=-1, ascending=False)],4)
        j0 = goodJets_ptsorted_4jpadded[:,0]
        j1 = goodJets_ptsorted_4jpadded[:,1]
        j2 = goodJets_ptsorted_4jpadded[:,2]
        j3 = goodJets_ptsorted_4jpadded[:,3]


        # Loose DeepJet WP
        if year == "2017":
            btagwpl = get_param("btag_wp_loose_UL17")
            btagwpm = get_param("btag_wp_medium_UL17")
        elif year == "2018":
            btagwpl = get_param("btag_wp_loose_UL18")
            btagwpm = get_param("btag_wp_medium_UL18")
        elif year=="2016":
            btagwpl = get_param("btag_wp_loose_UL16")          
            btagwpm = get_param("btag_wp_medium_UL16")
        elif year=="2016APV":
            btagwpl = get_param("btag_wp_loose_UL16APV")
            btagwpm = get_param("btag_wp_medium_UL16APV")
        else:
            raise ValueError(f"Error: Unknown year \"{year}\".")

        isBtagJetsLoose = (goodJets.btagDeepFlavB > btagwpl)
        nbtagsl = ak.num(goodJets[isBtagJetsLoose])

        isBtagJetsMedium = (goodJets.btagDeepFlavB > btagwpm)
        nbtagsm = ak.num(goodJets[isBtagJetsMedium])


        #################### Add variables into event object so that they persist ####################

        # Put njets and l_fo_conept_sorted into events
        events["njets"] = njets
        events["l_fo_conept_sorted"] = l_fo_conept_sorted

        # The event selection
        add2lMaskAndSFs(events, year, isData, sampleType)
        add3lMaskAndSFs(events, year, isData, sampleType)
        add4lMaskAndSFs(events, year, isData)
        addLepCatMasks(events)

        # Convenient to have l0, l1, l2 on hand
        l_fo_conept_sorted_padded = ak.pad_none(l_fo_conept_sorted, 3)
        l0 = l_fo_conept_sorted_padded[:,0]
        l1 = l_fo_conept_sorted_padded[:,1]
        l2 = l_fo_conept_sorted_padded[:,2]

        ######### Masks we need for the selection ##########

        # Get the lumi mask for data
        if year == "2016" or year == "2016APV":
            golden_json_path = topcoffea_path("data/goldenJsons/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")
        elif year == "2017":
            golden_json_path = topcoffea_path("data/goldenJsons/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt")
        elif year == "2018":
            golden_json_path = topcoffea_path("data/goldenJsons/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt")
        else:
            raise ValueError(f"Error: Unknown year \"{year}\".")
        lumi_mask = LumiMask(golden_json_path)(events.run,events.luminosityBlock)

        # Get mask for events that have two sf os leps close to z peak
        sfosz_3l_mask = get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:3],pt_window=10.0)

        # Pass trigger mask
        pass_trg = trgPassNoOverlap(events,isData,dataset,str(year))

        # b jet masks
        bmask_atleast1med_atleast2loose = ((nbtagsm>=1)&(nbtagsl>=2)) # Used for 2lss and 4l
        bmask_exactly1med = (nbtagsm==1) # Used for 3l SR and 2lss CR
        bmask_atleast2med = (nbtagsm>=2) # Used for 3l SR
        bmask_atmost2med  = (nbtagsm< 3) # Used to make 2lss mutually exclusive from tttt enriched
        bmask_atleast3med = (nbtagsm>=3) # Used for tttt enriched

        # Charge masks
        chargel0_p = ak.fill_none(((l0.charge)>0),False)
        chargel0_m = ak.fill_none(((l0.charge)<0),False)
        charge2l_1 = ak.fill_none(((l0.charge+l1.charge)!=0),False)
        charge3l_p = ak.fill_none(((l0.charge+l1.charge+l2.charge)>0),False)
        charge3l_m = ak.fill_none(((l0.charge+l1.charge+l2.charge)<0),False)

        # Store the masks for each of the 11 signal region categories using the PackedSelection object
        selections = PackedSelection(dtype='uint64')
        # 2lss selection
        selections.add("2lss_p_2b", (events.is2l & events.is2l_SR & chargel0_p & charge2l_1 & bmask_atleast1med_atleast2loose & bmask_atmost2med  & (njets>=4) & pass_trg))
        selections.add("2lss_m_2b", (events.is2l & events.is2l_SR & chargel0_m & charge2l_1 & bmask_atleast1med_atleast2loose & bmask_atmost2med  & (njets>=4) & pass_trg))
        selections.add("2lss_p_3b", (events.is2l & events.is2l_SR & chargel0_p & charge2l_1 & bmask_atleast1med_atleast2loose & bmask_atleast3med & (njets>=4) & pass_trg))
        selections.add("2lss_m_3b", (events.is2l & events.is2l_SR & chargel0_m & charge2l_1 & bmask_atleast1med_atleast2loose & bmask_atleast3med & (njets>=4) & pass_trg))
        # 3l selection
        selections.add("3l_p_offZ_1b", (events.is3l & events.is3l_SR & charge3l_p & ~sfosz_3l_mask & bmask_exactly1med & (njets>=2) & pass_trg))
        selections.add("3l_m_offZ_1b", (events.is3l & events.is3l_SR & charge3l_m & ~sfosz_3l_mask & bmask_exactly1med & (njets>=2) & pass_trg))
        selections.add("3l_p_offZ_2b", (events.is3l & events.is3l_SR & charge3l_p & ~sfosz_3l_mask & bmask_atleast2med & (njets>=2) & pass_trg))
        selections.add("3l_m_offZ_2b", (events.is3l & events.is3l_SR & charge3l_m & ~sfosz_3l_mask & bmask_atleast2med & (njets>=2) & pass_trg))
        selections.add("3l_onZ_1b",    (events.is3l & events.is3l_SR &               sfosz_3l_mask & bmask_exactly1med & (njets>=2) & pass_trg))
        selections.add("3l_onZ_2b",    (events.is3l & events.is3l_SR &               sfosz_3l_mask & bmask_atleast2med & (njets>=2) & pass_trg))
        # 4l selection
        selections.add("4l", (events.is4l & events.is4l_SR & bmask_atleast1med_atleast2loose & (njets>=2) & pass_trg))

        sr_category_lst = [
            "2lss_m_2b",
            "2lss_p_2b",
            "2lss_m_3b",
            "2lss_p_3b",
            "3l_onZ_1b",
            "3l_onZ_2b",
            "3l_p_offZ_1b",
            "3l_m_offZ_1b",
            "3l_p_offZ_2b",
            "3l_m_offZ_2b",
            "4l",
        ]

        ######### Get topcoffea SR mask and tight leptons ##########

        # Get the tight leptons
        tight_lep_mask = ak.fill_none(l_fo_conept_sorted_padded.isTightLep,False)
        tight_lep = l_fo_conept_sorted_padded[tight_lep_mask]

        # Construct a mask for events that passes the selection for any of our signal regions
        mask_topcoffea = selections.any(*sr_category_lst)
        mask_topcoffea_onZ = selections.any("3l_onZ_1b","3l_onZ_2b")
        if isData: mask_topcoffea = (mask_topcoffea & lumi_mask)
        if isData: mask_topcoffea_onZ = (mask_topcoffea_onZ & lumi_mask)


        ######### Get masks from the ttH legacy and UL selections ##########

        # Get the run, lumi, event info
        run = events.run
        lumi = events.luminosityBlock
        event = events.event
        rle_tup_arr = ak.to_list(ak.zip([run,lumi,event]))

        # Get the events from the ttH event lists
        ttH_leg_3lonZ  = topcoffea_path("data/eventLists/events_ttH_analysis_legacy_3l_onZ_aug02_2022.txt")
        ttH_uleg_3lonZ = topcoffea_path("data/eventLists/events_ttH_analysis_ultralegacy_3l_onZ_aug02_2022.txt")
        events_ttHanalysis_leg_3lonZ  = set(get_runlumievent_tup_from_strs(utils.read_lines_from_file(ttH_leg_3lonZ)))
        events_ttHanalysis_uleg_3lonZ = set(get_runlumievent_tup_from_strs(utils.read_lines_from_file(ttH_uleg_3lonZ)))
        events_3lonZ_uleg_unique = events_ttHanalysis_uleg_3lonZ.difference(events_ttHanalysis_leg_3lonZ)
        events_3lonZ_uleg_common = events_ttHanalysis_uleg_3lonZ.intersection(events_ttHanalysis_leg_3lonZ)
        events_3lonZ_leg_unique  = events_ttHanalysis_leg_3lonZ.difference(events_ttHanalysis_uleg_3lonZ)

        # Masks for the common events (i.e. in both ul and leg) and unique events (i.e. in ul but not leg)
        mask_3lonZ_ul         = (ak.Array(construct_mask(List(rle_tup_arr),List(events_ttHanalysis_uleg_3lonZ))) & pass_trg)
        mask_3lonZ_ul_unique  = (ak.Array(construct_mask(List(rle_tup_arr),List(events_3lonZ_uleg_unique))) & pass_trg)
        mask_3lonZ_ul_common  = (ak.Array(construct_mask(List(rle_tup_arr),List(events_3lonZ_uleg_common))) & pass_trg)
        mask_3lonZ_leg_unique = (ak.Array(construct_mask(List(rle_tup_arr),List(events_3lonZ_leg_unique)))  & pass_trg)



        ######### Calculating variables ##########

        #print("test_mask",mask_3lonZ_ul_unique)
        #print("test_mask",mask_3lonZ_ul_common)
        #print("tight_lep",tight_lep.pt)
        #print("")

        # Count tight leptons before padding
        ntightleps = ak.num(tight_lep)

        # Convenient to have l0, l1, l2 on hand
        tight_lep = ak.pad_none(tight_lep, 3)
        l0 = tight_lep[:,0]
        l1 = tight_lep[:,1]
        l2 = tight_lep[:,2]


        ######### Filling the histo ##########

        # Variables we will loop over when filling hists
        varnames = {

            # Multiplicities
            "ntightleps": ntightleps,
            "njets"     : njets,

            # Leptons
            "l0_pt": l0.pt,
            "l1_pt": l1.pt,
            "l2_pt": l2.pt,
            "l0_eta": l0.eta,
            "l1_eta": l1.eta,
            "l2_eta": l2.eta,
            "l0_mvaTTH"   : l0.mvaTTH,
            "l1_mvaTTH"   : l1.mvaTTH,
            "l2_mvaTTH"   : l2.mvaTTH,
            "l0_mvaTTHUL" : l0.mvaTTHUL,
            "l1_mvaTTHUL" : l1.mvaTTHUL,
            "l2_mvaTTHUL" : l2.mvaTTHUL,

            "l0_btagDeepFlavB" : l0.btagDeepFlavB,
            "l1_btagDeepFlavB" : l1.btagDeepFlavB,
            "l2_btagDeepFlavB" : l2.btagDeepFlavB,

            "l0_lostHits" : l0.lostHits,
            "l1_lostHits" : l1.lostHits,
            "l2_lostHits" : l2.lostHits,

            "l0_jetRelIso" : l0.jetRelIso,
            "l1_jetRelIso" : l1.jetRelIso,
            "l2_jetRelIso" : l2.jetRelIso,

            # Jets
            "j0_pt" : j0.pt,
            "j1_pt" : j1.pt,
            "j2_pt" : j2.pt,
            "j3_pt" : j3.pt,
            "j0_eta" : j0.eta,
            "j1_eta" : j1.eta,
            "j2_eta" : j2.eta,
            "j3_eta" : j3.eta,
            "j0_btagDeepFlavB" : j0.btagDeepFlavB,
            "j1_btagDeepFlavB" : j0.btagDeepFlavB,
            "j2_btagDeepFlavB" : j0.btagDeepFlavB,
            "j3_btagDeepFlavB" : j0.btagDeepFlavB,

        }

        # Selections to loop over, i.e. a list of masks to apply
        selection_dict = {
            "tthfrmwk_ul"            : mask_3lonZ_ul,
            "tthfrmwk_ul_common"     : mask_3lonZ_ul_common,
            "tthfrmwk_ul_unique"     : mask_3lonZ_ul_unique,
            "tthfrmwk_legacy_unique" : mask_3lonZ_leg_unique,
            "topcoffea"              : mask_topcoffea_onZ,
        }

        # Initialize the out object
        hout = self.accumulator.identity()

        # Loop over the hists we want to fill
        for dense_axis_name, dense_axis_vals in varnames.items():

            # Loop over the list of masks to apply
            for mask_name, mask in selection_dict.items():

                print("mask name:",mask_name)

                dense_vals_to_fill = dense_axis_vals[mask]
                dense_vals_to_fill = dense_vals_to_fill[~ak.is_none(dense_vals_to_fill)] # Get rid of None values (e.g. if this event passes the selection, but this variable is not defined for this event)

                # Fill the histos
                axes_fill_info_dict = {
                    dense_axis_name : dense_vals_to_fill,
                    "channel"       : mask_name,
                    "sample"        : histAxisName,
                }

                hout[dense_axis_name].fill(**axes_fill_info_dict)

        return hout


    def postprocess(self, accumulator):
        return accumulator
