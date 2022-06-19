#!/usr/bin/env python
import lz4.frame as lz4f
import cloudpickle
import json
import pprint
import copy
import coffea
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import hist, processor
from coffea.util import load, save
from optparse import OptionParser
from coffea.analysis_tools import PackedSelection
from coffea.lumi_tools import LumiMask

from topcoffea.modules.GetValuesFromJsons import get_param
from topcoffea.modules.objects import *
from topcoffea.modules.corrections import SFevaluator, GetBTagSF, ApplyJetCorrections, GetBtagEff, AttachMuonSF, AttachElectronSF, AttachPerLeptonFR, GetPUSF, ApplyRochesterCorrections, ApplyJetSystematics, AttachPSWeights, AttachPdfWeights, AttachScaleWeights, GetTriggerSF
from topcoffea.modules.selection import *
from topcoffea.modules.HistEFT import HistEFT
from topcoffea.modules.paths import topcoffea_path
import topcoffea.modules.eft_helper as efth


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, ecut_threshold=None, do_errors=False, do_systematics=False, split_by_lepton_flavor=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        # Create the histograms
        self._accumulator = processor.dict_accumulator({
            "ljptsum" : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Bin("ljptsum", "S$_{T}$ (GeV)", 11, 0, 1100)),
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
        vetos_tocleanjets = ak.with_name( l_fo, "PtEtaPhiMCandidate")
        tmp = ak.cartesian([ak.local_index(jets.pt), vetos_tocleanjets.jetIdx], nested=True)
        cleanedJets = jets[~ak.any(tmp.slot0 == tmp.slot1, axis=-1)] # this line should go before *any selection*, otherwise lep.jetIdx is not aligned with the jet index

        # Selecting jets and cleaning them
        jetptname = "pt_nom" if hasattr(cleanedJets, "pt_nom") else "pt"
        cleanedJets["isGood"] = isTightJet(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, jetPtCut=30.) # temporary at 25 for synch, TODO: Do we want 30 or 25?
        goodJets = cleanedJets[cleanedJets.isGood]

        # Count jets
        njets = ak.num(goodJets)
        ht = ak.sum(goodJets.pt,axis=-1)
        j0 = goodJets[ak.argmax(goodJets.pt,axis=-1,keepdims=True)]

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


        ######### Finding extreme objects ##########

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

        # Construct a mask for events that passes the selection for any of our signal regions
        sr_event_mask = selections.any(*sr_category_lst)
        if isData: sr_event_mask = (sr_event_mask & lumi_mask)

        # Get an array of objects we're interested in, e.g. number of tight leptons
        # Note that this array is still the same lengh as the number of events in the chunk
        tight_lep_mask = ak.fill_none(l_fo_conept_sorted_padded.isTightLep,False)
        tight_lep = l_fo_conept_sorted_padded[tight_lep_mask]

        # Now throw out all events that do not pass the selection cuts
        # What we're left with now should <= len(number of events)
        tight_lep = tight_lep[sr_event_mask]

        # Now find the number of tight leptons in each event, this array should look something like e.g. [3,2,2,4,2,3,2]
        #   - In this example, this would mean 7 events in this chunk passed our event selection criteria
        #   - For the first event that passed, there are 3 tight leptons in the event, for the second event that passed, there are 2 tight leptons in the event, etc.
        nleps = ak.num(tight_lep)

        # Next steps:
        #   - Order the events by nleps (can use ak.argsort)
        #   - Take the top ones (maybe top 5?) and put them into the output object
        #   - Use a dataframe as the output object?
        #   - Not sure what sort of info we can put into the dataframe, but would probably be useful to know e.g. run number, lumiblock, event number (events.run, events.luminosityBlock, events.event)
        #   - Could also be interesting to store the flavor of the leptons (or check for e.g. max number of electrons or max number of mu instead of just max number of leptons)
        #   - Then we could also do something similar for njets, number of b tagged jets, or other event quantities e.g. S_T, or invmass (i.e. find the events with the most extreme values of these variables and accumulate them as well)



        ######### PLACEHOLDER Filling the histo ##########

        # We probably don't want to use a histo for this processor
        # But I'll leave the histo filling stuff here for now as a placeholder

        # Collection of all objects (leptons and jets)
        l_j_collection = ak.with_name(ak.concatenate([l_fo_conept_sorted,goodJets], axis=1),"PtEtaPhiMCollection")
        ljptsum = ak.sum(l_j_collection.pt,axis=-1)

        # Variables we will loop over when filling hists
        varnames = {}
        varnames["ljptsum"] = ljptsum

        # Initialize the out object
        hout = self.accumulator.identity()

        # Loop over the hists we want to fill
        for dense_axis_name, dense_axis_vals in varnames.items():

            # Loop over the channels in each nlep cat (e.g. "3l_m_offZ_1b")
            for sr_category in sr_category_lst:

                    cuts_lst = [sr_category]
                    all_cuts_mask = selections.all(*cuts_lst)

                    # Fill the histos
                    axes_fill_info_dict = {
                        dense_axis_name : dense_axis_vals[all_cuts_mask],
                        "channel"       : sr_category,
                        "sample"        : histAxisName,
                    }

                    hout[dense_axis_name].fill(**axes_fill_info_dict)

        return hout

    def postprocess(self, accumulator):
        return accumulator
