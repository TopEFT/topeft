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

# Check if the values in an array are within a given range
def in_range_mask(in_var,lo_lim=None,hi_lim=None):

    # Make sure at least one of the cuts is not none
    if (lo_lim is None) and (hi_lim is None):
        raise Exception("Error: No cuts specified")

    # Check if the value is greater than the min
    if lo_lim is not None:
        above_min = (in_var > lo_lim)
    else:
        above_min = (ak.ones_like(in_var)==1)

    # Check if the value is less than or equal to the max
    if hi_lim is not None:
        below_max = (in_var <= hi_lim)
    else:
        below_max = (ak.ones_like(in_var)==1)

    # Return the mask
    return ak.fill_none((above_min & below_max),False)


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, dtype=np.float32):

        self._samples = samples
        self._dtype = dtype

        # Create the histograms
        self._accumulator = processor.dict_accumulator({
            "invmass" : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("kinematiccat", "kinematiccat"), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 100, 50, 150)),
            "njets"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("kinematiccat", "kinematiccat"), hist.Bin("njets", "njets", 10, 0, 10)),
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
        xsec               = self._samples[dataset]["xsec"]
        sow                = self._samples[dataset]["nSumOfWeights"]

        datasets = ["SingleMuon", "SingleElectron", "EGamma", "MuonEG", "DoubleMuon", "DoubleElectron", "DoubleEG"]
        for d in datasets: 
            if d in dataset: dataset = dataset.split('_')[0] 

        # Set the sampleType (used for MC matching requirement)
        # Does not really matter for this processor, but still need to pass it to the selection function anyway
        conversionDatasets=[x%y for x in ['TTGJets_centralUL%d'] for y in [16,17,18]]
        nonpromptDatasets =[x%y for x in ['TTJets_centralUL%d','DY50_centralUL%d','DY10to50_centralUL%d','tbarW_centralUL%d','tW_centralUL%d','tbarW_centralUL%d'] for y in [16,17,18]]
        sampleType = 'prompt'
        if isData:
            sampleType = 'data'
        elif dataset in conversionDatasets: 
            sampleType = 'conversions'
        elif dataset in nonpromptDatasets:
            sampleType = 'nonprompt'

        # Initialize objects
        met  = events.MET
        e    = events.Electron
        mu   = events.Muon
        jets = events.Jet

        e["idEmu"] = ttH_idEmu_cuts_E3(e.hoe, e.eta, e.deltaEtaSC, e.eInvMinusPInv, e.sieie)
        e["conept"] = coneptElec(e.pt, e.mvaTTH, e.jetRelIso)
        mu["conept"] = coneptMuon(mu.pt, mu.mvaTTH, mu.jetRelIso, mu.mediumId)
        e["btagDeepFlavB"] = ak.fill_none(e.matched_jet.btagDeepFlavB, -99)
        mu["btagDeepFlavB"] = ak.fill_none(mu.matched_jet.btagDeepFlavB, -99)

        if not isData:
            e["gen_pdgId"] = e.matched_gen.pdgId
            mu["gen_pdgId"] = mu.matched_gen.pdgId
            e["gen_parent_pdgId"] = e.matched_gen.distinctParent.pdgId
            mu["gen_parent_pdgId"] = mu.matched_gen.distinctParent.pdgId
            e["gen_gparent_pdgId"] = e.matched_gen.distinctParent.distinctParent.pdgId
            mu["gen_gparent_pdgId"] = mu.matched_gen.distinctParent.distinctParent.pdgId
        
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

        
        ################### Object selection ####################

        # Electron selection
        e["isPres"] = isPresElec(e.pt, e.eta, e.dxy, e.dz, e.miniPFRelIso_all, e.sip3d, getattr(e,"mvaFall17V2noIso_WPL"))
        e["isLooseE"] = isLooseElec(e.miniPFRelIso_all,e.sip3d,e.lostHits)
        e["isFO"] = isFOElec(e.conept, e.btagDeepFlavB, e.idEmu, e.convVeto, e.lostHits, e.mvaTTH, e.jetRelIso, e.mvaFall17V2noIso_WP80, year)
        e["isTightLep"] = tightSelElec(e.isFO, e.mvaTTH)
        # Muon selection
        mu["isPres"] = isPresMuon(mu.dxy, mu.dz, mu.sip3d, mu.eta, mu.pt, mu.miniPFRelIso_all)
        mu["isLooseM"] = isLooseMuon(mu.miniPFRelIso_all,mu.sip3d,mu.looseId)
        mu["isFO"] = isFOMuon(mu.pt, mu.conept, mu.btagDeepFlavB, mu.mvaTTH, mu.jetRelIso, year)
        mu["isTightLep"]= tightSelMuon(mu.isFO, mu.mediumId, mu.mvaTTH)
        # Build loose collections
        m_loose = mu[mu.isPres & mu.isLooseM]
        e_loose = e[e.isPres & e.isLooseE]
        l_loose = ak.with_name(ak.concatenate([e_loose, m_loose], axis=1), 'PtEtaPhiMCandidate')

        # Compute pair invariant masses, for all flavors all signes
        llpairs = ak.combinations(l_loose, 2, fields=["l0","l1"])
        events["minMllAFAS"] = ak.min( (llpairs.l0+llpairs.l1).mass, axis=-1)

        # Build FO collection
        m_fo = mu[mu.isPres & mu.isLooseM & mu.isFO]
        e_fo = e[e.isPres & e.isLooseE & e.isFO]

        # Attach the lepton SFs to the electron and muons collections (the event selection expect these to be present)
        AttachElectronSF(e_fo,year=year)
        AttachMuonSF(m_fo,year=year)

        # Attach per lepton fake rates
        AttachPerLeptonFR(e_fo, flavor = "Elec", year=year)
        AttachPerLeptonFR(m_fo, flavor = "Muon", year=year)
        m_fo['convVeto'] = ak.ones_like(m_fo.charge); 
        m_fo['lostHits'] = ak.zeros_like(m_fo.charge); 
        l_fo = ak.with_name(ak.concatenate([e_fo, m_fo], axis=1), 'PtEtaPhiMCandidate')
        l_fo_conept_sorted = l_fo[ak.argsort(l_fo.conept, axis=-1,ascending=False)]
        events["l_fo_conept_sorted"] = l_fo_conept_sorted

        # Convenient to have l0, l1, l2 on hand
        l_fo_conept_sorted_padded = ak.pad_none(l_fo_conept_sorted, 3)
        l0 = l_fo_conept_sorted_padded[:,0]
        l1 = l_fo_conept_sorted_padded[:,1]


        #################### Jets ####################

        # Jet cleaning, before any jet selection
        vetos_tocleanjets = ak.with_name( l_fo, "PtEtaPhiMCandidate")
        tmp = ak.cartesian([ak.local_index(jets.pt), vetos_tocleanjets.jetIdx], nested=True)
        cleanedJets = jets[~ak.any(tmp.slot0 == tmp.slot1, axis=-1)] # this line should go before *any selection*, otherwise lep.jetIdx is not aligned with the jet index

        # Selecting jets and cleaning them
        cleanedJets["isGood"] = isTightJet(getattr(cleanedJets, "pt"), cleanedJets.eta, cleanedJets.jetId, jetPtCut=30.) # temporary at 25 for synch, TODO: Do we want 30 or 25?
        goodJets = cleanedJets[cleanedJets.isGood]
        njets = ak.num(goodJets)


        #################### Event selection ####################

        # The event selection
        add2lMaskAndSFs(events, year, isData, sampleType)
        addLepCatMasks(events)


        ######### Weights ###########

        weights_object = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
        if not isData: weights_object.add("norm",(xsec/sow)*events["genWeight"])
        else: weights_object.add("norm",np.ones_like(events["event"]))

        # Apply the flip rate to OS as a cross check
        #weights_object.add("fliprate", events.flipfactor_2l)


        ######### Store boolean masks with PackedSelection ##########

        # Get mask for events that have two sf os leps close to z peak
        sfosz_2l_mask = get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:2],pt_window=30.0,flavor="os")
        sfssz_2l_mask = get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:2],pt_window=30.0,flavor="ss")

        # Pass trigger mask
        pass_trg = trgPassNoOverlap(events,isData,dataset,str(year))

        # Charge masks
        charge2l_0 = ak.fill_none(((l0.charge+l1.charge)==0),False)
        charge2l_1 = ak.fill_none(((l0.charge+l1.charge)!=0),False)

        # MC truth for flips
        flip_l0 = (l0.gen_pdgId == -l0.pdgId)
        flip_l1 = (l1.gen_pdgId == -l1.pdgId)
        from_g_l0 = (abs(l0.gen_parent_pdgId) == 22)
        from_g_l1 = (abs(l1.gen_parent_pdgId) == 22)
        from_z_l0 = (abs(l0.gen_parent_pdgId) == 23)
        from_z_l1 = (abs(l1.gen_parent_pdgId) == 23)
        from_h_l0 = (abs(l0.gen_parent_pdgId) == 25)
        from_h_l1 = (abs(l1.gen_parent_pdgId) == 25)
        from_tau_l0 = ((abs(l0.gen_parent_pdgId) == 15) & ((abs(l0.gen_gparent_pdgId) == 22) | (abs(l0.gen_gparent_pdgId) == 23) | (abs(l0.gen_gparent_pdgId) == 25)))
        from_tau_l1 = ((abs(l1.gen_parent_pdgId) == 15) & ((abs(l0.gen_gparent_pdgId) == 22) | (abs(l0.gen_gparent_pdgId) == 23) | (abs(l0.gen_gparent_pdgId) == 25)))
        isprompt_2l = ((from_z_l0 & from_z_l1) | (from_g_l0 & from_g_l1) | (from_h_l0 & from_h_l1) | (from_tau_l0 & from_tau_l1))
        truth_flip_mask = (isprompt_2l & (flip_l0 | flip_l1) & ~(flip_l0 & flip_l1)) # One or the other flips, but not both

        isprompt_2l_method2 = ( ((l0.genPartFlav==1) | (l0.genPartFlav == 15)) & ((l1.genPartFlav==1) | (l1.genPartFlav == 15)) )
        truth_flip_mask_method2 = (isprompt_2l_method2 & (flip_l0 | flip_l1) & ~(flip_l0 & flip_l1)) # One or the other flips, but not both

        # Print info about MC truth
        #print("isprompt_2l",isprompt_2l)
        #print("truth_flip_mask",truth_flip_mask)
        #for i,x in enumerate(truth_flip_mask):
        #    if x is not None:
        #        print("\n",i,x,l0[i].pdgId,l1[i].pdgId,l0[i].gen_pdgId,l1[i].gen_pdgId,l0[i].gen_parent_pdgId,l1[i].gen_parent_pdgId,l0[i].gen_gparent_pdgId,l1[i].gen_gparent_pdgId)
        #        print("from_z_l0, from_z_l1, from_g_l0, from_g_l1, from_tau_l0, from_tau_l1, isprompt_2l",from_z_l0[i], from_z_l1[i], from_g_l0[i], from_g_l1[i], from_tau_l0[i], from_tau_l1[i], isprompt_2l[i])
        #        print("truth_flip_mask",truth_flip_mask[i])

        # Selections
        selections = PackedSelection(dtype='uint64')
        selections.add("is_good_lumi",lumi_mask)
        selections.add("os",  (charge2l_0))
        selections.add("ss",  (charge2l_1))
        selections.add("osz", (charge2l_0 & sfosz_2l_mask))
        selections.add("ssz", (charge2l_1 & sfssz_2l_mask))
        selections.add("2e", (events.is2l_nozeeveto & events.is2l_SR & events.is_ee & (njets<4) & pass_trg))
        if not isData:
            selections.add("sszTruthFlip",  (charge2l_1 & sfssz_2l_mask & truth_flip_mask))
            selections.add("sszTruthFlip2", (charge2l_1 & sfssz_2l_mask & truth_flip_mask_method2))
            selections.add("ssTruthFlip2",  (charge2l_1 & truth_flip_mask_method2))

        # Masks for the pt ranges
        # Note that in_range_mask will be inclusive of abs(eta)=2.5
        # We want <2.5 (based on CMS AN-18-098)
        # But this is ok, since objects.py already requires <2.5
        selections.add("l0_E", in_range_mask(abs(l0.eta),lo_lim=1.479,hi_lim=2.5))
        selections.add("l1_E", in_range_mask(abs(l1.eta),lo_lim=1.479,hi_lim=2.5))
        selections.add("l0_B", in_range_mask(abs(l0.eta),lo_lim=None,hi_lim=1.479))
        selections.add("l1_B", in_range_mask(abs(l1.eta),lo_lim=None,hi_lim=1.479))

        # Masks for the pt ranges
        selections.add("l0_H", in_range_mask(l0.pt,lo_lim=50.0,hi_lim=None))
        selections.add("l1_H", in_range_mask(l1.pt,lo_lim=50.0,hi_lim=None))
        selections.add("l0_M", in_range_mask(l0.pt,lo_lim=25.0,hi_lim=50.0))
        selections.add("l1_M", in_range_mask(l1.pt,lo_lim=25.0,hi_lim=50.0))
        selections.add("l0_L", in_range_mask(l0.pt,lo_lim=15.0,hi_lim=25.0))
        selections.add("l1_L", in_range_mask(l1.pt,lo_lim=15.0,hi_lim=25.0))

        # Inclusive ranges for a cross check
        selections.add("l0_inclusive", (in_range_mask(l0.pt,lo_lim=15.0,hi_lim=None) & in_range_mask(abs(l0.eta),lo_lim=None,hi_lim=2.5)))
        selections.add("l1_inclusive", (in_range_mask(l1.pt,lo_lim=15.0,hi_lim=None) & in_range_mask(abs(l1.eta),lo_lim=None,hi_lim=2.5)))


        ######### Variables for the dense and sparse axes of the hists ##########

        dense_var_dict = {
            "invmass" : (l0+l1).mass,
            #"njets"   : njets,
        }

        # Cuts for the sparse "category" axis
        kinematic_cat_dict = {

            #"inclusive"  : ["l0_inclusive","l1_inclusive"], # Cross check that we're not missing anything

            "EH_EH" : ["l0_E","l0_H","l1_E","l1_H"],
            "BH_EH" : ["l0_B","l0_H","l1_E","l1_H"],
            "EM_EH" : ["l0_E","l0_M","l1_E","l1_H"],
            "BM_EH" : ["l0_B","l0_M","l1_E","l1_H"],
            "EL_EH" : ["l0_E","l0_L","l1_E","l1_H"],
            "BL_EH" : ["l0_B","l0_L","l1_E","l1_H"],

            "EH_BH" : ["l0_E","l0_H","l1_B","l1_H"],
            "BH_BH" : ["l0_B","l0_H","l1_B","l1_H"],
            "EM_BH" : ["l0_E","l0_M","l1_B","l1_H"],
            "BM_BH" : ["l0_B","l0_M","l1_B","l1_H"],
            "EL_BH" : ["l0_E","l0_L","l1_B","l1_H"],
            "BL_BH" : ["l0_B","l0_L","l1_B","l1_H"],

            "EH_EM" : ["l0_E","l0_H","l1_E","l1_M"],
            "BH_EM" : ["l0_B","l0_H","l1_E","l1_M"],
            "EM_EM" : ["l0_E","l0_M","l1_E","l1_M"],
            "BM_EM" : ["l0_B","l0_M","l1_E","l1_M"],
            "EL_EM" : ["l0_E","l0_L","l1_E","l1_M"],
            "BL_EM" : ["l0_B","l0_L","l1_E","l1_M"],

            "EH_BM" : ["l0_E","l0_H","l1_B","l1_M"],
            "BH_BM" : ["l0_B","l0_H","l1_B","l1_M"],
            "EM_BM" : ["l0_E","l0_M","l1_B","l1_M"],
            "BM_BM" : ["l0_B","l0_M","l1_B","l1_M"],
            "EL_BM" : ["l0_E","l0_L","l1_B","l1_M"],
            "BL_BM" : ["l0_B","l0_L","l1_B","l1_M"],

            "EH_EL" : ["l0_E","l0_H","l1_E","l1_L"],
            "BH_EL" : ["l0_B","l0_H","l1_E","l1_L"],
            "EM_EL" : ["l0_E","l0_M","l1_E","l1_L"],
            "BM_EL" : ["l0_B","l0_M","l1_E","l1_L"],
            "EL_EL" : ["l0_E","l0_L","l1_E","l1_L"],
            "BL_EL" : ["l0_B","l0_L","l1_E","l1_L"],

            "EH_BL" : ["l0_E","l0_H","l1_B","l1_L"],
            "BH_BL" : ["l0_B","l0_H","l1_B","l1_L"],
            "EM_BL" : ["l0_E","l0_M","l1_B","l1_L"],
            "BM_BL" : ["l0_B","l0_M","l1_B","l1_L"],
            "EL_BL" : ["l0_E","l0_L","l1_B","l1_L"],
            "BL_BL" : ["l0_B","l0_L","l1_B","l1_L"],
        }


        ########## Fill the histograms ##########

        hout = self.accumulator.identity()

        # Set the list of channels to loop over
        chan_lst = ["osz","ssz"]
        if not isData: chan_lst.append("sszTruthFlip")
        if not isData: chan_lst.append("sszTruthFlip2")

        #chan_lst = ["os","ss","ssTruthFlip2"]

        # Loop over histograms to fill (just invmass, njets for now)
        for dense_axis_name, dense_axis_vals in dense_var_dict.items():

            # Loop over the lepton channels
            for chan_name in chan_lst:

                # Loop over the kinematic categories
                for kinematic_cat_name in kinematic_cat_dict.keys():

                    # Get the cut mask object
                    cuts_lst = ["2e"]
                    cuts_lst.append(chan_name)
                    cuts_lst = cuts_lst + kinematic_cat_dict[kinematic_cat_name]
                    if isData: cuts_lst.append("is_good_lumi")
                    cuts_mask = selections.all(*cuts_lst)

                    # Fill the histo
                    axes_fill_info_dict = {
                        dense_axis_name  : dense_axis_vals[cuts_mask],
                        "channel"        : chan_name,
                        "kinematiccat"   : kinematic_cat_name,
                        "sample"         : histAxisName,
                        "weight"         : weights_object.weight()[cuts_mask],
                    }

                    hout[dense_axis_name].fill(**axes_fill_info_dict)

        return hout

    def postprocess(self, accumulator):
        return accumulator
