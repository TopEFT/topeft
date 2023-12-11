#gen processor written for photon studies
#Created on October 6, 2023

#!/usr/bin/env python
import copy
import coffea
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import hist, processor
from coffea.util import load
from coffea.analysis_tools import PackedSelection
from coffea.lumi_tools import LumiMask

from topcoffea.modules.paths import topcoffea_path
from topcoffea.modules.HistEFT import HistEFT
import topcoffea.modules.eft_helper as efth
import topcoffea.modules.event_selection as tc_es
import topcoffea.modules.object_selection as tc_os

from topeft.modules.paths import topeft_path
from topeft.modules.corrections import GetBTagSF, ApplyJetCorrections, GetBtagEff, AttachMuonSF, AttachElectronSF, AttachPerLeptonFR, GetPUSF, ApplyRochesterCorrections, ApplyJetSystematics, AttachPSWeights, AttachScaleWeights, GetTriggerSF
import topeft.modules.event_selection as te_es
import topeft.modules.object_selection as te_os

from topcoffea.modules.get_param_from_jsons import GetParam
get_tc_param = GetParam(topcoffea_path("params/params.json"))
get_te_param = GetParam(topeft_path("params/params.json"))

class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, ecut_threshold=None, do_errors=False, do_systematics=False, split_by_lepton_flavor=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        # Create the histograms
        self._accumulator = processor.dict_accumulator({
            "genphoton_pt"  : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("genphoton_pt",     "$p_{T}$ $Gen\gamma$ (GeV)", 10, 0, 200)),
            "genlep_pt" : HistEFT("Events", wc_names_lst, hist.Cat("sample","sample"), hist.Bin("genlep_pt",             "$p_{T}$ $Gen lepton$ (GeV)", 20,0,200)),
        })

       # Set the list of hists to fill
        if hist_lst is None:
            # If the hist list is none, assume we want to fill all hists
            self._hist_lst = list(self._accumulator.keys())
        else:
            # Otherwise, just fill the specified subset of hists
            for hist_to_include in hist_lst:
                if hist_to_include not in self._accumulator.keys():
                    raise Exception(f"Error: Cannot specify hist \"{hist_to_include}\", it is not defined in the processor.")
            self._hist_lst = hist_lst # Which hists to fill

        # Set the booleans
        self._do_errors = do_errors # Whether to calculate and store the w**2 coefficients

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, events):

        ### Dataset parameters ###
        dataset = events.metadata["dataset"]

        isData             = self._samples[dataset]["isData"]
        histAxisName       = self._samples[dataset]["histAxisName"]
        year               = self._samples[dataset]["year"]
        xsec               = self._samples[dataset]["xsec"]
        sow                = self._samples[dataset]["nSumOfWeights"]

        if isData: raise Exception("Error: This processor is not for data")

        ### Get gen particles collection ###
        genPart = events.GenPart
        genJet = events.GenJet

        ## reco ele and muon collection ###
        ele = events.Electron
        mu  = events.Muon

        is_final_mask = genPart.hasFlags(["fromHardProcess","isLastCopy"])

        ################## gen matched lepton selection #######################
        #genEle_mask = np.abs(ele.matched_gen.pdgId)==11
        #genEle_mask = abs(genPart.pdgId) == 11
        ele_genpartFlavmask = (ele.genPartFlav == 1 | 15)
        mu_genpartFlavmask = (mu.genPartFlav == 1 | 15)
        #genEle = genPart[is_final_mask & genEle_mask]
        #genMu_mask = np.abs(mu.matched_gen.pdgId)==13
        #genMu_mask = abs(genPart.pdgId) == 13
        #genMu = genPart[is_final_mask & genMu_mask]
        genEle = ele[ele_genpartFlavmask]
        genMu = mu[mu_genpartFlavmask]
        genleps = ak.concatenate([genEle,genMu],axis=1)
        genlep_pt_mask = genleps.pt > 20         #Require that the lepton pt be at least 20 GeV.
        genlep_eta_mask = abs(genleps.eta)<2.4   #Require that the lepton be within 2.4 eta
        genlep_pt_eta_mask = ak.fill_none(ak.pad_none((genlep_pt_mask & genlep_eta_mask),1),False)
        genleps = genleps[genlep_pt_eta_mask]
        genleps_ptsorted = genleps[ak.argsort(genleps.pt,axis=-1,ascending=False)]
        genleps_ptsorted_padded = ak.pad_none(genleps_ptsorted, 2)
        pt2515 = (ak.any(genleps_ptsorted_padded[:,0:1].pt>25,axis=1) & ak.any(genleps_ptsorted_padded[:,1:2].pt>15,axis=1)) #Require that the leading(sub-leading) lepton pT be at least 25(15)GeV. 
        #genleps_ptsorted_padded = genleps_ptsorted_padded[pt2515]
        genl0 = genleps_ptsorted_padded[:,0]
        genl1 = genleps_ptsorted_padded[:,1]

        #pt2515 = (ak.any(genleps_ptsorted_padded[:,0:1].pt>25,axis=1) & ak.any(genleps_ptsorted_padded[:,1:2].pt>15,axis=1)) #Require that the leading(sub-leading) lepton pT be at least 25(15)GeV. 
        
        genleps_num_mask = ak.num(genleps_ptsorted) == 2 #Require that there are exactly 2 gen_matched leptons
        #genleps_chargesum_zero = ak.fill_none(((genl0.charge + genl1.charge)==0),False)     #Require that the two leptons are of opposite signs
        genleps_chargesum_zero = (np.sign(genl0.pdgId) == - np.sign(genl1.pdgId))
        genleps_chargesum_zero = ak.fill_none(genleps_chargesum_zero, False)
        #genleps_final_mask = (genleps_num_mask & pt2515 & genleps_chargesum_zero) 
        

        invmass_ll = (genl0+genl1).mass
        mll_min20 = ak.fill_none((invmass_ll > 20),False)       #Require that the invariant mass of the lepton pairs be at least 20 GeV.
        all_genleps_mask = (genleps_num_mask & pt2515 & genleps_chargesum_zero & mll_min20)

        ## gen Photon selection ##
        genPh_mask = abs(genPart.pdgId) == 22
        genPhoton = genPart[is_final_mask & genPh_mask]
        is_clean_photon = te_os.isClean(genPhoton, genEle,drmin=0.4) & te_os.isClean(genPhoton,genMu,drmin=0.4)      #photon isolation from leps
        genPhoton = genPhoton[is_clean_photon]

        events['genPhoton'] = genPhoton
        te_es.GenPhotonSelection(events)       #Require photon pt be 20 GeV and abs(eta) be 1.44
        genPhoton = genPhoton[events.genPhoton_pT_eta_mask]

        genphoton_num_mask = ak.num(genPhoton) == 1     #Require that there is exactly 1 photon



        ############### Jet object selection #####################
        genJet_pT_eta_mask = ((genJet.pt > 30) & (abs(genJet.eta) < 2.44))
        genJet_pT_eta_mask = ak.fill_none(ak.pad_none(genJet_pT_eta_mask,1),False)
        genjet = genJet[genJet_pT_eta_mask]          #Require that the jet pt be at least 30 GeV.
        is_clean_jet = te_os.isClean(genjet, genEle, drmin=0.4) & te_os.isClean(genjet, genMu, drmin=0.4) & te_os.isClean(genjet, genPhoton, drmin=0.1)
        genjet_clean = genjet[is_clean_jet]

        genjets_num_mask = ak.num(genjet_clean) >= 1     #Require that there is at least 1 jet

        ################## Gen matched b jets selection ########################
        genbJets_mask = np.abs(genjet_clean.hadronFlavour)==5        #bjet selection from genJet collection
        genbJets = genJet[genbJets_mask]
        genbJets_pT_eta_mask = ((genbJets.pt > 30) & (abs(genbJets.eta) < 2.44))
        genbJets_pT_eta_mask = ak.fill_none(ak.pad_none(genbJets_pT_eta_mask,1),False)
        genbJets = genbJets[genbJets_pT_eta_mask]

        #gen bJet multiplicity mask
        genbJets_num_mask = ak.num(genbJets) >= 1    #Require that there is at least 1 btagged jet

        ################### Define dense axis variables #######################
        genphoton_pt = ak.fill_none(ak.firsts(genPhoton.pt),-1)      #highest pt in the genphoton collection
        genl0pt = ak.fill_none(ak.firsts(genleps_ptsorted.pt),-1)    #highest pt in the genlep collection

        # Dictionary of dense axis values
        dense_axis_dict = {
            "genphoton_pt" : genphoton_pt,
            "genlep_pt" : genl0pt,
        }

        #### Get weights ###
        # Extract the EFT quadratic coefficients and optionally use them to calculate the coefficients on the w**2 quartic function
        # eft_coeffs is never Jagged so convert immediately to numpy for ease of use.
        eft_coeffs = ak.to_numpy(events["EFTfitCoefficients"]) if hasattr(events, "EFTfitCoefficients") else None
        if eft_coeffs is not None:
            # Check to see if the ordering of WCs for this sample matches what want
            if self._samples[dataset]["WCnames"] != self._wc_names_lst:
                eft_coeffs = efth.remap_coeffs(self._samples[dataset]["WCnames"], self._wc_names_lst, eft_coeffs)
        eft_w2_coeffs = efth.calc_w2_coeffs(eft_coeffs,self._dtype) if (self._do_errors and eft_coeffs is not None) else None

        #If this is not an eft sample, get the genWeight
        if eft_coeffs is None: genw = events["genWeight"]
        else: genw = np.ones_like(events["event"])
        lumi = 1000.0*get_tc_param(f"lumi_{year}")
        event_weight = lumi*(xsec/sow)*genw

        #Final mask relevant to the "category" we are interested in.
        #genph_2lOS_mask = (genleps_num_mask & genleps_chargesum_zero & genjets_num_mask & genbJets_num_mask & events.genPhoton_pT_eta_mask & genphoton_num_mask)
        genphoton_2lOS_mask = (all_genleps_mask & genjets_num_mask & genbJets_num_mask & genphoton_num_mask)
        nogenphoton_2lOS_mask = (all_genleps_mask & genjets_num_mask & genbJets_num_mask) 

        ##### Loop over the hists we want to fill ###########

        hout = self.accumulator.identity()

        for dense_axis_name, dense_axis_vals in dense_axis_dict.items():

            #Mask out the none values
            isnotnone_mask = (ak.fill_none((dense_axis_vals != None),False))
            #all_cuts_mask = ak.any(genph_2lOS_mask,axis=1)# & isnotnone_mask)
            if dense_axis_name == "genphoton_pt":
                all_cuts_mask = genphoton_2lOS_mask
            elif dense_axis_name == "genlep_pt":
                all_cuts_mask = nogenphoton_2lOS_mask
            dense_axis_vals_cut = dense_axis_vals[all_cuts_mask]
            event_weights_cut = event_weight[all_cuts_mask]
            eft_coeffs_cut = eft_coeffs
            if eft_coeffs is not None: eft_coeffs_cut = eft_coeffs[all_cuts_mask]
            eft_w2_coeffs_cut = eft_w2_coeffs
            if eft_w2_coeffs is not None: eft_w2_coeffs_cut = eft_w2_coeffs[all_cuts_mask]

            # Fill the histos
            axes_fill_info_dict = {
                dense_axis_name : dense_axis_vals_cut,
                "sample"        : histAxisName,
                "weight"        : event_weights_cut,
                "eft_coeff"     : eft_coeffs_cut,
                "eft_err_coeff" : eft_w2_coeffs_cut,
            }

            hout[dense_axis_name].fill(**axes_fill_info_dict)

        return hout

    def postprocess(self, accumulator):
        return accumulator
