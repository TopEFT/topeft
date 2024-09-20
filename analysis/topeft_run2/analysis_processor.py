#!/usr/bin/env python
import copy
import coffea
import numpy as np
import awkward as ak
import json

import hist
from topcoffea.modules.histEFT import HistEFT
from coffea import processor
from coffea.util import load
from coffea.analysis_tools import PackedSelection
from coffea.lumi_tools import LumiMask

from topcoffea.modules.paths import topcoffea_path
import topcoffea.modules.eft_helper as efth
import topcoffea.modules.event_selection as tc_es
import topcoffea.modules.object_selection as tc_os
import topcoffea.modules.corrections as tc_cor

from topeft.modules.axes import info as axes_info
from topeft.modules.paths import topeft_path
from topeft.modules.corrections import ApplyJetCorrections, GetBtagEff, AttachMuonSF, AttachElectronSF, AttachTauSF, ApplyTES, ApplyTESSystematic, ApplyFESSystematic, AttachPerLeptonFR, ApplyRochesterCorrections, ApplyJetSystematics, GetTriggerSF
import topeft.modules.event_selection as te_es
import topeft.modules.object_selection as te_os

from topcoffea.modules.get_param_from_jsons import GetParam
get_tc_param = GetParam(topcoffea_path("params/params.json"))
get_te_param = GetParam(topeft_path("params/params.json"))

np.seterr(divide='ignore', invalid='ignore', over='ignore')

# Takes strings as inputs, constructs a string for the full channel name
# Try to construct a channel name like this: [n leptons]_[lepton flavors]_[p or m charge]_[on or off Z]_[n b jets]_[n jets]
    # chan_str should look something like "3l_p_offZ_1b", NOTE: This function assumes nlep comes first
    # njet_str should look something like "atleast_5j",   NOTE: This function assumes njets comes last
    # flav_str should look something like "emm"
def construct_cat_name(chan_str,njet_str=None,flav_str=None):
    # Get the component strings
    nlep_str = chan_str.split("_")[0] # Assumes n leps comes first in the str
    chan_str = "_".join(chan_str.split("_")[1:]) # The rest of the channel name is everything that comes after nlep
    if chan_str == "": chan_str = None # So that we properly skip this in the for loop below
    if flav_str is not None:
        flav_str = flav_str
    if njet_str is not None:
        njet_str = njet_str[-2:] # Assumes number of n jets comes at the end of the string
        if "j" not in njet_str:
            # The njet string should really have a "j" in it
            raise Exception(f"Something when wrong while trying to consturct channel name, is \"{njet_str}\" an njet string?")

    # Put the component strings into the channel name
    ret_str = nlep_str
    for component in [flav_str,chan_str,njet_str]:
        if component is None: continue
        ret_str = "_".join([ret_str,component])
    return ret_str


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, ecut_threshold=None, do_errors=False, do_systematics=False, split_by_lepton_flavor=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32, rebin=False, offZ_split=False, tau_flag=False):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype
        self._offZ_split = offZ_split
        self.tau_flag = tau_flag

        proc_axis = hist.axis.StrCategory([], name="process", growth=True)
        chan_axis = hist.axis.StrCategory([], name="channel", growth=True)
        syst_axis = hist.axis.StrCategory([], name="systematic", label=r"Systematic Uncertainty", growth=True)
        appl_axis = hist.axis.StrCategory([], name="appl", label=r"AR/SR", growth=True)

        histograms = {}
        for name, info in axes_info.items():
            if not rebin and "variable" in info:
                dense_axis = hist.axis.Variable(
                    info["variable"], name=name, label=info["label"]
                )
                sumw2_axis = hist.axis.Variable(
                    info["variable"], name=name+"_sumw2", label=info["label"] + " sum of w^2"
                )
            else:
                dense_axis = hist.axis.Regular(
                    *info["regular"], name=name, label=info["label"]
                )
                sumw2_axis = hist.axis.Regular(
                    *info["regular"], name=name+"_sumw2", label=info["label"] + " sum of w^2"
                )
            histograms[name] = HistEFT(
                proc_axis,
                chan_axis,
                syst_axis,
                appl_axis,
                dense_axis,
                wc_names=wc_names_lst,
                label=r"Events",
                rebin=rebin
            )
            histograms[name+"_sumw2"] = HistEFT(
                proc_axis,
                chan_axis,
                syst_axis,
                appl_axis,
                sumw2_axis,
                wc_names=wc_names_lst,
                label=r"Events",
                rebin=rebin
            )
        self._accumulator = histograms

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

        # Set the energy threshold to cut on
        self._ecut_threshold = ecut_threshold

        # Set the booleans
        self._do_errors = do_errors # Whether to calculate and store the w**2 coefficients
        self._do_systematics = do_systematics # Whether to process systematic samples
        self._split_by_lepton_flavor = split_by_lepton_flavor # Whether to keep track of lepton flavors individually
        self._skip_signal_regions = skip_signal_regions # Whether to skip the SR categories
        self._skip_control_regions = skip_control_regions # Whether to skip the CR categories



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
        isEFT   = self._samples[dataset]["WCnames"] != []

        isData             = self._samples[dataset]["isData"]
        histAxisName       = self._samples[dataset]["histAxisName"]
        year               = self._samples[dataset]["year"]
        xsec               = self._samples[dataset]["xsec"]
        sow                = self._samples[dataset]["nSumOfWeights"]

        # Get up down weights from input dict
        if (self._do_systematics and not isData):
            if histAxisName in get_te_param("lo_xsec_samples"):
                # We have a LO xsec for these samples, so for these systs we will have e.g. xsec_LO*(N_pass_up/N_gen_nom)
                # Thus these systs will cover the cross section uncty and the acceptance and effeciency and shape
                # So no NLO rate uncty for xsec should be applied in the text data card
                sow_ISRUp          = self._samples[dataset]["nSumOfWeights"]
                sow_ISRDown        = self._samples[dataset]["nSumOfWeights"]
                sow_FSRUp          = self._samples[dataset]["nSumOfWeights"]
                sow_FSRDown        = self._samples[dataset]["nSumOfWeights"]
                sow_renormUp       = self._samples[dataset]["nSumOfWeights"]
                sow_renormDown     = self._samples[dataset]["nSumOfWeights"]
                sow_factUp         = self._samples[dataset]["nSumOfWeights"]
                sow_factDown       = self._samples[dataset]["nSumOfWeights"]
                sow_renormfactUp   = self._samples[dataset]["nSumOfWeights"]
                sow_renormfactDown = self._samples[dataset]["nSumOfWeights"]
            else:
                # Otherwise we have an NLO xsec, so for these systs we will have e.g. xsec_NLO*(N_pass_up/N_gen_up)
                # Thus these systs should only affect acceptance and effeciency and shape
                # The uncty on xsec comes from NLO and is applied as a rate uncty in the text datacard
                sow_ISRUp          = self._samples[dataset]["nSumOfWeights_ISRUp"          ]
                sow_ISRDown        = self._samples[dataset]["nSumOfWeights_ISRDown"        ]
                sow_FSRUp          = self._samples[dataset]["nSumOfWeights_FSRUp"          ]
                sow_FSRDown        = self._samples[dataset]["nSumOfWeights_FSRDown"        ]
                sow_renormUp       = self._samples[dataset]["nSumOfWeights_renormUp"       ]
                sow_renormDown     = self._samples[dataset]["nSumOfWeights_renormDown"     ]
                sow_factUp         = self._samples[dataset]["nSumOfWeights_factUp"         ]
                sow_factDown       = self._samples[dataset]["nSumOfWeights_factDown"       ]
                sow_renormfactUp   = self._samples[dataset]["nSumOfWeights_renormfactUp"   ]
                sow_renormfactDown = self._samples[dataset]["nSumOfWeights_renormfactDown" ]
        else:
            sow_ISRUp          = -1
            sow_ISRDown        = -1
            sow_FSRUp          = -1
            sow_FSRDown        = -1
            sow_renormUp       = -1
            sow_renormDown     = -1
            sow_factUp         = -1
            sow_factDown       = -1
            sow_renormfactUp   = -1
            sow_renormfactDown = -1

        datasets = ["SingleMuon", "SingleElectron", "EGamma", "MuonEG", "DoubleMuon", "DoubleElectron", "DoubleEG"]
        for d in datasets:
            if d in dataset: dataset = dataset.split('_')[0]

        # Set the sampleType (used for MC matching requirement)
        sampleType = "prompt"
        if isData:
            sampleType = "data"
        elif histAxisName in get_te_param("conv_samples"):
            sampleType = "conversions"
        elif histAxisName in get_te_param("prompt_and_conv_samples"):
            # Just DY (since we care about prompt DY for Z CR, and conv DY for 3l CR)
            sampleType = "prompt_and_conversions"

        # Initialize objects
        met  = events.MET
        ele  = events.Electron
        mu   = events.Muon
        tau  = events.Tau
        jets = events.Jet

        # An array of lenght events that is just 1 for each event
        # Probably there's a better way to do this, but we use this method elsewhere so I guess why not..
        events.nom = ak.ones_like(events.MET.pt)

        ele["idEmu"] = te_os.ttH_idEmu_cuts_E3(ele.hoe, ele.eta, ele.deltaEtaSC, ele.eInvMinusPInv, ele.sieie)
        ele["conept"] = te_os.coneptElec(ele.pt, ele.mvaTTHUL, ele.jetRelIso)
        mu["conept"] = te_os.coneptMuon(mu.pt, mu.mvaTTHUL, mu.jetRelIso, mu.mediumId)
        ele["btagDeepFlavB"] = ak.fill_none(ele.matched_jet.btagDeepFlavB, -99)
        mu["btagDeepFlavB"] = ak.fill_none(mu.matched_jet.btagDeepFlavB, -99)
        if not isData:
            ele["gen_pdgId"] = ak.fill_none(ele.matched_gen.pdgId, 0)
            mu["gen_pdgId"] = ak.fill_none(mu.matched_gen.pdgId, 0)

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

        ######### EFT coefficients ##########

        # Extract the EFT quadratic coefficients and optionally use them to calculate the coefficients on the w**2 quartic function
        # eft_coeffs is never Jagged so convert immediately to numpy for ease of use.
        eft_coeffs = ak.to_numpy(events["EFTfitCoefficients"]) if hasattr(events, "EFTfitCoefficients") else None
        if eft_coeffs is not None:
            # Check to see if the ordering of WCs for this sample matches what want
            if self._samples[dataset]["WCnames"] != self._wc_names_lst:
                eft_coeffs = efth.remap_coeffs(self._samples[dataset]["WCnames"], self._wc_names_lst, eft_coeffs)
        eft_w2_coeffs = efth.calc_w2_coeffs(eft_coeffs,self._dtype) if (self._do_errors and eft_coeffs is not None) else None
        # Initialize the out object
        hout = self.accumulator

        ################### Electron selection ####################

        ele["isPres"] = te_os.isPresElec(ele.pt, ele.eta, ele.dxy, ele.dz, ele.miniPFRelIso_all, ele.sip3d, getattr(ele,"mvaFall17V2noIso_WPL"))
        ele["isLooseE"] = te_os.isLooseElec(ele.miniPFRelIso_all,ele.sip3d,ele.lostHits)
        ele["isFO"] = te_os.isFOElec(ele.pt, ele.conept, ele.btagDeepFlavB, ele.idEmu, ele.convVeto, ele.lostHits, ele.mvaTTHUL, ele.jetRelIso, ele.mvaFall17V2noIso_WP90, year)
        ele["isTightLep"] = te_os.tightSelElec(ele.isFO, ele.mvaTTHUL)

        ################### Muon selection ####################

        mu["pt"] = ApplyRochesterCorrections(year, mu, isData) # Need to apply corrections before doing muon selection
        mu["isPres"] = te_os.isPresMuon(mu.dxy, mu.dz, mu.sip3d, mu.eta, mu.pt, mu.miniPFRelIso_all)
        mu["isLooseM"] = te_os.isLooseMuon(mu.miniPFRelIso_all,mu.sip3d,mu.looseId)
        mu["isFO"] = te_os.isFOMuon(mu.pt, mu.conept, mu.btagDeepFlavB, mu.mvaTTHUL, mu.jetRelIso, year)
        mu["isTightLep"]= te_os.tightSelMuon(mu.isFO, mu.mediumId, mu.mvaTTHUL)

        ################### Loose selection ####################

        m_loose = mu[mu.isPres & mu.isLooseM]
        e_loose = ele[ele.isPres & ele.isLooseE]
        l_loose = ak.with_name(ak.concatenate([e_loose, m_loose], axis=1), 'PtEtaPhiMCandidate')

        # Compute pair invariant masses, for all flavors all signes
        llpairs = ak.combinations(l_loose, 2, fields=["l0","l1"])
        events["minMllAFAS"] = ak.min( (llpairs.l0+llpairs.l1).mass, axis=-1)

        # Build FO collection
        m_fo = mu[mu.isPres & mu.isLooseM & mu.isFO]
        e_fo = ele[ele.isPres & ele.isLooseE & ele.isFO]

        # Attach the lepton SFs to the electron and muons collections
        AttachElectronSF(e_fo,year=year)
        AttachMuonSF(m_fo,year=year)

        # Attach per lepton fake rates
        AttachPerLeptonFR(e_fo, flavor = "Elec", year=year)
        AttachPerLeptonFR(m_fo, flavor = "Muon", year=year)
        m_fo['convVeto'] = ak.ones_like(m_fo.charge)
        m_fo['lostHits'] = ak.zeros_like(m_fo.charge)
        l_fo = ak.with_name(ak.concatenate([e_fo, m_fo], axis=1), 'PtEtaPhiMCandidate')
        l_fo_conept_sorted = l_fo[ak.argsort(l_fo.conept, axis=-1,ascending=False)]

        ################### Tau selection ####################

        if self.tau_flag:
            tau["pt"] = ApplyTES(year, tau, isData)
            tau["isPres"]  = te_os.isPresTau(tau.pt, tau.eta, tau.dxy, tau.dz, tau.idDeepTau2017v2p1VSjet, tau.idDeepTau2017v2p1VSe, tau.idDeepTau2017v2p1VSmu, minpt=20)
            tau["isClean"] = te_os.isClean(tau, l_fo, drmin=0.3)
            tau["isGood"]  =  tau["isClean"] & tau["isPres"]
            tau = tau[tau.isGood]

            tau['DMflag'] = ((tau.decayMode==0) | (tau.decayMode==1) | (tau.decayMode==10) | (tau.decayMode==11))
            tau = tau[tau['DMflag']]
            tau["isVLoose"]  = te_os.isVLooseTau(tau.idDeepTau2017v2p1VSjet)
            tau["isLoose"]   = te_os.isLooseTau(tau.idDeepTau2017v2p1VSjet)
            tau["iseTight"]  = te_os.iseTightTau(tau.idDeepTau2017v2p1VSe)
            tau["ismTight"]  = te_os.ismTightTau(tau.idDeepTau2017v2p1VSmu)

            cleaning_taus = tau[tau["isLoose"]>0]
            nLtau  = ak.num(tau[tau["isLoose"]>0] )
            if not isData:
                AttachTauSF(events,tau,year=year)
            tau_padded = ak.pad_none(tau, 1)
            tau0 = tau_padded[:,0]
        else:

            tau["isPres"]  = te_os.isPresTau(tau.pt, tau.eta, tau.dxy, tau.dz, tau.idDeepTau2017v2p1VSjet, tau.idDeepTau2017v2p1VSe, tau.idDeepTau2017v2p1VSmu, minpt=20)
            tau["isClean"] = te_os.isClean(tau, l_loose, drmin=0.3)
            tau["isGood"]  =  tau["isClean"] & tau["isPres"]
            tau = tau[tau.isGood] # use these to clean jets
            tau["isTight"] = te_os.isVLooseTau(tau.idDeepTau2017v2p1VSjet) # use these to veto

        ######### Systematics ###########

        # Define the lists of systematics we include
        obj_correction_syst_lst = [
            f'JER_{year}Up',f'JER_{year}Down', # Systs that affect the kinematics of objects
            'JES_FlavorQCDUp', 'JES_AbsoluteUp', 'JES_RelativeBalUp', 'JES_BBEC1Up', 'JES_RelativeSampleUp', 'JES_FlavorQCDDown', 'JES_AbsoluteDown', 'JES_RelativeBalDown', 'JES_BBEC1Down', 'JES_RelativeSampleDown'
        ]
        if self.tau_flag:
            obj_correction_syst_lst.append("TESUp")
            obj_correction_syst_lst.append("TESDown")
            obj_correction_syst_lst.append("FESUp")
            obj_correction_syst_lst.append("FESDown")
        wgt_correction_syst_lst = [
            "lepSF_muonUp","lepSF_muonDown","lepSF_elecUp","lepSF_elecDown",f"btagSFbc_{year}Up",f"btagSFbc_{year}Down","btagSFbc_corrUp","btagSFbc_corrDown",f"btagSFlight_{year}Up",f"btagSFlight_{year}Down","btagSFlight_corrUp","btagSFlight_corrDown","PUUp","PUDown","PreFiringUp","PreFiringDown",f"triggerSF_{year}Up",f"triggerSF_{year}Down", # Exp systs
            "FSRUp","FSRDown","ISRUp","ISRDown","renormUp","renormDown","factUp","factDown", # Theory systs
        ]
        if self.tau_flag:
            wgt_correction_syst_lst.append("lepSF_taus_realUp")
            wgt_correction_syst_lst.append("lepSF_taus_realDown")
            wgt_correction_syst_lst.append("lepSF_taus_fakeUp")
            wgt_correction_syst_lst.append("lepSF_taus_fakeDown")

        data_syst_lst = [
            "FFUp","FFDown","FFptUp","FFptDown","FFetaUp","FFetaDown",f"FFcloseEl_{year}Up",f"FFcloseEl_{year}Down",f"FFcloseMu_{year}Up",f"FFcloseMu_{year}Down"
        ]

        # These weights can go outside of the outside sys loop since they do not depend on pt of mu or jets
        # We only calculate these values if not isData
        # Note: add() will generally modify up/down weights, so if these are needed for any reason after this point, we should instead pass copies to add()
        # Note: Here we will to the weights object the SFs that do not depend on any of the forthcoming loops
        weights_obj_base = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
        if not isData:

            # If this is no an eft sample, get the genWeight
            if eft_coeffs is None: genw = events["genWeight"]
            else: genw= np.ones_like(events["event"])

            # Normalize by (xsec/sow)*genw where genw is 1 for EFT samples
            # Note that for theory systs, will need to multiply by sow/sow_wgtUP to get (xsec/sow_wgtUp)*genw and same for Down
            lumi = 1000.0*get_tc_param(f"lumi_{year}")
            weights_obj_base.add("norm",(xsec/sow)*genw*lumi)

            # Attach PS weights (ISR/FSR) and scale weights (renormalization/factorization) and PDF weights
            tc_cor.AttachPSWeights(events)
            tc_cor.AttachScaleWeights(events)
            #AttachPdfWeights(events) # TODO
            # FSR/ISR weights
            weights_obj_base.add('ISR', events.nom, events.ISRUp*(sow/sow_ISRUp), events.ISRDown*(sow/sow_ISRDown))
            weights_obj_base.add('FSR', events.nom, events.FSRUp*(sow/sow_FSRUp), events.FSRDown*(sow/sow_FSRDown))
            # renorm/fact scale
            weights_obj_base.add('renorm', events.nom, events.renormUp*(sow/sow_renormUp), events.renormDown*(sow/sow_renormDown))
            weights_obj_base.add('fact', events.nom, events.factUp*(sow/sow_factUp), events.factDown*(sow/sow_factDown))
            # Prefiring and PU (note prefire weights only available in nanoAODv9)
            weights_obj_base.add('PreFiring', events.L1PreFiringWeight.Nom,  events.L1PreFiringWeight.Up,  events.L1PreFiringWeight.Dn)
            weights_obj_base.add('PU', tc_cor.GetPUSF((events.Pileup.nTrueInt), year), tc_cor.GetPUSF(events.Pileup.nTrueInt, year, 'up'), tc_cor.GetPUSF(events.Pileup.nTrueInt, year, 'down'))


        ######### The rest of the processor is inside this loop over systs that affect object kinematics  ###########

        # If we're doing systematics and this isn't data, we will loop over the obj_correction_syst_lst list
        if self._do_systematics and not isData: syst_var_list = ["nominal"] + obj_correction_syst_lst
        # Otherwise loop juse once, for nominal
        else: syst_var_list = ['nominal']

        # Loop over the list of systematic variations we've constructed
        met_raw=met
        for syst_var in syst_var_list:
            # Make a copy of the base weights object, so that each time through the loop we do not double count systs
            # In this loop over systs that impact kinematics, we will add to the weights objects the SFs that depend on the object kinematics
            weights_obj_base_for_kinematic_syst = copy.deepcopy(weights_obj_base)

            #################### Jets ####################

            # Jet cleaning, before any jet selection
            #vetos_tocleanjets = ak.with_name( ak.concatenate([tau, l_fo], axis=1), "PtEtaPhiMCandidate")
            if self.tau_flag:
                vetos_tocleanjets = ak.with_name( ak.concatenate([cleaning_taus, l_fo], axis=1), "PtEtaPhiMCandidate")
            else:
                vetos_tocleanjets = ak.with_name( l_fo, "PtEtaPhiMCandidate")
            tmp = ak.cartesian([ak.local_index(jets.pt), vetos_tocleanjets.jetIdx], nested=True)
            cleanedJets = jets[~ak.any(tmp.slot0 == tmp.slot1, axis=-1)] # this line should go before *any selection*, otherwise lep.jetIdx is not aligned with the jet index

            # Selecting jets and cleaning them
            jetptname = "pt_nom" if hasattr(cleanedJets, "pt_nom") else "pt"

            # Jet energy corrections
            if not isData:
                cleanedJets["pt_raw"] = (1 - cleanedJets.rawFactor)*cleanedJets.pt
                cleanedJets["mass_raw"] = (1 - cleanedJets.rawFactor)*cleanedJets.mass
                cleanedJets["pt_gen"] =ak.values_astype(ak.fill_none(cleanedJets.matched_gen.pt, 0), np.float32)
                cleanedJets["rho"] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, cleanedJets.pt)[0]
                events_cache = events.caches[0]
                cleanedJets = ApplyJetCorrections(year, corr_type='jets').build(cleanedJets, lazy_cache=events_cache)
                cleanedJets=ApplyJetSystematics(year,cleanedJets,syst_var)
                met=ApplyJetCorrections(year, corr_type='met').build(met_raw, cleanedJets, lazy_cache=events_cache)
                if self.tau_flag:
                    tau["pt"]      = ApplyTESSystematic(year, tau, isData, syst_var)
                    tau["pt"]      = ApplyFESSystematic(year, tau, isData, syst_var)
            cleanedJets["isGood"] = tc_os.is_tight_jet(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, pt_cut=30., eta_cut=get_te_param("eta_j_cut"), id_cut=get_te_param("jet_id_cut"))
            goodJets = cleanedJets[cleanedJets.isGood]

            # Count jets
            njets = ak.num(goodJets)
            ht = ak.sum(goodJets.pt,axis=-1)
            j0 = goodJets[ak.argmax(goodJets.pt,axis=-1,keepdims=True)]

            # Loose DeepJet WP
            if year == "2017":
                btagwpl = get_tc_param("btag_wp_loose_UL17")
            elif year == "2018":
                btagwpl = get_tc_param("btag_wp_loose_UL18")
            elif year=="2016":
                btagwpl = get_tc_param("btag_wp_loose_UL16")
            elif year=="2016APV":
                btagwpl = get_tc_param("btag_wp_loose_UL16APV")
            else:
                raise ValueError(f"Error: Unknown year \"{year}\".")
            isBtagJetsLoose = (goodJets.btagDeepFlavB > btagwpl)
            isNotBtagJetsLoose = np.invert(isBtagJetsLoose)
            nbtagsl = ak.num(goodJets[isBtagJetsLoose])

            # Medium DeepJet WP
            if year == "2017":
                btagwpm = get_tc_param("btag_wp_medium_UL17")
            elif year == "2018":
                btagwpm = get_tc_param("btag_wp_medium_UL18")
            elif year=="2016":
                btagwpm = get_tc_param("btag_wp_medium_UL16")
            elif year=="2016APV":
                btagwpm = get_tc_param("btag_wp_medium_UL16APV")
            else:
                raise ValueError(f"Error: Unknown year \"{year}\".")
            isBtagJetsMedium = (goodJets.btagDeepFlavB > btagwpm)
            isNotBtagJetsMedium = np.invert(isBtagJetsMedium)
            nbtagsm = ak.num(goodJets[isBtagJetsMedium])


            #################### Add variables into event object so that they persist ####################

            # Put njets and l_fo_conept_sorted into events
            events["njets"] = njets
            events["l_fo_conept_sorted"] = l_fo_conept_sorted

            # The event selection
            te_es.add2lMaskAndSFs(events, year, isData, sampleType)
            te_es.add3lMaskAndSFs(events, year, isData, sampleType)
            te_es.add4lMaskAndSFs(events, year, isData)
            te_es.addLepCatMasks(events)

            # Convenient to have l0, l1, l2 on hand
            l_fo_conept_sorted_padded = ak.pad_none(l_fo_conept_sorted, 3)
            l0 = l_fo_conept_sorted_padded[:,0]
            l1 = l_fo_conept_sorted_padded[:,1]
            l2 = l_fo_conept_sorted_padded[:,2]


            ######### Event weights that do not depend on the lep cat ##########

            if not isData:
                # Workaround to use UL16APV SFs for UL16 for light jets
                if year == "2016":
                    year_light = "2016APV"
                else:
                    year_light = year

                isBtagJetsLooseNotMedium = (isBtagJetsLoose & isNotBtagJetsMedium)

                light_mask = goodJets.hadronFlavour==0
                bc_mask = goodJets.hadronFlavour>0

                jets_light = goodJets[light_mask]
                jets_bc    = goodJets[bc_mask]

                btag_effM_light = GetBtagEff(jets_light, year, 'medium')
                btag_effM_bc = GetBtagEff(jets_bc, year, 'medium')
                btag_effL_light = GetBtagEff(jets_light, year, 'loose')
                btag_effL_bc = GetBtagEff(jets_bc, year, 'loose')
                btag_sfM_light = tc_cor.btag_sf_eval(jets_light, "M",year_light,"deepJet_incl","central")
                btag_sfM_bc    = tc_cor.btag_sf_eval(jets_bc,    "M",year,      "deepJet_comb","central")
                btag_sfL_light = tc_cor.btag_sf_eval(jets_light, "L",year_light,"deepJet_incl","central")
                btag_sfL_bc    = tc_cor.btag_sf_eval(jets_bc,    "L",year,      "deepJet_comb","central")

                pData_light, pMC_light = tc_cor.get_method1a_wgt_doublewp(btag_effM_light, btag_effL_light, btag_sfM_light, btag_sfL_light, isBtagJetsMedium[light_mask], isBtagJetsLooseNotMedium[light_mask], isNotBtagJetsLoose[light_mask])
                btag_w_light = pData_light/pMC_light
                pData_bc, pMC_bc = tc_cor.get_method1a_wgt_doublewp(btag_effM_bc, btag_effL_bc, btag_sfM_bc, btag_sfL_bc, isBtagJetsMedium[bc_mask], isBtagJetsLooseNotMedium[bc_mask], isNotBtagJetsLoose[bc_mask])
                btag_w_bc = pData_bc/pMC_bc
                btag_w = btag_w_light*btag_w_bc
                weights_obj_base_for_kinematic_syst.add("btagSF", btag_w)

                if self._do_systematics and syst_var=='nominal':
                    for b_syst in ["bc_corr","light_corr",f"bc_{year}",f"light_{year}"]:
                        if b_syst.endswith("_corr"):
                            corrtype = "correlated"
                        else:
                            corrtype = "uncorrelated"

                        if b_syst.startswith("light_"):
                            jets_flav = jets_light
                            flav_mask = light_mask
                            sys_year = year_light
                            dJ_tag = "incl"
                            btag_effM = btag_effM_light
                            btag_effL = btag_effL_light
                            pMC_flav = pMC_light
                            fixed_btag_w = btag_w_bc
                        elif b_syst.startswith("bc_"):
                            jets_flav = jets_bc
                            flav_mask = bc_mask
                            sys_year = year
                            dJ_tag = "comb"
                            btag_effM = btag_effM_bc
                            btag_effL = btag_effL_bc
                            pMC_flav = pMC_bc
                            fixed_btag_w = btag_w_light
                        else:
                            raise ValueError("btag systematics should be divided in flavor (bc or light)!")

                        btag_sfL_up   = tc_cor.btag_sf_eval(jets_flav, "L",sys_year,f"deepJet_{dJ_tag}",f"up_{corrtype}")
                        btag_sfL_down = tc_cor.btag_sf_eval(jets_flav, "L",sys_year,f"deepJet_{dJ_tag}",f"down_{corrtype}")
                        btag_sfM_up   = tc_cor.btag_sf_eval(jets_flav, "M",sys_year,f"deepJet_{dJ_tag}",f"up_{corrtype}")
                        btag_sfM_down = tc_cor.btag_sf_eval(jets_flav, "M",sys_year,f"deepJet_{dJ_tag}",f"down_{corrtype}")

                        pData_up, pMC_up = tc_cor.get_method1a_wgt_doublewp(btag_effM, btag_effL, btag_sfM_up, btag_sfL_up, isBtagJetsMedium[flav_mask], isBtagJetsLooseNotMedium[flav_mask], isNotBtagJetsLoose[flav_mask])
                        pData_down, pMC_down = tc_cor.get_method1a_wgt_doublewp(btag_effM, btag_effL, btag_sfM_down, btag_sfL_down, isBtagJetsMedium[flav_mask], isBtagJetsLooseNotMedium[flav_mask], isNotBtagJetsLoose[flav_mask])

                        btag_w_up = (pData_up/pMC_flav)
                        btag_w_down = (pData_down/pMC_flav)

                        btag_w_up = fixed_btag_w*btag_w_up/btag_w
                        btag_w_down = fixed_btag_w*btag_w_down/btag_w

                        weights_obj_base_for_kinematic_syst.add(f"btagSF{b_syst}", events.nom, btag_w_up, btag_w_down)

                # Trigger SFs
                GetTriggerSF(year,events,l0,l1)
                weights_obj_base_for_kinematic_syst.add(f"triggerSF_{year}", events.trigger_sf, copy.deepcopy(events.trigger_sfUp), copy.deepcopy(events.trigger_sfDown))            # In principle does not have to be in the lep cat loop


            ######### Event weights that do depend on the lep cat ###########

            # Loop over categories and fill the dict
            weights_dict = {}
            for ch_name in ["2l", "2l_4t", "3l", "4l", "2l_CR", "2l_CRflip", "3l_CR", "2los_CRtt", "2los_CRZ"]:

                # For both data and MC
                weights_dict[ch_name] = copy.deepcopy(weights_obj_base_for_kinematic_syst)
                if ch_name.startswith("2l"):
                    weights_dict[ch_name].add("FF", events.fakefactor_2l, copy.deepcopy(events.fakefactor_2l_up), copy.deepcopy(events.fakefactor_2l_down))
                    weights_dict[ch_name].add("FFpt",  events.nom, copy.deepcopy(events.fakefactor_2l_pt1/events.fakefactor_2l), copy.deepcopy(events.fakefactor_2l_pt2/events.fakefactor_2l))
                    weights_dict[ch_name].add("FFeta", events.nom, copy.deepcopy(events.fakefactor_2l_be1/events.fakefactor_2l), copy.deepcopy(events.fakefactor_2l_be2/events.fakefactor_2l))
                    weights_dict[ch_name].add(f"FFcloseEl_{year}", events.nom, copy.deepcopy(events.fakefactor_2l_elclosureup/events.fakefactor_2l), copy.deepcopy(events.fakefactor_2l_elclosuredown/events.fakefactor_2l))
                    weights_dict[ch_name].add(f"FFcloseMu_{year}", events.nom, copy.deepcopy(events.fakefactor_2l_muclosureup/events.fakefactor_2l), copy.deepcopy(events.fakefactor_2l_muclosuredown/events.fakefactor_2l))
                elif ch_name.startswith("3l"):
                    weights_dict[ch_name].add("FF", events.fakefactor_3l, copy.deepcopy(events.fakefactor_3l_up), copy.deepcopy(events.fakefactor_3l_down))
                    weights_dict[ch_name].add("FFpt",  events.nom, copy.deepcopy(events.fakefactor_3l_pt1/events.fakefactor_3l), copy.deepcopy(events.fakefactor_3l_pt2/events.fakefactor_3l))
                    weights_dict[ch_name].add("FFeta", events.nom, copy.deepcopy(events.fakefactor_3l_be1/events.fakefactor_3l), copy.deepcopy(events.fakefactor_3l_be2/events.fakefactor_3l))
                    weights_dict[ch_name].add(f"FFcloseEl_{year}", events.nom, copy.deepcopy(events.fakefactor_3l_elclosureup/events.fakefactor_3l), copy.deepcopy(events.fakefactor_3l_elclosuredown/events.fakefactor_3l))
                    weights_dict[ch_name].add(f"FFcloseMu_{year}", events.nom, copy.deepcopy(events.fakefactor_3l_muclosureup/events.fakefactor_3l), copy.deepcopy(events.fakefactor_3l_muclosuredown/events.fakefactor_3l))

                # For data only
                if isData:
                    if ch_name in ["2l","2l_4t","2l_CR","2l_CRflip"]:
                        weights_dict[ch_name].add("fliprate", events.flipfactor_2l)

                # For MC only
                if not isData:
                    if ch_name.startswith("2l"):
                        weights_dict[ch_name].add("lepSF_muon", events.sf_2l_muon, copy.deepcopy(events.sf_2l_hi_muon), copy.deepcopy(events.sf_2l_lo_muon))
                        weights_dict[ch_name].add("lepSF_elec", events.sf_2l_elec, copy.deepcopy(events.sf_2l_hi_elec), copy.deepcopy(events.sf_2l_lo_elec))
                        if self.tau_flag:
                            weights_dict[ch_name].add("lepSF_taus_real", events.sf_2l_taus_real, copy.deepcopy(events.sf_2l_taus_real_hi), copy.deepcopy(events.sf_2l_taus_real_lo))
                            weights_dict[ch_name].add("lepSF_taus_fake", events.sf_2l_taus_fake, copy.deepcopy(events.sf_2l_taus_fake_hi), copy.deepcopy(events.sf_2l_taus_fake_lo))
                    elif ch_name.startswith("3l"):
                        weights_dict[ch_name].add("lepSF_muon", events.sf_3l_muon, copy.deepcopy(events.sf_3l_hi_muon), copy.deepcopy(events.sf_3l_lo_muon))
                        weights_dict[ch_name].add("lepSF_elec", events.sf_3l_elec, copy.deepcopy(events.sf_3l_hi_elec), copy.deepcopy(events.sf_3l_lo_elec))
                        if self.tau_flag:
                            weights_dict[ch_name].add("lepSF_taus_real", events.sf_2l_taus_real, copy.deepcopy(events.sf_2l_taus_real_hi), copy.deepcopy(events.sf_2l_taus_real_lo))
                            weights_dict[ch_name].add("lepSF_taus_fake", events.sf_2l_taus_fake, copy.deepcopy(events.sf_2l_taus_fake_hi), copy.deepcopy(events.sf_2l_taus_fake_lo))
                    elif ch_name.startswith("4l"):
                        weights_dict[ch_name].add("lepSF_muon", events.sf_4l_muon, copy.deepcopy(events.sf_4l_hi_muon), copy.deepcopy(events.sf_4l_lo_muon))
                        weights_dict[ch_name].add("lepSF_elec", events.sf_4l_elec, copy.deepcopy(events.sf_4l_hi_elec), copy.deepcopy(events.sf_4l_lo_elec))
                    else:
                        raise Exception(f"Unknown channel name: {ch_name}")


            ######### Masks we need for the selection ##########

            # Get mask for events that have two sf os leps close to z peak
            sfosz_3l_OnZ_mask = tc_es.get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:3],pt_window=10.0)
            sfosz_3l_OffZ_mask = ~sfosz_3l_OnZ_mask
            sfosz_3l_OffZ_low_mask = tc_es.get_off_Z_mask_low(l_fo_conept_sorted_padded[:,0:3],pt_window=0.0)
            sfosz_3l_OffZ_any_mask = tc_es.get_any_sfos_pair(l_fo_conept_sorted_padded[:,0:3])
            sfosz_2l_mask = tc_es.get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:2],pt_window=10.0)
            sfasz_2l_mask = tc_es.get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:2],pt_window=30.0,flavor="as") # Any sign (do not enforce ss or os here)
            if self.tau_flag:
                tl_zpeak_mask = te_es.lt_Z_mask(l0, l1, tau0, 30.0)

            # Pass trigger mask
            pass_trg = tc_es.trg_pass_no_overlap(events,isData,dataset,str(year),te_es.dataset_dict_top22006,te_es.exclude_dict_top22006)

            # b jet masks
            bmask_atleast1med_atleast2loose = ((nbtagsm>=1)&(nbtagsl>=2)) # Used for 2lss and 4l
            bmask_exactly0med = (nbtagsm==0) # Used for 3l CR and 2los Z CR
            bmask_exactly1med = (nbtagsm==1) # Used for 3l SR and 2lss CR
            bmask_exactly2med = (nbtagsm==2) # Used for CRtt
            bmask_atleast2med = (nbtagsm>=2) # Used for 3l SR
            bmask_atmost2med  = (nbtagsm< 3) # Used to make 2lss mutually exclusive from tttt enriched
            bmask_atleast3med = (nbtagsm>=3) # Used for tttt enriched

            # Charge masks
            chargel0_p = ak.fill_none(((l0.charge)>0),False)
            chargel0_m = ak.fill_none(((l0.charge)<0),False)
            charge2l_0 = ak.fill_none(((l0.charge+l1.charge)==0),False)
            charge2l_1 = ak.fill_none(((l0.charge+l1.charge)!=0),False)
            charge3l_p = ak.fill_none(((l0.charge+l1.charge+l2.charge)>0),False)
            charge3l_m = ak.fill_none(((l0.charge+l1.charge+l2.charge)<0),False)
            if self.tau_flag:
                tau_L_mask  = (ak.num(tau[tau["isLoose"]>0]) ==1)
                no_tau_mask = (ak.num(tau[tau["isLoose"]>0])==0)


            ######### Store boolean masks with PackedSelection ##########

            selections = PackedSelection(dtype='uint64')
            preselections = PackedSelection(dtype='uint64')
            # Lumi mask (for data)
            selections.add("is_good_lumi",lumi_mask)
            preselections.add("is_good_lumi",lumi_mask)

            # 2lss selection
            preselections.add("chargedl0", (chargel0_p | chargel0_m))
            preselections.add("2lss", (events.is2l & pass_trg))
            preselections.add("2l_nozeeveto", (events.is2l_nozeeveto & pass_trg))
            preselections.add("2los", charge2l_0)
            preselections.add("2lem", events.is_em)
            preselections.add("2lee", events.is_ee)
            preselections.add("2lee", events.is_mm)
            preselections.add("2l_onZ_as", sfasz_2l_mask)
            preselections.add("2l_onZ", sfosz_2l_mask)
            preselections.add("bmask_atleast3m", (bmask_atleast3med))
            preselections.add("bmask_atleast1m2l", (bmask_atleast1med_atleast2loose))
            preselections.add("bmask_atmost2m", (bmask_atmost2med))
            preselections.add("2l_p", (chargel0_p))
            preselections.add("2l_m", (chargel0_m))
            if self.tau_flag:
                preselections.add("1tau", (tau_L_mask))
                preselections.add("0tau", (no_tau_mask))
                preselections.add("onZ_tau", (tl_zpeak_mask))
                preselections.add("offZ_tau", (~tl_zpeak_mask))

            # 3l selection
            preselections.add("3l", (events.is3l & pass_trg))
            preselections.add("bmask_exactly0m", (bmask_exactly0med))
            preselections.add("bmask_exactly1m", (bmask_exactly1med))
            preselections.add("bmask_exactly2m", (bmask_exactly2med))
            preselections.add("bmask_atleast2m", (bmask_atleast2med))
            preselections.add("3l_p", (events.is3l & pass_trg & charge3l_p))
            preselections.add("3l_m", (events.is3l & pass_trg & charge3l_m))
            preselections.add("3l_onZ", (sfosz_3l_OnZ_mask))

            if self._offZ_split:
                preselections.add("3l_offZ_low", (sfosz_3l_OffZ_mask & sfosz_3l_OffZ_any_mask & sfosz_3l_OffZ_low_mask))
                preselections.add("3l_offZ_high", (sfosz_3l_OffZ_mask & sfosz_3l_OffZ_any_mask & ~sfosz_3l_OffZ_low_mask))
                preselections.add("3l_offZ_none", (sfosz_3l_OffZ_mask & ~sfosz_3l_OffZ_any_mask))
            else:
                preselections.add("3l_offZ", (sfosz_3l_OffZ_mask))

            # 4l selection
            preselections.add("4l", (events.is4l & pass_trg))

            select_cat_dict = None
            with open(topeft_path("channels/ch_lst_test.json"), "r") as ch_json_test:
                select_cat_dict = json.load(ch_json_test)

            # This dictionary keeps track of which selections go with which SR categories
            if self._offZ_split:
                import_sr_cat_dict = select_cat_dict["OFFZ_SPLIT_CH_LST_SR"]
            elif self.tau_flag:
                import_sr_cat_dict = select_cat_dict["TAU_CH_LST_SR"]
            else:
                import_sr_cat_dict = select_cat_dict["TOP22_006_CH_LST_SR"]
            # This dictionary keeps track of which selections go with which CR categories
            import_cr_cat_dict = select_cat_dict["CH_LST_CR"]

            #Filling selections according to the json specifications for SRs
            for lep_cat, lep_cat_dict in import_sr_cat_dict.items():
                lep_ch_list = lep_cat_dict['lep_chan_lst']
                chtag = None

                #looping over each region within the lep category
                for lep_ch in lep_ch_list:
                    tempmask = None
                    #the first entry of the list is the region name to add in "selections"
                    chtag = lep_ch[0]

                    for chcut in lep_ch[1:]:
                        if not tempmask is None:
                            tempmask = tempmask & preselections.any(chcut)
                        else:
                            tempmask = preselections.any(chcut)
                    selections.add(chtag, tempmask)

            #Filling selections according to the json specifications for CRs
            for lep_cat, lep_cat_dict in import_cr_cat_dict.items():
                lep_ch_list = lep_cat_dict['lep_chan_lst']
                chtag = None

                #looping over each region within the lep category
                for lep_ch in lep_ch_list:
                    tempmask = None
                    #the first entry of the list is the region name to add in "selections"
                    chtag = lep_ch[0]

                    for chcut in lep_ch[1:]:
                        if not tempmask is None:
                            tempmask = tempmask & preselections.any(chcut)
                        else:
                            tempmask = preselections.any(chcut)
                    selections.add(chtag, tempmask)

            del preselections

            # Lep flavor selection
            selections.add("ee",  events.is_ee)
            selections.add("em",  events.is_em)
            selections.add("mm",  events.is_mm)
            selections.add("eee", events.is_eee)
            selections.add("eem", events.is_eem)
            selections.add("emm", events.is_emm)
            selections.add("mmm", events.is_mmm)
            selections.add("llll", (events.is_eeee | events.is_eeem | events.is_eemm | events.is_emmm | events.is_mmmm | events.is_gr4l)) # Not keepting track of these separately

            # Njets selection
            selections.add("exactly_0j", (njets==0))
            selections.add("exactly_1j", (njets==1))
            selections.add("exactly_2j", (njets==2))
            selections.add("exactly_3j", (njets==3))
            selections.add("exactly_4j", (njets==4))
            selections.add("exactly_5j", (njets==5))
            selections.add("exactly_6j", (njets==6))
            selections.add("atleast_1j", (njets>=1))
            selections.add("atleast_4j", (njets>=4))
            selections.add("atleast_5j", (njets>=5))
            selections.add("atleast_7j", (njets>=7))
            selections.add("atleast_0j", (njets>=0))
            selections.add("atmost_3j" , (njets<=3))

            # AR/SR categories
            selections.add("isSR_2lSS",    ( events.is2l_SR) & charge2l_1)
            selections.add("isAR_2lSS",    (~events.is2l_SR) & charge2l_1)
            selections.add("isAR_2lSS_OS", ( events.is2l_SR) & charge2l_0) # Sideband for the charge flip
            selections.add("isSR_2lOS",    ( events.is2l_SR) & charge2l_0)
            selections.add("isAR_2lOS",    (~events.is2l_SR) & charge2l_0)

            selections.add("isSR_3l",  events.is3l_SR)
            selections.add("isAR_3l", ~events.is3l_SR)
            selections.add("isSR_4l",  events.is4l_SR)


            ######### Variables for the dense axes of the hists ##########

            # Calculate ptbl
            ptbl_bjet = goodJets[(isBtagJetsMedium | isBtagJetsLoose)]
            ptbl_bjet = ptbl_bjet[ak.argmax(ptbl_bjet.pt,axis=-1,keepdims=True)] # Only save hardest b-jet
            ptbl_lep = l_fo_conept_sorted
            ptbl = (ptbl_bjet.nearest(ptbl_lep) + ptbl_bjet).pt
            ptbl = ak.values_astype(ak.fill_none(ptbl, -1), np.float32)

            # Z pt (pt of the ll pair that form the Z for the onZ categories)
            ptz = te_es.get_Z_pt(l_fo_conept_sorted_padded[:,0:3],10.0)
            if self.tau_flag:
                ptz_wtau = (l0+tau0).pt
            if self._offZ_split:
                ptz = te_es.get_ll_pt(l_fo_conept_sorted_padded[:,0:3],10.0)
            # Leading (b+l) pair pt
            bjetsl = goodJets[isBtagJetsLoose][ak.argsort(goodJets[isBtagJetsLoose].pt, axis=-1, ascending=False)]
            bl_pairs = ak.cartesian({"b":bjetsl,"l":l_fo_conept_sorted})
            blpt = (bl_pairs["b"] + bl_pairs["l"]).pt
            bl0pt = ak.flatten(blpt[ak.argmax(blpt,axis=-1,keepdims=True)])

            # Collection of all objects (leptons and jets)
            if self.tau_flag:
                l_j_collection = ak.with_name(ak.concatenate([l_fo_conept_sorted,goodJets,cleaning_taus], axis=1),"PtEtaPhiMCollection")
            else:
                l_j_collection = ak.with_name(ak.concatenate([l_fo_conept_sorted,goodJets], axis=1),"PtEtaPhiMCollection")

            # Leading object (j or l) pt
            o0pt = ak.max(l_j_collection.pt,axis=-1)

            # Pairs of l+j
            l_j_pairs = ak.combinations(l_j_collection,2,fields=["o0","o1"])
            l_j_pairs_pt = (l_j_pairs.o0 + l_j_pairs.o1).pt
            l_j_pairs_mass = (l_j_pairs.o0 + l_j_pairs.o1).mass
            lj0pt = ak.max(l_j_pairs_pt,axis=-1)

            # Define invariant mass hists
            mll_0_1 = (l0+l1).mass # Invmass for leading two leps

            # ST (but "st" is too hard to search in the code, so call it ljptsum)
            ljptsum = ak.sum(l_j_collection.pt,axis=-1)
            if self._ecut_threshold is not None:
                ecut_mask = (ljptsum<self._ecut_threshold)

            # Counts
            counts = np.ones_like(events['event'])

            # Variables we will loop over when filling hists
            varnames = {}
            varnames["ht"]      = ht
            varnames["met"]     = met.pt
            varnames["ljptsum"] = ljptsum
            varnames["l0pt"]    = l0.conept
            varnames["l0eta"]   = l0.eta
            varnames["l1pt"]    = l1.conept
            varnames["l1eta"]   = l1.eta
            varnames["j0pt"]    = ak.flatten(j0.pt)
            varnames["j0eta"]   = ak.flatten(j0.eta)
            varnames["njets"]   = njets
            varnames["nbtagsl"] = nbtagsl
            varnames["invmass"] = mll_0_1
            varnames["ptbl"]    = ak.flatten(ptbl)
            varnames["ptz"]     = ptz
            varnames["b0pt"]    = ak.flatten(ptbl_bjet.pt)
            varnames["bl0pt"]   = bl0pt
            varnames["o0pt"]    = o0pt
            varnames["lj0pt"]   = lj0pt

            ########## Fill the histograms ##########

            sr_cat_dict = {}
            cr_cat_dict = {}

            for lep_cat in import_sr_cat_dict.keys():
                sr_cat_dict[lep_cat] = {}
                for jet_cat in import_sr_cat_dict[lep_cat]["jet_lst"]:
                    jettag = None
                    if jet_cat.startswith("="):
                        jettag = "exactly_"
                    elif jet_cat.startswith("<"):
                        jettag = "atmost_"
                    elif jet_cat.startswith(">"):
                        jettag = "atleast_"
                    else:
                        raise RuntimeError(f"jet_cat {jet_cat} in {lep_cat} misses =,<,> !")
                    jet_key = jettag + str(jet_cat).replace("=", "").replace("<", "").replace(">", "") + "j"

                    sr_cat_dict[lep_cat][jet_key] = {}
                    sr_cat_dict[lep_cat][jet_key]["lep_chan_lst"] = []
                    for lep_chan_def in import_sr_cat_dict[lep_cat]["lep_chan_lst"]:
                        sr_cat_dict[lep_cat][jet_key]["lep_chan_lst"].append(lep_chan_def[0])
                    sr_cat_dict[lep_cat][jet_key]["lep_flav_lst"] = import_sr_cat_dict[lep_cat]["lep_flav_lst"]
                    if isData and "appl_lst_data" in import_sr_cat_dict[lep_cat].keys():
                        sr_cat_dict[lep_cat][jet_key]["appl_lst"] = import_sr_cat_dict[lep_cat]["appl_lst"] + import_sr_cat_dict[lep_cat]["appl_lst_data"]
                    else:
                        sr_cat_dict[lep_cat][jet_key]["appl_lst"] = import_sr_cat_dict[lep_cat]["appl_lst"]

            for lep_cat in import_cr_cat_dict.keys():
                cr_cat_dict[lep_cat] = {}
                for jet_cat in import_cr_cat_dict[lep_cat]["jet_lst"]:
                    jettag = None
                    if jet_cat.startswith("="):
                        jettag = "exactly_"
                    elif jet_cat.startswith("<"):
                        jettag = "atmost_"
                    elif jet_cat.startswith(">"):
                        jettag = "atleast_"
                    else:
                        raise RuntimeError(f"jet_cat {jet_cat} in {lep_cat} misses =,<,> !")
                    jet_key = jettag + str(jet_cat).replace("=", "").replace("<", "").replace(">", "") + "j"

                    cr_cat_dict[lep_cat][jet_key] = {}
                    cr_cat_dict[lep_cat][jet_key]["lep_chan_lst"] = []
                    for lep_chan_def in import_cr_cat_dict[lep_cat]["lep_chan_lst"]:
                        cr_cat_dict[lep_cat][jet_key]["lep_chan_lst"].append(lep_chan_def[0])
                    cr_cat_dict[lep_cat][jet_key]["lep_flav_lst"] = import_cr_cat_dict[lep_cat]["lep_flav_lst"]
                    if isData and "appl_lst_data" in import_cr_cat_dict[lep_cat].keys():
                        cr_cat_dict[lep_cat][jet_key]["appl_lst"] = import_cr_cat_dict[lep_cat]["appl_lst"] + import_cr_cat_dict[lep_cat]["appl_lst_data"]
                    else:
                        cr_cat_dict[lep_cat][jet_key]["appl_lst"] = import_cr_cat_dict[lep_cat]["appl_lst"]

            del import_sr_cat_dict, import_cr_cat_dict

            cat_dict = {}
            if not self._skip_signal_regions:
                cat_dict.update(sr_cat_dict)
            if not self._skip_control_regions:
                cat_dict.update(cr_cat_dict)
            if (not self._skip_signal_regions and not self._skip_control_regions):
                for k in sr_cat_dict:
                    if k in cr_cat_dict:
                        raise Exception(f"The key {k} is in both CR and SR dictionaries.")

            # Loop over the hists we want to fill
            for dense_axis_name, dense_axis_vals in varnames.items():
                if dense_axis_name not in self._hist_lst:
                    continue

                # Set up the list of syst wgt variations to loop over
                wgt_var_lst = ["nominal"]
                if self._do_systematics:
                    if not isData:
                        if (syst_var != "nominal"):
                            # In this case, we are dealing with systs that change the kinematics of the objs (e.g. JES)
                            # So we don't want to loop over up/down weight variations here
                            wgt_var_lst = [syst_var]
                        else:
                            # Otherwise we want to loop over the up/down weight variations
                            wgt_var_lst = wgt_var_lst + wgt_correction_syst_lst + data_syst_lst
                    else:
                        # This is data, so we want to loop over just up/down variations relevant for data (i.e. FF up and down)
                        wgt_var_lst = wgt_var_lst + data_syst_lst

                # Loop over the systematics
                for wgt_fluct in wgt_var_lst:

                    # Loop over nlep categories "2l", "3l", "4l"
                    for nlep_cat in cat_dict.keys():
                        # Get the appropriate Weights object for the nlep cat and get the weight to be used when filling the hist
                        # Need to do this inside of nlep cat loop since some wgts depend on lep cat
                        weights_object = weights_dict[nlep_cat]
                        if (wgt_fluct == "nominal") or (wgt_fluct in obj_correction_syst_lst):
                            # In the case of "nominal", or the jet energy systematics, no weight systematic variation is used
                            weight = weights_object.weight(None)
                        else:
                            # Otherwise get the weight from the Weights object
                            if wgt_fluct in weights_object.variations:
                                weight = weights_object.weight(wgt_fluct)
                            else:
                                # Note in this case there is no up/down fluct for this cateogry, so we don't want to fill a hist for it
                                continue

                        # This is a check ot make sure we guard against any unintentional variations being applied to data
                        if self._do_systematics and isData:
                            # Should not have any up/down variations for data in 4l (since we don't estimate the fake rate there)
                            if nlep_cat == "4l":
                                if weights_object.variations != set([]): raise Exception(f"Error: Unexpected wgt variations for data! Expected \"{[]}\" but have \"{weights_object.variations}\".")
                            # In all other cases, the up/down variations should correspond to only the ones in the data list
                            else:
                                if weights_object.variations != set(data_syst_lst): raise Exception(f"Error: Unexpected wgt variations for data! Expected \"{set(data_syst_lst)}\" but have \"{weights_object.variations}\".")

                        # Get a mask for events that pass any of the njet requiremens in this nlep cat
                        # Useful in cases like njets hist where we don't store njets in a sparse axis
                        njets_any_mask = selections.any(*cat_dict[nlep_cat].keys())

                        # Loop over the njets list for each channel
                        for njet_val in cat_dict[nlep_cat].keys():

                            # Loop over the appropriate AR and SR for this channel
                            for appl in cat_dict[nlep_cat][njet_val]["appl_lst"]:

                                # We don't want or need to fill SR histos with the FF variations
                                if appl.startswith("isSR") and wgt_fluct in data_syst_lst: continue

                                # Loop over the channels in each nlep cat (e.g. "3l_m_offZ_1b")
                                for lep_chan in cat_dict[nlep_cat][njet_val]["lep_chan_lst"]:
                                    # Loop over the lep flavor list for each channel
                                    for lep_flav in cat_dict[nlep_cat][njet_val]["lep_flav_lst"]:
                                        # Construct the hist name
                                        flav_ch = None
                                        njet_ch = None
                                        cuts_lst = [appl,lep_chan]

                                        if isData:
                                            cuts_lst.append("is_good_lumi")
                                        if self._split_by_lepton_flavor:
                                            flav_ch = lep_flav
                                            cuts_lst.append(lep_flav)
                                        if dense_axis_name != "njets":
                                            njet_ch = njet_val
                                            cuts_lst.append(njet_val)
                                        ch_name = construct_cat_name(lep_chan,njet_str=njet_ch,flav_str=flav_ch)

                                        # Get the cuts mask for all selections
                                        if dense_axis_name == "njets":
                                            all_cuts_mask = (selections.all(*cuts_lst) & njets_any_mask)
                                        else:
                                            all_cuts_mask = selections.all(*cuts_lst)

                                        # Apply the optional cut on energy of the event
                                        if self._ecut_threshold is not None:
                                            all_cuts_mask = (all_cuts_mask & ecut_mask)

                                        # Weights and eft coeffs
                                        weights_flat = weight[all_cuts_mask]
                                        eft_coeffs_cut = eft_coeffs[all_cuts_mask] if eft_coeffs is not None else None

                                        # Fill the histos
                                        axes_fill_info_dict = {
                                            dense_axis_name : dense_axis_vals[all_cuts_mask],
                                            "channel"       : ch_name,
                                            "appl"          : appl,
                                            "process"       : histAxisName,
                                            "systematic"    : wgt_fluct,
                                            "weight"        : weights_flat,
                                            "eft_coeff"     : eft_coeffs_cut,
                                        }
                                        axis_lst = ["njets", "ptz", "lj0pt"]
                                        if dense_axis_name not in axis_lst:
                                            continue

                                        # Skip histos that are not defined (or not relevant) to given categories
                                        if ((("j0" in dense_axis_name) and ("lj0pt" not in dense_axis_name)) & (("CRZ" in ch_name) or ("CRflip" in ch_name))): continue
                                        if ((("j0" in dense_axis_name) and ("lj0pt" not in dense_axis_name)) & ("0j" in ch_name)): continue
                                        if self._offZ_split:
                                            if (("ptz" in dense_axis_name) & ("onZ" not in lep_chan) & ("offZ_high" not in lep_chan) & ("offZ_low" not in lep_chan)):continue
                                        else:
                                            if (("ptz" in dense_axis_name) & ("onZ" not in lep_chan)): continue
                                            if (("ptz" in dense_axis_name) & ("2lss" in lep_chan)): continue
                                        if ((dense_axis_name in ["o0pt","b0pt","bl0pt"]) & ("CR" in ch_name)): continue

                                        hout[dense_axis_name].fill(**axes_fill_info_dict)
                                        axes_fill_info_dict = {
                                            dense_axis_name+"_sumw2" : dense_axis_vals[all_cuts_mask],
                                            "channel"       : ch_name,
                                            "appl"          : appl,
                                            "process"       : histAxisName,
                                            "systematic"    : wgt_fluct,
                                            "weight"        : np.square(weights_flat),
                                            "eft_coeff"     : eft_coeffs_cut,
                                        }
                                        hout[dense_axis_name+"_sumw2"].fill(**axes_fill_info_dict)

                                        # Do not loop over lep flavors if not self._split_by_lepton_flavor, it's a waste of time and also we'd fill the hists too many times
                                        if not self._split_by_lepton_flavor: break

                            # Do not loop over njets if hist is njets (otherwise we'd fill the hist too many times)
                            if dense_axis_name == "njets": break

        return hout

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    # Load the .coffea files
    outpath= './coffeaFiles/'
    samples     = load(outpath+'samples.coffea')
    topprocessor = AnalysisProcessor(samples)
