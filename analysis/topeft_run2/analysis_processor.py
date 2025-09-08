#!/usr/bin/env python
import copy
import coffea
import numpy as np
import awkward as ak
import json
import os
import yaml

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

    def __init__(self, sample, wc_names_lst=[], hist_key=None, var_info=None, ecut_threshold=None, do_errors=False, do_systematics=False, split_by_lepton_flavor=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32, rebin=False, offZ_split=False, tau_h_analysis=False, fwd_analysis=False, channel_dict=None):

        self._sample = sample
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype
        self.offZ_3l_split = offZ_split
        self.tau_h_analysis = tau_h_analysis
        self.fwd_analysis = fwd_analysis
        if channel_dict is None:
            raise ValueError("channel_dict must be provided and cannot be None")
        self._channel_dict = channel_dict

        histogram = {}

        metadata_path = topeft_path("params/metadata.yml")
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)

        if hist_key is None or var_info is None:
            raise ValueError("hist_key and var_info must be provided and cannot be None")

        var, ch, appl, sample_name, syst = hist_key
        info = var_info

        self._var = var
        self._channel = ch
        self._appregion = appl
        self._syst = syst
        self._var_def = info.get("definition")
        if self._var_def is None:
            raise ValueError(f"No definition provided for variable {var}")

        if var not in metadata["variables"]:
            raise ValueError(f"Unknown variable {var}")
        # if ch not in metadata["channels"]:
        #     raise ValueError(f"Unknown channel {ch}")
        # if appl not in metadata["applications"]:
        #     raise ValueError(f"Unknown application region {appl}")
        if syst not in metadata["systematics"]:
            raise ValueError(f"Unknown systematic {syst}")

        sumw2_key = (var + "_sumw2", ch, appl, sample_name, syst)

        if not rebin and "variable" in info:
            dense_axis = hist.axis.Variable(
                info["variable"], name=var, label=info["label"]
            )
            sumw2_axis = hist.axis.Variable(
                info["variable"], name=var+"_sumw2", label=info["label"] + " sum of w^2"
            )
        else:
            dense_axis = hist.axis.Regular(
                *info["regular"], name=var, label=info["label"]
            )
            sumw2_axis = hist.axis.Regular(
                *info["regular"], name=var+"_sumw2", label=info["label"] + " sum of w^2"
            )

        histogram[hist_key] = HistEFT(
            dense_axis,
            wc_names=wc_names_lst,
            label=r"Events",
        )
        histogram[sumw2_key] = HistEFT(
            sumw2_axis,
            wc_names=wc_names_lst,
            label=r"Events",
        )

        # Set the list of hists to fill
        self._hist_keys_to_fill = [hist_key, sumw2_key]

        self._accumulator = histogram

        # Set the energy threshold to cut on
        self._ecut_threshold = ecut_threshold

        # Set the booleans
        self._do_errors = do_errors # Whether to calculate and store the w**2 coefficients
        self._do_systematics = do_systematics # Whether to process systematic sample
        self._split_by_lepton_flavor = split_by_lepton_flavor # Whether to keep track of lepton flavors individually
        self._skip_signal_regions = skip_signal_regions # Whether to skip the SR categories
        self._skip_control_regions = skip_control_regions # Whether to skip the CR categories



    @property
    def accumulator(self):
        return self._accumulator

    @property
    def sample(self):
        return self._sample

    @property
    def hist_keys_to_fill(self):
        return self._hist_keys_to_fill

    @property
    def var(self):
        return self._var

    @property
    def var_def(self):
        return self._var_def

    @property
    def channel(self):
        return self._channel

    @property
    def appregion(self):    
        return self._appregion

    @property
    def syst(self):
        return self._syst

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, events):

        # Dataset parameters
        dataset = events.metadata["dataset"]
        isEFT   = self._sample["WCnames"] != []

        isData             = self._sample["isData"]
        histAxisName       = self._sample["histAxisName"]
        year               = self._sample["year"]
        xsec               = self._sample["xsec"]
        sow                = self._sample["nSumOfWeights"]

        is_run3 = False
        if year.startswith("202"):
            is_run3 = True
        is_run2 = not is_run3

        run_era = None
        if isData:
            run_era = self._sample["path"].split("/")[2].split("-")[0][-1]

        # Get up down weights from input dict
        if (self._do_systematics and not isData):
            if histAxisName in get_te_param("lo_xsec_samples"):
                # We have a LO xsec for these samples, so for these systs we will have e.g. xsec_LO*(N_pass_up/N_gen_nom)
                # Thus these systs will cover the cross section uncty and the acceptance and effeciency and shape
                # So no NLO rate uncty for xsec should be applied in the text data card
                sow_ISRUp          = self._sample["nSumOfWeights"]
                sow_ISRDown        = self._sample["nSumOfWeights"]
                sow_FSRUp          = self._sample["nSumOfWeights"]
                sow_FSRDown        = self._sample["nSumOfWeights"]
                sow_renormUp       = self._sample["nSumOfWeights"]
                sow_renormDown     = self._sample["nSumOfWeights"]
                sow_factUp         = self._sample["nSumOfWeights"]
                sow_factDown       = self._sample["nSumOfWeights"]
                sow_renormfactUp   = self._sample["nSumOfWeights"]
                sow_renormfactDown = self._sample["nSumOfWeights"]
            else:
                # Otherwise we have an NLO xsec, so for these systs we will have e.g. xsec_NLO*(N_pass_up/N_gen_up)
                # Thus these systs should only affect acceptance and effeciency and shape
                # The uncty on xsec comes from NLO and is applied as a rate uncty in the text datacard
                sow_ISRUp          = self._sample["nSumOfWeights_ISRUp"          ]
                sow_ISRDown        = self._sample["nSumOfWeights_ISRDown"        ]
                sow_FSRUp          = self._sample["nSumOfWeights_FSRUp"          ]
                sow_FSRDown        = self._sample["nSumOfWeights_FSRDown"        ]
                sow_renormUp       = self._sample["nSumOfWeights_renormUp"       ]
                sow_renormDown     = self._sample["nSumOfWeights_renormDown"     ]
                sow_factUp         = self._sample["nSumOfWeights_factUp"         ]
                sow_factDown       = self._sample["nSumOfWeights_factDown"       ]
                sow_renormfactUp   = self._sample["nSumOfWeights_renormfactUp"   ]
                sow_renormfactDown = self._sample["nSumOfWeights_renormfactDown" ]
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

        datasets = ["Muon", "SingleMuon", "SingleElectron", "EGamma", "MuonEG", "DoubleMuon", "DoubleElectron", "DoubleEG"]
        for d in datasets:
            if dataset.startswith(d):
                dataset = dataset.split('_')[0]

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


        if is_run3:
            leptonSelection = te_os.run3leptonselection()
            jetsRho = events.Rho["fixedGridRhoFastjetAll"]
        elif is_run2:
            leptonSelection = te_os.run2leptonselection()
            jetsRho = events.fixedGridRhoFastjetAll

        # An array of lenght events that is just 1 for each event
        # Probably there's a better way to do this, but we use this method elsewhere so I guess why not..
        events.nom = ak.ones_like(events.MET.pt)

        ele["idEmu"] = te_os.ttH_idEmu_cuts_E3(ele.hoe, ele.eta, ele.deltaEtaSC, ele.eInvMinusPInv, ele.sieie)
        ele["conept"] = leptonSelection.coneptElec(ele)
        mu["conept"] = leptonSelection.coneptMuon(mu)
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
        elif year == "2022":
            golden_json_path = topcoffea_path("data/goldenJsons/Cert_Collisions2022_355100_362760_Golden.txt")
        elif year == "2023":
            golden_json_path = topcoffea_path("data/goldenJsons/Cert_Collisions2023_366442_370790_Golden.txt")
        else:
            raise ValueError(f"Error: Unknown year \"{year}\".")
        lumi_mask = LumiMask(golden_json_path)(events.run,events.luminosityBlock)

        ######### EFT coefficients ##########

        # Extract the EFT quadratic coefficients and optionally use them to calculate the coefficients on the w**2 quartic function
        # eft_coeffs is never Jagged so convert immediately to numpy for ease of use.
        eft_coeffs = ak.to_numpy(events["EFTfitCoefficients"]) if hasattr(events, "EFTfitCoefficients") else None
        if eft_coeffs is not None:
            # Check to see if the ordering of WCs for this sample matches what want
            if self._sample["WCnames"] != self._wc_names_lst:
                eft_coeffs = efth.remap_coeffs(self._sample["WCnames"], self._wc_names_lst, eft_coeffs)
        eft_w2_coeffs = efth.calc_w2_coeffs(eft_coeffs,self._dtype) if (self._do_errors and eft_coeffs is not None) else None
        # Initialize the out object
        hout = self.accumulator

        ################### Electron selection ####################

        ele["isPres"] = leptonSelection.isPresElec(ele)
        ele["isLooseE"] = leptonSelection.isLooseElec(ele)
        ele["isFO"] = leptonSelection.isFOElec(ele, year)
        ele["isTightLep"] = leptonSelection.tightSelElec(ele)

        ################### Muon selection ####################

        mu["pt"] = ApplyRochesterCorrections(year, mu, isData) # Run3 ready
        mu["isPres"] = leptonSelection.isPresMuon(mu)
        mu["isLooseM"] = leptonSelection.isLooseMuon(mu)
        mu["isFO"] = leptonSelection.isFOMuon(mu, year)
        mu["isTightLep"]= leptonSelection.tightSelMuon(mu)

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
        AttachElectronSF(e_fo,year=year, looseWP="none" if is_run3 else "wpLnoiso") #Run3 ready
        AttachMuonSF(m_fo,year=year)

        # Attach per lepton fake rates
        AttachPerLeptonFR(e_fo, flavor = "Elec", year=year)
        AttachPerLeptonFR(m_fo, flavor = "Muon", year=year)
        m_fo['convVeto'] = ak.ones_like(m_fo.charge)
        m_fo['lostHits'] = ak.zeros_like(m_fo.charge)
        l_fo = ak.with_name(ak.concatenate([e_fo, m_fo], axis=1), 'PtEtaPhiMCandidate')
        l_fo_conept_sorted = l_fo[ak.argsort(l_fo.conept, axis=-1,ascending=False)]

        ################### Tau selection ####################

        if self.tau_h_analysis:
            tau["pt"], tau["mass"] = ApplyTES(year, tau, isData)
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
        obj_jes_entries = []
        with open(topeft_path('modules/jerc_dict.json'), 'r') as f:
            jerc_dict = json.load(f)
            for junc in jerc_dict[year]['junc']:
                junc = junc.replace("Regrouped_", "")
                obj_jes_entries.append(f'JES_{junc}Up')
                obj_jes_entries.append(f'JES_{junc}Down')
        obj_correction_syst_lst = [
            f'JER_{year}Up', f'JER_{year}Down'
        ] + obj_jes_entries
        if self.tau_h_analysis:
            obj_correction_syst_lst.append("TESUp")
            obj_correction_syst_lst.append("TESDown")
            obj_correction_syst_lst.append("FESUp")
            obj_correction_syst_lst.append("FESDown")

        wgt_correction_syst_lst = [
            "lepSF_muonUp","lepSF_muonDown","lepSF_elecUp","lepSF_elecDown",f"btagSFbc_{year}Up",f"btagSFbc_{year}Down","btagSFbc_corrUp","btagSFbc_corrDown",f"btagSFlight_{year}Up",f"btagSFlight_{year}Down","btagSFlight_corrUp","btagSFlight_corrDown","PUUp","PUDown","PreFiringUp","PreFiringDown",f"triggerSF_{year}Up",f"triggerSF_{year}Down", # Exp systs
            "FSRUp","FSRDown","ISRUp","ISRDown","renormUp","renormDown","factUp","factDown", # Theory systs
        ]
        if self.tau_h_analysis:
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
            if eft_coeffs is None:
                genw = events["genWeight"]
            else:
                genw = np.ones_like(events["event"])

            # Normalize by (xsec/sow)*genw where genw is 1 for EFT samples
            # Note that for theory systs, will need to multiply by sow/sow_wgtUP to get (xsec/sow_wgtUp)*genw and same for Down
            lumi = 1000.0*get_tc_param(f"lumi_{year}")
            weights_obj_base.add("norm",(xsec/sow)*genw*lumi)

            if is_run2:
                l1prefiring_args = [events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn]
            elif is_run3:
                l1prefiring_args = [ak.ones_like(events.nom), ak.ones_like(events.nom), ak.ones_like(events.nom)]

            # Attach PS weights (ISR/FSR) and scale weights (renormalization/factorization) and PDF weights
            tc_cor.AttachPSWeights(events) #Run3 ready
            tc_cor.AttachScaleWeights(events) #Run3 ready (with caveat on "nominal")
            #AttachPdfWeights(events) #TODO
            # FSR/ISR weights -- corrections come from AttachPSWeights
            weights_obj_base.add('ISR', events.nom, events.ISRUp*(sow/sow_ISRUp), events.ISRDown*(sow/sow_ISRDown))
            weights_obj_base.add('FSR', events.nom, events.FSRUp*(sow/sow_FSRUp), events.FSRDown*(sow/sow_FSRDown))
            # renorm/fact scale  -- corrections come from AttachScaleWeights
            weights_obj_base.add('renorm', events.nom, events.renormUp*(sow/sow_renormUp), events.renormDown*(sow/sow_renormDown))
            weights_obj_base.add('fact', events.nom, events.factUp*(sow/sow_factUp), events.factDown*(sow/sow_factDown))
            # Prefiring and PU (note prefire weights only available in nanoAODv9 and for Run2)
            weights_obj_base.add('PreFiring', *l1prefiring_args) #Run3 ready
            weights_obj_base.add('PU', tc_cor.GetPUSF((events.Pileup.nTrueInt), year), tc_cor.GetPUSF(events.Pileup.nTrueInt, year, 'up'), tc_cor.GetPUSF(events.Pileup.nTrueInt, year, 'down')) #Run3 ready


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
            if self.tau_h_analysis:
                vetos_tocleanjets = ak.with_name( ak.concatenate([cleaning_taus, l_fo], axis=1), "PtEtaPhiMCandidate")
            else:
                vetos_tocleanjets = ak.with_name( l_fo, "PtEtaPhiMCandidate")
            tmp = ak.cartesian([ak.local_index(jets.pt), vetos_tocleanjets.jetIdx], nested=True)
            cleanedJets = jets[~ak.any(tmp.slot0 == tmp.slot1, axis=-1)] # this line should go before *any selection*, otherwise lep.jetIdx is not aligned with the jet index

            # Selecting jets and cleaning them
            jetptname = "pt_nom" if hasattr(cleanedJets, "pt_nom") else "pt"

            cleanedJets["pt_raw"] = (1 - cleanedJets.rawFactor)*cleanedJets.pt
            cleanedJets["mass_raw"] = (1 - cleanedJets.rawFactor)*cleanedJets.mass
            cleanedJets["rho"] = ak.broadcast_arrays(jetsRho, cleanedJets.pt)[0]

            # Jet energy corrections
            if not isData:
                cleanedJets["pt_gen"] = ak.values_astype(ak.fill_none(cleanedJets.matched_gen.pt, 0), np.float32)
                if self.tau_h_analysis:
                    tau["pt"], tau["mass"]      = ApplyTESSystematic(year, tau, isData, syst_var)
                    tau["pt"], tau["mass"]      = ApplyFESSystematic(year, tau, isData, syst_var)

            events_cache = events.caches[0]
            cleanedJets = ApplyJetCorrections(year, corr_type='jets', isData=isData, era=run_era).build(cleanedJets, lazy_cache=events_cache)  #Run3 ready
            cleanedJets = ApplyJetSystematics(year,cleanedJets,syst_var)
            met = ApplyJetCorrections(year, corr_type='met', isData=isData, era=run_era).build(met_raw, cleanedJets, lazy_cache=events_cache)

            cleanedJets["isGood"] = tc_os.is_tight_jet(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, pt_cut=30., eta_cut=get_te_param("eta_j_cut"), id_cut=get_te_param("jet_id_cut"))
            cleanedJets["isFwd"] = te_os.isFwdJet(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, jetPtCut=40.)
            goodJets = cleanedJets[cleanedJets.isGood]
            fwdJets  = cleanedJets[cleanedJets.isFwd]

            # Count jets
            njets = ak.num(goodJets)
            nfwdj = ak.num(fwdJets)
            if "ht" in self._var_def:
                ht = ak.sum(goodJets.pt, axis=-1)
            if "j0" in self._var_def:
                j0 = goodJets[ak.argmax(goodJets.pt, axis=-1, keepdims=True)]

            # Loose DeepJet WP
            loose_tag = "btag_wp_loose_" + year.replace("201", "UL1")
            btagwpl = get_tc_param(loose_tag)
            isBtagJetsLoose = (goodJets.btagDeepFlavB > btagwpl)
            isNotBtagJetsLoose = np.invert(isBtagJetsLoose)
            nbtagsl = ak.num(goodJets[isBtagJetsLoose])

            # Medium DeepJet WP
            medium_tag = "btag_wp_medium_" + year.replace("201", "UL1")
            btagwpm = get_tc_param(medium_tag)
            isBtagJetsMedium = (goodJets.btagDeepFlavB > btagwpm)
            isNotBtagJetsMedium = np.invert(isBtagJetsMedium)
            nbtagsm = ak.num(goodJets[isBtagJetsMedium])

            #################### Add variables into event object so that they persist ####################

            # Put njets and l_fo_conept_sorted into events
            events["njets"] = njets
            events["l_fo_conept_sorted"] = l_fo_conept_sorted

            # The event selection
            te_es.add1lMaskAndSFs(events, year, isData, sampleType)
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
            cat_dict = self._channel_dict

            lep_cats = list(cat_dict.keys())
            lep_cats_data = [lep_cat for lep_cat in lep_cats if (lep_cat.startswith("2l") and not "os" in lep_cat)]

            weights_dict = {}

            for ch_name in lep_cats:
                # For both data and MC
                weights_dict[ch_name] = copy.deepcopy(weights_obj_base_for_kinematic_syst)
                if ch_name.startswith("1l"):
                    weights_dict[ch_name].add("FF", events.fakefactor_1l, copy.deepcopy(events.fakefactor_1l_up), copy.deepcopy(events.fakefactor_1l_down))
                    weights_dict[ch_name].add("FFpt",  events.nom, copy.deepcopy(events.fakefactor_1l_pt1/events.fakefactor_1l), copy.deepcopy(events.fakefactor_1l_pt2/events.fakefactor_1l))
                    weights_dict[ch_name].add("FFeta", events.nom, copy.deepcopy(events.fakefactor_1l_be1/events.fakefactor_1l), copy.deepcopy(events.fakefactor_1l_be2/events.fakefactor_1l))
                    weights_dict[ch_name].add(f"FFcloseEl_{year}", events.nom, copy.deepcopy(events.fakefactor_1l_elclosureup/events.fakefactor_1l), copy.deepcopy(events.fakefactor_1l_elclosuredown/events.fakefactor_1l))
                    weights_dict[ch_name].add(f"FFcloseMu_{year}", events.nom, copy.deepcopy(events.fakefactor_1l_muclosureup/events.fakefactor_1l), copy.deepcopy(events.fakefactor_1l_muclosuredown/events.fakefactor_1l))
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
                    if ch_name in lep_cats_data:
                        weights_dict[ch_name].add("fliprate", events.flipfactor_2l)

                # For MC only
                if not isData:
                    if ch_name.startswith("1l"):
                        weights_dict[ch_name].add("lepSF_muon", events.sf_1l_muon, copy.deepcopy(events.sf_1l_hi_muon), copy.deepcopy(events.sf_1l_lo_muon))
                        weights_dict[ch_name].add("lepSF_elec", events.sf_1l_elec, copy.deepcopy(events.sf_1l_hi_elec), copy.deepcopy(events.sf_1l_lo_elec))
                        if self.tau_h_analysis:
                            weights_dict[ch_name].add("lepSF_taus_real", events.sf_2l_taus_real, copy.deepcopy(events.sf_2l_taus_real_hi), copy.deepcopy(events.sf_2l_taus_real_lo))
                            weights_dict[ch_name].add("lepSF_taus_fake", events.sf_2l_taus_fake, copy.deepcopy(events.sf_2l_taus_fake_hi), copy.deepcopy(events.sf_2l_taus_fake_lo))
                    elif ch_name.startswith("2l"):
                        weights_dict[ch_name].add("lepSF_muon", events.sf_2l_muon, copy.deepcopy(events.sf_2l_hi_muon), copy.deepcopy(events.sf_2l_lo_muon))
                        weights_dict[ch_name].add("lepSF_elec", events.sf_2l_elec, copy.deepcopy(events.sf_2l_hi_elec), copy.deepcopy(events.sf_2l_lo_elec))
                        if self.tau_h_analysis:
                            weights_dict[ch_name].add("lepSF_taus_real", events.sf_2l_taus_real, copy.deepcopy(events.sf_2l_taus_real_hi), copy.deepcopy(events.sf_2l_taus_real_lo))
                            weights_dict[ch_name].add("lepSF_taus_fake", events.sf_2l_taus_fake, copy.deepcopy(events.sf_2l_taus_fake_hi), copy.deepcopy(events.sf_2l_taus_fake_lo))
                    elif ch_name.startswith("3l"):
                        weights_dict[ch_name].add("lepSF_muon", events.sf_3l_muon, copy.deepcopy(events.sf_3l_hi_muon), copy.deepcopy(events.sf_3l_lo_muon))
                        weights_dict[ch_name].add("lepSF_elec", events.sf_3l_elec, copy.deepcopy(events.sf_3l_hi_elec), copy.deepcopy(events.sf_3l_lo_elec))
                        if self.tau_h_analysis:
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
            if self.offZ_3l_split:
                sfosz_3l_OffZ_low_mask = tc_es.get_off_Z_mask_low(l_fo_conept_sorted_padded[:,0:3],pt_window=0.0)
                sfosz_3l_OffZ_any_mask = tc_es.get_any_sfos_pair(l_fo_conept_sorted_padded[:,0:3])
            sfosz_2l_mask = tc_es.get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:2],pt_window=10.0)
            sfasz_2l_mask = tc_es.get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:2],pt_window=30.0,flavor="as") # Any sign (do not enforce ss or os here)
            if self.tau_h_analysis:
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
            fwdjet_mask       = (nfwdj > 0)  # Used for ttW EWK enriched regions

            # Charge masks
            chargel0_p = ak.fill_none(((l0.charge)>0),False)
            chargel0_m = ak.fill_none(((l0.charge)<0),False)
            charge2l_0 = ak.fill_none(((l0.charge+l1.charge)==0),False)
            charge2l_1 = ak.fill_none(((l0.charge+l1.charge)!=0),False)
            charge3l_p = ak.fill_none(((l0.charge+l1.charge+l2.charge)>0),False)
            charge3l_m = ak.fill_none(((l0.charge+l1.charge+l2.charge)<0),False)
            if self.tau_h_analysis:
                tau_F_mask = (ak.num(tau[tau["isVLoose"]>0]) >=1)
                tau_L_mask  = (ak.num(tau[tau["isLoose"]>0]) >=1)
                no_tau_mask = (ak.num(tau[tau["isLoose"]>0])==0)


            ######### Store boolean masks with PackedSelection ##########

            selections = PackedSelection(dtype='uint64')
            preselections = PackedSelection(dtype='uint64')
            # Lumi mask (for data)
            selections.add("is_good_lumi",lumi_mask)
            preselections.add("is_good_lumi",lumi_mask)

            # 2lss selection
            preselections.add("chargedl0", (chargel0_p | chargel0_m))
            preselections.add("2l_nozeeveto", (events.is2l_nozeeveto & pass_trg))
            preselections.add("2los", charge2l_0)
            preselections.add("2lem", events.is_em)
            preselections.add("2lee", events.is_ee)
            preselections.add("2lmm", events.is_mm)
            preselections.add("2l_onZ_as", sfasz_2l_mask)
            preselections.add("2l_onZ", sfosz_2l_mask)
            preselections.add("bmask_atleast3m", (bmask_atleast3med))
            preselections.add("bmask_atleast1m2l", (bmask_atleast1med_atleast2loose))
            preselections.add("bmask_atmost2m", (bmask_atmost2med))
            preselections.add("fwdjet_mask", (fwdjet_mask))
            preselections.add("~fwdjet_mask", (~fwdjet_mask))
            if self.tau_h_analysis:
                preselections.add("1l", (events.is1l & pass_trg))
                preselections.add("1tau", (tau_L_mask))
                preselections.add("1Ftau", (tau_F_mask))
                preselections.add("0tau", (no_tau_mask))
                preselections.add("onZ_tau", (tl_zpeak_mask))
                preselections.add("offZ_tau", (~tl_zpeak_mask))
            if self.fwd_analysis:
                preselections.add("2lss_fwd", (events.is2l & pass_trg & fwdjet_mask))
                preselections.add("2l_fwd_p", (chargel0_p & fwdjet_mask))
                preselections.add("2l_fwd_m", (chargel0_m & fwdjet_mask))

            # 2lss selection
            preselections.add("2lss", (events.is2l & pass_trg))
            preselections.add("2l_p", (chargel0_p))
            preselections.add("2l_m", (chargel0_m))

            # 3l selection
            preselections.add("3l", (events.is3l & pass_trg))
            preselections.add("bmask_exactly0m", (bmask_exactly0med))
            preselections.add("bmask_exactly1m", (bmask_exactly1med))
            preselections.add("bmask_exactly2m", (bmask_exactly2med))
            preselections.add("bmask_atleast2m", (bmask_atleast2med))
            preselections.add("3l_p", (events.is3l & pass_trg & charge3l_p))
            preselections.add("3l_m", (events.is3l & pass_trg & charge3l_m))
            preselections.add("3l_onZ", (sfosz_3l_OnZ_mask))

            if self.offZ_3l_split:
                preselections.add("3l_offZ_low", (sfosz_3l_OffZ_mask & sfosz_3l_OffZ_any_mask & sfosz_3l_OffZ_low_mask))
                preselections.add("3l_offZ_high", (sfosz_3l_OffZ_mask & sfosz_3l_OffZ_any_mask & ~sfosz_3l_OffZ_low_mask))
                preselections.add("3l_offZ_none", (sfosz_3l_OffZ_mask & ~sfosz_3l_OffZ_any_mask))
            else:
                preselections.add("3l_offZ", (sfosz_3l_OffZ_mask))

            # 4l selection
            preselections.add("4l", (events.is4l & pass_trg))

            for lep_cat, jet_dict in cat_dict.items():
                # lep_chan definitions are the same for each jet bin, so take the first one
                lep_ch_list = next(iter(jet_dict.values()))["lep_chan_def_lst"]
                for lep_ch in lep_ch_list:
                    tempmask = None
                    chtag = lep_ch[0]
                    for chcut in lep_ch[1:]:
                        if tempmask is not None:
                            tempmask = tempmask & preselections.any(chcut)
                        else:
                            tempmask = preselections.any(chcut)
                    selections.add(chtag, tempmask)

            del preselections

            # Lep flavor selection
            selections.add("e",  events.is_e)
            selections.add("m",  events.is_m)
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
            selections.add("atleast_6j", (njets>=6))
            selections.add("atleast_7j", (njets>=7))
            selections.add("atleast_0j", (njets>=0))
            selections.add("atmost_3j" , (njets<=3))

            # AR/SR categories
            selections.add("isSR_2lSS",    ( events.is2l_SR) & charge2l_1)
            selections.add("isAR_2lSS",    (~events.is2l_SR) & charge2l_1)
            selections.add("isAR_2lSS_OS", ( events.is2l_SR) & charge2l_0) # Sideband for the charge flip
            selections.add("isSR_2lOS",    ( events.is2l_SR) & charge2l_0)
            selections.add("isAR_2lOS",    (~events.is2l_SR) & charge2l_0)
            if self.tau_h_analysis:
                selections.add("isSR_1l",    ( events.is1l_SR))

            selections.add("isSR_3l",  events.is3l_SR)
            selections.add("isAR_3l", ~events.is3l_SR)
            selections.add("isSR_4l",  events.is4l_SR)



            ######### Variables for the dense axes of the hists ##########

            var_def = self.var_def

            if ("ptbl" in var_def) or ("b0pt" in var_def) or ("bl0pt" in var_def):
                ptbl_bjet = goodJets[(isBtagJetsMedium | isBtagJetsLoose)]
                ptbl_bjet = ptbl_bjet[ak.argmax(ptbl_bjet.pt, axis=-1, keepdims=True)]
                ptbl_lep = l_fo_conept_sorted
                ptbl = (ptbl_bjet.nearest(ptbl_lep) + ptbl_bjet).pt
                ptbl = ak.values_astype(ak.fill_none(ptbl, -1), np.float32)

            if "ptz" in var_def:
                ptz = te_es.get_Z_pt(l_fo_conept_sorted_padded[:,0:3],10.0)
                if self.offZ_3l_split:
                    ptz = te_es.get_ll_pt(l_fo_conept_sorted_padded[:,0:3],10.0)
            if "ptz_wtau" in var_def:
                ptz_wtau = te_es.get_Zlt_pt(l0, l1, tau0)

            if "bl0pt" in var_def:
                bjetsl = goodJets[isBtagJetsLoose][ak.argsort(goodJets[isBtagJetsLoose].pt, axis=-1, ascending=False)]
                bl_pairs = ak.cartesian({"b": bjetsl, "l": l_fo_conept_sorted})
                blpt = (bl_pairs["b"] + bl_pairs["l"]).pt
                bl0pt = ak.flatten(blpt[ak.argmax(blpt, axis=-1, keepdims=True)])

            need_lj_collection = any(t in var_def for t in ["o0pt", "lj0pt", "ljptsum"]) or (self._ecut_threshold is not None)
            if need_lj_collection:
                if self.tau_h_analysis:
                    l_j_collection = ak.with_name(ak.concatenate([l_fo_conept_sorted, goodJets, cleaning_taus], axis=1), "PtEtaPhiMCollection")
                else:
                    l_j_collection = ak.with_name(ak.concatenate([l_fo_conept_sorted, goodJets], axis=1), "PtEtaPhiMCollection")
                if "o0pt" in var_def:
                    o0pt = ak.max(l_j_collection.pt, axis=-1)
                if ("ljptsum" in var_def) or (self._ecut_threshold is not None):
                    ljptsum = ak.sum(l_j_collection.pt, axis=-1)
                if "lj0pt" in var_def:
                    l_j_pairs = ak.combinations(l_j_collection, 2, fields=["o0","o1"])
                    l_j_pairs_pt = (l_j_pairs.o0 + l_j_pairs.o1).pt
                    lj0pt = ak.max(l_j_pairs_pt, axis=-1)

            if "lt" in var_def:
                lt = ak.sum(l_fo_conept_sorted_padded.pt, axis=-1) + met.pt

            if "mll_0_1" in var_def:
                mll_0_1 = (l0 + l1).mass

            if self._ecut_threshold is not None:
                if "ljptsum" not in locals():
                    ljptsum = ak.sum(l_j_collection.pt, axis=-1)
                ecut_mask = (ljptsum < self._ecut_threshold)

            counts = np.ones_like(events['event'])

            ########## Fill the histograms ##########
            cat_dict = self._channel_dict

            # Loop over the hists we want to fill
            dense_axis_name = self._var
            dense_axis_vals = eval(self._var_def, {"ak": ak, "np": np}, locals())
            
            # print("\n\n\n\n\n\n")
            # print(f"Filling histograms for variable: {dense_axis_name}")
            # print("dense_axis_vals:", dense_axis_vals)
            # print("self._var_def:", self._var_def)
            # print("\n\n\n\n\n\n")
            
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

                                    if ch_name != self.channel:
                                        continue

                                    # print("\n\n\n\n\ndense_axis_name:", dense_axis_name, "ch_name:", ch_name)
                                    # print("\n\n\n\n\n")

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
                                        #"channel"       : ch_name,
                                        #"appl"          : appl,
                                        #"process"       : histAxisName,
                                        #"systematic"    : wgt_fluct,
                                        "weight"        : weights_flat,
                                        "eft_coeff"     : eft_coeffs_cut,
                                    }

                                    # Skip histos that are not defined (or not relevant) to given categories
                                    if ((("j0" in dense_axis_name) and ("lj0pt" not in dense_axis_name)) & (("CRZ" in ch_name) or ("CRflip" in ch_name))): continue
                                    if ((("j0" in dense_axis_name) and ("lj0pt" not in dense_axis_name)) & ("0j" in ch_name)): continue
                                    if self.offZ_3l_split:
                                        if (("ptz" in dense_axis_name) & ("onZ" not in lep_chan) & ("offZ_high" not in lep_chan) & ("offZ_low" not in lep_chan)):continue
                                    elif self.tau_h_analysis:
                                        if (("ptz" in dense_axis_name) and ("onZ" not in lep_chan)): continue
                                        if (("ptz" in dense_axis_name) and ("2lss" in lep_chan) and ("ptz_wtau" not in dense_axis_name)): continue
                                        if (("ptz_wtau" in dense_axis_name) and (("1tau" not in lep_chan) or ("onZ" not in lep_chan) or ("2lss" not in lep_chan))): continue

                                    elif self.fwd_analysis:
                                        if (("ptz" in dense_axis_name) & ("onZ" not in lep_chan)): continue
                                        if (("lt" in dense_axis_name) and ("2lss" not in lep_chan)): continue
                                    else:
                                        if (("ptz" in dense_axis_name) & ("onZ" not in lep_chan)): continue
                                    if ((dense_axis_name in ["o0pt","b0pt","bl0pt"]) & ("CR" in ch_name)): continue
                                    histkey = (dense_axis_name, ch_name, appl, dataset, wgt_fluct)
                                    if histkey not in hout.keys():
                                        continue
                                    hout[histkey].fill(**axes_fill_info_dict)
                                    axes_fill_info_dict = {
                                        dense_axis_name+"_sumw2" : dense_axis_vals[all_cuts_mask],
                                        #"channel"       : ch_name,
                                        #"appl"          : appl,
                                        #"process"       : histAxisName,
                                        #"systematic"    : wgt_fluct,
                                        "weight"        : np.square(weights_flat),
                                        "eft_coeff"     : eft_coeffs_cut,
                                    }
                                    histkey = (dense_axis_name+"_sumw2", ch_name, appl, dataset, wgt_fluct)
                                    if histkey not in hout.keys():
                                        continue
                                    hout[histkey].fill(**axes_fill_info_dict)

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
    sample     = load(outpath+'sample.coffea')
    topprocessor = AnalysisProcessor(sample)
