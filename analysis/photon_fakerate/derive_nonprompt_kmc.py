#!/usr/bin/env python
import copy
import coffea
import numpy as np
import awkward as ak
import json

import hist
from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist

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
import topeft.modules.plotting_helper as plot_help
from topeft.modules.corrections import ApplyJetCorrections, GetBtagEff, AttachMuonSF, AttachElectronSF, AttachPhotonSF, AttachPerLeptonFR, ApplyRochesterCorrections, ApplyJetSystematics, GetTriggerSF, ApplyttgammaCF
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

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, ecut_threshold=None, do_errors=False, do_systematics=False, split_by_lepton_flavor=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32, rebin=False, validation_test=False):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype
        self.validation_test = False

        proc_axis = hist.axis.StrCategory([], name="process", growth=True)
        chan_axis = hist.axis.StrCategory([], name="channel", growth=True)
        syst_axis = hist.axis.StrCategory([], name="systematic", label=r"Systematic Uncertainty", growth=True)
        appl_axis = hist.axis.StrCategory([], name="appl", label=r"AR/SR", growth=True)

        histograms = {}
        pt_axis = hist.axis.Variable(axes_info["photon_pt_eta"]["pt"]["variable"],name='pt',label=r"$p_{T}$ $\gamma$ (GeV)")
        pt_sumw2_axis = hist.axis.Variable(axes_info["photon_pt_eta"]["pt"]["variable"],name='pt_sumw2',label=r"$p_{T}$ $\gamma$ (GeV)")
        abseta_axis = hist.axis.Variable(axes_info["photon_pt_eta"]["abseta"]["variable"],name='abseta',label=r"Photon abs. $\eta$")
        abseta_sumw2_axis = hist.axis.Variable(axes_info["photon_pt_eta"]["abseta"]["variable"],name='abseta_sumw2',label=r"Photon abs. $\eta$")

        histograms['photon_pt_eta'] = SparseHist(
            proc_axis,
            chan_axis,
            syst_axis,
            appl_axis,
            pt_axis,
            abseta_axis,
            label=r"Events",
        )

        histograms['photon_pt_eta_sumw2'] = SparseHist(
            proc_axis,
            chan_axis,
            syst_axis,
            appl_axis,
            pt_sumw2_axis,
            abseta_sumw2_axis,
            label=r"Events",
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

        is_run3 = False
        if year.startswith("202"):
            is_run3 = True
        is_run2 = not is_run3

        run_era = None
        if isData:
            run_era = self._samples[dataset]["path"].split("/")[2].split("-")[0][-1]

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
        ph   = events.Photon
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
            if self._samples[dataset]["WCnames"] != self._wc_names_lst:
                eft_coeffs = efth.remap_coeffs(self._samples[dataset]["WCnames"], self._wc_names_lst, eft_coeffs)
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

        tau["isPres"]  = te_os.isPresTau(tau.pt, tau.eta, tau.dxy, tau.dz, tau.idDeepTau2017v2p1VSjet, tau.idDeepTau2017v2p1VSe, tau.idDeepTau2017v2p1VSmu, minpt=20)
        tau["isClean"] = te_os.isClean(tau, l_loose, drmin=0.3)
        tau["isGood"]  =  tau["isClean"] & tau["isPres"]
        tau = tau[tau.isGood] # use these to clean jets
        tau["isTight"] = te_os.isVLooseTau(tau.idDeepTau2017v2p1VSjet) # use these to veto

        ################### Photon selection ###################
        #clean photons collection if there is an overlap with lepton collection
        ph["isClean"] = te_os.isClean(ph, l_fo, drmin=0.4)

        #We select photons that are already cleaned against leptons
        cleanPh = ph[ph.isClean]
        te_os.selectPhoton(cleanPh, pt_val=20.0, eta_val=1.44)

        #Depending on whether we are doing validation test or not, the SR and AR change
        if not self.validation_test:
            ph_fo = cleanPh[cleanPh.fakeablePhoton]
            ph_fo['in_regA'] = ph_fo.mediumPhoton
            ph_fo['in_regB'] = ph_fo.mediumPhoton_regB

        else:
            ph_fo = cleanPh[cleanPh.fakeablePhoton_LRCD_kMC]
            ph_fo['in_regL'] = ph_fo.mediumPhoton_regL
            ph_fo['in_regR'] = ph_fo.mediumPhoton_regR

        ph_fo['in_regC'] = ph_fo.mediumPhoton_regC
        ph_fo['in_regD'] = ph_fo.mediumPhoton_regD

        #Attach Photon SF
        AttachPhotonSF(ph_fo,year=year)

        #pT sort the photon collection
        ph_fo_pt_sorted = ph_fo[ak.argsort(ph_fo.pt,axis=-1,ascending=False)]

        # Photon pairs
        pppairs = ak.combinations(ph_fo, 2, fields=["p0","p1"])

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

        wgt_correction_syst_lst = [
            "lepSF_muonUp","lepSF_muonDown","lepSF_elecUp","lepSF_elecDown",f"btagSFbc_{year}Up",f"btagSFbc_{year}Down","btagSFbc_corrUp","btagSFbc_corrDown",f"btagSFlight_{year}Up",f"btagSFlight_{year}Down","btagSFlight_corrUp","btagSFlight_corrDown","PUUp","PUDown","PreFiringUp","PreFiringDown",f"triggerSF_{year}Up",f"triggerSF_{year}Down", # Exp systs
            "FSRUp","FSRDown","ISRUp","ISRDown","renormUp","renormDown","factUp","factDown", # Theory systs
        ]

        data_syst_lst = [
            "FFUp","FFDown","FFptUp","FFptDown","FFetaUp","FFetaDown",f"FFcloseEl_{year}Up",f"FFcloseEl_{year}Down",f"FFcloseMu_{year}Up",f"FFcloseMu_{year}Down"
        ]
        wgt_correction_syst_lst.extend(["phoSFUp","phoSFDown"])

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
                genw= np.ones_like(events["event"])

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
            vetos_tocleanjets = ak.with_name( l_fo, "PtEtaPhiMCandidate")
            tmp = ak.cartesian([ak.local_index(jets.pt), vetos_tocleanjets.jetIdx], nested=True)
            cleanedJets = jets[~ak.any(tmp.slot0 == tmp.slot1, axis=-1)] # this line should go before *any selection*, otherwise lep.jetIdx is not aligned with the jet index

            #Let's also clean jets against photon collection
            cleanedJets = cleanedJets[te_os.isClean(cleanedJets, ph_fo, drmin=0.4)]

            # Selecting jets and cleaning them
            jetptname = "pt_nom" if hasattr(cleanedJets, "pt_nom") else "pt"

            cleanedJets["pt_raw"] = (1 - cleanedJets.rawFactor)*cleanedJets.pt
            cleanedJets["mass_raw"] = (1 - cleanedJets.rawFactor)*cleanedJets.mass
            cleanedJets["rho"] = ak.broadcast_arrays(jetsRho, cleanedJets.pt)[0]

            # Jet energy corrections
            if not isData:
                cleanedJets["pt_gen"] = ak.values_astype(ak.fill_none(cleanedJets.matched_gen.pt, 0), np.float32)

            events_cache = events.caches[0]
            cleanedJets = ApplyJetCorrections(year, corr_type='jets', isData=isData, era=run_era).build(cleanedJets, lazy_cache=events_cache)  #Run3 ready
            cleanedJets = ApplyJetSystematics(year,cleanedJets,syst_var)
            met = ApplyJetCorrections(year, corr_type='met', isData=isData, era=run_era).build(met_raw, cleanedJets, lazy_cache=events_cache)

            cleanedJets["isGood"] = tc_os.is_tight_jet(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, pt_cut=30., eta_cut=get_te_param("eta_j_cut"), id_cut=get_te_param("jet_id_cut"))

            goodJets = cleanedJets[cleanedJets.isGood]

            # Count jets
            njets = ak.num(goodJets)
            ht = ak.sum(goodJets.pt,axis=-1)
            j0 = ak.firsts(goodJets[ak.argmax(goodJets.pt,axis=-1,keepdims=True)])

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
            events["ph_fo_pt_sorted"] = ph_fo_pt_sorted

            # The event selection
            te_es.add2lMaskAndSFs(events, year, isData, sampleType)
            te_es.addLepCatMasks(events)
            te_es.addPhotonSelection(events, sampleType, last_pt_bin=120.0 ,closureTest=False) #This is purely for photon SF and nothing else
            te_es.categorizePhotons_kMC(events, self.validation_test) #has non-prompt flag on by default

            # Convenient to have l0, l1, l2 on hand
            l_fo_conept_sorted_padded = ak.pad_none(l_fo_conept_sorted, 3)
            l0 = l_fo_conept_sorted_padded[:,0]
            l1 = l_fo_conept_sorted_padded[:,1]

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

                #photon pT correction factor for EFT ttgamma samples
                #Once we have the "good" ttgamma sample, no need to apply any corrections
                if "TTGamma_Dilept_private" in histAxisName:
                    ApplyttgammaCF(year,events)
                    weights_obj_base_for_kinematic_syst.add(f"photonptCF_{year}",events.photon_pt_cf)


            ######### Event weights that do depend on the lep cat ###########
            #This list keeps track of the lepton categories
            lep_cats = ["2los_AR","2los_MR"]

            weights_dict = {}

            for ch_name in lep_cats:
                # For both data and MC
                weights_dict[ch_name] = copy.deepcopy(weights_obj_base_for_kinematic_syst)
                if ch_name.startswith("2l"):
                    weights_dict[ch_name].add("FF", events.fakefactor_2l, copy.deepcopy(events.fakefactor_2l_up), copy.deepcopy(events.fakefactor_2l_down))
                    weights_dict[ch_name].add("FFpt",  events.nom, copy.deepcopy(events.fakefactor_2l_pt1/events.fakefactor_2l), copy.deepcopy(events.fakefactor_2l_pt2/events.fakefactor_2l))
                    weights_dict[ch_name].add("FFeta", events.nom, copy.deepcopy(events.fakefactor_2l_be1/events.fakefactor_2l), copy.deepcopy(events.fakefactor_2l_be2/events.fakefactor_2l))
                    weights_dict[ch_name].add(f"FFcloseEl_{year}", events.nom, copy.deepcopy(events.fakefactor_2l_elclosureup/events.fakefactor_2l), copy.deepcopy(events.fakefactor_2l_elclosuredown/events.fakefactor_2l))
                    weights_dict[ch_name].add(f"FFcloseMu_{year}", events.nom, copy.deepcopy(events.fakefactor_2l_muclosureup/events.fakefactor_2l), copy.deepcopy(events.fakefactor_2l_muclosuredown/events.fakefactor_2l))

                # For MC only
                if not isData:
                    if ch_name.startswith("2l"):
                        weights_dict[ch_name].add("phoSF", events.sf_2l_photon, copy.deepcopy(events.sf_2l_hi_photon), copy.deepcopy(events.sf_2l_lo_photon))
                        weights_dict[ch_name].add("lepSF_muon", events.sf_2l_muon, copy.deepcopy(events.sf_2l_hi_muon), copy.deepcopy(events.sf_2l_lo_muon))
                        weights_dict[ch_name].add("lepSF_elec", events.sf_2l_elec, copy.deepcopy(events.sf_2l_hi_elec), copy.deepcopy(events.sf_2l_lo_elec))
                    else:
                        raise Exception(f"Unknown channel name: {ch_name}")


            ######### Masks we need for the selection ##########

            # Get mask for events that have two sf os leps close to z peak
            sfosz_2los_ll_mask = tc_es.get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:2],pt_window=15.0)
            sfosz_2los_llg_mask_medph = te_es.get_Z_peak_mask_llg(l_fo_conept_sorted_padded[:,0:2],ph_fo_pt_sorted,pt_window=15.0)

            # Pass trigger mask
            pass_trg = tc_es.trg_pass_no_overlap(events,isData,dataset,str(year),te_es.dataset_dict_top22006,te_es.exclude_dict_top22006)

            # b jet masks
            bmask_atleast1med = (nbtagsm>=1) # Used for 2los cats with photons
            bmask_atleast0med = (nbtagsm>=0)

            # Charge masks
            charge2l_0 = ak.fill_none(((l0.charge+l1.charge)==0),False)

            #photon multiplicity mask
            exactly_1ph = (ak.num(ph_fo)==1)

            #Overlap removal
            vetoedbyOverlap = np.ones(len(events), dtype=bool)
            retainedbyOverlap = np.ones(len(events), dtype=bool)
            if not isData:
                if ("TTTo" in dataset) or ("TTGamma" in dataset):
                    te_es.generatorOverlapRemoval(dataset, events,ptCut=10, etaCut=5, deltaRCut=0.1)
                    vetoedbyOverlap = events.vetoedbyOverlap
                    retainedbyOverlap = events.retainedbyOverlap
                elif ("ZGToLLG" in dataset) or ("DY" in dataset):
                    te_es.generatorOverlapRemoval(dataset, events,ptCut=15, etaCut=2.6, deltaRCut=0.05)
                    vetoedbyOverlap = events.vetoedbyOverlap
                    retainedbyOverlap = events.retainedbyOverlap
                elif ("ST_top" in dataset) or ("ST_antitop" in dataset) or ("ST_TWGToLL") in dataset:
                    te_es.generatorOverlapRemoval(dataset, events,ptCut=10, etaCut=3, deltaRCut=0.4)
                    vetoedbyOverlap = events.vetoedbyOverlap
                    retainedbyOverlap = events.retainedbyOverlap

            ######### Store boolean masks with PackedSelection ##########

            selections = PackedSelection(dtype='uint64')

            # Lumi mask (for data)
            selections.add("is_good_lumi",lumi_mask)

            # Njets selection
            selections.add("atleast_1j", (njets>=1))
            selections.add("atleast_0j", (njets>=0))


            # AR/SR categories
            if not self.validation_test:
                selections.add("isSR_2lOS",    ( events.is2l_SR) & charge2l_0 & events.is_regA & bmask_atleast1med)   #the is_regA mask also has non-prompt photon mask baked in for MC
                selections.add("isAR_2lOS",    ( events.is2l_SR) & charge2l_0 & events.is_regB & bmask_atleast1med)   #the is_regB mask also has non-prompt photon mask baked in for MC

            else:
                selections.add("isSR_2lOS",    ( events.is2l_SR) & charge2l_0 & events.is_regL & bmask_atleast1med)   #the is_regL mask also has non-prompt photon mask baked in for MC
                selections.add("isAR_2lOS",    ( events.is2l_SR) & charge2l_0 & events.is_regR & bmask_atleast1med)   #the is_regR mask also has non-prompt photon mask baked in for MC

            selections.add("isMR_2lOS_C",    ( events.is2l_SR) & charge2l_0 & events.is_regC & bmask_atleast0med)   #the is_regC mask also has non-prompt photon mask baked in for MC
            selections.add("isMR_2lOS_D",    ( events.is2l_SR) & charge2l_0 & events.is_regD & bmask_atleast0med)   #the is_regD mask also has non-prompt photon mask baked in for MC

            selections.add("2los_sf",(retainedbyOverlap & events.is2l_nozeeveto & (events.is_ee | events.is_mm) & ~sfosz_2los_ll_mask & charge2l_0 & pass_trg & ~sfosz_2los_llg_mask_medph & exactly_1ph))
            selections.add("2los_of",(retainedbyOverlap & events.is2l & charge2l_0 & events.is_em & pass_trg & exactly_1ph))

            # Lep flavor selection
            selections.add("ee",  events.is_ee)
            selections.add("em",  events.is_em)
            selections.add("mm",  events.is_mm)

            # Njets selection
            selections.add("atleast_1j", (njets>=1))
            selections.add("atleast_0j", (njets>=0))

            ######### Variables for the dense axes of the hists ##########

            # Photon variables
            photon_pt = ak.fill_none(ak.firsts(ph_fo_pt_sorted.pt),-1)
            photon_abseta = ak.fill_none(ak.firsts(abs(ph_fo_pt_sorted.eta)),-1)
            pf_chIso = ak.flatten(ak.fill_none(ak.pad_none(((ph_fo_pt_sorted.pfRelIso03_chg) * (ph_fo_pt_sorted.pt)),1),-1))

            # Counts
            counts = np.ones_like(events['event'])
            ########## Fill the histograms ##########
            # Include SRs and CRs unless we asked to skip them
            cat_dict = {
                "2los_AR" : {
                    "atleast_1j" : {
                        "lep_chan_lst" : ["2los_sf","2los_of"],
                        "lep_flav_lst" : ["ee", "mm","em"],
                        "appl_lst"     : ["isSR_2lOS","isAR_2lOS"],
                    },
                },
                "2los_MR" : {
                    "atleast_1j" : {
                        "lep_chan_lst" : ["2los_sf"],
                        "lep_flav_lst" : ["ee","mm"],
                        "appl_lst"     : ["isMR_2lOS_C","isMR_2lOS_D"],
                    },

                    "atleast_0j": {
                        "lep_chan_lst" : ["2los_of"],
                        "lep_flav_lst" : ["em"],
                        "appl_lst"     : ["isMR_2lOS_C","isMR_2lOS_D"],
                    },
                },
            }

            # Variables we will loop over when filling hists
            varnames = {}
            varnames["photon_pt_eta"] = ak.ones_like(photon_pt)

            ########## Fill the histograms ##########

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
                        #Let's first make sure that we strip off that are not relevant to some lepton categories
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
                                #if the lepton category is not photon related, we need to edit data_syst_lst
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

                                        # Weights and eft coeffs
                                        weights_flat = weight[all_cuts_mask]
                                        eft_coeffs_cut = eft_coeffs[all_cuts_mask] if eft_coeffs is not None else None

                                        #plot
                                        weight_2DHist = weight
                                        if dense_axis_name == "photon_pt_eta":
                                            #This is needed because SparseHist doesn't handle EFT weights and so what we have to do is evaluate at SM point first and then modify the event weight
                                            if eft_coeffs is not None:
                                                wc_vals = np.zeros(len(max(self._samples[dataset]["WCnames"], self._wc_names_lst, key=len)))
                                                wgt_array_at_wc_vals = efth.calc_eft_weights(eft_coeffs,wc_vals)
                                            #else:
                                            #    wgt_array_at_wc_vals = np.ones(len(events))
                                                weight_2DHist = weight_2DHist * wgt_array_at_wc_vals
                                            plot_help.fill_2d_histogram(hout, dense_axis_name, "pt", "abseta", photon_pt, photon_abseta, ch_name, appl, histAxisName, wgt_fluct, weight_2DHist, None, all_cuts_mask, suffix="")  #absolutely make sure to set eft_coeffs to None
                                            plot_help.fill_2d_histogram(hout, dense_axis_name, "pt", "abseta", photon_pt, photon_abseta, ch_name, appl, histAxisName, wgt_fluct, np.square(weight_2DHist), None, all_cuts_mask, suffix="_sumw2") #absolutely make sure to set eft_coeffs to None


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
