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

from topcoffea.modules.GetValuesFromJsons import get_param, get_lumi
from topcoffea.modules.objects import *
from topcoffea.modules.corrections import GetBTagSF, ApplyJetCorrections, GetBtagEff, AttachMuonSF, AttachElectronSF, AttachPerLeptonFR, GetPUSF, ApplyRochesterCorrections, ApplyJetSystematics, AttachPSWeights, AttachScaleWeights, GetTriggerSF
from topcoffea.modules.selection import *
from topcoffea.modules.HistEFT import HistEFT
from topcoffea.modules.paths import topcoffea_path
import topcoffea.modules.eft_helper as efth

from coffea.nanoevents.methods import vector
from mt2 import mt2

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

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, ecut_threshold=None, do_errors=False, do_systematics=False, split_by_lepton_flavor=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        # Create the histograms
        self._accumulator = processor.dict_accumulator({
            "invmass" : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 20, 0, 1000)),
            "ptbl"    : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("ptbl",    "$p_{T}^{b\mathrm{-}jet+\ell_{min(dR)}}$ (GeV) ", 40, 0, 1000)),
            "ptz"     : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("ptz",     "$p_{T}$ Z (GeV)", 12, 0, 600)),
            "njets"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("njets",   "Jet multiplicity ", 10, 0, 10)),
            "nbtagsl" : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("nbtagsl", "Loose btag multiplicity ", 5, 0, 5)),
            "l0pt"    : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("l0pt",    "Leading lep $p_{T}$ (GeV)", 10, 0, 500)),
            "l1pt"    : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("l1pt",    "Subleading lep $p_{T}$ (GeV)", 10, 0, 100)),
            "l1eta"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("l1eta",   "Subleading $\eta$", 20, -2.5, 2.5)),
            "j0pt"    : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("j0pt",    "Leading jet  $p_{T}$ (GeV)", 10, 0, 500)),
            "b0pt"    : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("b0pt",    "Leading b jet  $p_{T}$ (GeV)", 10, 0, 500)),
            "l0eta"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("l0eta",   "Leading lep $\eta$", 20, -2.5, 2.5)),
            "j0eta"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("j0eta",   "Leading jet  $\eta$", 30, -3.0, 3.0)),
            "ht"      : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("ht",      "H$_{T}$ (GeV)", 20, 0, 1000)),
            "met"     : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("met",     "MET (GeV)", 20, 0, 400)),
            "ljptsum" : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("ljptsum", "S$_{T}$ (GeV)", 11, 0, 1100)),
            "o0pt"    : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("o0pt",    "Leading l or b jet $p_{T}$ (GeV)", 10, 0, 500)),
            "bl0pt"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("bl0pt",   "Leading (b+l) $p_{T}$ (GeV)", 10, 0, 500)),
            "lj0pt"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("lj0pt",   "Leading pt of pair from l+j collection (GeV)", 12, 0, 600)),
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

        isData             = self._samples[dataset]["isData"]
        histAxisName       = self._samples[dataset]["histAxisName"]
        year               = self._samples[dataset]["year"]
        xsec               = self._samples[dataset]["xsec"]
        sow                = self._samples[dataset]["nSumOfWeights"]

        # Get up down weights from input dict
        if (self._do_systematics and not isData):
            if histAxisName in get_param("lo_xsec_samples"):
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
        elif histAxisName in get_param("conv_samples"):
            sampleType = "conversions"
        elif histAxisName in get_param("prompt_and_conv_samples"):
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

        ele["idEmu"] = ttH_idEmu_cuts_E3(ele.hoe, ele.eta, ele.deltaEtaSC, ele.eInvMinusPInv, ele.sieie)
        ele["conept"] = coneptElec(ele.pt, ele.mvaTTHUL, ele.jetRelIso)
        mu["conept"] = coneptMuon(mu.pt, mu.mvaTTHUL, mu.jetRelIso, mu.mediumId)
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


        ################### Electron selection ####################

        # Do the object selection for the WWZ leptons
        ele_presl_mask = is_presel_wwz_ele(ele,tight=True)
        ele["topmva"] = get_topmva_score_ele(events, year)
        ele["is_tight_lep_for_wwz"] = ((ele.topmva > get_param("topmva_wp_t_e")) & ele_presl_mask)

        print("after")
        exit()

        ele["isPres"] = isPresElec(ele.pt, ele.eta, ele.dxy, ele.dz, ele.miniPFRelIso_all, ele.sip3d, getattr(ele,"mvaFall17V2noIso_WPL"))
        ele["isLooseE"] = isLooseElec(ele.miniPFRelIso_all,ele.sip3d,ele.lostHits)
        ele["isFO"] = isFOElec(ele.pt, ele.conept, ele.btagDeepFlavB, ele.idEmu, ele.convVeto, ele.lostHits, ele.mvaTTHUL, ele.jetRelIso, ele.mvaFall17V2noIso_WP90, year)
        ele["isTightLep"] = tightSelElec(ele.isFO, ele.mvaTTHUL)

        ################### Muon selection ####################

        # Do the object selection for the WWZ leptons
        mu_presl_mask = is_presel_wwz_mu(mu)
        mu["topmva"] = get_topmva_score_mu(events, year)
        mu["is_tight_lep_for_wwz"] = ((mu.topmva > get_param("topmva_wp_t_m")) & mu_presl_mask)

        mu["pt"] = ApplyRochesterCorrections(year, mu, isData) # Need to apply corrections before doing muon selection
        mu["isPres"] = isPresMuon(mu.dxy, mu.dz, mu.sip3d, mu.eta, mu.pt, mu.miniPFRelIso_all)
        mu["isLooseM"] = isLooseMuon(mu.miniPFRelIso_all,mu.sip3d,mu.looseId)
        mu["isFO"] = isFOMuon(mu.pt, mu.conept, mu.btagDeepFlavB, mu.mvaTTHUL, mu.jetRelIso, year)
        mu["isTightLep"]= tightSelMuon(mu.isFO, mu.mediumId, mu.mvaTTHUL)

        ################### Loose selection ####################

        m_loose = mu[mu.isPres & mu.isLooseM]
        e_loose = ele[ele.isPres & ele.isLooseE]
        l_loose = ak.with_name(ak.concatenate([e_loose, m_loose], axis=1), 'PtEtaPhiMCandidate')

        ################### Tau selection ####################

        tau["isPres"]  = isPresTau(tau.pt, tau.eta, tau.dxy, tau.dz, tau.idDeepTau2017v2p1VSjet, minpt=20)
        tau["isClean"] = isClean(tau, l_loose, drmin=0.3)
        tau["isGood"]  =  tau["isClean"] & tau["isPres"]
        tau = tau[tau.isGood] # use these to clean jets
        tau["isTight"] = isTightTau(tau.idDeepTau2017v2p1VSjet) # use these to veto

        # Compute pair invariant masses, for all flavors all signes
        llpairs = ak.combinations(l_loose, 2, fields=["l0","l1"])
        os_pairs_mask = (llpairs.l0.pdgId*llpairs.l1.pdgId < 0)
        ll_mass_pairs = (llpairs.l0+llpairs.l1).mass
        ll_mass_pairs_os = ll_mass_pairs[os_pairs_mask]
        events["min_mll_afos"] = ak.min(ll_mass_pairs_os,axis=-1) # For WWZ
        events["minMllAFAS"] = ak.min( (llpairs.l0+llpairs.l1).mass, axis=-1) # From TOP-22-006

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

        l_fo_conept_sorted_padded = ak.pad_none(l_fo_conept_sorted, 4) # Temp till we impliment new MVA
        l0 = l_fo_conept_sorted_padded[:,0]
        l1 = l_fo_conept_sorted_padded[:,1]
        l2 = l_fo_conept_sorted_padded[:,2]
        l3 = l_fo_conept_sorted_padded[:,3]

        ######### Systematics ###########


        # These weights can go outside of the outside sys loop since they do not depend on pt of mu or jets
        # We only calculate these values if not isData
        # Note: add() will generally modify up/down weights, so if these are needed for any reason after this point, we should instead pass copies to add()
        # Note: Here we will to the weights object the SFs that do not depend on any of the forthcoming loops
        weights_obj_base = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
        if not isData:
            genw = events["genWeight"]

            # Normalize by (xsec/sow)*genw where genw is 1 for EFT samples
            # Note that for theory systs, will need to multiply by sow/sow_wgtUP to get (xsec/sow_wgtUp)*genw and same for Down
            lumi = 1000.0*get_lumi(year)
            weights_obj_base.add("norm",(xsec/sow)*genw*lumi)

        syst_var_list = ['nominal']

        # Loop over the list of systematic variations we've constructed
        met_raw=met
        for syst_var in syst_var_list:
            # Make a copy of the base weights object, so that each time through the loop we do not double count systs
            # In this loop over systs that impact kinematics, we will add to the weights objects the SFs that depend on the object kinematics

            #################### Jets ####################

            # Jet cleaning, before any jet selection
            #vetos_tocleanjets = ak.with_name( ak.concatenate([tau, l_fo], axis=1), "PtEtaPhiMCandidate")
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
            elif year == "2018":
                btagwpl = get_param("btag_wp_loose_UL18")
            elif year=="2016":
                btagwpl = get_param("btag_wp_loose_UL16")
            elif year=="2016APV":
                btagwpl = get_param("btag_wp_loose_UL16APV")
            else:
                raise ValueError(f"Error: Unknown year \"{year}\".")
            isBtagJetsLoose = (goodJets.btagDeepFlavB > btagwpl)
            isNotBtagJetsLoose = np.invert(isBtagJetsLoose)
            nbtagsl = ak.num(goodJets[isBtagJetsLoose])

            # Medium DeepJet WP
            if year == "2017":
                btagwpm = get_param("btag_wp_medium_UL17")
            elif year == "2018":
                btagwpm = get_param("btag_wp_medium_UL18")
            elif year=="2016":
                btagwpm = get_param("btag_wp_medium_UL16")
            elif year=="2016APV":
                btagwpm = get_param("btag_wp_medium_UL16APV")
            else:
                raise ValueError(f"Error: Unknown year \"{year}\".")
            isBtagJetsMedium = (goodJets.btagDeepFlavB > btagwpm)
            isNotBtagJetsMedium = np.invert(isBtagJetsMedium)
            nbtagsm = ak.num(goodJets[isBtagJetsMedium])


            #################### Add variables into event object so that they persist ####################

            # Put njets and l_fo_conept_sorted into events
            events["njets"] = njets
            events["l_fo_conept_sorted"] = l_fo_conept_sorted

            add4lMaskAndSFs_wwz(events, year, isData)
            add4lMaskAndSFs(events, year, isData)


            ######### Masks we need for the selection ##########

            # Pass trigger mask
            pass_trg = trgPassNoOverlap(events,isData,dataset,str(year))

            # b jet masks
            bmask_atleast1med_atleast2loose = ((nbtagsm>=1)&(nbtagsl>=2)) # Used for 2lss and 4l
            bmask_exactly0loose = (nbtagsl==0) # Used for 4l WWZ SR
            bmask_exactly0med = (nbtagsm==0) # Used for 3l CR and 2los Z CR
            bmask_exactly1med = (nbtagsm==1) # Used for 3l SR and 2lss CR
            bmask_exactly2med = (nbtagsm==2) # Used for CRtt
            bmask_atleast2med = (nbtagsm>=2) # Used for 3l SR
            bmask_atmost2med  = (nbtagsm< 3) # Used to make 2lss mutually exclusive from tttt enriched
            bmask_atleast3med = (nbtagsm>=3) # Used for tttt enriched

            ######### WWZ stuff #########

            # Get some preliminary things we'll need
            attach_wwz_preselection_mask(events,l_fo_conept_sorted_padded[:,0:4])                                                  # Attach preselection sf and of flags to the events
            leps_from_z_candidate_ptordered, leps_not_z_candidate_ptordered = get_wwz_candidates(l_fo_conept_sorted_padded[:,0:4]) # Get a hold of the leptons from the Z and from the W
            w_candidates_mll = (leps_not_z_candidate_ptordered[:,0:1]+leps_not_z_candidate_ptordered[:,1:2]).mass                  # Will need to know mass of the leps from the W

            # Make masks for the SF regions
            w_candidates_mll_far_from_z = ak.fill_none(ak.any((abs(w_candidates_mll - 91.2) > 10.0),axis=1),False) # Will enforce this for SF in the PackedSelection
            ptl4 = (l0+l1+l2+l3).pt
            sf_A = (met.pt > 120.0)
            sf_B = ((met.pt > 70.0) & (met.pt < 120.0) & (ptl4 > 70.0))
            sf_C = ((met.pt > 70.0) & (met.pt < 120.0) & (ptl4 > 40.0) & (ptl4 < 70.0))

            # Make masks for the OF regions
            of_1 = ak.fill_none(ak.any((w_candidates_mll > 0.0) & (w_candidates_mll < 40.0),axis=1),False)
            of_2 = ak.fill_none(ak.any((w_candidates_mll > 40.0) & (w_candidates_mll < 60.0),axis=1),False)
            of_3 = ak.fill_none(ak.any((w_candidates_mll > 60.0) & (w_candidates_mll < 100.0),axis=1),False)
            of_4 = ak.fill_none(ak.any((w_candidates_mll > 100.0),axis=1),False)

            ### The mt2 stuff ###
            # Construct misspart vector, as implimented in c++: https://github.com/sgnoohc/mt2example/blob/main/main.cc#L7
            nevents = len(np.zeros_like(met))
            misspart = ak.zip(
                {
                    "pt": met.pt,
                    "eta": np.pi / 2,
                    "phi": met.phi,
                    "mass": np.full(nevents, 0),
                },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior,
            )
            # Do the boosts, as implimented in c++: https://github.com/sgnoohc/mt2example/blob/main/main.cc#L7
            w_lep0 = leps_not_z_candidate_ptordered[:,0:1]
            w_lep1 = leps_not_z_candidate_ptordered[:,1:2]
            rest_WW = w_lep0 + w_lep1 + misspart
            beta_from_miss_reverse = rest_WW.boostvec
            beta_from_miss = beta_from_miss_reverse.negative()
            w_lep0_boosted = w_lep0.boost(beta_from_miss)
            w_lep1_boosted = w_lep1.boost(beta_from_miss)
            misspart_boosted = misspart.boost(beta_from_miss)
            # Get the mt2 variable, use the mt2 package: https://pypi.org/project/mt2/
            mt2_var = mt2(
                w_lep0_boosted.mass, w_lep0_boosted.px, w_lep0_boosted.py,
                w_lep1_boosted.mass, w_lep1_boosted.px, w_lep1_boosted.py,
                misspart_boosted.px, misspart_boosted.py,
                np.zeros_like(events['event']), np.zeros_like(events['event']),
            )
            # Mask for mt2 cut
            mt2_mask = ak.fill_none(ak.any((mt2_var>25.0),axis=1),False)



            ######### Store boolean masks with PackedSelection ##########

            hout = self.accumulator.identity()

            selections = PackedSelection(dtype='uint64')

            # Lumi mask (for data)
            selections.add("is_good_lumi",lumi_mask)

            # 4l selection
            selections.add("4l", (events.is4lWWZ & bmask_exactly0loose & pass_trg))
            selections.add("isSR_4l",  events.is4l_SR)

            # For WWZ selection
            selections.add("4l_wwz_sf_A", (events.is4lWWZ & events.is4l_SR & bmask_exactly0med & pass_trg & events.wwz_presel_sf & w_candidates_mll_far_from_z & sf_A))
            selections.add("4l_wwz_sf_B", (events.is4lWWZ & events.is4l_SR & bmask_exactly0med & pass_trg & events.wwz_presel_sf & w_candidates_mll_far_from_z & sf_B))
            selections.add("4l_wwz_sf_C", (events.is4lWWZ & events.is4l_SR & bmask_exactly0med & pass_trg & events.wwz_presel_sf & w_candidates_mll_far_from_z & sf_C))
            selections.add("4l_wwz_of_1", (events.is4lWWZ & events.is4l_SR & bmask_exactly0med & pass_trg & events.wwz_presel_of & of_1 & mt2_mask))
            selections.add("4l_wwz_of_2", (events.is4lWWZ & events.is4l_SR & bmask_exactly0med & pass_trg & events.wwz_presel_of & of_2 & mt2_mask))
            selections.add("4l_wwz_of_3", (events.is4lWWZ & events.is4l_SR & bmask_exactly0med & pass_trg & events.wwz_presel_of & of_3 & mt2_mask))
            selections.add("4l_wwz_of_4", (events.is4lWWZ & events.is4l_SR & bmask_exactly0med & pass_trg & events.wwz_presel_of & of_4))

            # Topcoffea 4l SR
            selections.add("4l_tc", (events.is4l & events.is4l_SR & (njets>=2) & bmask_atleast1med_atleast2loose & pass_trg))

            sr_cat_dict = {
                #"lep_chan_lst" : ["4l_wwz_sf_A","4l_wwz_sf_B","4l_wwz_sf_C","4l_wwz_of_1","4l_wwz_of_2","4l_wwz_of_3","4l_wwz_of_4"],
                "lep_chan_lst" : ["4l_tc","4l_wwz_sf_A","4l_wwz_sf_B","4l_wwz_sf_C","4l_wwz_of_1","4l_wwz_of_2","4l_wwz_of_3","4l_wwz_of_4"],
            }


            ######### Fill histos #########

            dense_axes_dict = {
                "l0pt" : l0.pt
            }

            weights = weights_obj_base.partial_weight(include=["norm"])

            print("j0.pt",j0.pt)
            print("this")


            # Loop over the hists we want to fill
            for dense_axis_name, dense_axis_vals in dense_axes_dict.items():
                print("\ndense_axis_name,vals",dense_axis_name)
                print("dense_axis_name,vals",dense_axis_vals)

                for sr_cat in sr_cat_dict["lep_chan_lst"]:

                    # Make the cuts mask
                    cuts_lst = [sr_cat]
                    all_cuts_mask = selections.all(*cuts_lst)

                    # Fill the histos
                    axes_fill_info_dict = {
                        dense_axis_name : dense_axis_vals[all_cuts_mask],
                        "weight"        : weights[all_cuts_mask],
                        "channel"       : sr_cat,
                        "sample"        : histAxisName,
                        "systematic"    : "nominal",
                    }

                    hout[dense_axis_name].fill(**axes_fill_info_dict)

        return hout

    def postprocess(self, accumulator):
        return accumulator

