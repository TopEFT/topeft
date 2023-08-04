#!/usr/bin/env python
import copy
import coffea
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import processor
import hist
from hist import axis
from coffea.analysis_tools import PackedSelection
from coffea.lumi_tools import LumiMask

from topcoffea.modules.GetValuesFromJsons import get_param, get_lumi
from topcoffea.modules.paths import topcoffea_path
import topcoffea.modules.objects as objbase
import topcoffea.modules.objects_wwz as objwwz
import topcoffea.modules.selection_wwz as selwwz
import topcoffea.modules.selection as selbase


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, ecut_threshold=None, do_errors=False, do_systematics=False, split_by_lepton_flavor=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        # Create the histograms
        self._dense_hists_dict = {
            "njets"   :
                hist.Hist(
                    hist.axis.StrCategory([], growth=True, name="process", label="process"),
                    hist.axis.StrCategory([], growth=True, name="category", label="category"),
                    axis.Regular(20, 0, 20, name="njets",   label="Jet multiplicity"),
                    storage="weight", # Keeps track of sumw2
                    name="Counts",
                ),
            "nleps"   :
                hist.Hist(
                    hist.axis.StrCategory([], growth=True, name="process", label="process"),
                    hist.axis.StrCategory([], growth=True, name="category", label="category"),
                    axis.Regular(20, 0, 20, name="nleps",   label="Lep multiplicity"),
                    storage="weight", # Keeps track of sumw2
                    name="Counts",
                ),
            "nbtagsl"   :
                hist.Hist(
                    hist.axis.StrCategory([], growth=True, name="process", label="process"),
                    hist.axis.StrCategory([], growth=True, name="category", label="category"),
                    axis.Regular(20, 0, 20, name="nbtagsl",   label="Loose btag multiplicity"),
                    storage="weight", # Keeps track of sumw2
                    name="Counts",
                ),
        }

        # Set the list of hists to fill
        if hist_lst is None:
            # If the hist list is none, assume we want to fill all hists
            self._hist_lst = list(self._dense_hists_dict.keys())
        else:
            # Otherwise, just fill the specified subset of hists
            for hist_to_include in hist_lst:
                if hist_to_include not in self._dense_hists_dict.keys():
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


        ################### Lepton selection ####################

        # Do the object selection for the WWZ eleectrons
        ele_presl_mask = objwwz.is_presel_wwz_ele(ele,tight=True)
        ele["topmva"] = objwwz.get_topmva_score_ele(events, year)
        ele["is_tight_lep_for_wwz"] = ((ele.topmva > get_param("topmva_wp_t_e")) & ele_presl_mask)

        # Do the object selection for the WWZ muons
        mu_presl_mask = objwwz.is_presel_wwz_mu(mu)
        mu["topmva"] = objwwz.get_topmva_score_mu(events, year)
        mu["is_tight_lep_for_wwz"] = ((mu.topmva > get_param("topmva_wp_t_m")) & mu_presl_mask)

        # Get tight leptons for WWZ selection
        ele_wwz_t = ele[ele.is_tight_lep_for_wwz]
        mu_wwz_t = mu[mu.is_tight_lep_for_wwz]
        l_wwz_t = ak.with_name(ak.concatenate([ele_wwz_t,mu_wwz_t],axis=1),'PtEtaPhiMCandidate')
        l_wwz_t = l_wwz_t[ak.argsort(l_wwz_t.pt, axis=-1,ascending=False)] # Sort by pt

        # For WWZ: Compute pair invariant masses
        llpairs_wwz = ak.combinations(l_wwz_t, 2, fields=["l0","l1"])
        os_pairs_mask = (llpairs_wwz.l0.pdgId*llpairs_wwz.l1.pdgId < 0)
        ll_mass_pairs = (llpairs_wwz.l0+llpairs_wwz.l1).mass
        ll_mass_pairs_os = ll_mass_pairs[os_pairs_mask]
        events["min_mll_afos"] = ak.min(ll_mass_pairs_os,axis=-1) # For WWZ

        # For WWZ
        l_wwz_t_padded = ak.pad_none(l_wwz_t, 4)
        l0 = l_wwz_t_padded[:,0]
        l1 = l_wwz_t_padded[:,1]
        l2 = l_wwz_t_padded[:,2]
        l3 = l_wwz_t_padded[:,3]

        nleps = ak.num(l_wwz_t)

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

        # We do not have systematics yet
        syst_var_list = ['nominal']

        # Loop over the list of systematic variations we've constructed
        for syst_var in syst_var_list:
            # Make a copy of the base weights object, so that each time through the loop we do not double count systs
            # In this loop over systs that impact kinematics, we will add to the weights objects the SFs that depend on the object kinematics

            #################### Jets ####################

            # Jet cleaning, before any jet selection
            ##vetos_tocleanjets = ak.with_name( ak.concatenate([tau, l_fo], axis=1), "PtEtaPhiMCandidate")
            #vetos_tocleanjets = ak.with_name( l_wwz_t, "PtEtaPhiMCandidate")
            #tmp = ak.cartesian([ak.local_index(jets.pt), vetos_tocleanjets.jetIdx], nested=True)
            #cleanedJets = jets[~ak.any(tmp.slot0 == tmp.slot1, axis=-1)] # this line should go before *any selection*, otherwise lep.jetIdx is not aligned with the jet index

            # Clean with dr for now
            cleanedJets = objwwz.get_cleaned_collection(l_wwz_t,jets)

            # Selecting jets and cleaning them
            # NOTE: The jet id cut is commented for now in objects.py for the sync
            jetptname = "pt_nom" if hasattr(cleanedJets, "pt_nom") else "pt"
            cleanedJets["is_good"] = objbase.isTightJet(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, jetPtCut=20.)
            goodJets = cleanedJets[cleanedJets.is_good]

            # Count jets
            njets = ak.num(goodJets)
            ht = ak.sum(goodJets.pt,axis=-1)
            j0 = goodJets[ak.argmax(goodJets.pt,axis=-1,keepdims=True)]


            # Loose DeepJet WP
            #btagger = "btag" # For deep flavor WPs
            btagger = "btagcsv" # For deep CSV WPs
            if year == "2017":
                btagwpl = get_param(f"{btagger}_wp_loose_UL17")
                btagwpm = get_param(f"{btagger}_wp_medium_UL17")
            elif year == "2018":
                btagwpl = get_param(f"{btagger}_wp_loose_UL18")
                btagwpm = get_param(f"{btagger}_wp_medium_UL18")
            elif year=="2016":
                btagwpl = get_param(f"{btagger}_wp_loose_UL16")
                btagwpm = get_param(f"{btagger}_wp_medium_UL16")
            elif year=="2016APV":
                btagwpl = get_param(f"{btagger}_wp_loose_UL16APV")
                btagwpm = get_param(f"{btagger}_wp_medium_UL16APV")
            else:
                raise ValueError(f"Error: Unknown year \"{year}\".")

            if btagger == "btag":
                isBtagJetsLoose = (goodJets.btagDeepFlavB > btagwpl)
            if btagger == "btagcsv":
                isBtagJetsLoose = (goodJets.btagDeepB > btagwpl)
            isNotBtagJetsLoose = np.invert(isBtagJetsLoose)
            nbtagsl = ak.num(goodJets[isBtagJetsLoose])

            isBtagJetsMedium = (goodJets.btagDeepFlavB > btagwpm)
            isNotBtagJetsMedium = np.invert(isBtagJetsMedium)
            nbtagsm = ak.num(goodJets[isBtagJetsMedium])


            #################### Add variables into event object so that they persist ####################

            # Put njets and l_fo_conept_sorted into events
            events["njets"] = njets
            events["l_wwz_t"] = l_wwz_t

            selwwz.add4lmask_wwz(events, year, isData)


            ######### Masks we need for the selection ##########

            # Pass trigger mask
            pass_trg = selbase.trgPassNoOverlap(events,isData,dataset,str(year),dataset_dict=selwwz.dataset_dict,exclude_dict=selwwz.exclude_dict)
            pass_trg = (pass_trg & selwwz.trg_matching(events,year))

            # b jet masks
            bmask_atleast1med_atleast2loose = ((nbtagsm>=1)&(nbtagsl>=2)) # Used for 2lss and 4l
            bmask_exactly0loose = (nbtagsl==0) # Used for 4l WWZ SR
            bmask_exactly0med = (nbtagsm==0) # Used for 3l CR and 2los Z CR
            bmask_exactly1med = (nbtagsm==1) # Used for 3l SR and 2lss CR
            bmask_exactly2med = (nbtagsm==2) # Used for CRtt
            bmask_atleast2med = (nbtagsm>=2) # Used for 3l SR
            bmask_atmost2med  = (nbtagsm< 3) # Used to make 2lss mutually exclusive from tttt enriched
            bmask_atleast3med = (nbtagsm>=3) # Used for tttt enriched


            ######### WWZ event selection stuff #########

            # Get some preliminary things we'll need
            selwwz.attach_wwz_preselection_mask(events,l_wwz_t_padded[:,0:4])                                              # Attach preselection sf and of flags to the events
            leps_from_z_candidate_ptordered, leps_not_z_candidate_ptordered = selwwz.get_wwz_candidates(l_wwz_t_padded[:,0:4]) # Get a hold of the leptons from the Z and from the W
            w_candidates_mll = (leps_not_z_candidate_ptordered[:,0:1]+leps_not_z_candidate_ptordered[:,1:2]).mass       # Will need to know mass of the leps from the W

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

            # Mask for mt2 cut
            w_lep0 = leps_not_z_candidate_ptordered[:,0:1]
            w_lep1 = leps_not_z_candidate_ptordered[:,1:2]
            mt2_val = selwwz.get_mt2(w_lep0,w_lep1,met)
            mt2_mask = ak.fill_none(ak.any((mt2_val>25.0),axis=1),False)



            ######### Store boolean masks with PackedSelection ##########


            selections = PackedSelection(dtype='uint64')

            # Lumi mask (for data)
            selections.add("is_good_lumi",lumi_mask)

            zeroj = (njets==0)

            # For WWZ selection
            selections.add("4l_wwz_sf_A", (pass_trg & events.is4lWWZ & bmask_exactly0loose & events.wwz_presel_sf & w_candidates_mll_far_from_z & sf_A))
            selections.add("4l_wwz_sf_B", (pass_trg & events.is4lWWZ & bmask_exactly0loose & events.wwz_presel_sf & w_candidates_mll_far_from_z & sf_B))
            selections.add("4l_wwz_sf_C", (pass_trg & events.is4lWWZ & bmask_exactly0loose & events.wwz_presel_sf & w_candidates_mll_far_from_z & sf_C))
            selections.add("4l_wwz_of_1", (pass_trg & events.is4lWWZ & bmask_exactly0loose & events.wwz_presel_of & of_1 & mt2_mask))
            selections.add("4l_wwz_of_2", (pass_trg & events.is4lWWZ & bmask_exactly0loose & events.wwz_presel_of & of_2 & mt2_mask))
            selections.add("4l_wwz_of_3", (pass_trg & events.is4lWWZ & bmask_exactly0loose & events.wwz_presel_of & of_3 & mt2_mask))
            selections.add("4l_wwz_of_4", (pass_trg & events.is4lWWZ & bmask_exactly0loose & events.wwz_presel_of & of_4))

            selections.add("all_events", (events.is4lWWZ | (~events.is4lWWZ))) # All events.. this logic is a bit roundabout to just get an array of True
            selections.add("4l_presel", (events.is4lWWZ)) # This matches the VVV looper selection (object selection and event selection)

            sr_cat_dict = {
                "lep_chan_lst" : ["4l_wwz_sf_A","4l_wwz_sf_B","4l_wwz_sf_C","4l_wwz_of_1","4l_wwz_of_2","4l_wwz_of_3","4l_wwz_of_4","all_events","4l_presel"],
            }



            ######### Fill histos #########

            hout = self._dense_hists_dict

            dense_variables_dict = {
                #"met" : met.pt,
                "nleps" : nleps,
                "njets" : njets,
                "nbtagsl" : nbtagsl,
            }

            weights = weights_obj_base.partial_weight(include=["norm"])
            weights = events.nom

            # Loop over the hists we want to fill
            for dense_axis_name, dense_axis_vals in dense_variables_dict.items():
                #print("\ndense_axis_name,vals",dense_axis_name)
                #print("dense_axis_name,vals",dense_axis_vals)

                for sr_cat in sr_cat_dict["lep_chan_lst"]:

                    # Make the cuts mask
                    cuts_lst = [sr_cat]
                    all_cuts_mask = selections.all(*cuts_lst)

                    # Fill the histos
                    axes_fill_info_dict = {
                        dense_axis_name : dense_axis_vals[all_cuts_mask],
                        "weight"        : weights[all_cuts_mask],
                        "process"       : histAxisName,
                        "category"      : sr_cat,
                        #"systematic"    : "nominal",
                    }
                    hout[dense_axis_name].fill(**axes_fill_info_dict)

        return hout

    def postprocess(self, accumulator):
        return accumulator

