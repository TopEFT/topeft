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
from topcoffea.modules.objects import *
from topcoffea.modules.selection import *
from topcoffea.modules.paths import topcoffea_path

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
        #self._accumulator = processor.dict_accumulator({
        self._dense_hists_dict = {
            "njets"   :
                hist.Hist(
                    hist.axis.StrCategory([], growth=True, name="process", label="process"),
                    axis.Regular(20, 0, 20, name="njets",   label="Jet multiplicity"),
                    storage="weight",
                    name="Counts"
                ),
            "nleps"   :
                hist.Hist(
                    hist.axis.StrCategory([], growth=True, name="process", label="process"),
                    axis.Regular(20, 0, 20, name="nleps",   label="Lep multiplicity"),
                    storage="weight",
                    name="Counts"
                ),
            "nbtagsl"   :
                hist.Hist(
                    hist.axis.StrCategory([], growth=True, name="process", label="process"),
                    axis.Regular(20, 0, 20, name="nbtagsl",   label="Loose btag multiplicity"),
                    storage="weight",
                    name="Counts"
                ),
            #"njets"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("njets",   "Jet multiplicity ", 20, 0, 20)),
            #"nleps"   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("nleps",   "Lep multiplicity ", 10, 0, 10)),
            #"nbtagsl" : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("nbtagsl", "Loose btag multiplicity ", 20, 0, 20)),
            #"met"     : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("met",     "MET (GeV)", 20, 0, 400)),
        }

        # Set the list of hists to fill
        if hist_lst is None:
            # If the hist list is none, assume we want to fill all hists
            #self._hist_lst = list(self._accumulator.keys())
            self._hist_lst = list(self._dense_hists_dict.keys())
        else:
            # Otherwise, just fill the specified subset of hists
            for hist_to_include in hist_lst:
                #if hist_to_include not in self._accumulator.keys():
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


    #@property
    #def accumulator(self):
    #    return self._accumulator

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
        ele_presl_mask = is_presel_wwz_ele(ele,tight=True)
        ele["topmva"] = get_topmva_score_ele(events, year)
        ele["is_tight_lep_for_wwz"] = ((ele.topmva > get_param("topmva_wp_t_e")) & ele_presl_mask)

        # Do the object selection for the WWZ muons
        mu_presl_mask = is_presel_wwz_mu(mu)
        mu["topmva"] = get_topmva_score_mu(events, year)
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
            cleanedJets = get_cleaned_collection(l_wwz_t,jets)

            # Selecting jets and cleaning them
            # NOTE: The jet id cut is commented for now in objects.py for the sync
            jetptname = "pt_nom" if hasattr(cleanedJets, "pt_nom") else "pt"
            cleanedJets["isGood"] = is_tight_jet_wwz(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, jetPtCut=20.)
            goodJets = cleanedJets[cleanedJets.isGood]

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
            #isBtagJetsLoose = (goodJets.btagDeepFlavB > 0.1355)
            #isBtagJetsLoose = (goodJets.btagDeepB > 0.1355)
            isNotBtagJetsLoose = np.invert(isBtagJetsLoose)
            nbtagsl = ak.num(goodJets[isBtagJetsLoose])

            isBtagJetsMedium = (goodJets.btagDeepFlavB > btagwpm)
            isNotBtagJetsMedium = np.invert(isBtagJetsMedium)
            nbtagsm = ak.num(goodJets[isBtagJetsMedium])


            #################### Add variables into event object so that they persist ####################

            # Put njets and l_fo_conept_sorted into events
            events["njets"] = njets
            events["l_wwz_t"] = l_wwz_t # FOR WWZ

            add4lMaskAndSFs_wwz(events, year, isData)


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


            ######### WWZ event selection stuff #########

            # Get some preliminary things we'll need
            attach_wwz_preselection_mask(events,l_wwz_t_padded[:,0:4])                                                  # Attach preselection sf and of flags to the events
            leps_from_z_candidate_ptordered, leps_not_z_candidate_ptordered = get_wwz_candidates(l_wwz_t_padded[:,0:4]) # Get a hold of the leptons from the Z and from the W
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

            ### The mt2 stuff ###
            # Construct misspart vector, as implimented in c++: https://github.com/sgnoohc/mt2example/blob/main/main.cc#L7
            nevents = len(np.zeros_like(met))
            misspart = ak.zip(
                {
                    "pt": met.pt,
                    #"eta": np.pi / 2,
                    "eta": 0,
                    "phi": met.phi,
                    "mass": np.full(nevents, 0),
                },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior,
            )
            # Do the boosts, as implimented in c++: https://github.com/sgnoohc/mt2example/blob/main/main.cc#L7
            w_lep0 = leps_not_z_candidate_ptordered[:,0:1]
            w_lep1 = leps_not_z_candidate_ptordered[:,1:2]
            print("w_lep0.pt",w_lep0.pt)
            print("w_lep1.pt",w_lep1.pt)
            print("l0.pt",l0.pt)
            print("l0.pt",l1.pt)
            print("l0.pt",l2.pt)
            print("l0.pt",l3.pt)
            rest_WW = w_lep0 + w_lep1 + misspart
            beta_from_miss_reverse = rest_WW.boostvec
            beta_from_miss = beta_from_miss_reverse.negative()
            w_lep0_boosted = w_lep0.boost(beta_from_miss)
            w_lep1_boosted = w_lep1.boost(beta_from_miss)
            misspart_boosted = misspart.boost(beta_from_miss)

            #print("\nRest W")
            #for i,beta in enumerate(rest_WW):
            #    if beta is not None: print(i,rest_WW.x[i],rest_WW.y[i],rest_WW.z[i])

            #print("\nbeta_from_miss")
            #for i,beta in enumerate(beta_from_miss):
            #    if beta is not None: print(i,beta.x,beta.y,beta.z)

            #print("\n\nHERE!!!!!!",len(w_lep0_boosted.mass))
            #for i,x in enumerate(w_lep0_boosted.mass):
            #    print("")
            #    print(i,"m1"  ,w_lep0_boosted.mass[i],w_lep0.mass[i])
            #    print(i,"x1"  ,w_lep0_boosted.px[i],w_lep0.px[i])
            #    print(i,"y1"  ,w_lep0_boosted.py[i],w_lep0.py[i])
            #    print(i,"pt1" ,w_lep0_boosted.pt[i],w_lep0.pt[i])
            #    print(i,"eta1",w_lep0_boosted.eta[i],w_lep0.eta[i])

            #    print(i,"m2"  ,w_lep1_boosted.mass[i],w_lep1.mass[i])
            #    print(i,"x2"  ,w_lep1_boosted.px[i],w_lep1.px[i])
            #    print(i,"y2"  ,w_lep1_boosted.py[i],w_lep1.py[i])
            #    print(i,"pt2" ,w_lep1_boosted.pt[i],w_lep1.pt[i])
            #    print(i,"eta2",w_lep1_boosted.eta[i],w_lep1.eta[i])

            #    print(i,"metx",misspart_boosted.px[i],misspart.px[i])
            #    print(i,"mety",misspart_boosted.py[i],misspart.py[i])

            #print("\n\nHERE!!!!!!",len(w_lep0_boosted.mass))
            #for i,x in enumerate(w_lep0_boosted.mass):
            #    print("\n",i)
            #    print("l1pt    ",w_lep0.pt[i])
            #    print("l2pt    ",w_lep1.pt[i])
            #    print("l1eta   ",w_lep0.eta[i])
            #    print("l2eta   ",w_lep1.eta[i])
            #    print("l1phi   ",w_lep0.phi[i])
            #    print("l2phi   ",w_lep1.phi[i])
            #    print("l1energy",w_lep0.energy[i])
            #    print("l2energy",w_lep1.energy[i])
            #    print("met     ",met.pt[i])
            #    print("metphi  ",met.phi[i])

            # Get the mt2 variable, use the mt2 package: https://pypi.org/project/mt2/
            mt2_var = mt2(
                w_lep0.mass, w_lep0_boosted.px, w_lep0_boosted.py,
                w_lep1.mass, w_lep1_boosted.px, w_lep1_boosted.py,
                #w_lep0_boosted.mass, w_lep0_boosted.px, w_lep0_boosted.py,
                #w_lep1_boosted.mass, w_lep1_boosted.px, w_lep1_boosted.py,
                #np.zeros_like(events['event']), w_lep0_boosted.px, w_lep0_boosted.py,
                #np.zeros_like(events['event']), w_lep1_boosted.px, w_lep1_boosted.py,
                misspart_boosted.px, misspart_boosted.py,
                np.zeros_like(events['event']), np.zeros_like(events['event']),
            )
            # Mask for mt2 cut
            mt2_mask = ak.fill_none(ak.any((mt2_var>25.0),axis=1),False)

            #for i,x in enumerate(mt2_var):
            #    print(i,"mt2",mt2_var[i])
            #print("this")
            #exit()


            ######### Store boolean masks with PackedSelection ##########

            #hout = self.accumulator.identity()
            dense_hists_dict = self._dense_hists_dict

            hout = {
                "njets" : {},
                "nleps" : {},
                "nbtagsl" : {},
            }

            selections = PackedSelection(dtype='uint64')

            # Lumi mask (for data)
            selections.add("is_good_lumi",lumi_mask)

            zeroj = (njets==0)

            # For WWZ selection
            #selections.add("4l_wwz_sf_A", (events.is4lWWZ & bmask_exactly0loose & pass_trg & events.wwz_presel_sf & w_candidates_mll_far_from_z & sf_A))
            #selections.add("4l_wwz_sf_B", (events.is4lWWZ & bmask_exactly0loose & pass_trg & events.wwz_presel_sf & w_candidates_mll_far_from_z & sf_B))
            #selections.add("4l_wwz_sf_C", (events.is4lWWZ & bmask_exactly0loose & pass_trg & events.wwz_presel_sf & w_candidates_mll_far_from_z & sf_C))
            #selections.add("4l_wwz_of_1", (events.is4lWWZ & bmask_exactly0loose & pass_trg & events.wwz_presel_of & of_1 & mt2_mask))
            #selections.add("4l_wwz_of_2", (events.is4lWWZ & bmask_exactly0loose & pass_trg & events.wwz_presel_of & of_2 & mt2_mask))
            #selections.add("4l_wwz_of_3", (events.is4lWWZ & bmask_exactly0loose & pass_trg & events.wwz_presel_of & of_3 & mt2_mask))
            #selections.add("4l_wwz_of_4", (events.is4lWWZ & bmask_exactly0loose & pass_trg & events.wwz_presel_of & of_4))
            selections.add("4l_wwz_sf_A", (events.is4lWWZ & bmask_exactly0loose & events.wwz_presel_sf & w_candidates_mll_far_from_z & sf_A))
            selections.add("4l_wwz_sf_B", (events.is4lWWZ & bmask_exactly0loose & events.wwz_presel_sf & w_candidates_mll_far_from_z & sf_B))
            selections.add("4l_wwz_sf_C", (events.is4lWWZ & bmask_exactly0loose & events.wwz_presel_sf & w_candidates_mll_far_from_z & sf_C))
            selections.add("4l_wwz_of_1", (events.is4lWWZ & bmask_exactly0loose & events.wwz_presel_of & of_1 & mt2_mask))
            selections.add("4l_wwz_of_2", (events.is4lWWZ & bmask_exactly0loose & events.wwz_presel_of & of_2 & mt2_mask))
            selections.add("4l_wwz_of_3", (events.is4lWWZ & bmask_exactly0loose & events.wwz_presel_of & of_3 & mt2_mask))
            selections.add("4l_wwz_of_4", (events.is4lWWZ & bmask_exactly0loose & events.wwz_presel_of & of_4))

            selections.add("all_events", (events.is4lWWZ | (~events.is4lWWZ))) # All events.. this logic is a bit roundabout to just get an array of True
            selections.add("4l_presel", (events.is4lWWZ)) # This matches the VVV looper selection (object selection and event selection)

            sr_cat_dict = {
                "lep_chan_lst" : ["4l_wwz_sf_A","4l_wwz_sf_B","4l_wwz_sf_C","4l_wwz_of_1","4l_wwz_of_2","4l_wwz_of_3","4l_wwz_of_4","all_events","4l_presel"],
            }


            ######### Fill histos #########

            dense_axes_dict = {
                #"met" : met.pt,
                "nleps" : nleps,
                "njets" : njets,
                "nbtagsl" : nbtagsl,
            }

            weights = weights_obj_base.partial_weight(include=["norm"])
            weights = events.nom

            # Loop over the hists we want to fill
            for dense_axis_name, dense_axis_vals in dense_axes_dict.items():
                #print("\ndense_axis_name,vals",dense_axis_name)
                #print("dense_axis_name,vals",dense_axis_vals)

                for sr_cat in sr_cat_dict["lep_chan_lst"]:

                    hout[dense_axis_name][sr_cat] = {}
                    hout[dense_axis_name][sr_cat][histAxisName] = copy.deepcopy(dense_hists_dict[dense_axis_name])

                    # Make the cuts mask
                    cuts_lst = [sr_cat]
                    all_cuts_mask = selections.all(*cuts_lst)

                    # Fill the histos
                    axes_fill_info_dict = {
                        dense_axis_name : dense_axis_vals[all_cuts_mask],
                        "weight"        : weights[all_cuts_mask],
                        "process"       : histAxisName,
                        #"systematic"    : "nominal",
                    }
                    hout[dense_axis_name][sr_cat][histAxisName].fill(**axes_fill_info_dict)

        return hout

    def postprocess(self, accumulator):
        return accumulator

