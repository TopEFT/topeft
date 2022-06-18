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
            "invmass" : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 20, 0, 1000)),
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

        # An array of lenght events that is just 1 for each event
        # Probably there's a better way to do this, but we use this method elsewhere so I guess why not..
        events.nom = ak.ones_like(events.MET.pt)

        e["idEmu"] = ttH_idEmu_cuts_E3(e.hoe, e.eta, e.deltaEtaSC, e.eInvMinusPInv, e.sieie)
        e["conept"] = coneptElec(e.pt, e.mvaTTHUL, e.jetRelIso)
        mu["conept"] = coneptMuon(mu.pt, mu.mvaTTHUL, mu.jetRelIso, mu.mediumId)
        e["btagDeepFlavB"] = ak.fill_none(e.matched_jet.btagDeepFlavB, -99)
        mu["btagDeepFlavB"] = ak.fill_none(mu.matched_jet.btagDeepFlavB, -99)
        if not isData:
            e["gen_pdgId"] = ak.fill_none(e.matched_gen.pdgId, 0)
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

        # Initialize the out object
        hout = self.accumulator.identity()

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

        # These weights can go outside of the outside sys loop since they do not depend on pt of mu or jets
        # We only calculate these values if not isData
        # Note: add() will generally modify up/down weights, so if these are needed for any reason after this point, we should instead pass copies to add()
        # Note: Here we will to the weights object the SFs that do not depend on any of the forthcoming loops
        weights_obj_base = coffea.analysis_tools.Weights(len(events),storeIndividual=True)

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


        ######### Event weights that do depend on the lep cat ###########

        # Loop over categories and fill the dict
        weights_dict = {}
        for ch_name in ["2l", "2l_4t", "3l", "4l", "2l_CR", "2l_CRflip", "3l_CR", "2los_CRtt", "2los_CRZ"]:

            # For both data and MC
            weights_dict[ch_name] = copy.deepcopy(weights_obj_base)
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


        ######### Masks we need for the selection ##########

        # Get mask for events that have two sf os leps close to z peak
        sfosz_3l_mask = get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:3],pt_window=10.0)
        sfosz_2l_mask = get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:2],pt_window=10.0)
        sfasz_2l_mask = get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:2],pt_window=30.0,flavor="as") # Any sign (do not enforce ss or os here)

        # Pass trigger mask
        pass_trg = trgPassNoOverlap(events,isData,dataset,str(year))

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


        ######### Store boolean masks with PackedSelection ##########

        selections = PackedSelection(dtype='uint64')

        # Lumi mask (for data)
        selections.add("is_good_lumi",lumi_mask)

        # 2lss selection (drained of 4 top)
        selections.add("2lss_p", (events.is2l & chargel0_p & bmask_atleast1med_atleast2loose & pass_trg & bmask_atmost2med))  # Note: The ss requirement has NOT yet been made at this point! We take care of it later with the appl axis
        selections.add("2lss_m", (events.is2l & chargel0_m & bmask_atleast1med_atleast2loose & pass_trg & bmask_atmost2med))  # Note: The ss requirement has NOT yet been made at this point! We take care of it later with the appl axis

        # 2lss selection (enriched in 4 top)
        selections.add("2lss_4t_p", (events.is2l & chargel0_p & bmask_atleast1med_atleast2loose & pass_trg & bmask_atleast3med))  # Note: The ss requirement has NOT yet been made at this point! We take care of it later with the appl axis
        selections.add("2lss_4t_m", (events.is2l & chargel0_m & bmask_atleast1med_atleast2loose & pass_trg & bmask_atleast3med))  # Note: The ss requirement has NOT yet been made at this point! We take care of it later with the appl axis
    
        # 2lss selection for CR
        selections.add("2lss_CR", (events.is2l & (chargel0_p | chargel0_m) & bmask_exactly1med & pass_trg)) # Note: The ss requirement has NOT yet been made at this point! We take care of it later with the appl axis
        selections.add("2lss_CRflip", (events.is2l_nozeeveto & events.is_ee & sfasz_2l_mask & pass_trg)) # Note: The ss requirement has NOT yet been made at this point! We take care of it later with the appl axis, also note explicitly include the ee requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement

        # 2los selection
        selections.add("2los_CRtt", (events.is2l_nozeeveto & charge2l_0 & events.is_em & bmask_exactly2med & pass_trg)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("2los_CRZ", (events.is2l_nozeeveto & charge2l_0 & sfosz_2l_mask & bmask_exactly0med & pass_trg))

        # 3l selection
        selections.add("3l_p_offZ_1b", (events.is3l & charge3l_p & ~sfosz_3l_mask & bmask_exactly1med & pass_trg))
        selections.add("3l_m_offZ_1b", (events.is3l & charge3l_m & ~sfosz_3l_mask & bmask_exactly1med & pass_trg))
        selections.add("3l_p_offZ_2b", (events.is3l & charge3l_p & ~sfosz_3l_mask & bmask_atleast2med & pass_trg))
        selections.add("3l_m_offZ_2b", (events.is3l & charge3l_m & ~sfosz_3l_mask & bmask_atleast2med & pass_trg))
        selections.add("3l_onZ_1b", (events.is3l & sfosz_3l_mask & bmask_exactly1med & pass_trg))
        selections.add("3l_onZ_2b", (events.is3l & sfosz_3l_mask & bmask_atleast2med & pass_trg))
        selections.add("3l_CR", (events.is3l & bmask_exactly0med & pass_trg))

        # 4l selection
        selections.add("4l", (events.is4l & bmask_atleast1med_atleast2loose & pass_trg))

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
        ptz = get_Z_pt(l_fo_conept_sorted_padded[:,0:3],10.0)

        # Leading (b+l) pair pt
        bjetsl = goodJets[isBtagJetsLoose][ak.argsort(goodJets[isBtagJetsLoose].pt, axis=-1, ascending=False)]
        bl_pairs = ak.cartesian({"b":bjetsl,"l":l_fo_conept_sorted})
        blpt = (bl_pairs["b"] + bl_pairs["l"]).pt
        bl0pt = ak.flatten(blpt[ak.argmax(blpt,axis=-1,keepdims=True)])

        # Collection of all objects (leptons and jets)
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

        # Counts
        counts = np.ones_like(events['event'])

        # Variables we will loop over when filling hists
        varnames = {}
        varnames["invmass"] = mll_0_1


        ########## Fill the histograms ##########

        # This dictionary keeps track of which selections go with which SR categories
        sr_cat_dict = {
          "2l" : {
              "exactly_4j" : {
                  "lep_chan_lst" : ["2lss_p" , "2lss_m", "2lss_4t_p", "2lss_4t_m"],
                  "lep_flav_lst" : ["ee" , "em" , "mm"],
                  "appl_lst"     : ["isSR_2lSS" , "isAR_2lSS"] + (["isAR_2lSS_OS"] if isData else []),
              },
              "exactly_5j" : {
                  "lep_chan_lst" : ["2lss_p" , "2lss_m", "2lss_4t_p", "2lss_4t_m"],
                  "lep_flav_lst" : ["ee" , "em" , "mm"],
                  "appl_lst"     : ["isSR_2lSS" , "isAR_2lSS"] + (["isAR_2lSS_OS"] if isData else []),
              },
              "exactly_6j" : {
                  "lep_chan_lst" : ["2lss_p" , "2lss_m", "2lss_4t_p", "2lss_4t_m"],
                  "lep_flav_lst" : ["ee" , "em" , "mm"],
                  "appl_lst"     : ["isSR_2lSS" , "isAR_2lSS"] + (["isAR_2lSS_OS"] if isData else []),
              },
              "atleast_7j" : {
                  "lep_chan_lst" : ["2lss_p" , "2lss_m", "2lss_4t_p", "2lss_4t_m"],
                  "lep_flav_lst" : ["ee" , "em" , "mm"],
                  "appl_lst"     : ["isSR_2lSS" , "isAR_2lSS"] + (["isAR_2lSS_OS"] if isData else []),
              },
          },
          "3l" : {
              "exactly_2j" : {
                  "lep_chan_lst" : [
                      "3l_p_offZ_1b" , "3l_m_offZ_1b" , "3l_p_offZ_2b" , "3l_m_offZ_2b" , "3l_onZ_1b" , "3l_onZ_2b",
                  ],
                  "lep_flav_lst" : ["eee" , "eem" , "emm", "mmm"],
                  "appl_lst"     : ["isSR_3l", "isAR_3l"],
              },
              "exactly_3j" : {
                  "lep_chan_lst" : [
                      "3l_p_offZ_1b" , "3l_m_offZ_1b" , "3l_p_offZ_2b" , "3l_m_offZ_2b" , "3l_onZ_1b" , "3l_onZ_2b",
                  ],
                  "lep_flav_lst" : ["eee" , "eem" , "emm", "mmm"],
                  "appl_lst"     : ["isSR_3l", "isAR_3l"],
              },
              "exactly_4j" : {
                  "lep_chan_lst" : [
                      "3l_p_offZ_1b" , "3l_m_offZ_1b" , "3l_p_offZ_2b" , "3l_m_offZ_2b" , "3l_onZ_1b" , "3l_onZ_2b",
                  ],
                  "lep_flav_lst" : ["eee" , "eem" , "emm", "mmm"],
                  "appl_lst"     : ["isSR_3l", "isAR_3l"],
              },
              "atleast_5j" : {
                  "lep_chan_lst" : [
                      "3l_p_offZ_1b" , "3l_m_offZ_1b" , "3l_p_offZ_2b" , "3l_m_offZ_2b" , "3l_onZ_1b" , "3l_onZ_2b",
                  ],
                  "lep_flav_lst" : ["eee" , "eem" , "emm", "mmm"],
                  "appl_lst"     : ["isSR_3l", "isAR_3l"],
              },
          },
          "4l" : {
                  "exactly_2j" : {
                      "lep_chan_lst" : ["4l"],
                      "lep_flav_lst" : ["llll"], # Not keeping track of these separately
                      "appl_lst"     : ["isSR_4l"],
                  },
                  "exactly_3j" : {
                      "lep_chan_lst" : ["4l"],
                      "lep_flav_lst" : ["llll"], # Not keeping track of these separately
                      "appl_lst"     : ["isSR_4l"],
                  },
                  "atleast_4j" : {
                      "lep_chan_lst" : ["4l"],
                      "lep_flav_lst" : ["llll"], # Not keeping track of these separately
                      "appl_lst"     : ["isSR_4l"],
                  },
          },
        }


        cat_dict = sr_cat_dict

        # Loop over the hists we want to fill
        for dense_axis_name, dense_axis_vals in varnames.items():

            # Set up the list of syst wgt variations to loop over
            wgt_var_lst = ["nominal"]

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

                    # Get a mask for events that pass any of the njet requiremens in this nlep cat
                    # Useful in cases like njets hist where we don't store njets in a sparse axis
                    njets_any_mask = selections.any(*cat_dict[nlep_cat].keys())

                    # Loop over the njets list for each channel
                    for njet_val in cat_dict[nlep_cat].keys():

                        # Loop over the appropriate AR and SR for this channel
                        for appl in cat_dict[nlep_cat][njet_val]["appl_lst"]:

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

                                    # Fill the histos
                                    axes_fill_info_dict = {
                                        dense_axis_name : dense_axis_vals[all_cuts_mask],
                                        "channel"       : ch_name,
                                        "appl"          : appl,
                                        "sample"        : histAxisName,
                                        "systematic"    : wgt_fluct,
                                        "weight"        : weights_flat,
                                    }

                                    # Skip histos that are not defined (or not relevant) to given categories
                                    if ((("j0" in dense_axis_name) and ("lj0pt" not in dense_axis_name)) & (("CRZ" in ch_name) or ("CRflip" in ch_name))): continue
                                    if ((("j0" in dense_axis_name) and ("lj0pt" not in dense_axis_name)) & ("0j" in ch_name)): continue
                                    if (("ptz" in dense_axis_name) & ("onZ" not in lep_chan)): continue
                                    if ((dense_axis_name in ["o0pt","b0pt","bl0pt"]) & ("CR" in ch_name)): continue

                                    hout[dense_axis_name].fill(**axes_fill_info_dict)


                        # Do not loop over njets if hist is njets (otherwise we'd fill the hist too many times)
                        if dense_axis_name == "njets": break

        return hout

    def postprocess(self, accumulator):
        return accumulator
