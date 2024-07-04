#!/usr/bin/env python
import copy
import coffea
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import processor
from coffea.analysis_tools import PackedSelection
import hist

import topeft.modules.object_selection as te_os
from topcoffea.modules.histEFT import HistEFT
import topcoffea.modules.eft_helper as efth
import topcoffea.modules.event_selection as tc_es
from topcoffea.modules.get_param_from_jsons import GetParam
from topcoffea.modules.paths import topcoffea_path
get_tc_param = GetParam(topcoffea_path("params/params.json"))
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

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, ecut_threshold=None, do_errors=False, do_systematics=False, split_by_lepton_flavor=False, skip_signal_regions=False, skip_control_regions=False, dtype=np.float32):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype
        self._do_systematics = do_systematics # Whether to process systematic samples
        self._skip_signal_regions = skip_signal_regions # Whether to skip the SR categories
        self._skip_control_regions = skip_control_regions # Whether to skip the CR categories

        # Create the histograms
        proc_axis = hist.axis.StrCategory([], name="process", growth=True)
        chan_axis = hist.axis.StrCategory([], name="channel", growth=True)
        syst_axis = hist.axis.StrCategory([], name="systematic", growth=True)
        appl_axis = hist.axis.StrCategory([], name="appl", label=r"AR/SR", growth=True)
        self._accumulator = {
            "mll_fromzg_e" : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(40,  0, 200,  name="mll_fromzg_e", label=r"invmass ee from z/gamma"), wc_names=wc_names_lst, rebin=False),
            "mll_fromzg_m" : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(40,  0, 200,  name="mll_fromzg_m", label=r"invmass mm from z/gamma"), wc_names=wc_names_lst, rebin=False),
            "mll_fromzg_t" : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(40,  0, 200,  name="mll_fromzg_t", label=r"invmass tautau from z/gamma"), wc_names=wc_names_lst, rebin=False),
            "mll"          : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(60,  0, 600,  name="mll",          label=r"Invmass l0l1"), wc_names=wc_names_lst, rebin=False),
            "ht"           : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(100, 0, 1000, name="ht",           label=r"Scalar sum of genjet pt"), wc_names=wc_names_lst, rebin=False),
            "ht_clean"     : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(100, 0, 1000, name="ht_clean",     label=r"Scalar sum of clean genjet pt"), wc_names=wc_names_lst, rebin=False),
            "tops_pt"      : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(50,  0, 500,  name="tops_pt",      label=r"Pt of the sum of the tops"), wc_names=wc_names_lst, rebin=False),
            "photon_pt"    : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Variable([20,35,50,70,100,170,200,250,300],  name="photon_pt",      label=r"Pt of the sum of the photon"), wc_names=wc_names_lst, rebin=False),
            "l0_pt"        : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(50,  0, 500,    name="l0_pt",      label=r"Pt of leading lepton"), wc_names=wc_names_lst, rebin=False),
            "j0_pt"        : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(50,  0, 500,    name="j0_pt",      label=r"Pt of leading jet"), wc_names=wc_names_lst, rebin=False),
            "tX_pt"        : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(40,  0, 400,  name="tX_pt",        label=r"Pt of the t(t)X system"), wc_names=wc_names_lst, rebin=False),
            "njets"        : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(10,  0, 10,   name="njets",        label=r"njets"), wc_names=wc_names_lst, rebin=False),
        }

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

        selections = PackedSelection(dtype='uint64')

        ### Get gen particles collection ###
        genpart = events.GenPart
        genjet = events.GenJet


        ### Lepton object selection ###

        is_final_mask = genpart.hasFlags(["fromHardProcess","isLastCopy"])

        gen_top = ak.pad_none(genpart[is_final_mask & (abs(genpart.pdgId) == 6)],2)

        gen_l = genpart[is_final_mask & ((abs(genpart.pdgId) == 11) | (abs(genpart.pdgId) == 13) | (abs(genpart.pdgId) == 15))]
        gen_e = genpart[is_final_mask & (abs(genpart.pdgId) == 11)]
        gen_m = genpart[is_final_mask & (abs(genpart.pdgId) == 13)]
        gen_t = genpart[is_final_mask & (abs(genpart.pdgId) == 15)]

        gen_p = genpart[is_final_mask & (abs(genpart.pdgId) == 22)]
        ######### Systematics ###########

        # Define the lists of systematics we include
        wgt_correction_syst_lst = [
            "CMS_eff_gUp","CMS_eff_gDown" # Exp systs
        ]

        gen_l = gen_l[ak.argsort(gen_l.pt, axis=-1, ascending=False)]
        gen_l = ak.pad_none(gen_l,2)
        gen_e = gen_e[ak.argsort(gen_e.pt, axis=-1, ascending=False)]
        gen_m = gen_m[ak.argsort(gen_m.pt, axis=-1, ascending=False)]
        gen_t = gen_t[ak.argsort(gen_t.pt, axis=-1, ascending=False)]

        # Object selection after padding
        is_em = (np.abs(gen_l[:,0].pdgId)==11) & (np.abs(gen_l[:,1].pdgId)==13)
        is_me = (np.abs(gen_l[:,1].pdgId)==11) & (np.abs(gen_l[:,0].pdgId)==13)
        is2lOS = ak.any(gen_l[:,0].pdgId/np.abs(gen_l[:,1].pdgId)+gen_l[:,1].pdgId/np.abs(gen_l[:,1].pdgId)==0) & ((ak.num(gen_e) + ak.num(gen_m))==2)
        is2lOS_em = is2lOS & ak.any(is_em | is_me)

        gen_p = ak.fill_none(ak.pad_none(gen_p,1), 0)
        gen_p = gen_p[ak.argsort(gen_p.pt, axis=-1, ascending=False)]

        gen_l_from_zg = ak.pad_none(gen_l[(gen_l.distinctParent.pdgId == 23) | (gen_l.distinctParent.pdgId == 22)], 2)
        gen_e_from_zg = ak.pad_none(gen_e[(gen_e.distinctParent.pdgId == 23) | (gen_e.distinctParent.pdgId == 22)], 2)
        gen_m_from_zg = ak.pad_none(gen_m[(gen_m.distinctParent.pdgId == 23) | (gen_m.distinctParent.pdgId == 22)], 2)
        gen_t_from_zg = ak.pad_none(gen_t[(gen_t.distinctParent.pdgId == 23) | (gen_t.distinctParent.pdgId == 22)], 2)


        # Jet object selection
        genjet = genjet[genjet.pt > 30]
        is_clean_jet = te_os.isClean(genjet, gen_e, drmin=0.4) & te_os.isClean(genjet, gen_m, drmin=0.4) & te_os.isClean(genjet, gen_t, drmin=0.4)
        genjet_clean = genjet[is_clean_jet]
        genbjet_clean = genjet[np.abs(genjet.partonFlavour)==5]
        njets = ak.num(genjet_clean)
        nbjets = ak.num(genbjet_clean)
        atleast_1ph = ak.num(gen_p)>0


        selections.add("2los_CRtt", is2lOS_em & (nbjets==2) & (~atleast_1ph)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        #selections.add("2los_CRtt", (events.is2l_nozeeveto & charge2l_0 & events.is_em & bmask_exactly2med & pass_trg)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("2los_ph", is2lOS & (nbjets==2) & (ak.num(gen_p)>0) & (ak.firsts(gen_p).pt>20)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        #selections.add("2los_sf_ph", (retainedbyOverlap & events.is2l & charge2l_0 & (events.is_ee | events.is_mm) & ~sfosz_2los_ttg_mask & events.mask_SF_Zllgamma & bmask_atleast1med & pass_trg & exactly_1ph))
        sfosz_2los_ttg_mask = tc_es.get_Z_peak_mask(gen_l[:,0:2],pt_window=15.0) 
        Zllgamma_SF_mask = (abs( (gen_l[:,0] + gen_l[:,1] + gen_p[:,0]).mass -91.2) > 15)
        selections.add("2los_sf_Zg_CR_ULttg", (is2lOS_em & ~sfosz_2los_ttg_mask & Zllgamma_SF_mask & (nbjets==2) & atleast_1ph))
        #selections.add("2los_sf_Zg_CR_ULttg", (retainedbyOverlap & events.is2l_nozeeveto & charge2l_0 & (events.is_ee | events.is_mm) & ~sfosz_2los_ttg_mask & ~events.mask_SF_Zllgamma & bmask_atleast1med & pass_trg & atleast_1ph))


        # Njets selection
        selections.add("exactly_0j", (njets==0))
        selections.add("exactly_1j", (njets==1))
        selections.add("exactly_2j", (njets==2))
        selections.add("exactly_3j", (njets==3))
        selections.add("exactly_4j", (njets==4))
        selections.add("exactly_5j", (njets==5))
        selections.add("exactly_6j", (njets==6))
        selections.add("atleast_1j", (njets>=1))
        selections.add("atleast_2j", (njets>=2))
        selections.add("atleast_3j", (njets>=3))
        selections.add("atleast_4j", (njets>=4))
        selections.add("atleast_5j", (njets>=5))
        selections.add("atleast_7j", (njets>=7))
        selections.add("atleast_0j", (njets>=0))
        selections.add("atmost_1j" , (njets<=1))
        selections.add("atmost_3j" , (njets<=3))


        # AR/SR categories
        #selections.add("isSR_2lSS",    ( events.is2l_SR) & charge2l_1)
        #selections.add("isAR_2lSS",    (~events.is2l_SR) & charge2l_1)
        #selections.add("isAR_2lSS_OS", ( events.is2l_SR) & charge2l_0) # Sideband for the charge flip
        selections.add("isSR_2lOS",    is2lOS)#( events.is2l_SR) & charge2l_0)
        selections.add("isAR_2lOS",    is2lOS)#(~events.is2l_SR) & charge2l_0)

        #selections.add("isSR_3l",  events.is3l_SR)
        #selections.add("isAR_3l", ~events.is3l_SR)
        #selections.add("isSR_4l",  events.is4l_SR)


        ### Get dense axis variables ###

        ht = ak.sum(genjet.pt,axis=-1)
        ht_clean = ak.sum(genjet_clean.pt,axis=-1)

        tops_pt = gen_top.sum().pt

        # Pt of the t(t)X system
        tX_system = ak.concatenate([gen_top,gen_l_from_zg],axis=1)
        tX_pt = tX_system.sum().pt

        # Invmass distributions
        mll_e_from_zg = (gen_e_from_zg[:,0] + gen_e_from_zg[:,1]).mass
        mll_m_from_zg = (gen_m_from_zg[:,0] + gen_m_from_zg[:,1]).mass
        mll_t_from_zg = (gen_t_from_zg[:,0] + gen_t_from_zg[:,1]).mass
        mll_l0l1 = (gen_l[:,0] + gen_l[:,1]).mass

        # Dictionary of dense axis values
        dense_axis_dict = {
            "mll_fromzg_e" : mll_e_from_zg,
            "mll_fromzg_m" : mll_m_from_zg,
            "mll_fromzg_t" : mll_t_from_zg,
            "mll" : mll_l0l1,
            "ht" : ht,
            "ht_clean" : ht_clean,
            "tX_pt" : tX_pt,
            "tops_pt" : tops_pt,
            "photon_pt" : ak.firsts(gen_p).pt,
            "l0_pt" : ak.firsts(gen_l.pt),
            "j0_pt" : ak.firsts(genjet.pt),
            "njets" : njets,
        }
        cat_dict = {
            "2los_CRtt" : {
                "atmost_3j"   : {
                    "lep_chan_lst" : ["2los_ph"],
                    "lep_flav_lst" : ["em"],
                    "appl_lst"     : ["isSR_2lOS" , "isAR_2lOS"],
                },
            },
        }
        sr_cat_dict = {
            "2los_ph" : {
                "atleast_3j"   : {
                    "lep_chan_lst" : ["2los_ph"],
                    "lep_flav_lst" : ["ee", "mm", "em"],
                    "appl_lst"     : ["isSR_2lOS", "isAR_2lOS"],
                },
            },
        }
        cr_cat_dict = {
            "2los_CR_Zg_ULttg" : {
                "atleast_2j"   : {
                    "lep_chan_lst" : ["2los_sf_Zg_CR_ULttg"],
                    "lep_flav_lst" : ["ee", "mm"],
                    "appl_lst"     : ["isSR_2lOS", "isAR_2lOS"],
                },
            },
        }


        ### Get weights ###

        # Extract the EFT quadratic coefficients and optionally use them to calculate the coefficients on the w**2 quartic function
        # eft_coeffs is never Jagged so convert immediately to numpy for ease of use.
        eft_coeffs = ak.to_numpy(events["EFTfitCoefficients"]) if hasattr(events, "EFTfitCoefficients") else None
        if eft_coeffs is not None:
            # Check to see if the ordering of WCs for this sample matches what want
            if self._samples[dataset]["WCnames"] != self._wc_names_lst:
                eft_coeffs = efth.remap_coeffs(self._samples[dataset]["WCnames"], self._wc_names_lst, eft_coeffs)
        eft_w2_coeffs = efth.calc_w2_coeffs(eft_coeffs,self._dtype) if (self._do_errors and eft_coeffs is not None) else None

        # If this is not an eft sample, get the genWeight
        if eft_coeffs is None: genw = events["genWeight"]
        else: genw = np.ones_like(events["event"])
        lumi = 1000.0*get_tc_param(f"lumi_{year}")
        event_weight = lumi*xsec*genw/sow

        weights_dict = {}
        # For both data and MC
        weights_obj_base = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
        weights_obj_base.add("norm",(xsec/sow)*genw*lumi)
        for ch_name in ["2los_CRtt", "2los_ph", "2los_CR_Zg_ULttg"]:
            weights_dict[ch_name] = copy.deepcopy(weights_obj_base)
            weights_dict[ch_name].add("CMS_eff_g", ak.ones_like(ak.firsts(gen_p).pt), np.random.rand(*ak.to_numpy(ak.firsts(gen_p).pt).shape), np.random.rand(*ak.to_numpy(ak.firsts(gen_p).pt).shape))

        # Example of reweighting based on Ht
        #if "private" in histAxisName:
        #    ht_sf = get_ht_sf(ht,histAxisName)
        #    event_weight = event_weight*ht_sf


        ### Loop over the hists we want to fill ###

        hout = self.accumulator

        for dense_axis_name, dense_axis_vals in dense_axis_dict.items():
            if dense_axis_name not in self._hist_lst:
                #print(f"Skipping \"{dense_axis_name}\", it is not in the list of hists to include.")
                continue
            if not self._skip_signal_regions:
                cat_dict.update(sr_cat_dict)
            if not self._skip_control_regions:
                cat_dict.update(cr_cat_dict)
            if (not self._skip_signal_regions and not self._skip_control_regions):
                for k in sr_cat_dict:
                    if k in cr_cat_dict:
                        raise Exception(f"The key {k} is in both CR and SR dictionaries.")

            # Set up the list of syst wgt variations to loop over
            wgt_var_lst = ["nominal"]
            if self._do_systematics:
                if not isData:
                    wgt_var_lst = wgt_var_lst + wgt_correction_syst_lst

            for wgt_fluct in wgt_var_lst:
                for nlep_cat in cat_dict.keys():
                    flav_ch = None
                    njet_ch = None
                    njets_any_mask = selections.any(*cat_dict[nlep_cat].keys())
                    weights_object = weights_dict[nlep_cat]
                    if (wgt_fluct == "nominal"):
                        # In the case of "nominal", or the jet energy systematics, no weight systematic variation is used
                        weight = weights_object.weight(None)
                    else:
                        # Otherwise get the weight from the Weights object
                        if wgt_fluct in weights_object.variations:
                            weight = weights_object.weight(wgt_fluct)
                        else:
                            # Note in this case there is no up/down fluct for this cateogry, so we don't want to fill a hist for it
                            continue
                    # Loop over the njets list for each channel
                    for njet_val in cat_dict[nlep_cat]:

                        # Loop over the appropriate AR and SR for this channel
                        for appl in cat_dict[nlep_cat][njet_val]["appl_lst"]:
                            # Loop over the channels in each nlep cat (e.g. "3l_m_offZ_1b")
                            for lep_chan in cat_dict[nlep_cat][njet_val]["lep_chan_lst"]:

                                # Loop over the lep flavor list for each channel
                                for lep_flav in cat_dict[nlep_cat][njet_val]["lep_flav_lst"]:
                                    #cuts_lst = [chan, njet_val]
                                    cuts_lst = [appl,lep_chan]
                                    if dense_axis_name == "njets":
                                        all_cuts_mask = (selections.all(*cuts_lst) & njets_any_mask)
                                    else:
                                        njet_ch = njet_val
                                        all_cuts_mask = (selections.all(*cuts_lst))
                                    ch_name = construct_cat_name(lep_chan,njet_str=njet_ch,flav_str=flav_ch)

                                    # Mask out the none values
                                    isnotnone_mask = (ak.fill_none((dense_axis_vals != None),False))
                                    isnotnone_mask = isnotnone_mask & all_cuts_mask
                                    dense_axis_vals_cut = dense_axis_vals[isnotnone_mask]
                                    event_weight_cut = weight[isnotnone_mask]
                                    eft_coeffs_cut = eft_coeffs
                                    if eft_coeffs is not None: eft_coeffs_cut = eft_coeffs[isnotnone_mask]
                                    #eft_w2_coeffs_cut = eft_w2_coeffs
                                    #if eft_w2_coeffs is not None: eft_w2_coeffs_cut = eft_w2_coeffs[isnotnone_mask]

                                    # Fill the histos
                                    axes_fill_info_dict = {
                                        dense_axis_name : dense_axis_vals_cut,
                                        "process"       : histAxisName,
                                        "appl"          : appl,
                                        "channel"       : ch_name,
                                        "systematic"    : wgt_fluct,
                                        "weight"        : event_weight_cut,
                                        "eft_coeff"     : eft_coeffs_cut,
                                        #"eft_err_coeff" : eft_w2_coeffs_cut,
                                    }

                                    hout[dense_axis_name].fill(**axes_fill_info_dict)

        return hout

    def postprocess(self, accumulator):
        return accumulator
