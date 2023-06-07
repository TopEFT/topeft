#!/usr/bin/env python
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import hist, processor

from topcoffea.modules.GetValuesFromJsons import get_lumi
from topcoffea.modules.objects import *
#from topcoffea.modules.corrections import get_ht_sf
from topcoffea.modules.selection import *
from topcoffea.modules.HistEFT import HistEFT
import topcoffea.modules.eft_helper as efth


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, ecut_threshold=None, do_errors=False, do_systematics=False, split_by_lepton_flavor=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        # Create the histograms
        self._accumulator = processor.dict_accumulator({
            "mll_fromzg_e" : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("mll_fromzg_e", "invmass ee from z/gamma", 40, 0, 200)),
            "mll_fromzg_m" : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("mll_fromzg_m", "invmass mm from z/gamma", 40, 0, 200)),
            "mll_fromzg_t" : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("mll_fromzg_t", "invmass tautau from z/gamma", 40, 0, 200)),
            "mll"          : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("mll", "Invmass l0l1", 60, 0, 600)),
            "ht"           : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("ht", "Scalar sum of genjet pt", 100, 0, 1000)),
            "ht_clean"     : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("ht_clean", "Scalar sum of clean genjet pt", 100, 0, 1000)),
            "tops_pt"      : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("tops_pt", "Pt of the sum of the tops", 50, 0, 500)),
            "tX_pt"        : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("tX_pt", "Pt of the t(t)X system", 40, 0, 400)),
            "njets"        : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("njets", "njets", 10, 0, 10)),
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
        genpart = events.GenPart
        genjet = events.GenJet


        ### Lepton object selection ###

        is_final_mask = genpart.hasFlags(["fromHardProcess","isLastCopy"])

        gen_top = ak.pad_none(genpart[is_final_mask & (abs(genpart.pdgId) == 6)],2)

        gen_l = genpart[is_final_mask & ((abs(genpart.pdgId) == 11) | (abs(genpart.pdgId) == 13) | (abs(genpart.pdgId) == 15))]
        gen_e = genpart[is_final_mask & (abs(genpart.pdgId) == 11)]
        gen_m = genpart[is_final_mask & (abs(genpart.pdgId) == 13)]
        gen_t = genpart[is_final_mask & (abs(genpart.pdgId) == 15)]

        gen_l = gen_l[ak.argsort(gen_l.pt, axis=-1, ascending=False)]
        gen_l = ak.pad_none(gen_l,2)
        gen_e = gen_e[ak.argsort(gen_e.pt, axis=-1, ascending=False)]
        gen_m = gen_m[ak.argsort(gen_m.pt, axis=-1, ascending=False)]
        gen_t = gen_t[ak.argsort(gen_t.pt, axis=-1, ascending=False)]

        gen_l_from_zg = ak.pad_none(gen_l[(gen_l.distinctParent.pdgId == 23) | (gen_l.distinctParent.pdgId == 22)], 2)
        gen_e_from_zg = ak.pad_none(gen_e[(gen_e.distinctParent.pdgId == 23) | (gen_e.distinctParent.pdgId == 22)], 2)
        gen_m_from_zg = ak.pad_none(gen_m[(gen_m.distinctParent.pdgId == 23) | (gen_m.distinctParent.pdgId == 22)], 2)
        gen_t_from_zg = ak.pad_none(gen_t[(gen_t.distinctParent.pdgId == 23) | (gen_t.distinctParent.pdgId == 22)], 2)


        # Jet object selection
        genjet = genjet[genjet.pt > 30]
        is_clean_jet = isClean(genjet, gen_e, drmin=0.4) & isClean(genjet, gen_m, drmin=0.4) & isClean(genjet, gen_t, drmin=0.4)
        genjet_clean = genjet[is_clean_jet]
        njets = ak.num(genjet_clean)


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
            "njets" : njets,
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
        lumi = get_lumi(year)*1000.0
        event_weight = lumi*xsec*genw/sow

        # Example of reweighting based on Ht
        #if "private" in histAxisName:
        #    ht_sf = get_ht_sf(ht,histAxisName)
        #    event_weight = event_weight*ht_sf


        ### Loop over the hists we want to fill ###

        hout = self.accumulator.identity()

        for dense_axis_name, dense_axis_vals in dense_axis_dict.items():

            # Mask out the none values
            isnotnone_mask = (ak.fill_none((dense_axis_vals != None),False))
            dense_axis_vals_cut = dense_axis_vals[isnotnone_mask]
            event_weight_cut = event_weight[isnotnone_mask]
            eft_coeffs_cut = eft_coeffs
            if eft_coeffs is not None: eft_coeffs_cut = eft_coeffs[isnotnone_mask]
            eft_w2_coeffs_cut = eft_w2_coeffs
            if eft_w2_coeffs is not None: eft_w2_coeffs_cut = eft_w2_coeffs[isnotnone_mask]

            # Fill the histos
            axes_fill_info_dict = {
                dense_axis_name : dense_axis_vals_cut,
                "sample"        : histAxisName,
                "weight"        : event_weight_cut,
                "eft_coeff"     : eft_coeffs_cut,
                "eft_err_coeff" : eft_w2_coeffs_cut,
            }

            hout[dense_axis_name].fill(**axes_fill_info_dict)

        return hout

    def postprocess(self, accumulator):
        return accumulator
