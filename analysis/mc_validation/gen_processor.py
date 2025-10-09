#!/usr/bin/env python
import copy
import coffea
#from coffea.nanoevents.methods import candidate
import warnings
warnings.filterwarnings("ignore", message="Missing cross-reference index")
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
from topeft.modules.axes import info as axes_info
from topcoffea.modules.get_param_from_jsons import GetParam
from topcoffea.modules.paths import topcoffea_path
get_tc_param = GetParam(topcoffea_path("params/params.json"))

#import topcoffea.modules.corrections as tc_cor

def construct_cat_name(chan_str,nlep_cat,njet_str=None,flav_str=None):

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
    ret_str = nlep_cat
    for component in [flav_str,chan_str,njet_str]:
        if component is None: continue
        ret_str = "_".join([ret_str,component])
    ret_str = ret_str.replace('ph_ph', 'ph')
    return ret_str


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, ecut_threshold=None, do_errors=False, do_systematics=False, split_by_lepton_flavor=False, skip_signal_regions=False, skip_control_regions=False, dtype=np.float32):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype
        self._do_systematics = do_systematics # Whether to process systematic samples
        self._skip_signal_regions = skip_signal_regions # Whether to skip the SR categories
        self._skip_control_regions = skip_control_regions # Whether to skip the CR categories
        self._split_by_lepton_flavor = split_by_lepton_flavor # Whether to keep track of lepton flavors individually

        # Create the histograms
        axes_info["photon_pt"] = {
            "regular": (20, 0, 400),
            "variable": [20,35,50,70,100,170,200,250,300],
            "label": "$p_{T}$ $\gamma$ (GeV)"
        }
        proc_axis = hist.axis.StrCategory([], name="process", growth=True)
        chan_axis = hist.axis.StrCategory([], name="channel", growth=True)
        syst_axis = hist.axis.StrCategory([], name="systematic", growth=True)
        appl_axis = hist.axis.StrCategory([], name="appl", label=r"AR/SR", growth=True)
        self._accumulator = {
            "mll_fromzg_e" : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(40,  0, 200,  name="mll_fromzg_e", label=r"invmass ee from z/gamma"), wc_names=wc_names_lst, rebin=False),
            "mll_fromzg_m" : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(40,  0, 200,  name="mll_fromzg_m", label=r"invmass mm from z/gamma"), wc_names=wc_names_lst, rebin=False),
            "mll_fromzg_t" : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(40,  0, 200,  name="mll_fromzg_t", label=r"invmass tautau from z/gamma"), wc_names=wc_names_lst, rebin=False),
            "mll"          : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(30,  0, 300,  name="mll",          label=r"Invmass l0l1"), wc_names=wc_names_lst, rebin=False),
            "invm"          : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(100,  1000, 5000,  name="invm",        label=r"Invmass of system"), wc_names=wc_names_lst, rebin=False),
            "ht"           : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(100, 0, 1000, name="ht",           label=r"Scalar sum of genjet pt"), wc_names=wc_names_lst, rebin=False),
            "ht_clean"     : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(100, 0, 1000, name="ht_clean",     label=r"Scalar sum of clean genjet pt"), wc_names=wc_names_lst, rebin=False),
            "lhe_t_pt"      : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(50,  0, 500,   name="lhe_t_pt",     label=r"Pt of the leading LHE t"), wc_names=wc_names_lst, rebin=False),
            "t_pt"      : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(50,  0, 500,  name="t_pt",            label=r"Pt of the leading t"), wc_names=wc_names_lst, rebin=False),
            #"t_pt"      : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(15,  0, 800,  name="t_pt",            label=r"Pt of the leading t"), wc_names=wc_names_lst, rebin=False),
            "tops_pt"      : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(50,  0, 500,  name="tops_pt",      label=r"Pt of the sum of the tops"), wc_names=wc_names_lst, rebin=False),
            "dral"    : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(100,0,1, name="dral", label=r"$\Delta R(\gamma, \ell)$"), wc_names=wc_names_lst, rebin=False),
            "dral_sec"    : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(100,0,1, name="dral_sec", label=r"Sub $\Delta R(\gamma, \ell)$"), wc_names=wc_names_lst, rebin=False),
            "draj"    : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(100,0,1, name="draj", label=r"$\Delta R(\gamma, j)$"), wc_names=wc_names_lst, rebin=False),
            "photon_pt"    : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Variable(axes_info["photon_pt"]["variable"], name="photon_pt", label=r"$p_{\mathrm{T}}$ $\gamma$ (GeV)"), wc_names=wc_names_lst, rebin=False),
            #FIXME "lhe_photon_pt"    : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Variable(axes_info["photon_pt"]["variable"], name="lhe_photon_pt", label=r"LHE $p_{\mathrm{T}}$ $\gamma$ (GeV)"), wc_names=wc_names_lst, rebin=False),
            #"photon_pt"    : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Variable(axes_info["photon_pt"]["variable"], name="photon_pt", label=axes_info["photon_pt"]["label"]), wc_names=wc_names_lst, rebin=False),
            #"photon_pt"    : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Variable([20,35,50,70,100,170,200,250,300],  name="photon_pt",      label=r"$p_{\mathrm{T}}$ $\gamma$ (GeV)"), wc_names=wc_names_lst, rebin=False),
            "photon_pt_gen"    : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Variable([20,35,50,70,100,170,200,250,300],  name="photon_pt",      label=r"$p_{\mathrm{T}}$ $\gamma$ (GeV)"), wc_names=wc_names_lst, rebin=False),
            "photon_pt_cnt"    : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(28,20,300,  name="photon_pt",      label=r"$p_{\mathrm{T}}$ $\gamma$ (GeV) count"), wc_names=wc_names_lst, rebin=False),
            #"photon_pt_cnt"    : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Variable([20,35,50,70,100,170,200,250,300],  name="photon_pt",      label=r"$p_{\mathrm{T}}$ $\gamma$ (GeV) count"), wc_names=wc_names_lst, rebin=False),
            "photon_eta"        : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(30,  -1.5, 1.5,    name="photon_eta",      label=r"$\eta$ $\gamma$ (GeV)"), wc_names=wc_names_lst, rebin=False),
            "SumOfWeights": HistEFT(proc_axis, hist.axis.Regular(bins=1, start=0, stop=2, name="SumOfWeights", label="SumOfWeights"), wc_names=wc_names_lst),
            "SumOfWeights_eft": HistEFT(proc_axis, hist.axis.Regular(bins=1, start=0, stop=2, name="SumOfWeights", label="SumOfWeights"), wc_names=wc_names_lst),
            #"photon_l_pt"    : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Variable([20,35,50,70,100,170,200,250,300],  name="photon_pt",      label=r"$p_{\mathrm{T}}$ $\gamma$ (GeV)"), wc_names=wc_names_lst, rebin=False),
            "lhe_l0pt"        : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(25,  0, 250,    name="lhe_l0pt",      label=r"Pt of leading LHE lepton"), wc_names=wc_names_lst, rebin=False),
            #"l0pt"        : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(25,  0, 350,    name="l0pt",      label=r"Pt of leading lepton"), wc_names=wc_names_lst, rebin=False),
            #"l0pt"        : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(15,  0, 500,    name="l0pt",      label=r"Pt of leading lepton"), wc_names=wc_names_lst, rebin=False),
            "l0pt"        : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(25,  0, 250,    name="l0pt",      label=r"Pt of leading lepton"), wc_names=wc_names_lst, rebin=False),
            "j0pt"        : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(15,  0, 500,    name="j0pt",      label=r"Pt of leading jet"), wc_names=wc_names_lst, rebin=False),
            #"j0pt"        : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(30,  0, 300,    name="j0pt",      label=r"Pt of leading jet"), wc_names=wc_names_lst, rebin=False),
            "lj0pt"        : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(15,  0, 500,    name="lj0pt",      label=r"Pt of leading light jet"), wc_names=wc_names_lst, rebin=False),
            #"lj0pt"        : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(300,  0, 300,    name="lj0pt",      label=r"Pt of leading light jet"), wc_names=wc_names_lst, rebin=False),
            "bj0pt"        : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(15,  0, 500,    name="bj0pt",      label=r"Pt of leading b jet"), wc_names=wc_names_lst, rebin=False),
            #"bj0pt"        : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(300,  0, 300,    name="bj0pt",      label=r"Pt of leading b jet"), wc_names=wc_names_lst, rebin=False),
            "tX_pt"        : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(40,  0, 400,  name="tX_pt",        label=r"Pt of the t(t)X system"), wc_names=wc_names_lst, rebin=False),
            "njets"        : HistEFT(proc_axis, chan_axis, syst_axis, appl_axis, hist.axis.Regular(10,  0, 10,   name="njets",        label=r"njets"), wc_names=wc_names_lst, rebin=False),
        }
        self._accumulator['gen_reco'] = hist.Hist(
            proc_axis,
            chan_axis,
            syst_axis,
            appl_axis,
            hist.axis.Variable([20,35,50,70,100,170,200,250,300], name='ptgen',  label=r'gen $p_{\rm{T}}$'),
            hist.axis.Variable([20,35,50,70,100,170,200,250,300], name='ptreco', label=r'reco $p_{\rm{T}}$'),
            label=r"Events"
        )

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
        post_mortem        = self._samples[dataset]["post_mortem"]
        sow_renormUp       = self._samples[dataset]["nSumOfWeights"]
        sow_renormDown     = self._samples[dataset]["nSumOfWeights"]
        sow_factUp         = self._samples[dataset]["nSumOfWeights"]
        sow_factDown       = self._samples[dataset]["nSumOfWeights"]

        if isData: raise Exception("Error: This processor is not for data")

        selections = PackedSelection(dtype='uint64')

        ### Get gen particles collection ###
        genpart = events.GenPart
        lhepart = events.LHEPart
        #photon = events.Photon
        genjet = events.GenJet

        ### Lepton object selection ###

        gen_top = ak.pad_none(genpart[(abs(genpart.pdgId) == 6)],2)
        gen_bos = ak.pad_none(genpart[(abs(genpart.pdgId) == 23) | (abs(genpart.pdgId) == 24) | abs(genpart.pdgId) == 25],2)
        gen_bos = ak.pad_none(genpart[abs(genpart.pdgId) == 25],2)
        #dilep_mask = (np.abs(lhepart.pdgId)) == 13 | (np.abs(lhepart.pdgId) == 11)
        #nu_mask = (np.abs(lhepart.pdgId)) == 14 | (np.abs(lhepart.pdgId) == 12)
        #b_mask = (np.abs(lhepart.pdgId)) == 5
        #ttbar_mask = ak.num(lhepart[dilep_mask])==2
        ##lhe_part = ak.zip({'pt': lhepart.pt, 'eta': lhepart.eta, 'phi': lhepart.phi, 'mass': lhepart.mass, 'charge': lhepart.pdgId/np.abs(lhepart.pdgId)}, with_name='PtEtaPhiMCandidate', behavior=candidate.behavior)

        ###############################################################################################
        # Be careful with the LHE information. Not every process has the particles in the same order! #
        ###############################################################################################
        #lhe_top = ak.sum(ak.concatenate([lhepart[dilep_mask], lhepart[nu_mask], lhepart[b_mask]],axis=1), -1)
        lhe_top  = lhepart[:,2]+lhepart[:,3]+lhepart[:,4] #FIXME
        #lhe_top  = lhepart[:,2] # tWA top is always elemnt 2
        #lhe_top  = lhepart[:,2]+lhepart[:,3]+lhepart[:,5] #TWG central sample has b and leptons + neutrinos
        #lhe_top  = lhepart[np.abs(lhepart.pdgId) == 6]
        #lhe_top  = lhepart[:,2]
        #lhe_atop = lhepart[:,3]
        #lhe_l = ak.max([lhepart[:,2].pt,lhepart[:,5].pt], axis=0)
        #lhe_ph = lhepart[:,8]

        gen_l = genpart[((abs(genpart.pdgId) == 11) | (abs(genpart.pdgId) == 13) | (abs(genpart.pdgId) == 15))]
        gen_e = genpart[(abs(genpart.pdgId) == 11)]
        gen_m = genpart[(abs(genpart.pdgId) == 13)]
        gen_t = genpart[(abs(genpart.pdgId) == 15)]

        # Dressed leptons
        gen_l = events.GenDressedLepton
        gen_e = gen_l[(abs(gen_l.pdgId) == 11)]
        gen_m = gen_l[(abs(gen_l.pdgId) == 13)]
        gen_t = gen_l[(abs(gen_l.pdgId) == 15)]

        gen_p_l = genpart[(abs(genpart.pdgId) == 22)]
        gen_p = events.GenIsolatedPhoton
        ######### Systematics ###########

        # Define the lists of systematics we include
        wgt_correction_syst_lst = [
            "phoSFUp","phoSFDown", # Exp systs
            "renormUp","renormDown", # Exp systs
            "factUp","factDown", # Exp systs
            #"CMS_effUp","CMS_effDown" # Exp systs
        ]

        gen_l = gen_l[ak.argsort(gen_l.pt, axis=-1, ascending=False)]
        gen_l = ak.pad_none(gen_l,2)
        gen_e = gen_e[ak.argsort(gen_e.pt, axis=-1, ascending=False)]
        gen_m = gen_m[ak.argsort(gen_m.pt, axis=-1, ascending=False)]
        gen_t = gen_t[ak.argsort(gen_t.pt, axis=-1, ascending=False)]

        # Object selection after padding
        is_em = (np.abs(gen_l[:,0].pdgId)==11) & (np.abs(gen_l[:,1].pdgId)==13)
        is_me = (np.abs(gen_l[:,1].pdgId)==11) & (np.abs(gen_l[:,0].pdgId)==13)
        is2lOS = (gen_l[:,0].pdgId+gen_l[:,1].pdgId == 0) & ((ak.num(gen_e) + ak.num(gen_m))==2)
        #is2lOS = ak.any(gen_l[:,0].pdgId/np.abs(gen_l[:,0].pdgId)+gen_l[:,1].pdgId/np.abs(gen_l[:,1].pdgId)==0) & ((ak.num(gen_e) + ak.num(gen_m))==2) & (ak.firsts(gen_l.pt)>25)
        #is2lOS = ak.any(gen_l[:,0].pdgId/np.abs(gen_l[:,1].pdgId)+gen_l[:,1].pdgId/np.abs(gen_l[:,1].pdgId)==0) & ((ak.num(gen_e) + ak.num(gen_m))==2)
        #is2lOS = (gen_l[:,0].pdgId/np.abs(gen_l[:,1].pdgId)+gen_l[:,1].pdgId/np.abs(gen_l[:,1].pdgId)==0) & ((ak.num(gen_e) + ak.num(gen_m))==2)
        is2lSS = (gen_l[:,0].pdgId+gen_l[:,1].pdgId != 0) & ((ak.num(gen_e) + ak.num(gen_m))==2)
        #is2lSS = (gen_l[:,0].pdgId/np.abs(gen_l[:,0].pdgId)*gen_l[:,1].pdgId/np.abs(gen_l[:,1].pdgId)<0) & ((ak.num(gen_e) + ak.num(gen_m))==2)
        is2lOS_em = is2lOS & ak.any(is_em | is_me)
        is3l = (ak.num(gen_e) + ak.num(gen_m))==3

        #gen_p = ak.fill_none(ak.pad_none(gen_p,1), 0)
        gen_p = ak.pad_none(gen_p,1)
        gen_p = ak.with_name(gen_p, 'PtEtaPhiMCandidate')
        #gen_p_l = gen_p[(np.abs(gen_p.distinctParent.pdgId)==11) | (np.abs(gen_p.distinctParent.pdgId)==13)]
        #gen_p = gen_p[ak.argsort(gen_p.pt, axis=-1, ascending=False)]

        #gen_l_from_zg = ak.pad_none(gen_l[(gen_l.distinctParent.pdgId == 23) | (gen_l.distinctParent.pdgId == 22)], 2)
        #gen_e_from_zg = ak.pad_none(gen_e[(gen_e.distinctParent.pdgId == 23) | (gen_e.distinctParent.pdgId == 22)], 2)
        #gen_m_from_zg = ak.pad_none(gen_m[(gen_m.distinctParent.pdgId == 23) | (gen_m.distinctParent.pdgId == 22)], 2)
        #gen_t_from_zg = ak.pad_none(gen_t[(gen_t.distinctParent.pdgId == 23) | (gen_t.distinctParent.pdgId == 22)], 2)


        # Jet object selection
        genjet = genjet#[genjet.pt > 30]
        genjet_clean_near, genjet_clean_DR = gen_p.nearest(genjet, return_metric=True)
        genlep_clean_near, genlep_clean_DR = gen_p.nearest(gen_l, return_metric=True)
        no_colinear = ak.all(genjet_clean_DR > 0.1, axis=1)
        is_clean_jet = te_os.isClean(genjet, gen_e, drmin=0.4) & te_os.isClean(genjet, gen_m, drmin=0.4) & te_os.isClean(genjet, gen_t, drmin=0.4) & te_os.isClean(genjet, gen_p, drmin=0.1)
        is_clean_ph = te_os.isClean(gen_p, gen_e, drmin=0.1) & te_os.isClean(gen_p, gen_m, drmin=0.1)
        gen_p_clean = gen_p#[is_clean_ph & (np.abs(gen_p.eta)<2.4)]
        genjet_clean = genjet#[is_clean_jet & (np.abs(genjet.eta)<2.4)]
        genbjet_clean = genjet#[np.abs(genjet.partonFlavour)==5]
        njets = ak.num(genjet_clean)
        nbjets = ak.num(genbjet_clean)
        atleast_1ph = ak.num(gen_p_clean)>0
        exactly_1ph = ak.num(gen_p_clean)==1


        selections.add("2los_CRtt", is2lOS_em & (nbjets==2) & (~atleast_1ph)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        #selections.add("incl", no_colinear) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("incl", np.ones(len(events), dtype=bool)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("incl_l0pt", (ak.firsts(gen_l.pt)>25)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("incl_l0pt_dral0p4", (ak.firsts(gen_l.pt)>25) & ak.fill_none(ak.firsts(genlep_clean_DR)>0.4, False)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("incl_0p", ((events.LHE.Nc + events.LHE.Nuds + events.LHE.Nglu)==0)) # Events with no extra ME parton
        selections.add("incl_1p", ((events.LHE.Nc + events.LHE.Nuds + events.LHE.Nglu)==1)) # Events with 1 extra ME parton
        selections.add("incl_0p_dral0p4", (((events.LHE.Nc + events.LHE.Nuds + events.LHE.Nglu)==0) & ak.fill_none(ak.firsts(genlep_clean_DR)>0.4, False))) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("incl_1p_dral0p4", (((events.LHE.Nc + events.LHE.Nuds + events.LHE.Nglu)==1) & ak.fill_none(ak.firsts(genlep_clean_DR)>0.4, False))) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("incl_dral0p1", (ak.firsts(genlep_clean_DR)>0.1)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("incl_draj_dral0p1", (ak.firsts(genlep_clean_DR)>0.1) & (ak.firsts(genjet_clean_DR)>0.1)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("incl_draj_dral0p4", (ak.firsts(genlep_clean_DR)>0.4) & (ak.firsts(genjet_clean_DR)>0.4)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("incl_draj0p5_dral0p4", (ak.firsts(genlep_clean_DR)>0.4) & (ak.firsts(genjet_clean_DR)>0.5)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        #selections.add("2los_CRtt", (events.is2l_nozeeveto & charge2l_0 & events.is_em & bmask_exactly2med & pass_trg)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        #selections.add("2los_ph", is2lOS & (nbjets==2) & (ak.firsts(genjet_clean.pt)>30) & (ak.num(gen_p[~((np.abs(gen_p.distinctParent.pdgId)==11) | (np.abs(gen_p.distinctParent.pdgId)==13))])==1) & (ak.firsts(gen_p[~((np.abs(gen_p.distinctParent.pdgId)==11) | (np.abs(gen_p.distinctParent.pdgId)==13))]).pt>20)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("2lss", is2lSS & (nbjets>=1) & (ak.firsts(genjet_clean.pt)>30)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("2los_ph", is2lOS & (nbjets==2) & (exactly_1ph) & (ak.firsts(gen_p).pt>20)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        #selections.add("2los_ph", is2lOS & (nbjets==2) & (atleast_1ph) & (ak.firsts(gen_p).pt>20)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("2los_ph_dral0p1", is2lOS & (ak.firsts(genlep_clean_DR)>0.1) & (nbjets==2) & (atleast_1ph) & (ak.firsts(gen_p).pt>20)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("2los_ph_dral0p4", is2lOS & (ak.firsts(genlep_clean_DR)>0.4) & (nbjets==2) & (atleast_1ph) & (ak.firsts(gen_p).pt>20)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        #selections.add("2los_ph", is2lOS & (nbjets>=2) & (ak.firsts(genjet_clean.pt)>30) & (ak.num(gen_p)==1) & (ak.firsts(gen_p).pt>20) & no_colinear) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        #selections.add("2los_ph", is2lOS & (nbjets==2) & (ak.firsts(genjet_clean.pt)>30) & (ak.num(gen_p)==1) & (ak.firsts(gen_p).pt>20)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("2los_ph_l", is2lOS & (nbjets==2) & (ak.firsts(genjet_clean.pt)>30) & (ak.num(gen_p_l[(np.abs(gen_p_l.distinctParent.pdgId)==11) | (np.abs(gen_p_l.distinctParent.pdgId)==13)])==1) & (ak.firsts(gen_p_l[(np.abs(gen_p_l.distinctParent.pdgId)==11) | (np.abs(gen_p_l.distinctParent.pdgId)==13)]).pt>20)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        #selections.add("2los_sf_ph", (retainedbyOverlap & events.is2l & charge2l_0 & (events.is_ee | events.is_mm) & ~sfosz_2los_ttg_mask & events.mask_SF_Zllgamma & bmask_atleast1med & pass_trg & exactly_1ph))
        sfosz_2los_ttg_mask = tc_es.get_Z_peak_mask(gen_l[:,0:2],pt_window=15.0)
        Zllgamma_SF_mask = (abs( (gen_l[:,0] + gen_l[:,1] + gen_p[:,0]).mass -91.2) > 15)
        selections.add("2los_sf_Zg_CR_ULttg", (is2lOS_em & ~sfosz_2los_ttg_mask & Zllgamma_SF_mask & (nbjets==2) & atleast_1ph))
        selections.add("3l", is3l & (nbjets>=0) & (ak.firsts(genjet_clean.pt)>30)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        #selections.add("2los_sf_Zg_CR_ULttg", (retainedbyOverlap & events.is2l_nozeeveto & charge2l_0 & (events.is_ee | events.is_mm) & ~sfosz_2los_ttg_mask & ~events.mask_SF_Zllgamma & bmask_atleast1med & pass_trg & atleast_1ph))
        #skim_cut = "'nMuon+nElectron >=2 && Sum$( Muon_looseId && Muon_miniPFRelIso_all < 0.4 && Muon_sip3d <8) + Sum$(Electron_miniPFRelIso_all < 0.4 && Electron_sip3d <8 && Electron_mvaFall17V2noIso_WPL) >=2'"


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
        selections.add("isSR_2lSS",    is2lSS)#( events.is2l_SR) & charge2l_0)
        selections.add("isSR_2lOS",    is2lOS)#( events.is2l_SR) & charge2l_0)
        selections.add("isSR_3l",      is3l)#(~events.is2l_SR) & charge2l_0)
        selections.add("isAR_2lSS",    is2lSS)#(~events.is2l_SR) & charge2l_0)
        selections.add("isAR_2lOS",    is2lOS)#(~events.is2l_SR) & charge2l_0)
        selections.add("isAR_3l",      is3l)#(~events.is2l_SR) & charge2l_0)
        selections.add("isAR_incl",    np.ones(len(events), dtype=bool))#(~events.is2l_SR) & charge2l_0)

        #selections.add("isSR_3l",  events.is3l_SR)
        #selections.add("isAR_3l", ~events.is3l_SR)
        #selections.add("isSR_4l",  events.is4l_SR)


        ### Get dense axis variables ###

        ht = ak.sum(genjet.pt,axis=-1)
        ht_clean = ak.sum(genjet_clean.pt,axis=-1)

        tops_pt = gen_top.sum().pt
        tX      = gen_top.sum() + gen_bos.sum()

        # Pt of the t(t)X system
        #tX_system = ak.concatenate([gen_top,gen_l_from_zg],axis=1)
        #tX_pt = tX_system.sum().pt

        # Invmass distributions
        #mll_e_from_zg = (gen_e_from_zg[:,0] + gen_e_from_zg[:,1]).mass
        #mll_m_from_zg = (gen_m_from_zg[:,0] + gen_m_from_zg[:,1]).mass
        #mll_t_from_zg = (gen_t_from_zg[:,0] + gen_t_from_zg[:,1]).mass
        mll_l0l1 = (gen_l[:,0] + gen_l[:,1]).mass

        sfos_mask = gen_l[:,0].pdgId == -gen_l[:,1].pdgId
        mll_l0l1 = ak.where(sfos_mask, mll_l0l1, ak.ones_like(mll_l0l1)*-1)
        sfos30 = (gen_l[:,0] + gen_l[:,1]).mass >= 30

        selections.add("incl_sfos30", sfos30) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("incl_sfos30_dral0p4", sfos30 & (ak.firsts(genlep_clean_DR)>0.4)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("2los_ph_sfos30", is2lOS & sfos30 & (nbjets==2) & (atleast_1ph) & (ak.firsts(gen_p).pt>20)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("2los_ph_sfos30_dral0p4", is2lOS & sfos30 & (ak.firsts(genlep_clean_DR)>0.4) & (nbjets==2) & (atleast_1ph) & (ak.firsts(gen_p).pt>20)) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("incl_0p_sfos30_dral0p4", (sfos30 & ((events.LHE.Nc + events.LHE.Nuds + events.LHE.Nglu)==0) & ak.fill_none(ak.firsts(genlep_clean_DR)>0.4, False))) # Explicitly add the em requirement here, so we don't have to rely on running with _split_by_lepton_flavor turned on to enforce this requirement
        selections.add("ee",  ((np.abs(gen_l[:,0].pdgId)==11) & (np.abs(gen_l[:,1].pdgId)==11)))
        selections.add("em",  (((np.abs(gen_l[:,0].pdgId)==11) & (np.abs(gen_l[:,1].pdgId)==13)) | ((np.abs(gen_l[:,0].pdgId)==13) & (np.abs(gen_l[:,1].pdgId)==11))))
        selections.add("mm",  ((np.abs(gen_l[:,0].pdgId)==13) & (np.abs(gen_l[:,1].pdgId)==13)))

        gen_p_pt = ak.fill_none(ak.firsts(gen_p).pt, -1)
        gen_p_eta = ak.fill_none(ak.firsts(gen_p).eta, -999)

        # Dictionary of dense axis values
        dense_axis_dict = {
            "photon_pt" : gen_p_pt,
            #"photon_pt" : ph_pt,
            #"photon_pt" : ak.Array(gen_p_smear),
            #"photon_l_pt" : ak.fill_none(ak.firsts(gen_p_l).pt, -1),
            "photon_eta" : gen_p_eta,
            "mll"  : ak.fill_none(mll_l0l1, -1),
            "invm" : ak.fill_none(tX.mass, -1),
            "t_pt" : ak.fill_none(ak.firsts(gen_top).pt, -1),
            "lhe_t_pt" : lhe_top.pt,
            #"lhe_t_pt" : ak.fill_none(ak.firsts(lhe_top.pt), -1),
            #"lhe_l0pt" : lhe_l,
            #"lhe_photon_pt" : lhe_ph.pt,
            "l0pt" : ak.fill_none(ak.firsts(gen_l.pt), -1),
            "j0pt" : ak.fill_none(ak.firsts(genjet_clean.pt), -1),
            "lj0pt" : ak.fill_none(ak.firsts(genjet_clean[np.abs(genjet_clean.partonFlavour)!=5].pt), -1),
            "bj0pt" : ak.fill_none(ak.firsts(genbjet_clean.pt), -1),
            "draj" : ak.fill_none(ak.firsts(genjet_clean_DR), -1),
            "dral" : ak.fill_none(ak.firsts(genlep_clean_DR), -1),
            "dral_sec" : ak.fill_none(ak.pad_none(genlep_clean_DR, 2), -1)[:,1],
            "njets" : njets,
        }
        cat_dict = {}
        sr_cat_dict = {
            "2los_ph" : {
                "atleast_1j"   : {
                    "lep_chan_lst" : ["2los_ph", "2los_ph_l"],
                    "lep_flav_lst" : ["ee", "em", "mm"],# no split so only run once, "mm", "em"],
                    "appl_lst"     : ["isSR_2lOS", "isAR_2lOS"],
                },
                "exactly_2j"   : {
                    "lep_chan_lst" : ["2los_ph", "2los_ph_l"],
                    "lep_flav_lst" : ["ee", "em", "mm"],# no split so only run once, "mm", "em"],
                    "appl_lst"     : ["isSR_2lOS", "isAR_2lOS"],
                },
                "atleast_3j"   : {
                    "lep_chan_lst" : ["2los_ph", "2los_ph_l"],
                    "lep_flav_lst" : ["ee", "em", "mm"],# no split so only run once, "mm", "em"],
                    "appl_lst"     : ["isSR_2lOS", "isAR_2lOS"],
                },
            },
            "2lss" : {
                "exactly_2j"   : {
                    "lep_chan_lst" : ["2lss"],
                    "lep_flav_lst" : ["ee", "em", "mm"],# no split so only run once, "mm", "em"],
                    "appl_lst"     : ["isSR_2lSS", "isAR_2lSS"],
                },
                "atleast_3j"   : {
                    "lep_chan_lst" : ["2lss"],
                    "lep_flav_lst" : ["ee", "em", "mm"],# no split so only run once, "mm", "em"],
                    "appl_lst"     : ["isSR_2lSS", "isAR_2lSS"],
                },
            },
            "3l" : {
                "exactly_2j"   : {
                    "lep_chan_lst" : ["3l"],
                    "lep_flav_lst" : ["ee"],# no split so only run once, "mm", "em"],
                    "appl_lst"     : ["isSR_3l", "isAR_3l"],
                },
                "atleast_1j"   : {
                    "lep_chan_lst" : ["3l"],
                    "lep_flav_lst" : ["ee"],# no split so only run once, "mm", "em"],
                    "appl_lst"     : ["isSR_3l", "isAR_3l"],
                },
            },
            "incl": {
                "atleast_0j" : {
                    "lep_chan_lst" : ["incl", "incl_sfos30", "incl_draj_dral0p4", "incl_draj0p5_dral0p4", "incl_sfos30_dral0p4", "2los_ph", "2los_ph_dral0p4", "2los_ph_sfos30", "2los_ph_sfos30_dral0p4", "incl_0p", "incl_1p", "incl_0p_dral0p4", "incl_1p_dral0p4", "incl_0p_sfos30_dral0p4", "incl_l0pt", "incl_l0pt_dral0p4"],
                    #"lep_flav_lst" : ["em"],
                    "lep_flav_lst" : ["em", "ee", "mm"],
                    "appl_lst"     : ["isAR_incl", "isSR_2lOS"],
                },
                "atleast_1j" : {
                    "lep_chan_lst" : ["2los_ph", "2los_ph_dral0p4"],
                    #"lep_flav_lst" : ["em"],
                    "lep_flav_lst" : ["em", "ee", "mm"],
                    "appl_lst"     : ["isSR_2lOS"],
                }
            }
        }
        #sr_cat_dict = {k:v for k,v in sr_cat_dict.items() if 'incl' in k}
        cr_cat_dict = {
            "2los_CRtt" : {
                "atmost_3j"   : {
                    "lep_chan_lst" : ["2los_ph"],
                    #"lep_flav_lst" : ["em"],# no split so only run once, "ee", "mm"],
                    "lep_flav_lst" : ["em", "ee", "mm"],
                    "appl_lst"     : ["isSR_2lOS" , "isAR_2lOS"],
                },
            },
            "2los_CR_Zg_ULttg" : {
                "atleast_2j"   : {
                    "lep_chan_lst" : ["2los_sf_Zg_CR_ULttg"],
                    #"lep_flav_lst" : ["ee"],# no split so only run once, "mm"],
                    "lep_flav_lst" : ["em", "ee", "mm"],
                    "appl_lst"     : ["isSR_2lOS", "isAR_2lOS"],
                },
            },
        }


        ### Get weights ###

        # Extract the EFT quadratic coefficients and optionally use them to calculate the coefficients on the w**2 quartic function
        # eft_coeffs is never Jagged so convert immediately to numpy for ease of use.
        eft_coeffs = None
        if hasattr(events, "EFTfitCoefficients") or post_mortem:
            eft_coeffs = ak.to_numpy(events["EFTfitCoefficients"])
            eft_coeffs_fit = ak.to_numpy(events["EFTfitCoefficients"])
            #print('EFTfit', eft_coeffs)
            if post_mortem:
                eft_coeffs = ak.to_numpy(events["EFTPostMortem"])
                LHEWeight_originalXWGTUP = ak.broadcast_arrays(
                    events.LHEWeight.originalXWGTUP[:, None],
                    eft_coeffs
                )[0]
                #eft_coeffs = eft_coeffs/LHEWeight_originalXWGTUP
                eft_coeffs = eft_coeffs*LHEWeight_originalXWGTUP
                #print('PM', eft_coeffs, '\n\n\n')
                #print(eft_coeffs/eft_coeffs_fit, '\n\n\n')
                #print('SM', (eft_coeffs/eft_coeffs_fit)[:,0], '\n\n\n')
                #print('SM>1e-2', (eft_coeffs/eft_coeffs_fit)[:,0][(eft_coeffs/eft_coeffs_fit)[:,0]>1e-2], '\n\n\n')
                #print('SM>1e-2', ak.num((eft_coeffs/eft_coeffs_fit)[:,0][(eft_coeffs/eft_coeffs_fit)[:,0]>1e-2], axis=0), '\n\n\n')
                #print('SM<1e-2', ak.num((eft_coeffs/eft_coeffs_fit)[:,0][(eft_coeffs/eft_coeffs_fit)[:,0]<1e-2], axis=0), '\n\n\n')

        # Hack to change 11 WCs to 26
        #full_list = ["cpt", "ctp", "ctt1", "cptb", "ctG", "cQq11", "cQl3i", "ctlSi", "ctq8", "ctZ", "cQq83", "ctlTi", "ctq1", "cpQM", "cQq13", "cQt1", "cbW", "ctli", "cQt8", "ctei", "cQq81", "cQlMi", "cQQ1", "cpQ3", "cQei", "ctW"]
        #my_list = ["cQq11", "ctq8", "ctG", "ctW", "cpt", "ctq1", "cpQM", "cQq83", "cQq81", "cQq13", "ctZ"]

        if eft_coeffs is not None:
            # Check to see if the ordering of WCs for this sample matches what want
            if post_mortem and self._samples[dataset]["PMWCnames"] != self._wc_names_lst:
                eft_coeffs = efth.remap_coeffs(self._samples[dataset]["PMWCnames"], self._wc_names_lst, eft_coeffs)
            elif self._samples[dataset]["WCnames"] != self._wc_names_lst and not post_mortem:
                eft_coeffs = efth.remap_coeffs(self._samples[dataset]["WCnames"], self._wc_names_lst, eft_coeffs)

            # Hack to change 11 WCs to 26
            #if my_list != full_list:
            #    eft_coeffs = efth.remap_coeffs(my_list, full_list, eft_coeffs)

        #eft_coeffs = None

        eft_w2_coeffs = efth.calc_w2_coeffs(eft_coeffs,self._dtype) if (self._do_errors and eft_coeffs is not None) else None

        # If this is not an eft sample, get the genWeight
        if eft_coeffs is None: genw = events["genWeight"]
        else: genw = np.ones_like(events["event"])
        lumi = 1000.0*get_tc_param(f"lumi_{year}")
        event_weight = lumi*xsec*genw/sow
        #event_weight = genw/sow
        #print(event_weight, '\n\n\n')

        weights_dict = {}
        # For both data and MC
        weights_obj_base = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
        weights_obj_base.add("norm",event_weight)

        #tc_cor.AttachScaleWeights(events)

        # renorm/fact scale
        events.nom = np.ones(len(events))
        #weights_obj_base.add('renorm', events.nom, events.renormUp*(sow/sow_renormUp), events.renormDown*(sow/sow_renormDown))
        #weights_obj_base.add('fact', events.nom, events.factUp*(sow/sow_factUp), events.factDown*(sow/sow_factDown))

        for ch_name in ["2los_CRtt", "2lss", "2los_ph", "2los_ph_l", "2los_CR_Zg_ULttg", "3l", "incl"]:
            weights_dict[ch_name] = copy.deepcopy(weights_obj_base)
            #weights_dict[ch_name].add("phoShape", copy.deepcopy(photon_corr_lookup(gen_p_pt)), copy.deepcopy(photon_corr_lookup(gen_p_pt)), copy.deepcopy(photon_corr_lookup(gen_p_pt)))
            #weights_dict[ch_name].add("phoSF", ak.ones_like(ak.firsts(gen_p).pt), np.random.rand(*ak.to_numpy(ak.firsts(gen_p).pt).shape), np.random.rand(*ak.to_numpy(ak.firsts(gen_p).pt).shape))
            #weights_dict[ch_name].add("CMS_eff_g", ak.ones_like(ak.firsts(gen_p).pt), np.random.rand(*ak.to_numpy(ak.firsts(gen_p).pt).shape), np.random.rand(*ak.to_numpy(ak.firsts(gen_p).pt).shape))

        # Example of reweighting based on Ht
        #if "private" in histAxisName:
        #    ht_sf = get_ht_sf(ht,histAxisName)
        #    event_weight = event_weight*ht_sf


        ### Loop over the hists we want to fill ###

        hout = self.accumulator
        counts = np.ones_like(events['event'])
        #hout["SumOfWeights_eft"].fill(process=histAxisName, SumOfWeights=counts, weight=genw, eft_coeff=eft_coeffs)
        #hout["SumOfWeights"].fill(process=histAxisName, SumOfWeights=counts, weight=genw, eft_coeff=None)

        for dense_axis_name, dense_axis_vals in dense_axis_dict.items():
            if dense_axis_name not in self._hist_lst or '_cnt' in dense_axis_name:
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
                    #if 'photon_pt' in dense_axis_name:
                    #    weight = weight * photon_corr_lookup(dense_axis_vals)
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
                                    if 'incl' in nlep_cat:
                                        cuts_lst = [lep_chan]
                                    if self._split_by_lepton_flavor:
                                        flav_ch = lep_flav
                                        cuts_lst.append(lep_flav)
                                    if dense_axis_name == "njets":
                                        all_cuts_mask = (selections.all(*cuts_lst) & njets_any_mask)
                                    else:
                                        njet_ch = njet_val
                                        all_cuts_mask = (selections.all(*cuts_lst))
                                        #print(f'{nlep_cat=} {njet_val=} {appl=} {lep_flav=} {cuts_lst=}')
                                    ch_name = construct_cat_name(lep_chan,nlep_cat,njet_str=njet_ch,flav_str=flav_ch)
                                    #if 'incl' in nlep_cat:
                                    #    ch_name = nlep_cat
                                    #    if self._split_by_lepton_flavor:
                                    #        ch_name += '_' + flav_ch
                                    #    if 'incl' in ch_name:
                                    #        ch_name += '_' + njet_ch

                                    # Mask out the none values
                                    isnotnone_mask = (ak.fill_none((dense_axis_vals != None),False))
                                    isnotnone_mask = isnotnone_mask & all_cuts_mask
                                    dense_axis_vals_cut = dense_axis_vals[isnotnone_mask]
                                    event_weight_cut = weight[isnotnone_mask]
                                    dense_axis_vals_cut = dense_axis_vals[all_cuts_mask]#[isnotnone_mask]
                                    event_weight_cut = weight[all_cuts_mask]#[isnotnone_mask]
                                    #eft_coeffs_cut = eft_coeffs[all_cuts_mask]
                                    eft_coeffs_cut = eft_coeffs
                                    if eft_coeffs is not None: eft_coeffs_cut = eft_coeffs[all_cuts_mask]
                                    #if eft_coeffs is not None: eft_coeffs_cut = eft_coeffs[isnotnone_mask]
                                    #eft_w2_coeffs_cut = eft_w2_coeffs
                                    #if eft_w2_coeffs is not None: eft_w2_coeffs_cut = eft_w2_coeffs[isnotnone_mask]

                                    # Fill the histos
                                        #dense_axis_name.replace('photon_l_pt', 'photon_pt') : dense_axis_vals_cut,
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

                                    axes_fill_info_dict[dense_axis_name] = gen_p_pt[all_cuts_mask]
                                    if 'photon_pt' in dense_axis_name and 'lhe' not in dense_axis_name: hout[dense_axis_name+'_gen'].fill(**axes_fill_info_dict)

                                    axes_fill_info_dict['weight'] = np.ones_like(event_weight_cut)
                                    #axes_fill_info_dict['eft_coeff'] = None

                                    # Do not loop over lep flavors if not self._split_by_lepton_flavor, it's a waste of time and also we'd fill the hists too many times
                                    if not self._split_by_lepton_flavor: break
        return hout

    def postprocess(self, accumulator):
        return accumulator


if __name__ == '__main__':
    raise Exception('Please use `run_gen_analysis.py` to run this processor!')
