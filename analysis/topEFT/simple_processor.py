#!/usr/bin/env python
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import hist, processor
from coffea.analysis_tools import PackedSelection

from topcoffea.modules.objects import *
from topcoffea.modules.selection import *
from topcoffea.modules.HistEFT import HistEFT
import topcoffea.modules.eft_helper as efth

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, wc_names_lst=[], do_errors=False, do_systematics=False, dtype=np.float32):
        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        # Create the histograms
        # In general, histograms depend on 'sample', 'channel' (final state) and 'cut' (level of selection)
        self._accumulator = processor.dict_accumulator({
            'counts'  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("counts", "Counts", 1, 0, 2)),
            'njets'   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("njets",  "Jet multiplicity ", 12, 0, 12)),
            'j0pt'    : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("j0pt",   "Leading jet  $p_{T}$ (GeV)", 10, 0, 600)),
            'j0eta'   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("j0eta",  "Leading jet  $\eta$", 10, -3.0, 3.0)),
            'l0pt'    : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("l0pt",   "Leading lep $p_{T}$ (GeV)", 15, 0, 400)),
        })

        self._do_errors = do_errors # Whether to calculate and store the w**2 coefficients
        self._do_systematics = do_systematics # Whether to process systematic samples
        
    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, events):

        # Dataset parameters
        dataset = events.metadata['dataset']
        year   = self._samples[dataset]['year']
        xsec   = self._samples[dataset]['xsec']
        sow    = self._samples[dataset]['nSumOfWeights' ]
        isData = self._samples[dataset]['isData']
        datasets = ['SingleMuon', 'SingleElectron', 'EGamma', 'MuonEG', 'DoubleMuon', 'DoubleElectron']
        for d in datasets:
            if d in dataset: dataset = dataset.split('_')[0] 

        # Extract the EFT quadratic coefficients and optionally use them to calculate the coefficients on the w**2 quartic function
        # eft_coeffs is never Jagged so convert immediately to numpy for ease of use.
        eft_coeffs = ak.to_numpy(events['EFTfitCoefficients']) if hasattr(events, "EFTfitCoefficients") else None
        if eft_coeffs is not None:
            # Check to see if the ordering of WCs for this sample matches what want
            if self._samples[dataset]['WCnames'] != self._wc_names_lst:
                eft_coeffs = efth.remap_coeffs(self._samples[dataset]['WCnames'], self._wc_names_lst, eft_coeffs)
        eft_w2_coeffs = efth.calc_w2_coeffs(eft_coeffs,self._dtype) if (self._do_errors and eft_coeffs is not None) else None

        # Initialize objects (GEN objects)
        e = events.GenPart[abs(events.GenPart.pdgId)==11]
        m = events.GenPart[abs(events.GenPart.pdgId)==13]
        tau = events.GenPart[abs(events.GenPart.pdgId)==15]
        j = events.GenJet

        run = events.run
        luminosityBlock = events.luminosityBlock
        event = events.event

        print("\n\nInfo about events:")
        print("\trun:",run)
        print("\tluminosityBlock:",luminosityBlock)
        print("\tevent:",event)
 
        print("\nLeptons before selection:")
        print("\te pt",e.pt)
        print("\te eta",e.eta)
        print("\tm pt",m.pt)
        print("\tm eta",m.eta)

        ######## Lep selection  ########

        e_selec = ( (e.pt>15) & (abs(e.eta)<2.5) )
        m_selec = ( (m.pt>15) & (abs(m.eta)<2.5) )
        e = e[e_selec]
        m = m[m_selec]

        # Put the e and mu togheter
        l = ak.concatenate([e,m],axis=1)

        n_e = ak.num(e)
        n_m = ak.num(m)
        n_l = ak.num(l)

        at_least_two_leps = (n_l >= 2)

        e0 = e[ak.argmax(e.pt,axis=-1,keepdims=True)]
        m0 = m[ak.argmax(m.pt,axis=-1,keepdims=True)]
        l0 = l[ak.argmax(l.pt,axis=-1,keepdims=True)]

        print("\nLeptons after selection:")
        print("\te pt",e.pt)
        print("\tm pt",m.pt)
        print("\tl pt:",l.pt)
        print("\tn e", n_e)
        print("\tn m",n_m)
        print("\tn l",n_l)

        print("\nMask for at least two lep:",at_least_two_leps)

        print("\nLeading lepton info:")
        print("\te0",e0.pt)
        print("\tm0",m0.pt)
        print("\tl0",l0.pt)

        ######## Jet selection  ########

        print("\nJet info:")
        print("\tjpt before selection",j.pt)

        j_selec = ( (j.pt>30) & (abs(j.eta)<2.5) )
        print("\tjselect",j_selec)

        j = j[j_selec]
        print("\tjpt",j.pt)

        j['isClean'] = isClean(j, e, drmin=0.4)& isClean(j, m, drmin=0.4)
        j_isclean = isClean(j, e, drmin=0.4) & isClean(j, m, drmin=0.4)
        print("\tj is clean",j_isclean)

        j = j[j_isclean]
        print("\tclean jets pt",j.pt)

        n_j = ak.num(j)
        print("\tn_j",n_j)
        j0 = j[ak.argmax(j.pt,axis=-1,keepdims=True)]

        print("\tj0pt",j0.pt)

        at_least_two_jets = (n_j >= 2)
        print("\tat_least_two_jets",at_least_two_jets)

        ######## Selections and cuts ########

        event_selec = (at_least_two_leps & at_least_two_jets)
        print("\nEvent selection:",event_selec,"\n")

        selections = PackedSelection()
        selections.add('2l2j', event_selec)

        varnames = {}
        varnames['counts'] = np.ones_like(events.MET.pt)
        varnames['njets'] = n_j
        varnames['j0pt' ] = j0.pt
        varnames['j0eta'] = j0.eta
        varnames['l0pt' ] = l0.pt

        ######## Fill histos ########

        print("\nFilling hists now...\n")
        hout = self.accumulator.identity()
        for var, v in varnames.items():
            cut = selections.all("2l2j")
            values = v[cut]
            eft_coeffs_cut = eft_coeffs[cut] if eft_coeffs is not None else None
            eft_w2_coeffs_cut = eft_w2_coeffs[cut] if eft_w2_coeffs is not None else None
            if var == "counts":
                hout[var].fill(counts=values, sample=dataset, channel="2l", cut="2l")
            elif var == "njets":
                hout[var].fill(njets=values, sample=dataset, channel="2l", cut="2l", eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut)
            elif var == "j0pt":
                hout[var].fill(j0pt=values, sample=dataset, channel="2l", cut="2l", eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut)
            elif var == "j0eta":
                hout[var].fill(j0eta=values, sample=dataset, channel="2l", cut="2l", eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut)
            elif var == "l0pt":
                hout[var].fill(l0pt=values, sample=dataset, channel="2l", cut="2l", eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut)

        return hout

    def postprocess(self, accumulator):
        return accumulator

