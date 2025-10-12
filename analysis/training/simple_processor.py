#!/usr/bin/env python
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import hist, processor
from coffea.analysis_tools import PackedSelection
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from topcoffea.modules.objects import *
from topcoffea.modules.selection import *
from topcoffea.modules.HistEFT import HistEFT
import topcoffea.modules.eft_helper as efth


@dataclass
class ProcessingContext:
    """Container carrying shared state between processing stages."""

    events: Any
    dataset: str
    year: str
    xsec: float
    sow: float
    is_data: bool
    is_eft: bool
    eft_coeffs: Optional[np.ndarray] = None
    eft_w2_coeffs: Optional[np.ndarray] = None
    selections: Optional[PackedSelection] = None
    varnames: Dict[str, Any] = field(default_factory=dict)
    hist_output: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

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
        """Execute the full processing pipeline for a dataset."""

        context = self._prepare_context(events)
        self._prepare_collections(context)
        self._build_selections(context)
        self._compute_weights(context)
        self._fill_histograms(context)
        return context.hist_output

    def _prepare_context(self, events) -> ProcessingContext:
        """Collect dataset-level metadata and EFT coefficients."""

        dataset_key = events.metadata['dataset']
        sample_info = self._samples[dataset_key]
        dataset = dataset_key
        year = sample_info['year']
        xsec = sample_info['xsec']
        sow = sample_info['nSumOfWeights']
        is_data = sample_info['isData']
        datasets = ['SingleMuon', 'SingleElectron', 'EGamma', 'MuonEG', 'DoubleMuon', 'DoubleElectron']
        for d in datasets:
            if d in dataset:
                dataset = dataset.split('_')[0]

        eft_coeffs = ak.to_numpy(events['EFTfitCoefficients']) if hasattr(events, "EFTfitCoefficients") else None
        if eft_coeffs is not None:
            if sample_info['WCnames'] != self._wc_names_lst:
                eft_coeffs = efth.remap_coeffs(sample_info['WCnames'], self._wc_names_lst, eft_coeffs)
        eft_w2_coeffs = efth.calc_w2_coeffs(eft_coeffs, self._dtype) if (self._do_errors and eft_coeffs is not None) else None

        context = ProcessingContext(
            events=events,
            dataset=dataset,
            year=year,
            xsec=xsec,
            sow=sow,
            is_data=is_data,
            is_eft=sample_info['WCnames'] != [],
            eft_coeffs=eft_coeffs,
            eft_w2_coeffs=eft_w2_coeffs,
            hist_output=self.accumulator.identity(),
        )
        context.metadata['dataset_key'] = dataset_key
        return context

    def _prepare_collections(self, context: ProcessingContext) -> None:
        """Derive physics object collections and basic kinematic quantities."""

        events = context.events

        e = events.GenPart[abs(events.GenPart.pdgId) == 11]
        m = events.GenPart[abs(events.GenPart.pdgId) == 13]
        tau = events.GenPart[abs(events.GenPart.pdgId) == 15]
        j = events.GenJet

        run = events.run
        luminosityBlock = events.luminosityBlock
        event = events.event

        print("\n\nInfo about events:")
        print("\trun:", run)
        print("\tluminosityBlock:", luminosityBlock)
        print("\tevent:", event)

        print("\nLeptons before selection:")
        print("\te pt", e.pt)
        print("\te eta", e.eta)
        print("\tm pt", m.pt)
        print("\tm eta", m.eta)

        e_selec = ((e.pt > 15) & (abs(e.eta) < 2.5))
        m_selec = ((m.pt > 15) & (abs(m.eta) < 2.5))
        e = e[e_selec]
        m = m[m_selec]

        l = ak.concatenate([e, m], axis=1)

        n_e = ak.num(e)
        n_m = ak.num(m)
        n_l = ak.num(l)

        at_least_two_leps = (n_l >= 2)

        e0 = e[ak.argmax(e.pt, axis=-1, keepdims=True)]
        m0 = m[ak.argmax(m.pt, axis=-1, keepdims=True)]
        l0 = l[ak.argmax(l.pt, axis=-1, keepdims=True)]

        print("\nLeptons after selection:")
        print("\te pt", e.pt)
        print("\tm pt", m.pt)
        print("\tl pt:", l.pt)
        print("\tn e", n_e)
        print("\tn m", n_m)
        print("\tn l", n_l)

        print("\nMask for at least two lep:", at_least_two_leps)

        print("\nLeading lepton info:")
        print("\te0", e0.pt)
        print("\tm0", m0.pt)
        print("\tl0", l0.pt)

        print("\nJet info:")
        print("\tjpt before selection", j.pt)

        j_selec = ((j.pt > 30) & (abs(j.eta) < 2.5))
        print("\tjselect", j_selec)

        j = j[j_selec]
        print("\tjpt", j.pt)

        j['isClean'] = isClean(j, e, drmin=0.4) & isClean(j, m, drmin=0.4)
        j_isclean = isClean(j, e, drmin=0.4) & isClean(j, m, drmin=0.4)
        print("\tj is clean", j_isclean)

        j = j[j_isclean]
        print("\tclean jets pt", j.pt)

        n_j = ak.num(j)
        print("\tn_j", n_j)
        j0 = j[ak.argmax(j.pt, axis=-1, keepdims=True)]

        print("\tj0pt", j0.pt)

        at_least_two_jets = (n_j >= 2)
        print("\tat_least_two_jets", at_least_two_jets)

        context.e = e
        context.m = m
        context.tau = tau
        context.j = j
        context.l = l
        context.n_e = n_e
        context.n_m = n_m
        context.n_l = n_l
        context.at_least_two_leps = at_least_two_leps
        context.e0 = e0
        context.m0 = m0
        context.l0 = l0
        context.n_j = n_j
        context.j0 = j0
        context.at_least_two_jets = at_least_two_jets

    def _build_selections(self, context: ProcessingContext) -> None:
        """Construct event selections and store them in the context."""

        event_selec = (context.at_least_two_leps & context.at_least_two_jets)
        print("\nEvent selection:", event_selec, "\n")

        selections = PackedSelection()
        selections.add('2l2j', event_selec)

        context.selections = selections
        context.event_selection = event_selec

    def _compute_weights(self, context: ProcessingContext) -> None:
        """Populate variables to be histogrammed after selections."""

        events = context.events
        varnames: Dict[str, Any] = {}
        varnames['counts'] = np.ones_like(events.MET.pt)
        varnames['njets'] = context.n_j
        varnames['j0pt'] = context.j0.pt
        varnames['j0eta'] = context.j0.eta
        varnames['l0pt'] = context.l0.pt

        context.varnames = varnames

    def _fill_histograms(self, context: ProcessingContext) -> None:
        """Fill histograms using the prepared selections and variables."""

        print("\nFilling hists now...\n")
        selections = context.selections
        varnames = context.varnames
        eft_coeffs = context.eft_coeffs
        eft_w2_coeffs = context.eft_w2_coeffs
        dataset = context.dataset
        hout = context.hist_output

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

        context.hist_output = hout

    def postprocess(self, accumulator):
        return accumulator

