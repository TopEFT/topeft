#!/usr/bin/env python
import lz4.frame as lz4f
import cloudpickle
import json
import pprint
import numpy as np
import awkward
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea.arrays import Initialize
from coffea import hist, processor
from coffea.util import load, save
from optparse import OptionParser

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, objects, selection, corrections, functions, columns):
        self._samples = samples
        self._columns = columns
        self._objects = objects
        self._selection = selection
        self._corrections = corrections
        self._functions = functions

        # Object variables
        self._e   = {}
        self._mu  = {}
        self._jet = {}

        self._e['id' ]  = 'Electron_cutBased'
        self._e['dxy']  = 'Electron_dxy'
        self._e['dz' ]  = 'Electron_dz'

        self._mu['tight_id']  = 'Muon_tightId'
        self._mu['mediumId']  = 'Muon_mediumId'
        self._mu['dxy']  = 'Muon_dxy'
        self._mu['dz' ]  = 'Muon_dz'
        self._mu['iso']  = 'Muon_pfRelIso04_all'

        self._jet['id'] = 'Jet_jetId'

        # Create the histograms
        # 'name' : hist.Hist("Ytitle", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat("syst", "syst"), hist.Bin("name", "X axis (GeV)", 20, 0, 100)),
        self._accumulator = processor.dict_accumulator({
        'dummy'   : hist.Hist("Dummy", hist.Cat("sample", "sample"), hist.Bin("dummy", "Number of events", 1, 0, 1)),
        'lep0pt'  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Bin("lep0pt",  "Leading lepton $p_{T}$ (GeV)", 20, 0, 200)),
        'lep0eta' : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Bin("lep0eta", "Leading lepton $\eta$ ", 15, -2.5, 2.50)),
        'invmass' : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 20, 0, 200)),
        })

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, df):
        # Dataset parameters
        dataset = df['dataset']
        year   = self._samples[dataset]['year']
        xsec   = self._samples[dataset]['xsec']
        sow    = self._samples[dataset]['nSumOfWeights' ]
        isData = self._samples[dataset]['isData']

        ### Recover objects, selection, functions and others...
        # Objects
        isTightMuon     = self._objects['isTightMuon']
        isTightElectron = self._objects['isTightElectron']
        isGoodJet       = self._objects['isGoodJet']

        # Corrections
        GetMuonIsoSF    = self._corrections['getMuonIso']
        GetMuonIDSF     = self._corrections['getMuonID' ]

        # Selection
        passNJets  = self._selection['passNJets']
        passMETcut = self._selection['passMETcut']

        # Functions
        pow2 = self._functions['pow2']

        # Initialize objects
        met = Initialize({'pt' : df['MET_pt'],      'eta' : 0                 , 'phi' : df['MET_phi'],      'mass': 0                  })
        e   = Initialize({'pt' : df['Electron_pt'], 'eta' : df['Electron_eta'], 'phi' : df['Electron_phi'], 'mass': df['Electron_mass']})
        mu  = Initialize({'pt' : df['Muon_pt'],     'eta' : df['Muon_eta'],     'phi' : df['Muon_phi'],     'mass': df['Muon_mass']    })    
        j   = Initialize({'pt' : df['Jet_pt'],      'eta' : df['Jet_eta'],      'phi' : df['Jet_phi'],      'mass' : df['Jet_mass']})


        # Electron selection
        for key in self._e:
            e[key] = e.pt.zeros_like()
            if self._e[key] in df:
                e[key] = df[self._e[key]]
        e['istight'] = isTightElectron(e.pt, e.eta, e.dxy, e.dz, e.id, year)
        leading_e = e[e.pt.argmax()]
        leading_e = leading_e[leading_e.istight.astype(bool)]
        nElec = e.counts

        # Muon selection
        for key in self._mu:
            mu[key] = mu.pt.zeros_like()
            if self._mu[key] in df:
                mu[key] = df[self._mu[key]]
        mu['istight'] = isTightMuon(mu.pt, mu.eta, mu.dxy, mu.dz, mu.iso, mu.tight_id, year)
        leading_mu = mu[mu.pt.argmax()]
        leading_mu = leading_mu[leading_mu.istight.astype(bool)]
        nMuon = mu.counts

        # Jet selection
        j['deepcsv'] = df['Jet_btagDeepB']
        j['deepflv'] = df['Jet_btagDeepFlavB']
        for key in self._jet:
                j[key] = j.pt.zeros_like()
                if self._jet[key] in df:
                    j[key] = df[self._jet[key]]

        j['isgood']  = isGoodJet(j.pt, j.eta, j.id)
        j['isclean'] = ~j.match(e,0.4) & ~j.match(mu,0.4) & j.isgood.astype(bool)
        j0 = j[j.pt.argmax()]
        j0 = j0[j0.isclean.astype(bool)]
        nJets = j.counts

        # Dilepton pair
        ele_pairs = e.distincts()
        diele = leading_e
        leading_diele = leading_e
        if ele_pairs.i0.content.size>0:
            diele = ele_pairs.i0+ele_pairs.i1
            leading_diele = diele[diele.pt.argmax()]

        mu_pairs = mu.distincts()
        dimu = leading_mu
        leading_dimu = leading_mu
        if mu_pairs.i0.content.size>0:
            dimu = mu_pairs.i0+mu_pairs.i1
            leading_dimu = dimu[dimu.pt.argmax()]
        mmumu = leading_dimu.mass

        # Triggers

        # MET filters

        # Weights
        genw = np.ones_like(df['MET_pt']) if isData else df['genWeight']
        weights = processor.Weights(df.size)
        weights.add('norm',xsec/sow*genw)

        # Selections and cuts
        selections = processor.PackedSelection()
        channels = ['em', 'mm', 'ee']
        selections.add('em', (nElec==1)&(nMuon==1) )
        selections.add('ee', (nElec>=2))
        selections.add('mm', (nMuon>=2))

        levels = ['dilepton', '2jets']
        selections.add('dilepton', (nElec>=2)|(nMuon>=2)|((nElec+nMuon)>=2))
        selections.add('2jets', (nJets>=2))

        # Variables

        # Fill Histos
        hout = self.accumulator.identity()
        hout['dummy'].fill(sample=dataset, dummy=1, weight=df.size)

        for ch in channels:
          for lev in levels:
            weight = weights.weight()
            cuts = [ch] + [lev]
            cut = selections.all(*cuts)
            invmass_flat = mmumu[cut].flatten()
            weights_flat = (~np.isnan(mmumu[cut])*weight[cut]).flatten()

            hout['invmass'].fill(sample=dataset, channel=ch, level=lev, invmass=invmass_flat, weight=weights_flat)#*selections.all(*{'mm'})
        #flat_variables = {k: v[cut].flatten() for k, v in variables.items()}
        #flat_weights = {k: (~np.isnan(v[cut])*weight[cut]).flatten() for k, v in variables.items()}


        #hout['invmass'].fill(sample=dataset, channel='mm', level="dilepton", invmass=mmumu, weight=np.ones_like(df['MET_pt']))#weight=weights.weight())#*selections.all(*{'mm'})
        
        return hout

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    # Pour the coffea
    outpath= './coffeaFiles/'
    samples     = load(outpath+'samples.coffea')
    objects     = load(outpath+'objects.coffea')
    selection   = load(outpath+'selection.coffea')
    corrections = load(outpath+'corrections.coffea')
    functions   = load(outpath+'functions.coffea')

    columns = ''' 
    MET_pt
    MET_phi
    Electron_pt
    Electron_eta
    Electron_phi
    Electron_mass
    Electron_cutBased
    Electron_dxy
    Electron_dz
    Muon_pt
    Muon_eta
    Muon_phi
    Muon_mass
    Muon_pfRelIso04_all
    Muon_tightId
    Muon_mediumId
    Muon_dxy
    Muon_dz
    Jet_pt
    Jet_eta
    Jet_phi
    Jet_mass
    Jet_btagDeepB
    Jet_btagDeepFlavB
    Jet_jetId
    Jet_neHEF
    Jet_neEmEF
    Jet_chHEF
    Jet_chEmEF
    GenPart_pt
    GenPart_eta
    GenPart_phi
    GenPart_mass
    GenPart_pdgId
    GenPart_status
    GenPart_statusFlags
    GenPart_genPartIdxMother
    PV_npvs
 
    '''.split()

    topprocessor = AnalysisProcessor(samples, objects, selection, corrections, functions, columns)
    save(topprocessor, outpath+'top.coffea')


