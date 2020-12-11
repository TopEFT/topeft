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
import os, sys
basepath = os.path.abspath(__file__).rsplit('/topcoffea/',1)[0]+'/topcoffea/'
sys.path.append(basepath)
from modules.HistEFT import HistEFT

WCNames = ['ctW', 'ctp', 'cpQM', 'ctli', 'cQei', 'ctZ', 'cQlMi', 'cQl3i', 'ctG', 'ctlTi', 'cbW', 'cpQ3', 'ctei', 'cpt', 'ctlSi', 'cptb']

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, objects, selection, corrections):
        self._samples = samples
        self._objects = objects
        self._selection = selection
        self._corrections = corrections

        # Create the histograms
        # In general, histograms depend on 'sample', 'channel' (final state) and 'cut' (level of selection)
        self._accumulator = processor.dict_accumulator({
        'SumOfEFTweights'  : HistEFT("SumOfWeights", WCNames, hist.Cat("sample", "sample"), hist.Bin("SumOfEFTweights", "sow", 1, 0, 2)),
        'dummy'   : hist.Hist("Dummy" , hist.Cat("sample", "sample"), hist.Bin("dummy", "Number of events", 1, 0, 1)),
        'counts'  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("counts", "Counts", 1, 0, 2)),
        'invmass' : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut","cut"), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 20, 0, 200)),
        'njets'   : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("njets",  "Jet multiplicitu ", 10, 0, 10)),
        'nbtags'  : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("nbtags", "btag multiplicitu ", 5, 0, 5)),
        'met'     : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("met",    "MET (GeV)", 40, 0, 400)),
        'm3l'     : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("m3l",    "$m_{3\ell}$ (GeV) ", 20, 0, 200)),
        'wleppt'  : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("wleppt", "$p_{T}^{lepW}$ (GeV) ", 20, 0, 200)),
        'e0pt'    : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("e0pt",   "Leading elec $p_{T}$ (GeV)", 30, 0, 300)),
        'm0pt'    : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("m0pt",   "Leading muon $p_{T}$ (GeV)", 30, 0, 300)),
        'j0pt'    : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("j0pt",   "Leading jet  $p_{T}$ (GeV)", 20, 0, 400)),
        'e0eta'   : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("e0eta",  "Leading elec $\eta$", 20, -2.5, 2.5)),
        'm0eta'   : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("m0eta",  "Leading muon $\eta$", 20, -2.5, 2.5)),
        'j0eta'   : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("j0eta",  "Leading jet  $\eta$", 20, -2.5, 2.5)),
        'ht'      : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("ht",     "H$_{T}$ (GeV)", 40, 0, 800)),
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
        dataset = events.metadata['dataset']
        year   = self._samples[dataset]['year']
        xsec   = self._samples[dataset]['xsec']
        sow    = self._samples[dataset]['nSumOfWeights' ]
        isData = self._samples[dataset]['isData']
        datasets = ['SingleMuon', 'SingleElectron', 'EGamma', 'MuonEG', 'DoubleMuon', 'DoubleElectron']
        for d in datasets: 
          if d in dataset: dataset = dataset.split('_')[0] 

        ### Recover objects, selection, functions and others...
        # Objects
        isTightMuon     = self._objects['isTightMuonPOG']
        isTightElectron = self._objects['isTightElectronPOG']
        isGoodJet       = self._objects['isGoodJet']
        isClean         = self._objects['isClean']
        isMuonMVA       = self._objects['isMuonMVA'] #isMuonMVA(pt, eta, dxy, dz, miniIso, sip3D, mvaTTH, mediumPrompt, tightCharge, jetDeepB=0, minpt=15)
        isElecMVA       = self._objects['isElecMVA'] #isElecMVA(pt, eta, dxy, dz, miniIso, sip3D, mvaTTH, elecMVA, lostHits, convVeto, tightCharge, jetDeepB=0, minpt=15)

        # Corrections
        GetMuonIsoSF    = self._corrections['getMuonIso']
        GetMuonIDSF     = self._corrections['getMuonID' ]

        # Selection
        passNJets   = self._selection['passNJets']
        passMETcut  = self._selection['passMETcut']
        passTrigger = self._selection['passTrigger']

        # Functions
        #pow2            = self._functions['pow2']
        #IsClosestToZ    = self._functions['IsClosestToZ']
        #GetGoodTriplets = self._functions['GetGoodTriplets']

        # Initialize objects
        met = events.MET
        e   = events.Electron
        mu  = events.Muon
        j   = events.Jet
 

        # Electron selection
        #e['isGood'] = e.pt.zeros_like()
        e['isGood'] = isElecMVA(e.pt, e.eta, e.dxy, e.dz, e.miniPFRelIso_all, e.sip3d, e.mvaTTH, e.mvaFall17V2Iso, e.lostHits, e.convVeto, e.tightCharge, minpt=10)
        leading_e = e[e.pt.argmax()]
        leading_e = leading_e[leading_e.isGood.astype(np.bool)]

        # Muon selection
        mu['isGood'] = isMuonMVA(mu.pt, mu.eta, mu.dxy, mu.dz, mu.miniPFRelIso_all, mu.sip3d, mu.mvaTTH, mu.mediumPromptId, mu.tightCharge, minpt=10)
        leading_mu = mu[mu.pt.argmax()]
        leading_mu = leading_mu[leading_mu.isGood.astype(np.bool)]
        
        e  =  e[e .isGood.astype(np.bool)]
        mu = mu[mu.isGood.astype(np.bool)]
        nElec = e .counts
        nMuon = mu.counts

        twoLeps   = (nElec+nMuon) == 2
        threeLeps = (nElec+nMuon) == 3
        twoElec   = (nElec == 2)
        twoMuon   = (nMuon == 2)
        e0 = e[e.pt.argmax()]
        m0 = mu[mu.pt.argmax()]

        # Jet selection
        j['isgood']  = isGoodJet(j.pt, j.eta, j.jetId)
        j['isclean'] = isClean(j, e, mu)
        goodJets = j[(j.isclean)&(j.isgood)]
        njets = goodJets.counts
        ht = goodJets.pt.sum()
        j0 = goodJets[goodJets.pt.argmax()]
        nbtags = goodJets[goodJets.btagDeepFlavB > 0.2770].counts


        ##################################################################
        ### 2 same-sign leptons
        ##################################################################

        # emu
        singe = e [(nElec==1)&(nMuon==1)&(e .pt>-1)]
        singm = mu[(nElec==1)&(nMuon==1)&(mu.pt>-1)]
        em = singe.cross(singm)
        emSSmask = (em.i0.charge*em.i1.charge>0)
        emSS = em[emSSmask]
        nemSS = len(emSS.flatten())

        # ee and mumu
        # pt>-1 to preserve jagged dimensions
        ee = e [(nElec==2)&(nMuon==0)&(e.pt>-1)]
        mm = mu[(nElec==0)&(nMuon==2)&(mu.pt>-1)]

        eepairs = ee.distincts()
        eeSSmask = (eepairs.i0.charge*eepairs.i1.charge>0)
        eeonZmask  = (np.abs((eepairs.i0+eepairs.i1).mass-91)<15)
        eeoffZmask = (eeonZmask==0)

        mmpairs = mm.distincts()
        mmSSmask = (mmpairs.i0.charge*mmpairs.i1.charge>0)
        mmonZmask  = (np.abs((mmpairs.i0+mmpairs.i1).mass-91)<15)
        mmoffZmask = (mmonZmask==0)

        eeSSonZ  = eepairs[eeSSmask &  eeonZmask]
        eeSSoffZ = eepairs[eeSSmask & eeoffZmask]
        mmSSonZ  = mmpairs[mmSSmask &  mmonZmask]
        mmSSoffZ = mmpairs[mmSSmask & mmoffZmask]
        neeSS = len(eeSSonZ.flatten()) + len(eeSSoffZ.flatten())
        nmmSS = len(mmSSonZ.flatten()) + len(mmSSoffZ.flatten())

        #print('Same-sign events [ee, emu, mumu] = [%i, %i, %i]'%(neeSS, nemSS, nmmSS))

        # Cuts
        eeSSmask   = (eeSSmask[eeSSmask].counts>0)
        mmSSmask   = (mmSSmask[mmSSmask].counts>0)
        eeonZmask  = (eeonZmask[eeonZmask].counts>0)
        eeoffZmask = (eeoffZmask[eeoffZmask].counts>0)
        mmonZmask  = (mmonZmask[mmonZmask].counts>0)
        mmoffZmask = (mmoffZmask[mmoffZmask].counts>0)
        emSSmask    = (emSSmask[emSSmask].counts>0)

        # njets

        ##################################################################
        ### 3 leptons
        ##################################################################

        # eem
        muon_eem = mu[(nElec==2)&(nMuon==1)&(mu.pt>-1)]
        elec_eem =  e[(nElec==2)&(nMuon==1)&( e.pt>-1)]
        ee_eem   = elec_eem.distincts()
        ee_eemZmask     = (ee_eem.i0.charge*ee_eem.i1.charge<1)&(np.abs((ee_eem.i0+ee_eem.i1).mass-91)<15)
        ee_eemOffZmask  = (ee_eem.i0.charge*ee_eem.i1.charge<1)&(np.abs((ee_eem.i0+ee_eem.i1).mass-91)>15)
        ee_eemZmask     = (ee_eemZmask[ee_eemZmask].counts>0)
        ee_eemOffZmask  = (ee_eemOffZmask[ee_eemOffZmask].counts>0)

        eepair_eem     = (ee_eem.i0+ee_eem.i1)
        trilep_eem     = eepair_eem.cross(muon_eem)
        trilep_eem     = (trilep_eem.i0+trilep_eem.i1) 

        # mme
        muon_mme = mu[(nElec==1)&(nMuon==2)&(mu.pt>-1)]
        elec_mme =  e[(nElec==1)&(nMuon==2)&( e.pt>-1)]
        mm_mme   = muon_mme.distincts()
        mm_mmeZmask     = (mm_mme.i0.charge*mm_mme.i1.charge<1)&(np.abs((mm_mme.i0+mm_mme.i1).mass-91)<15)
        mm_mmeOffZmask  = (mm_mme.i0.charge*mm_mme.i1.charge<1)&(np.abs((mm_mme.i0+mm_mme.i1).mass-91)>15)
        mm_mmeZmask     = (mm_mmeZmask[mm_mmeZmask].counts>0)
        mm_mmeOffZmask  = (mm_mmeOffZmask[mm_mmeOffZmask].counts>0)

        mmpair_mme     = (mm_mme.i0+mm_mme.i1)
        trilep_mme     = mmpair_mme.cross(elec_mme)
        trilep_mme     = (trilep_mme.i0+trilep_mme.i1)
        mZ_mme  = mmpair_mme.mass
        mZ_eem  = eepair_eem.mass
        m3l_eem = trilep_eem.mass
        m3l_mme = trilep_mme.mass


        ### eee and mmm
        eee =   e[(nElec==3)&(nMuon==0)&( e.pt>-1)] 
        mmm =  mu[(nElec==0)&(nMuon==3)&(mu.pt>-1)] 
        # Create pairs
        ee_pairs = eee.argchoose(2)
        mm_pairs = mmm.argchoose(2)

        # Select pairs that are SFOS.
        eeSFOS_pairs = ee_pairs[(np.abs(eee[ee_pairs.i0].pdgId) == np.abs(eee[ee_pairs.i1].pdgId)) & (eee[ee_pairs.i0].charge != eee[ee_pairs.i1].charge)]
        mmSFOS_pairs = mm_pairs[(np.abs(mmm[mm_pairs.i0].pdgId) == np.abs(mmm[mm_pairs.i1].pdgId)) & (mmm[mm_pairs.i0].charge != mmm[mm_pairs.i1].charge)]
        # Find the pair with mass closest to Z.
        eeOSSFmask = eeSFOS_pairs[np.abs((eee[eeSFOS_pairs.i0] + eee[eeSFOS_pairs.i1]).mass - 91.2).argmin()]
        onZmask_ee = np.abs((eee[eeOSSFmask.i0] + eee[eeOSSFmask.i1]).mass - 91.2) < 15
        mmOSSFmask = mmSFOS_pairs[np.abs((mmm[mmSFOS_pairs.i0] + mmm[mmSFOS_pairs.i1]).mass - 91.2).argmin()]
        onZmask_mm = np.abs((mmm[mmOSSFmask.i0] + mmm[mmOSSFmask.i1]).mass - 91.2) < 15
        offZmask_ee = np.abs((eee[eeOSSFmask.i0] + eee[eeOSSFmask.i1]).mass - 91.2) > 15
        offZmask_mm = np.abs((mmm[mmOSSFmask.i0] + mmm[mmOSSFmask.i1]).mass - 91.2) > 15

        # Create masks
        eeeOnZmask  = onZmask_ee[onZmask_ee].counts>0
        eeeOffZmask = offZmask_ee[offZmask_ee].counts>0
        mmmOnZmask  = onZmask_mm[onZmask_mm].counts>0
        mmmOffZmask = offZmask_mm[offZmask_mm].counts>0
    
        # Leptons from Z
        eZ0= eee[eeOSSFmask.i0]
        eZ1= eee[eeOSSFmask.i1]
        mZ0= mmm[mmOSSFmask.i0]
        mZ1= mmm[mmOSSFmask.i1]

        # Leptons from W
        eW = eee[~eeOSSFmask.i0 | ~eeOSSFmask.i1]
        mW = mmm[~mmOSSFmask.i0 | ~mmOSSFmask.i1]

        eZ = eee[eeOSSFmask.i0] + eee[eeOSSFmask.i1]
        triElec = eZ + eW
        mZ = mmm[mmOSSFmask.i0] + mmm[mmOSSFmask.i1]
        triMuon = mZ + mW

        mZ_eee  = eZ.mass
        m3l_eee = triElec.mass
        mZ_mmm  = mZ.mass
        m3l_mmm = triMuon.mass
    
        # Triggers
        #passTrigger = lambda events, n, m, o : np.ones_like(events['MET_pt'], dtype=np.bool) # XXX
        trig_eeSS = passTrigger(events,'ee',isData,dataset)
        trig_mmSS = passTrigger(events,'mm',isData,dataset)
        trig_emSS = passTrigger(events,'em',isData,dataset)
        trig_eee  = passTrigger(events,'eee',isData,dataset)
        trig_mmm  = passTrigger(events,'mmm',isData,dataset)
        trig_eem  = passTrigger(events,'eem',isData,dataset)
        trig_mme  = passTrigger(events,'mme',isData,dataset)


        # MET filters

        # Weights
        genw = np.ones_like(events['MET_pt']) if isData else events['genWeight']
        weights = processor.Weights(events.size)
        weights.add('norm',genw if isData else (xsec/sow)*genw)
        eftweights = events['EFTfitCoefficients']

        # Selections and cuts
        selections = processor.PackedSelection()
        channels2LSS = ['eeSSonZ', 'eeSSoffZ', 'mmSSonZ', 'mmSSoffZ', 'emSS']
        selections.add('eeSSonZ',  (eeonZmask)&(eeSSmask)&(trig_eeSS))
        selections.add('eeSSoffZ', (eeoffZmask)&(eeSSmask)&(trig_eeSS))
        selections.add('mmSSonZ',  (mmonZmask)&(mmSSmask)&(trig_mmSS))
        selections.add('mmSSoffZ', (mmoffZmask)&(mmSSmask)&(trig_mmSS))
        selections.add('emSS',     (emSSmask)&(trig_emSS))

        channels3L = ['eemSSonZ', 'eemSSoffZ', 'mmeSSonZ', 'mmeSSoffZ']
        selections.add('eemSSonZ',   (ee_eemZmask)&(trig_eem))
        selections.add('eemSSoffZ',  (ee_eemOffZmask)&(trig_eem))
        selections.add('mmeSSonZ',   (mm_mmeZmask)&(trig_mme))
        selections.add('mmeSSoffZ',  (mm_mmeOffZmask)&(trig_mme))

        channels3L += ['eeeSSonZ', 'eeeSSoffZ', 'mmmSSonZ', 'mmmSSoffZ']
        selections.add('eeeSSonZ',   (eeeOnZmask)&(trig_eee))
        selections.add('eeeSSoffZ',  (eeeOffZmask)&(trig_eee))
        selections.add('mmmSSonZ',   (mmmOnZmask)&(trig_mmm))
        selections.add('mmmSSoffZ',  (mmmOffZmask)&(trig_mmm))

        levels = ['base', '2jets', '4jets', '4j1b', '4j2b']
        selections.add('base', (nElec+nMuon>=2))
        selections.add('2jets',(njets>=2))
        selections.add('4jets',(njets>=4))
        selections.add('4j1b',(njets>=4)&(nbtags>=1))
        selections.add('4j2b',(njets>=4)&(nbtags>=2))

        # Variables
        invMass_eeSSonZ  = ( eeSSonZ.i0+ eeSSonZ.i1).mass
        invMass_eeSSoffZ = (eeSSoffZ.i0+eeSSoffZ.i1).mass
        invMass_mmSSonZ  = ( mmSSonZ.i0+ mmSSonZ.i1).mass
        invMass_mmSSoffZ = (mmSSoffZ.i0+mmSSoffZ.i1).mass
        invMass_emSS     = (emSS.i0+emSS.i1).mass

        varnames = {}
        varnames['met'] = met.pt
        varnames['ht'] = ht
        varnames['njets'] = njets
        varnames['nbtags'] = nbtags
        varnames['invmass'] = {
          'eeSSonZ'   : invMass_eeSSonZ,
          'eeSSoffZ'  : invMass_eeSSoffZ,
          'mmSSonZ'   : invMass_mmSSonZ,
          'mmSSoffZ'  : invMass_mmSSoffZ,
          'emSS'      : invMass_emSS,
          'eemSSonZ'  : mZ_eem,
          'eemSSoffZ' : mZ_eem,
          'mmeSSonZ'  : mZ_mme,
          'mmeSSoffZ' : mZ_mme,
          'eeeSSonZ'  : mZ_eee,
          'eeeSSoffZ' : mZ_eee,
          'mmmSSonZ'  : mZ_mmm,
          'mmmSSoffZ' : mZ_mmm,
        }
        varnames['m3l'] = {
          'eemSSonZ'  : m3l_eem,
          'eemSSoffZ' : m3l_eem,
          'mmeSSonZ'  : m3l_mme,
          'mmeSSoffZ' : m3l_mme,
          'eeeSSonZ'  : m3l_eee,
          'eeeSSoffZ' : m3l_eee,
          'mmmSSonZ'  : m3l_mmm,
          'mmmSSoffZ' : m3l_mmm,
        }
        varnames['e0pt' ] = e0.pt
        varnames['e0eta'] = e0.eta
        varnames['m0pt' ] = m0.pt
        varnames['m0eta'] = m0.eta
        varnames['j0pt' ] = j0.pt
        varnames['j0eta'] = j0.eta
        varnames['counts'] = np.ones_like(events.MET.pt, dtype=np.int) 

        # fill Histos
        hout = self.accumulator.identity()
        allweights = weights.weight().flatten()
        #hout['dummy'].fill(sample=dataset, dummy=varnames['counts'], weight=np.ones_like(events.MET.pt, dtype=np.int))
        hout['SumOfEFTweights'].fill(eftweights, sample=dataset, SumOfEFTweights=varnames['counts'], weight=allweights)

        for var, v in varnames.items():
         for ch in channels2LSS+channels3L:
          for lev in levels:
            weight = weights.weight()
            cuts = [ch] + [lev]
            cut = selections.all(*cuts)
            weights_flat = weight[cut].flatten()
            weights_ones = np.ones_like(weights_flat, dtype=np.int)
            eftweightsvalues = eftweights[cut]
            if var == 'invmass':
              if   ch in ['eeeSSoffZ', 'mmmSSoffZ']: continue
              elif ch in ['eeeSSonZ' , 'mmmSSonZ' ]: continue #values = v[ch]
              else                                 : values = v[ch][cut].flatten()
              hout['invmass'].fill(sample=dataset, channel=ch, cut=lev, invmass=values, weight=weights_flat)
            elif var == 'm3l': 
              if ch in ['eeSSonZ','eeSSoffZ', 'mmSSonZ', 'mmSSoffZ','emSS', 'eeeSSoffZ', 'mmmSSoffZ', 'eeeSSonZ' , 'mmmSSonZ']: continue
              values = v[ch][cut].flatten()
              hout['m3l'].fill(eftweightsvalues, sample=dataset, channel=ch, cut=lev, m3l=values, weight=weights_flat)
            else:
              values = v[cut].flatten()
              if   var == 'ht'    : hout[var].fill(eftweightsvalues, ht=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'met'   : hout[var].fill(eftweightsvalues, met=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'njets' : hout[var].fill(eftweightsvalues, njets=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'nbtags': hout[var].fill(eftweightsvalues, nbtags=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'counts': hout[var].fill(counts=values, sample=dataset, channel=ch, cut=lev, weight=weights_ones)
              elif var == 'j0eta' : 
                if lev == 'base': continue
                hout[var].fill(eftweightsvalues, j0eta=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'e0pt'  : 
                if ch in ['mmSSonZ', 'mmSSoffZ', 'mmmSSoffZ', 'mmmSSonZ']: continue
                hout[var].fill(eftweightsvalues, e0pt=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'm0pt'  : 
                if ch in ['eeSSonZ', 'eeSSoffZ', 'eeeSSoffZ', 'eeeSSonZ']: continue
                hout[var].fill(eftweightsvalues, m0pt=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'e0eta' : 
                if ch in ['mmSSonZ', 'mmSSoffZ', 'mmmSSoffZ', 'mmmSSonZ']: continue
                hout[var].fill(eftweightsvalues, e0eta=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'm0eta':
                if ch in ['eeSSonZ', 'eeSSoffZ', 'eeeSSoffZ', 'eeeSSonZ']: continue
                hout[var].fill(eftweightsvalues, m0eta=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
              elif var == 'j0pt'  : 
                if lev == 'base': continue
                hout[var].fill(eftweightsvalues, j0pt=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)

        return hout

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    # Load the .coffea files
    outpath= './coffeaFiles/'
    samples     = load(outpath+'samples.coffea')
    objects     = load(outpath+'objects.coffea')
    selection   = load(outpath+'selection.coffea')
    corrections = load(outpath+'corrections.coffea')

    topprocessor = AnalysisProcessor(samples, objects, selection, corrections)
    save(topprocessor, outpath+'topeft.coffea')

