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

        self._e['id' ]           = 'Electron_cutBased'
        self._e['dxy']           = 'Electron_dxy'
        self._e['dz' ]           = 'Electron_dz'
        self._e['tightCharge' ]  = 'Electron_tightCharge'
        self._e['miniIso' ]      = 'Electron_miniPFRelIso_all'
        self._e['sip3d' ]        = 'Electron_sip3d'
        self._e['mvaTTH' ]       = 'Electron_mvaTTH'
        self._e['elecMVA' ]      = 'Electron_mvaFall17V2Iso'
        self._e['lostHits' ]     = 'Electron_lostHits'
        self._e['convVeto' ]     = 'Electron_convVeto'
        self._e['charge' ]       = 'Electron_charge'

        self._mu['tight_id']     = 'Muon_tightId'
        self._mu['mediumId']     = 'Muon_mediumId'
        self._mu['mediumPrompt'] = 'Muon_mediumPromptId'
        self._mu['dxy']          = 'Muon_dxy'
        self._mu['dz' ]          = 'Muon_dz'
        self._mu['iso']          = 'Muon_pfRelIso04_all'
        self._mu['tightCharge']  = 'Muon_tightCharge'
        self._mu['mvaTTH']       = 'Muon_mvaTTH'
        self._mu['miniIso']      = 'Muon_miniPFRelIso_all'
        self._mu['sip3d']        = 'Muon_sip3d'
        self._mu['charge' ]      = 'Muon_charge'

        self._jet['id'] = 'Jet_jetId'

        # Create the histograms
        # 'name' : hist.Hist("Ytitle", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat("syst", "syst"), hist.Bin("name", "X axis (GeV)", 20, 0, 100)),
        self._accumulator = processor.dict_accumulator({
        'dummy'   : hist.Hist("Dummy", hist.Cat("sample", "sample"), hist.Bin("dummy", "Number of events", 1, 0, 1)),
        #'lep0pt'  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("lepCat", "lepCat"), hist.Cat("Zcat", "Zcat"), hist.Bin("lep0pt",  "Leading lepton $p_{T}$ (GeV)", 20, 0, 200)),
        #'lep0eta' : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("lepCat", "lepCat"), hist.Cat("Zcat", "Zcat"), hist.Bin("lep0eta", "Leading lepton $\eta$ ", 15, -2.5, 2.50)),
        'invmass' : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("lepCat", "lepCat"), hist.Cat("Zcat", "Zcat"),hist.Cat("cut","cut"), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 20, 0, 200)),
        'njets'   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("lepCat", "lepCat"), hist.Cat("Zcat", "Zcat"), hist.Cat("cut", "cut"), hist.Bin("njets",   "Jet multiplicitu ", 10, 0, 10)),
        'nbtags'   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("lepCat", "lepCat"), hist.Cat("Zcat", "Zcat"), hist.Cat("cut", "cut"), hist.Bin("nbtags",   "btag multiplicitu ", 5, 0, 5)),
        'met'   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("lepCat", "lepCat"), hist.Cat("Zcat", "Zcat"), hist.Cat("cut", "cut"), hist.Bin("met",   "MET (GeV)", 40, 0, 400)),
        'm3l' : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("lepCat", "lepCat"), hist.Cat("Zcat", "Zcat"), hist.Cat("cut", "cut"), hist.Bin("m3l", "$m_{3\ell}$ (GeV) ", 20, 0, 200)),
        'wleppt' : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("lepCat", "lepCat"), hist.Cat("Zcat", "Zcat"), hist.Cat("cut", "cut"), hist.Bin("wleppt", "$p_{T}^{lepW}$ (GeV) ", 20, 0, 200)),
        'ht'   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("lepCat", "lepCat"), hist.Cat("Zcat", "Zcat"), hist.Cat("cut", "cut"), hist.Bin("ht",   "H$_{T}$ (GeV)", 40, 0, 800)),
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
        if dataset in ['SingleMuon', 'SingleElectron', 'EGamma', 'MuonEG', 'DoubleMuon', 'DoubleElectron']: dataset = dataset.split('_')[0]

        ### Recover objects, selection, functions and others...
        # Objects
        isTightMuon     = self._objects['isTightMuonPOG']
        isTightElectron = self._objects['isTightElectronPOG']
        isGoodJet       = self._objects['isGoodJet']
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
        pow2            = self._functions['pow2']
        IsClosestToZ    = self._functions['IsClosestToZ']
        GetGoodTriplets = self._functions['GetGoodTriplets']

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
        #e['isGood'] = isTightElectron(e.pt, e.eta, e.dxy, e.dz, e.id, e.tightChrage, year)
        e['isGood'] = isElecMVA(e.pt, e.eta, e.dxy, e.dz, e.miniIso, e.sip3d, e.mvaTTH, e.elecMVA, e.lostHits, e.convVeto, e.tightCharge, minpt=10)
        leading_e = e[e.pt.argmax()]
        leading_e = leading_e[leading_e.isGood.astype(np.bool)]

        # Muon selection
        for key in self._mu:
            mu[key] = mu.pt.zeros_like()
            if self._mu[key] in df:
                mu[key] = df[self._mu[key]]
        #mu['istight'] = isTightMuon(mu.pt, mu.eta, mu.dxy, mu.dz, mu.iso, mu.tight_id, mu.tightCharge, year)
        mu['isGood'] = isMuonMVA(mu.pt, mu.eta, mu.dxy, mu.dz, mu.miniIso, mu.sip3d, mu.mvaTTH, mu.mediumPrompt, mu.tightCharge, minpt=10)
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

        # Jet selection
        j['deepjet'] = df['Jet_btagDeepFlavB']
        for key in self._jet:
                j[key] = j.pt.zeros_like()
                if self._jet[key] in df:
                    j[key] = df[self._jet[key]]

        j['isgood']  = isGoodJet(j.pt, j.eta, j.id)
        j['isclean'] = ~j.match(e,0.4) & ~j.match(mu,0.4) & j.isgood.astype(np.bool)
        j0 = j[j.pt.argmax()]
        j0 = j0[j0.isclean.astype(np.bool)]
        nJets = j.counts



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
        goodJets = j[(j.isclean)&(j.isgood)]
        njets = j.counts
        ht = j.pt.sum()

        # nbtags
        nbtags = goodJets[goodJets.deepjet > 0.2770].counts



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


        ### eee and mmm
        eee =   e[(nElec==3)&(nMuon==0)&( e.pt>-1)] 
        mmm =  mu[(nElec==0)&(nMuon==3)&(mu.pt>-1)] 
        # Create pairs
        eee_groups = eee.distincts()
        mmm_groups = mmm.distincts()
        # Calculate the invariant mass of the pairs
        invMass_eee = ((eee_groups.i0+eee_groups.i1).mass)
        invMass_mmm = ((mmm_groups.i0+mmm_groups.i1).mass)
        # OS pairs
        isOSeee = ((eee_groups.i0.charge != eee_groups.i1.charge))
        isOSmmm = ((mmm_groups.i0.charge != mmm_groups.i1.charge))
        # Get the ones with a mass closest to the Z mass (and in a range of  thr)
        clos_eee = IsClosestToZ(invMass_eee, thr=15)
        clos_mmm = IsClosestToZ(invMass_mmm, thr=15)
        # Finally, the mask for eee/mmm with/without OS onZ pair
        eeeOnZmask  = (clos_eee)&(isOSeee)
        eeeOffZmask = (eeeOnZmask==0)
        mmmOnZmask  = (clos_mmm)&(isOSmmm)
        mmmOffZmask = (mmmOnZmask==0)
        eeeOnZmask  = (eeeOnZmask[eeeOnZmask].counts>0)
        eeeOffZmask = (eeeOffZmask[eeeOffZmask].counts>0)
        mmmOnZmask  = (mmmOnZmask[mmmOnZmask].counts>0)
        mmmOffZmask = (mmmOffZmask[mmmOffZmask].counts>0)
        
        # Get Z and W invariant masses
        goodPairs_eee = eee_groups[(eeeOnZmask)&(eee_groups.i0.pt>-1)]
        '''
        eZ0 = goodPairs_eee.i0[goodPairs_eee.counts>0].regular()
        eZ1 = goodPairs_eee.i1[goodPairs_eee.counts>0].regular()

        eee_reg = eee[goodPairs_eee.counts>0].regular()
        eW = np.append(eZ0, eee_reg,axis=1)
        eW = np.append(eW, eZ1,axis=1)
        mask = np.apply_along_axis(lambda a : [list(a).count(x)==1 for x in a], 1, eW)
        eW = eW[mask]
        #
        #trilep = eZ0+eZ1+eW
        #meee = trilep.mass
        #print('Trilep mass = ', meee)

        # Trilep
        #trilep = goodPairs_eee.cross(single_e)
        #meee = (trilep.i0+trilep.i1+trilep.i2).mass
        '''
        

        # Triggers
        #passTrigger = lambda df, n, m, o : np.ones_like(df['MET_pt'], dtype=np.bool) # XXX
        trig_eeSS = passTrigger(df,'ee',isData,dataset)
        trig_mmSS = passTrigger(df,'mm',isData,dataset)
        trig_emSS = passTrigger(df,'em',isData,dataset)
        trig_eee  = passTrigger(df,'eee',isData,dataset)
        trig_mmm  = passTrigger(df,'mmm',isData,dataset)
        trig_eem  = passTrigger(df,'eem',isData,dataset)
        trig_mme  = passTrigger(df,'mme',isData,dataset)

        # MET filters

        # Weights
        genw = np.ones_like(df['MET_pt']) if isData else df['genWeight']
        weights = processor.Weights(df.size)
        weights.add('norm',1 if isData else (xsec/sow)*genw)

        # Selections and cuts
        selections = processor.PackedSelection()
        channels2LSS = ['eeSSonZ', 'eeSSoffZ', 'mmSSonZ', 'mmSSoffZ', 'emSS']
        selections.add('eeSSonZ',  (eeonZmask)&(eeSSmask)&(trig_eeSS))
        selections.add('eeSSoffZ', (eeoffZmask)&(eeSSmask)&(trig_eeSS))
        selections.add('mmSSonZ',  (mmonZmask)&(mmSSmask)&(trig_mmSS))
        selections.add('mmSSoffZ', (mmoffZmask)&(mmSSmask)&(trig_mmSS))
        selections.add('emSS',     (emSSmask)&(trig_emSS))

        channels3L = ['eemSSonZ', 'eemSSoffZ', 'mmeSSonZ', 'mmeSSoffZ']#, 'eeeSSonZ', 'eeeSSoffZ', 'mmmSSonZ', 'mmmSSoffZ']
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

        #invMass_ee_eemSSonZ = (
        
        varnames = {}
        varnames['invmass'] = ''
        varnames['met'] = met.pt
        varnames['ht'] = ht
        varnames['njets'] = njets
        varnames['nbtags'] = nbtags

        # Fill Histos
        hout = self.accumulator.identity()
        hout['dummy'].fill(sample=dataset, dummy=1, weight=df.size)

        for var, v in varnames.items():
         for ch in channels2LSS+channels3L:
          for lev in levels:
            weight = weights.weight()
            cuts = [ch] + [lev]
            cut = selections.all(*cuts)
            #metflat = met.pt[cut].flatten()
            weights_flat = weight[cut].flatten()
            Zcat = 'onZ' if 'onZ' in ch else 'offZ'
            # Special case for invmass
            if var == 'invmass':
              if   ch == 'eeSSonZ' : hout['invmass'].fill(sample=dataset, channel=ch, cut=lev, lepCat='2lSS', Zcat=Zcat, invmass=invMass_eeSSonZ[cut].flatten(), weight=weights_flat)
              elif ch == 'eeSSoffZ': hout['invmass'].fill(sample=dataset, channel=ch, cut=lev, lepCat='2lSS', Zcat=Zcat, invmass=invMass_eeSSoffZ[cut].flatten(), weight=weights_flat)
              elif ch == 'mmSSonZ' : hout['invmass'].fill(sample=dataset, channel=ch, cut=lev, lepCat='2lSS', Zcat=Zcat, invmass=invMass_mmSSonZ[cut].flatten(), weight=weights_flat)
              elif ch == 'mmSSoffZ': hout['invmass'].fill(sample=dataset, channel=ch, cut=lev, lepCat='2lSS', Zcat=Zcat, invmass=invMass_mmSSoffZ[cut].flatten(), weight=weights_flat)
            else:
              lepCat = '2lSS' if ch in channels2LSS else '3l'
              values = v[cut].flatten()
              if   var == 'ht'    : hout[var].fill(ht=values, sample=dataset, channel=ch, cut=lev, lepCat=lepCat, Zcat=Zcat, weight=weights_flat)
              if   var == 'met'   : hout[var].fill(met=values, sample=dataset, channel=ch, cut=lev, lepCat=lepCat, Zcat=Zcat, weight=weights_flat)
              elif var == 'njets' : hout[var].fill(njets=values, sample=dataset, channel=ch, cut=lev, lepCat=lepCat, Zcat=Zcat, weight=weights_flat)
              elif var == 'nbtags': hout[var].fill(nbtags=values, sample=dataset, channel=ch, cut=lev, lepCat=lepCat, Zcat=Zcat, weight=weights_flat)

            #hout['invmass'].fill(sample=dataset, channel=ch, lepCat=lev, Zcat="all", invmass=invmass_flat, weight=weights_flat)#*selections.all(*{'mm'})
        #flat_variables = {k: v[cut].flatten() for k, v in variables.items()}
        #flat_weights = {k: (~np.isnan(v[cut])*weight[cut]).flatten() for k, v in variables.items()}



        #hout['invmass'].fill(sample=dataset, channel='mm', level="dilepton", invmass=mmumu, weight=np.ones_like(df['MET_pt']))#weight=weights.weight())#*selections.all(*{'mm'})
        
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
    functions   = load(outpath+'functions.coffea')

    # Branches
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
    Electron_sip3d
    Electron_convVeto
    Electron_lostHits
    Electron_pfRelIso03_all
    Electron_miniPFRelIso_all
    Electron_mvaTTH
    Electron_mvaFall17V2Iso
    Muon_pt
    Muon_ptErr
    Muon_eta
    Muon_phi
    Muon_mass
    Muon_pfRelIso04_all
    Muon_miniPFRelIso_all
    Muon_sip3d
    Muon_tightId
    Muon_mediumId
    Muon_mediumPromptId
    Muon_dxy
    Muon_dz
    Muon_tightCharge
    Muon_mvaTTH
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
    save(topprocessor, outpath+'topeft.coffea')


