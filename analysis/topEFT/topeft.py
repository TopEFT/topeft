#!/usr/bin/env python
import lz4.frame as lz4f
import cloudpickle
import json
import pprint
import coffea
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
#from coffea.arrays import Initialize # Not used and gives error
from coffea import hist, processor
from coffea.util import load, save
from optparse import OptionParser
from coffea.analysis_tools import PackedSelection

from topcoffea.modules.objects import *
from topcoffea.modules.corrections import SFevaluator, GetLeptonSF, GetBTagSF, jet_factory, GetBtagEff
from topcoffea.modules.selection import *
from topcoffea.modules.HistEFT import HistEFT, EFTHelper

#coffea.deprecations_as_errors = True

# In the future these names will be read from the nanoAOD files
#wc_names_lst= ['ctW', 'ctp', 'cpQM', 'ctli', 'cQei', 'ctZ', 'cQlMi', 'cQl3i', 'ctG', 'ctlTi', 'cbW', 'cpQ3', 'ctei', 'cpt', 'ctlSi', 'cptb']

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, wc_names_lst=[], do_errors=False, do_systematics=False):
        self._samples = samples
        self._wc_names_lst = wc_names_lst

        # Create the histograms
        # In general, histograms depend on 'sample', 'channel' (final state) and 'cut' (level of selection)
        self._accumulator = processor.dict_accumulator({
        'SumOfEFTweights'  : HistEFT("SumOfWeights", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("SumOfEFTweights", "sow", 1, 0, 2)),
        'dummy'   : hist.Hist("Dummy" , hist.Cat("sample", "sample"), hist.Bin("dummy", "Number of events", 1, 0, 1)),
        'counts'  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Bin("counts", "Counts", 1, 0, 2)),
        'invmass' : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 20, 0, 200)),
        'njets'   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("njets",  "Jet multiplicity ", 10, 0, 10)),
        'nbtags'  : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("nbtags", "btag multiplicity ", 5, 0, 5)),
        'met'     : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("met",    "MET (GeV)", 40, 0, 400)),
        'm3l'     : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("m3l",    "$m_{3\ell}$ (GeV) ", 50, 0, 500)),
        'wleppt'  : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("wleppt", "$p_{T}^{lepW}$ (GeV) ", 20, 0, 200)),
        'e0pt'    : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("e0pt",   "Leading elec $p_{T}$ (GeV)", 25, 0, 500)),
        'm0pt'    : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("m0pt",   "Leading muon $p_{T}$ (GeV)", 25, 0, 500)),
        'j0pt'    : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("j0pt",   "Leading jet  $p_{T}$ (GeV)", 25, 0, 500)),
        'e0eta'   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("e0eta",  "Leading elec $\eta$", 30, -3.0, 3.0)),
        'm0eta'   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("m0eta",  "Leading muon $\eta$", 30, -3.0, 3.0)),
        'j0eta'   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("j0eta",  "Leading jet  $\eta$", 30, -3.0, 3.0)),
        'ht'      : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("ht",     "H$_{T}$ (GeV)", 50, 0, 1000)),
        'njetsnbtags' : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("njets",  "Jet multiplicity ", 10, 0, 10), hist.Bin("nbtags", "btag multiplicity ", 5, 0, 5)), 
        })

        self._eft_helper = EFTHelper(wc_names_lst)
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
        histAxisName = self._samples[dataset]['histAxisName']
        year         = self._samples[dataset]['year']
        xsec         = self._samples[dataset]['xsec']
        sow          = self._samples[dataset]['nSumOfWeights' ]
        isData       = self._samples[dataset]['isData']
        datasets     = ['SingleMuon', 'SingleElectron', 'EGamma', 'MuonEG', 'DoubleMuon', 'DoubleElectron']
        for d in datasets: 
          if d in dataset: dataset = dataset.split('_')[0] 

        # Initialize objects
        met = events.MET
        e   = events.Electron
        mu  = events.Muon
        tau = events.Tau
        j   = events.Jet
 
        # Muon selection

        mu['isPres'] = isPresMuon(mu.dxy, mu.dz, mu.sip3d, mu.looseId)
        mu['isTight']= isTightMuon(mu.pt, mu.eta, mu.dxy, mu.dz, mu.pfRelIso03_all, mu.sip3d, mu.mvaTTH, mu.mediumPromptId, mu.tightCharge, mu.looseId, minpt=10)
        mu['isGood'] = mu['isPres'] & mu['isTight']

        leading_mu = mu[ak.argmax(mu.pt,axis=-1,keepdims=True)]
        leading_mu = leading_mu[leading_mu.isGood]
        
        mu = mu[mu.isGood]
        mu_pres = mu[mu.isPres]

        # Electron selection
        e['isPres']  = isPresElec(e.pt, e.eta, e.dxy, e.dz, e.miniPFRelIso_all, e.sip3d, e.lostHits, minpt=15)
        e['isTight'] = isTightElec(e.pt, e.eta, e.dxy, e.dz, e.miniPFRelIso_all, e.sip3d, e.mvaTTH, e.mvaFall17V2Iso, e.lostHits, e.convVeto, e.tightCharge, e.sieie, e.hoe, e.eInvMinusPInv, minpt=15)
        e['isClean'] = isClean(e, mu, drmin=0.05)
        e['isGood']  = e['isPres'] & e['isTight'] & e['isClean']

        leading_e = e[ak.argmax(e.pt,axis=-1,keepdims=True)]
        leading_e = leading_e[leading_e.isGood]

        e  =  e[e .isGood]
        e_pres = e[e .isPres & e .isClean]

        # Tau selection
        tau['isPres']  = isPresTau(tau.pt, tau.eta, tau.dxy, tau.dz, tau.leadTkPtOverTauPt, tau.idAntiMu, tau.idAntiEle, tau.rawIso, tau.idDecayModeNewDMs, minpt=20)
        tau['isClean'] = isClean(tau, e_pres, drmin=0.4) & isClean(tau, mu_pres, drmin=0.4)
        tau['isGood']  = tau['isPres']# & tau['isClean'], for the moment
        tau= tau[tau.isGood]

        nElec = ak.num(e)
        nMuon = ak.num(mu)
        nTau  = ak.num(tau)

        twoLeps   = (nElec+nMuon) == 2
        threeLeps = (nElec+nMuon) == 3
        twoElec   = (nElec == 2)
        twoMuon   = (nMuon == 2)
        e0 = e[ak.argmax(e.pt,axis=-1,keepdims=True)]
        m0 = mu[ak.argmax(mu.pt,axis=-1,keepdims=True)]

        # Jet selection
        jetptname = 'pt_nom' if hasattr(j, 'pt_nom') else 'pt'
        
        ### Jet energy corrections
        if not isData:
          j["pt_raw"]=(1 - j.rawFactor)*j.pt
          j["mass_raw"]=(1 - j.rawFactor)*j.mass
          j["pt_gen"]=ak.values_astype(ak.fill_none(j.matched_gen.pt, 0), np.float32)
          j["rho"]= ak.broadcast_arrays(events.fixedGridRhoFastjetAll, j.pt)[0]
          events_cache = events.caches[0]
          corrected_jets = jet_factory.build(j, lazy_cache=events_cache)
          #print('jet pt: ',j.pt)
          #print('cor pt: ',corrected_jets.pt)
          #print('jes up: ',corrected_jets.JES_jes.up.pt)
          #print('jes down: ',corrected_jets.JES_jes.down.pt)
          #print(ak.fields(corrected_jets))
          '''
          # SYSTEMATICS
          jets = corrected_jets
          if(self.jetSyst == 'JERUp'):
            jets = corrected_jets.JER.up
          elif(self.jetSyst == 'JERDown'):
            jets = corrected_jets.JER.down
          elif(self.jetSyst == 'JESUp'):
            jets = corrected_jets.JES_jes.up
          elif(self.jetSyst == 'JESDown'):
            jets = corrected_jets.JES_jes.down
          '''
        
        
        
        j['isGood']  = isTightJet(getattr(j, jetptname), j.eta, j.jetId, j.neHEF, j.neEmEF, j.chHEF, j.chEmEF, j.nConstituents)
        #j['isgood']  = isGoodJet(j.pt, j.eta, j.jetId)
        #j['isclean'] = isClean(j, e, mu)
        j['isClean'] = isClean(j, e, drmin=0.4)& isClean(j, mu, drmin=0.4)# & isClean(j, tau, drmin=0.4)
        goodJets = j[(j.isClean)&(j.isGood)]
        njets = ak.num(goodJets)
        ht = ak.sum(goodJets.pt,axis=-1)
        j0 = goodJets[ak.argmax(goodJets.pt,axis=-1,keepdims=True)]
        #nbtags = ak.num(goodJets[goodJets.btagDeepFlavB > 0.2770])
        if year == 2017: btagwp = 0.3040 #WP medium
        else: btagwp = 0.2783 #WP medium
        isBtagJets = (goodJets.btagDeepB > btagwp)
        isNotBtagJets = np.invert(isBtagJets)
        nbtags = ak.num(goodJets[isBtagJets])
        
        # Btag SF following 1a) in https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods
        btagSF   = np.ones_like(ht)
        btagSFUp = np.ones_like(ht)
        btagSFDo = np.ones_like(ht)
        if not isData:
          pt = goodJets.pt; abseta = np.abs(goodJets.eta); flav = goodJets.hadronFlavour
          bJetSF   = GetBTagSF(abseta, pt, flav)
          bJetSFUp = GetBTagSF(abseta, pt, flav, sys=1)
          bJetSFDo = GetBTagSF(abseta, pt, flav, sys=-1)
          bJetEff  = GetBtagEff(abseta, pt, flav, year)
          bJetEff_data   = bJetEff*bJetSF
          bJetEff_dataUp = bJetEff*bJetSFUp
          bJetEff_dataDo = bJetEff*bJetSFDo
   
          pMC     = ak.prod(bJetEff       [isBtagJets], axis=-1) * ak.prod((1-bJetEff       [isNotBtagJets]), axis=-1)
          pData   = ak.prod(bJetEff_data  [isBtagJets], axis=-1) * ak.prod((1-bJetEff_data  [isNotBtagJets]), axis=-1)
          pDataUp = ak.prod(bJetEff_dataUp[isBtagJets], axis=-1) * ak.prod((1-bJetEff_dataUp[isNotBtagJets]), axis=-1)
          pDataDo = ak.prod(bJetEff_dataDo[isBtagJets], axis=-1) * ak.prod((1-bJetEff_dataDo[isNotBtagJets]), axis=-1)

          pMC      = ak.where(pMC==0,1,pMC) # removeing zeroes from denominator...
          btagSF   = pData  /pMC
          btagSFUp = pDataUp/pMC
          btagSFDo = pDataUp/pMC

        ##################################################################
        ### 2 same-sign leptons
        ##################################################################

        # emu
        singe = e [(nElec==1)&(nMuon==1)&(e .pt>-1)]
        singm = mu[(nElec==1)&(nMuon==1)&(mu.pt>-1)]
        em = ak.cartesian({"e":singe,"m":singm})
        emSSmask = (em.e.charge*em.m.charge>0)
        emSS = em[emSSmask]
        nemSS = len(ak.flatten(emSS))
 
        year = 2018
        lepSF_emSS      = GetLeptonSF(singm.pt, singm.eta, 'm', singe.pt, singe.eta, 'e', year=year)
        lepSF_emSS_up   = GetLeptonSF(singm.pt, singm.eta, 'm', singe.pt, singe.eta, 'e', year=year, sys=1)
        lepSF_emSS_down = GetLeptonSF(singm.pt, singm.eta, 'm', singe.pt, singe.eta, 'e', year=year, sys=-1)
        # ee and mumu
        # pt>-1 to preserve jagged dimensions
        ee = e [(nElec==2)&(nMuon==0)&(e.pt>-1)]
        mm = mu[(nElec==0)&(nMuon==2)&(mu.pt>-1)]

        sumcharge = ak.sum(e.charge, axis=-1)+ak.sum(mu.charge, axis=-1)

        eepairs = ak.combinations(ee, 2, fields=["e0","e1"])
        eeSSmask = (eepairs.e0.charge*eepairs.e1.charge>0)
        eeonZmask  = (np.abs((eepairs.e0+eepairs.e1).mass-91.2)<10)
        eeoffZmask = (eeonZmask==0)

        mmpairs = ak.combinations(mm, 2, fields=["m0","m1"])
        mmSSmask = (mmpairs.m0.charge*mmpairs.m1.charge>0)
        mmonZmask = (np.abs((mmpairs.m0+mmpairs.m1).mass-91.2)<10)
        mmoffZmask = (mmonZmask==0)

        eeSSonZ  = eepairs[eeSSmask &  eeonZmask]
        eeSSoffZ = eepairs[eeSSmask & eeoffZmask]
        mmSSonZ  = mmpairs[mmSSmask &  mmonZmask]
        mmSSoffZ = mmpairs[mmSSmask & mmoffZmask]
        neeSS = len(ak.flatten(eeSSonZ)) + len(ak.flatten(eeSSoffZ))
        nmmSS = len(ak.flatten(mmSSonZ)) + len(ak.flatten(mmSSoffZ))
        
        lepSF_eeSS = GetLeptonSF(eepairs.e0.pt, eepairs.e0.eta, 'e', eepairs.e1.pt, eepairs.e1.eta, 'e', year=year)
        lepSF_mumuSS = GetLeptonSF(mmpairs.m0.pt, mmpairs.m0.eta, 'm', mmpairs.m1.pt, mmpairs.m1.eta, 'm', year=year)
        lepSF_eeSS_up = GetLeptonSF(eepairs.e0.pt, eepairs.e0.eta, 'e', eepairs.e1.pt, eepairs.e1.eta, 'e', year=year, sys=1)
        lepSF_mumuSS_up = GetLeptonSF(mmpairs.m0.pt, mmpairs.m0.eta, 'm', mmpairs.m1.pt, mmpairs.m1.eta, 'm', year=year, sys=1)
        lepSF_eeSS_down = GetLeptonSF(eepairs.e0.pt, eepairs.e0.eta, 'e', eepairs.e1.pt, eepairs.e1.eta, 'e', year=year, sys=-1)
        lepSF_mumuSS_down = GetLeptonSF(mmpairs.m0.pt, mmpairs.m0.eta, 'm', mmpairs.m1.pt, mmpairs.m1.eta, 'm', year=year, sys=-1)

        print('Same-sign events [ee, emu, mumu] = [%i, %i, %i]'%(neeSS, nemSS, nmmSS))

        # Cuts
        eeSSmask   = (ak.num(eeSSmask[eeSSmask])>0)
        mmSSmask   = (ak.num(mmSSmask[mmSSmask])>0)
        eeonZmask  = (ak.num(eeonZmask[eeonZmask])>0)
        eeoffZmask = (ak.num(eeoffZmask[eeoffZmask])>0)
        mmonZmask  = (ak.num(mmonZmask[mmonZmask])>0)
        mmoffZmask = (ak.num(mmoffZmask[mmoffZmask])>0)
        emSSmask   = (ak.num(emSSmask[emSSmask])>0)


        ##################################################################
        ### 3 leptons
        ##################################################################

        # eem
        muon_eem = mu[(nElec==2)&(nMuon==1)&(mu.pt>-1)]
        elec_eem =  e[(nElec==2)&(nMuon==1)&( e.pt>-1)]
        ee_eem = ak.combinations(elec_eem, 2, fields=["e0", "e1"])

        ee_eemZmask     = (ee_eem.e0.charge*ee_eem.e1.charge<1)&(np.abs((ee_eem.e0+ee_eem.e1).mass-91.2)<10)
        ee_eemOffZmask  = (ee_eem.e0.charge*ee_eem.e1.charge<1)&(np.abs((ee_eem.e0+ee_eem.e1).mass-91.2)>10)
        ee_eemZmask     = (ak.num(ee_eemZmask[ee_eemZmask])>0)
        ee_eemOffZmask  = (ak.num(ee_eemOffZmask[ee_eemOffZmask])>0)

        eepair_eem  = (ee_eem.e0+ee_eem.e1)
        trilep_eem = eepair_eem+muon_eem #ak.cartesian({"e0":ee_eem.e0,"e1":ee_eem.e1, "m":muon_eem})

        lepSF_eem = GetLeptonSF(ee_eem.e0.pt, ee_eem.e0.eta, 'e', ee_eem.e1.pt, ee_eem.e1.eta, 'e', muon_eem.pt, muon_eem.eta, 'm', year)
        lepSF_eem_up = GetLeptonSF(ee_eem.e0.pt, ee_eem.e0.eta, 'e', ee_eem.e1.pt, ee_eem.e1.eta, 'e', muon_eem.pt, muon_eem.eta, 'm', year, sys=1)
        lepSF_eem_down = GetLeptonSF(ee_eem.e0.pt, ee_eem.e0.eta, 'e', ee_eem.e1.pt, ee_eem.e1.eta, 'e', muon_eem.pt, muon_eem.eta, 'm', year, sys=-1)

        # mme
        muon_mme = mu[(nElec==1)&(nMuon==2)&(mu.pt>-1)]
        elec_mme =  e[(nElec==1)&(nMuon==2)&( e.pt>-1)]

        mm_mme = ak.combinations(muon_mme, 2, fields=["m0", "m1"])
        mm_mmeZmask     = (mm_mme.m0.charge*mm_mme.m1.charge<1)&(np.abs((mm_mme.m0+mm_mme.m1).mass-91.2)<10)
        mm_mmeOffZmask  = (mm_mme.m0.charge*mm_mme.m1.charge<1)&(np.abs((mm_mme.m0+mm_mme.m1).mass-91.2)>10)
        mm_mmeZmask     = (ak.num(mm_mmeZmask[mm_mmeZmask])>0)
        mm_mmeOffZmask  = (ak.num(mm_mmeOffZmask[mm_mmeOffZmask])>0)

        mmpair_mme     = (mm_mme.m0+mm_mme.m1)
        trilep_mme     = mmpair_mme+elec_mme

        mZ_mme  = mmpair_mme.mass
        mZ_eem  = eepair_eem.mass
        m3l_eem = trilep_eem.mass
        m3l_mme = trilep_mme.mass
        
        lepSF_mme = GetLeptonSF(mm_mme.m0.pt, mm_mme.m0.eta, 'm', mm_mme.m1.pt, mm_mme.m1.eta, 'm', elec_mme.pt, elec_mme.eta, 'e', year)
        lepSF_mme_up = GetLeptonSF(mm_mme.m0.pt, mm_mme.m0.eta, 'm', mm_mme.m1.pt, mm_mme.m1.eta, 'm', elec_mme.pt, elec_mme.eta, 'e', year, sys=1)
        lepSF_mme_down = GetLeptonSF(mm_mme.m0.pt, mm_mme.m0.eta, 'm', mm_mme.m1.pt, mm_mme.m1.eta, 'm', elec_mme.pt, elec_mme.eta, 'e', year, sys=-1)

        # eee and mmm
        eee =   e[(nElec==3)&(nMuon==0)&( e.pt>-1)] 
        mmm =  mu[(nElec==0)&(nMuon==3)&(mu.pt>-1)] 

        eee_leps = ak.combinations(eee, 3, fields=["e0", "e1", "e2"])
        mmm_leps = ak.combinations(mmm, 3, fields=["m0", "m1", "m2"])
        ee_pairs = ak.combinations(eee, 2, fields=["e0", "e1"])
        mm_pairs = ak.combinations(mmm, 2, fields=["m0", "m1"])
        ee_pairs_index = ak.argcombinations(eee, 2, fields=["e0", "e1"])
        mm_pairs_index = ak.argcombinations(mmm, 2, fields=["m0", "m1"])

        lepSF_eee = GetLeptonSF(eee_leps.e0.pt, eee_leps.e0.eta, 'e', eee_leps.e1.pt, eee_leps.e1.eta, 'e', eee_leps.e2.pt, eee_leps.e2.eta, 'e', year)
        lepSF_mmm = GetLeptonSF(mmm_leps.m0.pt, mmm_leps.m0.eta, 'm', mmm_leps.m1.pt, mmm_leps.m1.eta, 'm', mmm_leps.m2.pt, mmm_leps.m2.eta, 'm', year)
        lepSF_eee_up = GetLeptonSF(eee_leps.e0.pt, eee_leps.e0.eta, 'e', eee_leps.e1.pt, eee_leps.e1.eta, 'e', eee_leps.e2.pt, eee_leps.e2.eta, 'e', year, sys=1)
        lepSF_mmm_up = GetLeptonSF(mmm_leps.m0.pt, mmm_leps.m0.eta, 'm', mmm_leps.m1.pt, mmm_leps.m1.eta, 'm', mmm_leps.m2.pt, mmm_leps.m2.eta, 'm', year, sys=1)
        lepSF_eee_down = GetLeptonSF(eee_leps.e0.pt, eee_leps.e0.eta, 'e', eee_leps.e1.pt, eee_leps.e1.eta, 'e', eee_leps.e2.pt, eee_leps.e2.eta, 'e', year, sys=-1)
        lepSF_mmm_down = GetLeptonSF(mmm_leps.m0.pt, mmm_leps.m0.eta, 'm', mmm_leps.m1.pt, mmm_leps.m1.eta, 'm', mmm_leps.m2.pt, mmm_leps.m2.eta, 'm', year, sys=-1)
        mmSFOS_pairs = mm_pairs[(np.abs(mm_pairs.m0.pdgId) == np.abs(mm_pairs.m1.pdgId)) & (mm_pairs.m0.charge != mm_pairs.m1.charge)]
        offZmask_mm = ak.all(np.abs((mmSFOS_pairs.m0 + mmSFOS_pairs.m1).mass - 91.2)>10., axis=1, keepdims=True) & (ak.num(mmSFOS_pairs)>0)
        onZmask_mm  = ak.any(np.abs((mmSFOS_pairs.m0 + mmSFOS_pairs.m1).mass - 91.2)<10., axis=1, keepdims=True)
      
        eeSFOS_pairs = ee_pairs[(np.abs(ee_pairs.e0.pdgId) == np.abs(ee_pairs.e1.pdgId)) & (ee_pairs.e0.charge != ee_pairs.e1.charge)]
        offZmask_ee = ak.all(np.abs((eeSFOS_pairs.e0 + eeSFOS_pairs.e1).mass - 91.2)>10, axis=1, keepdims=True) & (ak.num(eeSFOS_pairs)>0)
        onZmask_ee  = ak.any(np.abs((eeSFOS_pairs.e0 + eeSFOS_pairs.e1).mass - 91.2)<10, axis=1, keepdims=True)

        # Create masks **for event selection**
        eeeOnZmask  = (ak.num(onZmask_ee[onZmask_ee])>0)
        eeeOffZmask = (ak.num(offZmask_ee[offZmask_ee])>0)
        mmmOnZmask  = (ak.num(onZmask_mm[onZmask_mm])>0)
        mmmOffZmask = (ak.num(offZmask_mm[offZmask_mm])>0)

        # Now we need to create masks for the leptons in order to select leptons from the Z boson candidate (in onZ categories)
        ZeeMask = ak.argmin(np.abs((eeSFOS_pairs.e0 + eeSFOS_pairs.e1).mass - 91.2),axis=1,keepdims=True)
        ZmmMask = ak.argmin(np.abs((mmSFOS_pairs.m0 + mmSFOS_pairs.m1).mass - 91.2),axis=1,keepdims=True)
  
        Zee = eeSFOS_pairs[ZeeMask]
        Zmm = mmSFOS_pairs[ZmmMask]
        eZ0= Zee.e0[ak.num(eeSFOS_pairs)>0]
        eZ1= Zee.e1[ak.num(eeSFOS_pairs)>0]
        eZ = eZ0+eZ1
        mZ0= Zmm.m0[ak.num(mmSFOS_pairs)>0]
        mZ1= Zmm.m1[ak.num(mmSFOS_pairs)>0]
        mZ = mZ0+mZ1
        mZ_eee  = eZ.mass
        mZ_mmm  = mZ.mass

        # And for the W boson
        ZmmIndices = mm_pairs_index[ZmmMask]
        ZeeIndices = ee_pairs_index[ZeeMask]
        eW = eee[~ZeeIndices.e0 | ~ZeeIndices.e1]
        mW = mmm[~ZmmIndices.m0 | ~ZmmIndices.m1]

        triElec = eee_leps.e0+eee_leps.e1+eee_leps.e2
        triMuon = mmm_leps.m0+mmm_leps.m1+mmm_leps.m2
        m3l_eee = triElec.mass
        m3l_mmm = triMuon.mass


        ##################################################################
        ### >=4 leptons
        ##################################################################

        # 4lep cat
        is4lmask = ((nElec+nMuon)>=4)
        muon_4l = mu[(is4lmask)&(mu.pt>-1)]
        elec_4l =  e[(is4lmask)&( e.pt>-1)]
        # selecting 4 leading leptons
        leptons=ak.concatenate([e,mu], axis=-1)
        leptons_sorted=leptons[ak.argsort(leptons.pt, axis=-1,ascending=False)]
        lep4l=leptons_sorted[:,0:4]
        e4l=lep4l[abs(lep4l.pdgId)==11]
        mu4l=lep4l[abs(lep4l.pdgId)==13]
        nElec4l=ak.num(e4l)
        nMuon4l=ak.num(mu4l)
       
        #eemm
        muon_eemm = mu4l[(nElec4l==2)&(nMuon4l==2)&(mu4l.pt>-1)]
        elec_eemm =  e4l[(nElec4l==2)&(nMuon4l==2)&( e4l.pt>-1)]
        ee_eemm = ak.combinations(elec_eemm, 2, fields=["e0", "e1"])
        mm_eemm = ak.combinations(muon_eemm, 2, fields=["m0", "m1"])
        lepSF_eemm = GetLeptonSF(ee_eemm.e0.pt, ee_eemm.e0.eta, 'e', ee_eemm.e1.pt, ee_eemm.e1.eta, 'e', mm_eemm.m0.pt, mm_eemm.m0.eta, 'm',mm_eemm.m1.pt, mm_eemm.m1.eta, 'm', year)
        lepSF_eemm_up = GetLeptonSF(ee_eemm.e0.pt, ee_eemm.e0.eta, 'e', ee_eemm.e1.pt, ee_eemm.e1.eta, 'e', mm_eemm.m0.pt, mm_eemm.m0.eta, 'm',mm_eemm.m1.pt, mm_eemm.m1.eta, 'm', year, sys=1)
        lepSF_eemm_down = GetLeptonSF(ee_eemm.e0.pt, ee_eemm.e0.eta, 'e', ee_eemm.e1.pt, ee_eemm.e1.eta, 'e', mm_eemm.m0.pt, mm_eemm.m0.eta, 'm',mm_eemm.m1.pt, mm_eemm.m1.eta, 'm', year, sys=-1)
        #eeem
        muon_eeem = mu4l[(nElec4l==3)&(nMuon4l==1)&(mu4l.pt>-1)]
        elec_eeem =  e4l[(nElec4l==3)&(nMuon4l==1)&( e4l.pt>-1)]
        eee_eeem = ak.combinations(elec_eeem, 3, fields=["e0", "e1", "e2"])
        lepSF_eeem = GetLeptonSF(eee_eeem.e0.pt, eee_eeem.e0.eta, 'e', eee_eeem.e1.pt, eee_eeem.e1.eta, 'e', eee_eeem.e2.pt, eee_eeem.e2.eta, 'e',muon_eeem.pt, muon_eeem.eta, 'm', year)
        lepSF_eeem_up = GetLeptonSF(eee_eeem.e0.pt, eee_eeem.e0.eta, 'e', eee_eeem.e1.pt, eee_eeem.e1.eta, 'e', eee_eeem.e2.pt, eee_eeem.e2.eta, 'e',muon_eeem.pt, muon_eeem.eta, 'm', year, sys=1)
        lepSF_eeem_down = GetLeptonSF(eee_eeem.e0.pt, eee_eeem.e0.eta, 'e', eee_eeem.e1.pt, eee_eeem.e1.eta, 'e', eee_eeem.e2.pt, eee_eeem.e2.eta, 'e',muon_eeem.pt, muon_eeem.eta, 'm', year, sys=-1)
        #mmme
        muon_mmme = mu4l[(nElec4l==1)&(nMuon4l==3)&(mu4l.pt>-1)]
        elec_mmme =  e4l[(nElec4l==1)&(nMuon4l==3)&( e4l.pt>-1)]
        mmm_mmme = ak.combinations(muon_eeem, 3, fields=["m0", "m1", "m2"])
        lepSF_mmme = GetLeptonSF(mmm_mmme.m0.pt, mmm_mmme.m0.eta, 'm', mmm_mmme.m1.pt, mmm_mmme.m1.eta, 'm', mmm_mmme.m2.pt, mmm_mmme.m2.eta, 'm',elec_mmme.pt, elec_mmme.eta, 'e', year)
        lepSF_mmme_up = GetLeptonSF(mmm_mmme.m0.pt, mmm_mmme.m0.eta, 'm', mmm_mmme.m1.pt, mmm_mmme.m1.eta, 'm', mmm_mmme.m2.pt, mmm_mmme.m2.eta, 'm',elec_mmme.pt, elec_mmme.eta, 'e', year, sys=1)
        lepSF_mmme_down = GetLeptonSF(mmm_mmme.m0.pt, mmm_mmme.m0.eta, 'm', mmm_mmme.m1.pt, mmm_mmme.m1.eta, 'm', mmm_mmme.m2.pt, mmm_mmme.m2.eta, 'm',elec_mmme.pt, elec_mmme.eta, 'e', year, sys=-1)
        #mmmm and eeee
        muon_mmmm = mu4l[(nElec4l==0)&(nMuon4l==4)&(mu4l.pt>-1)]
        elec_eeee =  e4l[(nElec4l==4)&(nMuon4l==0)&( e4l.pt>-1)]
        eeee = ak.combinations(elec_eeee, 4, fields=["e0", "e1", "e2","e3"])
        mmmm = ak.combinations(muon_mmmm, 4, fields=["m0", "m1", "m2","m3"])
        lepSF_mmmm = GetLeptonSF(mmmm.m0.pt, mmmm.m0.eta, 'm', mmmm.m1.pt, mmmm.m1.eta, 'm', mmmm.m2.pt, mmmm.m2.eta, 'm', mmmm.m3.pt, mmmm.m3.eta, 'm', year)
        lepSF_mmmm_up = GetLeptonSF(mmmm.m0.pt, mmmm.m0.eta, 'm', mmmm.m1.pt, mmmm.m1.eta, 'm', mmmm.m2.pt, mmmm.m2.eta, 'm', mmmm.m3.pt, mmmm.m3.eta, 'm', year, sys=1)
        lepSF_mmmm_down = GetLeptonSF(mmmm.m0.pt, mmmm.m0.eta, 'm', mmmm.m1.pt, mmmm.m1.eta, 'm', mmmm.m2.pt, mmmm.m2.eta, 'm', mmmm.m3.pt, mmmm.m3.eta, 'm', year, sys=-1)
        lepSF_eeee = GetLeptonSF(eeee.e0.pt, eeee.e0.eta, 'e', eeee.e1.pt, eeee.e1.eta, 'e', eeee.e2.pt, eeee.e2.eta, 'e', eeee.e3.pt, eeee.e3.eta, 'e', year)
        lepSF_eeee_up = GetLeptonSF(eeee.e0.pt, eeee.e0.eta, 'e', eeee.e1.pt, eeee.e1.eta, 'e', eeee.e2.pt, eeee.e2.eta, 'e', eeee.e3.pt, eeee.e3.eta, 'e', year, sys=1)
        lepSF_eeee_down = GetLeptonSF(eeee.e0.pt, eeee.e0.eta, 'e', eeee.e1.pt, eeee.e1.eta, 'e', eeee.e2.pt, eeee.e2.eta, 'e', eeee.e3.pt, eeee.e3.eta, 'e', year, sys=-1)

        # Triggers
        trig_eeSS = passTrigger(events,'ee',isData,dataset)
        trig_mmSS = passTrigger(events,'mm',isData,dataset)
        trig_emSS = passTrigger(events,'em',isData,dataset)
        trig_eee  = passTrigger(events,'eee',isData,dataset)
        trig_mmm  = passTrigger(events,'mmm',isData,dataset)
        trig_eem  = passTrigger(events,'eem',isData,dataset)
        trig_mme  = passTrigger(events,'mme',isData,dataset)
        trig_4l   = triggerFor4l(events, nMuon, nElec, isData, dataset)

        # MET filters

        # Weights
        genw = np.ones_like(events['event']) if (isData or len(self._wc_names_lst)>0) else events['genWeight']

        ### We need weights for: normalization, lepSF, triggerSF, pileup, btagSF...
        weights = {}
        for r in ['all', 'ee', 'mm', 'em', 'eee', 'mmm', 'eem', 'mme', 'eeee','eeem','eemm','mmme','mmmm']:
          weights[r] = coffea.analysis_tools.Weights(len(events))
          if len(self._wc_names_lst) > 0: sow = 1. # Not valid in nanoAOD for EFT samples, MUST use SumOfEFTweights at analysis level
          weights[r].add('norm',genw if isData else (xsec/sow)*genw)
          weights[r].add('btagSF', btagSF, btagSFUp, btagSFDo)
        
        weights['ee'].add('lepSF', lepSF_eeSS, lepSF_eeSS_up, lepSF_eeSS_down)
        weights['em'].add('lepSF', lepSF_emSS,lepSF_emSS_up, lepSF_emSS_down)
        weights['mm'].add('lepSF', lepSF_mumuSS, lepSF_mumuSS_up, lepSF_mumuSS_down)
        weights['eee'].add('lepSF', lepSF_eee, lepSF_eee_up, lepSF_eee_down)
        weights['mmm'].add('lepSF', lepSF_mmm, lepSF_mmm_up, lepSF_mmm_down)
        weights['mme'].add('lepSF', lepSF_mme, lepSF_mme_up, lepSF_mme_down)
        weights['eem'].add('lepSF', lepSF_eem, lepSF_eem_up, lepSF_eem_down)
        weights['eeee'].add('lepSF', lepSF_eeee, lepSF_eeee_up, lepSF_eeee_down)
        weights['mmmm'].add('lepSF', lepSF_mmmm, lepSF_mmmm_up, lepSF_mmmm_down)
        weights['mmme'].add('lepSF', lepSF_mmme, lepSF_mmme_up, lepSF_mmme_down)
        weights['eeem'].add('lepSF', lepSF_eeem, lepSF_eeem_up, lepSF_eeem_down)
        weights['eemm'].add('lepSF', lepSF_eemm, lepSF_eemm_up, lepSF_eemm_down)
        
        # Extract the EFT quadratic coefficients and optionally use them to calculate the coefficients on the w**2 quartic function
        # eft_coeffs is never Jagged so convert immediately to numpy for ease of use.
        eft_coeffs = ak.to_numpy(events['EFTfitCoefficients']) if hasattr(events, "EFTfitCoefficients") else None
        eft_w2_coeffs = self._eft_helper.calc_w2_coeffs(eft_coeffs) if (self._do_errors and eft_coeffs is not None) else None

        # Selections and cuts
        selections = PackedSelection()
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
        
        channels4L =['eeee','eeem','eemm','mmme','mmmm']
        selections.add('eeee',((nElec4l==4)&(nMuon4l==0))&(trig_4l))
        selections.add('eeem',((nElec4l==3)&(nMuon4l==1))&(trig_4l))
        selections.add('eemm',((nElec4l==2)&(nMuon4l==2))&(trig_4l))
        selections.add('mmme',((nElec4l==1)&(nMuon4l==3))&(trig_4l))
        selections.add('mmmm',((nElec4l==0)&(nMuon4l==4))&(trig_4l))
        
        selections.add('ch+', (sumcharge>0))
        selections.add('ch-', (sumcharge<0))

        levels = ['base', '2jets', '4jets', '4j1b', '4j2b']
        selections.add('base', (nElec+nMuon>=2))
        selections.add('2jets',(njets>=2))
        selections.add('4jets',(njets>=4))
        selections.add('4j1b',(njets>=4)&(nbtags>=1))
        selections.add('4j2b',(njets>=4)&(nbtags>=2))

        # Variables
        invMass_eeSSonZ  = ( eeSSonZ.e0+ eeSSonZ.e1).mass
        invMass_eeSSoffZ = (eeSSoffZ.e0+eeSSoffZ.e1).mass
        invMass_mmSSonZ  = ( mmSSonZ.m0+ mmSSonZ.m1).mass
        invMass_mmSSoffZ = (mmSSoffZ.m0+mmSSoffZ.m1).mass
        invMass_emSS     = (emSS.e+emSS.m).mass

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
        varnames['counts'] = np.ones_like(events['event'])

        # systematics
        systList = []
        if isData==False:
          systList = ['nominal']
          if self._do_systematics: systList = systList + ['lepSFUp','lepSFDown','btagSFUp', 'btagSFDown']
        else:
          systList = ['noweight']
        # fill Histos
        hout = self.accumulator.identity()
        normweights = weights['all'].weight().flatten() # Why does it not complain about .flatten() here?
        sowweights = np.ones_like(normweights) if len(self._wc_names_lst)>0 else normweights
        hout['SumOfEFTweights'].fill(sample=dataset, SumOfEFTweights=varnames['counts'], weight=sowweights, eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
    
        for syst in systList:
         for var, v in varnames.items():
          for ch in channels2LSS+channels3L+channels4L:
           for sumcharge in ['ch+', 'ch-']:
            for lev in levels:
             #find the event weight to be used when filling the histograms    
             weightSyst = syst
             #in the case of 'nominal', or the jet energy systematics, no weight systematic variation is used (weightSyst=None)
             if syst in ['nominal','JERUp','JERDown','JESUp','JESDown']:
              weightSyst = None # no weight systematic for these variations
             if syst=='noweight':
                weight = np.ones(len(events)) # for data
             else:
              # call weights.weight() with the name of the systematic to be varied
              if ch in channels3L: ch_w= ch[:3]
              elif ch in channels2LSS: ch_w =ch[:2]
              else: ch_w=ch
              weight = weights['all'].weight(weightSyst) if isData else weights[ch_w].weight(weightSyst)
             cuts = [ch] + [lev] + [sumcharge]
             cut = selections.all(*cuts)
             weights_flat = weight[cut].flatten() # Why does it not complain about .flatten() here?
             weights_ones = np.ones_like(weights_flat, dtype=np.int)
             eft_coeffs_cut = eft_coeffs[cut] if eft_coeffs is not None else None
             eft_w2_coeffs_cut = eft_w2_coeffs[cut] if eft_w2_coeffs is not None else None
             
             # filling histos
             if var == 'invmass':
              if ((ch in ['eeeSSoffZ', 'mmmSSoffZ','eeeSSonZ', 'mmmSSonZ']) or (ch in channels4L)): continue
              else                                 : values = ak.flatten(v[ch][cut])
              hout['invmass'].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, invmass=values, weight=weights_flat, systematic=syst)
             elif var == 'm3l': 
              if ((ch in channels2LSS) or (ch in ['eeeSSoffZ', 'mmmSSoffZ', 'eeeSSonZ' , 'mmmSSonZ']) or (ch in channels4L)): continue
              values = ak.flatten(v[ch][cut])
              hout['m3l'].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, m3l=values, weight=weights_flat, systematic=syst)
             else:
              values = v[cut] 
              if   var == 'ht'    : hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, ht=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'met'   : hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, met=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'njets' : hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, njets=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'nbtags': 
                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, nbtags=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
                hout['njetsnbtags'].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, njets=varnames['njets'][cut], nbtags=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'counts': hout[var].fill(counts=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_ones, systematic=syst)
              elif var == 'j0eta' : 
                if lev == 'base': continue
                values = ak.flatten(values)
                #values=np.asarray(values)
                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, j0eta=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'e0pt'  : 
                if ch in ['mmSSonZ', 'mmSSoffZ', 'mmmSSoffZ', 'mmmSSonZ','mmmm']: continue
                values = ak.flatten(values)
                #values=np.asarray(values)
                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, e0pt=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst) # Crashing here, not sure why. Related to values?
              elif var == 'm0pt'  : 
                if ch in ['eeSSonZ', 'eeSSoffZ', 'eeeSSoffZ', 'eeeSSonZ', 'eeee']: continue
                values = ak.flatten(values)
                #values=np.asarray(values)
                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, m0pt=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'e0eta' : 
                if ch in ['mmSSonZ', 'mmSSoffZ', 'mmmSSoffZ', 'mmmSSonZ', 'mmmm']: continue
                values = ak.flatten(values)
                #values=np.asarray(values)
                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, e0eta=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'm0eta':
                if ch in ['eeSSonZ', 'eeSSoffZ', 'eeeSSoffZ', 'eeeSSonZ', 'eeee']: continue
                values = ak.flatten(values)
                #values=np.asarray(values)
                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, m0eta=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'j0pt'  : 
                if lev == 'base': continue
                values = ak.flatten(values)
                #values=np.asarray(values)
                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, j0pt=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
        return hout

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    # Load the .coffea files
    outpath= './coffeaFiles/'
    samples     = load(outpath+'samples.coffea')
    topprocessor = AnalysisProcessor(samples)

