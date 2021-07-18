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
from topcoffea.modules.corrections import SFevaluator, GetBTagSF, jet_factory, GetBtagEff, AttachMuonSF, AttachElectronSF, AttachPerLeptonFR, fakeRateWeight2l, fakeRateWeight3l
from topcoffea.modules.selection import *
from topcoffea.modules.HistEFT import HistEFT
import topcoffea.modules.eft_helper as efth

#coffea.deprecations_as_errors = True


def add2lssMaskAndSFs(events, year, isData):
    FOs=events.l_fo_conept_sorted
    filter_flags=events.Flag
    filters=filter_flags.goodVertices & filter_flags.globalSuperTightHalo2016Filter & filter_flags.HBHENoiseFilter & filter_flags.HBHENoiseIsoFilter & filter_flags.EcalDeadCellTriggerPrimitiveFilter & filter_flags.BadPFMuonFilter & ((year == 2016) | filter_flags.ecalBadCalibFilter) & (isData | filter_flags.eeBadScFilter)
    cleanup=events.minMllAFAS > 12
    dilep  = ( ak.num(FOs)) >= 2 
    pt2515=ak.any(FOs[:,0:1].conept > 25.0, axis=1) & ak.any(FOs[:,1:2].conept > 15.0, axis=1)
    exclusive=ak.num( FOs[FOs.isTightLep],axis=-1)<3
    padded_FOs=ak.pad_none(FOs, 2)
    Zee_veto= (abs(padded_FOs[:,0].pdgId) != 11) | (abs(padded_FOs[:,1].pdgId) != 11) | ( abs ( (padded_FOs[:,0]+padded_FOs[:,1]).mass -91.2) > 10)
    # Z_veto=abs(events.mZ1-91.2)>10 not working yet :( 
    eleID1=(abs(padded_FOs[:,0].pdgId)!=11) | ((padded_FOs[:,0].convVeto != 0) & (padded_FOs[:,0].lostHits==0) & (padded_FOs[:,0].tightCharge>=2))
    eleID2=(abs(padded_FOs[:,1].pdgId)!=11) | ((padded_FOs[:,1].convVeto != 0) & (padded_FOs[:,1].lostHits==0) & (padded_FOs[:,1].tightCharge>=2))
    muTightCharge=((abs(padded_FOs[:,0].pdgId)!=13) | (padded_FOs[:,0].tightCharge>=1)) & ((abs(padded_FOs[:,1].pdgId)!=13) | (padded_FOs[:,1].tightCharge>=1))
    njet4=(events.njets>3)
    mask=(filters & cleanup & dilep & pt2515 & exclusive & Zee_veto & eleID1 & eleID2 & muTightCharge & njet4) #     & Z_veto
    events['is2lss']=ak.fill_none(mask,False)
    events['sf_2lss']=padded_FOs[:,0].sf_nom*padded_FOs[:,1].sf_nom
    events['sf_2lss_hi']=padded_FOs[:,0].sf_hi*padded_FOs[:,1].sf_hi
    events['sf_2lss_lo']=padded_FOs[:,0].sf_lo*padded_FOs[:,1].sf_lo
    fakeRateWeight2l(events, padded_FOs[:,0], padded_FOs[:,1])

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, wc_names_lst=[], do_errors=False, do_systematics=False, dtype=np.float32):
        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

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
        'l0pt'    : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("l0pt",   "Leading lep $p_{T}$ (GeV)", 25, 0, 500)),
        'j0pt'    : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("j0pt",   "Leading jet  $p_{T}$ (GeV)", 25, 0, 500)),
        'e0eta'   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("e0eta",  "Leading elec $\eta$", 30, -3.0, 3.0)),
        'm0eta'   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("m0eta",  "Leading muon $\eta$", 30, -3.0, 3.0)),
        'l0eta'   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("l0eta",  "Leading lep $\eta$", 30, -3.0, 3.0)),
        'j0eta'   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("j0eta",  "Leading jet  $\eta$", 30, -3.0, 3.0)),
        'ht'      : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Cat("sumcharge", "sumcharge"), hist.Cat("systematic", "Systematic Uncertainty"), hist.Bin("ht",     "H$_{T}$ (GeV)", 50, 0, 1000)),
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
        histAxisName = self._samples[dataset]['histAxisName']
        year         = self._samples[dataset]['year']
        xsec         = self._samples[dataset]['xsec']
        sow          = self._samples[dataset]['nSumOfWeights' ]
        isData       = self._samples[dataset]['isData']
        datasets     = ['SingleMuon', 'SingleElectron', 'EGamma', 'MuonEG', 'DoubleMuon', 'DoubleElectron']
        for d in datasets: 
          if d in dataset: dataset = dataset.split('_')[0] 

        # Initialize objects
        met  = events.MET
        e    = events.Electron
        mu   = events.Muon
        tau  = events.Tau
        jets = events.Jet
 
        e['idEmu'] = ttH_idEmu_cuts_E3(e.hoe, e.eta, e.deltaEtaSC, e.eInvMinusPInv, e.sieie)
        e['conept'] = coneptElec(e.pt, e.mvaTTH, e.jetRelIso)
        mu['conept'] = coneptMuon(mu.pt, mu.mvaTTH, mu.jetRelIso, mu.mediumId)
        e['btagDeepFlavB'] = ak.fill_none(e.matched_jet.btagDeepFlavB, -99)
        mu['btagDeepFlavB'] = ak.fill_none(mu.matched_jet.btagDeepFlavB, -99)

        # Muon selection
        mu['isPres'] = isPresMuon(mu.dxy, mu.dz, mu.sip3d, mu.eta, mu.pt, mu.miniPFRelIso_all)
        mu['isLooseM'] = isLooseMuon(mu.miniPFRelIso_all,mu.sip3d,mu.looseId)
        mu['isFO'] = isFOMuon(mu.pt, mu.conept, mu.btagDeepFlavB, mu.mvaTTH, mu.jetRelIso, year)
        mu['isTightLep']= tightSelMuon(mu.isFO, mu.mediumId, mu.mvaTTH)

        # Electron selection
        e['isPres'] = isPresElec(e.pt, e.eta, e.dxy, e.dz, e.miniPFRelIso_all, e.sip3d, getattr(e,"mvaFall17V2noIso_WPL"))
        e['isLooseE'] = isLooseElec(e.miniPFRelIso_all,e.sip3d,e.lostHits)
        e['isFO']  = isFOElec(e.conept, e.btagDeepFlavB, e.idEmu, e.convVeto, e.lostHits, e.mvaTTH, e.jetRelIso, e.mvaFall17V2noIso_WP80, year)
        e['isTightLep'] =  tightSelElec(e.isFO, e.mvaTTH)

        # build loose collections
        m_loose = mu[mu.isPres & mu.isLooseM]
        e_loose = e[e.isPres & e.isLooseE]
        l_loose = ak.with_name(ak.concatenate([e_loose, m_loose], axis=1), 'PtEtaPhiMCandidate')

        # compute pair invariant masses
        llpairs = ak.combinations(l_loose, 2, fields=["l0","l1"])
        events['minMllAFAS']=ak.min( (llpairs.l0+llpairs.l1).mass, axis=-1)
        osllpairs=llpairs[llpairs.l0.charge*llpairs.l1.charge<0]
        osllpairs_masses=(osllpairs.l0+osllpairs.l1).mass
        events['mZ1']=osllpairs_masses[ak.argmin( abs(osllpairs_masses-91.2), axis=-1)] # needs to be fixed, but not used yet

        # Build FO collection
        m_fo = mu[mu.isPres & mu.isLooseM & mu.isFO]
        e_fo = e[e.isPres & e.isLooseE & e.isFO]

        # Attach the lepton SFs to the electron and muons collections
        AttachElectronSF(e_fo,year=year)
        AttachMuonSF(m_fo,year=year)

        AttachPerLeptonFR(e_fo, flavor='Elec', year=year)
        AttachPerLeptonFR(m_fo, flavor='Muon', year=year)
        m_fo['convVeto']=ak.ones_like(m_fo.charge); 
        m_fo['lostHits']=ak.zeros_like(m_fo.charge); 
        l_fo = ak.with_name(ak.concatenate([e_fo, m_fo], axis=1), 'PtEtaPhiMCandidate')
        l_fo_conept_sorted = l_fo[ak.argsort(l_fo.conept, axis=-1,ascending=False)]

        ## Attach per lepton fake rates
        
        
        # Tau selection
        tau['isPres']  = isPresTau(tau.pt, tau.eta, tau.dxy, tau.dz, tau.idDecayModeNewDMs, tau.idDeepTau2017v2p1VSjet, minpt=20)
        tau['isClean'] = isClean(tau, l_loose, drmin=0.3)

        tau['isGood']  =  tau['isClean']  & tau['isPres']
        tau= tau[tau.isGood] # use these to clean jets
        tau['isTight']= isTightTau(tau.idDeepTau2017v2p1VSjet) # use these to veto

        # Jet cleaning, before any jet selection
        vetos_tocleanjets= ak.with_name( ak.concatenate([tau, l_fo], axis=1), 'PtEtaPhiMCandidate')
        tmp = ak.cartesian([ak.local_index(jets.pt), vetos_tocleanjets.jetIdx], nested=True)
        cleanedJets = jets[~ak.any(tmp.slot0 == tmp.slot1, axis=-1)] # this line should go before *any selection*, otherwise lep.jetIdx is not aligned with the jet index
        cleanedJets['isClean'] = isClean(cleanedJets, tau, drmin=0.3)
        cleanedJets=cleanedJets[cleanedJets.isClean]

        # Selecting jets and cleaning them
        jetptname = 'pt_nom' if hasattr(cleanedJets, 'pt_nom') else 'pt'

        ### Jet energy corrections
        if False: # for synch
          cleanedJets["pt_raw"]=(1 - cleanedJets.rawFactor)*cleanedJets.pt
          cleanedJets["mass_raw"]=(1 - cleanedJets.rawFactor)*cleanedJets.mass
          cleanedJets["pt_gen"]=ak.values_astype(ak.fill_none(cleanedJets.matched_gen.pt, 0), np.float32)
          cleanedJets["rho"]= ak.broadcast_arrays(events.fixedGridRhoFastjetAll, cleanedJets.pt)[0]
          events_cache = events.caches[0]
          corrected_jets = jet_factory.build(cleanedJets, lazy_cache=events_cache)
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

        cleanedJets['isGood']  = isTightJet(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, jetPtCut=25.) # temporary at 25 for synch
        goodJets = cleanedJets[cleanedJets.isGood]
        
        
        
        
        # count jets, jet 
        njets = ak.num(goodJets)
        ht = ak.sum(goodJets.pt,axis=-1)
        j0 = goodJets[ak.argmax(goodJets.pt,axis=-1,keepdims=True)]
        
        # to do: check these numbers are ok
        if year == 2017: btagwpl = 0.0532 #WP loose 
        else: btagwpl = 0.0490 #WP loose 
        isBtagJetsLoose = (goodJets.btagDeepB > btagwpl)
        isNotBtagJetsLoose = np.invert(isBtagJetsLoose)
        nbtagsl = ak.num(goodJets[isBtagJetsLoose])
        # Medium DeepJet WP
        if year == 2017: btagwpm = 0.3040 #WP medium
        else: btagwpm = 0.2783 #WP medium
        isBtagJetsMedium = (goodJets.btagDeepB > btagwpm)
        isNotBtagJetsMedium = np.invert(isBtagJetsMedium)
        nbtagsm = ak.num(goodJets[isBtagJetsMedium])

        ## Add the variables needed for event selection as columns to event, so they persist
        events['njets']=njets
        events['l_fo_conept_sorted']=l_fo_conept_sorted
        
        l_fo_conept_sorted_padded=ak.pad_none(l_fo_conept_sorted, 3)
        l0=l_fo_conept_sorted_padded[:,0]
        l1=l_fo_conept_sorted_padded[:,1]
        l2=l_fo_conept_sorted_padded[:,2]

        add2lssMaskAndSFs(events, year, isData)
        print('the number of events passing all cuts is', ak.num(events[events.is2lss],axis=0))
        events['l0']=l0; events['l1']=l1 # remove this 
        theevents=events[events.is2lss]
        #for fr, thel0, thel1 in zip(theevents['fakefactor_2l'], theevents['l0'], theevents['l1']):
        #  print(fr, thel0.isTightLep, thel1.isTightLep)

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
   
          pMC     = ak.prod(bJetEff       [isBtagJetsMedium], axis=-1) * ak.prod((1-bJetEff       [isNotBtagJetsMedium]), axis=-1)
          pData   = ak.prod(bJetEff_data  [isBtagJetsMedium], axis=-1) * ak.prod((1-bJetEff_data  [isNotBtagJetsMedium]), axis=-1)
          pDataUp = ak.prod(bJetEff_dataUp[isBtagJetsMedium], axis=-1) * ak.prod((1-bJetEff_dataUp[isNotBtagJetsMedium]), axis=-1)
          pDataDo = ak.prod(bJetEff_dataDo[isBtagJetsMedium], axis=-1) * ak.prod((1-bJetEff_dataDo[isNotBtagJetsMedium]), axis=-1)

          pMC      = ak.where(pMC==0,1,pMC) # removeing zeroes from denominator...
          btagSF   = pData  /pMC
          btagSFUp = pDataUp/pMC
          btagSFDo = pDataUp/pMC


        #sumcharge = ak.sum(e.charge, axis=-1)


        # Not sure I understand this for loop, it seems all the vals in the dict are the same, so why do we need a dict
        ### We need weights for: normalization, lepSF, triggerSF, pileup, btagSF...
        genw = np.ones_like(events['event']) if (isData or len(self._wc_names_lst)>0) else events['genWeight']
        weights = {}
        for r in ['2lss0tau', 'all', 'ee', 'mm', 'em', 'eee', 'mmm', 'eem', 'mme', 'eeee','eeem','eemm','mmme','mmmm']:
          # weights[r] = coffea.analysis_tools.Weights(len(events))
          weights[r] = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
          if len(self._wc_names_lst) > 0: sow = np.ones_like(sow) # Not valid in nanoAOD for EFT samples, MUST use SumOfEFTweights at analysis level
          weights[r].add('norm',genw if isData else (xsec/sow)*genw)
          weights[r].add('btagSF', btagSF, btagSFUp, btagSFDo)
          weights[r].add('lepSF', events.sf_2lss,  events.sf_2lss_hi,events.sf_2lss_lo)

        # Extract the EFT quadratic coefficients and optionally use them to calculate the coefficients on the w**2 quartic function
        # eft_coeffs is never Jagged so convert immediately to numpy for ease of use.
        eft_coeffs = ak.to_numpy(events['EFTfitCoefficients']) if hasattr(events, "EFTfitCoefficients") else None
        if eft_coeffs is not None:
            # Check to see if the ordering of WCs for this sample matches what want
            if self._samples[dataset]['WCnames'] != self._wc_names_lst:
                eft_coeffs = efth.remap_coeffs(self._samples[dataset]['WCnames'], self._wc_names_lst, eft_coeffs)
        eft_w2_coeffs = efth.calc_w2_coeffs(eft_coeffs,self._dtype) if (self._do_errors and eft_coeffs is not None) else None

        

        # Selections and cuts
        selections = PackedSelection(dtype='uint64')
        is2lss=ak.values_astype(events.is2lss,'bool')
        selections.add('2lss0tau', is2lss)

        # b jet masks
        bmask_atleast1med_atleast2loose = ((nbtagsm>=1)&(nbtagsl>=2)) # This is the requirement for 2lss and 4l
        bmask_exactly1med = (nbtagsm==1) # Used for 3l
        bmask_atleast2med = (nbtagsm>=2) # Used for 3l

        # Charge masks
        sumcharge = (l_fo_conept_sorted_padded.charge[:,0]+l_fo_conept_sorted_padded.charge[:,1])
        sumcharge_0 = ak.fill_none(sumcharge==0,False)
        sumcharge_p = ak.fill_none(sumcharge==1,False)
        sumcharge_m = ak.fill_none(sumcharge==-1,False)

        # Channels for the 2lss cat
        # Not sure if this is right (or not the way we want to do it)? 
        channels2LSS  = ["2lss_4j_p","2lss_5j_p","2lss_6j_p","2lss_7j_p","2lss_4j_m","2lss_5j_m","2lss_6j_m","2lss_7j_m"]
        selections.add("2lss_4j_p", (is2lss & (njets==4) & sumcharge_p & bmask_atleast1med_atleast2loose))
        selections.add("2lss_5j_p", (is2lss & (njets==5) & sumcharge_p & bmask_atleast1med_atleast2loose))
        selections.add("2lss_6j_p", (is2lss & (njets==6) & sumcharge_p & bmask_atleast1med_atleast2loose))
        selections.add("2lss_7j_p", (is2lss & (njets>=7) & sumcharge_p & bmask_atleast1med_atleast2loose))
        selections.add("2lss_4j_m", (is2lss & (njets==4) & sumcharge_m & bmask_atleast1med_atleast2loose))
        selections.add("2lss_5j_m", (is2lss & (njets==5) & sumcharge_m & bmask_atleast1med_atleast2loose))
        selections.add("2lss_6j_m", (is2lss & (njets==6) & sumcharge_m & bmask_atleast1med_atleast2loose))
        selections.add("2lss_7j_m", (is2lss & (njets>=7) & sumcharge_m & bmask_atleast1med_atleast2loose))

        varnames = {}
        varnames['ht']      = ht
        varnames['e0pt' ]  = l0.pt  # update
        varnames['e0eta']  = l0.eta # update
        varnames['m0pt' ]  = l0.pt  # update
        varnames['m0eta']  = l0.eta # update
        varnames['l0pt']  = l0.pt
        varnames['l0eta'] = l0.eta
        varnames['j0pt' ]  = j0.pt
        varnames['j0eta']  = j0.eta
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
        hout['SumOfEFTweights'].fill(sample=histAxisName, SumOfEFTweights=varnames['counts'], weight=sowweights, eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)
    
        for syst in systList:
         for var, v in varnames.items():
          print("var:",var)
          #continue
          for ch in ['2lss0tau'] + channels2LSS:
           for sumcharge in ['']:
            for lev in ['']:
             #find the event weight to be used when filling the histograms    
             weightSyst = syst
             #in the case of 'nominal', or the jet energy systematics, no weight systematic variation is used (weightSyst=None)
             if syst in ['nominal','JERUp','JERDown','JESUp','JESDown']:
              weightSyst = None # no weight systematic for these variations
             if syst=='noweight':
                weight = np.ones(len(events)) # for data
             else:
              # call weights.weight() with the name of the systematic to be varied
              # if ch in channels3L: ch_w= ch[:3]
              # elif ch in channels2LSS: ch_w =ch[:2]
              # elif ch in channels2LOS: ch_w =ch[:2]
              # else: ch_w=ch
              #ch_w=ch   
              #print(weightSyst)
              #weight = weights['all'].weight(weightSyst) if isData else weights[ch_w].weight(weightSyst)
              weight = weights['all'].weight(weightSyst) # All of the vals in the weights dict seem to be the same, so just use one that goes with the "all" key for now?
             cuts = []
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
              if ((ch in channels2LSS) or (ch in channels2LOS) or (ch in ['eeeSSoffZ', 'mmmSSoffZ', 'eeeSSonZ' , 'mmmSSonZ']) or (ch in channels4L)): continue
              values = ak.flatten(v[ch][cut])
              hout['m3l'].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, m3l=values, weight=weights_flat, systematic=syst)
             else:
              values = v[cut] 
              # These all look identical, do we need if/else here?
              if   var == 'ht'    : hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, ht=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'met'   : hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, met=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'njets' : hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, njets=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'nbtags': hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, nbtags=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'counts': hout[var].fill(counts=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_ones, systematic=syst)
              '''
              elif var == 'j0eta' : 
                if lev in ['base', 'CRZ', 'app']: continue
                values = ak.flatten(values)
                #values=np.asarray(values)
                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, j0eta=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'e0pt'  : 
                if ch in ['mmSSonZ', 'mmOSonZ', 'mmSSoffZ', 'mmOSoffZ', 'mmmSSoffZ', 'mmmSSonZ','mmmm']: continue
                values = ak.flatten(values)
                #values=np.asarray(values)
                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, e0pt=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst) # Crashing here, not sure why. Related to values?
              elif var == 'm0pt'  : 
                if ch in ['eeSSonZ', 'eeOSonZ', 'eeSSoffZ', 'eeOSoffZ', 'eeeSSoffZ', 'eeeSSonZ', 'eeee']: continue
                values = ak.flatten(values)
                #values=np.asarray(values)
                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, m0pt=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'l0pt'  : 
                values = ak.flatten(values)
                #values=np.asarray(values)
                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, l0pt=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'e0eta' : 
                if ch in ['mmSSonZ', 'mmOSonZ', 'mmSSoffZ', 'mmOSoffZ', 'mmmSSoffZ', 'mmmSSonZ', 'mmmm']: continue
                values = ak.flatten(values)
                #values=np.asarray(values)
                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, e0eta=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'm0eta':
                if ch in ['eeSSonZ', 'eeOSonZ', 'eeSSoffZ', 'eeOSoffZ', 'eeeSSoffZ', 'eeeSSonZ', 'eeee']: continue
                values = ak.flatten(values)
                #values=np.asarray(values)
                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, m0eta=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'l0eta'  : 
                values = ak.flatten(values)
                #values=np.asarray(values)
                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, l0eta=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              elif var == 'j0pt'  : 
                if lev in ['base', 'CRZ', 'app']: continue
                values = ak.flatten(values)
                #values=np.asarray(values)
                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, j0pt=values, sample=histAxisName, channel=ch, cut=lev, sumcharge=sumcharge, weight=weights_flat, systematic=syst)
              '''
        return hout

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    # Load the .coffea files
    outpath= './coffeaFiles/'
    samples     = load(outpath+'samples.coffea')
    topprocessor = AnalysisProcessor(samples)

