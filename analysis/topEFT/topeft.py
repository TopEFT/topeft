#!/usr/bin/env python
import lz4.frame as lz4f
import cloudpickle
import json
import pprint
import coffea
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import hist, processor
from coffea.util import load, save
from optparse import OptionParser
from coffea.analysis_tools import PackedSelection

from topcoffea.modules.GetValuesFromJsons import get_cut
from topcoffea.modules.objects import *
from topcoffea.modules.corrections import SFevaluator, GetBTagSF, jet_factory, GetBtagEff, AttachMuonSF, AttachElectronSF, AttachPerLeptonFR
from topcoffea.modules.selection import *
from topcoffea.modules.HistEFT import HistEFT
import topcoffea.modules.eft_helper as efth

class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], do_errors=False, do_systematics=False, dtype=np.float32):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        # Create the histograms
        self._accumulator = processor.dict_accumulator({
        'SumOfEFTweights'  : HistEFT("SumOfWeights", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("SumOfEFTweights", "sow", 1, 0, 2)),
        'counts'  : hist.Hist("Events",             hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("counts", "Counts", 1, 0, 2)),
        'invmass' : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 20, 0, 200)),
        'ptbl'     : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("ptbl",    "$p_{T}^{b\mathrm{-}jet+\ell_{min(dR)}}$ (GeV) ", 100, 0, 1000)),
        'invmass' : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ",50 , 60, 130)),
        'njets'   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("njets",  "Jet multiplicity ", 10, 0, 10)),
        'nbtags'  : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("nbtags", "btag multiplicity ", 5, 0, 5)),
        'met'     : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("met",    "MET (GeV)", 40, 0, 400)),
        'm3l'     : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("m3l",    "$m_{3\ell}$ (GeV) ", 50, 0, 500)),
        'wleppt'  : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("wleppt", "$p_{T}^{lepW}$ (GeV) ", 20, 0, 200)),
        'l0pt'    : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("l0pt",   "Leading lep $p_{T}$ (GeV)", 25, 0, 500)),
        'j0pt'    : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("j0pt",   "Leading jet  $p_{T}$ (GeV)", 25, 0, 500)),
        'l0eta'   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("l0eta",  "Leading lep $\eta$", 30, -3.0, 3.0)),
        'j0eta'   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("j0eta",  "Leading jet  $\eta$", 30, -3.0, 3.0)),
        'ht'      : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("ht",     "H$_{T}$ (GeV)", 200, 0, 2000)),
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
        datasets     = ['SingleMuon', 'SingleElectron', 'EGamma', 'MuonEG', 'DoubleMuon', 'DoubleElectron', 'DoubleEG']
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

        # Build loose collections
        m_loose = mu[mu.isPres & mu.isLooseM]
        e_loose = e[e.isPres & e.isLooseE]
        l_loose = ak.with_name(ak.concatenate([e_loose, m_loose], axis=1), 'PtEtaPhiMCandidate')

        # Compute pair invariant masses
        llpairs = ak.combinations(l_loose, 2, fields=["l0","l1"])
        events['minMllAFAS']=ak.min( (llpairs.l0+llpairs.l1).mass, axis=-1)
        osllpairs=llpairs[llpairs.l0.charge*llpairs.l1.charge<0]
        osllpairs_masses=(osllpairs.l0+osllpairs.l1).mass

        # Build FO collection
        m_fo = mu[mu.isPres & mu.isLooseM & mu.isFO]
        e_fo = e[e.isPres & e.isLooseE & e.isFO]

        # Attach the lepton SFs to the electron and muons collections
        AttachElectronSF(e_fo,year=year)
        AttachMuonSF(m_fo,year=year)

        # Attach per lepton fake rates
        AttachPerLeptonFR(e_fo, flavor='Elec', year=year)
        AttachPerLeptonFR(m_fo, flavor='Muon', year=year)
        m_fo['convVeto']=ak.ones_like(m_fo.charge); 
        m_fo['lostHits']=ak.zeros_like(m_fo.charge); 
        l_fo = ak.with_name(ak.concatenate([e_fo, m_fo], axis=1), 'PtEtaPhiMCandidate')
        l_fo_conept_sorted = l_fo[ak.argsort(l_fo.conept, axis=-1,ascending=False)]

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

        # Jet energy corrections
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

        cleanedJets['isGood']  = isTightJet(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, jetPtCut=30.) # temporary at 25 for synch, TODO: Do we want 30 or 25?
        goodJets = cleanedJets[cleanedJets.isGood]

        # Count jets, jet 
        njets = ak.num(goodJets)
        ht = ak.sum(goodJets.pt,axis=-1)
        j0 = goodJets[ak.argmax(goodJets.pt,axis=-1,keepdims=True)]

        # Loose DeepJet WP
        # TODO: Update these numbers when UL16 is available, and double check UL17 and UL18 at that time as well
        if year == "2017":
            btagwpl = get_cut("btag_wp_loose_UL17")
        elif year == "2018":
            btagwpl = get_cut("btag_wp_loose_UL18")
        elif ((year=="2016") or (year=="2016APV")):
            btagwpl = get_cut("btag_wp_loose_L16")
        else:
            raise ValueError(f"Error: Unknown year \"{year}\".")
        isBtagJetsLoose = (goodJets.btagDeepFlavB > btagwpl)
        isNotBtagJetsLoose = np.invert(isBtagJetsLoose)
        nbtagsl = ak.num(goodJets[isBtagJetsLoose])

        # Medium DeepJet WP
        # TODO: Update these numbers when UL16 is available, and double check UL17 and UL18 at that time as well
        if year == "2017": 
            btagwpm = get_cut("btag_wp_medium_UL17")
        elif year == "2018":
            btagwpm = get_cut("btag_wp_medium_UL18")
        elif ((year=="2016") or (year=="2016APV")):
            btagwpm = get_cut("btag_wp_medium_L16")
        else:
            raise ValueError(f"Error: Unknown year \"{year}\".")
        isBtagJetsMedium = (goodJets.btagDeepFlavB > btagwpm)
        isNotBtagJetsMedium = np.invert(isBtagJetsMedium)
        nbtagsm = ak.num(goodJets[isBtagJetsMedium])

        ## Add the variables needed for event selection as columns to event, so they persist
        events['njets'] = njets
        events['l_fo_conept_sorted'] = l_fo_conept_sorted

        l_fo_conept_sorted_padded = ak.pad_none(l_fo_conept_sorted, 3)
        l0 = l_fo_conept_sorted_padded[:,0]
        l1 = l_fo_conept_sorted_padded[:,1]
        l2 = l_fo_conept_sorted_padded[:,2]

        add2lssMaskAndSFs(events, year, isData)
        add3lMaskAndSFs(events, year, isData)
        add4lMaskAndSFs(events, year, isData)
        print('The number of events passing fo 2lss, 3l, and 4l selection is:', ak.num(events[events.is2lss],axis=0),ak.num(events[events.is3l],axis=0),ak.num(events[events.is4l],axis=0))

        # Get mask for events that have two sf os leps close to z peak
        ll_fo_pairs = ak.combinations(l_fo_conept_sorted_padded, 2, fields=["l0","l1"])
        zpeak_mask = (abs((ll_fo_pairs.l0+ll_fo_pairs.l1).mass - 91.2)<10.0) 
        sfos_mask = (ll_fo_pairs.l0.pdgId == -ll_fo_pairs.l1.pdgId)
        sfosz_mask = ak.flatten(ak.any((zpeak_mask & sfos_mask),axis=1,keepdims=True)) # Use flatten here not because it's jagged, but because it is too nested (i.e. it looks like this [[T],[F],[T],...], and want this [T,F,T,...]))

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


        # We need weights for: normalization, lepSF, triggerSF, pileup, btagSF...
        weights_dict = {}
        genw = np.ones_like(events['event']) if (isData or len(self._wc_names_lst)>0) else events['genWeight']
        if len(self._wc_names_lst) > 0: sow = np.ones_like(sow) # Not valid in nanoAOD for EFT samples, MUST use SumOfEFTweights at analysis level
        for ch_name in ["2lss","3l","4l"]:
            weights_dict[ch_name] = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
            weights_dict[ch_name].add('norm',genw if isData else (xsec/sow)*genw)
            weights_dict[ch_name].add('btagSF', btagSF, btagSFUp, btagSFDo)
            if ch_name == "2lss":
                weights_dict[ch_name].add('lepSF', events.sf_2lss, events.sf_2lss_hi, events.sf_2lss_lo)
            if ch_name == "3l":
                weights_dict[ch_name].add('lepSF', events.sf_3l, events.sf_3l_hi, events.sf_3l_lo)
            if ch_name == "4l":
                weights_dict[ch_name].add('lepSF', events.sf_4l, events.sf_4l_hi, events.sf_4l_lo)


        # Extract the EFT quadratic coefficients and optionally use them to calculate the coefficients on the w**2 quartic function
        # eft_coeffs is never Jagged so convert immediately to numpy for ease of use.
        eft_coeffs = ak.to_numpy(events['EFTfitCoefficients']) if hasattr(events, "EFTfitCoefficients") else None
        if eft_coeffs is not None:
            # Check to see if the ordering of WCs for this sample matches what want
            if self._samples[dataset]['WCnames'] != self._wc_names_lst:
                eft_coeffs = efth.remap_coeffs(self._samples[dataset]['WCnames'], self._wc_names_lst, eft_coeffs)
        eft_w2_coeffs = efth.calc_w2_coeffs(eft_coeffs,self._dtype) if (self._do_errors and eft_coeffs is not None) else None


        # Pass trigger
        pass_trg = trgPassNoOverlap(events,isData,dataset,str(year))

        # Selections and cuts
        selections = PackedSelection(dtype='uint64')

        # Lepton categories
        is2lss = events.is2lss
        is3l   = events.is3l
        is4l   = events.is4l

        # 2lss0tau things
        selections.add('2lss0tau', is2lss)

        # AR/SR categories
        selections.add('isSR_2lss',  ak.values_astype(events.is2lss_SR,'bool'))
        selections.add('isAR_2lss', ~ak.values_astype(events.is2lss_SR,'bool'))
        selections.add('isSR_3l',    ak.values_astype(events.is3l_SR,'bool'))
        selections.add('isAR_3l',   ~ak.values_astype(events.is3l_SR,'bool'))
        selections.add('isSR_4l',    ak.values_astype(events.is4l_SR,'bool'))

        # b jet masks
        bmask_atleast1med_atleast2loose = ((nbtagsm>=1)&(nbtagsl>=2)) # This is the requirement for 2lss and 4l
        bmask_exactly1med = (nbtagsm==1) # Used for 3l
        bmask_atleast2med = (nbtagsm>=2) # Used for 3l

        # Charge masks
        charge2l_p = ak.fill_none(((l0.charge+l1.charge)>0),False)
        charge2l_m = ak.fill_none(((l0.charge+l1.charge)<0),False)
        charge3l_p = ak.fill_none(((l0.charge+l1.charge+l2.charge)>0),False)
        charge3l_m = ak.fill_none(((l0.charge+l1.charge+l2.charge)<0),False)

        # Channels for the 2lss cat
        channels2LSS  = ["2lss_p", "2lss_m", "2lss_p_4j","2lss_p_5j","2lss_p_6j","2lss_p_7j","2lss_m_4j","2lss_m_5j","2lss_m_6j","2lss_m_7j"]
        selections.add("2lss_p",    (is2lss & charge2l_p & bmask_atleast1med_atleast2loose & pass_trg))
        selections.add("2lss_p_4j", (is2lss & charge2l_p & (njets==4) & bmask_atleast1med_atleast2loose & pass_trg))
        selections.add("2lss_p_5j", (is2lss & charge2l_p & (njets==5) & bmask_atleast1med_atleast2loose & pass_trg))
        selections.add("2lss_p_6j", (is2lss & charge2l_p & (njets==6) & bmask_atleast1med_atleast2loose & pass_trg))
        selections.add("2lss_p_7j", (is2lss & charge2l_p & (njets>=7) & bmask_atleast1med_atleast2loose & pass_trg))
        selections.add("2lss_m",    (is2lss & charge2l_m & bmask_atleast1med_atleast2loose & pass_trg))
        selections.add("2lss_m_4j", (is2lss & charge2l_m & (njets==4) & bmask_atleast1med_atleast2loose & pass_trg))
        selections.add("2lss_m_5j", (is2lss & charge2l_m & (njets==5) & bmask_atleast1med_atleast2loose & pass_trg))
        selections.add("2lss_m_6j", (is2lss & charge2l_m & (njets==6) & bmask_atleast1med_atleast2loose & pass_trg))
        selections.add("2lss_m_7j", (is2lss & charge2l_m & (njets>=7) & bmask_atleast1med_atleast2loose & pass_trg))

        # Channels for the 3l cat (we have a _lot_ of 3l categories...)
        channels3l  = [
            "3l_p_offZ_1b", "3l_m_offZ_1b", "3l_p_offZ_2b", "3l_m_offZ_2b", "3l_onZ_1b", "3l_onZ_2b",
            "3l_p_offZ_2j_1b","3l_p_offZ_3j_1b","3l_p_offZ_4j_1b","3l_p_offZ_5j_1b",
            "3l_m_offZ_2j_1b","3l_m_offZ_3j_1b","3l_m_offZ_4j_1b","3l_m_offZ_5j_1b",
            "3l_p_offZ_2j_2b","3l_p_offZ_3j_2b","3l_p_offZ_4j_2b","3l_p_offZ_5j_2b",
            "3l_m_offZ_2j_2b","3l_m_offZ_3j_2b","3l_m_offZ_4j_2b","3l_m_offZ_5j_2b",
            "3l_onZ_2j_1b","3l_onZ_3j_1b","3l_onZ_4j_1b","3l_onZ_5j_1b",
            "3l_onZ_2j_2b","3l_onZ_3j_2b","3l_onZ_4j_2b","3l_onZ_5j_2b",
        ]

        selections.add("3l_p_offZ_1b",    (is3l & charge3l_p & ~sfosz_mask & bmask_exactly1med & pass_trg))
        selections.add("3l_p_offZ_2j_1b", (is3l & charge3l_p & ~sfosz_mask & (njets==2) & bmask_exactly1med & pass_trg))
        selections.add("3l_p_offZ_3j_1b", (is3l & charge3l_p & ~sfosz_mask & (njets==3) & bmask_exactly1med & pass_trg))
        selections.add("3l_p_offZ_4j_1b", (is3l & charge3l_p & ~sfosz_mask & (njets==4) & bmask_exactly1med & pass_trg))
        selections.add("3l_p_offZ_5j_1b", (is3l & charge3l_p & ~sfosz_mask & (njets>=5) & bmask_exactly1med & pass_trg))

        selections.add("3l_m_offZ_1b",    (is3l & charge3l_m & ~sfosz_mask & bmask_exactly1med & pass_trg))
        selections.add("3l_m_offZ_2j_1b", (is3l & charge3l_m & ~sfosz_mask & (njets==2) & bmask_exactly1med & pass_trg))
        selections.add("3l_m_offZ_3j_1b", (is3l & charge3l_m & ~sfosz_mask & (njets==3) & bmask_exactly1med & pass_trg))
        selections.add("3l_m_offZ_4j_1b", (is3l & charge3l_m & ~sfosz_mask & (njets==4) & bmask_exactly1med & pass_trg))
        selections.add("3l_m_offZ_5j_1b", (is3l & charge3l_m & ~sfosz_mask & (njets>=5) & bmask_exactly1med & pass_trg))

        selections.add("3l_p_offZ_2b",    (is3l & charge3l_p & ~sfosz_mask & bmask_atleast2med & pass_trg))
        selections.add("3l_p_offZ_2j_2b", (is3l & charge3l_p & ~sfosz_mask & (njets==2) & bmask_atleast2med & pass_trg))
        selections.add("3l_p_offZ_3j_2b", (is3l & charge3l_p & ~sfosz_mask & (njets==3) & bmask_atleast2med & pass_trg))
        selections.add("3l_p_offZ_4j_2b", (is3l & charge3l_p & ~sfosz_mask & (njets==4) & bmask_atleast2med & pass_trg))
        selections.add("3l_p_offZ_5j_2b", (is3l & charge3l_p & ~sfosz_mask & (njets>=5) & bmask_atleast2med & pass_trg))

        selections.add("3l_m_offZ_2b",    (is3l & charge3l_m & ~sfosz_mask & bmask_atleast2med & pass_trg))
        selections.add("3l_m_offZ_2j_2b", (is3l & charge3l_m & ~sfosz_mask & (njets==2) & bmask_atleast2med & pass_trg))
        selections.add("3l_m_offZ_3j_2b", (is3l & charge3l_m & ~sfosz_mask & (njets==3) & bmask_atleast2med & pass_trg))
        selections.add("3l_m_offZ_4j_2b", (is3l & charge3l_m & ~sfosz_mask & (njets==4) & bmask_atleast2med & pass_trg))
        selections.add("3l_m_offZ_5j_2b", (is3l & charge3l_m & ~sfosz_mask & (njets>=5) & bmask_atleast2med & pass_trg))

        selections.add("3l_onZ_1b",    (is3l & sfosz_mask & bmask_exactly1med & pass_trg))
        selections.add("3l_onZ_2j_1b", (is3l & sfosz_mask & (njets==2) & bmask_exactly1med & pass_trg))
        selections.add("3l_onZ_3j_1b", (is3l & sfosz_mask & (njets==3) & bmask_exactly1med & pass_trg))
        selections.add("3l_onZ_4j_1b", (is3l & sfosz_mask & (njets==4) & bmask_exactly1med & pass_trg))
        selections.add("3l_onZ_5j_1b", (is3l & sfosz_mask & (njets>=5) & bmask_exactly1med & pass_trg))

        selections.add("3l_onZ_2b",    (is3l & sfosz_mask & bmask_atleast2med & pass_trg))
        selections.add("3l_onZ_2j_2b", (is3l & sfosz_mask & (njets==2) & bmask_atleast2med & pass_trg))
        selections.add("3l_onZ_3j_2b", (is3l & sfosz_mask & (njets==3) & bmask_atleast2med & pass_trg))
        selections.add("3l_onZ_4j_2b", (is3l & sfosz_mask & (njets==4) & bmask_atleast2med & pass_trg))
        selections.add("3l_onZ_5j_2b", (is3l & sfosz_mask & (njets>=5) & bmask_atleast2med & pass_trg))

        # Channels for the 4l cat
        channels4l  = ["4l", "4l_2j","4l_3j","4l_4j"]
        selections.add("4l",    (is4l & bmask_atleast1med_atleast2loose) & pass_trg)
        selections.add("4l_2j", (is4l & (njets==2) & bmask_atleast1med_atleast2loose) & pass_trg)
        selections.add("4l_3j", (is4l & (njets==3) & bmask_atleast1med_atleast2loose) & pass_trg)
        selections.add("4l_4j", (is4l & (njets>=4) & bmask_atleast1med_atleast2loose) & pass_trg)

        ptbl_bjet = goodJets[(isBtagJetsMedium | isBtagJetsLoose)]
        ptbl_bjet = ptbl_bjet[ak.argmax(ptbl_bjet.pt,axis=-1,keepdims=True)] # Only save hardest b-jet
        ptbl_lep = l_fo_conept_sorted
        ptbl = (ptbl_bjet.nearest(ptbl_lep) + ptbl_bjet).pt
        ptbl = ak.values_astype(ak.fill_none(ptbl, -1), np.float32)

        
        # Define invariant mass hists
        mll_0_1 = (l0+l1).mass     #invmass for leading two leps

        varnames = {}
        varnames['ht']     = ht
        varnames['l0pt']   = l0.conept
        varnames['l0eta']  = l0.eta
        varnames['j0pt' ]  = j0.pt
        varnames['j0eta']  = j0.eta
        varnames['njets']  = njets
        varnames['invmass'] = mll_0_1
        varnames['counts'] = np.ones_like(events['event'])
        varnames['ptbl']    = ptbl

        # Systematics
        systList = []
        if isData==False:
            systList = ['nominal']
            if self._do_systematics: systList = systList + ['lepSFUp','lepSFDown','btagSFUp', 'btagSFDown']
        else:
            systList = ['noweight']

        # Histograms
        hout = self.accumulator.identity()

        # Fill sum of weights hist
        normweights = weights_dict["2lss"].partial_weight(include=["norm"]) # Here we could have used 2lss, 3l, or 4l, as the "norm" weights should be identical for all three
        if len(self._wc_names_lst)>0: sowweights = np.ones_like(normweights)
        else: sowweights = normweights
        hout['SumOfEFTweights'].fill(sample=histAxisName, SumOfEFTweights=varnames['counts'], weight=sowweights, eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)

        # Fill the rest of the hists
        for syst in systList:
            for var, v in varnames.items():

                for ch in ['2lss0tau'] + channels2LSS + channels3l + channels4l:

                    if "2lss" in ch:
                        appl_lst = ['isSR_2lss', 'isAR_2lss']
                        weights_object = weights_dict["2lss"]
                    elif "3l" in ch:
                        appl_lst = ['isSR_3l', 'isAR_3l']
                        weights_object = weights_dict["3l"]
                    elif "4l" in ch:
                        appl_lst = ['isSR_4l']
                        weights_object = weights_dict["4l"]
                    else: raise Exception(f"Error: Unknown channel \"{ch}\". Exiting...")

                    # Find the event weight to be used when filling the histograms    
                    weight_fluct = syst
                    # In the case of 'nominal', or the jet energy systematics, no weight systematic variation is used (weight_fluct=None)
                    if syst in ['nominal','JERUp','JERDown','JESUp','JESDown']:
                        weight_fluct = None # no weight systematic for these variations
                    if syst=='noweight':
                        weight = np.ones(len(events)) # for data
                    else:
                        weight = weights_object.weight(weight_fluct)

                    for appl in appl_lst:

                        cuts = [ch,appl]
                        cut = selections.all(*cuts)
                        weights_flat = weight[cut]
                        weights_ones = np.ones_like(weights_flat, dtype=np.int)
                        eft_coeffs_cut = eft_coeffs[cut] if eft_coeffs is not None else None
                        eft_w2_coeffs_cut = eft_w2_coeffs[cut] if eft_w2_coeffs is not None else None

                        # Filling histos
                        if var == 'njets' :
                            if 'j' in ch: continue # Ignore sparse jet bins
                            values = v[cut]
                            hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, njets=values, sample=histAxisName, channel=ch, weight=weights_flat, systematic=syst,appl=appl)
                        elif 'j' not in ch: continue # Super channels for njets only
                        elif var == 'invmass':
                            values = v[cut]
                            hout['invmass'].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, sample=histAxisName, channel=ch, invmass=values, weight=weights_flat, systematic=syst,appl=appl)
                        elif var == 'ptbl' : 
                            values = ak.flatten(v[cut])
                            hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, ptbl=values, sample=histAxisName, channel=ch, weight=weights_flat, systematic=syst,appl=appl)
                        else:
                            values = v[cut] 
                            # These all look identical, do we need if/else here?
                            if   var == 'ht'    : hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, ht=values, sample=histAxisName, channel=ch, weight=weights_flat, systematic=syst,appl=appl)
                            elif var == 'met'   : hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, met=values, sample=histAxisName, channel=ch, weight=weights_flat, systematic=syst,appl=appl)
                            elif var == 'nbtags': hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, nbtags=values, sample=histAxisName, channel=ch, weight=weights_flat, systematic=syst,appl=appl)
                            elif var == 'counts': hout[var].fill(counts=values, sample=histAxisName, channel=ch, weight=weights_ones, systematic=syst,appl=appl)
                            elif var == 'j0eta' : 
                                values = ak.flatten(values) # Values here are not already flat, why?
                                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, j0eta=values, sample=histAxisName, channel=ch, weight=weights_flat, systematic=syst,appl=appl)
                            elif var == 'l0pt'  : 
                                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, l0pt=values, sample=histAxisName, channel=ch, weight=weights_flat, systematic=syst,appl=appl)
                            elif var == 'l0eta'  : 
                                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, l0eta=values, sample=histAxisName, channel=ch, weight=weights_flat, systematic=syst,appl=appl)
                            elif var == 'j0pt'  : 
                                values = ak.flatten(values) # Values here are not already flat, why?
                                hout[var].fill(eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut, j0pt=values, sample=histAxisName, channel=ch, weight=weights_flat, systematic=syst,appl=appl)
        return hout

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    # Load the .coffea files
    outpath= './coffeaFiles/'
    samples     = load(outpath+'samples.coffea')
    topprocessor = AnalysisProcessor(samples)

