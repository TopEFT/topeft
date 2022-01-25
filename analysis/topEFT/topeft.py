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
from coffea.lumi_tools import LumiMask

from topcoffea.modules.GetValuesFromJsons import get_param
from topcoffea.modules.objects import *
from topcoffea.modules.corrections import SFevaluator, GetBTagSF, ApplyJetCorrections, GetBtagEff, AttachMuonSF, AttachElectronSF, AttachPerLeptonFR, GetPUSF, ApplyRochesterCorrections, ApplyJetSystematics, AttachPSWeights, AttachPdfWeights, AttachScaleWeights
from topcoffea.modules.selection import *
from topcoffea.modules.HistEFT import HistEFT
from topcoffea.modules.paths import topcoffea_path
import topcoffea.modules.eft_helper as efth


# Takes strings as inputs, constructs a string for the full channel name
# Try to construct a channel name like this: [n leptons]_[lepton flavors]_[p or m charge]_[on or off Z]_[n b jets]_[n jets]
    # chan_str should look something like "3l_p_offZ_1b", NOTE: This function assumes nlep comes first
    # njet_str should look something like "atleast_5j",   NOTE: This function assumes njets comes last
    # flav_str should look something like "emm"
def construct_cat_name(chan_str,njet_str=None,flav_str=None):

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
    for component in [flav_str,chan_str,njet_str]:
        if component is None: continue
        ret_str = "_".join([ret_str,component])
    return ret_str


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, do_errors=False, do_systematics=False, split_by_lepton_flavor=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        # Create the histograms
        self._accumulator = processor.dict_accumulator({
        "invmass" : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 20, 0, 200)),
        "ptbl"    : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("ptbl",    "$p_{T}^{b\mathrm{-}jet+\ell_{min(dR)}}$ (GeV) ", 200, 0, 2000)),
        "ptz"     : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("ptz",      "$p_{T}$ Z (GeV)", 25, 0, 1000)),
        "invmass" : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ",50 , 60, 130)),
        "njets"   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("njets",   "Jet multiplicity ", 10, 0, 10)),
        "nbtagsl" : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("nbtagsl",  "Loose btag multiplicity ", 5, 0, 5)),
        "l0pt"    : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("l0pt",    "Leading lep $p_{T}$ (GeV)", 25, 0, 200)),
        "j0pt"    : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("j0pt",    "Leading jet  $p_{T}$ (GeV)", 25, 0, 200)),
        "l0eta"   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("l0eta",   "Leading lep $\eta$", 30, -3.0, 3.0)),
        "j0eta"   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("j0eta",   "Leading jet  $\eta$", 30, -3.0, 3.0)),
        "ht"      : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("ht",      "H$_{T}$ (GeV)", 200, 0, 2000)),
        "met"     : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("systematic", "Systematic Uncertainty"),hist.Cat("appl", "AR/SR"), hist.Bin("met",     "MET (GeV)", 40, 0, 400)),
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

        self._do_errors = do_errors # Whether to calculate and store the w**2 coefficients
        self._do_systematics = do_systematics # Whether to process systematic samples
        self._split_by_lepton_flavor = split_by_lepton_flavor # Whether to keep track of lepton flavors individually
        self._skip_signal_regions = skip_signal_regions # Whether to skip the SR categories
        self._skip_control_regions = skip_control_regions # Whether to skip the CR categories

        

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, events):

        # Dataset parameters
        dataset = events.metadata["dataset"]
        histAxisName = self._samples[dataset]["histAxisName"]
        year         = self._samples[dataset]["year"]
        xsec         = self._samples[dataset]["xsec"]
        sow          = self._samples[dataset]["nSumOfWeights"]
        isData       = self._samples[dataset]["isData"]
        datasets     = ["SingleMuon", "SingleElectron", "EGamma", "MuonEG", "DoubleMuon", "DoubleElectron", "DoubleEG"]
        for d in datasets: 
            if d in dataset: dataset = dataset.split('_')[0] 

        # Set the sampleType (used for MC matching requirement)
        conversionDatasets=[x%y for x in ['TTGJets_centralUL%d'] for y in [16,17,18]]
        nonpromptDatasets =[x%y for x in ['TTJets_centralUL%d','DY50_centralUL%d','DY10to50_centralUL%d','tbarW_centralUL%d','tW_centralUL%d','tbarW_centralUL%d'] for y in [16,17,18]]
        sampleType = 'prompt'
        if isData:
            sampleType = 'data'
        elif dataset in conversionDatasets: 
            sampleType = 'conversions'
        elif dataset in nonpromptDatasets:
            sampleType = 'nonprompt'

        # Initialize objects
        met  = events.MET
        e    = events.Electron
        mu   = events.Muon
        tau  = events.Tau
        jets = events.Jet

        e["idEmu"] = ttH_idEmu_cuts_E3(e.hoe, e.eta, e.deltaEtaSC, e.eInvMinusPInv, e.sieie)
        e["conept"] = coneptElec(e.pt, e.mvaTTH, e.jetRelIso)
        mu["conept"] = coneptMuon(mu.pt, mu.mvaTTH, mu.jetRelIso, mu.mediumId)
        e["btagDeepFlavB"] = ak.fill_none(e.matched_jet.btagDeepFlavB, -99)
        mu["btagDeepFlavB"] = ak.fill_none(mu.matched_jet.btagDeepFlavB, -99)
        
        # Get the lumi mask for data
        if year == "2016" or year == "2016APV":
            golden_json_path = topcoffea_path("data/goldenJsons/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")
        elif year == "2017":
            golden_json_path = topcoffea_path("data/goldenJsons/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt")
        elif year == "2018":
            golden_json_path = topcoffea_path("data/goldenJsons/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt")
        else:
            raise ValueError(f"Error: Unknown year \"{year}\".")
        lumi_mask = LumiMask(golden_json_path)(events.run,events.luminosityBlock)

        ######### EFT coefficients ##########

        # Extract the EFT quadratic coefficients and optionally use them to calculate the coefficients on the w**2 quartic function
        # eft_coeffs is never Jagged so convert immediately to numpy for ease of use.
        eft_coeffs = ak.to_numpy(events["EFTfitCoefficients"]) if hasattr(events, "EFTfitCoefficients") else None
        if eft_coeffs is not None:
          # Check to see if the ordering of WCs for this sample matches what want
          if self._samples[dataset]["WCnames"] != self._wc_names_lst:
              eft_coeffs = efth.remap_coeffs(self._samples[dataset]["WCnames"], self._wc_names_lst, eft_coeffs)
        eft_w2_coeffs = efth.calc_w2_coeffs(eft_coeffs,self._dtype) if (self._do_errors and eft_coeffs is not None) else None
        # Initialize the out object
        hout = self.accumulator.identity()
        
        ################### Object selection ####################
        # Electron selection
        e["isPres"] = isPresElec(e.pt, e.eta, e.dxy, e.dz, e.miniPFRelIso_all, e.sip3d, getattr(e,"mvaFall17V2noIso_WPL"))
        e["isLooseE"] = isLooseElec(e.miniPFRelIso_all,e.sip3d,e.lostHits)
        e["isFO"] = isFOElec(e.conept, e.btagDeepFlavB, e.idEmu, e.convVeto, e.lostHits, e.mvaTTH, e.jetRelIso, e.mvaFall17V2noIso_WP80, year)
        e["isTightLep"] = tightSelElec(e.isFO, e.mvaTTH)
        
        # Update muon kinematics with Rochester corrections
        mu["pt_raw"]=mu.pt
        met_raw=met
        if self._do_systematics : syst_var_list = ['ISRUp','ISRDown','FSRUp','FSRDown','renormUp','renormDown','factUp','factDown','renorm_factUp','renorm_factDown','MuonESUp','MuonESDown','JERUp','JERDown','JESUp','JESDown','nominal']
        else: syst_var_list = ['nominal']
        for syst_var in syst_var_list:
          mu["pt"]=mu.pt_raw
          if syst_var == 'MuonESUp': mu["pt"]=ApplyRochesterCorrections(mu, isData, var='up')
          elif syst_var == 'MuonESDown': mu["pt"]=ApplyRochesterCorrections(mu, isData, var='down')
          else: mu["pt"]=ApplyRochesterCorrections(mu, isData, var='nominal')
          # Muon selection
          mu["isPres"] = isPresMuon(mu.dxy, mu.dz, mu.sip3d, mu.eta, mu.pt, mu.miniPFRelIso_all)
          mu["isLooseM"] = isLooseMuon(mu.miniPFRelIso_all,mu.sip3d,mu.looseId)
          mu["isFO"] = isFOMuon(mu.pt, mu.conept, mu.btagDeepFlavB, mu.mvaTTH, mu.jetRelIso, year)
          mu["isTightLep"]= tightSelMuon(mu.isFO, mu.mediumId, mu.mvaTTH)
          # Build loose collections
          m_loose = mu[mu.isPres & mu.isLooseM]
          e_loose = e[e.isPres & e.isLooseE]
          l_loose = ak.with_name(ak.concatenate([e_loose, m_loose], axis=1), 'PtEtaPhiMCandidate')

          # Compute pair invariant masses, for all flavors all signes
          llpairs = ak.combinations(l_loose, 2, fields=["l0","l1"])
          events["minMllAFAS"] = ak.min( (llpairs.l0+llpairs.l1).mass, axis=-1)

          # Build FO collection
          m_fo = mu[mu.isPres & mu.isLooseM & mu.isFO]
          e_fo = e[e.isPres & e.isLooseE & e.isFO]

          # Attach the lepton SFs to the electron and muons collections
          AttachElectronSF(e_fo,year=year)
          AttachMuonSF(m_fo,year=year)

          # Attach per lepton fake rates
          AttachPerLeptonFR(e_fo, flavor = "Elec", year=year)
          AttachPerLeptonFR(m_fo, flavor = "Muon", year=year)
          m_fo['convVeto'] = ak.ones_like(m_fo.charge); 
          m_fo['lostHits'] = ak.zeros_like(m_fo.charge); 
          l_fo = ak.with_name(ak.concatenate([e_fo, m_fo], axis=1), 'PtEtaPhiMCandidate')
          l_fo_conept_sorted = l_fo[ak.argsort(l_fo.conept, axis=-1,ascending=False)]

          # Tau selection
          tau["isPres"]  = isPresTau(tau.pt, tau.eta, tau.dxy, tau.dz, tau.idDeepTau2017v2p1VSjet, minpt=20)
          tau["isClean"] = isClean(tau, l_loose, drmin=0.3)
          tau["isGood"]  =  tau["isClean"] & tau["isPres"]
          tau = tau[tau.isGood] # use these to clean jets
          tau["isTight"] = isTightTau(tau.idDeepTau2017v2p1VSjet) # use these to veto

          #################### Jets ####################

          # Jet cleaning, before any jet selection
          #vetos_tocleanjets = ak.with_name( ak.concatenate([tau, l_fo], axis=1), "PtEtaPhiMCandidate")
          vetos_tocleanjets = ak.with_name( l_fo, "PtEtaPhiMCandidate")
          tmp = ak.cartesian([ak.local_index(jets.pt), vetos_tocleanjets.jetIdx], nested=True)
          cleanedJets = jets[~ak.any(tmp.slot0 == tmp.slot1, axis=-1)] # this line should go before *any selection*, otherwise lep.jetIdx is not aligned with the jet index

          # Selecting jets and cleaning them
          jetptname = "pt_nom" if hasattr(cleanedJets, "pt_nom") else "pt"

          # Jet energy corrections
          if not isData:
            cleanedJets["pt_raw"] = (1 - cleanedJets.rawFactor)*cleanedJets.pt
            cleanedJets["mass_raw"] = (1 - cleanedJets.rawFactor)*cleanedJets.mass
            cleanedJets["pt_gen"] =ak.values_astype(ak.fill_none(cleanedJets.matched_gen.pt, 0), np.float32)
            cleanedJets["rho"] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, cleanedJets.pt)[0]
            events_cache = events.caches[0]
            cleanedJets = ApplyJetCorrections(year, corr_type='jets').build(cleanedJets, lazy_cache=events_cache)
            # SYSTEMATICS
            cleanedJets=ApplyJetSystematics(cleanedJets,syst_var)
            met=ApplyJetCorrections(year, corr_type='met').build(met_raw, cleanedJets, lazy_cache=events_cache)
          cleanedJets["isGood"] = isTightJet(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, jetPtCut=30.) # temporary at 25 for synch, TODO: Do we want 30 or 25?
          goodJets = cleanedJets[cleanedJets.isGood]

          # Count jets
          njets = ak.num(goodJets)
          ht = ak.sum(goodJets.pt,axis=-1)
          j0 = goodJets[ak.argmax(goodJets.pt,axis=-1,keepdims=True)]
          
          # Loose DeepJet WP
          # TODO: Update these numbers when UL16 is available, and double check UL17 and UL18 at that time as well
          if year == "2017":
            btagwpl = get_param("btag_wp_loose_UL17")
          elif year == "2018":
            btagwpl = get_param("btag_wp_loose_UL18")
          elif ((year=="2016") or (year=="2016APV")):
            btagwpl = get_param("btag_wp_loose_L16")
          else:
            raise ValueError(f"Error: Unknown year \"{year}\".")
          isBtagJetsLoose = (goodJets.btagDeepFlavB > btagwpl)
          isNotBtagJetsLoose = np.invert(isBtagJetsLoose)
          nbtagsl = ak.num(goodJets[isBtagJetsLoose])

          # Medium DeepJet WP
          # TODO: Update these numbers when UL16 is available, and double check UL17 and UL18 at that time as well
          if year == "2017": 
            btagwpm = get_param("btag_wp_medium_UL17")
          elif year == "2018":
            btagwpm = get_param("btag_wp_medium_UL18")
          elif ((year=="2016") or (year=="2016APV")):
            btagwpm = get_param("btag_wp_medium_L16")
          else:
            raise ValueError(f"Error: Unknown year \"{year}\".")
          isBtagJetsMedium = (goodJets.btagDeepFlavB > btagwpm)
          isNotBtagJetsMedium = np.invert(isBtagJetsMedium)
          nbtagsm = ak.num(goodJets[isBtagJetsMedium])


          #################### Add variables into event object so that they persist ####################

          # Put njets and l_fo_conept_sorted into events
          events["njets"] = njets
          events["l_fo_conept_sorted"] = l_fo_conept_sorted

          # The event selection
          add2lMaskAndSFs(events, year, isData, sampleType)
          add3lMaskAndSFs(events, year, isData, sampleType)
          add4lMaskAndSFs(events, year, isData)
          addLepCatMasks(events)

          # Convenient to have l0, l1, l2 on hand
          l_fo_conept_sorted_padded = ak.pad_none(l_fo_conept_sorted, 3)
          l0 = l_fo_conept_sorted_padded[:,0]
          l1 = l_fo_conept_sorted_padded[:,1]
          l2 = l_fo_conept_sorted_padded[:,2]

          print("The number of events passing FO 2l, 3l, and 4l selection:", ak.num(events[events.is2l],axis=0),ak.num(events[events.is3l],axis=0),ak.num(events[events.is4l],axis=0))

          ######### SFs, weights, systematics ##########

          # Btag SF following 1a) in https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods
          btagSF   = np.ones_like(ht)
          btagSFUp = np.ones_like(ht)
          btagSFDo = np.ones_like(ht)
          if not isData:
            pt = goodJets.pt; abseta = np.abs(goodJets.eta); flav = goodJets.hadronFlavour

            bJetSF   = GetBTagSF(abseta, pt, flav, year)
            bJetSFUp = GetBTagSF(abseta, pt, flav, year, sys='up')
            bJetSFDo = GetBTagSF(abseta, pt, flav, year, sys='down')
            bJetEff  = GetBtagEff(abseta, pt, flav, year)
            bJetEff_data   = bJetEff*bJetSF
            bJetEff_dataUp = bJetEff*bJetSFUp
            bJetEff_dataDo = bJetEff*bJetSFDo

            pMC     = ak.prod(bJetEff       [isBtagJetsMedium], axis=-1) * ak.prod((1-bJetEff       [isNotBtagJetsMedium]), axis=-1)
            pData   = ak.prod(bJetEff_data  [isBtagJetsMedium], axis=-1) * ak.prod((1-bJetEff_data  [isNotBtagJetsMedium]), axis=-1)
            pDataUp = ak.prod(bJetEff_dataUp[isBtagJetsMedium], axis=-1) * ak.prod((1-bJetEff_dataUp[isNotBtagJetsMedium]), axis=-1)
            pDataDo = ak.prod(bJetEff_dataDo[isBtagJetsMedium], axis=-1) * ak.prod((1-bJetEff_dataDo[isNotBtagJetsMedium]), axis=-1)           
            pMC      = ak.where(pMC==0,1,pMC) # removeing zeroes from denominator...

          # We need weights for: normalization, lepSF, triggerSF, pileup, btagSF...
          weights_dict = {}
          if (isData or (eft_coeffs is not None)):
            genw = np.ones_like(events["event"])
          else:
            genw = events["genWeight"]
          for ch_name in ["2l", "3l", "4l", "2l_CR", "3l_CR", "2los_CRtt", "2los_CRZ"]:
            weights_dict[ch_name] = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
            weights_dict[ch_name].add("norm",genw if isData else (xsec/sow)*genw)
            if not isData:

                ######### Event weights ###########

                # Attach PS weights (ISR/FSR)
                AttachPSWeights(events)
                # Attach scale weights (renormalization/factorization)
                AttachScaleWeights(events)
                # Attach PDF weights
                #AttachPdfWeights(events) # FIXME use these!

                # We only calculate these values if not isData
                weights_dict[ch_name].add("btagSF", pData/pMC, pDataUp/pMC, pDataDo/pMC)
                # Trying to calculate PU SFs for data causes a crash, and we don't apply this for data anyway, so just skip it in the case of data
                weights_dict[ch_name].add('PU', GetPUSF((events.Pileup.nTrueInt), year), GetPUSF(events.Pileup.nTrueInt, year, 'up'), GetPUSF(events.Pileup.nTrueInt, year, 'down'))
                # Prefiring weights only available in nanoAODv9**
                weights_dict[ch_name].add('PreFiring', events.L1PreFiringWeight.Nom,  events.L1PreFiringWeight.Up,  events.L1PreFiringWeight.Dn)
                # FSR/ISR weights
                weights_dict[ch_name].add('ISR', events.ISRnom, events.ISRUp, events.ISRDown)
                weights_dict[ch_name].add('FSR', events.FSRnom, events.FSRUp, events.FSRDown)
                # renorm/fact scale
                weights_dict[ch_name].add('renorm', events.nom, events.renormUp, events.renormDown)
                weights_dict[ch_name].add('fact',   events.nom, events.factUp,   events.factDown)
                weights_dict[ch_name].add('renorm_fact', events.nom, events.renorm_factUp, events.renorm_factDown)
            if "2l" in ch_name:
                weights_dict[ch_name].add("lepSF", events.sf_2l, events.sf_2l_hi, events.sf_2l_lo)
                weights_dict[ch_name].add("FF"   , events.fakefactor_2l, events.fakefactor_2l_up, events.fakefactor_2l_down )
                if isData:
                    weights_dict[ch_name].add("fliprate"   , events.flipfactor_2l)
            if "3l" in ch_name:
                weights_dict[ch_name].add("lepSF", events.sf_3l, events.sf_3l_hi, events.sf_3l_lo)
                weights_dict[ch_name].add("FF"   , events.fakefactor_3l, events.fakefactor_3l_up, events.fakefactor_3l_down)
            if "4l" in ch_name:
                weights_dict[ch_name].add("lepSF", events.sf_4l, events.sf_4l_hi, events.sf_4l_lo)

          # Systematics
          systList = ["nominal"]
          if (self._do_systematics and not isData and syst_var == "nominal"): systList = systList + ["lepSFUp","lepSFDown","btagSFUp", "btagSFDown","PUUp","PUDown","PreFiringUp","PreFiringDown","FSRUp","FSRDown","ISRUp","ISRDown","renormUp","renormDown","factUp","factDown","renorm_factUp","renorm_factDown"]
          elif (self._do_systematics and not isData and syst_var != 'nominal'): systList = [syst_var]

          ######### Masks we need for the selection ##########

          # Get mask for events that have two sf os leps close to z peak
          sfosz_3l_mask = get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:3],pt_window=10.0)
          sfosz_2l_mask = get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:2],pt_window=10.0)

          # Pass trigger mask
          pass_trg = trgPassNoOverlap(events,isData,dataset,str(year))

          # b jet masks
          bmask_atleast1med_atleast2loose = ((nbtagsm>=1)&(nbtagsl>=2)) # Used for 2lss and 4l
          bmask_exactly0med = (nbtagsm==0) # Used for 3l CR and 2los Z CR
          bmask_exactly1med = (nbtagsm==1) # Used for 3l SR and 2lss CR
          bmask_exactly2med = (nbtagsm==2) # Used for CRtt
          bmask_atleast2med = (nbtagsm>=2) # Used for 3l SR

          # Charge masks
          chargel0_p = ak.fill_none(((l0.charge)>0),False)
          chargel0_m = ak.fill_none(((l0.charge)<0),False)
          charge2l_0 = ak.fill_none(((l0.charge+l1.charge)==0),False)
          charge2l_1 = ak.fill_none(((l0.charge+l1.charge)!=0),False)
          charge3l_p = ak.fill_none(((l0.charge+l1.charge+l2.charge)>0),False)
          charge3l_m = ak.fill_none(((l0.charge+l1.charge+l2.charge)<0),False)


          ######### Store boolean masks with PackedSelection ##########

          selections = PackedSelection(dtype='uint64')

          # Lumi mask (for data)
          selections.add("is_good_lumi",lumi_mask)

          # 2lss selection
          selections.add("2lss_p", (events.is2l & chargel0_p & bmask_atleast1med_atleast2loose & pass_trg)) # Note: The ss requirement has NOT yet been made at this point! We take care of it later with the appl axis
          selections.add("2lss_m", (events.is2l & chargel0_m & bmask_atleast1med_atleast2loose & pass_trg)) # Note: The ss requirement has NOT yet been made at this point! We take care of it later with the appl axis
          selections.add("2lss_CR", (events.is2l & (chargel0_p| chargel0_m) & bmask_exactly1med & pass_trg)) # Note: The ss requirement has NOT yet been made at this point! We take care of it later with the appl axis
        
          # 2los selection
          selections.add("2los_CRtt", (events.is2l & charge2l_0 & bmask_exactly2med & pass_trg))
          selections.add("2los_CRZ", (events.is2l & charge2l_0 & sfosz_2l_mask & bmask_exactly0med & pass_trg))

          # 3l selection
          selections.add("3l_p_offZ_1b", (events.is3l & charge3l_p & ~sfosz_3l_mask & bmask_exactly1med & pass_trg))
          selections.add("3l_m_offZ_1b", (events.is3l & charge3l_m & ~sfosz_3l_mask & bmask_exactly1med & pass_trg))
          selections.add("3l_p_offZ_2b", (events.is3l & charge3l_p & ~sfosz_3l_mask & bmask_atleast2med & pass_trg))
          selections.add("3l_m_offZ_2b", (events.is3l & charge3l_m & ~sfosz_3l_mask & bmask_atleast2med & pass_trg))
          selections.add("3l_onZ_1b", (events.is3l & sfosz_3l_mask & bmask_exactly1med & pass_trg))
          selections.add("3l_onZ_2b", (events.is3l & sfosz_3l_mask & bmask_atleast2med & pass_trg))
          selections.add("3l_CR", (events.is3l & bmask_exactly0med & pass_trg))

          # 4l selection
          selections.add("4l", (events.is4l & bmask_atleast1med_atleast2loose & pass_trg))

          # Lep flavor selection
          selections.add("ee",  events.is_ee)
          selections.add("em",  events.is_em)
          selections.add("mm",  events.is_mm)
          selections.add("eee", events.is_eee)
          selections.add("eem", events.is_eem)
          selections.add("emm", events.is_emm)
          selections.add("mmm", events.is_mmm)
          selections.add("llll", (events.is_eeee | events.is_eeem | events.is_eemm | events.is_emmm | events.is_mmmm | events.is_gr4l)) # Not keepting track of these separately

          # Njets selection
          selections.add("exactly_1j", (njets==1))
          selections.add("exactly_2j", (njets==2))
          selections.add("exactly_3j", (njets==3))
          selections.add("exactly_4j", (njets==4))
          selections.add("exactly_5j", (njets==5))
          selections.add("exactly_6j", (njets==6))
          selections.add("atleast_1j", (njets>=1))
          selections.add("atleast_4j", (njets>=4))
          selections.add("atleast_5j", (njets>=5))
          selections.add("atleast_7j", (njets>=7))
          selections.add("atleast_0j", (njets>=0))

          # AR/SR categories
          selections.add("isSR_2lSS", ( events.is2l_SR) & charge2l_1) 
          selections.add("isAR_2lSS", (~events.is2l_SR) & charge2l_1) 
          selections.add("isAR_2lSS_OS", ( events.is2l_SR) & charge2l_0) # we need another naming for the sideband for the charge flip
          selections.add("isSR_2lOS", ( events.is2l_SR) & charge2l_0) 
          selections.add("isAR_2lOS", (~events.is2l_SR) & charge2l_0) 

          selections.add("isSR_3l",  events.is3l_SR)
          selections.add("isAR_3l", ~events.is3l_SR)
          selections.add("isSR_4l",  events.is4l_SR)


          ######### Variables for the dense axes of the hists ##########

          # Calculate ptbl
          ptbl_bjet = goodJets[(isBtagJetsMedium | isBtagJetsLoose)]
          ptbl_bjet = ptbl_bjet[ak.argmax(ptbl_bjet.pt,axis=-1,keepdims=True)] # Only save hardest b-jet
          ptbl_lep = l_fo_conept_sorted
          ptbl = (ptbl_bjet.nearest(ptbl_lep) + ptbl_bjet).pt
          ptbl = ak.values_astype(ak.fill_none(ptbl, -1), np.float32)

          # Z pt (pt of the ll pair that form the Z for the onZ categories) 
          ptz = get_Z_pt(l_fo_conept_sorted_padded[:,0:3],10.0)     
        
          # Define invariant mass hists
          mll_0_1 = (l0+l1).mass # Invmass for leading two leps

          # Counts
          counts = np.ones_like(events['event'])

          # Variables we will loop over when filling hists
          varnames = {}
          varnames["ht"]      = ht
          varnames["met"]     = met.pt
          varnames["l0pt"]    = l0.conept
          varnames["l0eta"]   = l0.eta
          varnames["j0pt"]    = ak.flatten(j0.pt)
          varnames["j0eta"]   = ak.flatten(j0.eta)
          varnames["njets"]   = njets
          varnames["nbtagsl"] = nbtagsl
          varnames["invmass"] = mll_0_1
          varnames["ptbl"]    = ak.flatten(ptbl)
          varnames["ptz"]     = ptz


          ########## Fill the histograms ##########

          # This dictionary keeps track of which selections go with which SR categories
          sr_cat_dict = {
            "2l" : {
                "lep_chan_lst" : ["2lss_p" , "2lss_m"],
                "lep_flav_lst" : ["ee" , "em" , "mm"],
                "njets_lst"    : ["exactly_4j" , "exactly_5j" , "exactly_6j" , "atleast_7j"],
                "appl_lst"     : ["isSR_2lSS" , "isAR_2lSS"] + (["isAR_2lSS_OS"] if isData else []),
            },
            "3l" : {
                "lep_chan_lst" : ["3l_p_offZ_1b" , "3l_m_offZ_1b" , "3l_p_offZ_2b" , "3l_m_offZ_2b" , "3l_onZ_1b" , "3l_onZ_2b"],
                "lep_flav_lst" : ["eee" , "eem" , "emm", "mmm"],
                "njets_lst"    : ["exactly_2j" , "exactly_3j" , "exactly_4j" , "atleast_5j"],
                "appl_lst"     : ["isSR_3l", "isAR_3l"],
            },
            "4l" : {
                "lep_chan_lst" : ["4l"],
                "lep_flav_lst" : ["llll"], # Not keeping track of these separately
                "njets_lst"    : ["exactly_2j" , "exactly_3j" , "atleast_4j"],
                "appl_lst"     : ["isSR_4l"],
            },
          }

          # This dictionary keeps track of which selections go with which CR categories
          cr_cat_dict = {
            "2l_CR" : {
                "lep_chan_lst" : ["2lss_CR"],
                "lep_flav_lst" : ["ee" , "em" , "mm"],
                "njets_lst"    : ["exactly_1j" , "exactly_2j"],
                "appl_lst"     : ["isSR_2lSS" , "isAR_2lSS"] + (["isAR_2lSS_OS"] if isData else []),
            },
            "3l_CR" : {
                "lep_chan_lst" : ["3l_CR"],
                "lep_flav_lst" : ["eee" , "eem" , "emm", "mmm"],
                "njets_lst"    : ["atleast_1j"],
                "appl_lst"     : ["isSR_3l" , "isAR_3l"],
            },
            "2los_CRtt" : {
                "lep_chan_lst" : ["2los_CRtt"],
                "lep_flav_lst" : ["em"],
                "njets_lst"    : ["exactly_2j"],
                "appl_lst"     : ["isSR_2lOS" , "isAR_2lOS"],
            },
            "2los_CRZ" : {
                "lep_chan_lst" : ["2los_CRZ"],
                "lep_flav_lst" : ["ee", "mm"],
                "njets_lst"    : ["atleast_0j"],
                "appl_lst"     : ["isSR_2lOS" , "isAR_2lOS"],
            }            
          }

          # Include SRs and CRs unless we asked to skip them
          cat_dict = {}
          if not self._skip_signal_regions:
            cat_dict.update(sr_cat_dict)
          if not self._skip_control_regions:
            cat_dict.update(cr_cat_dict)
          if (not self._skip_signal_regions and not self._skip_control_regions):
            for k in sr_cat_dict:
                if k in cr_cat_dict:
                    raise Exception(f"The key {k} is in both CR and SR dictionaries.")




          # Loop over the hists we want to fill
          for dense_axis_name, dense_axis_vals in varnames.items():
            if dense_axis_name not in self._hist_lst:
                print(f"Skipping \"{dense_axis_name}\", it is not in the list of hists to include.")
                continue

            # Loop over the systematics
            for syst in systList:
                # In the case of "nominal", or the jet energy systematics, no weight systematic variation is used (weight_fluct=None)
                weight_fluct = syst
                if syst in ["nominal","JERUp","JERDown","JESUp","JESDown","MuonESUp","MuonESDown"]: weight_fluct = None # No weight systematic for these variations

                # Loop over nlep categories "2l", "3l", "4l"
                for nlep_cat in cat_dict.keys():

                    # Get the appropriate Weights object for the nlep cat and get the weight to be used when filling the hist
                    weights_object = weights_dict[nlep_cat]
                    if isData : weight = weights_object.partial_weight(include=["FF"] + (["fliprate"] if nlep_cat in ["2l", "2l_CR"] else [])) # for data, must include the FF. The flip rate we only apply to 2lss regions
                    else      : weight = weights_object.weight(weight_fluct) # For MC

                    # Get a mask for events that pass any of the njet requiremens in this nlep cat
                    # Useful in cases like njets hist where we don't store njets in a sparse axis
                    njets_any_mask = selections.any(*cat_dict[nlep_cat]["njets_lst"])

                    # Loop over the appropriate AR and SR for this channel
                    for appl in cat_dict[nlep_cat]["appl_lst"]:

                        # Loop over the channels in each nlep cat (e.g. "3l_m_offZ_1b")
                        for lep_chan in cat_dict[nlep_cat]["lep_chan_lst"]:

                            # Loop over the njets list for each channel
                            for njet_val in cat_dict[nlep_cat]["njets_lst"]:

                                # Loop over the lep flavor list for each channel
                                for lep_flav in cat_dict[nlep_cat]["lep_flav_lst"]:

                                    # Construct the hist name
                                    flav_ch = None
                                    njet_ch = None
                                    cuts_lst = [appl,lep_chan]
                                    if isData:
                                        cuts_lst.append("is_good_lumi")
                                    if self._split_by_lepton_flavor:
                                        flav_ch = lep_flav
                                        cuts_lst.append(lep_flav)
                                    if dense_axis_name != "njets":
                                        njet_ch = njet_val
                                        cuts_lst.append(njet_val)
                                    ch_name = construct_cat_name(lep_chan,njet_str=njet_ch,flav_str=flav_ch)

                                    # Get the cuts mask for all selections
                                    if dense_axis_name == "njets":
                                        all_cuts_mask = (selections.all(*cuts_lst) & njets_any_mask)
                                    else:
                                        all_cuts_mask = selections.all(*cuts_lst)

                                    # Weights and eft coeffs
                                    weights_flat = weight[all_cuts_mask]
                                    eft_coeffs_cut = eft_coeffs[all_cuts_mask] if eft_coeffs is not None else None
                                    eft_w2_coeffs_cut = eft_w2_coeffs[all_cuts_mask] if eft_w2_coeffs is not None else None

                                    # Fill the histos
                                    axes_fill_info_dict = {
                                        dense_axis_name : dense_axis_vals[all_cuts_mask],
                                        "channel"       : ch_name,
                                        "appl"          : appl,
                                        "sample"        : histAxisName,
                                        "systematic"    : syst,
                                        "weight"        : weights_flat,
                                        "eft_coeff"     : eft_coeffs_cut,
                                        "eft_err_coeff" : eft_w2_coeffs_cut,
                                    }
                                    
                                    if (("j0" in dense_axis_name) & ("CRZ" in ch_name)): continue
                                    if (("ptz" in dense_axis_name) & ("onZ" not in lep_chan)): continue
                                    hout[dense_axis_name].fill(**axes_fill_info_dict)

                                    # Do not loop over lep flavors if not self._split_by_lepton_flavor, it's a waste of time and also we'd fill the hists too many times
                                    if not self._split_by_lepton_flavor: break

                                # Do not loop over njets if hist is njets (otherwise we'd fill the hist too many times)
                                if dense_axis_name == "njets": break

        return hout

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    # Load the .coffea files
    outpath= './coffeaFiles/'
    samples     = load(outpath+'samples.coffea')
    topprocessor = AnalysisProcessor(samples)

