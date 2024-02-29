#!/usr/bin/env python
import coffea
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
import hist
from coffea import processor
from coffea.analysis_tools import PackedSelection
from coffea.lumi_tools import LumiMask

import topcoffea.modules.object_selection as tc_os
from topcoffea.modules.paths import topcoffea_path
import topcoffea.modules.event_selection as tc_es

from topeft.modules.corrections import AttachMuonSF, AttachElectronSF, AttachPerLeptonFR
from topeft.modules.paths import topeft_path
import topeft.modules.event_selection as te_es
import topeft.modules.object_selection as te_os

from topcoffea.modules.get_param_from_jsons import GetParam
get_tc_param = GetParam(topcoffea_path("params/params.json"))
get_te_param = GetParam(topeft_path("params/params.json"))

# Check if the values in an array are within a given range
def in_range_mask(in_var,lo_lim=None,hi_lim=None):

    # Make sure at least one of the cuts is not none
    if (lo_lim is None) and (hi_lim is None):
        raise Exception("Error: No cuts specified")

    # Check if the value is greater than the min
    if lo_lim is not None:
        above_min = (in_var > lo_lim)
    else:
        above_min = (ak.ones_like(in_var)==1)

    # Check if the value is less than or equal to the max
    if hi_lim is not None:
        below_max = (in_var <= hi_lim)
    else:
        below_max = (ak.ones_like(in_var)==1)

    # Return the mask
    return ak.fill_none((above_min & below_max),False)


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, dtype=np.float32):

        self._samples = samples
        self._dtype = dtype

        # Create the histograms
        self._accumulator = {
            "invmass" : hist.Hist(hist.axis.StrCategory([], name="process", label="process", growth=True), hist.axis.StrCategory([], name="channel", label="channel", growth=True), hist.axis.Regular(name="invmass", label="$m_{\ell\ell}$ (GeV) ", bins=100, start=50,   stop=150)),
            "njets"   : hist.Hist(hist.axis.StrCategory([], name="process", label="process", growth=True), hist.axis.StrCategory([], name="channel", label="channel", growth=True), hist.axis.Regular(name="njets",   label="njets",                 bins=8,   start=0,    stop=8)),
            "l0pt"    : hist.Hist(hist.axis.StrCategory([], name="process", label="process", growth=True), hist.axis.StrCategory([], name="channel", label="channel", growth=True), hist.axis.Regular(name="l0pt",    label="l0pt",                  bins=20,  start=0,    stop=200)),
            "l0eta"   : hist.Hist(hist.axis.StrCategory([], name="process", label="process", growth=True), hist.axis.StrCategory([], name="channel", label="channel", growth=True), hist.axis.Regular(name="l0eta",   label="l0eta",                 bins=20,  start=-2.5, stop=2.5)),
            "l1pt"    : hist.Hist(hist.axis.StrCategory([], name="process", label="process", growth=True), hist.axis.StrCategory([], name="channel", label="channel", growth=True), hist.axis.Regular(name="l1pt",    label="l1pt",                  bins=20,  start=0,    stop=200)),
            "l1eta"   : hist.Hist(hist.axis.StrCategory([], name="process", label="process", growth=True), hist.axis.StrCategory([], name="channel", label="channel", growth=True), hist.axis.Regular(name="l1eta",   label="l1eta",                 bins=20,  start=-2.5, stop=2.5)),
        }

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
        isData             = self._samples[dataset]["isData"]
        histAxisName       = self._samples[dataset]["histAxisName"]
        year               = self._samples[dataset]["year"]
        xsec               = self._samples[dataset]["xsec"]
        sow                = self._samples[dataset]["nSumOfWeights"]

        datasets = ["SingleMuon", "SingleElectron", "EGamma", "MuonEG", "DoubleMuon", "DoubleElectron", "DoubleEG"]
        for d in datasets:
            if d in dataset: dataset = dataset.split('_')[0]

        # Set the sampleType (used for MC matching requirement)
        # Does not really matter for this processor, but still need to pass it to the selection function anyway
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
        ele  = events.Electron
        mu   = events.Muon
        jets = events.Jet

        ele["idEmu"] = te_os.ttH_idEmu_cuts_E3(ele.hoe, ele.eta, ele.deltaEtaSC, ele.eInvMinusPInv, ele.sieie)
        ele["conept"] = te_os.coneptElec(ele.pt, ele.mvaTTHUL, ele.jetRelIso)
        mu["conept"] = te_os.coneptMuon(mu.pt, mu.mvaTTHUL, mu.jetRelIso, mu.mediumId)
        ele["btagDeepFlavB"] = ak.fill_none(ele.matched_jet.btagDeepFlavB, -99)
        mu["btagDeepFlavB"] = ak.fill_none(mu.matched_jet.btagDeepFlavB, -99)

        if not isData:
            ele["gen_pdgId"] = ele.matched_gen.pdgId
            mu["gen_pdgId"] = mu.matched_gen.pdgId
            ele["gen_parent_pdgId"] = ele.matched_gen.distinctParent.pdgId
            mu["gen_parent_pdgId"] = mu.matched_gen.distinctParent.pdgId
            ele["gen_gparent_pdgId"] = ele.matched_gen.distinctParent.distinctParent.pdgId
            mu["gen_gparent_pdgId"] = mu.matched_gen.distinctParent.distinctParent.pdgId

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


        ################### Object selection ####################

        # Electron selection
        ele["isPres"] = te_os.isPresElec(ele.pt, ele.eta, ele.dxy, ele.dz, ele.miniPFRelIso_all, ele.sip3d, getattr(ele,"mvaFall17V2noIso_WPL"))
        ele["isLooseE"] = te_os.isLooseElec(ele.miniPFRelIso_all,ele.sip3d,ele.lostHits)
        ele["isFO"] = te_os.isFOElec(ele.pt, ele.conept, ele.btagDeepFlavB, ele.idEmu, ele.convVeto, ele.lostHits, ele.mvaTTHUL, ele.jetRelIso, ele.mvaFall17V2noIso_WP90, year)
        ele["isTightLep"] = te_os.tightSelElec(ele.isFO, ele.mvaTTHUL)
        # Muon selection
        mu["isPres"] = te_os.isPresMuon(mu.dxy, mu.dz, mu.sip3d, mu.eta, mu.pt, mu.miniPFRelIso_all)
        mu["isLooseM"] = te_os.isLooseMuon(mu.miniPFRelIso_all,mu.sip3d,mu.looseId)
        mu["isFO"] = te_os.isFOMuon(mu.pt, mu.conept, mu.btagDeepFlavB, mu.mvaTTHUL, mu.jetRelIso, year)
        mu["isTightLep"] = te_os.tightSelMuon(mu.isFO, mu.mediumId, mu.mvaTTHUL)
        # Build loose collections
        m_loose = mu[mu.isPres & mu.isLooseM]
        e_loose = ele[ele.isPres & ele.isLooseE]
        l_loose = ak.with_name(ak.concatenate([e_loose, m_loose], axis=1), 'PtEtaPhiMCandidate')

        # Compute pair invariant masses, for all flavors all signes
        llpairs = ak.combinations(l_loose, 2, fields=["l0","l1"])
        events["minMllAFAS"] = ak.min( (llpairs.l0+llpairs.l1).mass, axis=-1)

        # Build FO collection
        m_fo = mu[mu.isPres & mu.isLooseM & mu.isFO]
        e_fo = ele[ele.isPres & ele.isLooseE & ele.isFO]

        # Attach the lepton SFs to the electron and muons collections (the event selection expect these to be present)
        AttachElectronSF(e_fo,year=year)
        AttachMuonSF(m_fo,year=year)

        # Attach per lepton fake rates
        AttachPerLeptonFR(e_fo, flavor = "Elec", year=year)
        AttachPerLeptonFR(m_fo, flavor = "Muon", year=year)
        m_fo['convVeto'] = ak.ones_like(m_fo.charge)
        m_fo['lostHits'] = ak.zeros_like(m_fo.charge)
        l_fo = ak.with_name(ak.concatenate([e_fo, m_fo], axis=1), 'PtEtaPhiMCandidate')
        l_fo_conept_sorted = l_fo[ak.argsort(l_fo.conept, axis=-1,ascending=False)]
        events["l_fo_conept_sorted"] = l_fo_conept_sorted

        # Convenient to have l0, l1, l2 on hand
        l_fo_conept_sorted_padded = ak.pad_none(l_fo_conept_sorted, 3)
        l0 = l_fo_conept_sorted_padded[:,0]
        l1 = l_fo_conept_sorted_padded[:,1]


        #################### Jets ####################

        # Jet cleaning, before any jet selection
        vetos_tocleanjets = ak.with_name( l_fo, "PtEtaPhiMCandidate")
        tmp = ak.cartesian([ak.local_index(jets.pt), vetos_tocleanjets.jetIdx], nested=True)
        cleanedJets = jets[~ak.any(tmp.slot0 == tmp.slot1, axis=-1)] # this line should go before *any selection*, otherwise lep.jetIdx is not aligned with the jet index

        # Selecting jets and cleaning them
        cleanedJets["isGood"] = tc_os.is_tight_jet(getattr(cleanedJets, "pt"), cleanedJets.eta, cleanedJets.jetId, pt_cut=30., eta_cut=get_te_param("eta_j_cut"), id_cut=get_te_param("jet_id_cut"))
        goodJets = cleanedJets[cleanedJets.isGood]
        njets = ak.num(goodJets)


        #################### Event selection ####################

        # The event selection
        te_es.add2lMaskAndSFs(events, year, isData, sampleType)
        te_es.addLepCatMasks(events)


        ######### Weights ###########

        # Normalize by (xsec/sow)*genw where genw is 1 for EFT samples
        # Note that for theory systs, will need to multiply by sow/sow_wgtUP to get (xsec/sow_wgtUp)*genw and same for Down
        lumi = 1000.0*get_tc_param(f"lumi_{year}")
        weights_object = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
        if not isData: weights_object.add("norm",(xsec/sow)*events["genWeight"]*lumi)
        else: weights_object.add("norm",np.ones_like(events["event"]))

        # Apply the flip rate to OS as a cross check
        weights_object.add("fliprate", events.flipfactor_2l)

        # Print info
        #print("id0:",l0.pdgId)
        #print("pt0:",l0.pt)
        #print("eta0",l0.eta)
        #print("id1:",l1.pdgId)
        #print("pt1:",l1.pt)
        #print("eta1",l1.eta)
        #print(events.flipfactor_2l)


        ######### Store boolean masks with PackedSelection ##########

        # Get mask for events that have two sf os leps close to z peak
        sfosz_2l_mask = tc_es.get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:2],pt_window=30.0,flavor="os")
        sfssz_2l_mask = tc_es.get_Z_peak_mask(l_fo_conept_sorted_padded[:,0:2],pt_window=30.0,flavor="ss")

        # Pass trigger mask
        pass_trg = tc_es.trg_pass_no_overlap(events,isData,dataset,str(year),te_es.dataset_dict_top22006,te_es.exclude_dict_top22006)

        # Charge masks
        charge2l_0 = ak.fill_none(((l0.charge+l1.charge)==0),False)
        charge2l_1 = ak.fill_none(((l0.charge+l1.charge)!=0),False)

        # Flavor mask
        sameflav_mask = (abs(l0.pdgId) == abs(l1.pdgId))

        # MC truth for flips
        #flip_l0 = (l0.gen_pdgId == -l0.pdgId)
        #flip_l1 = (l1.gen_pdgId == -l1.pdgId)
        #noflip_l0 = (l0.gen_pdgId == l0.pdgId)
        #noflip_l1 = (l1.gen_pdgId == l1.pdgId)
        #isprompt_2l = ( ((l0.genPartFlav==1) | (l0.genPartFlav == 15)) & ((l1.genPartFlav==1) | (l1.genPartFlav == 15)) )
        #truth_flip_mask   = (isprompt_2l & (flip_l0 | flip_l1) & ~(flip_l0 & flip_l1)) # One or the other flips, but not both
        #truth_noflip_mask = (isprompt_2l & noflip_l0 & noflip_l1) # Neither flips

        # Selections
        selections = PackedSelection(dtype='uint64')
        selections.add("is_good_lumi",lumi_mask)
        selections.add("os",  (charge2l_0))
        selections.add("ss",  (charge2l_1))
        selections.add("osz", (charge2l_0 & sfosz_2l_mask))
        selections.add("ssz", (charge2l_1 & sfssz_2l_mask))
        selections.add("2e", (events.is2l_nozeeveto & events.is2l_SR & events.is_ee & (njets<4) & pass_trg))
        #if not isData:
            #selections.add("sszTruthFlip",   (charge2l_1 & sfssz_2l_mask & truth_flip_mask))
            #selections.add("oszTruthNoFlip", (charge2l_0 & sfosz_2l_mask & truth_noflip_mask))
            #selections.add("ssTruthFlip",    (charge2l_1 & truth_flip_mask))
            #selections.add("osTruthNoFlip",  (charge2l_0 & truth_noflip_mask))


        ######### Variables for the dense and sparse axes of the hists ##########

        dense_var_dict = {
            "invmass" : (l0+l1).mass,
            "njets"   : njets,
            "l0pt"    : l0.pt,
            "l0eta"   : l0.eta,
            "l1pt"    : l1.pt,
            "l1eta"   : l1.eta,
        }

        print(l0.pt)
        print(l1.eta)

        ########## Fill the histograms ##########

        hout = self.accumulator

        # Set the list of channels to loop over
        chan_lst = ["osz","ssz"]
        #if not isData: chan_lst.append("sszTruthFlip")
        #if not isData: chan_lst.append("sszTruthFlip2")
        #chan_lst = ["ss","os","ssz","osz","ssTruthFlip","osTruthNoFlip","sszTruthFlip","oszTruthNoFlip"]

        # Loop over histograms to fill (just invmass, njets for now)
        for dense_axis_name, dense_axis_vals in dense_var_dict.items():

            # Loop over the lepton channels
            for chan_name in chan_lst:

                # Get the cut mask object
                cuts_lst = ["2e"]
                cuts_lst.append(chan_name)
                if isData: cuts_lst.append("is_good_lumi")
                cuts_mask = selections.all(*cuts_lst)

                # Fill the histo
                axes_fill_info_dict = {
                    dense_axis_name  : dense_axis_vals[cuts_mask],
                    "channel"        : chan_name,
                    "process"         : histAxisName,
                    "weight"         : weights_object.weight()[cuts_mask],
                }

                hout[dense_axis_name].fill(**axes_fill_info_dict)

        return hout

    def postprocess(self, accumulator):
        return accumulator
