'''
 selection.py
 This script contains several functions that implement the some event selection.
 The functinos defined here can be used to define a selection, signal/control region, etc.
 The functions are called with (jagged)arrays as imputs plus some custom paramenters and return a boolean mask.

'''

import awkward as ak
import numpy as np

from topeft.modules.corrections import fakeRateWeight1l, fakeRateWeight2l, fakeRateWeight3l, additional_nonprompt_ph_unc
from topeft.modules.genParentage import maxHistoryPDGID


# The datasets we are using, and the triggers in them
dataset_dict_top22006 = {

    "2016" : {
        "SingleMuon" : [
            "IsoMu24",
            "IsoTkMu24",
            "IsoMu22_eta2p1",
            "IsoTkMu22_eta2p1",
            "IsoMu22",
            "IsoTkMu22",
            "IsoMu27",
        ],
        "SingleElectron" : [
            'Ele27_WPTight_Gsf',
            "Ele25_eta2p1_WPTight_Gsf",
            "Ele27_eta2p1_WPLoose_Gsf",
        ],
        "DoubleMuon" : [
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL",
            "Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL",
            "Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ",
            "TripleMu_12_10_5",
        ],
        "DoubleEG" : [
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Ele16_Ele12_Ele8_CaloIdL_TrackIdL",
        ],
        "MuonEG" : [
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL",
            "Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_DiEle12_CaloIdL_TrackIdL",
            "DiMu9_Ele9_CaloIdL_TrackIdL",
        ]
    },

    "2017" : {
        "SingleMuon" : [
            "IsoMu24",
            "IsoMu27",
        ],
        "SingleElectron" : [
            "Ele32_WPTight_Gsf",
            "Ele35_WPTight_Gsf",
        ],
        "DoubleMuon" : [
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
            "TripleMu_12_10_5",
        ],
        "DoubleEG" : [
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Ele16_Ele12_Ele8_CaloIdL_TrackIdL",
        ],
        "MuonEG" : [
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_DiEle12_CaloIdL_TrackIdL",
            "Mu8_DiEle12_CaloIdL_TrackIdL_DZ", # Note: Listed in Andrew's thesis, but not TOP-19-001 AN
            "DiMu9_Ele9_CaloIdL_TrackIdL_DZ",
        ]
    },

    "2018" : {
        "SingleMuon" : [
            "IsoMu24",
            "IsoMu27",
        ],
        "EGamma" : [
            "Ele32_WPTight_Gsf",
            "Ele35_WPTight_Gsf",
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Ele16_Ele12_Ele8_CaloIdL_TrackIdL",
        ],
        "DoubleMuon" : [
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
            "TripleMu_12_10_5",
        ],
        "MuonEG" : [
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_DiEle12_CaloIdL_TrackIdL",
            "Mu8_DiEle12_CaloIdL_TrackIdL_DZ",
            "DiMu9_Ele9_CaloIdL_TrackIdL_DZ",
        ]
    },

    #NEW TRIGGERS - Should pull from https://docs.google.com/document/d/1zm9EkFExonAO2upU1V7_lw8uKjHknFrnxNQMq75HWnw/edit?
    #Currently placeholders
    "2022" : {
        "Muon" : [
            "IsoMu24",
            "IsoMu27",
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
            "TripleMu_12_10_5",
        ],
        "EGamma" : [
            "Ele32_WPTight_Gsf",
            "Ele35_WPTight_Gsf",
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Ele16_Ele12_Ele8_CaloIdL_TrackIdL",
        ],
        "MuonEG" : [
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_DiEle12_CaloIdL_TrackIdL",
            "Mu8_DiEle12_CaloIdL_TrackIdL_DZ",
            "DiMu9_Ele9_CaloIdL_TrackIdL_DZ",
        ],
    },
}


# Hard coded dictionary for figuring out overlap...
#   - No unique way to do this
#   - Note: In order for this to work properly, you should be processing all of the datastes to be used in the analysis
#   - Otherwise, you may be removing events that show up in other datasets you're not using
exclude_dict_top22006 = {
    "2016": {
        "DoubleMuon"     : [],
        "DoubleEG"       : dataset_dict_top22006["2016"]["DoubleMuon"],
        "MuonEG"         : dataset_dict_top22006["2016"]["DoubleMuon"] + dataset_dict_top22006["2016"]["DoubleEG"],
        "SingleMuon"     : dataset_dict_top22006["2016"]["DoubleMuon"] + dataset_dict_top22006["2016"]["DoubleEG"] + dataset_dict_top22006["2016"]["MuonEG"],
        "SingleElectron" : dataset_dict_top22006["2016"]["DoubleMuon"] + dataset_dict_top22006["2016"]["DoubleEG"] + dataset_dict_top22006["2016"]["MuonEG"] + dataset_dict_top22006["2016"]["SingleMuon"],
    },
    "2017": {
        "DoubleMuon"     : [],
        "DoubleEG"       : dataset_dict_top22006["2017"]["DoubleMuon"],
        "MuonEG"         : dataset_dict_top22006["2017"]["DoubleMuon"] + dataset_dict_top22006["2017"]["DoubleEG"],
        "SingleMuon"     : dataset_dict_top22006["2017"]["DoubleMuon"] + dataset_dict_top22006["2017"]["DoubleEG"] + dataset_dict_top22006["2017"]["MuonEG"],
        "SingleElectron" : dataset_dict_top22006["2017"]["DoubleMuon"] + dataset_dict_top22006["2017"]["DoubleEG"] + dataset_dict_top22006["2017"]["MuonEG"] + dataset_dict_top22006["2017"]["SingleMuon"],
    },
    "2018": {
        "DoubleMuon"     : [],
        "EGamma"         : dataset_dict_top22006["2018"]["DoubleMuon"],
        "MuonEG"         : dataset_dict_top22006["2018"]["DoubleMuon"] + dataset_dict_top22006["2018"]["EGamma"],
        "SingleMuon"     : dataset_dict_top22006["2018"]["DoubleMuon"] + dataset_dict_top22006["2018"]["EGamma"] + dataset_dict_top22006["2018"]["MuonEG"],
    },
    "2022": {
        "Muon"           : [],
        "EGamma"         : dataset_dict_top22006["2022"]["Muon"],
        "MuonEG"         : dataset_dict_top22006["2022"]["Muon"] + dataset_dict_top22006["2022"]["EGamma"],
    },
}


# 1l selections
# STILL IN DEVELOPMENT!!!
def add1lMaskAndSFs(events, year, isData, sampleType):

    # FOs and padded FOs
    FOs = events.l_fo_conept_sorted
    padded_FOs = ak.pad_none(FOs,1)

    # Filters and cleanups
    filter_flags = events.Flag
    filters = filter_flags.goodVertices & filter_flags.globalSuperTightHalo2016Filter & filter_flags.HBHENoiseFilter & filter_flags.HBHENoiseIsoFilter & filter_flags.EcalDeadCellTriggerPrimitiveFilter & filter_flags.BadPFMuonFilter & (((year == "2016")|(year == "2016APV")) | filter_flags.ecalBadCalibFilter) & (isData | filter_flags.eeBadScFilter)
    cleanup = events.minMllAFAS > 12
    muTightCharge = ((abs(padded_FOs[:,0].pdgId)!=13) | (padded_FOs[:,0].tightCharge>=1))

    # IDs
    eleID1 = (abs(padded_FOs[:,0].pdgId)!=11) | ((padded_FOs[:,0].convVeto != 0) & (padded_FOs[:,0].lostHits==0) & (padded_FOs[:,0].tightCharge>=2))

    # 1l requirements:
    exclusive = ak.num( FOs[FOs.isTightLep],axis=-1)<2
    monlep = (ak.num(FOs)) >= 1
    pt2515 = (ak.any(FOs[:,0:1].conept > 25.0, axis=1))
    mask = (monlep & exclusive & eleID1 & muTightCharge)

    # MC matching requirement (already passed for data)
    if sampleType == "data":
        pass
    else:
        lep1_match_prompt = ((padded_FOs[:,0].genPartFlav==1) | (padded_FOs[:,0].genPartFlav == 15))
        lep1_charge       = ((padded_FOs[:,0].gen_pdgId*padded_FOs[:,0].pdgId) > 0)
        lep1_match_conv   = (padded_FOs[:,0].genPartFlav==22)
        prompt_mask = ( lep1_match_prompt )
        conv_mask   = ( lep1_match_conv )
        if sampleType == 'prompt':
            mask = (mask & prompt_mask)
        elif sampleType =='conversions':
            mask = (mask & conv_mask)
        elif sampleType =='prompt_and_conversions':
            # Samples that we use for both prompt and conv contributions (i.e. just DY)
            mask = (mask & (prompt_mask | conv_mask))
        else:
            raise Exception(f"Error: Unknown sampleType {sampleType}.")

    events['is1l'] = ak.fill_none(mask,False)

    # SFs
    events['sf_1l_muon'] = padded_FOs[:,0].sf_nom_2l_muon
    events['sf_1l_elec'] = padded_FOs[:,0].sf_nom_2l_elec
    events['sf_1l_hi_muon'] = padded_FOs[:,0].sf_hi_2l_muon
    events['sf_1l_hi_elec'] = padded_FOs[:,0].sf_hi_2l_elec
    events['sf_1l_lo_muon'] = padded_FOs[:,0].sf_lo_2l_muon
    events['sf_1l_lo_elec'] = padded_FOs[:,0].sf_lo_2l_elec

    # SR:
    events['is1l_SR'] = (padded_FOs[:,0].isTightLep)
    events['is1l_SR'] = ak.fill_none(events['is1l_SR'],False)

    # FF:
    fakeRateWeight1l(events, padded_FOs[:,0])

# 2l selection (we do not make the ss requirement here)
def add2lMaskAndSFs(events, year, isData, sampleType):
    # FOs and padded FOs
    FOs = events.l_fo_conept_sorted
    padded_FOs = ak.pad_none(FOs,2)
    l0 = padded_FOs[:,0]
    l1 = padded_FOs[:,1]
    # Filters and cleanups
    filter_flags = events.Flag
    filters = filter_flags.goodVertices & filter_flags.globalSuperTightHalo2016Filter & filter_flags.HBHENoiseFilter & filter_flags.HBHENoiseIsoFilter & filter_flags.EcalDeadCellTriggerPrimitiveFilter & filter_flags.BadPFMuonFilter & (((year == "2016")|(year == "2016APV")) | filter_flags.ecalBadCalibFilter) & (isData | filter_flags.eeBadScFilter)
    cleanup = events.minMllAFAS > 12
    muTightCharge = ((abs(padded_FOs[:,0].pdgId)!=13) | (padded_FOs[:,0].tightCharge>=1)) & ((abs(padded_FOs[:,1].pdgId)!=13) | (padded_FOs[:,1].tightCharge>=1))

    # Zee veto
    Zee_veto = (abs(padded_FOs[:,0].pdgId) != 11) | (abs(padded_FOs[:,1].pdgId) != 11) | ( abs ( (padded_FOs[:,0]+padded_FOs[:,1]).mass -91.2) > 10)

    # IDs
    eleID1 = (abs(padded_FOs[:,0].pdgId)!=11) | ((padded_FOs[:,0].convVeto != 0) & (padded_FOs[:,0].lostHits==0) & (padded_FOs[:,0].tightCharge>=2))
    eleID2 = (abs(padded_FOs[:,1].pdgId)!=11) | ((padded_FOs[:,1].convVeto != 0) & (padded_FOs[:,1].lostHits==0) & (padded_FOs[:,1].tightCharge>=2))

    # 2l requirements:
    exclusive = ak.num( FOs[FOs.isTightLep],axis=-1)<3
    dilep = (ak.num(FOs)) >= 2
    pt2515 = (ak.any(FOs[:,0:1].conept > 25.0, axis=1) & ak.any(FOs[:,1:2].conept > 15.0, axis=1))
    mask = (filters & cleanup & dilep & pt2515 & exclusive & eleID1 & eleID2 & muTightCharge)
    #the following mask "ptl0l1" is used for the new CR "2los_CR_lowleppt" to be used in ttgamma EFT work
    ptl0 = ak.any(FOs[:,0:1].conept > 25.0, axis=1)
    ptl1 = ak.any(abs(FOs[:,1:2].conept-12.5) < 2.5, axis=1) #the sub-leading lepton 10 GeV < l1pT < 15 GeV
    ptl0l1  = ptl0 & ptl1

    # MC matching requirement (already passed for data)
    if sampleType == "data":
        pass
    else:
        lep1_match_prompt = ((padded_FOs[:,0].genPartFlav==1) | (padded_FOs[:,0].genPartFlav == 15))
        lep2_match_prompt = ((padded_FOs[:,1].genPartFlav==1) | (padded_FOs[:,1].genPartFlav == 15))
        lep1_charge       = ((padded_FOs[:,0].gen_pdgId*padded_FOs[:,0].pdgId) > 0)
        lep2_charge       = ((padded_FOs[:,1].gen_pdgId*padded_FOs[:,1].pdgId) > 0)
        lep1_match_conv   = (padded_FOs[:,0].genPartFlav==22)
        lep2_match_conv   = (padded_FOs[:,1].genPartFlav==22)
        prompt_mask = ( lep1_match_prompt & lep2_match_prompt & lep1_charge & lep2_charge )
        conv_mask   = ( lep1_match_conv | lep2_match_conv )
        if sampleType == 'prompt':
            mask = (mask & prompt_mask)
        elif sampleType =='conversions':
            mask = (mask & conv_mask)
        elif sampleType =='prompt_and_conversions':
            # Samples that we use for both prompt and conv contributions (i.e. just DY)
            mask = (mask & (prompt_mask | conv_mask))
        else:
            raise Exception(f"Error: Unknown sampleType {sampleType}.")

    mask_nozeeveto = mask
    mask = mask & (  Zee_veto )
    events['is2l'] = ak.fill_none(mask,False)
    events['mll_12'] = ak.fill_none(cleanup,False)   #this is same thing as cleanups. Only implemented this separately for cutflow studies
    events['is2l_nozeeveto'] = ak.fill_none(mask_nozeeveto,False)

    # SFs
    events['sf_2l_muon'] = padded_FOs[:,0].sf_nom_2l_muon*padded_FOs[:,1].sf_nom_2l_muon
    events['sf_2l_elec'] = padded_FOs[:,0].sf_nom_2l_elec*padded_FOs[:,1].sf_nom_2l_elec
    events['sf_2l_hi_muon'] = padded_FOs[:,0].sf_hi_2l_muon*padded_FOs[:,1].sf_hi_2l_muon
    events['sf_2l_hi_elec'] = padded_FOs[:,0].sf_hi_2l_elec*padded_FOs[:,1].sf_hi_2l_elec
    events['sf_2l_lo_muon'] = padded_FOs[:,0].sf_lo_2l_muon*padded_FOs[:,1].sf_lo_2l_muon
    events['sf_2l_lo_elec'] = padded_FOs[:,0].sf_lo_2l_elec*padded_FOs[:,1].sf_lo_2l_elec

    # SR:
    events['is2l_SR'] = (padded_FOs[:,0].isTightLep) & (padded_FOs[:,1].isTightLep)
    events['is2l_SR'] = ak.fill_none(events['is2l_SR'],False)
    padded_photon = ak.pad_none(events.ph_fo_pt_sorted, 1)

    # SFs
    events['sf_2l_photon']    = padded_photon.sf_nom_photon[:,0]
    events['sf_2l_hi_photon'] = padded_photon.sf_hi_photon[:,0]
    events['sf_2l_lo_photon'] = padded_photon.sf_lo_photon[:,0]

    # FF:
    fakeRateWeight2l(events, padded_FOs[:,0], padded_FOs[:,1])


# 3l selection
def add3lMaskAndSFs(events, year, isData, sampleType):

    # FOs and padded FOs
    FOs = events.l_fo_conept_sorted
    padded_FOs = ak.pad_none(FOs,3)

    # Filters and cleanups
    filter_flags = events.Flag
    filters = filter_flags.goodVertices & filter_flags.globalSuperTightHalo2016Filter & filter_flags.HBHENoiseFilter & filter_flags.HBHENoiseIsoFilter & filter_flags.EcalDeadCellTriggerPrimitiveFilter & filter_flags.BadPFMuonFilter & (((year == "2016")|(year == "2016APV")) | filter_flags.ecalBadCalibFilter) & (isData | filter_flags.eeBadScFilter)
    cleanup=events.minMllAFAS > 12

    # IDs
    eleID1=(abs(padded_FOs[:,0].pdgId)!=11) | ((padded_FOs[:,0].convVeto != 0) & (padded_FOs[:,0].lostHits==0))
    eleID2=(abs(padded_FOs[:,1].pdgId)!=11) | ((padded_FOs[:,1].convVeto != 0) & (padded_FOs[:,1].lostHits==0))
    eleID3=(abs(padded_FOs[:,2].pdgId)!=11) | ((padded_FOs[:,2].convVeto != 0) & (padded_FOs[:,2].lostHits==0))

    # Pt requirements for 3rd lepton (different for e and m)
    pt3lmask = ak.any(ak.where(abs(FOs[:,2:3].pdgId)==11,FOs[:,2:3].conept>15.0,FOs[:,2:3].conept>10.0),axis=1)

    # 3l requirements:
    trilep = (ak.num(FOs)) >=3
    pt251510 = (ak.any(FOs[:,0:1].conept > 25.0, axis=1) & ak.any(FOs[:,1:2].conept > 15.0, axis=1) & pt3lmask)
    exclusive = ak.num( FOs[FOs.isTightLep],axis=-1)<4
    mask = (filters & cleanup & trilep & pt251510 & exclusive & eleID1 & eleID2 & eleID3 )

    # MC matching requirement (already passed for data)
    if sampleType == "data":
        pass
    else:
        lep1_match_prompt = ((padded_FOs[:,0].genPartFlav==1) | (padded_FOs[:,0].genPartFlav == 15))
        lep2_match_prompt = ((padded_FOs[:,1].genPartFlav==1) | (padded_FOs[:,1].genPartFlav == 15))
        lep3_match_prompt = ((padded_FOs[:,2].genPartFlav==1) | (padded_FOs[:,2].genPartFlav == 15))
        lep1_match_conv   = (padded_FOs[:,0].genPartFlav==22)
        lep2_match_conv   = (padded_FOs[:,1].genPartFlav==22)
        lep3_match_conv   = (padded_FOs[:,2].genPartFlav==22)
        prompt_mask = ( lep1_match_prompt & lep2_match_prompt & lep3_match_prompt )
        conv_mask   = ( lep1_match_conv | lep2_match_conv | lep3_match_conv)
        if sampleType == 'prompt':
            mask = (mask & prompt_mask)
        elif sampleType =='conversions':
            mask = (mask & conv_mask)
        elif sampleType =='prompt_and_conversions':
            # Samples that we use for both prompt and conv contributions (i.e. just DY)
            mask = (mask & (prompt_mask | conv_mask))
        else:
            raise Exception(f"Error: Unknown sampleType {sampleType}.")

    events['is3l'] = ak.fill_none(mask,False)

    # SFs
    events['sf_3l_muon'] = padded_FOs[:,0].sf_nom_3l_muon*padded_FOs[:,1].sf_nom_3l_muon*padded_FOs[:,2].sf_nom_3l_muon
    events['sf_3l_elec'] = padded_FOs[:,0].sf_nom_3l_elec*padded_FOs[:,1].sf_nom_3l_elec*padded_FOs[:,2].sf_nom_3l_elec
    events['sf_3l_hi_muon'] = padded_FOs[:,0].sf_hi_3l_muon*padded_FOs[:,1].sf_hi_3l_muon*padded_FOs[:,2].sf_hi_3l_muon
    events['sf_3l_hi_elec'] = padded_FOs[:,0].sf_hi_3l_elec*padded_FOs[:,1].sf_hi_3l_elec*padded_FOs[:,2].sf_hi_3l_elec
    events['sf_3l_lo_muon'] = padded_FOs[:,0].sf_lo_3l_muon*padded_FOs[:,1].sf_lo_3l_muon*padded_FOs[:,2].sf_lo_3l_muon
    events['sf_3l_lo_elec'] = padded_FOs[:,0].sf_lo_3l_elec*padded_FOs[:,1].sf_lo_3l_elec*padded_FOs[:,2].sf_lo_3l_elec

    # SR:
    events['is3l_SR'] = (padded_FOs[:,0].isTightLep)  & (padded_FOs[:,1].isTightLep) & (padded_FOs[:,2].isTightLep)
    events['is3l_SR'] = ak.fill_none(events['is3l_SR'],False)

    # FF:
    fakeRateWeight3l(events, padded_FOs[:,0], padded_FOs[:,1], padded_FOs[:,2])


# 4l selection
def add4lMaskAndSFs(events, year, isData):

    # FOs and padded FOs
    FOs = events.l_fo_conept_sorted
    padded_FOs = ak.pad_none(FOs,4)

    # Filters and cleanups
    filter_flags = events.Flag
    filters = filter_flags.goodVertices & filter_flags.globalSuperTightHalo2016Filter & filter_flags.HBHENoiseFilter & filter_flags.HBHENoiseIsoFilter & filter_flags.EcalDeadCellTriggerPrimitiveFilter & filter_flags.BadPFMuonFilter & (((year == "2016")|(year == "2016APV")) | filter_flags.ecalBadCalibFilter) & (isData | filter_flags.eeBadScFilter)
    cleanup = events.minMllAFAS > 12

    # IDs
    eleID1 = ((abs(padded_FOs[:,0].pdgId)!=11) | ((padded_FOs[:,0].convVeto != 0) & (padded_FOs[:,0].lostHits==0)))
    eleID2 = ((abs(padded_FOs[:,1].pdgId)!=11) | ((padded_FOs[:,1].convVeto != 0) & (padded_FOs[:,1].lostHits==0)))
    eleID3 = ((abs(padded_FOs[:,2].pdgId)!=11) | ((padded_FOs[:,2].convVeto != 0) & (padded_FOs[:,2].lostHits==0)))
    eleID4 = ((abs(padded_FOs[:,3].pdgId)!=11) | ((padded_FOs[:,3].convVeto != 0) & (padded_FOs[:,3].lostHits==0)))

    # Pt requirements for 3rd and 4th leptons (different for e and m)
    pt3lmask = ak.any(ak.where(abs(FOs[:,2:3].pdgId)==11,FOs[:,2:3].conept>15.0,FOs[:,2:3].conept>10.0),axis=1)
    pt4lmask = ak.any(ak.where(abs(FOs[:,3:4].pdgId)==11,FOs[:,3:4].conept>15.0,FOs[:,3:4].conept>10.0),axis=1)

    # 4l requirements:
    fourlep  = (ak.num(FOs)) >= 4
    pt25151510 = (ak.any(FOs[:,0:1].conept > 25.0, axis=1) & ak.any(FOs[:,1:2].conept > 15.0, axis=1) & pt3lmask & pt4lmask)
    tightleps = ((padded_FOs[:,0].isTightLep) & (padded_FOs[:,1].isTightLep) & (padded_FOs[:,2].isTightLep) & (padded_FOs[:,3].isTightLep))
    mask = (filters & cleanup & fourlep & pt25151510 & tightleps & eleID1 & eleID2 & eleID3 & eleID4)
    events['is4l'] = ak.fill_none(mask,False)

    # SFs:
    events['sf_4l_muon'] = padded_FOs[:,0].sf_nom_3l_muon*padded_FOs[:,1].sf_nom_3l_muon*padded_FOs[:,2].sf_nom_3l_muon*padded_FOs[:,3].sf_nom_3l_muon
    events['sf_4l_elec'] = padded_FOs[:,0].sf_nom_3l_elec*padded_FOs[:,1].sf_nom_3l_elec*padded_FOs[:,2].sf_nom_3l_elec*padded_FOs[:,3].sf_nom_3l_elec
    events['sf_4l_hi_muon'] = padded_FOs[:,0].sf_hi_3l_muon*padded_FOs[:,1].sf_hi_3l_muon*padded_FOs[:,2].sf_hi_3l_muon*padded_FOs[:,3].sf_hi_3l_muon
    events['sf_4l_hi_elec'] = padded_FOs[:,0].sf_hi_3l_elec*padded_FOs[:,1].sf_hi_3l_elec*padded_FOs[:,2].sf_hi_3l_elec*padded_FOs[:,3].sf_hi_3l_elec
    events['sf_4l_lo_muon'] = padded_FOs[:,0].sf_lo_3l_muon*padded_FOs[:,1].sf_lo_3l_muon*padded_FOs[:,2].sf_lo_3l_muon*padded_FOs[:,3].sf_lo_3l_muon
    events['sf_4l_lo_elec'] = padded_FOs[:,0].sf_lo_3l_elec*padded_FOs[:,1].sf_lo_3l_elec*padded_FOs[:,2].sf_lo_3l_elec*padded_FOs[:,3].sf_lo_3l_elec

    # SR: Don't really need this for 4l, but define it so we can treat 4l category similar to 2lss and 3l
    events['is4l_SR'] = tightleps
    events['is4l_SR'] = ak.fill_none(events['is4l_SR'],False)


def addLepCatMasks(events):

    # FOs and padded FOs
    fo = events.l_fo_conept_sorted
    padded_fo = ak.pad_none(fo,4)
    padded_fo_id = padded_fo.pdgId

    # Find the numbers of e and m in the event
    is_e_mask = (abs(padded_fo_id)==11)
    is_m_mask = (abs(padded_fo_id)==13)
    n_e_1l = ak.sum(is_e_mask[:,0:1],axis=-1)
    n_m_1l = ak.sum(is_m_mask[:,0:1],axis=-1)
    n_e_2l = ak.sum(is_e_mask[:,0:2],axis=-1) # Make sure we only look at first two leps
    n_m_2l = ak.sum(is_m_mask[:,0:2],axis=-1) # Make sure we only look at first two leps
    n_e_3l = ak.sum(is_e_mask[:,0:3],axis=-1) # Make sure we only look at first three leps
    n_m_3l = ak.sum(is_m_mask[:,0:3],axis=-1) # Make sure we only look at first three leps
    n_e_4l = ak.sum(is_e_mask,axis=-1)        # Look at all the leps
    n_m_4l = ak.sum(is_m_mask,axis=-1)        # Look at all the leps

    # 1l masks
    events["is_e"] = ((n_e_2l==1) & (n_m_2l==0))
    events["is_m"] = ((n_e_2l==0) & (n_m_2l==1))

    # 2l masks
    events['is_ee'] = ((n_e_2l==2) & (n_m_2l==0))
    events['is_em'] = ((n_e_2l==1) & (n_m_2l==1))
    events['is_mm'] = ((n_e_2l==0) & (n_m_2l==2))

    # 3l masks
    events['is_eee'] = ((n_e_3l==3) & (n_m_3l==0))
    events['is_eem'] = ((n_e_3l==2) & (n_m_3l==1))
    events['is_emm'] = ((n_e_3l==1) & (n_m_3l==2))
    events['is_mmm'] = ((n_e_3l==0) & (n_m_3l==3))

    # 4l masks
    events['is_eeee'] = ((n_e_4l==4) & (n_m_4l==0))
    events['is_eeem'] = ((n_e_4l==3) & (n_m_4l==1))
    events['is_eemm'] = ((n_e_4l==2) & (n_m_4l==2))
    events['is_emmm'] = ((n_e_4l==1) & (n_m_4l==3))
    events['is_mmmm'] = ((n_e_4l==0) & (n_m_4l==4))
    events['is_gr4l'] = ((n_e_4l+n_m_4l)>4)


def generatorOverlapRemoval(dataset, events, ptCut, etaCut, deltaRCut):
    """Filter generated events with overlapping phase space"""
    genMotherIdx = events.GenPart.genPartIdxMother
    genpdgId = events.GenPart.pdgId
    #calculate maxparent pdgId of the event
    idx = ak.to_numpy(ak.flatten(abs(events.GenPart.pdgId)))
    par = ak.to_numpy(ak.flatten(events.GenPart.genPartIdxMother))
    num = ak.to_numpy(ak.num(events.GenPart.pdgId))
    maxParentFlatten = maxHistoryPDGID(idx,par,num)
    events["GenPart","maxParent"] = ak.unflatten(maxParentFlatten, num)

    #Only the photons that pass the kinematic cuts are potential candidates for overlapping photons
    #If the overlap photon is actually from a non-prompt decay (maxParent > 37), it is not part of the phase space of the separate sample

    overlapPhoSelect = ((events.GenPart.pt>=ptCut) & (events.GenPart.status==1) & (events.GenPart.hasFlags(['isLastCopy'])) &
                        (abs(events.GenPart.eta) < etaCut) &
                        (abs(events.GenPart.pdgId)==22) &
                        ((events.GenPart.maxParent < 37) | (events.GenPart.maxParent == 2212))
                        )
    overlapPhotons = events.GenPart[overlapPhoSelect]

    #Also require that photons are separate from all other gen particles
    #Need not consider neutrinos and don't have to calculate dR between the OverlapPhoton and itself
    finalGen = events.GenPart[(events.GenPart.status==1) & (events.GenPart.pt > 5.0) &
                              ~((abs(events.GenPart.pdgId)==12) | (abs(events.GenPart.pdgId)==14) | (abs(events.GenPart.pdgId)==16)) &
                              ~(overlapPhoSelect)]

    #calculate dR between overlap photons and each gen particle
    phoGenDR = overlapPhotons.metric_table(finalGen)

    ph_iso_mask = ak.any(phoGenDR < deltaRCut, axis=-1)

    #the event is overlapping with the separate sample if there is an overlap photon passing the dR cut, kinematic cuts, and not coming from hadronic activity
    isolated_overlapPhotons = overlapPhotons[~ph_iso_mask]

    if any(x in dataset for x in ["TTTo","DY10to50","DY50"]):   #samples from which the events with well-isolated overlapping photons are to be vetoed
        criteria = (ak.num(isolated_overlapPhotons)==0)
        events["vetoedbyOverlap"] = ~criteria
        events["retainedbyOverlap"] = criteria

    elif any(x in dataset for x in ["TTGamma","ZGToLLG","DYGto2LG-1Jets"]):  #if these samples do not have well-isolated photon, then we remove such events from them
        criteria = (ak.num(isolated_overlapPhotons) >= 1)
        events["vetoedbyOverlap"] = ~criteria
        events["retainedbyOverlap"] = criteria

    else: #might not be necessary
        events["vetoedbyOverlap"] = np.ones(len(events), dtype=bool)
        events["retainedbyOverlap"] = np.ones(len(events), dtype=bool)


def select_nonpromptphoton(events):
    ph = events.photon

    """Filter generated events with overlapping phase space"""
    genMotherIdx = ph.matched_gen.genPartIdxMother
    genpdgId = ph.matched_gen.pdgId
    #calculate maxparent pdgId of the event
    idx = ak.to_numpy(ak.flatten(abs(ph.matched_gen.pdgId)))
    par = ak.to_numpy(ak.flatten(ph.matched_gen.genPartIdxMother))
    num = ak.to_numpy(ak.num(ph.matched_gen.pdgId))
    maxParentFlatten = maxHistoryPDGID(idx,par,num)
    ph["matched_gen","maxParent"] = ak.unflatten(maxParentFlatten, num)

    genmatchedPho = ak.fill_none(ph.matched_gen.pdgId == 22, False)

    # reco photons really generated as electrons
    matchedEle = ak.fill_none(abs(ph.matched_gen.pdgId) == 11, False)
    # if the gen photon has a PDG ID > 25 in its history, it has a hadronic parent
    hadronicParent = ak.fill_none(ph.matched_gen.maxParent > 25, False)

    # define the photon categories for tight photon events
    # a genuine photon is a reconstructed photon which is matched to a generator level photon, and does not have a hadronic parent
    isGenPho = genmatchedPho & ~hadronicParent
    # a hadronic photon is a reconstructed photon which is matched to a generator level photon, but has a hadronic parent
    isHadPho = genmatchedPho & hadronicParent
    # a misidentified electron is a reconstructed photon which is matched to a generator level electron
    isMisIDele = matchedEle
    # a hadronic/fake photon is a reconstructed photon that does not fall within any of the above categories
    isHadFake = ~isMisIDele & ~isHadPho & ~isGenPho

    #let's define a "nonprompt" photon mask
    isNonPromptPho = ~isGenPho

    events['isGenPho'] = isGenPho
    events['isNonPromptPho'] = isNonPromptPho


# Returns the pt of the l+l that form the Z peak
def get_Z_pt(lep_collection,pt_window):

    ll_pairs = ak.combinations(lep_collection, 2, fields=["l0","l1"])
    zpeak_mask = (abs((ll_pairs.l0+ll_pairs.l1).mass - 91.2)<pt_window)
    sfos_mask = (ll_pairs.l0.pdgId == -ll_pairs.l1.pdgId)
    sfosz_mask = ak.fill_none((sfos_mask & zpeak_mask),False)

    pair_invmass = (ll_pairs.l0 + ll_pairs.l1).mass
    pair_invmass_with_sfosz_mask = pair_invmass[sfosz_mask]
    pair_pt = (ll_pairs.l0 + ll_pairs.l1).pt
    pair_pt_with_sfosz_mask = pair_pt[sfosz_mask]

    zpeak_idx = ak.argmin(abs(pair_invmass_with_sfosz_mask - 91.2),keepdims=True,axis=1)
    pt_of_sfosz = pair_pt_with_sfosz_mask[zpeak_idx]

    return ak.flatten(pt_of_sfosz)

def get_ll_pt(lep_collection,pt_window):

    ll_pairs = ak.combinations(lep_collection, 2, fields=["l0","l1"])
    sfos_mask = (ll_pairs.l0.pdgId == -ll_pairs.l1.pdgId)
    sfosz_mask = ak.fill_none((sfos_mask),False)

    pair_invmass = (ll_pairs.l0 + ll_pairs.l1).mass
    pair_invmass_with_sfosz_mask = pair_invmass[sfosz_mask]
    pair_pt = (ll_pairs.l0 + ll_pairs.l1).pt
    pair_pt_with_sfosz_mask = pair_pt[sfosz_mask]

    zpeak_idx = ak.argmin(abs(pair_invmass_with_sfosz_mask - 91.2),keepdims=True,axis=1)
    pt_of_sfosz = pair_pt_with_sfosz_mask[zpeak_idx]

    return ak.flatten(pt_of_sfosz)

def lt_Z_mask(lep0, lep1, tau, pt_window):
    sfosz_l0t_mask = ((lep0.pdgId/abs(lep0.pdgId)) == tau.charge)
    zpeak_mask0 = (abs((lep0+tau).mass - 70.0)<20.0)
    sfosz_l1t_mask = ((lep1.pdgId/abs(lep1.pdgId)) == tau.charge)
    zpeak_mask1 = (abs((lep1+tau).mass - 70.0)<15.0)
    sfosz_mask0 = (sfosz_l0t_mask & zpeak_mask0)
    sfosz_mask1 = (sfosz_l1t_mask & zpeak_mask1)
    sfosz_mask = (sfosz_mask0 | sfosz_mask1)

    return sfosz_mask

def get_Z_peak_mask_llg(lep_collection,photon_collection,pt_window,flavor="os",zmass=91.2):
    #ll_pairs = ak.combinations(lep_collection, 2, fields=["l0","l1"])
    l0 = lep_collection[:,0]
    l1 = lep_collection[:,1]
    mediumcleanphotons_padded = ak.pad_none(photon_collection,1) #pads empty array with a single None value
    llg_Zmass_mask = (abs((l0+l1+mediumcleanphotons_padded[:,0]).mass - zmass) < pt_window)
    sf_lep_mask = (l0.pdgId == -l1.pdgId)
    sfosz_mask_llg = ak.fill_none((llg_Zmass_mask & sf_lep_mask),False)

    return sfosz_mask_llg

def addPhotonSelection(events, sampleType, last_pt_bin, closureTest):

    fo_ph = events.ph_fo_pt_sorted
    padded_fo_ph = ak.pad_none(fo_ph,1)
    a0 = padded_fo_ph[:,0]

    if not closureTest:
        SR_exclusive = (a0.inA_ABCD)
        AR_exclusive = (a0.inB_ABCD)

    else:
        SR_exclusive = (a0.inL_ABCD)
        AR_exclusive = (a0.inR_ABCD)

    #if MC, let's select prompt photons and if Data, do nothing
    if sampleType == "data":
        pass

    else:
        a0_prompt_match = (a0.genPartFlav == 1)

        SR_exclusive = SR_exclusive & a0_prompt_match
        AR_exclusive = AR_exclusive & a0_prompt_match

    events['isSR_ph'] = ak.fill_none(SR_exclusive,False)
    events['isAR_ph'] = ak.fill_none(AR_exclusive,False)

    #additional nonprompt photon uncertainty in the last bin
    last_bin_pt_mask = (a0.pt >= last_pt_bin)

    additional_nonprompt_ph_unc(events, last_bin_pt_mask)

#For Fake rate extraction for main non-prompt estimation, we want to identify prompt MC contribution. Plus, we don't care about Regions A and B
def categorizePhotonsInABCD_FR(events,sampleType):
    fo_ph = events.ph_fo_pt_sorted
    padded_fo_ph = ak.pad_none(fo_ph,1)
    a0 = padded_fo_ph[:,0]

    #if data, just categorize into C and D
    if sampleType=="data":
        C_exclusive = (a0.inC_ABCD)
        D_exclusive = (a0.inD_ABCD)

    #if MC, take the prompt piece only
    else:
        a0_prompt_match = (a0.genPartFlav == 1)

        C_exclusive = (a0.inC_ABCD) & a0_prompt_match
        D_exclusive = (a0.inD_ABCD) & a0_prompt_match

    events['isC_FR_ABCD'] = ak.fill_none(C_exclusive,False)
    events['isD_FR_ABCD'] = ak.fill_none(D_exclusive,False)
    #the following 2 masks are useful if we want to do Data-MC agreement study in the MRs
    events['isC_allph_ABCD'] = ak.fill_none((a0.inC_ABCD),False)
    events['isD_allph_ABCD'] = ak.fill_none((a0.inD_ABCD),False)

def categorize_into_ISRFSR_photon(events):
    ph_collection = events.ph_fo_pt_sorted

    #first make sure we are looking at true photon
    photon_is_true_ph = ak.fill_none(abs(ph_collection.matched_gen.pdgId)==22,False)
    true_ph = ph_collection[photon_is_true_ph]

    #look at the genPartIdx of the true photon and then find the mother of the particle at the genPartIdx
    genpartidx_of_true_ph = true_ph.genPartIdx
    genparticles_at_genpartidx = events.GenPart[genpartidx_of_true_ph]
    mother_of_gen_particle = genparticles_at_genpartidx.distinctParent

    #is the mother lepton, Z, W
    mother_is_lepton = ((abs(mother_of_gen_particle.pdgId)==11) | (abs(mother_of_gen_particle.pdgId)==13) | (abs(mother_of_gen_particle.pdgId)==15))
    mother_is_photon = (abs((mother_of_gen_particle.pdgId)==22)) #sometimes the parent of the photon is itself
    mother_is_Z_or_W = ((abs(mother_of_gen_particle.pdgId)==24) | (abs(mother_of_gen_particle.pdgId)==23))

    has_FSR_photon = ((mother_is_lepton) | (mother_is_Z_or_W))
    has_ISR_photon = ~(has_FSR_photon)

    has_FSR_photon = ak.fill_none(ak.pad_none(has_FSR_photon,1),False)
    has_FSR_photon = has_FSR_photon[:,0]

    has_ISR_photon = ak.fill_none(ak.pad_none(has_ISR_photon,1),False)
    has_ISR_photon = has_ISR_photon[:,0]

    return has_ISR_photon, has_FSR_photon
