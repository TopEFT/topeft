'''
 selection.py

 This script contains several functions that implement the some event selection.
 The functinos defined here can be used to define a selection, signal/control region, etc.
 The functions are called with (jagged)arrays as imputs plus some custom paramenters and return a boolean mask.

'''

import awkward as ak

from topeft.modules.corrections import fakeRateWeight1l, fakeRateWeight2l, fakeRateWeight3l


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
    }

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


# Returns a mask for events with a same flavor opposite (same) sign pair close to the Z
# Mask will be True if any combination of 2 leptons from within the given collection satisfies the requirement
def get_Z_peak_mask(lep_collection,pt_window,flavor="os"):
    ll_pairs = ak.combinations(lep_collection, 2, fields=["l0","l1"])
    zpeak_mask = (abs((ll_pairs.l0+ll_pairs.l1).mass - 91.2)<pt_window)
    if flavor == "os":
        sf_mask = (ll_pairs.l0.pdgId == -ll_pairs.l1.pdgId)
    elif flavor == "ss":
        sf_mask = (ll_pairs.l0.pdgId == ll_pairs.l1.pdgId)
    elif flavor == "as": # Same flav any sign
        sf_mask = ((ll_pairs.l0.pdgId == ll_pairs.l1.pdgId) | (ll_pairs.l0.pdgId == -ll_pairs.l1.pdgId))
    else:
        raise Exception(f"Error: flavor requirement \"{flavor}\" is unknown.")
    sfosz_mask = ak.flatten(ak.any((zpeak_mask & sf_mask),axis=1,keepdims=True)) # Use flatten here because it is too nested (i.e. it looks like this [[T],[F],[T],...], and want this [T,F,T,...]))
    return sfosz_mask

# Returns a mask for all events with mll of leading pt lepton within low region of off-Z
def get_off_Z_mask_low(lep_collection,pt_window):
    ll_pairs = ak.combinations(lep_collection[:,0:2], 2, fields=["l0","l1"])   
    zpeak_mask = ((91.2-(ll_pairs.l0+ll_pairs.l1).mass)>pt_window)
    sfosz_mask = ak.flatten(ak.any((zpeak_mask),axis=1,keepdims=True)) # Use flatten here because it is too nested (i.e. it looks like this [[T],[F],[T],...], and want this [T,F,T,...]))
    return sfosz_mask

# Returns a mask for all events with any os lepton pair within low region of off-Z
#def get_off_Z_mask_low(lep_collection,pt_window,flavor="os"):
#    ll_pairs = ak.combinations(lep_collection, 2, fields=["l0","l1"])
#    zpeak_mask = ((91.2-(ll_pairs.l0+ll_pairs.l1).mass)>pt_window)
#    if flavor == "os":
#        sf_mask = (ll_pairs.l0.pdgId == -ll_pairs.l1.pdgId)
#    elif flavor == "ss":
#        sf_mask = (ll_pairs.l0.pdgId == ll_pairs.l1.pdgId)
#    elif flavor == "as": # Same flav any sign
#        sf_mask = ((ll_pairs.l0.pdgId == ll_pairs.l1.pdgId) | (ll_pairs.l0.pdgId == -ll_pairs.l1.pdgId))
#    else:
#        raise Exception(f"Error: flavor requirement \"{flavor}\" is unknown.")
#    sfosz_mask = ak.flatten(ak.any((zpeak_mask & sf_mask),axis=1,keepdims=True)) # Use flatten here because it is too nested (i.e. it looks like this [[T],[F],[T],...], and want this [T,F,T,...]))
#    return sfosz_mask

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
