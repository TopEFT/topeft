'''
 selection.py

 This script contains several functions that implement the some event selection. 
 The functinos defined here can be used to define a selection, signal/control region, etc.
 The functions are called with (jagged)arrays as imputs plus some custom paramenters and return a boolean mask.

'''

import numpy as np
import awkward as ak

from topcoffea.modules.corrections import fakeRateWeight2l, fakeRateWeight3l

def passNJets(nJets, lim=2):
  return nJets >= lim

def passMETcut(met, metCut=40):
  return met >= metCut

# Datasets:
# SingleElec, SingleMuon
# DoubleElec, DoubleMuon, MuonEG
# Overlap removal at trigger level... singlelep, doublelep, triplelep

triggers = {
  'SingleMuonTriggers' : ['IsoMu24', 'IsoMu27'],
  'SingleElecTriggers' : ['Ele32_WPTight_Gsf', 'Ele35_WPTight_Gsf'],
  'DoubleMuonTrig' : ['Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ', 'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8'],
  'DoubleElecTrig' : ['Ele23_Ele12_CaloIdL_TrackIdL_IsoVL', 'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ'],
  'MuonEGTrig' : ['Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL', 'Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ', 'Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ'],
  'TripleElecTrig' : ['Ele16_Ele12_Ele8_CaloIdL_TrackIdL'],
  'TripleMuonTrig' : ['TripleMu_12_10_5'],
  'DoubleMuonElecTrig' : ['DiMu9_Ele9_CaloIdL_TrackIdL_DZ'],
  'DoubleElecMuonTrig' : ['Mu8_DiEle12_CaloIdL_TrackIdL'],
}

triggersForFinalState = {
  'ee' : {
      'MC': triggers['SingleElecTriggers']+triggers['DoubleElecTrig'],
      'EGamma'     : triggers['SingleElecTriggers']+triggers['DoubleElecTrig'],
  },
  'em' : {
      'MC': triggers['SingleElecTriggers']+triggers['SingleMuonTriggers']+triggers['MuonEGTrig'],
      'EGamma'     : triggers['SingleElecTriggers'],
      'MuonEG'     : triggers['MuonEGTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],
  },
  'mm' : {
      'MC': triggers['SingleMuonTriggers']+triggers['DoubleMuonTrig'],
      'DoubleMuon' : triggers['DoubleMuonTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],
  },
  'eee' : {
      'MC': triggers['SingleElecTriggers']+triggers['DoubleElecTrig']+triggers['TripleElecTrig'],
      'EGamma' : triggers['SingleElecTriggers']+triggers['DoubleElecTrig']+triggers['TripleElecTrig'],
  },
  'mmm' : {
      'MC': triggers['SingleMuonTriggers']+triggers['DoubleMuonTrig']+triggers['TripleMuonTrig'],
      'DoubleMuon' : triggers['DoubleMuonTrig']+triggers['TripleMuonTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],
  },
  'eem' : {
      'MC': triggers['SingleMuonTriggers']+triggers['SingleElecTriggers']+triggers['DoubleElecTrig']+triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig'],
      'MuonEG' : triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig'],
      'EGamma' : triggers['SingleElecTriggers']+triggers['DoubleElecTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],
  },
  'mme' : {
      'MC': triggers['SingleMuonTriggers']+triggers['SingleElecTriggers']+triggers['DoubleMuonTrig']+triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig'],
      'MuonEG' : triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig'],
      'EGamma' : triggers['SingleElecTriggers'],
      'DoubleMuon' : triggers['DoubleMuonTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],
  },
  'eeee' : {
      'MC': triggers['SingleElecTriggers']+triggers['DoubleElecTrig']+triggers['TripleElecTrig'],
      'EGamma' : triggers['SingleElecTriggers']+triggers['DoubleElecTrig']+triggers['TripleElecTrig'],
  },
  'mmmm' : {
      'MC': triggers['SingleMuonTriggers']+triggers['DoubleMuonTrig']+triggers['TripleMuonTrig'],
      'DoubleMuon' : triggers['DoubleMuonTrig']+triggers['TripleMuonTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],
  },
  'eeem' : {
      'MC': triggers['TripleElecTrig']+triggers['SingleMuonTriggers']+triggers['SingleElecTriggers']+triggers['DoubleElecTrig']+triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig'],
      'MuonEG' : triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig'],
      'EGamma' : triggers['TripleElecTrig']+triggers['SingleElecTriggers']+triggers['DoubleElecTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],
  },
  'eemm' : {
      'MC': triggers['SingleMuonTriggers']+triggers['SingleElecTriggers']+triggers['DoubleElecTrig']+triggers['DoubleMuonTrig']+triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['DoubleMuonElecTrig'],
      'MuonEG' : triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['DoubleMuonElecTrig'],
      'EGamma' : triggers['SingleElecTriggers']+triggers['DoubleElecTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],
      'DoubleMuon' : triggers['DoubleMuonTrig'],
  },
  'mmme' : {
      'MC': triggers['TripleMuonTrig']+triggers['SingleMuonTriggers']+triggers['SingleElecTriggers']+triggers['DoubleMuonTrig']+triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig'],
      'MuonEG' : triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig'],
      'EGamma' : triggers['SingleElecTriggers'],
      'DoubleMuon' : triggers['TripleMuonTrig']+triggers['DoubleMuonTrig'],
      'SingleMuon' : triggers['SingleMuonTriggers'],   
  }
}

triggersNotForFinalState = {
  'ee' : {'EGamma' : [],},
  'em' : {
      'MuonEG'     : [],
      'EGamma'     : triggers['MuonEGTrig'],
      'SingleMuon' : triggers['MuonEGTrig'],
  },
  'mm' : {
      'DoubleMuon' : [],
      'SingleMuon' : triggers['DoubleMuonTrig'],
  },
  'eee' : { 'EGamma' : [],},
  'mmm' : {
      'DoubleMuon' : [],
      'SingleMuon' : triggers['DoubleMuonTrig']+triggers['TripleMuonTrig'],
  },
  'eem' : {
      'MuonEG' : [], 
      'EGamma' : triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig'],
      'SingleMuon' :  triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['SingleElecTriggers']+triggers['DoubleElecTrig'],
  },
  'mme' : {
      'MuonEG' : [],
      'EGamma' : triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig'],
      'DoubleMuon' : triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig']+triggers['SingleElecTriggers'],
      'SingleMuon' : triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig']+triggers['SingleElecTriggers']+triggers['DoubleMuonTrig'],
  },
  'eeee' : { 'EGamma' : [],},
  'mmmm' : {
      'DoubleMuon' : [],
      'SingleMuon' : triggers['DoubleMuonTrig']+triggers['TripleMuonTrig'],   
  },
  'eeem' : {
      'MuonEG' : [], 
      'EGamma' : triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig'],
      'SingleMuon' :  triggers['TripleElecTrig']+triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['SingleElecTriggers']+triggers['DoubleElecTrig'],   
  },
  'eemm' : {
      'MuonEG' : [], 
      'EGamma' : triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['DoubleMuonElecTrig'],
      'SingleMuon' :  triggers['MuonEGTrig']+triggers['DoubleElecMuonTrig']+triggers['DoubleMuonElecTrig']+triggers['SingleElecTriggers']+triggers['DoubleElecTrig']+triggers['DoubleMuonTrig'], 
  },
  'mmme' : {
      'MuonEG' : [],
      'EGamma' : triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig'],
      'DoubleMuon' : triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig']+triggers['SingleElecTriggers'],
      'SingleMuon' : triggers['MuonEGTrig']+triggers['DoubleMuonElecTrig']+triggers['SingleElecTriggers']+triggers['DoubleMuonTrig']+triggers['TripleMuonTrig'],   
  }
}

def passTrigger(df, cat, isData=False, dataName=''):
  tpass = np.zeros_like(np.array(df.MET.pt), dtype=np.bool)
  df = df.HLT
  if not isData: 
    paths = triggersForFinalState[cat]['MC']
    for path in paths: tpass = tpass | df[path]
  else:
    passTriggers    = triggersForFinalState[cat][dataName] if dataName in triggersForFinalState[cat].keys() else []
    notPassTriggers = triggersNotForFinalState[cat][dataName] if dataName in triggersNotForFinalState[cat].keys() else []
    for path in passTriggers: tpass = tpass| df[path]
    for path in notPassTriggers: tpass = (tpass)&(df[path]==0)
  return tpass

def triggerFor4l(df, nMuon, nElec, isData, dataName=''):
  is4lmask = ((nElec+nMuon)>=4)
  is4l0m = (is4lmask)&(nMuon==0)
  is4l1m = (is4lmask)&(nMuon==1)
  is4l2m = (is4lmask)&(nMuon==2)
  is4l3m = (is4lmask)&(nMuon==3)
  is4l4m = (is4lmask)&(nMuon>=4)
  trig4l0m = passTrigger(df, 'eeee', isData, dataName)
  trig4l1m = passTrigger(df, 'eeem', isData, dataName)
  trig4l2m = passTrigger(df, 'eemm', isData, dataName)
  trig4l3m = passTrigger(df, 'mmme', isData, dataName)
  trig4l4m = passTrigger(df, 'mmmm', isData, dataName)
  trigMask = ( ( (is4l0m)&(trig4l0m) )|( (is4l1m)&(trig4l1m) )|( (is4l2m)&(trig4l2m) )|( (is4l3m)&(trig4l3m) )|( (is4l4m)&(trig4l4m) ) )
  return trigMask


# 2lss selection
def add2lssMaskAndSFs(events, year, isData):

    # FOs and padded FOs
    FOs = events.l_fo_conept_sorted
    padded_FOs = ak.pad_none(FOs, 2)

    # Filters and cleanups
    filter_flags = events.Flag
    filters = filter_flags.goodVertices & filter_flags.globalSuperTightHalo2016Filter & filter_flags.HBHENoiseFilter & filter_flags.HBHENoiseIsoFilter & filter_flags.EcalDeadCellTriggerPrimitiveFilter & filter_flags.BadPFMuonFilter & ((year == 2016) | filter_flags.ecalBadCalibFilter) & (isData | filter_flags.eeBadScFilter)
    cleanup = events.minMllAFAS > 12
    muTightCharge = ((abs(padded_FOs[:,0].pdgId)!=13) | (padded_FOs[:,0].tightCharge>=1)) & ((abs(padded_FOs[:,1].pdgId)!=13) | (padded_FOs[:,1].tightCharge>=1))

    # Zee veto
    Zee_veto = (abs(padded_FOs[:,0].pdgId) != 11) | (abs(padded_FOs[:,1].pdgId) != 11) | ( abs ( (padded_FOs[:,0]+padded_FOs[:,1]).mass -91.2) > 10)

    # IDs
    eleID1 = (abs(padded_FOs[:,0].pdgId)!=11) | ((padded_FOs[:,0].convVeto != 0) & (padded_FOs[:,0].lostHits==0) & (padded_FOs[:,0].tightCharge>=2))
    eleID2 = (abs(padded_FOs[:,1].pdgId)!=11) | ((padded_FOs[:,1].convVeto != 0) & (padded_FOs[:,1].lostHits==0) & (padded_FOs[:,1].tightCharge>=2))

    # Jet requirements:
    njet4 = (events.njets>3)

    # 2lss requirements:
    exclusive = ak.num( FOs[FOs.isTightLep],axis=-1)<3
    dilep = ( ak.num(FOs)) >= 2 
    pt2515 = ak.any(FOs[:,0:1].conept > 25.0, axis=1) & ak.any(FOs[:,1:2].conept > 15.0, axis=1)
    mask = (filters & cleanup & dilep & pt2515 & exclusive & Zee_veto & eleID1 & eleID2 & muTightCharge & njet4) #     & Z_veto
    events['is2lss'] = ak.fill_none(mask,False)

    # SFs
    events['sf_2lss'] = padded_FOs[:,0].sf_nom*padded_FOs[:,1].sf_nom
    events['sf_2lss_hi'] = padded_FOs[:,0].sf_hi*padded_FOs[:,1].sf_hi
    events['sf_2lss_lo'] = padded_FOs[:,0].sf_lo*padded_FOs[:,1].sf_lo

    # SR:
    events['is2lss_SR'] = (padded_FOs[:,0].isTightLep) & (padded_FOs[:,1].isTightLep)
    events['is2lss_SR'] = ak.fill_none(events['is2lss_SR'],False)

    # FF:
    fakeRateWeight2l(events, padded_FOs[:,0], padded_FOs[:,1])


# 3l selection
def add3lMaskAndSFs(events, year, isData):

    # FOs and padded FOs
    FOs=events.l_fo_conept_sorted
    padded_FOs = ak.pad_none(FOs, 3)

    # Filters and cleanups
    filter_flags = events.Flag
    filters = filter_flags.goodVertices & filter_flags.globalSuperTightHalo2016Filter & filter_flags.HBHENoiseFilter & filter_flags.HBHENoiseIsoFilter & filter_flags.EcalDeadCellTriggerPrimitiveFilter & filter_flags.BadPFMuonFilter & ((year == 2016) | filter_flags.ecalBadCalibFilter) & (isData | filter_flags.eeBadScFilter)
    cleanup=events.minMllAFAS > 12

    # IDs
    eleID1=(abs(padded_FOs[:,0].pdgId)!=11) | ((padded_FOs[:,0].convVeto != 0) & (padded_FOs[:,0].lostHits==0))
    eleID2=(abs(padded_FOs[:,1].pdgId)!=11) | ((padded_FOs[:,1].convVeto != 0) & (padded_FOs[:,1].lostHits==0))
    eleID3=(abs(padded_FOs[:,2].pdgId)!=11) | ((padded_FOs[:,2].convVeto != 0) & (padded_FOs[:,2].lostHits==0))

    # Jet requirements:
    njet2 = (events.njets>1)

    # 3l requirements:
    trilep = ( ak.num(FOs)) >=3
    pt251510 = ak.any(FOs[:,0:1].conept > 25.0, axis=1) & ak.any(FOs[:,1:2].conept > 15.0, axis=1) & ak.any(FOs[:,2:3].conept > 10.0, axis=1)
    exclusive = ak.num( FOs[FOs.isTightLep],axis=-1)<4
    mask = (filters & cleanup & trilep & pt251510 & exclusive & eleID1 & eleID2 & eleID3 & njet2) 
    events['is3l'] = ak.fill_none(mask,False)

    # SFs
    events['sf_3l'] = padded_FOs[:,0].sf_nom*padded_FOs[:,1].sf_nom*padded_FOs[:,2].sf_nom
    events['sf_3l_hi'] = padded_FOs[:,0].sf_hi*padded_FOs[:,1].sf_hi*padded_FOs[:,2].sf_hi
    events['sf_3l_lo'] = padded_FOs[:,0].sf_lo*padded_FOs[:,1].sf_lo*padded_FOs[:,2].sf_lo

    # SR:
    events['is3l_SR'] = (padded_FOs[:,0].isTightLep)  & (padded_FOs[:,1].isTightLep) & (padded_FOs[:,2].isTightLep)
    events['is3l_SR'] = ak.fill_none(events['is3l_SR'],False)
    
    # FF:
    fakeRateWeight3l(events, padded_FOs[:,0], padded_FOs[:,1], padded_FOs[:,2])

# 4l selection
def add4lMaskAndSFs(events, year, isData):

    # FOs and padded FOs
    FOs=events.l_fo_conept_sorted
    padded_FOs=ak.pad_none(FOs, 4)

    # Filters and cleanups
    filter_flags = events.Flag
    filters = filter_flags.goodVertices & filter_flags.globalSuperTightHalo2016Filter & filter_flags.HBHENoiseFilter & filter_flags.HBHENoiseIsoFilter & filter_flags.EcalDeadCellTriggerPrimitiveFilter & filter_flags.BadPFMuonFilter & ((year == 2016) | filter_flags.ecalBadCalibFilter) & (isData | filter_flags.eeBadScFilter)
    cleanup = events.minMllAFAS > 12

    # IDs
    eleID1 = ((abs(padded_FOs[:,0].pdgId)!=11) | ((padded_FOs[:,0].convVeto != 0) & (padded_FOs[:,0].lostHits==0)))
    eleID2 = ((abs(padded_FOs[:,1].pdgId)!=11) | ((padded_FOs[:,1].convVeto != 0) & (padded_FOs[:,1].lostHits==0)))
    eleID3 = ((abs(padded_FOs[:,2].pdgId)!=11) | ((padded_FOs[:,2].convVeto != 0) & (padded_FOs[:,2].lostHits==0)))
    eleID4 = ((abs(padded_FOs[:,3].pdgId)!=11) | ((padded_FOs[:,3].convVeto != 0) & (padded_FOs[:,3].lostHits==0)))

    # Jet requirements:
    njet2 = (events.njets>=2)

    # 4l requirements:
    fourlep  = (ak.num(FOs)) >= 4
    pt25151510 = ak.any(FOs[:,0:1].conept > 25.0, axis=1) & ak.any(FOs[:,1:2].conept > 15.0, axis=1) & ak.any(FOs[:,2:3].conept > 10.0, axis=1) & ak.any(FOs[:,3:4].conept > 10.0, axis=1) # TODO: Check on these thresholds!!!
    tightleps = (padded_FOs[:,0].isTightLep) & (padded_FOs[:,1].isTightLep) & (padded_FOs[:,2].isTightLep) & (padded_FOs[:,3].isTightLep) 
    mask = (filters & cleanup & fourlep & pt25151510 & tightleps & eleID1 & eleID2 & eleID3 & eleID4 & njet2)
    events['is4l'] = ak.fill_none(mask,False)

    # SFs:
    events['sf_4l'] = padded_FOs[:,0].sf_nom*padded_FOs[:,1].sf_nom*padded_FOs[:,2].sf_nom*padded_FOs[:,3].sf_nom
    events['sf_4l_hi'] = padded_FOs[:,0].sf_hi*padded_FOs[:,1].sf_hi*padded_FOs[:,2].sf_hi*padded_FOs[:,3].sf_hi
    events['sf_4l_lo'] = padded_FOs[:,0].sf_lo*padded_FOs[:,1].sf_lo*padded_FOs[:,2].sf_lo*padded_FOs[:,3].sf_lo

    # SR: Don't really need this for 4l, but define it so we can treat 4l category similar to 2lss and 3l
    events['is4l_SR'] = tightleps
    events['is4l_SR'] = ak.fill_none(events['is4l_SR'],False)
