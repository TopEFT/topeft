'''
 selection.py

 This script contains several functions that implement the some event selection. 
 The functinos defined here can be used to define a selection, signal/control region, etc.
 The functions are called with (jagged)arrays as imputs plus some custom paramenters and return a boolean mask.

'''

import numpy as np

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

##################################################################################
### Fake rates

import uproot
from coffea import hist, lookup_tools
import os, sys
from topcoffea.modules.paths import topcoffea_path
import awkward as ak

extFakeRates = lookup_tools.extractor()
basepathFromTTH = 'data/fromTTH/fakerate/'

# Electron reco
#histoName = ''
#for leptype in ['el', 'mu']:
#  for year in [2016, 2017, 2018]:
#    hname = 'FR_mva090_mu_data_comb' if leptype == 'mu' else 'FR_mva090_el_data_comb_NC'
#    extFakeRates.add_weight_sets( ["fr_%s_%i %s %s"%(leptype, year, hname, topcoffea_path(basepathFromTTH+'fr_%i.root'%year) )] ) # pt, abs(eta)


#extFakeRates.finalize()
#FRevaluatior = extFakeRates.make_evaluator()


### Charge flips
pathToChargeFlips = topcoffea_path('data/fromTTH/fliprates/')
flipFileName = lambda year : pathToChargeFlips+"ElectronChargeMisIdRates_era%s_2020Feb13.root"%str(year)[:4]
fliphname = 'chargeMisId'
# pt, abseta

extChargeFlips = lookup_tools.extractor()

for year in ['2016', '2016APV', '2017', '2018']:
  extChargeFlips.add_weight_sets(['fr_%s %s %s'%(year, fliphname, flipFileName(year))])

extChargeFlips.finalize()
ChargeFlipsEvaluator = extChargeFlips.make_evaluator()

def GetChargeFlipRate(pt, abseta, year):
  return ChargeFlipsEvaluator['fr_%s'%str(year)](pt, abseta)
