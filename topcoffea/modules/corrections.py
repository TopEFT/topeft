'''
 This script is used to transform scale factors, which are tipically provided as 2D histograms within root files,
 into coffea format of corrections.
'''

#import uproot, uproot_methods
import uproot
from coffea import hist, lookup_tools
import os, sys
from topcoffea.modules.paths import topcoffea_path
import numpy as np
import awkward as ak
import gzip
import pickle
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory
from coffea.btag_tools.btagscalefactor import BTagScaleFactor
from topcoffea.modules.GetValuesFromJsons import get_param
from coffea.lookup_tools import txt_converters, rochester_lookup

basepathFromTTH = 'data/fromTTH/'

###### Lepton scale factors
################################################################
extLepSF = lookup_tools.extractor()

# New UL Lepton SFs
# Muon: reco
extLepSF.add_weight_sets(["MuonRecoSF_2018 NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s"%topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2018_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2018_er NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s"%topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2018_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2017 NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s"%topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2017_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2017_er NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s"%topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2017_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2016 NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s"%topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2016postVFP_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2016_er NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s"%topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2016postVFP_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2016APV NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s"%topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2016preVFP_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2016APV_er NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s"%topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2016preVFP_UL_trackerMuon.json')])
# Muon: loose&tight
extLepSF.add_weight_sets(["MuonSF_2018 EGamma_SF2D %s"%topcoffea_path('data/leptonSF/muon/egammaEffi2018_EGM2D.root')])
extLepSF.add_weight_sets(["MuonSF_2018_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/muon/egammaEffi2018_EGM2D.root')])
extLepSF.add_weight_sets(["MuonSF_2017_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/muon/egammaEffi2017_EGM2D.root')])
extLepSF.add_weight_sets(["MuonSF_2017 EGamma_SF2D %s"%topcoffea_path('data/leptonSF/muon/egammaEffi2017_EGM2D.root')])
extLepSF.add_weight_sets(["MuonSF_2016_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/muon/egammaEffi2016_EGM2D.root')])
extLepSF.add_weight_sets(["MuonSF_2016 EGamma_SF2D %s"%topcoffea_path('data/leptonSF/muon/egammaEffi2016_EGM2D.root')])
extLepSF.add_weight_sets(["MuonSF_2016APV_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/muon/egammaEffi2016APV_EGM2D.root')])
extLepSF.add_weight_sets(["MuonSF_2016APV EGamma_SF2D %s"%topcoffea_path('data/leptonSF/muon/egammaEffi2016APV_EGM2D.root')])
# Elec: reco
extLepSF.add_weight_sets(["ElecRecoSFAb_2018 EGamma_SF2D %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2018_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2018_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2018_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2018 EGamma_SF2D %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2018_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2018_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2018_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2017 EGamma_SF2D %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2017_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2017_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2017_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2017 EGamma_SF2D %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2017_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2017_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2017_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2016 EGamma_SF2D %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2016_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2016_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2016_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2016 EGamma_SF2D %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2016_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2016_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2016_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2016APV EGamma_SF2D %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2016APV_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2016APV EGamma_SF2D %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2016APV_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_ptBelow20_EGM2D.root')])
# Elec: loose&tight
extLepSF.add_weight_sets(["ElecSF_2018_2lss EGamma_SF2D %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2018_2lss_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2018_2lss_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2018_2lss_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2018_3l EGamma_SF2D %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2018_3l_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2018_3l_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2018_3l_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2017_2lss EGamma_SF2D %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2017_2lss_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2017_2lss_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2017_2lss_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2017_3l EGamma_SF2D %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2017_3l_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2017_3l_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2017_3l_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2016_2lss EGamma_SF2D %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2016_2lss_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2016_2lss_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2016_2lss_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2016_3l EGamma_SF2D %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2016_3l_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2016_3l_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2016_3l_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2016APV_2lss EGamma_SF2D %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_2lss_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2016APV_2lss_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_2lss_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2016APV_3l EGamma_SF2D %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_3l_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2016APV_3l_er EGamma_SF2D_error %s"%topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_3l_EGM2D.root')])

# Fake rate 
# todo: check that these are the same as the "recorrected"
for year in [2016, 2017, 2018]:
  for syst in ['','_up','_down','_be1','_be2','_pt1','_pt2']:
    extLepSF.add_weight_sets([("MuonFR_{year}{syst} FR_mva085_mu_data_comb_recorrected{syst} %s"%topcoffea_path(basepathFromTTH+'fakerate/fr_{year}_recorrected.root')).format(year=year,syst=syst)])
    extLepSF.add_weight_sets([("ElecFR_{year}{syst} FR_mva080_el_data_comb_NC_recorrected{syst} %s"%topcoffea_path(basepathFromTTH+'fakerate/fr_{year}_recorrected.root')).format(year=year,syst=syst)])

# Flip rates                                                                                                                                                                                                       
for year in [2016, 2017, 2018]:
  extLepSF.add_weight_sets([("EleFlip_{year} chargeMisId %s"%topcoffea_path(basepathFromTTH+'fliprates/ElectronChargeMisIdRates_era{year}_2020Feb13.root')).format(year=year,syst=syst)])


extLepSF.finalize()
SFevaluator = extLepSF.make_evaluator()


ffSysts=['','_up','_down','_be1','_be2','_pt1','_pt2']
def AttachPerLeptonFR(leps, flavor, year):
  if year == '2016APV': year = '2016'
  for syst in ffSysts:
    fr=SFevaluator['{flavor}FR_{year}{syst}'.format(flavor=flavor,year=year,syst=syst)](leps.conept, np.abs(leps.eta) )
    leps['fakefactor%s'%syst]=ak.fill_none(-fr/(1-fr),0) # this is the factor that actually enters the expressions
  if flavor=="Elec":
    leps['fliprate']=SFevaluator["EleFlip_%s"%year]( np.maximum(25.,leps.pt), np.abs(leps.eta))
  else:
    leps['fliprate']=np.zeros_like(leps.pt)


def fakeRateWeight2l(events, lep1, lep2):
  for syst in ffSysts:
    fakefactor_2l =  (~lep1.isTightLep | ~lep2.isTightLep)*(-1) + (1)*(lep1.isTightLep & lep2.isTightLep) # if all are tight the FF is 1 because events are in the SR 
    fakefactor_2l =  fakefactor_2l*(lep1.isTightLep + (~lep1.isTightLep)*getattr(lep1,'fakefactor%s'%syst))
    fakefactor_2l =  fakefactor_2l*(lep2.isTightLep + (~lep2.isTightLep)*getattr(lep2,'fakefactor%s'%syst))
    events['fakefactor_2l%s'%syst]=fakefactor_2l
  events['flipfactor_2l']=1*((lep1.charge+lep2.charge)!=0) + (((lep1.fliprate+lep2.fliprate))*((lep1.charge+lep2.charge)==0)) # only apply fliprate for OS events. to handle the OS control regions later :) #  + 

def fakeRateWeight3l(events, lep1, lep2, lep3):
  for syst in ffSysts:
    fakefactor_3l = (~lep1.isTightLep | ~lep2.isTightLep | ~lep3.isTightLep)*(-1) + (1)*(lep1.isTightLep & lep2.isTightLep & lep3.isTightLep) # if all are tight the FF is 1 because events are in the SR  and we dont want to weight them
    fakefactor_3l = fakefactor_3l*(lep1.isTightLep + (~lep1.isTightLep)*getattr(lep1,'fakefactor%s'%syst))
    fakefactor_3l = fakefactor_3l*(lep2.isTightLep + (~lep2.isTightLep)*getattr(lep2,'fakefactor%s'%syst))
    fakefactor_3l = fakefactor_3l*(lep3.isTightLep + (~lep3.isTightLep)*getattr(lep3,'fakefactor%s'%syst))
    events['fakefactor_3l%s'%syst]=fakefactor_3l


def AttachMuonSF(muons, year):
  '''
    Description:
      Inserts 'sf_nom', 'sf_hi', and 'sf_lo' into the muons array passed to this function. These
      values correspond to the nominal, up, and down muon scalefactor values respectively.
  '''
  eta = np.abs(muons.eta)
  pt = muons.pt
  if year not in ['2016','2016APV','2017','2018']: raise Exception(f"Error: Unknown year \"{year}\".")
  reco_sf  = np.where(pt<20,SFevaluator['MuonRecoSF_{year}'.format(year=year)](eta,pt),1) #sf=1 when pt>20 becuase there is no reco SF available
  reco_err = np.where(pt<20,SFevaluator['MuonRecoSF_{year}_er'.format(year=year)](eta,pt),0) #sf error =0 when pt>20 becuase there is no reco SF available
  new_sf  = SFevaluator['MuonSF_{year}'.format(year=year)](eta,pt)
  new_err = SFevaluator['MuonSF_{year}_er'.format(year=year)](eta,pt)

  muons['sf_nom_2l'] = new_sf*reco_sf
  muons['sf_hi_2l']  = (new_sf+new_err)*(reco_sf+reco_err)
  muons['sf_lo_2l']  = (new_sf-new_err)*(reco_sf-reco_err)
  muons['sf_nom_3l'] = new_sf*reco_sf
  muons['sf_hi_3l']  = (new_sf+new_err)*(reco_sf+reco_err)
  muons['sf_lo_3l']  = (new_sf-new_err)*(reco_sf-reco_err)


def AttachElectronSF(electrons, year):
  '''
    Description:
      Inserts 'sf_nom', 'sf_hi', and 'sf_lo' into the electrons array passed to this function. These
      values correspond to the nominal, up, and down electron scalefactor values respectively.
  '''
  eta = electrons.eta
  pt = electrons.pt

  if year not in ['2016','2016APV','2017','2018']: raise Exception(f"Error: Unknown year \"{year}\".")
  
  reco_sf  = np.where(pt<20,SFevaluator['ElecRecoSFBe_{year}'.format(year=year)](eta,pt),SFevaluator['ElecRecoSFAb_{year}'.format(year=year)](eta,pt))
  reco_err = np.where(pt<20,SFevaluator['ElecRecoSFBe_{year}_er'.format(year=year)](eta,pt),SFevaluator['ElecRecoSFAb_{year}_er'.format(year=year)](eta,pt))
  new_sf_2l  = SFevaluator['ElecSF_{year}_2lss'.format(year=year)](np.abs(eta),pt)
  new_err_2l = SFevaluator['ElecSF_{year}_2lss_er'.format(year=year)](np.abs(eta),pt)
  new_sf_3l  = SFevaluator['ElecSF_{year}_3l'.format(year=year)](np.abs(eta),pt)
  new_err_3l = SFevaluator['ElecSF_{year}_3l_er'.format(year=year)](np.abs(eta),pt)

  electrons['sf_nom_2l'] = reco_sf*new_sf_2l
  electrons['sf_hi_2l']  = (reco_sf + reco_err) * (new_sf_2l + new_err_2l)
  electrons['sf_lo_2l']  = (reco_sf - reco_err) * (new_sf_2l - new_err_2l)
  electrons['sf_nom_3l'] = reco_sf*new_sf_3l
  electrons['sf_hi_3l']  = (reco_sf + reco_err) * (new_sf_3l + new_err_3l)
  electrons['sf_lo_3l']  = (reco_sf - reco_err) * (new_sf_3l - new_err_3l)


###### Btag scale factors
################################################################
# Hard-coded to DeepJet algorithm, medium WP

# MC efficiencies
def GetMCeffFunc(year, WP='medium', flav='b'):
  if year not in ['2016','2016APV','2017','2018']: raise Exception(f"Error: Unknown year \"{year}\".")

  pathToBtagMCeff = topcoffea_path('data/btagSF/UL/btagMCeff_%s.pkl.gz'%year)
  hists = {}
  with gzip.open(pathToBtagMCeff) as fin:
    hin = pickle.load(fin)
    for k in hin.keys():
      if k in hists: hists[k]+=hin[k]
      else:          hists[k]=hin[k]
  h = hists['jetptetaflav']
  hnum = h.integrate('WP', WP)
  hden = h.integrate('WP', 'all')
  getnum = lookup_tools.dense_lookup.dense_lookup(hnum.values(overflow='over')[()], [hnum.axis('pt').edges(), hnum.axis('abseta').edges(), hnum.axis('flav').edges()])
  getden = lookup_tools.dense_lookup.dense_lookup(hden.values(overflow='over')[()], [hden.axis('pt').edges(), hnum.axis('abseta').edges(), hden.axis('flav').edges()])
  values = hnum.values(overflow='over')[()]
  edges = [hnum.axis('pt').edges(), hnum.axis('abseta').edges(), hnum.axis('flav').edges()]
  fun = lambda pt, abseta, flav : getnum(pt,abseta,flav)/getden(pt,abseta,flav)
  return fun

def GetBtagEff(pt, eta, flavor, year):
  if year not in ['2016','2016APV','2017','2018']: raise Exception(f"Error: Unknown year \"{year}\".")
  return GetMCeffFunc(year,'medium')(pt, eta, flavor)

def GetBTagSF(eta, pt, flavor, year, sys='central'):
  if   year == '2016': SFevaluatorBtag = BTagScaleFactor(topcoffea_path("data/btagSF/UL/DeepJet_106XUL16postVFPSF_v2.csv"),"MEDIUM") 
  elif   year == '2016APV': SFevaluatorBtag = BTagScaleFactor(topcoffea_path("data/btagSF/UL/wp_deepJet_106XUL16preVFP_v2.csv"),"MEDIUM") 
  elif year == '2017': SFevaluatorBtag = BTagScaleFactor(topcoffea_path("data/btagSF/UL/wp_deepJet_106XUL17_v3.csv"),"MEDIUM")
  elif year == '2018': SFevaluatorBtag = BTagScaleFactor(topcoffea_path("data/btagSF/UL/wp_deepJet_106XUL18_v2.csv"),"MEDIUM")
  else: raise Exception(f"Error: Unknown year \"{year}\".")
  SF=SFevaluatorBtag.eval(sys,flavor,eta,pt)
  return (SF)

###### Pileup reweighing
##############################################
## Get central PU data and MC profiles and calculate reweighting
## Using the current UL recommendations in:
##   https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData
##   - 2018: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/
##   - 2017: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/
##   - 2016: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/UltraLegacy/
##
## MC histograms from:
##    https://github.com/CMS-LUMI-POG/PileupTools/

pudirpath = topcoffea_path('data/pileup/')

def GetDataPUname(year, var=0):
  ''' Returns the name of the file to read pu observed distribution '''
  if year == '2016APV': year = '2016'
  if   var== 'nominal': ppxsec = get_param("pu_w")
  elif var== 'up': ppxsec = get_param("pu_w_up")
  elif var== 'down': ppxsec = get_param("pu_w_down")
  year = str(year)
  return 'PileupHistogram-goldenJSON-13tev-%s-%sub-99bins.root'%((year), str(ppxsec))

MCPUfile = {'2016APV':'pileup_2016BF.root', '2016':'pileup_2016GH.root', '2017':'pileup_2017_shifts.root', '2018':'pileup_2018_shifts.root'}
def GetMCPUname(year):
  ''' Returns the name of the file to read pu MC profile '''
  return MCPUfile[str(year)]

PUfunc = {}
### Load histograms and get lookup tables (extractors are not working here...)
for year in ['2016', '2016APV', '2017', '2018']:
  PUfunc[year] = {}
  with uproot.open(pudirpath+GetMCPUname(year)) as fMC:
    hMC = fMC['pileup']
    PUfunc[year]['MC'] = lookup_tools.dense_lookup.dense_lookup(hMC .values(), hMC.axis(0).edges())
  with uproot.open(pudirpath+GetDataPUname(year,  'nominal')) as fData:
    hD   = fData  ['pileup']
    PUfunc[year]['Data'  ] = lookup_tools.dense_lookup.dense_lookup(hD  .values(), hD.axis(0).edges())
  with uproot.open(pudirpath+GetDataPUname(year,  'up')) as fDataUp:
    hDUp = fDataUp['pileup']
    PUfunc[year]['DataUp'] = lookup_tools.dense_lookup.dense_lookup(hDUp.values(), hD.axis(0).edges())
  with uproot.open(pudirpath+GetDataPUname(year, 'down')) as fDataDo:
    hDDo = fDataDo['pileup']
    PUfunc[year]['DataDo'] = lookup_tools.dense_lookup.dense_lookup(hDDo.values(), hD.axis(0).edges())

def GetPUSF(nTrueInt, year, var='nominal'):
  year = str(year)
  if year not in ['2016','2016APV','2017','2018']: raise Exception(f"Error: Unknown year \"{year}\".")
  nMC  =PUfunc[year]['MC'](nTrueInt+1)
  nData=PUfunc[year]['DataUp' if var == 'up' else ('DataDo' if var == 'down' else 'Data')](nTrueInt)
  weights = np.divide(nData,nMC)
  return weights

def AttachPSWeights(events):
  '''
  Return a list of PS weights
  PS weights (w_var / w_nominal); [0] is ISR=0.5 FSR=1; [1] is ISR=1 FSR=0.5; [2] is ISR=2 FSR=1; [3] is ISR=1 FSR=2
  '''
  ISR = 0; FSR = 1; ISRdown = 0; FSRdown = 1; ISRup = 2; FSRup = 3
  if events.PSWeight is None:
      raise Exception(f'PSWeight not found in {fname}!')
  # Add up variation event weights
  events['ISRUp']   = events.PSWeight[:, ISRup]
  events['FSRUp']   = events.PSWeight[:, FSRup]
  # Add down variation event weights
  events['ISRDown'] = events.PSWeight[:, ISRdown]
  events['FSRDown'] = events.PSWeight[:, FSRdown]

def AttachScaleWeights(events):
  '''
  Return a list of scale weights

  Case 1:
  LHE scale variation weights (w_var / w_nominal); [0] is renscfact=0.5d0 facscfact=0.5d0 ; [1] is renscfact=0.5d0 facscfact=1d0 ; [2] is renscfact=0.5d0 facscfact=2d0 ; [3] is renscfact=1d0 facscfact=0.5d0 ; [4] is renscfact=1d0 facscfact=1d0 ; [5] is renscfact=1d0 facscfact=2d0 ; [6] is renscfact=2d0 facscfact=0.5d0 ; [7] is renscfact=2d0 facscfact=1d0 ; [8] is renscfact=2d0 facscfact=2d0

  Case 2:
  [0] is MUF="0.5" MUR="0.5"; [1] is MUF="1.0" MUR="0.5"; [2] is MUF="2.0" MUR="0.5"; [3] is MUF="0.5" MUR="1.0"; [4] is MUF="2.0" MUR="1.0"; [5] is MUF="0.5" MUR="2.0"; [6] is MUF="1.0" MUR="2.0"; [7] is MUF="2.0" MUR="2.0"
  '''

  # Determine if we are in case 1 or case 2 by checking if we have 8 or 9 weights
  len_of_wgts = ak.count(events.LHEScaleWeight,axis=-1)
  all_len_9_or_0_bool = ak.all((len_of_wgts==9) | (len_of_wgts==0))
  all_len_8_or_0_bool = ak.all((len_of_wgts==8) | (len_of_wgts==0))
  if all_len_9_or_0_bool:
      scale_weights = ak.fill_none(ak.pad_none(events.LHEScaleWeight, 9), 1) # FIXME this is a bandaid until we understand _why_ some are empty 
      renormDown_factDown = 0
      renormDown          = 1
      renormDown_factUp   = 2
      factDown            = 3
      nominal             = 4
      factUp              = 5
      renormUp_factDown   = 6
      renormUp            = 7
      renormUp_factUp     = 8
  elif all_len_8_or_0_bool:
      scale_weights = ak.fill_none(ak.pad_none(events.LHEScaleWeight, 8), 1) # FIXME this is a bandaid until we understand _why_ some are empty 
      renormDown_factDown = 0
      renormDown          = 1
      renormDown_factUp   = 2
      factDown            = 3
      factUp              = 4
      renormUp_factDown   = 5
      renormUp            = 6
      renormUp_factUp     = 7
  else:
    raise Exception(f"Unknown weight type")

  # Get the weights from the event
  events['renormfactDown'] = scale_weights[:,renormDown_factDown]
  events['renormDown']     = scale_weights[:,renormDown]
  events['factDown']       = scale_weights[:,factDown]
  events['nom']            = ak.ones_like(scale_weights[:,0]) # Nominal not always included in scale_weights
  events['factUp']         = scale_weights[:,factUp]
  events['renormUp']       = scale_weights[:,renormUp]
  events['renormfactUp']   = scale_weights[:,renormUp_factUp]

  # We expect this to be 1, and use it as the "nominal" for ISR/FSR as well, so want to make sure it is never anything other than 1
  # We had been using the nominal from scale_weights, so we were checking to make sure it was just 1 (as we expected it to be)
  # But it seems nominal is not always included, so now we're creating our own array defined to be 1
  # But might as well leave the check in, just in case something weird goes wrong...
  if ak.any(events['nom'] != 1.0):
    raise Exception("ERROR: A LHEScaleWeight nominal value is not 1. Is this expected?")


def AttachPdfWeights(events):
  '''
  Return a list of PDF weights
  Should be 100 weights for NNPDF 3.1
  '''
  if events.LHEPdfWeight is None:
      raise Exception(f'LHEPdfWeight not found in {fname}!')
  pdf_weight = ak.Array(events.LHEPdfWeight)
  #events['Pdf'] = ak.Array(events.nLHEPdfWeight) # FIXME not working

####### JEC 
##############################################
# JER: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetResolution
# JES: https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC

def ApplyJetCorrections(year, corr_type):
  if year=='2016': jec_tag='16_V7'; jer_tag='Summer20UL16_JRV3'
  elif year=='2016APV': jec_tag='16APV_V7'; jer_tag='Summer20UL16APV_JRV3'
  elif year=='2017': jec_tag='17_V5'; jer_tag='Summer19UL17_JRV2'
  elif year=='2018': jec_tag='18_V5'; jer_tag='Summer19UL18_JRV2'
  else: raise Exception(f"Error: Unknown year \"{year}\".")
  extJEC = lookup_tools.extractor()
  extJEC.add_weight_sets(["* * "+topcoffea_path('data/JER/%s_MC_SF_AK4PFchs.jersf.txt'%jer_tag),"* * "+topcoffea_path('data/JER/%s_MC_PtResolution_AK4PFchs.jr.txt'%jer_tag),"* * "+topcoffea_path('data/JEC/Summer19UL%s_MC_L1FastJet_AK4PFchs.txt'%jec_tag),"* * "+topcoffea_path('data/JEC/Summer19UL%s_MC_L2Relative_AK4PFchs.txt'%jec_tag),"* * "+topcoffea_path('data/JEC/Summer19UL%s_MC_Uncertainty_AK4PFchs.junc.txt'%jec_tag)])
  jec_names = ["%s_MC_SF_AK4PFchs"%jer_tag,"%s_MC_PtResolution_AK4PFchs"%jer_tag,"Summer19UL%s_MC_L1FastJet_AK4PFchs"%jec_tag,"Summer19UL%s_MC_L2Relative_AK4PFchs"%jec_tag,"Summer19UL%s_MC_Uncertainty_AK4PFchs"%jec_tag]
  extJEC.finalize()
  JECevaluator = extJEC.make_evaluator()
  jec_inputs = {name: JECevaluator[name] for name in jec_names}
  jec_stack = JECStack(jec_inputs)
  name_map = jec_stack.blank_name_map
  name_map['JetPt'] = 'pt'
  name_map['JetMass'] = 'mass'
  name_map['JetEta'] = 'eta'
  name_map['JetPhi'] = 'phi'
  name_map['JetA'] = 'area'
  name_map['ptGenJet'] = 'pt_gen'
  name_map['ptRaw'] = 'pt_raw'
  name_map['massRaw'] = 'mass_raw'
  name_map['Rho'] = 'rho'
  name_map['METpt'] = 'pt'
  name_map['METphi'] = 'phi'
  name_map['UnClusteredEnergyDeltaX'] = 'MetUnclustEnUpDeltaX'
  name_map['UnClusteredEnergyDeltaY'] = 'MetUnclustEnUpDeltaY'
  if corr_type=='met': return CorrectedMETFactory(name_map)
  return CorrectedJetsFactory(name_map, jec_stack)

def ApplyJetSystematics(cleanedJets,syst_var):
  if(syst_var == 'JERUp'): return cleanedJets.JER.up
  elif(syst_var == 'JERDown'): return cleanedJets.JER.down
  elif(syst_var == 'JESUp'): return cleanedJets.JES_jes.up
  elif(syst_var == 'JESDown'): return cleanedJets.JES_jes.down
  elif(syst_var == 'nominal'): return cleanedJets
  elif(syst_var in ['nominal','MuonESUp','MuonESDown']): return cleanedJets
  else: raise Exception(f"Error: Unknown variation \"{syst_var}\".")

###### Muon Rochester corrections
################################################################
# https://gitlab.cern.ch/akhukhun/roccor
# https://github.com/CoffeaTeam/coffea/blob/master/coffea/lookup_tools/rochester_lookup.py

def ApplyRochesterCorrections(year, mu, is_data, var='nominal'):
    if year=='2016': rochester_data = txt_converters.convert_rochester_file(topcoffea_path("data/MuonScale/RoccoR2016bUL.txt"), loaduncs=True)
    elif year=='2016APV': rochester_data = txt_converters.convert_rochester_file(topcoffea_path("data/MuonScale/RoccoR2016aUL.txt"), loaduncs=True)
    elif year=='2017': rochester_data = txt_converters.convert_rochester_file(topcoffea_path("data/MuonScale/RoccoR2017UL.txt"), loaduncs=True)
    elif year=='2018': rochester_data = txt_converters.convert_rochester_file(topcoffea_path("data/MuonScale/RoccoR2018UL.txt"), loaduncs=True)
    rochester = rochester_lookup.rochester_lookup(rochester_data)
    if not is_data:
        hasgen = ~np.isnan(ak.fill_none(mu.matched_gen.pt, np.nan))
        mc_rand = np.random.rand(*ak.to_numpy(ak.flatten(mu.pt)).shape)
        mc_rand = ak.unflatten(mc_rand, ak.num(mu.pt, axis=1))
        corrections = np.array(ak.flatten(ak.ones_like(mu.pt)))
        errors = np.array(ak.flatten(ak.ones_like(mu.pt)))
        
        mc_kspread = rochester.kSpreadMC(mu.charge[hasgen],mu.pt[hasgen],mu.eta[hasgen],mu.phi[hasgen],mu.matched_gen.pt[hasgen])
        mc_ksmear = rochester.kSmearMC(mu.charge[~hasgen],mu.pt[~hasgen],mu.eta[~hasgen],mu.phi[~hasgen],mu.nTrackerLayers[~hasgen],mc_rand[~hasgen])
        errspread = rochester.kSpreadMCerror(mu.charge[hasgen],mu.pt[hasgen],mu.eta[hasgen],mu.phi[hasgen],mu.matched_gen.pt[hasgen])
        errsmear = rochester.kSmearMCerror(mu.charge[~hasgen],mu.pt[~hasgen],mu.eta[~hasgen],mu.phi[~hasgen],mu.nTrackerLayers[~hasgen],mc_rand[~hasgen])
        hasgen_flat = np.array(ak.flatten(hasgen))
        corrections[hasgen_flat] = np.array(ak.flatten(mc_kspread))
        corrections[~hasgen_flat] = np.array(ak.flatten(mc_ksmear))
        errors[hasgen_flat] = np.array(ak.flatten(errspread))
        errors[~hasgen_flat] = np.array(ak.flatten(errsmear))
        corrections = ak.unflatten(corrections, ak.num(mu.pt, axis=1))
        errors = ak.unflatten(errors, ak.num(mu.pt, axis=1))
    else:
        corrections = rochester.kScaleDT(mu.charge, mu.pt, mu.eta, mu.phi)
        errors = rochester.kScaleDTerror(mu.charge, mu.pt, mu.eta, mu.phi)
    if var=='nominal': return(mu.pt * corrections) #nominal
    elif var=='up': return(mu.pt * corrections + mu.pt * errors) #up 
    elif var=='down': return(mu.pt * corrections - mu.pt * errors) #down

###### Trigger SFs
################################################################

#### Functions needed
StackOverUnderflow = lambda v : [sum(v[0:2])] + v[2:-2] + [sum(v[-2:])]

def GetClopperPearsonInterval(hnum, hden):
  ''' Compute Clopper-Pearson interval from numerator and denominator histograms '''
  num = list(hnum.values(overflow='all')[()])
  den = list(hden.values(overflow='all')[()])
  if isinstance(num, list) and isinstance(num[0], np.ndarray):
    for i in range(len(num)):
      num[i] = np.array(StackOverUnderflow(list(num[i])), dtype=float)
      den[i] = np.array(StackOverUnderflow(list(den[i])), dtype=float)
    den = StackOverUnderflow(den)
    num = StackOverUnderflow(num)
  else: 
    num = np.array(StackOverUnderflow(num), dtype=float); den = np.array(StackOverUnderflow(den), dtype=float)
  num = np.array(num); den = np.array(den)
  num[num>den]=den[num>den]
  down, up = hist.clopper_pearson_interval(num, den)
  ratio = np.array(num, dtype=float)/den
  return [ratio, down, up]
  
def GetEff(num, den):
  ''' Compute efficiency values from numerator and denominator histograms '''
  ratio, down, up =  GetClopperPearsonInterval(num, den)
  axis = num.axes()[0].name
  bins = num.axis(axis).edges()
  x    = num.axis(axis).centers()
  xlo  = bins[:-1]
  xhi  = bins[1:]
  return [[x, xlo-x, xhi-x],[ratio, down-ratio, up-ratio]]

def GetSFfromCountsHisto(hnumMC, hdenMC, hnumData, hdenData):
  ''' Compute scale factors from efficiency histograms for data and MC '''
  Xmc, Ymc = GetEff(hnumMC, hdenMC)
  Xda, Yda = GetEff(hnumData, hdenData)
  ratio, do, up = GetRatioAssymetricUncertainties(Yda[0], Yda[1], Yda[2], Ymc[0], Ymc[1], Ymc[2])
  return ratio, do, up
    
def GetRatioAssymetricUncertainties(num, numDo, numUp, den, denDo, denUp):
  ''' Compute efficiencies from numerator and denominator counts histograms and uncertainties '''
  ratio = num/den
  uncUp = ratio*np.sqrt(numUp*numUp + denUp*denUp) 
  uncDo = ratio*np.sqrt(numDo*numDo + denDo*denDo) 
  return ratio, -uncDo, uncUp

######  Scale Factors

def LoadTriggerSF(year, ch='2l', flav='em'):
  pathToTriggerSF = topcoffea_path('data/triggerSF/triggerSF_%s.pkl.gz'%year)
  with gzip.open(pathToTriggerSF) as fin: hin = pickle.load(fin)
  if ch=='2l': axisY='l1pt'
  else: axisY='l0eta'
  h = hin[ch][flav]
  ratio, do, up = GetSFfromCountsHisto(h['hmn'], h['hmd'], h['hdn'], h['hdd'])
  ratio[np.isnan(ratio)]=1.0; do[np.isnan(do)]=0.0;up[np.isnan(up)]=0.0
  GetTrig   = lookup_tools.dense_lookup.dense_lookup(ratio, [h['hmn'].axis('l0pt').edges(), h['hmn'].axis(axisY).edges()])
  GetTrigUp = lookup_tools.dense_lookup.dense_lookup(up   , [h['hmn'].axis('l0pt').edges(), h['hmn'].axis(axisY).edges()])
  GetTrigDo = lookup_tools.dense_lookup.dense_lookup(do   , [h['hmn'].axis('l0pt').edges(), h['hmn'].axis(axisY).edges()])
  return [GetTrig, GetTrigDo, GetTrigUp]

def GetTriggerSF(year, events, lep0, lep1):
  ls=[]
  for syst in [0,1,2]:
    #2l
    SF_ee=np.where(events.is_ee==True,LoadTriggerSF(year,ch='2l',flav='ee')[syst](lep0.pt,lep1.pt),1.0)
    SF_em=np.where(events.is_em==True, LoadTriggerSF(year,ch='2l',flav='em')[syst](lep0.pt,lep1.pt),1.0)
    SF_mm=np.where(events.is_mm==True, LoadTriggerSF(year,ch='2l',flav='mm')[syst](lep0.pt,lep1.pt),1.0)
    #3l
    SF_eee=np.where(events.is_eee==True,LoadTriggerSF(year,ch='3l',flav='eee')[syst](lep0.pt,lep0.eta),1.0)
    SF_eem=np.where(events.is_eee==True,LoadTriggerSF(year,ch='3l',flav='eem')[syst](lep0.pt,lep0.eta),1.0)
    SF_emm=np.where(events.is_eee==True,LoadTriggerSF(year,ch='3l',flav='emm')[syst](lep0.pt,lep0.eta),1.0)
    SF_mmm=np.where(events.is_eee==True,LoadTriggerSF(year,ch='3l',flav='mmm')[syst](lep0.pt,lep0.eta),1.0)
    ls.append(SF_ee*SF_em*SF_mm*SF_eee*SF_eem*SF_emm*SF_mmm)
  ls[1]=np.where(ls[1]==1.0,0.0,ls[1]) # syst down
  ls[2]=np.where(ls[2]==1.0,0.0,ls[2]) # syst up
  events['trigger_sf']=ls[0] #nominal
  events['trigger_sfDown']=ls[0]+ls[1]
  events['trigger_sfUp']=ls[0]+ls[2]
