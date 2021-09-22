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
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.btag_tools.btagscalefactor import BTagScaleFactor
from topcoffea.modules.GetValuesFromJsons import get_param

basepathFromTTH = 'data/fromTTH/'

###### Lepton scale factors
################################################################
extLepSF = lookup_tools.extractor()

# Electron reco
extLepSF.add_weight_sets(["ElecRecoSFb20_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/reco/elec/2016/el_scaleFactors_gsf_ptLt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/reco/elec/2016/el_scaleFactors_gsf_ptGt20.root')])
extLepSF.add_weight_sets(["ElecRecoSFb20_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/reco/elec/2017/el_scaleFactors_gsf_ptLt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/reco/elec/2017/el_scaleFactors_gsf_ptGt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/reco/elec/2018/el_scaleFactors_gsf.root')])
extLepSF.add_weight_sets(["ElecRecoSFb20_2016_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/reco/elec/2016/el_scaleFactors_gsf_ptLt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2016_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/reco/elec/2016/el_scaleFactors_gsf_ptGt20.root')])
extLepSF.add_weight_sets(["ElecRecoSFb20_2017_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/reco/elec/2017/el_scaleFactors_gsf_ptLt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2017_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/reco/elec/2017/el_scaleFactors_gsf_ptGt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2018_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/reco/elec/2018/el_scaleFactors_gsf.root')])

# Electron loose
extLepSF.add_weight_sets(["ElecLooseSF_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loose_ele_2016.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loose_ele_2017.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loose_ele_2018.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loosettH_ele_2016.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loosettH_ele_2017.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loosettH_ele_2018.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2016_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loose_ele_2016.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2017_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loose_ele_2017.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2018_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loose_ele_2018.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2016_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loosettH_ele_2016.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2017_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loosettH_ele_2017.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2018_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loosettH_ele_2018.root')])

# Electron tight
extLepSF.add_weight_sets(["ElecTightSF_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/tight/elec/egammaEff2016_EGM2D.root')])
extLepSF.add_weight_sets(["ElecTightSF_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/tight/elec/egammaEff2017_EGM2D.root')])
extLepSF.add_weight_sets(["ElecTightSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/tight/elec/egammaEff2018_EGM2D.root')])
extLepSF.add_weight_sets(["ElecTightSF_2016_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/tight/elec/egammaEff2016_EGM2D.root')])
extLepSF.add_weight_sets(["ElecTightSF_2017_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/tight/elec/egammaEff2017_EGM2D.root')])
extLepSF.add_weight_sets(["ElecTightSF_2018_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/tight/elec/egammaEff2018_EGM2D.root')])

# Muon loose
extLepSF.add_weight_sets(["MuonLooseSF_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/muon/TnP_loose_muon_2016.root')])
extLepSF.add_weight_sets(["MuonLooseSF_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/muon/TnP_loose_muon_2017.root')])
extLepSF.add_weight_sets(["MuonLooseSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/muon/TnP_loose_muon_2018.root')])
extLepSF.add_weight_sets(["MuonLooseSF_2016_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/muon/TnP_loose_muon_2016.root')])
extLepSF.add_weight_sets(["MuonLooseSF_2017_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/muon/TnP_loose_muon_2017.root')])
extLepSF.add_weight_sets(["MuonLooseSF_2018_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/loose/muon/TnP_loose_muon_2018.root')])

# Muon tight
extLepSF.add_weight_sets(["MuonTightSF_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/tight/muon/egammaEff2016_EGM2D.root')])
extLepSF.add_weight_sets(["MuonTightSF_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/tight/muon/egammaEff2017_EGM2D.root')])
extLepSF.add_weight_sets(["MuonTightSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'lepSF/tight/muon/egammaEff2018_EGM2D.root')])
extLepSF.add_weight_sets(["MuonTightSF_2016_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/tight/muon/egammaEff2016_EGM2D.root')])
extLepSF.add_weight_sets(["MuonTightSF_2017_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/tight/muon/egammaEff2017_EGM2D.root')])
extLepSF.add_weight_sets(["MuonTightSF_2018_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'lepSF/tight/muon/egammaEff2018_EGM2D.root')])

# Fake rate 
# todo: check that these are the same as the "recorrected"
extLepSF.add_weight_sets(["MuonFR_2016 FR_mva085_mu_data_comb_recorrected %s"%topcoffea_path(basepathFromTTH+'fakerate/fr_2016_recorrected.root')])
extLepSF.add_weight_sets(["MuonFR_2017 FR_mva085_mu_data_comb_recorrected %s"%topcoffea_path(basepathFromTTH+'fakerate/fr_2017_recorrected.root')])
extLepSF.add_weight_sets(["MuonFR_2018 FR_mva085_mu_data_comb_recorrected %s"%topcoffea_path(basepathFromTTH+'fakerate/fr_2018_recorrected.root')])
extLepSF.add_weight_sets(["ElecFR_2016 FR_mva080_el_data_comb_NC_recorrected %s"%topcoffea_path(basepathFromTTH+'fakerate/fr_2016_recorrected.root')])
extLepSF.add_weight_sets(["ElecFR_2017 FR_mva080_el_data_comb_NC_recorrected %s"%topcoffea_path(basepathFromTTH+'fakerate/fr_2017_recorrected.root')])
extLepSF.add_weight_sets(["ElecFR_2018 FR_mva080_el_data_comb_NC_recorrected %s"%topcoffea_path(basepathFromTTH+'fakerate/fr_2018_recorrected.root')])


extLepSF.finalize()
SFevaluator = extLepSF.make_evaluator()

def AttachPerLeptonFR(leps, flavor, year=2018):
  if year == '2016APV': year = 2016
  fr=SFevaluator['{flavor}FR_{year}'.format(flavor=flavor,year=year)](np.abs(leps.eta), leps.conept )
  leps['fakefactor']=ak.fill_none(-fr/(1-fr),0) # this is the factor that actually enters the expressions

def fakeRateWeight2l(events, lep1, lep2):
  
  fakefactor_2l =  (~lep1.isTightLep | ~lep2.isTightLep)*(-1) # if all are tight the FF is 0 for safety reasons
  fakefactor_2l =  fakefactor_2l*(lep1.isTightLep + (~lep1.isTightLep)*lep1.fakefactor)
  fakefactor_2l =  fakefactor_2l*(lep2.isTightLep + (~lep2.isTightLep)*lep2.fakefactor)
  events['fakefactor_2l']=fakefactor_2l

def fakeRateWeight3l(events, lep1, lep2, lep3):
  fakefactor_3l = (~lep1.isTightLep | ~lep2.isTightLep | ~lep2.isTightLep)*(-1) # if all are tight the FF is 0 for safety reasons
  fakefactor_3l = fakefactor_3l*(lep1.isTightLep + (~lep1.isTightLep)*lep1.fakefactor)
  fakefactor_3l = fakefactor_3l*(lep2.isTightLep + (~lep2.isTightLep)*lep2.fakefactor)
  fakefactor_3l = fakefactor_3l*(lep3.isTightLep + (~lep3.isTightLep)*lep3.fakefactor)
  events['fakefactor_3l']=fakefactor_3l


def AttachMuonSF(muons, year=2018):
  '''
    Description:
      Inserts 'sf_nom', 'sf_hi', and 'sf_lo' into the muons array passed to this function. These
      values correspond to the nominal, up, and down muon scalefactor values respectively.
  '''
  eta = np.abs(muons.eta)
  pt = muons.pt
  if year == '2016APV': year = '2016'
  loose_sf  = SFevaluator['MuonLooseSF_{year}'.format(year=year)](eta,pt)
  loose_err = SFevaluator['MuonLooseSF_{year}_er'.format(year=year)](eta,pt)

  tight_sf  = SFevaluator['MuonTightSF_{year}'.format(year=year)](eta,pt)
  tight_err = SFevaluator['MuonTightSF_{year}_er'.format(year=year)](eta,pt)

  muons['sf_nom'] = loose_sf * tight_sf
  muons['sf_hi']  = (loose_sf + loose_err) * (tight_sf + tight_err)
  muons['sf_lo']  = (loose_sf - loose_err) * (tight_sf - tight_err)

def AttachElectronSF(electrons, year=2018):
  '''
    Description:
      Inserts 'sf_nom', 'sf_hi', and 'sf_lo' into the electrons array passed to this function. These
      values correspond to the nominal, up, and down electron scalefactor values respectively.
  '''
  # eta = np.abs(electrons.eta)
  eta = electrons.eta
  pt = electrons.pt
  if year == '2016APV': year = 2016
  # For the ElecRecoSF we dont take the absolute value of eta!
  reco_sf          = SFevaluator['ElecRecoSF_{year}'.format(year=year)](eta,pt)
  reco_sf_err      = SFevaluator['ElecRecoSF_{year}_er'.format(year=year)](eta,pt)

  loose_sf         = SFevaluator['ElecLooseSF_{year}'.format(year=year)](np.abs(eta),pt)
  loose_sf_err     = SFevaluator['ElecLooseSF_{year}_er'.format(year=year)](np.abs(eta),pt)

  loose_ttH_sf     = SFevaluator['ElecLoosettHSF_{year}'.format(year=year)](np.abs(eta),pt)
  loose_ttH_sf_err = SFevaluator['ElecLoosettHSF_{year}_er'.format(year=year)](np.abs(eta),pt)

  tight_sf         = SFevaluator['ElecTightSF_{year}'.format(year=year)](np.abs(eta),pt)
  tight_sf_err     = SFevaluator['ElecTightSF_{year}_er'.format(year=year)](np.abs(eta),pt)

  electrons['sf_nom'] = reco_sf * loose_sf * loose_ttH_sf * tight_sf
  electrons['sf_hi']  = (reco_sf + reco_sf_err) * (loose_sf + loose_sf_err) * (loose_ttH_sf + loose_ttH_sf_err) * (tight_sf + tight_sf_err)
  electrons['sf_lo']  = (reco_sf - reco_sf_err) * (loose_sf - loose_sf_err) * (loose_ttH_sf - loose_ttH_sf_err) * (tight_sf - tight_sf_err)


###### Btag scale factors
################################################################
# Hard-coded to DeepJet algorithm, medium WP

# MC efficiencies
def GetMCeffFunc(WP='medium', flav='b', year=2018):
  pathToBtagMCeff = topcoffea_path('data/btagSF/UL/btagMCeff_%i.pkl.gz'%year)
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

MCeffFunc_2018 = GetMCeffFunc('medium', 2018)
MCeffFunc_2017 = GetMCeffFunc('medium', 2017)

def GetBtagEff(eta, pt, flavor, year=2018):
  if year==2017: return MCeffFunc_2017(pt, eta, flavor)
  else         : return MCeffFunc_2018(pt, eta, flavor)

def GetBTagSF(eta, pt, flavor, year=2018, sys=0):

  # Efficiencies and SFs for UL only available for 2017 and 2018
  if year == '2016APV': year = 2016
  if   year == 2016: SFevaluatorBtag = BTagScaleFactor(topcoffea_path("data/btagSF/DeepFlav_2016.csv"),"MEDIUM")
  elif year == 2017: SFevaluatorBtag = BTagScaleFactor(topcoffea_path("data/btagSF/UL/DeepJet_UL17.csv"),"MEDIUM")
  elif year == 2018: SFevaluatorBtag = BTagScaleFactor(topcoffea_path("data/btagSF/UL/DeepJet_UL18.csv"),"MEDIUM")

  if   sys==0 : SF=SFevaluatorBtag.eval("central",flavor,eta,pt)
  elif sys==1 : SF=SFevaluatorBtag.eval("up",flavor,eta,pt)
  elif sys==-1: SF=SFevaluatorBtag.eval("down",flavor,eta,pt)

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

def GetDataPUname(year='2017', var=0):
  ''' Returns the name of the file to read pu observed distribution '''
  if year == '2016APV': year = 2016
  if   var== 0: ppxsec = get_param("pu_w")
  elif var== 1: ppxsec = get_param("pu_w_up")
  elif var==-1: ppxsec = get_param("pu_w_down")
  year = str(year)
  return 'PileupHistogram-goldenJSON-13tev-%s-%sub-99bins.root'%((year), str(ppxsec))

MCPUfile = {'2016APV':'pileup_2016BF.root', '2016':'pileup_2016GH.root', '2017':'pileup_2017_shifts.root', '2018':'pileup_2018_shifts.root'}
def GetMCPUname(year='2017'):
  ''' Returns the name of the file to read pu MC profile '''
  return MCPUfile[str(year)]

PUfunc = {}
### Load histograms and get lookup tables (extractors are not working here...)
for year in ['2016', '2016APV', '2017', '2018']:
  PUfunc[year] = {}
  with uproot.open(pudirpath+GetMCPUname(year)) as fMC:
    hMC = fMC['pileup']
    PUfunc[year]['MC'] = lookup_tools.dense_lookup.dense_lookup(hMC .values(), hMC.axis(0).edges())
  with uproot.open(pudirpath+GetDataPUname(year,  0)) as fData:
    hD   = fData  ['pileup']
    PUfunc[year]['Data'  ] = lookup_tools.dense_lookup.dense_lookup(hD  .values(), hD.axis(0).edges())
  with uproot.open(pudirpath+GetDataPUname(year,  1)) as fDataUp:
    hDUp = fDataUp['pileup']
    PUfunc[year]['DataUp'] = lookup_tools.dense_lookup.dense_lookup(hDUp.values(), hD.axis(0).edges())
  with uproot.open(pudirpath+GetDataPUname(year, -1)) as fDataDo:
    hDDo = fDataDo['pileup']
    PUfunc[year]['DataDo'] = lookup_tools.dense_lookup.dense_lookup(hDDo.values(), hD.axis(0).edges())

def GetPUSF(nTrueInt, year, var=0):
  year = str(year)
  nMC  =PUfunc[year]['MC'](nTrueInt+1)
  nData=PUfunc[year]['DataUp' if var == 1 else ('DataDo' if var == -1 else 'Data')](nTrueInt)
  weights = np.divide(nData,nMC)
  return weights

###### JEC corrections (2018)
##############################################
extJEC = lookup_tools.extractor()
extJEC.add_weight_sets(["* * "+topcoffea_path('data/JEC/Summer19UL18_V5_MC_L2Relative_AK4PFchs.txt'),"* * "+topcoffea_path('data/JEC/Summer19UL18_V5_MC_L2Residual_AK4PFchs.txt'),"* * "+topcoffea_path('data/JEC/Summer19UL18_V5_MC_L1FastJet_AK4PFchs.txt'),"* * "+topcoffea_path('data/JEC/Summer19UL18_V5_MC_L3Absolute_AK4PFchs.txt'),"* * "+topcoffea_path('data/JEC/Summer19UL18_V5_MC_L1RC_AK4PFchs.txt'),"* * "+topcoffea_path('data/JEC/Summer19UL18_V5_MC_Uncertainty_AK4PFchs.junc.txt'),"* * "+topcoffea_path('data/JEC/Summer19UL18_V5_MC_L2L3Residual_AK4PFchs.txt')])
extJEC.finalize()
JECevaluator = extJEC.make_evaluator()
jec_names = ["Summer19UL18_V5_MC_L2Relative_AK4PFchs","Summer19UL18_V5_MC_L2Residual_AK4PFchs","Summer19UL18_V5_MC_L1FastJet_AK4PFchs","Summer19UL18_V5_MC_L3Absolute_AK4PFchs","Summer19UL18_V5_MC_L1RC_AK4PFchs","Summer19UL18_V5_MC_Uncertainty_AK4PFchs","Summer19UL18_V5_MC_L2L3Residual_AK4PFchs"] 
jec_inputs = {name: JECevaluator[name] for name in jec_names}
jec_stack = JECStack(jec_inputs)
name_map = jec_stack.blank_name_map
name_map['JetPt'] = 'pt'
name_map['JetMass'] = 'mass'
name_map['JetEta'] = 'eta'
name_map['JetA'] = 'area'
name_map['ptGenJet'] = 'pt_gen'
name_map['ptRaw'] = 'pt_raw'
name_map['massRaw'] = 'mass_raw'
name_map['Rho'] = 'rho'
jet_factory = CorrectedJetsFactory(name_map, jec_stack)
# test
#val = evaluator['MuonTightSF_2016'](np.array([1.2, 0.3]),np.array([24.5, 51.3]))
#print('val = ', val)
