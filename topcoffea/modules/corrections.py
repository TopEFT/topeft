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

ext = lookup_tools.extractor()
basepathFromTTH = 'data/fromTTH/lepSF/'

# Electron reco
ext.add_weight_sets(["ElecRecoSFb20_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'reco/elec/2016/el_scaleFactors_gsf_ptLt20.root')])
ext.add_weight_sets(["ElecRecoSFg20_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'reco/elec/2016/el_scaleFactors_gsf_ptGt20.root')])
ext.add_weight_sets(["ElecRecoSFb20_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'reco/elec/2017/el_scaleFactors_gsf_ptLt20.root')])
ext.add_weight_sets(["ElecRecoSFg20_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'reco/elec/2017/el_scaleFactors_gsf_ptGt20.root')])
ext.add_weight_sets(["ElecRecoSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'reco/elec/2018/el_scaleFactors_gsf.root')])

# Electron loose
ext.add_weight_sets(["ElecLooseSF_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loose_ele_2016.root')])
ext.add_weight_sets(["ElecLooseSF_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loose_ele_2017.root')])
ext.add_weight_sets(["ElecLooseSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loose_ele_2018.root')])
ext.add_weight_sets(["ElecLoosettHSF_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loosettH_ele_2016.root')])
ext.add_weight_sets(["ElecLoosettHSF_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loosettH_ele_2017.root')])
ext.add_weight_sets(["ElecLoosettHSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loosettH_ele_2018.root')])

# Electron tight
ext.add_weight_sets(["ElecTightSF_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'tight/elec/egammaEff2016_EGM2D.root')])
ext.add_weight_sets(["ElecTightSF_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'tight/elec/egammaEff2016_EGM2D.root')])
ext.add_weight_sets(["ElecTightSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'tight/elec/egammaEff2016_EGM2D.root')])

# Muon loose
ext.add_weight_sets(["MuonLooseSF_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/muon/TnP_loose_muon_2016.root')])
ext.add_weight_sets(["MuonLooseSF_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/muon/TnP_loose_muon_2017.root')])
ext.add_weight_sets(["MuonLooseSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/muon/TnP_loose_muon_2018.root')])

# Muon tight
ext.add_weight_sets(["MuonTightSF_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'tight/muon/egammaEff2016_EGM2D.root')])
ext.add_weight_sets(["MuonTightSF_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'tight/muon/egammaEff2017_EGM2D.root')])
ext.add_weight_sets(["MuonTightSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'tight/muon/egammaEff2018_EGM2D.root')])

ext.finalize()
SFevaluator = ext.make_evaluator()

def GetLeptonSF(pt1, eta1, type1, pt2, eta2, type2, pt3=None, eta3=None, type3=None, year=2018):
  if type1 == 'm':
    SF1 = ak.prod(SFevaluator['MuonLooseSF_%i'%year](np.abs(eta1), pt1) * SFevaluator['MuonTightSF_%i'%year](np.abs(eta1), pt1), axis=-1)
  elif type1 == 'e':
    SF1 = ak.prod(SFevaluator['ElecRecoSF_%i'%year](eta1, pt1) * SFevaluator['ElecLooseSF_%i'%year](np.abs(eta1), pt1) * SFevaluator['ElecLoosettHSF_%i'%year](np.abs(eta1), pt1) * SFevaluator['ElecTightSF_%i'%year](np.abs(eta1), pt1), axis=-1)
  else: print(type1, ' is not a valid type. Valid types: "m" or "e"')
  if type2 == 'm':
    SF2 = ak.prod(SFevaluator['MuonLooseSF_%i'%year](np.abs(eta2), pt2) * SFevaluator['MuonTightSF_%i'%year](np.abs(eta2), pt2), axis=-1)
  elif type2 == 'e':
    SF2 = ak.prod(SFevaluator['ElecRecoSF_%i'%year](eta2, pt2) * SFevaluator['ElecLooseSF_%i'%year](np.abs(eta2), pt2) * SFevaluator['ElecLoosettHSF_%i'%year](np.abs(eta2), pt2) * SFevaluator['ElecTightSF_%i'%year](np.abs(eta2), pt2), axis=-1)
  else: print(type2, ' is not a valid type. Valid types: "m" or "e"')
  if type3==None:
    return( np.multiply(SF1,SF2) )
  elif type3 == 'm':
    SF3 = ak.prod(SFevaluator['MuonLooseSF_%i'%year](np.abs(eta3), pt3) * SFevaluator['MuonTightSF_%i'%year](np.abs(eta3), pt3), axis=-1)
  elif type3 == 'e':
    SF3 = ak.prod(SFevaluator['ElecRecoSF_%i'%year](eta3, pt3) * SFevaluator['ElecLooseSF_%i'%year](np.abs(eta3), pt3) * SFevaluator['ElecLoosettHSF_%i'%year](np.abs(eta3), pt3) * SFevaluator['ElecTightSF_%i'%year](np.abs(eta3), pt3), axis=-1)
  else: print(type3, ' is not a valid type. Valid types: "m" , "e" or None')
  if type3!=None:
    return( np.multiply(SF3, np.multiply(SF1,SF2)))


# test
#val = evaluator['MuonTightSF_2016'](np.array([1.2, 0.3]),np.array([24.5, 51.3]))
#print('val = ', val)
