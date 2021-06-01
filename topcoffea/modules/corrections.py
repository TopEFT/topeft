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

basepathFromTTH = 'data/fromTTH/lepSF/'

###### Lepton scale factors
################################################################
extLepSF = lookup_tools.extractor()

# Electron reco
extLepSF.add_weight_sets(["ElecRecoSFb20_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'reco/elec/2016/el_scaleFactors_gsf_ptLt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'reco/elec/2016/el_scaleFactors_gsf_ptGt20.root')])
extLepSF.add_weight_sets(["ElecRecoSFb20_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'reco/elec/2017/el_scaleFactors_gsf_ptLt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'reco/elec/2017/el_scaleFactors_gsf_ptGt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'reco/elec/2018/el_scaleFactors_gsf.root')])
extLepSF.add_weight_sets(["ElecRecoSFb20_2016_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'reco/elec/2016/el_scaleFactors_gsf_ptLt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2016_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'reco/elec/2016/el_scaleFactors_gsf_ptGt20.root')])
extLepSF.add_weight_sets(["ElecRecoSFb20_2017_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'reco/elec/2017/el_scaleFactors_gsf_ptLt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2017_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'reco/elec/2017/el_scaleFactors_gsf_ptGt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2018_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'reco/elec/2018/el_scaleFactors_gsf.root')])

# Electron loose
extLepSF.add_weight_sets(["ElecLooseSF_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loose_ele_2016.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loose_ele_2017.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loose_ele_2018.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loosettH_ele_2016.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loosettH_ele_2017.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loosettH_ele_2018.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2016_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loose_ele_2016.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2017_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loose_ele_2017.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2018_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loose_ele_2018.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2016_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loosettH_ele_2016.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2017_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loosettH_ele_2017.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2018_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'loose/elec/TnP_loosettH_ele_2018.root')])

# Electron tight
extLepSF.add_weight_sets(["ElecTightSF_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'tight/elec/egammaEff2016_EGM2D.root')])
extLepSF.add_weight_sets(["ElecTightSF_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'tight/elec/egammaEff2016_EGM2D.root')])
extLepSF.add_weight_sets(["ElecTightSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'tight/elec/egammaEff2016_EGM2D.root')])
extLepSF.add_weight_sets(["ElecTightSF_2016_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'tight/elec/egammaEff2016_EGM2D.root')])
extLepSF.add_weight_sets(["ElecTightSF_2017_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'tight/elec/egammaEff2016_EGM2D.root')])
extLepSF.add_weight_sets(["ElecTightSF_2018_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'tight/elec/egammaEff2016_EGM2D.root')])

# Muon loose
extLepSF.add_weight_sets(["MuonLooseSF_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/muon/TnP_loose_muon_2016.root')])
extLepSF.add_weight_sets(["MuonLooseSF_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/muon/TnP_loose_muon_2017.root')])
extLepSF.add_weight_sets(["MuonLooseSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'loose/muon/TnP_loose_muon_2018.root')])
extLepSF.add_weight_sets(["MuonLooseSF_2016_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'loose/muon/TnP_loose_muon_2016.root')])
extLepSF.add_weight_sets(["MuonLooseSF_2017_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'loose/muon/TnP_loose_muon_2017.root')])
extLepSF.add_weight_sets(["MuonLooseSF_2018_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'loose/muon/TnP_loose_muon_2018.root')])

# Muon tight
extLepSF.add_weight_sets(["MuonTightSF_2016 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'tight/muon/egammaEff2016_EGM2D.root')])
extLepSF.add_weight_sets(["MuonTightSF_2017 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'tight/muon/egammaEff2017_EGM2D.root')])
extLepSF.add_weight_sets(["MuonTightSF_2018 EGamma_SF2D %s"%topcoffea_path(basepathFromTTH+'tight/muon/egammaEff2018_EGM2D.root')])
extLepSF.add_weight_sets(["MuonTightSF_2016_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'tight/muon/egammaEff2016_EGM2D.root')])
extLepSF.add_weight_sets(["MuonTightSF_2017_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'tight/muon/egammaEff2017_EGM2D.root')])
extLepSF.add_weight_sets(["MuonTightSF_2018_er EGamma_SF2D_error %s"%topcoffea_path(basepathFromTTH+'tight/muon/egammaEff2018_EGM2D.root')])

extLepSF.finalize()
SFevaluator = extLepSF.make_evaluator()


def GetLeptonSF(pt1, eta1, type1, pt2, eta2, type2, pt3=None, eta3=None, type3=None, pt4=None, eta4=None, type4=None, year=2018, sys=0):
  if sys==0:
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
    if type3!=None and type4==None:
        return( np.multiply(SF3, np.multiply(SF1,SF2)))
    if type4 == 'm':
        SF4 = ak.prod(SFevaluator['MuonLooseSF_%i'%year](np.abs(eta4), pt4) * SFevaluator['MuonTightSF_%i'%year](np.abs(eta4), pt4), axis=-1)
    elif type4 == 'e':
        SF4 = ak.prod(SFevaluator['ElecRecoSF_%i'%year](eta4, pt4) * SFevaluator['ElecLooseSF_%i'%year](np.abs(eta4), pt4) * SFevaluator['ElecLoosettHSF_%i'%year](np.abs(eta4), pt4) * SFevaluator['ElecTightSF_%i'%year](np.abs(eta4), pt4), axis=-1)
    else: print(type4, ' is not a valid type. Valid types: "m" , "e" or None')
    if type4!=None:
        return( np.multiply(SF4,np.multiply(SF3, np.multiply(SF1,SF2))))
  elif sys==1:
    if type1 == 'm':
        SF1 = ak.prod((SFevaluator['MuonLooseSF_%i'%year](np.abs(eta1), pt1)+SFevaluator['MuonLooseSF_%i_er'%year](np.abs(eta1), pt1)) * (SFevaluator['MuonTightSF_%i'%year](np.abs(eta1), pt1) + SFevaluator['MuonTightSF_%i_er'%year](np.abs(eta1), pt1)), axis=-1)
    elif type1 == 'e':
        SF1 = ak.prod((SFevaluator['ElecRecoSF_%i'%year](eta1, pt1) + SFevaluator['ElecRecoSF_%i_er'%year](eta1, pt1)) * (SFevaluator['ElecLooseSF_%i'%year](np.abs(eta1), pt1) + SFevaluator['ElecLooseSF_%i_er'%year](np.abs(eta1), pt1)) * (SFevaluator['ElecLoosettHSF_%i'%year](np.abs(eta1), pt1) + SFevaluator['ElecLoosettHSF_%i_er'%year](np.abs(eta1), pt1)) * (SFevaluator['ElecTightSF_%i'%year](np.abs(eta1), pt1) + SFevaluator['ElecTightSF_%i_er'%year](np.abs(eta1), pt1)), axis=-1)
    else: print(type1, ' is not a valid type. Valid types: "m" or "e"')
    if type2 == 'm':
        SF2 = ak.prod((SFevaluator['MuonLooseSF_%i'%year](np.abs(eta2), pt2)+SFevaluator['MuonLooseSF_%i_er'%year](np.abs(eta2), pt2)) * (SFevaluator['MuonTightSF_%i'%year](np.abs(eta2), pt2) + SFevaluator['MuonTightSF_%i_er'%year](np.abs(eta2), pt2)), axis=-1)
    elif type2 == 'e':
        SF2 = ak.prod((SFevaluator['ElecRecoSF_%i'%year](eta2, pt2) + SFevaluator['ElecRecoSF_%i_er'%year](eta2, pt2)) * (SFevaluator['ElecLooseSF_%i'%year](np.abs(eta2), pt2) + SFevaluator['ElecLooseSF_%i_er'%year](np.abs(eta2), pt2)) * (SFevaluator['ElecLoosettHSF_%i'%year](np.abs(eta2), pt2) + SFevaluator['ElecLoosettHSF_%i_er'%year](np.abs(eta2), pt2)) * (SFevaluator['ElecTightSF_%i'%year](np.abs(eta2), pt2) + SFevaluator['ElecTightSF_%i_er'%year](np.abs(eta2), pt2)), axis=-1)
    else: print(type2, ' is not a valid type. Valid types: "m" or "e"')
    if type3==None:
        return( np.multiply(SF1,SF2) )
    if type3 == 'm':
        SF3 = ak.prod((SFevaluator['MuonLooseSF_%i'%year](np.abs(eta3), pt3)+SFevaluator['MuonLooseSF_%i_er'%year](np.abs(eta3), pt3)) * (SFevaluator['MuonTightSF_%i'%year](np.abs(eta3), pt3) + SFevaluator['MuonTightSF_%i_er'%year](np.abs(eta3), pt3)), axis=-1)
    elif type3 == 'e':
        SF3 = ak.prod((SFevaluator['ElecRecoSF_%i'%year](eta3, pt3) + SFevaluator['ElecRecoSF_%i_er'%year](eta3, pt3)) * (SFevaluator['ElecLooseSF_%i'%year](np.abs(eta3), pt3) + SFevaluator['ElecLooseSF_%i_er'%year](np.abs(eta3), pt3)) * (SFevaluator['ElecLoosettHSF_%i'%year](np.abs(eta3), pt3) + SFevaluator['ElecLoosettHSF_%i_er'%year](np.abs(eta3), pt3)) * (SFevaluator['ElecTightSF_%i'%year](np.abs(eta3), pt3) + SFevaluator['ElecTightSF_%i_er'%year](np.abs(eta3), pt3)), axis=-1)
    else: print(type3, ' is not a valid type. Valid types: "m" , "e" or None')
    if type3!=None and type4==None:
        return( np.multiply(SF3, np.multiply(SF1,SF2)))
    if type4 == 'm':
        SF4 = ak.prod((SFevaluator['MuonLooseSF_%i'%year](np.abs(eta4), pt4)+SFevaluator['MuonLooseSF_%i_er'%year](np.abs(eta4), pt4)) * (SFevaluator['MuonTightSF_%i'%year](np.abs(eta4), pt4) + SFevaluator['MuonTightSF_%i_er'%year](np.abs(eta4), pt4)), axis=-1)
    elif type4 == 'e':
        SF4 = ak.prod((SFevaluator['ElecRecoSF_%i'%year](eta4, pt4) + SFevaluator['ElecRecoSF_%i_er'%year](eta4, pt4)) * (SFevaluator['ElecLooseSF_%i'%year](np.abs(eta4), pt4) + SFevaluator['ElecLooseSF_%i_er'%year](np.abs(eta4), pt4)) * (SFevaluator['ElecLoosettHSF_%i'%year](np.abs(eta4), pt4) + SFevaluator['ElecLoosettHSF_%i_er'%year](np.abs(eta4), pt4)) * (SFevaluator['ElecTightSF_%i'%year](np.abs(eta4), pt4) + SFevaluator['ElecTightSF_%i_er'%year](np.abs(eta4), pt4)), axis=-1)
    else: print(type4, ' is not a valid type. Valid types: "m" , "e" or None')
    if type4!=None:
        return( np.multiply(SF4,np.multiply(SF3, np.multiply(SF1,SF2))))
  elif sys==-1:
    if type1 == 'm':
        SF1 = ak.prod((SFevaluator['MuonLooseSF_%i'%year](np.abs(eta1), pt1)-SFevaluator['MuonLooseSF_%i_er'%year](np.abs(eta1), pt1)) * (SFevaluator['MuonTightSF_%i'%year](np.abs(eta1), pt1) - SFevaluator['MuonTightSF_%i_er'%year](np.abs(eta1), pt1)), axis=-1)
    elif type1 == 'e':
        SF1 = ak.prod((SFevaluator['ElecRecoSF_%i'%year](eta1, pt1) - SFevaluator['ElecRecoSF_%i_er'%year](eta1, pt1)) * (SFevaluator['ElecLooseSF_%i'%year](np.abs(eta1), pt1) - SFevaluator['ElecLooseSF_%i_er'%year](np.abs(eta1), pt1)) * (SFevaluator['ElecLoosettHSF_%i'%year](np.abs(eta1), pt1) - SFevaluator['ElecLoosettHSF_%i_er'%year](np.abs(eta1), pt1)) * (SFevaluator['ElecTightSF_%i'%year](np.abs(eta1), pt1) - SFevaluator['ElecTightSF_%i_er'%year](np.abs(eta1), pt1)), axis=-1)
    else: print(type1, ' is not a valid type. Valid types: "m" or "e"')
    if type2 == 'm':
        SF2 = ak.prod((SFevaluator['MuonLooseSF_%i'%year](np.abs(eta2), pt2)-SFevaluator['MuonLooseSF_%i_er'%year](np.abs(eta2), pt2)) * (SFevaluator['MuonTightSF_%i'%year](np.abs(eta2), pt2) - SFevaluator['MuonTightSF_%i_er'%year](np.abs(eta2), pt2)), axis=-1)
    elif type2 == 'e':
        SF2 = ak.prod((SFevaluator['ElecRecoSF_%i'%year](eta2, pt2) - SFevaluator['ElecRecoSF_%i_er'%year](eta2, pt2)) * (SFevaluator['ElecLooseSF_%i'%year](np.abs(eta2), pt2) - SFevaluator['ElecLooseSF_%i_er'%year](np.abs(eta2), pt2)) * (SFevaluator['ElecLoosettHSF_%i'%year](np.abs(eta2), pt2) - SFevaluator['ElecLoosettHSF_%i_er'%year](np.abs(eta2), pt2)) * (SFevaluator['ElecTightSF_%i'%year](np.abs(eta2), pt2) - SFevaluator['ElecTightSF_%i_er'%year](np.abs(eta2), pt2)), axis=-1)
    else: print(type2, ' is not a valid type. Valid types: "m" or "e"')
    if type3==None:
        return( np.multiply(SF1,SF2) )
    if type3 == 'm':
        SF3 = ak.prod((SFevaluator['MuonLooseSF_%i'%year](np.abs(eta3), pt3)-SFevaluator['MuonLooseSF_%i_er'%year](np.abs(eta3), pt3)) * (SFevaluator['MuonTightSF_%i'%year](np.abs(eta3), pt3) - SFevaluator['MuonTightSF_%i_er'%year](np.abs(eta3), pt3)), axis=-1)
    elif type3 == 'e':
        SF3 = ak.prod((SFevaluator['ElecRecoSF_%i'%year](eta3, pt3) - SFevaluator['ElecRecoSF_%i_er'%year](eta3, pt3)) * (SFevaluator['ElecLooseSF_%i'%year](np.abs(eta3), pt3) - SFevaluator['ElecLooseSF_%i_er'%year](np.abs(eta3), pt3)) * (SFevaluator['ElecLoosettHSF_%i'%year](np.abs(eta3), pt3) - SFevaluator['ElecLoosettHSF_%i_er'%year](np.abs(eta3), pt3)) * (SFevaluator['ElecTightSF_%i'%year](np.abs(eta3), pt3) - SFevaluator['ElecTightSF_%i_er'%year](np.abs(eta3), pt3)), axis=-1)
    else: print(type3, ' is not a valid type. Valid types: "m" , "e" or None')
    if type3!=None and type4==None:
        return( np.multiply(SF3, np.multiply(SF1,SF2)))
    if type4 == 'm':
        SF4 = ak.prod((SFevaluator['MuonLooseSF_%i'%year](np.abs(eta4), pt4)-SFevaluator['MuonLooseSF_%i_er'%year](np.abs(eta4), pt4)) * (SFevaluator['MuonTightSF_%i'%year](np.abs(eta4), pt4) - SFevaluator['MuonTightSF_%i_er'%year](np.abs(eta4), pt4)), axis=-1)
    elif type4 == 'e':
        SF4 = ak.prod((SFevaluator['ElecRecoSF_%i'%year](eta4, pt4) - SFevaluator['ElecRecoSF_%i_er'%year](eta4, pt4)) * (SFevaluator['ElecLooseSF_%i'%year](np.abs(eta4), pt4) - SFevaluator['ElecLooseSF_%i_er'%year](np.abs(eta4), pt4)) * (SFevaluator['ElecLoosettHSF_%i'%year](np.abs(eta4), pt4) - SFevaluator['ElecLoosettHSF_%i_er'%year](np.abs(eta4), pt4)) * (SFevaluator['ElecTightSF_%i'%year](np.abs(eta4), pt4) - SFevaluator['ElecTightSF_%i_er'%year](np.abs(eta4), pt4)), axis=-1)
    else: print(type4, ' is not a valid type. Valid types: "m" , "e" or None')
    if type4!=None:
        return( np.multiply(SF4,np.multiply(SF3, np.multiply(SF1,SF2))))


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

# Efficiencies and SFs for UL only available for 2017 and 2018
extBtagSF = lookup_tools.extractor()
extBtagSF.add_weight_sets(["BTag_2016 * %s"%topcoffea_path("data/btagSF/DeepFlav_2016.csv")])
extBtagSF.add_weight_sets(["BTag_2017 * %s"%topcoffea_path("data/btagSF/UL/DeepJet_UL17.csv")])#DeepFlav_2017.csv")])
extBtagSF.add_weight_sets(["BTag_2018 * %s"%topcoffea_path("data/btagSF/UL/DeepJet_UL18.csv")])#DeepFlav_2018.csv")])
extBtagSF.finalize()
SFevaluatorBtag = extBtagSF.make_evaluator()

MCeffFunc_2018 = GetMCeffFunc('medium', 2018)
MCeffFunc_2017 = GetMCeffFunc('medium', 2017)

def GetBtagEff(eta, pt, flavor, year=2018):
  if year==2017: return MCeffFunc_2017(pt, eta, flavor)
  else         : return MCeffFunc_2018(pt, eta, flavor)

def GetBTagSF(eta, pt, flavor, year=2018, sys=0):
  if   sys==0:  SF=SFevaluatorBtag['BTag_%iDeepJet_1_comb_central_0'%year](eta,pt,flavor)
  elif sys==1:  SF=SFevaluatorBtag['BTag_%iDeepJet_1_comb_up_0'%year](eta,pt,flavor)
  elif sys==-1: SF=SFevaluatorBtag['BTag_%iDeepJet_1_comb_down_0'%year](eta,pt,flavor)
  return (SF)

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
