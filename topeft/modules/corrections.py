'''
 This script is used to transform scale factors, which are tipically provided as 2D histograms within root files,
 into coffea format of corrections.
'''

from coffea import lookup_tools
from topcoffea.modules.paths import topcoffea_path
from topeft.modules.paths import topeft_path
import numpy as np
import awkward as ak
import scipy
import gzip
import pickle
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory
from coffea.btag_tools.btagscalefactor import BTagScaleFactor
from coffea.lookup_tools import txt_converters, rochester_lookup

from topcoffea.modules.get_param_from_jsons import GetParam
get_tc_param = GetParam(topcoffea_path("params/params.json"))
get_te_param = GetParam(topeft_path("params/params.json"))

basepathFromTTH = 'data/fromTTH/'

###### Lepton scale factors
################################################################
extLepSF = lookup_tools.extractor()

# New UL Lepton SFs
# Muon: reco
extLepSF.add_weight_sets(["MuonRecoSF_2018 NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2018_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2018_er NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2018_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2017 NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2017_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2017_er NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2017_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2016 NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2016postVFP_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2016_er NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2016postVFP_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2016APV NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2016preVFP_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2016APV_er NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2016preVFP_UL_trackerMuon.json')])
# Muon: loose POG
extLepSF.add_weight_sets(["MuonLooseSF_2018 NUM_LooseID_DEN_TrackerMuons/abseta_pt_value %s" % topcoffea_path('data/leptonSF/muon/Efficiencies_muon_generalTracks_Z_Run2018_UL_ID.json')])
extLepSF.add_weight_sets(["MuonLooseSF_2018_stat NUM_LooseID_DEN_TrackerMuons/abseta_pt_stat %s" % topcoffea_path('data/leptonSF/muon/Efficiencies_muon_generalTracks_Z_Run2018_UL_ID.json')])
extLepSF.add_weight_sets(["MuonLooseSF_2018_syst NUM_LooseID_DEN_TrackerMuons/abseta_pt_syst %s" % topcoffea_path('data/leptonSF/muon/Efficiencies_muon_generalTracks_Z_Run2018_UL_ID.json')])
extLepSF.add_weight_sets(["MuonLooseSF_2017 NUM_LooseID_DEN_TrackerMuons/abseta_pt_value %s" % topcoffea_path('data/leptonSF/muon/Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.json')])
extLepSF.add_weight_sets(["MuonLooseSF_2017_stat NUM_LooseID_DEN_TrackerMuons/abseta_pt_stat %s" % topcoffea_path('data/leptonSF/muon/Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.json')])
extLepSF.add_weight_sets(["MuonLooseSF_2017_syst NUM_LooseID_DEN_TrackerMuons/abseta_pt_syst %s" % topcoffea_path('data/leptonSF/muon/Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.json')])
extLepSF.add_weight_sets(["MuonLooseSF_2016 NUM_LooseID_DEN_TrackerMuons/abseta_pt_value %s" % topcoffea_path('data/leptonSF/muon/Efficiencies_muon_generalTracks_Z_Run2016_UL_ID.json')])
extLepSF.add_weight_sets(["MuonLooseSF_2016_stat NUM_LooseID_DEN_TrackerMuons/abseta_pt_stat %s" % topcoffea_path('data/leptonSF/muon/Efficiencies_muon_generalTracks_Z_Run2016_UL_ID.json')])
extLepSF.add_weight_sets(["MuonLooseSF_2016_syst NUM_LooseID_DEN_TrackerMuons/abseta_pt_syst %s" % topcoffea_path('data/leptonSF/muon/Efficiencies_muon_generalTracks_Z_Run2016_UL_ID.json')])
extLepSF.add_weight_sets(["MuonLooseSF_2016APV NUM_LooseID_DEN_TrackerMuons/abseta_pt_value %s" % topcoffea_path('data/leptonSF/muon/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID.json')])
extLepSF.add_weight_sets(["MuonLooseSF_2016APV_stat NUM_LooseID_DEN_TrackerMuons/abseta_pt_stat %s" % topcoffea_path('data/leptonSF/muon/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID.json')])
extLepSF.add_weight_sets(["MuonLooseSF_2016APV_syst NUM_LooseID_DEN_TrackerMuons/abseta_pt_syst %s" % topcoffea_path('data/leptonSF/muon/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID.json')])
# Muon: ISO + IP (Barbara)
extLepSF.add_weight_sets(["MuonIsoSF_2018 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2018_iso_EGM2D.root')])
extLepSF.add_weight_sets(["MuonIsoSF_2018_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2018_iso_EGM2D.root')])
extLepSF.add_weight_sets(["MuonIsoSF_2017_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2017_iso_EGM2D.root')])
extLepSF.add_weight_sets(["MuonIsoSF_2017 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2017_iso_EGM2D.root')])
extLepSF.add_weight_sets(["MuonIsoSF_2016_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2016_iso_EGM2D.root')])
extLepSF.add_weight_sets(["MuonIsoSF_2016 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2016_iso_EGM2D.root')])
extLepSF.add_weight_sets(["MuonIsoSF_2016APV_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2016APV_iso_EGM2D.root')])
extLepSF.add_weight_sets(["MuonIsoSF_2016APV EGamma_SF2D %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2016APV_iso_EGM2D.root')])
# Muon: looseMVA&tight (Barbara)
extLepSF.add_weight_sets(["MuonSF_2018 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2018_EGM2D.root')])
extLepSF.add_weight_sets(["MuonSF_2018_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2018_EGM2D.root')])
extLepSF.add_weight_sets(["MuonSF_2017_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2017_EGM2D.root')])
extLepSF.add_weight_sets(["MuonSF_2017 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2017_EGM2D.root')])
extLepSF.add_weight_sets(["MuonSF_2016_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2016_EGM2D.root')])
extLepSF.add_weight_sets(["MuonSF_2016 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2016_EGM2D.root')])
extLepSF.add_weight_sets(["MuonSF_2016APV_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2016APV_EGM2D.root')])
extLepSF.add_weight_sets(["MuonSF_2016APV EGamma_SF2D %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2016APV_EGM2D.root')])
# Elec: reco
extLepSF.add_weight_sets(["ElecRecoSFAb_2018 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2018_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2018 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2018_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2017 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2017_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2017 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2017_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2016 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2016_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2016 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2016_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2016APV EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2016APV_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2016APV EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2016APV_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_ptBelow20_EGM2D.root')])
# Elec: loose (Barbara)
extLepSF.add_weight_sets(["ElecLooseSF_2018 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_recoToloose_EGM2D.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2018_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_recoToloose_EGM2D.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2017_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_recoToloose_EGM2D.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2017 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_recoToloose_EGM2D.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2016_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_recoToloose_EGM2D.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2016 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_recoToloose_EGM2D.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2016APV_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_recoToloose_EGM2D.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2016APV EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_recoToloose_EGM2D.root')])
# Elec: ISO + IP (Barbara)
extLepSF.add_weight_sets(["ElecIsoSF_2018 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_iso_EGM2D.root')])
extLepSF.add_weight_sets(["ElecIsoSF_2018_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_iso_EGM2D.root')])
extLepSF.add_weight_sets(["ElecIsoSF_2017_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_iso_EGM2D.root')])
extLepSF.add_weight_sets(["ElecIsoSF_2017 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_iso_EGM2D.root')])
extLepSF.add_weight_sets(["ElecIsoSF_2016_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_iso_EGM2D.root')])
extLepSF.add_weight_sets(["ElecIsoSF_2016 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_iso_EGM2D.root')])
extLepSF.add_weight_sets(["ElecIsoSF_2016APV_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_iso_EGM2D.root')])
extLepSF.add_weight_sets(["ElecIsoSF_2016APV EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_iso_EGM2D.root')])
# Elec: looseMVA&tight (Barbara)
extLepSF.add_weight_sets(["ElecSF_2018_2lss EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_2lss_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2018_2lss_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_2lss_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2018_3l EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_3l_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2018_3l_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_3l_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2017_2lss EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_2lss_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2017_2lss_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_2lss_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2017_3l EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_3l_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2017_3l_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_3l_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2016_2lss EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_2lss_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2016_2lss_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_2lss_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2016_3l EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_3l_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2016_3l_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_3l_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2016APV_2lss EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_2lss_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2016APV_2lss_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_2lss_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2016APV_3l EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_3l_EGM2D.root')])
extLepSF.add_weight_sets(["ElecSF_2016APV_3l_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_3l_EGM2D.root')])
#Tau SF
extLepSF.add_weight_sets(["TauSF_2016APV_Loose Tau_SF/dm_pt_value %s"%topcoffea_path('data/TauSF/TauSFUL2016_preVFPLoose.json')])
extLepSF.add_weight_sets(["TauSF_2016_Loose Tau_SF/dm_pt_value %s"%topcoffea_path('data/TauSF/TauSFUL2016_postVFPLoose.json')])
extLepSF.add_weight_sets(["TauSF_2017_Loose Tau_SF/dm_pt_value %s"%topcoffea_path('data/TauSF/TauSFUL2017Loose.json')])
extLepSF.add_weight_sets(["TauSF_2018_Loose Tau_SF/dm_pt_value %s"%topcoffea_path('data/TauSF/TauSFUL2018Loose.json')])

extLepSF.add_weight_sets(["TauSF_2016APV_Loose_up Tau_SF/dm_pt_up %s"%topcoffea_path('data/TauSF/TauSFUL2016_preVFPLoose.json')])
extLepSF.add_weight_sets(["TauSF_2016_Loose_up Tau_SF/dm_pt_up %s"%topcoffea_path('data/TauSF/TauSFUL2016_postVFPLoose.json')])
extLepSF.add_weight_sets(["TauSF_2017_Loose_up Tau_SF/dm_pt_up %s"%topcoffea_path('data/TauSF/TauSFUL2017Loose.json')])
extLepSF.add_weight_sets(["TauSF_2018_Loose_up Tau_SF/dm_pt_up %s"%topcoffea_path('data/TauSF/TauSFUL2018Loose.json')])

extLepSF.add_weight_sets(["TauSF_2016APV_Loose_down Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2016_preVFPLoose.json')])
extLepSF.add_weight_sets(["TauSF_2016_Loose_down Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2016_postVFPLoose.json')])
extLepSF.add_weight_sets(["TauSF_2017_Loose_down Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2017Loose.json')])
extLepSF.add_weight_sets(["TauSF_2018_Loose_down Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2018Loose.json')])

extLepSF.add_weight_sets(["TauSF_2016APV_Medium Tau_SF/dm_pt_value %s"%topcoffea_path('data/TauSF/TauSFUL2016_preVFPMedium.json')])
extLepSF.add_weight_sets(["TauSF_2016_Medium Tau_SF/dm_pt_value %s"%topcoffea_path('data/TauSF/TauSFUL2016_postVFPMedium.json')])
extLepSF.add_weight_sets(["TauSF_2017_Medium Tau_SF/dm_pt_value %s"%topcoffea_path('data/TauSF/TauSFUL2017Medium.json')])
extLepSF.add_weight_sets(["TauSF_2018_Medium Tau_SF/dm_pt_value %s"%topcoffea_path('data/TauSF/TauSFUL2018Medium.json')])

extLepSF.add_weight_sets(["TauSF_2016APV_Medium_up Tau_SF/dm_pt_up %s"%topcoffea_path('data/TauSF/TauSFUL2016_preVFPMedium.json')])
extLepSF.add_weight_sets(["TauSF_2016_Medium_up Tau_SF/dm_pt_up %s"%topcoffea_path('data/TauSF/TauSFUL2016_postVFPMedium.json')])
extLepSF.add_weight_sets(["TauSF_2017_Medium_up Tau_SF/dm_pt_up %s"%topcoffea_path('data/TauSF/TauSFUL2017Medium.json')])
extLepSF.add_weight_sets(["TauSF_2018_Medium_up Tau_SF/dm_pt_up %s"%topcoffea_path('data/TauSF/TauSFUL2018Medium.json')])

extLepSF.add_weight_sets(["TauSF_2016APV_Medium_down Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2016_preVFPMedium.json')])
extLepSF.add_weight_sets(["TauSF_2016_Medium_down Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2016_postVFPMedium.json')])
extLepSF.add_weight_sets(["TauSF_2017_Medium_down Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2017Medium.json')])
extLepSF.add_weight_sets(["TauSF_2018_Medium_down Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2018Medium.json')])

extLepSF.add_weight_sets(["TauSF_2016APV_Tight Tau_SF/dm_pt_value %s"%topcoffea_path('data/TauSF/TauSFUL2016_preVFPTight.json')])
extLepSF.add_weight_sets(["TauSF_2016_Tight Tau_SF/dm_pt_value %s"%topcoffea_path('data/TauSF/TauSFUL2016_postVFPTight.json')])
extLepSF.add_weight_sets(["TauSF_2017_Tight Tau_SF/dm_pt_value %s"%topcoffea_path('data/TauSF/TauSFUL2017Tight.json')])
extLepSF.add_weight_sets(["TauSF_2018_Tight Tau_SF/dm_pt_value %s"%topcoffea_path('data/TauSF/TauSFUL2018Tight.json')])

extLepSF.add_weight_sets(["TauSF_2016APV_Tight_up Tau_SF/dm_pt_up %s"%topcoffea_path('data/TauSF/TauSFUL2016_preVFPTight.json')])
extLepSF.add_weight_sets(["TauSF_2016_Tight_up Tau_SF/dm_pt_up %s"%topcoffea_path('data/TauSF/TauSFUL2016_postVFPTight.json')])
extLepSF.add_weight_sets(["TauSF_2017_Tight_up Tau_SF/dm_pt_up %s"%topcoffea_path('data/TauSF/TauSFUL2017Tight.json')])
extLepSF.add_weight_sets(["TauSF_2018_Tight_up Tau_SF/dm_pt_up %s"%topcoffea_path('data/TauSF/TauSFUL2018Tight.json')])

extLepSF.add_weight_sets(["TauSF_2016APV_Tight_down Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2016_preVFPTight.json')])
extLepSF.add_weight_sets(["TauSF_2016_Tight_down Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2016_postVFPTight.json')])
extLepSF.add_weight_sets(["TauSF_2017_Tight_down Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2017Tight.json')])
extLepSF.add_weight_sets(["TauSF_2018_Tight_down Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2018Tight.json')])

extLepSF.add_weight_sets(["TauTES_2016APV Tau_TES/dm_pt_value %s"%topcoffea_path('data/TauSF/TauTESUL2016_preVFP.json')])
extLepSF.add_weight_sets(["TauTES_2016 Tau_TES/dm_pt_value %s"%topcoffea_path('data/TauSF/TauTESUL2016_postVFP.json')])
extLepSF.add_weight_sets(["TauTES_2017 Tau_TES/dm_pt_value %s"%topcoffea_path('data/TauSF/TauTESUL2017.json')])
extLepSF.add_weight_sets(["TauTES_2018 Tau_TES/dm_pt_value %s"%topcoffea_path('data/TauSF/TauTESUL2018.json')])

extLepSF.add_weight_sets(["TauTES_2016APV_up Tau_TES/dm_pt_up %s"%topcoffea_path('data/TauSF/TauTESUL2016_preVFP.json')])
extLepSF.add_weight_sets(["TauTES_2016_up Tau_TES/dm_pt_up %s"%topcoffea_path('data/TauSF/TauTESUL2016_postVFP.json')])
extLepSF.add_weight_sets(["TauTES_2017_up Tau_TES/dm_pt_up %s"%topcoffea_path('data/TauSF/TauTESUL2017.json')])
extLepSF.add_weight_sets(["TauTES_2018_up Tau_TES/dm_pt_up %s"%topcoffea_path('data/TauSF/TauTESUL2018.json')])

extLepSF.add_weight_sets(["TauTES_2016APV_down Tau_TES/dm_pt_down %s"%topcoffea_path('data/TauSF/TauTESUL2016_preVFP.json')])
extLepSF.add_weight_sets(["TauTES_2016_down Tau_TES/dm_pt_down %s"%topcoffea_path('data/TauSF/TauTESUL2016_postVFP.json')])
extLepSF.add_weight_sets(["TauTES_2017_down Tau_TES/dm_pt_down %s"%topcoffea_path('data/TauSF/TauTESUL2017.json')])
extLepSF.add_weight_sets(["TauTES_2018_down Tau_TES/dm_pt_down %s"%topcoffea_path('data/TauSF/TauTESUL2018.json')])

extLepSF.add_weight_sets(["TauFakeSF_2016APV TauFake/pt_value %s"%topcoffea_path('data/TauSF/TauFakeSF_2016APV.json')])
extLepSF.add_weight_sets(["TauFakeSF_2016 TauFake/pt_value %s"%topcoffea_path('data/TauSF/TauFakeSF_2016.json')])
extLepSF.add_weight_sets(["TauFakeSF_2017 TauFake/pt_value %s"%topcoffea_path('data/TauSF/TauFakeSF_2017.json')])
extLepSF.add_weight_sets(["TauFakeSF_2018 TauFake/pt_value %s"%topcoffea_path('data/TauSF/TauFakeSF_2018.json')])

extLepSF.add_weight_sets(["TauFake_2016 Tau_SF/eta_gen_value %s"%topcoffea_path('data/TauSF/TauFakeUL2016.json')])
extLepSF.add_weight_sets(["TauFake_2017 Tau_SF/eta_gen_value %s"%topcoffea_path('data/TauSF/TauFakeUL2017.json')])
extLepSF.add_weight_sets(["TauFake_2018 Tau_SF/eta_gen_value %s"%topcoffea_path('data/TauSF/TauFakeUL2018.json')])

# Fake rate
for year in ['2016APV_2016', 2017, 2018]:
    for syst in ['','_up','_down','_be1','_be2','_pt1','_pt2']:
        extLepSF.add_weight_sets([("MuonFR_{year}{syst} FR_mva085_mu_data_comb_recorrected{syst} %s" % topcoffea_path(basepathFromTTH + 'fakerate/fr_{year}_recorrected.root')).format(year=year,syst=syst)])
        extLepSF.add_weight_sets([("ElecFR_{year}{syst} FR_mva090_el_data_comb_NC_recorrected{syst} %s" % topcoffea_path(basepathFromTTH + 'fakerate/fr_{year}_recorrected.root')).format(year=year,syst=syst)])

extLepSF.finalize()
SFevaluator = extLepSF.make_evaluator()

###### Photon scale factors
################################################################
extPhoSF = lookup_tools.extractor()

# New UL Photon SFs
# Muon: reco
# pT vs super cluster eta
extPhoSF.add_weight_sets(["PhotonTightSF_2016 EGamma_SF2D %s"    % topcoffea_path('data/photonSF/egammaEffi_EGM2D_Pho_Tight_UL16.root')])
extPhoSF.add_weight_sets(["PhotonTightSF_2016_err EGamma_SF2D_err %s"    % topcoffea_path('data/photonSF/egammaEffi_EGM2D_Pho_Tight_UL16.root')])
extPhoSF.add_weight_sets(["PhotonTightSF_2016APV EGamma_SF2D %s" % topcoffea_path('data/photonSF/egammaEffi_EGM2D_Pho_Tight_UL16_postVFP.root')])
extPhoSF.add_weight_sets(["PhotonTightSF_2016APV_err EGamma_SF2D_err %s" % topcoffea_path('data/photonSF/egammaEffi_EGM2D_Pho_Tight_UL16_postVFP.root')])
extPhoSF.add_weight_sets(["PhotonTightSF_2017 EGamma_SF2D %s"    % topcoffea_path('data/photonSF/egammaEffi_EGM2D_PHO_Tight_UL17.root')])
extPhoSF.add_weight_sets(["PhotonTightSF_2017_err EGamma_SF2D_err %s"    % topcoffea_path('data/photonSF/egammaEffi_EGM2D_PHO_Tight_UL17.root')])
extPhoSF.add_weight_sets(["PhotonTightSF_2018 EGamma_SF2D %s"    % topcoffea_path('data/photonSF/egammaEffi_EGM2D_Pho_Tight_UL18.root')])
extPhoSF.add_weight_sets(["PhotonTightSF_2018_err EGamma_SF2D_err %s"    % topcoffea_path('data/photonSF/egammaEffi_EGM2D_Pho_Tight_UL18.root')])

extPhoSF.finalize()
PhoSFevaluator = extPhoSF.make_evaluator()

ffSysts=['','_up','_down','_be1','_be2','_pt1','_pt2']

def ApplyTES(events, Taus, isData):
    if isData:
        return Taus.pt
    #padded_Taus = ak.pad_none(Taus,1)
    #padded_Taus = ak.with_name(padded_Taus, "TauCandidate")
    pt  = Taus.pt
    dm  = Taus.decayMode
    gen = Taus.genPartFlav

    whereFlag = ((pt>20) & (pt<205) & (gen==5))
    tes = np.where(whereFlag, SFevaluator['TauTES_{year}'.format(year=year)](dm,pt), 1)
    return (Taus.pt*tes, Taus.mass*tes)
    #return(Taus.pt*tes)

def AttachTauSF(events, Taus, year):
    padded_Taus = ak.pad_none(Taus,1)
    padded_Taus = ak.with_name(padded_Taus, "TauCandidate")
    padded_Taus["sf_tau"] = 1.0
    padded_Taus["sf_tau_up"] = 1.0
    padded_Taus["sf_tau_down"] = 1.0

    pt  = padded_Taus.pt
    dm  = padded_Taus.decayMode
    wp  = padded_Taus.idDeepTau2017v2p1VSjet
    eta = padded_Taus.eta
    gen = padded_Taus.genPartFlav
    mass= padded_Taus.mass

    whereFlag = ((pt>20) & (pt<205) & (gen==5) & (padded_Taus["isLoose"]) & (~padded_Taus["isMedium"]))
    real_sf_loose = np.where(whereFlag, SFevaluator['TauSF_{year}_Loose'.format(year=year)](dm,pt), 1)
    real_sf_loose_up = np.where(whereFlag, SFevaluator['TauSF_{year}_Loose_up'.format(year=year)](dm,pt), 1)
    real_sf_loose_down = np.where(whereFlag, SFevaluator['TauSF_{year}_Loose_down'.format(year=year)](dm,pt), 1)
    whereFlag = ((pt>20) & (pt<205) & (gen==5) & (padded_Taus["isMedium"]) & (~padded_Taus["isTight"]))
    real_sf_medium = np.where(whereFlag, SFevaluator['TauSF_{year}_Medium'.format(year=year)](dm,pt), 1)
    real_sf_medium_up = np.where(whereFlag, SFevaluator['TauSF_{year}_Medium_up'.format(year=year)](dm,pt), 1)
    real_sf_medium_down = np.where(whereFlag, SFevaluator['TauSF_{year}_Medium_down'.format(year=year)](dm,pt), 1)
    whereFlag = ((pt>20) & (pt<205) & (gen==5) & (padded_Taus["isTight"]))
    real_sf_tight = np.where(whereFlag, SFevaluator['TauSF_{year}_Tight'.format(year=year)](dm,pt), 1)
    real_sf_tight_up = np.where(whereFlag, SFevaluator['TauSF_{year}_Tight_up'.format(year=year)](dm,pt), 1)
    real_sf_tight_down = np.where(whereFlag, SFevaluator['TauSF_{year}_Tight_down'.format(year=year)](dm,pt), 1)
    whereFlag = ((pt>20) & (pt<205) & (gen!=5) & (gen!=0) & (gen!=6))
    if year == "2016APV":
        year = "2016"
    fake_sf = np.where(whereFlag, SFevaluator['TauFake_{year}'.format(year=year)](np.abs(eta),gen), 1)
    whereFlag = ((pt>20) & (pt<205) & (gen!=5) & (gen!=4) & (gen!=3) & (gen!=2) & (gen!=1) & (~padded_Taus["isLoose"]) & (padded_Taus["isVLoose"]))
    faker_sf = np.where(whereFlag, SFevaluator['TauFakeSF_{year}'.format(year=year)](pt), 1)
    padded_Taus["sf_tau"] = real_sf_loose*real_sf_medium*real_sf_tight*fake_sf*faker_sf
    padded_Taus["sf_tau_up"] = real_sf_loose_up*real_sf_medium_up*real_sf_tight_up
    padded_Taus["sf_tau_down"] = real_sf_loose_down*real_sf_medium_down*real_sf_tight_down

    events["sf_2l_taus"] = padded_Taus.sf_tau[:,0]
    events["sf_2l_taus_hi"] = padded_Taus.sf_tau_up[:,0]
    events["sf_2l_taus_lo"] = padded_Taus.sf_tau_down[:,0]

def AttachPerLeptonFR(leps, flavor, year):
    # Get the flip rates lookup object
    if year == "2016APV": flip_year_name = "UL16APV"
    elif year == "2016": flip_year_name = "UL16"
    elif year == "2017": flip_year_name = "UL17"
    elif year == "2018": flip_year_name = "UL18"
    else: raise Exception(f"Not a known year: {year}")
    with gzip.open(topeft_path(f"data/fliprates/flip_probs_topcoffea_{flip_year_name}.pkl.gz")) as fin:
        flip_hist = pickle.load(fin)
        flip_lookup = lookup_tools.dense_lookup.dense_lookup(flip_hist.values()[()],[flip_hist.axes["pt"].edges,flip_hist.axes["eta"].edges])

    # Get the fliprate scaling factor for the given year
    chargeflip_sf = get_te_param("chargeflip_sf_dict")[flip_year_name]

    # For FR filepath naming conventions
    if '2016' in year:
        year = '2016APV_2016'

    # Add the flip/fake info into the leps opject
    for syst in ffSysts:
        fr = SFevaluator['{flavor}FR_{year}{syst}'.format(flavor=flavor,year=year,syst=syst)](leps.conept, np.abs(leps.eta) )
        leps['fakefactor%s' % syst] = ak.fill_none(-fr/(1-fr),0) # this is the factor that actually enters the expressions

    if year == '2016APV_2016':
        leps['fakefactor_elclosurefactor'] = (np.abs(leps.pdgId)==11) * ((np.abs(leps.eta) > 1.5)*0.5 + (np.abs(leps.eta) < 1.5)*0.1) + 1.0
        leps['fakefactor_muclosurefactor'] = (np.abs(leps.pdgId)==13)*0.05 + 1.0
    if year == '2017':
        leps['fakefactor_elclosurefactor'] = (np.abs(leps.pdgId)==11)*0.2 + 1.0
        leps['fakefactor_muclosurefactor'] = (np.abs(leps.pdgId)==13)*0.2 + 1.0
    if year == '2018':
        leps['fakefactor_elclosurefactor'] = (np.abs(leps.pdgId)==11) * ((np.abs(leps.eta) > 1.5)*0.5 + (np.abs(leps.eta) < 1.5)*0.1) + 1.0
        leps['fakefactor_muclosurefactor'] = (np.abs(leps.pdgId)==13)*0.05 + 1.0

    for flav in ['el','mu']:
        leps['fakefactor_%sclosuredown' % flav] = leps['fakefactor'] / leps['fakefactor_%sclosurefactor' % flav]
        leps['fakefactor_%sclosureup' % flav]   = leps['fakefactor'] * leps['fakefactor_%sclosurefactor' % flav]

    if flavor == "Elec":
        leps['fliprate'] = (chargeflip_sf)*(flip_lookup(leps.pt,abs(leps.eta)))
    else:
        leps['fliprate'] = np.zeros_like(leps.pt)

def fakeRateWeight1l(events, lep1):
    for syst in ffSysts+['_elclosureup','_elclosuredown','_muclosureup','_muclosuredown']:
        fakefactor_2l =  (~lep1.isTightLep + (1)*(lep1.isTightLep)) # if all are tight the FF is 1 because events are in the SR
        fakefactor_2l =  fakefactor_2l*(lep1.isTightLep + (~lep1.isTightLep)*getattr(lep1,'fakefactor%s'%syst))
        events['fakefactor_1l%s'%syst]=fakefactor_2l
    # Calculation of flip factor: flip_factor_2l = 1*(isSS) + (fliprate1 + fliprate2)*(isOS):
    #     - For SS events = 1
    #     - For OS events = (fliprate1 + fliprate2)
    events['flipfactor_1l']=1*((lep1.charge)!=0) + (((lep1.fliprate))*((lep1.charge)==0))

def fakeRateWeight2l(events, lep1, lep2):
    for syst in ffSysts+['_elclosureup','_elclosuredown','_muclosureup','_muclosuredown']:
        fakefactor_2l = -1*(~lep1.isTightLep | ~lep2.isTightLep) + 1*(lep1.isTightLep & lep2.isTightLep) # if all are tight the FF is 1 because events are in the SR
        fakefactor_2l = fakefactor_2l * (lep1.isTightLep + (~lep1.isTightLep) * getattr(lep1,'fakefactor%s' % syst))
        fakefactor_2l = fakefactor_2l * (lep2.isTightLep + (~lep2.isTightLep) * getattr(lep2,'fakefactor%s' % syst))
        events['fakefactor_2l%s' % syst] = fakefactor_2l
    # Calculation of flip factor: flip_factor_2l = 1*(isSS) + (fliprate1 + fliprate2)*(isOS):
    #     - For SS events = 1
    #     - For OS events = (fliprate1 + fliprate2)
    events['flipfactor_2l'] = 1*((lep1.charge + lep2.charge) != 0) + (((lep1.fliprate + lep2.fliprate)) * ((lep1.charge + lep2.charge) == 0)) # only apply fliprate for OS events. to handle the OS control regions later :)

def fakeRateWeight3l(events, lep1, lep2, lep3):
    for syst in ffSysts+['_elclosureup','_elclosuredown','_muclosureup','_muclosuredown'] :
        fakefactor_3l = -1*(~lep1.isTightLep | ~lep2.isTightLep | ~lep3.isTightLep) + 1*(lep1.isTightLep & lep2.isTightLep & lep3.isTightLep) # if all are tight the FF is 1 because events are in the SR  and we dont want to weight them
        fakefactor_3l = fakefactor_3l * (lep1.isTightLep + (~lep1.isTightLep) * getattr(lep1,'fakefactor%s' % syst))
        fakefactor_3l = fakefactor_3l * (lep2.isTightLep + (~lep2.isTightLep) * getattr(lep2,'fakefactor%s' % syst))
        fakefactor_3l = fakefactor_3l * (lep3.isTightLep + (~lep3.isTightLep) * getattr(lep3,'fakefactor%s' % syst))
        events['fakefactor_3l%s' % syst] = fakefactor_3l

def AttachMuonSF(muons, year):
    '''
      Description:
          Inserts 'sf_nom', 'sf_hi', and 'sf_lo' into the muons array passed to this function. These
          values correspond to the nominal, up, and down muon scalefactor values respectively.
    '''
    eta = np.abs(muons.eta)
    pt = muons.pt
    if year not in ['2016','2016APV','2017','2018']: raise Exception(f"Error: Unknown year \"{year}\".")
    reco_sf  = np.where(pt < 20,SFevaluator['MuonRecoSF_{year}'.format(year=year)](eta,pt),1) # sf=1 when pt>20 becuase there is no reco SF available
    reco_err = np.where(pt < 20,SFevaluator['MuonRecoSF_{year}_er'.format(year=year)](eta,pt),0) # sf error =0 when pt>20 becuase there is no reco SF available
    loose_sf  = SFevaluator['MuonLooseSF_{year}'.format(year=year)](eta,pt)
    loose_err = np.sqrt(
        SFevaluator['MuonLooseSF_{year}_stat'.format(year=year)](eta,pt) * SFevaluator['MuonLooseSF_{year}_stat'.format(year=year)](eta,pt) +
        SFevaluator['MuonLooseSF_{year}_syst'.format(year=year)](eta,pt) * SFevaluator['MuonLooseSF_{year}_syst'.format(year=year)](eta,pt)
    )
    iso_sf  = SFevaluator['MuonIsoSF_{year}'.format(year=year)](eta,pt)
    iso_err = SFevaluator['MuonIsoSF_{year}_er'.format(year=year)](eta,pt)
    new_sf  = SFevaluator['MuonSF_{year}'.format(year=year)](eta,pt)
    new_err = SFevaluator['MuonSF_{year}_er'.format(year=year)](eta,pt)

    muons['sf_nom_2l_muon'] = new_sf * reco_sf * loose_sf * iso_sf
    muons['sf_hi_2l_muon']  = (new_sf + new_err) * (reco_sf + reco_err) * (loose_sf + loose_err) * (iso_sf + iso_err)
    muons['sf_lo_2l_muon']  = (new_sf - new_err) * (reco_sf - reco_err) * (loose_sf - loose_err) * (iso_sf - iso_err)
    muons['sf_nom_3l_muon'] = new_sf * reco_sf * loose_sf
    muons['sf_hi_3l_muon']  = (new_sf + new_err) * (reco_sf + reco_err) * (loose_sf + loose_err) * (iso_sf + iso_err)
    muons['sf_lo_3l_muon']  = (new_sf - new_err) * (reco_sf - reco_err) * (loose_sf - loose_err) * (iso_sf - iso_err)
    muons['sf_nom_2l_elec'] = ak.ones_like(new_sf)
    muons['sf_hi_2l_elec']  = ak.ones_like(new_sf)
    muons['sf_lo_2l_elec']  = ak.ones_like(new_sf)
    muons['sf_nom_3l_elec'] = ak.ones_like(new_sf)
    muons['sf_hi_3l_elec']  = ak.ones_like(new_sf)
    muons['sf_lo_3l_elec']  = ak.ones_like(new_sf)

def AttachElectronSF(electrons, year):
    '''
      Description:
          Inserts 'sf_nom', 'sf_hi', and 'sf_lo' into the electrons array passed to this function. These
          values correspond to the nominal, up, and down electron scalefactor values respectively.
    '''
    eta = electrons.eta
    pt = electrons.pt

    if year not in ['2016','2016APV','2017','2018']:
        raise Exception(f"Error: Unknown year \"{year}\".")

    reco_sf  = np.where(
        pt < 20,
        SFevaluator['ElecRecoSFBe_{year}'.format(year=year)](eta,pt),
        SFevaluator['ElecRecoSFAb_{year}'.format(year=year)](eta,pt)
    )
    reco_err = np.where(
        pt < 20,
        SFevaluator['ElecRecoSFBe_{year}_er'.format(year=year)](eta,pt),
        SFevaluator['ElecRecoSFAb_{year}_er'.format(year=year)](eta,pt)
    )
    new_sf_2l  = SFevaluator['ElecSF_{year}_2lss'.format(year=year)](np.abs(eta),pt)
    new_err_2l = SFevaluator['ElecSF_{year}_2lss_er'.format(year=year)](np.abs(eta),pt)
    new_sf_3l  = SFevaluator['ElecSF_{year}_3l'.format(year=year)](np.abs(eta),pt)
    new_err_3l = SFevaluator['ElecSF_{year}_3l_er'.format(year=year)](np.abs(eta),pt)
    loose_sf  = SFevaluator['ElecLooseSF_{year}'.format(year=year)](np.abs(eta),pt)
    loose_err = SFevaluator['ElecLooseSF_{year}_er'.format(year=year)](np.abs(eta),pt)
    iso_sf  = SFevaluator['ElecIsoSF_{year}'.format(year=year)](np.abs(eta),pt)
    iso_err = SFevaluator['ElecIsoSF_{year}_er'.format(year=year)](np.abs(eta),pt)

    electrons['sf_nom_2l_elec'] = reco_sf * new_sf_2l * loose_sf * iso_sf
    electrons['sf_hi_2l_elec']  = (reco_sf + reco_err) * (new_sf_2l + new_err_2l) * (loose_sf + loose_err) * (iso_sf + iso_err)
    electrons['sf_lo_2l_elec']  = (reco_sf - reco_err) * (new_sf_2l - new_err_2l) * (loose_sf - loose_err) * (iso_sf - iso_err)
    electrons['sf_nom_3l_elec'] = reco_sf * new_sf_3l * loose_sf
    electrons['sf_hi_3l_elec']  = (reco_sf + reco_err) * (new_sf_3l + new_err_3l) * (loose_sf + loose_err) * (iso_sf + iso_err)
    electrons['sf_lo_3l_elec']  = (reco_sf - reco_err) * (new_sf_3l - new_err_3l) * (loose_sf - loose_err) * (iso_sf - iso_err)
    electrons['sf_nom_2l_muon'] = ak.ones_like(reco_sf)
    electrons['sf_hi_2l_muon']  = ak.ones_like(reco_sf)
    electrons['sf_lo_2l_muon']  = ak.ones_like(reco_sf)
    electrons['sf_nom_3l_muon'] = ak.ones_like(reco_sf)
    electrons['sf_hi_3l_muon']  = ak.ones_like(reco_sf)
    electrons['sf_lo_3l_muon']  = ak.ones_like(reco_sf)

def AttachPhotonSF(photons, year):
    '''
      Description:
          Inserts 'sf_nom', 'sf_hi', and 'sf_lo' into the photons array passed to this function. These
          values correspond to the nominal, up, and down photon scalefactor values respectively.
    '''
    sieie = np.abs(photons.sieie)
    pt = photons.pt
    if year not in ['2016','2016APV','2017','2018']: raise Exception(f"Error: Unknown year \"{year}\".")
    tight_sf  = PhoSFevaluator['PhotonTightSF_{year}'.format(year=year)](sieie,pt)
    tight_err = PhoSFevaluator['PhotonTightSF_{year}_err'.format(year=year)](sieie,pt)

    photons['sf_nom_photon'] = tight_sf
    photons['sf_hi_photon']  = (tight_sf + tight_err)
    photons['sf_lo_photon']  = (tight_sf - tight_err)

###### Btag scale factors
################################################################
# Hard-coded to DeepJet algorithm, loose and medium WPs

# MC efficiencies
def GetMCeffFunc(year, wp='medium', flav='b'):
    if year not in ['2016','2016APV','2017','2018']:
        raise Exception(f"Error: Unknown year \"{year}\".")
    pathToBtagMCeff = topeft_path('data/btagSF/UL/btagMCeff_%s.pkl.gz'%year)
    hists = {}
    with gzip.open(pathToBtagMCeff) as fin:
        hin = pickle.load(fin)
        for k in hin.keys():
            if k in hists:
                hists[k] += hin[k]
            else:
                hists[k] = hin[k]
    h = hists['jetptetaflav']
    hnum = h[{'WP': wp}]
    hden = h[{'WP': 'all'}]
    getnum = lookup_tools.dense_lookup.dense_lookup(
        hnum.values(flow=True)[1:,1:,1:], # Strip off underflow
        [
            hnum.axes['pt'].edges,
            hnum.axes['abseta'].edges,
            hnum.axes['flav'].edges
        ]
    )
    getden = lookup_tools.dense_lookup.dense_lookup(
        hden.values(flow=True)[1:,1:,1:],
        [
            hden.axes['pt'].edges,
            hnum.axes['abseta'].edges,
            hden.axes['flav'].edges
        ]
    )
    values = hnum.values(flow=True)[1:,1:,1:]
    edges = [hnum.axes['pt'].edges, hnum.axes['abseta'].edges, hnum.axes['flav'].edges]
    fun = lambda pt, abseta, flav: getnum(pt,abseta,flav)/getden(pt,abseta,flav)
    return fun

def GetBtagEff(jets, year, wp='medium'):
    if year not in ['2016','2016APV','2017','2018']:
        raise Exception(f"Error: Unknown year \"{year}\".")
    return GetMCeffFunc(year,wp)(jets.pt, np.abs(jets.eta), jets.hadronFlavour)

def GetBTagSF(jets, year, wp='MEDIUM', syst='central'):
    if   year == '2016': SFevaluatorBtag = BTagScaleFactor(topeft_path("data/btagSF/UL/DeepJet_106XUL16postVFPSF_v2.csv"),wp)
    elif year == '2016APV': SFevaluatorBtag = BTagScaleFactor(topeft_path("data/btagSF/UL/wp_deepJet_106XUL16preVFP_v2.csv"),wp)
    elif year == '2017': SFevaluatorBtag = BTagScaleFactor(topeft_path("data/btagSF/UL/wp_deepJet_106XUL17_v3.csv"),wp)
    elif year == '2018': SFevaluatorBtag = BTagScaleFactor(topeft_path("data/btagSF/UL/wp_deepJet_106XUL18_v2.csv"),wp)
    else: raise Exception(f"Error: Unknown year \"{year}\".")

    pt = jets.pt
    SF = SFevaluatorBtag.eval('central',jets.hadronFlavour,np.abs(jets.eta),jets.pt)

    # Workaround: For UL16, use the SFs from the UL16APV for light flavor jets
    SFevaluatorBtag_UL16APV = BTagScaleFactor(topeft_path("data/btagSF/UL/wp_deepJet_106XUL16preVFP_v2.csv"),wp)
    if year == "2016":
        had_flavor = jets.hadronFlavour
        SF_UL16APV = SFevaluatorBtag_UL16APV.eval('central',jets.hadronFlavour,np.abs(jets.eta),jets.pt)
        SF = ak.where(had_flavor == 0,SF_UL16APV,SF)

    if syst == 'central':
        # If we are just getting the central, return here
        return (SF)
    else:
        # We are calculating up and down
        flavors = {
            0: ["light_corr", f"light_{year}"],
            4: ["bc_corr",f"bc_{year}"],
            5: ["bc_corr",f"bc_{year}"]
        }
        jets[f"btag_{syst}_up"] = SF
        jets[f"btag_{syst}_down"] = SF
        for f, f_syst in flavors.items():
            if syst in f_syst:
                # Workaround: For UL16, use the SFs from the UL16APV for light flavor jets
                if (f == 0) and (year == "2016"):
                    if f"{year}" in syst:
                        jets[f"btag_{syst}_up"] = np.where(
                            abs(jets.hadronFlavour) == f,
                            SFevaluatorBtag_UL16APV.eval("up_uncorrelated",jets.hadronFlavour,np.abs(jets.eta),pt,jets.btagDeepFlavB,True),
                            jets[f"btag_{syst}_up"]
                        )
                        jets[f"btag_{syst}_down"] = np.where(
                            abs(jets.hadronFlavour) == f,
                            SFevaluatorBtag_UL16APV.eval("down_uncorrelated", jets.hadronFlavour,np.abs(jets.eta),pt,jets.btagDeepFlavB,True),
                            jets[f"btag_{syst}_down"]
                        )
                    else:
                        jets[f"btag_{syst}_up"] = np.where(
                            abs(jets.hadronFlavour) == f,
                            SFevaluatorBtag_UL16APV.eval("up_correlated", jets.hadronFlavour,np.abs(jets.eta),pt,jets.btagDeepFlavB,True),
                            jets[f"btag_{syst}_up"]
                        )
                        jets[f"btag_{syst}_down"] = np.where(
                            abs(jets.hadronFlavour) == f,
                            SFevaluatorBtag_UL16APV.eval("down_correlated", jets.hadronFlavour,np.abs(jets.eta),pt,jets.btagDeepFlavB,True),
                            jets[f"btag_{syst}_down"]
                        )
                # Otherwise, proceed as usual
                else:
                    if f"{year}" in syst:
                        jets[f"btag_{syst}_up"] = np.where(
                            abs(jets.hadronFlavour) == f,
                            SFevaluatorBtag.eval("up_uncorrelated", jets.hadronFlavour,np.abs(jets.eta),pt,jets.btagDeepFlavB,True),
                            jets[f"btag_{syst}_up"]
                        )
                        jets[f"btag_{syst}_down"] = np.where(
                            abs(jets.hadronFlavour) == f,
                            SFevaluatorBtag.eval("down_uncorrelated", jets.hadronFlavour,np.abs(jets.eta),pt,jets.btagDeepFlavB,True),
                            jets[f"btag_{syst}_down"]
                        )
                    else:
                        jets[f"btag_{syst}_up"] = np.where(
                            abs(jets.hadronFlavour) == f,
                            SFevaluatorBtag.eval("up_correlated", jets.hadronFlavour,np.abs(jets.eta),pt,jets.btagDeepFlavB,True),
                            jets[f"btag_{syst}_up"]
                        )
                        jets[f"btag_{syst}_down"] = np.where(
                            abs(jets.hadronFlavour) == f,
                            SFevaluatorBtag.eval("down_correlated", jets.hadronFlavour,np.abs(jets.eta),pt,jets.btagDeepFlavB,True),
                            jets[f"btag_{syst}_down"]
                        )
    return ([jets[f"btag_{syst}_up"],jets[f"btag_{syst}_down"]])


def AttachPdfWeights(events):
    '''
        Return a list of PDF weights
        Should be 100 weights for NNPDF 3.1
    '''
    if events.LHEPdfWeight is None:
        raise Exception('LHEPdfWeight not found!')
    pdf_weight = ak.Array(events.LHEPdfWeight)
    #events['Pdf'] = ak.Array(events.nLHEPdfWeight) # FIXME not working

####### JEC
##############################################
# JER: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetResolution
# JES: https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC

def ApplyJetCorrections(year, corr_type):
    if year == '2016':
        jec_tag = '16_V7'
        jer_tag = 'Summer20UL16_JRV3'
    elif year == '2016APV':
        jec_tag = '16APV_V7'
        jer_tag = 'Summer20UL16APV_JRV3'
    elif year == '2017':
        jec_tag = '17_V5'
        jer_tag = 'Summer19UL17_JRV2'
    elif year == '2018':
        jec_tag = '18_V5'
        jer_tag = 'Summer19UL18_JRV2'
    else:
        raise Exception(f"Error: Unknown year \"{year}\".")
    extJEC = lookup_tools.extractor()
    extJEC.add_weight_sets([
        "* * " + topcoffea_path('data/JER/%s_MC_SF_AK4PFchs.jersf.txt' % jer_tag),
        "* * " + topcoffea_path('data/JER/%s_MC_PtResolution_AK4PFchs.jr.txt' % jer_tag),
        "* * " + topcoffea_path('data/JEC/Summer19UL%s_MC_L1FastJet_AK4PFchs.txt' % jec_tag),
        "* * " + topcoffea_path('data/JEC/Summer19UL%s_MC_L2Relative_AK4PFchs.txt' % jec_tag),
        "* * " + topcoffea_path('data/JEC/Quad_Summer19UL%s_MC_UncertaintySources_AK4PFchs.junc.txt' % jec_tag)
    ])
    jec_types = [
        'FlavorQCD', 'FlavorPureBottom', 'FlavorPureQuark', 'FlavorPureGluon', 'FlavorPureCharm',
        'BBEC1', 'Absolute', 'RelativeBal', 'RelativeSample'
    ]
    jec_regroup = ["Quad_Summer19UL%s_MC_UncertaintySources_AK4PFchs_%s" % (jec_tag,jec_type) for jec_type in jec_types]
    jec_names = [
        "%s_MC_SF_AK4PFchs" % jer_tag,
        "%s_MC_PtResolution_AK4PFchs" % jer_tag,
        "Summer19UL%s_MC_L1FastJet_AK4PFchs" % jec_tag,
        "Summer19UL%s_MC_L2Relative_AK4PFchs" % jec_tag
    ]
    jec_names.extend(jec_regroup)
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
    if corr_type == 'met':
        return CorrectedMETFactory(name_map)
    return CorrectedJetsFactory(name_map, jec_stack)

def ApplyJetSystematics(year,cleanedJets,syst_var):
    if (syst_var == f'JER_{year}Up'):
        return cleanedJets.JER.up
    elif (syst_var == f'JER_{year}Down'):
        return cleanedJets.JER.down
    elif (syst_var == 'JESUp'):
        return cleanedJets.JES_jes.up
    elif (syst_var == 'JESDown'):
        return cleanedJets.JES_jes.down
    elif (syst_var == 'nominal'):
        return cleanedJets
    elif (syst_var in ['nominal','MuonESUp','MuonESDown']):
        return cleanedJets
    elif ('JES_FlavorQCD' in syst_var in syst_var):# and (('Up' in syst_var and syst_var.replace('Up', '') in cleanedJets.fields) or ('Down' in syst_var and syst_var.replace('Down', '') in cleanedJets.fields))):
        # Overwrite FlavorQCD with the proper jet flavor uncertainty
        bmask = np.array(ak.flatten(abs(cleanedJets.partonFlavour)==5))
        cmask = abs(cleanedJets.partonFlavour)==4
        cmask = np.array(ak.flatten(cmask))
        qmask = abs(cleanedJets.partonFlavour)<=3
        qmask = np.array(ak.flatten(qmask))
        gmask = abs(cleanedJets.partonFlavour)==21
        gmask = np.array(ak.flatten(gmask))
        corrections = np.array(np.zeros_like(ak.flatten(cleanedJets.JES_FlavorQCD.up.pt)))
        if 'Up' in syst_var:
            corrections[bmask] = corrections[bmask] + np.array(ak.flatten(cleanedJets.JES_FlavorQCD.up.pt))[bmask]
            corrections[cmask] = corrections[cmask] + np.array(ak.flatten(cleanedJets.JES_FlavorQCD.up.pt))[cmask]
            corrections[qmask] = corrections[qmask] + np.array(ak.flatten(cleanedJets.JES_FlavorQCD.up.pt))[qmask]
            corrections[gmask] = corrections[gmask] + np.array(ak.flatten(cleanedJets.JES_FlavorQCD.up.pt))[gmask]
            corrections = ak.unflatten(corrections, ak.num(cleanedJets.JES_FlavorQCD.up.pt))
            cleanedJets['JES_FlavorQCD']['up']['pt'] = corrections
            return cleanedJets.JES_FlavorQCD.up
        if 'Down' in syst_var:
            corrections[bmask] = corrections[bmask] + np.array(ak.flatten(cleanedJets.JES_FlavorQCD.down.pt))[bmask]
            corrections[cmask] = corrections[cmask] + np.array(ak.flatten(cleanedJets.JES_FlavorQCD.down.pt))[cmask]
            corrections[qmask] = corrections[qmask] + np.array(ak.flatten(cleanedJets.JES_FlavorQCD.down.pt))[qmask]
            corrections[gmask] = corrections[gmask] + np.array(ak.flatten(cleanedJets.JES_FlavorQCD.down.pt))[gmask]
            corrections = ak.unflatten(corrections, ak.num(cleanedJets.JES_FlavorQCD.down.pt))
            cleanedJets['JES_FlavorQCD']['down']['pt'] = corrections
            return cleanedJets.JES_FlavorQCD.down
    # Save `2016APV` as `2016APV` but look up `2016` corrections (no separate APV corrections available)
    elif ('Up' in syst_var and syst_var.replace('Up', '').replace('APV', '') in cleanedJets.fields):
        return cleanedJets[syst_var.replace('Up', '').replace('APV', '')].up
    elif ('Down' in syst_var and syst_var.replace('Down', '').replace('APV', '') in cleanedJets.fields):
        return cleanedJets[syst_var.replace('Down', '').replace('APV', '')].down
    else:
        raise Exception(f"Error: Unknown variation \"{syst_var}\".")

###### Muon Rochester corrections
################################################################
# https://gitlab.cern.ch/akhukhun/roccor
# https://github.com/CoffeaTeam/coffea/blob/master/coffea/lookup_tools/rochester_lookup.py
def ApplyRochesterCorrections(year, mu, is_data):
    if year == '2016':
        rochester_data = txt_converters.convert_rochester_file(topcoffea_path("data/MuonScale/RoccoR2016bUL.txt"), loaduncs=True)
    elif year == '2016APV':
        rochester_data = txt_converters.convert_rochester_file(topcoffea_path("data/MuonScale/RoccoR2016aUL.txt"), loaduncs=True)
    elif year == '2017':
        rochester_data = txt_converters.convert_rochester_file(topcoffea_path("data/MuonScale/RoccoR2017UL.txt"), loaduncs=True)
    elif year == '2018':
        rochester_data = txt_converters.convert_rochester_file(topcoffea_path("data/MuonScale/RoccoR2018UL.txt"), loaduncs=True)
    rochester = rochester_lookup.rochester_lookup(rochester_data)
    if not is_data:
        hasgen = ~np.isnan(ak.fill_none(mu.matched_gen.pt, np.nan))
        mc_rand = np.random.rand(*ak.to_numpy(ak.flatten(mu.pt)).shape)
        mc_rand = ak.unflatten(mc_rand, ak.num(mu.pt, axis=1))
        corrections = np.array(ak.flatten(ak.ones_like(mu.pt)))
        mc_kspread = rochester.kSpreadMC(
            mu.charge[hasgen],mu.pt[hasgen],
            mu.eta[hasgen],
            mu.phi[hasgen],
            mu.matched_gen.pt[hasgen]
        )
        mc_ksmear = rochester.kSmearMC(
            mu.charge[~hasgen],
            mu.pt[~hasgen],
            mu.eta[~hasgen],
            mu.phi[~hasgen],
            mu.nTrackerLayers[~hasgen],
            mc_rand[~hasgen]
        )
        hasgen_flat = np.array(ak.flatten(hasgen))
        corrections[hasgen_flat] = np.array(ak.flatten(mc_kspread))
        corrections[~hasgen_flat] = np.array(ak.flatten(mc_ksmear))
        corrections = ak.unflatten(corrections, ak.num(mu.pt, axis=1))
    else:
        corrections = rochester.kScaleDT(mu.charge, mu.pt, mu.eta, mu.phi)
    return (mu.pt * corrections)

###### Trigger SFs
################################################################

#### Functions needed
StackOverUnderflow = lambda v : [sum(v[0:2])] + v[2:-2] + [sum(v[-2:])]

_coverage1sd = scipy.stats.norm.cdf(1) - scipy.stats.norm.cdf(-1)
def clopper_pearson_interval(num, denom, coverage=_coverage1sd):
    """Compute Clopper-Pearson coverage interval for a binomial distribution

    Parameters
    ----------
        num : numpy.ndarray
            Numerator, or number of successes, vectorized
        denom : numpy.ndarray
            Denominator or number of trials, vectorized
        coverage : float, optional
            Central coverage interval, defaults to 68%

    c.f. http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    """
    if np.any(num > denom):
        raise ValueError(
            "Found numerator larger than denominator while calculating binomial uncertainty"
        )
    lo = scipy.stats.beta.ppf((1 - coverage) / 2, num, denom - num + 1)
    hi = scipy.stats.beta.ppf((1 + coverage) / 2, num + 1, denom - num)
    interval = np.array([lo, hi])
    interval[:, num == 0.0] = 0.0
    interval[1, num == denom] = 1.0
    return interval

def GetClopperPearsonInterval(hnum, hden):
    ''' Compute Clopper-Pearson interval from numerator and denominator histograms '''
    num = list(hnum.values(flow=True)[()])
    den = list(hden.values(flow=True)[()])
    if isinstance(num, list) and isinstance(num[0], np.ndarray):
        for i in range(len(num)):
            num[i] = np.array(StackOverUnderflow(list(num[i])), dtype=float)
            den[i] = np.array(StackOverUnderflow(list(den[i])), dtype=float)
        den = StackOverUnderflow(den)
        num = StackOverUnderflow(num)
    else:
        num = np.array(StackOverUnderflow(num), dtype=float)
        den = np.array(StackOverUnderflow(den), dtype=float)
    num = np.array(num)
    den = np.array(den)
    num[num>den] = den[num > den]
    down, up = clopper_pearson_interval(num, den)
    ratio = np.array(num, dtype=float) / den
    return [ratio, down, up]

def GetEff(num, den):
    ''' Compute efficiency values from numerator and denominator histograms '''
    ratio, down, up = GetClopperPearsonInterval(num, den)
    axis = num.axes[0].name
    bins = num.axes[axis].edges
    x    = num.axes[axis].centers
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
    ratio = num / den
    uncUp = ratio * np.sqrt(numUp * numUp + denUp * denUp)
    uncDo = ratio * np.sqrt(numDo * numDo + denDo * denDo)
    return ratio, -uncDo, uncUp

######  Scale Factors

def LoadTriggerSF(year, ch='2l', flav='em'):
    pathToTriggerSF = topeft_path('data/triggerSF/triggerSF_%s.pkl.gz' % year)
    with gzip.open(pathToTriggerSF) as fin:
        hin = pickle.load(fin)
    if ch == '2l':
        axisY = 'l1pt'
    else:
        axisY = 'l0eta'
    h = hin[ch][flav]
    ratio, do, up = GetSFfromCountsHisto(h['hmn'], h['hmd'], h['hdn'], h['hdd'])
    ratio[np.isnan(ratio)] = 1.0
    do[np.isnan(do)] = 0.0
    up[np.isnan(up)] = 0.0
    GetTrig   = lookup_tools.dense_lookup.dense_lookup(ratio, [h['hmn'].axes['l0pt'].edges, h['hmn'].axes[axisY].edges])
    GetTrigUp = lookup_tools.dense_lookup.dense_lookup(up   , [h['hmn'].axes['l0pt'].edges, h['hmn'].axes[axisY].edges])
    GetTrigDo = lookup_tools.dense_lookup.dense_lookup(do   , [h['hmn'].axes['l0pt'].edges, h['hmn'].axes[axisY].edges])
    return [GetTrig, GetTrigDo, GetTrigUp]

def GetTriggerSF(year, events, lep0, lep1):
    ls = []
    for syst in [0,1]:
        #2l
        SF_ee = np.where((events.is2l & events.is_ee), LoadTriggerSF(year,ch='2l',flav='ee')[syst](lep0.pt,lep1.pt), 1.0)
        SF_em = np.where((events.is2l & events.is_em), LoadTriggerSF(year,ch='2l',flav='em')[syst](lep0.pt,lep1.pt), 1.0)
        SF_mm = np.where((events.is2l & events.is_mm), LoadTriggerSF(year,ch='2l',flav='mm')[syst](lep0.pt,lep1.pt), 1.0)
        #3l
        '''
        SF_eee=np.where((events.is3l & events.is_eee),LoadTriggerSF(year,ch='3l',flav='eee')[syst](lep0.pt,lep0.eta),1.0)
        SF_eem=np.where((events.is3l & events.is_eem),LoadTriggerSF(year,ch='3l',flav='eem')[syst](lep0.pt,lep0.eta),1.0)
        SF_emm=np.where((events.is3l & events.is_emm),LoadTriggerSF(year,ch='3l',flav='emm')[syst](lep0.pt,lep0.eta),1.0)
        SF_mmm=np.where((events.is3l & events.is_mmm),LoadTriggerSF(year,ch='3l',flav='mmm')[syst](lep0.pt,lep0.eta),1.0)
        ls.append(SF_ee*SF_em*SF_mm*SF_eee*SF_eem*SF_emm*SF_mmm)
        '''
        ls.append(SF_ee * SF_em * SF_mm)
    ls[1] = np.where(ls[1] == 1.0, 0.0, ls[1]) # stat unc. down
    events['trigger_sf'] = ls[0] # nominal
    events['trigger_sfDown'] = ls[0] - np.sqrt(ls[1] * ls[1] + ls[0]*0.02*ls[0]*0.02)
    events['trigger_sfUp'] = ls[0] + np.sqrt(ls[1] * ls[1] + ls[0]*0.02*ls[0]*0.02)
