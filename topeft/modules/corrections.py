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
import correctionlib
import json
from coffea.jetmet_tools import CorrectedMETFactory
### workaround while waiting the correcion-lib integration will be provided in the coffea package
from topcoffea.modules.CorrectedJetsFactory import CorrectedJetsFactory
from topcoffea.modules.JECStack import JECStack
from coffea.btag_tools.btagscalefactor import BTagScaleFactor
from coffea.lookup_tools import txt_converters, rochester_lookup

from topcoffea.modules.get_param_from_jsons import GetParam
get_tc_param = GetParam(topcoffea_path("params/params.json"))
get_te_param = GetParam(topeft_path("params/params.json"))

from collections import OrderedDict

basepathFromTTH = 'data/fromTTH/'

###### Lepton scale factors
################################################################
extLepSF = lookup_tools.extractor()

clib_year_map = {
    "2016APV": "2016preVFP_UL",
    "2016preVFP": "2016preVFP_UL",
    "2016": "2016postVFP_UL",
    "2017": "2017_UL",
    "2018": "2018_UL",
    "2022": "2022_Summer22",
    "2022EE": "2022_Summer22EE",
    "2023": "2023_Summer23",
    "2023BPix": "2023_Summer23BPix",
}

egm_tag_map = {
    "2016preVFP_UL": "2016preVFP",
    "2016postVFP_UL": "2016postVFP",
    "2017_UL": "2017",
    "2018_UL": "2018",
    "2022_Summer22": "2022Re-recoBCD",
    "2022_Summer22EE": "2022Re-recoE+PromptFG",
    "2023_Summer23": "2023PromptC",
    "2023_Summer23BPix": "2023PromptD",
}

egm_et_map = {
    "2022_Summer22": "2022preEE",
    "2022_Summer22EE": "2022postEE",
    "2023_Summer23": "2023preBPIX",
    "2023_Summer23BPix": "2023postBPIX",
}

egm_pt_bins = {
    "Run2": OrderedDict([
        ("RecoBelow20", [10, 20]),
        ("RecoAbove20", [20, 1000])
    ]),
    "Run3": OrderedDict([
        ("RecoBelow20", [10, 20]),
        ("Reco20to75", [20, 75]),
        ("RecoAbove75", [75, 1000])
    ]),
}

#Leftover for the old JERC implementation
jerc_tag_map = {
    '2016': [
        '16_V7',
        'Summer20UL16_JRV3',
    ],
    '2016APV': [
        '16APV_V7',
        'Summer20UL16APV_JRV3',
    ],
    '2017': [
        '17_V5',
        'Summer19UL17_JRV2',
    ],
    '2018': [
        '18_V5',
        'Summer19UL18_JRV2',
    ],
    "2022": [],
    "2022EE": [],
    "2023": [],
    "2023BPix": [],
}

#JERC dictionary for various keys
jerc_dict = {
    "2016": {
        "jec_mc"  : "Summer19UL16_V7_MC",
        "jec_data": "Summer19UL16_RunFGH_V7_DATA",
        "jec_levels": [
            "L1FastJet",
            "L2Relative",
        ],
        "jer"     : "Summer20UL16_JRV3_MC",
        "junc"    : [
            'FlavorQCD', 'FlavorPureBottom', 'FlavorPureQuark', 'FlavorPureGluon', 'FlavorPureCharm',
            'Regrouped_BBEC1', 'Regrouped_Absolute', 'Regrouped_RelativeBal', 'RelativeSample'
        ]
    },
    "2016APV": {
        "jec_mc": "Summer19UL16APV_V7_MC",
        "jec_data": {
            "B": "Summer19UL16APV_RunBCD_V7_DATA",
            "C": "Summer19UL16APV_RunBCD_V7_DATA",
            "D": "Summer19UL16APV_RunBCD_V7_DATA",
            "E": "Summer19UL16APV_RunEF_V7_DATA",
            "F": "Summer19UL16APV_RunEF_V7_DATA",
        },
        "jec_levels": [
            "L1FastJet",
            "L2Relative",
        ],
        "jer": "Summer20UL16APV_JRV3_MC",
        "junc"    : [
            'FlavorQCD', 'FlavorPureBottom', 'FlavorPureQuark', 'FlavorPureGluon', 'FlavorPureCharm',
            'Regrouped_BBEC1', 'Regrouped_Absolute', 'Regrouped_RelativeBal', 'RelativeSample'
        ]
    },
    "2017": {
        "jec_mc": "Summer19UL17_V5_MC",
        "jec_data": {
            "B": "Summer19UL17_RunB_V5_DATA",
            "C": "Summer19UL17_RunC_V5_DATA",
            "D": "Summer19UL17_RunD_V5_DATA",
            "E": "Summer19UL17_RunE_V5_DATA",
            "F": "Summer19UL17_RunF_V5_DATA",
        },
        "jec_levels": [
            "L1FastJet",
            "L2Relative",
        ],
        "jer": "Summer19UL17_JRV2_MC",
        "junc"    : [
            'FlavorQCD', 'FlavorPureBottom', 'FlavorPureQuark', 'FlavorPureGluon', 'FlavorPureCharm',
            'Regrouped_BBEC1', 'Regrouped_Absolute', 'Regrouped_RelativeBal', 'RelativeSample'
        ]
    },
    "2018": {
        "jec_mc": "Summer19UL18_V5_MC",
        "jec_data": {
            "A": "Summer19UL18_RunA_V5_DATA",
            "B": "Summer19UL18_RunB_V5_DATA",
            "C": "Summer19UL18_RunC_V5_DATA",
            "D": "Summer19UL18_RunD_V5_DATA",
        },
        "jec_levels": [
            "L1FastJet",
            "L2Relative",
        ],
        "jer": "Summer19UL18_JRV2_MC",
        "junc"    : [
            'FlavorQCD', 'FlavorPureBottom', 'FlavorPureQuark', 'FlavorPureGluon', 'FlavorPureCharm',
            'Regrouped_BBEC1', 'Regrouped_Absolute', 'Regrouped_RelativeBal', 'RelativeSample'
        ]

    },
    "2022": {
        "jec_mc"  : "Summer22_22Sep2023_V2_MC",
        "jec_data": "Summer22_22Sep2023_RunCD_V2_DATA",
        "jec_levels": [
            "L1FastJet",
            "L2Relative",
            "L3Absolute",
            "L2L3Residual",
        ],
        "jer"     : "Summer22_22Sep2023_JRV1_MC",
        "junc"    : [
            "AbsoluteMPFBias","AbsoluteScale","FlavorQCD","Fragmentation","PileUpDataMC",
            "PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF","PileUpPtRef",
            "RelativeFSR","RelativeJERHF","RelativePtBB","RelativePtHF","RelativeBal",
            "SinglePionECAL","SinglePionHCAL",
            "AbsoluteStat","RelativeJEREC1","RelativeJEREC2","RelativePtEC1","RelativePtEC2",
            "TimePtEta","RelativeSample","RelativeStatEC","RelativeStatFSR","RelativeStatHF",
            "Total",
        ]
    },
    "2022EE": {
        "jec_mc": "Summer22EE_22Sep2023_V2_MC",
        "jec_data": {
            "E": "Summer22EE_22Sep2023_RunE_V2_DATA",
            "F": "Summer22EE_22Sep2023_RunF_V2_DATA",
            "G": "Summer22EE_22Sep2023_RunG_V2_DATA",
        },
        "jec_levels": [
            "L1FastJet",
            "L2Relative",
            "L3Absolute",
            "L2L3Residual",
        ],
        "jer": "Summer22EE_22Sep2023_JRV1_MC",
        "junc"    : [
            "AbsoluteMPFBias","AbsoluteScale","FlavorQCD","Fragmentation","PileUpDataMC",
            "PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF","PileUpPtRef",
            "RelativeFSR","RelativeJERHF","RelativePtBB","RelativePtHF","RelativeBal",
            "SinglePionECAL","SinglePionHCAL",
            "AbsoluteStat","RelativeJEREC1","RelativeJEREC2","RelativePtEC1","RelativePtEC2",
            "TimePtEta","RelativeSample","RelativeStatEC","RelativeStatFSR","RelativeStatHF",
            "Total",
        ]
    },
    "2023": {
        "jec_mc": "Summer23Prompt23_V1_MC",
        "jec_data": {
            "C1": "Summer23Prompt23_RunCv123_V1_DATA",
            "C2": "Summer23Prompt23_RunCv123_V1_DATA",
            "C3": "Summer23Prompt23_RunCv123_V1_DATA",
            "C4": "Summer23Prompt23_RunCv4_V1_DATA",
        },
        "jec_levels": [
            "L1FastJet",
            "L2Relative",
            "L3Absolute",
            "L2L3Residual",
        ],
        "jer": "Summer23Prompt23_RunCv1234_JRV1_MC",
        "junc"    : [
            "AbsoluteMPFBias","AbsoluteScale","FlavorQCD","Fragmentation","PileUpDataMC",
            "PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF","PileUpPtRef",
            "RelativeFSR","RelativeJERHF","RelativePtBB","RelativePtHF","RelativeBal",
            "SinglePionECAL","SinglePionHCAL",
            "AbsoluteStat","RelativeJEREC1","RelativeJEREC2","RelativePtEC1","RelativePtEC2",
            "TimePtEta","RelativeSample","RelativeStatEC","RelativeStatFSR","RelativeStatHF",
            "Total",
        ]
    },
    "2023BPix": {
        "jec_mc"  : "Summer23BPixPrompt23_V1_MC",
        "jec_data": "Summer23BPixPrompt23_RunD_V1_DATA",
        "jec_levels": [
            "L1FastJet",
            "L2Relative",
            "L3Absolute",
            "L2L3Residual",
        ],
        "jer"     : "Summer23BPixPrompt23_RunD_JRV1_MC",
        "junc"    : [
            "AbsoluteMPFBias","AbsoluteScale","FlavorQCD","Fragmentation","PileUpDataMC",
            "PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF","PileUpPtRef",
            "RelativeFSR","RelativeJERHF","RelativePtBB","RelativePtHF","RelativeBal",
            "SinglePionECAL","SinglePionHCAL",
            "AbsoluteStat","RelativeJEREC1","RelativeJEREC2","RelativePtEC1","RelativePtEC2",
            "TimePtEta","RelativeSample","RelativeStatEC","RelativeStatFSR","RelativeStatHF",
            "Total",
        ]
    }
}

jet_veto_dict = {
    "2016APV": "Summer19UL16_V1",
    "2016": "Summer19UL16_V1",
    "2017": "Summer19UL17_V1",
    "2018": "Summer19UL18_V1",
    "2022": "Summer22_23Sep2023_RunCD_V1",
    "2022EE": "Summer22EE_23Sep2023_RunEFG_V1",
    "2023": "Summer23Prompt23_RunC_V1",
    "2023BPix": "Summer23BPixPrompt23_RunD_V1"
}

with open(topeft_path('modules/jerc_dict.json'), 'r') as f:
    jerc_dict = json.load(f)

def get_jerc_keys(year, isdata, era=None):
    # Jet Algorithm
    if year.startswith("202"):
        jet_algo = 'AK4PFPuppi'
    else:
        jet_algo = 'AK4PFchs'

    #jec levels
    jec_levels = jerc_dict[year]['jec_levels']

    # jerc keys and junc types
    if not isdata:
        jec_key    = jerc_dict[year]['jec_mc']
        jer_key    = jerc_dict[year]['jer']
        junc_types = jerc_dict[year]['junc']
    else:
        #if year in ['2016','2022','2023BPix']:
        #    jec_key = jerc_dict[year]['jec_data']
        #else:
        jec_key = jerc_dict[year]['jec_data'][era]
        jer_key     = None
        junc_types  = None

    return jet_algo, jec_key, jec_levels, jer_key, junc_types

def get_corr_inputs(objs, corr_obj, name_map):
    """
    Helper function for getting values of input variables
    given a dictionary and a correction object.
    """
    input_values = [ak.flatten(objs[inp.name]) for inp in corr_obj.inputs if inp.name != "systematic"]
    return input_values

# New UL Lepton SFs
# Muon: reco ##clib ready
extLepSF.add_weight_sets(["MuonRecoSF_2018 NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2018_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2018_er NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2018_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2017 NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2017_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2017_er NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2017_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2016 NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2016postVFP_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2016_er NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2016postVFP_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2016APV NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2016preVFP_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2016APV_er NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2016preVFP_UL_trackerMuon.json')])
# Muon: loose POG ##clib ready
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
# Muon: ISO + IP (Barbara) ##personal
extLepSF.add_weight_sets(["MuonIsoSF_2018 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2018_iso_EGM2D.root')])
extLepSF.add_weight_sets(["MuonIsoSF_2018_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2018_iso_EGM2D.root')])
extLepSF.add_weight_sets(["MuonIsoSF_2017_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2017_iso_EGM2D.root')])
extLepSF.add_weight_sets(["MuonIsoSF_2017 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2017_iso_EGM2D.root')])
extLepSF.add_weight_sets(["MuonIsoSF_2016_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2016_iso_EGM2D.root')])
extLepSF.add_weight_sets(["MuonIsoSF_2016 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2016_iso_EGM2D.root')])
extLepSF.add_weight_sets(["MuonIsoSF_2016APV_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2016APV_iso_EGM2D.root')])
extLepSF.add_weight_sets(["MuonIsoSF_2016APV EGamma_SF2D %s" % topcoffea_path('data/leptonSF/muon/egammaEffi2016APV_iso_EGM2D.root')])
# Muon: looseMVA&tight (Barbara) ##personal
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
# Elec: looseID (Barbara) ##personal
extLepSF.add_weight_sets(["ElecLooseSF_2018 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_recoToloose_EGM2D.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2018_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_recoToloose_EGM2D.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2017_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_recoToloose_EGM2D.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2017 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_recoToloose_EGM2D.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2016_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_recoToloose_EGM2D.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2016 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_recoToloose_EGM2D.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2016APV_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_recoToloose_EGM2D.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2016APV EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_recoToloose_EGM2D.root')])
# Elec: ISO + IP (Barbara) ##personal
extLepSF.add_weight_sets(["ElecIsoSF_2018 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_iso_EGM2D.root')])
extLepSF.add_weight_sets(["ElecIsoSF_2018_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_iso_EGM2D.root')])
extLepSF.add_weight_sets(["ElecIsoSF_2017_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_iso_EGM2D.root')])
extLepSF.add_weight_sets(["ElecIsoSF_2017 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_iso_EGM2D.root')])
extLepSF.add_weight_sets(["ElecIsoSF_2016_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_iso_EGM2D.root')])
extLepSF.add_weight_sets(["ElecIsoSF_2016 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_iso_EGM2D.root')])
extLepSF.add_weight_sets(["ElecIsoSF_2016APV_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_iso_EGM2D.root')])
extLepSF.add_weight_sets(["ElecIsoSF_2016APV EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_iso_EGM2D.root')])
# Elec: looseMVA&tight (Barbara) ##personal
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

extLepSF.add_weight_sets(["TauSF_2016APV_VTight_down Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2016_preVFPVTight.json')])
extLepSF.add_weight_sets(["TauSF_2016_VTight_down Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2016_postVFPVTight.json')])
extLepSF.add_weight_sets(["TauSF_2017_VTight_down Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2017VTight.json')])
extLepSF.add_weight_sets(["TauSF_2018_VTight_down Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2018VTight.json')])

extLepSF.add_weight_sets(["TauSF_2016APV_VTight_up Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2016_preVFPVTight.json')])
extLepSF.add_weight_sets(["TauSF_2016_VTight_up Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2016_postVFPVTight.json')])
extLepSF.add_weight_sets(["TauSF_2017_VTight_up Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2017VTight.json')])
extLepSF.add_weight_sets(["TauSF_2018_VTight_up Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2018VTight.json')])

extLepSF.add_weight_sets(["TauSF_2016APV_VTight Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2016_preVFPVTight.json')])
extLepSF.add_weight_sets(["TauSF_2016_VTight Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2016_postVFPVTight.json')])
extLepSF.add_weight_sets(["TauSF_2017_VTight Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2017VTight.json')])
extLepSF.add_weight_sets(["TauSF_2018_VTight Tau_SF/dm_pt_down %s"%topcoffea_path('data/TauSF/TauSFUL2018VTight.json')])

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

extLepSF.add_weight_sets(["TauFES_2016APV Tau_FES/eta_dm_value %s"%topcoffea_path('data/TauSF/TauFESUL2016_preVFP.json')])
extLepSF.add_weight_sets(["TauFES_2016 Tau_FES/eta_dm_value %s"%topcoffea_path('data/TauSF/TauFESUL2016_postVFP.json')])
extLepSF.add_weight_sets(["TauFES_2017 Tau_FES/eta_dm_value %s"%topcoffea_path('data/TauSF/TauFESUL2017.json')])
extLepSF.add_weight_sets(["TauFES_2018 Tau_FES/eta_dm_value %s"%topcoffea_path('data/TauSF/TauFESUL2018.json')])

extLepSF.add_weight_sets(["TauFES_2016APV_up Tau_FES/eta_dm_up %s"%topcoffea_path('data/TauSF/TauFESUL2016_preVFP.json')])
extLepSF.add_weight_sets(["TauFES_2016_up Tau_FES/eta_dm_up %s"%topcoffea_path('data/TauSF/TauFESUL2016_postVFP.json')])
extLepSF.add_weight_sets(["TauFES_2017_up Tau_FES/eta_dm_up %s"%topcoffea_path('data/TauSF/TauFESUL2017.json')])
extLepSF.add_weight_sets(["TauFES_2018_up Tau_FES/eta_dm_up %s"%topcoffea_path('data/TauSF/TauFESUL2018.json')])

extLepSF.add_weight_sets(["TauFES_2016APV_down Tau_FES/eta_dm_down %s"%topcoffea_path('data/TauSF/TauFESUL2016_preVFP.json')])
extLepSF.add_weight_sets(["TauFES_2016_down Tau_FES/eta_dm_down %s"%topcoffea_path('data/TauSF/TauFESUL2016_postVFP.json')])
extLepSF.add_weight_sets(["TauFES_2017_down Tau_FES/eta_dm_down %s"%topcoffea_path('data/TauSF/TauFESUL2017.json')])
extLepSF.add_weight_sets(["TauFES_2018_down Tau_FES/eta_dm_down %s"%topcoffea_path('data/TauSF/TauFESUL2018.json')])

extLepSF.add_weight_sets(["TauFakeSF_2016APV TauFake/pt_value %s"%topcoffea_path('data/TauSF/TauFakeSF_2016APV.json')])
extLepSF.add_weight_sets(["TauFakeSF_2016 TauFake/pt_value %s"%topcoffea_path('data/TauSF/TauFakeSF_2016.json')])
extLepSF.add_weight_sets(["TauFakeSF_2017 TauFake/pt_value %s"%topcoffea_path('data/TauSF/TauFakeSF_2017.json')])
extLepSF.add_weight_sets(["TauFakeSF_2018 TauFake/pt_value %s"%topcoffea_path('data/TauSF/TauFakeSF_2018.json')])

extLepSF.add_weight_sets(["TauFakeSF_2016APV_up TauFake/pt_up %s"%topcoffea_path('data/TauSF/TauFakeSF_2016APV.json')])
extLepSF.add_weight_sets(["TauFakeSF_2016_up TauFake/pt_up %s"%topcoffea_path('data/TauSF/TauFakeSF_2016.json')])
extLepSF.add_weight_sets(["TauFakeSF_2017_up TauFake/pt_up %s"%topcoffea_path('data/TauSF/TauFakeSF_2017.json')])
extLepSF.add_weight_sets(["TauFakeSF_2018_up TauFake/pt_up %s"%topcoffea_path('data/TauSF/TauFakeSF_2018.json')])

extLepSF.add_weight_sets(["TauFakeSF_2016APV_down TauFake/pt_down %s"%topcoffea_path('data/TauSF/TauFakeSF_2016APV.json')])
extLepSF.add_weight_sets(["TauFakeSF_2016_down TauFake/pt_down %s"%topcoffea_path('data/TauSF/TauFakeSF_2016.json')])
extLepSF.add_weight_sets(["TauFakeSF_2017_down TauFake/pt_down %s"%topcoffea_path('data/TauSF/TauFakeSF_2017.json')])
extLepSF.add_weight_sets(["TauFakeSF_2018_down TauFake/pt_down %s"%topcoffea_path('data/TauSF/TauFakeSF_2018.json')])

extLepSF.add_weight_sets(["TauFake_2016 Tau_SF/eta_gen_value %s"%topcoffea_path('data/TauSF/TauFakeUL2016.json')])
extLepSF.add_weight_sets(["TauFake_2017 Tau_SF/eta_gen_value %s"%topcoffea_path('data/TauSF/TauFakeUL2017.json')])
extLepSF.add_weight_sets(["TauFake_2018 Tau_SF/eta_gen_value %s"%topcoffea_path('data/TauSF/TauFakeUL2018.json')])

extLepSF.add_weight_sets(["Tau_muonFakeSF_2018 TauSF/eta_value %s"%topcoffea_path('data/TauSF/TauSF_muonfake_eta_UL2018.json')])
extLepSF.add_weight_sets(["Tau_muonFakeSF_2017 TauSF/eta_value %s"%topcoffea_path('data/TauSF/TauSF_muonfake_eta_UL2017.json')])
extLepSF.add_weight_sets(["Tau_muonFakeSF_2016 TauSF/eta_value %s"%topcoffea_path('data/TauSF/TauSF_muonfake_eta_UL2016_postVFP.json')])
extLepSF.add_weight_sets(["Tau_muonFakeSF_2016APV TauSF/eta_value %s"%topcoffea_path('data/TauSF/TauSF_muonfake_eta_UL2016_preVFP.json')])

extLepSF.add_weight_sets(["Tau_elecFakeSF_2018 TauSF/eta_value %s"%topcoffea_path('data/TauSF/TauSF_elecfake_eta_UL2018.json')])
extLepSF.add_weight_sets(["Tau_elecFakeSF_2017 TauSF/eta_value %s"%topcoffea_path('data/TauSF/TauSF_elecfake_eta_UL2017.json')])
extLepSF.add_weight_sets(["Tau_elecFakeSF_2016 TauSF/eta_value %s"%topcoffea_path('data/TauSF/TauSF_elecfake_eta_UL2016_postVFP.json')])
extLepSF.add_weight_sets(["Tau_elecFakeSF_2016APV TauSF/eta_value %s"%topcoffea_path('data/TauSF/TauSF_elecfake_eta_UL2016_preVFP.json')])

extLepSF.add_weight_sets(["Tau_muonFakeSF_2018_up TauSF/eta_up %s"%topcoffea_path('data/TauSF/TauSF_muonfake_eta_UL2018.json')])
extLepSF.add_weight_sets(["Tau_muonFakeSF_2017_up TauSF/eta_up %s"%topcoffea_path('data/TauSF/TauSF_muonfake_eta_UL2017.json')])
extLepSF.add_weight_sets(["Tau_muonFakeSF_2016_up TauSF/eta_up %s"%topcoffea_path('data/TauSF/TauSF_muonfake_eta_UL2016_postVFP.json')])
extLepSF.add_weight_sets(["Tau_muonFakeSF_2016APV_up TauSF/eta_up %s"%topcoffea_path('data/TauSF/TauSF_muonfake_eta_UL2016_preVFP.json')])

extLepSF.add_weight_sets(["Tau_elecFakeSF_2018_up TauSF/eta_up %s"%topcoffea_path('data/TauSF/TauSF_elecfake_eta_UL2018.json')])
extLepSF.add_weight_sets(["Tau_elecFakeSF_2017_up TauSF/eta_up %s"%topcoffea_path('data/TauSF/TauSF_elecfake_eta_UL2017.json')])
extLepSF.add_weight_sets(["Tau_elecFakeSF_2016_up TauSF/eta_up %s"%topcoffea_path('data/TauSF/TauSF_elecfake_eta_UL2016_postVFP.json')])
extLepSF.add_weight_sets(["Tau_elecFakeSF_2016APV_up TauSF/eta_up %s"%topcoffea_path('data/TauSF/TauSF_elecfake_eta_UL2016_preVFP.json')])

extLepSF.add_weight_sets(["Tau_muonFakeSF_2018_down TauSF/eta_down %s"%topcoffea_path('data/TauSF/TauSF_muonfake_eta_UL2018.json')])
extLepSF.add_weight_sets(["Tau_muonFakeSF_2017_down TauSF/eta_down %s"%topcoffea_path('data/TauSF/TauSF_muonfake_eta_UL2017.json')])
extLepSF.add_weight_sets(["Tau_muonFakeSF_2016_down TauSF/eta_down %s"%topcoffea_path('data/TauSF/TauSF_muonfake_eta_UL2016_postVFP.json')])
extLepSF.add_weight_sets(["Tau_muonFakeSF_2016APV_down TauSF/eta_down %s"%topcoffea_path('data/TauSF/TauSF_muonfake_eta_UL2016_preVFP.json')])

extLepSF.add_weight_sets(["Tau_elecFakeSF_2018_down TauSF/eta_down %s"%topcoffea_path('data/TauSF/TauSF_elecfake_eta_UL2018.json')])
extLepSF.add_weight_sets(["Tau_elecFakeSF_2017_down TauSF/eta_down %s"%topcoffea_path('data/TauSF/TauSF_elecfake_eta_UL2017.json')])
extLepSF.add_weight_sets(["Tau_elecFakeSF_2016_down TauSF/eta_down %s"%topcoffea_path('data/TauSF/TauSF_elecfake_eta_UL2016_postVFP.json')])
extLepSF.add_weight_sets(["Tau_elecFakeSF_2016APV_down TauSF/eta_down %s"%topcoffea_path('data/TauSF/TauSF_elecfake_eta_UL2016_preVFP.json')])

extLepSF.add_weight_sets(["TauSF_pt_2016APV TauSF/pt_value %s"%topcoffea_path('data/TauSF/TauSF_pt_UL2016_preVFP.json')])
extLepSF.add_weight_sets(["TauSF_pt_2016 TauSF/pt_value %s"%topcoffea_path('data/TauSF/TauSF_pt_UL2016_postVFP.json')])
extLepSF.add_weight_sets(["TauSF_pt_2017 TauSF/pt_value %s"%topcoffea_path('data/TauSF/TauSF_pt_UL2017.json')])
extLepSF.add_weight_sets(["TauSF_pt_2018 TauSF/pt_value %s"%topcoffea_path('data/TauSF/TauSF_pt_UL2018.json')])

extLepSF.add_weight_sets(["TauSF_dm_2016APV TauSF/dm_value %s"%topcoffea_path('data/TauSF/TauSF_dm_UL2016_preVFP.json')])
extLepSF.add_weight_sets(["TauSF_dm_2016 TauSF/dm_value %s"%topcoffea_path('data/TauSF/TauSF_dm_UL2016_postVFP.json')])
extLepSF.add_weight_sets(["TauSF_dm_2017 TauSF/dm_value %s"%topcoffea_path('data/TauSF/TauSF_dm_UL2017.json')])
extLepSF.add_weight_sets(["TauSF_dm_2018 TauSF/dm_value %s"%topcoffea_path('data/TauSF/TauSF_dm_UL2018.json')])

extLepSF.add_weight_sets(["TauFakeSF TauSF/pt_value %s"%topcoffea_path('data/TauSF/TauFakeSF.json')])
extLepSF.add_weight_sets(["TauFakeSF_up TauSF/pt_up %s"%topcoffea_path('data/TauSF/TauFakeSF.json')])
extLepSF.add_weight_sets(["TauFakeSF_down TauSF/pt_down %s"%topcoffea_path('data/TauSF/TauFakeSF.json')])

# Jet Veto Maps
def ApplyJetVetoMaps(jets, year):
    jme_year = clib_year_map[year]
    key = jet_veto_dict[year]
    json_path = topcoffea_path(f"data/POG/JME/{jme_year}/jetvetomaps.json.gz")

    # Grab the json
    ceval = correctionlib.CorrectionSet.from_file(json_path)

    # Flatten the inputs
    eta_flat = ak.flatten(jets.eta)
    phi_flat = ak.flatten(jets.phi)

    #Put mins and maxes on the accepted values
    eta_flat_bound = ak.where(eta_flat>5.19, 5.19, ak.where(eta_flat<-5.19, -5.19, eta_flat))
    phi_flat_bound = ak.where(phi_flat>3.14159,3.14159, ak.where(phi_flat<-3.14159,-3.14159, phi_flat))

    #Get pass/fail values for each jet (0 is pass and >0 is fail)
    jet_vetomap_flat = ceval[key].evaluate('jetvetomap',eta_flat_bound,phi_flat_bound)
    
    #Unflatten the array
    jet_vetomap_score = ak.unflatten(jet_vetomap_flat,ak.num(jets.phi))

    #Sum the outputs for each event (if the sum is >0, the event will fail)
    veto_map_event = ak.sum(jet_vetomap_score, axis=-1)

    return veto_map_event

# Fake rate
for year in ['2016APV_2016', 2017, 2018]:
    for syst in ['','_up','_down','_be1','_be2','_pt1','_pt2']:
        extLepSF.add_weight_sets([("MuonFR_{year}{syst} FR_mva085_mu_data_comb_recorrected{syst} %s" % topcoffea_path(basepathFromTTH + 'fakerate/fr_{year}_recorrected.root')).format(year=year,syst=syst)])
        extLepSF.add_weight_sets([("ElecFR_{year}{syst} FR_mva090_el_data_comb_NC_recorrected{syst} %s" % topcoffea_path(basepathFromTTH + 'fakerate/fr_{year}_recorrected.root')).format(year=year,syst=syst)])
extLepSF.finalize()
SFevaluator = extLepSF.make_evaluator()

ffSysts=['','_up','_down','_be1','_be2','_pt1','_pt2']

def ApplyTES(year, taus, isData, vsJetWP="Loose"):
    if isData:
        return (taus.pt, taus.mass)

    pt  = taus.pt
    dm  = taus.decayMode
    gen = taus.genPartFlav
    eta = taus.eta

    clib_year = clib_year_map[year]
    is_run2 = False
    if year.startswith("201"):
        is_run2 = True

    is_run3 = not is_run2

    if is_run2:

        kinFlag = (pt>20) & (pt<205) & (gen==5)
        dmFlag = ((dm==0) | (dm==1) | (dm==10) | (dm==11))
        whereFlag = kinFlag & dmFlag #((pt>20) & (pt<205) & (gen==5) & (dm==0 | dm==1 | dm==10 | dm==11))
        tes = np.where(whereFlag, SFevaluator['TauTES_{year}'.format(year=year)](dm,pt), 1)

        kinFlag = (pt>20) & (pt<205) & (gen>=1) & (gen<=4)
        dmFlag = ((dm==0) | (dm==1))
        whereFlag = kinFlag & dmFlag
        fes = np.where(whereFlag, SFevaluator['TauFES_{year}'.format(year=year)](eta,dm), 1)

    if is_run3:

        json_path = topcoffea_path(f"data/POG/TAU/{clib_year}/tau.json.gz")
        ceval = correctionlib.CorrectionSet.from_file(json_path)
        corr = ceval["tau_energy_scale"]

        flat_pt  = ak.flatten(ak.fill_none(pt, 0.0), axis=1)
        flat_eta = ak.flatten(ak.fill_none(eta, 0.0), axis=1)
        flat_dm  = ak.flatten(ak.fill_none(dm, -1), axis=1)
        flat_gen = ak.flatten(ak.fill_none(gen, 0), axis=1)

        flat_all_pt = ak.flatten(ak.fill_none(pt, 0.0), axis=1)
        flat_all_pt_np = ak.to_numpy(flat_all_pt)
        counts = ak.num(pt, axis=1)

        # Genuine taus (genmatch==5) receive the TES weights
        tes_kin = (flat_pt > 20) & (flat_pt < 205)
        tes_dm  = (flat_dm == 0) | (flat_dm == 1) | (flat_dm == 2) | (flat_dm == 10) | (flat_dm == 11)
        tes_where = tes_kin & tes_dm & (flat_gen == 5)

        full_tes = np.ones_like(flat_all_pt_np, dtype=np.float32)
        tes_indices = np.nonzero(ak.to_numpy(tes_where))[0]
        if len(tes_indices) > 0:
            tes_values = corr.evaluate(
                ak.to_numpy(flat_pt[tes_where]),
                ak.to_numpy(flat_eta[tes_where]),
                ak.to_numpy(flat_dm[tes_where]),
                ak.to_numpy(flat_gen[tes_where]),
                "DeepTau2018v2p5",
                vsJetWP,
                "VVLoose",
                "nom",
            )
            full_tes = ak.to_numpy(full_tes)
            full_tes[tes_indices] = tes_values
        tes = ak.unflatten(full_tes, counts)

        # Electron/muon fakes (genmatch 1-4) receive the FES weights
        fes_kin = (flat_pt > 20) & (flat_pt < 205)
        fes_dm  = (flat_dm == 0) | (flat_dm == 1)
        fes_where = fes_kin & fes_dm & (flat_gen >= 1) & (flat_gen <= 4)

        full_fes = np.ones_like(flat_all_pt_np, dtype=np.float32)
        fes_indices = np.nonzero(ak.to_numpy(fes_where))[0]
        if len(fes_indices) > 0:
            fes_values = corr.evaluate(
                ak.to_numpy(flat_pt[fes_where]),
                ak.to_numpy(flat_eta[fes_where]),
                ak.to_numpy(flat_dm[fes_where]),
                ak.to_numpy(flat_gen[fes_where]),
                "DeepTau2018v2p5",
                vsJetWP,
                "VVLoose",
                "nom",
            )
            full_fes = ak.to_numpy(full_fes)
            full_fes[fes_indices] = fes_values
        fes = ak.unflatten(full_fes, counts)

    return (taus.pt*tes*fes, taus.mass*tes*fes)

def ApplyTESSystematic(year, taus, isData, syst_name, vsJetWP="Loose"):
    if not syst_name.startswith('TES') or isData:
        return (taus.pt, taus.mass)

    pt  = taus.pt
    dm  = taus.decayMode
    gen = taus.genPartFlav
    eta = taus.eta

    clib_year = clib_year_map[year]
    is_run2 = False
    if year.startswith("201"):
        is_run2 = True

    is_run3 = not is_run2

    syst_lab = f'TauTES_{year}'
    syst = "nom"
    if syst_name.endswith("Up"):
        syst = "up"
        syst_lab += '_up'
    elif syst_name.endswith("Down"):
        syst = "down"
        syst_lab += '_down'

    if is_run2:

        kinFlag = (pt>20) & (pt<205) & (gen==5)
        dmFlag = ((dm==0) | (dm==1) | (dm==10) | (dm==11))
        whereFlag = kinFlag & dmFlag

        tes_syst = np.where(whereFlag, SFevaluator[syst_lab](dm,pt), 1)

    if is_run3:
        json_path = topcoffea_path(f"data/POG/TAU/{clib_year}/tau.json.gz")
        ceval = correctionlib.CorrectionSet.from_file(json_path)
        corr   = ceval['tau_energy_scale']

        flat_pt  = ak.flatten(ak.fill_none(pt, 0.0), axis=1)
        flat_eta = ak.flatten(ak.fill_none(eta, 0.0), axis=1)
        flat_dm  = ak.flatten(ak.fill_none(dm, -1), axis=1)
        flat_gen = ak.flatten(ak.fill_none(gen, 0), axis=1)

        kinFlag = (flat_pt>20) & (flat_pt<205) & (flat_gen==5)
        dmFlag = ((flat_dm==0) | (flat_dm==1) | (flat_dm==10) | (flat_dm==11))
        whereFlag = kinFlag & dmFlag

        flat_all_pt = ak.flatten(ak.fill_none(pt, 0.0), axis=1)
        full_tes_syst = np.ones_like(flat_all_pt, dtype=np.float32)
        indices = np.nonzero(ak.to_numpy(whereFlag))[0]

        if len(indices) > 0:
            tes_syst_values = corr.evaluate(
                ak.to_numpy((flat_pt[whereFlag])),
                ak.to_numpy((flat_eta[whereFlag])),
                ak.to_numpy((flat_dm[whereFlag])),
                1,
                "DeepTau2018v2p5",
                vsJetWP,
                "VVLoose",
                syst
            )

            full_tes_syst = ak.to_numpy(full_tes_syst)
            full_tes_syst[indices] = tes_syst_values

        counts = ak.num(pt,axis=1)
        tes_syst = ak.unflatten(full_tes_syst, counts)

    return (taus.pt*tes_syst, taus.mass*tes_syst)

def ApplyFESSystematic(year, taus, isData, syst_name, vsJetWP="Loose"):
    if not syst_name.startswith('FES') or isData:
        return (taus.pt, taus.mass)

    pt  = taus.pt
    eta  = taus.eta
    dm  = taus.decayMode
    gen = taus.genPartFlav

    clib_year = clib_year_map[year]
    is_run2 = False
    if year.startswith("201"):
        is_run2 = True

    is_run3 = not is_run2

    syst_lab = f'TauFES_{year}'
    syst = "nom"

    if syst_name.endswith("Up"):
        syst = "up"
        syst_lab += '_up'
    elif syst_name.endswith("Down"):
        syst = "down"
        syst_lab += '_down'

    if is_run2:
        kinFlag = (pt>20) & (pt<205) & (gen>=1) & (gen<=4)
        dmFlag = ((taus.decayMode==0) | (taus.decayMode==1))
        whereFlag = kinFlag & dmFlag

        fes_syst = np.where(whereFlag, SFevaluator[syst_lab](eta,dm), 1)

    if is_run3:
        json_path = topcoffea_path(f"data/POG/TAU/{clib_year}/tau.json.gz")
        ceval = correctionlib.CorrectionSet.from_file(json_path)
        corr   = ceval['tau_energy_scale']

        flat_pt  = ak.flatten(ak.fill_none(pt, 0.0), axis=1)
        flat_eta = ak.flatten(ak.fill_none(eta, 0.0), axis=1)
        flat_dm  = ak.flatten(ak.fill_none(dm, -1), axis=1)
        flat_gen = ak.flatten(ak.fill_none(gen, 0), axis=1)

        kinFlag = (flat_pt>20) & (flat_pt<205) & (flat_gen>=1) & (flat_gen<=4)
        dmFlag = ((flat_dm==0) | (flat_dm==1))
        whereFlag = kinFlag & dmFlag

        flat_all_pt = ak.flatten(ak.fill_none(pt, 0.0), axis=1)
        full_fes_syst = np.ones_like(flat_all_pt, dtype=np.float32)
        indices = np.nonzero(ak.to_numpy(whereFlag))[0]

        if len(indices) > 0:
            fes_syst_values = corr.evaluate(
                ak.to_numpy((flat_pt[whereFlag])),
                ak.to_numpy((flat_eta[whereFlag])),
                ak.to_numpy((flat_dm[whereFlag])),
                1,
                "DeepTau2018v2p5",
                vsJetWP,
                "VVLoose",
                syst
            )

            full_fes_syst = ak.to_numpy(full_fes_syst)
            full_fes_syst[indices] = fes_syst_values

        counts = ak.num(pt,axis=1)
        fes_syst = ak.unflatten(full_fes_syst, counts)

    return (taus.pt*fes_syst, taus.mass*fes_syst)

def AttachTauSF(events, taus, year, vsJetWP="Loose"):
    pt   = taus.pt
    dm   = taus.decayMode
    eta  = taus.eta
    gen  = taus.genPartFlav
    mass = taus.mass

    DT_sf_list = []
    DT_up_list = []
    DT_do_list = []

    pt_mask_flat = ak.flatten((pt>20) & (pt<205))

    flat_pt = ak.flatten(ak.fill_none(pt,0))
    flat_dm = ak.flatten(ak.fill_none(dm,0))
    flat_gen = ak.flatten(ak.fill_none(gen,0))
    flat_eta = ak.flatten(ak.fill_none(eta,0))
    is_run2 = False
    if year.startswith("201"):
        is_run2 = True

    is_run3 = not is_run2

    ## Correction-lib implementation - MUST BE TESTED WHEN TAU IN THE MASTER BRANCH PROCESSOR
    if is_run2:
        clib_year = clib_year_map[year]
        json_path = topcoffea_path(f"data/POG/TAU/{clib_year}/tau.json.gz")
        ceval = correctionlib.CorrectionSet.from_file(json_path)
 
        wp   = taus.idDeepTau2017v2p1VSjet

        ## legacy
        whereFlag = ((pt>20) & (pt<205) & (gen==5) & (taus[f"is{vsJetWP}"]>0))
        real_sf_loose = np.where(whereFlag, SFevaluator[f'TauSF_{year}_{vsJetWP}'](dm,pt), 1)
        real_sf_loose_up = np.where(whereFlag, SFevaluator[f'TauSF_{year}_{vsJetWP}_up'](dm,pt), 1)
        real_sf_loose_down = np.where(whereFlag, SFevaluator[f'TauSF_{year}_{vsJetWP}_down'](dm,pt), 1)

        whereFlag = ((pt>20) & (pt<205) & ((gen==1)|(gen==3)) & (taus["iseTight"]>0))
        fake_elec_sf = np.where(whereFlag, SFevaluator[f'Tau_elecFakeSF_{year}'](np.abs(eta)), 1)
        fake_elec_sf_up = np.where(whereFlag, SFevaluator[f'Tau_elecFakeSF_{year}_up'](np.abs(eta)), 1)
        fake_elec_sf_down = np.where(whereFlag, SFevaluator[f'Tau_elecFakeSF_{year}_down'](np.abs(eta)), 1)
        whereFlag = ((pt>20) & (pt<205) & ((gen==2)|(gen==4))  & (taus["ismTight"]>0))
        fake_muon_sf = np.where(whereFlag, SFevaluator[f'Tau_muonFakeSF_{year}'](np.abs(eta)), 1)
        fake_muon_sf_up = np.where(whereFlag, SFevaluator[f'Tau_muonFakeSF_{year}_up'](np.abs(eta)), 1)
        fake_muon_sf_down = np.where(whereFlag, SFevaluator[f'Tau_muonFakeSF_{year}_down'](np.abs(eta)), 1)

        whereFlag = ((pt>20) & (pt<205) & (gen!=5) & (gen!=4) & (gen!=3) & (gen!=2) & (gen!=1) & (taus[f"is{vsJetWP}"]>0))
        new_fake_sf = np.where(whereFlag, SFevaluator['TauFakeSF'](pt), 1)
        new_fake_sf_up = np.where(whereFlag, SFevaluator['TauFakeSF_up'](pt), 1)
        new_fake_sf_down = np.where(whereFlag, SFevaluator['TauFakeSF_down'](pt), 1)

        real_sf = real_sf_loose
        real_sf_up = real_sf_loose_up
        real_sf_down = real_sf_loose_down

    if is_run3:
        clib_year = clib_year_map[year]
        json_path = topcoffea_path(f"data/POG/TAU/{clib_year}/tau.json.gz")
        ceval = correctionlib.CorrectionSet.from_file(json_path)
        corr_jet = ceval["DeepTau2018v2p5VSjet"]
        corr_e = ceval["DeepTau2018v2p5VSe"]

        vsjet_raw_mask = taus[f"is{vsJetWP}"] > 0        
        vsjet_mask     = ak.fill_none(vsjet_raw_mask, False)             
        vsjet_flat_mask = ak.flatten(vsjet_mask)     

        vse_raw_mask = taus[f"iseTight"] > 0        
        vse_mask     = ak.fill_none(vse_raw_mask, False)             
        vse_flat_mask = ak.flatten(vse_mask)            


        deep_tau_cuts = [
            (
                "DeepTau2018v2p5VSjet",
                vsjet_flat_mask,
                (flat_pt, flat_dm, flat_gen, vsJetWP, "Tight"),
                (flat_gen == 5),
            ),
            (
                "DeepTau2018v2p5VSe",
                vse_flat_mask,
                (flat_eta, flat_dm, flat_gen, "VVLoose"),
                ((flat_gen == 1) | (flat_gen == 3)),
            ),
        ]

        for idx, deep_tau_cut in enumerate(deep_tau_cuts):
            discr = deep_tau_cut[0]
            id_mask_flat = ak.fill_none(deep_tau_cut[1], False)
            arg_list = (deep_tau_cut[2])
            gen_mask_flat = ak.fill_none(deep_tau_cut[3], False)
            tau_mask_flat = ak.fill_none(id_mask_flat & pt_mask_flat & gen_mask_flat, False)

            if "VSjet" in discr:
                arg_sf = arg_list + ("nom", "pt")
            else:
                arg_sf = arg_list + ("nom",)
            DT_sf_list.append(
                ak.where(
                    ~tau_mask_flat,
                    1,
                    ceval[discr].evaluate(*arg_sf)
                )
            )

            if "VSjet" in discr:
                arg_up = arg_list + ("up", "pt")
            else:
                arg_up = arg_list + ("up",)
            DT_up_list.append(
                ak.where(
                    ~tau_mask_flat,
                    1,
                    ceval[discr].evaluate(*arg_up)
                )
            )
            if "VSjet" in discr:
                arg_down = arg_list + ("down", "pt")
            else:
                arg_down = arg_list + ("down",)
            DT_do_list.append(
                ak.where(
                    ~tau_mask_flat,
                    1,
                    ceval[discr].evaluate(*arg_down)
                )
            )

        DT_sf_flat = None
        DT_up_flat = None
        DT_do_flat = None

        for idr, DT_sf_discr in enumerate(DT_sf_list):
            DT_sf_discr = ak.to_numpy(DT_sf_discr)
            DT_up_discr = ak.to_numpy(DT_up_list[idr])
            DT_do_discr = ak.to_numpy(DT_do_list[idr])

            if idr == 0:
                real_sf = ak.unflatten(DT_sf_discr, ak.num(pt))
                real_sf_up = ak.unflatten(DT_up_discr, ak.num(pt))
                real_sf_down = ak.unflatten(DT_do_discr, ak.num(pt))
            if idr == 1:
                fake_elec_sf = ak.unflatten(DT_sf_discr, ak.num(pt))
                fake_elec_sf_up = ak.unflatten(DT_up_discr, ak.num(pt))
                fake_elec_sf_down = ak.unflatten(DT_do_discr, ak.num(pt))

        new_fake_sf = ak.fill_none(np.ones_like(pt, dtype=np.float32), 1.0)
        new_fake_sf_up = ak.fill_none(np.ones_like(pt, dtype=np.float32), 1.0)
        new_fake_sf_down = ak.fill_none(np.ones_like(pt, dtype=np.float32), 1.0)
        fake_muon_sf = ak.fill_none(np.ones_like(pt, dtype=np.float32), 1.0)
        fake_muon_sf_up = ak.fill_none(np.ones_like(pt, dtype=np.float32), 1.0)
        fake_muon_sf_down = ak.fill_none(np.ones_like(pt, dtype=np.float32), 1.0)

    taus["sf_tau_real"] = real_sf
    taus["sf_tau_real_up"] = real_sf_up
    taus["sf_tau_real_down"] = real_sf_down
    taus["sf_tau_fake"] = fake_elec_sf*fake_muon_sf*new_fake_sf
    taus["sf_tau_fake_up"] = fake_elec_sf_up*fake_muon_sf_up*new_fake_sf_up
    taus["sf_tau_fake_down"] = fake_elec_sf_down*fake_muon_sf_down*new_fake_sf_down

    padded_taus = ak.pad_none(taus, 1)
    events["sf_2l_taus_real"] = padded_taus.sf_tau_real[:,0]
    events["sf_2l_taus_real_hi"] = padded_taus.sf_tau_real_up[:,0]
    events["sf_2l_taus_real_lo"] = padded_taus.sf_tau_real_down[:,0]
    events["sf_2l_taus_fake"] = padded_taus.sf_tau_fake[:,0]
    events["sf_2l_taus_fake_hi"] = padded_taus.sf_tau_fake_up[:,0]
    events["sf_2l_taus_fake_lo"] = padded_taus.sf_tau_fake_down[:,0]

def AttachPerLeptonFR(leps, flavor, year):
    # Get the flip rates lookup object
    if year not in clib_year_map.keys():
        raise Exception(f"Error: Unknown year \"{year}\"\n"".")

    #if year == "2016APV": flip_year_name = "UL16APV"
    #elif year == "2016": flip_year_name = "UL16"
    #elif year == "2017": flip_year_name = "UL17"
    #elif year == "2018": flip_year_name = "UL18"
    is_run2 = False
    if year.startswith("201"):
        flip_year_name = year.replace("20", "UL")
        is_run2 = True

    is_run3 = not is_run2
    #else: flip_year_name = "UL18" #TO READAPT when fakefactors are ready #raise Exception(f"Not a known year: {year}")
    #Run2 is not implemented with correction_lib
    if is_run2:
        with gzip.open(topeft_path(f"data/fliprates/flip_probs_topcoffea_{flip_year_name}.pkl.gz")) as fin:
            flip_hist = pickle.load(fin)
            flip_lookup = lookup_tools.dense_lookup.dense_lookup(flip_hist.values()[()],[flip_hist.axes["pt"].edges,flip_hist.axes["eta"].edges])

        # Get the fliprate scaling factor for the given year
        chargeflip_sf = get_te_param("chargeflip_sf_dict")[flip_year_name]

        if flavor == "Elec":
            leps['fliprate'] = (chargeflip_sf)*(flip_lookup(leps.pt,abs(leps.eta)))
        else:
            leps['fliprate'] = np.zeros_like(leps.pt)

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
        elif year == '2017':
            leps['fakefactor_elclosurefactor'] = (np.abs(leps.pdgId)==11)*0.2 + 1.0
            leps['fakefactor_muclosurefactor'] = (np.abs(leps.pdgId)==13)*0.2 + 1.0
        elif year == '2018':
            leps['fakefactor_elclosurefactor'] = (np.abs(leps.pdgId)==11) * ((np.abs(leps.eta) > 1.5)*0.5 + (np.abs(leps.eta) < 1.5)*0.1) + 1.0
            leps['fakefactor_muclosurefactor'] = (np.abs(leps.pdgId)==13)*0.05 + 1.0

    #Run3 is implemented with correction_lib
    if is_run3:
        flip_year_name = year
        with gzip.open(topeft_path(f"data/fliprates/flip_probs_topcoffea_{flip_year_name}.pkl.gz")) as fin:
            flip_hist = pickle.load(fin)
            flip_lookup = lookup_tools.dense_lookup.dense_lookup(flip_hist.values()[()],[flip_hist.axes["pt"].edges,flip_hist.axes["eta"].edges])

        # Apply scaling factor for electrons
        if flavor == "Elec":
            leps['fliprate'] = (get_flipsf(leps.eta, year))*(flip_lookup(leps.pt,abs(leps.eta)))
        else:
            leps['fliprate'] = np.zeros_like(leps.pt)

        json_path = topeft_path("data/fakerates/fake_rates_Run3.json")
        ceval = correctionlib.CorrectionSet.from_file(json_path)
        pt = ak.flatten(leps.pt)
        abseta = ak.flatten(abs(leps.eta))
        abspdgid = ak.flatten(abs(leps.pdgId))

        minpt = 0.
        maxpt = 100.

        if flavor == "Elec":
            minpt = 15.
        elif flavor == "Muon":
            minpt = 10.

        pt_mask_low = (pt > minpt)
        pt_mask_hi = (pt < maxpt)
        pt_masked = ak.where(~pt_mask_low, minpt, pt)
        pt_masked = ak.where(~pt_mask_hi, maxpt-0.5, pt_masked)

        chargeflip_sf = ak.ones_like(leps.pdgId, dtype=np.float64) #get_te_param("chargeflip_sf_dict")[flip_year_name]

        for syst in ffSysts:
            fr = ak.unflatten(ceval["fakeRate_2022_2022EE"].evaluate(pt_masked, abseta, syst, abspdgid), ak.num(leps.pt))
            leps['fakefactor%s' % syst] = ak.fill_none(-fr/(1-fr),0)
            leps['fakefactor_elclosurefactor'] = (np.abs(leps.pdgId)==11)*0.0 + 1.0
            leps['fakefactor_muclosurefactor'] = (np.abs(leps.pdgId)==13)*0.0 + 1.0

    #Common part
    for flav in ['el','mu']:
        leps['fakefactor_%sclosuredown' % flav] = leps['fakefactor'] / leps['fakefactor_%sclosurefactor' % flav]
        leps['fakefactor_%sclosureup' % flav]   = leps['fakefactor'] * leps['fakefactor_%sclosurefactor' % flav]

def get_flipsf(eta_array, year):
    # Get flip scaling factors for run3

    json_path = topeft_path(f"data/fliprates/flip_sf_{year}.json")
    with open(json_path, 'r') as f:
        chargeflip_sf_dict = json.load(f)

    flip_sf = ak.full_like(eta_array, 1.0)  # default value

    for bin_str, sf in chargeflip_sf_dict["FlipSF_eta"].items():
        # Parse bin string like "[-3,-1.479]"
        low, high = map(float, bin_str.strip("[]").split(","))
        # Apply mask
        mask = ((eta_array >= low) & (eta_array < high)) | ((eta_array == 2.5) & (high == 2.5))
        flip_sf = ak.where(mask, sf, flip_sf)

    return flip_sf

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

def AttachMuonSF(muons, year, useRun3MVA=True):
    '''
      Description:
          Inserts 'sf_nom', 'sf_hi', and 'sf_lo' into the muons array passed to this function. These
          values correspond to the nominal, up, and down muon scalefactor values respectively.
    Run2 strategy:
    - Use reco from TOP-22-006
    - use loose from correction-lib
    Run3 strategy
    - use loose from correction-lib
    - reco not available yet, but MUO don't bother about that
    '''
    is_run3 = False
    if year.startswith("202"):
        is_run3 = True
    is_run2 = not is_run3

    eta = np.abs(muons.eta)
    pt = muons.pt
    if year not in clib_year_map.keys():
        raise Exception(f"Error: Unknown year \"{year}\"\n"".")

    ## Run2 pieces
    new_sf = ak.ones_like(pt)
    new_up = ak.ones_like(pt)
    new_do = ak.ones_like(pt)
    reco_sf = ak.ones_like(pt)
    reco_up = ak.ones_like(pt)
    reco_do = ak.ones_like(pt)
    loose_sf = ak.ones_like(pt)
    loose_up = ak.ones_like(pt)
    loose_do = ak.ones_like(pt)
    iso_sf = ak.ones_like(pt)
    iso_up = ak.ones_like(pt)
    iso_do = ak.ones_like(pt)
    ## Run3 pieces
    reco_loose_sf = ak.ones_like(pt)
    reco_loose_up = ak.ones_like(pt)
    reco_loose_do = ak.ones_like(pt)

    ## Run2:
    ## only loose_sf can be consistently used with correction-lib, for the other we use the TOP-22-006 original SFs
    ## Run3:
    ## reco_loose_sf will be used in place of reco*loose, both for nominal and systematics

    clib_year = clib_year_map[year]
    json_path = topcoffea_path(f"data/POG/MUO/{clib_year}/muon_Z.json.gz")
    ceval = correctionlib.CorrectionSet.from_file(json_path)

    pt_flat = ak.flatten(pt)
    abseta_flat = ak.flatten(eta)
    pdgid_flat = ak.flatten(abs(muons.pdgId))

    pt_mask = ak.flatten((pt >= 15))
    pt_mask_reco = ak.flatten((pt >= 40))
    pt_flat_reco = ak.where(~pt_mask_reco, 40., pt_flat)
    pt_flat_loose = ak.where(~pt_mask, 15., pt_flat)

    if is_run2:
        ## The only one to be actually got from clib for Run2<
        loose_sf_flat = ak.where(
            ~pt_mask,
            1,
            ceval["NUM_LooseID_DEN_TrackerMuons"].evaluate(abseta_flat, pt_flat_loose, "nominal")
        )
        loose_err_flat = ak.where(
            ~pt_mask,
            1,
            np.sqrt(
                ceval["NUM_LooseID_DEN_TrackerMuons"].evaluate(abseta_flat, pt_flat_loose, "syst") * ceval["NUM_LooseID_DEN_TrackerMuons"].evaluate(abseta_flat, pt_flat_loose, "syst") + ceval["NUM_LooseID_DEN_TrackerMuons"].evaluate(abseta_flat, pt_flat_loose, "stat") * ceval["NUM_LooseID_DEN_TrackerMuons"].evaluate(abseta_flat, pt_flat_loose, "stat")
            )
        )
        loose_sf  = ak.unflatten(loose_sf_flat, ak.num(pt))
        loose_err = ak.unflatten(loose_err_flat, ak.num(pt))
        loose_up = loose_sf + loose_err
        loose_do = loose_sf - loose_err

        ## these are the reco and loose TOP-22-006 SFs
        reco_sf  = np.where(pt < 20,SFevaluator['MuonRecoSF_{year}'.format(year=year)](eta,pt),1) # sf=1 when pt>20 becuase there is no reco SF available
        reco_err = np.where(pt < 20,SFevaluator['MuonRecoSF_{year}_er'.format(year=year)](eta,pt),0) # sf error =0 when pt>20 becuase there is no reco SF available
        reco_up  = reco_sf + reco_err
        reco_do  = reco_sf - reco_err

        ## For Run2, reco and loose are multiplied
        reco_loose_sf = reco_sf * loose_sf
        reco_loose_up = reco_up * loose_up
        reco_loose_do = reco_do * loose_do

        ## ad-hoc from TOP-22-006 for Run2 (not clib ready)
        iso_sf  = SFevaluator['MuonIsoSF_{year}'.format(year=year)](eta,pt)
        iso_err = SFevaluator['MuonIsoSF_{year}_er'.format(year=year)](eta,pt)
        iso_up  = iso_sf + iso_err
        iso_do  = iso_sf - iso_err

        new_sf  = SFevaluator['MuonSF_{year}'.format(year=year)](eta,pt)
        new_err = SFevaluator['MuonSF_{year}_er'.format(year=year)](eta,pt)
        new_up = new_sf + new_err
        new_do = new_sf - new_err

    elif is_run3:
        loose_sf_flat = ak.where(
            ~pt_mask,
            1,
            ceval["NUM_LooseID_DEN_TrackerMuons"].evaluate(abseta_flat, pt_flat_loose, "nominal")
        )
        loose_err_flat = ak.where(
            ~pt_mask,
            1,
            np.sqrt(
                ceval["NUM_LooseID_DEN_TrackerMuons"].evaluate(abseta_flat, pt_flat_loose, "syst") * ceval["NUM_LooseID_DEN_TrackerMuons"].evaluate(abseta_flat, pt_flat_loose, "syst") + ceval["NUM_LooseID_DEN_TrackerMuons"].evaluate(abseta_flat, pt_flat_loose, "stat") * ceval["NUM_LooseID_DEN_TrackerMuons"].evaluate(abseta_flat, pt_flat_loose, "stat")
            )
        )
        loose_sf  = ak.unflatten(loose_sf_flat, ak.num(pt))
        loose_err = ak.unflatten(loose_err_flat, ak.num(pt))
        loose_up = loose_sf + loose_err
        loose_do = loose_sf - loose_err

        if "mvaTTHrun3" in muons.fields and useRun3MVA:
            #clib integration of the lepMVA Run3 SFs
            if year.startswith("2022"):
                lepmva_json_path = topeft_path(f"data/lepMVASF/leptonSF_{year}.json.gz")
            elif year.startswith("2023"):
                lepmva_json_path = topeft_path(f"data/lepMVASF/muon_mvaTTH_{year}.json.gz")
            else:
                raise ValueError(f"{year} is not supported for the lepMVA SFs.")
            lepmva_ceval = correctionlib.CorrectionSet.from_file(lepmva_json_path)

            #lep mva SFs in clib format
            minpt = 15.0
            maxpt = 500.
            pt_mask_low = (pt_flat > minpt)
            pt_mask_hi = (pt_flat < maxpt)
            pt_lepmva_mask = pt_mask_low & pt_mask_hi
            pt_masked = ak.where(~pt_mask_low, minpt+0.1, pt_flat)
            pt_masked = ak.where(~pt_mask_hi, maxpt-0.5, pt_masked)
            pt_lepmva_flat = pt_masked

            if year.startswith("2022"):
                muo_tag  = "mu_allflavor"
                lepmva_vals_nom= lepmva_ceval[muo_tag].evaluate(abseta_flat, pt_lepmva_flat, "", pdgid_flat)
                lepmva_vals_up= lepmva_ceval[muo_tag].evaluate(abseta_flat, pt_lepmva_flat, "_muup", pdgid_flat)
                lepmva_vals_down= lepmva_ceval[muo_tag].evaluate(abseta_flat, pt_lepmva_flat, "_mudn", pdgid_flat)
            elif year.startswith("2023"):
                muo_tag = "NUM_TightmvaTTH_DEN_LooseMuons"
                lepmva_vals_nom = lepmva_ceval[muo_tag].evaluate(abseta_flat, pt_lepmva_flat, "nominal")
                lepmva_vals_up = lepmva_ceval[muo_tag].evaluate(abseta_flat, pt_lepmva_flat, "systup")
                lepmva_vals_down = lepmva_ceval[muo_tag].evaluate(abseta_flat, pt_lepmva_flat, "systdown")

            new_sf_flat = ak.where(
                ~pt_lepmva_mask,
                1,
                lepmva_vals_nom,
            )
            new_up_flat = ak.where(
                ~pt_lepmva_mask,
                1,
                lepmva_vals_up,
            )
            new_do_flat = ak.where(
                ~pt_lepmva_mask,
                1,
                lepmva_vals_down,
            )
            new_sf = ak.unflatten(new_sf_flat, ak.num(pt))
            new_up = ak.unflatten(new_up_flat, ak.num(pt))
            new_do = ak.unflatten(new_do_flat, ak.num(pt))

        ##Not available yet for Run3
        #reco_loose_sf_flat = ak.where(
        #    ~pt_mask,
        #    1,
        #    ceval["NUM_LooseID_DEN_genTracks"].evaluate(abseta_flat, pt_flat_loose, "nominal")
        #)
        #reco_loose_err_flat = ak.where(
        #    ~pt_mask,
        #    1,
        #    ceval["NUM_LooseID_DEN_genTracks"].evaluate(abseta_flat, pt_flat_loose, "syst") * ceval["NUM_LooseID_DEN_genTracks"].evaluate(abseta_flat, pt_flat_loose, "syst") + ceval["NUM_LooseID_DEN_genTracks"].evaluate(abseta_flat, pt_flat_loose, "stat") * ceval["NUM_LooseID_DEN_genTracks"].evaluate(abseta_flat, pt_flat_loose, "stat")
        #)

        #reco_sf = ak.unflatten(reco_sf_flat, ak.num(pt))
        #reco_err = ak.unflatten(reco_err_flat, ak.num(pt))
        #reco_up = reco_sf + reco_err
        #reco_do = reco_sf - reco_err
        #reco_loose_sf = ak.unflatten(reco_loose_sf_flat, ak.num(pt))
        #reco_loose_err = ak.unflatten(reco_loose_err_flat, ak.num(pt))
        #reco_loose_up = reco_loose_sf + reco_loose_err
        #reco_loose_do = reco_loose_sf - reco_loose_err

    #Run2+Run3 implementation
    muons['sf_nom_2l_muon'] = new_sf * reco_loose_sf * iso_sf
    muons['sf_hi_2l_muon']  = new_up * reco_loose_up * iso_up
    muons['sf_lo_2l_muon']  = new_do * reco_loose_do * iso_do
    muons['sf_nom_3l_muon'] = new_sf * reco_loose_sf
    muons['sf_hi_3l_muon']  = new_up * reco_loose_up * iso_up
    muons['sf_lo_3l_muon']  = new_do * reco_loose_do * iso_do
    muons['sf_nom_2l_elec'] = ak.ones_like(new_sf)
    muons['sf_hi_2l_elec']  = ak.ones_like(new_sf)
    muons['sf_lo_2l_elec']  = ak.ones_like(new_sf)
    muons['sf_nom_3l_elec'] = ak.ones_like(new_sf)
    muons['sf_hi_3l_elec']  = ak.ones_like(new_sf)
    muons['sf_lo_3l_elec']  = ak.ones_like(new_sf)

def AttachElectronSF(electrons, year, looseWP=None, useRun3MVA=True):
    '''
      Description:
          Inserts 'sf_nom', 'sf_hi', and 'sf_lo' into the electrons array passed to this function. These
          values correspond to the nominal, up, and down electron scalefactor values respectively.
    '''

    if looseWP is None:
        raise ValueError('when calling AttachElectronSF, a looseWP value must be provided according to the ele ID isPres selection')

    is_run3 = False
    if year.startswith("202"):
        is_run3 = True
    is_run2 = not is_run3
    dt_era = "Run3" if is_run3 else "Run2"

    eta = electrons.eta
    pt = electrons.pt
    phi = electrons.phi
    pdgid = abs(electrons.pdgId)

    #initializing the sf and up/down array to 1, so they are fixed to unit values if the a given SF is not available yet
    ## Run2 and Run3
    reco_sf = ak.ones_like(pt)
    reco_up = ak.ones_like(pt)
    reco_do = ak.ones_like(pt)
    loose_sf = ak.ones_like(pt)
    loose_up = ak.ones_like(pt)
    loose_do = ak.ones_like(pt)
    new_sf_2l = ak.ones_like(pt)
    new_up_2l = ak.ones_like(pt)
    new_do_2l = ak.ones_like(pt)
    new_sf_3l = ak.ones_like(pt)
    new_up_3l = ak.ones_like(pt)
    new_do_3l = ak.ones_like(pt)
    iso_sf = ak.ones_like(pt)
    iso_up = ak.ones_like(pt)
    iso_do = ak.ones_like(pt)

    if year not in clib_year_map.keys():
        raise Exception(f"Error: Unknown year \"{year}\".")

    clib_year = clib_year_map[year]
    json_path = topcoffea_path(f"data/POG/EGM/{clib_year}/electron.json.gz")
    ceval = correctionlib.CorrectionSet.from_file(json_path)

    eta_flat   = ak.flatten(eta)
    pt_flat    = ak.flatten(pt)
    phi_flat   = ak.flatten(phi)
    pdgid_flat = ak.flatten(pdgid)

    pt_bins = egm_pt_bins[dt_era]

    reco_sf_perbin = []
    reco_up_perbin = []
    reco_do_perbin = []

    egm_year = egm_tag_map[clib_year]
    egm_tag = "Electron-ID-SF"
    if is_run2:
        egm_tag = "UL-" + "Electron-ID-SF"

    for bintag, bin_edges in pt_bins.items():
        pt_mask = ak.flatten((pt >= bin_edges[0]) & (pt < bin_edges[1]))
        pt_bin_flat = ak.where(~pt_mask, bin_edges[1]-0.1, pt_flat)
        egm_args = [bintag, eta_flat, pt_bin_flat]
        if "2023" in year:
            egm_args.append(phi_flat)
        reco_sf_perbin.append(
            ak.where(
                ~pt_mask,
                1,
                ceval[egm_tag].evaluate(egm_year, "sf", *egm_args)
            )
        )
        reco_up_perbin.append(
            ak.where(
                ~pt_mask,
                1,
                ceval[egm_tag].evaluate(egm_year, "sfup", *egm_args)
            )
        )
        reco_do_perbin.append(
            ak.where(
                ~pt_mask,
                1,
                ceval[egm_tag].evaluate(egm_year, "sfdown", *egm_args)
            )
        )

    reco_sf_flat = None
    reco_up_flat = None
    reco_do_flat = None

    for idr, reco_sf_bin_flat in enumerate(reco_sf_perbin):
        reco_sf_bin_flat = ak.to_numpy(reco_sf_bin_flat)
        reco_up_bin_flat = ak.to_numpy(reco_up_perbin[idr])
        reco_do_bin_flat = ak.to_numpy(reco_do_perbin[idr])
        if idr == 0:
            reco_sf_flat = reco_sf_bin_flat
            reco_up_flat = reco_up_bin_flat
            reco_do_flat = reco_do_bin_flat
        else:
            reco_sf_flat *= reco_sf_bin_flat
            reco_up_flat *= reco_up_bin_flat
            reco_do_flat *= reco_do_bin_flat

    reco_sf = ak.unflatten(reco_sf_flat, ak.num(pt))
    reco_up = ak.unflatten(reco_up_flat, ak.num(pt))
    reco_do = ak.unflatten(reco_do_flat, ak.num(pt))

    if is_run3:
        if looseWP != "none":
            #print("\n\n\n\n\nI'm applying EGM loose SFs\n\n\n\n\n")
            loose_sf_flat = None
            loose_up_flat = None
            loose_do_flat = None
            pt_mask = ak.flatten((pt >= 10))
            egm_args = [eta_flat, pt_bin_flat]
            if "2023" in year:
                egm_args.append(phi_flat)
            loose_sf_flat = ak.where(
                ~pt_mask,
                1,
                ceval[egm_tag].evaluate(egm_year, "sf", looseWP, *egm_args)
            )
            loose_up_flat = ak.where(
                ~pt_mask,
                1,
                ceval[egm_tag].evaluate(egm_year, "sfup", looseWP, *egm_args)
            )
            loose_do_flat = ak.where(
                ~pt_mask,
                1,
                ceval[egm_tag].evaluate(egm_year, "sfdown", looseWP, *egm_args)
            )
            loose_sf = ak.unflatten(loose_sf_flat, ak.num(pt))
            loose_up = ak.unflatten(loose_up_flat, ak.num(pt))
            loose_do = ak.unflatten(loose_do_flat, ak.num(pt))
        else:
            #print("\n\n\n\n\nI'm NOT applying EGM loose SFs\n\n\n\n\n")
            loose_sf = ak.ones_like(reco_sf)
            loose_up = ak.ones_like(reco_sf)
            loose_do = ak.ones_like(reco_sf)

        if "mvaTTHrun3" in electrons.fields and useRun3MVA:
            #clib integration of the lepMVA Run3 SFs
            if year.startswith("2022"):
                lepmva_json_path = topeft_path(f"data/lepMVASF/leptonSF_{year}.json.gz")
            elif year.startswith("2023"):
                lepmva_json_path = topeft_path(f"data/lepMVASF/electron_mvaTTH_{year}.json.gz")
            else:
                raise ValueError(f"{year} is not supported for the lepMVA SFs.")
            lepmva_ceval = correctionlib.CorrectionSet.from_file(lepmva_json_path)

            minpt = 15.0
            maxpt = 500.
            pt_mask_low = (pt_flat > minpt)
            pt_mask_hi = (pt_flat < maxpt)
            pt_lepmva_mask = pt_mask_low & pt_mask_hi
            pt_masked = ak.where(~pt_mask_low, minpt+0.1, pt_flat)
            pt_masked = ak.where(~pt_mask_hi, maxpt-0.5, pt_masked)
            pt_lepmva_flat = pt_masked

            if year.startswith("2022"):
                egm_tag  = "el_allflavor"
                lepmva_vals_nom= lepmva_ceval[egm_tag].evaluate(abs(eta_flat), pt_lepmva_flat, "", pdgid_flat)
                lepmva_vals_up= lepmva_ceval[egm_tag].evaluate(abs(eta_flat), pt_lepmva_flat, "_elup", pdgid_flat)
                lepmva_vals_down= lepmva_ceval[egm_tag].evaluate(abs(eta_flat), pt_lepmva_flat, "_eldn", pdgid_flat)
            elif year.startswith("2023"):
                egm_tag = "NUM_TightmvaTTH_DEN_LooseElectrons"
                lepmva_vals_nom = lepmva_ceval[egm_tag].evaluate(abs(eta_flat), pt_lepmva_flat, "nominal")
                lepmva_vals_up = lepmva_ceval[egm_tag].evaluate(abs(eta_flat), pt_lepmva_flat, "systup")
                lepmva_vals_down = lepmva_ceval[egm_tag].evaluate(abs(eta_flat), pt_lepmva_flat, "systdown")

            new_sf_flat = ak.where(
                ~pt_lepmva_mask,
                1,
                lepmva_vals_nom,
            )
            new_up_flat = ak.where(
                ~pt_lepmva_mask,
                1,
                lepmva_vals_up,
            )
            new_do_flat = ak.where(
                ~pt_lepmva_mask,
                1,
                lepmva_vals_down,
            )
            new_sf = ak.unflatten(new_sf_flat, ak.num(pt))
            new_up = ak.unflatten(new_up_flat, ak.num(pt))
            new_do = ak.unflatten(new_do_flat, ak.num(pt))
            new_sf_2l = new_sf
            new_up_2l = new_up
            new_do_2l = new_do
            new_sf_3l = new_sf
            new_up_3l = new_up
            new_do_3l = new_do

    else:
        loose_sf  = SFevaluator['ElecLooseSF_{year}'.format(year=year)](np.abs(eta),pt)
        loose_err = SFevaluator['ElecLooseSF_{year}_er'.format(year=year)](np.abs(eta),pt)
        loose_up = loose_sf + loose_err
        loose_do = loose_sf - loose_err

        new_sf_2l  = SFevaluator['ElecSF_{year}_2lss'.format(year=year)](np.abs(eta),pt)
        new_err_2l = SFevaluator['ElecSF_{year}_2lss_er'.format(year=year)](np.abs(eta),pt)
        new_up_2l = new_sf_2l + new_err_2l
        new_do_2l = new_sf_2l - new_err_2l
        new_sf_3l  = SFevaluator['ElecSF_{year}_3l'.format(year=year)](np.abs(eta),pt)
        new_err_3l = SFevaluator['ElecSF_{year}_3l_er'.format(year=year)](np.abs(eta),pt)
        new_up_3l = new_sf_3l + new_err_3l
        new_do_3l = new_sf_3l - new_err_3l

        iso_sf  = SFevaluator['ElecIsoSF_{year}'.format(year=year)](np.abs(eta),pt)
        iso_err = SFevaluator['ElecIsoSF_{year}_er'.format(year=year)](np.abs(eta),pt)
        iso_up = iso_sf + iso_err
        iso_do = iso_sf - iso_err

    #print("\n\n\n\n\n\n\n")
    #print('new_sf', new_sf)
    #print('new_sf_2l', new_sf_2l)
    #print('new_sf_3l', new_sf_3l)
    #print('sf_nom_2l_elec', ak.to_list(reco_sf * new_sf_2l * loose_sf * iso_sf))
    #print('sf_nom_3l_elec', ak.to_list(reco_sf * new_sf_2l * loose_sf))
    #print("\n\n\n\n\n\n\n")

    electrons['sf_nom_2l_elec'] = reco_sf * new_sf_2l * loose_sf * iso_sf
    electrons['sf_hi_2l_elec']  = (reco_up) * new_up_2l * loose_up * iso_up
    electrons['sf_lo_2l_elec']  = (reco_do) * new_do_2l * loose_do * iso_do
    electrons['sf_nom_3l_elec'] = reco_sf * new_sf_3l * loose_sf
    electrons['sf_hi_3l_elec']  = (reco_up) * new_up_3l * loose_up * iso_up
    electrons['sf_lo_3l_elec']  = (reco_do) * new_do_3l * loose_do * iso_do
    electrons['sf_nom_2l_muon'] = ak.ones_like(reco_sf)
    electrons['sf_hi_2l_muon']  = ak.ones_like(reco_sf)
    electrons['sf_lo_2l_muon']  = ak.ones_like(reco_sf)
    electrons['sf_nom_3l_muon'] = ak.ones_like(reco_sf)
    electrons['sf_hi_3l_muon']  = ak.ones_like(reco_sf)
    electrons['sf_lo_3l_muon']  = ak.ones_like(reco_sf)

def AttachElectronCorrections(electrons, run, year, isData=False):
    """
    Attach per-electron scale (data) or smear+scale (MC) corrections.
    """

    # Common electron vars
    pt    = electrons.pt
    eta   = electrons.deltaEtaSC
    r9    = electrons.r9
    gain  = electrons.seedGain
    absEta = abs(eta)

    if year not in clib_year_map.keys() and not year.startswith("202"):
        raise Exception(f"Error: Unknown year \"{year}\".")

    clib_year = clib_year_map[year]
    scale_json = topcoffea_path(f"data/POG/EGM/{clib_year}/electronSS_EtDependent.json.gz")
    smear_json = scale_json  # same file

    # Load correction sets
    cset_scale = correctionlib.CorrectionSet.from_file(scale_json)
    cset_smear = correctionlib.CorrectionSet.from_file(smear_json)

    et_tag = egm_et_map[clib_year]
    # Data: only scale
    if isData:
        scale_eval = cset_scale.compound[f"EGMScale_Compound_Ele_{et_tag}"]
        # flatten arrays for evaluation
        pt_flat = ak.flatten(pt)
        sceta_flat = ak.flatten(eta)
        r9_flat = ak.flatten(r9)
        gain_flat = ak.flatten(gain)
        absEta_flat = ak.flatten(absEta)

        # have a 'run' array with the same structure as electrons.pt, then flatten it
        run_per_electron = ak.full_like(pt, 1, dtype=int) * run
        run_flat = ak.flatten(run_per_electron)

        scale_flat = scale_eval.evaluate(
            "scale",
            run_flat,
            sceta_flat,
            r9_flat,
            absEta_flat,
            pt_flat,
            gain_flat
        )
        # re‐nest to original jagged structure
        electrons["pt_raw"] = pt
        electrons["pt"] = ak.unflatten(scale_flat * pt_flat, ak.num(pt))

    # MC: smear + scale uncertainties
    else:
        # Smear
        smear_eval = cset_smear[f"EGMSmearAndSyst_ElePTsplit_{et_tag}"]
        pt_flat   = ak.flatten(pt)
        r9_flat   = ak.flatten(r9)
        sceta_flat = ak.flatten(eta)
        absEta_flat = ak.flatten(absEta)

        # nominal smear width
        smear_nom = smear_eval.evaluate("smear", pt_flat, r9_flat, absEta_flat)
        # random numbers per event
        rng = np.random.default_rng(12345)
        rnd = rng.normal(size=len(pt_flat))

        pt_smeared_nom = pt_flat * (1 + smear_nom * rnd)

        # systematic up/down on smear
        dsmear = smear_eval.evaluate("esmear", pt_flat, r9_flat, absEta_flat)
        pt_smeared_up   = pt_flat * (1 + (smear_nom + dsmear) * rnd)
        pt_smeared_down = pt_flat * (1 + (smear_nom - dsmear) * rnd)

        # re‐nest
        smeared_nom = ak.unflatten(pt_smeared_nom,   ak.num(pt))
        smeared_up  = ak.unflatten(pt_smeared_up,    ak.num(pt))
        smeared_dn  = ak.unflatten(pt_smeared_down, ak.num(pt))

        electrons["pt_raw"] = pt
        electrons["pt"]  = smeared_nom
        electrons["pt_smear_nom"]  = smeared_nom
        electrons["pt_smear_up"]   = smeared_up
        electrons["pt_smear_down"] = smeared_dn

        # 2) Scale uncertainties on the *smeared* pt
        scale_eval = smear_eval  # same JSON holds "escale"
        escale = scale_eval.evaluate("escale", pt_flat, r9_flat, absEta_flat)

        scale_up   = (1 + escale) * pt_smeared_nom
        scale_down = (1 - escale) * pt_smeared_nom

        electrons["pt_scale_up"]   = ak.unflatten(scale_up,   ak.num(pt))
        electrons["pt_scale_down"] = ak.unflatten(scale_down, ak.num(pt))


###### Btag scale factors
################################################################
# Hard-coded to DeepJet algorithm, loose and medium WPs

# MC efficiencies
def GetMCeffFunc(year, wp='medium', btagalgo="btagDeepFlavB", flav='b'):
    if year not in clib_year_map.keys():
        raise Exception(f"Error: Unknown year \"{year}\".")
    if not year.startswith("202"):
        pathToBtagMCeff = topeft_path('data/btagSF/UL/btagMCeff_%s.pkl.gz' % year)
    else:
        pkltag = year[0:4]
        if btagalgo == "btagPNetB":
            pkltag += "_PNet"
        pathToBtagMCeff = topeft_path(f'data/btagSF/Run3/btagMCeff_{pkltag}.pkl.gz')

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

def GetBtagEff(jets, year, wp='medium', btagalgo="btagDeepFlavB"):
    if year not in clib_year_map.keys():
        raise Exception(f"Error: Unknown year \"{year}\".")
    result = GetMCeffFunc(year, wp, btagalgo)(jets.pt, np.abs(jets.eta), jets.hadronFlavour)
    return result

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

def ApplyJetCorrections(year, corr_type, isData, era, useclib=True, savelevels=False):
    usejecstack = not useclib

    if year not in clib_year_map.keys():
        raise Exception(f"Error: Unknown year \"{year}\".")

    jec_year = clib_year_map[year]
    if usejecstack:
        jec_tag = jerc_tag_map[year][0]
        jer_tag = jerc_tag_map[year][1]
        jet_algo = "AK4PFchs"
        extJEC = lookup_tools.extractor()
        weight_sets = []
        if not isData:
            weight_sets += [
                "* * " + topcoffea_path(f'data/JER/{jer_tag}_MC_SF_{jet_algo}.jersf.txt'),
                "* * " + topcoffea_path(f'data/JER/{jer_tag}_MC_PtResolution_{jet_algo}.jr.txt'),
            ]
        weight_sets += [
            "* * " + topcoffea_path(f'data/JEC/Summer19UL{jec_tag}_MC_L1FastJet_{jet_algo}.txt'),
            "* * " + topcoffea_path(f'data/JEC/Summer19UL{jec_tag}_MC_L2Relative_{jet_algo}.txt'),
        ]
        if not isData:
            weight_sets += [
                "* * " + topcoffea_path(f'data/JEC/Quad_Summer19UL{jec_tag}_MC_UncertaintySources_{jet_algo}.junc.txt')
            ]
        extJEC.add_weight_sets(weight_sets)
        jec_types = [
            'FlavorQCD', 'FlavorPureBottom', 'FlavorPureQuark', 'FlavorPureGluon', 'FlavorPureCharm',
            'BBEC1', 'Absolute', 'RelativeBal', 'RelativeSample'
        ]
        jec_regroup = [f"Quad_Summer19UL%s_MC_UncertaintySources_{jet_algo}_%s" % (jec_tag,jec_type) for jec_type in jec_types]
        jec_names = []
        if not isData:
            jec_names += [
                f"{jer_tag}_MC_SF_{jet_algo}",
                f"{jer_tag}_MC_PtResolution_{jet_algo}",
            ]
        jec_names += [
            f"Summer19UL{jec_tag}_MC_L1FastJet_{jet_algo}",
            f"Summer19UL{jec_tag}_MC_L2Relative_{jet_algo}",
        ]
        if not isData:
            jec_names.extend(jec_regroup)

        extJEC.finalize()
        JECevaluator = extJEC.make_evaluator()
        jec_inputs = {name: JECevaluator[name.replace("Regrouped_", "")] for name in jec_names}
        jec_stack = JECStack(jec_inputs)

    elif useclib:
        # Handle clib case
        jet_algo, jec_tag, jec_levels, jer_tag, junc_types = get_jerc_keys(year, isData, era)
        json_path = topcoffea_path(f"data/POG/JME/{jec_year}/jet_jerc.json.gz")

        # Create JECStack for clib scenario
        jec_stack = JECStack(
            jec_tag=jec_tag,
            jec_levels=jec_levels,
            jer_tag=jer_tag,
            jet_algo=jet_algo,
            junc_types=junc_types,
            json_path=json_path,
            use_clib=useclib,
            savecorr=savelevels
        )

    # Name map for jet or MET corrections
    name_map = {
        'JetPt': 'pt',
        'JetMass': 'mass',
        'JetEta': 'eta',
        'JetPhi': 'phi',
        'JetA': 'area',
        'ptGenJet': 'pt_gen',
        'ptRaw': 'pt_raw',
        'massRaw': 'mass_raw',
        'Rho': 'rho',
        'METpt': 'pt',
        'METphi': 'phi',
        'UnClusteredEnergyDeltaX': 'MetUnclustEnUpDeltaX',
        'UnClusteredEnergyDeltaY': 'MetUnclustEnUpDeltaY'
    }

    # Return appropriate factory based on correction type
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
    elif (syst_var in ['nominal','MuonESUp','MuonESDown', 'TESUp', 'TESDown', 'FESUp', 'FESDown']):
        return cleanedJets
    elif ('JES_FlavorQCD' in syst_var in syst_var):
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
    elif ('Up' in syst_var and syst_var[:-2].replace('APV', '') in cleanedJets.fields):
        return cleanedJets[syst_var.replace('Up', '').replace("Pile", "PileUp").replace('APV', '')].up
    elif ('Down' in syst_var and syst_var[:-4].replace('APV', '') in cleanedJets.fields):
        return cleanedJets[syst_var.replace('Down', '').replace('APV', '')].down
    else:
        raise Exception(f"Error: Unknown variation \"{syst_var}\".")

###### Muon Rochester corrections
################################################################
# https://gitlab.cern.ch/akhukhun/roccor
# https://github.com/CoffeaTeam/coffea/blob/master/coffea/lookup_tools/rochester_lookup.py
def ApplyRochesterCorrections(year, mu, is_data):
    if year.startswith('201'): #Run2 scenario
        rocco_tag = None
        if year == '2016':
            rocco_tag = "2016bUL"
        elif year == '2016APV':
            rocco_tag = "2016aUL"
        elif year == '2017':
            rocco_tag = "2017UL"
        elif year == '2018':
            rocco_tag = "2018UL"
        rochester_data = txt_converters.convert_rochester_file(topcoffea_path(f"data/MuonScale/RoccoR{rocco_tag}.txt"), loaduncs=True)
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
    else:
        corrections = ak.ones_like(mu.pt)
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

# helper function for the Run3 SFs
def ApplyBinnedSF(pt, edges, centers, unc, var):
    default = centers[-1] + var * unc
    sf = ak.full_like(pt, default)
    for low, high, cen in zip(edges[:-1], edges[1:], centers[:-1]):
        sf = ak.where((pt >= low) & (pt < high), cen + var * np.sqrt(unc**2 + (0.02*cen)*0.02), sf)
    return sf

# Vectorized Run3 SF functions for 2-lepton channels
def ComputeTriggerSFRun3(year, pdg0, pt0, pdg1, pt1, var=0):
    """
    A single dispatcher for all run3 variants:
    - year:       string, e.g. "2022", "2022EE", "2023", "2023BPix"
    - pdg0, pdg1: ak.Array of ints
    - pt0, pt1:   ak.Array of floats (conept)
    - var:        integer or array of ±1/0
    """
    # Only valid when exactly two leptons
    prod = abs(pdg0 * pdg1)
    out  = ak.ones_like(pt0)

    # define bins once
    edges = [20, 40, 50, 65, 80, 100, 200]

    # pick the right centers & uncertainties by year/channel
    # Format: (centers_ee, unc_ee, centers_em, unc_em, centers_mm, unc_mm)
    sf_defs = {
        "2022": (
            [1.0115, 1.0105, 1.0042, 0.9850, 1.0012, 1.0000, 1.0], 0.0146,   # ee
            [0.9850, 0.9889, 0.9885, 0.9717, 0.9674, 0.9679, 1.0], 0.0052,   # em
            [0.9881, 0.9944, 0.9937, 0.9868, 1.0022, 0.9841, 1.0], 0.0098    # mm
        ),
        "2022EE": (
            [0.9845, 1.0004, 1.0025, 0.9857, 0.9965, 1.0044, 1.0], 0.0037,
            [0.9833, 0.9818, 0.9841, 0.9806, 0.9777, 0.9807, 1.0], 0.0023,
            [0.9788, 0.9856, 0.9850, 0.9963, 0.9909, 0.9873, 1.0], 0.0039
        ),
        "2023": (
            [0.9453, 0.9791, 0.9953, 0.9822, 1.0025, 0.9948, 1.0], 0.0107,
            [0.9748, 0.9799, 0.9712, 0.9716, 0.9724, 0.9616, 1.0], 0.0028,
            [0.9821, 0.9936, 0.9941, 0.9863, 0.9905, 0.9786, 1.0], 0.0051
        ),
        "2023BPix": (
            [0.9672, 1.0001, 0.9852, 0.9928, 0.9981, 0.9954, 1.0], 0.0155,
            [0.9765, 0.9801, 0.9692, 0.9735, 0.9665, 0.9587, 1.0], 0.0041,
            [0.9890, 0.9956, 0.9869, 0.9907, 0.9950, 0.9646, 1.0], 0.0080
        ),
    }

    centers_ee, unc_ee, centers_em, unc_em, centers_mm, unc_mm = sf_defs[year]

    # apply ee (uses pt0) where |pdg0*pdg1|==121
    mask_ee = (prod == 121)
    sf_ee = ApplyBinnedSF(pt0, edges, centers_ee, unc_ee, var)
    out = ak.where(mask_ee, sf_ee, out)

    # apply em (uses pt1) for 143
    mask_em = (prod == 143)
    sf_em = ApplyBinnedSF(pt1, edges, centers_em, unc_em, var)
    out = ak.where(mask_em, sf_em, out)

    # apply mm (uses pt1) for 169
    mask_mm = (prod == 169)
    sf_mm = ApplyBinnedSF(pt1, edges, centers_mm, unc_mm, var)
    out = ak.where(mask_mm, sf_mm, out)

    return out


def GetTriggerSF(year, events, lep0, lep1):
    is_run3 = year.startswith("202")
    is_run2 = not is_run3

    pdg0     = lep0.pdgId
    pdg1     = lep1.pdgId
    conept0  = lep0.conept
    conept1  = lep1.conept

    #trigger SFs are applied only to the 2l events, since for >2l channels the trigger SFs are compatible with 1
    ls = []
    if is_run2:
        for syst in [0, 1]:
            # Run 2: fall back to the JSON‐loaded functions
            SF_ee = np.where(events.is2l & events.is_ee,
                             LoadTriggerSF(year, ch='2l', flav='ee')[syst](conept0, conept1),
                             1.0)
            SF_em = np.where(events.is2l & events.is_em,
                             LoadTriggerSF(year, ch='2l', flav='em')[syst](conept0, conept1),
                             1.0)
            SF_mm = np.where(events.is2l & events.is_mm,
                             LoadTriggerSF(year, ch='2l', flav='mm')[syst](conept0, conept1),
                             1.0)

            ls.append(SF_ee * SF_em * SF_mm)

        ls[1] = np.where(ls[1] == 1.0, 0.0, ls[1])
        sf_nominal = ls[0]
        sf_up = ls[0] + np.sqrt(ls[1]**2 + (0.02 * ls[0])**2)
        sf_down = ls[0] - np.sqrt(ls[1]**2 + (0.02 * ls[0])**2)
    else:
        # Run 3: vectorized awkward functions
        mask2l     = events.is2l
        sf_nominal = ak.where(mask2l,
                              ComputeTriggerSFRun3(year, pdg0, conept0, pdg1, conept1, var=0),
                              1.0)
        sf_up      = ak.where(mask2l,
                              ComputeTriggerSFRun3(year, pdg0, conept0, pdg1, conept1, var=1),
                              1.02)
        sf_down    = ak.where(mask2l,
                              ComputeTriggerSFRun3(year, pdg0, conept0, pdg1, conept1, var=-1),
                              0.98)

    events['trigger_sf'] = sf_nominal
    events['trigger_sfUp'] = sf_up
    events['trigger_sfDown'] = sf_down
