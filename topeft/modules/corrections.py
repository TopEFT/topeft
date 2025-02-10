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
    "2023": "2022_Summer23",
    "2023BPix": "2022_Summer23BPix",
}

egm_tag_map = {
    "2016preVFP_UL": "2016preVFP",
    "2016postVFP_UL": "2016postVFP",
    "2017_UL": "2017",
    "2018_UL": "2018",
    "2022_Summer22": "2022Re-recoBCD",
    "2022_Summer22EE": "2022Re-recoE+PromptFG",
    "2022_Summer23": "2023PromptC",
    "2022_Summer23BPix": "2023PromptD",
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
        if year in ['2016','2022','2023BPix']:
            jec_key = jerc_dict[year]['jec_data']
        else:
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
extPhoSF.add_weight_sets(["PhotonMediumSF_2016 EGamma_SF2D %s"    % topcoffea_path('data/photonSF/egammaEffi_EGM2D_Pho_Medium_UL16.root')])
extPhoSF.add_weight_sets(["PhotonMediumSF_2016_err EGamma_SF2D_err %s"    % topcoffea_path('data/photonSF/egammaEffi_EGM2D_Pho_Medium_UL16.root')])
extPhoSF.add_weight_sets(["PhotonMediumSF_2016APV EGamma_SF2D %s" % topcoffea_path('data/photonSF/egammaEffi_EGM2D_Pho_Medium_UL16_postVFP.root')])
extPhoSF.add_weight_sets(["PhotonMediumSF_2016APV_err EGamma_SF2D_err %s" % topcoffea_path('data/photonSF/egammaEffi_EGM2D_Pho_Medium_UL16_postVFP.root')])
extPhoSF.add_weight_sets(["PhotonMediumSF_2017 EGamma_SF2D %s"    % topcoffea_path('data/photonSF/egammaEffi_EGM2D_PHO_Medium_UL17.root')])
extPhoSF.add_weight_sets(["PhotonMediumSF_2017_err EGamma_SF2D_err %s"    % topcoffea_path('data/photonSF/egammaEffi_EGM2D_PHO_Medium_UL17.root')])
extPhoSF.add_weight_sets(["PhotonMediumSF_2018 EGamma_SF2D %s"    % topcoffea_path('data/photonSF/egammaEffi_EGM2D_Pho_Med_UL18.root')])
extPhoSF.add_weight_sets(["PhotonMediumSF_2018_statData statData %s"    % topcoffea_path('data/photonSF/egammaEffi_EGM2D_Pho_Med_UL18.root')])
extPhoSF.add_weight_sets(["PhotonMediumSF_2018_err EGamma_SF2D_err %s"    % topcoffea_path('data/photonSF/egammaEffi_EGM2D_Pho_Med_UL18.root')])
extPhoSF.add_weight_sets(["SF_CSEV_Medium_2016_postVFP MediumID/SF_CSEV_MediumID %s"    % topcoffea_path('data/photonSF/CSEV_SummaryPlot_UL16_postVFP.root')])
extPhoSF.add_weight_sets(["Staunc_CSEV_Medium_2016_postVFP MediumID/Staunc_CSEV_MediumID %s"    % topcoffea_path('data/photonSF/CSEV_SummaryPlot_UL16_postVFP.root')])
extPhoSF.add_weight_sets(["PUunc_CSEV_Medium_2016_postVFP MediumID/PUunc_CSEV_MediumID %s"    % topcoffea_path('data/photonSF/CSEV_SummaryPlot_UL16_postVFP.root')])
#Model only in UL2017 and 18 extPhoSF.add_weight_sets(["Modelunc_CSEV_Medium_16_postVFP MediumID/Modelunc_CSEV_MediumID %s"    % topcoffea_path('data/photonSF/CSEV_SummaryPlot_UL16_postVFP.root')])
extPhoSF.add_weight_sets(["SF_HasPix_Medium_2016_postVFP MediumID/SF_HasPix_MediumID %s"    % topcoffea_path('data/photonSF/HasPix_SummaryPlot_UL16_postVFP.root')])
extPhoSF.add_weight_sets(["Staunc_HasPix_Medium_2016_postVFP MediumID/Staunc_HasPix_MediumID %s"    % topcoffea_path('data/photonSF/HasPix_SummaryPlot_UL16_postVFP.root')])
extPhoSF.add_weight_sets(["PUunc_HasPix_Medium_2016_postVFP MediumID/PUunc_HasPix_MediumID %s"    % topcoffea_path('data/photonSF/HasPix_SummaryPlot_UL16_postVFP.root')])
#Model only in UL2017 and 18 extPhoSF.add_weight_sets(["Modelunc_HasPix_Medium_16_postVFP MediumID/Modelunc_HasPix_MediumID %s"    % topcoffea_path('data/photonSF/HasPix_SummaryPlot_UL16_postVFP.root')])
extPhoSF.add_weight_sets(["SF_CSEV_Medium_2016_preVFP MediumID/SF_CSEV_MediumID %s"    % topcoffea_path('data/photonSF/CSEV_SummaryPlot_UL16_preVFP.root')])
extPhoSF.add_weight_sets(["Staunc_CSEV_Medium_2016_preVFP MediumID/Staunc_CSEV_MediumID %s"    % topcoffea_path('data/photonSF/CSEV_SummaryPlot_UL16_preVFP.root')])
extPhoSF.add_weight_sets(["PUunc_CSEV_Medium_2016_preVFP MediumID/PUunc_CSEV_MediumID %s"    % topcoffea_path('data/photonSF/CSEV_SummaryPlot_UL16_preVFP.root')])
#Model only in UL2017 and 18 extPhoSF.add_weight_sets(["Modelunc_CSEV_Medium_16_preVFP MediumID/Modelunc_CSEV_MediumID %s"    % topcoffea_path('data/photonSF/CSEV_SummaryPlot_UL16_preVFP.root')])
extPhoSF.add_weight_sets(["SF_HasPix_Medium_2016_preVFP MediumID/SF_HasPix_MediumID %s"    % topcoffea_path('data/photonSF/HasPix_SummaryPlot_UL16_preVFP.root')])
extPhoSF.add_weight_sets(["Staunc_HasPix_Medium_2016_preVFP MediumID/Staunc_HasPix_MediumID %s"    % topcoffea_path('data/photonSF/HasPix_SummaryPlot_UL16_preVFP.root')])
extPhoSF.add_weight_sets(["PUunc_HasPix_Medium_2016_preVFP MediumID/PUunc_HasPix_MediumID %s"    % topcoffea_path('data/photonSF/HasPix_SummaryPlot_UL16_preVFP.root')])
#Model only in UL2017 and 18 extPhoSF.add_weight_sets(["Modelunc_HasPix_Medium_16_preVFP MediumID/Modelunc_HasPix_MediumID %s"    % topcoffea_path('data/photonSF/HasPix_SummaryPlot_UL16_preVFP.root')])
extPhoSF.add_weight_sets(["SF_CSEV_Medium_2017 MediumID/SF_CSEV_MediumID %s"    % topcoffea_path('data/photonSF/CSEV_SummaryPlot_UL17.root')])
extPhoSF.add_weight_sets(["Staunc_CSEV_Medium_2017 MediumID/Staunc_CSEV_MediumID %s"    % topcoffea_path('data/photonSF/CSEV_SummaryPlot_UL17.root')])
extPhoSF.add_weight_sets(["PUunc_CSEV_Medium_2017 MediumID/PUunc_CSEV_MediumID %s"    % topcoffea_path('data/photonSF/CSEV_SummaryPlot_UL17.root')])
extPhoSF.add_weight_sets(["Modelunc_CSEV_Medium_2017 MediumID/Modelunc_CSEV_MediumID %s"    % topcoffea_path('data/photonSF/CSEV_SummaryPlot_UL17.root')])
extPhoSF.add_weight_sets(["SF_HasPix_Medium_2017 MediumID/SF_HasPix_MediumID %s"    % topcoffea_path('data/photonSF/HasPix_SummaryPlot_UL17.root')])
extPhoSF.add_weight_sets(["Staunc_HasPix_Medium_2017 MediumID/Staunc_HasPix_MediumID %s"    % topcoffea_path('data/photonSF/HasPix_SummaryPlot_UL17.root')])
extPhoSF.add_weight_sets(["PUunc_HasPix_Medium_2017 MediumID/PUunc_HasPix_MediumID %s"    % topcoffea_path('data/photonSF/HasPix_SummaryPlot_UL17.root')])
extPhoSF.add_weight_sets(["Modelunc_HasPix_Medium_2017 MediumID/Modelunc_HasPix_MediumID %s"    % topcoffea_path('data/photonSF/HasPix_SummaryPlot_UL17.root')])
extPhoSF.add_weight_sets(["SF_CSEV_Medium_2018 MediumID/SF_CSEV_MediumID %s"    % topcoffea_path('data/photonSF/CSEV_SummaryPlot_UL18.root')])
extPhoSF.add_weight_sets(["Staunc_CSEV_Medium_2018 MediumID/Staunc_CSEV_MediumID %s"    % topcoffea_path('data/photonSF/CSEV_SummaryPlot_UL18.root')])
extPhoSF.add_weight_sets(["PUunc_CSEV_Medium_2018 MediumID/PUunc_CSEV_MediumID %s"    % topcoffea_path('data/photonSF/CSEV_SummaryPlot_UL18.root')])
extPhoSF.add_weight_sets(["Modelunc_CSEV_Medium_2018 MediumID/Modelunc_CSEV_MediumID %s"    % topcoffea_path('data/photonSF/CSEV_SummaryPlot_UL18.root')])
extPhoSF.add_weight_sets(["SF_HasPix_Medium_2018 MediumID/SF_HasPix_MediumID %s"    % topcoffea_path('data/photonSF/HasPix_SummaryPlot_UL18.root')])
extPhoSF.add_weight_sets(["Staunc_HasPix_Medium_2018 MediumID/Staunc_HasPix_MediumID %s"    % topcoffea_path('data/photonSF/HasPix_SummaryPlot_UL18.root')])
extPhoSF.add_weight_sets(["PUunc_HasPix_Medium_2018 MediumID/PUunc_HasPix_MediumID %s"    % topcoffea_path('data/photonSF/HasPix_SummaryPlot_UL18.root')])
extPhoSF.add_weight_sets(["Modelunc_HasPix_Medium_2018 MediumID/Modelunc_HasPix_MediumID %s"    % topcoffea_path('data/photonSF/HasPix_SummaryPlot_UL18.root')])

extPhoSF.finalize()
PhoSFevaluator = extPhoSF.make_evaluator()

ffSysts=['','_up','_down','_be1','_be2','_pt1','_pt2']

def ApplyTES(year, taus, isData, tagger, syst_name, vsJetWP):
    if isData:
        return (taus.pt, taus.mass)
    pt  = taus.pt
    dm  = taus.decayMode
    gen = taus.genPartFlav

    kinFlag = (pt>20) & (pt<205) & (gen==5)
    dmFlag = ((taus.decayMode==0) | (taus.decayMode==1) | (taus.decayMode==10) | (taus.decayMode==11))
    whereFlag = kinFlag & dmFlag #((pt>20) & (pt<205) & (gen==5) & (dm==0 | dm==1 | dm==10 | dm==11))
    tes = np.where(whereFlag, SFevaluator['TauTES_{year}'.format(year=year)](dm,pt), 1)

    ## Correction-lib implementation - MUST BE TESTED WHEN TAU IN THE MASTER BRANCH PROCESSOR
    padded_taus = ak.pad_none(taus,1)
    padded_taus = ak.with_name(padded_taus, "TauCandidate")

    clib_year = clib_year_map[year]
    json_path = topcoffea_path(f"data/POG/TAU/{clib_year}/tau.json.gz")
    ceval = correctionlib.CorrectionSet.from_file(json_path)

    pt  = padded_taus.pt
    dm  = padded_taus.decayMode
    wp  = padded_taus.idDeepTau2017v2p1VSjet
    eta = padded_taus.eta
    gen = padded_taus.genPartFlav
    mass= padded_taus.mass

    arg_tau = ["pt", "eta", "decayMode", "genPartFlav"]
    pt_mask_flat = ak.flatten((pt>0) & (pt<1000))

    ## Correction-lib implementation - MUST BE TESTED WHEN TAU IN THE MASTER BRANCH PROCESSOR
    deep_tau_cuts = [
        ("DeepTau2017v2p1VSjet", ak.flatten(padded_taus[f"is{vsJetWP}"]>0), ("pt", "decayMode", "genPartFlav", vsJetWP)),
        ("DeepTau2017v2p1VSe", ak.flatten(padded_taus["iseTight"]>0), ("eta", "genPartFlav", "VVLoose")),
        ("DeepTau2017v2p1VSmu", ak.flatten(padded_taus["ismTight"]>0), ("eta", "genPartFlav", "Loose")),
    ]

    DT_sf_list = []
    DT_up_list = []
    DT_do_list = []

    for deep_tau_cut in deep_tau_cuts:
        discr = deep_tau_cut[0]
        id_mask = deep_tau_cut[1]
        arg_list = deep_tau_cut[2]
        args = []
        for ttag in arg_list:
            args.append(
                ak.flatten(padded_taus[ttag])
            )
        tau_mask = id_mask & pt_mask_flat

        args_sf = args + ["sf"]
        DT_sf_list.append(
            ak.where(
                ~tau_mask,
                1,
                ceval[discr].evaluate(*args_sf)
            )
        )
        args_up = args + ["up"]
        DT_up_list.append(
            ak.where(
                ~tau_mask,
                1,
                ceval[discr].evaluate(*args_up)
            )
        )
        args_down = args + ["down"]
        DT_do_list.append(
            ak.where(
                ~tau_mask,
                1,
                ceval[discr].evaluate(*args_down)
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
            DT_sf_flat = DT_sf_discr
            DT_up_flat = DT_up_discr
            DT_do_flat = DT_do_discr
        else:
            DT_sf_flat *= DT_sf_discr
            DT_up_flat *= DT_up_discr
            DT_do_flat *= DT_do_discr

    DT_sf = ak.unflatten(DT_sf_flat, ak.num(pt))
    DT_up = ak.unflatten(DT_up_flat, ak.num(pt))
    DT_do = ak.unflatten(DT_do_flat, ak.num(pt))
    ## end of correction-lib implementation

    return (taus.pt*tes, taus.mass*tes)

def ApplyFES(year, taus, isData):
    if isData:
        return (taus.pt, taus.mass)

    eta  = taus.eta
    pt  = taus.pt
    dm  = taus.decayMode
    gen = taus.genPartFlav

    kinFlag = (pt>20) & (pt<205) & (gen==5)
    dmFlag = ((taus.decayMode==0) | (taus.decayMode==1))
    whereFlag = kinFlag & dmFlag
    fes = np.where(whereFlag, SFevaluator['TauFES_{year}'.format(year=year)](eta,dm), 1)
    return (taus.pt*fes, taus.mass*fes)

def ApplyTESSystematic(year, taus, syst_name):
    if not syst_name.startswith('TES'):
        return (taus.pt, taus.mass)

    pt  = taus.pt
    dm  = taus.decayMode
    gen = taus.genPartFlav

    kinFlag = (pt>20) & (pt<205) & (gen==5)
    dmFlag = ((taus.decayMode==0) | (taus.decayMode==1) | (taus.decayMode==10) | (taus.decayMode==11))
    whereFlag = kinFlag & dmFlag
    syst_lab = f'TauTES_{year}'

    if syst_name.endswith("Up"):
        syst_lab += '_up'
    elif syst_name.endswith("Down"):
        syst_lab += '_down'

    tes_syst = np.where(whereFlag, SFevaluator['TauTES_{year}'.format(year=year)](dm,pt), 1)
    return (taus.pt*tes_syst, taus.mass*tes_syst)

def ApplyFESSystematic(year, taus, syst_name):
    if not syst_name.startswith('FES'):
        return (taus.pt, taus.mass)

    pt  = taus.pt
    eta  = taus.eta
    dm  = taus.decayMode
    gen = taus.genPartFlav

    kinFlag = (pt>20) & (pt<205) & (gen==5)
    dmFlag = ((taus.decayMode==0) | (taus.decayMode==1))
    whereFlag = kinFlag & dmFlag

    syst_lab = f'TauFES_{year}'

    if syst_name.endswith("Up"):
        syst_lab += '_up'
    elif syst_name.endswith("Down"):
        syst_lab += '_down'

    fes_syst = np.where(whereFlag, SFevaluator['TauFES_{year}'.format(year=year)](eta,dm), 1)
    return (taus.pt*fes_syst, taus.mass*fes_syst)

def AttachTauSF(events, taus, year, vsJetWP="Loose"):
    padded_taus = ak.pad_none(taus,1)
    padded_taus = ak.with_name(padded_taus, "TauCandidate")
    padded_taus["sf_tau"] = 1.0
    padded_taus["sf_tau_up"] = 1.0
    padded_taus["sf_tau_down"] = 1.0

    clib_year = clib_year_map[year]
    json_path = topcoffea_path(f"data/POG/TAU/{clib_year}/tau.json.gz")
    ceval = correctionlib.CorrectionSet.from_file(json_path)

    pt  = padded_taus.pt
    dm  = padded_taus.decayMode
    wp  = padded_taus.idDeepTau2017v2p1VSjet
    eta = padded_taus.eta
    gen = padded_taus.genPartFlav
    mass= padded_taus.mass

    DT_sf_list = []
    DT_up_list = []
    DT_do_list = []

    pt_mask_flat = ak.flatten((pt>20) & (pt<205))

    ## Correction-lib implementation - MUST BE TESTED WHEN TAU IN THE MASTER BRANCH PROCESSOR
    deep_tau_cuts = [
        ("DeepTau2017v2p1VSjet", ak.flatten(padded_taus[f"is{vsJetWP}"]>0), ("pt", "decayMode", "genPartFlav", vsJetWP)),
        ("DeepTau2017v2p1VSe", ak.flatten(padded_taus["iseTight"]>0), ("eta", "genPartFlav", "VVLoose")),
        ("DeepTau2017v2p1VSmu", ak.flatten(padded_taus["ismTight"]>0), ("eta", "genPartFlav", "Loose")),
    ]

    for deep_tau_cut in deep_tau_cuts:
        discr = deep_tau_cut[0]
        id_mask = deep_tau_cut[1]
        arg_list = deep_tau_cut[2]
        args = []
        for ttag in arg_list:
            args.append(
                ak.flatten(padded_taus[ttag])
            )
        tau_mask = id_mask & pt_mask_flat

        args_sf = args + ["sf"]
        DT_sf_list.append(
            ak.where(
                ~tau_mask,
                1,
                ceval[discr].evaluate(*args_sf)
            )
        )
        args_up = args + ["up"]
        DT_up_list.append(
            ak.where(
                ~tau_mask,
                1,
                ceval[discr].evaluate(*args_up)
            )
        )
        args_down = args + ["down"]
        DT_do_list.append(
            ak.where(
                ~tau_mask,
                1,
                ceval[discr].evaluate(*args_down)
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
            DT_sf_flat = DT_sf_discr
            DT_up_flat = DT_up_discr
            DT_do_flat = DT_do_discr
        else:
            DT_sf_flat *= DT_sf_discr
            DT_up_flat *= DT_up_discr
            DT_do_flat *= DT_do_discr

    DT_sf = ak.unflatten(DT_sf_flat, ak.num(pt))
    DT_up = ak.unflatten(DT_up_flat, ak.num(pt))
    DT_do = ak.unflatten(DT_do_flat, ak.num(pt))
    ## end of correction-lib implementation

    ## legacy
    whereFlag = ((pt>20) & (pt<205) & (gen==5) & (padded_taus[f"is{vsJetWP}"]>0))
    real_sf_loose = np.where(whereFlag, SFevaluator[f'TauSF_{year}_{vsJetWP}'](dm,pt), 1)
    real_sf_loose_up = np.where(whereFlag, SFevaluator[f'TauSF_{year}_{vsJetWP}_up'](dm,pt), 1)
    real_sf_loose_down = np.where(whereFlag, SFevaluator[f'TauSF_{year}_{vsJetWP}_down'](dm,pt), 1)

    whereFlag = ((pt>20) & (pt<205) & ((gen==1)|(gen==3)) & (padded_taus["iseTight"]>0))
    fake_elec_sf = np.where(whereFlag, SFevaluator[f'Tau_elecFakeSF_{year}'](np.abs(eta)), 1)
    fake_elec_sf_up = np.where(whereFlag, SFevaluator[f'Tau_elecFakeSF_{year}_up'](np.abs(eta)), 1)
    fake_elec_sf_down = np.where(whereFlag, SFevaluator[f'Tau_elecFakeSF_{year}_down'](np.abs(eta)), 1)
    whereFlag = ((pt>20) & (pt<205) & ((gen==2)|(gen==4))  & (padded_taus["ismTight"]>0))
    fake_muon_sf = np.where(whereFlag, SFevaluator[f'Tau_muonFakeSF_{year}'](np.abs(eta)), 1)
    fake_muon_sf_up = np.where(whereFlag, SFevaluator[f'Tau_muonFakeSF_{year}_up'](np.abs(eta)), 1)
    fake_muon_sf_down = np.where(whereFlag, SFevaluator[f'Tau_muonFakeSF_{year}_down'](np.abs(eta)), 1)

    whereFlag = ((pt>20) & (pt<205) & (gen!=5) & (gen!=4) & (gen!=3) & (gen!=2) & (gen!=1) & (padded_taus["is{vsJetWP}"]>0))
    new_fake_sf = np.where(whereFlag, SFevaluator['TauFakeSF'](pt), 1)
    new_fake_sf_up = np.where(whereFlag, SFevaluator['TauFakeSF_up'](pt), 1)
    new_fake_sf_down = np.where(whereFlag, SFevaluator['TauFakeSF_down'](pt), 1)

    real_sf = real_sf_loose
    real_sf_up = real_sf_loose_up
    real_sf_down = real_sf_loose_down
    padded_taus["sf_tau"] = real_sf*fake_elec_sf*fake_muon_sf*new_fake_sf
    padded_taus["sf_tau_up"] = real_sf_up*fake_elec_sf_up*fake_muon_sf_up*new_fake_sf_up
    padded_taus["sf_tau_down"] = real_sf_down*fake_elec_sf_down*fake_muon_sf_down*new_fake_sf_down

    ## final step for correction-lib
    DT_sf *= new_fake_sf
    DT_up *= new_fake_sf_up
    DT_do *= new_fake_sf_down

    events["sf_2l_taus"] = padded_taus.sf_tau[:,0]
    events["sf_2l_taus_hi"] = padded_taus.sf_tau_up[:,0]
    events["sf_2l_taus_lo"] = padded_taus.sf_tau_down[:,0]

def AttachPerLeptonFR(leps, flavor, year):
    # Get the flip rates lookup object
    if year not in clib_year_map.keys():
        raise Exception(f"Error: Unknown year \"{year}\"\n"".")

    if year == "2016APV": flip_year_name = "UL16APV"
    elif year == "2016": flip_year_name = "UL16"
    elif year == "2017": flip_year_name = "UL17"
    elif year == "2018": flip_year_name = "UL18"
    else: flip_year_name = "UL18" #TO READAPT when fakefactors are ready #raise Exception(f"Not a known year: {year}")
    with gzip.open(topeft_path(f"data/fliprates/flip_probs_topcoffea_{flip_year_name}.pkl.gz")) as fin:
        flip_hist = pickle.load(fin)
        flip_lookup = lookup_tools.dense_lookup.dense_lookup(flip_hist.values()[()],[flip_hist.axes["pt"].edges,flip_hist.axes["eta"].edges])

    # Get the fliprate scaling factor for the given year
    chargeflip_sf = get_te_param("chargeflip_sf_dict")[flip_year_name]

    # For FR filepath naming conventions
    if '2016' in year:
        year = '2016APV_2016'
    elif year.startswith("202"):
        year = '2018'

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


def AddPerPhotonFR(events,ph,year,closureTest=False):
    # Get the fake rates lookup object
    if year == "2016APV": year_name = "UL16APV"
    elif year == "2016": year_name = "UL16"
    elif year == "2017": year_name = "UL17"
    elif year == "2018": year_name = "UL18"
    else: raise Exception(f"Not a known year: {year}")

    pt_edges = np.array([20,30,45,70,120])
    eta_edges = np.array([0,0.435,0.783,1.13,1.50])

    #NOTE for future: The FR and kMC numpy files will change in the future. The alphanumeric code in the path is also temporary and will eventually be removed.
    #We just need a single fake-rate file
    #fr_file = np.load(topeft_path(f"data/photon_fakerates_gyR6uGhvfy/fr_ph_{year_name}.npz"))
    fr_file = np.load(topeft_path(f"data/photon_fakerates_o7cANdcK3j/fr_ph_{year_name}.npz"))

    fr_value = fr_file[fr_file.files[0]]
    #We need to clip the underflow bins along both axes.
    fr_value = fr_value[1:,1:]
    fr_val_lookup = lookup_tools.dense_lookup.dense_lookup(fr_value,[pt_edges, eta_edges])
    ph['fr_val'] = (fr_val_lookup(ph.pt,abs(ph.eta)))

    fr_error = fr_file[fr_file.files[1]]
    fr_error = fr_error[1:,1:]
    fr_err_lookup = lookup_tools.dense_lookup.dense_lookup(fr_error,[pt_edges, eta_edges])
    ph['fr_err'] = (fr_err_lookup(ph.pt,abs(ph.eta)))

    #Depending on whether we are doing closure test or not, we need different kmc files
    if not closureTest:
        #kmc_file = np.load(topeft_path(f"data/photon_fakerates_gB29WFMqFb/kmc_ph_{year_name}.npz"))
        kmc_file = np.load(topeft_path(f"data/photon_fakerates_ZxY8lNRB2E/kmc_ph_{year_name}.npz"))

    else:
        #kmc_file = np.load(topeft_path(f"data/photon_fakerates_jeJHI2cDh5/kmc_ph_{year_name}.npz"))
        kmc_file = np.load(topeft_path(f"data/photon_fakerates_5QFHsLhmPF/kmc_ph_{year_name}.npz"))

    kmc_value = kmc_file[kmc_file.files[0]]
    #We need to clip the underflow bins along both axes.
    kmc_value = kmc_value[1:,1:]
    kmc_val_lookup = lookup_tools.dense_lookup.dense_lookup(kmc_value, [pt_edges, eta_edges])
    ph['kmc_val'] = (kmc_val_lookup(ph.pt,abs(ph.eta)))

    kmc_error = kmc_file[kmc_file.files[1]]
    #We need to clip the underflow bins along both axes.
    kmc_error = kmc_error[1:,1:]
    kmc_err_lookup = lookup_tools.dense_lookup.dense_lookup(kmc_error, [pt_edges, eta_edges])
    ph['kmc_err'] = (kmc_err_lookup(ph.pt,abs(ph.eta)))

    fakerate_val = (ph.fr_val * ph.kmc_val)
    fakerate_ph_val = ak.pad_none(fakerate_val, 1)

    fakerate_err = fakerate_val * np.sqrt(pow((ph.fr_err/ph.fr_val),2)+pow((ph.kmc_err/ph.kmc_val),2))
    fakerate_ph_err = ak.pad_none(fakerate_err, 1)

    fakerate_ph_val = fakerate_ph_val[:,0]  #ALERT: Fine for our case since we need exactly 1 photon but need to revisit this if we relax photon multiplicity cut in the future
    fakerate_ph_err = fakerate_ph_err[:,0]

    events['fakerate_ph_val'] = fakerate_ph_val
    events['fakerate_ph_err'] = fakerate_ph_err

#this is where we assign an additional 67% systematic uncertainty on the non-prompt photon yield in the photon pT distribution
#the 67% uncertainty was determined using a dedicated data-driven closure test for ABCD method (summed over all years)
def additional_nonprompt_ph_unc(events, last_bin_pt_mask):
    nominal = np.ones(len(events))
    #If last bin, scale the yield up/down by 15% and if other bins, keep nominal yield
    up_variation = ak.where(last_bin_pt_mask, nominal * 1.67, nominal)
    down_variation = ak.where(last_bin_pt_mask, nominal * 0.33, nominal)

    events['np_ph_up'] = up_variation
    events['np_ph_down'] = down_variation

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

def AttachElectronSF(electrons, year, looseWP=None):
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

    #initializing the sf and up/down array to 1, so they are fixed to unit values if the a given SF is not available yet
    ## Run2 and Run3
    reco_sf = ak.ones_like(pt)
    reco_up = ak.ones_like(pt)
    reco_do = ak.ones_like(pt)
    loose_sf = ak.ones_like(pt)
    loose_up = ak.ones_like(pt)
    loose_do = ak.ones_like(pt)
    ## Only Run2 for now
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

    eta_flat = ak.flatten(eta)
    pt_flat = ak.flatten(pt)
    phi_flat = ak.flatten(phi)

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

def AttachPhotonSF(photons, year):
    '''
      Description:
          Inserts 'sf_nom', 'sf_hi', and 'sf_lo' into the photons array passed to this function. These
          values correspond to the nominal, up, and down photon scalefactor values respectively.
    '''
    sieie = np.abs(photons.sieie)
    pt = photons.pt
    eta = photons.eta
    aeta = np.abs(photons.eta)
    r9 = photons.r9
    if year not in ['2016','2016APV','2017','2018']: raise Exception(f"Error: Unknown year \"{year}\".")
    photon_sf  = PhoSFevaluator['PhotonMediumSF_{year}'.format(year=year)](sieie,pt)
    photon_err = np.power(PhoSFevaluator['PhotonMediumSF_{year}_err'.format(year=year)](sieie,pt), 2)

    sf_year = year
    if '2016APV' in year:
        sf_year = '2016_preVFP'
    elif '2016' in year:
        sf_year = '2016_postVFP'
    '''
    High R9 [R9 > 0.96], Low R9 [R9 < 0.96]
    EB: |eta| < 1.5
    EE: 1.5 > |eta| > 2.5
    '''
    r9_cut  = 0.96
    eta_cut = 1.5
    EBhR9 = (1 * ak.ones_like(r9))
    EBlR9 = (2 * ak.ones_like(r9))
    EEhR9 = (4 * ak.ones_like(r9))
    EElR9 = (5 * ak.ones_like(r9))
    for sf_type in ['CSEV', 'HasPix']:
        EBlR9_sf = PhoSFevaluator[f'SF_{sf_type}_Medium_{sf_year}'](EBlR9)
        EBhR9_sf = PhoSFevaluator[f'SF_{sf_type}_Medium_{sf_year}'](EBhR9)
        EElR9_sf = PhoSFevaluator[f'SF_{sf_type}_Medium_{sf_year}'](EElR9)
        EEhR9_sf = PhoSFevaluator[f'SF_{sf_type}_Medium_{sf_year}'](EEhR9)
        tmp_sf = ak.ones_like(photon_sf)
        tmp_sf = ak.where((aeta < eta_cut) & (r9<r9_cut), EBlR9_sf[aeta<eta_cut], tmp_sf)
        tmp_sf = ak.where((aeta < eta_cut) & (r9>r9_cut), EBhR9_sf[aeta<eta_cut], tmp_sf)
        tmp_sf = ak.where((aeta > eta_cut) & (r9<r9_cut), EElR9_sf[aeta<eta_cut], tmp_sf)
        tmp_sf = ak.where((aeta > eta_cut) & (r9>r9_cut), EEhR9_sf[aeta<eta_cut], tmp_sf)
        photon_sf = photon_sf * tmp_sf
        for syst_type in ['Staunc', 'PUunc', 'Modelunc']:
            if '16' in sf_year and 'Model' in syst_type:
                #Model only in UL17 and 18
                continue
            tmp_err = ak.ones_like(photon_err)
            tmp_err  = ak.where((aeta < eta_cut) & (r9<r9_cut),
                                PhoSFevaluator[f'{syst_type}_{sf_type}_Medium_{sf_year}'](EBlR9)[aeta<eta_cut], tmp_err)
            tmp_err  = ak.where((aeta < eta_cut) & (r9>r9_cut),
                                PhoSFevaluator[f'{syst_type}_{sf_type}_Medium_{sf_year}'](EBhR9)[aeta<eta_cut], tmp_err)
            tmp_err  = ak.where((aeta > eta_cut) & (r9<r9_cut),
                                PhoSFevaluator[f'{syst_type}_{sf_type}_Medium_{sf_year}'](EElR9)[aeta<eta_cut], tmp_err)
            tmp_err  = ak.where((aeta > eta_cut) & (r9>r9_cut),
                                PhoSFevaluator[f'{syst_type}_{sf_type}_Medium_{sf_year}'](EEhR9)[aeta<eta_cut], tmp_err)
            photon_err = photon_err + np.power(tmp_err, 2)

    photons['sf_nom_photon'] = photon_sf
    photons['sf_hi_photon']  = photon_sf + np.sqrt(photon_err)
    photons['sf_lo_photon']  = photon_sf - np.sqrt(photon_err)

###### Btag scale factors
################################################################
# Hard-coded to DeepJet algorithm, loose and medium WPs

# MC efficiencies
def GetMCeffFunc(year, wp='medium', flav='b'):
    if year not in clib_year_map.keys():
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
    if year not in clib_year_map.keys():
        raise Exception(f"Error: Unknown year \"{year}\".")
    result = GetMCeffFunc(year,wp)(jets.pt, np.abs(jets.eta), jets.hadronFlavour) if year[2] != "2" else ak.ones_like(jets.pt)
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
    elif (syst_var in ['nominal','MuonESUp','MuonESDown']):
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

def GetTriggerSF(year, events, lep0, lep1):
    is_run3 = False
    if year.startswith("202"):
        is_run3 = True
    is_run2 = not is_run3

    ls = []
    for syst in [0,1]:
        #2l
        if is_run2:
            SF_ee = np.where((events.is2l & events.is_ee), LoadTriggerSF(year,ch='2l',flav='ee')[syst](lep0.pt,lep1.pt), 1.0)
            SF_em = np.where((events.is2l & events.is_em), LoadTriggerSF(year,ch='2l',flav='em')[syst](lep0.pt,lep1.pt), 1.0)
            SF_mm = np.where((events.is2l & events.is_mm), LoadTriggerSF(year,ch='2l',flav='mm')[syst](lep0.pt,lep1.pt), 1.0)
        elif is_run3:
            SF_ee = ak.ones_like(events.is2l)
            SF_em = ak.ones_like(events.is2l)
            SF_mm = ak.ones_like(events.is2l)
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
