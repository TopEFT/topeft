# This file is essentially a wrapper for createJSON.py:
#   - It runs createJSON.py for each sample that you include in a dictionary, and moves the resulting json file to the directory you specify
#   - If the private NAOD has to be remade, the version numbers should be updated in the dictionaries here, then just rerun the script to remake the jsons

import json
import subprocess
import os
from topcoffea.modules.paths import topcoffea_path
from topcoffea.modules.samples import loadxsecdic
from topcoffea.modules.combine_json_ext  import combine_json_ext

########### The XSs from xsec.cfg ###########
XSECDIC = loadxsecdic("../../topcoffea/cfg/xsec.cfg",True)

########### Private UL signal samples ###########

private_UL17_dict = {


    "UL17_ttHJet_b1"   : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v4/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttHJet_privateUL17",
        "xsecName": "ttHnobb",
    },
    "UL17_ttHJet_b2"   : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch2/naodOnly_step/v3/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttHJet_privateUL17",
        "xsecName": "ttHnobb",
    },
    "UL17_ttHJet_b3"   : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch3/naodOnly_step/v4/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttHJet_privateUL17",
        "xsecName": "ttHnobb",
    },

    "UL17_ttlnuJet_b1" : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v4/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttlnuJet_privateUL17",
        "xsecName": "TTWJetsToLNu",
    },
    "UL17_ttlnuJet_b2" : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch2/naodOnly_step/v3/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttlnuJet_privateUL17",
        "xsecName": "TTWJetsToLNu",
    },
    "UL17_ttlnuJet_b3" : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch3/naodOnly_step/v4/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttlnuJet_privateUL17",
        "xsecName": "TTWJetsToLNu",
    },

    "UL17_ttllJet_b1"  : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v4/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttllJet_privateUL17",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL17_ttllJet_b2"  : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch2/naodOnly_step/v3/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttllJet_privateUL17",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL17_ttllJet_b3"  : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch3/naodOnly_step/v4/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttllJet_privateUL17",
        "xsecName": "TTZToLLNuNu_M_10",
    },

    "UL17_tllq_b1"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v4/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
        "histAxisName" : "tllq_privateUL17",
        "xsecName": "tZq",
    },
    "UL17_tllq_b2"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch2/naodOnly_step/v3/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
        "histAxisName": "tllq_privateUL17",
        "xsecName": "tZq",
    },
    "UL17_tllq_b3"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch3/naodOnly_step/v4/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
        "histAxisName": "tllq_privateUL17",
        "xsecName": "tZq",
    },

    "UL17_tHq_b1"      : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v4/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "tHq_privateUL17",
        "xsecName": "tHq",
    },
    "UL17_tHq_b2"      : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch2/naodOnly_step/v3/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "tHq_privateUL17",
        "xsecName": "tHq",
    },
    "UL17_tHq_b3"      : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch3/naodOnly_step/v4/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "tHq_privateUL17",
        "xsecName": "tHq",
    },

    "UL17_tttt_b4"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch4/naodOnly_step/v2/nAOD_step_tttt_FourtopsMay3v1_run0",
        "histAxisName": "tttt_privateUL17",
        "xsecName": "tttt",
    },
}

private_UL18_dict = {

    "UL18_ttHJet_b1"   : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch1/naodOnly_step/v5/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttHJet_privateUL18",
        "xsecName": "ttHnobb",
    },
    "UL18_ttHJet_b2"   : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch2/naodOnly_step/v2/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttHJet_privateUL18",
        "xsecName": "ttHnobb",
    },
    "UL18_ttHJet_b3"   : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch3/naodOnly_step/v2/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttHJet_privateUL18",
        "xsecName": "ttHnobb",
    },

    "UL18_ttlnuJet_b1" : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch1/naodOnly_step/v5/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttlnuJet_privateUL18",
        "xsecName": "TTWJetsToLNu",
    },
    "UL18_ttlnuJet_b2" : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch2/naodOnly_step/v2/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttlnuJet_privateUL18",
        "xsecName": "TTWJetsToLNu",
    },
    "UL18_ttlnuJet_b3" : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch3/naodOnly_step/v2/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttlnuJet_privateUL18",
        "xsecName": "TTWJetsToLNu",
    },

    "UL18_ttllJet_b1"  : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch1/naodOnly_step/v5/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttllJet_privateUL18",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL18_ttllJet_b2"  : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch2/naodOnly_step/v2/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttllJet_privateUL18",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL18_ttllJet_b3"  : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch3/naodOnly_step/v2/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttllJet_privateUL18",
        "xsecName": "TTZToLLNuNu_M_10",
    },

    "UL18_tllq_b1"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch1/naodOnly_step/v5/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
        "histAxisName": "tllq_privateUL18",
        "xsecName": "tZq",
    },
    "UL18_tllq_b2"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch2/naodOnly_step/v2/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
        "histAxisName": "tllq_privateUL18",
        "xsecName": "tZq",
    },
    "UL18_tllq_b3"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch3/naodOnly_step/v2/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
        "histAxisName": "tllq_privateUL18",
        "xsecName": "tZq",
    },

    "UL18_tHq_b1"      : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch1/naodOnly_step/v5/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "tHq_privateUL18",
        "xsecName": "tHq",
    },
    "UL18_tHq_b2"      : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch2/naodOnly_step/v2/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "tHq_privateUL18",
        "xsecName": "tHq",
    },
    "UL18_tHq_b3"      : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch3/naodOnly_step/v2/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "tHq_privateUL18",
        "xsecName": "tHq",
    },

    "UL18_tttt_b4"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch4/naodOnly_step/v2/nAOD_step_tttt_FourtopsMay3v1_run0",
        "histAxisName": "tttt_privateUL18",
        "xsecName": "tttt",
    },
}

private_UL16_dict = {

    "UL16_ttHJet_b1"   : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16/Round1/Batch1/naodOnly_step/v2/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttHJet_privateUL16",
        "xsecName": "ttHnobb",
    },
    "UL16_ttlnuJet_b1" : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16/Round1/Batch1/naodOnly_step/v2/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttlnuJet_privateUL16",
        "xsecName": "TTWJetsToLNu",
    },
    "UL16_ttllJet_b1"  : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16/Round1/Batch1/naodOnly_step/v2/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttllJet_privateUL16",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL16_tllq_b1"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16/Round1/Batch1/naodOnly_step/v2/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
        "histAxisName": "tllq_privateUL16",
        "xsecName": "tZq",
    },
    "UL16_tHq_b1"      : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16/Round1/Batch1/naodOnly_step/v2/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "tHq_privateUL16",
        "xsecName": "tHq",
    },
    "UL16_tttt_b1"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16/Round1/Batch1/naodOnly_step/v2/nAOD_step_tttt_FourtopsMay3v1_run0",
        "histAxisName": "tttt_privateUL16",
        "xsecName": "tttt",
    },
}

private_UL16APV_dict = {

    "UL16APV_ttHJet_b1"   : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16APV/Round1/Batch1/naodOnly_step/v2/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttHJet_privateUL16APV",
        "xsecName": "ttHnobb",
    },
    "UL16APV_ttlnuJet_b1" : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16APV/Round1/Batch1/naodOnly_step/v2/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttlnuJet_privateUL16APV",
        "xsecName": "TTWJetsToLNu",
    },
    "UL16APV_ttllJet_b1"  : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16APV/Round1/Batch1/naodOnly_step/v2/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttllJet_privateUL16APV",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL16APV_tllq_b1"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16APV/Round1/Batch1/naodOnly_step/v2/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
        "histAxisName": "tllq_privateUL16APV",
        "xsecName": "tZq",
    },
    "UL16APV_tHq_b1"      : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16APV/Round1/Batch1/naodOnly_step/v2/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "tHq_privateUL16APV",
        "xsecName": "tHq",
    },
    "UL16APV_tttt_b1"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16APV/Round1/Batch1/naodOnly_step/v2/nAOD_step_tttt_FourtopsMay3v1_run0",
        "histAxisName": "tttt_privateUL16APV",
        "xsecName": "tttt",
    },
}


########### TOP-19-001 samples and subsets of UL17 locally at ND on /scratch365 and at UNL ###########

private_2017_dict = {
    "2017_ttHJet" : {
        "path" : "/scratch365/kmohrman/mc_files/TOP-19-001/ttH/",
        "histAxisName": "ttH_private2017",
        "xsecName": "ttHnobb",
    },
    "2017_ttlnuJet" : {
        "path" : "/scratch365/kmohrman/mc_files/TOP-19-001/ttlnu/",
        "histAxisName": "ttlnu_private2017",
        "xsecName": "TTWJetsToLNu",
    },
    "2017_ttllJet" : {
        "path" : "/scratch365/kmohrman/mc_files/TOP-19-001/ttll/",
        "histAxisName": "ttll_private2017",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "2017_tllqJet" : {
        "path" : "/scratch365/kmohrman/mc_files/TOP-19-001/tllq/",
        "histAxisName": "tllq_private2017",
        "xsecName": "tZq",
    },
    "2017_tHqJet" : {
        "path" : "/scratch365/kmohrman/mc_files/TOP-19-001/tHq/",
        "histAxisName": "tHq_private2017",
        "xsecName": "tHq",
    },
}

private_UL17_dict_b1b4_local = {
    "UL17_ttHJet" : {
        "path" : "/scratch365/kmohrman/mc_files/full_r2_files/round1/ttH_b1/",
        "histAxisName": "ttHJet_privateUL17",
        "xsecName": "ttHnobb",
    },
    "UL17_ttlnuJet" : {
        "path" : "/scratch365/kmohrman/mc_files/full_r2_files/round1/ttlnu_b1/",
        "histAxisName": "ttlnuJet_privateUL17",
        "xsecName": "TTWJetsToLNu",
    },
    "UL17_ttllJet" : {
        "path" : "/scratch365/kmohrman/mc_files/full_r2_files/round1/ttll_b1/",
        "histAxisName": "ttllJet_privateUL17",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL17_tllq" : {
        "path" : "/scratch365/kmohrman/mc_files/full_r2_files/round1/tllq_b1/",
        "histAxisName": "tllq_privateUL17",
        "xsecName": "tZq",
    },
    "UL17_tHq" : {
        "path" : "/scratch365/kmohrman/mc_files/full_r2_files/round1/tHq_b1/",
        "histAxisName": "tHq_privateUL17",
        "xsecName": "tHq",
    },
    "UL17_tttt" : {
        "path" : "/scratch365/kmohrman/mc_files/full_r2_files/round1/tttt_b4/",
        "histAxisName": "tttt_privateUL17",
        "xsecName": "tttt",
    },
}

private_UL17_dict_b1b4_UNL = {
    "UL17_ttHJet" : {
        "path" : "/store/user/awightma/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v2/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttHJet_privateUL17",
        "xsecName": "ttHnobb",
    },
    "UL17_ttlnuJet" : {
        "path" : "/store/user/awightma/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v2/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttlnuJet_privateUL17",
        "xsecName": "TTWJetsToLNu",
    },
    "UL17_ttllJet" : {
        "path" : "/store/user/awightma/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v2/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttllJet_privateUL17",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL17_tllqJet" : {
        "path" : "/store/user/awightma/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v2/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
        "histAxisName": "tllq_privateUL17",
        "xsecName": "tZq",
    },
    "UL17_tHqJet" : {
        "path" : "/store/user/awightma/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v2/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "tHq_privateUL17",
        "xsecName": "tHq",
    },
    "UL17_tttt" : {
        "path" : "/store/user/awightma/FullProduction/FullR2/UL17/Round1/Batch4/naodOnly_step/v1/nAOD_step_tttt_FourtopsMay3v1_run0",
        "histAxisName": "tttt_privateUL17",
        "xsecName": "tttt",
    },
}


########### Central signal samples ###########

sync_dict = {
    "ttHJetToNonbb_sync" : {
        "path" : "/store/mc/RunIIFall17NanoAODv7/ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1/110000/BB506088-A858-A24D-B27C-0D31058D3125.root",
        "histAxisName": "ttHJetToNonbb_sync",
        "xsecName": "ttHnobb",
    },
}

central_2017_correctnPartonsInBorn_dict = {
    "2017_TTZToLLNuNu_M_10_correctnPartonsInBorn" : {
        "path" : "/TTZToLLNuNu_M-10_TuneCP5_PSweights_correctnPartonsInBorn_13TeV-amcatnlo-pythia8/RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/NANOAODSIM",
        "histAxisName": "ttZ_central2017_correctnPartonsInBorn",
        "xsecName" : "tZq",
    }
}

central_2017_dict = {
    "2017_ttHnobb" : {
        "path" : "/ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1/NANOAODSIM",
        "histAxisName": "ttH_central2017",
        "xsecName": "ttHnobb",
    },
    "2017_TTWJetsToLNu" : {
        "path" : "/TTWJetsToLNu_TuneCP5_PSweights_13TeV-amcatnloFXFX-madspin-pythia8/RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1/NANOAODSIM",
        "histAxisName": "ttW_central2017",
        "xsecName": "TTWJetsToLNu",
    },
    "2017_TTZToLLNuNu_M_10" : {
        "path" : "/TTZToLLNuNu_M-10_TuneCP5_PSweights_13TeV-amcatnlo-pythia8/RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/NANOAODSIM",
        "histAxisName": "ttZ_central2017",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "2017_tZq" : {
        "path" : "/tZq_ll_4f_ckm_NLO_TuneCP5_PSweights_13TeV-amcatnlo-pythia8/RunIIFall17NanoAOD-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/NANOAODSIM",
        "histAxisName": "tZq_central2017",
        "xsecName": "tZq",
    },
    "2017_THQ" : {
        "path" : "/THQ_ctcvcp_4f_Hincl_13TeV_madgraph_pythia8/RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/NANOAODSIM",
        "histAxisName": "tHq_central2017",
        "xsecName": "tHq",
    },
    "2017_TTTT" : {
        "path" : "/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/NANOAODSIM",
        "histAxisName": "tttt_central2017",
        "xsecName": "tttt",
    },
}

central_UL16_dict = {
    "UL16_ttHnobb" : {
        "path" : "/ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "histAxisName": "ttHJet_centralUL16",
        "xsecName": "ttHnobb",
    },
}

central_UL16APV_dict = {
    "UL16APV_ttHnobb" : {
        "path" : "/ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL16NanoAODAPVv2-106X_mcRun2_asymptotic_preVFP_v9-v1/NANOAODSIM",
        "histAxisName": "ttHJet_centralUL16APV",
        "xsecName": "ttHnobb",
    },
}

central_UL17_dict = {
    "UL17_ttHnobb" : {
        "path" : "/ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL17NanoAODv2-106X_mc2017_realistic_v8-v1/NANOAODSIM",
        "histAxisName": "ttH_centralUL17",
        "xsecName": "ttHnobb",
    },
    "UL17_TTWJetsToLNu" : {
        "path" : "/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL17NanoAODv2-106X_mc2017_realistic_v8-v1/NANOAODSIM",
        "histAxisName": "ttW_centralUL17",
        "xsecName": "TTWJetsToLNu",
    },
    "UL17_TTZToLLNuNu_M_10" : {
        "path" : "/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv2-106X_mc2017_realistic_v8-v1/NANOAODSIM",
        "histAxisName": "ttZ_centralUL17",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL17_tZq" : {
        "path" : "/tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer19UL17NanoAODv2-106X_mc2017_realistic_v8-v1/NANOAODSIM",
        "histAxisName": "tZq_centralUL17",
        "xsecName": "tZq",
    },
}

central_UL18_dict = {
    "UL18_ttHnobb" : {
        "path" : "/ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM",
        "histAxisName": "ttH_centralUL18",
        "xsecName": "ttHnobb",
    },
    "UL18_TTWJetsToLNu" : {
        "path" : "/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM", # Could update to v16?
        "histAxisName": "ttW_centralUL18",
        "xsecName": "TTWJetsToLNu",
    },
    "UL18_TTZToLLNuNu_M_10" : {
        "path" : "/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM",
        "histAxisName": "ttZ_centralUL18",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL18_tZq" : {
        "path" : "/tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer19UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM",
        "histAxisName": "tZq_centralUL18",
        "xsecName": "tZq",
    },
}


########### Central background samples ###########

central_UL17_bkg_dict = {
    "UL17_DY10to50" : {
        "path" : "/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "histAxisName": "DYJetsToLL_centralUL17",
        "xsecName": "DYJetsToLL_M_10to50_MLM",
    },
    "UL17_DY50" : {
        "path" : "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "histAxisName": "DYJetsToLL_centralUL17",
        "xsecName": "DYJetsToLL_M_50_MLM",
    },
    "UL17_ST_top_s-channel" : {
        "path" : "/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "histAxisName": "ST_top_s-channel_centralUL17",
        "xsecName": "ST_top_s-channel",
    },
    "UL17_ST_top_t-channel" : {
        "path" : "/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM", 
        "histAxisName": "ST_top_t-channel_centralUL17",
        "xsecName": "ST_top_t-channel",
    },
    "UL17_ST_antitop_t-channel" : {
        "path" : "/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM", 
        "histAxisName": "ST_antitop_t-channel_centralUL17",
        "xsecName": "ST_antitop_t-channel",
    },
    "UL17_tbarW" : {
        "path" : "/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "histAxisName": "tbarW_centralUL17",
        "xsecName": "ST_tW_antitop_5f_inclusiveDecays",
    },
    "UL17_tW" : {
        "path" : "/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "histAxisName": "tW_centralUL17",
        "xsecName": "ST_tW_antitop_5f_inclusiveDecays",
    },
    "UL17_TTGJets" : {
        "path" : "/TTGJets_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "histAxisName": "TTGJets_centralUL17",
        "xsecName": "TTGJets",
    },
    "UL17_TTJets" : {
        "path" : "/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "histAxisName": "TTJets_centralUL17",
        "xsecName": "TT",
    },
    "UL17_WJetsToLNu" : {
        "path" : "/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "histAxisName": "WJetsToLNu_centralUL17",
        "xsecName": "WJetsToLNu",
    },
    "UL17_WWTo2L2Nu" : {
        "path" : "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "histAxisName": "WWTo2L2Nu_centralUL17",
        "xsecName": "WWTo2L2Nu",
    },
    "UL17_WWW_4F" : {
        "path" : "/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "histAxisName": "WWW_centralUL17",
        "xsecName": "WWW",
    },
    "UL17_WWW_4F_ext" : {
        "path" : "/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9_ext1-v2/NANOAODSIM",
        "histAxisName": "WWW_centralUL17",
        "xsecName": "WWW",
    },
    "UL17_WWZ_4F" : {
        "path" : "/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "histAxisName": "WWZ_centralUL17",
        "xsecName": "WWZ",
    },
    "UL17_WZTo3LNu" : {
        "path" : "/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "histAxisName": "WZTo3LNu_centralUL17",
        "xsecName": "WZTo3LNu",
    },
    "UL17_WZZ" : {
        "path" : "/WZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "histAxisName": "WZZ_centralUL17",
        "xsecName": "WZZ",
    },
    "UL17_WZZ_ext" : {
        "path" : "/WZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9_ext1-v2/NANOAODSIM",
        "histAxisName": "WZZ_centralUL17",
        "xsecName": "WZZ",
    },
    "UL17_ZZTo4L" : {
        "path" : "/ZZTo4L_13TeV_powheg_pythia8/RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1/NANOAODSIM", # NOTE: PLACEHOLDER till a UL sample is available (last checked Jan 17, 2022)
        "histAxisName": "ZZTo4L_centralUL17",
        "xsecName": "ZZTo4L",
    },
    "UL17_ZZZ" : {
        "path" : "/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "histAxisName": "ZZZ_centralUL17",
        "xsecName": "ZZZ",
    },
    "UL17_ZZZ_ext" : {
        "path" : "/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9_ext1-v2/NANOAODSIM",
        "histAxisName": "ZZZ_centralUL17",
        "xsecName": "ZZZ",
    },
}


central_UL18_bkg_dict = {
    "UL18_DY10to50" : {
        "path" : "/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "histAxisName": "DY10to50_centralUL18",
        "xsecName": "DYJetsToLL_M_10to50_MLM",
    },
    "UL18_DY50" : {
        "path" : "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "histAxisName": "DY50_centralUL18",
        "xsecName": "DYJetsToLL_M_50_MLM",
    },
    "UL18_ST_top_s-channel" : {
        "path" : "/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "histAxisName": "ST_top_s-channel_centralUL18",
        "xsecName": "ST_top_s-channel",
    },
    "UL18_ST_top_t-channel" : {
        "path" : "/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM", 
        "histAxisName": "ST_top_t-channel_centralUL18",
        "xsecName": "ST_top_t-channel",
    },
    "UL18_ST_antitop_t-channel" : {
        "path" : "/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM", 
        "histAxisName": "ST_antitop_t-channel_centralUL18",
        "xsecName": "ST_antitop_t-channel",
    },
    "UL18_tbarW" : {
        "path" : "/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "histAxisName": "tbarW_centralUL18",
        "xsecName": "ST_tW_antitop_5f_inclusiveDecays",
    },
    "UL18_tW" : {
        "path" : "/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "histAxisName": "tW_centralUL18",
        "xsecName": "ST_tW_top_5f_inclusiveDecays",
    },
    "UL18_TTGJets" : {
        "path" : "/TTGJets_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "histAxisName": "TTGJets_centralUL18",
        "xsecName": "TTGJets",
    },
    "UL18_TTJets" : {
        "path" : "/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM", # NOTE: PLACEHOLDER till v9 is available (last checked Jan 17, 2022)
        "histAxisName": "TTJets_centralUL18",
        "xsecName": "TT",
    },
    "UL18_WJetsToLNu" : {
        "path" : "/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "histAxisName": "WJetsToLNu_centralUL18",
        "xsecName": "WJetsToLNu",
    },
    "UL18_WWTo2L2Nu" : {
        "path" : "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "histAxisName": "WWTo2L2Nu_centralUL18",
        "xsecName": "WWTo2L2Nu",
    },
    "UL18_WWW_4F" : {
        "path" : "/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "histAxisName": "WWW_4F_centralUL18",
        "xsecName": "WWW",
    },
    "UL18_WWZ_4F" : {
        "path" : "/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "histAxisName": "WWZ_4F_centralUL18",
        "xsecName": "WWZ",
    },
    "UL18_WZTo3LNu" : {
        "path" : "/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "histAxisName": "WZTo3LNu_centralUL18",
        "xsecName": "WZTo3LNu",
    },
    "UL18_WZZ" : {
        "path" : "/WZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "histAxisName": "WZZ_centralUL18",
        "xsecName": "WZZ",
    },
    "UL18_WZZ_ext" : {
        "path" : "/WZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1_ext1-v2/NANOAODSIM",
        "histAxisName": "WZZ_ext_centralUL18",
        "xsecName": "WZZ",
    },
    "UL18_ZZTo4L" : {
        "path" : "/ZZTo4L_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM", # NOTE: PLACEHOLDER till v9 is available
        "histAxisName": "ZZTo4L_centralUL18",
        "xsecName": "ZZTo4L",
    },
    "UL18_ZZZ" : {
        "path" : "/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "histAxisName": "ZZZ_centralUL18",
        "xsecName": "ZZZ",
    },
}


central_UL16_bkg_dict = {
    "UL16_DY10to50" : {
        "path" : "/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv2-106X_mcRun2_asymptotic_v15-v1/NANOAODSIM", # NOTE: PLACEHOLDER till v9 is available (last checked Jan 17, 2022)
        "histAxisName": "DY10to50_centralUL16",
        "xsecName": "DYJetsToLL_M_10to50_MLM",
    },
    "UL16_DY50" : {
        "path" : "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM", 
        "histAxisName": "DY50_centralUL16",
        "xsecName": "DYJetsToLL_M_50_MLM",
    },
    "UL16_ST_top_s-channel" : {
        "path" : "/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv2-106X_mcRun2_asymptotic_v15-v1/NANOAODSIM", # NOTE: PLACEHOLDER till v9 is available (last checked Jan 17, 2022)
        "histAxisName": "ST_top_s-channel_centralUL16",
        "xsecName": "ST_top_s-channel",
    },
    "UL16_ST_top_t-channel" : {
        "path" : "/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "histAxisName": "ST_top_t-channel_centralUL16",
        "xsecName": "ST_top_t-channel",
    },
    "UL16_ST_antitop_t-channel" : {
        "path" : "/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "histAxisName": "ST_antitop_t-channel_centralUL16",
        "xsecName": "ST_antitop_t-channel",
    },
    "UL16_tbarW" : {
        "path" : "/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
        "histAxisName": "tbarW_centralUL16",
        "xsecName": "ST_tW_antitop_5f_inclusiveDecays",
    },
    "UL16_tW" : {
        "path" : "/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
        "histAxisName": "tW_centralUL16",
        "xsecName": "ST_tW_top_5f_inclusiveDecays",
    },
    "UL16_TTGJets" : {
        "path" : "/TTGJets_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "histAxisName": "TTGJets_centralUL16",
        "xsecName": "TTGJets",
    },
    "UL16_TTJets" : {
        "path" : "/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv2-106X_mcRun2_asymptotic_v15-v1/NANOAODSIM", # NOTE: PLACEHOLDER till v9 is available (last checked Jan 17, 2022)
        "histAxisName": "TTJets_centralUL16",
        "xsecName": "TT",
    },
    "UL16_WWTo2L2Nu" : {
        "path" : "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "histAxisName": "WWTo2L2Nu_centralUL16",
        "xsecName": "WWTo2L2Nu",
    },
    "UL16_WJetsToLNu" : {
        "path" : "/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "histAxisName": "WJetsToLNu_centralUL16",
        "xsecName": "WJetsToLNu",
    },
    "UL16_WWW_4F" : {
        "path" : "/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "histAxisName": "WWW_4F_centralUL16",
        "xsecName": "WWW",
    },
    "UL16_WWW_4F_ext" : {
        "path" : "/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17_ext1-v1/NANOAODSIM",
        "histAxisName": "WWW_4F_centralUL16",
        "xsecName": "WWW",
    },
    "UL16_WWZ_4F" : {
        "path" : "/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "histAxisName": "WWZ_4F_centralUL16",
        "xsecName": "WWZ",
    },
    "UL16_WWZ_4F_ext" : {
        "path" : "/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17_ext1-v1/NANOAODSIM",
        "histAxisName": "WWZ_4F_centralUL16",
        "xsecName": "WWZ",
    },
    "UL16_WZTo3LNu" : {
        "path" : "/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "histAxisName": "WZTo3LNu_centralUL16",
        "xsecName": "WZTo3LNu",
    },
    "UL16_WZZ" : {
        "path" : "/WZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "histAxisName": "WZZ_centralUL16",
        "xsecName": "WZZ",
    },
    "UL16_WZZ_ext" : {
        "path" : "/WZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17_ext1-v1/NANOAODSIM",
        "histAxisName": "WZZ_centralUL16",
        "xsecName": "WZZ",
    },
    "UL16_ZZTo4L" : {
        "path" : "/ZZTo4L_13TeV_powheg_pythia8/RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1/NANOAODSIM", # NOTE: PLACEHOLDER till a UL sample is available (last checked Jan 17, 2022)
        "histAxisName": "ZZTo4L_centralUL16",
        "xsecName": "ZZTo4L",
    },
    "UL16_ZZZ" : {
        "path" : "/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "histAxisName": "ZZZ_centralUL16",
        "xsecName": "ZZZ",
    },
    "UL16_ZZZ_ext" : {
        "path" : "/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17_ext1-v1/NANOAODSIM",
        "histAxisName": "ZZZ_centralUL16",
        "xsecName": "ZZZ",
    },

}

central_UL16APV_bkg_dict = {
    "UL16APV_DY10to50" : {
        "path" : "/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv2-106X_mcRun2_asymptotic_preVFP_v9-v1/NANOAODSIM", # NOTE: PLACEHOLDER till v9 is available (last checked Jan 17, 2022)
        "histAxisName": "DY10to50_centralUL16APV",
        "xsecName": "DYJetsToLL_M_10to50_MLM",
    },
    "UL16APV_DY50" : {
        "path" : "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "histAxisName": "DY50_centralUL16APV",
        "xsecName": "DYJetsToLL_M_50_MLM",
    },
    "UL16APV_ST_top_s-channel" : {
        "path" : "/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "histAxisName": "ST_top_s-channel_centralUL16APV",
        "xsecName": "ST_top_s-channel",
    },
    "UL16APV_ST_top_t-channel" : {
        "path" : "/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM", 
        "histAxisName": "ST_top_t-channel_centralUL16APV",
        "xsecName": "ST_top_t-channel",
    },
    "UL16APV_ST_antitop_t-channel" : {
        "path" : "/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM", 
        "histAxisName": "ST_antitop_t-channel_centralUL16APV",
        "xsecName": "ST_antitop_t-channel",
    },
    "UL16APV_tbarW" : {
        "path" : "/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "histAxisName": "tbarW_centralUL16APV",
        "xsecName": "ST_tW_antitop_5f_inclusiveDecays",
    },
    "UL16APV_tW" : {
        "path" : "/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "histAxisName": "tW_centralUL16APV",
        "xsecName": "ST_tW_top_5f_inclusiveDecays",
    },
    "UL16APV_TTGJets" : {
        "path" : "/TTGJets_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM", 
        "histAxisName": "TTGJets_centralUL16APV",
        "xsecName": "TTGJets",
    },
    "UL16APV_TTJets" : {
        "path" : "/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv2-106X_mcRun2_asymptotic_preVFP_v9-v1/NANOAODSIM", # NOTE: PLACEHOLDER till v9 is available (last checked Jan 17, 2022)
        "histAxisName": "TTJets_centralUL16APV",
        "xsecName": "TT",
    },
    "UL16APV_WWTo2L2Nu" : {
        "path" : "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "histAxisName": "WWTo2L2Nu_centralUL16APV",
        "xsecName": "WWTo2L2Nu",
    },
    "UL16APV_WJetsToLNu" : {
        "path" : "/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer19UL16NanoAODAPVv2-106X_mcRun2_asymptotic_preVFP_v9-v1/NANOAODSIM", # NOTE: PLACEHOLDER till v9 is available (last checked Jan 17, 2022)
        "histAxisName": "WJetsToLNu_centralUL16APV",
        "xsecName": "WJetsToLNu",
    },
    "UL16APV_WWW_4F" : {
        "path" : "/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "histAxisName": "WWW_centralUL16APV",
        "xsecName": "WWW",
    },
    "UL16APV_WWW_4F_ext" : {
        "path" : "/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11_ext1-v1/NANOAODSIM",
        "histAxisName": "WWW_centralUL16APV",
        "xsecName": "WWW",
    },
    "UL16APV_WWZ_4F" : {
        "path" : "/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "histAxisName": "WWZ_4F_centralUL16APV",
        "xsecName": "WWZ",
    },
    "UL16APV_WWZ_4F_ext" : {
        "path" : "/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11_ext1-v1/NANOAODSIM",
        "histAxisName": "WWZ_4F_centralUL16APV",
        "xsecName": "WWZ",
    },
    "UL16APV_WZTo3LNu" : {
        "path" : "/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv2-106X_mcRun2_asymptotic_preVFP_v9-v1/NANOAODSIM", # NOTE: PLACEHOLDER till v9 is available (last checked Jan 17, 2022)
        "histAxisName": "WZTo3LNu_centralUL16APV",
        "xsecName": "WZTo3LNu",
    },
    "UL16APV_WZZ" : {
        "path" : "/WZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "histAxisName": "WZZ_centralUL16APV",
        "xsecName": "WZZ",
    },
    "UL16APV_WZZ_ext" : {
        "path" : "/WZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11_ext1-v1/NANOAODSIM",
        "histAxisName": "WZZ_centralUL16APV",
        "xsecName": "WZZ",
    },
    "UL16APV_ZZTo4L" : {
        "path" : "/ZZTo4L_13TeV_powheg_pythia8/RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1/NANOAODSIM", # NOTE: PLACEHOLDER till a ULAPV sample is available (last checked Jan 17, 2022)
        "histAxisName": "ZZTo4L_centralUL16APV",
        "xsecName": "ZZTo4L",
    },
    "UL16APV_ZZZ" : {
        "path" : "/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "histAxisName": "ZZZ_centralUL16APV",
        "xsecName": "ZZZ",
    },
    "UL16APV_ZZZ_ext" : {
        "path" : "/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11_ext1-v1/NANOAODSIM",
        "histAxisName": "ZZZ_centralUL16APV",
        "xsecName": "ZZZ",
    },
}

########### Data ##############
from collections import defaultdict

# For more info on the datasets and eras for each year
# See: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun2LegacyAnalysis

### 2016 ###
year = 'Run2016'
naod_version  = "MiniAODv1_NanoAODv2-v1"
dataset_names = ["DoubleEG","DoubleMuon","SingleElectron","SingleMuon"]
dataset_eras = [# See: https://twiki.cern.ch/twiki/bin/view/CMS/PdmVDatasetsUL2016
    'B-ver1_HIPM_UL2016',
    'B-ver2_HIPM_UL2016',
    'C-UL2016',
    'D-UL2016',
    'E-UL2016',
    'F-HIPM_UL2016',
    'F-UL2016',
    'G-UL2016',
    'H-UL2016',
]

version_overwrite = {
    'DoubleEG': {# See: https://pdmv-pages.web.cern.ch/rereco_ul/?input_dataset=DoubleEG%2FRun2016
        'F-UL2016': 'MiniAODv1_NanoAODv2-v2',
    },
    'DoubleMuon': {# See: https://pdmv-pages.web.cern.ch/rereco_ul/?input_dataset=DoubleMuon%2FRun2016
        'F-UL2016': 'MiniAODv1_NanoAODv2-v2',
    },
    'SingleElectron': {# See: https://pdmv-pages.web.cern.ch/rereco_ul/?input_dataset=SingleElectron%2FRun2016
        'E-UL2016': 'MiniAODv1_NanoAODv2-v2',
        'F-UL2016': 'MiniAODv1_NanoAODv2-v2',
    },
    'SingleMuon': {# See: https://pdmv-pages.web.cern.ch/rereco_ul/?input_dataset=SingleMuon%2FRun2016
        'F-UL2016': 'MiniAODv1_NanoAODv2-v4',
    },
}

data_2016_dict = defaultdict(lambda: {'path': '','histAxisName': 'dataUL16', 'xsecName': 'data'})
for era in dataset_eras:
    for ds_name in dataset_names:
        key_name = "{name}_{era}".format(name=ds_name,era=era)
        version = naod_version
        if ds_name in version_overwrite:
            if era in version_overwrite[ds_name]:
                version = version_overwrite[ds_name][era]
        ds_path = "/{ds}/{year}{era}_{ver}/NANOAOD".format(year=year,ds=ds_name,era=era,ver=version)
        data_2016_dict[key_name]['path'] = ds_path

### 2017 ###
year = 'Run2017'
naod_version  = "MiniAODv1_NanoAODv2-v1"
dataset_names = ["SingleMuon","SingleElectron","DoubleMuon","DoubleEG","MuonEG"]
dataset_eras = [# Note: Eras G and H correspond to 5 TeV and lowPU, so ignore them
    'B-UL2017',
    'C-UL2017',
    'D-UL2017',
    'E-UL2017',
    'F-UL2017',
]

version_overwrite = {
    'SingleMuon': {# See: https://pdmv-pages.web.cern.ch/rereco_ul/?input_dataset=SingleMuon%2FRun2017
        'E-UL2017': 'MiniAODv1_NanoAODv2-v2',
        'F-UL2017': 'MiniAODv1_NanoAODv2-v2',
    },
    'SingleElectron': {# See: https://pdmv-pages.web.cern.ch/rereco_ul/?input_dataset=SingleElectron%2FRun2017
        'E-UL2017': 'MiniAODv1_NanoAODv2-v2',
        'F-UL2017': 'MiniAODv1_NanoAODv2-v3',
    }
}

data_2017_dict = defaultdict(lambda: {'path': '','histAxisName': 'dataUL17', 'xsecName': 'data'})
for era in dataset_eras:
    for ds_name in dataset_names:
        key_name = "{name}_{era}".format(name=ds_name,era=era)
        version = naod_version
        if ds_name in version_overwrite:
            if era in version_overwrite[ds_name]:
                version = version_overwrite[ds_name][era]
        ds_path = "/{ds}/{year}{era}_{ver}/NANOAOD".format(year=year,ds=ds_name,era=era,ver=version)
        data_2017_dict[key_name]['path'] = ds_path

### 2018 ###
year = 'Run2018'
naod_version  = "MiniAODv1_NanoAODv2-v1"
dataset_names = ["SingleMuon","EGamma","DoubleMuon","MuonEG"]
dataset_eras = [
    'A-UL2018',
    'B-UL2018',
    'C-UL2018',
    'D-UL2018',
]
version_overwrite = {
    'SingleMuon': {# See: https://pdmv-pages.web.cern.ch/rereco_ul/?input_dataset=SingleMuon%2FRun2018
        'A-UL2018': 'MiniAODv1_NanoAODv2-v2',
        'B-UL2018': 'MiniAODv1_NanoAODv2-v2',
        'C-UL2018': 'MiniAODv1_NanoAODv2-v2',
        'D-UL2018': 'MiniAODv1_NanoAODv2-v2',
    },
    'EGamma': {# See: https://pdmv-pages.web.cern.ch/rereco_ul/?input_dataset=EGamma%2FRun2018
        'D-UL2018': 'MiniAODv1_NanoAODv2-v2',
    },
    'DoubleMuon': {# See: https://pdmv-pages.web.cern.ch/rereco_ul/?input_dataset=DoubleMuon%2FRun2018
        'B-UL2018': 'MiniAODv1_NanoAODv2-v2',
    },
}

data_2018_dict = defaultdict(lambda: {'path': '','histAxisName': 'dataUL18', 'xsecName': 'data'})
for era in dataset_eras:
    for ds_name in dataset_names:
        key_name = "{name}_{era}".format(name=ds_name,era=era)
        version = naod_version
        if ds_name in version_overwrite:
            if era in version_overwrite[ds_name]:
                version = version_overwrite[ds_name][era]
        ds_path = "/{ds}/{year}{era}_{ver}/NANOAOD".format(year=year,ds=ds_name,era=era,ver=version)
        data_2018_dict[key_name]['path'] = ds_path

########### TESTING ########### 

test_dict = {
    "test_dy_sample" : {
        "path" : "/store/user/jrgonzal/nanoAODcrab/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/mc2017_28apr2021_DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/210427_231758/0000/",
        "histAxisName": "central_DY",
        "xsecName" : "DYJetsToLL_M_50_MLM", # Not sure if this is actually the right xsec, I just picked one of the DY ones
    }
}

########### Functions for makign the jsons ###########

# Replace a value in one of the JSONs
def replace_val_in_json(path_to_json_file,key,new_val,verbose=True):

    # Replace value if it's different than what's in the JSON
    with open(path_to_json_file) as json_file:
       json_dict = json.load(json_file)
    if new_val == json_dict[key]:
        if verbose:
            print(f"\tValues already agree, both are {new_val}")
    else:
        if verbose:
            print(f"\tOld value for {key}: {json_dict[key]}")
            print(f"\tNew value for {key}: {new_val}")
        json_dict[key] = new_val

        # Save new json
        with open(path_to_json_file, "w") as out_file:
            json.dump(json_dict, out_file, indent=2)


# Loop through a dictionary of samples and replace the xsec in the JSON with what's in xsec.cfg
def replace_xsec_for_dict_of_samples(samples_dict,out_dir):
    for sample_name,sample_info in samples_dict.items():
        path_to_json = os.path.join(out_dir,sample_name+".json")
        xsecName = sample_info["xsecName"]
        new_xsec = XSECDIC[xsecName]
        print(f"\nReplacing XSEC for {sample_name} JSON with the value from xsec.cfg for \"{xsecName}\":")
        print("\tPath to json:",path_to_json)
        replace_val_in_json(path_to_json,"xsec",new_xsec)

# Wrapper for createJSON.py
def make_json(sample_dir,sample_name,prefix,sample_yr,xsec_name,hist_axis_name,on_das=False):

    # If the sample is on DAS, inclue the DAS flag in the createJSON.py arguments
    das_flag = ""
    if on_das: das_flag = "--DAS"

    args = [
        "python",
        "../../topcoffea/modules/createJSON.py",
        sample_dir,
        das_flag,
        "--sampleName"   , sample_name,
        "--prefix"       , prefix,
        "--xsec"         , "../../topcoffea/cfg/xsec.cfg",
        "--year"         , sample_yr,
        "--histAxisName" , hist_axis_name,
    ]

    if xsec_name:
        args.extend(['--xsecName',xsec_name])

    # Run createJSON.py
    subprocess.run(args)

# Convenience function for running make_json() on all entries in a dictionary of samples, and moving the results to out_dir
def make_jsons_for_dict_of_samples(samples_dict,prefix,year,out_dir,on_das=False):
    for sample_name,sample_info in samples_dict.items():
        print(f"\n\nMaking JSON for {sample_name}...")
        path = sample_info["path"]
        hist_axis_name = sample_info["histAxisName"]
        xsec_name = sample_info["xsecName"]
        make_json(
            sample_dir = path,
            sample_name = sample_name,
            prefix = prefix,
            sample_yr = year,
            xsec_name = xsec_name,
            hist_axis_name = hist_axis_name,
            on_das = on_das,
        )
        out_name = sample_name+".json"

        subprocess.run(["mv",out_name,out_dir]) 
        if '_ext' in out_name:
          combine_json_ext(out_dir+'/'+out_name) # Merge with non-ext version
          os.remove(out_dir+'/'+out_name) # Remove (now) outdated ext version

        print("sample name:",sample_name)
        print("\tpath:",path,"\n\thistAxisName:",hist_axis_name,"\n\txsecName",xsec_name,"\n\tout name:",out_name,"\n\tout dir:",out_dir)


# Uncomment the make_jsons_for_dict_of_samples() lines for the jsons you want to make/remake
def main():

    # Specify some output dirs
    out_dir_private_UL     = os.path.join(topcoffea_path("json"),"signal_samples/private_UL/")
    out_dir_private_UL_subset_local = os.path.join(topcoffea_path("json"),"signal_samples/subsets_of_private_UL_for_debugging/private_UL17_b1b4_at_NDscratch365/")
    out_dir_private_UL_subset_unl = os.path.join(topcoffea_path("json"),"signal_samples/subsets_of_private_UL_for_debugging/private_UL17_b1b4_at_NDscratch365/")
    out_dir_top19001_local = os.path.join(topcoffea_path("json"),"signal_samples/private_top19001_local")
    out_dir_central_UL     = os.path.join(topcoffea_path("json"),"signal_samples/central_UL/")
    out_dir_central_bkg_UL = os.path.join(topcoffea_path("json"),"background_samples/central_UL/")
    out_dir_central_2017   = os.path.join(topcoffea_path("json"),"signal_samples/central_2017/")
    out_dir_central_sync   = os.path.join(topcoffea_path("json"),"sync_samples/")

    out_dir_data_2016 = os.path.join(topcoffea_path("json"),"data_samples/2016/")
    out_dir_data_2017 = os.path.join(topcoffea_path("json"),"data_samples/2017/")
    out_dir_data_2018 = os.path.join(topcoffea_path("json"),"data_samples/2018/")

    ######### Make/remake JSONs #########

    # Private UL samples
    #make_jsons_for_dict_of_samples(private_UL17_dict,"/hadoop","2017",out_dir_private_UL)
    #make_jsons_for_dict_of_samples(private_UL18_dict,"/hadoop","2018",out_dir_private_UL)
    #make_jsons_for_dict_of_samples(private_UL16_dict,"/hadoop","2016",out_dir_private_UL)
    #make_jsons_for_dict_of_samples(private_UL16APV_dict,"/hadoop","2016APV",out_dir_private_UL) # Not sure what we need here for the year, can remake the JSONs later to update when we have SFs etc set up for 2016 stuff (right now I think it's mostly just 18)

    # Subsets of files for small debugging tests local files (scratch365 at ND)
    #make_jsons_for_dict_of_samples(private_2017_dict,"","2017",out_dir_top19001_local)
    #make_jsons_for_dict_of_samples(private_UL17_dict_b1b4_local,"","2017",out_dir_private_UL_subset_local)

    # Central signal samples
    #make_jsons_for_dict_of_samples(central_2016_dict,"root://ndcms.crc.nd.edu/","2016",out_dir_central_2016,on_das=True)
    #make_jsons_for_dict_of_samples(central_UL16_dict,"root://ndcms.crc.nd.edu/","2016",out_dir_central_UL,on_das=True)
    #make_jsons_for_dict_of_samples(central_2016APV_dict,"root://ndcms.crc.nd.edu/","2016APV",out_dir_central_2016APV,on_das=True)
    #make_jsons_for_dict_of_samples(central_UL16APV_dict,"root://ndcms.crc.nd.edu/","2016APV",out_dir_central_UL,on_das=True)
    #make_jsons_for_dict_of_samples(central_2017_correctnPartonsInBorn_dict,"root://ndcms.crc.nd.edu/","2017",out_dir_central_2017,on_das=True)
    #make_jsons_for_dict_of_samples(central_2017_dict,"root://ndcms.crc.nd.edu/","2017",out_dir_central_2017,on_das=True)
    #make_jsons_for_dict_of_samples(central_UL17_dict,"root://ndcms.crc.nd.edu/","2017",out_dir_central_UL,on_das=True)
    #make_jsons_for_dict_of_samples(central_UL18_dict,"root://ndcms.crc.nd.edu/","2018",out_dir_central_UL,on_das=True)
    #make_jsons_for_dict_of_samples(sync_dict,"root://ndcms.crc.nd.edu/","2017",out_dir_central_sync)

    # Central background samples
    make_jsons_for_dict_of_samples(central_UL17_bkg_dict,"root://ndcms.crc.nd.edu/","2017",out_dir_central_bkg_UL,on_das=True)
    make_jsons_for_dict_of_samples(central_UL18_bkg_dict,"root://ndcms.crc.nd.edu/","2018",out_dir_central_bkg_UL,on_das=True)
    make_jsons_for_dict_of_samples(central_UL16_bkg_dict,"root://ndcms.crc.nd.edu/","2016",out_dir_central_bkg_UL,on_das=True)
    make_jsons_for_dict_of_samples(central_UL16APV_bkg_dict,"root://ndcms.crc.nd.edu/","2016APV",out_dir_central_bkg_UL,on_das=True)

    # Data samples
    #make_jsons_for_dict_of_samples(data_2016_dict,"root://ndcms.crc.nd.edu/","2016",out_dir_data_2016,on_das=True)
    #make_jsons_for_dict_of_samples(data_2017_dict,"root://ndcms.crc.nd.edu/","2017",out_dir_data_2017,on_das=True)
    #make_jsons_for_dict_of_samples(data_2018_dict,"root://ndcms.crc.nd.edu/","2018",out_dir_data_2018,on_das=True)

    # Testing finding list of files with xrdfs ls
    #make_jsons_for_dict_of_samples(test_dict,"root://xrootd-local.unl.edu/","2017",".")


    ######### Just replace xsec in JSON with whatever is in xsec.cfg #########
    #replace_xsec_for_dict_of_samples(central_UL17_bkg_dict,out_dir_central_bkg_UL)
    #replace_xsec_for_dict_of_samples(central_UL18_bkg_dict,out_dir_central_bkg_UL)
    #replace_xsec_for_dict_of_samples(central_UL16_bkg_dict,out_dir_central_bkg_UL)
    #replace_xsec_for_dict_of_samples(central_UL16APV_bkg_dict,out_dir_central_bkg_UL)


if __name__ == "__main__":
    main()

