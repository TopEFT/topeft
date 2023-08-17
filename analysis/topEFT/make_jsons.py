# This file is essentially a wrapper for createJSON.py:
#   - It runs createJSON.py for each sample that you include in a dictionary, and moves the resulting json file to the directory you specify
#   - If the private NAOD has to be remade, the version numbers should be updated in the dictionaries here, then just rerun the script to remake the jsons

import os
from topcoffea.modules.paths import topcoffea_path
from topcoffea.modules.samples import loadxsecdic
import topcoffea.modules.sample_lst_jsons_tools as sjt

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

# Testing only a single batch from each
test_private_UL17_dict = {


    "UL17_ttHJet_b1"   : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v4/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttHJet_privateUL17",
        "xsecName": "ttHnobb",
    },
    "UL17_ttlnuJet_b1" : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v4/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttlnuJet_privateUL17",
        "xsecName": "TTWJetsToLNu",
    },
    "UL17_ttllJet_b1"  : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v4/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttllJet_privateUL17",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL17_tllq_b1"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v4/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
        "histAxisName" : "tllq_privateUL17",
        "xsecName": "tZq",
    },
    "UL17_tHq_b1"      : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v4/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "tHq_privateUL17",
        "xsecName": "tHq",
    },
    "UL17_tttt_b4"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch4/naodOnly_step/v2/nAOD_step_tttt_FourtopsMay3v1_run0",
        "histAxisName": "tttt_privateUL17",
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
        "path_local" : "/store/mc/RunIIFall17NanoAODv7/ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1",
        "histAxisName": "ttH_central2017",
        "xsecName": "ttHnobb",
    },
    "2017_TTWJetsToLNu" : {
        "path" : "/TTWJetsToLNu_TuneCP5_PSweights_13TeV-amcatnloFXFX-madspin-pythia8/RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIIFall17NanoAODv7/TTWJetsToLNu_TuneCP5_PSweights_13TeV-amcatnloFXFX-madspin-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1",
        "histAxisName": "ttW_central2017",
        "xsecName": "TTWJetsToLNu",
    },
    "2017_TTZToLLNuNu_M_10" : {
        "path" : "/TTZToLLNuNu_M-10_TuneCP5_PSweights_13TeV-amcatnlo-pythia8/RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIIFall17NanoAODv7/TTZToLLNuNu_M-10_TuneCP5_PSweights_13TeV-amcatnlo-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1",
        "histAxisName": "ttZ_central2017",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "2017_tZq" : {
        "path" : "/tZq_ll_4f_ckm_NLO_TuneCP5_PSweights_13TeV-amcatnlo-pythia8/RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIIFall17NanoAODv7/tZq_ll_4f_ckm_NLO_TuneCP5_PSweights_13TeV-amcatnlo-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1",
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

########### Central signal samples UL ###########

central_UL16_dict = {
    "UL16_ttHnobb" : {
        "path" : "/ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName": "ttHJet_centralUL16",
        "xsecName": "ttHnobb",
    },
    "UL16_TTWJetsToLNu" : {
        "path" : "/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName": "ttW_centralUL16",
        "xsecName": "TTWJetsToLNu",
    },
    "UL16_TTZToLLNuNu_M_10" : {
        "path" : "/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName": "ttZ_centralUL16",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL16_tZq" : {
        "path" : "/tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName": "tZq_centralUL16",
        "xsecName": "tZq",
    },
    "UL16_tttt" : {
        "path" : "/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v2",
        "histAxisName": "tttt_centralUL16",
        "xsecName": "tttt",
    },
}

central_UL16APV_dict = {
    "UL16APV_ttHnobb" : {
        "path" : "/ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1",
        "histAxisName": "ttHJet_centralUL16APV",
        "xsecName": "ttHnobb",
    },
    "UL16APV_TTWJetsToLNu" : {
        "path" : "/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v2",
        "histAxisName": "ttW_centralUL16APV",
        "xsecName": "TTWJetsToLNu",
    },
    "UL16APV_TTZToLLNuNu_M_10" : {
        "path" : "/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1",
        "histAxisName": "ttZ_centralUL16APV",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL16APV_tZq" : {
        "path" : "/tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1",
        "histAxisName": "tZq_centralUL16APV",
        "xsecName": "tZq",
    },
    "UL16APV_tttt" : {
        "path" : "/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v2",
        "histAxisName": "tttt_centralUL16APV",
        "xsecName": "tttt",
    },
}

central_UL17_dict = {
    "UL17_ttHnobb" : {
        "path" : "/ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1",
        "histAxisName": "ttH_centralUL17",
        "xsecName": "ttHnobb",
    },
    "UL17_TTWJetsToLNu" : {
        "path" : "/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1",
        "histAxisName": "ttW_centralUL17",
        "xsecName": "TTWJetsToLNu",
    },
    "UL17_TTZToLLNuNu_M_10" : {
        "path" : "/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1",
        "histAxisName": "ttZ_centralUL17",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL17_tZq" : {
        "path" : "/tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1",
        "histAxisName": "tZq_centralUL17",
        "xsecName": "tZq",
    },
    "UL17_tttt" : {
        "path" : "/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v2",
        "histAxisName": "tttt_centralUL17",
        "xsecName": "tttt",
    },
}

central_UL18_dict = {
    "UL18_ttHnobb" : {
        "path" : "/ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1",
        "histAxisName": "ttH_centralUL18",
        "xsecName": "ttHnobb",
    },
    "UL18_TTWJetsToLNu" : {
        "path" : "/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1",
        "histAxisName": "ttW_centralUL18",
        "xsecName": "TTWJetsToLNu",
    },
    "UL18_TTZToLLNuNu_M_10" : {
        "path" : "/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1",
        "histAxisName": "ttZ_centralUL18",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL18_tZq" : {
        "path" : "/tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1",
        "histAxisName": "tZq_centralUL18",
        "xsecName": "tZq",
    },
    "UL18_tttt" : {
        "path" : "/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2",
        "histAxisName": "tttt_centralUL18",
        "xsecName": "tttt",
    },
}


########### Central background samples ###########

central_UL17_bkg_dict = {
    "UL17_ZGToLLG" : {
        "path" : "/ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM",
        "histAxisName": "ZGToLLG_centralUL17",
        "xsecName": "ZGToLLG",
    },
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
    "UL17_TTGamma_SingleLept" : {
        "path" : "/TTGamma_SingleLept_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/TTGamma_SingleLept_TuneCP5_13TeV-madgraph-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1",
        "histAxisName": "TTGamma_centralUL17",
        "xsecName": "TTGamma_SingleLept",
    },
    "UL17_TTGamma_Dilept" : {
        "path" : "/TTGamma_Dilept_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/TTGamma_Dilept_TuneCP5_13TeV-madgraph-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1",
        "histAxisName": "TTGamma_centralUL17",
        "xsecName": "TTGamma_Dilept",
    },
    "UL17_TTJets" : {
        "path" : "/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "histAxisName": "TTJets_centralUL17",
        "xsecName": "TT",
    },
    "UL17_TTTo2L2Nu" : {
        "path" : "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1",
        "histAxisName": "TTTo2L2Nu_centralUL17",
        "xsecName": "TTTo2L2Nu",
    },
    "UL17_TTToSemiLeptonic" : {
        "path" : "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1",
        "histAxisName": "TTToSemiLeptonic_centralUL17",
        "xsecName": "TTToSemiLeptonic",
    },
    "UL17_WJetsToLNu" : {
        "path" : "/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "histAxisName": "WJetsToLNu_centralUL17",
        "xsecName": "WJetsToLNu",
    },
    "UL17_WWTo2L2Nu" : {
        "path" : "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v2/",
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
    "UL17_WWZ_4F_ext" : {
        "path" : "/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9_ext1-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mc2017_realistic_v9_ext1-v2",
        "histAxisName": "WWZ_centralUL17",
        "xsecName": "WWZ",
    },
    "UL17_WZTo3LNu" : {
        "path" : "/WZTo3LNu_mllmin4p0_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/WZTo3LNu_mllmin4p0_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v2",
        "histAxisName": "WZTo3LNu_centralUL17",
        "xsecName": "WZTo3LNu",
    },
    "UL17_WLLJJ_WToLNu_EWK" : {
        "path" : "/WLLJJ_WToLNu_EWK_TuneCP5_13TeV_madgraph-madspin-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/WLLJJ_WToLNu_EWK_TuneCP5_13TeV_madgraph-madspin-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v2",
        "histAxisName": "WZTo3LNu_centralUL17",
        "xsecName": "WLLJJ_WToLNu_EWK",
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
        "path" : "/ZZTo4L_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/ZZTo4L_TuneCP5_13TeV_powheg_pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v2",
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
    "UL17_TWZToLL_thad_Wlept" : {
        "path" : "/TWZToLL_thad_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/TWZToLL_thad_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1",
        "histAxisName": "TWZToLL_centralUL17",
        "xsecName": "TWZToLL_thad_Wlept",
    },
    "UL17_TWZToLL_tlept_Whad" : {
        "path" : "/TWZToLL_tlept_Whad_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/TWZToLL_tlept_Whad_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1",
        "histAxisName": "TWZToLL_centralUL17",
        "xsecName": "TWZToLL_tlept_Whad",
    },
    "UL17_TWZToLL_tlept_Wlept" : {
        "path" : "/TWZToLL_tlept_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/TWZToLL_tlept_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1",
        "histAxisName": "TWZToLL_centralUL17",
        "xsecName": "TWZToLL_tlept_Wlept",
    },
    # NOTE: This should really be part of the signal, but no EFT effects, so it's included in the bkg samples
    "UL17_TTZToLL_M-1to10" : {
        "path" : "/TTZToLL_M-1to10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/TTZToLL_M-1to10_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1",
        "histAxisName": "TTZToLL_M1to10_centralUL17",
        "xsecName": "TTZToLL_M1to10",
    },
    "UL17_GluGluToContinToZZTo2e2mu":{
        "path" : "/GluGluToContinToZZTo2e2mu_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/GluGluToContinToZZTo2e2mu_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v2",
        "histAxisName" : "ZZTo4L_centralUL17",
        "xsecName" : "ggToZZTo2e2mu",
    },
    "UL17_GluGluToContinToZZTo2e2nu":{
        "path" : "/GluGluToContinToZZTo2e2nu_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/GluGluToContinToZZTo2e2nu_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v2",
        "histAxisName" : "ZZTo4L_centralUL17",
        "xsecName" : "ggToZZTo2e2nu",
    },
    "UL17_GluGluToContinToZZTo2e2tau":{
        "path" : "/GluGluToContinToZZTo2e2tau_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/GluGluToContinToZZTo2e2tau_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v2",
        "histAxisName" : "ZZTo4L_centralUL17",
        "xsecName" : "ggToZZTo2e2tau",
    },
    "UL17_GluGluToContinToZZTo2mu2tau":{
        "path" : "/GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v2",
        "histAxisName" : "ZZTo4L_centralUL17",
        "xsecName" : "ggToZZTo2mu2tau",
    },
    "UL17_GluGluToContinToZZTo4e":{
        "path" : "/GluGluToContinToZZTo4e_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/GluGluToContinToZZTo4e_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v2",
        "histAxisName" : "ZZTo4L_centralUL17",
        "xsecName" : "ggToZZTo4e",
    },
    "UL17_GluGluToContinToZZTo4mu":{
        "path" : "/GluGluToContinToZZTo4mu_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/GluGluToContinToZZTo4mu_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v2",
        "histAxisName" : "ZZTo4L_centralUL17",
        "xsecName" : "ggToZZTo4mu",
    },
    "UL17_GluGluToContinToZZTo4tau":{
        "path" : "/GluGluToContinToZZTo4tau_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL17NanoAODv9/GluGluToContinToZZTo4tau_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v2",
        "histAxisName" : "ZZTo4L_centralUL17",
        "xsecName" : "ggToZZTo4tau",
    },
}


central_UL18_bkg_dict = {
    "UL18_ZGToLLG" : {
        "path" : "/ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM",
        "histAxisName": "ZGToLLG_centralUL18",
        "xsecName": "ZGToLLG",
    },
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
    "UL18_TTGJets_ext" : {
        "path" : "/TTGJets_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1_ext1-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/TTGJets_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1_ext1-v1/",
        "histAxisName": "TTGJets_centralUL18",
        "xsecName": "TTGJets",
    },
    "UL18_TTJets" : {
        "path" : "/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1",
        "histAxisName": "TTJets_centralUL18",
        "xsecName": "TT",
    },
    "UL18_TTGamma_SingleLept" : {
        "path" : "/TTGamma_SingleLept_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/TTGamma_SingleLept_TuneCP5_13TeV-madgraph-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1",
        "histAxisName": "TTGamma_centralUL18",
        "xsecName": "TTGamma_SingleLept",
    },
    "UL18_TTGamma_Dilept" : {
        "path" : "/TTGamma_Dilept_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/TTGamma_Dilept_TuneCP5_13TeV-madgraph-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1",
        "histAxisName": "TTGamma_centralUL18",
        "xsecName": "TTGamma_Dilept",
    },
    "UL18_TTTo2L2Nu" : {
        "path" : "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1",
        "histAxisName": "TTTo2L2Nu_centralUL18",
        "xsecName": "TTTo2L2Nu",
    },
    "UL18_TTToSemiLeptonic" : {
        "path" : "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1",
        "histAxisName": "TTToSemiLeptonic_centralUL18",
        "xsecName": "TTToSemiLeptonic",
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
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1",
        "histAxisName": "WWW_4F_centralUL18",
        "xsecName": "WWW",
    },
    "UL18_WWW_4F_ext" : {
        "path" : "/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1_ext1-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1_ext1-v2",
        "histAxisName": "WWW_4F_centralUL18",
        "xsecName": "WWW",
    },
    "UL18_WWZ_4F" : {
        "path" : "/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1",
        "histAxisName": "WWZ_4F_centralUL18",
        "xsecName": "WWZ",
    },
    "UL18_WWZ_4F_ext" : {
        "path" : "/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1_ext1-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1_ext1-v2",
        "histAxisName": "WWZ_4F_centralUL18",
        "xsecName": "WWZ",
    },
    "UL18_WZTo3LNu" : {
        "path" : "/WZTo3LNu_mllmin4p0_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/WZTo3LNu_mllmin4p0_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2",
        "histAxisName": "WZTo3LNu_centralUL18",
        "xsecName": "WZTo3LNu",
    },
    "UL18_WLLJJ_WToLNu_EWK" : {
        "path" : "/WLLJJ_WToLNu_EWK_TuneCP5_13TeV_madgraph-madspin-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/WLLJJ_WToLNu_EWK_TuneCP5_13TeV_madgraph-madspin-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2",
        "histAxisName": "WZTo3LNu_centralUL18",
        "xsecName": "WLLJJ_WToLNu_EWK",
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
        "path" : "/ZZTo4L_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/ZZTo4L_TuneCP5_13TeV_powheg_pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/",
        "histAxisName": "ZZTo4L_centralUL18",
        "xsecName": "ZZTo4L",
    },
    "UL18_ZZZ" : {
        "path" : "/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/",
        "histAxisName": "ZZZ_centralUL18",
        "xsecName": "ZZZ",
    },
    "UL18_ZZZ_ext" : {
        "path" : "/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1_ext1-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1_ext1-v2",
        "histAxisName": "ZZZ_centralUL18",
        "xsecName": "ZZZ",
    },
    "UL18_TWZToLL_thad_Wlept" : {
        "path" : "/TWZToLL_thad_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/TWZToLL_thad_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1",
        "histAxisName": "TWZToLL_centralUL18",
        "xsecName": "TWZToLL_thad_Wlept",
    },
    "UL18_TWZToLL_tlept_Whad" : {
        "path" : "/TWZToLL_tlept_Whad_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/TWZToLL_tlept_Whad_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1",
        "histAxisName": "TWZToLL_centralUL18",
        "xsecName": "TWZToLL_tlept_Whad",
    },
    "UL18_TWZToLL_tlept_Wlept" : {
        "path" : "/TWZToLL_tlept_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/TWZToLL_tlept_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1",
        "histAxisName": "TWZToLL_centralUL18",
        "xsecName": "TWZToLL_tlept_Wlept",
    },
    # NOTE: This should really be part of the signal, but no EFT effects, so it's included in the bkg samples
    "UL18_TTZToLL_M-1to10" : {
        "path" : "/TTZToLL_M-1to10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/TTZToLL_M-1to10_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1",
        "histAxisName": "TTZToLL_M1to10_centralUL18",
        "xsecName": "TTZToLL_M1to10",
    },
    "UL18_GluGluToContinToZZTo2e2mu":{
        "path" : "/GluGluToContinToZZTo2e2mu_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/GluGluToContinToZZTo2e2mu_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2",
        "histAxisName" : "ZZTo4L_centralUL18",
        "xsecName" : "ggToZZTo2e2mu",
    },
    "UL18_GluGluToContinToZZTo2e2nu":{
        "path" : "/GluGluToContinToZZTo2e2nu_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/GluGluToContinToZZTo2e2nu_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2",
        "histAxisName" : "ZZTo4L_centralUL18",
        "xsecName" : "ggToZZTo2e2nu",
    },
    "UL18_GluGluToContinToZZTo2e2tau":{
        "path" : "/GluGluToContinToZZTo2e2tau_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/GluGluToContinToZZTo2e2tau_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2",
        "histAxisName" : "ZZTo4L_centralUL18",
        "xsecName" : "ggToZZTo2e2tau",
    },
    "UL18_GluGluToContinToZZTo2mu2tau":{
        "path" : "/GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2",
        "histAxisName" : "ZZTo4L_centralUL18",
        "xsecName" : "ggToZZTo2mu2tau",
    },
    "UL18_GluGluToContinToZZTo4e":{
        "path" : "/GluGluToContinToZZTo4e_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/GluGluToContinToZZTo4e_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2",
        "histAxisName" : "ZZTo4L_centralUL18",
        "xsecName" : "ggToZZTo4e",
    },
    "UL18_GluGluToContinToZZTo4mu":{
        "path" : "/GluGluToContinToZZTo4mu_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/GluGluToContinToZZTo4mu_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2",
        "histAxisName" : "ZZTo4L_centralUL18",
        "xsecName" : "ggToZZTo4mu",
    },
    "UL18_GluGluToContinToZZTo4tau":{
        "path" : "/GluGluToContinToZZTo4tau_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL18NanoAODv9/GluGluToContinToZZTo4tau_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2",
        "histAxisName" : "ZZTo4L_centralUL18",
        "xsecName" : "ggToZZTo4tau",
    },
}


central_UL16_bkg_dict = {
    "UL16_ZGToLLG" : {
        "path" : "/ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM",
        "histAxisName": "ZGToLLG_centralUL16",
        "xsecName": "ZGToLLG",
    },
    "UL16_DY10to50" : {
        "path" : "/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName": "DY10to50_centralUL16",
        "xsecName": "DYJetsToLL_M_10to50_MLM",
    },
    "UL16_DY50" : {
        "path" : "/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "histAxisName": "DY50_centralUL16",
        "xsecName": "DYJetsToLL_M_50_MLM",
    },
    "UL16_ST_top_s-channel" : {
        "path" : "/ST_s-channel_4f_leptonDecays_TuneCP5up_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/ST_s-channel_4f_leptonDecays_TuneCP5up_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v2",
        "histAxisName": "ST_top_s-channel_centralUL16",
        "xsecName": "ST_top_s-channel",
    },
    "UL16_ST_top_t-channel" : {
        "path" : "/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
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
    "UL16_TTGamma_SingleLept" : {
        "path" : "/TTGamma_SingleLept_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/TTGamma_SingleLept_TuneCP5_13TeV-madgraph-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName": "TTGamma_centralUL16",
        "xsecName": "TTGamma_SingleLept",
    },
    "UL16_TTGamma_Dilept" : {
        "path" : "/TTGamma_Dilept_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/TTGamma_Dilept_TuneCP5_13TeV-madgraph-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName": "TTGamma_centralUL16",
        "xsecName": "TTGamma_Dilept",
    },
    "UL16_TTJets" : {
        "path" : "/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName": "TTJets_centralUL16",
        "xsecName": "TT",
    },
    "UL16_TTTo2L2Nu" : {
        "path" : "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName": "TTTo2L2Nu_centralUL16",
        "xsecName": "TTTo2L2Nu",
    },
    "UL16_TTToSemiLeptonic" : {
        "path" : "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName": "TTToSemiLeptonic_centralUL16",
        "xsecName": "TTToSemiLeptonic",
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
        "path" : "/WZTo3LNu_mllmin4p0_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/WZTo3LNu_mllmin4p0_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v2",
        "histAxisName": "WZTo3LNu_centralUL16",
        "xsecName": "WZTo3LNu",
    },
    "UL16_WLLJJ_WToLNu_EWK" : {
        "path" : "/WLLJJ_WToLNu_EWK_TuneCP5_13TeV_madgraph-madspin-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/WLLJJ_WToLNu_EWK_TuneCP5_13TeV_madgraph-madspin-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v2",
        "histAxisName": "WZTo3LNu_centralUL16",
        "xsecName": "WLLJJ_WToLNu_EWK",
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
        "path" : "/ZZTo4L_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/ZZTo4L_TuneCP5_13TeV_powheg_pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
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
    "UL16_TWZToLL_thad_Wlept" : {
        "path" : "/TWZToLL_thad_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/TWZToLL_thad_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName": "TWZToLL_centralUL16",
        "xsecName": "TWZToLL_thad_Wlept",
    },
    "UL16_TWZToLL_tlept_Whad" : {
        "path" : "/TWZToLL_tlept_Whad_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/TWZToLL_tlept_Whad_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName": "TWZToLL_centralUL16",
        "xsecName": "TWZToLL_tlept_Whad",
    },
    "UL16_TWZToLL_tlept_Wlept" : {
        "path" : "/TWZToLL_tlept_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/TWZToLL_tlept_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName": "TWZToLL_centralUL16",
        "xsecName": "TWZToLL_tlept_Wlept",
    },
    # NOTE: This should really be part of the signal, but no EFT effects, so it's included in the bkg samples
    "UL16_TTZToLL_M-1to10" : {
        "path" : "/TTZToLL_M-1to10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/TTZToLL_M-1to10_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName": "TTZToLL_M1to10_centralUL16",
        "xsecName": "TTZToLL_M1to10",
    },
    "UL16_GluGluToContinToZZTo2e2mu":{
        "path" : "/GluGluToContinToZZTo2e2mu_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/GluGluToContinToZZTo2e2mu_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName" : "ZZTo4L_centralUL16",
        "xsecName" : "ggToZZTo2e2mu",
    },
    "UL16_GluGluToContinToZZTo2e2nu":{
        "path" : "/GluGluToContinToZZTo2e2nu_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/GluGluToContinToZZTo2e2nu_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName" : "ZZTo4L_centralUL16",
        "xsecName" : "ggToZZTo2e2nu",
    },
    "UL16_GluGluToContinToZZTo2e2tau":{
        "path" : "/GluGluToContinToZZTo2e2tau_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/GluGluToContinToZZTo2e2tau_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName" : "ZZTo4L_centralUL16",
        "xsecName" : "ggToZZTo2e2tau",
    },
    "UL16_GluGluToContinToZZTo2mu2tau":{
        "path" : "/GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName" : "ZZTo4L_centralUL16",
        "xsecName" : "ggToZZTo2mu2tau",
    },
    "UL16_GluGluToContinToZZTo4e":{
        "path" : "/GluGluToContinToZZTo4e_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/GluGluToContinToZZTo4e_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v2",
        "histAxisName" : "ZZTo4L_centralUL16",
        "xsecName" : "ggToZZTo4e",
    },
    "UL16_GluGluToContinToZZTo4mu":{
        "path" : "/GluGluToContinToZZTo4mu_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/GluGluToContinToZZTo4mu_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v2",
        "histAxisName" : "ZZTo4L_centralUL16",
        "xsecName" : "ggToZZTo4mu",
    },
    "UL16_GluGluToContinToZZTo4tau":{
        "path" : "/GluGluToContinToZZTo4tau_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODv9/GluGluToContinToZZTo4tau_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1",
        "histAxisName" : "ZZTo4L_centralUL16",
        "xsecName" : "ggToZZTo4tau",
    },
}

central_UL16APV_bkg_dict = {
    "UL16APV_ZGToLLG" : {
        "path" : "/ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM",
        "histAxisName": "ZGToLLG_centralUL16APV",
        "xsecName": "ZGToLLG",
    },
    "UL16APV_DY10to50" : {
        "path" : "/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1",
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
    "UL16APV_TTGamma_SingleLept" : {
        "path" : "/TTGamma_SingleLept_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/TTGamma_SingleLept_TuneCP5_13TeV-madgraph-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1",
        "histAxisName": "TTGamma_centralUL16APV",
        "xsecName": "TTGamma_SingleLept",
    },
    "UL16APV_TTGamma_Dilept" : {
        "path" : "/TTGamma_Dilept_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/TTGamma_Dilept_TuneCP5_13TeV-madgraph-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1",
        "histAxisName": "TTGamma_centralUL16APV",
        "xsecName": "TTGamma_Dilept",
    },
    "UL16APV_TTJets" : {
        "path" : "/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1",
        "histAxisName": "TTJets_centralUL16APV",
        "xsecName": "TT",
    },
    "UL16APV_TTTo2L2Nu" : {
        "path" : "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1",
        "histAxisName": "TTTo2L2Nu_centralUL16APV",
        "xsecName": "TTTo2L2Nu",
    },
    "UL16APV_TTToSemiLeptonic" : {
        "path" : "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1",
        "histAxisName": "TTToSemiLeptonic_centralUL16APV",
        "xsecName": "TTToSemiLeptonic",
    },
    "UL16APV_WWTo2L2Nu" : {
        "path" : "/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "histAxisName": "WWTo2L2Nu_centralUL16APV",
        "xsecName": "WWTo2L2Nu",
    },
    "UL16APV_WJetsToLNu" : {
        "path" : "/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v2",
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
        "path" : "/WZTo3LNu_mllmin4p0_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/WZTo3LNu_mllmin4p0_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1",
        "histAxisName": "WZTo3LNu_centralUL16APV",
        "xsecName": "WZTo3LNu",
    },
    "UL16APV_WLLJJ_WToLNu_EWK" : {
        "path" : "/WLLJJ_WToLNu_EWK_TuneCP5_13TeV_madgraph-madspin-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/WLLJJ_WToLNu_EWK_TuneCP5_13TeV_madgraph-madspin-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v2",
        "histAxisName": "WZTo3LNu_centralUL16APV",
        "xsecName": "WLLJJ_WToLNu_EWK",
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
        "path" : "/ZZTo4L_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/ZZTo4L_TuneCP5_13TeV_powheg_pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1",
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
    "UL16APV_TWZToLL_thad_Wlept" : {
        "path" : "/TWZToLL_thad_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/TWZToLL_thad_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1",
        "histAxisName": "TWZToLL_centralUL16APV",
        "xsecName": "TWZToLL_thad_Wlept",
    },
    "UL16APV_TWZToLL_tlept_Whad" : {
        "path" : "/TWZToLL_tlept_Whad_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/TWZToLL_tlept_Whad_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1",
        "histAxisName": "TWZToLL_centralUL16APV",
        "xsecName": "TWZToLL_tlept_Whad",
    },
    "UL16APV_TWZToLL_tlept_Wlept" : {
        "path" : "/TWZToLL_tlept_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/TWZToLL_tlept_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1",
        "histAxisName": "TWZToLL_centralUL16APV",
        "xsecName": "TWZToLL_tlept_Wlept",
    },
    # NOTE: This should really be part of the signal, but no EFT effects, so it's included in the bkg samples
    "UL16APV_TTZToLL_M-1to10" : {
        "path" : "/TTZToLL_M-1to10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/TTZToLL_M-1to10_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1",
        "histAxisName": "TTZToLL_M1to10_centralUL16APV",
        "xsecName": "TTZToLL_M1to10",
    },
    "UL16APV_GluGluToContinToZZTo2e2mu":{
        "path" : "/GluGluToContinToZZTo2e2mu_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/GluGluToContinToZZTo2e2mu_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v2",
        "histAxisName" : "ZZTo4L_centralUL16APV",
        "xsecName" : "ggToZZTo2e2mu",
    },
    "UL16APV_GluGluToContinToZZTo2e2nu":{
        "path" : "/GluGluToContinToZZTo2e2nu_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v3/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/GluGluToContinToZZTo2e2nu_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v3",
        "histAxisName" : "ZZTo4L_centralUL16APV",
        "xsecName" : "ggToZZTo2e2nu",
    },
    "UL16APV_GluGluToContinToZZTo2e2tau":{
        "path" : "/GluGluToContinToZZTo2e2tau_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/GluGluToContinToZZTo2e2tau_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v2",
        "histAxisName" : "ZZTo4L_centralUL16APV",
        "xsecName" : "ggToZZTo2e2tau",
    },
    "UL16APV_GluGluToContinToZZTo2mu2tau":{
        "path" : "/GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v2",
        "histAxisName" : "ZZTo4L_centralUL16APV",
        "xsecName" : "ggToZZTo2mu2tau",
    },
    "UL16APV_GluGluToContinToZZTo4e":{
        "path" : "/GluGluToContinToZZTo4e_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/GluGluToContinToZZTo4e_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v2",
        "histAxisName" : "ZZTo4L_centralUL16APV",
        "xsecName" : "ggToZZTo4e",
    },
    "UL16APV_GluGluToContinToZZTo4mu":{
        "path" : "/GluGluToContinToZZTo4mu_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/GluGluToContinToZZTo4mu_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v2",
        "histAxisName" : "ZZTo4L_centralUL16APV",
        "xsecName" : "ggToZZTo4mu",
    },
    "UL16APV_GluGluToContinToZZTo4tau":{
        "path" : "/GluGluToContinToZZTo4tau_TuneCP5_13TeV-mcfm701-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM",
        "path_local" : "/store/mc/RunIISummer20UL16NanoAODAPVv9/GluGluToContinToZZTo4tau_TuneCP5_13TeV-mcfm701-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v2",
        "histAxisName" : "ZZTo4L_centralUL16APV",
        "xsecName" : "ggToZZTo4tau",
    },
}

########### Data ##############
from collections import defaultdict

# For more info on the datasets and eras for each year
# See: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun2LegacyAnalysis

### 2016APV ###
year = 'Run2016'
naod_version  = "MiniAODv2_NanoAODv9-v1"
dataset_names = ["DoubleEG","DoubleMuon","SingleElectron","SingleMuon","MuonEG"]
dataset_eras = [# See: https://twiki.cern.ch/twiki/bin/view/CMS/PdmVDatasetsUL2016
    'B-ver1_HIPM_UL2016',
    'B-ver2_HIPM_UL2016',
    'C-HIPM_UL2016',
    'D-HIPM_UL2016',
    'E-HIPM_UL2016',
    'F-HIPM_UL2016',
]

version_overwrite = {
    'DoubleEG': {# See: https://pdmv-pages.web.cern.ch/rereco_ul/?input_dataset=DoubleEG%2FRun2016
        'B-ver1_HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'B-ver2_HIPM_UL2016': 'MiniAODv2_NanoAODv9-v3',
        'C-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'D-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'E-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'F-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
    },
    'DoubleMuon': {# See: https://pdmv-pages.web.cern.ch/rereco_ul/?input_dataset=DoubleMuon%2FRun2016
        'B-ver1_HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'B-ver2_HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'C-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'D-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'E-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'F-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
    },
    'MuonEG': {
        'B-ver1_HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'B-ver2_HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'C-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'D-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'E-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'F-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
    },
    'SingleElectron': {
        'B-ver1_HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'B-ver2_HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'C-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'D-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'E-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'F-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
    },
    'SingleMuon': {
        'B-ver1_HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'B-ver2_HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'C-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'D-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'E-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
        'F-HIPM_UL2016': 'MiniAODv2_NanoAODv9-v2',
    },
}

data_2016APV_dict = defaultdict(lambda: {'path': '','histAxisName': 'dataUL16APV', 'xsecName': 'data'})
for era in dataset_eras:
    for ds_name in dataset_names:
        key_name = "{name}_{era}".format(name=ds_name,era=era)
        version = naod_version
        if ds_name in version_overwrite:
            if era in version_overwrite[ds_name]:
                version = version_overwrite[ds_name][era]
        ds_path = "/{ds}/{year}{era}_{ver}/NANOAOD".format(year=year,ds=ds_name,era=era,ver=version)
        data_2016APV_dict[key_name]['path'] = ds_path


### 2016 ###
year = 'Run2016'
naod_version  = "MiniAODv2_NanoAODv9-v1"
dataset_names = ["DoubleEG","DoubleMuon","SingleElectron","SingleMuon","MuonEG"]
dataset_eras = [# See: https://twiki.cern.ch/twiki/bin/view/CMS/PdmVDatasetsUL2016
    'F-UL2016',
    'G-UL2016',
    'H-UL2016',
]

version_overwrite = {
    'DoubleMuon': {# See: https://pdmv-pages.web.cern.ch/rereco_ul/?input_dataset=DoubleMuon%2FRun2016
        'G-UL2016': 'MiniAODv2_NanoAODv9-v2',
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
naod_version  = "MiniAODv2_NanoAODv9-v1"
dataset_names = ["SingleMuon","SingleElectron","DoubleMuon","DoubleEG","MuonEG"]
dataset_eras = [# Note: Eras G and H correspond to 5 TeV and lowPU, so ignore them
    'B-UL2017',
    'C-UL2017',
    'D-UL2017',
    'E-UL2017',
    'F-UL2017',
]

version_overwrite = {
    "DoubleEG": {},
    "DoubleMuon": {},
    "MuonEG": {},
    "SingleElectron": {},
    "SingleMuon": {},
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
naod_version  = "MiniAODv2_NanoAODv9-v1"
dataset_names = ["SingleMuon","EGamma","DoubleMuon","MuonEG"]
dataset_eras = [
    'A-UL2018',
    'B-UL2018',
    'C-UL2018',
    'D-UL2018',
]
version_overwrite = {
    'SingleMuon': {# See: https://pdmv-pages.web.cern.ch/rereco_ul/?input_dataset=SingleMuon%2FRun2018
        'A-UL2018': 'MiniAODv2_NanoAODv9-v2',
        'B-UL2018': 'MiniAODv2_NanoAODv9-v2',
        'C-UL2018': 'MiniAODv2_NanoAODv9-v2',
    },
    'EGamma': {# See: https://pdmv-pages.web.cern.ch/rereco_ul/?input_dataset=EGamma%2FRun2018
        'D-UL2018': 'MiniAODv2_NanoAODv9-v3',
    },
    'DoubleMuon': {# See: https://pdmv-pages.web.cern.ch/rereco_ul/?input_dataset=DoubleMuon%2FRun2018
        'D-UL2018': 'MiniAODv2_NanoAODv9-v2',
    },
    'MuonEG': {}
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




# Uncomment the make_jsons_for_dict_of_samples() lines for the jsons you want to make/remake
def main():

    # Specify some output dirs
    out_dir_test_private_UL     = os.path.join(topcoffea_path("json"),"signal_samples/test_UL/")
    out_dir_private_UL          = os.path.join(topcoffea_path("json"),"signal_samples/private_UL/")
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
    #sjt.make_jsons_for_dict_of_samples(test_private_UL17_dict,"/hadoop","2017",out_dir_test_private_UL)
    #sjt.make_jsons_for_dict_of_samples(private_UL17_dict,"/hadoop","2017",out_dir_private_UL)
    #sjt.make_jsons_for_dict_of_samples(private_UL18_dict,"/hadoop","2018",out_dir_private_UL)
    #sjt.make_jsons_for_dict_of_samples(private_UL16_dict,"/hadoop","2016",out_dir_private_UL)
    #sjt.make_jsons_for_dict_of_samples(private_UL16APV_dict,"/hadoop","2016APV",out_dir_private_UL) # Not sure what we need here for the year, can remake the JSONs later to update when we have SFs etc set up for 2016 stuff (right now I think it's mostly just 18)

    # Subsets of files for small debugging tests local files (scratch365 at ND)
    #sjt.make_jsons_for_dict_of_samples(private_2017_dict,"","2017",out_dir_top19001_local)
    #sjt.make_jsons_for_dict_of_samples(private_UL17_dict_b1b4_local,"","2017",out_dir_private_UL_subset_local)

    # Central signal samples
    #sjt.make_jsons_for_dict_of_samples(central_2016_dict,"root://ndcms.crc.nd.edu/","2016",out_dir_central_2016,on_das=True)
    #sjt.make_jsons_for_dict_of_samples(central_2016APV_dict,"root://ndcms.crc.nd.edu/","2016APV",out_dir_central_2016APV,on_das=True)
    #sjt.make_jsons_for_dict_of_samples(central_2017_correctnPartonsInBorn_dict,"root://ndcms.crc.nd.edu/","2017",out_dir_central_2017,on_das=True)
    #sjt.make_jsons_for_dict_of_samples(central_2017_dict,"root://ndcms.crc.nd.edu/","2017",out_dir_central_2017,on_das=True)
    #sjt.make_jsons_for_dict_of_samples(central_2017_dict,"/hadoop","2017",out_dir_central_2017,on_das=False) # ttH, ttW, ttZ, and tZq are at ND
    #sjt.make_jsons_for_dict_of_samples(sync_dict,"root://ndcms.crc.nd.edu/","2017",out_dir_central_sync)
    #sjt.make_jsons_for_dict_of_samples(central_UL16_dict,    "/hadoop","2016",    out_dir_central_UL,on_das=False) # Central signal samples ar at ND now
    #sjt.make_jsons_for_dict_of_samples(central_UL16APV_dict, "/hadoop","2016APV", out_dir_central_UL,on_das=False) # Central signal samples ar at ND now
    #sjt.make_jsons_for_dict_of_samples(central_UL17_dict,    "/hadoop","2017",    out_dir_central_UL,on_das=False) # Central signal samples ar at ND now
    #sjt.make_jsons_for_dict_of_samples(central_UL18_dict,    "/hadoop","2018",    out_dir_central_UL,on_das=False) # Central signal samples ar at ND now

    # Central background samples
    # Note: Some of the bkg dicts have both a "path" and a "path_local" (these are samples that generated JSONs for after moving the samples to ND),
    #       while the others only have a "path" (i.e. dataset name), these were the ones we produced JSONs for prior to moving the samples to ND, but
    #       these should also be located at ND, so if the samples need to be remade, you can add a "local_path" with the path starting with /store
    sjt.make_jsons_for_dict_of_samples(central_UL17_bkg_dict,   "/hadoop","2017",   out_dir_central_bkg_UL,on_das=False) # Background samples are at ND now
    sjt.make_jsons_for_dict_of_samples(central_UL18_bkg_dict,   "/hadoop","2018",   out_dir_central_bkg_UL,on_das=False) # Background samples are at ND now
    sjt.make_jsons_for_dict_of_samples(central_UL16_bkg_dict,   "/hadoop","2016",   out_dir_central_bkg_UL,on_das=False) # Background samples are at ND now
    sjt.make_jsons_for_dict_of_samples(central_UL16APV_bkg_dict,"/hadoop","2016APV",out_dir_central_bkg_UL,on_das=False) # Background samples are at ND now

    # Data samples
    #sjt.make_jsons_for_dict_of_samples(data_2016APV_dict,"root://ndcms.crc.nd.edu/","2016APV",out_dir_data_2016,on_das=True)
    #sjt.make_jsons_for_dict_of_samples(data_2016_dict,"root://ndcms.crc.nd.edu/","2016",out_dir_data_2016,on_das=True)
    #sjt.make_jsons_for_dict_of_samples(data_2017_dict,"root://ndcms.crc.nd.edu/","2017",out_dir_data_2017,on_das=True)
    #sjt.make_jsons_for_dict_of_samples(data_2018_dict,"root://ndcms.crc.nd.edu/","2018",out_dir_data_2018,on_das=True)

    # Testing finding list of files with xrdfs ls
    #sjt.make_jsons_for_dict_of_samples(test_dict,"root://xrootd-local.unl.edu/","2017",".")


    ######### Just replace xsec in JSON with whatever is in xsec.cfg #########

    #sjt.replace_xsec_for_dict_of_samples(test_private_UL17_dict,out_dir_test_private_UL)
    #sjt.replace_xsec_for_dict_of_samples(private_UL17_dict,out_dir_private_UL)
    #sjt.replace_xsec_for_dict_of_samples(private_UL18_dict,out_dir_private_UL)
    #sjt.replace_xsec_for_dict_of_samples(private_UL16_dict,out_dir_private_UL)
    #sjt.replace_xsec_for_dict_of_samples(private_UL16APV_dict,out_dir_private_UL)

    #sjt.replace_xsec_for_dict_of_samples(private_2017_dict,out_dir_top19001_local)
    #sjt.replace_xsec_for_dict_of_samples(private_UL17_dict_b1b4_local,out_dir_private_UL_subset_local)

    ##sjt.replace_xsec_for_dict_of_samples(central_2016_dict,out_dir_central_2016)
    ##sjt.replace_xsec_for_dict_of_samples(central_2016APV_dict,out_dir_central_2016APV)
    #sjt.replace_xsec_for_dict_of_samples(central_2017_correctnPartonsInBorn_dict,out_dir_central_2017)
    #sjt.replace_xsec_for_dict_of_samples(central_2017_dict,out_dir_central_2017)
    #sjt.replace_xsec_for_dict_of_samples(central_2017_dict,out_dir_central_2017)
    #sjt.replace_xsec_for_dict_of_samples(sync_dict,out_dir_central_sync)
    #sjt.replace_xsec_for_dict_of_samples(central_UL16_dict,out_dir_central_UL)
    #sjt.replace_xsec_for_dict_of_samples(central_UL16APV_dict,out_dir_central_UL)
    #sjt.replace_xsec_for_dict_of_samples(central_UL17_dict,out_dir_central_UL)
    #sjt.replace_xsec_for_dict_of_samples(central_UL18_dict,out_dir_central_UL)

    #sjt.replace_xsec_for_dict_of_samples(central_UL17_bkg_dict,out_dir_central_bkg_UL)
    #sjt.replace_xsec_for_dict_of_samples(central_UL18_bkg_dict,out_dir_central_bkg_UL)
    #sjt.replace_xsec_for_dict_of_samples(central_UL16_bkg_dict,out_dir_central_bkg_UL)
    #sjt.replace_xsec_for_dict_of_samples(central_UL16APV_bkg_dict,out_dir_central_bkg_UL)


if __name__ == "__main__":
    main()
