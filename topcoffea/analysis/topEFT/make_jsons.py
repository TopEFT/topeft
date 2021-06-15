# This file is essentially a wrapper for createJSON.py:
#   - It runs createJSON.py for each sample that you include in a dictionary, and moves the resulting json file to the directory you specify
#   - If the private NAOD has to be remade, the version numbers should be updated in the dictionaries here, then just rerun the script to remake the jsons

import subprocess
import os
from topcoffea.modules.paths import topcoffea_path


########### Private UL signal samples ###########

private_UL17_dict = {

    "UL17_ttHJet_b1"   : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v2/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttHJet_privateUL17",
        "xsecName": "ttHnobb",
    },
    "UL17_ttHJet_b2"   : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch2/naodOnly_step/v2/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttHJet_privateUL17",
        "xsecName": "ttHnobb",
    },
    "UL17_ttHJet_b3"   : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch3/naodOnly_step/v3/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttHJet_privateUL17",
        "xsecName": "ttHnobb",
    },

    "UL17_ttlnuJet_b1" : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v2/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttlnuJet_privateUL17",
        "xsecName": "TTWJetsToLNu",
    },
    "UL17_ttlnuJet_b2" : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch2/naodOnly_step/v2/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttlnuJet_privateUL17",
        "xsecName": "TTWJetsToLNu",
    },
    "UL17_ttlnuJet_b3" : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch3/naodOnly_step/v3/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttlnuJet_privateUL17",
        "xsecName": "TTWJetsToLNu",
    },

    "UL17_ttllJet_b1"  : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v2/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttllJet_privateUL17",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL17_ttllJet_b2"  : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch2/naodOnly_step/v2/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttllJet_privateUL17",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL17_ttllJet_b3"  : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch3/naodOnly_step/v3/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttllJet_privateUL17",
        "xsecName": "TTZToLLNuNu_M_10",
    },

    "UL17_tllq_b1"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v2/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
        "histAxisName" : "tllq_privateUL17",
        "xsecName": "tZq",
    },
    "UL17_tllq_b2"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch2/naodOnly_step/v2/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
        "histAxisName": "tllq_privateUL17",
        "xsecName": "tZq",
    },
    "UL17_tllq_b3"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch3/naodOnly_step/v3/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
        "histAxisName": "tllq_privateUL17",
        "xsecName": "tZq",
    },

    "UL17_tHq_b1"      : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v2/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "tHq_privateUL17",
        "xsecName": "tHq",
    },
    "UL17_tHq_b2"      : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch2/naodOnly_step/v2/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "tHq_privateUL17",
        "xsecName": "tHq",
    },
    "UL17_tHq_b3"      : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch3/naodOnly_step/v3/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "tHq_privateUL17",
        "xsecName": "tHq",
    },

    "UL17_tttt_b4"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch4/naodOnly_step/v1/nAOD_step_tttt_FourtopsMay3v1_run0",
        "histAxisName": "tttt_privateUL17",
        "xsecName": "tttt",
    },
}

private_UL18_dict = {

    "UL18_ttHJet_b1"   : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch1/naodOnly_step/v4/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttHJet_privateUL18",
        "xsecName": "ttHnobb",
    },
    "UL18_ttHJet_b2"   : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch2/naodOnly_step/v1/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttHJet_privateUL18",
        "xsecName": "ttHnobb",
    },
    "UL18_ttHJet_b3"   : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch3/naodOnly_step/v1/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttHJet_privateUL18",
        "xsecName": "ttHnobb",
    },

    "UL18_ttlnuJet_b1" : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch1/naodOnly_step/v4/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttlnuJet_privateUL18",
        "xsecName": "TTWJetsToLNu",
    },
    "UL18_ttlnuJet_b2" : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch2/naodOnly_step/v1/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttlnuJet_privateUL18",
        "xsecName": "TTWJetsToLNu",
    },
    "UL18_ttlnuJet_b3" : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch3/naodOnly_step/v1/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttlnuJet_privateUL18",
        "xsecName": "TTWJetsToLNu",
    },

    "UL18_ttllJet_b1"  : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch1/naodOnly_step/v4/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttllJet_privateUL18",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL18_ttllJet_b2"  : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch2/naodOnly_step/v1/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttllJet_privateUL18",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL18_ttllJet_b3"  : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch3/naodOnly_step/v1/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttllJet_privateUL18",
        "xsecName": "TTZToLLNuNu_M_10",
    },

    "UL18_tllq_b1"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch1/naodOnly_step/v4/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
        "histAxisName": "tllq_privateUL18",
        "xsecName": "tZq",
    },
    "UL18_tllq_b2"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch2/naodOnly_step/v1/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
        "histAxisName": "tllq_privateUL18",
        "xsecName": "tZq",
    },
    "UL18_tllq_b3"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch3/naodOnly_step/v1/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
        "histAxisName": "tllq_privateUL18",
        "xsecName": "tZq",
    },

    "UL18_tHq_b1"      : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch1/naodOnly_step/v4/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "tHq_privateUL18",
        "xsecName": "tHq",
    },
    "UL18_tHq_b2"      : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch2/naodOnly_step/v1/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "tHq_privateUL18",
        "xsecName": "tHq",
    },
    "UL18_tHq_b3"      : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch3/naodOnly_step/v1/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "tHq_privateUL18",
        "xsecName": "tHq",
    },

    "UL18_tttt_b4"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL18/Round1/Batch4/naodOnly_step/v1/nAOD_step_tttt_FourtopsMay3v1_run0",
        "histAxisName": "tttt_privateUL18",
        "xsecName": "tttt",
    },
}

private_UL16_dict = {

    "UL16_ttHJet_b1"   : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16/Round1/Batch1/naodOnly_step/v1/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttHJet_privateUL16",
        "xsecName": "ttHnobb",
    },
    "UL16_ttlnuJet_b1" : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16/Round1/Batch1/naodOnly_step/v1/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttlnuJet_privateUL16",
        "xsecName": "TTWJetsToLNu",
    },
    "UL16_ttllJet_b1"  : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16/Round1/Batch1/naodOnly_step/v1/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "ttllJet_privateUL16",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL16_tllq_b1"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16/Round1/Batch1/naodOnly_step/v1/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
        "histAxisName": "tllq_privateUL16",
        "xsecName": "tZq",
    },
    "UL16_tHq_b1"      : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16/Round1/Batch1/naodOnly_step/v1/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
        "histAxisName": "tHq_privateUL16",
        "xsecName": "tHq",
    },
    "UL16_tttt_b1"     : {
        "path" : "/store/user/kmohrman/FullProduction/FullR2/UL16/Round1/Batch1/naodOnly_step/v1/nAOD_step_tttt_FourtopsMay3v1_run0",
        "histAxisName": "tttt_privateUL16",
        "xsecName": "tttt",
    },
}

private_UL16APV_dict = {

    "UL16APV_ttHJet_b1"   : {
        "path" : "",
        "histAxisName": "ttHJet_privateUL16APV",
        "xsecName": "ttHnobb",
    },
    "UL16APV_ttlnuJet_b1" : {
        "path" : "",
        "histAxisName": "ttlnuJet_privateUL16APV",
        "xsecName": "TTWJetsToLNu",
    },
    "UL16APV_ttllJet_b1"  : {
        "path" : "",
        "histAxisName": "ttllJet_privateUL16APV",
        "xsecName": "TTZToLLNuNu_M_10",
    },
    "UL16APV_tllq_b1"     : {
        "path" : "",
        "histAxisName": "tllq_privateUL16APV",
        "xsecName": "tZq",
    },
    "UL16APV_tHq_b1"      : {
        "path" : "",
        "histAxisName": "tHq_privateUL16APV",
        "xsecName": "tHq",
    },
    "UL16APV_tttt_b1"     : {
        "path" : "",
        "histAxisName": "tttt_privateUL16APV",
        "xsecName": "tttt",
    },
}


########### TOP-19-001 samples (locally at ND on /scratch365) ###########

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
    #"2017_ttllJet" : {
    #    "path" : "/scratch365/kmohrman/mc_files/TOP-19-001/ttll/",
    #    "histAxisName": "ttll_TOP-19-001",
    #    "xsecName": "TTZToLLNuNu_M_10",
    #},
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



########### Central signal samples ###########

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
        "path" : "/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM",
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

########### TESTING ########### 

test_dict = {
    "test_dy_sample" : {
        "path" : "/store/user/jrgonzal/nanoAODcrab/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/mc2017_28apr2021_DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/210427_231758/0000/",
        "histAxisName": "central_DY",
        "xsecName" : "DYJetsToLL_M_50_MLM", # Not sure if this is actually the right xsec, I just picked one of the DY ones
    }
}

########### Functions for makign the jsons ###########

# Wrapper for createJSON.py
def make_json(sample_dir,sample_name,prefix,sample_yr,xsec_name,hist_axis_name,on_das=False):

    # If the sample is on DAS, inclue the DAS flag in the createJSON.py arguments
    das_flag = ""
    if on_das: das_flag = "--DAS"

    # Run createJSON.py
    subprocess.run([
        "python",
        "../../topcoffea/modules/createJSON.py",
        sample_dir,
        das_flag,
        "--sampleName"   , sample_name,
        "--prefix"       , prefix,
        "--xsec"         , "../../topcoffea/cfg/xsec.cfg",
        "--xsecName"     , xsec_name,
        "--year"         , sample_yr,
        "--histAxisName" , hist_axis_name,
    ])


# Convenience function for running make_json() on all entries in a dictionary of samples, and moving the results to out_dir
def make_jsons_for_dict_of_samples(samples_dict,prefix,year,out_dir,on_das=False):
    for sample_name,sample_info in samples_dict.items():
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
        print("sample name:",sample_name)
        print("\tpath:",path,"\n\thistAxisName:",hist_axis_name,"\n\txsecName",xsec_name,"\n\tout name:",out_name,"\n\tout dir:",out_dir)


# Uncomment the make_jsons_for_dict_of_samples() lines for the jsons you want to make/remake
def main():

    # Specify some output dirs
    out_dir_private_UL     = os.path.join(topcoffea_path("json"),"signal_samples/private_UL/")
    out_dir_top19001_local = os.path.join(topcoffea_path("json"),"signal_samples/private_top19001_local")
    out_dir_central_UL     = os.path.join(topcoffea_path("json"),"signal_samples/central_UL/")
    out_dir_central_2017   = os.path.join(topcoffea_path("json"),"signal_samples/central_2017/")

    # Private UL
    #make_jsons_for_dict_of_samples(private_UL17_dict,"/hadoop","2017",out_dir_private_UL)
    #make_jsons_for_dict_of_samples(private_UL18_dict,"/hadoop","2018",out_dir_private_UL)
    #make_jsons_for_dict_of_samples(private_UL16_dict,"/hadoop","2016",out_dir_private_UL)

    # TOP-19-001 ttll
    make_jsons_for_dict_of_samples(private_2017_dict,"","2017",out_dir_top19001_local)

    # Central
    #make_jsons_for_dict_of_samples(central_2017_correctnPartonsInBorn_dict,"root://ndcms.crc.nd.edu/","2017",out_dir_central_2017,on_das=True)
    #make_jsons_for_dict_of_samples(central_2017_dict,"root://ndcms.crc.nd.edu/","2017",out_dir_central_2017,on_das=True)
    #make_jsons_for_dict_of_samples(central_UL17_dict,"root://ndcms.crc.nd.edu/","2017",out_dir_central_UL,on_das=True)
    #make_jsons_for_dict_of_samples(central_UL18_dict,"root://ndcms.crc.nd.edu/","2018",out_dir_central_UL,on_das=True)

    # Testing finding list of files with xrdfs ls
    #make_jsons_for_dict_of_samples(test_dict,"root://xrootd-local.unl.edu/","2017",".")


main()
