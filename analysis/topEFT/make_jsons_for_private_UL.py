# This file is essentially a wrapper for createJSON.py:
#   - It runs createJSON.py for each batch of the private UL samples, and moves the resulting json file to the topcoffea/json/signal_samples directory
#   - If the private NAOD has to be remade, the version numbers should be updated in the dictionaries here, then just rerun the script to remake the jsons

import subprocess
import os

test_UL17_dict = {
    "UL17_ttHJet_b1"   : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v1/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
}

private_UL17_dict = {

    "UL17_ttHJet_b1"   : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v2/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
    "UL17_ttHJet_b2"   : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch2/naodOnly_step/v2/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
    "UL17_ttHJet_b3"   : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch3/naodOnly_step/v3/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0",

    "UL17_ttlnuJet_b1" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v2/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
    "UL17_ttlnuJet_b2" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch2/naodOnly_step/v2/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",
    "UL17_ttlnuJet_b3" : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch3/naodOnly_step/v3/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0",

    "UL17_ttllJet_b1"  : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v2/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
    "UL17_ttllJet_b2"  : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch2/naodOnly_step/v2/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",
    "UL17_ttllJet_b3"  : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch3/naodOnly_step/v3/nAOD_step_ttllNuNuJetNoHiggs_all22WCsStartPtCheckdim6TopMay20GST_run0",

    "UL17_tllq_b1"     : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v2/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
    "UL17_tllq_b2"     : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch2/naodOnly_step/v2/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",
    "UL17_tllq_b3"     : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch3/naodOnly_step/v3/nAOD_step_tllq4fNoSchanWNoHiggs0p_all22WCsStartPtCheckV2dim6TopMay20GST_run0",

    "UL17_tHq_b1"      : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v2/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
    "UL17_tHq_b2"      : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch2/naodOnly_step/v2/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",
    "UL17_tHq_b3"      : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch3/naodOnly_step/v3/nAOD_step_tHq4f_all22WCsStartPtCheckdim6TopMay20GST_run0",

    "UL17_tttt_b4"     : "/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch4/naodOnly_step/v1/nAOD_step_tttt_FourtopsMay3v1_run0",
}

private_UL18_dict = {

    "UL18_ttHJet_b1"   : "",
    "UL18_ttHJet_b2"   : "",
    "UL18_ttHJet_b3"   : "",

    "UL18_ttlnuJet_b1" : "",
    "UL18_ttlnuJet_b2" : "",
    "UL18_ttlnuJet_b3" : "",

    "UL18_ttllJet_b1"  : "",
    "UL18_ttllJet_b2"  : "",
    "UL18_ttllJet_b3"  : "",

    "UL18_tllq_b1"     : "",
    "UL18_tllq_b2"     : "",
    "UL18_tllq_b3"     : "",

    "UL18_tHq_b1"      : "",
    "UL18_tHq_b2"      : "",
    "UL18_tHq_b3"      : "",

    "UL18_tttt_b4"     : "",
}

private_UL16_dict = {

    "UL16_ttHJet_b1"   : "",
    "UL16_ttllJet_b1"  : "",
    "UL16_ttlnuJet_b1" : "",
    "UL16_tllq_b1"     : "",
    "UL16_tHq_b1"      : "",
    "UL16_tttt_b1"     : "",
}

private_UL16APV_dict = {

    "UL16APV_ttHJet_b1"   : "",
    "UL16APV_ttllJet_b1"  : "",
    "UL16APV_ttlnuJet_b1" : "",
    "UL16APV_tllq_b1"     : "",
    "UL16APV_tHq_b1"      : "",
    "UL16APV_tttt_b1"     : "",
}

def find_xsec_name(sample_name):
    name_map = {
        "ttH"   : "ttHnobb",
        "ttlnu" : "TTWJetsToLNu",
        "ttll"  : "TTZToLLNuNu_M_10",
        "tllq"  : "tZq",
        "tHq"   : "tHq",
        "tttt"  : "tttt",
    }
    for process_name,xsec_name in name_map.items():
        if process_name in sample_name:
            xsec_name_match = xsec_name
            return xsec_name

def make_json(sample_dir,sample_name,prefix,sample_yr,xsec_name):
    subprocess.run([
        "python",
        "../../topcoffea/modules/createJSON.py",
        sample_dir,
        "--sampleName", sample_name,
        "--prefix"    , prefix,
        "--xsec"      , "../../topcoffea/cfg/xsec.cfg",
        "--xsecName"  , xsec_name,
        "--year"      , sample_yr,
    ])


def main():

    # Local example
    #sample_dir = "/scratch365/kmohrman/mc_files/all_17_18_with_subdirs/ttH_top19001/"
    #sample_name = "Tree_ttH"
    #sample_yr = "2018"
    #xsec_name = "TTTo2L2Nu"
    #make_json(sample_dir,sample_name,sample_yr,xsec_name)

    out_dir = "../../topcoffea/json/signal_samples/"

    for sname,sdir in private_UL17_dict.items():
        make_json(sdir,sname,"/hadoop","2017",find_xsec_name(sname)) # Takes about 30 min
        out_name = sname+".json"
        subprocess.run(["mv",out_name,out_dir]) 

main()
