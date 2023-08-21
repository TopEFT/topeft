import os
from topcoffea.modules.paths import topcoffea_path
import topcoffea.modules.sample_lst_jsons_tools as sjt


############################ Bkg samples ############################


central_UL16APV_bkg_dict = {

    "UL16APV_ZZTo4L" : {
        "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1_NANOAODSIM_3LepTau_4Lep",
        "histAxisName": "UL16APV_ZZTo4l",
        "xsecName": "ZZTo4L",
    },

    "UL16APV_ggToZZTo2e2mu"   : { "histAxisName" : "UL16APV_ggToZZTo2e2mu"    , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo2e2mu_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2_NANOAODSIM_3LepTau_4Lep"    , "xsecName" : "ggToZZTo2e2muK" , } ,
    "UL16APV_ggToZZTo2e2tau"  : { "histAxisName" : "UL16APV_ggToZZTo2e2tau"   , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo2e2tau_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2_NANOAODSIM_3LepTau_4Lep"   , "xsecName" : "ggToZZTo2e2tauK" , } ,
    "UL16APV_ggToZZTo2mu2tau" : { "histAxisName" : "UL16APV_ggToZZTo2mu2tau"  , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2_NANOAODSIM_3LepTau_4Lep"  , "xsecName" : "ggToZZTo2mu2tauK" , } ,
    "UL16APV_ggToZZTo4e"      : { "histAxisName" : "UL16APV_ggToZZTo4e"       , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo4e_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2_NANOAODSIM_3LepTau_4Lep"       , "xsecName" : "ggToZZTo4eK" , } ,
    "UL16APV_ggToZZTo4mu"     : { "histAxisName" : "UL16APV_ggToZZTo4mu"      , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo4mu_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2_NANOAODSIM_3LepTau_4Lep"      , "xsecName" : "ggToZZTo4muK" , } ,
    "UL16APV_ggToZZTo4tau"    : { "histAxisName" : "UL16APV_ggToZZTo4tau"     , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo4tau_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2_NANOAODSIM_3LepTau_4Lep"     , "xsecName" : "ggToZZTo4tauK" , } ,
}

central_UL16_bkg_dict = {

    "UL16_ZZTo4L" : {
        "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1_NANOAODSIM_3LepTau_4Lep",
        "histAxisName": "UL16_ZZTo4l",
        "xsecName": "ZZTo4L",
    },

    "UL16_ggToZZTo2e2mu"      : { "histAxisName" : "UL16_ggToZZTo2e2mu"       , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo2e2mu_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1_NANOAODSIM_3LepTau_4Lep"              , "xsecName" : "ggToZZTo2e2muK" , } ,
    "UL16_ggToZZTo2e2tau"     : { "histAxisName" : "UL16_ggToZZTo2e2tau"      , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo2e2tau_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1_NANOAODSIM_3LepTau_4Lep"             , "xsecName" : "ggToZZTo2e2tauK" , } ,
    "UL16_ggToZZTo2mu2tau"    : { "histAxisName" : "UL16_ggToZZTo2mu2tau"     , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1_NANOAODSIM_3LepTau_4Lep"            , "xsecName" : "ggToZZTo2mu2tauK" , } ,
    "UL16_ggToZZTo4e"         : { "histAxisName" : "UL16_ggToZZTo4e"          , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo4e_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2_NANOAODSIM_3LepTau_4Lep"                 , "xsecName" : "ggToZZTo4eK" , } ,
    "UL16_ggToZZTo4mu"        : { "histAxisName" : "UL16_ggToZZTo4mu"         , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo4mu_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2_NANOAODSIM_3LepTau_4Lep"                , "xsecName" : "ggToZZTo4muK" , } ,
    "UL16_ggToZZTo4tau"       : { "histAxisName" : "UL16_ggToZZTo4tau"        , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo4tau_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1_NANOAODSIM_3LepTau_4Lep"               , "xsecName" : "ggToZZTo4tauK" , } ,
}

central_UL17_bkg_dict = {

    "UL17_ZZTo4L" : {
        "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2_NANOAODSIM_3LepTau_4Lep",
        "histAxisName": "UL17_ZZTo4l",
        "xsecName": "ZZTo4L",
    },

    "UL17_ggToZZTo2e2mu"      : { "histAxisName" : "UL17_ggToZZTo2e2mu"       , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo2e2mu_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2_NANOAODSIM_3LepTau_4Lep"                , "xsecName" : "ggToZZTo2e2muK" , } ,
    "UL17_ggToZZTo2e2tau"     : { "histAxisName" : "UL17_ggToZZTo2e2tau"      , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo2e2tau_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2_NANOAODSIM_3LepTau_4Lep"               , "xsecName" : "ggToZZTo2e2tauK" , } ,
    "UL17_ggToZZTo2mu2tau"    : { "histAxisName" : "UL17_ggToZZTo2mu2tau"     , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2_NANOAODSIM_3LepTau_4Lep"              , "xsecName" : "ggToZZTo2mu2tauK" , } ,
    "UL17_ggToZZTo4e"         : { "histAxisName" : "UL17_ggToZZTo4e"          , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo4e_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2_NANOAODSIM_3LepTau_4Lep"                   , "xsecName" : "ggToZZTo4eK" , } ,
    "UL17_ggToZZTo4mu"        : { "histAxisName" : "UL17_ggToZZTo4mu"         , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo4mu_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2_NANOAODSIM_3LepTau_4Lep"                  , "xsecName" : "ggToZZTo4muK" , } ,
    "UL17_ggToZZTo4tau"       : { "histAxisName" : "UL17_ggToZZTo4tau"        , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo4tau_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2_NANOAODSIM_3LepTau_4Lep"                 , "xsecName" : "ggToZZTo4tauK" , } ,
}

central_UL18_bkg_dict = {

    "UL18_ZZTo4L" : {
        "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2_NANOAODSIM_3LepTau_4Lep",
        "histAxisName": "UL18_ZZTo4l",
        "xsecName": "ZZTo4L",
    },

    "UL18_ggToZZTo2e2mu"      : { "histAxisName" : "UL18_ggToZZTo2e2mu"       , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo2e2mu_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2_NANOAODSIM_3LepTau_4Lep"     , "xsecName" : "ggToZZTo2e2muK" , } ,
    "UL18_ggToZZTo2e2tau"     : { "histAxisName" : "UL18_ggToZZTo2e2tau"      , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo2e2tau_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2_NANOAODSIM_3LepTau_4Lep"    , "xsecName" : "ggToZZTo2e2tauK" , } ,
    "UL18_ggToZZTo2mu2tau"    : { "histAxisName" : "UL18_ggToZZTo2mu2tau"     , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2_NANOAODSIM_3LepTau_4Lep"   , "xsecName" : "ggToZZTo2mu2tauK" , } ,
    "UL18_ggToZZTo4e"         : { "histAxisName" : "UL18_ggToZZTo4e"          , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo4e_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2_NANOAODSIM_3LepTau_4Lep"        , "xsecName" : "ggToZZTo4eK" , } ,
    "UL18_ggToZZTo4mu"        : { "histAxisName" : "UL18_ggToZZTo4mu"         , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo4mu_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2_NANOAODSIM_3LepTau_4Lep"       , "xsecName" : "ggToZZTo4muK" , } ,
    "UL18_ggToZZTo4tau"       : { "histAxisName" : "UL18_ggToZZTo4tau"        , "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluToContinToZZTo4tau_TuneCP5_13TeV-mcfm701-pythia8_RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2_NANOAODSIM_3LepTau_4Lep"      , "xsecName" : "ggToZZTo4tauK" , } ,
}


############################ Signal samples ############################

central_UL16APV_sig_dict = {
    "UL16APV_WWZJetsTo4L2Nu" : {
        "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/WWZJetsTo4L2Nu_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2_NANOAODSIM_3LepTau_4Lep",
        "histAxisName": "UL16APV_WWZJetsTo4L2Nu",
        "xsecName": "WWZ4l",
    },
    "UL16APV_GluGluZH" : {
        "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluZH_HToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1_NANOAODSIM_3LepTau_4Lep",
        "histAxisName": "UL16APV_GluGluZH",
        "xsecName": "ZH",
    },
}

central_UL16_sig_dict = {
    "UL16_WWZJetsTo4L2Nu" : {
        "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/WWZJetsTo4L2Nu_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2_NANOAODSIM_3LepTau_4Lep",
        "histAxisName": "UL16_WWZJetsTo4L2Nu",
        "xsecName": "WWZ4l",
    },
    "UL16_GluGluZH" : {
        "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluZH_HToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2_NANOAODSIM_3LepTau_4Lep",
        "histAxisName": "UL16_GluGluZH",
        "xsecName": "ZH",
    },
}

central_UL17_sig_dict = {
    "UL17_WWZJetsTo4L2Nu" : {
        "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/WWZJetsTo4L2Nu_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2_NANOAODSIM_3LepTau_4Lep",
        "histAxisName": "UL17_WWZJetsTo4L2Nu",
        "xsecName": "WWZ4l",
    },
    "UL17_GluGluZH" : {
        "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluZH_HToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2_NANOAODSIM_3LepTau_4Lep",
        "histAxisName": "UL17_GluGluZH",
        "xsecName": "ZH",
    },
}

central_UL18_sig_dict = {
    "UL18_WWZJetsTo4L2Nu" : {
        "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/WWZJetsTo4L2Nu_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2_NANOAODSIM_3LepTau_4Lep",
        "histAxisName": "UL18_WWZJetsTo4L2Nu",
        "xsecName": "WWZ4l",
    },
    "UL18_GluGluZH" : {
        "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluZH_HToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2_NANOAODSIM_3LepTau_4Lep",
        "histAxisName": "UL18_GluGluZH",
        "xsecName": "ZH",
    },
}




############################ Test example samples ############################

# Test dict
test_wwz_dict = {
    "UL17_WWZJetsTo4L2Nu" : {
        "path" : "/store/user/kmohrman/samples/from_keegan_skims_3LepTau_4Lep/WWZJetsTo4L2Nu_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2_NANOAODSIM_3LepTau_4Lep",
        "histAxisName": "UL17_WWZJetsTo4L2Nu",
        "xsecName": "WWZ4l",
    },
}


############################ Main ############################

# Uncomment the make_jsons_for_dict_of_samples() lines for the jsons you want to make/remake
def main():

    # A simple example
    #sjt.make_jsons_for_dict_of_samples(test_wwz_dict, "/ceph/cms/","2017",".",on_das=False) # An example

    # Specify output paths
    out_dir_sig = os.path.join(topcoffea_path("json"),"wwz_analysis_samples/sig_samples/")
    out_dir_bkg = os.path.join(topcoffea_path("json"),"wwz_analysis_samples/bkg_samples/")

    # Make configs for signal samples
    #sjt.make_jsons_for_dict_of_samples(central_UL16APV_bkg_dict, "/ceph/cms/","2016APV", out_dir_sig,on_das=False)
    #sjt.make_jsons_for_dict_of_samples(central_UL16_bkg_dict, "/ceph/cms/","2016", out_dir_sig,on_das=False)
    #sjt.make_jsons_for_dict_of_samples(central_UL17_bkg_dict, "/ceph/cms/","2017", out_dir_sig,on_das=False)
    #sjt.make_jsons_for_dict_of_samples(central_UL18_bkg_dict, "/ceph/cms/","2018", out_dir_sig,on_das=False)

    # Make configs for bkg samples
    sjt.make_jsons_for_dict_of_samples(central_UL16APV_sig_dict, "/ceph/cms/","2016APV", out_dir_bkg,on_das=False)
    sjt.make_jsons_for_dict_of_samples(central_UL16_sig_dict, "/ceph/cms/","2016", out_dir_bkg,on_das=False)
    sjt.make_jsons_for_dict_of_samples(central_UL17_sig_dict, "/ceph/cms/","2017", out_dir_bkg,on_das=False)
    sjt.make_jsons_for_dict_of_samples(central_UL18_sig_dict, "/ceph/cms/","2018", out_dir_bkg,on_das=False)

    # Replace xsec numbers
    #sjt.replace_xsec_for_dict_of_samples(central_UL16APV_bkg_dict,out_dir_bkg)
    #sjt.replace_xsec_for_dict_of_samples(central_UL16_bkg_dict,out_dir_bkg)
    #sjt.replace_xsec_for_dict_of_samples(central_UL17_bkg_dict,out_dir_bkg)
    #sjt.replace_xsec_for_dict_of_samples(central_UL18_bkg_dict,out_dir_bkg)


if __name__ == "__main__":
    main()
