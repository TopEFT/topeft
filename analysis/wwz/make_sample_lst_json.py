import os
from topcoffea.modules.paths import topcoffea_path
import topcoffea.modules.sample_lst_jsons_tools as sjt

############################ Signal samples ############################

central_UL16APV_sig_dict = {
    #"UL16APV_WWZJetsTo4L2Nu" : {
    #    "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/WWZJetsTo4L2Nu_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2_NANOAODSIM_3LepTau_4Lep",
    #    "histAxisName": "UL16APV_WWZJetsTo4L2Nu",
    #    "xsecName": "WWZ4l",
    #},
    "UL16APV_GluGluZH" : {
        "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluZH_HToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1_NANOAODSIM_3LepTau_4Lep",
        "histAxisName": "UL16APV_GluGluZH",
        "xsecName": "ZH",
    },
}

central_UL16_sig_dict = {
    #"UL16_WWZJetsTo4L2Nu" : {
    #    "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/WWZJetsTo4L2Nu_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2_NANOAODSIM_3LepTau_4Lep",
    #    "histAxisName": "UL16_WWZJetsTo4L2Nu",
    #    "xsecName": "WWZ4l",
    #},
    "UL16_GluGluZH" : {
        "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluZH_HToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2_NANOAODSIM_3LepTau_4Lep",
        "histAxisName": "UL16_GluGluZH",
        "xsecName": "ZH",
    },
}

central_UL17_sig_dict = {
    #"UL17_WWZJetsTo4L2Nu" : {
    #    "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/WWZJetsTo4L2Nu_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2_NANOAODSIM_3LepTau_4Lep",
    #    "histAxisName": "UL17_WWZJetsTo4L2Nu",
    #    "xsecName": "WWZ4l",
    #},
    "UL17_GluGluZH" : {
        "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluZH_HToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2_NANOAODSIM_3LepTau_4Lep",
        "histAxisName": "UL17_GluGluZH",
        "xsecName": "ZH",
    },
}

central_UL18_sig_dict = {
    #"UL18_WWZJetsTo4L2Nu" : {
    #    "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/WWZJetsTo4L2Nu_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2_NANOAODSIM_3LepTau_4Lep",
    #    "histAxisName": "UL18_WWZJetsTo4L2Nu",
    #    "xsecName": "WWZ4l",
    #},
    "UL18_GluGluZH" : {
        "path" : "/store/user/kdownham/skimOutput/3LepTau_4Lep/GluGluZH_HToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1",
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

    out_dir_sig = os.path.join(topcoffea_path("json"),"wwz_analysis_samples/sig_samples/")

    sjt.make_jsons_for_dict_of_samples(central_UL16APV_sig_dict, "/ceph/cms/","2016APV", out_dir_sig,on_das=False)
    sjt.make_jsons_for_dict_of_samples(central_UL16_sig_dict, "/ceph/cms/","2016", out_dir_sig,on_das=False)
    sjt.make_jsons_for_dict_of_samples(central_UL17_sig_dict, "/ceph/cms/","2017", out_dir_sig,on_das=False)
    sjt.make_jsons_for_dict_of_samples(central_UL18_sig_dict, "/ceph/cms/","2018", out_dir_sig,on_das=False)

    sjt.make_jsons_for_dict_of_samples(test_wwz_dict, "/ceph/cms/","2017",".",on_das=False) # An example


if __name__ == "__main__":
    main()
