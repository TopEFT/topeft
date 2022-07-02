# This script makes plots to compare private and central RECO level distributions
#   - Should be run on the output of the topeft processor
#   - Was used during the June 2022 MC validation studies (for TOP-22-006 pre approval checks)

import numpy as np
import os
import copy
import datetime
import argparse
import matplotlib.pyplot as plt
from cycler import cycler

import uproot
from coffea import hist

from topcoffea.modules.HistEFT import HistEFT
from topcoffea.modules.paths import topcoffea_path
from topcoffea.modules.YieldTools import YieldTools
from topcoffea.plotter.make_html import make_html

import topcoffea.modules.GetValuesFromJsons as getj

# This script should maybe just be a part of make_cr_and_sr_plots, though that script is getting really long
# Probably I should move the utility functions out of that script and put them in modules
# Anyway, not good practice to just import it here as if it were a library, but I'm doing it anyway (for now)
import make_cr_and_sr_plots as mcp

yt = YieldTools()

# Opens the missing parton SF root file and returns a dict of the values
def get_missing_parton_sf_dict(per_jet_bin=True):

    fparton = uproot.open(topcoffea_path("data/missing_parton/missing_parton.root"))

    # Which njets values the lepton categories begin with
    njets_start_dict = {
        "2l" : 4,
        "3l" : 2,
        "4l" : 2,
    }

    # Mapping between categories from the datacard to cateogries in the HistEFT
    cat_name_map = {
        "2lss_4t_m_2b" : "2lss_4t_m",
        "2lss_4t_p_2b" : "2lss_4t_p",
        "2lss_m_2b"    : "2lss_m",
        "2lss_p_2b"    : "2lss_p",
        "3l_sfz_1b"    : "3l_onZ_1b",
        "3l_sfz_2b"    : "3l_onZ_2b",
        "3l1b_p"       : "3l_p_offZ_1b",
        "3l1b_m"       : "3l_m_offZ_1b",
        "3l2b_p"       : "3l_p_offZ_2b",
        "3l2b_m"       : "3l_m_offZ_2b",
        "4l_2b"        : "4l",
    }

    # Loop over the keys in the root file and fill the dict of SFs for each of the 43 categoreis
    if per_jet_bin:
        sf_dict = {}
        for k in fparton.keys():
            cat_name = k.split(";")[0] # Keys seem to be e.g. 2lss_m_2b;1
            lep_cat = k[0:2] # Assumes the keys start with "nl" where n is the number of leptons
            sf_arr = np.array(fparton[k]['tllq'].array())
            njet = njets_start_dict[lep_cat]
            for sf in sf_arr:
                histeft_cat_name = cat_name_map[cat_name]
                histeft_cat_name_with_j = f"{histeft_cat_name}_{njet}j"
                sf_dict[histeft_cat_name_with_j] = sf
                njet = njet +1

    # Loop over the keys in the root file and fill the dict of arrays of SFs for each of the 11 categories
    else:
        sf_dict = {}
        for k in fparton.keys():
            cat_name = k.split(";")[0] # Keys seem to be e.g. 2lss_m_2b;1
            lep_cat = k[0:2] # Assumes the keys start with "nl" where n is the number of leptons
            sf_arr = np.array(fparton[k]['tllq'].array())
            histeft_cat_name = cat_name_map[cat_name]

            # Pad the array for the 10 jet njets histo
            sf_lst = []
            njets_cat_highest_multiplicity = njets_start_dict[lep_cat] + len(sf_arr) - 1
            for njet in range(10):
                if njet < njets_start_dict[lep_cat]:
                    sf_lst.append(0)
                elif njet >= njets_start_dict[lep_cat] and njet < njets_cat_highest_multiplicity:
                    sf_lst.append(sf_arr[njet-njets_start_dict[lep_cat]])
                elif njet >= njets_cat_highest_multiplicity:
                    sf_lst.append(sf_arr[-1])
                else:
                    raise Exception("This should not be possible, something is wrong with the logic.")

            sf_dict[histeft_cat_name] = np.array(sf_lst)

    return sf_dict


# Main wrapper script for making the private vs central comparison plots
def make_mc_validation_plots(dict_of_hists,year,skip_syst_errs,save_dir_path):
    sample_lst = yt.get_cat_lables(dict_of_hists,"sample")
    cat_lst = yt.get_cat_lables(dict_of_hists,"channel")
    vars_lst = dict_of_hists.keys()
    print("\nSamples:",sample_lst)
    print("\nVariables:",vars_lst)
    print("\nChannels:",cat_lst)

    # Get the dictionary of histograms that we want to group together in the plots
    # This is way more hard coded than it probably should be
    proc_dict = {
        "ttHJet_private" : [],
        "ttlnuJet_private" : [],
        "ttllJet_private" : [],
        "tllq_private" : [],
        "tttt_private" : [],
        "ttH_central" : [],
        "ttW_central" : [],
        "ttZ_central" : [],
        "tZq_central" : [],
        "tttt_central" : [],
    }
    for sample_name in sample_lst:
        if not sample_name.endswith(year): continue
        # Get name of sample without the year in it
        if "APV" in sample_name: sample_name_noyear = sample_name[:-7]
        else: sample_name_noyear = sample_name[:-4]
        # Put the sample into the process dictionary
        if sample_name_noyear in proc_dict: proc_dict[sample_name_noyear].append(sample_name)
        else: print(f"Skipping sample {sample_name}")
    comp_proc_dict = {
        "ttH" : {
            "central" : proc_dict["ttH_central"],
            "private": proc_dict["ttHJet_private"],
        },
        "ttlnu" : {
            "central" : proc_dict["ttW_central"],
            "private" : proc_dict["ttlnuJet_private"],
        },
        "ttll": {
            "central" : proc_dict["ttZ_central"],
            "private" : proc_dict["ttllJet_private"],
        },
        "tttt": {
            "central" : proc_dict["tttt_central"],
            "private" : proc_dict["tttt_private"],
        },
        "tllq": {
            "central" : proc_dict["tZq_central"],
            "private" : proc_dict["tllq_private"],
        }
    }
    print(comp_proc_dict)



    # Loop over variables
    for var_name in vars_lst:
        print("\nVar name:",var_name)
        #if var_name != "njets": continue
        #if var_name != "lj0pt": continue


        # Sum over channels, and just grab the nominal from the syst axis
        histo_base = dict_of_hists[var_name]

        # Normalize by lumi (important to do this before grouping by year)
        sample_lumi_dict = {}
        for sample_name in sample_lst:
            sample_lumi_dict[sample_name] = mcp.get_lumi_for_sample(sample_name)
        histo_base.scale(sample_lumi_dict,axis="sample")

        # Now loop over processes and make plots
        for proc in comp_proc_dict.keys():
            if "tllq" not in proc: continue
            print(f"\nProcess: {proc}")

            histo = histo_base.sum("channel")

            #for cat in cat_lst:
            #histo = histo_base.integrate("channel",cat)

            # Get the nominal private
            private_proc_histo = mcp.group_bins(histo,{proc+"_private":comp_proc_dict[proc]["private"]},drop_unspecified=True)
            nom_arr_all = private_proc_histo.sum("sample").integrate("systematic","nominal").values()[()]

            # Get the systematic shape and rate uncertainties
            group_map = {"Conv":[], "Diboson":[], "Triboson":[], "Flips":[], "Signal":[proc+"_private"]} # A group map is expected by the code that gets the rate systs
            rate_systs_summed_arr_m , rate_systs_summed_arr_p = mcp.get_rate_syst_arrs(private_proc_histo,group_map)
            shape_systs_summed_arr_m , shape_systs_summed_arr_p = mcp.get_shape_syst_arrs(private_proc_histo)

            # Get the missing parton uncertainty, add it to the rate uncertainties
            histo_private_all_cats = histo_base.integrate("sample",comp_proc_dict[proc]["private"]).integrate("systematic","nominal")
            if proc == "tllq" and var_name != "njets":
                histo_private_all_cats.scale(get_missing_parton_sf_dict(),axis="channel")
                missing_parton_err_summed = histo_private_all_cats.sum("channel").values()[()]
                #missing_parton_err_summed = histo_private_all_cats.integrate("channel",cat).values()[()]
                rate_systs_summed_arr_p = rate_systs_summed_arr_p + missing_parton_err_summed*missing_parton_err_summed
                rate_systs_summed_arr_m = rate_systs_summed_arr_m + missing_parton_err_summed*missing_parton_err_summed
            if proc == "tllq" and var_name == "njets":
                missing_parton_dict_of_arrs = get_missing_parton_sf_dict(per_jet_bin=False)
                missing_parton_err_arr_dict = {}
                missing_parton_err_arr_lst = []
                for cat_name,sf_arr in missing_parton_dict_of_arrs.items():
                    missing_parton_err_arr_dict[cat_name] = missing_parton_dict_of_arrs[cat_name]*histo_private_all_cats.values()[(cat_name,)] # Store in a dict in case we want to look channel by channel
                    missing_parton_err_arr_lst.append(missing_parton_dict_of_arrs[cat_name]*histo_private_all_cats.values()[(cat_name,)]) # Also store in a list since it's easier to sum all the arrays... this is not good code.
                missing_parton_err_arr_summed = sum(missing_parton_err_arr_lst)
                rate_systs_summed_arr_p = rate_systs_summed_arr_p + missing_parton_err_arr_summed*missing_parton_err_arr_summed
                rate_systs_summed_arr_m = rate_systs_summed_arr_m + missing_parton_err_arr_summed*missing_parton_err_arr_summed

            # Find the plus and minus arrays
            p_err_arr = nom_arr_all + np.sqrt(shape_systs_summed_arr_p + rate_systs_summed_arr_p) # This goes in the main plot
            m_err_arr = nom_arr_all - np.sqrt(shape_systs_summed_arr_m + rate_systs_summed_arr_m) # This goes in the main plot
            #print("shape m",shape_systs_summed_arr_m)
            #print("shape p",shape_systs_summed_arr_p)
            #print("rate m",rate_systs_summed_arr_m)
            #print("rate p",rate_systs_summed_arr_p)
            #print("\nnom_arr_all:",nom_arr_all)
            #print("p_err_arr",p_err_arr)
            #print("m_err_arr",m_err_arr)
            p_err_arr_ratio = np.where(nom_arr_all>0,p_err_arr/nom_arr_all,1) # This goes in the ratio plot
            m_err_arr_ratio = np.where(nom_arr_all>0,m_err_arr/nom_arr_all,1) # This goes in the ratio plot

            # Make the plots
            proc_histo = mcp.group_bins(histo,comp_proc_dict[proc],drop_unspecified=True).integrate("systematic","nominal")
            fig = mcp.make_single_fig_with_ratio(
                proc_histo,"sample","private",
                err_p = p_err_arr,
                err_m = m_err_arr,
                err_ratio_p = p_err_arr_ratio,
                err_ratio_m = m_err_arr_ratio
            )
            fig.savefig(os.path.join(save_dir_path,proc+"_"+var_name))
            #fig.savefig(os.path.join(save_dir_path,proc+"_"+var_name+"_"+cat))
            if "www" in save_dir_path: make_html(save_dir_path)




def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    parser.add_argument("-o", "--output-path", default=".", help = "The path the output files should be saved to")
    parser.add_argument("-n", "--output-name", default="plots", help = "A name for the output directory")
    parser.add_argument("-t", "--include-timestamp-tag", action="store_true", help = "Append the timestamp to the out dir name")
    parser.add_argument("-y", "--year", default="UL18", help = "The year of the sample")
    parser.add_argument("-s", "--skip-syst", default=False, action="store_true", help = "Skip syst errs in plots, only relevant for CR plots right now")
    args = parser.parse_args()

    # Make a tmp output directory in curren dir a different dir is not specified
    timestamp_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    save_dir_path = args.output_path
    outdir_name = args.output_name
    if args.include_timestamp_tag:
        outdir_name = outdir_name + "_" + timestamp_tag
    save_dir_path = os.path.join(save_dir_path,outdir_name)
    os.mkdir(save_dir_path)

    # Get the histograms
    hin_dict = yt.get_hist_from_pkl(args.pkl_file_path,allow_empty=False)

    # Make the plots
    make_mc_validation_plots(hin_dict,args.year,args.skip_syst,save_dir_path)

if __name__ == "__main__":
    main()
