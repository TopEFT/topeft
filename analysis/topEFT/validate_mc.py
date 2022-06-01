import numpy as np
import os
import copy
import datetime
import argparse
import matplotlib.pyplot as plt
from cycler import cycler

from coffea import hist
from topcoffea.modules.HistEFT import HistEFT

from topcoffea.modules.YieldTools import YieldTools
import topcoffea.modules.GetValuesFromJsons as getj
from topcoffea.plotter.make_html import make_html

# This script should maybe just be a part of make_cr_and_sr_plots, though that script is getting really long
# Probably I should move the utility functions out of that script and put them in modules
# Anyway, not good practice to just import it here as if it were a library, but I'm doing it anyway (for now)
import make_cr_and_sr_plots as mcp

yt = YieldTools()

def make_mc_validation_plots(dict_of_hists,year,skip_syst_errs,unit_norm_bool,save_dir_path):
    sample_lst = yt.get_cat_lables(dict_of_hists,"sample")
    vars_lst = dict_of_hists.keys()
    print("\nVariables:",sample_lst)
    print("\nVariables:",vars_lst)

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
        "ttW" : {
            "central" : proc_dict["ttW_central"],
            "private" : proc_dict["ttlnuJet_private"],
        },
        "ttZ": {
            "central" : proc_dict["ttZ_central"],
            "private" : proc_dict["ttllJet_private"],
        },
        "tttt": {
            "central" : proc_dict["tttt_central"],
            "private" : proc_dict["tttt_private"],
        },
        "tZq": {
            "central" : proc_dict["tZq_central"],
            "private" : proc_dict["tllq_private"],
        }
    }

    print(comp_proc_dict)

    # Loop over variables
    for var_name in vars_lst:
        #if var_name != "njets": continue

        # Sum over channels, and just grab the nominal from the syst axis
        histo = dict_of_hists[var_name].sum("channel").integrate("systematic","nominal")

        # Normalize by lumi (important to do this before grouping by year)
        sample_lumi_dict = {}
        for sample_name in sample_lst:
            sample_lumi_dict[sample_name] = mcp.get_lumi_for_sample(sample_name)
        histo.scale(sample_lumi_dict,axis="sample")

        # Now loop over processes and make plots
        for proc in comp_proc_dict.keys():
            print(f"\nProcess: {proc}")

            # Group the histos
            proc_histo = mcp.group_bins(histo,comp_proc_dict[proc],drop_unspecified=True)

            # Make the plots
            fig = mcp.make_single_fig(proc_histo,False)
            fig.savefig(os.path.join(save_dir_path,proc+"_"+var_name))
            if "www" in save_dir_path: make_html(save_dir_path)




def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    parser.add_argument("-o", "--output-path", default=".", help = "The path the output files should be saved to")
    parser.add_argument("-n", "--output-name", default="plots", help = "A name for the output directory")
    parser.add_argument("-t", "--include-timestamp-tag", action="store_true", help = "Append the timestamp to the out dir name")
    parser.add_argument("-y", "--year", default=None, help = "The year of the sample")
    parser.add_argument("-u", "--unit-norm", action="store_true", help = "Unit normalize the plots")
    parser.add_argument("-s", "--skip-syst", default=False, action="store_true", help = "Skip syst errs in plots, only relevant for CR plots right now")
    args = parser.parse_args()

    # Whether or not to unit norm the plots
    unit_norm_bool = args.unit_norm

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
    make_mc_validation_plots(hin_dict,args.year,args.skip_syst,unit_norm_bool,save_dir_path)

if __name__ == "__main__":
    main()
