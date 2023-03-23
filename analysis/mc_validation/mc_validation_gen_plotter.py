# This script makes plots to compare private and central GEN level distributions
#   - Should be run on the output of the mc_validation_gen_processor.py processor
#   - Was used during the June 2022 MC validation studies (for TOP-22-006 pre approval checks)

import numpy as np
import os
import copy
import datetime
import argparse
import matplotlib.pyplot as plt
from cycler import cycler
import gzip
import cloudpickle

import uproot
from coffea import hist

from topcoffea.modules.HistEFT import HistEFT
from topcoffea.modules.paths import topcoffea_path
from topcoffea.modules.YieldTools import YieldTools
from topcoffea.scripts.make_html import make_html

import topcoffea.modules.GetValuesFromJsons as getj

# Probably I should move the utility functions out of this script and put them in modules
# Anyway, not good practice to just import it here as if it were a library, but I'm doing it anyway (for now)
import make_cr_and_sr_plots as mcp

yt = YieldTools()

def save_pkl_for_arr(sf_arr,tag):
    sf_histo = hist.Hist(
        "Ratio",
        hist.Bin("ht","ht",25,0,1000)
    )
    sf_histo._sumw = {(): sf_arr}
    save_pkl_str = "ht_rwgt_sf_" + tag + ".pkl.gz"
    with gzip.open(save_pkl_str, "wb") as fout:
        cloudpickle.dump(sf_histo, fout)
    

# Main wrapper script for making the private vs central comparison plots
def make_mc_validation_plots(dict_of_hists,year,skip_syst_errs,save_dir_path):
    sample_lst = yt.get_cat_lables(dict_of_hists,"sample")
    vars_lst = dict_of_hists.keys()
    print("\nSamples:",sample_lst)
    print("\nVariables:",vars_lst)

    comp_proc_dict = {
        "ttH" : {
            "central_nonUL" : f"ttH_central2017",
            "central" : f"ttH_central{year}",
            "private": f"ttHJet_private{year}",
        },
        "ttlnu" : {
            "central_nonUL" : f"ttW_central2017",
            "central" : f"ttW_central{year}",
            "private": f"ttlnuJet_private{year}",
        },
        "ttll" : {
            "central_nonUL" : f"ttZ_central2017",
            "central" : f"ttZ_central{year}",
            "private": f"ttllJet_private{year}",
        },
        "tllq" : {
            "central_nonUL" : f"tZq_central2017",
            "central" : f"tZq_central{year}",
            "private": f"tllq_private{year}",
        },
    }


    # Loop over variables
    for var_name in vars_lst:
        print("\nVar name:",var_name)

        # Sum over channels, and just grab the nominal from the syst axis
        histo_base = dict_of_hists[var_name]

        # Now loop over processes and make plots
        for proc in comp_proc_dict.keys():
            print(f"\nProcess: {proc}")

            # Group bins
            proc_histo = mcp.group_bins(histo_base,comp_proc_dict[proc],drop_unspecified=True)
            print(comp_proc_dict[proc])

            # Dump SF dictionary (for the HT reweighting tests)
            #central_arr = proc_histo["central"].values(overflow="allnan")[("central",)]
            #private_arr = proc_histo["private"].values(overflow="allnan")[("private",)]
            #ratio_arr = np.where( central_arr>0, central_arr/private_arr, 0)
            #save_pkl_for_arr(ratio_arr,comp_proc_dict[proc]["private"])
            #continue

            # Make the plots
            fig = mcp.make_single_fig_with_ratio(proc_histo,"sample","private")
            #fig = mcp.make_single_fig(proc_histo,unit_norm_bool=True)
            fig.savefig(os.path.join(save_dir_path,proc+"_"+var_name))
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
