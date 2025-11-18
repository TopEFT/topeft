# This script makes plots to compare private and central GEN level distributions
#   - Should be run on the output of the mc_validation_gen_processor.py processor
#   - Was used during the June 2022 MC validation studies (for TOP-22-006 pre approval checks)

import os
import sys
import datetime
import argparse
import gzip
import cloudpickle

import hist
from hist import axis, storage

from pathlib import Path

from topcoffea.modules.YieldTools import YieldTools
from topcoffea.scripts.make_html import make_html

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from analysis.mc_validation.plot_utils import (  # noqa: E402
    build_dataset_histograms,
    component_labels,
    component_values,
    require_tuple_histogram_items,
)


# Probably I should move the utility functions out of this script and put them in modules
# Anyway, not good practice to just import it here as if it were a library, but I'm doing it anyway (for now)
import make_cr_and_sr_plots as mcp

yt = YieldTools()

def save_pkl_for_arr(sf_arr,tag):
    sf_histo = hist.Hist(
        axis.Regular(25, 0, 1000, name="ht", label="ht", flow=True),
        storage=storage.Double(),
        label="Ratio",
    )
    sf_histo.view()[...] = sf_arr
    save_pkl_str = "ht_rwgt_sf_" + tag + ".pkl.gz"
    with gzip.open(save_pkl_str, "wb") as fout:
        cloudpickle.dump(sf_histo, fout)


# Main wrapper script for making the private vs central comparison plots
def make_mc_validation_plots(dict_of_hists,year,skip_syst_errs,save_dir_path):
    tuple_entries = require_tuple_histogram_items(dict_of_hists)
    rebuilt_hists = build_dataset_histograms(dict_of_hists)

    vars_lst = sorted(rebuilt_hists.keys())
    sample_lst = component_values(tuple_entries, "sample")
    sample_labels = component_labels(tuple_entries, "sample", include_application=True)
    dataset_axis_name = "dataset"

    print("\nSamples:",sample_labels)
    print("\nVariables:",vars_lst)

    comp_proc_dict = {
        "ttH" : {
            "central_nonUL" : "ttH_central2017",
            "central" : f"ttH_central{year}",
            "private": f"ttHJet_private{year}",
        },
        "ttlnu" : {
            "central_nonUL" : "ttW_central2017",
            "central" : f"ttW_central{year}",
            "private": f"ttlnuJet_private{year}",
        },
        "ttll" : {
            "central_nonUL" : "ttZ_central2017",
            "central" : f"ttZ_central{year}",
            "private": f"ttllJet_private{year}",
        },
        "tllq" : {
            "central_nonUL" : "tZq_central2017",
            "central" : f"tZq_central{year}",
            "private": f"tllq_private{year}",
        },
    }


    # Loop over variables
    for var_name in vars_lst:
        print("\nVar name:",var_name)

        # Sum over channels, and just grab the nominal from the syst axis
        histo_base = rebuilt_hists.get(var_name)
        if histo_base is None:
            histo_base = dict_of_hists.get(var_name)
        if histo_base is None:
            raise KeyError(f"Histogram '{var_name}' not found in rebuilt or original mapping")

        # Collapse categorical axes that are not part of the plotting layout so downstream
        # grouping returns the expected 1D histogram.  The rebuilt tuple histograms may
        # include both channel and application axes, and keeping either around results in
        # `.values()[()]` lookups failing when the grouped histogram still has extra
        # dimensions.  Sum over these axes if they are present.
        axes_names = {ax.name for ax in getattr(histo_base, "axes", ())}
        histo_collapsed = histo_base
        if "channel" in axes_names:
            histo_collapsed = histo_collapsed.sum("channel")
        if "application" in axes_names:
            histo_collapsed = histo_collapsed.sum("application")

        # Now loop over processes and make plots
        for proc in comp_proc_dict.keys():
            print(f"\nProcess: {proc}")

            # Group bins
            proc_histo = mcp.group_bins(
                histo_collapsed,
                comp_proc_dict[proc],
                axis_name=dataset_axis_name,
                drop_unspecified=True,
            )
            print(comp_proc_dict[proc])

            # Dump SF dictionary (for the HT reweighting tests)
            #central_arr = proc_histo["central"].values(overflow="allnan")[("central",)]
            #private_arr = proc_histo["private"].values(overflow="allnan")[("private",)]
            #ratio_arr = np.where( central_arr>0, central_arr/private_arr, 0)
            #save_pkl_for_arr(ratio_arr,comp_proc_dict[proc]["private"])
            #continue

            # Make the plots
            fig = mcp.make_single_fig_with_ratio(proc_histo,dataset_axis_name,"private")
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
