import os
import copy
import datetime
import argparse
import matplotlib.pyplot as plt
import uproot3
from coffea import hist

import numpy as np
np.seterr(divide='ignore', invalid='ignore', over='ignore')

import topcoffea.modules.utils as utils
from topcoffea.plotter.make_html import make_html
from topcoffea.modules.YieldTools import YieldTools
yt = YieldTools()

# This script is a very rough plotter for the output of the data_comp processor

parser = argparse.ArgumentParser(description='You can customize your run')
parser.add_argument("filepath",default='histos/data_comp.pkl.gz', help = 'path of file with histograms')
parser.add_argument("--output-path" ,'-o', default='.', help = 'Path to the output directory')
parser.add_argument("-n", "--output-name", default="plots", help = "A name for the output directory")
parser.add_argument("-t", "--include-timestamp-tag", action="store_true", help = "Append the timestamp to the out dir name")
args = parser.parse_args()
outpath = args.output_path

# Takes a hist with one sparse axis and one dense axis, overlays everything on the sparse axis
def make_single_fig(histo,unit_norm_bool):
    #print("\nPlotting values:",histo.values())
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    hist.plot1d(
        histo,
        stack=False,
        density=unit_norm_bool,
        clear=False,
    )
    ax.autoscale(axis='y')
    return fig


# Main function
def main():

    # Channels: ['topcoffea', 'tthfrmwk_legacy_unique', 'tthfrmwk_ul', 'tthfrmwk_ul_common', 'tthfrmwk_ul_unique']

    # Make a tmp output directory in curren dir a different dir is not specified
    timestamp_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    save_dir_path = args.output_path
    outdir_name = args.output_name
    if args.include_timestamp_tag:
        outdir_name = outdir_name + "_" + timestamp_tag
    save_dir_path = os.path.join(save_dir_path,outdir_name)
    os.mkdir(save_dir_path)

    hin_dict = utils.get_hist_from_pkl(args.filepath)
    sample_names_lst = yt.get_cat_lables(hin_dict,"sample")
    chan_names_lst = yt.get_cat_lables(hin_dict,"channel")
    var_lst = list(hin_dict.keys())

    print("Samples:",sample_names_lst)
    print("Channels:",chan_names_lst)
    print("Variables:",var_lst)

    #histo = hin_dict["l0pt"]
    for var_name in var_lst:
        histo = hin_dict[var_name]
        histo = histo.sum("sample")

        print("\n",var_name,histo.values())

        #unit_norm_bool = False
        unit_norm_bool = True
        # Make a sub dir for this category
        save_dir_path_tmp = os.path.join(save_dir_path)
        if not os.path.exists(save_dir_path_tmp):
            os.mkdir(save_dir_path_tmp)

        fig = make_single_fig(histo,unit_norm_bool)
        title = var_name
        if unit_norm_bool: title = title + "_unitnorm"
        fig.savefig(os.path.join(save_dir_path_tmp,title))

        # Make an index.html file if saving to web area
        if "www" in save_dir_path_tmp: make_html(save_dir_path_tmp)

main()
