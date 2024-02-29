# Notes on this script:
#   - This script runs on the output of flip_ar_processor.py
#   - It opens the pkl file and plots the SS data and the prediction
#   - The prediction was calculated in the processor by applying flip probabilities to the OS data

import os
import copy
import matplotlib.pyplot as plt
from coffea import hist

from topcoffea.modules.YieldTools import YieldTools
yt = YieldTools()

import argparse
parser = argparse.ArgumentParser(description='You can customize your run')
parser.add_argument("filepath",default='histos/flipTopEFT.pkl.gz', help = 'path of file with histograms')
parser.add_argument("--outpath" ,'-o'   , default='.', help = 'Path to the output directory')
args = parser.parse_args()
outpath = args.outpath

# Plots one or two histos (optionally along with an "up" and "down" variation) and returns the fig
def make_fig(histo1,histo2=None,hup=None,hdo=None):
    #print("\nPlotting values:",histo1.values())
    #print("\nPlotting values:",histo2.values())
    fig, ax = plt.subplots(1, 1, figsize=(7,7))

    # Plot the "up" and "down" histos
    if hup is not None and hdo is not None:
        hist.plot1d(
            hup,
            stack=False,
            line_opts={'color': 'lightgrey'},
            clear=False,
        )
        hist.plot1d(
            hdo,
            stack=False,
            line_opts={'color': 'lightgrey'},
            clear=False,
        )

    # Plot the main histos
    hist.plot1d(
        histo1,
        stack=False,
        clear=False,
    )
    if histo2 is not None:
        hist.plot1d(
            histo2,
            stack=False,
            clear=False,
        )

    ax.autoscale(axis='y')
    return fig


# Print summed values from a histo
def print_summed_hist_vals(in_hist,ret="ssz",quiet=False):
    val_dict = {}
    for k,v in in_hist.values().items():
        val_dict[k[0]] = sum(v)
    for k,v in val_dict.items():
        if not quiet: print(f"\t{k}: {v}")
    fliprate = val_dict["sszTruthFlip"]/(val_dict["sszTruthFlip"] + val_dict["oszTruthNoFlip"])
    #print("\tFlip rate:", val_dict["sszTruthFlip"]/(val_dict["osz"] + val_dict["ssz"]))
    #print("\tFlip rate:", val_dict["ssTruthFlip2"]/(val_dict["os"] + val_dict["ss"]))
    if not quiet: print("\tFlip rate:", fliprate)
    return val_dict[ret]


# Main wrapper function
def make_plot():

    hin_dict = yt.get_hist_from_pkl(args.filepath)
    sample_names_lst = yt.get_cat_lables(hin_dict,"process")
    chan_names_lst = yt.get_cat_lables(hin_dict,"channel")

    print("Samples:",sample_names_lst)
    print("Channels:",chan_names_lst)

    for histo_name,histo_orig in hin_dict.items():
        print(f"\nName: {histo_name}")

        # Loop over samples
        for sample_name in sample_names_lst:
            print("sample_name",sample_name)

            # Copy (and rebin the ss)
            histo = copy.deepcopy(histo_orig)
            if histo_name == "invmass": histo = histo.rebin("invmass",2)

            # Integrate and make plot (overlay the categories)
            savename = "_".join([sample_name,histo_name])
            histo = histo.integrate("process",sample_name)
            h_up = copy.deepcopy(histo)
            h_do = copy.deepcopy(histo)
            h_up.scale(1.3)
            h_do.scale(0.7)
            fig = make_fig(histo["ssz"],histo2=histo["osz"],hup=h_up["osz"],hdo=h_do["osz"])
            fig.savefig(os.path.join(outpath,savename))


# Main function
def main():
    make_plot()

main()
