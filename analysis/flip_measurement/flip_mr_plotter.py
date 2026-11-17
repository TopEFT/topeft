# Notes on this script:
#   - This script runs on the output of flip_mr_processor.py
#   - It opens the pkl file, extracts the flip and no flip histos, calculates the flip prob, and saves that to a histo (also saves the 2d hists to png for reference)
#   - The output histo is then placed in topcoffea/data so that corrections.py can read in the values using dense lookup

import copy
import matplotlib.pyplot as plt
import cloudpickle
import gzip
#from coffea import hist
import hist
from topcoffea.modules.histEFT import HistEFT
from boost_histogram import storage
from topeft.modules.yield_tools import YieldTools
#from topcoffea.modules.YieldTools import YieldTools
import topcoffea.modules.utils as utils
yt = YieldTools()

import seaborn as sns
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("filepath",default='histos/flip_rate_2023BPix_SkimSampleDYtt.pkl.gz', help = 'path of file with histograms')
args = parser.parse_args()

hin_dict = utils.get_hist_from_pkl(args.filepath)

# This binning should match what is defined in the flip measurement processor
PT_BINS = [0, 30.0, 45.0, 60.0, 100.0, 200.0]
ABSETA_BINS = [0, 0.4, 0.8, 1.1, 1.4, 1.6, 1.9, 2.2, 2.5]
#ABSETA_BINS = [-2.5, -2.2, -1.9, -1.6, -1.4, -1.1, -0.8, -0.4, 0, 0.4, 0.8, 1.1, 1.4, 1.6, 1.9, 2.2, 2.5]

# These scale factors are determined by comparing prediction to data in the the flip CR
# Though now we apply this in corrections.py, so don't scale here
SCALE_DICT = {
    "UL16APV" : 1.0,
    "UL16"    : 1.0,
    "UL17"    : 1.0,
    "UL18"    : 1.0,
    "2022"    : 1.0,
    "2023"    : 1.0,
    "2022EE"  : 1.0,
    "2023BPix": 1.0,
}

def make_ratio_hist(ratio_arr, pt_bin_lst, abseta_bin_lst):
    # Create a new histogram with Variable axes
    hist_ratio = hist.Hist(
        hist.axis.Variable(pt_bin_lst, name="pt"),
        hist.axis.Variable(abseta_bin_lst, name="abseta"),
        #storage=storage.Weight()
    )

    # Fill the histogram with the ratio array
    # Note: ratio_arr must match shape (len(pt_bin_lst)-1, len(abseta_bin_lst)-1)
    hist_ratio.view()[...] = ratio_arr

    return hist_ratio


# Plot and save a png for a given 2d histo
#def make_2d_fig(histo,xaxis_var,save_name,title=None):
#    if title is not None: title_str = title
#    else: title_str = save_name
#    ax = hist.plot2d(histo,xaxis=xaxis_var)
#    plt.title(title_str)
#    plt.savefig(save_name)

def make_2d_fig(histo, xaxis_var, save_name, title=None):
    if title is None:
        title = save_name

    # Get the two axes of the projected histogram
    axes = histo.axes

    # Assuming 2D projection, xaxis is the first axis, yaxis is the second
    xaxis = axes[0]
    yaxis = axes[1]

    # Bin edges
    x_edges = xaxis.edges
    y_edges = yaxis.edges

    # Get content
    data = histo.view()

    # Ensure that data dimensions match edges for pcolormesh
    data = data.T  # This transposes the data if necessary (adjust depending on how your data is structured)
    
    # If the number of bins in x and y doesn't match the data shape, pad with zeros or use the appropriate method
    if data.shape[1] != len(x_edges) - 1:  # Check if data columns match x_edges
        data = np.pad(data, ((0, 0), (0, 1)), mode='constant', constant_values=0)  # Padding for x axis
    if data.shape[0] != len(y_edges) - 1:  # Check if data rows match y_edges
        data = np.pad(data, ((0, 1), (0, 0)), mode='constant', constant_values=0)  # Padding for y axis

    # Plotting
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(x_edges, y_edges, data, shading='auto', cmap='viridis')
    fig.colorbar(mesh, ax=ax, label="Content")
    ax.set_xlabel(xaxis.label or xaxis.name)
    ax.set_ylabel(yaxis.label or yaxis.name)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_name + ".png")
    plt.close(fig)

def scale_histogram(histogram, scale_factor):
    # Get the current values of the histogram (sum of weights)
    data = histogram.view()

    # Scale the histogram's data
    data *= scale_factor

    # Return the scaled histogram
    return histogram

# Get ratio, flip, and noflip histos from a given input histo
def get_flip_histos(in_hist_orig):

    # Copy and rebin the histo
    in_hist = copy.deepcopy(in_hist_orig)
    print("in_hist", in_hist.values())
#    in_hist = in_hist.rebin("abseta", hist.Bin("abseta","abseta",ABSETA_BINS))
#    in_hist = in_hist.rebin("pt", hist.Bin("pt","pt",PT_BINS))
    # Grab the histo of flipped e and histo of not flipped e

    arr = in_hist.values()
    print(in_hist.axes)
    flip_axis = in_hist.axes[0][0]
    print("flip_axis",flip_axis)

    hist_flip = in_hist.integrate("flipstatus", "truthFlip")
    hist_noflip = in_hist.integrate("flipstatus",'truthNoFlip')

    print("hist_flip value", hist_flip.sum(flow=True))
    print(hist_noflip.sum(flow=True))

    flip_axis = in_hist.axes[in_hist.axes.name.index("flipstatus")]

# Get categories by iterating

    categories = list(flip_axis)

    print("Categories in flipstatus axis:", categories)
    print("'truthFlip' in categories?", "truthFlip" in categories)
    print("Types of categories:", [type(c) for c in categories])
    print("Type of 'truthFlip':", type("truthFlip"))

    # Calculate ratio and make ratio histo
    #flip_sumw_arr = hist_flip._sumw[()]
    #noflip_sumw_arr = hist_noflip._sumw[()]
    flip_sumw_arr = hist_flip.view()
    noflip_sumw_arr = hist_noflip.view()

    print("hist_flip",hist_flip)
    print("hist_noflip",hist_noflip)
    print("flip_sumw_arr",flip_sumw_arr)
    print("noflip_sumw_arr",noflip_sumw_arr)
    flip_sumw_arr = flip_sumw_arr - noflip_sumw_arr
    print("flip_sumw_arr subtract", flip_sumw_arr)
    ratio_sumw_arr = flip_sumw_arr/(flip_sumw_arr+noflip_sumw_arr)
    hist_ratio = make_ratio_hist(ratio_sumw_arr,PT_BINS,ABSETA_BINS)
    print("ratio_sumw_arr", ratio_sumw_arr)
#    uncertainties = np.sqrt(ratio_sumw_arr*( 1 - ratio_sumw_arr)/(flip_sumw_arr+noflip_sumw_arr))
#    print("uncertainties", uncertainties)
#
#
 #   relative_uncertainty = uncertainties / ratio_sumw_arr

    A = flip_sumw_arr
    B = flip_sumw_arr + noflip_sumw_arr

    # Avoid divide-by-zero
    safe_A = np.where(A == 0, 1e-10, A)
    safe_B = np.where(B == 0, 1e-10, B)

    # Flip rate
    flip_rate = A / B
    
    # Standard ratio uncertainty for uncorrelated inputs
    uncertainties = flip_rate * np.sqrt((1 / safe_A) + (1 / safe_B))

    # Relative uncertainty
    #relative_uncertainty = uncertainties / flip_rate
    relative_uncertainty = flip_rate
    #print("flip_rate",flip_rate)
    #print("relative_uncertainty", relative_uncertainty)
    # Optional: mask zero-rate bins to avoid division by zero warning
    relative_uncertainty = np.where(ratio_sumw_arr == 0, np.nan, relative_uncertainty)

    # --- Plot ---
    plt.figure(figsize=(12, 5))
    sns.heatmap(relative_uncertainty, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5)
    #plt.title("Relative Uncertainty of Flip Rate (Δf / f)", fontsize=14)
    #plt.title("Δnoflip / Nnoflip")
    #plt.title("Uncertainty of Flip Rate (Δf) ")
    plt.title("flip yields")

    plt.xlabel("Bin Index")
    plt.ylabel("η / pT bin")

    # --- Save to file ---
    plt.tight_layout()
    plt.savefig("2022_flip_yields.png", dpi=300)
    plt.close()

    return [hist_flip,hist_noflip,hist_ratio]


# Main function
def main():

    # Print info about the in dict

    sample_names_lst = yt.get_cat_lables(hin_dict, axis='Sample')
    flip_names_lst = yt.get_cat_lables(hin_dict, axis='flipstatus')
    print("Samples:",sample_names_lst)
    print("Flipstatus:",flip_names_lst)

    # Get the histos from the input dict
    histo_ptabseta = hin_dict["ptabseta"]
    print(histo_ptabseta.axes["flipstatus"])
    print("Total sum:", histo_ptabseta.sum())
    print("Flip sum:", histo_ptabseta.project("flipstatus").values())
    # Loop over the years
    # NOTE: Will need to hanld UL16APV too
#    for year in ["UL16APV","UL16","UL17","UL18"]:
#
#        # Integrate just the samples for the given year
#        blacklist = []
#        if year != "UL16APV": blacklist = "APV"
#        samples_to_include = yt.filter_lst_of_strs(sample_names_lst,substr_whitelist=year,substr_blacklist=blacklist)
#        histo_ptabseta_year = copy.deepcopy(histo_ptabseta)
#        histo_ptabseta_year = histo_ptabseta_year.integrate("sample",samples_to_include)
    histo_ptabseta = histo_ptabseta.integrate("Sample",sample_names_lst)
    year = "2023BPix"
    print(f"For year {year}, including samples: {sample_names_lst}")
     # Get the flip histos
#    hist_flip, hist_noflip, hist_ratio = get_flip_histos(histo_ptabseta_year)
    hist_flip, hist_noflip, hist_ratio = get_flip_histos(histo_ptabseta)
    # Save figs
    hist_flip_2d = hist_flip.project("pt", "abseta")
    hist_noflip_2d = hist_noflip.project("pt", "abseta")
    hist_ratio_2d = hist_ratio.project("pt", "abseta")
    make_2d_fig(hist_flip_2d,"pt",year+"_truth_flip")
    make_2d_fig(hist_noflip_2d,"pt",year+"_truth_noflip")
    make_2d_fig(hist_ratio_2d,"pt",year+"_truth_ratio","Flip ratio = flip/(flip+noflip)")

    # Scale ratio histo and save a fig for that one too
    hist_ratio = scale_histogram(hist_ratio, SCALE_DICT[year])
    make_2d_fig(hist_ratio,"pt",year+"_truth_ratio_scaled","Flip ratio = flip/(flip+noflip)")

    # Save output histo
    save_pkl_str = "flip_prob_topcoffea_test" + year + ".pkl.gz"
    print("Saving file %s" % save_pkl_str)
    with gzip.open(save_pkl_str, "wb") as fout:
        cloudpickle.dump(hist_ratio, fout)


if __name__ == "__main__":
    main()
