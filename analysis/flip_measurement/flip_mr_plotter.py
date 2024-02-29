# Notes on this script:
#   - This script runs on the output of flip_mr_processor.py
#   - It opens the pkl file, extracts the flip and no flip histos, calculates the flip prob, and saves that to a histo (also saves the 2d hists to png for reference)
#   - The output histo is then placed in topcoffea/data so that corrections.py can read in the values using dense lookup

import copy
import matplotlib.pyplot as plt
import cloudpickle
import gzip

from topeft.modules.yield_tools import YieldTools
import topcoffea.modules.utils as utils
yt = YieldTools()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("filepath",default='histos/flipMR_TopEFT.pkl.gz', help = 'path of file with histograms')
args = parser.parse_args()

hin_dict = utils.get_hist_from_pkl(args.filepath)

# This binning should match what is defined in the flip measurement processor
PT_BINS = [0, 30.0, 45.0, 60.0, 100.0, 200.0]
ABSETA_BINS = [0, 0.4, 0.8, 1.1, 1.4, 1.6, 1.9, 2.2, 2.5]

# These scale factors are determined by comparing prediction to data in the the flip CR
# Though now we apply this in corrections.py, so don't scale here
SCALE_DICT = {
    "UL16APV" : 1.0,
    "UL16"    : 1.0,
    "UL17"    : 1.0,
    "UL18"    : 1.0,
}

# Given an array of values and a pt and eta bin list, make a histo
'''
def make_ratio_hist(ratio_arr,pt_bin_lst,eta_bin_lst):
    hist_ratio = hist.Hist(
        hist.axis.Variable(name="pt",  label="pt",  edges=pt_bin_lst),
        hist.axis.Variable(name="eta", label="eta", edges=eta_bin_lst),
    )
    hist_ratio.view(flow=True) = {(): ratio_arr}
    return hist_ratio
'''

# Plot and save a png for a given 2d histo
def make_2d_fig(histo,xaxis_var,save_name,title=None):
    if title is not None: title_str = title
    else: title_str = save_name
    histo.plot2d(flow='none')
    plt.title(title_str)
    plt.savefig(save_name)
    plt.close()


# Get ratio, flip, and noflip histos from a given input histo
def get_flip_histos(in_hist_orig):

    # Copy and rebin the histo
    in_hist = copy.deepcopy(in_hist_orig)
    #in_hist = in_hist.rebin("abseta", hist.Bin("abseta","abseta",ABSETA_BINS))
    #in_hist = in_hist.rebin("pt", hist.Bin("pt","pt",PT_BINS))

    # Grab the histo of flipped e and histo of not flipped e
    hist_flip = in_hist.integrate("flipstatus","truthFlip")
    hist_noflip = in_hist.integrate("flipstatus","truthNoFlip")

    # Calculate ratio and make ratio histo
    flip_sumw_arr = hist_flip.values(flow=True)[()]
    noflip_sumw_arr = hist_noflip.values(flow=True)[()]
    #ratio_sumw_arr = flip_sumw_arr/(flip_sumw_arr+noflip_sumw_arr)
    #hist_ratio = make_ratio_hist(ratio_sumw_arr,PT_BINS,ABSETA_BINS)

    # Print info
    #print("\nflip:",flip_sumw_arr)
    #print("\nnoflip:",noflip_sumw_arr)
    #print("\nflipratio:",ratio_sumw_arr)

    return [hist_flip,hist_noflip]#,hist_ratio]


# Main function
def main():

    # Print info about the in dict
    sample_names_lst = yt.get_cat_lables(hin_dict,"process")
    flip_names_lst = yt.get_cat_lables(hin_dict,"flipstatus")
    print("Samples:",sample_names_lst)
    print("Flipstatus:",flip_names_lst)

    # Get the histos from the input dict
    histo_ptabseta = hin_dict["ptabseta"]

    # Loop over the years
    # NOTE: Will need to hanld UL16APV too
    for year in ["UL16APV","UL16","UL17","UL18"]:

        # Integrate just the samples for the given year
        blacklist = []
        if year != "UL16APV": blacklist = "APV"
        samples_to_include = utils.filter_lst_of_strs(sample_names_lst,substr_whitelist=year,substr_blacklist=blacklist)
        if not samples_to_include:
            continue
        print(f"For year {year}, including samples: {samples_to_include}")
        histo_ptabseta_year = copy.deepcopy(histo_ptabseta)
        histo_ptabseta_year = histo_ptabseta_year.integrate("process",samples_to_include)

        # Get the flip histos
        hist_flip, hist_noflip = get_flip_histos(histo_ptabseta_year)
        #hist_flip, hist_noflip, hist_ratio = get_flip_histos(histo_ptabseta_year)

        hist_ratio = hist_flip / (hist_noflip + hist_flip)

        # Save figs
        make_2d_fig(hist_flip,"pt",year+"_truth_flip")
        make_2d_fig(hist_noflip,"pt",year+"_truth_noflip")
        make_2d_fig(hist_ratio,"pt",year+"_truth_ratio","Flip ratio = flip/(flip+noflip)")

        # Scale ratio histo and save a fig for that one too
        hist_ratio *= SCALE_DICT[year]
        make_2d_fig(hist_ratio,"pt",year+"_truth_ratio_scaled","Flip ratio = flip/(flip+noflip)")

        # Save output histo
        save_pkl_str = "flip_probs_topcoffea_" + year + ".pkl.gz"
        with gzip.open(save_pkl_str, "wb") as fout:
            cloudpickle.dump(hist_ratio, fout)


if __name__ == "__main__":
    main()
