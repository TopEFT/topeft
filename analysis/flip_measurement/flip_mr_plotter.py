import os
import copy
import json
import matplotlib.pyplot as plt
import cloudpickle
import gzip

import uproot3
from coffea import hist

from topcoffea.modules.YieldTools import YieldTools

import make_cr_and_sr_plots as mp

yt = YieldTools()

#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument("filepath",default='histos/flipTopEFT.pkl.gz', help = 'path of file with histograms')
#args = parser.parse_args()

#hin_dict = yt.get_hist_from_pkl("histos/apr01_UL17DY_2d_00_mvaTTHUL.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/apr04_UL17-dy-dy1050-ttbar_test01.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/apr05_UL17-dy-dy1050-ttbar_withTightChReq_test01.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/apr05_UL17-dy-dy1050-ttbar_withTightChReq_test03.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/apr06_UL16-UL17-dy-dy1050-ttbar-UL18-dy-dy1050_test01.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/apr06_UL16-UL17-dy-dy1050-ttbar-UL18-dy-dy1050_anbinning_test02.pkl.gz")
hin_dict = yt.get_hist_from_pkl("histos/apr06_FullR2-dy-dy1050-ttbar_test00.pkl.gz")


#PT_BINS = [0,20,30,40,50,60,70,100,200]
#ABSETA_BINS = [0,0.5,1.0,1.5,2.0,2.5]
PT_BINS = [0, 30.0, 45.0, 60.0, 100.0, 200.0]
ABSETA_BINS = [0, 0.4, 0.8, 1.1, 1.4, 1.6, 1.9, 2.2, 2.5]


# Given an array of values and a pt and eta bin list, make a histo
def make_ratio_hist(ratio_arr,pt_bin_lst,eta_bin_lst):
    hist_ratio = hist.Hist(
        "Ratio",
        hist.Bin("pt", "pt", pt_bin_lst),
        hist.Bin("eta", "eta", eta_bin_lst),
    )
    hist_ratio._sumw = {(): ratio_arr}
    return hist_ratio


# Plot and save a png for a given 2d histo
def make_2d_fig(histo,xaxis_var,save_name,title=None):
    if title is not None: title_str = title
    else: title_str = save_name
    ax = hist.plot2d(histo,xaxis=xaxis_var)
    plt.title(title_str)
    plt.savefig(save_name)


# Get ratio, flip, and noflip histos from a given input histo
def get_flip_histos(in_hist_orig):

    # Copy and rebin the histo
    in_hist = copy.deepcopy(in_hist_orig)
    in_hist = in_hist.rebin("abseta", hist.Bin("abseta","abseta",ABSETA_BINS))
    in_hist = in_hist.rebin("pt", hist.Bin("pt","pt",PT_BINS))

    # Grab the histo of flipped e and histo of not flipped e
    hist_flip = in_hist.integrate("flipstatus","truthFlip")
    hist_noflip = in_hist.integrate("flipstatus","truthNoFlip")

    # Calculate ratio and make ratio histo
    flip_sumw_arr = hist_flip._sumw[()]
    noflip_sumw_arr = hist_noflip._sumw[()]
    ratio_sumw_arr = flip_sumw_arr/(flip_sumw_arr+noflip_sumw_arr)
    hist_ratio = make_ratio_hist(ratio_sumw_arr,PT_BINS,ABSETA_BINS)

    # Print info
    #print("\nflip:",flip_sumw_arr)
    #print("\nnoflip:",noflip_sumw_arr)
    #print("\nflipratio:",ratio_sumw_arr)

    return [hist_flip,hist_noflip,hist_ratio]


# Main function
def main():

    # Print info about the in dict
    sample_names_lst = yt.get_cat_lables(hin_dict,"sample")
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
        samples_to_include = mp.filter_lst_of_strs(sample_names_lst,substr_whitelist=year,substr_blacklist=blacklist)
        print(f"For year {year}, including samples: {samples_to_include}")
        histo_ptabseta_year = copy.deepcopy(histo_ptabseta)
        histo_ptabseta_year = histo_ptabseta_year.integrate("sample",samples_to_include)

        # Get the flip histos
        hist_flip, hist_noflip, hist_ratio = get_flip_histos(histo_ptabseta_year)

        # Save figs
        make_2d_fig(hist_flip,"pt",year+"_truth_flip")
        make_2d_fig(hist_noflip,"pt",year+"_truth_noflip")
        make_2d_fig(hist_ratio,"pt",year+"_truth_ratio","Flip ratio = flip/(flip+noflip)")

        # Save output histo
        save_pkl_str = "flip_probs_apr06BinningAN19127_" + year + ".pkl.gz"
        with gzip.open(save_pkl_str, "wb") as fout:
            cloudpickle.dump(hist_ratio, fout)


if __name__ == "__main__":
    main()
