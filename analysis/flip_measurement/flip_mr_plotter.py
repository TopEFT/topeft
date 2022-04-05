import os
import copy
import json
import matplotlib.pyplot as plt
import cloudpickle
import gzip

import uproot3
from coffea import hist

from topcoffea.modules.YieldTools import YieldTools

yt = YieldTools()


#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument("filepath",default='histos/flipTopEFT.pkl.gz', help = 'path of file with histograms')
#args = parser.parse_args()

#hin_dict = yt.get_hist_from_pkl("histos/apr01_UL17DY_2d_00_mvaTTHUL.pkl.gz")
hin_dict = yt.get_hist_from_pkl("histos/apr04_UL17-dy-dy1050-ttbar_test01.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/apr04_fullR2-dy-dy1050-ttbar_test01.pkl.gz")


PT_BINS = [0,20,30,40,50,60,70,100,200]
ETA_BINS = [0,0.5,1.0,1.5,2.0,2.5]


# Given an array of values and a pt and eta bin list, make a histo
def make_ratio_hist(ratio_arr,pt_bin_lst,eta_bin_lst):
    hist_ratio = hist.Hist(
        "Ratio",
        hist.Bin("pt", "pt", pt_bin_lst),
        hist.Bin("eta", "eta", eta_bin_lst),
    )
    hist_ratio._sumw = {(): ratio_arr}
    return hist_ratio


# Plto and save a png for a given 2d histo
def make_2d_fig(histo,xaxis_var,save_name,title=None):
    if title is not None: title_str = title
    else: title_str = save_name
    ax = hist.plot2d(histo,xaxis=xaxis_var)
    plt.title(title_str)
    plt.savefig(save_name)


# Main wrapper function
def make_plot(in_hist):

    # Integrate sample axis
    #in_hist = in_hist.integrate("sample","DYJetsToLL_centralUL17")
    #in_hist = in_hist.integrate("sample","TTJets_centralUL17")
    in_hist = in_hist.sum("sample")

    # Rebin the histo
    in_hist = in_hist.rebin("eta", 2)
    in_hist = in_hist.rebin("pt", hist.Bin("pt","pt",PT_BINS))

    # Grab the histo of flipped e and histo of not flipped e
    hist_flip = in_hist.integrate("flipstatus","truthFlip")
    hist_noflip = in_hist.integrate("flipstatus","truthNoFlip")

    # Calculate ratio and make ratio histo
    flip_sumw_arr = hist_flip._sumw[()]
    noflip_sumw_arr = hist_noflip._sumw[()]
    ratio_sumw_arr = flip_sumw_arr/(flip_sumw_arr+noflip_sumw_arr)
    hist_ratio = make_ratio_hist(ratio_sumw_arr,PT_BINS,ETA_BINS)

    # Print info
    #print("\nflip:",flip_sumw_arr)
    #print("\nnoflip:",noflip_sumw_arr)

    # Save figs
    make_2d_fig(hist_flip,"pt","truth_flip")
    make_2d_fig(hist_noflip,"pt","truth_noflip")
    make_2d_fig(hist_ratio,"pt","truth_ratio","Flip ratio = flip/(flip+noflip)")

    # Save output histo
    #with gzip.open("test_ratio.pkl.gz", "wb") as fout:
        #cloudpickle.dump(hist_ratio, fout)


# Main function
def main():

    # Print info about the in dict
    sample_names_lst = yt.get_cat_lables(hin_dict,"sample")
    flip_names_lst = yt.get_cat_lables(hin_dict,"flipstatus")
    print("Samples:",sample_names_lst)
    print("Flipstatus:",flip_names_lst)

    # Get the hito from the input dict
    histo_pteta = hin_dict["pteta"]

    make_plot(histo_pteta)


if __name__ == "__main__":
    main()
