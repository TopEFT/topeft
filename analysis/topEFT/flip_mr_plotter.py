import os
import copy
import json
import matplotlib.pyplot as plt
import uproot3
from coffea import hist

from topcoffea.modules.YieldTools import YieldTools
import make_cr_and_sr_plots as mp


yt = YieldTools()

import argparse
parser = argparse.ArgumentParser(description='You can customize your run')
#parser.add_argument("filepath"          , default='histos/plotsTopEFT.pkl.gz', help = 'path of file with histograms')
parser.add_argument("--outpath" ,'-o'   , default='.', help = 'Path to the output directory')
args = parser.parse_args()

#hin_dict = yt.get_hist_from_pkl("histos/apr01_UL17DY_2d_00_mvaTTHUL.pkl.gz")
hin_dict = yt.get_hist_from_pkl("histos/apr04_UL17-dy-dy1050-ttbar_test01.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/apr04_fullR2-dy-dy1050-ttbar_test01.pkl.gz")

'''
    'DY10to50_centralUL16', 
    'DY50_centralUL16', 
    'TTJets_centralUL16', 

    'DYJetsToLL_centralUL17', 
    'TTJets_centralUL17'

    'DY10to50_centralUL18', 
    'DY50_centralUL18', 
'''

PT_BINS = [0,20,30,40,50,60,70,100,200]

def build_out_dict(histo2d):
    out_dict = {}


# Main wrapper function
def make_plot():

    sample_names_lst = yt.get_cat_lables(hin_dict,"sample")
    flip_names_lst = yt.get_cat_lables(hin_dict,"flipstatus")

    print("Samples:",sample_names_lst)
    print("Flipstatus:",flip_names_lst)

    histo_pteta = hin_dict["pteta"]
    #histo_pteta = histo_pteta.integrate("sample","DYJetsToLL_centralUL17")
    #histo_pteta = histo_pteta.integrate("sample","TTJets_centralUL17")
    histo_pteta = histo_pteta.sum("sample")

    #histo_pteta = h_base.rebin(variable, hist.Bin(variable,  h.axis(variable).label, self.analysis_bins[variable][lep_bin]))
    #self.analysis_bins['ptbl'] = [0, 100, 200, 400, self.hists['ptbl'].axis('ptbl').edges()[-1]]
    histo_pteta = histo_pteta.rebin("eta", 2)
    histo_pteta = histo_pteta.rebin("pt", hist.Bin("pt","pt",PT_BINS))

    hist_flip = histo_pteta.integrate("flipstatus","truthFlip")
    hist_noflip = histo_pteta.integrate("flipstatus","truthNoFlip")

    # Try to get errors
    #print(hist_flip.values(sumw2=True))
    #print(hist_flip.values(sumw2=False))
    #exit()

    # Get ratios
    flip_sumw_arr = hist_flip._sumw[()]
    noflip_sumw_arr = hist_noflip._sumw[()]
    ratio_sumw_arr = flip_sumw_arr/(flip_sumw_arr+noflip_sumw_arr)

    print("\nflip:",flip_sumw_arr)
    print("\nnoflip:",noflip_sumw_arr)
    #print("\nflip err:",flip_sumw2_arr)
    #print("\nnoflip err:",noflip_sumw2_arr)


    # Flips
    #ax = hist.plot2d(histo_pteta.integrate("flipstatus","truthFlip"),xaxis="eta")
    ax = hist.plot2d(hist_flip,xaxis="pt")
    plt.title("truth_flip")
    plt.savefig("truth_flip")

    # No flips
    ax = hist.plot2d(hist_noflip,xaxis="pt")
    plt.title("truth_noflip")
    plt.savefig("truth_noflip")

    # Ratio
    hist_ratio = hist.Hist(
        "Ratio",
        #hist.Bin("pt", "pt", 20, 0, 200),
        hist.Bin("pt", "pt", PT_BINS),
        hist.Bin("eta", "eta", 5, 0, 2.5),
    )
    hist_ratio._sumw = {(): ratio_sumw_arr}
    ax = hist.plot2d(hist_ratio,xaxis="pt")
    plt.title("Flip ratio = flip/(flip+noflip)")
    plt.savefig("truth_ratio")

    #plt.show()

    #def export2d(h)
    #return h.to_hist().to_numpy()

    print("---")
    print(hist_ratio.values())
    #x = hist_ratio.to_hist().to_numpy()
    x = hist_ratio.to_hist()
    print("x",x)



make_plot()
