import os
import copy
import topcoffea.modules.GetValuesFromJsons as getVal
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

#hin_dict = yt.get_hist_from_pkl("/afs/crc.nd.edu/user/k/kmohrman/coffea_dir/check_PRs/sergio_lepmva/topcoffea/analysis/topEFT/histos/mar31_UL17DY_withSSOSTruth_minPtl15_mvaTTHUL.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/apr05_xcheck_00.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/apr05_xcheck_ttHProbs_00.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/apr05_ar_UL17-dy-dy1050-ttbar_withTightChReq_test02.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/apr05_ar_ttHProbs_test02.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/apr05_ar_ttHProbs_test03.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/apr05_ar_UL17-dy-dy1050-ttbar_withTightChReq_test03.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/apr06_ar_UL16UL17UL18_apr06BinningAN19127_00.pkl.gz")
hin_dict = yt.get_hist_from_pkl("histos/apr06_ar_FullR2_apr06BinningAN19127_00.pkl.gz")


outpath = args.outpath

# Get a MC name that corresponds to whatever data we're looking at
def get_mc_name(data_name):
    if "UL17" in data_name:
        return "DYJetsToLL_centralUL17"
    elif "UL18" in data_name:
        return "DY50_centralUL18"
    else: raise Exception

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

    make_plots = True

    out_dict_ss = {}
    out_dict_os = {}

    sample_names_lst = yt.get_cat_lables(hin_dict,"sample")
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
            #if histo_name == "invmass": histo = histo.rebin("invmass",10)
            #if histo_name == "invmass": histo = histo.rebin("invmass",5)
            if histo_name == "invmass": histo = histo.rebin("invmass",2)

            # Integrate and make plot (overlay the categories)
            savename = "_".join([sample_name,histo_name])
            histo = histo.integrate("sample",sample_name)
            h_up = copy.deepcopy(histo)
            h_do = copy.deepcopy(histo)
            h_up.scale(1.3)
            h_do.scale(0.7)
            fig = mp.make_single_fig(histo["ssz"],histo2=histo["osz"],hup=h_up["osz"],hdo=h_do["osz"])
            fig.savefig(os.path.join(outpath,savename))


# Main function
def main():
    make_plot()

main()
