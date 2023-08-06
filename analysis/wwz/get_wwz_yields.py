import argparse
import json
import pickle
import gzip

# This script opens a pkl file of histograms produced by wwz processor
# Reads the histograms and dumps out the yields for each group of processes
# Example usage: python get_yld_check.py -f histos/tmp_histo.pkl.gz

sample_dict = {
    "WWZ" : ["UL16APV_WWZJetsTo4L2Nu","UL16_WWZJetsTo4L2Nu","UL17_WWZJetsTo4L2Nu","UL18_WWZJetsTo4L2Nu"],
    "ZH"  : ["UL16APV_GluGluZH","UL16_GluGluZH","UL17_GluGluZH","UL18_GluGluZH"],
    "ZZ"  : ["UL16APV_ZZTo4l","UL16_ZZTo4l","UL17_ZZTo4l","UL18_ZZTo4l"],
}

# Get the yields in the SR
def get_yields(histos_dict):

    yld_dict = {}

    # Look at the yields in one histo (e.g. njets)
    dense_axis = "njets"
    for proc_name in sample_dict.keys():
        yld_dict[proc_name] = {}
        for cat_name in histos_dict[dense_axis].axes["category"]:
            val = sum(sum(histos_dict[dense_axis][{"category":cat_name,"process":sample_dict[proc_name]}].values(flow=True)))
            yld_dict[proc_name][cat_name] = val

    for proc in yld_dict.keys():
        print(f"\n{proc}:")
        for cat in yld_dict[proc].keys():
            val = yld_dict[proc][cat]
            print(f"\t{cat}: {val}")


def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    parser.add_argument("-o", "--output-path", default=".", help = "The path the output files should be saved to")
    parser.add_argument("-n", "--output-name", default="counts_wwz_sync", help = "A name for the output directory")
    args = parser.parse_args()

    # Get the counts from the input hiso
    histo_dict = pickle.load(gzip.open(args.pkl_file_path))

    # Wrapper around the code for getting the yields for sr and bkg samples
    get_yields(histo_dict)




if __name__ == "__main__":
    main()

