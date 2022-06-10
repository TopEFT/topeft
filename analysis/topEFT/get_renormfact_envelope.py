import numpy as np
import argparse
import copy

import gzip
import cloudpickle

from coffea.hist import StringBin, Cat, Bin

from topcoffea.modules.YieldTools import YieldTools
yt = YieldTools()


def dump_to_pkl(out_name,out_histo):
    if not out_name.endswith(".pkl.gz"):
        out_name = out_name + ".pkl.gz"
    print(f"Saving output to {out_name}...")
    with gzip.open(out_name, "wb") as fout:
        cloudpickle.dump(out_histo, fout)
    print("Done.")


def get_rf_envelope(dict_of_hists):

    # Get the list of samples we want to plot
    sample_lst = yt.get_cat_lables(dict_of_hists,"sample")
    cat_lst = yt.get_cat_lables(dict_of_hists,"channel")
    print("\nAll samples:",sample_lst)
    print("\nAll cats:",cat_lst)
    print("\nVariables:",dict_of_hists.keys())

    rf_var_lst = [
        "renormfactUp",
        "renormfactDown",
        "renormUp",
        "renormDown",
        "factUp",
        "factDown"
    ]

    out_hist = {}

    no_renormfact_samples_lst = ["nonpromptUL17"] # Alos flips and data, for all years

    ##########################
    # Test getting the most extreme rf variation

    # Test print some info
    for var_name in dict_of_hists.keys():
        print("\nVar name:",var_name)

        histo = dict_of_hists[var_name]

        sample_lst = yt.get_cat_lables(histo,"sample")
        cat_lst = yt.get_cat_lables(histo,"channel")

        # Loop over samples and channels and find the bins with the most extreme rf variations
        out_dict = {}
        for sample_name in sample_lst:
            if sample_name in no_renormfact_samples_lst: continue

            for cat_name in cat_lst:

                print("\n\n",sample_name,cat_name)
                key_tup_nom = (sample_name, cat_name, "nominal")
                dense_arr_nom = histo.values(overflow="allnan")[key_tup_nom]

                # Find the 6 variations
                diff_wrt_nom_arr_lst = []
                for rf_variation in rf_var_lst:
                    key_tup     = (sample_name, cat_name, rf_variation)
                    dense_arr_diffwrtnom = histo.values(overflow="allnan")[key_tup] - dense_arr_nom
                    diff_wrt_nom_arr_lst.append(dense_arr_diffwrtnom)

                # Get the indices (of the list of the renomr/fact strings) corresponding to the most extreme variation
                max_var_idx = np.argmax(diff_wrt_nom_arr_lst,axis=0)
                min_var_idx = np.argmin(diff_wrt_nom_arr_lst,axis=0)

                # Get the renorm/fact variation string corresponding to the most extreme variation
                rf_vars_extreme_max = np.take(rf_var_lst,max_var_idx)
                rf_vars_extreme_min = np.take(rf_var_lst,min_var_idx)

                # Print stuff
                max_val = np.max(diff_wrt_nom_arr_lst,axis=0)
                min_val = np.min(diff_wrt_nom_arr_lst,axis=0)

                key_tup_rf_env_up = (StringBin(sample_name), StringBin(cat_name), StringBin("renormfactUp"))
                key_tup_rf_env_do = (StringBin(sample_name), StringBin(cat_name), StringBin("renormfactDown"))
                extreme_vals_up_lst = []
                extreme_vals_do_lst = []
                for bin_idx in range(len(rf_vars_extreme_max)):
                    key_tup_extreme_up = (StringBin(sample_name), StringBin(cat_name), StringBin(rf_vars_extreme_max[bin_idx]))
                    key_tup_extreme_do = (StringBin(sample_name), StringBin(cat_name), StringBin(rf_vars_extreme_min[bin_idx]))
                    extreme_vals_up_lst.append(histo._sumw[key_tup_extreme_up][bin_idx])
                    extreme_vals_do_lst.append(histo._sumw[key_tup_extreme_do][bin_idx])

                histo._sumw[key_tup_rf_env_up] = np.array(extreme_vals_up_lst)
                histo._sumw[key_tup_rf_env_do] = np.array(extreme_vals_do_lst)

        histo = histo.remove(["factUp","factDown","renormUp","renormDown"],"systematic")

        out_hist[var_name] = histo

    dump_to_pkl("test_rf_env",out_hist)


def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file_path", help = "The path to the pkl file")
    parser.add_argument("-n", "--output-name", default="plots", help = "A name for the output directory")
    parser.add_argument("-t", "--include-timestamp-tag", action="store_true", help = "Append the timestamp to the out dir name")
    args = parser.parse_args()

    # Get the histograms
    hin_dict = yt.get_hist_from_pkl(args.pkl_file_path,allow_empty=False)

    ref_env = get_rf_envelope(hin_dict)


main()

