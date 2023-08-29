import numpy as np
import argparse

from coffea.hist import StringBin

import topcoffea.modules.utils as utils
from topeft.modules.yield_tools import YieldTools
yt = YieldTools()


# The names of the 6 renorm and fact variations
RENORMFACT_VAR_LST = [
    "renormfactUp",
    "renormfactDown",
    "renormUp",
    "renormDown",
    "factUp",
    "factDown"
]

# Samples that do not include renorm and fact variations
NO_RENORMFACT_LST = [
    "dataUL16",
    "dataUL16APV",
    "dataUL17",
    "dataUL18",
    "flipsUL16",
    "flipsUL16APV",
    "flipsUL17",
    "flipsUL18",
    "nonpromptUL16",
    "nonpromptUL16APV",
    "nonpromptUL17",
    "nonpromptUL18",
]


# Get the most extreme renorm fact variations
def get_renormfact_envelope(dict_of_hists):

    sample_lst = yt.get_cat_lables(dict_of_hists,"process")
    cat_lst = yt.get_cat_lables(dict_of_hists,"channel")
    print("\nAll processes:",sample_lst)
    print("\nAll cats:",cat_lst)
    print("\nAll vars:",dict_of_hists.keys())

    # Get the most extreme renorm fact variations
    out_hist_dict = {}
    for var_name in dict_of_hists.keys():
        print("\tVar name:",var_name)

        # Get the histo for this variable from the input dict
        histo = dict_of_hists[var_name]
        sample_lst = yt.get_cat_lables(histo,"process")
        cat_lst = yt.get_cat_lables(histo,"channel")

        # Loop over samples and channels and find the bins with the most extreme rf variations
        out_dict = {}
        for sample_name in sample_lst:
            if sample_name in NO_RENORMFACT_LST: continue
            for cat_name in cat_lst:
                print("\t\t",sample_name,cat_name)

                # Get the nominal arr
                # Use sumw not values() since it's way faster
                key_tup_nom = (StringBin(sample_name), StringBin(cat_name), StringBin("nominal"))
                dense_arr_nom = histo._sumw[key_tup_nom]
                if dense_arr_nom.ndim == 2:
                    # If this is an EFT bin, just take SM part
                    dense_arr_nom = dense_arr_nom[:,0]

                # Get the 6 renorm/fact variation arrs, and find difference with respect to nominal, appending resulting arrays to a list
                diff_wrt_nom_arr_lst = []
                for rf_variation in RENORMFACT_VAR_LST:
                    key_tup = (StringBin(sample_name), StringBin(cat_name), StringBin(rf_variation))
                    dense_arr_var = histo._sumw[key_tup]
                    if dense_arr_var.ndim == 2:
                        # If this is an EFT bin, just take SM part
                        dense_arr_var = dense_arr_var[:,0]
                    dense_arr_diffwrtnom = dense_arr_var - dense_arr_nom
                    diff_wrt_nom_arr_lst.append(dense_arr_diffwrtnom)

                # Get the indices (of the list of the renomr/fact strings) corresponding to the most extreme variation with respect to nominal
                max_var_idx = np.argmax(diff_wrt_nom_arr_lst,axis=0)
                min_var_idx = np.argmin(diff_wrt_nom_arr_lst,axis=0)

                # Get the renorm/fact variation string corresponding to the most extreme variation
                rf_vars_extreme_max = np.take(RENORMFACT_VAR_LST,max_var_idx)
                rf_vars_extreme_min = np.take(RENORMFACT_VAR_LST,min_var_idx)

                # Now loop over the dense bins (probably a way to do this without a loop)
                # For each dense bin, get the values corresponding to the most extreme variation, and append those to a list (whihc we'll later turn into an array)
                extreme_sumw_up_lst = []
                extreme_sumw_do_lst = []
                extreme_sumw2_up_lst = []
                extreme_sumw2_do_lst = []
                all_sumw2_exists = True # Keep track if all sumw2 arrays are meaningful (otherwise just leave leave sumw2 as None)
                for bin_idx in range(len(rf_vars_extreme_max)):
                    # Get the key tuple for the bin for this category
                    key_tup_extreme_up = (StringBin(sample_name), StringBin(cat_name), StringBin(rf_vars_extreme_max[bin_idx]))
                    key_tup_extreme_do = (StringBin(sample_name), StringBin(cat_name), StringBin(rf_vars_extreme_min[bin_idx]))
                    # Append the sum2 for each bin to the list
                    extreme_sumw_up_lst.append(histo._sumw[key_tup_extreme_up][bin_idx])
                    extreme_sumw_do_lst.append(histo._sumw[key_tup_extreme_do][bin_idx])
                    # Also might as well get the sumw2 for the bin (though we don't really use this right now)
                    extreme_sumw2_up = histo._sumw2[key_tup_extreme_up]
                    extreme_sumw2_do = histo._sumw2[key_tup_extreme_do]
                    if (extreme_sumw2_up is not None) and (extreme_sumw2_do is not None):
                        extreme_sumw2_up_lst.append(extreme_sumw2_up[bin_idx])
                        extreme_sumw2_do_lst.append(extreme_sumw2_do[bin_idx])
                    else:
                        all_sumw2_exists = False

                # This is a bit of a hack, probably not the best way to do this :(
                # Since it's apparently very hard to add categories to a coffea hist, let's just overwrite the sumw values of an exisitng category
                # We won't need renorm or fact or renormfact once we've found the evelope, so just overwrite renormfact
                # At the end we'll remove the renorm and fact categories
                # So what we'll be left with is a renormfact category, whose values are now the evelope of the renorm, fact, and renormfact systeamtics
                key_tup_rf_env_up = (StringBin(sample_name), StringBin(cat_name), StringBin("renormfactUp"))
                key_tup_rf_env_do = (StringBin(sample_name), StringBin(cat_name), StringBin("renormfactDown"))
                histo._sumw[key_tup_rf_env_up] = np.array(extreme_sumw_up_lst)
                histo._sumw[key_tup_rf_env_do] = np.array(extreme_sumw_do_lst)
                if all_sumw2_exists:
                    histo._sumw2[key_tup_rf_env_up] = np.array(extreme_sumw2_up_lst)
                    histo._sumw2[key_tup_rf_env_do] = np.array(extreme_sumw2_do_lst)
                else:
                    histo._sumw2[key_tup_rf_env_up] = None
                    histo._sumw2[key_tup_rf_env_do] = None

        # Remove the left over renorm/fact variations, and put the histo into the output dictionary
        histo = histo.remove(["factUp","factDown","renormUp","renormDown"],"systematic")
        out_hist_dict[var_name] = histo

    return out_hist_dict


# Example standalone usage of get_renormfact_envelope()
# Generally this function will be called from the run script
def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file_path", help = "The path to the pkl file")
    parser.add_argument("-n", "--output-name", default="histos_dict", help = "A name for the output file")
    args = parser.parse_args()

    # Get the envelope and write to an out pkl
    hin_dict = utils.get_hist_from_pkl(args.pkl_file_path,allow_empty=False)
    hout_dict = get_renormfact_envelope(hin_dict)
    utils.dump_to_pkl(args.output_name,hout_dict)

if __name__ == "__main__":
    main()

