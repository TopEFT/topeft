import os
import copy
import datetime
import argparse
import json

import topcoffea.modules.MakeLatexTable as mlt

BKG_PROC_LST = ["tWZ_sm", "convs_sm","Diboson_sm","Triboson_sm","charge_flips_sm","fakes_sm"]
SIG_PROC_LST = ["ttH_sm", "ttlnu_sm", "ttll_sm", "tllq_sm", "tHq_sm", "tttt_sm"]

PROC_ORDER = [
    "tWZ_sm",
    "Diboson_sm",
    "Triboson_sm",
    "charge_flips_sm",
    "fakes_sm",
    "convs_sm",
    "Sum_bkg",
    "ttlnu_sm",
    "ttll_sm",
    "ttH_sm",
    "tllq_sm",
    "tHq_sm",
    "tttt_sm",
    "Sum_sig",
    "Sum_expected",
    "Observation",
    "Pdiff",
]
CAT_ORDER = [
    "2lss_m_3b",
    "2lss_p_3b",
    "2lss_m_2b",
    "2lss_p_2b",
    "3l_m_1b",
    "3l_p_1b",
    "3l_m_2b",
    "3l_p_2b",
    "3l_onZ_1b",
    "3l_onZ_2b",
    "4l_2b",
]

RENAME_CAT_MAP = {
    # name_in_card : name_we_want_want_in_the_table
    "2lss_4t_m"    : "2lss_m_3b",
    "2lss_4t_p"    : "2lss_p_3b",
    "2lss_m"       : "2lss_m_2b",
    "2lss_p"       : "2lss_p_2b",
    "3l_m_offZ_1b" : "3l_m_1b",
    "3l_m_offZ_2b" : "3l_m_2b",
    "3l_p_offZ_1b" : "3l_p_1b",
    "3l_p_offZ_2b" : "3l_p_2b",
    "3l_onZ_1b"    : "3l_onZ_1b",
    "3l_onZ_2b"    : "3l_onZ_2b",
    "4l"           : "4l_2b",
}


########################################
# General functions
########################################

# Dump contents of a file
def read_file(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

# Print a dictionary
def printd(in_dict):
    for k,v in in_dict.items():
        print("\t",k,":",v)

# Returns a dict with val -> [val, None] for a nested dict
# Useful if passing dict to a function that expects errs
# Assumes two layers of nesting
def append_none_errs(in_dict):
    out_dict = {}
    for k in in_dict.keys():
        out_dict[k] = {}
        for subk in in_dict[k].keys():
            out_dict[k][subk] = [in_dict[k][subk], None]
    return out_dict

# Adds values of two dictionaries that have the same keys
# Not checking to make sure dicts agree, just assuming this
def add_dict_vals(d1,d2):
    out_dict = {}
    for k in d1.keys():
        out_dict[k] = d1[k] + d2[k]
    return out_dict


########################################
# Functions specific for our datacards
########################################

# From a list of filenames, get the list that looks like the text datacards
def get_dc_file_names(flst,ext=".txt"):
    dclst = []
    for fname in flst:
        if fname.startswith("ttx") and fname.endswith(ext):
            dclst.append(fname)
    return (dclst)

# Take the name of a datacard and extract the process name
def get_cat_name_from_dc_name(dc_name,ext=".txt"):
    ret_str = dc_name.replace("ttx_multileptons-","")
    ret_str = ret_str.replace(ext,"")
    return ret_str

# Get the process' rates from the contents of a datacard
def get_rates(dc_lines_lst):

    # Get the rate and processes from the datacard
    dc_proc_line = None
    dc_rate_line = None
    dc_observation_line = None
    for dc_line in dc_lines_lst:
        # The first line starting with "process" is the one we're after
        if dc_proc_line is None and dc_line.startswith("process"):
            dc_proc_line = dc_line
        # Also get the rate line (note there's only one of these)
        if dc_line.startswith("rate"):
            dc_rate_line = dc_line
        # Get the observation line
        if dc_line.startswith("observation"):
            dc_observation_line = dc_line

    # Get lst from lines (drop the "process" and "rate" string first elements)
    proc_lst = dc_proc_line.split()[1:]
    rate_lst = dc_rate_line.split()[1:]
    observation = dc_observation_line.split()[1] # This just has one number

    # Check length
    n_cats = len(proc_lst)
    if len(proc_lst) != len(rate_lst):
        raise Exception("Something is wrong, lenghts do not match.")

    # Put the info into a dictionary
    rate_dict = {}
    for i in range(n_cats):
        rate_dict[proc_lst[i]] = float(rate_lst[i])
    rate_dict["Observation"] = float(observation)

    return (rate_dict)


# Takes a rate dict (as returned by get_rates) and returns just the sm terms
def get_just_sm(rate_dict):
    sm_dict = {}
    for proc_name, rate in rate_dict.items():
        if not proc_name.endswith("_sm") and proc_name != "Observation": continue
        sm_dict[proc_name] = rate
    return sm_dict


# Find sums for e.g. all bkg and all signal
# Retrun a new dict with these extra terms included
def add_proc_sums(rates_dict):

    ret_dict = copy.deepcopy(rates_dict)

    # Sum the background samples
    def sum_rates(rate_dict,lst_to_sum):
        sum_rate = 0
        for proc_name,rate in rates_dict.items():
            if proc_name in lst_to_sum:
                sum_rate += float(rate)
        return sum_rate

    # If a cat (i.e. flips) are not in a category, include them with 0 yield
    def include_skipped_procs_as_zero(rates_dict,proc_lst_to_include):
        ret_dict = copy.deepcopy(rates_dict)
        for proc_name in proc_lst_to_include:
            if proc_name not in rates_dict:
                ret_dict[proc_name] = 0.0
        return ret_dict

    # Put the sub sums into the dict
    bkg_sum = sum_rates(rates_dict,BKG_PROC_LST)
    sig_sum = sum_rates(rates_dict,SIG_PROC_LST)
    exp_sum = sum_rates(rates_dict,BKG_PROC_LST+SIG_PROC_LST)
    ret_dict["Sum_bkg"] = bkg_sum
    ret_dict["Sum_sig"] = sig_sum
    ret_dict["Sum_expected"] = exp_sum

    # Make sure all bkg procs are listed for every category (if they are not relevant, just list 0 yield)
    ret_dict = include_skipped_procs_as_zero(ret_dict,BKG_PROC_LST)

    return ret_dict


# Takes a string (corresponding to a cateogry name), returns the name with the njets and kinematic var info removed
# E.g. "2lss_4t_p_4j_2b_lj0pt" -> "2lss_4t_p_2b"
def get_base_cat_name(cat_name,var_lst=["ptz","lj0pt"]):
    cat_name_split = cat_name.split("_")
    substr_lst_nojets_novarname = []
    for substr in cat_name_split:
        if (len(substr) == 2) and substr.endswith("j"): continue
        if substr in var_lst: continue
        substr_lst_nojets_novarname.append(substr)
    out_name = "_".join(substr_lst_nojets_novarname)
    return out_name


# Take a dictionary of rates and combine jet categories
def comb_dict(in_dict):
    out_dict = {}
    cat_name_lst = in_dict.keys()
    for cat_name in in_dict.keys():
        cat_name_base = get_base_cat_name(cat_name)
        if cat_name_base not in out_dict:
            out_dict[cat_name_base] = in_dict[cat_name]
        else:
            old_vals_dict = out_dict[cat_name_base]
            new_vals_dict = in_dict[cat_name]
            out_dict[cat_name_base] = add_dict_vals(old_vals_dict,new_vals_dict)
    return out_dict

# Replace key names in dictionary of yields
def replace_key_names(in_dict,key_names_map):
    out_dict = {}
    for k in in_dict.keys():
        new_k = key_names_map[k]
        out_dict[new_k] = in_dict[k]
    return out_dict

# Take a dictionary of rates and get rid of the Observation line
# This is useful when ignoring real data if still blinded
def remove_observed_rates(in_dict,comparison_threshold=None):

    out_dict = {}
    ok = True
    for cat_name in in_dict.keys():
        out_dict[cat_name] = {}

        # Check if the observation and prediction are very different
        if comparison_threshold is not None:
            mu = in_dict[cat_name]["Observation"]/in_dict[cat_name]["Sum_expected"]
            if (mu > float(comparison_threshold)) or (mu < 1.0/float(comparison_threshold)):
                print(f"\nWARNING: Observation and Sum_expected are different than more than a factor of {comparison_threshold} in category {cat_name}.")
                ok = False

        # Create the out dict, setting the observation equal to -999
        for proc_name in in_dict[cat_name].keys():
            if proc_name == "Observation":
                out_dict[cat_name][proc_name] = -999
            else:
                out_dict[cat_name][proc_name] = in_dict[cat_name][proc_name]

    # If we were checking the threshold, print if there was agreeemnt or not
    if comparison_threshold is not None:
        if ok: print(f"\nOK: Observation and Sum_expected are within a factor of {comparison_threshold} in all bins.")
        else: print(f"\nWARNING: Observation and Sum_expected are NOT within a factor of {comparison_threshold} in at least one bin.")

    return out_dict


########################################
# Convenience functions
########################################

# Convenience function to get sm yields from a datacard
def get_sm_rates(dc_fullpath):
    dc_content = read_file(dc_fullpath)
    rate_dict = get_rates(dc_content)
    rate_dict_sm = get_just_sm(rate_dict)
    rate_dict_sm_with_sums = add_proc_sums(rate_dict_sm)
    return rate_dict_sm_with_sums


########################################
# Main function
########################################

def main():

    timestamp_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("datacards_dir_path", help = "The path to the directory where the datacards live")
    parser.add_argument("-n", "--json-name", default="dc_yields", help = "Name of the json file to save")
    parser.add_argument("--save-json",action="store_true", help = "Dump output to a json file")
    parser.add_argument("--unblind",action="store_true", help = "Do not remove Observation numbers")
    args = parser.parse_args()

    # Get the list of files in the dc dir
    dc_files_all = os.listdir(args.datacards_dir_path)
    dc_files = get_dc_file_names(dc_files_all)

    # Get rate dicts and cat names
    all_rates_dict = {}
    for dc_fname in dc_files:
        rate_dict_sm = get_sm_rates(os.path.join(args.datacards_dir_path,dc_fname))
        cat_name = get_cat_name_from_dc_name(dc_fname)
        all_rates_dict[cat_name] = rate_dict_sm

    #printd(all_rates_dict)

    # Sum over jet bins and rename the keys, i.e. just some "post processing"
    all_rates_dict = comb_dict(all_rates_dict)
    all_rates_dict = replace_key_names(all_rates_dict,RENAME_CAT_MAP)

    # If we're blind, get rid of the Observation numbers (give warning if off by more than a factor of 2 from prediction)
    if not args.unblind:
        all_rates_dict = remove_observed_rates(all_rates_dict,2)

    # Get pdiff
    for cat in all_rates_dict.keys():
        sm = all_rates_dict[cat]["Sum_expected"]
        ob = all_rates_dict[cat]["Observation"]
        pdiff = 100.0*(sm-ob)/sm
        print(cat,pdiff)
        all_rates_dict[cat]["Pdiff"] = pdiff

    # Dump to the screen text for a latex table
    all_rates_dict_none_errs = append_none_errs(all_rates_dict) # Get a dict that will work for the latex table (i.e. need None for errs)
    mlt.print_latex_yield_table(
        all_rates_dict_none_errs,
        tag="SM yields",
        key_order=CAT_ORDER,
        subkey_order=PROC_ORDER,
        print_begin_info=True,
        print_end_info=True,
        column_variable="keys",
        hz_line_lst=[5,6,12,13,14,15],
    )

    # Save yields to a json
    if args.save_json:
        out_json_name = args.json_name
        if args.json_name == parser.get_default("json_name"):
            out_json_name = out_json_name + "_" + timestamp_tag
        with open(out_json_name+".json", "w") as out_file:
            json.dump(all_rates_dict, out_file, indent=4)
        print(f"Saved json file: {out_json_name}.json\n")


if __name__ == "__main__":
    main()
