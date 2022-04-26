import os
import copy
import datetime
import argparse
import json

import topcoffea.modules.MakeLatexTable as mlt

BKG_PROC_LST = ["convs_sm","Diboson_sm","Triboson_sm","charge_flips_sm","fakes_sm"]
SIG_PROC_LST = ["ttH_sm", "ttlnu_sm", "ttll_sm", "tllq_sm", "tHq_sm", "tttt_sm"]

PROC_ORDER = [
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
]
CAT_ORDER = [
    "2lss_4t_m_2b",
    "2lss_4t_p_2b",
    "2lss_m_2b",
    "2lss_p_2b",
    "3l1b_m",
    "3l1b_p",
    "3l2b_m",
    "3l2b_p",
    "3l_sfz_1b",
    "3l_sfz_2b",
    "4l_2b",
]


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


########################################
# Functions specific for our datacards
########################################

# From a list of filenames, get the list that looks like the text datacards
def get_dc_file_names(flst,ext=".txt"):
    dclst = []
    for fname in flst:
        if fname.startswith("ttx") and fname.endswith(ext):
            dclst.append(fname)
    return(dclst)

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
    for dc_line in dc_lines_lst:
        # The first line starting with "process" is the one we're after
        if dc_proc_line is None and dc_line.startswith("process"):
            dc_proc_line = dc_line
        # Also get the rate line (note there's only one of these)
        if dc_line.startswith("rate"):
            dc_rate_line = dc_line

    # Get lst from lines (drop the "process" and "rate" string first elements)
    proc_lst = dc_proc_line.split()[1:]
    rate_lst = dc_rate_line.split()[1:]

    # Check length
    n_cats = len(proc_lst)
    if len(proc_lst) != len(rate_lst):
        raise Exception("Something is wrong, lenghts do not match.")

    # Put the info into a dictionary
    rate_dict = {}
    for i in range(n_cats):
        rate_dict[proc_lst[i]] = float(rate_lst[i])

    return(rate_dict)


# Takes a rate dict (as returned by get_rates) and returns just the sm terms
def get_just_sm(rate_dict):
    sm_dict = {}
    for proc_name, rate in rate_dict.items():
        if not proc_name.endswith("_sm"): continue
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
    args = parser.parse_args()

    # Get the list of files in the dc dir
    dc_files_all = os.listdir(args.datacards_dir_path)
    dc_files = get_dc_file_names(dc_files_all)

    # Get rate dicts and cat names
    all_rates_dict = {}
    for dc_fname in dc_files:
        rate_dict_sm = get_sm_rates(os.path.join(args.datacards_dir_path,dc_fname))
        cat_name = get_cat_name_from_dc_name(dc_fname)
        print(cat_name)
        printd(rate_dict_sm)
        all_rates_dict[cat_name] = rate_dict_sm

    # Dump to the screen text for a latex table
    all_rates_dict_none_errs = append_none_errs(all_rates_dict) # Get a dict that will work for the latex table (i.e. need None for errs)
    mlt.print_latex_yield_table(all_rates_dict_none_errs,tag="SM yields",key_order=CAT_ORDER,subkey_order=PROC_ORDER,print_begin_info=True,print_end_info=True,column_variable="keys")

    # Save yields to a json
    out_json_name = args.json_name
    if args.json_name == parser.get_default("json_name"):
        out_json_name = out_json_name + "_" + timestamp_tag
    with open(out_json_name+".json", "w") as out_file:
        json.dump(all_rates_dict, out_file, indent=4)
    print(f"Saved json file: {out_json_name}.json\n")


if __name__ == "__main__":
    main()
