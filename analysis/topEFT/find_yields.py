import gzip
import json
import pickle
import numpy as np
from coffea import hist
#from topcoffea.modules.HistEFT import HistEFT
from topcoffea.modules.paths import topcoffea_path

CAT_LST = ["cat_2lss_p","cat_2lss_m","cat_3l_1b_offZ_p","cat_3l_1b_offZ_m","cat_3l_2b_offZ_p","cat_3l_2b_offZ_m","cat_3l_1b_onZ","cat_3l_2b_onZ","cat_4l"]

PROC_MAP = {
    "ttlnu" : ["ttW_centralUL17","ttlnu_private2017","ttlnuJet_privateUL17","ttlnuJet_privateUL18"],
    "ttll"  : ["ttZ_centralUL17","ttll_TOP-19-001","ttllJet_privateUL17","ttllJet_privateUL18"],
    "ttH"   : ["ttH_centralUL17","ttH_private2017","ttHJet_privateUL17","ttHJet_privateUL18"],
    "tllq"  : ["tZq_centralUL17","tllq_private2017","tllq_privateUL17","tllq_privateUL18"],
    "tHq"   : ["tHq_central2017","tHq_privateUL17"],
    "tttt"  : ["tttt_central2017","tttt_privateUL17"],
}

JET_BINS = {
    "2lss" : [4,5,6,7],
    "3l"   : [2,3,4,5],
    "4l"   : [2,3,4],
}

ch_3l_onZ = ["eemSSonZ", "mmeSSonZ", "eeeSSonZ", "mmmSSonZ"]
ch_3l_offZ = ["eemSSoffZ", "mmeSSoffZ", "eeeSSoffZ", "mmmSSoffZ"]
ch_2lss = ["eeSSonZ", "eeSSoffZ", "mmSSonZ", "mmSSoffZ", "emSS"]
ch_4l = ["eeee","eeem","eemm","mmme","mmmm"]

CATEGORIES = {
    "cat_2lss_p" : {
        "channel": ch_2lss,
        "sumcharge": ["ch+"],
        "cut": ["1+bm2+bl"],
    },
    "cat_2lss_m" : {
        "channel": ch_2lss,
        "sumcharge": ["ch-"],
        "cut": ["1+bm2+bl"],
    },

    "cat_3l_1b_onZ" : {
        "channel": ch_3l_onZ,
        "sumcharge": ["ch+","ch-"],
        "cut": ["1bm"],
    },
    "cat_3l_1b_offZ_p" : {
        "channel": ch_3l_offZ,
        "sumcharge": ["ch+"],
        "cut": ["1bm"],
    },
    "cat_3l_1b_offZ_m" : {
        "channel": ch_3l_offZ,
        "sumcharge": ["ch-"],
        "cut": ["1bm"],
    },

    "cat_3l_2b_onZ" : {
        "channel": ch_3l_onZ,
        "sumcharge": ["ch+","ch-"],
        "cut": ["2+bm"],
    },
    "cat_3l_2b_offZ_p" : {
        "channel": ch_3l_offZ,
        "sumcharge": ["ch+"],
        "cut": ["2+bm"],
    },
    "cat_3l_2b_offZ_m" : {
        "channel": ch_3l_offZ,
        "sumcharge": ["ch-"],
        "cut": ["2+bm"],
    },

    "cat_4l" : {
        "channel": ch_4l,
        "sumcharge": ["ch+","ch-","ch0"],
        "cut": ["1+bm2+bl"],
    },
}

######### Functions for getting process names from PROC_MAP #########

# What this function does:
#   - Takes a full process name (i.e. the name of the category on the samples axis)
#   - Then loops through PROC_MAP and returns the short (i.e. standard) version of the process name
def get_short_name(long_name):
    ret_name = None
    for short_name,long_name_lst in PROC_MAP.items():
        if long_name in long_name_lst:
            ret_name = short_name
            break
    return ret_name

# What this function does:
#   - Takes a list of full process names (i.e. all of the categories on samples axis) and a key from PROC_MAP
#   - Returns the long (i.e. the name of the cateogry in the smples axis) corresponding to the short name
def get_long_name(long_name_lst_in,short_name_in):
    ret_name = None
    for long_name in PROC_MAP[short_name_in]:
        for long_name_in in long_name_lst_in:
            if long_name_in == long_name:
                ret_name = long_name
    return ret_name


######### General functions #########

# Get percent difference
def get_pdiff(a,b):
    #p = (float(a)-float(b))/((float(a)+float(b))/2)
    if ((a is None) or (b is None)):
        p = None
    elif b == 0:
        p = None
    else:
        p = (float(a)-float(b))/float(b)
    return p

# Get the dictionary of hists from the pkl file (that the processor outputs)
def get_hist_from_pkl(path_to_pkl):
    h = pickle.load( gzip.open(path_to_pkl) )
    return h

# Return the lumi from the json/lumi.json file for a given year
def get_lumi(year):
    lumi_json = topcoffea_path("json/lumi.json")
    with open(lumi_json) as f_lumi:
       lumi = json.load(f_lumi)
       lumi = lumi[year]
    return lumi

# Takes a hist dictionary (i.e. from the pkl file that the processor makes) and an axis name, retrun the list of categories for that axis
def get_cat_lables(hin_dict,axis,h_name=None):
    cats = []
    if h_name is None: h_name = "njets" # Guess a hist that we usually have
    for i in range(len(hin_dict[h_name].axes())):
        if axis == hin_dict[h_name].axes()[i].name:
            for cat in hin_dict[h_name].axes()[i].identifiers():
                cats.append(cat.name)
    return cats

# Takes a histogram and a dictionary that specifies categories, integrates out the categories listed in the dictionry
def integrate_out_cats(h,cuts_dict):
    h_ret = h
    for axis_name,cat_lst in cuts_dict.items():
        h_ret = h_ret.integrate(axis_name,cat_lst)
    return h_ret

# Takes a histogram and a bin, rebins the njets axis to consist of only that bin (all other bins combined into under/overvlow)
def select_njet_bin(h,bin_val):
    if not isinstance(bin_val,int):
        raise Exception(f"Need to pass an int to this function, got a {type(bin_val)} instead. Exiting...")
    h = h.rebin('njets', hist.Bin("njets",  "Jet multiplicity ", [bin_val,bin_val+1]))
    return h

# Get the percent difference between values in nested dictionary of the following format.
# Returns a dictionary in the same formate (cuttently does not propagate errors, just returns None)
#   dict = {
#       k : {
#           subk : (val,err)
#       }
#   }
def get_pdiff_between_nested_dicts(dict1,dict2):

    ret_dict = {}
    for k1 in dict1.keys():

        if k1 in dict2.keys():
            ret_dict[k1] = {}
            for subk1 in dict1[k1].keys():
                if subk1 not in dict2[k1].keys():
                    raise Exception("These dictionaries do not have the same structure, exiting...")
                v1,e1 = dict1[k1][subk1]
                v2,e1 = dict2[k1][subk1]
                pdiff = get_pdiff(v1,v2)
                ret_dict[k1][subk1] = (pdiff,None)
        else:
            print(f"Warning, key {k1} is not in both dictionaries, continuing...")
            continue

    return ret_dict


######### Functions specifically for getting yields #########

# Sum all the values of a hist 
#    - The hist you pass should have two axes (all other should already be integrated out)
#    - The two axes should be the samples axis, and the dense axis (e.g. ht)
#    - You pass a process name, and we select just that category from the sample axis
def get_yield(h,proc,overflow_str="none"):
    h_vals = h[proc].values(sumw2=True,overflow=overflow_str)
    for i,(k,v) in enumerate(h_vals.items()):
        v_sum = v[0].sum()
        e_sum = v[1].sum()
        if i > 0: raise Exception("Why is i greater than 0? The hist is not what this function is expecting. Exiting...")
    e_sum = np.sqrt(e_sum)
    return (v_sum,e_sum)

# Uses integrate_out_cats() and select_njet_bin() to return h for a particular analysis cateogry
#def get_h_for_cat(h,cat_dict,njet):
def select_hist_for_ana_cat(h,cat_dict,njet):
    h_ret = integrate_out_cats(h,cat_dict)
    h_ret = h_ret.integrate("systematic","nominal") # For now anyway...
    h_ret = select_njet_bin(h_ret,njet)
    return h_ret

# This is really just a wrapper for get_yield(). Note:
#   - This fucntion now also rebins the njets hists
#   - Maybe that does not belong in this function
def get_scaled_yield(hin_dict,year,proc,cat,njets_cat,overflow_str):

    h = hin_dict["njets"]

    if isinstance(njets_cat,str):
        njet = JET_BINS[njets_cat][0]
    elif isinstance(njets_cat,int):
        njet = njets_cat
    else:
        raise Exception(f"Wrong type, expected str or int, but got a {type(njets_cat)}, exiting...")

    h = select_hist_for_ana_cat(h,CATEGORIES[cat],njet)

    lumi = 1000.0*get_lumi(year)
    h_sow = hin_dict["SumOfEFTweights"]
    nwc = h_sow._nwc

    if nwc > 0:
        sow_val , sow_err = get_yield(h_sow,proc)
        h.scale(1.0/sow_val) # Divide EFT samples by sum of weights at SM, ignore error propagation for now
        #print("sow_val,sow_err",sow_val,sow_err,"->",sow_err/sow_val)
        #print("Num of WCs:",nwc)
        #print("Sum of weights:",sow)

    h.scale(lumi)
    return get_yield(h,proc,overflow_str)


# This function:
#   - Takes as input a hist dict (i.e. what the processor outptus)
#   - Retruns a dictionary of yields for the categories in CATEGORIES
#   - Making use of get_scaled_yield()
#   - If you pass a key from JET_BINS for yields_for_njets_cats, will make a dict of yields for all the jet bins in that lep cat
def get_yld_dict(hin_dict,year,yields_for_njets_cats=None):

    yld_dict = {}
    proc_lst = get_cat_lables(hin_dict,"sample")
    for proc in proc_lst:
        proc_name_short = get_short_name(proc)
        yld_dict[proc_name_short] = {}
        for cat,cuts_dict in CATEGORIES.items():
            if "2lss" in cat: njet_cat = "2lss"
            elif "3l" in cat: njet_cat = "3l"
            elif "4l" in cat: njet_cat = "4l"

            # We want to sum over the jet bins, and look at all of the let cats
            if yields_for_njets_cats is None:
                yld_dict[proc_name_short][cat] = get_scaled_yield(hin_dict,year,proc,cat,njet_cat,overflow_str="over")

            # We want to look at all the jet bins in the give lep cat
            elif yields_for_njets_cats == njet_cat:
                for njet in JET_BINS[njet_cat]:
                    if njet == max(JET_BINS[njet_cat]): include_overflow = "over"
                    else: include_overflow = "none"
                    cat_name_full = cat+"_"+str(njet)+"j"
                    yld_dict[proc_name_short][cat_name_full] = get_scaled_yield(hin_dict,year,proc,cat,njet,overflow_str=include_overflow)

    return yld_dict


######### Functions that just print out information #########

# Print out all the info about all the axes in a hist
def print_hist_info(path):

    if type(path) is str: hin_dict = get_hist_from_pkl(path)
    else: hin_dict = path

    h_name = "njets"
    #h_name = "SumOfEFTweights"
    for i in range(len(hin_dict[h_name].axes())):
        print(f"\n{i} Aaxis name:",hin_dict[h_name].axes()[i].name)
        for cat in hin_dict[h_name].axes()[i].identifiers():
            print(cat)

# Print a latex table for the yields (assumes the rows are PROC_MAP.keys())
def print_latex_yield_table(yld_dict,col_order_lst,tag,print_begin_info=False,print_end_info=False,print_errs=False,column_variable="cats"):

    def format_header(column_lst):
        s = "\\hline "
        for i,col in enumerate(column_lst):
            col = col.replace("_"," ")
            col = col.replace("cat","")
            s = s + " & " + col
        s = s + " \\\\ \\hline"
        return s

    def print_begin():
        print("\n")
        print("\\documentclass[10pt]{article}")
        print("\\usepackage[margin=0.05in]{geometry}")
        print("\\begin{document}")

    def print_end():
        print("\\end{document}")
        print("\n")

    def print_table(proc_lst,cat_lst,columns):
        print("\\begin{table}[hbtp!]")
        print("\\setlength\\tabcolsep{5pt}")
        print(f"\\caption{{{tag}}}") # Need to escape the "{" with another "{"


        # Print categories as columns
        if columns == "cats":
            tabular_info = "c"*(len(cat_lst)+1)
            print(f"\\begin{{tabular}}{{{tabular_info}}}")
            print(format_header(cat_lst))
            for proc in proc_lst:
                if proc not in yld_dict.keys():
                    print("\\rule{0pt}{3ex} ","-",end=' ')
                    for cat in cat_lst:
                        print("&","-",end=' ')
                    print("\\\ ")
                else:
                    print("\\rule{0pt}{3ex} ",proc.replace("_"," "),end=' ')
                    for cat in cat_lst:
                        yld , err = yld_dict[proc][cat]
                        if yld is not None: yld = round(yld,2)
                        if err is not None: err = round(err,2)
                        if print_errs:
                            print("&",yld,"$\pm$",err,end=' ')
                        else:
                            print("&",yld,end=' ')
                    print("\\\ ")

        # Print processes as columns
        if columns == "procs":
            tabular_info = "c"*(len(proc_lst)+1)
            print(f"\\begin{{tabular}}{{{tabular_info}}}")
            print(format_header(proc_lst))
            for cat in cat_lst:
                print("\\rule{0pt}{3ex} ",cat.replace("_"," "),end=' ')
                for proc in proc_lst:
                    if proc not in yld_dict.keys():
                        print("& - ",end=' ')
                    else:
                        yld , err = yld_dict[proc][cat]
                        if yld is not None: yld = round(yld,2)
                        if err is not None: err = round(err,2)
                        if print_errs:
                            print("&",yld,"$\pm$",err,end=' ')
                        else:
                            print("&",yld,end=' ')
                print("\\\ ")

        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")

    if print_begin_info: print_begin()
    print_table(PROC_MAP.keys(),col_order_lst,columns=column_variable)
    if print_end_info: print_end()

# Takes yield dicts (i.e. what get_yld_dict() returns) and prints it
def print_yld_dicts(ylds_dict,tag,show_errs=False):
    print(f"\n--- {tag} ---\n")
    for proc in ylds_dict.keys():
        print(proc)
        for cat in ylds_dict[proc].keys():
            print(f"    {cat}")
            val , err = ylds_dict[proc][cat]
            if show_errs:
                #print(f"\t{val} +- {err}")
                print(f"\t{val} +- {err} -> {err/val}")
            else:
                print(f"\t{val}")

# This function:
#    - Takes as input a yld dict
#    - Sums ylds over processes to find total for each cat
#    - Rreturns a dict with the same structure, where the vals are the cat's fractional contribution to the total
def find_relative_contributions(yld_dict):

    sum_dict = {}
    for proc in yld_dict.keys():
        for cat in yld_dict[proc].keys():
            if cat not in sum_dict.keys(): sum_dict[cat] = 0
            yld,err = yld_dict[proc][cat]
            sum_dict[cat] = sum_dict[cat] + yld

    ret_dict = {}
    for proc in yld_dict.keys():
        ret_dict[proc] = {}
        for cat in yld_dict[proc].keys():
            yld,err = yld_dict[proc][cat]
            ret_dict[proc][cat] = (yld/sum_dict[cat],None) # No propagation of errors

    return ret_dict


######### The main() function #########

def main():

    # Paths to the input pkl files
    fpath_default  = "histos/plotsTopEFT.pkl.gz"
    fpath_cuts_centralUl17_test = "histos/plotsTopEFT_centralUL17_fix4l.pkl.gz"
    #fpath_cuts_centralUl17_test = "histos/plotsTopEFT_centralUL17_all-UL-but-TTTT-THQ.pkl.gz"
    fpath_cuts_privateUl17_test = "histos/plotsTopEFT_privateUL17_fix4l.pkl.gz"
    #fpath_witherrors = "histos/test_privateUL17_1c_doerrors.pkl.gz"

    # Get the histograms from the files
    hin_dict_central = get_hist_from_pkl(fpath_cuts_centralUl17_test)
    hin_dict_private = get_hist_from_pkl(fpath_cuts_privateUl17_test)

    # Get the yield dictionaries and percent difference
    ylds_central_dict = get_yld_dict(hin_dict_central,"2017")
    ylds_private_dict = get_yld_dict(hin_dict_private,"2017")
    pdiff_dict = get_pdiff_between_nested_dicts(ylds_private_dict,ylds_central_dict)

    # Print out yields and percent differences
    print_yld_dicts(ylds_central_dict,"Central UL17 yields")
    print_yld_dicts(ylds_private_dict,"Private UL17 yields")
    #print_yld_dicts(pdiff_dict,"Percent diff between private and central")

    # Print latex table
    print_latex_yield_table(ylds_central_dict,CAT_LST,"Central UL17",print_begin_info=True,print_errs=True)
    print_latex_yield_table(ylds_private_dict,CAT_LST,"Private UL17")
    print_latex_yield_table(pdiff_dict,CAT_LST,"Percent diff between central and private UL17: (private-central)/private",print_end_info=True)


    ###### Print info for the jet cats ######
    for lep_cat in JET_BINS.keys():
        print("lep_cat:",lep_cat)
        ylds_private_dict_jets = get_yld_dict(hin_dict_private,"2017",lep_cat)
        #print_yld_dicts(ylds_central_dict_test_jets,"Test")
        print_latex_yield_table(ylds_private_dict_jets,ylds_private_dict_jets["ttH"].keys(),"Private UL17 with jet cats",print_begin_info=True,print_end_info=True,column_variable="procs")
        relative_contributions = find_relative_contributions(ylds_private_dict_jets)
        print_latex_yield_table(relative_contributions,relative_contributions["ttH"].keys(),"Relative contributions",print_begin_info=True,print_end_info=True,column_variable="procs")

    #print_hist_info(hin_dict)
    #exit()

if __name__ == "__main__":
    main()
