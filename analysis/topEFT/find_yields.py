import gzip
import json
import pickle
from coffea import hist
#from topcoffea.modules.HistEFT import HistEFT
from topcoffea.modules.paths import topcoffea_path

CAT_LST = ["cat_2lss_p","cat_2lss_m","cat_3l_1b_offZ_p","cat_3l_1b_offZ_m","cat_3l_2b_offZ_p","cat_3l_2b_offZ_m","cat_3l_1b_onZ","cat_3l_2b_onZ","cat_4l"]

PROC_MAP = {
    "ttlnu" : ["ttW_centralUL17","ttlnu_private2017","ttlnuJet_privateUL17","ttlnuJet_privateUL18"],
    "ttll" : ["ttZ_centralUL17","ttll_TOP-19-001","ttllJet_privateUL17","ttllJet_privateUL18"],
    "ttH" : ["ttH_centralUL17","ttH_private2017","ttHJet_privateUL17","ttHJet_privateUL18"],
    "tllq" : ["tZq_centralUL17","tllq_private2017","tllq_privateUL17","tllq_privateUL18"],
    "tHq" : ["tHq_privateUL17"],
}

ch_3l_onZ = ["eemSSonZ", "mmeSSonZ", "eeeSSonZ", "mmmSSonZ"]
ch_3l_offZ = ["eemSSoffZ", "mmeSSoffZ", "eeeSSoffZ", "mmmSSoffZ"]
ch_2lss = ["eeSSonZ", "eeSSoffZ", "mmSSonZ", "mmSSoffZ", "emSS"]
ch_4l = ["eeee","eeem","eemm","mmme","mmmm"]

CATEGORIES = {
    "cat_2lss_p" : {
        "channel": ch_2lss,
        "sumcharge": ["ch+"],
        "nbjet": ["1+bm2+bl"],
    },
    "cat_2lss_m" : {
        "channel": ch_2lss,
        "sumcharge": ["ch-"],
        "nbjet": ["1+bm2+bl"],
    },

    "cat_3l_1b_onZ" : {
        "channel": ch_3l_onZ,
        "sumcharge": ["ch+","ch-"],
        "nbjet": ["1bm"],
    },
    "cat_3l_1b_offZ_p" : {
        "channel": ch_3l_offZ,
        "sumcharge": ["ch+"],
        "nbjet": ["1bm"],
    },
    "cat_3l_1b_offZ_m" : {
        "channel": ch_3l_offZ,
        "sumcharge": ["ch-"],
        "nbjet": ["1bm"],
    },

    "cat_3l_2b_onZ" : {
        "channel": ch_3l_onZ,
        "sumcharge": ["ch+","ch-"],
        "nbjet": ["2+bm"],
    },
    "cat_3l_2b_offZ_p" : {
        "channel": ch_3l_offZ,
        "sumcharge": ["ch+"],
        "nbjet": ["2+bm"],
    },
    "cat_3l_2b_offZ_m" : {
        "channel": ch_3l_offZ,
        "sumcharge": ["ch-"],
        "nbjet": ["2+bm"],
    },

    "cat_4l" : {
        "channel": ch_4l,
        "sumcharge": ["ch+","ch-"],
        "nbjet": ["1+bm2+bl"],
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


# Get the percent difference between values in nested dictionary of the format:
#   dict1 = {
#       k1 : {
#           subk1 : v1
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
                v1 = dict1[k1][subk1]
                v2 = dict2[k1][subk1]
                pdiff = get_pdiff(v1,v2)
                ret_dict[k1][subk1] = pdiff
        else:
            print(f"Warning, key {k1} is not in both dictionaries, continuing...")
            continue

    return ret_dict


######### Functions specifically for getting yields #########

# Sum all the values of a hist 
#    - The hist you pass should have two axes (all other should already be integrated out)
#    - The two axes should be the samples axis, and the dense axis (e.g. ht)
#    - You pass a process name, and we select just that category from the sample axis
def get_yield(h,proc,cat_lst=None):
    #print("\nIn get_yields()")
    if cat_lst is not None:
        h = h.integrate(axis_name,cats_lst)
        # I don't want to use the function like this right now, but don't want to deleate this in case we want it in the future
        print("I don't think we should be here !!!")
        raise Exception
    #h_vals = h[proc].values(overflow='all')
    h_vals = h[proc].values(overflow='over')
    #h_vals = h[proc].values()
    for i,(k,v) in enumerate(h_vals.items()):
        v_sum = v.sum()
        if i > 0: raise Exception("Why is i greater than 0? Something is not what you thought it was.")
    return v_sum

# This is really just a wrapper for get_yield(). Note:
#   - This fucntion now also rebins the njets hists
#   - Maybe that does not belong in this function
def get_scaled_yield(hin_dict,year,proc,cat):

    #h = hin_dict["ht"]
    h = hin_dict["njets"]

    h = integrate_out_cats(h,CATEGORIES[cat])
    h = h.integrate("cut","base").integrate("systematic","nominal")

    if '2l' in cat:
        h = h.rebin('njets', hist.Bin("njets",  "Jet multiplicity ", [4,5,6,7,8,9,10]))
    elif '3l' in cat:
        h = h.rebin('njets', hist.Bin("njets",  "Jet multiplicity ", [2,3,4,5,6,7,8,9,10]))
    elif '4l' in cat: 
        h = h.rebin('njets', hist.Bin("njets",  "Jet multiplicity ", [2,3,4,5,6,7,8,9,10]))

    lumi = 1000.0*get_lumi(year)
    h_sow = hin_dict["SumOfEFTweights"]
    nwc = h_sow._nwc

    if nwc > 0:
        sow = get_yield(h_sow,proc)
        h.scale(1.0/sow) # Divide EFT samples by sum of weights at SM
        #print("Num of WCs:",nwc)
        #print("Sum of weights:",sow)

    yld = lumi*get_yield(h,proc)
    return yld

# This function:
#   - Takes as input a hist dict (i.e. what the processor outptus)
#   - Retruns a dictionary of yields for the categories in CATEGORIES
#   - Making use of get_scaled_yield()
def get_yld_dict(hin_dict,year):
    yld_dict = {}
    proc_lst = get_cat_lables(hin_dict,"sample")
    for proc in proc_lst:
        proc_name_short = get_short_name(proc)
        #yld_dict[proc] = {}
        yld_dict[proc_name_short] = {}
        for cat,cuts_dict in CATEGORIES.items():
            yld = get_scaled_yield(hin_dict,year,proc,cat)
            #yld_dict[proc][cat]= yld
            yld_dict[proc_name_short][cat]= yld
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

# Print a latex table for the yields
def print_latex_yield_table(year,hin_dict1,hin_dict2=None):

    def format_header(cat_lst):
        s = "\\hline "
        for i,cat in enumerate(cat_lst):
            cat = cat.replace("_"," ")
            cat = cat.replace("cat","")
            s = s + " & " + cat
        s = s + " \\\\ \\hline"
        return s

    def print_begin():
        print("\\documentclass[10pt]{article}")
        print("\\usepackage[margin=0.1in]{geometry}")
        print("\\begin{document}")

    def print_end():
        print("\\end{document}")

    def print_table(cat_lst,hin_dict,year):
        print("\\begin{table}[hbtp!]")
        print("\\caption{}")
        print("\\begin{tabular}{cccccccccc}") # Hard coded :(
        print(format_header(cat_lst))

        proc_lst = get_cat_lables(hin_dict,"sample")
        #for proc in proc_lst:
        for ptag,ptag_lst in PROC_MAP.items():
            proc = None
            for p in proc_lst:
                if p in ptag_lst:
                    proc = p
                    break
            if proc is None:
                print("\\rule{0pt}{3ex} ","-",end=' ')
                for cat in cat_lst:
                    print("&","-",end=' ')
                print("\\\ ")
            else:
                print("\\rule{0pt}{3ex} ",proc.replace("_"," "),end=' ')
                for cat in cat_lst:
                    yld = get_scaled_yield(hin_dict,year,proc,cat)
                    yld = round(yld,2)
                    print("&",yld,end=' ')
                print("\\\ ")

        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")

    print_begin()
    print_table(CAT_LST,hin_dict1,year)
    #print_table(CAT_LST,hin_dict2,year)
    print_end()

# Takes yield dicts (i.e. what get_yld_dict() returns) and prints it
def print_yld_dicts(ylds_dict,tag):
    print(f"\n---{tag}---\n")
    for proc in ylds_dict.keys():
        print(proc)
        for cat in ylds_dict[proc].keys():
            print(f"    {cat}")
            print("\t",ylds_dict[proc][cat])



######### The main() function #########

def main():

    fpath_default  = "histos/plotsTopEFT.pkl.gz"
    #fpath_cuts_privateUl17_test = "histos/plotsTopEFT_privateUL17_ttH-ttll-b1.pkl.gz"
    fpath_cuts_privateUl17_test = "histos/plotsTopEFT_privateUL17_allBut4t_b1.pkl.gz"
    #fpath_cuts_centralUl17_test = "histos/plotsTopEFT_centralUl17_ttH-ttll.pkl.gz"
    fpath_cuts_centralUl17_test = "histos/plotsTopEFT_centralUL17_allButtHq4t.pkl.gz"

    #hin_dict = get_hist_from_pkl(fpath_default)
    hin_dict = get_hist_from_pkl(fpath_cuts_centralUl17_test)
    hin_dict_private = get_hist_from_pkl(fpath_cuts_privateUl17_test)

    ylds_central_dict = get_yld_dict(hin_dict,"2017")
    ylds_private_dict = get_yld_dict(hin_dict_private,"2017")

    # Get percent differenes
    pdiff_dict = get_pdiff_between_nested_dicts(ylds_private_dict,ylds_central_dict)

    # Print out yields and percent differences
    print_yld_dicts(ylds_central_dict,"Central UL17 yields")
    print_yld_dicts(ylds_private_dict,"Private UL17 yields")
    print_yld_dicts(pdiff_dict,"Percent diff between private and central")

    # Print out info about the hists
    #print_hist_info(hin_dict)
    #exit()

    # Print latex table
    #print_latex_yield_table("2017",hin_dict,hin_dict_private)
    #print_latex_yield_table("2017",hin_dict)


main()

