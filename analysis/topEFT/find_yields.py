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

PROC_MAP_CENTRAL = {
    "ttlnu" : ["ttW_centralUL17"],
    "ttll"  : ["ttZ_centralUL17"],
    "ttH"   : ["ttH_centralUL17"],
    "tllq"  : ["tZq_centralUL17"],
}

PROC_MAP_PRIVATE = {
    "ttlnu" : ["ttlnuJet_privateUL17"],
    "ttll"  : ["ttllJet_privateUL17"],
    "ttH"   : ["ttHJet_privateUL17"],
    "tllq"  : ["tllq_privateUL17"],
    "tHq"   : ["tHq_privateUL17"],
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

def pdiff(a,b):
    #p = (float(a)-float(b))/((float(a)+float(b))/2)
    if b == 0:
        return None
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

# Takes a hist (from the pkl file) and an axis name, retrun the list of categories for that axis
def get_cat_lables(hin_dict,axis,h_name=None):
    cats = []
    if h_name is None: h_name = "njets" # Guess a hist that we usually have
    for i in range(len(hin_dict[h_name].axes())):
        if axis == hin_dict[h_name].axes()[i].name:
            for cat in hin_dict[h_name].axes()[i].identifiers():
                cats.append(cat.name)
    return cats

# Takes a histogram and a dictionary, integrates out the categories listed in the dictionry
def integrate_out_cats(h,cuts_dict):
    h_ret = h
    for axis_name,cat_lst in cuts_dict.items():
        h_ret = h_ret.integrate(axis_name,cat_lst)
    return h_ret

# Sum all the values of a hist 
#    - The hist you pass should have two axes (all other should already be integrated out)
#    - The two axes should be the samples axis, and the dense axis (e.g. ht)
#    - You pass a process name, and we select just that category from the sample axis
def get_yield(h,proc,cat_lst=None):
    #print("\nIn get_yields()")
    if cat_lst is not None:
        h = h.integrate(axis_name,cats_lst)
        print("I don't think we should be here !!!")
        raise Exception
    #h_vals = h[proc].values(overflow='all')
    h_vals = h[proc].values(overflow='over')
    #h_vals = h[proc].values()
    #print("proc:",proc)
    #print("\th_vals:",h_vals)
    for i,(k,v) in enumerate(h_vals.items()):
        v_sum = v.sum()
        if i > 0: raise Exception("Why is i greater than 0? Something is not what you thought it was.")
    #print("\tSum of vals:",v_sum)
    return v_sum

# Print out all the info about all the axes in a hist
def check_hist_info(path):

    if type(path) is str:
        hin_dict = get_hist_from_pkl(path)
    else:
        hin_dict = path

    h_name = "njets"
    #h_name = "SumOfEFTweights"
    for i in range(len(hin_dict[h_name].axes())):
        print(f"\n{i} Aaxis name:",hin_dict[h_name].axes()[i].name)
        for cat in hin_dict[h_name].axes()[i].identifiers():
            print(cat)

# Really just a wrapper for get_yield()
def get_scaled_yield(hin_dict,year,proc,cat):

    #h = hin_dict["ht"]
    h = hin_dict["njets"]

    h = integrate_out_cats(h,CATEGORIES[cat])
    h = h.integrate("cut","base").integrate("systematic","nominal")

    #print("pre:",h[proc].values())
    if '2l' in cat:
        h = h.rebin('njets', hist.Bin("njets",  "Jet multiplicity ", [4,5,6,7,8,9,10]))
    elif '3l' in cat:
        h = h.rebin('njets', hist.Bin("njets",  "Jet multiplicity ", [2,3,4,5,6,7,8,9,10]))
    elif '4l' in cat: 
        h = h.rebin('njets', hist.Bin("njets",  "Jet multiplicity ", [2,3,4,5,6,7,8,9,10]))
    #print("post:",h[proc].values())

    lumi = 1000.0*get_lumi(year)
    h_sow = hin_dict["SumOfEFTweights"]
    nwc = h_sow._nwc

    if nwc > 0:
        sow = get_yield(h_sow,proc)
        h.scale(1.0/sow) # Divide EFT samples by sum of weights at SM
        #print("Num of WCs:",nwc)
        #print("Sum of weights:",sow)

    yld = lumi*get_yield(h,proc)
    #print("\tN events:",yld)
    #print("\t",yld)
    return yld

# Make a latex table for the yields
def make_latex_yield_table(year,hin_dict1,hin_dict2=None):

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

def get_yld_dict(hin_dict,year):
    yld_dict = {}
    proc_lst = get_cat_lables(hin_dict,"sample")
    for proc in proc_lst:
        yld_dict[proc] = {}
        for cat,cuts_dict in CATEGORIES.items():
            yld = get_scaled_yield(hin_dict,year,proc,cat)
            yld_dict[proc][cat]= yld
    return yld_dict


def main():

    fpath_default  = "histos/plotsTopEFT.pkl.gz"
    #fpath_cuts_privateUl17_test = "histos/plotsTopEFT_privateUL17_ttH-ttll-b1.pkl.gz"
    fpath_cuts_privateUl17_test = "histos/plotsTopEFT_privateUL17_allBut4t_b1.pkl.gz"
    #fpath_cuts_centralUl17_test = "histos/plotsTopEFT_centralUl17_ttH-ttll.pkl.gz"
    fpath_cuts_centralUl17_test = "histos/plotsTopEFT_centralUL17_allButtHq4t.pkl.gz"

    hin_dict = get_hist_from_pkl(fpath_default)
    #hin_dict = get_hist_from_pkl(fpath_cuts_centralUl17_test)
    #hin_dict_private = get_hist_from_pkl(fpath_cuts_privateUl17_test)

    ###
    # From Brent:
    #hists = hin_dict_private
    #ch2lss = ['eeSSonZ', 'eeSSoffZ', 'mmSSonZ', 'mmSSoffZ', 'emSS']
    #x = 1000 * 41.3 * hists['njets'].rebin('njets', hist.Bin("njets",  "Jet multiplicity ", [4,5,6,7])).integrate('channel', ch2lss).integrate('cut', 'base').integrate('sumcharge', 'ch+').integrate('nbjet', '1+bm2+bl').integrate('systematic', 'nominal').integrate('sample', 'ttHJet_privateUL17').values(overflow='over')[()].sum() / hists['SumOfEFTweights'].sum('sample').values()[()].sum()
    #print("Value:",x)
    #exit()
    ###


    # Print out info about the hists
    check_hist_info(hin_dict)
    exit()

    proc_lst = get_cat_lables(hin_dict,"sample")

    '''
    for proc in proc_lst:
        print(f"\n{proc}")
        for cat,cuts_dict in CATEGORIES.items():
            yld = get_scaled_yield(hin_dict,"2017",proc,cat)
            print("\t",cat,":",yld)
    '''

    #make_latex_yield_table("2017",hin_dict,hin_dict_private)
    #make_latex_yield_table("2017",hin_dict)

    ylds_central_dict = get_yld_dict(hin_dict,"2017")
    ylds_private_dict = get_yld_dict(hin_dict_private,"2017")

    for ptag in list(PROC_MAP_CENTRAL.keys()):
        print("\n",ptag)
        for cat_name in CAT_LST:
            print("\n",cat_name)
            yld_central = ylds_central_dict[PROC_MAP_CENTRAL[ptag][0]][cat_name]
            yld_private = ylds_private_dict[PROC_MAP_PRIVATE[ptag][0]][cat_name]
            print("\t",yld_private,yld_central)
            p = pdiff(yld_private,yld_central)
            print("\t",p)

main()
