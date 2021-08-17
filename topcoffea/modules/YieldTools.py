import gzip
import json
import pickle
import numpy as np
from coffea import hist
#from topcoffea.modules.HistEFT import HistEFT
from topcoffea.modules.paths import topcoffea_path
from topcoffea.modules.GetValuesFromJsons import get_lumi

class YieldTools():

    def __init__(self):

        # The order of the categories in the TOP-19-001 AN yield tables
        self.CAT_LST = ["cat_2lss_p", "cat_2lss_m", "cat_3l_p_offZ_1b", "cat_3l_m_offZ_1b", "cat_3l_p_offZ_2b", "cat_3l_m_offZ_2b", "cat_3l_onZ_1b", "cat_3l_onZ_2b", "cat_4l"]

        # A dictionary mapping names of samples in the samples axis to a short version of the name
        self.PROC_MAP = {
            "ttlnu" : ["ttW_centralUL17" , "ttlnuJet_privateUL18" , "ttlnuJet_privateUL17" , "ttlnuJet_privateUL16" , "ttlnuJet_privateUL16APV"],
            "ttll"  : ["ttZ_centralUL17" , "ttllJet_privateUL18"  , "ttllJet_privateUL17"  , "ttllJet_privateUL16"  , "ttllJet_privateUL16APV"],
            "ttH"   : ["ttH_centralUL17" , "ttHJet_privateUL18"   , "ttHJet_privateUL17"   , "ttHJet_privateUL16"   , "ttHJet_privateUL16APV"],
            "tllq"  : ["tZq_centralUL17" , "tllq_privateUL18"     , "tllq_privateUL17"     , "tllq_privateUL16"     , "tllq_privateUL16APV"],
            "tHq"   : ["tHq_central2017" , "tHq_privateUL18"      , "tHq_privateUL17"      , "tHq_privateUL16"      , "tHq_privateUL16APV"],
            "tttt"  : ["tttt_central2017", "tttt_privateUL18"     , "tttt_privateUL17"     , "tttt_privateUL16"     , "tttt_privateUL16APV"],
        }

        # The jet bins we define for the lep categories (Not currently used)
        self.JET_BINS = {
            "2lss" : [4,5,6,7],
            "3l"   : [2,3,4,5],
            "4l"   : [2,3,4],
        }


        # The ratios we can scale our yields by to see if we get agreemnet with TOP-19-001
        e_over_mu_old = 0.75
        e_over_mu_new = 0.56
        flav_factor_dict = {
            "ee"  : (e_over_mu_old**(-2))*(e_over_mu_new**(2)),
            "em"  : (e_over_mu_old**(-1))*(e_over_mu_new**(1)),
            "mm"  : (e_over_mu_old**(-0))*(e_over_mu_new**(0)),
            "eee" : (e_over_mu_old**(-3))*(e_over_mu_new**(3)),
            "eem" : (e_over_mu_old**(-2))*(e_over_mu_new**(2)),
            "emm" : (e_over_mu_old**(-1))*(e_over_mu_new**(1)),
            "mmm" : (e_over_mu_old**(-0))*(e_over_mu_new**(0)),
        }

        # A dictionary specifying which categories from the hists create the analysis categories
        self.CATEGORIES = {
            "cat_2lss_p" : {
                "channel": ["2lss_p_4j","2lss_p_5j","2lss_p_6j","2lss_p_7j"],
                "appl": ["isSR_2l"],
            },
            "cat_2lss_m" : {
                "channel": ["2lss_m_4j","2lss_m_5j","2lss_m_6j","2lss_m_7j"],
                "appl": ["isSR_2l"],
            },
            "cat_3l_p_offZ_1b" : {
                "channel": ["3l_p_offZ_1b_2j", "3l_p_offZ_1b_3j", "3l_p_offZ_1b_4j", "3l_p_offZ_1b_5j"],
                "appl": ["isSR_3l"],
            },
            "cat_3l_m_offZ_1b" : {
                "channel": ["3l_m_offZ_1b_2j", "3l_m_offZ_1b_3j", "3l_m_offZ_1b_4j", "3l_m_offZ_1b_5j"],
                "appl": ["isSR_3l"],
            },
            "cat_3l_p_offZ_2b" : {
                "channel": ["3l_p_offZ_2b_2j", "3l_p_offZ_2b_3j", "3l_p_offZ_2b_4j", "3l_p_offZ_2b_5j"],
                "appl": ["isSR_3l"],
            },
            "cat_3l_m_offZ_2b" : {
                "channel": ["3l_m_offZ_2b_2j", "3l_m_offZ_2b_3j", "3l_m_offZ_2b_4j", "3l_m_offZ_2b_5j"],
                "appl": ["isSR_3l"],
            },
            "cat_3l_onZ_1b" : {
                "channel": ["3l_onZ_1b_2j", "3l_onZ_1b_3j", "3l_onZ_1b_4j", "3l_onZ_1b_5j"],
                "appl": ["isSR_3l"],
            },
            "cat_3l_onZ_2b" : {
                "channel": ["3l_onZ_2b_2j", "3l_onZ_2b_3j", "3l_onZ_2b_4j", "3l_onZ_2b_5j"],
                "appl": ["isSR_3l"],
            },
            "cat_4l" : {
                "channel": ["4l_2j", "4l_3j", "4l_4j"],
                "appl": ["isSR_4l"],
            },
        }


        # Yields from TOP-19-001 AN table 15
        self.TOP19001_YLDS = {
            "ttlnu" : {
                "cat_2lss_p" : (68.7,None),
                "cat_2lss_m" : (37.1,None),
                "cat_3l_p_offZ_1b" : (14.4,None),
                "cat_3l_m_offZ_1b" : (8.0,None),
                "cat_3l_p_offZ_2b" : (10.8,None),
                "cat_3l_m_offZ_2b" : (5.9,None),
                "cat_3l_onZ_1b" : (2.9,None),
                "cat_3l_onZ_2b" : (2.3,None),
                "cat_4l" : (0.0,None),
            },
            "ttll"  : {
                "cat_2lss_p" : (19.3,None),
                "cat_2lss_m" : (19.0,None),
                "cat_3l_p_offZ_1b" : (12.7,None),
                "cat_3l_m_offZ_1b" : (13.3,None),
                "cat_3l_p_offZ_2b" : (9.1,None),
                "cat_3l_m_offZ_2b" : (8.5,None),
                "cat_3l_onZ_1b" : (95.5,None),
                "cat_3l_onZ_2b" : (63.2,None),
                "cat_4l" : (9.4,None),
            },
            "ttH"   : {
                "cat_2lss_p" : (24.7,None),
                "cat_2lss_m" : (24.1,None),
                "cat_3l_p_offZ_1b" : (7.9,None),
                "cat_3l_m_offZ_1b" : (7.6,None),
                "cat_3l_p_offZ_2b" : (5.1,None),
                "cat_3l_m_offZ_2b" : (5.2,None),
                "cat_3l_onZ_1b" : (3.2,None),
                "cat_3l_onZ_2b" : (2.2,None),
                "cat_4l" : (1.0,None),
            },
            "tllq"  : {
                "cat_2lss_p" : (2.7,None),
                "cat_2lss_m" : (1.5,None),
                "cat_3l_p_offZ_1b" : (3.5,None),
                "cat_3l_m_offZ_1b" : (1.8,None),
                "cat_3l_p_offZ_2b" : (1.2,None),
                "cat_3l_m_offZ_2b" : (0.6,None),
                "cat_3l_onZ_1b" : (39.8,None),
                "cat_3l_onZ_2b" : (13.3,None),
                "cat_4l" : (0.0,None),
            },
            "tHq"   : {
                "cat_2lss_p" : (0.8,None),
                "cat_2lss_m" : (0.4,None),
                "cat_3l_p_offZ_1b" : (0.3,None),
                "cat_3l_m_offZ_1b" : (0.2,None),
                "cat_3l_p_offZ_2b" : (0.2,None),
                "cat_3l_m_offZ_2b" : (0.1,None),
                "cat_3l_onZ_1b" : (0.1,None),
                "cat_3l_onZ_2b" : (0.1,None),
                "cat_4l" : (0.0,None),
            },
        }


    ######### Functions for getting process names from PROC_MAP #########

    # What this function does:
    #   - Takes a full process name (i.e. the name of the category on the samples axis)
    #   - Then loops through PROC_MAP and returns the short (i.e. standard) version of the process name
    def get_short_name(self,long_name):
        ret_name = None
        for short_name,long_name_lst in self.PROC_MAP.items():
            if long_name in long_name_lst:
                ret_name = short_name
                break
        return ret_name

    # What this function does:
    #   - Takes a list of full process names (i.e. all of the categories on samples axis) and a key from PROC_MAP
    #   - Returns the long (i.e. the name of the category in the smples axis) corresponding to the short name
    def get_long_name(self,long_name_lst_in,short_name_in):
        ret_name = None
        for long_name in PROC_MAP[short_name_in]:
            for long_name_in in long_name_lst_in:
                if long_name_in == long_name:
                    ret_name = long_name
        return ret_name


    ######### General functions #########

    # Get percent difference
    def get_pdiff(self,a,b):
        #p = (float(a)-float(b))/((float(a)+float(b))/2)
        if ((a is None) or (b is None)):
            p = None
        elif b == 0:
            p = None
        else:
            p = (float(a)-float(b))/float(b)
        return p


    # Get the dictionary of hists from the pkl file (that the processor outputs)
    def get_hist_from_pkl(self,path_to_pkl):
        h = pickle.load( gzip.open(path_to_pkl) )
        return h


    # Takes a hist dictionary (i.e. from the pkl file that the processor makes) and an axis name, returns the list of categories for that axis. Defaults to 'njets' histogram if none given.
    def get_cat_lables(self,hin_dict,axis,h_name=None):
        cats = []
        #if h_name is None: h_name = "njets" # Guess a hist that we usually have
        if h_name is None: h_name = "ht" # Guess a hist that we usually have
        for i in range(len(hin_dict[h_name].axes())):
            if axis == hin_dict[h_name].axes()[i].name:
                for cat in hin_dict[h_name].axes()[i].identifiers():
                    cats.append(cat.name)
        return cats


    # Takes a histogram and a dictionary that specifies categories, integrates out the categories listed in the dictionry
    def integrate_out_cats(self,h,cuts_dict):
        h_ret = h
        for axis_name,cat_lst in cuts_dict.items():
            h_ret = h_ret.integrate(axis_name,cat_lst)
        return h_ret


    # Takes a histogram and a bin, rebins the njets axis to consist of only that bin (all other bins combined into under/overvlow)
    def select_njet_bin(self,h,bin_val):
        if not isinstance(bin_val,int):
            raise Exception(f"Need to pass an int to this function, got a {type(bin_val)} instead. Exiting...")
        h = h.rebin('njets', hist.Bin("njets",  "Jet multiplicity ", [bin_val,bin_val+1]))
        return h


    # Get the difference between values in nested dictionary, currently can get either percent diff, or absolute diff
    # Returns a dictionary in the same format (currently does not propagate errors, just returns None)
    #   dict = {
    #       k : {
    #           subk : (val,err)
    #       }
    #   }
    def get_diff_between_nested_dicts(self,dict1,dict2,difftype):

        ret_dict = {}
        for k1 in dict1.keys():

            if k1 in dict2.keys():
                ret_dict[k1] = {}
                for subk1 in dict1[k1].keys():
                    if subk1 not in dict2[k1].keys():
                        raise Exception("These dictionaries do not have the same structure. Exiting...")
                    v1,e1 = dict1[k1][subk1]
                    v2,e1 = dict2[k1][subk1]
                    if difftype == "percent_diff":
                        ret_diff = self.get_pdiff(v1,v2)
                    elif difftype == "absolute_diff":
                        ret_diff = v1 - v2
                    else:
                        raise Exception(f"Unknown diff type: {difftype}. Exiting...")

                    ret_dict[k1][subk1] = (ret_diff,None)
            else:
                print(f"WARNING, key {k1} is not in both dictionaries. Continuing...")
                continue

        return ret_dict


    ######### Functions specifically for getting yields #########

    # Sum all the values of a hist 
    #    - The hist you pass should have two axes (all other should already be integrated out)
    #    - The two axes should be the samples axis, and the dense axis (e.g. ht)
    #    - You pass a process name, and we select just that category from the sample axis
    def get_yield(self,h,proc,overflow_str="none"):
        h_vals = h[proc].values(sumw2=True,overflow=overflow_str)
        for i,(k,v) in enumerate(h_vals.items()):
            v_sum = v[0].sum()
            e_sum = v[1].sum()
            if i > 0: raise Exception("Why is i greater than 0? The hist is not what this function is expecting. Exiting...")
        e_sum = np.sqrt(e_sum)
        return (v_sum,e_sum)


    # Integrates out categories, normalizes, then calls get_yield()
    def get_scaled_yield(self,hin_dict,year,proc,cat,overflow_str,rwgt_pt=None):

        cat_dict = self.CATEGORIES[cat]

        # Integrate out cateogries
        #h = hin_dict["njets"]
        h = hin_dict["ht"]
        h = self.integrate_out_cats(h,cat_dict)
        h = h.integrate("systematic","nominal") # For now anyway...

        # Integrate over the categories for the analysis category we are interested in
        #h = self.select_hist_for_ana_cat(h,self.CATEGORIES[cat])
        #cats_dict = {}
        #cats_dict["channel"] = self.CATEGORIES[cat]["channel"][0][:-3]
        #cats_dict["appl"] = self.CATEGORIES[cat]["appl"]
        #h = self.select_hist_for_ana_cat(h,cats_dict)

        # Reweight the hist
        if rwgt_pt is not None:
            hist.set_wilson_coefficients(**wc_vals)
        else:
            h.set_sm()

        # Scale by sum of weights (if EFT hist)
        h_sow = hin_dict["SumOfEFTweights"]
        nwc = h_sow._nwc
        if nwc > 0:
            h.set_sm()
            sow_val , sow_err = self.get_yield(h_sow,proc)
            h.scale(1.0/sow_val) # Divide EFT samples by sum of weights at SM, ignore error propagation for now

        # Scale by lumi
        lumi = 1000.0*get_lumi(year)
        h.scale(lumi)

        return self.get_yield(h,proc,overflow_str)


    # This function:
    #   - Takes as input a hist dict (i.e. what the processor outptus)
    #   - Returns a dictionary of yields for the categories in CATEGORIES
    #   - Making use of get_scaled_yield()
    #   - If you pass a key from JET_BINS for yields_for_njets_cats, will make a dict of yields for all the jet bins in that lep cat
    def get_yld_dict(self,hin_dict,year,yields_for_njets_cats=None):

        yld_dict = {}
        proc_lst = self.get_cat_lables(hin_dict,"sample")
        for proc in proc_lst:
            proc_name_short = self.get_short_name(proc)
            yld_dict[proc_name_short] = {}
            for cat,cuts_dict in self.CATEGORIES.items():
                if "2lss" in cat: njet_cat = "2lss"
                elif "3l" in cat: njet_cat = "3l"
                elif "4l" in cat: njet_cat = "4l"

                # We want to sum over the jet bins, and look at all of the lep cats
                if yields_for_njets_cats is None:
                    yld_dict[proc_name_short][cat] = self.get_scaled_yield(hin_dict,year,proc,cat,overflow_str="over") # Important to keep overflow

                else:
                    raise Exception(f"Error, invalid input for yields_for_njets_cats \"{yields_for_njets_cats}\". Exiting...")

        return yld_dict

    # This function:
    #    - Takes as input a yld dict
    #    - Sums ylds over processes to find total for each cat
    #    - Rreturns a dict with the same structure, where the vals are the cat's fractional contribution to the total
    def find_relative_contributions(self,yld_dict):

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


    ######### Functions that just print out information #########

    # Print out all the info about all the axes in a hist
    def print_hist_info(self,path):

        if type(path) is str: hin_dict = self.get_hist_from_pkl(path)
        else: hin_dict = path

        #h_name = "ht"
        h_name = "njets"
        #h_name = "SumOfEFTweights"
        for i in range(len(hin_dict[h_name].axes())):
            print(f"\n{i} Aaxis name:",hin_dict[h_name].axes()[i].name)
            for cat in hin_dict[h_name].axes()[i].identifiers():
                print(f"\t{cat}")


    # Takes yield dicts (i.e. what get_yld_dict() returns) and prints it
    # Note:
    #   - This function also now optionally takes a tolerance value
    #   - Checks if the differences are larger than that value
    #   - Returns False if any of the values are too large
    #   - Should a different function handle this stuff?
    def print_yld_dicts(self,ylds_dict,tag,show_errs=False,tolerance=None):
        ret = True
        print(f"\n--- {tag} ---\n")
        for proc in ylds_dict.keys():
            print(proc)
            for cat in ylds_dict[proc].keys():
                print(f"    {cat}")
                val , err = ylds_dict[proc][cat]

                # We don't want to check if the val is small
                if tolerance is None:
                    if show_errs:
                        #print(f"\t{val} +- {err}")
                        print(f"\t{val} +- {err} -> {err/val}")
                    else:
                        print(f"\t{val}")

                # We want to check if the val is small
                else:
                    if (val is None) or (abs(val) < abs(tolerance)):
                        print(f"\t{val}")
                    else:
                        print(f"\t{val} -> NOTE: This is larger than tolerance ({tolerance})!")
                        ret = False
        return ret
