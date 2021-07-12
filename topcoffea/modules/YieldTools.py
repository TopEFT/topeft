import gzip
import json
import pickle
import numpy as np
from coffea import hist
#from topcoffea.modules.HistEFT import HistEFT
from topcoffea.modules.paths import topcoffea_path

class YieldTools():

    def __init__(self):

        self.CAT_LST = ["cat_2lss_p","cat_2lss_m","cat_3l_1b_offZ_p","cat_3l_1b_offZ_m","cat_3l_2b_offZ_p","cat_3l_2b_offZ_m","cat_3l_1b_onZ","cat_3l_2b_onZ","cat_4l"]

        self.PROC_MAP = {
            "ttlnu" : ["ttW_centralUL17","ttlnu_private2017","ttlnuJet_privateUL17","ttlnuJet_privateUL18"],
            "ttll"  : ["ttZ_centralUL17","ttll_TOP-19-001","ttllJet_privateUL17","ttllJet_privateUL18"],
            "ttH"   : ["ttH_centralUL17","ttH_private2017","ttHJet_privateUL17","ttHJet_privateUL18"],
            "tllq"  : ["tZq_centralUL17","tllq_private2017","tllq_privateUL17","tllq_privateUL18"],
            "tHq"   : ["tHq_central2017","tHq_privateUL17"],
            "tttt"  : ["tttt_central2017","tttt_privateUL17"],
        }

        self.JET_BINS = {
            "2lss" : [4,5,6,7],
            "3l"   : [2,3,4,5],
            "4l"   : [2,3,4],
        }

        self.ch_3l_onZ = ["eemSSonZ", "mmeSSonZ", "eeeSSonZ", "mmmSSonZ"]
        self.ch_3l_offZ = ["eemSSoffZ", "mmeSSoffZ", "eeeSSoffZ", "mmmSSoffZ"]
        self.ch_2lss = ["eeSSonZ", "eeSSoffZ", "mmSSonZ", "mmSSoffZ", "emSS"]
        self.ch_4l = ["eeee","eeem","eemm","mmme","mmmm"]

        self.CATEGORIES = {
            "cat_2lss_p" : {
                "channel": self.ch_2lss,
                "sumcharge": ["ch+"],
                "cut": ["1+bm2+bl"],
            },
            "cat_2lss_m" : {
                "channel": self.ch_2lss,
                "sumcharge": ["ch-"],
                "cut": ["1+bm2+bl"],
            },

            "cat_3l_1b_onZ" : {
                "channel": self.ch_3l_onZ,
                "sumcharge": ["ch+","ch-"],
                "cut": ["1bm"],
            },
            "cat_3l_1b_offZ_p" : {
                "channel": self.ch_3l_offZ,
                "sumcharge": ["ch+"],
                "cut": ["1bm"],
            },
            "cat_3l_1b_offZ_m" : {
                "channel": self.ch_3l_offZ,
                "sumcharge": ["ch-"],
                "cut": ["1bm"],
            },

            "cat_3l_2b_onZ" : {
                "channel": self.ch_3l_onZ,
                "sumcharge": ["ch+","ch-"],
                "cut": ["2+bm"],
            },
            "cat_3l_2b_offZ_p" : {
                "channel": self.ch_3l_offZ,
                "sumcharge": ["ch+"],
                "cut": ["2+bm"],
            },
            "cat_3l_2b_offZ_m" : {
                "channel": self.ch_3l_offZ,
                "sumcharge": ["ch-"],
                "cut": ["2+bm"],
            },

            "cat_4l" : {
                "channel": self.ch_4l,
                "sumcharge": ["ch+","ch-","ch0"],
                "cut": ["1+bm2+bl"],
            },
        }

        self.TOP19001_YLDS = {
            "ttlnu" : {
                "cat_2lss_p" : (81.1,None),
                "cat_2lss_m" : (44.0,None),
                "cat_3l_1b_offZ_p" : (16.6,None),
                "cat_3l_1b_offZ_m" : (9.1,None),
                "cat_3l_2b_offZ_p" : (12.1,None),
                "cat_3l_2b_offZ_m" : (6.7,None),
                "cat_3l_1b_onZ" : (3.4,None),
                "cat_3l_2b_onZ" : (2.5,None),
                "cat_4l" : (0.0,None),
            },
            "ttll"  : {
                "cat_2lss_p" : (22.6,None),
                "cat_2lss_m" : (22.5,None),
                "cat_3l_1b_offZ_p" : (14.2,None),
                "cat_3l_1b_offZ_m" : (14.7,None),
                "cat_3l_2b_offZ_p" : (10.1,None),
                "cat_3l_2b_offZ_m" : (9.4,None),
                "cat_3l_1b_onZ" : (106.5,None),
                "cat_3l_2b_onZ" : (70.9,None),
                "cat_4l" : (10.4,None),
            },
            "ttH"   : {
                "cat_2lss_p" : (28.6,None),
                "cat_2lss_m" : (27.9,None),
                "cat_3l_1b_offZ_p" : (8.5,None),
                "cat_3l_1b_offZ_m" : (8.1,None),
                "cat_3l_2b_offZ_p" : (5.5,None),
                "cat_3l_2b_offZ_m" : (5.6,None),
                "cat_3l_1b_onZ" : (3.5,None),
                "cat_3l_2b_onZ" : (2.4,None),
                "cat_4l" : (1.1,None),
            },
            "tllq"  : {
                "cat_2lss_p" : (2.9,None),
                "cat_2lss_m" : (1.7,None),
                "cat_3l_1b_offZ_p" : (3.8,None),
                "cat_3l_1b_offZ_m" : (1.9,None),
                "cat_3l_2b_offZ_p" : (1.3,None),
                "cat_3l_2b_offZ_m" : (0.6,None),
                "cat_3l_1b_onZ" : (42.1,None),
                "cat_3l_2b_onZ" : (14.1,None),
                "cat_4l" : (0.0,None),
            },
            "tHq"   : {
                "cat_2lss_p" : (0.9,None),
                "cat_2lss_m" : (0.5,None),
                "cat_3l_1b_offZ_p" : (0.3,None),
                "cat_3l_1b_offZ_m" : (0.2,None),
                "cat_3l_2b_offZ_p" : (0.2,None),
                "cat_3l_2b_offZ_m" : (0.1,None),
                "cat_3l_1b_onZ" : (0.1,None),
                "cat_3l_2b_onZ" : (0.1,None),
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
    #   - Returns the long (i.e. the name of the cateogry in the smples axis) corresponding to the short name
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

    # Return the lumi from the json/lumi.json file for a given year
    def get_lumi(self,year):
        lumi_json = topcoffea_path("json/lumi.json")
        with open(lumi_json) as f_lumi:
           lumi = json.load(f_lumi)
           lumi = lumi[year]
        return lumi

    # Takes a hist dictionary (i.e. from the pkl file that the processor makes) and an axis name, retrun the list of categories for that axis
    def get_cat_lables(self,hin_dict,axis,h_name=None):
        cats = []
        if h_name is None: h_name = "njets" # Guess a hist that we usually have
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
    # Returns a dictionary in the same formate (cuttently does not propagate errors, just returns None)
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

    # Uses integrate_out_cats() and select_njet_bin() to return h for a particular analysis cateogry
    #def get_h_for_cat(h,cat_dict,njet):
    def select_hist_for_ana_cat(self,h,cat_dict,njet):
        h_ret = self.integrate_out_cats(h,cat_dict)
        h_ret = h_ret.integrate("systematic","nominal") # For now anyway...
        h_ret = self.select_njet_bin(h_ret,njet)
        return h_ret

    # This is really just a wrapper for get_yield(). Note:
    #   - This fucntion now also rebins the njets hists
    #   - Maybe that does not belong in this function
    def get_scaled_yield(self,hin_dict,year,proc,cat,njets_cat,overflow_str):

        h = hin_dict["njets"]

        if isinstance(njets_cat,str):
            njet = self.JET_BINS[njets_cat][0]
        elif isinstance(njets_cat,int):
            njet = njets_cat
        else:
            raise Exception(f"Wrong type, expected str or int, but got a {type(njets_cat)}, exiting...")

        h = self.select_hist_for_ana_cat(h,self.CATEGORIES[cat],njet)

        '''
        # Reweight h (TODO: This part will need to be cleaned up.. should we pass the wc point to  this function?)
        if h._nwc == 22:
            #print("22 WCs")
            wc_vals = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22])
        if h._nwc == 26:
            if h._wcnames[-1] == "cQQ1":
                #print("26 WCs, sample came from mix-and-match-wc branch branch")
                wc_vals = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,0,0,0,0]) # If comparing against a 22 WC sample
                #wc_vals = np.array([1,2,4,22,21,6,7,9,12,13,18,15,17,14,24,20,25,26,8,11,10,5,3,16,19,23]) # To match the tttt WCs order on master branch
            if h._wcnames[-1] == "ctW":
                #print("26 WCs, sample came from master branch")
                wc_vals = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26])
                #wc_vals = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,0,0,0,0]) # If comparing against a 22 WC sample
        #h.set_wilson_coefficients(wc_vals)
        '''
        h.set_sm()

        lumi = 1000.0*self.get_lumi(year)
        h_sow = hin_dict["SumOfEFTweights"]
        nwc = h_sow._nwc

        if nwc > 0:
            #h_sow.set_wilson_coefficients([0]*h._nwc)
            h.set_sm()
            sow_val , sow_err = self.get_yield(h_sow,proc)
            h.scale(1.0/sow_val) # Divide EFT samples by sum of weights at SM, ignore error propagation for now
            #print("sow_val,sow_err",sow_val,sow_err,"->",sow_err/sow_val)
            #print("Num of WCs:",nwc)
            #print("Sum of weights:",sow)

        h.scale(lumi)
        return self.get_yield(h,proc,overflow_str)

    # This function:
    #   - Takes as input a hist dict (i.e. what the processor outptus)
    #   - Retruns a dictionary of yields for the categories in CATEGORIES
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
                    yld_dict[proc_name_short][cat] = self.get_scaled_yield(hin_dict,year,proc,cat,njet_cat,overflow_str="over")

                # We want to look at all the jet bins in the give lep cat
                elif yields_for_njets_cats == njet_cat:
                    for njet in self.JET_BINS[njet_cat]:
                        if njet == max(self.JET_BINS[njet_cat]): include_overflow = "over"
                        else: include_overflow = "none"
                        cat_name_full = cat+"_"+str(njet)+"j"
                        yld_dict[proc_name_short][cat_name_full] = self.get_scaled_yield(hin_dict,year,proc,cat,njet,overflow_str=include_overflow)

        return yld_dict


    ######### Functions that just print out information #########

    # Print out all the info about all the axes in a hist
    def print_hist_info(self,path):

        if type(path) is str: hin_dict = self.get_hist_from_pkl(path)
        else: hin_dict = path

        h_name = "njets"
        #h_name = "SumOfEFTweights"
        for i in range(len(hin_dict[h_name].axes())):
            print(f"\n{i} Aaxis name:",hin_dict[h_name].axes()[i].name)
            for cat in hin_dict[h_name].axes()[i].identifiers():
                print(cat)



    # Takes yield dicts (i.e. what get_yld_dict() returns) and prints it
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
                    if abs(val) < abs(tolerance): # If these are differences between yields, coudl be negative
                        print(f"\t{val}")
                    else:
                        print(f"\t{val} -> NOTE: This is larger than tolerance ({tolerance})!")
                        ret = False
        return ret

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
