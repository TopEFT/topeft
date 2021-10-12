import gzip
import json
import pickle
import numpy as np
import copy
import coffea
from coffea import hist
#from topcoffea.modules.HistEFT import HistEFT
from topcoffea.modules.paths import topcoffea_path
from topcoffea.modules.GetValuesFromJsons import get_lumi

class YieldTools():

    def __init__(self):

        # The order of the categories in the TOP-19-001 AN yield tables
        self.CAT_LST = ["2lss_p", "2lss_m", "3l_p_offZ_1b", "3l_m_offZ_1b", "3l_p_offZ_2b", "3l_m_offZ_2b", "3l_onZ_1b", "3l_onZ_2b", "4l"]

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

        self.APPL_DICT = {
            "2lss" : "isSR_2lSS",
            "2los" : "isSR_2lOS",
            "3l"   : "isSR_3l",
            "4l"   : "isSR_4l",
        }

        # Yields from TOP-19-001 AN table 15
        self.TOP19001_YLDS = {
            "ttlnu" : {
                "2lss_p" : (68.7,None),
                "2lss_m" : (37.1,None),
                "3l_p_offZ_1b" : (14.4,None),
                "3l_m_offZ_1b" : (8.0,None),
                "3l_p_offZ_2b" : (10.8,None),
                "3l_m_offZ_2b" : (5.9,None),
                "3l_onZ_1b" : (2.9,None),
                "3l_onZ_2b" : (2.3,None),
                "4l" : (0.0,None),
            },
            "ttll"  : {
                "2lss_p" : (19.3,None),
                "2lss_m" : (19.0,None),
                "3l_p_offZ_1b" : (12.7,None),
                "3l_m_offZ_1b" : (13.3,None),
                "3l_p_offZ_2b" : (9.1,None),
                "3l_m_offZ_2b" : (8.5,None),
                "3l_onZ_1b" : (95.5,None),
                "3l_onZ_2b" : (63.2,None),
                "4l" : (9.4,None),
            },
            "ttH"   : {
                "2lss_p" : (24.7,None),
                "2lss_m" : (24.1,None),
                "3l_p_offZ_1b" : (7.9,None),
                "3l_m_offZ_1b" : (7.6,None),
                "3l_p_offZ_2b" : (5.1,None),
                "3l_m_offZ_2b" : (5.2,None),
                "3l_onZ_1b" : (3.2,None),
                "3l_onZ_2b" : (2.2,None),
                "4l" : (1.0,None),
            },
            "tllq"  : {
                "2lss_p" : (2.7,None),
                "2lss_m" : (1.5,None),
                "3l_p_offZ_1b" : (3.5,None),
                "3l_m_offZ_1b" : (1.8,None),
                "3l_p_offZ_2b" : (1.2,None),
                "3l_m_offZ_2b" : (0.6,None),
                "3l_onZ_1b" : (39.8,None),
                "3l_onZ_2b" : (13.3,None),
                "4l" : (0.0,None),
            },
            "tHq"   : {
                "2lss_p" : (0.8,None),
                "2lss_m" : (0.4,None),
                "3l_p_offZ_1b" : (0.3,None),
                "3l_m_offZ_1b" : (0.2,None),
                "3l_p_offZ_2b" : (0.2,None),
                "3l_m_offZ_2b" : (0.1,None),
                "3l_onZ_1b" : (0.1,None),
                "3l_onZ_2b" : (0.1,None),
                "4l" : (0.0,None),
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

    # Takes two dictionaries, returns the list of lists [common keys, keys unique to d1, keys unique to d2]
    def get_common_keys(self,dict1,dict2):

        common_lst = []
        unique_1_lst = []
        unique_2_lst = []

        # Find common keys, and keys unique to d1
        for k1 in dict1.keys():
            if k1 in dict2.keys():
                common_lst.append(k1)
            else:
                unique_1_lst.append(k1)

        # Find keys unique to d2
        for k2 in dict2.keys():
            if k2 not in common_lst:
                unique_2_lst.append(k2)

        return [common_lst,unique_1_lst,unique_2_lst]


    # Get the per lepton e/m factor from e.g. eee and mmm yields
    def get_em_factor(self,e_val,m_val,nlep):
        return (e_val/m_val)**(1.0/nlep)


    # Get the dictionary of hists from the pkl file (that the processor outputs)
    def get_hist_from_pkl(self,path_to_pkl):
        h = pickle.load( gzip.open(path_to_pkl) )
        return h

    # Takes a hist, and retruns a list of the axis names
    def get_axis_list(self,histo):
        axis_lst = []
        for axis in histo.axes():
            axis_lst.append(axis.name)
        return axis_lst


    # Takes a hist dictionary (i.e. from the pkl file that the processor makes) and an axis name, returns the list of categories for that axis. Defaults to 'njets' histogram if none given.
    def get_cat_lables(self,hin_dict,axis,h_name=None):

        #if h_name is None: h_name = "njets" # Guess a hist that we usually have
        if h_name is None: h_name = "ht" # Guess a hist that we usually have

        # Chek if what we have is the output of the processsor, if so, get a specific hist from it
        if isinstance(hin_dict,coffea.processor.accumulator.dict_accumulator):
            hin_dict = hin_dict[h_name]
        elif isinstance(hin_dict,dict):
            hin_dict = hin_dict[h_name]

        # Note: Use h.identifiers('axis') here, not axis.identifiers() (since according to Nick Smith "the axis may hold identifiers longer than the hist that uses it (hists can share axes)", but h.identifiers('axis') will get the ones actually contained in the histogram)
        cats_lst = []
        for identifier in hin_dict.identifiers(axis):
            cats_lst.append(identifier.name)

        return cats_lst


    # Remove the njet component of a category name, returns a new str
    def get_str_without_njet(self,in_str):

        # Check if the substring is an njet substr e.g. "2j"
        def is_jet_str(substr):
            if len(substr) != 2:
                is_a_jet_str = False
            elif not substr[0].isdigit():
                is_a_jet_str = False
            elif not substr[1] == "j":
                is_a_jet_str = False
            else:
                is_a_jet_str = True
            return is_a_jet_str

        # Assumes the str is separated by underscores 
        str_components_lst = in_str.split("_")
        keep_lst = []
        for component in str_components_lst:
            if not is_jet_str(component):
                keep_lst.append(component)
        ret_str  = "_".join(keep_lst)
        return(ret_str)


    # Remove the lepflav component of a category name, returns a new str
    def get_str_without_lepflav(self,in_str):
        # The list of lep flavors we consider (this is a bit hardcoded...)
        lepflav_lst = ["ee","em","mm","eee","eem","emm","mmm"]
        # Assumes the str is separated by underscores 
        str_components_lst = in_str.split("_")
        keep_lst = []
        for component in str_components_lst:
            if not component in lepflav_lst:
                keep_lst.append(component)
        ret_str  = "_".join(keep_lst)
        return(ret_str)


    # This should return true if the hist is split by lep flavor, definitely not a bullet proof check..
    def is_split_by_lepflav(self,hin_dict):
        ch_names_lst = self.get_cat_lables(hin_dict,h_name="ht",axis="channel")
        lep_flav_lst = ["ee","em","mm","eee","eem","emm","mmm"]
        for ch_name in ch_names_lst:
            for lep_flav_name in lep_flav_lst:
                if lep_flav_name in ch_name:
                    return True
        return False


    # Takes a histogram and a dictionary that specifies categories, integrates out the categories listed in the dictionry
    def integrate_out_cats(self,h,cuts_dict):
        h_ret = h.copy()
        for axis_name,cat_lst in cuts_dict.items():
            h_ret = h_ret.integrate(axis_name,cat_lst)
        return h_ret


    # Takes a histogram and a bin, rebins the njets axis to consist of only that bin (all other bins combined into under/overvlow)
    def select_njet_bin(self,h,bin_val):
        if not isinstance(bin_val,int):
            raise Exception(f"Need to pass an int to this function, got a {type(bin_val)} instead. Exiting...")
        h = h.rebin('njets', hist.Bin("njets",  "Jet multiplicity ", [bin_val,bin_val+1]))
        return h


    # Get a dictionary with the sum of weights values the EFT samples need to be scaled by
    def get_eft_sow_scale_dict(self,hin_dict):

        # Chek if what we have is the output of the processsor, if so, get the sow hist from it
        if isinstance(hin_dict,coffea.processor.accumulator.dict_accumulator): sow_hist = hin_dict["SumOfEFTweights"]
        else: sow_hist = hin_dict

        # Get the scale dictionary for each sample in the sample axis
        # If a sample is eft, should scale by 1/(sum of weights at sm), if it's not an eft hist, should scale by 1
        sow_scale_dict = {}
        for sample_name in sow_hist.identifiers('sample'):
            sow_scale_dict[sample_name.name] = 1.0
            is_eft = (len(sow_hist[sample_name]._sumw[sample_name,].shape) == 2)
            if is_eft:
                norm_val = sow_hist[sample_name].values()[sample_name.name,][0]
                sow_scale_dict[sample_name.name] = 1.0/norm_val

        return sow_scale_dict


    # Integrate appl axis if present, keeping only SR
    def integrate_out_appl(self,histo,lep_cat):
        histo_integrated = copy.deepcopy(histo)
        if ("appl" in self.get_axis_list(histo)):
            if "2l" in lep_cat:
                histo_integrated = histo.integrate("appl","isSR_2l")
            elif "3l" in lep_cat:
                histo_integrated = histo.integrate("appl","isSR_3l")
            elif "4l" in lep_cat:
                histo_integrated = histo.integrate("appl","isSR_4l")
            else: raise Exception(f"Error: Category \"{lep_cat}\" is not known.")
        else: print("Already integrated out the appl axis. Continuing...")
        return histo_integrated
            

    # Get the difference between values in nested dictionary, currently can get either percent diff, or absolute diff
    # Returns a dictionary in the same format (currently does not propagate errors, just returns None)
    #   dict = {
    #       k : {
    #           subk : (val,err)
    #       }
    #   }
    def get_diff_between_nested_dicts(self,dict1,dict2,difftype):

        # Get list of keys common to both dictionaries
        common_keys, d1_keys, d2_keys = self.get_common_keys(dict1,dict2)
        if len(d1_keys+d2_keys) > 0:
            print(f"\nWARNING, keys {d1_keys+d2_keys} are not in both dictionaries.")

        ret_dict = {}
        for k in common_keys:

            ret_dict[k] = {}

            # Get list of sub keys common to both sub dictionaries
            common_subkeys, d1_subkeys, d2_subkeys = self.get_common_keys(dict1[k],dict2[k])
            if len(d1_subkeys+d2_subkeys) > 0:
                print(f"\tWARNING, sub keys {d1_subkeys+d2_subkeys} are not in both dictionaries.")

            for subk in common_subkeys:
                v1,e1 = dict1[k][subk]
                v2,e1 = dict2[k][subk]
                if difftype == "percent_diff":
                    ret_diff = self.get_pdiff(v1,v2)
                elif difftype == "absolute_diff":
                    ret_diff = v1 - v2
                else:
                    raise Exception(f"Unknown diff type: {difftype}. Exiting...")

                ret_dict[k][subk] = (ret_diff,None)

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
    def get_normalized_yield(self,hin_dict,year,proc,cat_dict,overflow_str,rwgt_pt=None,h_name="ht"):

        # Integrate out cateogries
        h = hin_dict[h_name]
        h = self.integrate_out_cats(h,cat_dict)
        h = h.integrate("systematic","nominal") # For now anyway...

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
    #   - Returns a dictionary of yields for the categories in the "channel" axis
    #   - Optionally sums over njets or lep flavs
    def get_yld_dict(self,hin_dict,year,njets=False,lepflav=False):

        # Check for options that do not make sense
        if lepflav and not self.is_split_by_lepflav(hin_dict):
            raise Exception("Error: Cannot split by lep flav if the input file is not split by lep flav.")

        # If we want to seperate by njets, don't use njets hist since njets are not in it's sparse axis
        hist_to_use = "njets"
        if njets: hist_to_use = "ht"

        # Get the cat dict (that we will integrate over)
        cat_dict = {}
        for ch in self.get_cat_lables(hin_dict,"channel",h_name=hist_to_use):
            cat_dict[ch] = {}
            nlep_str = ch.split("_")[0]
            cat_dict[ch]["appl"] = self.APPL_DICT[nlep_str]
            cat_dict[ch]["channel"] = ch

        # Find the yields
        yld_dict = {}
        proc_lst = self.get_cat_lables(hin_dict,"sample")
        for proc in proc_lst:
            proc_name_short = self.get_short_name(proc)
            yld_dict[proc_name_short] = {}
            for cat,cuts_dict in cat_dict.items():
                yld_dict[proc_name_short][cat] = self.get_normalized_yield(hin_dict,year,proc,cuts_dict,overflow_str="over",h_name=hist_to_use) # Important to keep overflow

        # If the file is split by lepton flav, but we don't want that, sum over lep flavors:
        if self.is_split_by_lepflav(hin_dict) and not lepflav:
            yld_dict = self.sum_over_lepcats(yld_dict)

        return yld_dict


    # This function takes as input a yld_dict that's seperated by lep flavor, and sums categories over lepton flavor
    def sum_over_lepcats(self,yld_dict):

        sum_dict = {}
        for proc in yld_dict.keys():
            sum_dict[proc] = {}
            for cat_name in yld_dict[proc].keys():

                # Get name without lepflav in it
                name_components = cat_name.split("_")
                lepflav = name_components[1] # Assumes lepflav comes right after nlep e.g. "3l_eee_..."
                name_components.remove(lepflav)
                cat_name_sans_leplfav = "_".join(name_components)

                # Sum the values
                yld,err = yld_dict[proc][cat_name]
                if cat_name_sans_leplfav not in sum_dict[proc].keys(): sum_dict[proc][cat_name_sans_leplfav] = (0,None)
                sum_dict[proc][cat_name_sans_leplfav] = (sum_dict[proc][cat_name_sans_leplfav][0] + yld, None)

        return sum_dict


    # This function:
    #    - Takes as input a yld dict
    #    - Scales each val by the given factor to the power of the number of e in the event
    #    - Rreturns a dict with the same structure, where the vals are scaled
    def scale_ylds_by_em_factor(self,yld_dict,factor):
        ret_dict = {}
        for proc in yld_dict.keys():
            ret_dict[proc] = {}
            for cat_name in yld_dict[proc].keys():
                yld,err = yld_dict[proc][cat_name]
                lepflav = cat_name.split("_")[1] # Assumes lepflav comes right after nlep e.g. "3l_eee_..."
                power_of_e = lepflav.count("e")
                ret_dict[proc][cat_name] = (yld*((factor)**(power_of_e)),None)
        return ret_dict


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
    def print_hist_info(self,path,h_name="njets"):

        if type(path) is str: hin_dict = self.get_hist_from_pkl(path)
        else: hin_dict = path

        for i in range(len(hin_dict[h_name].axes())):
            print(f"\n{i} Aaxis name:",hin_dict[h_name].axes()[i].name)
            for cat in hin_dict[h_name].axes()[i].identifiers():
                print(f"\t{cat}")


    # Print the ratios of e to m from a given yld dict
    def print_em_ratios(self,yld_dict):

        def print_ratios(e_val,m_val,nlep):
            ratio = self.get_em_factor(e_val,m_val,nlep)
            print(f"\te/m from {nlep}l: ({e_val}/{m_val})^(1/{nlep}) = {ratio}")

        for proc in yld_dict.keys():
            print("Proc:",proc)

            yld_sum_dict = {"ee":0, "mm":0, "eee":0, "mmm":0}
            for cat_name in yld_dict[proc].keys():
                yld,err = yld_dict[proc][cat_name]
                lepflav = cat_name.split("_")[1] # Assumes lepflav comes right after nlep e.g. "3l_eee_..."
                if lepflav in yld_sum_dict.keys():
                    yld_sum_dict[lepflav] = yld_sum_dict[lepflav] + yld

            e_over_m_from_2l = (yld_sum_dict["ee"]/yld_sum_dict["mm"])**(1./2.)
            e_over_m_from_3l = (yld_sum_dict["eee"]/yld_sum_dict["mmm"])**(1./3.)

            print_ratios(yld_sum_dict["ee"],yld_sum_dict["mm"],2)
            print_ratios(yld_sum_dict["eee"],yld_sum_dict["mmm"],3)


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
