import numpy as np
import copy
import coffea
from coffea import hist
from topcoffea.modules.HistEFT import HistEFT
from topcoffea.modules.GetValuesFromJsons import get_lumi
import topcoffea.modules.utils as utils

class YieldTools():

    def __init__(self):

        # The order of the categories in the TOP-19-001 AN yield tables
        self.CAT_LST = ["2lss_p", "2lss_m", "2lss_4t_p", "2lss_4t_m", "3l_p_offZ_1b", "3l_m_offZ_1b", "3l_p_offZ_2b", "3l_m_offZ_2b", "3l_onZ_1b", "3l_onZ_2b", "4l"]
        self.CAT_LST_TOP19001 = ["2lss_p", "2lss_m", "3l_p_offZ_1b", "3l_m_offZ_1b", "3l_p_offZ_2b", "3l_m_offZ_2b", "3l_onZ_1b", "3l_onZ_2b", "4l"]

        self.SIG      = ["ttH","ttlnu","ttll","tllq","tHq","tttt"]
        self.BKG      = ["flips","fakes","conv","VV","VVV","tWZ"]
        self.DIBOSON  = ["WW","WZ","ZZ"]
        self.TRIBOSON = ["WWW","WWZ","WZZ","ZZZ"]

        self.DATA_MC_COLUMN_ORDER = ["tWZ", "VV", "VVV", "flips", "fakes", "conv", "bkg", "ttlnu", "ttll", "ttH", "tllq", "tHq", "tttt", "sig", "pred", "data", "pdiff"]

        # A dictionary mapping names of samples in the samples axis to a short version of the name
        self.PROC_MAP = {

            "ttlnu" : ["ttW_centralUL16APV"    ,"ttW_centralUL16"    ,"ttW_centralUL17" ,"ttW_centralUL18" , "ttlnuJet_privateUL18" , "ttlnuJet_privateUL17" , "ttlnuJet_privateUL16" , "ttlnuJet_privateUL16APV"],
            "ttll"  : ["ttZ_centralUL16APV"    ,"ttZ_centralUL16"    ,"ttZ_centralUL17" ,"ttZ_centralUL18" , "ttllJet_privateUL18"  , "ttllJet_privateUL17"  , "ttllJet_privateUL16"  , "ttllJet_privateUL16APV"],
            "ttH"   : ["ttHJet_centralUL16APV" ,"ttHJet_centralUL16" ,"ttH_centralUL17" ,"ttH_centralUL18" , "ttHJet_privateUL18"   , "ttHJet_privateUL17"   , "ttHJet_privateUL16"   , "ttHJet_privateUL16APV"],
            "tllq"  : ["tZq_centralUL16APV"    ,"tZq_centralUL16"    ,"tZq_centralUL17" ,"tZq_centralUL18" , "tllq_privateUL18"     , "tllq_privateUL17"     , "tllq_privateUL16"     , "tllq_privateUL16APV"],
            "tHq"   : ["tHq_centralUL16APV"    ,"tHq_centralUL16"    ,"tHq_centralUL17" ,"tHq_centralUL18" , "tHq_privateUL18"      , "tHq_privateUL17"      , "tHq_privateUL16"      , "tHq_privateUL16APV"],
            "tttt"  : ["tttt_centralUL16APV"   ,"tttt_centralUL16"   ,"tttt_centralUL17","tttt_centralUL18", "tttt_privateUL18"     , "tttt_privateUL17"     , "tttt_privateUL16"     , "tttt_privateUL16APV"],

            "flips" : ["flipsUL16"            ,"flipsUL16APV"            ,"flipsUL17"            ,"flipsUL18"            ],
            "fakes" : ["nonpromptUL16"        ,"nonpromptUL16APV"        ,"nonpromptUL17"        ,"nonpromptUL18"        ],
            "conv"  : ["TTGamma_centralUL16"  ,"TTGamma_centralUL16APV"  ,"TTGamma_centralUL17"  ,"TTGamma_centralUL18"  ],
            "WW"    : ["WWTo2L2Nu_centralUL16","WWTo2L2Nu_centralUL16APV","WWTo2L2Nu_centralUL17","WWTo2L2Nu_centralUL18"],
            "WZ"    : ["WZTo3LNu_centralUL16" ,"WZTo3LNu_centralUL16APV" ,"WZTo3LNu_centralUL17" ,"WZTo3LNu_centralUL18" ],
            "ZZ"    : ["ZZTo4L_centralUL16"   ,"ZZTo4L_centralUL16APV"   ,"ZZTo4L_centralUL17"   ,"ZZTo4L_centralUL18"   ],
            "WWW"   : ["WWW_4F_centralUL16"   ,"WWW_centralUL16APV"      ,"WWW_centralUL17"      ,"WWW_4F_centralUL18"   ],
            "WWZ"   : ["WWZ_4F_centralUL16"   ,"WWZ_4F_centralUL16APV"   ,"WWZ_centralUL17"      ,"WWZ_4F_centralUL18"   ],
            "WZZ"   : ["WZZ_centralUL16"      ,"WZZ_centralUL16APV"      ,"WZZ_centralUL17"      ,"WZZ_centralUL18"      ],
            "ZZZ"   : ["ZZZ_centralUL16"      ,"ZZZ_centralUL16APV"      ,"ZZZ_centralUL17"      ,"ZZZ_centralUL18"      ],
            "tWZ"   : ["TWZToLL_centralUL16"  ,"TWZToLL_centralUL16APV"  ,"TWZToLL_centralUL17"  ,"TWZToLL_centralUL18"  ],

            "ttZlowMll"   : ["TTZToLL_M1to10_centralUL16"  ,"TTZToLL_M1to10_centralUL16APV"  ,"TTZToLL_M1to10_centralUL17"  ,"TTZToLL_M1to10_centralUL18"  ],
            "ttbarll" : ["TTTo2L2Nu_centralUL16", "TTTo2L2Nu_centralUL16APV", "TTTo2L2Nu_centralUL17", "TTTo2L2Nu_centralUL18"],
            "ttbarsl" : ["TTToSemiLeptonic_centralUL16", "TTToSemiLeptonic_centralUL16APV", "TTToSemiLeptonic_centralUL17", "TTToSemiLeptonic_centralUL18"],

            "data"   : ["dataUL16","dataUL16APV","dataUL17","dataUL18"],
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
                "2lss_4t_p"    : (0,None),
                "2lss_4t_m"    : (0,None),
                "2lss_p"       : (68.7,None),
                "2lss_m"       : (37.1,None),
                "3l_p_offZ_1b" : (14.4,None),
                "3l_m_offZ_1b" : (8.0,None),
                "3l_p_offZ_2b" : (10.8,None),
                "3l_m_offZ_2b" : (5.9,None),
                "3l_onZ_1b"    : (2.9,None),
                "3l_onZ_2b"    : (2.3,None),
                "4l"           : (0.0,None),
            },
            "ttll"  : {
                "2lss_4t_p"    : (0,None),
                "2lss_4t_m"    : (0,None),
                "2lss_p"       : (19.3,None),
                "2lss_m"       : (19.0,None),
                "3l_p_offZ_1b" : (12.7,None),
                "3l_m_offZ_1b" : (13.3,None),
                "3l_p_offZ_2b" : (9.1,None),
                "3l_m_offZ_2b" : (8.5,None),
                "3l_onZ_1b"    : (95.5,None),
                "3l_onZ_2b"    : (63.2,None),
                "4l"           : (9.4,None),
            },
            "ttH"   : {
                "2lss_4t_p"    : (0,None),
                "2lss_4t_m"    : (0,None),
                "2lss_p"       : (24.7,None),
                "2lss_m"       : (24.1,None),
                "3l_p_offZ_1b" : (7.9,None),
                "3l_m_offZ_1b" : (7.6,None),
                "3l_p_offZ_2b" : (5.1,None),
                "3l_m_offZ_2b" : (5.2,None),
                "3l_onZ_1b"    : (3.2,None),
                "3l_onZ_2b"    : (2.2,None),
                "4l"           : (1.0,None),
            },
            "tllq"  : {
                "2lss_4t_p"    : (0,None),
                "2lss_4t_m"    : (0,None),
                "2lss_p"       : (2.7,None),
                "2lss_m"       : (1.5,None),
                "3l_p_offZ_1b" : (3.5,None),
                "3l_m_offZ_1b" : (1.8,None),
                "3l_p_offZ_2b" : (1.2,None),
                "3l_m_offZ_2b" : (0.6,None),
                "3l_onZ_1b"    : (39.8,None),
                "3l_onZ_2b"    : (13.3,None),
                "4l"           : (0.0,None),
            },
            "tHq"   : {
                "2lss_4t_p"    : (0,None),
                "2lss_4t_m"    : (0,None),
                "2lss_p"       : (0.8,None),
                "2lss_m"       : (0.4,None),
                "3l_p_offZ_1b" : (0.3,None),
                "3l_m_offZ_1b" : (0.2,None),
                "3l_p_offZ_2b" : (0.2,None),
                "3l_m_offZ_2b" : (0.1,None),
                "3l_onZ_1b"    : (0.1,None),
                "3l_onZ_2b"    : (0.1,None),
                "4l"           : (0.0,None),
            },

            "VV"   : {
                "2lss_4t_p"    : (0,None),
                "2lss_4t_m"    : (0,None),
                "2lss_p"       : (1.6,None),
                "2lss_m"       : (1.2,None),
                "3l_p_offZ_1b" : (5.9,None),
                "3l_m_offZ_1b" : (4.7,None),
                "3l_p_offZ_2b" : (0.4,None),
                "3l_m_offZ_2b" : (0.3,None),
                "3l_onZ_1b"    : (52.1,None),
                "3l_onZ_2b"    : (4.1,None),
                "4l"           : (0.6,None),
            },
            "VVV"   : {
                "2lss_4t_p"    : (0,None),
                "2lss_4t_m"    : (0,None),
                "2lss_p"       : (0.5,None),
                "2lss_m"       : (0.5,None),
                "3l_p_offZ_1b" : (0.2,None),
                "3l_m_offZ_1b" : (0.2,None),
                "3l_p_offZ_2b" : (0,None),
                "3l_m_offZ_2b" : (0.1,None),
                "3l_onZ_1b"    : (3.5,None),
                "3l_onZ_2b"    : (0.6,None),
                "4l"           : (0.1,None),
            },
            "flips"   : {
                "2lss_4t_p"    : (0,None),
                "2lss_4t_m"    : (0,None),
                "2lss_p"       : (8.5,None),
                "2lss_m"       : (8.5,None),
                "3l_p_offZ_1b" : (0,None),
                "3l_m_offZ_1b" : (0,None),
                "3l_p_offZ_2b" : (0,None),
                "3l_m_offZ_2b" : (0,None),
                "3l_onZ_1b"    : (0,None),
                "3l_onZ_2b"    : (0,None),
                "4l"           : (0,None),
            },
            "fakes"   : {
                "2lss_4t_p"    : (0,None),
                "2lss_4t_m"    : (0,None),
                "2lss_p"       : (25.6,None),
                "2lss_m"       : (26.8,None),
                "3l_p_offZ_1b" : (11.3,None),
                "3l_m_offZ_1b" : (13.0,None),
                "3l_p_offZ_2b" : (3.3,None),
                "3l_m_offZ_2b" : (2.5,None),
                "3l_onZ_1b"    : (16.9,None),
                "3l_onZ_2b"    : (3.8,None),
                "4l"           : (0,None),
            },
            "conv"   : {
                "2lss_4t_p"    : (0,None),
                "2lss_4t_m"    : (0,None),
                "2lss_p"       : (10.9,None),
                "2lss_m"       : (9.2,None),
                "3l_p_offZ_1b" : (2.3,None),
                "3l_m_offZ_1b" : (2.6,None),
                "3l_p_offZ_2b" : (1.7,None),
                "3l_m_offZ_2b" : (1.9,None),
                "3l_onZ_1b"    : (0.8,None),
                "3l_onZ_2b"    : (0.4,None),
                "4l"           : (0,None),
            },
            "data"   : {
                "2lss_4t_p"    : (0,None),
                "2lss_4t_m"    : (0,None),
                "2lss_p"       : (192,None),
                "2lss_m"       : (171,None),
                "3l_p_offZ_1b" : (85,None),
                "3l_m_offZ_1b" : (64,None),
                "3l_p_offZ_2b" : (32,None),
                "3l_m_offZ_2b" : (28,None),
                "3l_onZ_1b"    : (239,None),
                "3l_onZ_2b"    : (95,None),
                "4l"           : (12,None),
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
        for long_name in self.PROC_MAP[short_name_in]:
            for long_name_in in long_name_lst_in:
                if long_name_in == long_name:
                    ret_name = long_name
        return ret_name


    ######### General functions #########

    # Get percent difference
    def get_pdiff(self,a,b,in_percent=False):
        #p = (float(a)-float(b))/((float(a)+float(b))/2)
        if ((a is None) or (b is None)):
            p = None
        elif b == 0:
            p = None
        else:
            p = (float(a)-float(b))/float(b)
            if in_percent:
                p = p*100.0
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

    # For a nested dict {k:{subk:v}} reorganizes to be {subk:{k:v}}
    def swap_keys_subkeys(self,in_dict):
        out_dict = {}
        for k in in_dict.keys():
            for subk in in_dict[k].keys():
                if subk not in out_dict: out_dict[subk] = {}
                if k in out_dict[subk].keys():
                    raise Exception("Cannot invert this dict")
                else:
                    out_dict[subk][k] = in_dict[k][subk]
        return out_dict

    # Get a subset of the elements from a list of strings given a whitelist and/or blacklist of substrings
    def filter_lst_of_strs(self,in_lst,substr_whitelist=[],substr_blacklist=[]):

        # Check all elements are strings
        if not (all(isinstance(x,str) for x in in_lst) and all(isinstance(x,str) for x in substr_whitelist) and all(isinstance(x,str) for x in substr_blacklist)):
            raise Exception("Error: This function only filters lists of strings, one of the elements in one of the input lists is not a str.")
        for elem in substr_whitelist:
            if elem in substr_blacklist:
                raise Exception(f"Error: Cannot whitelist and blacklist the same element (\"{elem}\").")

        # Append to the return list
        out_lst = []
        for element in in_lst:
            blacklisted = False
            whitelisted = True
            for substr in substr_blacklist:
                if substr in element:
                    # If any of the substrings are in the element, blacklist it
                    blacklisted = True
            for substr in substr_whitelist:
                if substr not in element:
                    # If any of the substrings are NOT in the element, do not whitelist it
                    whitelisted = False
            if whitelisted and not blacklisted:
                out_lst.append(element)

        return out_lst


    # Get the per lepton e/m factor from e.g. eee and mmm yields
    def get_em_factor(self,e_val,m_val,nlep):
        return (e_val/m_val)**(1.0/nlep)


    # Takes a hist, and retruns a list of the axis names
    def get_axis_list(self,histo):
        axis_lst = []
        for axis in histo.axes():
            axis_lst.append(axis.name)
        return axis_lst


    # Find the list of hists in a pkl file
    def get_hist_list(self,path,allow_empty=True):

        # Get the dict
        if type(path) is str: hin_dict = utils.get_hist_from_pkl(path,allow_empty)
        else: hin_dict = path

        # Get list of keys
        return list(hin_dict.keys())


    # Takes a hist dictionary (i.e. from the pkl file that the processor makes) and an axis name, returns the list of categories for that axis. Defaults to 'njets' histogram if none given.
    def get_cat_lables(self,hin_dict,axis,h_name=None):

        # If the hin is not a histo, then get one of the histos from inside of it
        if not isinstance(hin_dict,HistEFT):

            # If no hist specified, just choose the first one
            if h_name is None:
                all_hists = self.get_hist_list(hin_dict)
                for h in all_hists:
                    if h != "SumOfEFTweights":
                        h_name = h
                        break

                # If we failed to find a hist, raise exception
                if h_name is None:
                    raise Exception("There are no hists in this hist dict")

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
        return (ret_str)


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
        return (ret_str)


    # This should return true if the hist is split by lep flavor, definitely not a bullet proof check..
    def is_split_by_lepflav(self,hin_dict):
        ch_names_lst = self.get_cat_lables(hin_dict,axis="channel")
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




    # Integrate appl axis if present, keeping only SR
    def integrate_out_appl(self,histo,lep_cat):
        histo_integrated = copy.deepcopy(histo)
        if ("appl" in self.get_axis_list(histo)):
            if "2lss" in lep_cat:
                sr_bin = self.APPL_DICT["2lss"]
            elif "2los" in lep_cat:
                sr_bin = self.APPL_DICT["2los"]
            elif "3l" in lep_cat:
                sr_bin = self.APPL_DICT["3l"]
            elif "4l" in lep_cat:
                sr_bin = self.APPL_DICT["4l"]
            else:
                raise Exception(f"Error: Category \"{lep_cat}\" is not known.")
            histo_integrated = histo.integrate("appl",sr_bin)
        else:
            print("Already integrated out the appl axis. Continuing...")
        return histo_integrated


    # Get the difference between values in nested dictionary, currently can get either percent diff, or absolute diff
    # Returns a dictionary in the same format (currently does not propagate errors, just returns None)
    #   dict = {
    #       k : {
    #           subk : (val,err)
    #       }
    #   }
    def get_diff_between_nested_dicts(self,dict1,dict2,difftype,inpercent=False):

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
                    ret_diff = self.get_pdiff(v1,v2,in_percent=inpercent)
                elif difftype == "absolute_diff":
                    ret_diff = v1 - v2
                else:
                    raise Exception(f"Unknown diff type: {difftype}. Exiting...")

                ret_dict[k][subk] = (ret_diff,None)

        return ret_dict

    # Takes as input a dictionary {"k": {"subk":[val,err]}} and returns {"k":{"subk":val}}
    def strip_errs(self,in_dict):
        out_dict = {}
        for k in in_dict.keys():
            out_dict[k] = {}
            for subk in in_dict[k]:
                out_dict[k][subk] = in_dict[k][subk][0]
        return out_dict

    # Takes as input a dictionary {"k":{"subk":val}} and returns {"k": {"subk":[val,None]}}
    def put_none_errs(self,in_dict):
        out_dict = {}
        for k in in_dict.keys():
            out_dict[k] = {}
            for subk in in_dict[k]:
                out_dict[k][subk] = [in_dict[k][subk],None]
        return out_dict


    # This function:
    #   - Takes as input yield dict {"proc":{"cat":yld}}
    #   - For each process we cobine categories that have the same name except for the given str (e.g. combine across charges if you pass ["p","m"])
    #   - Can also take as input {"proc":{"cat":[yld,err]}}, but note we will not propagate errors, and instead will just replace errors with None
    def sum_over_cats(self,in_dict,combine_str_lst):
        out_dict = {}
        for proc_name in in_dict.keys():
            tmp_dict = {}
            for cat_name in in_dict[proc_name]:
                # Do we have errors or not
                if isinstance(in_dict[proc_name][cat_name],float): errs = False
                else: errs = True # Assumes we have [yld,err]
                # Get the stripped name
                cat_name_component_lst = cat_name.split("_")
                for s in combine_str_lst:
                    if s in cat_name_component_lst: cat_name_component_lst.remove(s)
                cat_name_stripped = "_".join(cat_name_component_lst)
                # Add the common cateogires to the out dict
                if cat_name_stripped not in tmp_dict:
                    if errs: tmp_dict[cat_name_stripped] = [in_dict[proc_name][cat_name][0],None]
                    else   : tmp_dict[cat_name_stripped] =  in_dict[proc_name][cat_name]
                else:
                    if errs: tmp_dict[cat_name_stripped][0] = tmp_dict[cat_name_stripped][0] + in_dict[proc_name][cat_name][0]
                    else   : tmp_dict[cat_name_stripped]    = tmp_dict[cat_name_stripped]    + in_dict[proc_name][cat_name]
            out_dict[proc_name] = tmp_dict

        return out_dict



    ######### Functions specifically for getting yields #########

    # Sum all the values of a hist
    #    - The hist you pass should have two axes (all other should already be integrated out)
    #    - The two axes should be the samples axis, and the dense axis (e.g. ht)
    #    - You pass a process name, and we select just that category from the sample axis
    def get_yield(self,h,proc,overflow_str="none"):
        h_vals = h[proc].values(sumw2=True,overflow=overflow_str)
        if len(h_vals) != 0: # I.e. dict is not empty, this process exists in this dict
            for i,(k,v) in enumerate(h_vals.items()):
                v_sum = v[0].sum()
                e_sum = v[1].sum()
                if i > 0: raise Exception("Why is i greater than 0? The hist is not what this function is expecting. Exiting...")
        else:
            v_sum = 0.0
            e_sum = 0.0
        e_sum = np.sqrt(e_sum)
        return [v_sum,e_sum]


    # Integrates out categories, normalizes, then calls get_yield()
    def get_normalized_yield(self,hin_dict,proc,cat_dict,overflow_str,rwgt_pt=None,h_name="ht"):

        # Integrate out cateogries
        h = hin_dict[h_name]
        h = self.integrate_out_cats(h,cat_dict)
        h = h.integrate("systematic","nominal") # For now anyway...

        # Reweight the hist
        if rwgt_pt is not None:
            hist.set_wilson_coefficients(**rwgt_pt)
        else:
            h.set_sm()

        return self.get_yield(h,proc,overflow_str)


    # This function:
    #   - Takes as input a hist dict (i.e. what the processor outptus)
    #   - Returns a dictionary of yields for the categories in the "channel" axis
    #   - Optionally sums over njets or lep flavs
    def get_yld_dict(self,hin_dict,year=None,njets=False,lepflav=False):

        # Check for options that do not make sense
        if lepflav and not self.is_split_by_lepflav(hin_dict):
            raise Exception("Error: Cannot split by lep flav if the input file is not split by lep flav.")

        # If we want to seperate by njets, don't use njets hist since njets are not in it's sparse axis
        hist_to_use = "njets"
        if njets: hist_to_use = "lj0pt"

        # Get the cat dict (that we will integrate over)
        cat_dict = {}
        for ch in self.get_cat_lables(hin_dict,"channel",h_name=hist_to_use):
            cat_dict[ch] = {}
            nlep_str = ch.split("_")[0]
            if "appl" in self.get_axis_list(hin_dict[hist_to_use]):
                cat_dict[ch]["appl"] = self.APPL_DICT[nlep_str]
            cat_dict[ch]["channel"] = ch

        # Find the yields
        yld_dict = {}
        proc_lst = self.get_cat_lables(hin_dict,"sample")
        #if "flipsUL17" not in proc_lst: proc_lst = proc_lst + ["flipsUL16","flipsUL16APV","flipsUL17","flipsUL18"] # Very bad workaround for _many_ reasons.. leaving it in since it's useful for getting yields of the full pkl file (but we don't need it for e.g. the CI, so leave it commented), note this entire class is a mess and should be totally rewritten before the next analysis
        print("proc_lst",proc_lst)
        for proc in proc_lst:
            p = self.get_short_name(proc)
            print("Name:",p,proc) # Print what name the sample has been matched to

        for proc in proc_lst:
            if year is not None:
                if not proc.endswith(year): continue
            proc_name_short = self.get_short_name(proc)
            if proc_name_short not in yld_dict:
                yld_dict[proc_name_short] = {}
                for cat,cuts_dict in cat_dict.items():
                    yld_dict[proc_name_short][cat] = self.get_normalized_yield(hin_dict,proc,cuts_dict,overflow_str="over",h_name=hist_to_use) # Important to keep overflow
            else:
                for cat,cuts_dict in cat_dict.items():
                    yld_dict[proc_name_short][cat][0] += self.get_normalized_yield(hin_dict,proc,cuts_dict,overflow_str="over",h_name=hist_to_use)[0] # Important to keep overflow
                    yld_dict[proc_name_short][cat][1] = None # Ok, let's just forget the sumw2...

        # If the file is split by lepton flav, but we don't want that, sum over lep flavors:
        if self.is_split_by_lepflav(hin_dict) and not lepflav:
            yld_dict = self.sum_over_lepcats(yld_dict)

        return yld_dict



    ######### Functions specifically for manipulating the "yld_dict" object  #########

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



    # This function is a tool to sum a subset of the values in a dict (a subdict of the yield_dict)
    #   - Assumes all keys in keys_to_sum list are in the dict
    #   - Assumes the keys are tuples of val,err and we ignore the err in the sum
    def get_subset_sum(self,in_dict,keys_to_sum,skip_missing_keys=False):
        out_vals = {}
        for k in keys_to_sum:
            if k not in in_dict:
                if skip_missing_keys:
                    print(f"Warning: key {k} is missing from dict with keys {list(in_dict.keys())}")
                    continue
                else:
                    raise Exception(f"Error: key {k} is missing from dict with keys {list(in_dict.keys())}")
            for subk in in_dict[k].keys():
                if subk not in out_vals:
                    out_vals[subk] = [in_dict[k][subk][0],None]
                else:
                    out_vals[subk][0] = out_vals[subk][0] + in_dict[k][subk][0]
        return out_vals

    # This function
    #   - Takes as input a yld_dict
    #   - Combines the bkg processes (e.g. combine all diboson processes)
    #   - Returns a new yld dict with these combined processes
    def comb_bkg_procs(self,yld_dict):
        ret_dict = {}
        for proc in self.BKG:
            if proc in yld_dict.keys():
                ret_dict[proc] = yld_dict[proc]
            else:
                if proc == "VV":
                    ret_dict["VV"] = self.get_subset_sum(yld_dict,self.DIBOSON)
                if proc == "VVV":
                    ret_dict["VVV"] = self.get_subset_sum(yld_dict,self.TRIBOSON)
        for proc in self.SIG + ["data"]:
            # Pass through
            ret_dict[proc] = yld_dict[proc]

        return ret_dict

    # This function
    #   - takes as input a yld_dict
    #   - Assumes that the yld dict has all of the processes relevent for the analysis (sig, bkg, data)
    #   - Combines calculates sub category sums
    #   - Returns a new yld dict with these combined sums
    def sum_over_ana_procs(self,yld_dict,skip4t=False,comb2lss=False):

        # Fill the out dict with the orig set of processes
        ret_dict = {}
        for proc in yld_dict.keys():
            if skip4t and proc == "tttt": continue
            ret_dict[proc] = yld_dict[proc]

        # Get sums and add those to the dict
        bkg_lst = copy.deepcopy(self.BKG)
        sig_lst = copy.deepcopy(self.SIG)
        if skip4t:
            sig_lst.remove("tttt")
        ret_dict["bkg"] = self.get_subset_sum(yld_dict,bkg_lst)
        ret_dict["sig"] = self.get_subset_sum(yld_dict,sig_lst)
        ret_dict["pred"] = self.get_subset_sum(yld_dict,sig_lst+bkg_lst)

        # Note that this assumes the 2lss_p, 2lss_m, 2lss_4t_p, and 2lss_4t_m keys are in the dict
        # Adds the 4t category to the regular 2lss cat
        if comb2lss:
            ret_dict_comb_2lss = {}
            for proc_name in ret_dict.keys():
                ret_dict_comb_2lss[proc_name] = {}
                for cat_name in ret_dict[proc_name].keys():
                    val1 = ret_dict[proc_name][cat_name][0]
                    val2 = None
                    if cat_name == "2lss_p":
                        val2 = ret_dict[proc_name]["2lss_4t_p"][0]
                    elif cat_name == "2lss_m":
                        val2 = ret_dict[proc_name]["2lss_4t_m"][0]
                    elif cat_name == "2lss_4t_p" or cat_name == "2lss_4t_m":
                        # We won't have keys in the out dict for this cat
                        continue

                    # Here number+None=number, and None+None=None
                    if val1 is None and val2 is None:
                        val = None
                    elif val1 is None:
                        val = val2
                    elif val2 is None:
                        val = val1
                    else:
                        val = val1 + val2
                    ret_dict_comb_2lss[proc_name][cat_name] = (val,None)

            ret_dict = ret_dict_comb_2lss

        ret_dict["pdiff"] = {}
        for cat in ret_dict["pred"].keys():
            obs = ret_dict["data"][cat][0]
            exp = ret_dict["pred"][cat][0]
            pdiff = self.get_pdiff(exp,obs,in_percent=True)
            ret_dict["pdiff"][cat] = [pdiff,None]

        return ret_dict



    ######### Functions that just print out information #########


    # Print out all the info about all the axes in a hist
    def print_hist_info(self,path,h_name="njets",verbose=False):

        # Get the dict
        if type(path) is str: hin_dict = utils.get_hist_from_pkl(path)
        else: hin_dict = path

        # Print info about all keys
        print("\nThe keys of the dict are:",list(hin_dict.keys()))
        if verbose:
            for k in hin_dict.keys():
                print(f"\n{k}: {hin_dict[k].values()}")

        # Print info about axes for one key
        print(f"\nPrinting info for key \"{h_name}\":")
        for i in range(len(hin_dict[h_name].axes())):
            print(f"\n    {i} Aaxis name:",hin_dict[h_name].axes()[i].name)
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
