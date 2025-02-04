import numpy as np
import os
import copy
import datetime
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler

import mplhep as hep
import hist
from topcoffea.modules.histEFT import HistEFT
from topeft.modules.axes import info as axes_info

from topcoffea.scripts.make_html import make_html
import topcoffea.modules.utils as utils
from topeft.modules.yield_tools import YieldTools

from topcoffea.modules.paths import topcoffea_path
import topeft.modules.get_rate_systs as grs
from topcoffea.modules.get_param_from_jsons import GetParam
get_tc_param = GetParam(topcoffea_path("params/params.json"))


# This script takes an input pkl file that should have both data and background MC included.
# Use the -y option to specify a year, if no year is specified, all years will be included.
# There are various other options available from the command line.
# For example, to make unit normalized plots for 2018, with the timestamp appended to the directory name, you would run:
#     python make_cr_plots.py -f histos/your.pkl.gz -o ~/www/somewhere/in/your/web/dir -n some_dir_name -y 2018 -t -u

# Some options for plotting the data and MC
DATA_ERR_OPS = {'linestyle':'none', 'marker': '.', 'markersize': 10., 'color':'k', 'elinewidth': 1,}
MC_ERROR_OPS = {'label': 'Stat. Unc.', 'hatch': '////', 'facecolor': 'none', 'edgecolor': (0,0,0,.5), 'linewidth': 0}
FILL_OPS = {}

# The channels that define the CR categories
CR_CHAN_DICT = {
    "cr_2los_Z" : [
        "2los_ee_CRZ_0j",
        "2los_mm_CRZ_0j",
    ],
    "cr_2los_tt" : [
        "2los_em_CRtt_2j",
    ],
    "cr_2lss" : [
        "2lss_ee_CR_1j",
        "2lss_em_CR_1j",
        "2lss_mm_CR_1j",
        "2lss_ee_CR_2j",
        "2lss_em_CR_2j",
        "2lss_mm_CR_2j",
    ],
    "cr_2lss_flip" : [
        "2lss_ee_CRflip_3j",
    ],
    "cr_3l" : [
        "3l_eee_CR_0j",
        "3l_eem_CR_0j",
        "3l_emm_CR_0j",
        "3l_mmm_CR_0j",
        "3l_eee_CR_1j",
        "3l_eem_CR_1j",
        "3l_emm_CR_1j",
        "3l_mmm_CR_1j",
    ],
}


SR_CHAN_DICT = {
    "2lss_SR": [
        "2lss_4t_m_4j", "2lss_4t_m_5j", "2lss_4t_m_6j", "2lss_4t_m_7j",
        "2lss_4t_p_4j", "2lss_4t_p_5j", "2lss_4t_p_6j", "2lss_4t_p_7j",
        "2lss_m_4j", "2lss_m_5j", "2lss_m_6j", "2lss_m_7j",
        "2lss_p_4j", "2lss_p_5j", "2lss_p_6j", "2lss_p_7j",
    ],
    "2lss_fwd_SR": [
        "2lss_fwd_m_4j", "2lss_fwd_m_5j", "2lss_fwd_m_6j", "2lss_fwd_m_7j",
        "2lss_fwd_p_4j", "2lss_fwd_p_5j", "2lss_fwd_p_6j", "2lss_fwd_p_7j",
    ],
    "3l_offZ_SR" : [
        "3l_m_offZ_1b_2j", "3l_m_offZ_1b_3j", "3l_m_offZ_1b_4j", "3l_m_offZ_1b_5j",
        "3l_m_offZ_2b_2j", "3l_m_offZ_2b_3j", "3l_m_offZ_2b_4j", "3l_m_offZ_2b_5j",
        "3l_p_offZ_1b_2j", "3l_p_offZ_1b_3j", "3l_p_offZ_1b_4j", "3l_p_offZ_1b_5j",
        "3l_p_offZ_2b_2j", "3l_p_offZ_2b_3j", "3l_p_offZ_2b_4j", "3l_p_offZ_2b_5j",
    ],
    "3l_offZ_fwd_SR" : [
        "3l_m_offZ_1b_fwd_2j","3l_m_offZ_1b_fwd_3j", "3l_m_offZ_1b_fwd_4j", "3l_m_offZ_1b_fwd_5j",
        #"3l_m_offZ_2b_fwd_2j","3l_m_offZ_2b_fwd_3j", "3l_m_offZ_2b_fwd_4j", "3l_m_offZ_2b_fwd_5j",
        "3l_p_offZ_1b_fwd_2j","3l_p_offZ_1b_fwd_3j", "3l_p_offZ_1b_fwd_4j", "3l_p_offZ_1b_fwd_5j",
        #"3l_p_offZ_2b_fwd_2j","3l_p_offZ_2b_fwd_3j", "3l_p_offZ_2b_fwd_4j", "3l_p_offZ_2b_fwd_5j",
    ],
    "3l_offZ_fwd_1j_SR" : [
        "3l_m_offZ_1b_fwd_1j",
        #"3l_m_offZ_2b_fwd_1j",
        "3l_p_offZ_1b_fwd_1j",
        #"3l_p_offZ_2b_fwd_1j",
    ],
    "3l_offZ_fwd_2j_SR" : [
        "3l_m_offZ_1b_fwd_2j",
        #"3l_m_offZ_2b_fwd_2j",
        "3l_p_offZ_1b_fwd_2j",
        #"3l_p_offZ_2b_fwd_2j",
    ],
    "3l_offZ_fwd_3j_SR" : [
        "3l_m_offZ_1b_fwd_3j",
        #"3l_m_offZ_2b_fwd_3j",
        "3l_p_offZ_1b_fwd_3j",
        #"3l_p_offZ_2b_fwd_3j",
    ],
    "3l_offZ_fwd_4j_SR" : [
        "3l_m_offZ_1b_fwd_4j",
        #"3l_m_offZ_2b_fwd_4j",
        "3l_p_offZ_1b_fwd_4j",
        #"3l_p_offZ_2b_fwd_4j",
    ],
    "3l_offZ_fwd_5j_SR" : [
        "3l_m_offZ_1b_fwd_5j",
        #"3l_m_offZ_2b_fwd_5j",
        "3l_p_offZ_1b_fwd_5j",
        #"3l_p_offZ_2b_fwd_5j",
    ],
    "3l_onZ_SR" : [
        "3l_onZ_1b_2j"   , "3l_onZ_1b_3j"   , "3l_onZ_1b_4j"   , "3l_onZ_1b_5j",
        "3l_onZ_2b_2j"   , "3l_onZ_2b_3j"   , "3l_onZ_2b_4j"   , "3l_onZ_2b_5j",
    ],
    "3l_onZ_fwd_SR" : [
        "3l_onZ_1b_fwd_1j"   , "3l_onZ_1b_fwd_2j"   , "3l_onZ_1b_3j"   , "3l_onZ_1b_4j"   , "3l_onZ_1b_5j",
        #"3l_onZ_2b_fwd_1j"   , "3l_onZ_2b_fwd_2j"   , "3l_onZ_2b_3j"   , "3l_onZ_2b_4j"   , "3l_onZ_2b_5j",
    ],
    "3l_onZ_fwd_1j_SR" : [
        "3l_onZ_1b_fwd_1j"  ,
        #"3l_onZ_2b_fwd_1j"  ,
    ],
    "3l_onZ_fwd_2j_SR" : [
        "3l_onZ_1b_fwd_2j"  ,
        #"3l_onZ_2b_fwd_2j"  ,
    ],
    "3l_onZ_fwd_3j_SR" : [
        "3l_onZ_1b_fwd_3j"  ,
        #"3l_onZ_2b_fwd_3j"  ,
    ],
    "3l_onZ_fwd_4j_SR" : [
        "3l_onZ_1b_fwd_4j"  ,
        #"3l_onZ_2b_fwd_4j"  ,
    ],
    "3l_onZ_fwd_5j_SR" : [
        "3l_onZ_1b_fwd_5j"  ,
        #"3l_onZ_2b_fwd_5j"  ,
    ],
    "4l_SR" : [
        "4l_2j", "4l_3j", "4l_4j",
    ],
    "4l_2j" : [
        "4l_2j",
    ],
    "4l_3j" : [
        "4l_3j",
    ],
    "4l_4j" : [
        "4l_4j",
    ]
}


CR_GRP_MAP = {
    "DY" : [],
    "Ttbar" : [],
    "Ttbarpowheg" : [],
    "ZGamma" : [],
    "Diboson" : [],
    "Triboson" : [],
    "Single top" : [],
    "Singleboson" : [],
    "Conv": [],
    "Nonprompt" : [],
    "Flips" : [],
    "Signal" : [],
    "Data" : [],
}

SR_GRP_MAP = {
    "Data": [],
    "Conv": [],
    "Diboson" : [],
    "Multiboson" : [],
    "Nonprompt" : [],
    "Flips" : [],
    "ttH" : [],
    "ttlnu" : [],
    "ttll" : [],
    "tttt" : [],
    "tXq" : [],
}

# Best fit point from TOP-19-001 with madup numbers for the 10 new WCs
WCPT_EXAMPLE = {
    "ctW": -0.74,
    "ctZ": -0.86,
    "ctp": 24.5,
    "cpQM": -0.27,
    "ctG": -0.81,
    "cbW": 3.03,
    "cpQ3": -1.71,
    "cptb": 0.13,
    "cpt": -3.72,
    "cQl3i": -4.47,
    "cQlMi": 0.51,
    "cQei": 0.05,
    "ctli": 0.33,
    "ctei": 0.33,
    "ctlSi": -0.07,
    "ctlTi": -0.01,
    "cQq13"  : -0.05,
    "cQq83"  : -0.15,
    "cQq11"  : -0.15,
    "ctq1"   : -0.20,
    "cQq81"  : -0.50,
    "ctq8"   : -0.50,
    "ctt1"   : -0.71,
    "cQQ1"   : -1.35,
    "cQt8"   : -2.89,
    "cQt1"   : -1.24,
}

# Some of our processes do not have rate systs split into qcd and pdf, so we just list them under qcd
# This list keeps track of those, so we can handle them when extracting the numbers from the rate syst json
PROC_WITHOUT_PDF_RATE_SYST = ["tttt","ttll","ttlnu","Triboson","tWZ","convs"]

yt = YieldTools()


######### Utility functions #########

# Takes a dictionary where the keys are catetory names and keys are lists of bin names in the category, and a string indicating what type of info (njets, or lepflav) to remove
# Returns a dictionary of the same structure, except with njet or lepflav info stripped off of the bin names
# E.g. if a value was ["cat_a_1j","cat_b_1j","cat_b_2j"] and we passed "njets", we should return ["cat_a","cat_b"]
def get_dict_with_stripped_bin_names(in_chan_dict,type_of_info_to_strip):
    out_chan_dict = {}
    for cat,bin_names in in_chan_dict.items():
        out_chan_dict[cat] = []
        for bin_name in bin_names:
            if type_of_info_to_strip == "njets":
                bin_name_no_njet = yt.get_str_without_njet(bin_name)
            elif type_of_info_to_strip == "lepflav":
                bin_name_no_njet = yt.get_str_without_lepflav(bin_name)
            else:
                raise Exception(f"Error: Unknown type of string to remove \"{type_of_info_to_strip}\".")
            if bin_name_no_njet not in out_chan_dict[cat]:
                out_chan_dict[cat].append(bin_name_no_njet)
    return (out_chan_dict)

def group(h: HistEFT, oldname: str, newname: str, grouping: dict[str, list[str]]):
    hnew = HistEFT(
        hist.axis.StrCategory(grouping, name=newname),
        *(ax for ax in h.axes if ax.name != oldname),
        wc_names=h.wc_names,
    )
    for i, indices in enumerate(grouping.values()):
        ind = [c for c in indices if c in h.axes[0]]
        hnew.view(flow=True)[i] = h[{oldname: ind}][{oldname: sum}].view(flow=True)

    return hnew
# Group bins in a hist, returns a new hist
def group_bins(histo,bin_map,axis_name="process",drop_unspecified=False):

    bin_map = copy.deepcopy(bin_map) # Don't want to edit the original

    # Construct the map of bins to remap
    bins_to_remap_lst = []
    for grp_name,bins_in_grp in bin_map.items():
        bins_to_remap_lst.extend(bins_in_grp)
    if not drop_unspecified:
        for bin_name in yt.get_cat_lables(histo,axis_name):
            if bin_name not in bins_to_remap_lst:
                bin_map[bin_name] = bin_name
    bin_map = {m:bin_map[m] for m in bin_map if any(a in list(histo.axes[axis_name]) for a in bin_map[m])}

    # Remap the bins
    old_ax = histo.axes[axis_name]
    #new_ax = hist.axis.StrCategory([], name=old_ax.name, label=old_ax.label, growth=True)
    new_histo = group(histo, axis_name, axis_name, bin_map)
    #new_histo = histo.group(axis_name, bin_map)

    return new_histo


######### Functions for getting info from the systematics json #########

# Match a given sample name to whatever it is called in the json
# Will return None if a match is not found
def get_scale_name(sample_name,sample_group_map,group_type="CR"):
    scale_name_for_json = None
    if group_type == "CR":
        if sample_name in sample_group_map["Conv"]:
            scale_name_for_json = "convs"
        elif sample_name in sample_group_map["Diboson"]:
            scale_name_for_json = "Diboson"
        elif sample_name in sample_group_map["Triboson"]:
            scale_name_for_json = "Triboson"
        elif sample_name in sample_group_map["Signal"]:
            for proc_str in ["ttH","tllq","ttlnu","ttll","tHq","tttt"]:
                if proc_str in sample_name:
                    # This should only match once, but maybe we should put a check to enforce this
                    scale_name_for_json = proc_str
    else:
        if sample_name in sample_group_map["Conv"]:
            scale_name_for_json = "convs"
        elif sample_name in sample_group_map["Diboson"]:
            scale_name_for_json = "Diboson"
        #elif "Multiboson" in sample_group_map and sample_name in sample_group_map["Multiboson"]:
        #    scale_name_for_json = "Multiboson"
        else:
            for proc_str in ["ttH","tllq","ttlnu","ttll","tHq","tttt"]:
                if proc_str in sample_name:
                    # This should only match once, but maybe we should put a check to enforce this
                    scale_name_for_json = proc_str
    return scale_name_for_json

# This function gets the tag that indicates how a particualr systematic is correlated
#   - For pdf_scale this corresponds to the initial state (e.g. gg)
#   - For qcd_scale this corresponds to the process type (e.g. VV)
# For any systemaitc or process that is not included in the correlations json we return None
def get_correlation_tag(uncertainty_name,proc_name,sample_group_map,group_type):
    proc_name_in_json = get_scale_name(proc_name,sample_group_map,group_type)
    corr_tag = None
    # Right now we only have two types of uncorrelated rate systematics
    if uncertainty_name in ["qcd_scale","pdf_scale"]:
        if proc_name_in_json is not None:
            if proc_name_in_json == "convs":
                # Special case for conversions since we estimate from LO sample, we do not have qcd uncty
                # Would be better to handle this in a more general way
                corr_tag = None
            else:
                corr_tag = grs.get_correlation_tag(uncertainty_name,proc_name_in_json)
    return corr_tag

# This function gets all of the the rate systematics from the json file
# Returns a dictionary with all of the uncertainties
# If the sample does not have an uncertainty in the json, an uncertainty of 0 is returned for that category
def get_rate_systs(sample_name,sample_group_map,group_type):

    # Figure out the name of the appropriate sample in the syst rate json (if the proc is in the json)
    scale_name_for_json = get_scale_name(sample_name,sample_group_map,group_type)

    # Get the lumi uncty for this sample (same for all samles)
    lumi_uncty = grs.get_syst("lumi")

    # Get the flip uncty from the json (if there is not an uncertainty for this sample, return 1 since the uncertainties are multiplicative)
    if sample_name in sample_group_map["Flips"]:
        flip_uncty = grs.get_syst("charge_flips","charge_flips_sm")
    else:
        flip_uncty = [1.0,1,0]

    # Get the scale uncty from the json (if there is not an uncertainty for this sample, return 1 since the uncertainties are multiplicative)
    if scale_name_for_json is not None:
        if scale_name_for_json in PROC_WITHOUT_PDF_RATE_SYST:
            # Special cases for when we do not have a pdf uncty (this is a really brittle workaround)
            # NOTE Someday should fix this, it's a really hardcoded and brittle and bad workaround
            pdf_uncty = [1.0,1,0]
        else:
            pdf_uncty = grs.get_syst("pdf_scale",scale_name_for_json)
        if scale_name_for_json == "convs":
            # Special case for conversions, since we estimate these from a LO sample, so we don't have an NLO uncty here
            # Would be better to handle this in a more general way
            qcd_uncty = [1.0,1,0]
        else:
            # In all other cases, use the qcd scale uncty that we have for the process
            qcd_uncty = grs.get_syst("qcd_scale",scale_name_for_json)
    else:
        pdf_uncty = [1.0,1,0]
        qcd_uncty = [1.0,1,0]

    out_dict = {"pdf_scale":pdf_uncty, "qcd_scale":qcd_uncty, "lumi":lumi_uncty, "charge_flips":flip_uncty}
    return out_dict


# Wrapper for getting plus and minus rate arrs
def get_rate_syst_arrs(base_histo,proc_group_map,group_type="CR"):

    # Fill dictionary with the rate uncertainty arrays (with correlated ones organized together)
    rate_syst_arr_dict = {}
    for rate_sys_type in grs.get_syst_lst():
        rate_syst_arr_dict[rate_sys_type] = {}
        for sample_name in yt.get_cat_lables(base_histo,"process"):

            # Build the plus and minus arrays from the rate uncertainty number and the nominal arr
            rate_syst_dict = get_rate_systs(sample_name,proc_group_map,group_type)
            thissample_nom_arr = base_histo.integrate("process",sample_name).integrate("systematic","nominal").eval({})[()]
            p_arr = thissample_nom_arr*(rate_syst_dict[rate_sys_type][1]) - thissample_nom_arr # Difference between positive fluctuation and nominal
            m_arr = thissample_nom_arr*(rate_syst_dict[rate_sys_type][0]) - thissample_nom_arr # Difference between positive fluctuation and nominal

            # Put the arrays into the correlation dict (organizing correlated ones together)
            correlation_tag = get_correlation_tag(rate_sys_type,sample_name,proc_group_map,group_type)
            out_key_name = rate_sys_type
            if correlation_tag is not None: out_key_name += "_"+correlation_tag
            if out_key_name not in rate_syst_arr_dict[rate_sys_type]:
                rate_syst_arr_dict[rate_sys_type][out_key_name] = {"p":[],"m":[]}
            rate_syst_arr_dict[rate_sys_type][out_key_name]["p"].append(p_arr)
            rate_syst_arr_dict[rate_sys_type][out_key_name]["m"].append(m_arr)

    # Now sum the linearly correlated ones and then square everything
    all_rates_p_sumw2_lst = []
    all_rates_m_sumw2_lst = []
    for syst_name in rate_syst_arr_dict.keys():
        for correlated_syst_group in rate_syst_arr_dict[syst_name]:
            sum_p_arrs = sum(rate_syst_arr_dict[syst_name][correlated_syst_group]["p"])
            sum_m_arrs = sum(rate_syst_arr_dict[syst_name][correlated_syst_group]["m"])
            all_rates_p_sumw2_lst.append(sum_p_arrs*sum_p_arrs)
            all_rates_m_sumw2_lst.append(sum_m_arrs*sum_m_arrs)

    return [sum(all_rates_m_sumw2_lst),sum(all_rates_p_sumw2_lst)]

# Wrapper for getting plus and minus shape arrs
def get_shape_syst_arrs(base_histo):

    # Get the list of systematic base names (i.e. without the up and down tags)
    # Assumes each syst has a "systnameUp" and a "systnameDown" category on the systematic axis
    syst_var_lst = []
    all_syst_var_lst = yt.get_cat_lables(base_histo,"systematic")
    for syst_var_name in all_syst_var_lst:
        if syst_var_name.endswith("Up"):
            syst_name_base = "Up".join(syst_var_name.split("Up")[:-1])
            if syst_name_base not in syst_var_lst:
                syst_var_lst.append(syst_name_base)

    # Sum each systematic's contribtuions for all samples together (e.g. the ISR for all samples is summed linearly)
    p_arr_rel_lst = []
    m_arr_rel_lst = []
    for syst_name in syst_var_lst:
        # Skip the variation of renorm and fact together, since we're treating as independent
        if syst_name == "renormfact": continue

        relevant_samples_lst = yt.get_cat_lables(base_histo.integrate("systematic",syst_name+"Up"), "process") # The samples relevant to this syst
        n_arr = base_histo.integrate("process",relevant_samples_lst)[{'process': sum}].integrate("systematic","nominal").eval({})[()] # Sum of all samples for nominal variation

        # Special handling of renorm and fact
        # Uncorrelate these systs across the processes (though leave processes in groups like dibosons correlated to be consistent with SR)
        if (syst_name == "renorm") or (syst_name == "fact"):
            p_arr_rel,m_arr_rel = get_decorrelated_uncty(syst_name,CR_GRP_MAP,relevant_samples_lst,base_histo,n_arr)

        # If the syst is not renorm or fact, just treat it normally (correlate across all processes)
        else:
            u_arr_sum = base_histo.integrate("process",relevant_samples_lst)[{"process": sum}].integrate("systematic",syst_name+"Up").eval({})[()]   # Sum of all samples for up variation
            d_arr_sum = base_histo.integrate("process",relevant_samples_lst)[{"process": sum}].integrate("systematic",syst_name+"Down").eval({})[()] # Sum of all samples for down variation

            u_arr_rel = u_arr_sum - n_arr # Diff with respect to nominal
            d_arr_rel = d_arr_sum - n_arr # Diff with respect to nominal
            p_arr_rel = np.where(u_arr_rel>0,u_arr_rel,d_arr_rel) # Just the ones that increase the yield
            m_arr_rel = np.where(u_arr_rel<0,u_arr_rel,d_arr_rel) # Just the ones that decrease the yield

        # Square and append this syst to the return lists
        p_arr_rel_lst.append(p_arr_rel*p_arr_rel) # Square each element in the arr and append the arr to the out list
        m_arr_rel_lst.append(m_arr_rel*m_arr_rel) # Square each element in the arr and append the arr to the out list

    return [sum(m_arr_rel_lst), sum(p_arr_rel_lst)]


# Special case for renorm and fact, as these are decorrelated across processes
# Sorry to anyone who tries to read this in the future, this function is very ad hoc and messy and hard to follow
# Just used in get_shape_syst_arrs()
# Here are a few notes:
#   - This is complicated, so I just symmetrized the errors
#   - The processes are generally correlated across groups (e.g. WZ and ZZ) since this is what's done in the datacard maker for the SR
#   - So the grouping generally follows what's in the CR group map, except in the case of signal
#       - Since in the SR all signal processes are uncorrelated for these systs, we also uncorrelate here
#       - Note there are caveats to this:
#           * In the SR, TTZToLL_M1to10 and TTToSemiLeptonic and TTTo2L2Nu are all grouped into ttll
#           * Here in the CR TTZToLL_M1to10 is part of signal group, but TTToSemiLeptonic and TTTo2L2Nu are in their own ttbar group
#           * So there are two differences with respect to how these processes are grouped in the SR:
#               1) Here TTToSemiLeptonic and TTTo2L2Nu are correlated with each other, but not with ttll
#               2) Here TTZToLL_M1to10 is grouped as part of signal (as in SR) but here _all_ signal processes are uncorrleated so here TTZToLL_M1to10 is uncorrelated with ttll while in SR they would be correlated
def get_decorrelated_uncty(syst_name,grp_map,relevant_samples_lst,base_histo,template_zeros_arr):

    # Initialize the array we will return (ok technically we return sqrt of this arr squared..)
    a_arr_sum = np.zeros_like(template_zeros_arr) # Just using this template_zeros_arr for its size

    # Loop over the groups of processes, generally the processes in the groups will be correlated and the different groups will be uncorrelated
    for proc_grp in grp_map.keys():
        proc_lst = grp_map[proc_grp]
        if proc_grp in ["Nonprompt","Flips","Data"]: continue # Renorm and fact not relevant here
        if proc_lst == []: continue # Nothing here

        # We'll keep all signal processes as uncorrelated, similar to what's done in SR
        if proc_grp == "Signal":
            for proc_name in proc_lst:
                if proc_name not in relevant_samples_lst: continue

                n_arr_proc = base_histo.integrate("process",proc_name).integrate("systematic","nominal").eval({})[()]
                u_arr_proc = base_histo.integrate("process",proc_name).integrate("systematic",syst_name+"Up").eval({})[()]
                d_arr_proc = base_histo.integrate("process",proc_name).integrate("systematic",syst_name+"Down").eval({})[()]

                u_arr_proc_rel = u_arr_proc - n_arr_proc
                d_arr_proc_rel = d_arr_proc - n_arr_proc
                a_arr_proc_rel = (abs(u_arr_proc_rel) + abs(d_arr_proc_rel))/2.0

                a_arr_sum += a_arr_proc_rel*a_arr_proc_rel

        # Otherwise corrleated across groups (e.g. ZZ and WZ, as datacard maker does in SR)
        else:
            n_arr_grp = base_histo.integrate("process",proc_lst)[{"process": sum}].integrate("systematic","nominal").eval({})[()]
            u_arr_grp = base_histo.integrate("process",proc_lst)[{"process": sum}].integrate("systematic",syst_name+"Up").eval({})[()]
            d_arr_grp = base_histo.integrate("process",proc_lst)[{"process": sum}].integrate("systematic",syst_name+"Down").eval({})[()]
            u_arr_grp_rel = u_arr_grp - n_arr_grp
            d_arr_grp_rel = d_arr_grp - n_arr_grp
            a_arr_grp_rel = (abs(u_arr_grp_rel) + abs(d_arr_grp_rel))/2.0

            a_arr_sum += a_arr_grp_rel*a_arr_grp_rel

    # Before we move on, need to sqrt the outcome since later we'll square before adding in quadrature with other systs
    p_arr_rel =  np.sqrt(a_arr_sum)
    m_arr_rel = -np.sqrt(a_arr_sum)

    return [p_arr_rel,m_arr_rel]


# Get the squared arr for the jet dependent syst (e.g. for diboson jet dependent syst)
def get_diboson_njets_syst_arr(njets_histo_vals_arr,bin0_njets):

    # Get the list of njets vals for which we have SFs
    sf_int_lst = []
    diboson_njets_dict = grs.get_jet_dependent_syst_dict()
    sf_str_lst = list(diboson_njets_dict.keys())
    for s in sf_str_lst: sf_int_lst.append(int(s))
    min_njets = min(sf_int_lst) # The lowest njets bin we have a SF for
    max_njets = max(sf_int_lst) # The highest njets bin we have a SF for

    # Put the SFs into an array that matches the njets hist array
    sf_lst = []
    jet_idx = bin0_njets
    for idx in range(len(njets_histo_vals_arr)):
        if jet_idx < min_njets:
            # We do not apply the syst for these low jet bins
            sf_lst.append(1.0)
        elif jet_idx > max_njets:
            # For jet bins higher than the highest one in the dict, just use the val of the highest one
            sf_lst.append(diboson_njets_dict[str(max_njets)])
        else:
            # In this case, the exact jet bin should be included in the dict so use it directly
            sf_lst.append(diboson_njets_dict[str(jet_idx)])
        jet_idx = jet_idx + 1
    sf_arr = np.array(sf_lst)

    shift = abs(njets_histo_vals_arr - sf_arr*njets_histo_vals_arr)
    shift_sq = shift*shift # Return shift squared so we can combine with other syts in quadrature
    return shift*shift


######### Plotting functions #########

# Takes two histograms and makes a plot (with only one sparse axis, whihc should be "process"), one hist should be mc and one should be data
def make_cr_fig(h_mc,h_data,unit_norm_bool,axis='process',var='lj0pt',bins=[],group=[],set_x_lim=None,err_p=None,err_m=None,err_ratio_p=None,err_ratio_m=None,unblind=False):

    colors = ["tab:blue","darkgreen","tab:orange",'tab:cyan',"tab:purple","tab:pink","tan","mediumseagreen","tab:red","brown"]

    # Decide if we're plotting stat or syst uncty for mc
    # In principle would be better to combine them
    # But for our cases syst is way bigger than mc stat, so if we're plotting syst, ignore stat
    plot_syst_err = False
    mc_err_ops = MC_ERROR_OPS
    if (err_p is not None) and (err_m is not None) and (err_ratio_p is not None) and (err_ratio_m is not None):
        plot_syst_err = True
        mc_err_ops = None

    # Create the figure
    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10,10),
        gridspec_kw={"height_ratios": (3, 1)},
        sharex=True
    )
    fig.subplots_adjust(hspace=.07)

    # Set up the colors
    ax.set_prop_cycle(cycler(color=colors))

    # Normalize if we want to do that
    if unit_norm_bool:
        sum_mc = 0
        sum_data = 0
        for sample in h_mc.eval({}):
            sum_mc = sum_mc + sum(h_mc.eval({})[sample])
        for sample in h_data.eval({}):
            sum_data = sum_data + sum(h_data.eval({})[sample])
        h_mc.scale(1.0/sum_mc)
        h_data.scale(1.0/sum_data)

    # Plot the MC
    years = {}
    for axis_name in h_mc.axes[axis]:
        name = axis_name.split('UL')[0].replace('_private', '').replace('_central', '')
        if name in years:
            years[name].append(axis_name)
        else:
            years[name] = [axis_name]
    hep.style.use("CMS")
    plt.sca(ax)
    hep.cms.label(lumi='138')
    # Hack for grouping until fixed
    grouping = {proc: [good_proc for good_proc in group[proc] if good_proc in h_mc.axes['process']] for proc in group if any(p in h_mc.axes['process'] for p in group[proc])}
    if group:
        vals = [h_mc[{'process': grouping[proc]}][{'process': sum}].eval({})[()][1:-1] for proc in grouping]
        mc_vals = {proc: h_mc[{'process': grouping[proc]}][{'process': sum}].as_hist({}).values(flow=True)[1:] for proc in grouping}
    else:
        vals = [h_mc[{'process': proc}].eval({})[()][1:-1] for proc in grouping]
        mc_vals = {proc: h_mc[{'process': proc}].as_hist({}).values(flow=True)[1:] for proc in grouping}
    bins = h_data[{'process': sum}].as_hist({}).axes[var].edges
    bins = np.append(bins, [bins[-1] + (bins[-1] - bins[-2])*0.3])
    hep.histplot(
        list(mc_vals.values()),
        ax=ax,
        bins=bins,
        stack=True,
        density=unit_norm_bool,
        label=list(mc_vals.keys()),
        histtype='fill',
    )

    # Plot the data
    hep.histplot(
        h_data[{'process':sum}].as_hist({}).values(flow=True)[1:],
        #error_opts = DATA_ERR_OPS,
        ax=ax,
        bins=bins,
        stack=False,
        density=unit_norm_bool,
        label=('Data' if unblind else 'Asimov Data'),
        #flow='show',
        histtype='errorbar',
        **DATA_ERR_OPS,
    )

    # Make the ratio plot
    hep.histplot(
        (h_data[{'process':sum}].as_hist({}).values(flow=True)/h_mc[{"process": sum}].as_hist({}).values(flow=True))[1:],
        yerr=(np.sqrt(h_data[{'process':sum}].as_hist({}).values(flow=True)) / h_data[{'process':sum}].as_hist({}).values(flow=True))[1:],
        #error_opts = DATA_ERR_OPS,
        ax=rax,
        bins=bins,
        stack=False,
        density=unit_norm_bool,
        #flow='show',
        histtype='errorbar',
        **DATA_ERR_OPS,
    )

    # Plot the syst error
    if plot_syst_err:
        bin_edges_arr = h_mc.axes[var].edges
        #err_p = np.append(err_p,0) # Work around off by one error
        #err_m = np.append(err_m,0) # Work around off by one error
        #err_ratio_p = np.append(err_ratio_p,0) # Work around off by one error
        #err_ratio_m = np.append(err_ratio_m,0) # Work around off by one error
        ax.fill_between(bin_edges_arr,err_m,err_p, step='post', facecolor='none', edgecolor='gray', label='Syst err', hatch='////')
        rax.fill_between(bin_edges_arr,err_ratio_m,err_ratio_p,step='post', facecolor='none', edgecolor='gray', label='Syst err', hatch='////')
    err_m = np.append(h_mc[{'process': sum}].as_hist({}).values(flow=True)[1:]-np.sqrt(h_mc[{'process': sum}].as_hist({}).values(flow=True)[1:]), 1)
    err_p = np.append(h_mc[{'process': sum}].as_hist({}).values(flow=True)[1:]+np.sqrt(h_mc[{'process': sum}].as_hist({}).values(flow=True)[1:]), 1)
    err_ratio_m = np.append(1-1/np.sqrt(h_mc[{'process': sum}].as_hist({}).values(flow=True)[1:]), 1)
    err_ratio_p = np.append(1+1/np.sqrt(h_mc[{'process': sum}].as_hist({}).values(flow=True)[1:]), 1)
    rax.fill_between(bins,err_ratio_m,err_ratio_p,step='post', facecolor='none', edgecolor='gray', label='Stat err', hatch='////')

    # Scale the y axis and labels
    ax.autoscale(axis='y')
    ax.set_xlabel(None)
    rax.set_ylabel('Ratio', loc='center')
    rax.set_ylim(0.5,1.5)
    labels = [item.get_text() for item in rax.get_xticklabels()]
    labels[-1] = '>500'
    rax.set_xticklabels(labels)

    # Set the x axis lims
    if set_x_lim: plt.xlim(set_x_lim)

    ax.legend(ncol=3)
    return fig

# Takes a hist with one sparse axis and one dense axis, overlays everything on the sparse axis
def make_single_fig(histo,unit_norm_bool,axis=None,bins=[],group=[]):
    #print("\nPlotting values:",histo.eval({}))
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    hep.style.use("CMS")
    plt.sca(ax)
    hep.cms.label(lumi='138')
    if axis is None:
        hep.histplot(
            histo.eval({})[()][1:-1],
            ax=ax,
            bins=bins,
            stack=False,
            density=unit_norm_bool,
            #clear=False,
            histtype='fill',
        )
    else:
        for axis_name in histo.axes[axis]:
            print(axis_name)
            hep.histplot(
                histo[{axis: axis_name}].eval({})[()][1:-1],
                bins=bins,
                stack=True,
                density=unit_norm_bool,
                label=axis_name,
            )
    plt.legend()
    ax.autoscale(axis='y')
    return fig

# Takes a hist with one sparse axis (axis_name) and one dense axis, overlays everything on the sparse axis
# Makes a ratio of each cateogory on the sparse axis with respect to ref_cat
def make_single_fig_with_ratio(histo,axis_name,cat_ref,var='lj0pt',err_p=None,err_m=None,err_ratio_p=None,err_ratio_m=None):
    #print("\nPlotting values:",histo.eval({}))

    # Create the figure
    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7,7),
        gridspec_kw={"height_ratios": (3, 1)},
        sharex=True
    )
    fig.subplots_adjust(hspace=.07)

    # Make the main plot
    hep.histplot(
        histo,
        ax=ax,
        stack=False,
        clear=False,
    )

    # Make the ratio plot
    # TODO similar to L554
    #for cat_name in yt.get_cat_lables(histo,axis_name):
    #    hist.plotratio(
    #        num = histo.integrate(axis_name,cat_name),
    #        denom = histo.integrate(axis_name,cat_ref),
    #        ax = rax,
    #        unc = 'num',
    #        error_opts= {'linestyle': 'none','marker': '.', 'markersize': 10, 'elinewidth': 0},
    #        clear = False,
    #    )

    # Plot the syst error (if we have the necessary up/down variations)
    plot_syst_err = False
    if (err_p is not None) and (err_m is not None) and (err_ratio_p is not None) and (err_ratio_m is not None): plot_syst_err = True
    if plot_syst_err:
        bin_edges_arr = histo.axes[var].edges
        #err_p = np.append(err_p,0) # Work around off by one error
        #err_m = np.append(err_m,0) # Work around off by one error
        #err_ratio_p = np.append(err_ratio_p,0) # Work around off by one error
        #err_ratio_m = np.append(err_ratio_m,0) # Work around off by one error
        ax.fill_between(bin_edges_arr,err_m,err_p, step='post', facecolor='none', edgecolor='gray', label='Syst err', hatch='////')
        ax.set_ylim(0.0,1.2*max(err_p))
        rax.fill_between(bin_edges_arr,err_ratio_m,err_ratio_p,step='post', facecolor='none', edgecolor='gray', label='Syst err', hatch='////')

    # Style
    ax.set_xlabel('')
    rax.axhline(1.0,linestyle="-",color="k",linewidth=1)
    rax.set_ylabel('Ratio')
    rax.autoscale(axis='y')

    return fig



###################### Wrapper function for example SR plots with systematics ######################
# Wrapper function to loop over all SR categories and make plots for all variables
# Right now this function will only plot the signal samples
# By default, will make plots that show all systematics in the pkl file
def make_all_sr_sys_plots(dict_of_hists,year,save_dir_path):

    # If selecting a year, append that year to the wight list
    sig_wl = ["private"]
    if year is None: pass
    elif year == "2017": sig_wl.append("UL17")
    elif year == "2018": sig_wl.append("UL18")
    elif year == "2016": sig_wl.append("UL16") # NOTE: Right now this will plot both UL16 an UL16APV
    elif year == "2022": sig_wl.append("central2022")
    else: raise Exception

    # Get the list of samples to actually plot (finding sample list from first hist in the dict)
    all_samples = yt.get_cat_lables(dict_of_hists,"process",h_name=yt.get_hist_list(dict_of_hists)[0])
    sig_sample_lst = utils.filter_lst_of_strs(all_samples,substr_whitelist=sig_wl)
    if len(sig_sample_lst) == 0: raise Exception("Error: No signal samples to plot.")
    samples_to_rm_from_sig_hist = []
    for sample_name in all_samples:
        if sample_name not in sig_sample_lst:
            samples_to_rm_from_sig_hist.append(sample_name)
    print("\nAll samples:",all_samples)
    print("\nSig samples:",sig_sample_lst)
    print("\nAll systematics:",yt.get_cat_lables(dict_of_hists,"systematic",h_name=yt.get_hist_list(dict_of_hists)[0]))

    # Loop over hists and make plots
    skip_lst = [] # Skip this hist
    for idx,var_name in enumerate(dict_of_hists.keys()):
        if 'sumw2' in var_name: continue
        if yt.is_split_by_lepflav(dict_of_hists): raise Exception("Not set up to plot lep flav for SR, though could probably do it without too much work")
        if (var_name in skip_lst): continue
        if (var_name == "njets"):
            # We do not keep track of jets in the sparse axis for the njets hists
            sr_cat_dict = get_dict_with_stripped_bin_names(SR_CHAN_DICT,"njets")
        else:
            sr_cat_dict = SR_CHAN_DICT
        print("\nVar name:",var_name)
        print("sr_cat_dict:",sr_cat_dict)

        # Extract the signal hists
        hist_sig = dict_of_hists[var_name].remove("process", samples_to_rm_from_sig_hist)

        # If we only want to look at a subset of the systematics (Probably should be an option? For now, just uncomment if you want to use it)
        syst_subset_dict = {
            "nominal":["nominal"],
            "renormfactUp":["renormfactUp"],"renormfactDown":["renormfactDown"],
        }
        #hist_sig  = group_bins(hist_sig,syst_subset_dict,"systematic",drop_unspecified=True)

        # Make plots for each process
        for proc_name in sig_sample_lst:

            # Make a sub dir for this category
            save_dir_path_tmp = os.path.join(save_dir_path,proc_name)
            if not os.path.exists(save_dir_path_tmp):
                os.mkdir(save_dir_path_tmp)

            # Group categories
            hist_sig_grouped = group_bins(hist_sig,sr_cat_dict,"channel",drop_unspecified=True)

            # Make the plots
            for grouped_hist_cat in yt.get_cat_lables(hist_sig_grouped,axis="channel",h_name=var_name):

                # Integrate
                hist_sig_grouped_tmp = copy.deepcopy(hist_sig_grouped)
                hist_sig_grouped_tmp = yt.integrate_out_appl(hist_sig_grouped_tmp,grouped_hist_cat)
                hist_sig_grouped_tmp = hist_sig_grouped_tmp.integrate("process",proc_name[{'process': sum}])
                hist_sig_grouped_tmp = hist_sig_grouped_tmp.integrate("channel",grouped_hist_cat[{'channel': sum}])

                # Reweight (Probably should be an option? For now, just uncomment if you want to use it)
                #hist_sig_grouped_tmp.set_wilson_coefficients(**WCPT_EXAMPLE)

                # Make plots
                fig = make_single_fig_with_ratio(hist_sig_grouped_tmp,"systematic","nominal",var=var_name)
                title = proc_name+"_"+grouped_hist_cat+"_"+var_name
                fig.savefig(os.path.join(save_dir_path_tmp,title))

            # Make an index.html file if saving to web area
            if "www" in save_dir_path_tmp: make_html(save_dir_path_tmp)


###################### Wrapper function for simple plots ######################
# Wrapper function to loop over categories and make plots for all variables
def make_simple_plots(dict_of_hists,year,save_dir_path):

    all_samples = yt.get_cat_lables(dict_of_hists,"process",h_name="njets")

    for idx,var_name in enumerate(dict_of_hists.keys()):
        if 'sumw2' in var_name: continue
        #if var_name == "njets": continue
        #if "parton" in var_name: save_tag = "partonFlavour"
        #if "hadron" in var_name: save_tag = "hadronFlavour"
        #if "hadron" not in var_name: continue
        #if var_name != "j0hadronFlavour": continue
        if var_name != "j0partonFlavour": continue

        histo_orig = dict_of_hists[var_name]

        # Loop over channels
        channels_lst = yt.get_cat_lables(dict_of_hists[var_name],"channel")
        for chan_name in channels_lst:

            histo = copy.deepcopy(histo_orig)

            histo = yt.integrate_out_appl(histo,chan_name)
            histo = histo.integrate("systematic","nominal")
            histo = histo.integrate("channel",chan_name)

            print("\n",chan_name)
            print(histo.eval({}))
            summed_histo = histo[{"process": sum}]
            print("sum:",sum(summed_histo.eval({})[()]))
            continue

            # Make a sub dir for this category
            save_tag = "placeholder" # Flake8 pointed out that save_tag is not defined, should figure out why at some point if this function is ever used again
            save_dir_path_tmp = os.path.join(save_dir_path,save_tag)
            if not os.path.exists(save_dir_path_tmp):
                os.mkdir(save_dir_path_tmp)

            fig = make_single_fig(histo, unit_norm_bool=False)
            title = chan_name + "_" + var_name
            fig.savefig(os.path.join(save_dir_path_tmp,title))

            # Make an index.html file if saving to web area
            if "www" in save_dir_path: make_html(save_dir_path_tmp)


###################### Wrapper function for SR data and mc plots (unblind!) ######################
# Wrapper function to loop over all SR categories and make plots for all variables
def make_all_sr_data_mc_plots(dict_of_hists,year,save_dir_path,unblind=False,skip_syst_errs=False):

    # Construct list of MC samples
    mc_wl = []
    mc_bl = ["data"]
    data_wl = ["data"]
    data_bl = []
    if year is None:
        pass # Don't think we actually need to do anything here?
    elif year == "2017":
        mc_wl.append("UL17")
        data_wl.append("UL17")
    elif year == "2018":
        mc_wl.append("UL18")
        data_wl.append("UL18")
    elif year == "2016":
        mc_wl.append("UL16")
        mc_bl.append("UL16APV")
        data_wl.append("UL16")
        data_bl.append("UL16APV")
    elif year == "2016APV":
        mc_wl.append("UL16APV")
        data_wl.append("UL16APV")
    elif year == "2022":
        mc_wl.append("central2022")
        data_wl.append("2022")
    else:
        raise Exception(f"Error: Unknown year \"{year}\".")

    # Get the list of samples we want to plot
    samples_to_rm_from_mc_hist = []
    samples_to_rm_from_data_hist = []
    all_samples = yt.get_cat_lables(dict_of_hists,"process",h_name="lj0pt")
    mc_sample_lst = utils.filter_lst_of_strs(all_samples,substr_whitelist=mc_wl,substr_blacklist=mc_bl)
    data_sample_lst = utils.filter_lst_of_strs(all_samples,substr_whitelist=data_wl,substr_blacklist=data_bl)
    for sample_name in all_samples:
        if sample_name not in mc_sample_lst:
            samples_to_rm_from_mc_hist.append(sample_name)
        if sample_name not in data_sample_lst:
            samples_to_rm_from_data_hist.append(sample_name)
    print("\nAll samples:",all_samples)
    print("\nMC samples:",mc_sample_lst)
    print("\nData samples:",data_sample_lst)
    print("\nVariables:",dict_of_hists.keys())

    # Very hard coded :(
    for proc_name in mc_sample_lst + data_sample_lst:
        if "data" in proc_name:
            SR_GRP_MAP["Data"].append(proc_name)
        elif "nonprompt" in proc_name:
            SR_GRP_MAP["Nonprompt"].append(proc_name)
        elif "flips" in proc_name:
            SR_GRP_MAP["Flips"].append(proc_name)
        elif ("ttH" in proc_name):
            SR_GRP_MAP["ttH"].append(proc_name)
        elif ("ttlnu" in proc_name):
            SR_GRP_MAP["ttlnu"].append(proc_name)
        elif ("ttll" in proc_name) or ("TTZToLL_M1to10" in proc_name) or ("TTToSemiLeptonic" in proc_name) or ("TTTo2L2Nu" in proc_name):
            CR_GRP_MAP["Signal"].append(proc_name)
            SR_GRP_MAP["ttll"].append(proc_name)
        elif (("tllq" in proc_name) or ("tHq" in proc_name)):
            SR_GRP_MAP["tXq"].append(proc_name)
        elif ("tttt" in proc_name):
            SR_GRP_MAP["tttt"].append(proc_name)
        elif "ST" in proc_name or "tW" in proc_name or "tbarW" in proc_name or "TWZToLL" in proc_name:
            CR_GRP_MAP["Single top"].append(proc_name)
        elif "TTTo" in proc_name or "TTto" in proc_name:
            CR_GRP_MAP["Ttbar"].append(proc_name)
        elif "TTG" in proc_name:
            SR_GRP_MAP["Conv"].append(proc_name)
        elif "WWW" in proc_name or "WWZ" in proc_name or "WZZ" in proc_name or "ZZZ" in proc_name:
            SR_GRP_MAP["Multiboson"].append(proc_name)
        elif "WWTo2L2Nu" in proc_name or "ZZTo4L" in proc_name or "WZTo3LNu" in proc_name:
            SR_GRP_MAP["Diboson"].append(proc_name)
        elif "TWZ" in proc_name:
            SR_GRP_MAP["Multiboson"].append(proc_name)
        else:
            raise Exception(f"Error: Process name \"{proc_name}\" is not known.")

    # The analysis bins
    analysis_bins = {}
    # Skipping for now
    #    'njets': {
    #        '2l': [4,5,6,7,dict_of_hists['njets'].axis('njets').edges()[-1]], # Last bin in topeft.py is 10, this should grab the overflow
    #        '3l': [2,3,4,5,dict_of_hists['njets'].axis('njets').edges()[-1]],
    #        '4l': [2,3,4,dict_of_hists['njets'].axis('njets').edges()[-1]]
    #    }
    #}
    analysis_bins['ptz'] = axes_info['ptz']['variable']
    analysis_bins['lj0pt'] = axes_info['lj0pt']['variable']

    # Loop over hists and make plots
    skip_lst = ['ptz', 'njets'] # Skip this hist
    #keep_lst = ["njets","lj0pt","ptz","nbtagsl","nbtagsm","l0pt","j0pt"] # Skip all but these hists
    for idx,var_name in enumerate(dict_of_hists.keys()):
        if 'sumw2' in var_name: continue
        if (var_name in skip_lst): continue
        #if (var_name not in keep_lst): continue
        print("\nVariable:",var_name)

        # Extract the MC and data hists
        hist_mc_orig = dict_of_hists[var_name].remove("process", samples_to_rm_from_mc_hist)
        hist_data_orig = dict_of_hists[var_name].remove("process", samples_to_rm_from_data_hist)
        if not unblind: hist_data_orig = hist_mc_orig

        # Loop over channels
        channels_lst = yt.get_cat_lables(dict_of_hists[var_name],"channel")
        print("channels:",channels_lst)
        #for chan_name in channels_lst: # For each channel individually
        for chan_name in SR_CHAN_DICT.keys():
            #hist_mc = hist_mc_orig.integrate("systematic","nominal").integrate("channel",chan_name) # For each channel individually
            #hist_data = hist_data_orig.integrate("systematic","nominal").integrate("channel",chan_name) # For each channel individually
            # Skip missing channels (histEFT throws an exception)
            channels = [chan for chan in SR_CHAN_DICT[chan_name] if chan in hist_mc_orig.axes['channel']]
            if not channels:
                continue
            hist_mc = hist_mc_orig.integrate("systematic","nominal").integrate("channel",channels)[{'channel': sum}]
            channels = [chan for chan in SR_CHAN_DICT[chan_name] if chan in hist_data_orig.axes['channel']]
            hist_data = hist_data_orig.integrate("systematic","nominal").integrate("channel",channels)[{'channel': sum}]

            # Calculate the syst errors
            p_err_arr = None
            m_err_arr = None
            p_err_arr_ratio = None
            m_err_arr_ratio = None
            if not skip_syst_errs:
                # Get plus and minus rate and shape arrs
                rate_systs_summed_arr_m , rate_systs_summed_arr_p = get_rate_syst_arrs(hist_mc_orig[{"channel": channels}][{"channel": sum}], SR_GRP_MAP, group_type="SR")
                shape_systs_summed_arr_m , shape_systs_summed_arr_p = get_shape_syst_arrs(hist_mc_orig[{"channel": channels}][{"channel": sum}])
                if (var_name == "njets"):
                    # This is a special case for the diboson jet dependent systematic
                    db_hist = hist_mc_orig[{"process": SR_GRP_MAP["Diboson"], "channel": channels, "systematic": "nominal"}][{"process":  sum, "channel":  sum}].eval({})[()]
                    shape_systs_summed_arr_p = shape_systs_summed_arr_p + get_diboson_njets_syst_arr(db_hist,bin0_njets=0 + (1 if 'fwd' in proc_name else 0)) # Njets histos are assumed to start at njets=0
                    shape_systs_summed_arr_m = shape_systs_summed_arr_m + get_diboson_njets_syst_arr(db_hist,bin0_njets=0 + (1 if 'fwd' in proc_name else 0)) # Njets histos are assumed to start at njets=0
                # Get the arrays we will actually put in the CR plot
                nom_arr_all = hist_mc_orig[{"process": sum, "channel":channels, "systematic": "nominal"}][{"channel": sum}].eval({})[()][1:]
                p_err_arr = nom_arr_all + np.sqrt(shape_systs_summed_arr_p + rate_systs_summed_arr_p)[1:] # This goes in the main plot
                m_err_arr = nom_arr_all - np.sqrt(shape_systs_summed_arr_m + rate_systs_summed_arr_m)[1:] # This goes in the main plot
                p_err_arr_ratio = np.where(nom_arr_all>0,p_err_arr/nom_arr_all,1) # This goes in the ratio plot
                m_err_arr_ratio = np.where(nom_arr_all>0,m_err_arr/nom_arr_all,1) # This goes in the ratio plot

            #print(var_name, chan_name, f'grouping {SR_GRP_MAP=}')
            # Using new grouping approach in plot functions
            #hist_mc = group_bins(hist_mc,SR_GRP_MAP,"process",drop_unspecified=False)
            #hist_data = group_bins(hist_data,SR_GRP_MAP,"process",drop_unspecified=False)

            # Make a sub dir for this category
            save_dir_path_tmp = os.path.join(save_dir_path,chan_name)
            if not os.path.exists(save_dir_path_tmp):
                os.mkdir(save_dir_path_tmp)

            # Rebin into analysis bins
            '''
            if var_name in analysis_bins.keys():
                lep_bin = chan_name[:2]
                # histEFT doesn't support rebinning for now
                if var_name == "njets":
                    hist_mc = hist_mc.rebin(var_name, hist.Bin(var_name,  hist_mc.axes[var_name].label, analysis_bins[var_name][lep_bin]))
                    hist_data = hist_data.rebin(var_name, hist.Bin(var_name,  hist_data.axes[var_name].label, analysis_bins[var_name][lep_bin]))
                else:
                    hist_mc = hist_mc.rebin(var_name, hist.Bin(var_name,  hist_mc.axes[var_name].label, analysis_bins[var_name]))
                    hist_data = hist_data.rebin(var_name, hist.Bin(var_name,  hist_data.axes[var_name].label, analysis_bins[var_name]))
            '''

            if not hist_mc.eval({}):
                print("Warning: empty mc histo, continuing")
                continue
            if not hist_data.eval({}):
                print("Warning: empty data histo, continuing")
                continue

            fig = make_cr_fig(hist_mc,
                              hist_data,
                              var=var_name,
                              unit_norm_bool=False,
                              bins=axes_info[var_name]['variable'],
                              group=SR_GRP_MAP,
                              unblind=unblind,
                              err_p = p_err_arr,
                              err_m = m_err_arr,
                              err_ratio_p = p_err_arr_ratio,
                              err_ratio_m = m_err_arr_ratio)
            if year is not None: year_str = year
            else: year_str = "ULall"
            title = chan_name + "_" + var_name + "_" + year_str
            fig.savefig(os.path.join(save_dir_path_tmp,title))

            # Make an index.html file if saving to web area
            if "www" in save_dir_path_tmp: make_html(save_dir_path_tmp)





###################### Wrapper function for example SR plots ######################
# Wrapper function to loop over all SR categories and make plots for all variables
# Right now this function will only plot the signal samples
# By default, will make two sets of plots: One with process overlay, one with channel overlay
def make_all_sr_plots(dict_of_hists,year,unit_norm_bool,save_dir_path,split_by_chan=True,split_by_proc=True,skip_syst_errs=False):

    # If selecting a year, append that year to the wight list
    sig_wl = ["private"]
    if year is None: pass
    elif year == "2017": sig_wl.append("UL17")
    elif year == "2018": sig_wl.append("UL18")
    elif year == "2016": sig_wl.append("UL16") # NOTE: Right now this will plot both UL16 an UL16APV
    elif year == "2022": sig_wl.append("central2022")
    else: raise Exception

    # Get the list of samples to actually plot (finding sample list from first hist in the dict)
    all_samples = yt.get_cat_lables(dict_of_hists,"process",h_name=yt.get_hist_list(dict_of_hists)[0])
    sig_sample_lst = utils.filter_lst_of_strs(all_samples,substr_whitelist=sig_wl)
    if len(sig_sample_lst) == 0: raise Exception("Error: No signal samples to plot.")
    samples_to_rm_from_sig_hist = []
    for sample_name in all_samples:
        if sample_name not in sig_sample_lst:
            samples_to_rm_from_sig_hist.append(sample_name)
    print("\nAll samples:",all_samples)
    print("\nSig samples:",sig_sample_lst)


    # Loop over hists and make plots
    skip_lst = [] # Skip this hist
    for idx,var_name in enumerate(dict_of_hists.keys()):
        #if yt.is_split_by_lepflav(dict_of_hists): raise Exception("Not set up to plot lep flav for SR, though could probably do it without too much work")
        if 'sumw2' in var_name: continue
        if (var_name in skip_lst): continue
        if (var_name == "njets"):
            continue
            # We do not keep track of jets in the sparse axis for the njets hists
            sr_cat_dict = get_dict_with_stripped_bin_names(SR_CHAN_DICT,"njets")
        else:
            sr_cat_dict = SR_CHAN_DICT
        print("\nVar name:",var_name)
        print("sr_cat_dict:",sr_cat_dict)

        # Extract the signal hists, and integrate over systematic axis
        hist_sig = dict_of_hists[var_name].remove("process", samples_to_rm_from_sig_hist)
        hist_sig = hist_sig.integrate("systematic","nominal")

        # Make plots for each SR category
        if split_by_chan:
            for hist_cat in SR_CHAN_DICT.keys():

                # Calculate the syst errors
                p_err_arr = None
                m_err_arr = None
                p_err_arr_ratio = None
                m_err_arr_ratio = None
                if not skip_syst_errs:
                    # Get plus and minus rate and shape arrs
                    rate_systs_summed_arr_m , rate_systs_summed_arr_p = get_rate_syst_arrs(hist_sig, CR_GRP_MAP)
                    shape_systs_summed_arr_m , shape_systs_summed_arr_p = get_shape_syst_arrs(hist_sig)
                    if (var_name == "njets"):
                        # This is a special case for the diboson jet dependent systematic
                        db_hist = hist_sig.integrate("process",CR_GRP_MAP["Diboson"])[{"process": sum}].integrate("systematic","nominal").eval({})[()]
                        shape_systs_summed_arr_p = shape_systs_summed_arr_p + get_diboson_njets_syst_arr(db_hist,bin0_njets=0 + (1 if 'fwd' in hist_cat else 0)) # Njets histos are assumed to start at njets=0
                        shape_systs_summed_arr_m = shape_systs_summed_arr_m + get_diboson_njets_syst_arr(db_hist,bin0_njets=0 + (1 if 'fwd' in hist_cat else 0)) # Njets histos are assumed to start at njets=0
                    # Get the arrays we will actually put in the CR plot
                    nom_arr_all = hist_sig[{"process": sum}].integrate("systematic","nominal").eval({})[()][1:]
                    p_err_arr = nom_arr_all + np.sqrt(shape_systs_summed_arr_p + rate_systs_summed_arr_p)[1:] # This goes in the main plot
                    m_err_arr = nom_arr_all - np.sqrt(shape_systs_summed_arr_m + rate_systs_summed_arr_m)[1:] # This goes in the main plot
                    p_err_arr_ratio = np.where(nom_arr_all>0,p_err_arr/nom_arr_all,1) # This goes in the ratio plot
                    m_err_arr_ratio = np.where(nom_arr_all>0,m_err_arr/nom_arr_all,1) # This goes in the ratio plot

                if ((var_name == "ptz") and ("3l" not in hist_cat)): continue

                # Make a sub dir for this category
                save_dir_path_tmp = os.path.join(save_dir_path,hist_cat)
                if not os.path.exists(save_dir_path_tmp):
                    os.mkdir(save_dir_path_tmp)

                # Integrate to get the SR category we want to plot
                hist_sig_integrated_ch = yt.integrate_out_appl(hist_sig,hist_cat)
                # Skip missing channels (histEFT throws an exception)
                channels = [chan for chan in sr_cat_dict[hist_cat] if chan in hist_sig_integrated_ch.axes['channel']]
                if not channels: # Skip empty channels
                    continue
                hist_sig_integrated_ch = hist_sig_integrated_ch.integrate("channel",channels)[{'channel': sum}]
                hist_sig_integrated_ch = hist_sig_integrated_ch.integrate("process")

                # Make the plots
                if not hist_sig_integrated_ch.eval({}):
                    print("Warning: empty mc histo, continuing")
                    continue
                fig = make_single_fig(hist_sig_integrated_ch,unit_norm_bool,bins=axes_info[var_name]['variable'])
                #fig = make_cr_fig(
                #    hist_sig_integrated_ch,
                #    hist_data_integrated,
                #    unit_norm_bool,
                #    var=var_name,
                #    group=group,#CR_GRP_MAP,
                #    bins=axes_info[var_name]['variable'],
                #    set_x_lim = x_range,
                #    err_p = p_err_arr,
                #    err_m = m_err_arr,
                #    err_ratio_p = p_err_arr_ratio,
                #    err_ratio_m = m_err_arr_ratio
                #)
                title = hist_cat+"_"+var_name
                if unit_norm_bool: title = title + "_unitnorm"
                fig.savefig(os.path.join(save_dir_path_tmp,title))

                # Make an index.html file if saving to web area
                if "www" in save_dir_path_tmp: make_html(save_dir_path_tmp)


        # Make plots for each process
        if split_by_proc:
            for proc_name in sig_sample_lst:

                # Make a sub dir for this category
                save_dir_path_tmp = os.path.join(save_dir_path,proc_name)
                if not os.path.exists(save_dir_path_tmp):
                    os.mkdir(save_dir_path_tmp)

                # Group categories
                # Using new grouping approach in plot functions
                #hist_sig_grouped = group_bins(hist_sig,sr_cat_dict,"channel",drop_unspecified=True)
                hist_sig_grouped = hist_sig

                # Make the plots
                # Using new grouping approach in plot functions
                #for grouped_hist_cat in yt.get_cat_lables(hist_sig_grouped,axis="channel",h_name=var_name):
                for grouped_hist_cat in sr_cat_dict:
                    if not any(cat in hist_sig_grouped.axes['channel'] for cat in sr_cat_dict[grouped_hist_cat]):
                        continue

                    # Integrate
                    hist_sig_grouped_tmp = copy.deepcopy(hist_sig_grouped)
                    hist_sig_grouped_tmp = yt.integrate_out_appl(hist_sig_grouped_tmp,grouped_hist_cat)
                    if proc_name not in list(hist_sig_grouped_tmp.axes["process"]):
                        print(f"Warning: mc histo missing {proc_name}, continuing")
                        continue
                    hist_sig_grouped_tmp = hist_sig_grouped_tmp.integrate("process",proc_name)
                    if not hist_sig_grouped_tmp.eval({}):
                        print("Warning: empty mc histo, continuing")
                        continue

                    # Make plots
                    fig = make_single_fig(hist_sig_grouped_tmp[{'channel': sr_cat_dict[grouped_hist_cat]}][{'channel': sum}],unit_norm_bool,bins=axes_info[var_name]['variable'])
                    #fig = make_cr_fig(
                    #    hist_mc_integrated,
                    #    hist_data_integrated,
                    #    unit_norm_bool,
                    #    var=var_name,
                    #    group=group,#CR_GRP_MAP,
                    #    bins=axes_info[var_name]['variable'],
                    #    set_x_lim = x_range,
                    #    err_p = p_err_arr,
                    #    err_m = m_err_arr,
                    #    err_ratio_p = p_err_arr_ratio,
                    #    err_ratio_m = m_err_arr_ratio
                    #)
                    title = proc_name+"_"+grouped_hist_cat+"_"+var_name
                    if unit_norm_bool: title = title + "_unitnorm"
                    fig.savefig(os.path.join(save_dir_path_tmp,title))

                # Make an index.html file if saving to web area
                if "www" in save_dir_path_tmp: make_html(save_dir_path_tmp)



###################### Wrapper function for all CR plots ######################
# Wrapper function to loop over all CR categories and make plots for all variables
# The input hist should include both the data and MC
def make_all_cr_plots(dict_of_hists,year,skip_syst_errs,unit_norm_bool,save_dir_path):

    # Construct list of MC samples
    mc_wl = []
    mc_bl = ["data"]
    data_wl = ["data"]
    data_bl = []
    if year is None:
        pass # Don't think we actually need to do anything here?
    elif year == "2017":
        mc_wl.append("UL17")
        data_wl.append("UL17")
    elif year == "2018":
        mc_wl.append("UL18")
        data_wl.append("UL18")
    elif year == "2016":
        mc_wl.append("UL16")
        mc_bl.append("UL16APV")
        data_wl.append("UL16")
        data_bl.append("UL16APV")
    elif year == "2016APV":
        mc_wl.append("UL16APV")
        data_wl.append("UL16APV")
    elif year == "2022":
        mc_wl.append("central2022")
        data_wl.append("2022")
    else:
        raise Exception(f"Error: Unknown year \"{year}\".")

    # Get the list of samples we want to plot
    samples_to_rm_from_mc_hist = []
    samples_to_rm_from_data_hist = []
    all_samples = yt.get_cat_lables(dict_of_hists,"process")
    mc_sample_lst = utils.filter_lst_of_strs(all_samples,substr_whitelist=mc_wl,substr_blacklist=mc_bl)
    data_sample_lst = utils.filter_lst_of_strs(all_samples,substr_whitelist=data_wl,substr_blacklist=data_bl)
    for sample_name in all_samples:
        if sample_name not in mc_sample_lst:
            samples_to_rm_from_mc_hist.append(sample_name)
        if sample_name not in data_sample_lst:
            samples_to_rm_from_data_hist.append(sample_name)
    print("\nAll samples:",all_samples)
    print("\nMC samples:",mc_sample_lst)
    print("\nData samples:",data_sample_lst)
    print("\nVariables:",dict_of_hists.keys())

    # Fill group map (should we just fully hard code this?)
    for proc_name in all_samples:
        if "data" in proc_name:
            CR_GRP_MAP["Data"].append(proc_name)
        elif "nonprompt" in proc_name:
            CR_GRP_MAP["Nonprompt"].append(proc_name)
        elif "flips" in proc_name:
            CR_GRP_MAP["Flips"].append(proc_name)
        elif ("ttH" in proc_name) or ("ttlnu" in proc_name) or ("ttll" in proc_name) or ("tllq" in proc_name) or ("tHq" in proc_name) or ("tttt" in proc_name) or ("TTZToLL_M1to10" in proc_name):
            CR_GRP_MAP["Signal"].append(proc_name)
        elif "ST" in proc_name or "tW" in proc_name or "tbarW" in proc_name or "TWZToLL" in proc_name:
            CR_GRP_MAP["Single top"].append(proc_name)
        elif "DY" in proc_name:
            CR_GRP_MAP["DY"].append(proc_name)
        elif "TTG" in proc_name:
            CR_GRP_MAP["Conv"].append(proc_name)
        elif "TTTo" in proc_name or "TTto" in proc_name:
            CR_GRP_MAP["Ttbar"].append(proc_name)
        elif "ZGTo" in proc_name:
            CR_GRP_MAP["ZGamma"].append(proc_name)
        elif "WWW" in proc_name or "WWZ" in proc_name or "WZZ" in proc_name or "ZZZ" in proc_name:
            CR_GRP_MAP["Triboson"].append(proc_name)
        elif "WWTo2L2Nu" in proc_name or "ZZTo4L" in proc_name or "WZTo3LNu" in proc_name:
            CR_GRP_MAP["Diboson"].append(proc_name)
        elif "TWZ" in proc_name:
            CR_GRP_MAP["Diboson"].append(proc_name)
        elif "WJets" in proc_name:
            CR_GRP_MAP["Singleboson"].append(proc_name)
        else:
            raise Exception(f"Error: Process name \"{proc_name}\" is not known.")

    # Loop over hists and make plots
    skip_lst = [] # Skip these hists
    #skip_wlst = ["njets"] # Skip all but these hists
    for idx,var_name in enumerate(dict_of_hists.keys()):
        if 'sumw2' in var_name: continue
        if (var_name in skip_lst): continue
        #if (var_name not in skip_wlst): continue
        if (var_name == "njets"):
            # We do not keep track of jets in the sparse axis for the njets hists
            cr_cat_dict = get_dict_with_stripped_bin_names(CR_CHAN_DICT,"njets")
        else:  cr_cat_dict = CR_CHAN_DICT
        # If the hist is not split by lepton flavor, the lep flav info should not be in the channel names we try to integrate over
        if not yt.is_split_by_lepflav(dict_of_hists):
            cr_cat_dict = get_dict_with_stripped_bin_names(cr_cat_dict,"lepflav")
        print("\nVar name:",var_name)
        print("cr_cat_dict:",cr_cat_dict)

        # Extract the MC and data hists
        hist_mc = dict_of_hists[var_name].remove("process", samples_to_rm_from_mc_hist)
        hist_data = dict_of_hists[var_name].remove("process", samples_to_rm_from_data_hist)

        # Loop over the CR categories
        for hist_cat in cr_cat_dict.keys():
            if (hist_cat == "cr_2los_Z" and (("j0" in var_name) and ("lj0pt" not in var_name))): continue # The 2los Z category does not require jets (so leading jet plots do not make sense)
            if (hist_cat == "cr_2lss_flip" and (("j0" in var_name) and ("lj0pt" not in var_name))): continue # The flip category does not require jets (so leading jet plots do not make sense)
            print("\n\tCategory:",hist_cat)

            # Make a sub dir for this category
            save_dir_path_tmp = os.path.join(save_dir_path,hist_cat)
            if not os.path.exists(save_dir_path_tmp):
                os.mkdir(save_dir_path_tmp)

            # Integrate to get the categories we want
            axes_to_integrate_dict = {}
            axes_to_integrate_dict["channel"] = cr_cat_dict[hist_cat]
            hist_mc_integrated   = yt.integrate_out_cats(yt.integrate_out_appl(hist_mc,hist_cat)   ,axes_to_integrate_dict)[{"channel": sum}]
            hist_data_integrated = yt.integrate_out_cats(yt.integrate_out_appl(hist_data,hist_cat) ,axes_to_integrate_dict)[{"channel": sum}]

            # Remove samples that are not relevant for the given category
            samples_to_rm = []
            if hist_cat == "cr_2los_tt":
                samples_to_rm += copy.deepcopy(CR_GRP_MAP["Nonprompt"])
            hist_mc_integrated = hist_mc_integrated.remove("process", samples_to_rm)


            # Calculate the syst errors
            p_err_arr = None
            m_err_arr = None
            p_err_arr_ratio = None
            m_err_arr_ratio = None
            if not skip_syst_errs:
                # Get plus and minus rate and shape arrs
                rate_systs_summed_arr_m , rate_systs_summed_arr_p = get_rate_syst_arrs(hist_mc_integrated, CR_GRP_MAP)
                shape_systs_summed_arr_m , shape_systs_summed_arr_p = get_shape_syst_arrs(hist_mc_integrated)
                if (var_name == "njets"):
                    # This is a special case for the diboson jet dependent systematic
                    db_hist = hist_mc_integrated.integrate("process",CR_GRP_MAP["Diboson"])[{"process": sum}].integrate("systematic","nominal").eval({})[()]
                    shape_systs_summed_arr_p = shape_systs_summed_arr_p + get_diboson_njets_syst_arr(db_hist,bin0_njets=0 + (1 if 'fwd' in proc_name else 0)) # Njets histos are assumed to start at njets=0
                    shape_systs_summed_arr_m = shape_systs_summed_arr_m + get_diboson_njets_syst_arr(db_hist,bin0_njets=0 + (1 if 'fwd' in proc_name else 0)) # Njets histos are assumed to start at njets=0
                # Get the arrays we will actually put in the CR plot
                nom_arr_all = hist_mc_integrated[{"process": sum}].integrate("systematic","nominal").eval({})[()][1:]
                p_err_arr = nom_arr_all + np.sqrt(shape_systs_summed_arr_p + rate_systs_summed_arr_p)[1:] # This goes in the main plot
                m_err_arr = nom_arr_all - np.sqrt(shape_systs_summed_arr_m + rate_systs_summed_arr_m)[1:] # This goes in the main plot
                p_err_arr_ratio = np.where(nom_arr_all>0,p_err_arr/nom_arr_all,1) # This goes in the ratio plot
                m_err_arr_ratio = np.where(nom_arr_all>0,m_err_arr/nom_arr_all,1) # This goes in the ratio plot


            # Group the samples by process type, and grab just nominal syst category
            #hist_mc_integrated = group_bins(hist_mc_integrated,CR_GRP_MAP)
            #hist_data_integrated = group_bins(hist_data_integrated,CR_GRP_MAP)
            hist_mc_integrated = hist_mc_integrated.integrate("systematic","nominal")
            hist_data_integrated = hist_data_integrated.integrate("systematic","nominal")
            if hist_mc_integrated.empty():
                print(f'Empty {hist_mc_integrated=}')
                continue
            if hist_data_integrated.empty():
                print(f'Empty {hist_data_integrated=}')
                continue

            # Print out total MC and data and the sf between them
            # For extracting the factors we apply to the flip contribution
            # Probably should be an option not just a commented block...
            #if hist_cat != "cr_2lss_flip": continue
            #tot_data = sum(sum(hist_data_integrated.eval({}).eval({})))
            #tot_mc   = sum(sum(hist_mc_integrated.eval({}).eval({})))
            #flips    = sum(sum(hist_mc_integrated["Flips"].eval({}).eval({})))
            #tot_mc_but_flips = tot_mc - flips
            #sf = (tot_data - tot_mc_but_flips)/flips
            #print(f"\nComp: data/pred = {tot_data}/{tot_mc} = {tot_data/tot_mc}")
            #print(f"Flip sf needed = (data - (pred - flips))/flips = {sf}")
            #exit()

            # Create and save the figure
            x_range = None
            if var_name == "ht": x_range = (0,250)
            group = {k:v for k,v in CR_GRP_MAP.items() if v} # Remove empty groups
            fig = make_cr_fig(
                hist_mc_integrated,
                hist_data_integrated,
                unit_norm_bool,
                var=var_name,
                group=group,#CR_GRP_MAP,
                set_x_lim = x_range,
                err_p = p_err_arr,
                err_m = m_err_arr,
                err_ratio_p = p_err_arr_ratio,
                err_ratio_m = m_err_arr_ratio
            )
            title = hist_cat+"_"+var_name
            if unit_norm_bool: title = title + "_unitnorm"
            fig.savefig(os.path.join(save_dir_path_tmp,title))

            # Make an index.html file if saving to web area
            if "www" in save_dir_path_tmp: make_html(save_dir_path_tmp)


def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    parser.add_argument("-o", "--output-path", default=".", help = "The path the output files should be saved to")
    parser.add_argument("-n", "--output-name", default="plots", help = "A name for the output directory")
    parser.add_argument("-t", "--include-timestamp-tag", action="store_true", help = "Append the timestamp to the out dir name")
    parser.add_argument("-y", "--year", default=None, help = "The year of the sample")
    parser.add_argument("-u", "--unit-norm", action="store_true", help = "Unit normalize the plots")
    parser.add_argument("-s", "--skip-syst", default=False, action="store_true", help = "Skip syst errs in plots, only relevant for CR plots right now")
    args = parser.parse_args()

    # Whether or not to unit norm the plots
    unit_norm_bool = args.unit_norm

    # Make a tmp output directory in curren dir a different dir is not specified
    timestamp_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    save_dir_path = args.output_path
    outdir_name = args.output_name
    if args.include_timestamp_tag:
        outdir_name = outdir_name + "_" + timestamp_tag
    save_dir_path = os.path.join(save_dir_path,outdir_name)
    os.mkdir(save_dir_path)

    # Get the histograms
    hin_dict = utils.get_hist_from_pkl(args.pkl_file_path,allow_empty=False)

    # Print info about histos
    #yt.print_hist_info(args.pkl_file_path,"nbtagsl")
    #exit()

    # Make the plots
    #make_all_cr_plots(hin_dict,args.year,args.skip_syst,unit_norm_bool,save_dir_path)
    #make_all_sr_plots(hin_dict,args.year,unit_norm_bool,save_dir_path)
    # Blinded plots (Asimov data)
    make_all_sr_data_mc_plots(hin_dict,args.year,save_dir_path)
    # Unblinded plots (real data)
    #make_all_sr_data_mc_plots(hin_dict,args.year,save_dir_path,unblid=True)
    #make_all_sr_sys_plots(hin_dict,args.year,save_dir_path)
    #make_simple_plots(hin_dict,args.year,save_dir_path)

    # Make unblinded SR data MC comparison plots by year
    #make_all_sr_data_mc_plots(hin_dict,"2016",save_dir_path)
    #make_all_sr_data_mc_plots(hin_dict,"2016APV",save_dir_path)
    #make_all_sr_data_mc_plots(hin_dict,"2017",save_dir_path)
    #make_all_sr_data_mc_plots(hin_dict,"2018",save_dir_path)
    #make_all_sr_data_mc_plots(hin_dict,None,save_dir_path)

if __name__ == "__main__":
    main()
