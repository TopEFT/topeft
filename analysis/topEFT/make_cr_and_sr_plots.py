import numpy as np
import os
import copy
import datetime
import argparse
import matplotlib.pyplot as plt
from cycler import cycler

from coffea import hist
from topcoffea.modules.HistEFT import HistEFT

from topcoffea.modules.YieldTools import YieldTools
import topcoffea.modules.GetValuesFromJsons as getj
from topcoffea.plotter.make_html import make_html
import topcoffea.modules.utils as utils

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
    "3l_offZ_SR" : [
        "3l_m_offZ_1b_2j", "3l_m_offZ_1b_3j", "3l_m_offZ_1b_4j", "3l_m_offZ_1b_5j",
        "3l_m_offZ_2b_2j", "3l_m_offZ_2b_3j", "3l_m_offZ_2b_4j", "3l_m_offZ_2b_5j",
        "3l_p_offZ_1b_2j", "3l_p_offZ_1b_3j", "3l_p_offZ_1b_4j", "3l_p_offZ_1b_5j",
        "3l_p_offZ_2b_2j", "3l_p_offZ_2b_3j", "3l_p_offZ_2b_4j", "3l_p_offZ_2b_5j",
    ],
    "3l_onZ_SR" : [
        "3l_onZ_1b_2j"   , "3l_onZ_1b_3j"   , "3l_onZ_1b_4j"   , "3l_onZ_1b_5j",
        "3l_onZ_2b_2j"   , "3l_onZ_2b_3j"   , "3l_onZ_2b_4j"   , "3l_onZ_2b_5j",
    ],
    "4l_SR" : [
        "4l_2j", "4l_3j", "4l_4j",
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
PROC_WITH_JUST_QCD_RATE_SYST = ["tttt","ttll","ttlnu","Triboson","tWZ"]

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
    return(out_chan_dict)


# Figures out which year a sample is from, retruns the lumi for that year
def get_lumi_for_sample(sample_name):
    if "UL17" in sample_name:
        lumi = 1000.0*getj.get_lumi("2017")
    elif "UL18" in sample_name:
        lumi = 1000.0*getj.get_lumi("2018")
    elif "UL16APV" in sample_name:
        lumi = 1000.0*getj.get_lumi("2016APV")
    elif "UL16" in sample_name:
        # Should not be here unless "UL16APV" not in sample_name
        lumi = 1000.0*getj.get_lumi("2016")
    else:
        raise Exception(f"Error: Unknown year \"{year}\".")
    return lumi

# Group bins in a hist, returns a new hist
def group_bins(histo,bin_map,axis_name="sample",drop_unspecified=False):

    bin_map = copy.deepcopy(bin_map) # Don't want to edit the original

    # Construct the map of bins to remap
    bins_to_remap_lst = []
    for grp_name,bins_in_grp in bin_map.items():
        bins_to_remap_lst.extend(bins_in_grp)
    if not drop_unspecified:
        for bin_name in yt.get_cat_lables(histo,axis_name):
            if bin_name not in bins_to_remap_lst:
                bin_map[bin_name] = bin_name

    # Remap the bins
    old_ax = histo.axis(axis_name)
    new_ax = hist.Cat(old_ax.name,old_ax.label)
    new_histo = histo.group(old_ax,new_ax,bin_map,overflow="over")

    return new_histo


######### Functions for getting info from the systematics json #########

# Match a given sample name to whatever it is called in the json
# Will return None if a match is not found
def get_scale_name(sample_name,sample_group_map):
    scale_name_for_json = None
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
    return scale_name_for_json

# This function gets the tag that indicates how a particualr systematic is correlated
#   - For pdf_scale this corresponds to the initial state (e.g. gg)
#   - For qcd_scale this corresponds to the process type (e.g. VV)
# For any systemaitc or process that is not included in the correlations json we return None
def get_correlation_tag(uncertainty_name,proc_name,sample_group_map):
    proc_name_in_json = get_scale_name(proc_name,sample_group_map)
    corr_tag = None
    # Right now we only have two types of uncorrelated rate systematics
    if uncertainty_name in ["qcd_scale","pdf_scale"]:
        if proc_name_in_json is not None:
            if proc_name_in_json == "convs":
                # Special case for conversions since we estimate from LO sample, we have _only_ pdf key not qcd
                # Would be better to handle this in a more general way
                corr_tag = None
            else:
                corr_tag = getj.get_correlation_tag(uncertainty_name,proc_name_in_json)
    return corr_tag

# This function gets all of the the rate systematics from the json file
# Returns a dictionary with all of the uncertainties
# If the sample does not have an uncertainty in the json, an uncertainty of 0 is returned for that category
def get_rate_systs(sample_name,sample_group_map):

    # Figure out the name of the appropriate sample in the syst rate json (if the proc is in the json)
    scale_name_for_json = get_scale_name(sample_name,sample_group_map)

    # Get the lumi uncty for this sample (same for all samles)
    lumi_uncty = getj.get_syst("lumi")

    # Get the flip uncty from the json (if there is not an uncertainty for this sample, return 1 since the uncertainties are multiplicative)
    if sample_name in sample_group_map["Flips"]:
        flip_uncty = getj.get_syst("charge_flips","charge_flips_sm")
    else:
        flip_uncty = [1.0,1,0]

    # Get the scale uncty from the json (if there is not an uncertainty for this sample, return 1 since the uncertainties are multiplicative)
    if scale_name_for_json is not None:
        if scale_name_for_json in PROC_WITH_JUST_QCD_RATE_SYST:
            # Special cases for when we do not have a pdf uncty (this is a really brittle workaround)
            # NOTE Someday should fix this, it's a really hardcoded and brittle and bad workaround
            pdf_uncty = [1.0,1,0]
        else:
            pdf_uncty = getj.get_syst("pdf_scale",scale_name_for_json)
        if scale_name_for_json == "convs":
            # Special case for conversions, since we estimate these from a LO sample, so we don't have an NLO uncty here
            # Would be better to handle this in a more general way
            qcd_uncty = [1.0,1,0]
        else:
            # In all other cases, use the qcd scale uncty that we have for the process
            qcd_uncty = getj.get_syst("qcd_scale",scale_name_for_json)
    else:
        pdf_uncty = [1.0,1,0]
        qcd_uncty = [1.0,1,0]

    out_dict = {"pdf_scale":pdf_uncty, "qcd_scale":qcd_uncty, "lumi":lumi_uncty, "charge_flips":flip_uncty}
    return out_dict


# Wrapper for getting plus and minus rate arrs
def get_rate_syst_arrs(base_histo,proc_group_map):

    # Fill dictionary with the rate uncertainty arrays (with correlated ones organized together)
    rate_syst_arr_dict = {}
    for rate_sys_type in getj.get_syst_lst():
        rate_syst_arr_dict[rate_sys_type] = {}
        for sample_name in yt.get_cat_lables(base_histo,"sample"):

            # Build the plus and minus arrays from the rate uncertainty number and the nominal arr
            rate_syst_dict = get_rate_systs(sample_name,proc_group_map)
            thissample_nom_arr = base_histo.integrate("sample",sample_name).integrate("systematic","nominal").values()[()]
            p_arr = thissample_nom_arr*(rate_syst_dict[rate_sys_type][1]) - thissample_nom_arr # Difference between positive fluctuation and nominal
            m_arr = thissample_nom_arr*(rate_syst_dict[rate_sys_type][0]) - thissample_nom_arr # Difference between positive fluctuation and nominal

            # Put the arrays into the correlation dict (organizing correlated ones together)
            correlation_tag = get_correlation_tag(rate_sys_type,sample_name,proc_group_map)
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
            syst_name_base = syst_var_name.replace("Up","")
            if syst_name_base not in syst_var_lst:
                syst_var_lst.append(syst_name_base)

    # Sum each systematic's contribtuions for all samples together (e.g. the ISR for all samples is summed linearly)
    p_arr_rel_lst = []
    m_arr_rel_lst = []
    for syst_name in syst_var_lst:
        relevant_samples_lst = yt.get_cat_lables(base_histo.integrate("systematic",syst_name+"Up"), "sample") # The samples relevant to this syst
        n_arr     = base_histo.integrate("sample",relevant_samples_lst).integrate("systematic","nominal").values()[()]        # Sum of all samples for nominal variation
        u_arr_sum = base_histo.integrate("sample",relevant_samples_lst).integrate("systematic",syst_name+"Up").values()[()]   # Sum of all samples for up variation
        d_arr_sum = base_histo.integrate("sample",relevant_samples_lst).integrate("systematic",syst_name+"Down").values()[()] # Sum of all samples for down variation
        u_arr_rel = u_arr_sum - n_arr # Diff with respect to nominal
        d_arr_rel = d_arr_sum - n_arr # Diff with respect to nominal
        p_arr_rel = np.where(u_arr_rel>0,u_arr_rel,d_arr_rel) # Just the ones that increase the yield
        m_arr_rel = np.where(u_arr_rel<0,u_arr_rel,d_arr_rel) # Just the ones that decrease the yield
        p_arr_rel_lst.append(p_arr_rel*p_arr_rel) # Square each element in the arr and append the arr to the out list
        m_arr_rel_lst.append(m_arr_rel*m_arr_rel) # Square each element in the arr and append the arr to the out list

    return [sum(m_arr_rel_lst), sum(p_arr_rel_lst)]

# Get the squared arr for the jet dependent syst (e.g. for diboson jet dependent syst)
def get_diboson_njets_syst_arr(njets_histo_vals_arr,bin0_njets):

    # Get the list of njets vals for which we have SFs
    sf_int_lst = []
    diboson_njets_dict = getj.get_jet_dependent_syst_dict()
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

# Takes two histograms and makes a plot (with only one sparse axis, whihc should be "sample"), one hist should be mc and one should be data
def make_cr_fig(h_mc,h_data,unit_norm_bool,set_x_lim=None,err_p=None,err_m=None,err_ratio_p=None,err_ratio_m=None):

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
        figsize=(7,7),
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
        for sample in h_mc.values():
            sum_mc = sum_mc + sum(h_mc.values()[sample])
        for sample in h_data.values():
            sum_data = sum_data + sum(h_data.values()[sample])
        h_mc.scale(1.0/sum_mc)
        h_data.scale(1.0/sum_data)

    # Plot the MC
    hist.plot1d(
        h_mc,
        ax=ax,
        stack=True,
        line_opts=None,
        fill_opts=FILL_OPS,
        error_opts=mc_err_ops,
        clear=False,
    )

    # Plot the data
    hist.plot1d(
        h_data,
        ax=ax,
        error_opts = DATA_ERR_OPS,
        stack=False,
        clear=False,
    )

    # Make the ratio plot
    hist.plotratio(
        num = h_data.sum("sample"),
        denom = h_mc.sum("sample"),
        ax = rax,
        error_opts = DATA_ERR_OPS,
        denom_fill_opts = {},
        guide_opts = {},
        unc = 'num',
        clear = False,
    )

    # Plot the syst error
    if plot_syst_err:
        dense_axes = h_mc.dense_axes()
        bin_edges_arr = h_mc.axis(dense_axes[0]).edges()
        err_p = np.append(err_p,0) # Work around off by one error
        err_m = np.append(err_m,0) # Work around off by one error
        err_ratio_p = np.append(err_ratio_p,0) # Work around off by one error
        err_ratio_m = np.append(err_ratio_m,0) # Work around off by one error
        ax.fill_between(bin_edges_arr,err_m,err_p, step='post', facecolor='none', edgecolor='gray', label='Syst err', hatch='////')
        rax.fill_between(bin_edges_arr,err_ratio_m,err_ratio_p,step='post', facecolor='none', edgecolor='gray', label='Syst err', hatch='////')

    # Scale the y axis and labels
    ax.autoscale(axis='y')
    ax.set_xlabel(None)
    rax.set_ylabel('Ratio')
    rax.set_ylim(0.5,1.5)

    # Set the x axis lims
    if set_x_lim: plt.xlim(set_x_lim)

    return fig

# Takes a hist with one sparse axis and one dense axis, overlays everything on the sparse axis
def make_single_fig(histo,unit_norm_bool):
    #print("\nPlotting values:",histo.values())
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    hist.plot1d(
        histo,
        stack=False,
        density=unit_norm_bool,
        clear=False,
    )
    ax.autoscale(axis='y')
    return fig

# Takes a hist with one sparse axis (axis_name) and one dense axis, overlays everything on the sparse axis
# Makes a ratio of each cateogory on the sparse axis with respect to ref_cat
def make_single_fig_with_ratio(histo,axis_name,cat_ref,err_p=None,err_m=None,err_ratio_p=None,err_ratio_m=None):
    #print("\nPlotting values:",histo.values())

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
    hist.plot1d(
        histo,
        ax=ax,
        stack=False,
        clear=False,
    )

    # Make the ratio plot
    for cat_name in yt.get_cat_lables(histo,axis_name):
        hist.plotratio(
            num = histo.integrate(axis_name,cat_name),
            denom = histo.integrate(axis_name,cat_ref),
            ax = rax,
            unc = 'num',
            error_opts= {'linestyle': 'none','marker': '.', 'markersize': 10, 'elinewidth': 0},
            clear = False,
        )

    # Plot the syst error (if we have the necessary up/down variations)
    plot_syst_err = False
    if (err_p is not None) and (err_m is not None) and (err_ratio_p is not None) and (err_ratio_m is not None): plot_syst_err = True
    if plot_syst_err:
        dense_axes = histo.dense_axes()
        bin_edges_arr = histo.axis(dense_axes[0]).edges()
        err_p = np.append(err_p,0) # Work around off by one error
        err_m = np.append(err_m,0) # Work around off by one error
        err_ratio_p = np.append(err_ratio_p,0) # Work around off by one error
        err_ratio_m = np.append(err_ratio_m,0) # Work around off by one error
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
    else: raise Exception

    # Get the list of samples to actually plot (finding sample list from first hist in the dict)
    all_samples = yt.get_cat_lables(dict_of_hists,"sample",h_name=yt.get_hist_list(dict_of_hists)[0])
    sig_sample_lst = yt.filter_lst_of_strs(all_samples,substr_whitelist=sig_wl)
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
        hist_sig = dict_of_hists[var_name].remove(samples_to_rm_from_sig_hist,"sample")

        # Normalize the hists
        sample_lumi_dict = {}
        for sample_name in sig_sample_lst:
            sample_lumi_dict[sample_name] = get_lumi_for_sample(sample_name)
        hist_sig.scale(sample_lumi_dict,axis="sample")

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
                hist_sig_grouped_tmp = hist_sig_grouped_tmp.integrate("sample",proc_name)
                hist_sig_grouped_tmp = hist_sig_grouped_tmp.integrate("channel",grouped_hist_cat)

                # Reweight (Probably should be an option? For now, just uncomment if you want to use it)
                #hist_sig_grouped_tmp.set_wilson_coefficients(**WCPT_EXAMPLE)

                # Make plots
                fig = make_single_fig_with_ratio(hist_sig_grouped_tmp,"systematic","nominal")
                title = proc_name+"_"+grouped_hist_cat+"_"+var_name
                fig.savefig(os.path.join(save_dir_path_tmp,title))

            # Make an index.html file if saving to web area
            if "www" in save_dir_path_tmp: make_html(save_dir_path_tmp)


###################### Wrapper function for simple plots ######################
# Wrapper function to loop over categories and make plots for all variables
def make_simple_plots(dict_of_hists,year,save_dir_path):

    all_samples = yt.get_cat_lables(dict_of_hists,"sample",h_name="njets")

    for idx,var_name in enumerate(dict_of_hists.keys()):
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

            # Normalize the MC hists
            sample_lumi_dict = {}
            for sample_name in all_samples:
                sample_lumi_dict[sample_name] = get_lumi_for_sample(sample_name)
            histo.scale(sample_lumi_dict,axis="sample")

            histo = yt.integrate_out_appl(histo,chan_name)
            histo = histo.integrate("systematic","nominal")
            histo = histo.integrate("channel",chan_name)

            print("\n",chan_name)
            print(histo.values())
            summed_histo = histo.sum("sample")
            print("sum:",sum(summed_histo.values()[()]))
            continue

            # Make a sub dir for this category
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
def make_all_sr_data_mc_plots(dict_of_hists,year,save_dir_path):

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
    else: raise Exception(f"Error: Unknown year \"{year}\".")

    # Get the list of samples we want to plot
    samples_to_rm_from_mc_hist = []
    samples_to_rm_from_data_hist = []
    all_samples = yt.get_cat_lables(dict_of_hists,"sample",h_name="lj0pt")
    mc_sample_lst = yt.filter_lst_of_strs(all_samples,substr_whitelist=mc_wl,substr_blacklist=mc_bl)
    data_sample_lst = yt.filter_lst_of_strs(all_samples,substr_whitelist=data_wl,substr_blacklist=data_bl)
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
        elif ("ttll" in proc_name):
            SR_GRP_MAP["ttll"].append(proc_name)
        elif (("tllq" in proc_name) or ("tHq" in proc_name)):
            SR_GRP_MAP["tXq"].append(proc_name)
        elif ("tttt" in proc_name):
            SR_GRP_MAP["tttt"].append(proc_name)
        elif "TTG" in proc_name:
            SR_GRP_MAP["Conv"].append(proc_name)
        elif "WWW" in proc_name or "WWZ" in proc_name or "WZZ" in proc_name or "ZZZ" in proc_name:
            SR_GRP_MAP["Multiboson"].append(proc_name)
        elif "WWTo2L2Nu" in proc_name or "ZZTo4L" in proc_name or "WZTo3LNu" in proc_name:
            SR_GRP_MAP["Multiboson"].append(proc_name)
        else:
            raise Exception(f"Error: Process name \"{proc_name}\" is not known.")

    # The analysis bins
    analysis_bins = {
        'njets': {
            '2l': [4,5,6,7,dict_of_hists['njets'].axis('njets').edges()[-1]], # Last bin in topeft.py is 10, this should grab the overflow
            '3l': [2,3,4,5,dict_of_hists['njets'].axis('njets').edges()[-1]],
            '4l': [2,3,4,dict_of_hists['njets'].axis('njets').edges()[-1]]
        }
    }
    analysis_bins['ptz'] = [0, 200, 300, 400, 500, dict_of_hists['ptz'].axis('ptz').edges()[-1]]
    analysis_bins['lj0pt'] = [0, 150, 250, 500, dict_of_hists['lj0pt'].axis('lj0pt').edges()[-1]]

    # Loop over hists and make plots
    skip_lst = [] # Skip this hist
    #keep_lst = ["njets","lj0pt","ptz","nbtagsl","nbtagsm","l0pt","j0pt"] # Skip all but these hists
    for idx,var_name in enumerate(dict_of_hists.keys()):
        if (var_name in skip_lst): continue
        #if (var_name not in keep_lst): continue
        print("\nVariable:",var_name)

        # Extract the MC and data hists
        hist_mc_orig = dict_of_hists[var_name].remove(samples_to_rm_from_mc_hist,"sample")
        hist_data_orig = dict_of_hists[var_name].remove(samples_to_rm_from_data_hist,"sample")

        # Loop over channels
        channels_lst = yt.get_cat_lables(dict_of_hists[var_name],"channel")
        print("channels:",channels_lst)
        #for chan_name in channels_lst: # For each channel individually
        for chan_name in SR_CHAN_DICT.keys():

            #hist_mc = hist_mc_orig.integrate("systematic","nominal").integrate("channel",chan_name) # For each channel individually
            #hist_data = hist_data_orig.integrate("systematic","nominal").integrate("channel",chan_name) # For each channel individually
            hist_mc = hist_mc_orig.integrate("systematic","nominal").integrate("channel",SR_CHAN_DICT[chan_name],overflow="over")
            hist_data = hist_data_orig.integrate("systematic","nominal").integrate("channel",SR_CHAN_DICT[chan_name],overflow="over")

            # Normalize the MC hists
            sample_lumi_dict = {}
            for sample_name in mc_sample_lst:
                sample_lumi_dict[sample_name] = get_lumi_for_sample(sample_name)
            hist_mc.scale(sample_lumi_dict,axis="sample")

            hist_mc = group_bins(hist_mc,SR_GRP_MAP)
            hist_data = group_bins(hist_data,SR_GRP_MAP)

            # Make a sub dir for this category
            save_dir_path_tmp = os.path.join(save_dir_path,chan_name)
            if not os.path.exists(save_dir_path_tmp):
                os.mkdir(save_dir_path_tmp)

            # Rebin into analysis bins
            if var_name in analysis_bins.keys():
                lep_bin = chan_name[:2]
                if var_name == "njets":
                    hist_mc = hist_mc.rebin(var_name, hist.Bin(var_name,  hist_mc.axis(var_name).label, analysis_bins[var_name][lep_bin]))
                    hist_data = hist_data.rebin(var_name, hist.Bin(var_name,  hist_data.axis(var_name).label, analysis_bins[var_name][lep_bin]))
                else:
                    hist_mc = hist_mc.rebin(var_name, hist.Bin(var_name,  hist_mc.axis(var_name).label, analysis_bins[var_name]))
                    hist_data = hist_data.rebin(var_name, hist.Bin(var_name,  hist_data.axis(var_name).label, analysis_bins[var_name]))

            if hist_mc.values() == {}:
                print("Warning: empty mc histo, continuing")
                continue
            if hist_data.values() == {}:
                print("Warning: empty data histo, continuing")
                continue

            fig = make_cr_fig(hist_mc, hist_data, unit_norm_bool=False)
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
def make_all_sr_plots(dict_of_hists,year,unit_norm_bool,save_dir_path,split_by_chan=True,split_by_proc=True):

    # If selecting a year, append that year to the wight list
    sig_wl = ["private"]
    if year is None: pass
    elif year == "2017": sig_wl.append("UL17")
    elif year == "2018": sig_wl.append("UL18")
    elif year == "2016": sig_wl.append("UL16") # NOTE: Right now this will plot both UL16 an UL16APV
    else: raise Exception

    # Get the list of samples to actually plot (finding sample list from first hist in the dict)
    all_samples = yt.get_cat_lables(dict_of_hists,"sample",h_name=yt.get_hist_list(dict_of_hists)[0])
    sig_sample_lst = yt.filter_lst_of_strs(all_samples,substr_whitelist=sig_wl)
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
        if (var_name in skip_lst): continue
        if (var_name == "njets"):
            # We do not keep track of jets in the sparse axis for the njets hists
            sr_cat_dict = get_dict_with_stripped_bin_names(SR_CHAN_DICT,"njets")
        else:
            sr_cat_dict = SR_CHAN_DICT
        print("\nVar name:",var_name)
        print("sr_cat_dict:",sr_cat_dict)

        # Extract the signal hists, and integrate over systematic axis
        hist_sig = dict_of_hists[var_name].remove(samples_to_rm_from_sig_hist,"sample")
        hist_sig = hist_sig.integrate("systematic","nominal")

        # Normalize the hists
        sample_lumi_dict = {}
        for sample_name in sig_sample_lst:
            sample_lumi_dict[sample_name] = get_lumi_for_sample(sample_name)
        hist_sig.scale(sample_lumi_dict,axis="sample")


        # Make plots for each SR category
        if split_by_chan:
            for hist_cat in SR_CHAN_DICT.keys(): 
                if ((var_name == "ptz") and ("3l" not in hist_cat)): continue

                # Make a sub dir for this category
                save_dir_path_tmp = os.path.join(save_dir_path,hist_cat)
                if not os.path.exists(save_dir_path_tmp):
                    os.mkdir(save_dir_path_tmp)

                # Integrate to get the SR category we want to plot
                hist_sig_integrated_ch = yt.integrate_out_appl(hist_sig,hist_cat)
                hist_sig_integrated_ch = hist_sig_integrated_ch.integrate("channel",sr_cat_dict[hist_cat])

                # Make the plots
                fig = make_single_fig(hist_sig_integrated_ch,unit_norm_bool)
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
                hist_sig_grouped = group_bins(hist_sig,sr_cat_dict,"channel",drop_unspecified=True)

                # Make the plots
                for grouped_hist_cat in yt.get_cat_lables(hist_sig_grouped,axis="channel",h_name=var_name):

                    # Integrate
                    hist_sig_grouped_tmp = copy.deepcopy(hist_sig_grouped)
                    hist_sig_grouped_tmp = yt.integrate_out_appl(hist_sig_grouped_tmp,grouped_hist_cat)
                    hist_sig_grouped_tmp = hist_sig_grouped_tmp.integrate("sample",proc_name)

                    # Make plots
                    fig = make_single_fig(hist_sig_grouped_tmp[grouped_hist_cat],unit_norm_bool)
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
    else: raise Exception(f"Error: Unknown year \"{year}\".")

    # Get the list of samples we want to plot
    samples_to_rm_from_mc_hist = []
    samples_to_rm_from_data_hist = []
    all_samples = yt.get_cat_lables(dict_of_hists,"sample")
    mc_sample_lst = yt.filter_lst_of_strs(all_samples,substr_whitelist=mc_wl,substr_blacklist=mc_bl)
    data_sample_lst = yt.filter_lst_of_strs(all_samples,substr_whitelist=data_wl,substr_blacklist=data_bl)
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
        elif "TTTo" in proc_name:
            CR_GRP_MAP["Ttbar"].append(proc_name)
        elif "ZGTo" in proc_name:
            CR_GRP_MAP["ZGamma"].append(proc_name)
        elif "WWW" in proc_name or "WWZ" in proc_name or "WZZ" in proc_name or "ZZZ" in proc_name:
            CR_GRP_MAP["Triboson"].append(proc_name)
        elif "WWTo2L2Nu" in proc_name or "ZZTo4L" in proc_name or "WZTo3LNu" in proc_name:
            CR_GRP_MAP["Diboson"].append(proc_name)
        elif "WJets" in proc_name:
            CR_GRP_MAP["Singleboson"].append(proc_name)
        else:
            raise Exception(f"Error: Process name \"{proc_name}\" is not known.")

    # Loop over hists and make plots
    skip_lst = [] # Skip these hists
    #skip_wlst = ["njets"] # Skip all but these hists
    for idx,var_name in enumerate(dict_of_hists.keys()):
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
        hist_mc = dict_of_hists[var_name].remove(samples_to_rm_from_mc_hist,"sample")
        hist_data = dict_of_hists[var_name].remove(samples_to_rm_from_data_hist,"sample")

        # Normalize the MC hists
        sample_lumi_dict = {}
        for sample_name in mc_sample_lst:
            sample_lumi_dict[sample_name] = get_lumi_for_sample(sample_name)
        hist_mc.scale(sample_lumi_dict,axis="sample")

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
            hist_mc_integrated   = yt.integrate_out_cats(yt.integrate_out_appl(hist_mc,hist_cat)   ,axes_to_integrate_dict)
            hist_data_integrated = yt.integrate_out_cats(yt.integrate_out_appl(hist_data,hist_cat) ,axes_to_integrate_dict)

            # Remove samples that are not relevant for the given category
            samples_to_rm = []
            if hist_cat == "cr_2los_tt":
                samples_to_rm += copy.deepcopy(CR_GRP_MAP["Nonprompt"])
            hist_mc_integrated = hist_mc_integrated.remove(samples_to_rm,"sample")


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
                    db_hist = hist_mc_integrated.integrate("sample",CR_GRP_MAP["Diboson"]).integrate("systematic","nominal").values()[()]
                    shape_systs_summed_arr_p = shape_systs_summed_arr_p + get_diboson_njets_syst_arr(db_hist,bin0_njets=0) # Njets histos are assumed to start at njets=0
                    shape_systs_summed_arr_m = shape_systs_summed_arr_m + get_diboson_njets_syst_arr(db_hist,bin0_njets=0) # Njets histos are assumed to start at njets=0
                # Get the arrays we will actually put in the CR plot
                nom_arr_all = hist_mc_integrated.sum("sample").integrate("systematic","nominal").values()[()]
                p_err_arr = nom_arr_all + np.sqrt(shape_systs_summed_arr_p + rate_systs_summed_arr_p) # This goes in the main plot
                m_err_arr = nom_arr_all - np.sqrt(shape_systs_summed_arr_m + rate_systs_summed_arr_m) # This goes in the main plot
                p_err_arr_ratio = np.where(nom_arr_all>0,p_err_arr/nom_arr_all,1) # This goes in the ratio plot
                m_err_arr_ratio = np.where(nom_arr_all>0,m_err_arr/nom_arr_all,1) # This goes in the ratio plot


            # Group the samples by process type, and grab just nominal syst category
            hist_mc_integrated = group_bins(hist_mc_integrated,CR_GRP_MAP)
            hist_data_integrated = group_bins(hist_data_integrated,CR_GRP_MAP)
            hist_mc_integrated = hist_mc_integrated.integrate("systematic","nominal")
            hist_data_integrated = hist_data_integrated.integrate("systematic","nominal")

            # Print out total MC and data and the sf between them
            # For extracting the factors we apply to the flip contribution
            # Probably should be an option not just a commented block...
            #if hist_cat != "cr_2lss_flip": continue
            #tot_data = sum(sum(hist_data_integrated.values().values()))
            #tot_mc   = sum(sum(hist_mc_integrated.values().values()))
            #flips    = sum(sum(hist_mc_integrated["Flips"].values().values()))
            #tot_mc_but_flips = tot_mc - flips
            #sf = (tot_data - tot_mc_but_flips)/flips
            #print(f"\nComp: data/pred = {tot_data}/{tot_mc} = {tot_data/tot_mc}")
            #print(f"Flip sf needed = (data - (pred - flips))/flips = {sf}")
            #exit()

            # Create and save the figure
            x_range = None
            if var_name == "ht": x_range = (0,250)
            fig = make_cr_fig(
                hist_mc_integrated,
                hist_data_integrated,
                unit_norm_bool,
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
    make_all_cr_plots(hin_dict,args.year,args.skip_syst,unit_norm_bool,save_dir_path)
    #make_all_sr_plots(hin_dict,args.year,unit_norm_bool,save_dir_path)
    #make_all_sr_data_mc_plots(hin_dict,args.year,save_dir_path)
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
