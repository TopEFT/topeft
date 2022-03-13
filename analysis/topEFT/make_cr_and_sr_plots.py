import os
import copy
import datetime
import argparse
import matplotlib.pyplot as plt
from cycler import cycler

from coffea import hist
from topcoffea.modules.HistEFT import HistEFT

from topcoffea.modules.YieldTools import YieldTools
from topcoffea.modules.GetValuesFromJsons import get_lumi
from topcoffea.plotter.make_html import make_html

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

    "cr_3l" : [
        "3l_eee_CR_1j",
        "3l_eem_CR_1j",
        "3l_emm_CR_1j",
        "3l_mmm_CR_1j",
    ],
}


SR_CHAN_DICT = {
    "2lss_4t_SR": [
        "2lss_4t_p_4j", "2lss_4t_m_5j", "2lss_4t_m_6j", "2lss_4t_m_7j",
        "2lss_4t_p_4j", "2lss_4t_p_5j", "2lss_4t_p_6j", "2lss_4t_p_7j",
    ],
    "2lss_SR" : [
        "2lss_m_4j", "2lss_m_5j", "2lss_m_6j", "2lss_m_7j",
        "2lss_p_4j", "2lss_p_5j", "2lss_p_6j", "2lss_p_7j",
    ],
    "3l_SR" : [
        "3l_m_offZ_1b_2j", "3l_m_offZ_1b_3j", "3l_m_offZ_1b_4j", "3l_m_offZ_1b_5j",
        "3l_m_offZ_2b_2j", "3l_m_offZ_2b_3j", "3l_m_offZ_2b_4j", "3l_m_offZ_2b_5j",
        "3l_p_offZ_1b_2j", "3l_p_offZ_1b_3j", "3l_p_offZ_1b_4j", "3l_p_offZ_1b_5j",
        "3l_p_offZ_2b_2j", "3l_p_offZ_2b_3j", "3l_p_offZ_2b_4j", "3l_p_offZ_2b_5j",
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


yt = YieldTools()

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

# Get a subset of the elements from a list of strings given a whitelist and/or blacklist of substrings
def filter_lst_of_strs(in_lst,substr_whitelist=[],substr_blacklist=[]):

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


# Figures out which year a sample is from, retruns the lumi for that year
def get_lumi_for_sample(sample_name):
    if "UL17" in sample_name:
        lumi = 1000.0*get_lumi("2017")
    elif "UL18" in sample_name:
        lumi = 1000.0*get_lumi("2018")
    elif "UL16APV" in sample_name:
        lumi = 1000.0*get_lumi("2016APV")
    elif "UL16" in sample_name:
        # Should not be here unless "UL16APV" not in sample_name
        lumi = 1000.0*get_lumi("2016")
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
    new_histo = histo.group(old_ax,new_ax,bin_map)

    return new_histo


# Takes two histograms and makes a plot (with only one sparse axis, whihc should be "sample"), one hist should be mc and one should be data
def make_cr_fig(h_mc,h_data,unit_norm_bool,set_x_lim=None):

    #colors = ['#e31a1c','#fb9a99','#a6cee3','#1f78b4','#b2df8a','#33a02c']
    colors = ["tab:blue","brown","tab:orange",'tan',"tab:purple","tab:pink","tab:cyan","tab:green","tab:red"]

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
        #overlay="sample",
        ax=ax,
        stack=True,
        line_opts=None,
        fill_opts=FILL_OPS,
        error_opts=MC_ERROR_OPS,
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
def make_single_fig_with_ratio(histo,axis_name,cat_ref):

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
    sig_sample_lst = filter_lst_of_strs(all_samples,substr_whitelist=sig_wl)
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
                #hist_sig_grouped_tmp.set_wilson_coefficients(**WCPT)

                # Make plots
                fig = make_single_fig_with_ratio(hist_sig_grouped_tmp,"systematic","nominal")
                title = proc_name+"_"+grouped_hist_cat+"_"+var_name
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
    sig_sample_lst = filter_lst_of_strs(all_samples,substr_whitelist=sig_wl)
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
def make_all_cr_plots(dict_of_hists,year,unit_norm_bool,save_dir_path):

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
        mc_wl.append("UL16")   # Includes UL16 and UL16APV
        data_wl.append("UL16") # Includes UL16 and UL16APV
    else: raise Exception(f"Error: Unknown year \"{year}\".")

    # Get the list of samples we want to plot
    samples_to_rm_from_mc_hist = []
    samples_to_rm_from_data_hist = []
    all_samples = yt.get_cat_lables(dict_of_hists,"sample")
    mc_sample_lst = filter_lst_of_strs(all_samples,substr_whitelist=mc_wl,substr_blacklist=mc_bl)
    data_sample_lst = filter_lst_of_strs(all_samples,substr_whitelist=data_wl,substr_blacklist=data_bl)
    for sample_name in all_samples:
        if sample_name not in mc_sample_lst:
            samples_to_rm_from_mc_hist.append(sample_name)
        if sample_name not in data_sample_lst:
            samples_to_rm_from_data_hist.append(sample_name)
    print("\nAll samples:",all_samples)
    print("\nMC samples:",mc_sample_lst)
    print("\nData samples:",data_sample_lst)

    # Fill group map (should we just fully hard code this?)
    for proc_name in all_samples:
        if "data" in proc_name:
            CR_GRP_MAP["Data"].append(proc_name)
        elif "nonprompt" in proc_name:
            CR_GRP_MAP["Nonprompt"].append(proc_name)
        elif "flips" in proc_name:
            CR_GRP_MAP["Flips"].append(proc_name)
        elif ("ttH" in proc_name) or ("ttlnu" in proc_name) or ("ttll" in proc_name) or ("tllq" in proc_name) or ("tHq" in proc_name) or ("tttt" in proc_name):
            CR_GRP_MAP["Signal"].append(proc_name)
        elif "ST" in proc_name or "tW" in proc_name or "tbarW" in proc_name:
            CR_GRP_MAP["Single top"].append(proc_name)
        elif "DY" in proc_name:
            CR_GRP_MAP["DY"].append(proc_name)
        elif "TTG" in proc_name:
            CR_GRP_MAP["Conv"].append(proc_name)
        elif "TTJets" in proc_name:
            CR_GRP_MAP["Ttbar"].append(proc_name)
        elif "WWW" in proc_name or "WWZ" in proc_name or "WZZ" in proc_name or "ZZZ" in proc_name:
            CR_GRP_MAP["Triboson"].append(proc_name)
        elif "WWTo2L2Nu" in proc_name or "ZZTo4L" in proc_name or "WZTo3LNu" in proc_name:
            CR_GRP_MAP["Diboson"].append(proc_name)
        elif "WJets" in proc_name:
            CR_GRP_MAP["Singleboson"].append(proc_name)
        else:
            raise Exception(f"Error: Process name \"{proc_name}\" is not known.")

    # Get the eft sum of weights at SM norm dict

    # Loop over hists and make plots
    skip_lst = [] # Skip this hist
    for idx,var_name in enumerate(dict_of_hists.keys()):
        if (var_name in skip_lst): continue
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

        # Group the samples by process type
        hist_mc = group_bins(hist_mc,CR_GRP_MAP)
        hist_data = group_bins(hist_data,CR_GRP_MAP)

        # Loop over the CR categories
        for hist_cat in cr_cat_dict.keys():
            if (hist_cat == "cr_2los_Z" and "j0" in var_name): continue # The 2los Z category does not require jets
            print("\n\tCategory:",hist_cat)

            # Make a sub dir for this category
            save_dir_path_tmp = os.path.join(save_dir_path,hist_cat)
            if not os.path.exists(save_dir_path_tmp):
                os.mkdir(save_dir_path_tmp)

            # Integrate to get the categories we want
            axes_to_integrate_dict = {}
            axes_to_integrate_dict["systematic"] = "nominal"
            axes_to_integrate_dict["channel"] = cr_cat_dict[hist_cat]
            hist_mc_integrated   = yt.integrate_out_cats(yt.integrate_out_appl(hist_mc,hist_cat)   ,axes_to_integrate_dict)
            hist_data_integrated = yt.integrate_out_cats(yt.integrate_out_appl(hist_data,hist_cat) ,axes_to_integrate_dict)

            # Remove samples that are not relevant for the given category
            if hist_cat == "cr_2los_tt":
                hist_mc_integrated = hist_mc_integrated.remove(["Nonprompt"],"sample")
            if hist_cat == "cr_2lss":
                hist_mc_integrated = hist_mc_integrated.remove(["Ttbar"],"sample")
                hist_mc_integrated = hist_mc_integrated.remove(["DY"],"sample")
            if hist_cat == "cr_3l":
                hist_mc_integrated = hist_mc_integrated.remove(["DY"],"sample")

            # Create and save the figure
            x_range = None
            if var_name == "ht": x_range = (0,250)
            fig = make_cr_fig(hist_mc_integrated,hist_data_integrated,unit_norm_bool,set_x_lim=x_range)
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
    hin_dict = yt.get_hist_from_pkl(args.pkl_file_path,allow_empty=False)

    # Print info about histos
    #yt.print_hist_info(args.pkl_file_path,"nbtagsl")
    #exit()

    # Make the plots
    #make_all_cr_plots(hin_dict,args.year,unit_norm_bool,save_dir_path)
    #make_all_sr_plots(hin_dict,args.year,unit_norm_bool,save_dir_path)
    make_all_sr_sys_plots(hin_dict,args.year,save_dir_path)

if __name__ == "__main__":
    main()
