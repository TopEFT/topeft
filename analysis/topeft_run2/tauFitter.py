"""Tau fake-rate fitter utilities.

This module consumes histograms containing both the nominal yields and the
associated ``_sumw2`` accumulators (storing the sum of weights squared per bin).
The fake-rate and scale-factor uncertainties are derived directly from the
square root of the aggregated ``sumw2`` bins, so the input pkl must provide the
``tau0pt`` and ``tau0pt_sumw2`` histograms with matching axes.
"""

import os
import copy
import datetime
import argparse
import math
from cycler import cycler

#from coffea import hist
import hist

import sys
import re
import numpy as np
import matplotlib
#matplotlib.use('Qt4Agg')

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from  numpy.linalg import eig
from scipy.odr import *

from topeft.modules.yield_tools import YieldTools
import topcoffea.modules.utils as utils

yt = YieldTools()


Ftau = ["2los_ee_1tau_Ftau_2j", "2los_em_1tau_Ftau_2j", "2los_mm_1tau_Ftau_2j", "2los_ee_1tau_Ftau_3j", "2los_em_1tau_Ftau_3j", "2los_mm_1tau_Ftau_3j", "2los_ee_1tau_Ftau_4j", "2los_em_1tau_Ftau_4j", "2los_mm_1tau_Ftau_4j"]
Ttau = ["2los_ee_1tau_Ttau_2j", "2los_em_1tau_Ttau_2j", "2los_mm_1tau_Ttau_2j", "2los_ee_1tau_Ttau_3j", "2los_em_1tau_Ttau_3j", "2los_mm_1tau_Ttau_3j", "2los_ee_1tau_Ttau_4j", "2los_em_1tau_Ttau_4j", "2los_mm_1tau_Ttau_4j"]

#CR_GRP_MAP = {
#    "DY" : [],
#    "Ttbar" : [],
#    "Ttbarpowheg" : [],
#    "ZGamma" : [],
#    "Diboson" : [],
#    "Triboson" : [],
#    "Single top" : [],
#    "Singleboson" : [],
#    "Conv": [],
#    "Nonprompt" : [],
#    "Flips" : [],
#    "Signal" : [],
#    "Data" : [],
#}

CR_GRP_MAP = {
        "Data" : [],
        "Ttbar" : [],
    }

CR_GRP_MAP_full = {
    "DY": [
        "DYJetsToLL_MLL-50_central2022",
        "DYJetsToLL_MLL-10to50_central2022"
    ],
    "Ttbar": [
        "TTto2L2Nu_central2022",
        "TTtoLNu2Q_central2022",
        "TTto4Q_central2022",
        "TTLL_MLL-4to50_central2022"
    ],
    "ZGamma": [
        "ZGto2LG-1Jets_ntgc_5f_central2022"
    ],
    "Diboson": [
        "ZZTo4L_central2022",
        "WWTo2L2Nu_central2022",
        "WZTo3LNu_central2022",
        "WWZ_central2022",
        "WZZ_central2022",
        "ggToZZTo2mu2tau_central2022",
        "ggToZZTo2e2tau_central2022",
        "ggToZZTo4e_central2022",
        "ggToZZTo4mu_central2022",
        "ggToZZTo4tau_central2022"
    ],
    "Triboson": [
        "WWW_central2022",
        "ZZZ_central2022",
        "TWZ_Tto2Q_WtoLNu_Zto2L_central2022",
        "TWZ_TtoLNu_WtoLNu_Zto2L_central2022"
    ],
    "Single top": [
        "ST_tW_Leptonic_central2022",
        "ST_tW_Semileptonic_central2022",
        "ST_tbarW_Leptonic_central2022",
        "ST_tbarW_Semileptonic_central2022",
        "ST_top_s-channel_central2022",
        "ST_top_t-channel_central2022",
        "ST_antitop_t-channel_central2022"
    ],
    "Singleboson": [
        "WJetsToLNu_central2022"
    ],
    "TtG": [
        "TTG-1Jets_PTG-10to100_central2022",
        "TTG-1Jets_PTG-100to200_central2022",
        "TTG-1Jets_PTG-200_central2022"
    ],
    "Nonprompt": ["nonprompt2022"],
    "Flips": ["flips2022"],
    "Signal": [
        "TWZ_TtoLNu_Wto2Q_Zto2L_central2022"  # fill with your actual signal samples if needed
    ],
    "Data": [
        "data2022"
    ]
}


def linear(x,a,b):
    return b*x+a

def linear2(B,x):
    return B[1]*x+B[0]

def SF_fit(SF,SF_e,x):

    params, cov = curve_fit(linear,x,SF,sigma=SF_e,absolute_sigma=True)
    return params[0],params[1], cov

def SF_fit_alt(SF,SF_e,x):
    x_err = [0.1]*len(x)
    linear_model = Model(linear2)
    data = RealData(x, SF, sx=x_err, sy=SF_e)
    odr = ODR(data, linear_model, beta0=[0.4, 0.4])
    out = odr.run()
    c0,c1,cov, = out.Output()
    return c0,c1,cov

def group_bins_original(histo,bin_map,axis_name="sample",drop_unspecified=True):

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

def group_bins(histo, bin_map, axis_name="process", drop_unspecified=False):
    bin_map = copy.deepcopy(bin_map)  # Avoid editing original

    axis_cats = list(histo.axes[axis_name])

    # Build new bin_map that only contains categories that exist in the hist
    new_bin_map = {}
    for grp_name, cat_list in bin_map.items():
        filtered = [c for c in cat_list if c in axis_cats]  # Only keep existing categories
        if filtered:
            new_bin_map[grp_name] = filtered

    if not drop_unspecified:
        specified_cats = [c for lst in new_bin_map.values() for c in lst]
        for cat in axis_cats:
            if cat not in specified_cats:
                new_bin_map[cat] = [cat]

    new_histo = histo.group(axis_name, new_bin_map)

    return new_histo

def unwrap(hist, flow=True):
    """
    Unwrap a coffea.hist.Hist or HistEFT object into numpy arrays for values and errors.
    """
    # If it's already a dict (from coffea), use it directly
    if isinstance(hist, dict):
        vals = list(hist.values())[0]
        vars_ = list(hist.values())[0]  # if variances already computed elsewhere
    else:
        vals = hist.values(flow=flow)
        vars_ = hist.variances(flow=flow)
        if isinstance(vals, dict):
            vals = list(vals.values())[0]
        if isinstance(vars_, dict):
            vars_ = list(vars_.values())[0]

    errs = np.sqrt(vars_)
    return vals, errs

def getPoints(dict_of_hists):
    # Construct list of MC samples
    mc_wl = []
    mc_bl = ["data"]
    data_wl = ["data"]
    data_bl = []

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

    for proc_name in all_samples:
        if "data" in proc_name:
            CR_GRP_MAP["Data"].append(proc_name)
        #elif "TTTo" in proc_name:
        #    CR_GRP_MAP["Ttbar"].append(proc_name)
        else:
            CR_GRP_MAP["Ttbar"].append(proc_name)

    var_name = "tau0pt"
    var_name_sumw2 = f"{var_name}_sumw2"
    if var_name_sumw2 not in dict_of_hists:
        raise KeyError(f"Missing required histogram '{var_name_sumw2}' in input")

    hist_mc_nominal = dict_of_hists[var_name].copy()
    hist_mc_sumw2 = dict_of_hists[var_name_sumw2].copy()
    hist_data_nominal = dict_of_hists[var_name].copy()
    hist_data_sumw2 = dict_of_hists[var_name_sumw2].copy()

    # Apply the common filtering and grouping directly for each histogram variant.
    mc_fake = hist_mc_nominal.copy()
    mc_fake = mc_fake.remove("process", samples_to_rm_from_mc_hist)
    mc_fake = mc_fake.integrate("channel", Ftau)
    mc_fake = group_bins(mc_fake, CR_GRP_MAP, "process", drop_unspecified=True)
    mc_fake = mc_fake.integrate("systematic", "nominal")

    mc_tight = hist_mc_nominal.copy()
    mc_tight = mc_tight.remove("process", samples_to_rm_from_mc_hist)
    mc_tight = mc_tight.integrate("channel", Ttau)
    mc_tight = group_bins(mc_tight, CR_GRP_MAP, "process", drop_unspecified=True)
    mc_tight = mc_tight.integrate("systematic", "nominal")

    mc_fake_sumw2 = hist_mc_sumw2.copy()
    mc_fake_sumw2 = mc_fake_sumw2.remove("process", samples_to_rm_from_mc_hist)
    mc_fake_sumw2 = mc_fake_sumw2.integrate("channel", Ftau)
    mc_fake_sumw2 = group_bins(mc_fake_sumw2, CR_GRP_MAP, "process", drop_unspecified=True)
    mc_fake_sumw2 = mc_fake_sumw2.integrate("systematic", "nominal")

    mc_tight_sumw2 = hist_mc_sumw2.copy()
    mc_tight_sumw2 = mc_tight_sumw2.remove("process", samples_to_rm_from_mc_hist)
    mc_tight_sumw2 = mc_tight_sumw2.integrate("channel", Ttau)
    mc_tight_sumw2 = group_bins(mc_tight_sumw2, CR_GRP_MAP, "process", drop_unspecified=True)
    mc_tight_sumw2 = mc_tight_sumw2.integrate("systematic", "nominal")

    data_fake = hist_data_nominal.copy()
    data_fake = data_fake.remove("process", samples_to_rm_from_data_hist)
    data_fake = data_fake.integrate("channel", Ftau)
    data_fake = group_bins(data_fake, CR_GRP_MAP, "process", drop_unspecified=True)
    data_fake = data_fake.integrate("systematic", "nominal")

    data_tight = hist_data_nominal.copy()
    data_tight = data_tight.remove("process", samples_to_rm_from_data_hist)
    data_tight = data_tight.integrate("channel", Ttau)
    data_tight = group_bins(data_tight, CR_GRP_MAP, "process", drop_unspecified=True)
    data_tight = data_tight.integrate("systematic", "nominal")

    data_fake_sumw2 = hist_data_sumw2.copy()
    data_fake_sumw2 = data_fake_sumw2.remove("process", samples_to_rm_from_data_hist)
    data_fake_sumw2 = data_fake_sumw2.integrate("channel", Ftau)
    data_fake_sumw2 = group_bins(data_fake_sumw2, CR_GRP_MAP, "process", drop_unspecified=True)
    data_fake_sumw2 = data_fake_sumw2.integrate("systematic", "nominal")

    data_tight_sumw2 = hist_data_sumw2.copy()
    data_tight_sumw2 = data_tight_sumw2.remove("process", samples_to_rm_from_data_hist)
    data_tight_sumw2 = data_tight_sumw2.integrate("channel", Ttau)
    data_tight_sumw2 = group_bins(data_tight_sumw2, CR_GRP_MAP, "process", drop_unspecified=True)
    data_tight_sumw2 = data_tight_sumw2.integrate("systematic", "nominal")

    # Collapse everything onto the tau0pt axis, summing over the remaining axes.
    hist_pairs = {
        "mc_fake": (mc_fake, mc_fake_sumw2),
        "mc_tight": (mc_tight, mc_tight_sumw2),
        "data_fake": (data_fake, data_fake_sumw2),
        "data_tight": (data_tight, data_tight_sumw2),
    }

    collapsed = {}
    for key, (hist_nominal, hist_sumw2) in hist_pairs.items():
        values = np.asarray(hist_nominal.values(flow=False), dtype=float)
        sumw2 = np.asarray(hist_sumw2.values(flow=False), dtype=float)

        if values.shape != sumw2.shape:
            raise ValueError("Nominal histogram and sumw2 histogram shapes do not match")

        axis_names = [axis.name for axis in hist_nominal.axes]
        # Sum over every axis except for tau0pt so we are left with a single dimension.
        for axis_name in list(axis_names):
            if axis_name == "tau0pt":
                continue
            axis_idx = axis_names.index(axis_name)
            values = values.sum(axis=axis_idx)
            sumw2 = sumw2.sum(axis=axis_idx)
            axis_names.pop(axis_idx)

        collapsed[key] = (values, sumw2)

    tau_edges = mc_fake.axes["tau0pt"].edges

    # Regroup the tau pt bins into the historical coarse boundaries while carrying sumw2.
    coarse_boundaries = [30, 40, 50, 60, 80, 100, 200]
    min_edge = 20

    regrouped = {}
    for key, (values, sumw2_vals) in collapsed.items():
        regrouped_values = []
        regrouped_sumw2 = []

        boundary_iter = iter(coarse_boundaries)
        current_boundary = next(boundary_iter, None)
        accumulator = 0.0
        accumulator_sumw2 = 0.0

        for _, high, val, var in zip(tau_edges[:-1], tau_edges[1:], values, sumw2_vals):
            if high <= min_edge:
                continue

            accumulator += val
            accumulator_sumw2 += var

            while current_boundary is not None and (high >= current_boundary or np.isclose(high, current_boundary)):
                regrouped_values.append(accumulator)
                regrouped_sumw2.append(accumulator_sumw2)
                accumulator = 0.0
                accumulator_sumw2 = 0.0
                current_boundary = next(boundary_iter, None)
                break

        if accumulator or accumulator_sumw2:
            regrouped_values.append(accumulator)
            regrouped_sumw2.append(accumulator_sumw2)

        regrouped[key] = (np.array(regrouped_values, dtype=float), np.array(regrouped_sumw2, dtype=float))

    mc_fake_vals, mc_fake_sumw2_vals = regrouped["mc_fake"]
    mc_tight_vals, mc_tight_sumw2_vals = regrouped["mc_tight"]
    data_fake_vals, data_fake_sumw2_vals = regrouped["data_fake"]
    data_tight_vals, data_tight_sumw2_vals = regrouped["data_tight"]

    # Compute fake rates and their uncertainties from the aggregated values.
    mc_fake_rates = []
    mc_fake_rate_errs = []
    for tight, tight_var, fake, fake_var in zip(mc_tight_vals, mc_tight_sumw2_vals, mc_fake_vals, mc_fake_sumw2_vals):
        if fake <= 0:
            mc_fake_rates.append(0.0)
            mc_fake_rate_errs.append(0.0)
            continue

        tight_err = math.sqrt(max(tight_var, 0.0))
        fake_err = math.sqrt(max(fake_var, 0.0))
        rate = tight / fake
        variance = (tight_err / fake) ** 2 + (tight * fake_err / (fake ** 2)) ** 2
        mc_fake_rates.append(rate)
        mc_fake_rate_errs.append(math.sqrt(max(variance, 0.0)))

    data_fake_rates = []
    data_fake_rate_errs = []
    for tight, tight_var, fake, fake_var in zip(data_tight_vals, data_tight_sumw2_vals, data_fake_vals, data_fake_sumw2_vals):
        if fake <= 0:
            data_fake_rates.append(0.0)
            data_fake_rate_errs.append(0.0)
            continue

        tight_err = math.sqrt(max(tight_var, 0.0))
        fake_err = math.sqrt(max(fake_var, 0.0))
        rate = tight / fake
        variance = (tight_err / fake) ** 2 + (tight * fake_err / (fake ** 2)) ** 2
        data_fake_rates.append(rate)
        data_fake_rate_errs.append(math.sqrt(max(variance, 0.0)))

    mc_fake_rates = np.array(mc_fake_rates, dtype=float)
    mc_fake_rate_errs = np.array(mc_fake_rate_errs, dtype=float)
    data_fake_rates = np.array(data_fake_rates, dtype=float)
    data_fake_rate_errs = np.array(data_fake_rate_errs, dtype=float)

    mc_x = np.array([20, 30, 40, 50, 60, 80, 100], dtype=float)
    data_x = np.array([20, 30, 40, 50, 60, 80, 100], dtype=float)

    return mc_x, mc_fake_rates, mc_fake_rate_errs, data_x, data_fake_rates, data_fake_rate_errs

def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    args = parser.parse_args()

    # Whether or not to unit norm the plots
    #unit_norm_bool = args.unit_norm

    # Make a tmp output directory in curren dir a different dir is not specified
    timestamp_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    #save_dir_path = args.output_path
    #outdir_name = args.output_name
    #if args.include_timestamp_tag:
    #    outdir_name = outdir_name + "_" + timestamp_tag
    #save_dir_path = os.path.join(save_dir_path,outdir_name)
    #os.mkdir(save_dir_path)

    # Get the histograms
    hin_dict = utils.get_hist_from_pkl(args.pkl_file_path,allow_empty=False)
    x_mc,y_mc,yerr_mc,x_data,y_data,yerr_data = getPoints(hin_dict)

    y_data = np.array(y_data, dtype=float).flatten()
    y_mc   = np.array(y_mc, dtype=float).flatten()
    yerr_data = np.array(yerr_data, dtype=float).flatten()
    yerr_mc   = np.array(yerr_mc, dtype=float).flatten()
    x_data    = np.array(x_data, dtype=float).flatten()


    print("fr data = ", y_data)
    print("fr mc = ", y_mc)
    with np.errstate(divide='ignore', invalid='ignore'):
        SF = np.divide(y_data, y_mc, out=np.zeros_like(y_data), where=y_mc != 0)
        sf_var = (np.divide(yerr_data, y_mc, out=np.zeros_like(yerr_data), where=y_mc != 0) ** 2 +
                  (np.divide(y_data * yerr_mc, y_mc**2, out=np.zeros_like(y_data), where=y_mc != 0)) ** 2)
    SF_e = np.sqrt(np.clip(sf_var, 0.0, None))
    # Guard against zero-uncertainty bins (e.g. empty high-pt tails) that
    # would otherwise make curve_fit's sigma division blow up.
    SF_e = np.maximum(SF_e, 1e-3)
    print('SF',SF)
    print('sfERR',SF_e)
    print('x',x_data)

    #fitting...
    c0,c1,cov = SF_fit(SF,SF_e,x_data)
    print(c0)
    print(c1)
    print(cov)


    eigenvalues, eigenvectors = eig(cov)
    print('eige',eigenvalues,eigenvectors)
    #eval y using fit:
    y_fit = c1*x_data+c0

    lv0 = np.sqrt(abs(eigenvalues.dot(eigenvectors[0])))
    lv1 = np.sqrt(abs(eigenvalues.dot(eigenvectors[1])))
    #systunc_up = (1 + lv0)*c0 + (1 + lv1)*c1*x_data
    #systunc_dn = (1 - lv0)*c0 + (1 - lv1)*c1*x_data
    ##systunc_1st_up =  (c0 + lv0) + c1*x_data
    ##systunc_1st_dn =  (c0 - lv0) + c1*x_data
    ##systunc_2nd_up =  c0 + (c1 + lv1)*x_data
    ##systunc_2nd_dn =  c0 + (c1 - lv1)*x_data
    l0 =  eigenvalues[0]
    l1 =  eigenvalues[1]
    v00 = eigenvectors[0][0]
    v01 = eigenvectors[0][1]
    v10 = eigenvectors[1][0]
    v11 = eigenvectors[1][1]
    print(l0,l1,v00,v01,v10,v11)
    perr = np.sqrt(np.diag(cov))
    print(perr)
    print(lv0,lv1)
    systunc_1st_up = c0 + np.sqrt(l0)*v00   +  (c1 + np.sqrt(l0)*v01)*x_data
    systunc_1st_dn = c0 - np.sqrt(l0)*v00   +  (c1 - np.sqrt(l0)*v01)*x_data
    systunc_2nd_up = c0 + np.sqrt(l1)*v10   +  (c1 + np.sqrt(l1)*v11)*x_data
    systunc_2nd_dn = c0 - np.sqrt(l1)*v10   +  (c1 - np.sqrt(l1)*v11)*x_data
    print('           c0,c1')
    print('nom',c0,c1)
    print('up1',c0 + np.sqrt(l0)*v00,(c1 + np.sqrt(l0)*v01))
    print('up2',c0 + np.sqrt(l1)*v10,(c1 + np.sqrt(l0)*v01))
    #c0 = 1.16534
    #c1 = -0.0017
    c2 = (c1 + np.sqrt(l0)*v01)
    c3 = np.sqrt(l0)*v00+c0
    bin_div = [30, 40, 50, 60, 80, 100, 200]
    for p in bin_div:
        print(p, " SF= ", c1*(p)+c0)
        print(p, " SFup = ", (1 + lv0)*c0 + (1 + lv1)*c1*p)
        print(p, " SFdown = ", (1 - lv0)*c0 + (1 - lv1)*c1*p)

if __name__ == "__main__":
    main()
