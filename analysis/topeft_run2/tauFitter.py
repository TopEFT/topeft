##############################################################
# Script for creating the fake tau scale factors
# To use, run command python tauFitter.py -f /path/to/pkl/file
# pkl file should have CRs listed below and have all other
# corrections aside from fake tau SFs
# output is in the form of linear fit y = mx+b
# where m and b are in numerical form, y is the SF, and x is the tau pt
# pt bins are from [20, 30], [30, 40], [40, 50], [50, 60], [60, 80], [80, 100], [100, 200]

import numpy as np
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

#Ftau = ["2los_ee_1tau_Ftau_2j"]
#Ttau = ["2los_ee_1tau_Ttau_2j"]

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


#def sqrt_list(numbers):
#    return [math.sqrt(num) for num in numbers]


def sqrt_list(numbers):
    return [math.sqrt(max(num, 0)) for num in numbers]

def sqrt_list2(numbers):
    numbers = np.array(numbers)
    return np.sqrt(np.clip(numbers, 0, None))

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

    bin_map = copy.deepcopy(bin_map)

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
    bin_map = copy.deepcopy(bin_map)

    axis_cats = list(histo.axes[axis_name])

    new_bin_map = {}
    for grp_name, cat_list in bin_map.items():
        filtered = [c for c in cat_list if c in axis_cats]
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
        else:
            CR_GRP_MAP["Ttbar"].append(proc_name)

    var_name = "tau0pt"

    hist_mc = dict_of_hists[var_name].remove("process",samples_to_rm_from_mc_hist)
    hist_data = dict_of_hists[var_name].remove("process",samples_to_rm_from_data_hist)

    hist_mc_sumw2 = dict_of_hists[var_name + '_sumw2'].remove("process",samples_to_rm_from_mc_hist)
    hist_data_sumw2 = dict_of_hists[var_name + '_sumw2'].remove("process",samples_to_rm_from_data_hist)


    mc_fake_all = None
    mc_fake_sumw2_all = None
    mc_tight_all = None
    mc_tight_sumw2_all = None

    for ftau in Ftau:
        mc_fake = hist_mc.integrate("channel", ftau)
        mc_fake_sumw2 = hist_mc_sumw2.integrate("channel", ftau)
        data_fake = hist_data.integrate("channel", ftau)
        data_fake_sumw2 = hist_data_sumw2.integrate("channel", ftau)

        mc_fake = group_bins(mc_fake, CR_GRP_MAP, "process", drop_unspecified=True)
        mc_fake_sumw2 = group_bins(mc_fake_sumw2, CR_GRP_MAP, "process", drop_unspecified=True)
        data_fake = group_bins(data_fake, CR_GRP_MAP, "process", drop_unspecified=True)
        data_fake_sumw2 = group_bins(data_fake_sumw2, CR_GRP_MAP, "process", drop_unspecified=True)

        mc_fake = mc_fake.integrate("systematic", "nominal")
        mc_fake_sumw2 = mc_fake_sumw2.integrate("systematic", "nominal")
        data_fake = data_fake.integrate("systematic", "nominal")
        data_fake_sumw2 = data_fake_sumw2.integrate("systematic", "nominal")

        if mc_fake_all is None:
            mc_fake_all = mc_fake.copy()
            mc_fake_sumw2_all = mc_fake_sumw2.copy()
            data_fake_all = data_fake.copy()
            data_fake_sumw2_all = data_fake_sumw2.copy()
        else:
            mc_fake_all += mc_fake
            mc_fake_sumw2_all += mc_fake_sumw2
            data_fake_all += data_fake
            data_fake_sumw2_all += data_fake_sumw2

    mc_fake_view = mc_fake_all.view()
    mc_fake_sumw2_view = mc_fake_sumw2_all.view()
    data_fake_view = data_fake_all.view()
    data_fake_sumw2_view = data_fake_sumw2_all.view()



    for ttau in Ttau:
        mc_tight = hist_mc.integrate("channel", ttau)
        mc_tight_sumw2 = hist_mc_sumw2.integrate("channel", ttau)
        data_tight = hist_data.integrate("channel", ttau)
        data_tight_sumw2 = hist_data_sumw2.integrate("channel", ttau)

        mc_tight = group_bins(mc_tight, CR_GRP_MAP, "process", drop_unspecified=True)
        mc_tight_sumw2 = group_bins(mc_tight_sumw2, CR_GRP_MAP, "process", drop_unspecified=True)
        data_tight = group_bins(data_tight, CR_GRP_MAP, "process", drop_unspecified=True)
        data_tight_sumw2 = group_bins(data_tight_sumw2, CR_GRP_MAP, "process", drop_unspecified=True)

        mc_tight = mc_tight.integrate("systematic", "nominal")
        mc_tight_sumw2 = mc_tight_sumw2.integrate("systematic", "nominal")
        data_tight = data_tight.integrate("systematic", "nominal")
        data_tight_sumw2 = data_tight_sumw2.integrate("systematic", "nominal")

        if mc_tight_all is None:
            mc_tight_all = mc_tight.copy()
            mc_tight_sumw2_all = mc_tight_sumw2.copy()
            data_tight_all = data_tight.copy()
            data_tight_sumw2_all = data_tight_sumw2.copy()
        else:
            mc_tight_all += mc_tight
            mc_tight_sumw2_all += mc_tight_sumw2
            data_tight_all += data_tight
            data_tight_sumw2_all += data_tight_sumw2


    mc_tight_view = mc_tight_all.view()
    mc_tight_sumw2_view = mc_tight_sumw2_all.view()
    data_tight_view = data_tight_all.view()
    data_tight_sumw2_view = data_tight_sumw2_all.view()

    for key, vals in mc_fake_view.items():
        mc_fake_vals = vals


    for key, vals in mc_fake_sumw2_view.items():
        mc_fake_e = sqrt_list(vals)


    for key, vals in mc_tight_view.items():
        mc_tight_vals = vals

    for key, vals in mc_tight_sumw2_view.items():
        mc_tight_e = sqrt_list(vals)

    for key, vals in data_fake_view.items():
        data_fake_vals = vals

    for key, vals in data_fake_sumw2_view.items():
        data_fake_e = sqrt_list(vals)

    for key, vals in data_tight_view.items():
        data_tight_vals = vals

    for key, vals in data_tight_sumw2_view.items():
        data_tight_e = sqrt_list(vals)

    mc_x = [20, 30, 40, 50, 60, 80, 100]
    mc_y = []
    mc_e = []
    x = 20
    bin_div = [30, 40, 50, 60, 80, 100, 200]
    fake = 0
    tight = 0
    f_err = 0
    t_err = 0

    for index in range(2, len(mc_fake_vals)):
        fake  += mc_fake_vals[index]
        tight += mc_tight_vals[index]
        f_err += mc_fake_e[index]
        t_err += mc_tight_e[index]
        x += 10
        if x in bin_div:
            if fake != 0.0:
                y = tight/fake
                #y_err = t_err/fake + tight*f_err/(fake*fake)
                y_err = math.sqrt((t_err / fake)**2 + (tight * f_err / (fake**2))**2)
            else:
                y = 0.0
                y_err = 0.0
            mc_y.append(y)
            if y != 0.0:
                if (y + y_err) / y < 1.02:
                    mc_e.append(1.02 * y - y)
                else:
                    mc_e.append(y_err)
            else:
                mc_e.append(0.0)
            fake = 0.0
            tight = 0.0
            f_err = 0.0
            t_err = 0.0
    data_x = [20, 30, 40, 50, 60, 80, 100]
    data_y = []
    data_e = []
    x = 20
    fake = 0.0
    tight = 0.0

    for index in range(2, len(data_fake_vals)):
        fake  += data_fake_vals[index]
        tight += data_tight_vals[index]
        f_err += data_fake_e[index]
        t_err += data_tight_e[index]
        x += 10
        if x in bin_div:
            if fake != 0.0:
                y = tight/fake
                print("check t/f: ", y)
                #y_err =t_err/fake + tight*f_err/(fake*fake)
                y_err = math.sqrt((t_err / fake)**2 + (tight * f_err / (fake**2))**2)

            else:
                y = 0.0
                y_err =0.0
            data_y.append(y)
            if y != 0.0:
                if (y + y_err) / y < 1.02:
                    data_e.append(1.02 * y - y)
                else:
                    data_e.append(y_err)
            else:
                data_e.append(0.0)
            fake = 0.0
            tight = 0.0
            f_err = 0.0
            t_err = 0.0
    return np.array(mc_x), np.array(mc_y), np.array(mc_e), np.array(data_x), np.array(data_y), np.array(data_e)

def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    args = parser.parse_args()

    timestamp_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    hin_dict = utils.get_hist_from_pkl(args.pkl_file_path,allow_empty=False)

    x_mc,y_mc,yerr_mc,x_data,y_data,yerr_data = getPoints(hin_dict)

    y_data = np.array(y_data, dtype=float).flatten()
    y_mc   = np.array(y_mc, dtype=float).flatten()
    yerr_data = np.array(yerr_data, dtype=float).flatten()
    yerr_mc   = np.array(yerr_mc, dtype=float).flatten()
    x_data    = np.array(x_data, dtype=float).flatten()


    print("fr data = ", y_data)
    print("fr mc = ", y_mc)
    print("d error", yerr_data)
    print("mc error", yerr_mc)
    print("x data", x_data)
    SF = y_data/y_mc
    #SF_e = yerr_data/y_mc + y_data*yerr_mc/(y_mc**2)
    SF_e = np.sqrt((yerr_data / y_mc)**2 + (y_data * yerr_mc / (y_mc**2))**2)
    
    SF_e = np.where(SF_e <= 0, 1e-3, SF_e)
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
