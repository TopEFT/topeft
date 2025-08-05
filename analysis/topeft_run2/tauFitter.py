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

from coffea import hist

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

def sqrt_list(numbers):
    return [math.sqrt(num) for num in numbers]

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
    #new_histo = group(histo, axis_name, axis_name, bin_map)
    new_histo = histo.group(axis_name, bin_map)

    return new_histo


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
    print(mc_sample_lst)
    print(data_sample_lst)
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
    #cr_cat_dict = CR_CHAN_DICT
    for sample in samples_to_rm_from_mc_hist:
        print(sample)
    for	sample in samples_to_rm_from_data_hist:
        print(sample)

    hist_mc = dict_of_hists[var_name].remove("process",samples_to_rm_from_mc_hist)
    hist_data = dict_of_hists[var_name].remove("process",samples_to_rm_from_data_hist)

    for ax in hist_mc.axes:
        print(f"{ax.name}: {type(ax)}")
    if hasattr(ax, "categories"):
        print(f"  categories: {ax.categories}")
    elif hasattr(ax, "edges"):
        print(f"  bin edges: {ax.edges}")

    # Integrate to get the categories we want
    mc_fake     = hist_mc.integrate("channel", Ftau)
    mc_tight    = hist_mc.integrate("channel", Ttau)
    data_fake     = hist_data.integrate("channel", Ftau)
    data_tight    = hist_data.integrate("channel", Ttau)
    mc_fake     = group_bins(mc_fake,CR_GRP_MAP,"channel",drop_unspecified=True)
    mc_tight    = group_bins(mc_tight,CR_GRP_MAP,"channel",drop_unspecified=True)
    data_fake   = group_bins(data_fake,CR_GRP_MAP,"channel",drop_unspecified=True)
    data_tight  = group_bins(data_tight,CR_GRP_MAP,"channel",drop_unspecified=True)

    mc_fake     = mc_fake.integrate("systematic","nominal")
    mc_tight    = mc_tight.integrate("systematic","nominal")
    data_fake   = data_fake.integrate("systematic","nominal")
    data_tight  = data_tight.integrate("systematic","nominal")

    #print(mc_tight.values(sumw2=True))

    #for sample in mc_fake._sumw2:
    for item in (mc_tight.values(sumw2=True)):
        mc_fake_e = sqrt_list(mc_fake.values(sumw2=True)[item][1])
        mc_tight_e = sqrt_list(mc_tight.values(sumw2=True)[item][1])
    for sample in mc_tight.values():
        mc_fake_vals  = mc_fake.values()[sample]
        mc_tight_vals = mc_tight.values()[sample]
        print("mc fake = ", mc_fake.values()[sample])
        print("mc tight = ", mc_tight.values()[sample])
    for sample in data_fake.values():
        print("data fake = ", data_fake.values()[sample])
        print("data tight = ", data_tight.values()[sample])
        data_fake_vals  = data_fake.values()[sample]
        data_tight_vals = data_tight.values()[sample]
        data_fake_e = []
        data_tight_e = []
        for item in data_fake_vals:
            data_fake_e.append(math.sqrt(item*(1-(item/sum(data_fake_vals)))))
        for item in data_tight_vals:
            data_tight_e.append(math.sqrt(item*(1-(item/sum(data_tight_vals)))))
        

    mc_x = [20, 30, 40, 50, 60, 80, 100]
    mc_y = []
    mc_e = []
    x = 20
    bin_div = [30, 40, 50, 60, 80, 100, 200]
    fake = 0
    tight = 0
    f_err = 0
    t_err = 0
    print(mc_fake_e)
    print(mc_tight_e)
    for index in range(2, len(mc_fake_vals)):
        fake  += mc_fake_vals[index]
        tight += mc_tight_vals[index]
        f_err += mc_fake_e[index]
        t_err += mc_tight_e[index]
        x += 10
        if x in bin_div:
            if fake != 0.0:
                y = tight/fake
                y_err = t_err/fake + tight*f_err/(fake*fake)
            else:
                y = 0.0
                y_err = 0.0
            mc_y.append(y)
            if (y+y_err)/y < 1.02:
                mc_e.append(1.02*y-y)
            else:
                mc_e.append(y_err)
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
                y_err =t_err/fake + tight*f_err/(fake*fake)
            else:
                y = 0.0
                y_err =0.0
            data_y.append(y)
            if (y+y_err)/y < 1.02:
                data_e.append(1.02*y-y)
            else:
                data_e.append(y_err)
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
    #print("Available axes:", hin_dict)
    x_mc,y_mc,yerr_mc,x_data,y_data,yerr_data = getPoints(hin_dict)

    print("fr data = ", y_data)
    print("fr mc = ", y_mc)
    SF = y_data/y_mc
    SF_e = yerr_data/y_mc + y_data*yerr_mc/(y_mc**2)
        

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
    c0 = 1.16534
    c1 = -0.0017
    c2 = (c1 + np.sqrt(l0)*v01)
    c3 = np.sqrt(l0)*v00+c0
    bin_div = [30, 40, 50, 60, 80, 100, 200]
    for p in bin_div:
        print(p, " SF= ", c1*(p)+c0)
        print(p, " SFup = ", (1 + lv0)*c0 + (1 + lv1)*c1*p)
        print(p, " SFdown = ", (1 - lv0)*c0 + (1 - lv1)*c1*p)

if __name__ == "__main__":
    main()
