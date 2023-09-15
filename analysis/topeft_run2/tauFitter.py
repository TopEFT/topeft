import numpy as np
import os
import copy
import datetime
import argparse
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

from topcoffea.modules.YieldTools import YieldTools
import topcoffea.modules.utils as utils

yt = YieldTools()

CR_CHAN_DICT = [
    "2los_CRtt_Ftau_2j",
    "2los_CRtt_Ttau_2j",
]

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


def getPoints(dict_of_hists):
    # Construct list of MC samples
    mc_wl = []
    mc_bl = ["data"]
    data_wl = ["data"]
    data_bl = []

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

    var_name = "taupt"
    cr_cat_dict = CR_CHAN_DICT
    #cr_cat_dict = get_dict_with_stripped_bin_names(cr_cat_dict,"lepflav")
    hist_mc = dict_of_hists[var_name].remove(samples_to_rm_from_mc_hist,"sample")
    hist_data = dict_of_hists[var_name].remove(samples_to_rm_from_data_hist,"sample")
    
    # Integrate to get the categories we want
    Fake_axes  = {}
    Fake_axes["channe"] = "2los_CRtt_Ftau_2j"
    Tight_axes = {}
    Tight_axes["channel"] = "2los_CRtt_Ttau_2j"
    mc_fake     = hist_mc.integrate("appl","isSR_2lOS").integrate("channel", "2los_CRtt_Ftau_2j")
    mc_tight    = hist_mc.integrate("appl","isSR_2lOS").integrate("channel", "2los_CRtt_Ttau_2j")
    data_fake     = hist_data.integrate("appl","isSR_2lOS").integrate("channel", "2los_CRtt_Ftau_2j")
    data_tight    = hist_data.integrate("appl","isSR_2lOS").integrate("channel", "2los_CRtt_Ttau_2j")
    mc_fake     = group_bins(mc_fake,CR_GRP_MAP)
    mc_tight    = group_bins(mc_tight,CR_GRP_MAP)
    data_fake   = group_bins(data_fake,CR_GRP_MAP)
    data_tight  = group_bins(data_tight,CR_GRP_MAP)
    mc_fake     = mc_fake.integrate("systematic","nominal")
    mc_tight    = mc_tight.integrate("systematic","nominal")
    data_fake   = data_fake.integrate("systematic","nominal")
    data_tight  = data_tight.integrate("systematic","nominal")

    for sample in mc_fake.values():
        mc_fake_vals  = mc_fake.values()[sample]
        mc_tight_vals = mc_tight.values()[sample]
        print(mc_fake.values()[sample])
        print(mc_tight.values()[sample])
    for sample in data_fake.values():
        print(data_fake.values()[sample])
        print(data_tight.values()[sample])
        data_fake_vals  = data_fake.values()[sample]
        data_tight_vals = data_tight.values()[sample]

    mc_x = []
    mc_y = []
    mc_e = []
    x = 0
    for index in range(2, len(mc_fake_vals)-12):
        fake  = mc_fake_vals[index]
        tight = mc_tight_vals[index]
        mc_x.append(x)
        x += 10
        if fake != 0.0:
            y = tight/fake
        else:
            y = 0
        mc_y.append(y)
        mc_e.append(1.02*y-y)
    data_x = []
    data_y = []
    data_e = []
    x = 0
    for index in range(2, len(data_fake_vals)-12):
        fake  = data_fake_vals[index]
        tight = data_tight_vals[index]
        data_x.append(x)
        x += 10
        if fake != 0.0:
            y = tight/fake
        else:
            y = 0
        data_y.append(y)
        data_e.append(1.02*y-y)
    return np.array(mc_x), np.array(mc_y), np.array(mc_e), np.array(data_x), np.array(data_y), np.array(data_e)

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
    x_data,y_data,yerr_data,x_mc,y_mc,yerr_mc = getPoints(hin_dict)

    print(y_data)
    print(y_mc)
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

if __name__ == "__main__":
    main()
