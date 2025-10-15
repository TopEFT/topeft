##############################################################
# Script for creating the fake tau scale factors
#
# Statistical and systematic uncertainties on the fake-rate points are
# combined in quadrature, matching the expectation for uncorrelated
# sources and avoiding the overly conservative behaviour of the
# historical linear addition.
#
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
import json
import logging
import math
from collections import OrderedDict
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

from topeft.modules.paths import topeft_path
from topeft.modules.yield_tools import YieldTools
import topcoffea.modules.utils as utils

yt = YieldTools()


LOGGER = logging.getLogger(__name__)

_TAU_HISTOGRAM_REQUIRED_AXES = ("process", "channel", "systematic", "tau0pt")


def _extract_jet_suffix(jet_label):
    jet_digits = "".join(ch for ch in jet_label if ch.isdigit())
    if not jet_digits:
        raise RuntimeError(
            f"Unable to determine jet multiplicity from label '{jet_label}' in tau channel configuration."
        )
    return f"{jet_digits}j"


def _insert_flavor(base_name, flavor):
    if "_" not in base_name:
        return f"{base_name}_{flavor}"
    prefix, remainder = base_name.split("_", 1)
    return f"{prefix}_{flavor}_{remainder}"


def load_tau_control_channels(channels_json_path=None):
    """Build the Ftau and Ttau channel lists from the channel configuration JSON."""

    json_path = channels_json_path or topeft_path("channels/ch_lst.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Tau channel configuration JSON not found at '{json_path}'."
        )

    with open(json_path, "r") as ch_json:
        try:
            channel_config = json.load(ch_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse tau channel configuration at '{json_path}': {exc}."
            ) from exc

    try:
        tau_controls = channel_config["TAU_CH_LST_CR"]["2los_1tau"]
    except KeyError as exc:
        missing_key = exc.args[0] if exc.args else "TAU_CH_LST_CR"
        if missing_key == "TAU_CH_LST_CR":
            missing_path = "TAU_CH_LST_CR"
        else:
            missing_path = f"TAU_CH_LST_CR -> {missing_key}"
        raise RuntimeError(
            "The tau control-region definition is missing from the channel configuration. "
            f"Expected to find '{missing_path}' in '{json_path}'."
        ) from exc

    required_fields = ["lep_chan_lst", "lep_flav_lst", "jet_lst"]
    for field in required_fields:
        if field not in tau_controls:
            raise RuntimeError(
                f"Tau control-region definition in '{json_path}' is missing the '{field}' field."
            )

    lep_chans = tau_controls["lep_chan_lst"]
    lep_flavs = tau_controls["lep_flav_lst"]
    jet_bins = tau_controls["jet_lst"]

    if not lep_chans or not lep_flavs or not jet_bins:
        raise RuntimeError(
            "Tau control-region configuration must define non-empty channel, flavor, and jet lists."
        )

    ftau_channels = []
    ttau_channels = []

    for chan_def in lep_chans:
        if not chan_def:
            continue
        base_name = chan_def[0]
        if base_name.endswith("_Ftau"):
            target_list = ftau_channels
        elif base_name.endswith("_Ttau"):
            target_list = ttau_channels
        else:
            raise RuntimeError(
                f"Unexpected tau control channel name '{base_name}'. "
                "Expected names ending with '_Ftau' or '_Ttau'."
            )

        for flavor in lep_flavs:
            for jet_label in jet_bins:
                channel_name = f"{_insert_flavor(base_name, flavor)}_{_extract_jet_suffix(jet_label)}"
                target_list.append(channel_name)

    if not ftau_channels or not ttau_channels:
        raise RuntimeError(
            "Failed to build Ftau/Ttau channel lists from the tau control-region configuration."
        )

    return ftau_channels, ttau_channels

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


#def sqrt_list(numbers):
#    return [math.sqrt(num) for num in numbers]

def sqrt_list(numbers):
    arr = np.array(numbers.flatten())
    arr = np.clip(arr, 0, None)
    return arr.tolist()


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

def _ensure_list(values):
    if isinstance(values, str):
        return [values]
    return list(values)


def group_bins(histo, bin_map, axis_name="process", drop_unspecified=False):
    bin_map_copy = copy.deepcopy(bin_map)  # Avoid editing original
    normalized_map = OrderedDict(
        (group, _ensure_list(categories))
        for group, categories in bin_map_copy.items()
    )

    axis_categories = list(histo.axes[axis_name])
    axis_category_set = set(axis_categories)

    if not drop_unspecified:
        specified = {item for bins in normalized_map.values() for item in bins}
        for category in axis_categories:
            if category not in specified:
                normalized_map.setdefault(category, [category])

    requested = {item for bins in normalized_map.values() for item in bins}
    missing = sorted(requested - axis_category_set)
    if missing:
        raise ValueError(
            f"Requested {axis_name} bins not found in histogram: {', '.join(missing)}"
        )

    return histo.group(axis_name, normalized_map)

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

TAU_PT_BIN_START = 20
TAU_PT_BIN_STEP = 10
TAU_PT_BIN_DIVIDERS = [30, 40, 50, 60, 80, 100, 200]


def _accumulate_error(total, value):
    return total + value * value


def _finalize_error(total):
    return math.sqrt(total)


def _combine_ratio_uncertainty(num, num_err, den, den_err):
    if den == 0.0:
        return 0.0
    return math.sqrt((num_err / den) ** 2 + (num * den_err / (den ** 2)) ** 2)


def _combine_ratio_uncertainty_array(num, num_err, den, den_err):
    num = np.asarray(num, dtype=float)
    num_err = np.asarray(num_err, dtype=float)
    den = np.asarray(den, dtype=float)
    den_err = np.asarray(den_err, dtype=float)

    result = np.empty_like(num, dtype=float)
    result.fill(np.inf)
    valid = den != 0.0
    if not np.any(valid):
        return result

    result[valid] = np.sqrt(
        (num_err[valid] / den[valid]) ** 2
        + (num[valid] * den_err[valid] / (den[valid] ** 2)) ** 2
    )
    return result


def compute_fake_rates(
    fake_vals,
    fake_errs,
    tight_vals,
    tight_errs,
    start_index=2,
    bin_start=TAU_PT_BIN_START,
    bin_step=TAU_PT_BIN_STEP,
    bin_dividers=TAU_PT_BIN_DIVIDERS,
):
    fake_vals = np.asarray(fake_vals, dtype=float)
    fake_errs = np.asarray(fake_errs, dtype=float)
    tight_vals = np.asarray(tight_vals, dtype=float)
    tight_errs = np.asarray(tight_errs, dtype=float)

    x = bin_start
    fake_sum = 0.0
    tight_sum = 0.0
    fake_err_total = 0.0
    tight_err_total = 0.0
    ratios = []
    errors = []

    for index in range(start_index, len(fake_vals)):
        fake_sum += fake_vals[index]
        tight_sum += tight_vals[index]
        fake_err_total = _accumulate_error(fake_err_total, fake_errs[index])
        tight_err_total = _accumulate_error(tight_err_total, tight_errs[index])
        x += bin_step
        if x in bin_dividers:
            if fake_sum != 0.0:
                fake_err = _finalize_error(fake_err_total)
                tight_err = _finalize_error(tight_err_total)
                ratio = tight_sum / fake_sum
                ratio_err = _combine_ratio_uncertainty(
                    tight_sum, tight_err, fake_sum, fake_err
                )
            else:
                ratio = 0.0
                ratio_err = 0.0
            ratios.append(ratio)
            if ratio != 0.0 and (ratio + ratio_err) / ratio < 1.02:
                errors.append(1.02 * ratio - ratio)
            else:
                errors.append(ratio_err)
            fake_sum = 0.0
            tight_sum = 0.0
            fake_err_total = 0.0
            tight_err_total = 0.0

    return np.array(ratios, dtype=float), np.array(errors, dtype=float)


def _validate_histogram_axes(histogram, expected_axes, hist_name):
    """Ensure the histogram contains the expected axes for the tau fake-rate workflow."""

    present_axes = {axis.name for axis in histogram.axes}
    missing_axes = [axis for axis in expected_axes if axis not in present_axes]

    if missing_axes:
        available = ", ".join(sorted(present_axes)) if present_axes else "<none>"
        summary = (
            f"The '{hist_name}' histogram is missing required axes: {', '.join(missing_axes)}. "
            f"Available axes: {available}. "
            "Regenerate the histogram pickle with these axes enabled before running tauFitter."
        )
        LOGGER.error(summary)
        raise RuntimeError(summary)


def _validate_tau_channel_coverage(
    histogram,
    channel_axis_name,
    ftau_channels,
    ttau_channels,
    hist_name,
):
    """Verify that all Ftau/Ttau channels are present in the histogram's channel axis."""

    channel_axis = None
    for axis in histogram.axes:
        if axis.name == channel_axis_name:
            channel_axis = axis
            break

    if channel_axis is None:
        raise RuntimeError(
            f"Histogram '{hist_name}' does not define a '{channel_axis_name}' axis."
        )

    available_channels = {str(category) for category in channel_axis}

    missing_summary = {
        "Ftau": sorted(
            {channel for channel in ftau_channels if channel not in available_channels}
        ),
        "Ttau": sorted(
            {channel for channel in ttau_channels if channel not in available_channels}
        ),
    }

    missing_lines = [
        f"  {label}: {', '.join(channels)}"
        for label, channels in missing_summary.items()
        if channels
    ]

    if missing_lines:
        summary_lines = [
            f"The '{hist_name}' histogram is missing required tau control-region categories.",
            "Missing bins summary:",
            *missing_lines,
            "Available channel bins: "
            + (
                ", ".join(sorted(available_channels))
                if available_channels
                else "<none>"
            ),
            "Regenerate the histogram pickle with complete Ftau/Ttau coverage before rerunning tauFitter.",
        ]
        summary = "\n".join(summary_lines)
        LOGGER.error(summary)
        raise RuntimeError(summary)


def getPoints(dict_of_hists, ftau_channels, ttau_channels):
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
    tau_hist = dict_of_hists[var_name]

    _validate_histogram_axes(tau_hist, _TAU_HISTOGRAM_REQUIRED_AXES, var_name)
    _validate_tau_channel_coverage(
        tau_hist,
        "channel",
        ftau_channels,
        ttau_channels,
        var_name,
    )

    hist_mc = tau_hist.remove("process",samples_to_rm_from_mc_hist)
    hist_data = tau_hist.remove("process",samples_to_rm_from_data_hist)


    # Integrate to get the categories we want
    mc_fake     = hist_mc.integrate("channel", ftau_channels)
    mc_tight    = hist_mc.integrate("channel", ttau_channels)
    data_fake     = hist_data.integrate("channel", ftau_channels)
    data_tight    = hist_data.integrate("channel", ttau_channels)
  
    mc_fake     = group_bins(mc_fake,CR_GRP_MAP,"process",drop_unspecified=True)
    mc_tight    = group_bins(mc_tight,CR_GRP_MAP,"process",drop_unspecified=True)
    data_fake   = group_bins(data_fake,CR_GRP_MAP,"process",drop_unspecified=True)
    data_tight  = group_bins(data_tight,CR_GRP_MAP,"process",drop_unspecified=True)

    mc_fake     = mc_fake.integrate("systematic","nominal")
    mc_tight    = mc_tight.integrate("systematic","nominal")
    data_fake   = data_fake.integrate("systematic","nominal")

    data_tight  = data_tight.integrate("systematic","nominal")

    mc_fake_view = mc_fake.view()  # dictionary: keys are SparseHistTuple, values are arrays
    mc_tight_view = mc_tight.view()
    for key, vals in mc_fake_view.items():
        proc = key[0]
        chan = key[1]
        mc_fake_e = np.asarray(sqrt_list(vals), dtype=float)
        mc_fake_vals = np.asarray(vals, dtype=float)

    for key, vals in mc_tight_view.items():
        proc = key[0]
        chan = key[1]
        mc_tight_e = np.asarray(sqrt_list(vals), dtype=float)
        mc_tight_vals = np.asarray(vals, dtype=float)


    data_fake_view = data_fake.view()  # dictionary: keys are SparseHistTuple, values are arrays
    data_tight_view = data_tight.view()
    for key, vals in data_fake_view.items():
        proc = key[0]
        chan = key[1]
        data_fake_e = np.asarray(sqrt_list(vals), dtype=float)
        data_fake_vals = np.asarray(vals, dtype=float)

    for key, vals in data_tight_view.items():
        proc = key[0]
        chan = key[1]
        data_tight_e = np.asarray(sqrt_list(vals), dtype=float)
        data_tight_vals = np.asarray(vals, dtype=float)

    mc_y, mc_e = compute_fake_rates(
        mc_fake_vals,
        mc_fake_e,
        mc_tight_vals,
        mc_tight_e,
    )
    data_y, data_e = compute_fake_rates(
        data_fake_vals,
        data_fake_e,
        data_tight_vals,
        data_tight_e,
    )

    mc_x = np.array([TAU_PT_BIN_START] + TAU_PT_BIN_DIVIDERS[:-1], dtype=float)
    data_x = np.array([TAU_PT_BIN_START] + TAU_PT_BIN_DIVIDERS[:-1], dtype=float)

    return mc_x, mc_y, mc_e, data_x, data_y, data_e

def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    parser.add_argument(
        "--channels-json",
        default=None,
        help=(
            "Optional path to a channel configuration JSON file."
            " Defaults to topeft/channels/ch_lst.json."
        ),
    )
    parser.add_argument(
        "--dump-channels",
        nargs="?",
        const="-",
        metavar="OUTPUT",
        help=(
            "Dump the resolved Ftau/Ttau channel lists to stdout or the specified file."
            " The script continues after dumping."
        ),
    )
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

    ftau_channels, ttau_channels = load_tau_control_channels(args.channels_json)

    if args.dump_channels is not None:
        dump_payload = {"Ftau": ftau_channels, "Ttau": ttau_channels}
        if args.dump_channels in ("-", ""):
            json.dump(dump_payload, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            with open(args.dump_channels, "w") as dump_file:
                json.dump(dump_payload, dump_file, indent=2)
            print(f"Tau channel lists written to {args.dump_channels}")

    # Get the histograms
    hin_dict = utils.get_hist_from_pkl(args.pkl_file_path,allow_empty=False)
    x_mc,y_mc,yerr_mc,x_data,y_data,yerr_data = getPoints(
        hin_dict,
        ftau_channels,
        ttau_channels,
    )

    y_data = np.array(y_data, dtype=float).flatten()
    y_mc   = np.array(y_mc, dtype=float).flatten()
    yerr_data = np.array(yerr_data, dtype=float).flatten()
    yerr_mc   = np.array(yerr_mc, dtype=float).flatten()
    x_data    = np.array(x_data, dtype=float).flatten()


    print("fr data = ", y_data)
    print("fr mc = ", y_mc)
    SF = y_data/y_mc
    SF_e = _combine_ratio_uncertainty_array(
        y_data,
        yerr_data,
        y_mc,
        yerr_mc,
    )
        
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
