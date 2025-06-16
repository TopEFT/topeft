'''Helper function to do the filling of the histogram in the main processor. Needed for code clarity'''
#Use of "all_mask": Can be simply just "all_cuts_mask" or boolean & with other relevant masks (e.g. ISR/FSR mask for ZGamma samples)
#"suffix" is either empty string (default) or "_sumw2" for the sumw2 histograms
#"weights" and "eft_coeffs" should be arrays without any cuts applied

import numpy as np

#This is needed for ZGammaISR and ZGammaFSR split
#e.g. before modification it is ZGToLLG_centralUL17ISR, but we want it to be ZGToLLGISR_centralUL17
def modify_histAxisName(histAxisName):
    # Split at '_'
    parts = histAxisName.split('_')  # ['abc', 'centralUL17ISR']

    # Move "ISR" or "FSR" from the second part to the first part
    for tag in ["ISR","FSR"]:
        if tag in parts[1]:
            parts[0] += tag
            parts[1] = parts[1].replace(tag, "")

        # Join back together
        new_histAxisname = "_".join(parts)

    return new_histAxisname

def fill_1d_histogram(hout, dense_axis_name, dense_axis_vals, ch_name, appl, histAxisName, wgt_fluct, weights, eft_coeffs, all_mask, suffix=""):
    """Helper function to fill histograms with appropriate parameters."""

    #first, if the sample name has 'ISR' or 'FSR' string (only true for ZGamma sample), then we need to modify the sample name
    if "ISR" in histAxisName or "FSR" in histAxisName:
        histAxisName = modify_histAxisName(histAxisName)

    axes_fill_info_dict = {
        dense_axis_name + suffix: dense_axis_vals[all_mask],
        "channel": ch_name,
        "appl": appl,
        "process": histAxisName,
        "systematic": wgt_fluct,
        "weight": weights[all_mask] if not suffix.endswith("_sumw2") else np.square(weights[all_mask]),
        "eft_coeff": eft_coeffs[all_mask] if eft_coeffs is not None else None,
    }
    hout[dense_axis_name + suffix].fill(**axes_fill_info_dict)

#for example: hist_name is "photon_pt_eta", dense_axis1_name is "pt", and dense_axis2_name is "abseta"
def fill_2d_histogram(hout, hist_name, dense_axis1_name, dense_axis2_name, dense_axis1_vals, dense_axis2_vals, ch_name, appl, histAxisName, wgt_fluct, weights, eft_coeffs, all_mask, suffix=""):

    #first, if the sample name has 'ISR' or 'FSR' string (only true for ZGamma sample), then we need to modify the sample name
    if "ISR" in histAxisName or "FSR" in histAxisName:
        histAxisName = modify_histAxisName(histAxisName)

    axes_fill_info_dict = {
        dense_axis1_name + suffix: dense_axis1_vals[all_mask],
        dense_axis2_name + suffix: dense_axis2_vals[all_mask],
        "channel": ch_name,
        "appl": appl,
        "process": histAxisName,
        "systematic": wgt_fluct,
        "weight": weights[all_mask] if not suffix.endswith("_sumw2") else np.square(weights[all_mask]),
        "eft_coeff": eft_coeffs[all_mask] if eft_coeffs is not None else None,
    }
    hout[hist_name + suffix].fill(**axes_fill_info_dict)
