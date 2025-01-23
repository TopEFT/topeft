'''Helper function to do the filling of the histogram in the main processor. Needed for code clarity'''
#Use of "all_mask": Can be simply just "all_cuts_mask" or boolean & with other relevant masks (e.g. ISR/FSR mask)
#"suffix" is either empty string (default) or "_sumw2" for the sumw2 histograms
#"weights" and "eft_coeffs" should be arrays without any cuts applied

import numpy as np

def fill_histogram(hout, dense_axis_name, dense_axis_vals, ch_name, appl, histAxisName, wgt_fluct, weights, eft_coeffs, all_mask, suffix=""):
    """Helper function to fill histograms with appropriate parameters."""
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
