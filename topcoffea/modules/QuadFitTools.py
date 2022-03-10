import os
from coffea.nanoevents import NanoEventsFactory
import awkward as ak

import matplotlib.pyplot as plt
import numpy as np

import topcoffea.modules.fileReader as fr

# Some useful dictionaries (should these go somewhere else?)

# Maps WC names to latex formatting
WC_NAME_LATEX_MAP = {

    "ctW"  : "$c_{tW}$",
    "ctZ"  : "$c_{tZ}$",
    "ctp"  : "$c_{t\\varphi}$",
    "cpQM" : "$c^{-}_{\\varphi Q}$",
    "ctG"  : "$c_{tG}$",
    "cbW"  : "$c_{bW}$",
    "cpQ3" : "$c^{3}_{\\varphi Q}$",
    "cptb" : "$c_{\\varphi t b}$",
    "cpt"  : "$c_{\\varphi t}$",

    "cQl3i": "$c^{3(l)}_{Ql}$",
    "cQlMi": "$c^{M(l)}_{Ql}$",
    "cQei" : "$c^{l}_{Qe}$",
    "ctli" : "$c^{(l)}_{tl}$",
    "ctei" : "$c^{(l)}_{te}$",
    "ctlSi": "$c^{S(l)}_{t}$",
    "ctlTi": "$c^{T(l)}_{t}$",

    "cQq13": "$c^{3,1}_{Qq}$",
    "cQq83": "$c^{3,8}_{Qq}$",
    "cQq11": "$c^{1,1}_{Qq}$",
    "ctq1" : "$c^{1}_{tq}$",
    "cQq81": "$c^{1,8}_{Qq}$",
    "ctq8" : "$c^{8}_{tq}$",
}

# TOP-19-001 limits
TOP19001_LIMS = {
    "ctW"   : [-3.08, 2.87],
    "ctZ"   : [-3.32, 3.15],
    "ctp"   : [-16.98, 44.26],
    "cpQM"  : [-7.59, 21.65],
    "ctG"   : [-1.38, 1.18],
    "cbW"   : [-4.95, 4.95],
    "cpQ3"  : [-7.37, 3.48],
    "cptb"  : [-12.72, 12.63],
    "cpt"   : [-18.62, 12.31],
    "cQl3i" : [-9.67, 8.97],
    "cQlMi" : [-4.02, 4.99],
    "cQei"  : [-4.38, 4.59],
    "ctli"  : [-4.29, 4.82],
    "ctei"  : [-4.24, 4.86],
    "ctlSi" : [-6.52, 6.52],
    "ctlTi" : [-0.84, 0.84],
}

ARXIV1901_LIMS = {
    "ctG"  : [-0.4, 0.4],
    "ctW"  : [-1.8, 0.9],
    "cbW"  : [-2.6, 3.1],
    "ctZ"  : [-2.1, 4.0],
    "cptb" : [-27, 8.7],
    "cpQ3" : [-5.5, 5.8],
    "cpQM" : [-3.5, 3],
    "cpt"  : [-13, 18],
    "ctp"  : [-60, 10],
    "cQq13": [-1.1, 1.3],
    "cQq83": [-1.3, 1.6],
    "cQq11": [-6.8, 7.4],
    "ctq1" : [-5.3, 7.5],
    "cQq81": [-4.7, 7.8],
    "ctq8" : [-3.7, 4.1],
    "ctt1" : [-11.0,11.0],
    "cQQ1" : [-9.4,9.4],
    "cQt1" : [-13.0,12.0],
    "cQt8" : [-12.0,10.0]
}

FIT_RANGES = {  
    'ctW':(-4,4),     'ctZ':(-5,5),
    'cpt':(-40,30),   'ctp':(-35,65),
    'ctli':(-10,10),  'ctlSi':(-10,10),
    'cQl3i':(-10,10), 'cptb':(-20,20),
    'ctG':(-2,2),     'cpQM':(-10,30),
    'ctlTi':(-2,2),   'ctei':(-10,10),
    'cQei':(-10,10),  'cQlMi':(-10,10),
    'cpQ3':(-15,10),  'cbW':(-5,5),
    'cQq13': (-1,1),  'cQq83': (-2,2),
    'cQq11': (-2,2),  'ctq1': (-2,2),
    'cQq81': (-5,5),  'ctq8': (-5,5),
    'ctt1': (-5,5),   'cQQ1': (-10,10),
    'cQt8': (-20,20), 'cQt1': (-10,10)
}


########## Plotting tools ##########

# Takes two arrays and returnes a shifted differences
# E.g. for finding up/down fluctuations extrapolated beyond 1 sigma
def get_shifted_arr(y_arr_1,y_arr_2,x_arr,shift_factor):
    diff_arr = y_arr_1 - y_arr_2
    diff_arr = diff_arr*float(shift_factor)
    shift_arr = y_arr_2 + diff_arr
    return shift_arr

# Make a 1d plot
def make_1d_quad_plot(quad_params_dict,xaxis_name,yaxis_name,title,xaxis_range=[-10,10],save_dir=".",shift_var=None):

    # Get a string of the fit equation
    def get_fit_str(tag,xvar,s0,s1,s2):
        s0 = str(round(s0,2))
        s1 = str(round(s1,2))
        s2 = str(round(s2,2))
        rstr = f"{tag}: {s0} + {s1}*{xvar} + {s2}*{xvar}$^2$"
        return rstr

    # Make the figure, set the plot style
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.grid()
    ax.set_xlabel(WC_NAME_LATEX_MAP.get(xaxis_name,xaxis_name)) # Use the latex version of the wc name if we have it, otherwise just use the string directly
    ax.set_ylabel(yaxis_name)
    ax.set_title(title)

    # Get x and y arr from quad params
    ymax = 0
    ymin = 99999999
    quad_arr_dict = {}
    for key_name, quad_params in quad_params_dict.items():

        # Get the quad fit params from the list
        if len(quad_params) != 3:
            raise Exception(f"Error: Wrong number of parameters specified for 1d quadratic. Require 3, received {len(quad_params)}.")
        s0 = quad_params[0] 
        s1 = quad_params[1]
        s2 = quad_params[2]

        # Find x and y points
        x_arr = np.linspace(xaxis_range[0], xaxis_range[1], 1000)
        y_arr = s0 + s1*x_arr + s2*x_arr*x_arr

        # Keep track of overall max y
        ymax = max(ymax,max(y_arr))
        ymin = min(ymin,min(y_arr))

        quad_arr_dict[key_name] = [x_arr,y_arr]

    # Get shifted up/down array (if using this to plot nom/up/down)
    # Note: We can only do this if pass exactly three sets of quad params corresponding to nom, up, down
    if shift_var != None:
        if (len(quad_params_dict) !=3) or ("nom" not in quad_params_dict.keys()) or ("up" not in quad_params_dict.keys()) or ("down" not in quad_params_dict.keys()):
            raise Exception("Cannot plot shifted up/down variations, do not have the right info in quad_params_dict.")
        up_shift_str = "up"+str(shift_var)
        do_shift_str = "down"+str(shift_var)
        quad_arr_dict[up_shift_str] = {}
        quad_arr_dict[do_shift_str] = {}
        quad_arr_dict[up_shift_str][0] = quad_arr_dict["nom"][0]
        quad_arr_dict[do_shift_str][0] = quad_arr_dict["nom"][0]
        quad_arr_dict[up_shift_str][1] = get_shifted_arr(quad_arr_dict["up"][1],quad_arr_dict["nom"][1],quad_arr_dict["nom"][0],shift_var)
        quad_arr_dict[do_shift_str][1] = get_shifted_arr(quad_arr_dict["down"][1],quad_arr_dict["nom"][1],quad_arr_dict["nom"][0],shift_var)

    # Loop over arr dict and make plots
    for key_name, quad_arr in quad_arr_dict.items():

        # Get info needed for the legend
        if key_name != "none": tag = key_name + " "
        else: tag = ""
        if key_name in quad_params_dict.keys():
            s0 = quad_params_dict[key_name][0] 
            s1 = quad_params_dict[key_name][1]
            s2 = quad_params_dict[key_name][2]
            leg_str = get_fit_str(tag+"fit",xaxis_name,s0,s1,s2)
        else: leg_str = tag

        # Plot the points
        x_arr = quad_arr[0]
        y_arr = quad_arr[1]
        ax.plot(x_arr,y_arr,label=leg_str)

    # Set x and y ranges (just use default for y unless it's really tiny)
    if ((ymax-ymin)<1e-6): ax.set_ylim([0.0,1.5])
    ax.set_xlim(xaxis_range)
    ax.set_ylim([0,ymax*1.2])

    #ax.plot(x_arr,y_arr,label="TEST")
    ax.legend()

    # Save the figure
    #plt.show()
    plt.savefig(os.path.join(save_dir,title))
    plt.close()


########## Get info from nano aod file ##########

# Given a nano events object, get the arary of WCs and sum over all events
def get_summed_quad_fit_arr(events):

    # Raise an error if this is not an EFT sample
    if not hasattr(events, "EFTfitCoefficients"):
        raise Exception("Error: This file does not have any EFT fit coefficients.")

    # Get array of quad coeffs
    quad_coeffs_arr = ak.to_numpy(events["EFTfitCoefficients"]) 
    quad_coeffs_arr = ak.sum(quad_coeffs_arr,axis=0)

    return quad_coeffs_arr


# Given a quad fit array and a list of WCs, make a dictionary mapping the quad fit terms to their WCs
def get_quad_fit_dict(wc_names_lst,quad_coeffs_arr):

    # Prepend "sm" to wc names list
    wc_names_lst = ["sm"] + wc_names_lst

    # Fill a dict with wc names and the corresponding quad fit terms
    # The order of the quad coeff array is the "lower triangle" of the matrix
    # I.e. if the list of WC names is [c1,c2,...,cn], the order of the quad terms is:
    #     quad terms = [
    #         sm*sm,
    #         c1*sm, c1*c1,
    #         c2*sm, c2*c1, c2*c2,
    #         ...
    #         cn*sm,  ... , cn*cn
    #     ]
    wc_fit_dict = {}
    idx = 0
    for i in range(len(wc_names_lst)):
        for j in range(i+1):
            key_str = wc_names_lst[i]+"*"+wc_names_lst[j]
            wc_fit_dict[key_str] = quad_coeffs_arr[idx]
            idx+=1

    return wc_fit_dict


########## Manipulate quad fit dict ##########

# Get constant, linear, quadratic term for a give WC from a fit dict
def get_1d_fit(fit_dict,wc):
    s0_key = "sm*sm"
    s1_key = wc+"*sm"
    s2_key = wc+"*"+wc
    s0 = fit_dict[s0_key]
    s1 = fit_dict[s1_key]
    s2 = fit_dict[s2_key]
    return [s0,s1,s2]

# Scale all values in a fit dictionary by a given value, returns a new dictionary
def scale_fit_dict(fit_dict,scale_val):
    ret_dict = {}
    for k,v in fit_dict.items():
       ret_dict[k] = v*scale_val 
    return ret_dict

# Scale all values in a fit to the SM value, returns a new dictionary
def scale_to_sm(fit_dict):
    return scale_fit_dict(fit_dict,1.0/fit_dict["sm*sm"])

# Evalueate a fit dictionary at some point in the wc phase space
def eval_fit(fit_dict,wcpt_dict):
    xsec = 0
    for wc_str, coeff_val in fit_dict.items():
        wc1,wc2 = wc_str.split("*")
        if wc1 not in wcpt_dict:
            if wc1 == "sm":
                wc1_val = 1.0
            else:
                print(f"WARNING: No value specified for WC {wc}. Setting it to 0.")
                wc1_val = 0.0
        else:
            wc1_val = wcpt_dict[wc1]
        if wc2 not in wcpt_dict.keys():
            if wc2 == "sm":
                wc2_val = 1.0
            else:
                print(f"WARNING: No value specified for WC {wc}. Setting it to 0.")
                wc2_val = 0.0
        else:
            wc2_val = wcpt_dict[wc2]
        xsec = xsec + wc1_val*wc2_val*coeff_val
    return xsec


# Evaluate a 1d quadratic at a given point
def eval_1d_quad( quad_params_1d,x):
    if len(quad_params_1d) != 3:
        raise Exception(f"Error: Wrong number of parameters specified for 1d quadratic. Require 3, received {len(quad_params)}.")
    y = quad_params_1d[0] + quad_params_1d[1]*x + quad_params_1d[2]*x*x
    return y


# Takes as input 1d quadratic fit params, and returns the x value where y crosses some threshold
def find_where_fit_crosses_threshold(quad_params_1d,threshold):

    # Get the individual params
    if len(quad_params_1d) != 3:
        raise Exception(f"Error: Wrong number of parameters specified for 1d quadratic. Require 3, received {len(quad_params)}.")
    s0 = quad_params_1d[0]
    s1 = quad_params_1d[1]
    s2 = quad_params_1d[2]

    x_p = (-s1 + np.sqrt(s1*s1 - 4.0*s2*(s0-threshold)) )/(2.0*s2)
    x_m = (-s1 - np.sqrt(s1*s1 - 4.0*s2*(s0-threshold)) )/(2.0*s2)

    return [x_m,x_p]
