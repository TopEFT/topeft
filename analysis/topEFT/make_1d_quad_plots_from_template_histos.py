# About this script:
#     - This script takes as input the information from the template histograms
#     - The goal is to reconstruct the quadratic parameterizations from the templates
#     - The script extracts the full 26-dimensional quadratic, but currently just plots the 1d quadratics 
# About the input to this script:
#     - The relevant templates are the ones produced by topcoffea's datacard_maker.py, which should be passed 
#       to EFTFit's look_at_templates.C (which opens the templates, optionally extrapolates the up/down beyond +-1sigma,
#       and dumps the info into a python dictionary), so it is the output of look_at_templates.C that this script runs on
#     - It would probably be better for look_at_templates.C to dump the info into e.g. a json, but right now it just prints 
#       the info to the screen in the form of a python dictionary, so this script assumes that dictionary has been pasted 
#       into a .py file and we can just directly import the dictionary
#     - That dictionary that we import is a global variable called IN_DICT in the script
#     - Note that so far this script assumes the templates are just njets (i.e. none of the naming and conventions etc. are 
#       currently set up to work with e.g. bins in lj0pt)

# Some notes on the naming conventions for the decomposed and reconstructed quadratic parameterization
# This is based on Eqn 5 of CMS AN-20-204 ("EFT model for SMP measurements")
# The terms of the quadratic are: S, L, Q, M, and the decomposed terms in the template dict are defines as follows:
#     sm       = S
#     quad_i   = Qi
#     lin_i    = S + Li + Qi
#     mixed_ij = S + Li + Lj + Qi + Qj + 2Mij
# Thus, to reconstruct the quadratic parameterization, we need the following combinations of decomposed terms:
#     S    = sm
#     Qi   = quad_i
#     Li   = lin_i - S - Qi 
#          = lin_i - sm - quad_i
#     2Mij = mixed_ij - S - Qi - Qj - Li - Lj
#          = mixed_ij - sm - quad_i - quad_j - (lin_i - sm - Qi) - (lin_j - sm - quad_j)

# To run this script:
#     - The script is currently very basic and hard coded
#     - The inputs dictionary is hardcoded, also where to save the output plots is hard coded
#     - So to run it is just: 
#       python make_1d_quad_plots_from_template_histos.py


import numpy as np
import os
import topcoffea.modules.QuadFitTools as qft
from topcoffea.plotter.make_html import make_html

# Load the input dict (with all of the values from the template histos)
import fit_params_mar06_fullR2_sig_njets_lj0pt_withSys_renormfact_njets.ttx_multileptons_2lss_p_2b_theta01 as template_vals_dict_file
#IN_DICT = template_vals_dict_file.template_vals_dict
IN_DICT = template_vals_dict_file.p


PROC_LST = [ "ttH", "ttlnu", "ttll", "tllq", "tHq", "tttt" ]
WC_LST = ['ctW','ctZ','ctp','cpQM','ctG','cbW','cpQ3','cptb','cpt','cQl3i','cQlMi','cQei','ctli','ctei','ctlSi','ctlTi', 'cQq13', 'cQq83', 'cQq11', 'ctq1', 'cQq81', 'ctq8', 'ctt1', 'cQQ1', 'cQt8', 'cQt1']


######### Functions that get info from the list of template names #########

# Get WC lst for each process (since not all processes will have templates for all WCs since the datacard maker skips the ones that do not contribute)
def get_wc_lst_for_proc(proc,all_decomp_terms):
    proc_lst = []
    for decomp_term in all_decomp_terms.keys():
        if proc not in decomp_term: continue
        for wc in WC_LST:
            substr_lst = decomp_term.split("_")
            if wc in substr_lst:
                if wc not in proc_lst:
                    proc_lst.append(wc)
    return proc_lst

# Takes the input dict and returns a list of the relevant terms (given a sys variation and njets value)
def get_term_lst(in_dict,proc_to_get,var_to_get,njets_to_get):

    # Get up/down/nom variation of term
    def get_variation(in_str):
        if   ("Up" in in_str): return "up"
        elif ("Down" in in_str): return "down"
        else: return "nom"

    # Fill the dict
    term_lst = []
    for term_name, val in in_dict.items():
        #print("\n",term_name)
        # Get the info
        var_str = get_variation(term_name)
        njets = term_name[-1]
        # Skip ones we don't care about
        if proc_to_get not in term_name: continue
        if var_str != var_to_get: continue
        if njets != str(njets_to_get): continue
        # Append to return lst
        term_lst.append(term_name)

    return term_lst

# Get sm term (sm = S)
def get_decomp_term_sm(decomp_lst):
    ret = None
    for decomp_term_name in decomp_lst:
        if "sm" in decomp_term_name:
            ret = decomp_term_name
    if ret is None:
        print(f"\nError: Can't find term for wc \"{wc}\" in this list of terms: {decomp_lst}")
        raise Exception("Error, no sm term found")
    else: return ret

# Get lin term (lin = S + Li + Qi)
def get_decomp_term_lin(decomp_lst,wc):
    ret = None
    for decomp_term_name in decomp_lst:
        substr_lst = decomp_term_name.split("_")
        if ("lin" in decomp_term_name) and (wc in substr_lst):
            ret = decomp_term_name
    if ret is None:
        print(f"\nError: Can't find term for wc \"{wc}\" in this list of terms: {decomp_lst}")
        raise Exception("Error, no lin term found")
    else: return ret

# Get quad term (quad = Qi)
def get_decomp_term_quad(decomp_lst,wc):
    ret = None
    for decomp_term_name in decomp_lst:
        substr_lst = decomp_term_name.split("_")
        if ("quad" in decomp_term_name) and ("mix" not in decomp_term_name) and (wc in substr_lst):
            ret = decomp_term_name
    if ret is None:
        print(f"\nError: Can't find term for wc \"{wc}\" in this list of terms: {decomp_lst}")
        raise Exception("Error, no mixed term found")
    else: return ret

# Get mix term (mix = S + Li + lj + Qi + Qj + 2Mij)
def get_decomp_term_mix(decomp_lst,wc0,wc1):
    ret = None
    for decomp_term_name in decomp_lst:
        substr_lst = decomp_term_name.split("_")
        if ("mix" in decomp_term_name) and (wc0 in substr_lst) and (wc1 in substr_lst):
            ret = decomp_term_name
    if ret is None:
        print(f"\nError: Can't find term for wc \"{wc}\" in this list of terms: {decomp_lst}")
        raise Exception("Error, no mixed term found")
    else: return ret


######### Functions that find fit coeffecients from the decomposed components #########

# Takes a quad fit param e.g. "ctG*ctp", finds the relevant terms from input dict, and builds the value from those terms
def get_param_value(quad_term,decomp_term_name_lst):

    wc0 = quad_term.split("*")[0]
    wc1 = quad_term.split("*")[1]

    decomp_name_sm = get_decomp_term_sm(decomp_term_name_lst)
    val_sm = IN_DICT[decomp_name_sm]

    # Looking for the S term
    if quad_term == "sm*sm":
        val_ret = val_sm

    # Looking for the Q term
    elif wc0 == wc1:
        val_ret = IN_DICT[get_decomp_term_quad(decomp_term_name_lst,wc0)]

    # Looking for the M term
    elif "sm" not in quad_term:
        val_mix   = IN_DICT[get_decomp_term_mix(decomp_term_name_lst,wc0,wc1)]
        val_lin0  = IN_DICT[get_decomp_term_lin(decomp_term_name_lst,wc0)]
        val_lin1  = IN_DICT[get_decomp_term_lin(decomp_term_name_lst,wc1)]
        val_quad0 = IN_DICT[get_decomp_term_quad(decomp_term_name_lst,wc0)]
        val_quad1 = IN_DICT[get_decomp_term_quad(decomp_term_name_lst,wc1)]

        # Note, don't divide by 2 here (since we don't keep e.g. ctG*ctp and ctp*ctG, the ctG*ctp term needs the factor of 2)
        val_ret = val_mix - val_sm - (val_lin0 - val_sm - val_quad0) - (val_lin1 - val_sm - val_quad1) - val_quad0 - val_quad1

    # looking for the L term
    else:
        if wc0 != "sm": wc_nonsm = wc0
        elif wc1 != "sm": wc_nonsm = wc1
        else: raise Exception("This should not be possible")
        val_lin  = IN_DICT[get_decomp_term_lin(decomp_term_name_lst,wc_nonsm)]
        val_quad = IN_DICT[get_decomp_term_quad(decomp_term_name_lst,wc_nonsm)]
        val_ret = val_lin - val_sm - val_quad

    return val_ret

# Find quad param values for a given list of quad terms (e.g. ["sm*sm","sm*ctG"...])
# Really this is just a wrapper for get_param_value() (here we call that function in a loop over a list of term names)
def get_fit_param_val_dict(quad_term_name_lst,decomp_terms_lst):
    ret_dict = {}
    for quad_term_name in quad_term_name_lst:
        val = get_param_value(quad_term_name,decomp_terms_lst)
        ret_dict[quad_term_name] = val
    return ret_dict


# Main wrapper to read the values and make plots (this function should maybe be split up)
def quad_wrapper(proc,njets,save_path="quad_fits"):

    wc_lst_for_proc = get_wc_lst_for_proc(proc,IN_DICT)

    # Get the names of the fit params for the given list of WCs
    quad_terms_lst = qft.get_quad_keys(wc_lst_for_proc)

    # Get an example subset
    decomp_terms_dict = {
        "nom"  : get_term_lst(IN_DICT,proc,"nom",njets),
        "up"   : get_term_lst(IN_DICT,proc,"up",njets),
        "down" : get_term_lst(IN_DICT,proc,"down",njets),
    }

    # For each fit param calculate val
    proc_fit_dict = {}
    for sys_var,decomp_terms_lst in decomp_terms_dict.items():
        proc_fit_dict[sys_var] = get_fit_param_val_dict(quad_terms_lst,decomp_terms_lst)
        #proc_fit_dict[sys_var] = qft.scale_to_sm(proc_fit_dict[sys_var]) # If we want to scale to sm

    # For validation, can evaluate the fit at a given point in EFT space
    #wcpt = {"cpt":11.1}
    #print(proc_fit_dict["nom"])
    #print(qft.eval_fit(proc_fit_dict["nom"],wcpt))
    #exit()

    # Make the plots
    for wc in wc_lst_for_proc:
        plot_name = "fit_" + proc + "_njets" + str(njets) + "_" + wc
        qft.make_1d_quad_plot(
            quad_params_dict = {
                "nom" : qft.get_1d_fit(proc_fit_dict["nom"],wc),
                "up"  : qft.get_1d_fit(proc_fit_dict["up"],wc),
                "down": qft.get_1d_fit(proc_fit_dict["down"],wc),
            },
            xaxis_name = wc,
            yaxis_name = "Yld",
            title = plot_name,
            xaxis_range = qft.FIT_RANGES[wc],
            save_dir = save_path,
        )



######### Main function, get values for the quad fits #########

def main():

    # Example for a single set of plots
    #os.mkdir("tmp_quad_fits")
    #quad_wrapper("ttH",4,"tmp_quad_fits")

    # Specify the save dir (would be a lot better to pass this as a command line argument)
    www_loc = "/afs/crc.nd.edu/user/k/kmohrman/www/EFT/TopCoffea/testing/mar04"
    base_dir = os.path.join(www_loc,"ttx_multileptons-2lss_p_2b") # For example

    # Run the wrapper for all processes and all jet cateogires and all WCs
    os.mkdir(base_dir)
    for proc in PROC_LST:
        print("proc:",proc)
        os.mkdir(os.path.join(base_dir,proc))
        for njets in [4,5,6,7]: # NOTE This is super hard coded (for jet bins for this category) someday should fix it
            print("njets:",njets)
            save_dir = os.path.join(os.path.join(base_dir,proc),"njets"+str(njets))
            os.mkdir(save_dir)
            quad_wrapper(proc,njets,save_dir) # Run the wrapper
            make_html(save_dir)


if __name__ == "__main__":
    main()
