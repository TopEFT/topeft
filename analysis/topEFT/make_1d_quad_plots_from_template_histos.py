import numpy as np
import os
import topcoffea.modules.QuadFitTools as qft
from topcoffea.plotter.make_html import make_html

# Load the in dict
#import tmp_quad as q
#IN_DICT = q.fit_pieces_dict
#import decomp_fit_terms_test as q
#IN_DICT = q.decomp_terms_in_dict

#import fit_params_mar06_fullR2_sig_njets_lj0pt_withSys_renormfact_njets_theta22.ttx_multileptons_2lss_4t_m_2b as q
#import fit_params_mar06_fullR2_sig_njets_lj0pt_withSys_renormfact_njets_theta22.ttx_multileptons_2lss_4t_p_2b as q
#import fit_params_mar06_fullR2_sig_njets_lj0pt_withSys_renormfact_njets_theta22.ttx_multileptons_2lss_m_2b as q
#import fit_params_mar06_fullR2_sig_njets_lj0pt_withSys_renormfact_njets_theta22.ttx_multileptons_2lss_p_2b as q
#import fit_params_mar06_fullR2_sig_njets_lj0pt_withSys_renormfact_njets_theta22.ttx_multileptons_3l1b_m as q
#import fit_params_mar06_fullR2_sig_njets_lj0pt_withSys_renormfact_njets_theta22.ttx_multileptons_3l1b_p as q
#import fit_params_mar06_fullR2_sig_njets_lj0pt_withSys_renormfact_njets_theta22.ttx_multileptons_3l2b_m as q
#import fit_params_mar06_fullR2_sig_njets_lj0pt_withSys_renormfact_njets_theta22.ttx_multileptons_3l2b_p as q
#import fit_params_mar06_fullR2_sig_njets_lj0pt_withSys_renormfact_njets_theta22.ttx_multileptons_3l_sfz_1b as q
#import fit_params_mar06_fullR2_sig_njets_lj0pt_withSys_renormfact_njets_theta22.ttx_multileptons_3l_sfz_2b as q
#import fit_params_mar06_fullR2_sig_njets_lj0pt_withSys_renormfact_njets_theta22.ttx_multileptons_4l_2b as q
#IN_DICT = q.p

import fit_params_mar06_fullR2_sig_njets_lj0pt_withSys_renormfact_njets.ttx_multileptons_2lss_p_2b_theta01 as q
#import fit_params_mar06_fullR2_sig_njets_lj0pt_withSys_renormfact_njets.ttx_multileptons_2lss_p_2b_theta05 as q
#import fit_params_mar06_fullR2_sig_njets_lj0pt_withSys_renormfact_njets.ttx_multileptons_2lss_p_2b_theta22 as q
#import fit_params_mar06_fullR2_sig_njets_lj0pt_withSys_renormfact_njets.ttx_multileptons_2lss_p_2b_theta22_FLIP as q
IN_DICT = q.p


PROC_LST = [ "ttH", "ttlnu", "ttll", "tllq", "tHq", "tttt" ]
#WC_LST = ['ctW','ctZ','ctp','cpQM','ctG','cbW','cpQ3','cptb','cpt','cQl3i','cQlMi','cQei','ctli','ctei','ctlSi','ctlTi', 'cQq13', 'cQq83', 'cQq11', 'ctq1', 'cQq81', 'ctq8', 'ctt1', 'cQQ1', 'cQt8', 'cQt1']
#WC_LST = ['ctG','ctq8']
#WC_LST = ['cpt']
WC_LST = ['ctZ','cpt']

# Get WC lst for each process
def get_wc_lst_for_proc(proc,all_decomp_terms):
    proc_lst = []
    for decomp_term in all_decomp_terms.keys():
        if proc not in decomp_term: continue
        for wc in WC_LST:
            if wc in decomp_term:
                if wc not in proc_lst:
                    proc_lst.append(wc)
    return proc_lst


######### Getting names from list of decomposed names #########

# Takes dict dumped by cpp and returns a list of the relevant ones
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

# Get term: sm = S
def get_decomp_term_sm(decomp_lst):
    ret = None
    for decomp_term_name in decomp_lst:
        if "sm" in decomp_term_name:
            ret = decomp_term_name
    if ret is None: raise Exception("Error, no sm term found")
    else: return ret

# Get term: lin = S + Li + Qi
def get_decomp_term_lin(decomp_lst,wc):
    ret = None
    for decomp_term_name in decomp_lst:
        if ("lin" in decomp_term_name) and (wc in decomp_term_name):
            ret = decomp_term_name
    if ret is None: raise Exception("Error, no lin term found")
    else: return ret

# Get term: quad = Qi
def get_decomp_term_quad(decomp_lst,wc):
    ret = None
    for decomp_term_name in decomp_lst:
        if ("quad" in decomp_term_name) and ("mix" not in decomp_term_name) and (wc in decomp_term_name):
            ret = decomp_term_name
    if ret is None: raise Exception("Error, no mixed term found")
    else: return ret

# Get term: mix = S + Li + lj + Qi + Qj + 2Mij
def get_decomp_term_mix(decomp_lst,wc0,wc1):
    ret = None
    for decomp_term_name in decomp_lst:
        if ("mix" in decomp_term_name) and (wc0 in decomp_term_name) and (wc1 in decomp_term_name):
            ret = decomp_term_name
    if ret is None: raise Exception("Error, no mixed term found")
    else: return ret


######### Finding fit coeffecients from the decomposed components #########

# Construct the keys e.g. ctG*ctp for the quad fit for a given set of WCs
def get_quad_keys(wc_lst):
    quad_terms_lst = []
    if "sm" not in wc_lst: wc_lst = ["sm"] + wc_lst
    for i,wc_row in enumerate(wc_lst):
        for j,wc_col in enumerate(wc_lst):
            if j>i: continue
            quad_terms_lst.append(wc_row+"*"+wc_col)
    return quad_terms_lst

# Takes e.g. "ctG*ctp" and reconstructs the value from the values in the decomposed dict
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

        val_ret = (val_mix - val_sm - val_lin0 - val_lin1 - val_quad0 - val_quad1)/2.0

    # looking for the L term
    else:
        if wc0 != "sm": wc_nonsm = wc0
        elif wc1 != "sm": wc_nonsm = wc1
        else: raise Exception("This should not be possible")
        val_lin  = IN_DICT[get_decomp_term_lin(decomp_term_name_lst,wc_nonsm)]
        val_quad = IN_DICT[get_decomp_term_quad(decomp_term_name_lst,wc_nonsm)]
        val_ret = val_lin - val_sm - val_quad

    return val_ret

# Find fit param values for terms e.g. ["sm*sm","sm*ctG"...] from the values mapped to the decomposed lst names
# Really just a wrapper for get_param_value()
def get_fit_param_val_dict(quad_term_name_lst,decomp_terms_lst):
    ret_dict = {}
    for quad_term_name in quad_term_name_lst:
        val = get_param_value(quad_term_name,decomp_terms_lst)
        ret_dict[quad_term_name] = val
    return ret_dict


# Main wrapper to read the values and make plots
def quad_wrapper(proc,njets,save_path="quad_fits",shift=None):

    wc_lst_for_proc = get_wc_lst_for_proc(proc,IN_DICT)

    # Get the names of the fit params for the given list of WCs
    quad_terms_lst = get_quad_keys(wc_lst_for_proc)

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

    wcpt = {
        #"ctW": -0.74,
        #"ctZ": -0.86,
        #"ctp": 24.5,
        #"cpQM": -0.27,
        #"ctG": -0.81,
        #"cbW": 3.03,
        #"cpQ3": -1.71,
        #"cptb": 0.13,
        #"cpt": -3.72,
        #"cQl3i": -4.47,
        #"cQlMi": 0.51,
        #"cQei": 0.05,
        #"ctli": 0.33,
        #"ctei": 0.33,
        #"ctlSi": -0.07,
        #"ctlTi": -0.01,
        #"cQq13"  : -0.05,
        #"cQq83"  : -0.15,
        #"cQq11"  : -0.15,
        #"ctq1"   : -0.20,
        #"cQq81"  : -0.50,
        #"ctq8"   : -0.50,
        #"ctt1"   : -0.71,
        #"cQQ1"   : -1.35,
        #"cQt8"   : -2.89,
        #"cQt1"   : -1.24,

    }
    #wcpt = {"ctZ":-5.0,"cpt":3.0}
    #wcpt = {"ctZ":-5.0}
    wcpt = {"cpt":3.0}
    print(proc_fit_dict["nom"])
    print(qft.eval_fit(proc_fit_dict["nom"],wcpt))
    exit()

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
            shift_var = shift,
        )



######### Main function, get values for the quad fits #########

def main():

    # Run the wrapper for all processes and all jet cateogires and all WCs
    #www_loc = "/afs/crc.nd.edu/user/k/kmohrman/www/EFT/TopCoffea/fitting/sys_checks/mar03_fullR2-sig_processorMaster-dcmakerBeforeCor_fitting/sys_individual/check_renormfact/mar06_fullR2-sig_njets-lj0pt_withSys-renormfact/njets_template-hisots/test"
    #base_dir = os.path.join(www_loc,"ttx_multileptons-2lss_p_2b_fits")
    #base_dir = os.path.join(www_loc,"ttx_multileptons-2lss_p_2b_fits_theta22_test_flip_u_d")

    '''
    www_loc = "/afs/crc.nd.edu/user/k/kmohrman/www/EFT/TopCoffea/fitting/sys_checks/mar03_fullR2-sig_processorMaster-dcmakerBeforeCor_fitting/sys_individual/check_renormfact/mar06_fullR2-sig_njets-lj0pt_withSys-renormfact/njets_template-hisots/all_cats_theta22"
    #base_dir = os.path.join(www_loc,"ttx_multileptons-2lss_4t_m_2b")
    #base_dir = os.path.join(www_loc,"ttx_multileptons-2lss_4t_p_2b")
    #base_dir = os.path.join(www_loc,"ttx_multileptons-2lss_m_2b")
    #base_dir = os.path.join(www_loc,"ttx_multileptons-2lss_p_2b")
    #base_dir = os.path.join(www_loc,"ttx_multileptons-3l1b_m")
    #base_dir = os.path.join(www_loc,"ttx_multileptons-3l1b_p")
    #base_dir = os.path.join(www_loc,"ttx_multileptons-3l2b_m")
    #base_dir = os.path.join(www_loc,"ttx_multileptons-3l2b_p")
    #base_dir = os.path.join(www_loc,"ttx_multileptons-3l_sfz_1b")
    #base_dir = os.path.join(www_loc,"ttx_multileptons-3l_sfz_2b")
    base_dir = os.path.join(www_loc,"ttx_multileptons-4l_2b")

    #os.mkdir(base_dir)
    save_dir = os.path.join(base_dir) # TMP
    os.makedirs(save_dir) # TMP
    for proc in PROC_LST:
        print("proc:",proc)
        ##os.mkdir(os.path.join(base_dir,proc))
        #for njets in [4,5,6,7]:
        #for njets in [2,3,4,5]:
        for njets in [2,3,4]:
            print("njets:",njets)
            ##save_dir = os.path.join(os.path.join(base_dir,proc),"njets"+str(njets))
            ##os.mkdir(save_dir)
            quad_wrapper(proc,njets,save_dir) # Run the wrapper
            make_html(save_dir)
    '''

    # Make a single set of plots
    #os.mkdir("tmp_quad_fits")
    quad_wrapper("ttH",4,"tmp_quad_fits")


main()
