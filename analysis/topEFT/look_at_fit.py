import numpy as np
import dicts

CAT_ORDER = ["BH", "BM", "BL", "EH", "EM", "EL"]

# Build the ratio dict from this template
ratio_dict = {
    "EH_EH" : {}, # Not really getting the shape right # P!

    "EH_BH" : {}, # Not really getting the shape right # P!
    ##"BH_BH" : {}, # Weird shape in ss

    "EH_EM" : {}, # P!
    "BH_EM" : {}, # Not really getting the shape right # P!
    "EM_EM" : {}, # P!

    "EH_BM" : {}, # P!
    "BH_BM" : {}, # P!
    #"EM_BM" : {},
    "BM_BM" : {}, # P!

    #"EH_EL" : {},
    "BH_EL" : {}, # P!
    ##"EM_EL" : {}, # No Z peak
    "BM_EL" : {}, # P!
    ##"EL_EL" : {}, # Our selection can't fill

    #"EH_BL" : {},
    "BH_BL" : {}, # P!
    "EM_BL" : {}, # P!
    "BM_BL" : {}, # P!
    ##"EL_BL" : {}, # Our selection can't fill
    ##"BL_BL" : {}, # Our selection can't fill
}

# Gives positive answer:
#EH_EH, EM_EM, BM_BM, BM_EL, BH_BL, BM_BL

# Build the row for the matrix we want to invert
def get_lin_eqn_row(lin_eqn_term_lst,cat_pair_name):
    row_lst = []
    cats_in_pair = cat_pair_name.split("_")
    for cat in lin_eqn_term_lst:
        occurrences = cats_in_pair.count(cat)
        row_lst.append(float(occurrences))
    return row_lst


# Main function
def main():

    ss_dict = dicts.d_ss_05
    os_dict = dicts.d_os_01
    #ss_dict = dicts.d_ss_29
    #os_dict = dicts.d_os_29

    ### Fill the ratio dict ###

    # Fill SS terms
    for fit_name,param_dict in ss_dict.items():
        kin_cat = fit_name[-5:]
        if kin_cat not in ratio_dict: continue
        ratio_dict[kin_cat]["ss"] = param_dict["nsig"]

    # Fill OS terms
    for fit_name,param_dict in os_dict.items():
        kin_cat = fit_name[-5:]
        if kin_cat not in ratio_dict: continue
        ratio_dict[kin_cat]["os"] = param_dict["nsig"]

    # Calculate ratios
    for kin_cat in ratio_dict.keys():
        nsig_ss = ratio_dict[kin_cat]["ss"]
        nsig_os = ratio_dict[kin_cat]["os"]
        ratio = nsig_ss/(nsig_ss+nsig_os)
        ratio_dict[kin_cat]["ratio"] = ratio

    # Print ratios dict
    for k,v in ratio_dict.items():
        print(k,v)


    ### Build the matrix of coeff an solve for the probs ###

    # Buld M and r (to solve for x in M*p = r)
    print("CAT_ORDER",CAT_ORDER)
    coef_nested_lst = []
    ratio_lst = []
    for kin_cat_name in ratio_dict.keys():
        lin_eqn_row = get_lin_eqn_row(CAT_ORDER,kin_cat_name )
        coef_nested_lst.append(lin_eqn_row)
        ratio_lst.append(ratio_dict[kin_cat_name]["ratio"])
        print(kin_cat_name,ratio_dict[kin_cat_name]["ratio"])
    coef_arr = np.array(coef_nested_lst) # M
    ratio_arr = np.array(ratio_lst) # r

    # Solve for the probabilities (use svd to find pseudo inverse of matrix)
    coef_arr_inv = np.linalg.pinv(coef_arr)
    #coef_arr_inv = np.linalg.pinv(coef_arr,rcond=0.05)
    prob_arr = np.dot(coef_arr_inv,ratio_arr)

    # Print arrays
    print("\nM:",coef_arr)
    print("\nM inv:",coef_arr_inv)
    print("\nr:",ratio_arr)
    print("\np:",prob_arr)

main()
