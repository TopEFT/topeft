from topcoffea.modules.YieldTools import YieldTools

# Convenience function for printing out two sets of yields and percent differene
def comp_ylds(hin_dict1,hin_dict2,year1,year2,str1,str2):

    yt = YieldTools()

    ylds_dict1 = yt.get_yld_dict(hin_dict1,year1)
    ylds_dict2 = yt.get_yld_dict(hin_dict2,year2)

    pdiff_dict = yt.get_diff_between_nested_dicts(ylds_dict1,ylds_dict2,difftype="percent_diff")
    diff_dict  = yt.get_diff_between_nested_dicts(ylds_dict1,ylds_dict2,difftype="absolute_diff")

    yt.print_yld_dicts(ylds_dict1,str1)
    yt.print_yld_dicts(ylds_dict2,str2)
    yt.print_yld_dicts(diff_dict,"Diff between dicts",tolerance=1e-9)

    yt.print_latex_yield_table(ylds_dict1,yt.CAT_LST,str1,print_begin_info=True)
    yt.print_latex_yield_table(ylds_dict2,yt.CAT_LST,str2)
    yt.print_latex_yield_table(pdiff_dict,yt.CAT_LST,f"Percent diff between central {str1} and {str2}",print_end_info=True)


######### The main() function #########

def main():

    yt = YieldTools() # Where to put this?

    # TODO: This main() has gotten super messy, will need to clean it up

    # Paths to the input pkl files
    fpath_default  = "histos/plotsTopEFT.pkl.gz"

    fpath_cuts_centralUl17_test = "histos/plotsTopEFT_centralUL17_all-UL-but-TTTT-THQ.pkl.gz"
    fpath_cuts_privateUl17_test = "histos/plotsTopEFT_privateUL17_all.pkl.gz"
    #fpath_cuts_centralUl17_test = "histos/plotsTopEFT_centralUL17_fix4l.pkl.gz"
    #fpath_cuts_privateUl17_test = "histos/plotsTopEFT_privateUL17_fix4l.pkl.gz"
    fpath_kevin = "/afs/crc.nd.edu/user/k/kmohrman/coffea_dir/topcoffea_mix-and-match-wc/topcoffea/analysis/topEFT/histos/plotsTopEFT_privateUL17_all.pkl.gz" # From Keivn's mix-and-match-wc branch

    # The dicts of 5 procs and tttt seperately
    fpath_test = "histos/plotsTopEFT_privateUL17_all-but-tttt.pkl.gz"
    #fpath_test = "histos/plotsTopEFT_privateUL17_tttt.pkl.gz"
    hin_test = yt.get_hist_from_pkl(fpath_test)

    # Get the histograms from the files
    hin_dict_central = yt.get_hist_from_pkl(fpath_cuts_centralUl17_test)
    hin_dict_private = yt.get_hist_from_pkl(fpath_cuts_privateUl17_test)
    hin_dict_kevin = yt.get_hist_from_pkl(fpath_kevin)


    #comp_ylds(hin_dict_private,hin_dict_central,"2017","2017","Private UL17","Central UL17") # The comparison in the notes

    #comp_ylds(hin_dict_private,hin_dict_kevin,"2017","2017","privateUL17","privateUL17 mixandmatchwc") # Compare sample from Kevin's branch with sample where I appended zeros (so only look at SM)
    comp_ylds(hin_dict_kevin,hin_test,"2017","2017","privateUL17","privateUL17 test")

    '''
    # Get the yield dictionaries and percent difference
    ylds_central_dict = get_yld_dict(hin_dict_central,"2017")
    ylds_private_dict = get_yld_dict(hin_dict_private,"2017")
    pdiff_dict = get_diff_between_nested_dicts(ylds_private_dict,ylds_central_dict,difftype="percent_diff")

    # Print out yields and percent differences
    print_yld_dicts(ylds_central_dict,"Central UL17 yields")
    print_yld_dicts(ylds_private_dict,"Private UL17 yields")
    #print_yld_dicts(pdiff_dict,"Percent diff between private and central")

    # Print latex table
    print_latex_yield_table(ylds_central_dict,CAT_LST,"Central UL17",print_begin_info=True,print_errs=True)
    print_latex_yield_table(ylds_private_dict,CAT_LST,"Private UL17")
    print_latex_yield_table(pdiff_dict,CAT_LST,"Percent diff between central and private UL17: (private-central)/private",print_end_info=True)
    '''


    ####### Print info for the jet cats ######
    #for lep_cat in JET_BINS.keys():
    #    print("lep_cat:",lep_cat)
    #    ylds_private_dict_jets = get_yld_dict(hin_dict_private,"2017",lep_cat)
    #    #print_yld_dicts(ylds_central_dict_test_jets,"Test")
    #    print_latex_yield_table(ylds_private_dict_jets,ylds_private_dict_jets["ttH"].keys(),"Private UL17 with jet cats",print_begin_info=True,print_end_info=True,column_variable="procs")
    #    relative_contributions = find_relative_contributions(ylds_private_dict_jets)
    #    print_latex_yield_table(relative_contributions,relative_contributions["ttH"].keys(),"Relative contributions",print_begin_info=True,print_end_info=True,column_variable="procs")

    #print_hist_info(hin_dict)
    #exit()

if __name__ == "__main__":
    main()
