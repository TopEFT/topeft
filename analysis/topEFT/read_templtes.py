# About this script (Apr 21, 2022)
#   - This script takes as input the path to a dir that has all of the template root files produced by the datacard maker
#       - All of the files in the dir should correspond to a single variable
#       - So far the script has only been tested with njets, but in principle could be extended to other variables too
#   - The script then grabs all of the nom, up, and down histos from the templates, and plots them and saves to png

import os
import ROOT

from topcoffea.plotter.make_html import make_html


# Get the list of histo names from a root file
def get_histo_names(in_file,only_sm=True):
    histo_name_lst = [] 
    in_file_keys = in_file.GetListOfKeys()
    for key_object in in_file_keys:
        hname = key_object.GetName()
        if only_sm:
            if "_sm" in hname:
                histo_name_lst.append(hname)
        else:
            histo_name_lst.append(hname)
    return histo_name_lst


# For all of the hisos in the root file, get the dict of all of the nominal, up, and down histos
def get_dict_of_nom_up_do_names(in_root_file):

    syst_name_dict = {}
    histo_name_lst = get_histo_names(in_root_file)

    # Loop over all of the histos in the dict and find the nom, up, and down
    for hname in histo_name_lst:
        histo = in_root_file.Get(hname)

        # Plot up down and nom for each histo
        if "Up" in hname:

            # Get the up down and nom histo names based on the up name
            hname_smsplit = hname.split("_sm_")
            if len(hname_smsplit) != 2: raise Exception("Something wrong, expected two parts of string")
            up_substr  = hname_smsplit[1]
            do_substr  = up_substr.replace("Up","Down")
            syst_name  = up_substr.replace("Up","")  # E.g. btagSFlight_2017
            hname_base = hname.replace(up_substr,"") # E.g. ttlnu_sm_
            hname_key  = hname_base+syst_name        # E.g. ttlnu_sm_btagSFlight_2017
            hname_nom  = hname_base[:-1]             # E.g. ttlnu_sm
            hname_do   = hname_base + do_substr      # E.g. ttlnu_sm_btagSFlight_2017Down
            hname_up   = hname_base + up_substr      # E.g. ttlnu_sm_btagSFlight_2017Up
            if hname_up != hname: raise Exception(f"Something went wrong in the name construction, expected {hname}, but built {hname_up}")

            # Add these variations to the out dictionary
            #if syst_name in syst_name_dict: raise Exception(f"Error, {syst_name} is included twice")
            if syst_name in syst_name_dict: print(f"Warning: {syst_name} is included twice")
            syst_name_dict[hname_key] = {}
            syst_name_dict[hname_key]["nom"] = hname_nom
            syst_name_dict[hname_key]["up"] = hname_up
            syst_name_dict[hname_key]["do"] = hname_do

    return syst_name_dict


# For a nom, up, down histo, draw all three to
def draw_nom_up_do_overlay(h_n,h_u,h_d,save_path):

    # Draw to a canvas
    canvas = ROOT.TCanvas("canvas")
    h_u.Draw("E")
    h_d.Draw("E SAME")
    h_n.Draw("E SAME")

    # Set the colors
    h_u.SetLineColor(3);
    h_n.SetLineColor(1);
    h_d.SetLineColor(4);

    # Set y range
    max_u = h_u.GetMaximum()
    max_d = h_d.GetMaximum()
    max_n = h_n.GetMaximum()
    max_y = max(max_n, max(max_u,max_d))
    h_u.GetYaxis().SetRangeUser(0.0,1.3*max_y)

    # Save
    canvas.Print(save_path)


# Main function
def main():

    ROOT.gROOT.SetBatch()

    #f = "/afs/crc.nd.edu/user/k/kmohrman/coffea_dir/check_PRs/master_for_benchmarking/topcoffea/datacards/apr18_checks/apr18_fullR2-data-bkg-sig_njets-lj0pt_onBranchWeightsUpdates_np/njets_withSys/ttx_multileptons-2lss_m_2b.root"
    f = "/afs/crc.nd.edu/user/k/kmohrman/coffea_dir/check_PRs/master_for_benchmarking/topcoffea/datacards/apr20_fullR2-data-bkg-sig_njets-lj0pt_np/njets_withSys/ttx_multileptons-2lss_m_2b.root"
    in_file  = ROOT.TFile.Open(f,"READ")

    out_basepath = "/afs/crc.nd.edu/user/k/kmohrman/coffea_dir/check_PRs/kmohrman_dc_yields_table/topcoffea/analysis/topEFT/test_apr21"

    # Get the dictionary of the variations
    syst_name_dict = get_dict_of_nom_up_do_names(in_file)

    # Make plot for each variation
    for proc_syst_var_name in syst_name_dict.keys():
        save_fpath = os.path.join(out_basepath,proc_syst_var_name+".png")
        draw_nom_up_do_overlay(
            h_n = in_file.Get(syst_name_dict[proc_syst_var_name]["nom"]),
            h_u = in_file.Get(syst_name_dict[proc_syst_var_name]["up"]),
            h_d = in_file.Get(syst_name_dict[proc_syst_var_name]["do"]),
            save_path = save_fpath,
        )

    make_html(out_basepath)


if __name__ == "__main__":
    main()
