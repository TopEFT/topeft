# About this script (Apr 21, 2022)
#   - This script takes as input the path to a dir that has all of the template root files produced by the datacard maker
#       - All of the files in the dir should correspond to a single variable
#       - So far the script has only been tested with njets, but in principle could be extended to other variables too
#   - The script then grabs all of the nom, up, and down histos from the templates, and plots them and saves to png

import os
import argparse
import ROOT

from topcoffea.plotter.make_html import make_html
import get_datacard_yields as dy # Note the functions we're using from this script should probably go in topocffea/modules


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
            if hname_key in syst_name_dict: raise Exception(f"Error, {hname_key} is included twice")
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
    min_u = h_u.GetMinimum()
    min_d = h_d.GetMinimum()
    min_n = h_n.GetMinimum()
    max_y = max(max_n, max(max_u,max_d))
    min_y = min(min_n, min(min_u,min_d))
    h_u.GetYaxis().SetRangeUser(min(1.3*min_y,0),1.3*max_y)

    # Save
    print("Saviang",save_path)
    canvas.Print(save_path)




# Main function
def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("datacards_dir_path", help = "The path to the directory where the datacards templates live")
    parser.add_argument("-o", "--out-path", default=".", help = "Out file location")
    args = parser.parse_args()

    out_basepath = args.out_path

    # Very crude way of switching between run modes, maybe should put into the command line options
    print_all_templates = 0
    dump_negative = 0
    make_plots = 1

    # Get the list of template root files in the dc dir
    files_all = os.listdir(args.datacards_dir_path)
    template_files = dy.get_dc_file_names(files_all,ext=".root")

    ### Get list of all histos for a given category, just for ref ###
    if print_all_templates:
        example_cat = "ttx_multileptons-2lss_p_2b.root"
        all_histos = get_histo_names(ROOT.TFile.Open(os.path.join(args.datacards_dir_path,example_cat),"READ"),only_sm=True)
        print(f"Printing all histos for cat {example_cat}:")
        for name in all_histos: print(name)
        print(f"({len(all_histos)} total)")

    ### Get info about any negative bins ###
    if dump_negative:
        for template_name in template_files:
            # Get root file and cat name
            template_path_full = os.path.join(args.datacards_dir_path,template_name)
            in_file  = ROOT.TFile.Open(template_path_full,"READ")
            cat_name = dy.get_cat_name_from_dc_name(template_name,".root")
            print("Cat name:",cat_name)
            all_histos = get_histo_names(in_file,only_sm=True)
            for h_name in all_histos:
                h = in_file.Get(h_name)
                m = h.GetMinimum()
                a = h.Integral()
                if a < 0:
                    print(f"\t{h_name} sum val: {a}")
                #if m < 0:
                #    print(f"\t{h_name} min val: {m}")

    ### Make plots for the nominal up and down ###
    if make_plots:
        # Loop over templates
        for template_name in template_files:

            # Get root file and cat name
            template_path_full = os.path.join(args.datacards_dir_path,template_name)
            in_file  = ROOT.TFile.Open(template_path_full,"READ")
            cat_name = dy.get_cat_name_from_dc_name(template_name,".root")
            print("Cat name:",cat_name)

            # Get the dictionary of the variations
            syst_name_dict = get_dict_of_nom_up_do_names(in_file)

            # Make an output subdir for this category
            out_basepath_forthiscat = os.path.join(out_basepath,cat_name)
            os.mkdir(out_basepath_forthiscat)

            # Make plot for each variation
            ROOT.gROOT.SetBatch()
            for proc_syst_var_name in syst_name_dict.keys():
                print("proc_syst_var_name",proc_syst_var_name)
                save_fpath = os.path.join(out_basepath_forthiscat,proc_syst_var_name+".png")
                n_dict = draw_nom_up_do_overlay(
                    h_n = in_file.Get(syst_name_dict[proc_syst_var_name]["nom"]),
                    h_u = in_file.Get(syst_name_dict[proc_syst_var_name]["up"]),
                    h_d = in_file.Get(syst_name_dict[proc_syst_var_name]["do"]),
                    save_path = save_fpath,
                )

            make_html(out_basepath_forthiscat)


if __name__ == "__main__":
    main()
