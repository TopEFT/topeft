import os
import argparse
import datetime
import matplotlib.pyplot as plt
from coffea.nanoevents import NanoEventsFactory

import topcoffea.modules.fileReader as fr
import topcoffea.modules.QuadFitTools as qft
from topcoffea.scripts.make_html import make_html

# This is more or less a placeholder script
#   - It shows  an example of how we might want to access the quadratic fit information using topcoffea.modules.QuadFitTools
#   - Currently the script doesn't do much (just processes a single ttH file, prints where fits cross a threshold, and makes 1d quadratic plots)
#   - So this is probably not all that useful right now, but might give us a place to build from in the future
#   - Example usage: python make_1d_quad_plots.py -o ~/www/some/dir/in/your/web/area

def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", default="/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch2/naodOnly_step/v2/nAOD_step_ttHJet_all22WCsStartPtCheckdim6TopMay20GST_run0/NAOD-00000_10769.root", help = "The path the input files should be saved to")
    parser.add_argument("-o", "--output-path", default=".", help = "The path the output files should be saved to")
    parser.add_argument("-r", "--redirector", default="root://deepthought.crc.nd.edu/", help = "Redirector to use for XrootD")
    args = parser.parse_args()

    # Make a tmp output directory in curren dir a different dir is not specified
    timestamp_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    save_dir_path = args.output_path
    if args.output_path == ".":
        outdir_name = "tmp_quad_plos_"+timestamp_tag
        os.mkdir(outdir_name)
        save_dir_path = os.path.join(args.output_path,outdir_name)

    # Open an example naod file
    redirector = args.redirector
    in_file = args.input_path
    in_file = redirector + in_file

    # Get the events object and wc names from the input file
    events = NanoEventsFactory.from_root(in_file).events()
    wc_names_lst = fr.GetListOfWCs(in_file)

    # Get the wc fit dict
    wc_fit_arr = qft.get_summed_quad_fit_arr(events)
    wc_fit_dict = qft.get_quad_fit_dict(wc_names_lst,wc_fit_arr)

    # Scale the WC fit dict to the SM
    wc_fit_dict = qft.scale_to_sm(wc_fit_dict)

    # Print where the fits cross 1.1 (i.e. scaling the SM up by 10%)
    # Not really useful right now, but at some point might need to do a variation of this for the equivalent of TOP-19-001's Table 1
    threshold = 1.1
    for wc_name in wc_names_lst:
        quad_fit_1d = qft.get_1d_fit(wc_fit_dict,wc_name)
        values_at_threshold = qft.find_where_fit_crosses_threshold(quad_fit_1d,threshold)
        print(f"({wc_name} crosses {threshold} at: {values_at_threshold}")

    # Make 1d quad plots for all the WCs
    yaxis_str = "$\sigma/\sigma_{SM}$"
    for wc_name in wc_names_lst:
        fit_coeffs_1d = qft.get_1d_fit(wc_fit_dict,wc_name)
        xaxis_lims = qft.ARXIV1901_LIMS.get(wc_name,qft.TOP19001_LIMS.get(wc_name,[-10,10])) # Use lim from 1901 theory paper if it exists, or TOP-19-001 if it exists, or -10,10 otherwise
        qft.make_1d_quad_plot(
            {wc_name: fit_coeffs_1d},
            wc_name,
            yaxis_str,
            title=wc_name,
            xaxis_range = xaxis_lims,
            save_dir=save_dir_path,
        )

    # Make an index.html file if saving to web area
    if "www" in save_dir_path:
        make_html(save_dir_path)


if __name__ == "__main__":
    main()
