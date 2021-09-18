import os
import datetime
import argparse
import matplotlib.pyplot as plt

from coffea import hist
from topcoffea.modules.HistEFT import HistEFT

from topcoffea.modules.YieldTools import YieldTools
from topcoffea.modules.GetValuesFromJsons import get_lumi
from topcoffea.plotter.make_html import make_html

# Some options for plotting the data
DATA_ERR_OPS = {'linestyle':'none', 'marker': '.', 'markersize': 10., 'color':'k', 'elinewidth': 1,}

# The channels that define the CR categories
CR_CHAN_DICT = {
    "cr_2los_Z" : [
        "2los_ee_CRZ_0j",
        "2los_mm_CRZ_0j",
    ],
    "cr_2los_tt" : [
        "2los_em_CRtt_2j",
    ],
    "cr_2lss" : [
        "2lss_ee_CR_1j",
        "2lss_em_CR_1j",
        "2lss_mm_CR_1j",
        "2lss_ee_CR_2j",
        "2lss_em_CR_2j",
        "2lss_mm_CR_2j",
    ],

    "cr_3l" : [
        "3l_eee_CR_1j",
        "3l_eem_CR_1j",
        "3l_emm_CR_1j",
        "3l_mmm_CR_1j",
    ],
}

# The channels that define the CR categories nor the njets hist (where we do not keep track of jets in the sparse axis)
CR_CHAN_DICT_NO_J = {
    "cr_2los_Z" : [
        "2los_ee_CRZ",
        "2los_mm_CRZ",
    ],
    "cr_2los_tt" : [
        "2los_em_CRtt",
    ],
    "cr_2lss" : [
        "2lss_ee_CR",
        "2lss_em_CR",
        "2lss_mm_CR",
    ],
    "cr_3l" : [
        "3l_eee_CR",
        "3l_eem_CR",
        "3l_emm_CR",
        "3l_mmm_CR",
    ],
}

def main():

    yt = YieldTools()

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    parser.add_argument("-u", "--unit-norm", action="store_true", help = "Unit normalize the plots")
    parser.add_argument("-y", "--year", default="2017", help = "The year of the sample")
    parser.add_argument("-t", "--tag", default="Sample", help = "A string to describe the pkl file")
    parser.add_argument("-o", "--output-path", default=".", help = "The path the output files should be saved to")
    args = parser.parse_args()

    # Whether or not to unit norm the plots
    unit_norm_bool = args.unit_norm

    # Make a tmp output directory in curren dir a different dir is not specified
    timestamp_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    save_dir_path = args.output_path
    outdir_name = "cr_plots_"+timestamp_tag
    save_dir_path = os.path.join(save_dir_path,outdir_name)
    os.mkdir(save_dir_path)

    # Get the histograms
    hin_dict = yt.get_hist_from_pkl(args.pkl_file_path)

    # Print info about histos
    #yt.print_hist_info(args.pkl_file_path,"nbtagsl")
    #exit()

    # Construct list of MC samples
    sample_lst = yt.get_cat_lables(hin_dict,"sample")
    mc_sample_lst = []
    for sample_name in sample_lst:
        if "data" not in sample_name:
            mc_sample_lst.append(sample_name)
    print("\nMC samples:",sample_lst)

    # Loop over hists and make plots
    skip_lst = ["SumOfEFTweights"] # Skip this hist
    for var_name in hin_dict.keys():
        if (var_name in skip_lst): continue
        if (var_name == "njets"):
            cat_hist = CR_CHAN_DICT_NO_J
        else: cat_hist = CR_CHAN_DICT
        print("\nVar name:",var_name)

        # Extract the MC and data hists
        hist_mc = hin_dict[var_name].copy()
        hist_mc = hist_mc.remove(["data_UL17"],"sample")
        hist_mc = hist_mc.integrate("systematic","nominal")
        hist_data = hin_dict[var_name].copy()
        hist_data = hist_data.remove(mc_sample_lst,"sample")
        hist_data = hist_data.integrate("systematic","nominal")

        # Normalize the MC hists
        hist_mc.scale(1000.0*get_lumi(args.year))

        # Loop over the CR categories
        for hist_cat in cat_hist.keys():

            print("\n\tCategory:",hist_cat)

            # The 2los Z category does not require jets
            if (hist_cat == "cr_2los_Z" and "j0" in var_name): continue

            # Make a sub dir for this category
            save_dir_path_tmp = os.path.join(save_dir_path,hist_cat)
            if not os.path.exists(save_dir_path_tmp):
                os.mkdir(save_dir_path_tmp)

            # Integrate over the channels
            histo_data_tmp = hist_data.copy()
            histo_data_tmp = histo_data_tmp.integrate("channel",cat_hist[hist_cat])
            histo_mc_tmp = hist_mc.copy()
            histo_mc_tmp = histo_mc_tmp.integrate("channel",cat_hist[hist_cat])

            # Integrate over the appl category
            # NOTE: Once we merge PR #98 this should not be necessary
            if "2l" in hist_cat:
                if "appl" in histo_mc_tmp.axes():
                    histo_mc_tmp = histo_mc_tmp.integrate("appl","isSR_2l")
                if "appl" in histo_data_tmp.axes():
                    histo_data_tmp = histo_data_tmp.integrate("appl","isSR_2l")
            elif "3l" in hist_cat:
                histo_mc_tmp = histo_mc_tmp.integrate("appl","isSR_3l")
                histo_data_tmp = histo_data_tmp.integrate("appl","isSR_3l")
            else:
                raise Exception

            # Create the plots
            fig, ax = plt.subplots(1, 1, figsize=(11,7))
            cm = plt.get_cmap('tab20')
            NUM_COLORS = 16
            ax.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

            hist.plot1d(
                histo_mc_tmp,
                stack=True,
                density=unit_norm_bool,
                clear=False,
            )

            hist.plot1d(
                histo_data_tmp,
                error_opts = DATA_ERR_OPS,
                stack=False,
                density=unit_norm_bool,
                clear=False,
            )

            ax.autoscale(axis='y')
            title = hist_cat+"_"+var_name
            if unit_norm_bool:
                title = title + "_unitnorm"
            fig.savefig(os.path.join(save_dir_path_tmp,title))
            ax.clear()

            # Make an index.html file if saving to web area
            if "www" in save_dir_path_tmp:
                make_html(save_dir_path_tmp)


if __name__ == "__main__":
    main()
