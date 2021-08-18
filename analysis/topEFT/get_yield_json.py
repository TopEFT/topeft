import argparse
import json
import datetime
from topcoffea.modules.YieldTools import YieldTools
import topcoffea.modules.MakeLatexTable as mlt

# This script takes a pkl file, finds the yields in the analysis categories, saves the yields to a json
#   - If you do not specify a pkl file path, will default to "hists/plotsTopEFT.pkl.gz"
#   - Example usage: python get_yield_json.py -f histos/plotsTopEFT.pkl.gz

def main():

    yt = YieldTools()

    timestamp_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    parser.add_argument("-y", "--year", default="2017", help = "The year of the sample")
    parser.add_argument("-t", "--tag", default="Sample", help = "A string to describe the pkl file")
    parser.add_argument("-n", "--json_name", default="yields", help = "Name of the json file to save")
    parser.add_argument("-q", "--quiet", action="store_true", help = "Do not print out anything")
    parser.add_argument("-l", "--lepflav", action="store_true", help = "Do not sum over the lep flav categories, cannot use unless input file is split by lep flavors")
    parser.add_argument("-j", "--njets", action="store_true", help = "Do not sum over the njets categories")
    args = parser.parse_args()

    # Get the histograms, check if split into lep flavors
    hin_dict = yt.get_hist_from_pkl(args.pkl_file_path)
    if not yt.is_split_by_lepflav(hin_dict) and args.lepflav:
        raise Exception("Cannot specify --lepflav option, the yields file is not split by lepton flavor")

    # Put the yields into a dict
    yld_dict = yt.get_yld_dict(hin_dict,args.year,njets=args.njets,lepflav=args.lepflav)

    #yt.print_em_ratios(yld_dict)
    #yld_dict = yt.scale_ylds_by_em_factor(yld_dict,0.75/0.45)
    #yld_dict = yt.sum_over_lepcats(yld_dict)
    #yt.print_yld_dicts(yld_sums,"Sum over lep cats")
    #print("\n\nAfter scaling the dict:")
    #yt.print_em_ratios(yld_dict)

    # Print info about the file
    if not args.quiet:
        yt.print_hist_info(args.pkl_file_path)
        yt.print_yld_dicts(yld_dict,args.tag)
        if not args.lepflav and not args.njets:
            mlt.print_latex_yield_table(yld_dict,key_order=yt.PROC_MAP.keys(),subkey_order=yt.CAT_LST,tag=args.tag,print_begin_info=True,print_end_info=True)
        else:
            mlt.print_latex_yield_table(yld_dict,tag=args.tag,print_begin_info=True,print_end_info=True,column_variable="keys")


    # Save to a json
    out_json_name = args.json_name
    if args.json_name == parser.get_default("json_name"):
        out_json_name = out_json_name + "_" + timestamp_tag
    with open(out_json_name+".json", "w") as out_file:
        json.dump(yld_dict, out_file, indent=4)
    print(f"Saved json file: {out_json_name}.json\n")
    
if __name__ == "__main__":
    main()
