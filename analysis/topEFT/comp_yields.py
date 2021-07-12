import argparse
import json
import sys

import topcoffea.modules.MakeLatexTable as mlt
from topcoffea.modules.YieldTools import YieldTools

# This script takes two json files of yields, and prints out information about how they compare
# Example usage:
# python comp_yields.py -f1 TOP-19-001 -f2 yields.json

def main():

    yt = YieldTools()

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", "--yields_file_1", default="yields.json", help = "The name of the first json file")
    parser.add_argument("-f2", "--yields_file_2", default="yields.json", help = "The name of the second json file")
    parser.add_argument("-t1", "--tag_1", default="Yields 1", help = "A string to describe the first set of yields")
    parser.add_argument("-t2", "--tag_2", default="Yields 2", help = "A string to describe the second set of yields")
    parser.add_argument("-q" , "--quiet", action='store_true', help = "Do not print out info about sample")
    args = parser.parse_args()

    # Load the jsons
    if args.yields_file_1 == "TOP-19-001":
        yld_dict_1 = yt.TOP19001_YLDS
    else:
        with open(args.yields_file_1,"r") as f1:
            data_1 = f1.read()
        yld_dict_1 = json.loads(data_1)

    if args.yields_file_2 == "TOP-19-001":
        yld_dict_2 = yt.TOP19001_YLDS
    else:
        with open(args.yields_file_2,"r") as f2:
            data_2 = f2.read()
        yld_dict_2 = json.loads(data_2)

    # Get the difference between the yields
    pdiff_dict = yt.get_diff_between_nested_dicts(yld_dict_1,yld_dict_2,difftype="percent_diff")
    diff_dict  = yt.get_diff_between_nested_dicts(yld_dict_1,yld_dict_2,difftype="absolute_diff")

    # Print the yields
    if not args.quiet:

        yt.print_yld_dicts(yld_dict_1,args.tag_1)
        yt.print_yld_dicts(yld_dict_2,args.tag_2)
        yt.print_yld_dicts(pdiff_dict,f"Percent diff between {args.tag_1} and {args.tag_2}")
        yt.print_yld_dicts(diff_dict,f"Diff between {args.tag_1} and {args.tag_2}")

        mlt.print_begin()
        mlt.print_latex_yield_table(yld_dict_1,key_order=yt.PROC_MAP.keys(),subkey_order=yt.CAT_LST,tag=args.tag_1)
        mlt.print_latex_yield_table(yld_dict_2,key_order=yt.PROC_MAP.keys(),subkey_order=yt.CAT_LST,tag=args.tag_2)
        mlt.print_latex_yield_table(pdiff_dict,key_order=yt.PROC_MAP.keys(),subkey_order=yt.CAT_LST,tag=f"Percent diff between {args.tag_1} and {args.tag_2}")
        mlt.print_latex_yield_table(diff_dict, key_order=yt.PROC_MAP.keys(),subkey_order=yt.CAT_LST,tag=f"Diff between {args.tag_1} and {args.tag_2}")
        mlt.print_end()

    # Raise errors if yields are too different
    yields_agree_bool = yt.print_yld_dicts(pdiff_dict,f"Percent diff between {args.tag_1} and {args.tag_2}",tolerance=1e-8)
    if not yields_agree_bool:
        sys.exit(1)


if __name__ == "__main__":
    main()
