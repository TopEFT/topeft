import argparse
import json
import sys

import topcoffea.modules.YieldTools as yt

# This script takes two json files of yields, and prints out information about how they compare

def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("yields_file_1", help = "The name of the first json file")
    parser.add_argument("yields_file_2", help = "The name of the second json file")
    parser.add_argument("-t1", "--tag1", default="Yields 1", help = "A string to describe the first set of yields")
    parser.add_argument("-t2", "--tag2", default="Yields 2", help = "A string to describe the second set of yields")
    parser.add_argument("-q" , "--quiet", action='store_true', help = "Do not print out info about sample")
    parser.add_argument("--tolerance", type=float, default=1e-5, help = "Maximum absolute difference allowed between the sets of yields")
    args = parser.parse_args()

    # Load the jsons
    with open(args.yields_file_1,"r") as f1: data_1 = f1.read()
    with open(args.yields_file_2,"r") as f2: data_2 = f2.read()
    yld_dict_1 = json.loads(data_1)
    yld_dict_2 = json.loads(data_2)

    # Get the difference between the yields
    print("yld_dict_1",yld_dict_1)
    pdiff_dict = yt.get_diff_between_nested_dicts(yld_dict_1,yld_dict_2,difftype="percent_diff")
    diff_dict  = yt.get_diff_between_nested_dicts(yld_dict_1,yld_dict_2,difftype="absolute_diff")

    # Print the yields
    if not args.quiet:

        yt.print_yld_dicts(yld_dict_1,args.tag1)
        yt.print_yld_dicts(yld_dict_2,args.tag2)
        yt.print_yld_dicts(pdiff_dict,f"Percent diff between {args.tag1} and {args.tag2}")
        yt.print_yld_dicts(diff_dict,f"Diff between {args.tag1} and {args.tag2}")

    # Raise errors if yields are too different
    yields_agree_bool = yt.print_yld_dicts(pdiff_dict,f"Percent diff between {args.tag1} and {args.tag2}",tolerance=args.tolerance)
    if not yields_agree_bool:
        sys.exit(1)


if __name__ == "__main__":
    main()
