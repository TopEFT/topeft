import argparse
import json
from topcoffea.modules.YieldTools import YieldTools

def main():

    yt = YieldTools()

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", "--yields_file_1", default="yields.json", help = "The first json file that you would like to look at")
    parser.add_argument("-f2", "--yields_file_2", default="yields.json", help = "The second json that you would like to look at")
    parser.add_argument("-t1", "--tag_1", default="Yields 1", help = "A string to describe the first set of yields")
    parser.add_argument("-t2", "--tag_2", default="Yields 2", help = "A string to describe the second set of yields")
    parser.add_argument("-q", "--quiet", action='store_true', help = "Do not print out info about sample")
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

        # TODO: Need to fix the print_latex_yield_table stuff...
        #yt.print_latex_yield_table({},[],"",print_begin_info=True)
        yt.print_latex_yield_table(yld_dict_1,yt.CAT_LST,args.tag_1,print_begin_info=True)
        yt.print_latex_yield_table(yld_dict_2,yt.CAT_LST,args.tag_2)
        yt.print_latex_yield_table(pdiff_dict,yt.CAT_LST,f"Percent diff between {args.tag_1} and {args.tag_2}")
        yt.print_latex_yield_table(diff_dict,yt.CAT_LST,f"Diff between {args.tag_1} and {args.tag_2}",print_end_info=True)
        #yt.print_latex_yield_table({},[],"",print_end_info=True)

    # Raise errors if yields are too different

if __name__ == "__main__":
    main()
