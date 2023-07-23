import argparse
import json
import pickle
import gzip

# This script opens a pkl file of histograms produced by wwz processor
# Reads the histograms and dumps out the event counts
# Example usage: python get_yld_check.py -f histos/tmp_histo.pkl.gz


# Get number of evens in SRs
def get_counts(histos_dict):

    wwz_sync_sample = 'UL17_WWZJetsTo4L2Nu'

    out_dict = {}
    out_dict[wwz_sync_sample] = {}

    # Get object multiplicity counts (nleps, njets, nbtags)
    ojb_lst = ["nleps","njets","nbtagsl"]
    for obj in ojb_lst:
        nobjs_hist = histos_dict[obj]["all_events"][wwz_sync_sample].values(flow=True)[0]
        tot_objs = 0
        for i,v in enumerate(nobjs_hist):
            tot_objs = tot_objs + (i-1.)*v # Have to adjust for the fact that first bin is underflow, second bin is 0 objects
        out_dict[wwz_sync_sample][obj] = (tot_objs,None) # Save err as None
        #print("\ntotobj",obj,tot_objs)

    # Look at the event counts in one histo (e.g. njets)
    dense_axis = "njets"
    for cat_name in histos_dict[dense_axis].keys():
        val = sum(histos_dict[dense_axis][cat_name][wwz_sync_sample].values()[0])
        out_dict[wwz_sync_sample][cat_name] = (val,None) # Save err as None
        #print(dense_axis,cat_name,val)

    return out_dict


def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    parser.add_argument("-o", "--output-path", default=".", help = "The path the output files should be saved to")
    parser.add_argument("-n", "--output-name", default="counts_wwz_sync", help = "A name for the output directory")
    args = parser.parse_args()

    # Get the counts from the input hiso
    histo_dict = pickle.load(gzip.open(args.pkl_file_path))
    counts_dict = get_counts(histo_dict)

    # Print the counts
    print("\nCounts:")
    for proc in counts_dict.keys():
        for cat,val in counts_dict[proc].items(): print(f"  {cat}:{val[0]}")

    # Dump counts dict to json
    if "json" not in args.output_name: output_name = args.output_name + ".json"
    else: output_name = args.output_name
    with open(output_name,"w") as out_file: json.dump(counts_dict, out_file, indent=4)
    print(f"\nSaved json file: {output_name}\n")


if __name__ == "__main__":
    main()

