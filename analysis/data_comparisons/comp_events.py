import json

# Return a dictionary of the contents of a json
def get_dict_from_json(json_path):
    with open(json_path,"r") as f:
        data = f.read()
    out_dict = json.loads(data)
    return out_dict

# Retrun a list of lines from a file
def read_file(file_path):
    with open(file_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

# Takes a list of strings, returns as a list of tuples
def get_runlumievent_tup_from_strs(in_lst):
    out_lst = []
    for rle_str in in_lst:
        run,lumi,event = rle_str.split(":")
        out_lst.append((int(run),int(lumi),int(event)))
    return out_lst

# Takes as input a dict with keys of years, subkeys of channels
# Returns a dict with all years summed
def sum_over_years(in_dict):
    out_dict = {}
    for year in in_dict.keys():
        for chan_name in in_dict[year].keys():
            if chan_name not in out_dict:
                out_dict[chan_name] = []
            event_lst = in_dict[year][chan_name]["events"]
            out_dict[chan_name] = out_dict[chan_name] + event_lst

    return out_dict

# Takes as input a dict with keys of channels
# Returns a list for only the specified list of channels
def get_channel_subset(in_dict,chan_lst):
    out_lst = []
    for chan_name in in_dict.keys():
        if chan_name in chan_lst:
            out_lst = out_lst + in_dict[chan_name]
    return out_lst

# Compares two sets and prints out info about how they compare
def print_set_comp_info(set1,set2,tag1,tag2):

    set1_unique = set1.difference(set2)
    set2_unique = set2.difference(set1)
    set1set2_common = set1.intersection(set2)
    set1set2_total = set1.union(set2)

    n_set1 = float(len(set1))
    n_set2 = float(len(set2))

    n_set1_unique = len(set1_unique)
    n_set2_unique = len(set2_unique)

    n_set1set2_common = len(set1set2_common)
    n_set1set2_total = len(set1set2_total)

    print(f"Comparing {tag1} and {tag2}:")

    print(f"\tTotal evens in {tag1}: {n_set1}")
    print(f"\tTotal evens in {tag2}: {n_set2}")
    print(f"\tCommon events  : {n_set1set2_common}")
    print(f"\tUnion of events: {n_set1set2_total}")

    print(f"\tUnique to {tag1}: {n_set1_unique} ({100*n_set1_unique/n_set1}% of {tag1})")
    print(f"\tUnique to {tag2}: {n_set2_unique} ({100*n_set2_unique/n_set2}% of {tag2})")
    print(f"\tPercent of {tag1} events that overlap with {tag2}: {100*n_set1set2_common/n_set1}")
    print(f"\tPercent of {tag2} events that overlap with {tag1}: {100*n_set1set2_common/n_set2}")


# Main function
def main():

    # Read in the top 22006 event numbers
    top22006_event_json           = "event_lists/fullR2_data_passing_events_summary_with_categories.json"
    ttH_framework_legacy_txt      = "event_lists/events_ttH_analysis_legacy.txt"
    ttH_framework_ultralegacy_txt = "event_lists/events_ttH_analysis_ultralegacy.txt"

    # Get the 3l on Z dict from top22006
    top22006_dict = get_dict_from_json(top22006_event_json)
    top22006_dict = sum_over_years(top22006_dict)

    # Get the 3l on Z lists 
    top22006_onZ_lst        = get_channel_subset(top22006_dict,["3l_onZ_1b","3l_onZ_2b"])
    tth_legacy_onZ_lst      = read_file(ttH_framework_legacy_txt)
    tth_ultralegacy_onZ_lst = read_file(ttH_framework_ultralegacy_txt)

    # Get the 3l on Z sets
    top22006_onZ_set        = set(top22006_onZ_lst)
    tth_legacy_onZ_set      = set(tth_legacy_onZ_lst)
    tth_ultralegacy_onZ_set = set(tth_ultralegacy_onZ_lst)

    print("\n-----------------------\n")
    print_set_comp_info(tth_legacy_onZ_set,tth_ultralegacy_onZ_set,"tthLeg","tthUL")
    print("\n-----------------------\n")
    print_set_comp_info(top22006_onZ_set,tth_ultralegacy_onZ_set,"topcoffea","tthUL")
    print("\n-----------------------\n")

    tth_legacy_onZ_tup_lst = get_runlumievent_tup_from_strs(tth_legacy_onZ_lst)
    tth_ultralegacy_onZ_tup_lst = get_runlumievent_tup_from_strs(tth_ultralegacy_onZ_lst)


# A function for just comparing the mmm yields
def comp_mmm():

    ttHframwk_onZ_mmm        = "event_lists/ul_mmm.txt"
    topcoffea_onZ_mmm        = "event_lists/topcoffea_lepflav_mmm.txt"
    topcoffea_onZ_mmm_cleanWithTau = "event_lists/topcoffea_lepflav_mmm_cleanJetsWithTaus.txt"

    ttHframwk_onZ_mmm_set = set(read_file(ttHframwk_onZ_mmm))
    topcoffea_onZ_mmm_set = set(read_file(topcoffea_onZ_mmm))
    topcoffea_onZ_mmm_cleanWithTau_set = set(read_file(topcoffea_onZ_mmm_cleanWithTau))

    print_set_comp_info(ttHframwk_onZ_mmm_set,topcoffea_onZ_mmm_set,"tthframwk","topcoffea")
    print_set_comp_info(ttHframwk_onZ_mmm_set,topcoffea_onZ_mmm_cleanWithTau_set,"ttH","tau")


if __name__ == "__main__":
    main()
    #comp_mmm()
