import os
import shutil
import argparse
import json
from topeft.modules.paths import topeft_path

# This script does some basic checks of the cards and templates produced by the `make_cards.py` script.
#   - It also can parse the condor log files and dump a summary of the contents
#   - Additionally, it can also grab the right set of ptz and lj0pt templates (for the right categories) used in TOP-22-006

# Lines that show up in the condor err files that we want to ignore
IGNORE_LINES = [
    "FutureWarning: In coffea version v2023.3.0 (target date: 31 Mar 2023), this will be an error.",
    "(Set coffea.deprecations_as_errors = True to get a stack trace now.)",
    "ImportError: coffea.hist is deprecated",
    "warnings.warn(message, FutureWarning)",
]

# Return list of lines in a file
def read_file(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

# Check if we want to ignore the line or not (based on whether or not any of a list of strings we don't care about shows up in the line)
def ignore_line(line_to_check,list_of_str_to_ignore=IGNORE_LINES):
    ignore = False
    for str_to_ignore in list_of_str_to_ignore:
        if str_to_ignore in line_to_check:
            ignore = True
    return ignore

def extract_number(item):
    return str(''.join(char for char in item if char.isdigit()))

# Check the output of the datacard maekr
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("datacards_path", help = "The path to the directory with the datacards in it.")
    parser.add_argument("-c", "--check-condor-logs", action="store_true", help = "Check the contents of the condor err files.")
    parser.add_argument("-s", "--set-up-top22006", action="store_true", help = "Copy the ptz and lj0pt cards used in TOP-22-006 into their own directory.")
    parser.add_argument("-z", "--set-up-offZdivision", action="store_true", help = "Copy the ptz and lj0pt cards with 3l offZ division.")
    parser.add_argument("-t", "--tau-flag", action="store_true", help = "Copy the ptz, lj0pt, and ptz_wtau cards for tau channels.")
    parser.add_argument("-f", "--fwd-flag", action="store_true", help = "Copy the ptz, lj0pt, and lt cards for forward channels.")
    args = parser.parse_args()
    print(args)

    ###### Check that you run one only type of analysis ######

    # collect your booleans
    flags = [
        args.set_up_top22006,
        args.set_up_offZdivision,
        args.tau_flag,
        args.fwd_flag,
    ]
    
    # check exactly one is True
    if sum(flags) != 1:
        raise ValueError(
            "Exactly one of --set_up_top22006, "
            "--set_up_offZdivision, --tau_flag, --fwd_flag must be set."
        )
    
    # now you can safely branch
    if args.set_up_top22006:
        import_sr_ch_lst = select_ch_lst["TOP22_006_CH_LST_SR"]
    elif args.set_up_offZdivision:
        import_sr_ch_lst = select_ch_lst["OFFZ_SPLIT_CH_LST_SR"]
    elif args.tau_flag:
        import_sr_ch_lst = select_ch_lst["TAU_CH_LST_SR"]
    elif args.fwd_flag:
        import_sr_ch_lst = select_ch_lst["FWD_CH_LST_SR"]

    ###### Print out general info ######

    with open(os.path.join(args.datacards_path,'scalings-preselect.json'), 'r') as file:
        scalings_content = json.load(file)

    # Count the number of text data cards and root templates
    n_text_cards = 0
    n_root_templates = 0
    datacard_files = os.listdir(args.datacards_path)
    for fname in datacard_files:
        if fname.startswith("ttx_multileptons") and fname.endswith(".txt"):
            n_text_cards += 1
        if fname.startswith("ttx_multileptons") and fname.endswith(".root"):
            n_root_templates += 1

    # Print out what we learned
    print(f"\nSummary of cards and templates in {args.datacards_path}:")
    print(f"\tNumber of text cards    : {n_text_cards}")
    print(f"\tNumber of root templates: {n_root_templates}")


    # Check the condor err files
    if args.check_condor_logs:
        lines_from_condor_err_to_print = []
        lines_from_condor_out_to_print = []
        condor_logs_path = os.path.join(args.datacards_path,"job_logs")
        condor_log_files = os.listdir(condor_logs_path)
        for fname in condor_log_files:
            # Parse the .err files
            if fname.endswith(".err"):
                err_file_lines = read_file(os.path.join(condor_logs_path,fname))
                for line in err_file_lines:
                    if not ignore_line(line):
                        lines_from_condor_err_to_print.append((fname,line))
            # Parse the .out files
            if fname.endswith(".out"):
                out_file_lines = read_file(os.path.join(condor_logs_path,fname))
                for line in out_file_lines:
                    if "ERROR" in line:
                        lines_from_condor_out_to_print.append((fname,line))

        # Print out what we learned
        print(f"\nSummary of condor err files in {condor_logs_path}:")
        print(f"\tNumber of non-ingnored lines in condor err files: {len(lines_from_condor_err_to_print)}")
        for line in lines_from_condor_err_to_print:
            print(f"\t\t* In {line[0]}: {line[1]}")
        print(f"\tNumber of ERROR lines in condor out files: {len(lines_from_condor_out_to_print)}")
        for line in lines_from_condor_out_to_print:
            print(f"\t\t* In {line[0]}: {line[1]}")

    ####### Copy the TOP-22-006 relevant files to their own dir ######


    with open(topeft_path("channels/ch_lst_test.json"), "r") as ch_json:
        select_ch_lst = json.load(ch_json)
        #reading the macro analysis setup
        if args.set_up_top22006:
            import_sr_ch_lst = select_ch_lst["TOP22_006_CH_LST_SR"]
        if args.set_up_offZdivision:
            import_sr_ch_lst = select_ch_lst["OFFZ_SPLIT_CH_LST_SR"]
        if args.tau_flag:
            import_sr_ch_lst = select_ch_lst["TAU_CH_LST_SR"]
        if args.fwd_flag:
            import_sr_ch_lst = select_ch_lst["FWD_CH_LST_SR"]

        CATSELECTED = []

        #looping over 2l, 3l and 4l
        for lep_cat, lep_cat_dict in import_sr_ch_lst.items():
            lep_ch_list = lep_cat_dict['lep_chan_lst']
            jet_list = lep_cat_dict['jet_lst']
            jet_list = [extract_number(item) for item in jet_list]
            #looping over each region within the lep category
            for lep_ch in lep_ch_list:
                lep_ch_name = lep_ch[0]
                for jet in jet_list:
                    # special channels to be binned by ptz instead of lj0pt
                    if lep_ch_name == "3l_onZ_1b" or (lep_ch_name == "3l_onZ_2b" and (int(jet) == 4 or int(jet) == 5)):
                        channelname = lep_ch_name + "_" + jet + "j_ptz"
                    elif args.set_up_offZdivision and ( "high" in lep_ch_name  or "low" in lep_ch_name ): # extra channels from offZ division binned by ptz
                        channelname = lep_ch_name + "_" + jet + "j_ptz"
                    elif args.tau_flag and ("2los" in lep_ch_name):
                        channelname = lep_ch_name + "_" + jet + "j_ptz"
                    elif args.tau_flag and ("1tau_onZ" in lep_ch_name):
                        channelname = lep_ch_name + "_" + jet + "j_ptz_wtau"
                    elif args.fwd_flag and ("fwd" in lep_ch_name or "2lss_p" in lep_ch_name or "2lss_m" in lep_ch_name):
                        channelname = lep_ch_name + "_" + jet + "j_lt"
                    else:
                        channelname = lep_ch_name + "_" + jet + "j_lj0pt"
                    CATSELECTED.append(channelname)

    CATSELECTED = sorted(CATSELECTED)
    print("\nCATSELECTED", CATSELECTED, len(CATSELECTED), "\n")
    # Grab the ptz-lj0pt cards we want for TOP-22-006, copy into a dir
    n_txt = 0
    n_root = 0
    ptzlj0pt_path = os.path.join(args.datacards_path,"ptz-lj0pt_withSys")
    os.mkdir(ptzlj0pt_path)
    if args.set_up_top22006:
        print(f"\nCopying TOP-22-006 relevant files to {ptzlj0pt_path}...")

    if args.set_up_offZdivision:
        print(f"\nCopying 3l-offZ-division relevant files to {ptzlj0pt_path}...")
    for fname in datacard_files:
        file_name_strip_ext = os.path.splitext(fname)[0]
        for file in CATSELECTED:
            if file in file_name_strip_ext:
                shutil.copyfile(os.path.join(args.datacards_path,fname),os.path.join(ptzlj0pt_path,fname))
                if fname.endswith(".txt"): n_txt += 1
                if fname.endswith(".root"): n_root += 1
    #also copy the selectedWCs.txt file
    shutil.copyfile(os.path.join(args.datacards_path,"selectedWCs.txt"),os.path.join(ptzlj0pt_path,"selectedWCs.txt"))

    for item in scalings_content:
        channel_name = item.get("channel")
        if channel_name in CATSELECTED:
            ch_index = CATSELECTED.index(channel_name) + 1
            item["channel"] = "ch" + str(ch_index)
        else:
            scalings_content = [d for d in scalings_content if d != item]

    with open(os.path.join(ptzlj0pt_path, 'scalings.json'), 'w') as file:
        json.dump(scalings_content, file, indent=4)

    # Check that we got the expected number and print what we learn
    print(f"\tNumber of text templates copied: {n_txt}")
    print(f"\tNumber of root templates copied: {n_txt}")
    print(args.tau_flag)
    print((n_txt != 60) or (n_root != 60))
    print((args.tau_flag and ((n_txt != 60) or (n_root != 60))))
    if (args.set_up_top22006 and ((n_txt != 43) or (n_root != 43)))   or   (args.set_up_offZdivision and ((n_txt != 75) or (n_root != 75))   or   (args.tau_flag and ((n_txt != 60) or (n_root != 60)))):
        raise Exception(f"Error, unexpected number of text ({n_txt}) or root ({n_root}) files copied")
    print("Done.\n")


main()
