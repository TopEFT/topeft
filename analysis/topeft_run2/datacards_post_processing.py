import os
import subprocess
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
    "UserWarning: Numba extension module 'awkward.numba' failed to load due to 'AttributeError(module 'awkward.numba' has no attribute '_register')",
    "entrypoints.init_all()",
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
    parser.add_argument("-a", "--all-analysis", action="store_true", help = "Copy all channels in with fwd and offZ contributions.")
    args = parser.parse_args()

    ###### Check that you run one only type of analysis ######

    # collect your booleans
    flags = [
        args.set_up_top22006,
        args.set_up_offZdivision,
        args.tau_flag,
        args.fwd_flag,
        args.all_analysis,
    ]

    # check exactly one is True
    if sum(flags) != 1:
        raise ValueError(
            "Exactly one of --set_up_top22006, "
            "--set_up_offZdivision, --tau_flag, --fwd_flag must be set."
        )

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
    with open(topeft_path("channels/ch_lst.json"), "r") as ch_json:
        select_ch_lst = json.load(ch_json)
        #reading the macro analysis setup
        if args.set_up_top22006:
            import_sr_ch_lst = select_ch_lst["TOP22_006_CH_LST_SR"]
        elif args.set_up_offZdivision:
            import_sr_ch_lst = select_ch_lst["OFFZ_SPLIT_CH_LST_SR"]
        elif args.tau_flag:
            import_sr_ch_lst = select_ch_lst["TAU_CH_LST_SR"]
        elif args.fwd_flag:
            import_sr_ch_lst = select_ch_lst["FWD_CH_LST_SR"]
        elif args.all_analysis:
            import_sr_ch_lst = select_ch_lst["ALL_CH_LST_SR"]

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
                    if "3l" in lep_ch_name and int(jet) == 1 and 'fwd' not in lep_ch_name: # 1j for fwd only
                        continue
                    #elif "3l_onZ_1b" in lep_ch_name or ("3l_onZ_2b" in lep_ch_name and (int(jet) == 4 or int(jet) == 5)) and 'fwd' not in lep_ch_name:
                    elif (("3l_onZ_1b" in lep_ch_name) or ("3l_onZ_2b" in lep_ch_name and int(jet) in [4, 5])) and 'fwd' not in lep_ch_name:
                        channelname = lep_ch_name + "_" + jet + "j_ptz"
                    elif args.all_analysis and (
                        ("3l_onZ_2b" in lep_ch_name and int(jet) == 1) or
                        ("3l_onZ_1b" in lep_ch_name and int(jet) == 1) or
                        ("offZ_2b_fwd" in lep_ch_name and int(jet) == 1) or
                        ("fwd_p_1tau_offZ" in lep_ch_name and int(jet) > 5)
                        ):
                        continue
                    elif (args.set_up_offZdivision or args.all_analysis) and ( "high" in lep_ch_name  or "low" in lep_ch_name ): # extra channels from offZ division binned by ptz
                        channelname = lep_ch_name + "_" + jet + "j_ptz"
                    elif (args.tau_flag or args.all_analysis) and ("2los" in lep_ch_name):
                        channelname = lep_ch_name + "_" + jet + "j_ptz"
                    elif (args.tau_flag or args.all_analysis) and ("1tau_onZ" in lep_ch_name):
                        channelname = lep_ch_name + "_" + jet + "j_ptz_wtau"
                    elif (args.fwd_flag or args.all_analysis) and ("fwd" in lep_ch_name):
                        channelname = lep_ch_name + "_" + jet + "j_lt"
                    else:
                        channelname = lep_ch_name + "_" + jet + "j_lj0pt"
                    CATSELECTED.append(channelname)

    CATSELECTED = sorted(CATSELECTED)

    # Grab the ptz-lj0pt cards we want for TOP-22-006, copy into a dir
    n_txt = 0
    n_root = 0
    ptzlj0pt_path = os.path.join(args.datacards_path,"ptz-lj0pt_withSys")
    os.mkdir(ptzlj0pt_path)
    if args.set_up_top22006:
        print(f"\nCopying TOP-22-006 relevant files to {ptzlj0pt_path}...")
    elif args.set_up_offZdivision:
        print(f"\nCopying 3l-offZ-division relevant files to {ptzlj0pt_path}...")
    elif args.tau_flag:
        print(f"\nCopying tau analysis relevant files to {ptzlj0pt_path}...")
    elif args.fwd_flag:
        print(f"\nCopying forward jets analysis relevant files to {ptzlj0pt_path}...")

    for fname in datacard_files:
        file_name_strip_ext = os.path.splitext(fname)[0]
        for file in CATSELECTED:
            if file in file_name_strip_ext:
                if fname.endswith(".txt"):
                    bad = subprocess.call([f'grep "observation 0.00" {os.path.join(args.datacards_path,fname)}'], shell=True, stdout=subprocess.DEVNULL)
                    #if bad == 0:
                    #    raise Exception(f"Warning: {file} has 0 observation!")
                shutil.copyfile(os.path.join(args.datacards_path,fname),os.path.join(ptzlj0pt_path,fname))
                if fname.endswith(".txt"): n_txt += 1
                if fname.endswith(".root"): n_root += 1
    #also copy the selectedWCs.txt file
    shutil.copyfile(os.path.join(args.datacards_path,"selectedWCs.txt"),os.path.join(ptzlj0pt_path,"selectedWCs.txt"))

    new_scalings = []
    for ch_index, channel_name in enumerate(CATSELECTED, start=1):
        # find all items that match this channel
        matches = [item for item in scalings_content if item.get("channel") == channel_name]

        if not matches:
            raise ValueError(f"Channel '{channel_name}' not found in scalings_content")

        # update and append in order
        for item in matches:
            new_item = item.copy()
            new_item["channel"] = f"ch{ch_index}"
            new_scalings.append(new_item)

    scalings_content = new_scalings

    with open(os.path.join(ptzlj0pt_path, 'scalings.json'), 'w') as file:
        json.dump(scalings_content, file, indent=4)

    # Check that we got the expected number and print what we learn
    print(f"\tNumber of text templates copied: {n_txt}")
    print(f"\tNumber of root templates copied: {n_root}")
    if (args.set_up_top22006 and ((n_txt != 43) or (n_root != 43)))   or   (args.set_up_offZdivision and ((n_txt != 75) or (n_root != 75))):
        raise Exception(f"Error, unexpected number of text ({n_txt}) or root ({n_root}) files copied")
    print("Done.\n")

main()
