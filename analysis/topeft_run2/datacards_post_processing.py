import os
import shutil
import argparse
import json
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

# The list of ptz and lj0pt we choose to use in each category for TOP-22-006
TOP22006_CATEGORIES = [

    "2lss_4t_m_4j_lj0pt",
    "2lss_4t_m_5j_lj0pt",
    "2lss_4t_m_6j_lj0pt",
    "2lss_4t_m_7j_lj0pt",
    "2lss_4t_p_4j_lj0pt",
    "2lss_4t_p_5j_lj0pt",
    "2lss_4t_p_6j_lj0pt",
    "2lss_4t_p_7j_lj0pt",
    "2lss_m_4j_lj0pt",
    "2lss_m_5j_lj0pt",
    "2lss_m_6j_lj0pt",
    "2lss_m_7j_lj0pt",
    "2lss_p_4j_lj0pt",
    "2lss_p_5j_lj0pt",
    "2lss_p_6j_lj0pt",
    "2lss_p_7j_lj0pt",
    "3l_m_offZ_1b_2j_lj0pt",
    "3l_m_offZ_1b_3j_lj0pt",
    "3l_m_offZ_1b_4j_lj0pt",
    "3l_m_offZ_1b_5j_lj0pt",
    "3l_m_offZ_2b_2j_lj0pt",
    "3l_m_offZ_2b_3j_lj0pt",
    "3l_m_offZ_2b_4j_lj0pt",
    "3l_m_offZ_2b_5j_lj0pt",
    "3l_onZ_1b_2j_ptz",
    "3l_onZ_1b_3j_ptz",
    "3l_onZ_1b_4j_ptz",
    "3l_onZ_1b_5j_ptz",
    "3l_onZ_2b_2j_lj0pt",
    "3l_onZ_2b_3j_lj0pt",
    "3l_onZ_2b_4j_ptz",
    "3l_onZ_2b_5j_ptz",
    "3l_p_offZ_1b_2j_lj0pt",
    "3l_p_offZ_1b_3j_lj0pt",
    "3l_p_offZ_1b_4j_lj0pt",
    "3l_p_offZ_1b_5j_lj0pt",
    "3l_p_offZ_2b_2j_lj0pt",
    "3l_p_offZ_2b_3j_lj0pt",
    "3l_p_offZ_2b_4j_lj0pt",
    "3l_p_offZ_2b_5j_lj0pt",
    "4l_2j_lj0pt",
    "4l_3j_lj0pt",
    "4l_4j_lj0pt",
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


# Check the output of the datacard maekr
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("datacards_path", help = "The path to the directory with the datacards in it.")
    parser.add_argument("-c", "--check-condor-logs", action="store_true", help = "Check the contents of the condor err files.")
    parser.add_argument("-s", "--set-up-top22006", action="store_true", help = "Copy the ptz and lj0pt cards used in TOP-22-006 into their own directory.")
    args = parser.parse_args()

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

    # Grab the ptz-lj0pt cards we want for TOP-22-006, copy into a dir
    n_txt = 0
    n_root = 0
    if args.set_up_top22006:
        ptzlj0pt_path = os.path.join(args.datacards_path,"ptz-lj0pt_withSys")
        os.mkdir(ptzlj0pt_path)
        print(f"\nCopying TOP-22-006 relevant files to {ptzlj0pt_path}...")
        for fname in datacard_files:
            file_name_strip_ext = os.path.splitext(fname)[0]
            for file in TOP22006_CATEGORIES:
                if file in file_name_strip_ext:
                    shutil.copyfile(os.path.join(args.datacards_path,fname),os.path.join(ptzlj0pt_path,fname))
                    if fname.endswith(".txt"): n_txt += 1
                    if fname.endswith(".root"): n_root += 1
        #also copy the selectedWCs.txt file
        shutil.copyfile(os.path.join(args.datacards_path,"selectedWCs.txt"),os.path.join(ptzlj0pt_path,"selectedWCs.txt"))

        for item in scalings_content:
            channel_name = item.get("channel") 
            if channel_name in TOP22006_CATEGORIES:
                ch_index = TOP22006_CATEGORIES.index(channel_name) + 1
                item["channel"] = "ch" + str(ch_index)
            else:
                scalings_content = [d for d in scalings_content if d != item]

        with open(os.path.join(ptzlj0pt_path, 'scalings.json'), 'w') as file:
            json.dump(scalings_content, file, indent=4)            

        # Check that we got the expected number and print what we learn
        print(f"\tNumber of text templates copied: {n_txt}")
        print(f"\tNumber of root templates copied: {n_txt}")
        if ((n_txt != 43) or (n_root != 43)):
            raise Exception(f"Error, unexpected number of text ({n_txt}) or root ({n_root}) files copied")
        print("Done.\n")



main()


