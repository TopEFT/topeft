import os
import argparse

from topcoffea.modules.paths import topcoffea_path
from topcoffea.modules.utils import regex_match, load_sample_json_file, get_files, get_hist_from_pkl
from topcoffea.modules.update_json import update_json
from topeft.modules.yield_tools import YieldTools

pjoin = os.path.join

# Description:
#   This script runs on a histogram file produced by the sow_processor. It extracts the individual
#   SumOfWeights histograms and then tries to find a matching json file from somewhere in the
#   'topcoffea/json' directory. If it finds a matching file it replaces the value for the
#   'nSumOfWeights' key in the file with the one computed from the histogram.
#
#   This is typically a pre-processing step to running the topeft processor so that both the central
#   and private MC samples can be normalized and treated in the same way. The issue is that when the
#   JSON files for the EFT samples are initially made, the 'nSumOfWeights' that is computed is not
#   correct. This means we had to recompute this value to get the correct normaliztion, but only for
#   the EFT samples! After running this script the EFT sample JSON files will have the correct value
#   and can be treated the same as the central MC samples.
#
# Usage:
#
# Note:
#   A current limitation of this script is that if it encounters two JSON files with the exact
#   same name, but in different sub-directories, then it skips updating that file and instead prints
#   an error displaying the confounding JSONs.

MAX_PDIFF = 1e-7

WEIGHTS_NAME_LST = [
    'nom',
    'ISRUp',
    'ISRDown',
    'FSRUp',
    'FSRDown',
    'renormUp',
    'renormDown',
    'factUp',
    'factDown',
    #'renormfactUp',
    #'renormfactDown',
]

# Construct a dict to hold the hist name and json name, format:
# d = {
#   'varUp': {
#       'hist_name': 'SumOfWeights_varUp',
#       'jsn_key_name': 'nSumOfWeights_varUp',
#   }
# }
def construct_wgt_name_dict(wgt_name_lst):

    def construct_hist_name(wgt_var_str):
        if wgt_var_str == 'nom': wgt_var_str = ''
        else:  wgt_var_str = '_' + wgt_var_str
        return 'SumOfWeights' + wgt_var_str

    def construct_jsn_key_name(wgt_var_str):
        if wgt_var_str == 'nom': wgt_var_str = ''
        else:  wgt_var_str = '_' + wgt_var_str
        return 'nSumOfWeights' + wgt_var_str

    wgt_name_dict = {}
    for wgt_name in wgt_name_lst:
        wgt_name_dict[wgt_name] = {}
        wgt_name_dict[wgt_name]['hist_name'] = construct_hist_name(wgt_name)
        wgt_name_dict[wgt_name]['jsn_key_name'] = construct_jsn_key_name(wgt_name)

    return wgt_name_dict


def main():
    parser = argparse.ArgumentParser(description='You want options? We got options!')
    parser.add_argument('hist_paths', nargs='+', help = 'Paths to the histogram pkl.gz files that we want to load')
    parser.add_argument('--json-dir', nargs='?', default=topcoffea_path("json"), help = 'Path to the directory with JSON files you want to update. Will recurse down into all sub-directories looking for any .json files along the way')
    parser.add_argument('--ignore-dirs', nargs='*', default=[])
    parser.add_argument('--match-files', nargs='*', default=[])
    parser.add_argument('--ignore-files', nargs='*', default=[])
    parser.add_argument('--dry-run', '-n', action='store_true', help = 'Process the histogram files, but do not actually modify the JSON files')
    parser.add_argument('--verbose', '-v', action='store_true', help = 'Prints out the old and new values of the json as it is being updated')

    args = parser.parse_args()
    hist_paths   = args.hist_paths
    json_dir     = args.json_dir
    ignore_dirs  = args.ignore_dirs
    match_files  = args.match_files
    ignore_files = args.ignore_files
    dry_run      = args.dry_run
    verbose      = args.verbose

    yt  = YieldTools()

    ignore_dirs.extend(['subsets_of_private_UL_.*','private_UL_backup'])    # These sub-directories have duplicated JSON names with those from private_UL
    match_files.extend(['.*\\.json'])                   # Make sure to always only find .json files
    ignore_files.extend(['lumi.json','params.json'])    # These are not sample json files

    json_fpaths = get_files(
        json_dir,
        ignore_dirs=ignore_dirs,
        match_files=match_files,
        ignore_files=ignore_files,
        recursive=True,
        verbose=True
    )

    # Get dictionary of names
    wgt_name_dict = construct_wgt_name_dict(WEIGHTS_NAME_LST)

    # Find JSONs and update weights
    for fpath in hist_paths:
        h = get_hist_from_pkl(fpath)
        h_sow_nom = h[wgt_name_dict['nom']['hist_name']] # Note, just using nom here (so we assume all histos include the same samples)
        idents = h_sow_nom.axes['process'] # This is the list of identifiers for the sample axis
        for sname in idents:
            match = regex_match(json_fpaths,regex_lst=[f"{sname}\\.json$"])
            if len(match) != 1:
                print(f"ERROR: Found {len(match)} matches for {sname}! Don't know which json should be modified, so skipping")
                for x in match:
                    print(f"\t{x}")
                continue
            match = match[0]
            jsn = load_sample_json_file(match)

            # Loop over each wgt variation and update JSON
            for wgt_var in wgt_name_dict.keys():

                # Get value from sow hist
                hist_name = wgt_name_dict[wgt_var]['hist_name']
                jsn_key_name = wgt_name_dict[wgt_var]['jsn_key_name']
                new_sow,err = yt.get_yield(h[hist_name],sname)

                # If key already in dict, check if new number looks different than old
                if jsn_key_name in jsn:
                    old = jsn[jsn_key_name]
                    diff = new_sow - old
                    pdiff = diff / old
                    if abs(pdiff) < MAX_PDIFF:
                        continue

                # Update the JSON
                updates = {
                    jsn_key_name: float(new_sow)
                }
                update_json(match,dry_run=dry_run,verbose=verbose,**updates)

if __name__ == "__main__":
    main()
