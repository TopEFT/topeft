import os
import argparse

from topcoffea.modules.YieldTools import YieldTools
from topcoffea.modules.paths import topcoffea_path
from topcoffea.modules.utils import regex_match, load_sample_json_file, update_json, get_files

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

WEIGHTS_NAME = 'SumOfWeights'
JSON_KEY_NAME = 'nSumOfWeights'
MAX_PDIFF = 1e-7

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

    tools = YieldTools()

    ignore_dirs.extend(['subsets_of_private_UL_.*','private_UL_backup'])    # These sub-directories have duplicated JSON names with those from private_UL
    match_files.extend(['.*\\.json'])                   # Make sure to always only find .json files
    ignore_files.extend(['lumi.json','params.json'])    # These are not sample json files

    json_fpaths = get_files(json_dir,
        ignore_dirs=ignore_dirs,
        match_files=match_files,
        ignore_files=ignore_files
    )

    for fpath in hist_paths:
        h = tools.get_hist_from_pkl(fpath)
        h_sow = h[WEIGHTS_NAME]
        idents = h_sow.identifiers('sample')
        for sname in idents:
            match = regex_match(json_fpaths,regex_lst=[f"{sname}\\.json$"])
            if len(match) != 1:
                print(f"ERROR: Found {len(match)} matches for {sname}! Don't know which json should be modified, so skipping")
                for x in match:
                    print(f"\t{x}")
                continue
            match = match[0]
            jsn = load_sample_json_file(match)
            old = jsn[JSON_KEY_NAME]
            yld,err = tools.get_yield(h_sow,sname)
            diff = yld - old
            pdiff = diff / old
            if abs(pdiff) < MAX_PDIFF:
                continue

            updates = {
                JSON_KEY_NAME: float(yld)
            }

            update_json(match,dry_run=dry_run,verbose=verbose,**updates)

if __name__ == "__main__":
    main()
