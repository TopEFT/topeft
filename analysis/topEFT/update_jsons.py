import os
import argparse

import json

from topcoffea.modules.YieldTools import YieldTools
from topcoffea.modules.paths import topcoffea_path
from topcoffea.modules.utils import regex_match, load_sample_json_file

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

def update_json(fname,dry_run=False,outname=None,verbose=False,**kwargs):
    '''
        Description:
            Attempts to open a json file, modify one or more of the outermost keys, and then save
            the new json. If dry_run is set to true, then skip writing to an output file. If outname
            is None then the file name will be set to the original and overwrite it.

        Note:
            fname will in general will be the full file path to the desired file, so don't expect it
            to be saved in the same directory as the original w/o making sure the file path is correct
    '''
    jsn = load_sample_json_file(fname)
    jsn.pop('redirector')   # Don't currently store this info in the json
    if verbose:
        h,t = os.path.split(fname)
        print(f"Updating {t}")
    for k,new in kwargs.items():
        if not k in jsn:
            raise KeyError(f"Unknown json key specified: {k}")
        old = jsn[k]
        # if type(old) != type(new):
        if not isinstance(old,type(new)):
            raise TypeError(f"New should at least be a base class of old: {type(old)} vs {type(new)}")
        if verbose:
            print(f"\t{k}: {old} --> {new}")
        jsn[k] = new
    if dry_run:
        return
    new_file = fname if outname is None else outname
    with open(new_file,'w') as f:
        print(f'>> Writing updated file to {new_file}')
        json.dump(jsn,f,indent=2)

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
    ignore_dirs  = args.ignore_dirs
    match_files  = args.match_files
    ignore_files = args.ignore_files
    dry_run      = args.dry_run
    verbose      = args.verbose

    def get_files(top_dir,ignore_dirs=[],match_files=[],ignore_files=[]):
        '''
            Description:
                Walks through an entire directory structure searching for files. Returns a list of
                matching files with absolute path included.

                Can optionally be given list of regular
                expressions to skip certain directories/files or only match certain types of files
        '''
        found = []
        for root, dirs, files in os.walk(top_dir):
            dir_matches = regex_match(dirs,regex_lst=ignore_dirs)
            for m in dir_matches:
                print(f"Skipping directory: {m}")
                dirs.remove(m)
            if match_files:
                files = regex_match(files,match_files)
            file_matches = regex_match(files,regex_lst=ignore_files)
            for m in file_matches:
                print(f"Skipping file: {m}")
                files.remove(m)     # Removes 'm' from the file list, not the actual file on disk
            for f in files:
                fpath = os.path.join(root,f)
                found.append(fpath)
        return found

    tools = YieldTools()

    ignore_dirs.extend(['subsets_of_private_UL_.*','private_UL_backup'])    # These sub-directories have duplicated JSON names with those from private_UL
    match_files.extend(['.*\\.json'])                   # Make sure to always only find .json files
    ignore_files.extend(['lumi.json','params.json'])    # These are not sample json files

    json_fpaths = get_files(topcoffea_path("json"),
        ignore_dirs=['subsets_of_private_UL_.*','private_UL_backup'],
        match_files=['.*\\.json'],
        ignore_files=['lumi.json','params.json']
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
