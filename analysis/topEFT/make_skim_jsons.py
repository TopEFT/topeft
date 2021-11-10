import os
import argparse

from topcoffea.modules.paths import topcoffea_path
from topcoffea.modules.utils import regex_match, load_sample_json_file, update_json, get_files

pjoin = os.path.join

def main():
    parser = argparse.ArgumentParser(description='You want options? We got options!')
    parser.add_argument('--json-dir', nargs='?', default=topcoffea_path("json"), help = 'Path to the directory with JSON files you want to update. Will recurse down into all sub-directories looking for any .json files along the way')
    parser.add_argument('--ignore-dirs', nargs='*', default=[])
    parser.add_argument('--match-files', nargs='*', default=[])
    parser.add_argument('--ignore-files', nargs='*', default=[])
    parser.add_argument('--dry-run', '-n', action='store_true', help = 'Process the histogram files, but do not actually modify the JSON files')

    args = parser.parse_args()
    json_dir     = args.json_dir
    ignore_dirs  = args.ignore_dirs
    match_files  = args.match_files
    ignore_files = args.ignore_files
    dry_run      = args.dry_run

    ignore_dirs.extend(['subsets_of_private_UL_.*','private_UL_backup'])    # These sub-directories have duplicated JSON names with those from private_UL
    match_files.extend(['.*\\.json'])                   # Make sure to always only find .json files
    ignore_files.extend(['lumi.json','params.json'])    # These are not sample json files

    json_fpaths = get_files(json_dir,
        ignore_dirs  = ignore_dirs,
        match_files  = match_files,
        ignore_files = ignore_files
    )

    # Example
    new_fdir  = "/hadoop/store/user/awightma/skims/NanoAOD_ULv8/v1/SingleMuon_C_UL2017"
    to_match  = [".*\\.root"]
    new_files = [x.replace("/hadoop","") for x in get_files(new_fdir,match_files=to_match)]

    template_json_fpath = get_files(
        top_dir = pjoin(json_dir,"data_samples/2017"),
        match_files  = ["SingleMuon_C.*UL2017\\.json"],
        ignore_files = [".*_atPSI\\.json"]
    )
    for fp in template_json_fpath:
        print(fp)

if __name__ == "__main__":
    main()

