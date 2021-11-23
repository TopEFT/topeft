import os
import argparse
import json

from topcoffea.modules.paths import topcoffea_path
from topcoffea.modules.utils import regex_match, load_sample_json_file, update_json, get_files

pjoin = os.path.join

# Attempts to match a hadoop dataset to a json dataset based on the file name
def find_json_match(hadoop_skim_dir,json_fpaths):
    hadoop_dataset_name = os.path.split(hadoop_skim_dir)[1]
    for json_fpath in json_fpaths:
        json_dataset_name = os.path.split(json_fpath)[1].replace(".json","")
        # This is a result of the fact that lobster doesnt allow workflows to contain hyphens in the name
        json_dataset_name = json_dataset_name.replace("-","_")
        if hadoop_dataset_name == json_dataset_name:
            return json_fpath
    return None

def main():
    parser = argparse.ArgumentParser(description='Utility script for creating/updating JSON files for skimmed data')
    parser.add_argument('src_dirs',       nargs='*', default=[], metavar='SRC_DIR', help='Path(s) to toplevel directory that contains the lobster skims we want to match to, can also be specified with the "--file" option instead')
    parser.add_argument('--file','-f',    nargs='?', metavar='FILE', help='Text file with paths to the src directories that contain the lobster skims')
    parser.add_argument('--json-dir',     nargs='?', default=topcoffea_path("json"), metavar='DIR', help='Path to the directory with JSON files you want to update. Will recurse down into all sub-directories looking for any .json files along the way')
    parser.add_argument('--output-dir',   nargs='?', default='', metavar='DIR', help='Path to an output directory to save the generated json skim files, if not specified will save to the same directory that the template json is located')
    parser.add_argument('--skim-postfix', default='_NDSkim',metavar='NAME', help='Postfix string to differentiate the skim json from the original, defaults to "_NDSkim"')
    parser.add_argument('--ignore-dirs',  nargs='*', default=[], metavar='PATTERN')
    parser.add_argument('--match-files',  nargs='*', default=[], metavar='PATTERN')
    parser.add_argument('--ignore-files', nargs='*', default=[], metavar='PATTERN')
    parser.add_argument('--dry-run', '-n', action='store_true', help='Run the script, but do not actually modify the JSON files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()
    src_dirs     = args.src_dirs
    json_dir     = args.json_dir
    out_dir      = args.output_dir
    src_file     = args.file
    postfix      = args.skim_postfix
    ignore_dirs  = args.ignore_dirs
    match_files  = args.match_files
    ignore_files = args.ignore_files
    dry_run      = args.dry_run
    verbose      = args.verbose

    if src_file:
        with open(src_file) as f:
            for l in f.readlines():
                l = l.strip()
                l = l.split('#')[0]
                if len(l) == 0:
                    continue
                if not os.path.isdir(l):
                    print(f"[WARNING]: Not a valid src directory, skipping -- {l}")
                    continue
                src_dirs.append(l)

    if len(src_dirs) == 0:
        print("[ERROR] No src directories have been specified!")
        parser.print_help()
        return

    if out_dir and not os.path.exists(out_dir):
        print(f"[ERROR] output directory does not exist: {out_dir}")
        return

    # These sub-directories have duplicated JSON names with those from private_UL
    ignore_dirs.extend(['subsets_of_private_UL_.*','private_UL_backup'])
    # Make sure to always only find .json files
    match_files.extend(['.*\\.json'])
    # These are not sample json files, so skip them
    ignore_files.extend(['lumi.json','params.json'])
    # These are json files for already produced skims, so skip them as well
    ignore_files.extend([".*_atPSI\\.json",".*_NDSkim\\.json"])
    template_json_fpaths = get_files(json_dir,
        ignore_dirs  = ignore_dirs,
        match_files  = match_files,
        ignore_files = ignore_files,
        recursive = True,
        verbose = verbose
    )
    missing_templates = []
    hadoop_dataset_dirs = []
    for src_dir in src_dirs:
        for d in os.listdir(src_dir):
            dir_fpath = pjoin(src_dir,d)
            if not os.path.isdir(dir_fpath): continue
            hadoop_dataset_dirs.append(dir_fpath)
    print("Skim directories:")
    for hdir in hadoop_dataset_dirs:
        print(f"\t{hdir}")
    print("Attempting to find matching json templates...")
    for hdir in hadoop_dataset_dirs:
        dataset = os.path.split(hdir)[1]
        matched_json_fp = find_json_match(hdir,template_json_fpaths)
        print(f"\tMatch: {matched_json_fp}")
        if not matched_json_fp:
            missing_templates.append(hdir)
            continue
        template_json_dir = os.path.split(matched_json_fp)[0]
        updates = {
            "files": [x.replace("/hadoop","") for x in get_files(hdir,match_files=[".*\\.root"])]
        }
        outname = os.path.split(matched_json_fp)[1].replace(".json",f"{postfix}.json")
        if out_dir:
            outname = pjoin(out_dir,outname)
        else:
            outname = pjoin(template_json_dir,outname)
        update_json(matched_json_fp,dry_run=dry_run,outname=outname,verbose=verbose,**updates)
        template_json_fpaths.remove(matched_json_fp)
    # These are lobster skims for which we couldn't find a matching json template
    if missing_templates:
        print(f"Skims with no matching json template found:")
        for x in missing_templates:
            print(f"\t{x}")
    else:
        print(f"Skims with no matching json template found: {missing_templates}")
    # These are json templates for which we couldn't find a lobster skim
    if template_json_fpaths:
        print(f"Json templates with no matching skim found:")
        for x in template_json_fpaths:
            print(f"\t{x}")
    else:
        print(f"Json templates with no matching skim found: {template_json_fpaths}")


if __name__ == "__main__":
    main()

