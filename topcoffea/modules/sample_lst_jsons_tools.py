# This file is essentially a wrapper for createJSON.py:
#   - It runs createJSON.py for each sample that you include in a dictionary, and moves the resulting json file to the directory you specify

import json
import subprocess
import os
from topcoffea.modules.samples import loadxsecdic
from topcoffea.modules.combine_json_ext import combine_json_ext
from topcoffea.modules.combine_json_batch import combine_json_batch
import re

########### The XSs from xsec.cfg ###########
XSECDIC = loadxsecdic("../../topcoffea/cfg/xsec.cfg",True)

########### Functions for makign the jsons ###########

# Replace a value in one of the JSONs
def replace_val_in_json(path_to_json_file,key,new_val,verbose=True):

    # Replace value if it's different than what's in the JSON
    with open(path_to_json_file) as json_file:
        json_dict = json.load(json_file)
    if new_val == json_dict[key]:
        if verbose:
            print(f"\tValues already agree, both are {new_val}")
    else:
        if verbose:
            print(f"\tOld value for {key}: {json_dict[key]}")
            print(f"\tNew value for {key}: {new_val}")
        json_dict[key] = new_val

        # Save new json
        with open(path_to_json_file, "w") as out_file:
            json.dump(json_dict, out_file, indent=2)


# Loop through a dictionary of samples and replace the xsec in the JSON with what's in xsec.cfg
def replace_xsec_for_dict_of_samples(samples_dict,out_dir):
    for sample_name,sample_info in samples_dict.items():
        path_to_json = os.path.join(out_dir,sample_name+".json")
        if not os.path.exists(path_to_json):
            print(f"\nWARNING: This json does not exist, continuing ({path_to_json})")
            continue
        xsecName = sample_info["xsecName"]
        new_xsec = XSECDIC[xsecName]
        print(f"\nReplacing XSEC for {sample_name} JSON with the value from xsec.cfg for \"{xsecName}\":")
        print("\tPath to json:",path_to_json)
        replace_val_in_json(path_to_json,"xsec",new_xsec)

# Wrapper for createJSON.py
def make_json(sample_dir,sample_name,prefix,sample_yr,xsec_name,hist_axis_name,on_das=False):

    # If the sample is on DAS, inclue the DAS flag in the createJSON.py arguments
    das_flag = ""
    if on_das: das_flag = "--DAS"

    args = [
        "python",
        "../../topcoffea/modules/createJSON.py",
        sample_dir,
        das_flag,
        "--sampleName"   , sample_name,
        "--prefix"       , prefix,
        "--xsec"         , "../../topcoffea/cfg/xsec.cfg",
        "--year"         , sample_yr,
        "--histAxisName" , hist_axis_name,
    ]

    if xsec_name:
        args.extend(['--xsecName',xsec_name])

    # Run createJSON.py
    subprocess.run(args)

# Convenience function for running make_json() on all entries in a dictionary of samples, and moving the results to out_dir
def make_jsons_for_dict_of_samples(samples_dict,prefix,year,out_dir,on_das=False):
    failed = []
    for sample_name,sample_info in sorted(samples_dict.items()):
        print(f"\n\nMaking JSON for {sample_name}...")
        path = sample_info["path"]
        if not on_das and "path_local" in sample_info:
            # The bkg samples are now at ND, but we wanted to leave the dataset names in the dictionaries as well (in case we want to access remotely)
            # So for these samples we have a "path" (i.e. dataset name to be used when on_das=True), as well as a "local_path" for acessing locally
            # Note, it would probably make more sense to call "path" something like "path_das" (and "path_local" just "path"), but did not want to change the existing names..
            path = sample_info["path_local"]
        hist_axis_name = sample_info["histAxisName"]
        xsec_name = sample_info["xsecName"]
        make_json(
            sample_dir = path,
            sample_name = sample_name,
            prefix = prefix,
            sample_yr = year,
            xsec_name = xsec_name,
            hist_axis_name = hist_axis_name,
            on_das = on_das,
        )
        out_name = sample_name+".json"
        if not os.path.exists(out_name):
            failed.append(sample_name)

        subprocess.run(["mv",out_name,out_dir])
        if '_ext' in out_name:
            combine_json_ext(out_dir+'/'+out_name) # Merge with non-ext version
            os.remove(out_dir+'/'+out_name) # Remove (now) outdated ext version
        # Only run if more than one file exists (differentiates between `*_b2.json` and `*_b2_atPSI.json`
        r = re.compile(re.sub(r'_b[1-9]', '_b[1-9]', out_name))
        matches = [b for b in str(subprocess.check_output(["ls",'.'], shell=True)).split('\\n') if bool(r.match(b))]
        if re.search('_b[2-9]', out_name) and len(matches)>1:
            combine_json_batch(out_dir+'/'+out_name) # Merge batches
            os.remove(out_dir+'/'+out_name) # Remove (now) outdated batch version

        print("sample name:",sample_name)
        print("\tpath:",path,"\n\thistAxisName:",hist_axis_name,"\n\txsecName",xsec_name,"\n\tout name:",out_name,"\n\tout dir:",out_dir)
    if len(failed):
        print("Failed:")
        for l in failed:
            print(f"\t{l}")
    else:
        print("Failed: None")
