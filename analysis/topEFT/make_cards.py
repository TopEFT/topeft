import os
import time
import json
import shutil
import argparse
import numpy as np

from topcoffea.modules.datacard_tools import *
from topcoffea.modules.utils import regex_match,clean_dir,dict_comp

# Note:
#   Not sure if constructing the condor related files this way is good or bad practice. It already
#   feels clunky in a number of places with having to manually hardcode what options are used by the
#   condor jobs.
# Note 2:
#   The single quotes on some of the arguments are to ensure the submit file passes the string of
#   (potentially multiple) options as a single argument to the executable script, i.e. we don't want
#   it to split the string up by spaces, but still have the executable see it as a space spearated
#   list of arguments
sub_fragment = """\
universe   = vanilla
executable = condor.sh
arguments  = "{usr_dir} {inf} {out_dir} {var_lst} '{ch_lst}' '{other}'"
output = {condor_dir}/job_{idx}.out
error  = {condor_dir}/job_{idx}.err
log    = {condor_dir}/job_{idx}.log

request_cpus = 1
request_memory = 4096
request_disk = 1024

transfer_input_files = make_cards.py,selectedWCs.txt
should_transfer_files = yes
transfer_executable = true

getenv = true
queue 1
"""

sh_fragment = r"""#!/bin/sh
USR_DIR=${1}
INF=${2}
OUT_DIR=${3}
VAR_LST=${4}
CH_LST=${5}
OTHER=${6}

echo "USR_DIR: ${USR_DIR}"
echo "INF: ${INF}"
echo "OUT_DIR: ${OUT_DIR}"
echo "VAR_LST: ${VAR_LST}"
echo "CH_LST: ${CH_LST}"
echo "OTHER: ${OTHER}"

source ${USR_DIR}/miniconda3/etc/profile.d/conda.sh
unset PYTHONPATH
conda activate ${CONDA_DEFAULT_ENV}

python make_cards.py ${INF} -d ${OUT_DIR} --var-lst ${VAR_LST} --ch-lst ${CH_LST} --use-selected "selectedWCs.txt" --do-nuisance ${OTHER}
"""

def run_local(dc,km_dists,channels,selected_wcs, crop_negative_bins):
    for km_dist in km_dists:
        all_chs = dc.channels(km_dist)
        matched_chs = regex_match(all_chs,channels)
        if channels:
            print(f"Channels to process: {matched_chs}")
        for ch in matched_chs:
            r = dc.analyze(km_dist,ch,selected_wcs, crop_negative_bins)

# VERY IMPORTANT:
#   This setup assumes the output directory is mounted on the remote condor machines
# Note:
#   The condor jobs currently have to read the various .json files from the default locations, which
#   means that they will probably be getting read from the user's AFS area (or wherever their TopEFT
#   repo is located).
# TODO: Currently there's no way to transparently passthrough parent arguments to the condor ones.
#   There's also no clear way to pass customized options to different sub-sets of condor jobs
def run_condor(dc,pkl_fpath,out_dir,var_lst,ch_lst,chunk_size):
    import subprocess
    import stat

    home = os.getcwd()

    condor_dir = os.path.join(out_dir,"job_logs")

    if not os.path.exists(condor_dir):
        print(f"Making condor output directory {condor_dir}")
        os.mkdir(condor_dir)

    clean_dir(condor_dir,targets=["job_.*log","job_.*err","job_.*out","^condor.*sub$"])

    condor_exe_fname = "condor.sh"
    if not os.path.samefile(home,out_dir):
        condor_exe_fname = os.path.join(out_dir,"condor.sh")

    print(f"Generating condor executable script")
    with open(condor_exe_fname,"w") as f:
        f.write(sh_fragment)

    usr_perms = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
    grp_perms = stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP
    all_perms = stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH

    os.chmod(condor_exe_fname,usr_perms | grp_perms | all_perms)    # equiv. to 777

    other_opts = []
    if dc.do_mc_stat:
        other_opts.append("--do-mc-stat")
    if dc.verbose:
        other_opts.append("--verbose")
    if dc.use_real_data:
        other_opts.append("--unblind")
    if dc.year_lst:
        other_opts.extend(["--year"," ".join(dc.year_lst)])
    other_opts = " ".join(other_opts)

    idx = 0
    for km_dist in var_lst:
        all_chs = dc.channels(km_dist)
        matched_chs = regex_match(all_chs,ch_lst)
        n = max(chunk_size,1)
        chunks = np.split(matched_chs,[i for i in range(n,len(matched_chs),n)])
        for chnk in chunks:
            print(f"[{idx+1:0>3}] Variable: {km_dist} -- Channels: {chnk}")
            s = sub_fragment.format(
                idx=idx,
                usr_dir=os.path.expanduser("~"),
                inf=pkl_fpath,
                out_dir=os.path.realpath(out_dir),
                var_lst=km_dist,
                ch_lst=" ".join(chnk),
                condor_dir=condor_dir,
                other=f"{other_opts}",
            )
            condor_submit_fname = os.path.join(condor_dir,f"condor.{idx}.sub")
            with open(condor_submit_fname,"w") as f:
                f.write(s)
            cmd = ["condor_submit",condor_submit_fname]
            print(f"{'':>5} Condor command: {' '.join(cmd)}")
            os.chdir(out_dir)
            p = subprocess.run(cmd)
            os.chdir(home)
            idx += 1

def main():
    parser = argparse.ArgumentParser(description="You can select which file to run over")
    parser.add_argument("pkl_file",nargs="?",help="Pickle file with histograms to run over")
    parser.add_argument("--lumi-json","-l",default="json/lumi.json",help="Lumi json file, path relative to topcoffea_path()")
    parser.add_argument("--rate-syst-json","-s",default="json/rate_systs.json",help="Rate related systematics json file, path relative to topcoffea_path()")
    parser.add_argument("--miss-parton-file","-m",default="data/missing_parton/missing_parton.root",help="File for missing parton systematic, path relative to topcoffea_path()")
    parser.add_argument("--selected-wcs-ref",default="test/selectedWCs.json",help="Reference file for selected wcs")
    parser.add_argument("--out-dir","-d",default=".",help="Output directory to write root and text datacard files to")
    parser.add_argument("--var-lst",default=[],action="extend",nargs="+",help="Specify a list of variables to make cards for.")
    parser.add_argument("--ch-lst","-c",default=[],action="extend",nargs="+",help="Specify a list of channels to process.")
    parser.add_argument("--do-mc-stat",action="store_true",help="Add bin-by-bin statistical uncertainties with the autoMCstats option (for background)")
    parser.add_argument("--ignore","-i",default=[],action="extend",nargs="+",help="Specify a list of processes to exclude, must match name from 'sample' axis modulo UL year")
    parser.add_argument("--drop-syst",default=[],action="extend",nargs="+",help="Specify one or more template systematics to remove from the datacard")
    parser.add_argument("--POI",default=[],help="List of WCs (comma separated)")
    parser.add_argument("--year","-y",default=[],action="extend",nargs="+",help="Run over a subset of years")
    parser.add_argument("--do-nuisance",action="store_true",help="Include nuisance parameters")
    parser.add_argument("--unblind",action="store_true",help="If set, use real data, otherwise use asimov data")
    parser.add_argument("--verbose","-v",action="store_true",help="Set to verbose output")
    parser.add_argument("--select-only",action="store_true",help="Only run the WC selection step")
    parser.add_argument("--skip-selected-wcs-check",action="store_true",help="Do not raise an error if the selected WCs disagree with ref")
    parser.add_argument("--use-selected",default="",help="Load selected process+WC combs from a file. Skips doing the normal selection step.")
    parser.add_argument("--condor","-C",action="store_true",help="Split up the channels into multiple condor jobs")
    parser.add_argument("--chunks","-n",default=1,help="The number of channels each condor job should process")
    parser.add_argument("--keep-negative-bins",action="store_true",help="Don't crop negative bins")

    args = parser.parse_args()
    pkl_file   = args.pkl_file
    lumi_json  = args.lumi_json
    rs_json    = args.rate_syst_json
    mp_file    = args.miss_parton_file
    out_dir    = args.out_dir
    years      = args.year
    var_lst    = args.var_lst
    ch_lst     = args.ch_lst
    do_mc_stat = args.do_mc_stat
    wcs        = args.POI
    ignore     = args.ignore
    do_nuis    = args.do_nuisance
    drop_syst  = args.drop_syst
    unblind    = args.unblind
    verbose    = args.verbose


    select_only = args.select_only
    use_selected = args.use_selected

    use_condor = args.condor
    chunks = int(args.chunks)

    if isinstance(wcs,str):
        wcs = wcs.split(",")

    if use_condor:
        # Note:
        #   The dc in the parent submission is only used to generate the selectedWCs file and figure
        #   out what channels/samples are available, so we can just drop the systematics to speed
        #   things up
        do_nuis = False

    kwargs = {
        "wcs": wcs,
        "lumi_json_path": lumi_json,
        "rate_syst_path": rs_json,
        "missing_parton_path": mp_file,
        "out_dir": out_dir,
        "var_lst": var_lst,
        "do_mc_stat": do_mc_stat,
        "ignore": ignore,
        "do_nuisance": do_nuis,
        "drop_syst": drop_syst,
        "unblind": unblind,
        "verbose": verbose,
        "year_lst": years,
    }

    if out_dir != "." and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Copy over make_cards.py ASAP so a user can't accidentally modify it before the submit jobs run
    if use_condor and not os.path.samefile(os.getcwd(),out_dir):
        shutil.copy("make_cards.py",out_dir)

    tic = time.time()
    dc = DatacardMaker(pkl_file,**kwargs)

    dists = var_lst if len(var_lst) else dc.hists.keys()
    if use_selected:
        # Use a pre-generated selectionWCs.txt file
        with open(use_selected) as f:
            selected_wcs = json.load(f)
        # This is needed since when we load WCs from a file, the background processes aren't included
        for km_dist in dists:
            all_procs = dc.processes(km_dist)
            for p in all_procs:
                if not p in selected_wcs:
                    selected_wcs[p] = []
        print(f"Loading WCs from {use_selected}")
        for p,wcs in selected_wcs.items():
            print(f"\t{p}: {wcs}")
    else:
        # Generate the selectedWCs file based on ch-lst and var-lst
        selected_wcs = {}
        for km_dist in dists:
            all_chs = dc.channels(km_dist)
            matched_chs = regex_match(all_chs,ch_lst)
            if select_only and ch_lst:
                print(f"Channels to process: {matched_chs}")
            dist_wcs = dc.get_selected_wcs(km_dist,matched_chs)
            # TODO: This could be made a lot more elegant, but for now is a quick and dirty way of making it work
            for p,wcs in dist_wcs.items():
                if not p in selected_wcs:
                    selected_wcs[p] = []
                for wc in wcs:
                    if not wc in selected_wcs[p]:
                        selected_wcs[p].append(wc)
        with open(os.path.join(out_dir,f"selectedWCs.txt"),"w") as f:
            selected_wcs_for_json = {}
            for p,v in selected_wcs.items():
                if not dc.is_signal(p):
                    # WC selection will include backgrounds in the dict (always with an empty list), so remove them here
                    continue
                selected_wcs_for_json[p] = list(v)
            json.dump(selected_wcs_for_json,f)

    # Check selected WCs against what's currently the list being assumed by the physcis model
    # Right now we're set to raise an exception if these files differ (warnings are easy to miss, and we really want the user to notice)
    # If you know what you're doing and expet them to differ, then just bypass this
    if not args.skip_selected_wcs_check and not use_selected:
        with open(args.selected_wcs_ref,"r") as selected_wcs_ref_f:
            selected_wcs_ref_data = selected_wcs_ref_f.read()
        selected_wcs_ref = json.loads(selected_wcs_ref_data)
        wcs_agree = dict_comp(selected_wcs_ref,selected_wcs_for_json)
        if not wcs_agree:
            raise Exception(f"The selected WCs do not agree. Please check if this is expected.\n\tRef:{selected_wcs_ref}\n\tNew:{selected_wcs_for_json}")

    if select_only:
        return

    if use_condor:
        run_condor(dc,pkl_file,out_dir,dists,ch_lst,chunks)
    else:
        run_local(dc,dists,ch_lst,selected_wcs, not args.keep_negative_bins)
    dt = time.time() - tic
    print(f"Total Time: {dt:.2f} s")
    print("Finished!")

main()
