import os
import time
import json
import shutil
import argparse
import gzip
import pickle
import numpy as np

from topcoffea.modules.utils import regex_match,clean_dir,dict_comp
from topeft.modules.datacard_tools import *
from topeft.modules.histogram_artifact import write_histogram_artifact

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
arguments  = "{usr_dir} {pkl_list} {out_dir} {var_lst} '{ch_lst}' '{other}'"
output = {condor_dir}/job_{idx}.out
error  = {condor_dir}/job_{idx}.err
log    = {condor_dir}/job_{idx}.log

request_cpus = 1
request_memory = 20000
request_disk = 4096

transfer_input_files = {transfer_inputs}
should_transfer_files = yes
transfer_executable = true

getenv = true
queue 1
"""

sh_fragment = r"""#!/bin/sh
USR_DIR=${1}
PKL_LIST_FILE=${2}
OUT_DIR=${3}
VAR_LST=${4}
CH_LST=${5}
OTHER=${6}

echo "USR_DIR: ${USR_DIR}"
echo "PKL_LIST_FILE: ${PKL_LIST_FILE}"
echo "OUT_DIR: ${OUT_DIR}"
echo "VAR_LST: ${VAR_LST}"
echo "CH_LST: ${CH_LST}"
echo "OTHER: ${OTHER}"

export CONDA_DIR="$(conda info --base)"
echo "CONDA_DIR: ${CONDA_DIR}"
source ${CONDA_DIR}/etc/profile.d/conda.sh
unset PYTHONPATH
conda activate ${CONDA_DEFAULT_ENV}

ulimit -s unlimited
python make_cards.py --pkl-list-file "${PKL_LIST_FILE}" -d ${OUT_DIR} --var-lst ${VAR_LST} --ch-lst ${CH_LST} --use-selected "selectedWCs.txt" ${OTHER}
"""

def build_arg_parser():
    parser = argparse.ArgumentParser(description="You can select which file to run over")
    parser.add_argument("pkl_file",nargs="*",help="One or more pickle files with histograms to run over")
    parser.add_argument("--pkl-list-file",default="",help="Optional text file with one pkl path per line")
    parser.add_argument("--rate-syst-json","-s",default="params/rate_systs.json",help="Rate related systematics json file, path relative to topeft_path()")
    parser.add_argument(
        "--miss-parton-file",
        "-m",
        default=None,
        help=(
            "Optional missing-parton payload path relative to topeft_path(); when "
            "omitted, select missing_parton_run2.root or missing_parton_run3.root "
            "from the resolved card era."
        ),
    )
    from topeft.modules.missing_parton_contract import SUPPORTED_SR_REGISTRIES, DEFAULT_SR_REGISTRY
    parser.add_argument("--sr-registry", choices=SUPPORTED_SR_REGISTRIES, default=DEFAULT_SR_REGISTRY,
                        help=("SR registry associated with missing-parton payload selection "
                              f"(default: {DEFAULT_SR_REGISTRY})."))
    parser.add_argument("--skip-missing-parton-rate-syst",action="store_true",default=False,help="Skip loading/inserting only the missing-parton rate systematic; preserves other nuisances.")
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
    parser.add_argument("--use-AAC","-A",action="store_true",help="Include all EFT templates in datacards for AAC model")
    parser.add_argument("--wc-vals", default="",action="store", nargs="+", help="Specify the corresponding wc values to set for the wc list")
    parser.add_argument("--wc-scalings", default=[],action="extend",nargs="+",help="Specify a list of wc ordering for scalings.json")
    parser.add_argument(
        "--on-process-collision",
        choices=["error","warn","allow"],
        default="error",
        help=(
            "Policy for process-label overlaps when merging multiple input pkl files. "
            "Default is strict `error`. Expert-only escape hatches: `warn`/`allow`, "
            "to be used only when overlaps are intentional (e.g. chunked outputs)."
        ),
    )
    parser.add_argument("--merge-report",default="-",help="Path for merge diagnostic report JSON, or '-' for stdout")
    parser.add_argument("--merge-only",action="store_true",help="Only load+merge+validate input histograms and exit")
    parser.add_argument("--cache-merged-pkl",default="",help="Optional output path for merged histogram dictionary (.pkl.gz)")
    return parser


def _resolve_pkl_paths(args, parser):
    pkl_files = list(args.pkl_file)
    if args.pkl_list_file:
        if pkl_files:
            parser.error("Specify either positional pkl files or --pkl-list-file, not both.")
        with open(args.pkl_list_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pkl_files.append(line)
    if not pkl_files:
        parser.error("No input pkl files were provided.")
    return pkl_files


def _emit_merge_report(report_obj, report_path, out_dir):
    if report_path == "-":
        print("Merge report:")
        print(json.dumps(report_obj, indent=2, sort_keys=True))
        return

    report_fpath = report_path
    if not os.path.isabs(report_fpath):
        report_fpath = os.path.join(out_dir, report_fpath)
    report_parent = os.path.dirname(report_fpath)
    if report_parent:
        os.makedirs(report_parent, exist_ok=True)
    with open(report_fpath, "w") as f:
        json.dump(report_obj, f, indent=2, sort_keys=True)
    print(f"Wrote merge report: {report_fpath}")


def _cache_merged_histograms(merged_hists, cache_path, out_dir, merge_report=None):
    out_fpath = cache_path
    if not os.path.isabs(out_fpath):
        out_fpath = os.path.join(out_dir, out_fpath)
    if not out_fpath.endswith(".pkl.gz"):
        out_fpath += ".pkl.gz"
    out_parent = os.path.dirname(out_fpath)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    print(f"Caching merged histograms to {out_fpath}")
    if merge_report and merge_report.get("schema") == "split_sibling_v1":
        write_histogram_artifact(
            out_fpath,
            histograms=merged_hists,
            artifact_kind=merge_report["artifact_kind"],
            sumw2_storage_provenance=merge_report["sumw2_storage_provenance"],
            production_sample_contract=merge_report[
                "production_sample_contract"
            ],
            merged=True,
            lineage_inputs=merge_report["lineage_inputs"],
            required_sumw2_processes=merge_report["required_sumw2_processes"],
            transformation_contract=merge_report["transformation_contract"],
            requested_data_driven_products=merge_report[
                "requested_data_driven_products"
            ],
            resolved_data_driven_contract=merge_report[
                "resolved_data_driven_contract"
            ],
        )
    else:
        with gzip.open(out_fpath, "wb") as fout:
            pickle.dump(merged_hists, fout, protocol=pickle.HIGHEST_PROTOCOL)
    return out_fpath

def run_local(dc,km_dists,channels,selected_wcs, crop_negative_bins, wcs_dict):
    for km_dist in km_dists:
        all_chs = dc.channels(km_dist)
        matched_chs = regex_match(all_chs,channels)
        if channels:
            print(f"Channels to process: {matched_chs}")
        for ch in matched_chs:
            r = dc.analyze(km_dist,ch,selected_wcs, crop_negative_bins, wcs_dict)

def _build_condor_base_other_opts(dc,on_process_collision):
    base_other_opts = []
    if dc.do_mc_stat:
        base_other_opts.append("--do-mc-stat")
    if dc.verbose:
        base_other_opts.append("--verbose")
    if dc.use_real_data:
        base_other_opts.append("--unblind")
    if dc.do_nuisance:
        base_other_opts.append("--do-nuisance")
    if dc.year_lst:
        base_other_opts.extend(["--year"," ".join(dc.year_lst)])
    if dc.drop_syst:
        base_other_opts.extend(["--drop-syst"," ".join(dc.drop_syst)])
    missing_parton_payload_path = getattr(dc, "missing_parton_payload_path", None)
    if missing_parton_payload_path is not None:
        base_other_opts.extend(["--miss-parton-file", missing_parton_payload_path])
    base_other_opts.extend(["--sr-registry", dc.sr_registry])
    if getattr(dc, "skip_missing_parton_rate_syst", False):
        base_other_opts.append("--skip-missing-parton-rate-syst")
    base_other_opts.extend(["--on-process-collision",on_process_collision])
    return base_other_opts

# VERY IMPORTANT:
#   This setup assumes the output directory is mounted on the remote condor machines
# Note:
#   The condor jobs currently have to read the various .json files from the default locations, which
#   means that they will probably be getting read from the user's AFS area (or wherever their TopEFT
#   repo is located).
# TODO: Currently there's no way to transparently passthrough parent arguments to the condor ones.
#   There's also no clear way to pass customized options to different sub-sets of condor jobs
def run_condor(dc,pkl_paths,out_dir,var_lst,ch_lst,chunk_size,on_process_collision="error",merge_report="-"):
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

    print("Generating condor executable script")
    with open(condor_exe_fname,"w") as f:
        f.write(sh_fragment)

    pkl_list_basename = "pkl_inputs.txt"
    pkl_list_fpath = os.path.join(out_dir,pkl_list_basename)
    print(f"Writing pkl list file for condor: {pkl_list_fpath}")
    with open(pkl_list_fpath,"w") as f:
        for p in pkl_paths:
            f.write(f"{p}\n")

    usr_perms = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
    grp_perms = stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP
    all_perms = stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH

    os.chmod(condor_exe_fname,usr_perms | grp_perms | all_perms)    # equiv. to 777

    base_other_opts = _build_condor_base_other_opts(dc,on_process_collision)

    idx = 0
    for km_dist in var_lst:
        all_chs = dc.channels(km_dist)
        matched_chs = regex_match(all_chs,ch_lst)
        n = max(chunk_size,1)
        chunks = np.split(matched_chs,[i for i in range(n,len(matched_chs),n)])
        for chnk in chunks:
            print(f"[{idx+1:0>3}] Variable: {km_dist} -- Channels: {chnk}")
            other_opts = list(base_other_opts)
            if merge_report == "-":
                other_opts.extend(["--merge-report","-"])
            else:
                report_path = merge_report
                if report_path.endswith(".json"):
                    report_path = report_path.replace(".json",f".job_{idx}.json")
                else:
                    report_path = f"{report_path}.job_{idx}.json"
                other_opts.extend(["--merge-report",report_path])
            other_opts = " ".join(other_opts)

            s = sub_fragment.format(
                idx=idx,
                usr_dir=os.path.expanduser("~"),
                pkl_list=pkl_list_basename,
                out_dir=os.path.realpath(out_dir),
                var_lst=km_dist,
                ch_lst=" ".join(chnk),
                condor_dir=condor_dir,
                other=f"{other_opts}",
                transfer_inputs=f"make_cards.py,selectedWCs.txt,{pkl_list_basename}",
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
    parser = build_arg_parser()
    args = parser.parse_args()
    pkl_files  = _resolve_pkl_paths(args, parser)
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
    skip_missing_parton_rate_syst = args.skip_missing_parton_rate_syst
    sr_registry = args.sr_registry
    unblind    = args.unblind
    verbose    = args.verbose
    use_AAC     = args.use_AAC
    wc_vals    = args.wc_vals

    wc_scalings = args.wc_scalings
    select_only = args.select_only
    use_selected = args.use_selected

    use_condor = args.condor
    chunks = int(args.chunks)

    if isinstance(wcs,str):
        wcs = wcs.split(",")

    kwargs = {
        "wcs": wcs,
        "rate_syst_path": rs_json,
        "missing_parton_path": mp_file,
        "sr_registry": sr_registry,
        "out_dir": out_dir,
        "var_lst": var_lst,
        "do_mc_stat": do_mc_stat,
        "ignore": ignore,
        "do_nuisance": do_nuis,
        "drop_syst": drop_syst,
        "skip_missing_parton_rate_syst": skip_missing_parton_rate_syst,
        "unblind": unblind,
        "verbose": verbose,
        "year_lst": years,
        "use_AAC":  use_AAC,
        "wc_vals": wc_vals,
        "wc_scalings": wc_scalings,
    }

    if out_dir != "." and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Copy over make_cards.py ASAP so a user can't accidentally modify it before the submit jobs run
    if use_condor and not os.path.samefile(os.getcwd(),out_dir):
        shutil.copy("make_cards.py",out_dir)

    if args.condor and args.merge_only:
        parser.error("--merge-only and --condor cannot be used together.")

    merged_hists, merge_report = load_and_merge_histogram_pkls(
        pkl_files,
        on_process_collision=args.on_process_collision,
        require_sumw2=True,
    )
    _emit_merge_report(merge_report, args.merge_report, out_dir)
    if args.cache_merged_pkl:
        _cache_merged_histograms(
            merged_hists, args.cache_merged_pkl, out_dir, merge_report
        )
    if args.merge_only:
        print("Merge-only mode enabled, stopping after successful merge validation.")
        return

    tic = time.time()
    dc = DatacardMaker(hists=merged_hists,**kwargs)

    # convert wc_vals string to a dictionary
    wc_vals = ''.join(wc_vals)
    wcs_dict = eval("dict({})".format(wc_vals))

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
        with open(os.path.join(out_dir,"selectedWCs.txt"),"w") as f:
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
        run_condor(
            dc,
            pkl_files,
            out_dir,
            dists,
            ch_lst,
            chunks,
            on_process_collision=args.on_process_collision,
            merge_report=args.merge_report,
        )
    else:
        run_local(dc,dists,ch_lst,selected_wcs, not args.keep_negative_bins, wcs_dict)

    # make pre-selection scalings.json
    print("Making scalings-preselect.json file...")
    with open(os.path.join(out_dir,"scalings-preselect.json"),"w") as f:
        json.dump(dc.scalings, f, indent=4)

    dt = time.time() - tic
    print(f"Total Time: {dt:.2f} s")
    print("Finished!")

if __name__ == "__main__":
    main()
