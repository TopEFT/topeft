import os
import json
import argparse

from topcoffea.modules.datacard_tools import *
from topcoffea.modules.utils import regex_match

def main():
    parser = argparse.ArgumentParser(description="You can select which file to run over")
    parser.add_argument("pkl_file",nargs="?",help="Pickle file with histograms to run over")
    parser.add_argument("--lumi-json","-l",default="json/lumi.json",help="Lumi json file, path relative to topcoffea_path()")
    parser.add_argument("--rate-syst-json","-s",default="json/rate_systs.json",help="Rate related systematics json file, path relative to topcoffea_path()")
    parser.add_argument("--miss-parton-file","-m",default="data/missing_parton/missing_parton.root",help="File for missing parton systematic, path relative to topcoffea_path()")
    parser.add_argument("--out-dir","-d",default=".",help="Output directory to write root and text datacard files to")
    parser.add_argument("--var-lst",default=[],action="extend",nargs="+",help="Specify a list of variables to make cards for.")
    parser.add_argument("--ch-lst","-c",default=[],action="extend",nargs="+",help="Specify a list of channels to process.")
    parser.add_argument("--do-mc-stat",action="store_true",help="Add bin-by-bin statistical uncertainties with the autoMCstats option (for background)")
    parser.add_argument("--ignore","-i",default=[],action="extend",nargs="+",help="Specify a list of processes to exclude, must match name from 'sample' axis modulo UL year")
    parser.add_argument("--POI",default=[],help="List of WCs (comma separated)")
    parser.add_argument("--year","-y",default="",help="Run over single year")
    parser.add_argument("--do-nuisance",action="store_true",help="Include nuisance parameters")
    parser.add_argument("--unblind",action="store_true",help="If set, use real data, otherwise use asimov data")
    parser.add_argument("--verbose","-v",action="store_true",help="Set to verbose output")
    parser.add_argument("--select-only",action="store_true",help="Only run the WC selection step")
    parser.add_argument("--use-selected",default="",help="Load selected process+WC combs from a file. Skips doing the normal selection step.")

    args = parser.parse_args()
    pkl_file   = args.pkl_file
    lumi_json  = args.lumi_json
    rs_json    = args.rate_syst_json
    mp_file    = args.miss_parton_file
    out_dir    = args.out_dir
    # year      = args.year     # NOT IMPLEMENTED YET
    var_lst    = args.var_lst
    ch_lst     = args.ch_lst
    do_mc_stat = args.do_mc_stat
    wcs        = args.POI
    ignore     = args.ignore
    do_nuis    = args.do_nuisance
    unblind    = args.unblind
    verbose    = args.verbose

    select_only = args.select_only
    use_selected = args.use_selected

    if isinstance(wcs,str):
        wcs = wcs.split(",")

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
        "unblind": unblind,
        "verbose": verbose,
    }

    if out_dir != "." and not os.path.exists(out_dir):
        os.makedirs(out_dir)

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

    if select_only:
        return

    for km_dist in dists:
        all_chs = dc.channels(km_dist)
        matched_chs = regex_match(all_chs,ch_lst)
        if ch_lst:
            print(f"Channels to process: {matched_chs}")
        for ch in matched_chs:
            r = dc.analyze(km_dist,ch,selected_wcs)
    dt = time.time() - tic
    print(f"Total Time: {dt:.2f} s")
    print("Finished!")

main()
