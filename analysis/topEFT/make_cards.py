import argparse

from topcoffea.modules.datacard_tools import *

def main():
    parser = argparse.ArgumentParser(description="You can select which file to run over")
    parser.add_argument("pkl_file",nargs="?",help="Pickle ifle with histograms to run over")
    parser.add_argument("--lumi-json","-l",default="json/lumi.json",help="Lumi json file, path relative to topcoffea_path()")
    parser.add_argument("--rate-syst-json","-s",default="json/rate_systs.json",help="Rate related systematics json file, path relative to topcoffea_path()")
    parser.add_argument("--miss-parton-file","-m",default="data/missing_parton/missing_parton.root",help="File for missing parton systematic, path relative to topcoffea_path()")
    parser.add_argument("--var-lst",default=[],action="extend",nargs="+",help="Specify a list of variables to make cards for.")
    parser.add_argument("--POI",default=[],help="List of WCs (comma separated)")
    parser.add_argument("--year","-y",default="",help="Run over single year")
    parser.add_argument("--do-nuisance",action="store_true",help="Include nuisance parameters")
    parser.add_argument("--unblind",action="store_true",help="If set, use real data, otherwise use asimov data")
    parser.add_argument("--verbose","-v",action="store_true",help="Set to verbose output")

    args = parser.parse_args()
    pkl_file  = args.pkl_file
    lumi_json = args.lumi_json
    rs_json   = args.rate_syst_json
    mp_file   = args.miss_parton_file
    # year      = args.year     # NOT IMPLEMENTED YET
    var_lst   = args.var_lst
    wcs       = args.POI
    do_nuis   = args.do_nuisance
    unblind   = args.unblind
    verbose   = args.verbose

    if isinstance(wcs,str):
        wcs = wcs.split(",")

    kwargs = {
        "wcs": wcs,
        "lumi_json_path": lumi_json,
        "rate_syst_path": rs_json,
        "missing_parton_path": mp_file,
        "do_nuisance": do_nuis,
        "unblind": unblind,
        "verbose": verbose,
    }

    tic = time.time()
    dc = DatacardMaker(pkl_file,**kwargs)

    dists = var_lst if len(var_lst) else dc.hists.keys()
    for km_dist in dists:
        selected_wcs = dc.get_selected_wcs(km_dist)
        for ch in dc.channels(km_dist):
            r = dc.analyze(km_dist,ch,selected_wcs)
    dt = time.time() - tic
    print(f"Total Time: {dt:.2f} s")
    print("Finished!")

main()