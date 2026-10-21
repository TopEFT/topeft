import os
import time
import json
import argparse
import sys
import gzip
import pickle
import numpy as np

from topcoffea.modules.utils import regex_match,dict_comp
import topeft.modules.datacard_tools as datacard_tools
from topeft.modules.histogram_artifact import write_histogram_artifact
from topeft.modules.axis_binning import BINNING_MODES

def build_arg_parser():
    parser = argparse.ArgumentParser(description="You can select which file to run over")
    parser.add_argument("pkl_file",nargs="*",help="One or more pickle files with histograms to run over")
    parser.add_argument("--pkl-list-file",default="",help="Optional text file with one pkl path per line")
    parser.add_argument(
        "--rate-syst-json",
        "-s",
        default=None,
        help=(
            "Override the run-era rate-systematics JSON path relative to "
            "topeft_path(); omission keeps DatacardMaker's Run-2/Run-3 default."
        ),
    )
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
    parser.add_argument("--skip-missing-parton-rate-syst",action="store_true",default=False,help="Skip loading/inserting only the missing-parton rate systematic; preserves other nuisances.")
    parser.add_argument(
        "--enable-legacy-run3-vsmu-recovery", action="store_true", default=False,
        help="Explicitly allow recovery of structurally supported pre-fix Run-3 VSmu labels.",
    )
    parser.add_argument("--selected-wcs-ref",default="test/selectedWCs.json",help="Reference file for selected wcs")
    parser.add_argument("--out-dir","-d",default=".",help="Output directory to write root and text datacard files to")
    parser.add_argument("--var-lst",default=[],action="extend",nargs="+",help="Specify a list of variables to make cards for.")
    parser.add_argument("--ch-lst","-c",default=[],action="extend",nargs="+",help="Specify a list of channels to process.")
    parser.add_argument("--do-mc-stat",action="store_true",help="Add bin-by-bin statistical uncertainties with the autoMCstats option (for background)")
    parser.add_argument("--ignore","-i",default=[],action="extend",nargs="+",help="Specify a list of processes to exclude, must match name from 'sample' axis modulo UL year")
    parser.add_argument("--drop-syst",default=[],action="extend",nargs="+",help="Specify one or more template systematics to remove from the datacard")
    parser.add_argument(
        "--suppress-nominal-like-tau-shapes", action="store_true", default=False,
        help="Omit complete local Tau shape pairs below the strict relative 1e-5 threshold (default: off; unsupported with --use-AAC)",
    )
    parser.add_argument("--POI",default=[],help="List of WCs (comma separated)")
    parser.add_argument("--year","-y",default=[],action="extend",nargs="+",help="Run over a subset of years")
    parser.add_argument("--do-nuisance",action="store_true",help="Include nuisance parameters")
    parser.add_argument(
        "--lepton-mva-correlation",
        choices=datacard_tools.lepton_mva_correlation_modes,
        default=datacard_tools.default_lepton_mva_correlation,
        help="Correlation identity for electron and muon MVA shapes (default: per-run)",
    )
    parser.add_argument("--unblind",action="store_true",help="If set, use real data, otherwise use asimov data")
    parser.add_argument("--verbose","-v",action="store_true",help="Set to verbose output")
    parser.add_argument("--select-only",action="store_true",help="Only run the WC selection step")
    parser.add_argument("--skip-selected-wcs-check",action="store_true",help="Do not raise an error if the selected WCs disagree with ref")
    parser.add_argument("--use-selected",default="",help="Load selected process+WC combs from a file. Skips doing the normal selection step.")
    parser.add_argument("--condor", "-C", action="store_true", help="Submit canonical datacard units to HTCondor from --matrix-manifest")
    parser.add_argument("--prepare-only", action="store_true", help="With --condor, write and validate dispatch material without reserving or submitting")
    parser.add_argument("--row-id", action="append", help="With --condor, dispatch an existing manifest row (repeatable)")
    parser.add_argument("--rescue-ambiguous", action="store_true", help="With one --row-id, explicitly supersede an ambiguous row reservation")
    parser.add_argument("--matrix-manifest", help="Canonical v3 workload for Condor preparation")
    parser.add_argument("--condor-output", help="Fresh shared-filesystem directory for dispatch material")
    parser.add_argument("--request-cpus", type=int)
    parser.add_argument("--request-memory")
    parser.add_argument("--request-disk")
    parser.add_argument("--max-runtime", type=int, help="Optional AllowedExecuteDuration seconds per activation, excluding transfer time; timeout holds the job")
    parser.add_argument("--max-retries", type=int, help="Explicit terminal executable retry budget (zero disables retries)")
    parser.add_argument("--max-starts", type=int, help="Explicit total physical start budget, including eviction restarts")
    parser.add_argument("--condor-batch-name", help="Optional HTCondor batch label for operator visibility")
    parser.add_argument("--keep-negative-bins",action="store_true",help="Don't crop negative bins")
    parser.add_argument("--use-AAC","-A",action="store_true",help="Include all EFT templates in datacards for AAC model")
    parser.add_argument("--wc-vals", default="",action="store", nargs="+", help="Specify the corresponding wc values to set for the wc list")
    parser.add_argument("--wc-scalings", default=[],action="extend",nargs="+",help="Specify a list of wc ordering for scalings.json")
    parser.add_argument(
        "--year-coverage-policy",
        choices=["warn", "error", "off"],
        default="warn",
        help=(
            "Structural process/year coverage policy for each histogram-family and "
            "final-channel slice (default: warn)."
        ),
    )
    parser.add_argument("--merge-report",default="-",help="Path for merge diagnostic report JSON, or '-' for stdout")
    parser.add_argument("--merge-only",action="store_true",help="Only load+merge+validate input histograms and exit")
    parser.add_argument("--cache-merged-pkl",default="",help="Optional output path for merged histogram dictionary (.pkl.gz)")
    parser.add_argument(
        "--binning",
        choices=BINNING_MODES,
        default="fitting",
        help=(
            "Histogram binning used for card templates: 'fitting' applies the "
            "current exact channel fitting contract; 'processing' uses the "
            "dense axis stored in the input PKL (default: fitting)."
        ),
    )
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
            histogram_applicability=merge_report.get("histogram_applicability"),
        )
    else:
        with gzip.open(out_fpath, "wb") as fout:
            pickle.dump(merged_hists, fout, protocol=pickle.HIGHEST_PROTOCOL)
    return out_fpath


def _signal_only_selected_wcs(dc, selected_wcs):
    return {
        process: list(wcs)
        for process, wcs in selected_wcs.items()
        if dc.is_signal(process)
    }


def _materialize_selected_wcs(
    out_dir,
    selected_wcs_for_json,
    source_path=None,
):
    output_path = os.path.join(out_dir, "selectedWCs.txt")
    source_is_output = False
    if source_path:
        source_is_output = os.path.abspath(source_path) == os.path.abspath(output_path)
        if os.path.exists(output_path):
            source_is_output = source_is_output or os.path.samefile(
                source_path,
                output_path,
            )

    if source_is_output:
        with open(output_path) as selected_wcs_file:
            existing_selected_wcs = json.load(selected_wcs_file)
        if existing_selected_wcs != selected_wcs_for_json:
            raise ValueError(
                "--use-selected points to the output selectedWCs.txt, but its "
                "content is not the canonical signal-only representation."
            )
        return output_path

    with open(output_path, "w") as selected_wcs_file:
        json.dump(selected_wcs_for_json, selected_wcs_file)
    return output_path

def run_local(dc,km_dists,channels,selected_wcs, crop_negative_bins, wcs_dict):
    for km_dist in km_dists:
        all_chs = dc.channels(km_dist)
        matched_chs = regex_match(all_chs,channels)
        if channels:
            print(f"Channels to process: {matched_chs}")
        for ch in matched_chs:
            r = dc.analyze(km_dist,ch,selected_wcs, crop_negative_bins, wcs_dict)

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.condor:
        # Dispatch before resolving PKLs, loading histograms or producer setup.
        from pathlib import Path
        try:
            from .datacard_matrix_condor import prepare_material, submit_material, runner
        except ImportError:
            from datacard_matrix_condor import prepare_material, submit_material, runner
        if not args.matrix_manifest or not args.condor_output:
            parser.error("--condor requires --matrix-manifest and --condor-output; generate the workload with make_datacard_matrix_manifest.py")
        allowed = {"condor", "prepare_only", "row_id", "rescue_ambiguous", "matrix_manifest", "condor_output", "request_cpus", "request_memory", "request_disk", "max_runtime", "max_retries", "max_starts", "condor_batch_name"}
        allowed_options = {option for action in parser._actions if action.dest in allowed for option in action.option_strings}
        for token in sys.argv[1:]:
            if token.startswith("-") and token.split("=", 1)[0] not in allowed_options:
                parser.error("producer options belong in the canonical manifest; --condor accepts only workload, dispatch and resource options")
        if args.pkl_file:
            parser.error("producer inputs belong in the canonical manifest")
        if args.rescue_ambiguous and (args.prepare_only or not args.row_id or len(args.row_id) != 1):
            parser.error("--rescue-ambiguous requires exactly one --row-id and cannot use --prepare-only")
        try:
            result = prepare_material(Path(args.matrix_manifest), Path(args.condor_output),
                                      request_cpus=args.request_cpus, request_memory=args.request_memory,
                                      request_disk=args.request_disk, max_runtime=args.max_runtime,
                                      max_retries=args.max_retries, max_starts=args.max_starts,
                                      row_ids=args.row_id)
            if not args.prepare_only:
                result = {**result, **submit_material(Path(args.matrix_manifest),
                            Path(args.condor_output), result["scheduler"], batch_name=args.condor_batch_name,
                            row_ids=args.row_id, rescue_ambiguous=args.rescue_ambiguous)}
        except (runner.RunnerError, OSError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2, sort_keys=True))
        if not args.prepare_only and not result["submitted"]:
            raise SystemExit(1)
        return
    if args.prepare_only or args.rescue_ambiguous or args.row_id or any(value is not None for value in (args.matrix_manifest, args.condor_output, args.request_cpus, args.request_memory, args.request_disk, args.max_runtime, args.max_retries, args.max_starts, args.condor_batch_name)):
        parser.error("dispatch and resource options require --condor")
    pkl_files  = _resolve_pkl_paths(args, parser)
    from pathlib import Path
    try:
        from .datacard_matrix_runner import direct_output_lease, RunnerError
    except ImportError:
        from datacard_matrix_runner import direct_output_lease, RunnerError
    try:
        with direct_output_lease(Path(args.out_dir)):
            return _run_producer(args, pkl_files)
    except RunnerError as exc:
        parser.error(str(exc))


def _run_producer(args, pkl_files):
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
    unblind    = args.unblind
    verbose    = args.verbose
    use_AAC     = args.use_AAC
    wc_vals    = args.wc_vals
    binning_mode = args.binning

    wc_scalings = args.wc_scalings
    select_only = args.select_only
    use_selected = args.use_selected


    if isinstance(wcs,str):
        wcs = wcs.split(",")

    kwargs = {
        "wcs": wcs,
        "missing_parton_path": mp_file,
        "out_dir": out_dir,
        "var_lst": var_lst,
        "channel_patterns": ch_lst,
        "do_mc_stat": do_mc_stat,
        "ignore": ignore,
        "do_nuisance": do_nuis,
        "lepton_mva_correlation": args.lepton_mva_correlation,
        "drop_syst": drop_syst,
        "suppress_nominal_like_tau_shapes": args.suppress_nominal_like_tau_shapes,
        "skip_missing_parton_rate_syst": skip_missing_parton_rate_syst,
        "enable_legacy_run3_vsmu_recovery": args.enable_legacy_run3_vsmu_recovery,
        "unblind": unblind,
        "verbose": verbose,
        "year_lst": years,
        "use_AAC":  use_AAC,
        "wc_vals": wc_vals,
        "wc_scalings": wc_scalings,
        "binning_mode": binning_mode,
    }
    if rs_json is not None:
        kwargs["rate_systs_path"] = rs_json

    if out_dir != "." and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if var_lst and args.cache_merged_pkl:
        raise ValueError(
            "A selected histogram closure cannot be cached as a complete artifact."
        )
    merged_hists, merge_report = datacard_tools.load_and_merge_histogram_pkls(
        pkl_files,
        require_sumw2=True,
        year_coverage_policy=args.year_coverage_policy,
        **({"selected_families": var_lst, "channel_patterns": ch_lst} if var_lst else {}),
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
    dc = datacard_tools.DatacardMaker(hists=merged_hists,**kwargs)

    # convert wc_vals string to a dictionary
    wc_vals = ''.join(wc_vals)
    wcs_dict = eval("dict({})".format(wc_vals))

    dists = var_lst if len(var_lst) else dc.hists.keys()
    if use_selected:
        # Use a pre-generated selectionWCs.txt file
        with open(use_selected) as f:
            loaded_selected_wcs = json.load(f)
        selected_wcs_for_json = _signal_only_selected_wcs(
            dc,
            loaded_selected_wcs,
        )
        _materialize_selected_wcs(
            out_dir,
            selected_wcs_for_json,
            source_path=use_selected,
        )
        selected_wcs = {
            process: list(process_wcs)
            for process, process_wcs in loaded_selected_wcs.items()
        }
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
        selected_wcs_for_json = _signal_only_selected_wcs(dc, selected_wcs)
        _materialize_selected_wcs(out_dir, selected_wcs_for_json)

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
