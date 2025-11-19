#!/usr/bin/env python

import argparse
import json
import time
import cloudpickle
import gzip
import os
import shlex

from coffea import processor
from coffea.nanoevents import NanoAODSchema

import topcoffea.modules.utils as utils
import topcoffea.modules.remote_environment as remote_environment
from topcoffea.modules.paths import topcoffea_path

from topeft.modules.dataDrivenEstimation import DataDrivenProducer
from topeft.modules.get_renormfact_envelope import get_renormfact_envelope
import analysis_processor

LST_OF_KNOWN_EXECUTORS = ["futures", "work_queue", "taskvine"]

WGT_VAR_LST = [
    "nSumOfWeights_ISRUp",
    "nSumOfWeights_ISRDown",
    "nSumOfWeights_FSRUp",
    "nSumOfWeights_FSRDown",
    "nSumOfWeights_renormUp",
    "nSumOfWeights_renormDown",
    "nSumOfWeights_factUp",
    "nSumOfWeights_factDown",
    #"nSumOfWeights_renormfactUp",
    #"nSumOfWeights_renormfactDown",
]


def _ensure_topcoffea_data_available(skip_check=False):
    if skip_check:
        return

    target_relpath = "data/pileup/pileup_2016GH.root"
    guidance = (
        "Topcoffea shared data files are missing. Re-run scripts/install_topcoffea.sh "
        "from the topeft checkout so the matching topcoffea branch (e.g. run3_test_mmerged) "
        "and its data bundles are installed, or pass --skip-topcoffea-data-check if your setup "
        "provides the resources elsewhere."
    )

    try:
        pileup_path = topcoffea_path(target_relpath)
    except FileNotFoundError as exc:
        raise SystemExit(f"{guidance} (lookup failed for {target_relpath}).") from exc

    if not os.path.exists(pileup_path):
        raise SystemExit(f"{guidance} (expected {pileup_path}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="You can customize your run")
    parser.add_argument(
        "jsonFiles",
        nargs="?",
        default="",
        help="Json file(s) containing files and metadata",
    )
    parser.add_argument(
        "--executor",
        "-x",
        default="work_queue",
        help="Which executor to use",
    )
    parser.add_argument(
        "--prefix",
        "-r",
        nargs="?",
        default="",
        help="Prefix or redirector to look for the files",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="To perform a test, run over a few events in a couple of chunks",
    )
    parser.add_argument(
        "--pretend",
        action="store_true",
        help="Read json files but do not execute the analysis",
    )
    parser.add_argument(
        "--nworkers",
        "-n",
        default=8,
        type=int,
        help="Number of workers",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Alias for --nworkers (kept for backwards compatibility)",
    )
    parser.add_argument(
        "--chunksize",
        "-s",
        default=100000,
        help="Number of events per chunk",
    )
    parser.add_argument(
        "--nchunks",
        "-c",
        default=None,
        help="You can choose to run only a number of chunks",
    )
    parser.add_argument(
        "--outname",
        "-o",
        default="plotsTopEFT",
        help="Name of the output file with histograms",
    )
    parser.add_argument(
        "--outpath",
        "-p",
        default="histos",
        help="Name of the output directory",
    )
    parser.add_argument(
        "--years",
        "-y",
        nargs="+",
        help="Limit processing to the specified data-taking years",
    )
    parser.add_argument(
        "--treename",
        default="Events",
        help="Name of the tree inside the files",
    )
    parser.add_argument(
        "--no-sumw2",
        action="store_true",
        help="Skip filling sum of weight-squared histograms",
    )
    parser.add_argument(
        "--do-systs",
        action="store_true",
        help="Compute systematic variations",
    )
    parser.add_argument(
        "--split-lep-flavor",
        action="store_true",
        help="Split up categories by lepton flavor",
    )
    parser.add_argument(
        "--offZ-split",
        action="store_true",
        help="Split up 3l offZ categories",
    )
    parser.add_argument(
        "--tau_h_analysis",
        action="store_true",
        help="Add tau channels",
    )
    parser.add_argument(
        "--fwd-analysis",
        action="store_true",
        help="Add fwd channels",
    )
    parser.add_argument(
        "--skip-sr",
        action="store_true",
        help="Skip all signal region categories",
    )
    parser.add_argument(
        "--skip-cr",
        action="store_true",
        help="Skip all control region categories",
    )
    parser.add_argument(
        "--do-np",
        action="store_true",
        help=(
            "Perform nonprompt estimation on the output hist, and save a new hist "
            "with the np contribution included. Signal, background and data samples "
            "must all be processed together."
        ),
    )
    parser.add_argument(
        "--np-postprocess",
        choices=["inline", "defer", "skip"],
        default="inline",
        help=(
            "Control when the nonprompt post-processing step runs. "
            "Use 'inline' (default) to run immediately, 'defer' to emit metadata "
            "for a follow-up job, or 'skip' to omit the step entirely."
        ),
    )
    parser.add_argument(
        "--do-renormfact-envelope",
        action="store_true",
        help=(
            "Perform renorm/fact envelope calculation on the output hist "
            "(saves the modified with the same name as the original)."
        ),
    )
    parser.add_argument(
        "--wc-list",
        action="extend",
        nargs="+",
        help="Specify a list of Wilson coefficients to use in filling histograms.",
    )
    parser.add_argument(
        "--hist-list",
        action="extend",
        nargs="+",
        help="Specify a list of histograms to fill.",
    )
    parser.add_argument(
        "--ecut",
        default=None,
        help="Energy cut threshold i.e. throw out events above this (GeV)",
    )
    parser.add_argument(
        "--port",
        default="9123-9130",
        help="Specify the Work Queue port. An integer PORT or an integer range PORT_MIN-PORT_MAX.",
    )
    parser.add_argument(
        "--noRun3MVA",
        action='store_false',
        default=True,
        help = 'Do not use the Run3 MVA for lepton selection. Default is to use it.',
    )
    parser.add_argument(
        "--options",
        default=None,
        help="YAML file that specifies command-line options. Options explicitly set at command-line take precedence",
    )
    parser.add_argument(
        "--analysis-mode",
        choices=["standard", "taufitter"],
        default="standard",
        help=(
            "Select the analysis workflow configuration. Use 'standard' for the default behaviour "
            "or 'taufitter' to enable tau fitter specific handling."
        ),
    )
    parser.add_argument(
        "--skip-topcoffea-data-check",
        action="store_true",
        help=(
            "Bypass the startup sanity check that verifies the shared topcoffea data files are present. "
            "Use only for expert/custom setups."
        ),
    )

    args = parser.parse_args()
    if args.workers is not None:
        args.nworkers = args.workers
    _ensure_topcoffea_data_available(args.skip_topcoffea_data_check)
    jsonFiles = args.jsonFiles
    prefix = args.prefix
    executor_name = args.executor
    dotest = args.test
    nworkers = int(args.nworkers)
    chunksize = int(args.chunksize)
    nchunks = int(args.nchunks) if not args.nchunks is None else args.nchunks
    outname = args.outname
    outpath = args.outpath
    pretend = args.pretend
    treename = args.treename
    fill_sumw2 = not args.no_sumw2
    do_systs = args.do_systs
    split_lep_flavor = args.split_lep_flavor
    offZ_split = args.offZ_split
    tau_h_analysis = args.tau_h_analysis
    fwd_analysis = args.fwd_analysis
    skip_sr    = args.skip_sr
    skip_cr    = args.skip_cr
    do_np      = args.do_np
    np_postprocess_mode = args.np_postprocess
    useRun3MVA = args.noRun3MVA #NB: default value is True, the arg starts with no because its usage prevents making selections with the run3 MVA
    do_renormfact_envelope = args.do_renormfact_envelope
    wc_lst = args.wc_list if args.wc_list is not None else []
    ecut = args.ecut
    port = args.port
    hist_list = args.hist_list
    analysis_mode = args.analysis_mode

    if args.options:
        import yaml
        with open(args.options,'r') as f:
            ops = yaml.load(f,Loader=yaml.Loader)
        jsonFiles = ops.pop("jsonFiles",jsonFiles)
        prefix = ops.pop("prefix",prefix)
        executor_name = ops.pop("executor",executor_name)
        dotest = ops.pop("test",dotest)
        nworkers = ops.pop("nworkers",nworkers)
        chunksize = ops.pop("chunksize",chunksize)
        nchunks = ops.pop("nchunks",nchunks)
        outname = ops.pop("outname",outname)
        outpath = ops.pop("outpath",outpath)
        pretend = ops.pop("pretend",pretend)
        treename = ops.pop("treename",treename)
        no_sumw2_opt = ops.pop("no_sumw2", None)
        if no_sumw2_opt is not None:
            fill_sumw2 = not no_sumw2_opt
        else:
            legacy_do_errors = ops.pop("do_errors", None)
            if legacy_do_errors is not None:
                fill_sumw2 = bool(legacy_do_errors)
        do_systs = ops.pop("do_systs",do_systs)
        split_lep_flavor = ops.pop("split_lep_flavor",split_lep_flavor)
        offZ_split = ops.pop("offZ_split",offZ_split)
        tau_h_analysis = ops.pop("tau_h_analysis",tau_h_analysis)
        fwd_analysis = ops.pop("fwd_analysis",fwd_analysis)
        skip_sr = ops.pop("skip_sr",skip_sr)
        skip_cr = ops.pop("skip_cr",skip_cr)
        do_np = ops.pop("do_np",do_np)
        np_postprocess_mode = ops.pop("np_postprocess", np_postprocess_mode)
        do_renormfact_envelope = ops.pop("do_renormfact_envelope",do_renormfact_envelope)
        wc_lst = ops.pop("wc_list",wc_lst)
        hist_list = ops.pop("hist_list",hist_list)
        port = ops.pop("port",port)
        ecut = ops.pop("ecut",ecut)
        analysis_mode = ops.pop("analysis_mode", analysis_mode)

    out_pkl_file = os.path.join(outpath, outname + ".pkl.gz")
    out_pkl_file_name_np = os.path.join(outpath, outname + "_np.pkl.gz")
    np_metadata_file = out_pkl_file_name_np + ".metadata.json"

    # Check if we have valid options
    if executor_name not in LST_OF_KNOWN_EXECUTORS:
        raise Exception(
            f'The "{executor_name}" executor is not known. Please specify an executor from the known executors ({LST_OF_KNOWN_EXECUTORS}). Exiting.'
        )
    if do_renormfact_envelope:
        if not do_systs:
            raise Exception(
                "Error: Cannot specify do_renormfact_envelope if we are not including systematics."
            )
        if not do_np:
            raise Exception(
                "Error: Cannot specify do_renormfact_envelope if we have not already done the integration across the appl axis that occurs in the data driven estimator step."
            )
        if np_postprocess_mode != "inline":
            raise Exception(
                "Error: Renorm/fact envelope requires inline nonprompt post-processing."
            )
    if dotest:
        if executor_name == "futures":
            nchunks = 2
            chunksize = 10000
            nworkers = 1
            print(
                "Running a fast test with %i workers, %i chunks of %i events"
                % (nworkers, nchunks, chunksize)
            )
        else:
            raise Exception(
                f'The "test" option is not set up to work with the {executor_name} executor. Exiting.'
            )

    # Set the threshold for the ecut (if not applying a cut, should be None)
    ecut_threshold = ecut
    if ecut_threshold is not None:
        ecut_threshold = float(ecut)

    if executor_name in ["work_queue", "taskvine"]:
        # construct wq port range
        port = list(map(int, port.split("-")))
        if len(port) < 1:
            raise ValueError("At least one port value should be specified.")
        if len(port) > 2:
            raise ValueError("More than one port range was specified.")
        if len(port) == 1:
            # convert singale values into a range of one element
            port.append(port[0])

    # Figure out which hists to include
    if hist_list == ["ana"]:
        # Here we hardcode a list of hists used for the analysis
        hist_lst = ["njets", "lj0pt", "ptz"]
        if tau_h_analysis:
            hist_lst.append("ptz_wtau")
            hist_lst.append("tau0pt")
        if fwd_analysis:
            hist_lst.append("lt")
        if "lepton_pt_vs_eta" not in hist_lst:
            hist_lst.append("lepton_pt_vs_eta")
        if "l0_SeedEtaOrX_vs_SeedPhiOrY" not in hist_lst:
            hist_lst.append("l0_SeedEtaOrX_vs_SeedPhiOrY")
        if "l0_eta_vs_phi" not in hist_lst:
            hist_lst.append("l0_eta_vs_phi")
        if "l1_SeedEtaOrX_vs_SeedPhiOrY" not in hist_lst:
            hist_lst.append("l1_SeedEtaOrX_vs_SeedPhiOrY")
        if "l1_eta_vs_phi" not in hist_lst:
            hist_lst.append("l1_eta_vs_phi")
        if fill_sumw2 and "lepton_pt_vs_eta_sumw2" not in hist_lst:
            hist_lst.append("lepton_pt_vs_eta_sumw2")
        if fill_sumw2 and "l0_SeedEtaOrX_vs_SeedPhiOrY_sumw2" not in hist_lst:
            hist_lst.append("l0_SeedEtaOrX_vs_SeedPhiOrY_sumw2")
        if fill_sumw2 and "l0_eta_vs_phi_sumw2" not in hist_lst:
            hist_lst.append("l0_eta_vs_phi_sumw2")
        if fill_sumw2 and "l1_SeedEtaOrX_vs_SeedPhiOrY_sumw2" not in hist_lst:
            hist_lst.append("l1_SeedEtaOrX_vs_SeedPhiOrY_sumw2")
        if fill_sumw2 and "l1_eta_vs_phi_sumw2" not in hist_lst:
            hist_lst.append("l1_eta_vs_phi_sumw2")
    elif args.hist_list == ["cr"]:
        # Here we hardcode a list of hists used for the CRs
        hist_lst = [
            "lj0pt",
            # "ptz",
            "met",
            # "ljptsum",
            # "l0pt",
            # "l0ptcorr",
            "l0conept",
            "l0eta",
            # "l1pt",
            # "l1ptcorr",
            "l1conept",
            # "l1eta",
            "j0pt",
            "j0eta",
            "njets",
            # "nbtagsl",
            "invmass",
            # "npvs",
            # "npvsGood",
            # "l0_gen_pdgId",
            # "l1_gen_pdgId",
            # "l2_gen_pdgId",
            # "l0_genParent_pdgId",
            # "l1_genParent_pdgId",
            # "l2_genParent_pdgId",
            # "b0l_hFlav",
            # "b0m_hFlav",
            # "b0l_pFlav",
            # "b0m_pFlav",
            # "b1l_hFlav",
            # "b1m_hFlav",
            # "b1l_pFlav",
            # "b1m_pFlav",
            # "b0l_genhFlav",
            # "b0m_genhFlav",
            # "b0l_genpFlav",
            # "b0m_genpFlav",
            # "b1l_genhFlav",
            # "b1m_genhFlav",
            # "b1l_genpFlav",
            # "b1m_genpFlav",
            # "lepton_pt_vs_eta",
            # "l0_SeedEtaOrX_vs_SeedPhiOrY",
            # "l0_eta_vs_phi",
            # "l1_SeedEtaOrX_vs_SeedPhiOrY",
            # "l1_eta_vs_phi",
        ]
        if tau_h_analysis:
            hist_lst.append("tau0pt")
    else:
        # We want to specify a custom list
        # If we don't specify this argument, it will be None, and the processor will fill all hists
        hist_lst = args.hist_list

    ### Load samples from json
    samplesdict = {}
    allInputFiles = []

    def LoadJsonToSampleName(jsonFile, prefix):
        sampleName = (
            jsonFile if not "/" in jsonFile else jsonFile[jsonFile.rfind("/") + 1 :]
        )
        if sampleName.endswith(".json"):
            sampleName = sampleName[:-5]
        with open(jsonFile) as jf:
            samplesdict[sampleName] = json.load(jf)
            samplesdict[sampleName]["redirector"] = prefix

    if isinstance(jsonFiles, str) and "," in jsonFiles:
        jsonFiles = jsonFiles.replace(" ", "").split(",")
    elif isinstance(jsonFiles, str):
        jsonFiles = [jsonFiles]
    for jsonFile in jsonFiles:
        if os.path.isdir(jsonFile):
            if not jsonFile.endswith("/"):
                jsonFile += "/"
            for f in os.path.listdir(jsonFile):
                if f.endswith(".json"):
                    allInputFiles.append(jsonFile + f)
        else:
            allInputFiles.append(jsonFile)

    resolved_input_jsons = list(allInputFiles)

    # Read from cfg files
    for f in allInputFiles:
        if not os.path.isfile(f):
            raise Exception(f"[ERROR] Input file {f} not found!")
        # This input file is a json file, not a cfg
        if f.endswith(".json"):
            LoadJsonToSampleName(f, prefix)
        # Open cfg files
        else:
            with open(f) as fin:
                print(" >> Reading json from cfg file...")
                lines = fin.readlines()
                for l in lines:
                    if "#" in l:
                        l = l[: l.find("#")]
                    l = l.replace(" ", "").replace("\n", "")
                    if l == "":
                        continue
                    if "," in l:
                        l = l.split(",")
                        for nl in l:
                            if not os.path.isfile(l):
                                prefix = nl
                            else:
                                LoadJsonToSampleName(nl, prefix)
                    else:
                        if not os.path.isfile(l):
                            prefix = l
                        else:
                            LoadJsonToSampleName(l, prefix)

    requested_years = None
    if args.years:
        valid_year_choices = {
            "UL16",
            "UL16APV",
            "UL17",
            "UL18",
            "2016",
            "2016APV",
            "2017",
            "2018",
            "2022",
            "2022EE",
            "2023",
            "2023BPix",
            "run2",
            "run3",
        }
        year_synonyms = {
            "2016": {"2016", "UL16"},
            "UL16": {"2016", "UL16"},
            "2016APV": {"2016APV", "UL16APV"},
            "UL16APV": {"2016APV", "UL16APV"},
            "2017": {"2017", "UL17"},
            "UL17": {"2017", "UL17"},
            "2018": {"2018", "UL18"},
            "UL18": {"2018", "UL18"},
            "run2": {
                "2016",
                "2016APV",
                "2017",
                "2018",
                "UL16",
                "UL16APV",
                "UL17",
                "UL18",
            },
            "run3": {"2022", "2022EE", "2023", "2023BPix"},
        }

        requested_years = set()
        for year in args.years:
            year_str = str(year)
            if year_str not in valid_year_choices:
                raise ValueError(
                    "Invalid year selection \"{}\". Valid choices are: {}".format(
                        year_str, ", ".join(sorted(valid_year_choices))
                    )
                )

            requested_years.update(year_synonyms.get(year_str, {year_str}))

    if requested_years is not None:
        samplesdict = {
            name: sample
            for name, sample in samplesdict.items()
            if str(sample.get("year")) in requested_years
        }

        if not samplesdict:
            raise ValueError(
                "No samples remaining after applying the requested year filter."
            )

        
    flist = {}
    nevts_total = 0
    for sname in samplesdict.keys():

        samplesdict[sname]["files"] = samplesdict[sname]["files"]  # [0:1]

        redirector = samplesdict[sname]["redirector"]
        flist[sname] = [(redirector + f) for f in samplesdict[sname]["files"]]
        samplesdict[sname]["year"] = samplesdict[sname]["year"]
        samplesdict[sname]["xsec"] = float(samplesdict[sname]["xsec"])
        samplesdict[sname]["nEvents"] = int(samplesdict[sname]["nEvents"])
        nevts_total += samplesdict[sname]["nEvents"]
        samplesdict[sname]["nGenEvents"] = int(samplesdict[sname]["nGenEvents"])
        samplesdict[sname]["nSumOfWeights"] = float(samplesdict[sname]["nSumOfWeights"])
        if not samplesdict[sname]["isData"]:
            for wgt_var in WGT_VAR_LST:
                # Check that MC samples have all needed weight sums (only needed if doing systs)
                if do_systs:
                    if wgt_var not in samplesdict[sname]:
                        raise Exception(f'Missing weight variation "{wgt_var}".')
                    else:
                        samplesdict[sname][wgt_var] = float(samplesdict[sname][wgt_var])
        # Print file info
        print(">> " + sname)
        print(
            "   - isData?      : %s" % ("YES" if samplesdict[sname]["isData"] else "NO")
        )
        print("   - year         : %s" % samplesdict[sname]["year"])
        print("   - xsec         : %f" % samplesdict[sname]["xsec"])
        print("   - histAxisName : %s" % samplesdict[sname]["histAxisName"])
        print("   - options      : %s" % samplesdict[sname]["options"])
        print("   - tree         : %s" % samplesdict[sname]["treeName"])
        print("   - nEvents      : %i" % samplesdict[sname]["nEvents"])
        print("   - nGenEvents   : %i" % samplesdict[sname]["nGenEvents"])
        print("   - SumWeights   : %i" % samplesdict[sname]["nSumOfWeights"])
        if not samplesdict[sname]["isData"]:
            for wgt_var in WGT_VAR_LST:
                if wgt_var in samplesdict[sname]:
                    print(f"   - {wgt_var}: {samplesdict[sname][wgt_var]}")
        print("   - Prefix       : %s" % samplesdict[sname]["redirector"])
        print("   - nFiles       : %i" % len(samplesdict[sname]["files"]))
        for fname in samplesdict[sname]["files"]:
            print("     %s" % fname)

        if executor_name == "futures":
            break

    sample_years_from_inputs = sorted(
        {
            str(sample.get("year"))
            for sample in samplesdict.values()
            if sample.get("year") is not None
        }
    )

    def _build_np_followup_command():
        followup_snippet = (
            "from topeft.modules.dataDrivenEstimation import DataDrivenProducer; "
            f"DataDrivenProducer({out_pkl_file!r}, {out_pkl_file_name_np!r}).dumpToPickle()"
        )
        return f"python -c {shlex.quote(followup_snippet)}"

    def _build_np_metadata_payload():
        resolved_year_list = (
            sorted(requested_years) if requested_years is not None else sample_years_from_inputs
        )
        payload = {
            "metadata_version": 1,
            "timestamp": time.time(),
            "input_histogram": out_pkl_file,
            "output_histogram": out_pkl_file_name_np,
            "metadata_path": np_metadata_file,
            "np_postprocess": np_postprocess_mode,
            "pretend_mode": pretend,
            "do_np": do_np,
            "resolved_years": resolved_year_list,
            "sample_years": sample_years_from_inputs,
            "input_jsons": resolved_input_jsons,
            "analysis_mode": analysis_mode,
            "hist_list": hist_lst,
            "wc_list": wc_lst,
            "executor": executor_name,
            "options_file": args.options,
            "flags": {
                "split_lep_flavor": split_lep_flavor,
                "offZ_split": offZ_split,
                "tau_h_analysis": tau_h_analysis,
                "fwd_analysis": fwd_analysis,
                "skip_sr": skip_sr,
                "skip_cr": skip_cr,
                "do_systs": do_systs,
                "fill_sumw2": fill_sumw2,
                "useRun3MVA": useRun3MVA,
            },
            "followup_command": _build_np_followup_command(),
        }
        return payload

    def _write_np_metadata_sidecar(*, pretend_override=None):
        payload = _build_np_metadata_payload()
        if pretend_override is not None:
            payload["pretend_mode"] = pretend_override
        os.makedirs(outpath, exist_ok=True)
        with open(np_metadata_file, "w") as metadata_stream:
            json.dump(payload, metadata_stream, indent=2, sort_keys=True)
        return payload

    def _print_np_defer_instructions(metadata_payload):
        followup_command = metadata_payload.get("followup_command", _build_np_followup_command())
        print(
            "Nonprompt estimation deferred. Metadata saved to {}.\n"
            "Run the following command to finalize the nonprompt histograms:\n  {}".format(
                metadata_payload.get("metadata_path", np_metadata_file), followup_command
            )
        )
            
    if pretend:
        print("pretending...")
        if do_np and np_postprocess_mode == "defer":
            metadata_payload = _write_np_metadata_sidecar(pretend_override=True)
            _print_np_defer_instructions(metadata_payload)
        exit()

    # Extract the list of all WCs, as long as we haven't already specified one.
    if len(wc_lst) == 0:
        for k in samplesdict.keys():
            for wc in samplesdict[k]["WCnames"]:
                if wc not in wc_lst:
                    wc_lst.append(wc)

    if len(wc_lst) > 0:
        # Yes, why not have the output be in correct English?
        if len(wc_lst) == 1:
            wc_print = wc_lst[0]
        elif len(wc_lst) == 2:
            wc_print = wc_lst[0] + " and " + wc_lst[1]
        else:
            wc_print = ", ".join(wc_lst[:-1]) + ", and " + wc_lst[-1]
            print("Wilson Coefficients: {}.".format(wc_print))
    else:
        print("No Wilson coefficients specified")

    print("Variables to be histogrammed: {}".format(", ".join(hist_lst)))

    processor_instance = analysis_processor.AnalysisProcessor(
        samplesdict,
        wc_lst,
        hist_lst,
        ecut_threshold,
        fill_sumw2,
        do_systs,
        split_lep_flavor,
        skip_sr,
        skip_cr,
        offZ_split=offZ_split,
        tau_h_analysis=tau_h_analysis,
        fwd_analysis=fwd_analysis,
        useRun3MVA=useRun3MVA,
        tau_run_mode=analysis_mode
    )

    if executor_name in ["work_queue", "taskvine"]:
        executor_args = {
            "manager_name": f"{os.environ['USER']}-workqueue-coffea",
            # find a port to run work queue in this range:
            "port": port,
            "debug_log": "debug.log",
            "transactions_log": "tr.log",
            "stats_log": "stats.log",
            "tasks_accum_log": "tasks.log",
            "environment_file": remote_environment.get_environment(
                extra_pip_local={"topeft": ["topeft", "setup.py"]},
            ),
            "extra_input_files": ["analysis_processor.py"],
            "retries": 15,
            # use mid-range compression for chunks results.
            # Valid values are 0 (minimum compression, less memory
            # usage) to 16 (maximum compression, more memory usage).
            "compression": 8,
            # automatically find an adequate resource allocation for tasks.
            # tasks are first tried using the maximum resources seen of previously ran
            # tasks. on resource exhaustion, they are retried with the maximum resource
            # values, if specified below. if a maximum is not specified, the task waits
            # forever until a larger worker connects.
            # 'resource_monitor': True,
            "resource_monitor": "measure",
            "resources_mode": "auto",
            #'filepath': f'/tmp/{os.environ["USER"]}-workers', ##Placeholder to comment out if you don't want to save wq-factory dirs in $HOME
            "filepath": '/tmp',
            # this resource values may be omitted when using
            # resources_mode: 'auto', but they do make the initial portion
            # of a workflow run a little bit faster.
            # Rather than using whole workers in the exploratory mode of
            # resources_mode: auto, tasks are forever limited to a maximum
            # of 8GB of mem and disk.
            #
            # NOTE: The very first tasks in the exploratory
            # mode will use the values specified here, so workers need to be at least
            # this large. If left unspecified, tasks will use whole workers in the
            # exploratory mode.
            # 'cores': 1,
            # 'disk': 8000,   #MB
            # 'memory': 10000, #MB
            # control the size of accumulation tasks.
            "treereduction": 10,
            # terminate workers on which tasks have been running longer than average.
            # This is useful for temporary conditions on worker nodes where a task will
            # be finish faster is ran in another worker.
            # the time limit is computed by multipliying the average runtime of tasks
            # by the value of 'fast_terminate_workers'.  Since some tasks can be
            # legitimately slow, no task can trigger the termination of workers twice.
            #
            # warning: small values (e.g. close to 1) may cause the workflow to misbehave,
            # as most tasks will be terminated.
            #
            # Less than 1 disables it.
            "fast_terminate_workers": 0,
            # print messages when tasks are submitted, finished, etc.,
            # together with their resource allocation and usage. If a task
            # fails, its standard output is also printed, so we can turn
            # off print_stdout for all tasks.
            "verbose": True,
            "print_stdout": False,
        }

    # Run the processor and get the output
    tstart = time.time()

    if executor_name == "futures":
        futures_factory = getattr(processor, "futures_executor", None)
        if callable(futures_factory):
            exec_instance = futures_factory(workers=nworkers)
        else:
            exec_instance = processor.FuturesExecutor(workers=nworkers)
        runner = processor.Runner(
            exec_instance, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks
        )
    elif executor_name == "work_queue":
        executor_instance = processor.WorkQueueExecutor(**executor_args)
        runner = processor.Runner(
            executor_instance,
            schema=NanoAODSchema,
            chunksize=chunksize,
            maxchunks=nchunks,
            skipbadfiles=False,
            xrootdtimeout=180,
        )
    elif executor_name == "taskvine":
        try:
            executor_instance = processor.TaskVineExecutor(**executor_args)
        except AttributeError:
            raise RuntimeError("TaskVineExecutor not available.")
        runner = processor.Runner(
            executor_instance,
            schema=NanoAODSchema,
            chunksize=chunksize,
            maxchunks=nchunks,
            skipbadfiles=True,
            xrootdtimeout=300,
        )

    output = runner(flist, treename, processor_instance)

    print("Finished running the processor...")

    dt = time.time() - tstart

    if executor_name in ["work_queue", "taskvine"]:
        print(
            "Processed {} events in {} seconds ({:.2f} evts/sec).".format(
                nevts_total, dt, nevts_total / dt
            )
        )

    # nbins = sum(sum(arr.size for arr in h.eval({}).values()) for h in output.values() if isinstance(h, hist.Hist))
    # nfilled = sum(sum(np.sum(arr > 0) for arr in h.eval({}).values()) for h in output.values() if isinstance(h, hist.Hist))
    # print("Filled %.0f bins, nonzero bins: %1.1f %%" % (nbins, 100*nfilled/nbins,))

    if executor_name == "futures":
        print(
            "Processing time: %1.2f s with %i workers (%.2f s cpu overall)"
            % (
                dt,
                nworkers,
                dt * nworkers,
            )
        )

    # Save the output
    os.makedirs(outpath, exist_ok=True)
    print(f"\nSaving output in {out_pkl_file}...")
    with gzip.open(out_pkl_file, "wb") as fout:
        cloudpickle.dump(output, fout)
    print("Done!")

    # Run the data driven estimation, save the output
    if do_np:
        if np_postprocess_mode == "inline":
            print("\nDoing the nonprompt estimation...")
            ddp = DataDrivenProducer(out_pkl_file, out_pkl_file_name_np)
            print(f"Saving output in {out_pkl_file_name_np}...")
            ddp.dumpToPickle()
            print("Done!")
            if do_renormfact_envelope:
                print("\nDoing the renorm. fact. envelope calculation...")
                dict_of_histos = utils.get_hist_from_pkl(
                    out_pkl_file_name_np, allow_empty=False
                )
                dict_of_histos_after_applying_envelope = get_renormfact_envelope(
                    dict_of_histos
                )
                utils.dump_to_pkl(
                    out_pkl_file_name_np, dict_of_histos_after_applying_envelope
                )
        elif np_postprocess_mode == "defer":
            print("\nDeferring the nonprompt estimation and writing metadata...")
            metadata_payload = _write_np_metadata_sidecar()
            _print_np_defer_instructions(metadata_payload)
        else:
            print("\nSkipping the nonprompt estimation as requested (--np-postprocess=skip).")
