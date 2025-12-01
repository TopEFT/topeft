## topEFT
This directory contains scripts for the Full Run 2 EFT analysis. This README documents and explains how to run the scrips.

### Scripts for remaking reference files that the CI tests against

* `remake_ci_ref_datacard.py`:
    - This script runs the datacard maker tests.
    - Example usage: `python remake_ci_ref_datacard.py`
    
* `remake_ci_ref_datacard.sh`:
    - This script runs `remake_ci_ref_datacard.py` and copies the resulting reference files to the `analysis/topEFT/test`
    - Example usage: `sh remake_ci_ref_datacard.sh`


### Scripts that check or plot things directly from the NAOD files

* `check_for_lepMVA.py`:
    - Checks if the NanoAOD root file has the ttH lepMVA in it

* `make_1d_quad_plots.py`:
    - Makes plots of the inclusive 1d parameterization for the events in an input root file 


### Scripts for making things that are inputs to the processors

* `make_jsons.py`:
    - The purpose of this script is to function as a wrapper for the `topcoffea/modules/createJSON.py` script. That script can also be run from the command line, but many options must be specified, so if you would like to make multiple JSON files or if you will need to remake the JSON files at some point, it is easier to use this script.
    - To make JSON files for samples that you would like to process:
        * Make a dictionary, where the key is the name of the JSON file you would like to produce, and the value is another dictionary, specifying the path to the sample, the name you would like the sample to have on the `sample` axis of the coffea histogram, and the cross section that the sample should correspond to in the `topcoffea/cfg/xsec.cfg` file. The path to the sample should start with `/store`. The existing dictionaries in the file can be used as examples.
        * In the `main()` function, call `make_jsons_for_dict_of_samples()`, and pass your dictionary, along with a redirector (if you are accessing the sample via xrootd), the year of the sample, and the path to an output directory.
        * After the JSON file is produced, it will be moved to the output directory that you specified.
    - Make sure to run `run_sow.py` and `update_json_sow.py` to update the sum of weights before committing and pushing any updates to the json files
    - Once you have produced the JSON file, you should consider committing it to the repository (so that other people can easily process the sample as well), along with the updated `make_jsons.py` script (so that if you have to reproduce the JSON in the future, you will not have to redo any work).
    - Example usage: `python make_jsons.py`

* `make_skim_jsons.py`:
    - Makes JSON files for skimmed samples to be used as input for the processor

* `update_json_sow.py`:
    - This script updates the actual json files corresponding to the samples run with `run_sow.py`
    - Example usage: `python update_json_sow.py histos/sowTopEFT.pkl.gz --json-dir ../../input_samples/sample_jsons/some_json_dir`

* `missing_parton.py`:
    - This script compares two sets of datacards (central NLO and private LO) and computes the necessary uncertainty to bring them into agreement (after account for all included systematics).
    - Datacards should be copied to `histos/central_sm` and `histos/private_sm` respectively.
    - Example usage: `python analysis/topEFT/missing_parton.py --output-path ~/www/coffea/master/1D/ --years 2017`
    - :warning: The part of this script that gets the lumi has not been updated since the `topcoffea` refactoring. 


### Run scripts and processors

* `run_topeft.py` for `topeft.py`:
    - This is the run script for the main `topeft.py` processor. Its usage is documented on the repository's main README. It supports the TaskVine executor for distributed deployments (the default) and the Coffea `futures` executor for single-node testing. TaskVine managers expect workers launched via `vine_submit_workers` (see the main README for command examples). You can configure the run with a number of command line arguments, but the most important one is the config file, where you list the samples you would like to process (by pointing to the JSON files for each sample, located inside of `topcoffea/json`.
    - Example usage: `python run_topeft.py ../../topcoffea/cfg/your_cfg.cfg`

* `run_analysis.py`:
    - Provides the CLI entrypoint for the Run 2 Coffea workflow. The heavy lifting now lives in ``analysis.topeft_run2.workflow`` where the ``RunWorkflow`` class orchestrates channel planning, histogram scheduling, and executor setup.
    - Jet and MET corrections rely on the cache-free factory interfaces shipped with the ``ch_update_calcoffea`` branch of ``topcoffea`` (or a release that includes those helpers). Keep the matching checkout available so ``run_analysis.py`` can build corrected jets/MET without attaching a ``lazy_cache`` to the NanoEvents input.
    - The Run 2 helpers assume Awkward Array ``>=2`` and rely on the library's native ``ak.stack`` implementation rather than a local compatibility shim.
    - YAML-driven presets live in ``analysis/topeft_run2/configs/``.  The ``fullR2_run.yml`` profile mirrors the historic ``fullR2_run.sh`` wrapper and includes both control-region (``default``) and signal-region (``sr``) option bundles.  Launch the control-region pass with::

        python run_analysis.py ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
            --options configs/fullR2_run.yml

      Switch to the signal-region preset by appending ``:sr`` to the YAML path (``--options configs/fullR2_run.yml:sr``).  When ``--options`` is present the YAML file becomes the single source of truth—embed any extra overrides (for example executor choices or log verbosity) directly in the configuration or drop ``--options`` for an ad-hoc CLI run.
    - Metadata scenarios from ``topeft/params/metadata.yml`` can be selected via ``--scenario`` when running without YAML (defaults to ``TOP_22_006``).  Additional bundles include ``tau_analysis`` for the tau-enriched categories and ``fwd_analysis`` for the forward-jet study.  Repeat the argument to combine scenarios on CLI-only runs or edit your YAML presets to keep the combinations version-controlled.
    - When you need to test a custom metadata bundle, clone ``topeft/params/metadata.yml`` and reference it from the profile you launch with ``--options`` via the top-level ``metadata`` key.  This keeps overrides version-controlled while letting scenario-only runs continue to draw from the registry-backed defaults.

    - When running with the futures executor you can now tune the local workflow directly from the CLI (or the matching YAML keys).  The most common toggles are::

          --futures-prefetch 1        # number of ROOT files to stage per sample (0 uses all files)
          --futures-retries 2         # automatic retry attempts when a task fails
          --futures-retry-wait 10     # seconds to wait between retries
          --futures-status            # show or hide the coffea progress bar (--no-futures-status disables it)
          --futures-tail-timeout 300  # cancel stuck tasks after 5 minutes of inactivity
          --futures-memory 2000       # hint (in MB) for dynamic chunk sizing

      YAML profiles accept the same names with underscores (for example ``futures_prefetch`` and ``futures_retry_wait``).  The defaults keep single-node debugging lightweight—only the first ROOT file is staged and no retries are attempted—while still allowing larger local tests before handing work off to TaskVine.

    - Python API example::

        from topeft.analysis import run_workflow
        from analysis.topeft_run2.run_analysis_helpers import RunConfig

        config = RunConfig(json_files=["/path/to/sample.json"])
        run_workflow(config)

      The Run 2 workflow always instantiates NanoEvents factories in explicit
      ``"numpy"`` mode, so local futures runs use in-memory arrays without
      relying on Dask. The coffea runner wiring enforces the mode when
      building factories, which avoids ``_mode`` attribute errors seen with
      coffea >= 0.7 and clarifies that Dask is not a requirement for the
      standard Run 2 configuration.

    - A step-by-step walkthrough of the command-line interface is available in
      the [TOP-22-006 quickstart guide](../docs/quickstart_top22_006.md).  See
      also the [Run 2 quickstart overview](../docs/quickstart_run2.md), the
      [analysis processing primer](../docs/analysis_processing.md), and the
      [YAML configuration guide](../docs/run_analysis_configuration.md) for
      deeper dives into the helper modules and configuration schema.  For a
      minimal end-to-end smoke test, run ``python -m topeft.quickstart`` with a
      JSON from ``input_samples/sample_jsons/``—the helper resolves your samples
      and dispatches a lightweight futures job before returning the recorded
      ``RunConfig`` instance.
    - Logging controls:
        - ``--log-level`` accepts the standard Python logging levels
          (DEBUG/INFO/WARNING/ERROR/CRITICAL, case-insensitive) and defaults to
          ``INFO`` when no explicit flag is provided.  The legacy
          ``--debug-logging`` flag remains available; when present it forces the
          logging level to ``DEBUG`` and turns on the extra
          ``AnalysisProcessor._debug_logging`` instrumentation even if
          ``--log-level`` was also supplied.
        - ``AnalysisProcessor._debug_logging`` is automatically synchronized with
          the effective logging level: it is ``True`` when the level is
          ``DEBUG`` and ``False`` otherwise.
        - Most analysis-level diagnostics (dataset context, task planning, MET/JEC summaries)
          now emit at ``INFO`` so you can see them without enabling the extremely verbose
          DEBUG logs from dependencies; reserve ``--log-level DEBUG`` for deep dives into
          third-party internals.
        - Progress bars and logging output no longer trample each other: the CLI uses a
          ``tqdm``-aware logging handler so ``INFO`` lines remain readable even while coffea
          updates its per-task progress display.
        - Each histogram task now ends with a concise “variation recap” line that lists the
          requested variation labels, the object/weight variations that actually executed,
          and the histogram labels that were filled; search for ``Completed histogram task`` to
          see the summary for a given sample/channel combination.  Internally the processor
          stashes these recaps in a reserved accumulator key, which the workflow pops before
          writing outputs so downstream consumers only see the usual histograms.
        - The combined l–j variables (e.g., ``o0pt``, ``ljptsum``, ``lj0pt``) rely on
          ``_ensure_object_collection_layout`` to normalize the FO lepton and jet inputs
          into a shared ``[events][objects]`` layout before concatenation; if the inputs
          wrap multiple collections or have mismatched event counts, the helper now raises
          an explicit error indicating the offending layout rather than failing deeper inside
          Awkward.
        - Example 5-event futures run with full diagnostics::

            python run_analysis.py ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
                --executor futures --nworkers 1 --chunksize 5 --nchunks 1 --log-level DEBUG

* `full_run.sh`: Unified TaskVine and futures wrapper that mirrors the
  ``fullR3_run.sh`` behavior from the Run 3 development branch.  The script
  expands ``run2``/``run3`` bundles, resolves the appropriate cfg/json inputs,
  and builds the ``run_analysis.py`` invocation for both distributed and
  single-node runs.

    - TaskVine example:

      ``./full_run.sh --cr -y run3 --outdir histos/run3_taskvine --tag dev_validation``

    - Futures example with a single JSON override:

      ``./full_run.sh --sr -y UL17 --executor futures --samples \\
          ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \\
          --outdir histos/local_debug --tag quickstart``

    - Add ``--dry-run`` to print the resolved command without launching Python
      (handy for CI smoke tests and argument validation).

* `run_sow.py` for `sow_processor.py`:
    - This script runs over the provided json files and calculates the properer sum of weights
    - Example usage: `python run_sow.py ../../topcoffea/json/signal_samples/private_UL/UL17_tHq_b1.json --xrd root://deepthought.crc.nd.edu/`


### Scripts for finding, comparing and plotting yields from histograms (from the processor)

* `make_cr_and_sr_plots.py`:
    - This script makes plots for all CRs categories and can also make SR plots.
    - The script takes as input the 5-tuple histogram pickle produced by the current topcoffea runners (e.g. TaskVine output that has both data and background MC).
    - Ensure `python -c "import topcoffea"` succeeds in the active environment. Install the sibling checkout via `pip install -e ../topcoffea` **after running** `git -C ../topcoffea switch ch_update_calcoffea` (or checking out the matching release tag) when developing locally so the processors, packaged environments, and guard all agree on the dependency baseline.
    - Example usage (smoke test with TaskVine-style output): `python make_cr_and_sr_plots.py -f /path/to/taskvine/output/plotsTopEFT.pkl.gz -o /tmp/cr_sr_smoke -n plots -y 2018 --skip-syst`
    - See `examples/run_make_cr_and_sr_plots_smoke.sh` for a ready-to-run wrapper that expects an existing runner output pickle.

* `get_yield_json.py`:
    - This script takes a pkl file produced by the processor, finds the yields in the analysis categories, and saves the yields to a json file. It can also print the info to the screen. The default pkl file to process is `hists/plotsTopEFT.pkl.gz`.
    - Example usage: `python get_yield_json.py -f histos/your_pkl_file.pkl.gz`

* `comp_yields.py`:
    - This script takes two json files of yields (produced by `get_yield_json.py`), finds the difference and percent difference between them in each category, and prints out all of the information. You can also compare to the TOP-19-001 yields by specifying `TOP-19-001` as one of the inputs. Specifying the second file is optional, and it will default to the reference yield file. The script returns a non-zero exit code if any of the percent differences are larger than a given value (currently set to 1e-8). 
    - Example usage: `python comp_yields.py your_yields_1.json your_yields_2.json`


### Scripts for making and checking the datacards

* `make_cards.py`
    - Example usage: `time python make_cards.py /path/to/your.pkl.gz -C --do-nuisance --var-lst lj0pt ptz -d /path/to/output/dir --unblind --do-mc-stat`

* `parse_datacard_templtes.py`:
    - Takes as input the path to a dir that has all of the template root files produced by the datacard maker, can output info about the templates or plots of the templates

* `get_datacard_yields.py`:
    - Gets SM yields from template histograms, dumps the yields (in latex table format) to the screen
    - Example usage: `python get_datacard_yields.py /path/to/dir/with/your/templates/`

* `make_1d_quad_plots_from_template_histos.py`:
    - The purpose of this script was to help to understand the quadratic dependence of the systematics on the WCs. This script takes as input the information from the template histograms, and the goal is to reconstruct the quadratic parameterizations from the templates. The relevant templates are the ones produced by topcoffea's datacard maker, which should be passed to `EFTFit`'s `look_at_templates.C` (which opens the templates, optionally extrapolates the up/down beyond +-1sigma, and dumps the info into a python dictionary). The comments in the script have more information about how to run it. 

* `datacards_post_processing.py`:
    - This script does some basic checks of the cards and templates produced by the `make_cards.py` script.
    - It also can parse the condor log files and dump a summary of the contents
    - Additionally, it can also grab the right set of ptz and lj0pt templates (for the right categories) used in TOP-22-006
    - Example: `python datacards_post_processing.py /path/to/your/datacards/dir -c -s`
