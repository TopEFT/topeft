## topEFT
This directory contains scripts for the Full Run 2 EFT analysis. This README documents and explains how to run the scrips.

### Table of Contents
- [Scripts for remaking reference files that the CI tests against](#scripts-for-remaking-reference-files-that-the-ci-tests-against)
- [Scripts that check or plot things directly from the NAOD files](#scripts-that-check-or-plot-things-directly-from-the-naod-files)
- [Scripts for making things that are inputs to the processors](#scripts-for-making-things-that-are-inputs-to-the-processors)
- [Run scripts and processors](#run-scripts-and-processors)
- [Scripts for finding, comparing and plotting yields from histograms (from the processor)](#scripts-for-finding-comparing-and-plotting-yields-from-histograms-from-the-processor)
- [Scripts for making and checking the datacards](#scripts-for-making-and-checking-the-datacards)
- [CR/SR plotting CLI quickstart](#crsr-plotting-cli-quickstart)
  - [run\_plotter.sh shell wrapper quickstart](#run_plottersh-shell-wrapper-quickstart)
- [HTCondor plotting on Glados](#htcondor-plotting-on-glados)
- [make_cr_and_sr_plots.py internals](#make_cr_and_sr_plotspy-internals)
- [CR/SR metadata reference](#crsr-metadata-reference)

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
    - This is the run script for the main `topeft.py` processor. Its usage is documented on the repository's main README. It uses either the `work_queue` or the `futures` executors (with `futures` it uses 8 cores by default). The `work_queue` executor makes use of remote resources, and you will need to submit workers using a `condor_submit_workers` command as explained on the main `topcoffea` README. You can configure the run with a number of command line arguments, but the most important one is the config file, where you list the samples you would like to process (by pointing to the JSON files for each sample, located inside of `topcoffea/json`.
    - Example usage: `python run_topeft.py ../../topcoffea/cfg/your_cfg.cfg`

* `run_analysis.py`:
    - Thin wrapper around `analysis_processor.py` used for the standard CR/analysis histogram production. The canned histogram lists now include the 2D `lepton_pt_vs_eta` observable (and keep the matching `_sumw2` companion unless `--no-sumw2` is passed) so downstream tools can rely on a consistent pt vs $|\eta|$ binning description.
    - Leave the default `sumw²` companions enabled whenever you plan to run downstream uncertainty-aware tooling such as the tau fake-rate fitter or the diboson scale-factor extractor. Disabling them with `--no-sumw2` drops the `*_sumw2` histograms (for example `tau0pt_sumw2`), which causes those utilities to fail or to lose their statistical error propagation. If you need to trim the histogram list, remove individual observables instead of the sumw² accumulators.
    - Pass `--years YEAR [YEAR ...]` to filter the loaded JSON samples to the requested campaign tokens. Supported values are `2016`, `2016APV`, `2017`, `2018`, `2022`, `2022EE`, `2023`, `2023BPix`, their UL aliases (`UL16`, `UL16APV`, `UL17`, `UL18`), and the aggregate shorthands `run2` (`UL16 UL16APV UL17 UL18`) and `run3` (`2022 2022EE 2023 2023BPix`). Legacy tokens remain valid, so existing command snippets do not require changes. When the option is absent every sample in the configuration is retained as before.
    - The data-driven helper now supports inline and deferred workflows. Keep the historical behaviour by relying on the default `--np-postprocess=inline` (paired with `--do-np`) so the `_np.pkl.gz` file appears immediately. Choose `--np-postprocess=defer` **together with `--do-np`** to emit the base pickle along with `histos/<outname>_np.pkl.gz.metadata.json`, which records the resolved years, follow-up command, and absolute histogram paths for later processing. Setting `--np-postprocess=skip` suppresses the data-driven step entirely.
    - Startup now includes a quick sanity check that resolves `data/pileup/pileup_2016GH.root` via `topcoffea_path` and ensures the file exists. When it fails the CLI exits with instructions to re-run `scripts/install_topcoffea.sh`, verify the `external/topcoffea` checkout (currently `run3_test_mmerged`) is available, and try again. Use `--skip-topcoffea-data-check` only when you intentionally manage the shared pileup files outside of the helper script.
    - The metadata sidecar allows the deferred helper to reconstruct the `_np.pkl.gz` file without repeating the whole analysis: `python run_data_driven.py --metadata-json histos/<outname>_np.pkl.gz.metadata.json`. Pass `--input-pkl / --output-pkl` directly if you prefer not to use the metadata. The helper also exposes `--apply-renormfact-envelope` so the deferred runs match the inline envelope path.

* `run_sow.py` for `sow_processor.py`:
    - This script runs over the provided json files and calculates the properer sum of weights
    - Example usage: `python run_sow.py ../../topcoffea/json/signal_samples/private_UL/UL17_tHq_b1.json --xrd root://deepthought.crc.nd.edu/`

* `fullR3_run.sh`: Recommended wrapper script for both Run 2 and Run 3 histogram production. It expands the aggregate campaign aliases (`run2` → `UL16 UL16APV UL17 UL18`, `run3` → `2022 2022EE 2023 2023BPix`) before dispatching to `run_analysis.py`, superseding the legacy helper while keeping the historical single-year tokens functioning as before.
    - Whenever the Run 2 bundle is activated (any of `2016`, `2016APV`, `2017`, `2018`, `UL16`, `UL16APV`, `UL17`, or `UL18` appear in `-y/--year`), the wrapper forwards the matching Run 2 payload to `run_analysis.py` via `--years`. Aliases are resolved so that `UL16` behaves like `2016`, `UL16APV` like `2016APV`, and similarly for `UL17`/`2017` and `UL18`/`2018`.
    - Add both `--do-np` and `--defer-np` when you want the wrapper to append `--do-np --np-postprocess=defer` to the delegated `run_analysis.py` command. The first flag enables the nonprompt producer, and the second switches it to deferred mode so the wrapper prints the metadata path (`histos/<outname>_np.pkl.gz.metadata.json`) and the follow-up helper has everything it needs. Passing only `--defer-np` leaves the producer disabled, so neither the metadata nor the `_np.pkl.gz` histogram will be created.
    - The wrapper inherits the same `topcoffea` data probe as `run_analysis.py`. If the command exits before queueing any jobs, re-run `scripts/install_topcoffea.sh` (or confirm that `external/topcoffea` tracks the branch advertised in the repository README) so the shared pileup payloads are restored. Expert setups can add `--skip-topcoffea-data-check` to the forwarded arguments, but keep the default enabled to avoid wasting Run 3 campaigns on misconfigured environments.
* `fullR2_run.sh`: Historical wrapper for the original TOP-22-006 pickle production. Keep it around for archival reproducibility; new workflows should prefer `fullR3_run.sh`.

* `run_data_driven.py`:
    - Finalizes deferred nonprompt/flips histograms using either the metadata emitted by `run_analysis.py --np-postprocess=defer` or manually specified pickle paths. See the dedicated usage notes below.

#### `run_data_driven.py` usage and recovery paths

- **Metadata-driven:** when `run_analysis.py` was run with `--np-postprocess=defer`, point the helper at the recorded sidecar to reconstruct the `_np.pkl.gz` output and (optionally) add the renorm/fact envelope:

  ```bash
  python run_data_driven.py --metadata-json histos/plotsTopEFT_np.pkl.gz.metadata.json \
      --apply-renormfact-envelope
  ```

- **Direct pickle path:** skip metadata entirely by forwarding the original histogram pickle and your desired destination explicitly:

  ```bash
  python run_data_driven.py --input-pkl histos/plotsTopEFT.pkl.gz \
      --output-pkl histos/plotsTopEFT_np.pkl.gz --apply-renormfact-envelope
  ```

  The helper streams `.pkl`/`.pkl.gz` inputs one histogram at a time, so even multi-GB dictionaries can be processed without holding everything in memory. Expect the `--input-pkl` file to be the base (pre-nonprompt) histograms and the `--output-pkl` path to receive the `_np.pkl.gz` variant ready for datacard production.

- **Troubleshooting missing metadata or moved pickles:** if the sidecar no longer matches your filesystem (for example, after relocating the histogram directory), re-run the helper with explicit `--input-pkl`/`--output-pkl` paths. You can also pass an absolute path to `--metadata-json` so relative entries resolve correctly when the metadata lives in a different folder than the pickle.

> **Sourcing helpers:** `run_plotter.sh`, `submit_plotter_condor.sh`, `fullR3_run.sh`, `fullR3_run_diboson.sh`, and `condor_plotter_entry.sh` now funnel their work through a `main()` function. They return non-zero statuses instead of exiting outright when validation fails, so sourcing them in an interactive shell will surface the error without tearing down your session. Executing the scripts directly still exits with the same return codes as before.


### Scripts for finding, comparing and plotting yields from histograms (from the processor)

* `make_cr_and_sr_plots.py`:
    - This script produces stacked yield and ratio plots for the configured analysis regions and can also drive dedicated comparison overlays.
    - The script takes as input a pkl file that should have both data and background MC included.
    - Example usage: `python make_cr_and_sr_plots.py -f histos/your.pkl.gz -o ~/www/some/dir -n some_dir_name -y 2017 2018 -t -u --variables lj0pt ptz`
    - Omitting `--variable/--variables` processes every histogram in the input pickle. Add a single histogram with `--variable name` or pass multiple tokens through `--variables name1 name2 ...` to focus the render on a shortlist.
    - `--year YEAR [YEAR ...]` filters both MC and data histograms to the selected campaign tokens before plotting. The resolver mirrors the datacard utilities, accepts the Run 2 (`run2` → `UL16 UL16APV UL17 UL18`) and Run 3 (`run3` → `2022 2022EE 2023 2023BPix`) aggregates, and prints a summary of the samples that were retained or vetoed alongside the traditional single-year tokens.
    - `--channel-output {merged,split,both,merged-njets,split-njets,both-njets}` selects how channel categories are rendered. `merged` integrates every category into the legacy combined templates and automatically drops split-only folders (for example the per-flavour CR variations) so the directory layout matches historical outputs, `split` preserves each individual channel when the input histograms are flavour-split and otherwise emits a warning while skipping the per-channel plots, and `both` renders the two sets back-to-back. When the inputs contain flavour-split channel labels, `both` always emits the merged category alongside every matching split directory. Append `-njets` to any mode to keep the per-njet bins defined in `cr_sr_plots_metadata.yml` instead of collapsing them into their aggregate parents. The default is `merged`.
      When requesting `both` or `both-njets`, expect two parallel directory trees: the merged view mirrors the split view's variable list even though the channel bins are aggregated (or grouped by jet multiplicity), so you can always find the full set of rendered histograms under both outputs.
    - `--workers N` enables multiprocessing when `N>1`. The plotter distributes the requested variables across worker processes and, when spare capacity remains, further fans out over `(variable, category)` pairs so SR-sized channel maps can render in parallel. Start with 2–4 workers; each process keeps a full copy of the histogram dictionary so memory usage still grows roughly linearly with `N`.
    - Pass `--log-y` to draw the stacked yields with a logarithmic y-axis (the ratio panel remains linear). The flag defaults to off so existing plots keep their linear scale unless explicitly requested, and is available both on the Python CLI and via `run_plotter.sh`.
    - Pass `--verbose` when you need detailed diagnostics (sample inventories, per-variable channel dumps). The default `--quiet` mode keeps the console output to high-level progress summaries.
    - Histograms with multiple dense axes (e.g. the `SparseHist`-based `lepton_pt_vs_eta`) are automatically rendered as CMS-style 2D heatmaps, while the 1D rebinning and systematic envelopes quietly skip them. The heatmap canvas now includes a dedicated Data/MC ratio panel so comparisons are available at a glance alongside the nominal MC and data projections.

### CR/SR plotting CLI quickstart

#### Outputs

Plots land under the directory you pass via `-o/--output-dir`. The plotter keeps things tidy by creating per-category subfolders when a histogram spans several channels, so the rendered figures stay grouped with their companions.

Each render currently emits the stat-only view plus the stat⊕syst variant when the inputs include systematics. Files are written using Matplotlib's default extension for the active backend (typically `.png`), so you can preview them with any standard image viewer.

Once a batch finishes, the CLI asks `make_html` to rebuild an `index.html` page in every output folder. Open those summaries directly from the filesystem with your web browser or serve the directory with `python -m http.server` if you prefer to share a link.

Re-running the same command rewrites matching figures and extends the same directory tree, making it easy to update plots without hunting for old outputs.

The `make_cr_and_sr_plots.py` entry point auto-detects whether the supplied pickle corresponds to control- or signal-region histograms by looking for `CR` or `SR` tokens in the filename. Detection is case-insensitive and accepts suffixes such as `SR2018`; it defaults to control-region mode when no clear token is present. If both tokens are found the script falls back to the control-region configuration and prints a warning recommending an explicit override.

Two new mutually exclusive switches, `--cr` and `--sr`, allow you to override the auto-detected mode. They are especially useful when the filename contains multiple year or campaign tags that would otherwise confuse the heuristic, or when a generic filename (e.g. `plotsTopEFT.pkl.gz`) is reused for multiple region exports.

Filtering the pickle to a subset of campaigns is now built into both the Python CLI and the wrapper. Pass the mandatory `-y/--year` flag with one or more tokens (e.g. `2017 2018 2022EE`) to restrict the MC and data samples before any plotting or yield aggregation. The script echoes a summary of the retained and vetoed samples so it is easy to verify the filter matched the intended years.

Run-aggregation shortcuts are available when you need the full campaigns: `run2` expands to `UL16 UL16APV UL17 UL18`, while `run3` expands to `2022 2022EE 2023 2023BPix`. Mix them freely with individual years—the CLI deduplicates the final list before the plots render and the legacy tokens remain available.

> **Note:** Omitting `-y/--year` now raises an error. Every invocation must include at least one campaign token so the plotter can perform the correct filtering.

Blinding is now governed by a single flag pair: `--unblind` always renders the data layer regardless of the region defaults, and `--blind` hides the data. When neither flag is provided the tool unblinds control-region plots and blinds signal-region plots, matching the standard analysis policy. The resolved region and blinding choice are echoed on start-up for clarity.

Long pickle sweeps can opt into multiprocessing with `--workers N`. When set above one the script fans the variable list out across a `ProcessPoolExecutor`, pre-creates the output directories, and aggregates the per-worker statistics before printing the summary counts. If idle slots remain, the work queue expands to `(variable, category)` pairs so that categories render in parallel. Each worker unpickles the histogram dictionary, so memory consumption increases roughly linearly with the worker count—start with a small value (e.g. `--workers 2` or `--workers 4` on machines with plenty of RAM) and scale up only if the host has headroom.

Add `--log-y` to either entry point when you need the stacked yields on a logarithmic scale. The plotter automatically rescales bins with zero or negative MC content so the log axis is well-defined while leaving the ratio panel on a linear scale for readability.

Console verbosity is now controlled by mutually exclusive `--verbose` and `--quiet` switches. Quiet mode remains the default and prints only high-level progress (region resolution, worker counts, summary statistics). Add `--verbose` to include the per-variable headings, sample inventories, and channel lists that previously flooded the terminal.

| Entry point | When to use |
| --- | --- |
| `python make_cr_and_sr_plots.py` | Direct access to every CLI flag for notebook or batch workflows. Remember to include `-y` with your desired years or aliases (e.g. `-y run2`). |
| [`./run_plotter.sh`](#run_plottersh-shell-wrapper-quickstart) | Convenience wrapper that mirrors the auto-detection logic and common flags, and accepts the same `-y` campaigns/aliases. |

Common invocation patterns (`-y/--year` now accepts multiple tokens for combined campaigns and must always be provided):

* Control-region scan with automatic blinding: `python make_cr_and_sr_plots.py -f histos/plotsCR_Run2.pkl.gz -y run2`
* Summing luminosities across multiple years: `python make_cr_and_sr_plots.py -f histos/plotsCR_Run2.pkl.gz -y 2016APV 2016 2017 2018`
* Signal-region pass where the filename already encodes `SR`: `python make_cr_and_sr_plots.py -f histos/SR2018.pkl.gz -o ~/www/sr -y 2018 --variable lj0pt --variable ptz`
* Overriding the heuristic and forcing a blinded SR workflow: `python make_cr_and_sr_plots.py -f histos/plotsTopEFT.pkl.gz -y run3 --sr --blind`
* Producing unblinded CR plots with explicit tagging and timestamped directories: `python make_cr_and_sr_plots.py -f histos/CR2018.pkl.gz -y 2018 --cr -t -n cr_2018_scan`
* Switching the stacked panel to a log scale: `python make_cr_and_sr_plots.py -f histos/plotsCR_Run2.pkl.gz -y run2 --log-y`

#### run_plotter.sh shell wrapper quickstart

The `run_plotter.sh` helper script lives alongside `make_cr_and_sr_plots.py` and reproduces the same filename-based auto-detection for control vs. signal regions. After resolving the region it appends the corresponding `--cr` or `--sr` flag before delegating to the Python CLI. When both `CR` and `SR` tokens appear in the filename the wrapper prints a warning and falls back to the control-region defaults unless you pass an explicit override.

Wrapper options match the Python interface so that README guidance applies verbatim. The required `-y/--year` flag shares the same individual years and `run2`/`run3` aggregates as the Python CLI (`run2` → `UL16 UL16APV UL17 UL18`, `run3` → `2022 2022EE 2023 2023BPix`), so you can reuse the shortcuts when hopping between Run 2 and Run 3 payloads. `--channel-output` forwards the merged/split/both selection along with the `*-njets` variants that preserve the per-njet bins from `cr_sr_plots_metadata.yml`, `--variable` adds a single histogram name per invocation while `--variables` continues to accept the whitespace-delimited list, and `--blind` / `--unblind` toggle data visibility after the wrapper has selected a region. You can still provide manual `--cr` or `--sr` overrides, and any other switches the wrapper does not understand are forwarded untouched to `make_cr_and_sr_plots.py`. The historical `--` passthrough marker remains accepted for backward compatibility but is no longer required.

The wrapper also exposes the new `--workers` flag; the argument is forwarded directly to the Python CLI, so the same variable/category fan-out and memory-usage caveats apply when you request more than one worker.

Use `-v/--verbose` with the wrapper when you need the Python CLI's detailed logging, or `--quiet` to enforce concise output explicitly.

Example commands:

* Auto-detected control-region plotting with timestamped outputs: `./run_plotter.sh -f histos/plotsCR_Run2.pkl.gz -o ~/www/cr_plots -y run2 --timestamp`
* Combining Run-3 campaigns in one call: `./run_plotter.sh -f histos/CR2022_combo.pkl.gz -o ~/www/cr_run3 -y run3`
* Enforcing a blinded SR pass with specific variables: `./run_plotter.sh -f histos/plotsTopEFT.pkl.gz -o ~/www/sr -n sr_scan -y run3 --sr --blind --variable lj0pt --variable ptz`
* Passing additional CLI flags through the wrapper: `./run_plotter.sh -f histos/SR2018.pkl.gz -o ~/www/sr_2018 -y 2018 --unblind --no-sumw2`
* Switching the stacked panel to a log scale via the wrapper: `./run_plotter.sh -f histos/plotsCR_Run2.pkl.gz -o ~/www/cr_plots -y run2 --log-y`

#### HTCondor plotting on Glados

##### Running on Glados HTCondor

`submit_plotter_condor.sh` builds a Condor submit description around `run_plotter.sh` so the same plotting CLI can run on Glados batch slots. The helper performs a `--dry-run` validation, stages a copy of `condor_plotter_entry.sh` next to the generated `.sub` file (spooled automatically with the job), and records the commands it will execute before handing everything to `condor_submit`.

**Prerequisites**

* A Glados login with valid UW–Madison Kerberos/AFS tokens (`kinit <netid>@AD.WISC.EDU` followed by `aklog`).
* A CephFS checkout of this repository that the worker nodes can reach. The helper defaults to `/users/apiccine/work/correction-lib/topeft`; override it with `--ceph-root /cephfs/<group>/<netid>/topeft` if your clone lives elsewhere. Make sure the path you provide is readable from the execute node—the flag should reference the worker-visible checkout rather than a login-only mount.
* An accessible Conda installation that contains the `clib-env` environment. Pass its prefix with `--conda-prefix /cephfs/<group>/<netid>/mambaforge/envs/clib-env`; the script discovers `conda.sh`, normalises the path, and activates `clib-env` inside the job. Make sure `etc/profile.d/conda.sh` is readable.
* Input histogram pickles, log directories, and optional sandboxes placed on shared storage (CephFS or AFS) with world-readable permissions so the execute node can fetch them.

**Example submission**

```bash
./submit_plotter_condor.sh \
  --ceph-root /cephfs/<group>/<netid>/topeft \
  --conda-prefix /cephfs/<group>/<netid>/mambaforge/envs/clib-env \
  --request-cpus 2 --request-memory 6GB \
  --log-dir /cephfs/<group>/<netid>/topeft/logs \
  -f /cephfs/<group>/<netid>/topeft/pickles/plotsCR_Run2.pkl.gz \
  -o /cephfs/<group>/<netid>/topeft/plots/run2_combo \
  -y run2 --variable lj0pt --variable ptz
```

Prefix the command with `--dry-run` when you want to review the generated job wrapper and `.sub` file without actually queueing the job. Adjust the batch resources with `--request-cpus`, `--request-memory`, or `--request-disk`, and add `--queue N` to launch an array of identical submissions. The optional `--sandbox /cephfs/.../templates` flag ships extra payload files alongside the job so the execute node can pick up custom style sheets or metadata.

`--request-cpus` requires a positive integer and `--request-memory` must be a non-empty HTCondor size string; the helper validates both before submitting so typos are caught locally during the dry-run step. The generated submit file exports `TOPEFT_REPO_ROOT` (the parent directory of `analysis/topeft_run2`) and `TOPEFT_ENTRY_DIR` (`analysis/topeft_run2` itself), mirroring the `${analysis_dir}/..` and `${analysis_dir}` values in the helper, so the entry script can override its working tree automatically; add `--conda-prefix ...` when you also need the helper to append `TOPEFT_CONDA_PREFIX` for environment activation. A literal `--` separator is still tolerated if you have scripts that emit it, but new invocations can omit it entirely.

**Entry-script environment steps**

Jobs land in `analysis/topeft_run2/condor_plotter_entry.sh`, which unsets `PYTHONPATH`, honours `TOPEFT_REPO_ROOT`/`TOPEFT_ENTRY_DIR` to pick the checkout and working directory, and activates `clib-env` via either the discovered Conda installation or an explicit `TOPEFT_CONDA_PREFIX`. Override those environment variables in the submit script when you need to point at a different checkout, wrapper directory, or Conda stack, or if you prefer to activate a bespoke environment before calling `run_plotter.sh`. The entry script shares the same `main()`-style return handling as the other helpers, so sourcing it during local smoke tests or unit checks surfaces failures without exiting your shell.

**Inspecting jobs and logs**

`submit_plotter_condor.sh` prints the Condor cluster ID on success. Use `condor_q <netid>` or `condor_q -af:j ClusterId ProcId JobStatus` to watch the queue; status codes follow the standard convention (1 = idle, 2 = running, 4 = completed). Each job writes `plotter.<cluster>.<proc>.{log,out,err}` into the `--log-dir` directory. The `.out` file streams the `condor_plotter_entry.sh` chatter—including the `unset PYTHONPATH` guard and `conda activate clib-env` activation—followed by the `run_plotter.sh` logs, so `tail -f` is the quickest way to monitor progress.

**Retrieving the plots**

Outputs appear directly under the `-o/--output-dir` you forwarded through the wrapper (e.g. `/afs/.../plots/run2_combo`). Condor populates the folder once the job finishes, so you can browse the rendered plots or host them with `python -m http.server` without additional copy steps.

* `get_yield_json.py`:
    - This script takes a pkl file produced by the processor, finds the yields in the analysis categories, and saves the yields to a json file. It can also print the info to the screen. The default pkl file to process is `hists/plotsTopEFT.pkl.gz`.
    - Example usage: `python get_yield_json.py -f histos/your_pkl_file.pkl.gz`

  * `comp_yields.py`:
      - This script takes two json files of yields (produced by `get_yield_json.py`), finds the difference and percent difference between them in each category, and prints out all of the information. You can also compare to the TOP-19-001 yields by specifying `TOP-19-001` as one of the inputs. Specifying the second file is optional, and it will default to the reference yield file. The script returns a non-zero exit code if any of the percent differences are larger than a given value (currently set to 1e-8).
      - Example usage: `python comp_yields.py your_yields_1.json your_yields_2.json`


### make_cr_and_sr_plots.py internals

Under the hood the CLI defers to a unified region runner so that both CR and SR workflows share the same plumbing. The `main()` function normalizes the CLI arguments, resolves the target region (auto-detected from the filename unless `--cr/--sr` is supplied), and prepares the output directory before handing control to `run_plots_for_region()`. That helper builds a `RegionContext` object via `build_region_context()`, which bundles together the histogram dictionary, lists of MC/data samples, per-region channel maps, and all style defaults. The context embeds the metadata derived from `topeft/params/cr_sr_plots_metadata.yml`, ensuring that channel definitions, grouping patterns, and region-specific overrides are all evaluated once and reused throughout the plotting loop.

`produce_region_plots()` then iterates over the requested histograms, applies the appropriate channel transformations, and orchestrates the per-category plotting. In aggregate (CR) mode the channel axis is integrated before rendering, while the SR configuration keeps each channel separate. During this sweep the code also:

* Removes samples that do not belong to the selected MC/data view and applies optional group-specific removals and category skips defined in the metadata.
* Fetches `sumw2` histograms for statistical uncertainties and combines them with shape/rate systematics where requested.
* Switches between raw 1D plotting and the dedicated 2D heatmap path when sparse histograms are encountered.

Because everything flows through the same `RegionContext`, adding a new region or adjusting behaviour in the YAML automatically updates both CR and SR plotting passes without touching the CLI.


### CR/SR metadata reference

The plotting behaviour is configured by `topeft/params/cr_sr_plots_metadata.yml`. The most commonly tuned blocks are:

* **Channel maps (`CR_CHAN_DICT` / `SR_CHAN_DICT`)** – map human-readable category labels to the underlying histogram channel bins. Add or remove entries here when categories are renamed or regrouped; the CLI enforces that every plotted channel appears in these lists.
* **Group patterns (`CR_GRP_MAP` / `SR_GRP_MAP`)** – define how raw process names are clustered into stacked contributions. Each group contains a color token and a list of substring patterns; new MC samples inherit the colour/styling of the group whose pattern matches their dataset name.
* **Region overrides (`REGION_PLOTTING`)** – per-region knobs that adjust plotting mechanics. Highlights include `channel_mode` (aggregate CR vs. per-channel SR figures), `channel_transformations` (string rewrites such as removing jet- or flavour-suffixes before matching), sample removal/category skip rules, and blinding-specific controls like `sumw2_remove_signal_when_blinded` and `use_mc_as_data_when_blinded`.

Other keys provide cohesive styling—e.g. `DATA_ERR_OPS`, `MC_ERROR_OPS`, `LUMI_COM_PAIRS`, and `WCPT_EXAMPLE`—and are consumed when building the `RegionContext`. Treat the YAML as the single source of truth for both category definitions and plot appearance to keep CR and SR outputs synchronized.

#### Styling quickstart: `STACKED_RATIO_STYLE`

If you want to tweak how the stacked-yield + ratio figures look, start with the `STACKED_RATIO_STYLE` block in `topeft/params/cr_sr_plots_metadata.yml`. The `defaults.figure` keys (such as `figsize`, `height_ratios`, and `hspace`) set the canvas geometry, so widening the plot is as simple as changing the first value in `figsize`. Axis cosmetics live under `defaults.axes`: adjust `label_fontsize`, `tick_labelsize`, or the `tick_length`/`tick_width` pair to make the layout friendlier for talks, and toggle `apply_secondary_ticks.x`/`.y` if you prefer primary ticks only. Legends are controlled by `defaults.legend` and its siblings—`ncol`, `fontsize`, and `bbox_to_anchor` reposition the main legend, while `uncertainty_legend` and `ratio_band_legend` handle the smaller annotation boxes. Once you have a layout you like, keep the edits inside the relevant nested dictionary so future readers can relate each number directly to the YAML keys mentioned here.

#### Understanding `analysis_bins`

The `analysis_bins` map inside `REGION_PLOTTING` (for example the `SR` block’s `analysis_bins` entry pointing to `ptz` and `lj0pt`) tells the plotter to replace the default histogram binning with the analysis-approved bin definitions in `axes_info`. Add a new key/value pair whenever you introduce a histogram that should adopt those curated bin edges—typically because it feeds a datacard or a physics note. If a variable can rely on the raw processor binning, leave it out; expanding the map is only necessary when the plotting output must match a named entry in the axes metadata. Remember to reuse the exact axis name from `cr_sr_plots_metadata.yml` so the lookup succeeds.


### Scripts for making and checking the datacards

All of the utilities in this section expect the nonprompt-enhanced histogram pickle (filename ending in `_np.pkl.gz`). Produce it inline via `run_analysis.py --do-np --np-postprocess=inline` or, when using the deferred workflow, call `python run_data_driven.py --metadata-json histos/<outname>_np.pkl.gz.metadata.json` before pointing the datacard maker at the pickle.

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

