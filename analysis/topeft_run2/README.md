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
  - [Plot-Time Variable Rebinning](#plot-time-variable-rebinning)
  - [Negative MC Contribution Reports](#negative-mc-contribution-reports)
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

For fake-tau SF extraction, including `tau0Fpt`/`tau0Tpt` input requirements and
the split-first/aggregate-fallback channel contract, see
[`README_faketau_sf_fitter.md`](README_faketau_sf_fitter.md).

* `run_topeft.py` for `topeft.py`:
    - This is the run script for the main `topeft.py` processor. Its usage is documented on the repository's main README. It uses either the `work_queue` or the `futures` executors (with `futures` it uses 8 cores by default). The `work_queue` executor makes use of remote resources, and you will need to submit workers using a `condor_submit_workers` command as explained on the main `topcoffea` README. You can configure the run with a number of command line arguments, but the most important one is the config file, where you list the samples you would like to process (by pointing to the JSON files for each sample, located inside of `topcoffea/json`.
    - Example usage: `python run_topeft.py ../../topcoffea/cfg/your_cfg.cfg`

* `run_analysis.py`:
    - Thin wrapper around `analysis_processor.py` used for the standard CR/analysis histogram production. The canned histogram lists now include the 2D `lepton_pt_vs_eta` observable (and keep the matching `_sumw2` companion unless `--no-sumw2` is passed) so downstream tools can rely on a consistent pt vs $|\eta|$ binning description.
    - Leave the default `sumw²` companions enabled whenever you plan to run downstream uncertainty-aware tooling such as the tau fake-rate fitter or the diboson scale-factor extractor. Disabling them with `--no-sumw2` drops the `*_sumw2` histograms (for example `tau0pt_sumw2`), which causes those utilities to fail or to lose their statistical error propagation. If you need to trim the histogram list, remove individual observables instead of the sumw² accumulators.
    - Pass `--years YEAR [YEAR ...]` to filter the loaded JSON samples to the requested campaign tokens. Supported values are `2016`, `2016APV`, `2017`, `2018`, `2022`, `2022EE`, `2023`, `2023BPix`, their UL aliases (`UL16`, `UL16APV`, `UL17`, `UL18`), and the aggregate shorthands `run2` (`UL16 UL16APV UL17 UL18`) and `run3` (`2022 2022EE 2023 2023BPix`). Legacy tokens remain valid, so existing command snippets do not require changes. When the option is absent every sample in the configuration is retained as before.
    - Pass `--category-groups GROUP [GROUP ...]` to narrow the resolved `ch_lst.json` SR/CR block selection to one or more named groups before `analysis_processor.py` starts any downstream work. The names are validated in `run_analysis.py` against the active block(s) chosen by the analysis-mode flags, so `ch_lst.json` remains the only source of truth. Omitting the option preserves the historical behavior and keeps every group in each active block. Example: `python analysis/topeft_run2/run_analysis.py ... --category-groups 3l_fwd 4l`. When both SR and CR are active, a requested group may exist in only one resolved block; that region is filtered normally and the other region may intentionally become empty (for example `--category-groups 4l` keeps the default SR `4l` group while the default CR block resolves to no selected groups).
    - `sumw2_storage.mode` accepts six values: `production`, `production_central`, `taufitter`, `full_diagnostics`, `disabled`, and `full_custom`. Omission still selects `production`. The two production modes use the same selective allocation rules but certify different signal variants: use `production` with the maintained private-signal cfg bundle and `production_central` with a central-signal cfg bundle. The cfg remains authoritative and is never rewritten by the mode. A paired private/central mismatch or overlap is rejected before `AnalysisProcessor` construction. For example:

      ```yaml
      sumw2_storage:
        mode: production_central
        rules:
          - process_prefixes: [data, TTTo, tZq_central]
      ```

      Switching only the cfg or only the mode is intentionally an error. Unpaired signals and signal groups absent from both variants remain governed by the active cfg and do not become false requirements.
    - The data-driven helper supports inline and deferred workflows. Keep the historical behaviour by relying on the default `--np-postprocess=inline` (paired with `--do-np`) so the `_np.pkl.gz` file appears immediately. Choose `--np-postprocess=defer` **together with `--do-np`** to emit the base pickle and print the direct `run_data_driven.py --input-pkl ... --output-pkl ...` follow-up command. Setting `--np-postprocess=skip` suppresses the data-driven step entirely.
    - Startup now includes a quick sanity check that resolves `data/pileup/pileup_2016GH.root` via `topcoffea_path` and ensures the file exists. When it fails the CLI exits with instructions to re-run `scripts/install_topcoffea.sh`, verify the `external/topcoffea` checkout (currently `run3_test_mmerged`) is available, and try again. Use `--skip-topcoffea-data-check` only when you intentionally manage the shared pileup files outside of the helper script.
    - Deferred follow-up uses the printed direct command: `python run_data_driven.py --input-pkl histos/<outname>.pkl.gz --output-pkl histos/<outname>_np.pkl.gz`. The helper still defaults to the streaming iterator path. The maintained renorm/fact workflow retains the independent `renormUp`, `renormDown`, `factUp`, and `factDown` templates as separate `renorm` and `fact` nuisances; the historical combined envelope is unsupported.

* `run_sow.py` for `sow_processor.py`:
    - This script runs over the provided json files and calculates the properer sum of weights
    - Example usage: `python run_sow.py ../../topcoffea/json/signal_samples/private_UL/UL17_tHq_b1.json --xrd root://deepthought.crc.nd.edu/`

* `fullR3_run.sh`: Recommended wrapper script for both Run 2 and Run 3 histogram production. It expands the aggregate campaign aliases (`run2` → `UL16 UL16APV UL17 UL18`, `run3` → `2022 2022EE 2023 2023BPix`) before dispatching to `run_analysis.py`, superseding the legacy helper while keeping the historical single-year tokens functioning as before.
    - Whenever the Run 2 bundle is activated (any of `2016`, `2016APV`, `2017`, `2018`, `UL16`, `UL16APV`, `UL17`, or `UL18` appear in `-y/--year`), the wrapper forwards the matching Run 2 payload to `run_analysis.py` via `--years`. Aliases are resolved so that `UL16` behaves like `2016`, `UL16APV` like `2016APV`, and similarly for `UL17`/`2017` and `UL18`/`2018`.
    - Add both `--do-np` and `--defer-np` when you want the wrapper to append `--do-np --np-postprocess=defer` to the delegated `run_analysis.py` command. The first flag enables the nonprompt producer, and the second switches it to deferred mode so the wrapper prints the direct `run_data_driven.py --input-pkl ... --output-pkl ...` follow-up command. Passing only `--defer-np` leaves the producer disabled, so no `_np.pkl.gz` histogram will be created.
    - The wrapper inherits the same `topcoffea` data probe as `run_analysis.py`. If the command exits before queueing any jobs, re-run `scripts/install_topcoffea.sh` (or confirm that `external/topcoffea` tracks the branch advertised in the repository README) so the shared pileup payloads are restored. Expert setups can add `--skip-topcoffea-data-check` to the forwarded arguments, but keep the default enabled to avoid wasting Run 3 campaigns on misconfigured environments.
* `fullR2_run.sh`: Historical wrapper for the original TOP-22-006 pickle production. Keep it around for archival reproducibility; new workflows should prefer `fullR3_run.sh`.

* `run_data_driven.py`:
    - Finalizes deferred nonprompt/flips histograms using either the metadata emitted by `run_analysis.py --np-postprocess=defer` or manually specified pickle paths. See the dedicated usage notes below.

#### `run_data_driven.py` usage and recovery paths

- **Deferred direct pickle path:** when `run_analysis.py` was run with `--np-postprocess=defer`, use the printed follow-up command (or forward the original histogram pickle and your desired destination explicitly):

  ```bash
  python run_data_driven.py --input-pkl histos/plotsTopEFT.pkl.gz \
      --output-pkl histos/plotsTopEFT_np.pkl.gz
  ```

  The helper streams `.pkl`/`.pkl.gz` inputs one histogram at a time, so even multi-GB dictionaries can be processed without holding everything in memory. Expect the `--input-pkl` file to be the base (pre-nonprompt) histograms and the `--output-pkl` path to receive the `_np.pkl.gz` variant ready for datacard production.

- **Default vs legacy mode:** the helper now defaults to the streaming iterator path, including when launched from metadata. To restore the original materialized-dict behavior, pass `--legacy-dict-mode`.

- **Hardcoded streaming writer settings:** iterator/default mode writes through `dump_dict_streaming` with hardcoded `protocol=3` and `clear_memo_interval=1`. This is intentional for bounded-memory safety; protocols `>=4` are not currently used in this memo-clearing streaming path.

- **Troubleshooting moved pickles:** if an input or output pickle has moved, re-run the helper with explicit `--input-pkl`/`--output-pkl` paths. The deferred workflow has no metadata JSON transport.

> **Sourcing helpers:** `run_plotter.sh`, `submit_plotter_condor.sh`, `fullR3_run.sh`, `fullR3_run_diboson.sh`, and `condor_plotter_entry.sh` now funnel their work through a `main()` function. They return non-zero statuses instead of exiting outright when validation fails, so sourcing them in an interactive shell will surface the error without tearing down your session. Executing the scripts directly still exits with the same return codes as before.


### Scripts for finding, comparing and plotting yields from histograms (from the processor)

* `make_cr_and_sr_plots.py`:
    - This script produces stacked yield and ratio plots for the configured analysis regions and can also drive dedicated comparison overlays.
    - The script takes as input a pkl file that should have both data and background MC included.
    - Example usage:

      ```bash
      PYTHON_ENV="/users/apiccine/work/miniconda3/envs/clib-env/bin/python"
      $PYTHON_ENV analysis/topeft_run2/make_cr_and_sr_plots.py \
          -f histos/your.pkl.gz \
          -o ~/www/some/dir \
          -n some_dir_name \
          -y 2017 2018 \
          -t -u \
          --variables lj0pt ptz
      ```

    - Omitting `--variables` processes every histogram in the input pickle. Use `--variables name1 name2 ...` to focus the render on a shortlist. (`--variable` is only supported by `run_plotter.sh`, which forwards it as `--variables`.)
    - `--year YEAR [YEAR ...]` filters both MC and data histograms to the selected campaign tokens before plotting. The resolver mirrors the datacard utilities and accepts the Run 2 (`run2` → `UL16 UL16APV UL17 UL18`) and Run 3 (`run3` → `2022 2022EE 2023 2023BPix`) aggregates. If omitted, no filtering is applied; use `--verbose` to inspect the resulting sample lists.
- `--channel-output {merged,split,both,merged-njets,split-njets,both-njets}` selects how channel categories are rendered. `merged` integrates every category into the legacy combined templates and automatically drops split-only folders (for example the per-flavour CR variations) so the directory layout matches historical outputs, `split` preserves each individual channel when the input histograms are flavour-split and otherwise emits a warning while skipping the per-channel plots, and `both` renders the two sets back-to-back. When the inputs contain flavour-split channel labels, `both` always emits the merged category alongside every matching split directory (including the `both-njets` variant). Append `-njets` to any mode to keep the per-njet bins defined in `cr_sr_plots_metadata.yml` instead of collapsing them into their aggregate parents. The default is `merged`.
      When requesting `both` or `both-njets`, expect two parallel directory trees: the merged view mirrors the split view's variable list even though the channel bins are aggregated (or grouped by jet multiplicity), so you can always find the full set of rendered histograms under both outputs.
    - `--workers N` enables multiprocessing when `N>1`. The plotter distributes the requested variables across worker processes and, when spare capacity remains, further fans out over `(variable, category)` pairs so SR-sized channel maps can render in parallel. Start with 2–4 workers; each process keeps a full copy of the histogram dictionary so memory usage still grows roughly linearly with `N`.
    - Pass `--log-y` to draw the stacked yields with a logarithmic y-axis (the ratio panel remains linear). The flag defaults to off so existing plots keep their linear scale unless explicitly requested, and is available both on the Python CLI and via `run_plotter.sh`.
    - Pass `--verbose` when you need detailed diagnostics (sample inventories, per-variable channel dumps). The default `--quiet` mode keeps the console output to high-level progress summaries.
    - `--report-zero-yields` emits a detailed summary of processes with zero or missing yields after plotting.
    - `--rebin-plot-vars j0pt:2,l1conept=2` rebins only the listed variables at plot/report time. The input pickle is not modified.
    - The negative MC contribution report is enabled by default and writes `negative_weight_contribution_report.csv` plus `negative_weight_contribution_summary.md` under the plot output directory. Add `--no-negative-weight-report` to suppress those files.
    - Histograms with multiple dense axes (e.g. the `SparseHist`-based `lepton_pt_vs_eta`) are automatically rendered as CMS-style 2D heatmaps, while the 1D rebinning and systematic envelopes quietly skip them. The heatmap canvas now includes a dedicated Data/MC ratio panel so comparisons are available at a glance alongside the nominal MC and data projections.

### CR/SR plotting CLI quickstart

#### Outputs

Plots land under `<output-path>/<output-name>` for the Python CLI (`-o/--output-path` plus `-n/--output-name`). The wrapper uses `-o/--output-dir` and `-n/--name` but lands in the same `<output-dir>/<name>` layout. The plotter keeps things tidy by creating per-category subfolders when a histogram spans several channels, so the rendered figures stay grouped with their companions.

Each render currently emits the stat-only view plus the stat⊕syst variant when the inputs include systematics. Files are written using Matplotlib's default extension for the active backend (typically `.png`), so you can preview them with any standard image viewer.

Once a batch finishes, the CLI refreshes HTML indices only for output folders whose path includes `www` (via `topcoffea.scripts.make_html`). Open those summaries directly from the filesystem with your web browser or serve the directory with `python -m http.server` if you prefer to share a link.

Re-running the same command rewrites matching figures and extends the same directory tree, making it easy to update plots without hunting for old outputs.

The `make_cr_and_sr_plots.py` entry point auto-detects whether the supplied pickle corresponds to control- or signal-region histograms by looking for `CR` or `SR` tokens in the filename. Detection is case-insensitive and accepts suffixes such as `SR2018`; it defaults to control-region mode when no clear token is present. If both tokens are found the script falls back to the control-region configuration and prints a warning recommending an explicit override.

Two new mutually exclusive switches, `--cr` and `--sr`, allow you to override the auto-detected mode. They are especially useful when the filename contains multiple year or campaign tags that would otherwise confuse the heuristic, or when a generic filename (e.g. `plotsTopEFT.pkl.gz`) is reused for multiple region exports.

Filtering the pickle to a subset of campaigns is supported by both entry points. Pass `-y/--year` with one or more tokens (e.g. `2017 2018 2022EE`) to restrict the MC and data samples before any plotting or yield aggregation. `run_plotter.sh` requires `-y`, while the Python CLI treats it as optional; use `--verbose` to inspect the resulting sample lists.

Run-aggregation shortcuts are available when you need the full campaigns: `run2` expands to `UL16 UL16APV UL17 UL18`, while `run3` expands to `2022 2022EE 2023 2023BPix`. Mix them freely with individual years—the CLI deduplicates the final list before the plots render and the legacy tokens remain available.

> **Note:** Omitting `-y/--year` only raises an error in `run_plotter.sh`. The Python CLI keeps all samples when `-y` is absent.

Blinding is now governed by a single flag pair: `--unblind` always renders the data layer regardless of the region defaults, and `--blind` hides the data. When neither flag is provided the tool unblinds control-region plots and blinds signal-region plots, matching the standard analysis policy. The resolved region and blinding choice are echoed on start-up for clarity.

Long pickle sweeps can opt into multiprocessing with `--workers N`. When set above one the script fans the variable list out across a `ProcessPoolExecutor`, pre-creates the output directories, and aggregates the per-worker statistics before printing the summary counts. If idle slots remain, the work queue expands to `(variable, category)` pairs so that categories render in parallel. Each worker unpickles the histogram dictionary, so memory consumption increases roughly linearly with the worker count—start with a small value (e.g. `--workers 2` or `--workers 4` on machines with plenty of RAM) and scale up only if the host has headroom.

Add `--log-y` to either entry point when you need the stacked yields on a logarithmic scale. The plotter automatically rescales bins with zero or negative MC content so the log axis is well-defined while leaving the ratio panel on a linear scale for readability.

Console verbosity is now controlled by mutually exclusive `--verbose` and `--quiet` switches. Quiet mode remains the default and prints only high-level progress (region resolution, worker counts, summary statistics). Add `--verbose` to include the per-variable headings, sample inventories, and channel lists that previously flooded the terminal.

Every histogram variable available in the pickle is plotted for the selected channel-output mode. By default the plotter runs with `--channel-output merged`; use `split`, `both`, or the `*-njets` variants to include per-channel outputs. The plotter now ignores the YAML `skip_variables`, `skip_sparse_2d`, and `category_skips` lists unless you opt in with `--enable-category-skips`, keeping the default runs aligned with the full histogram payload.

| Entry point | When to use |
| --- | --- |
| `python make_cr_and_sr_plots.py` | Direct access to every CLI flag for notebook or batch workflows. Use `-y` to filter by years/aliases (optional but recommended). |
| [`./run_plotter.sh`](#run_plottersh-shell-wrapper-quickstart) | Convenience wrapper that mirrors the auto-detection logic and common flags, and requires `-y` campaigns/aliases. |

Common invocation patterns (`-y/--year` accepts multiple tokens for combined campaigns; it is required only for `run_plotter.sh`):

* Control-region scan with automatic blinding: `python make_cr_and_sr_plots.py -f histos/plotsCR_Run2.pkl.gz -y run2`
* Summing luminosities across multiple years: `python make_cr_and_sr_plots.py -f histos/plotsCR_Run2.pkl.gz -y 2016APV 2016 2017 2018`
* Signal-region pass where the filename already encodes `SR`: `python make_cr_and_sr_plots.py -f histos/SR2018.pkl.gz -o ~/www/sr -y 2018 --variables lj0pt ptz`
* Overriding the heuristic and forcing a blinded SR workflow: `python make_cr_and_sr_plots.py -f histos/plotsTopEFT.pkl.gz -y run3 --sr --blind`
* Producing unblinded CR plots with explicit tagging and timestamped directories: `python make_cr_and_sr_plots.py -f histos/CR2018.pkl.gz -y 2018 --cr -t -n cr_2018_scan`
* Switching the stacked panel to a log scale: `python make_cr_and_sr_plots.py -f histos/plotsCR_Run2.pkl.gz -y run2 --log-y`

#### Plot-Time Variable Rebinning

Use `--rebin-plot-vars` when a plot needs coarser visible bins but the input pickle should remain untouched. The option takes comma-separated `variable:factor` or `variable=factor` entries, and each factor must be an integer greater than or equal to 2:

```bash
python make_cr_and_sr_plots.py \
  -f histos/plotsCR_Run2.pkl.gz \
  -y 2017 \
  --variables j0pt l1conept \
  --rebin-plot-vars j0pt:2,l1conept=2
```

Rebinning happens only inside the plotting/reporting path. The source histograms and input pickle are not rewritten. Visible-bin contents are summed, variances and `sumw2` values are summed, and any leftover visible bins that do not fill a complete factor group are merged into the final rebinned bin rather than being dropped.

The rebinned edges are used for the stacked plot, ratio inputs, statistical and systematic uncertainty-band inputs, and the `post_rebin` rows in the negative MC contribution report. Variables not listed in `--rebin-plot-vars` keep their normal binning and, when applicable, use the configured `analysis_bins` edges from `cr_sr_plots_metadata.yml`. Rebinning is most useful together with `--variables` so focused checks do not render the entire pickle.

#### Negative MC Contribution Reports

The plotter writes negative MC contribution diagnostics by default. Disable them with `--no-negative-weight-report` if you only want figures. The output files are written under the selected plot output directory:

```text
negative_weight_contribution_report.csv
negative_weight_contribution_summary.md
```

The CSV contains one row per negative bin contribution at two levels:

* `level = process` identifies the raw process that contributed the negative bin.
* `level = group` identifies the stacked plot group affected by the negative bin.

The `stage` column records which binning was inspected:

* `nominal_no_rebin` is used when no plot-time rebinning applies to that variable.
* `pre_rebin` records the original visible bins when plot-time rebinning is requested.
* `post_rebin` records the merged bins after plot-time rebinning.

Important columns include `yield`, `sumw2`, `total_mc_yield`, `data_yield`, `yield_over_total_mc`, and `abs_yield_over_total_mc`. When `sumw2 >= 0`, `error = sqrt(sumw2)`. When `sumw2 > 0`, `effective_entries = yield^2 / sumw2`. The boolean flags use:

* `is_compatible_with_zero_1sigma`: `abs(yield) <= error`;
* `is_single_effective_entry_like`: `effective_entries <= 1.05`;
* `is_low_effective_entries`: `effective_entries <= 5`.

These reports are diagnostic aids, not physics corrections. Single-effective-entry-like rows often point to one or very few high-weight signed events. Rows compatible with zero at one sigma can be ordinary statistical fluctuations. Use the process and group rows to decide whether to rebin, annotate, or investigate a sample.

`--report-zero-yields` is separate. It reports processes with zero or missing yields after plotting: it answers "what is absent or zero?" The negative contribution report answers "which signed MC components drive negative bins?"

#### Focused CR Rebinning Example

For a focused CR check of the known narrow-bin variables, run only the variables under study and request plot-time rebinning explicitly:

```bash
YR=2017
PKL=/path/to/plotsCR_${YR}.pkl.gz

python make_cr_and_sr_plots.py \
  -f "$PKL" \
  -n CR_preappr_rebin_check_${YR} \
  -o ../../histos/ \
  -y "$YR" \
  --verbose \
  --cr \
  --workers 1 \
  --channel-output both \
  --variables j0pt l1conept \
  --rebin-plot-vars j0pt:2,l1conept:2
```

The figures and negative-weight report files land under `../../histos/CR_preappr_rebin_check_${YR}`. The command does not modify `$PKL`.

#### run_plotter.sh shell wrapper quickstart

The `run_plotter.sh` helper script lives alongside `make_cr_and_sr_plots.py` and reproduces the same filename-based auto-detection for control vs. signal regions. After resolving the region it appends the corresponding `--cr` or `--sr` flag before delegating to the Python CLI. When both `CR` and `SR` tokens appear in the filename the wrapper prints a warning and falls back to the control-region defaults unless you pass an explicit override.

Wrapper options mirror the Python interface with a few naming differences: `run_plotter.sh` uses `--input`/`--output-dir`/`--name` (vs `--pkl-file-path`/`--output-path`/`--output-name` on the Python CLI), requires `-y/--year`, and accepts `--variable` as a shorthand that it forwards as `--variables`. The required `-y/--year` flag shares the same individual years and `run2`/`run3` aggregates as the Python CLI (`run2` → `UL16 UL16APV UL17 UL18`, `run3` → `2022 2022EE 2023 2023BPix`), so you can reuse the shortcuts when hopping between Run 2 and Run 3 payloads. `--channel-output` forwards the merged/split/both selection along with the `*-njets` variants that preserve the per-njet bins from `cr_sr_plots_metadata.yml`, and `--blind` / `--unblind` toggle data visibility after the wrapper has selected a region. The wrapper also forwards `--rebin-plot-vars` and `--no-negative-weight-report`; any other switches the wrapper does not understand are passed untouched to `make_cr_and_sr_plots.py`. The historical `--` passthrough marker remains accepted for backward compatibility but is no longer required.

If you need to control the Python interpreter, export `PYTHON_BIN="$PYTHON_ENV"` (or `PYTHON="$PYTHON_ENV"`) before calling the wrapper.

The wrapper also exposes the new `--workers` flag; the argument is forwarded directly to the Python CLI, so the same variable/category fan-out and memory-usage caveats apply when you request more than one worker.

Use `-v/--verbose` with the wrapper when you need the Python CLI's detailed logging, or `--quiet` to enforce concise output explicitly.

Example commands:

* Auto-detected control-region plotting with timestamped outputs: `./run_plotter.sh -f histos/plotsCR_Run2.pkl.gz -o ~/www/cr_plots -y run2 --timestamp`
* Combining Run-3 campaigns in one call: `./run_plotter.sh -f histos/CR2022_combo.pkl.gz -o ~/www/cr_run3 -y run3`
* Enforcing a blinded SR pass with specific variables: `./run_plotter.sh -f histos/plotsTopEFT.pkl.gz -o ~/www/sr -n sr_scan -y run3 --sr --blind --variable lj0pt --variable ptz`
* Passing additional CLI flags through the wrapper: `./run_plotter.sh -f histos/SR2018.pkl.gz -o ~/www/sr_2018 -y 2018 --unblind --skip-syst`
* Switching the stacked panel to a log scale via the wrapper: `./run_plotter.sh -f histos/plotsCR_Run2.pkl.gz -o ~/www/cr_plots -y run2 --log-y`
* Focused rebinning with the default negative report: `./run_plotter.sh -f histos/plotsCR_2017.pkl.gz -o ~/www/cr_plots -y 2017 --cr --variables j0pt l1conept --rebin-plot-vars j0pt:2,l1conept:2`
* Suppressing the negative report when only figures are needed: `./run_plotter.sh -f histos/plotsCR_Run2.pkl.gz -o ~/www/cr_plots -y run2 --no-negative-weight-report`

#### HTCondor plotting on Glados

##### Running on Glados HTCondor

`submit_plotter_condor.sh` builds a Condor submit description around `run_plotter.sh` so the same plotting CLI can run on Glados batch slots. The helper performs a `--dry-run` validation, writes a submit file that executes `analysis/topeft_run2/condor_plotter_entry.sh` directly from the shared checkout (`should_transfer_files = NO`), and records the commands it will execute before handing everything to `condor_submit`.

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

Prefix the command with `--dry-run` when you want to review the generated job wrapper and `.sub` file without actually queueing the job. Adjust the batch resources with `--request-cpus` and `--request-memory`, and add `--queue N` to launch an array of identical submissions. The optional `--sandbox /cephfs/.../templates` flag ships extra payload files alongside the job so the execute node can pick up custom style sheets or metadata. Use `--condor-ulimit` if you want the entry script to apply the same ulimit safeguards outside of Condor.

`--request-cpus` requires a positive integer and `--request-memory` must be a non-empty HTCondor size string; the helper validates both before submitting so typos are caught locally during the dry-run step. The generated submit file exports `TOPEFT_REPO_ROOT` (the parent directory of `analysis/topeft_run2`) and `TOPEFT_ENTRY_DIR` (`analysis/topeft_run2` itself) to mirror the `initialdir` specified in the submit description, so the entry script can derive its working directory deterministically; add `--conda-prefix ...` when you also need the helper to append `TOPEFT_CONDA_PREFIX` for environment activation. A literal `--` separator is still tolerated if you have scripts that emit it, but new invocations can omit it entirely.

Plotter options such as `--variables`, `--rebin-plot-vars`, and `--no-negative-weight-report` are forwarded through `submit_plotter_condor.sh` to `run_plotter.sh`. For example, keep Condor options before the delimiter and plotting options after it when you want to make the split explicit:

```bash
./submit_plotter_condor.sh \
  --request-cpus 2 --request-memory 6GB \
  -- \
  -f /cephfs/<group>/<netid>/topeft/pickles/plotsCR_2017.pkl.gz \
  -o /cephfs/<group>/<netid>/topeft/plots/cr_2017_rebin \
  -y 2017 --cr \
  --variables j0pt l1conept \
  --rebin-plot-vars j0pt:2,l1conept:2
```

**Entry-script environment steps**

Jobs land in `analysis/topeft_run2/condor_plotter_entry.sh`, which unsets `PYTHONPATH`, resolves its working directory from `TOPEFT_ENTRY_DIR`/Condor's `initialdir`/the script path, logs the choice, and activates `clib-env` via either the discovered Conda installation or an explicit `TOPEFT_CONDA_PREFIX`. Override those environment variables in the submit script when you need to point at a different checkout, wrapper directory, or Conda stack, or if you prefer to activate a bespoke environment before calling `run_plotter.sh`. The entry script shares the same `main()`-style return handling as the other helpers, so sourcing it during local smoke tests or unit checks surfaces failures without exiting your shell.

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

* Removes samples that do not belong to the selected MC/data view and applies optional group-specific removals. Category skip rules in the metadata are ignored unless you explicitly add `--enable-category-skips`; sample/group removals and category skip rules are separate controls.
* Fetches `sumw2` histograms for statistical uncertainties and combines them with shape/rate systematics where requested.
* Switches between raw 1D plotting and the dedicated 2D heatmap path when sparse histograms are encountered.
* Applies requested plot-time rebinning and collects negative MC contribution rows without mutating the input histograms.

Because everything flows through the same `RegionContext`, adding a new region or adjusting behaviour in the YAML automatically updates both CR and SR plotting passes without touching the CLI.

#### Sumw2 Companion Histograms

The plotter uses `sumw2` companion histograms for statistical uncertainty bands and for the negative-report `error` and `effective_entries` diagnostics. Nominal histograms normally expose a dense axis named after the variable, such as `j0pt`. Companion `sumw2` histograms may use that same dense-axis name or a companion name such as `j0pt_sumw2`. If neither name exists and the histogram has exactly one unambiguous dense numeric axis, the plotter can use that axis. Missing or ambiguous dense axes raise a clear error so uncertainty bands and effective-entry calculations do not silently use the wrong binning.

This naming flexibility matters most after plot-time rebinning: nominal contents, data contents, and summed `sumw2` values must be merged with the same target edges even when the companion histogram uses a `_sumw2` dense-axis name.

#### Missing-Process-Tolerant Shape Systematics

Shape-systematic group maps can name processes that are absent from a currently plotted histogram because of metadata skips, sample removals, year filtering, or category-specific process availability. During shape-systematic evaluation, absent process labels are ignored for the affected histogram/group. If a group has no available processes after intersecting with the histogram process axis, it contributes zero/absent uncertainty for that systematic rather than causing a global shape-systematic failure.

This keeps ordinary skipped or absent processes from disabling unrelated shape systematics such as `renorm` or `fact`. It does not hide genuine problems: missing systematic labels, malformed axes, or failures unrelated to process-axis availability can still warn or fail normally.


### CR/SR metadata reference

The plotting behaviour is configured by `topeft/params/cr_sr_plots_metadata.yml`. The most commonly tuned blocks are:

* **Channel maps (`CR_CHAN_DICT` / `SR_CHAN_DICT`)** – map human-readable category labels to the underlying histogram channel bins. Add or remove entries here when categories are renamed or regrouped; the CLI enforces that every plotted channel appears in these lists.
* **Group patterns (`CR_GRP_MAP` / `SR_GRP_MAP`)** – define how raw process names are clustered into stacked contributions. Each group contains a color token and a list of substring patterns; new MC samples inherit the colour/styling of the group whose pattern matches their dataset name.
* **Region overrides (`REGION_PLOTTING`)** – per-region knobs that adjust plotting mechanics. Highlights include `channel_mode` (aggregate CR vs. per-channel SR figures), `channel_transformations` (string rewrites such as removing jet- or flavour-suffixes before matching), sample removal rules (and opt-in category skip rules), and blinding-specific controls like `sumw2_remove_signal_when_blinded` and `use_mc_as_data_when_blinded`.

Other keys provide cohesive styling—e.g. `DATA_ERR_OPS`, `MC_ERROR_OPS`, `LUMI_COM_PAIRS`, and `WCPT_EXAMPLE`—and are consumed when building the `RegionContext`. Treat the YAML as the single source of truth for both category definitions and plot appearance to keep CR and SR outputs synchronized.

#### Styling quickstart: `STACKED_RATIO_STYLE`

If you want to tweak how the stacked-yield + ratio figures look, start with the `STACKED_RATIO_STYLE` block in `topeft/params/cr_sr_plots_metadata.yml`. The `defaults.figure` keys (such as `figsize`, `height_ratios`, and `hspace`) set the canvas geometry, so widening the plot is as simple as changing the first value in `figsize`. Axis cosmetics live under `defaults.axes`: adjust `label_fontsize`, `tick_labelsize`, or the `tick_length`/`tick_width` pair to make the layout friendlier for talks, and toggle `apply_secondary_ticks.x`/`.y` if you prefer primary ticks only. Legends are controlled by `defaults.legend` and its siblings—`ncol`, `fontsize`, and `bbox_to_anchor` reposition the main legend, while `uncertainty_legend` and `ratio_band_legend` handle the smaller annotation boxes. Once you have a layout you like, keep the edits inside the relevant nested dictionary so future readers can relate each number directly to the YAML keys mentioned here.

#### Understanding `analysis_bins`

The `analysis_bins` map inside `REGION_PLOTTING` (for example the `SR` block’s `analysis_bins` entry pointing to `ptz` and `lj0pt`) tells the plotter to replace the default histogram binning with the analysis-approved bin definitions in `axes_info`. Add a new key/value pair whenever you introduce a histogram that should adopt those curated bin edges—typically because it feeds a datacard or a physics note. If a variable can rely on the raw processor binning, leave it out; expanding the map is only necessary when the plotting output must match a named entry in the axes metadata. Remember to reuse the exact axis name from `cr_sr_plots_metadata.yml` so the lookup succeeds.


### Scripts for making and checking the datacards

All of the utilities in this section expect the nonprompt-enhanced histogram pickle (filename ending in `_np.pkl.gz`). Produce it inline via `run_analysis.py --do-np --np-postprocess=inline` or, when using the deferred workflow, run the printed `run_data_driven.py --input-pkl ... --output-pkl ...` command before pointing the datacard maker at the pickle.

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
