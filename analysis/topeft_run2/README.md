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

* `run_sow.py` for `sow_processor.py`:
    - This script runs over the provided json files and calculates the properer sum of weights
    - Example usage: `python run_sow.py ../../topcoffea/json/signal_samples/private_UL/UL17_tHq_b1.json --xrd root://deepthought.crc.nd.edu/`

* `fullR2_run.sh`: Wrapper script for making the full TOP-22-006 pkl file with `run_topeft.py`. 


### Scripts for finding, comparing and plotting yields from histograms (from the processor)

* `make_cr_and_sr_plots.py`:
    - This script produces stacked yield and ratio plots for the configured analysis regions and can also drive dedicated comparison overlays.
    - The script takes as input a pkl file that should have both data and background MC included.
    - Example usage: `python make_cr_and_sr_plots.py -f histos/your.pkl.gz -o ~/www/some/dir -n some_dir_name -y 2017 2018 -t -u --variables lj0pt ptz`
    - Omitting `--variables` processes every histogram in the input pickle, while providing one or more names limits the run to those histograms.
    - `--year YEAR [YEAR ...]` filters both MC and data histograms to the selected campaign tokens before plotting. The resolver mirrors the datacard utilities, accepts Run 2 and Run 3 tokens, and prints a summary of the samples that were retained or vetoed.
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

Filtering the pickle to a subset of campaigns is now built into both the Python CLI and the wrapper. Pass `-y/--year` with one or more tokens (e.g. `2017 2018 2022EE`) to restrict the MC and data samples before any plotting or yield aggregation. The script echoes a summary of the retained and vetoed samples so it is easy to verify the filter matched the intended years.

Blinding is now governed by a single flag pair: `--unblind` always renders the data layer regardless of the region defaults, and `--blind` hides the data. When neither flag is provided the tool unblinds control-region plots and blinds signal-region plots, matching the standard analysis policy. The resolved region and blinding choice are echoed on start-up for clarity.

Long pickle sweeps can opt into multiprocessing with `--workers N`. When set above one the script fans the variable list out across a `ProcessPoolExecutor`, pre-creates the output directories, and aggregates the per-worker statistics before printing the summary counts. If idle slots remain, the work queue expands to `(variable, category)` pairs so that categories render in parallel. Each worker unpickles the histogram dictionary, so memory consumption increases roughly linearly with the worker count—start with a small value (e.g. `--workers 2` or `--workers 4` on machines with plenty of RAM) and scale up only if the host has headroom.

Add `--log-y` to either entry point when you need the stacked yields on a logarithmic scale. The plotter automatically rescales bins with zero or negative MC content so the log axis is well-defined while leaving the ratio panel on a linear scale for readability.

Console verbosity is now controlled by mutually exclusive `--verbose` and `--quiet` switches. Quiet mode remains the default and prints only high-level progress (region resolution, worker counts, summary statistics). Add `--verbose` to include the per-variable headings, sample inventories, and channel lists that previously flooded the terminal.

| Entry point | When to use |
| --- | --- |
| `python make_cr_and_sr_plots.py` | Direct access to every CLI flag for notebook or batch workflows. |
| [`./run_plotter.sh`](#run_plottersh-shell-wrapper-quickstart) | Convenience wrapper that mirrors the auto-detection logic and common flags. |

Common invocation patterns (`-y/--year` now accepts multiple tokens for combined campaigns):

* Control-region scan with automatic blinding: `python make_cr_and_sr_plots.py -f histos/plotsCR_Run2.pkl.gz`
* Summing luminosities across multiple years: `python make_cr_and_sr_plots.py -f histos/plotsCR_Run2.pkl.gz -y 2016APV 2016 2017 2018`
* Signal-region pass where the filename already encodes `SR`: `python make_cr_and_sr_plots.py -f histos/SR2018.pkl.gz -o ~/www/sr --variables lj0pt ptz`
* Overriding the heuristic and forcing a blinded SR workflow: `python make_cr_and_sr_plots.py -f histos/plotsTopEFT.pkl.gz --sr --blind`
* Producing unblinded CR plots with explicit tagging and timestamped directories: `python make_cr_and_sr_plots.py -f histos/CR2018.pkl.gz --cr -t -n cr_2018_scan`
* Switching the stacked panel to a log scale: `python make_cr_and_sr_plots.py -f histos/plotsCR_Run2.pkl.gz --log-y`

#### run_plotter.sh shell wrapper quickstart

The `run_plotter.sh` helper script lives alongside `make_cr_and_sr_plots.py` and reproduces the same filename-based auto-detection for control vs. signal regions. After resolving the region it appends the corresponding `--cr` or `--sr` flag before delegating to the Python CLI. When both `CR` and `SR` tokens appear in the filename the wrapper prints a warning and falls back to the control-region defaults unless you pass an explicit override.

Wrapper options match the Python interface so that README guidance applies verbatim. `--variables` accepts the same list of histogram names, and `--blind` / `--unblind` toggle data visibility after the wrapper has selected a region. You can still provide manual `--cr` or `--sr` overrides, and everything after a literal `--` is forwarded untouched to `make_cr_and_sr_plots.py` for less common tweaks.

The wrapper also exposes the new `--workers` flag; the argument is forwarded directly to the Python CLI, so the same variable/category fan-out and memory-usage caveats apply when you request more than one worker.

Use `-v/--verbose` with the wrapper when you need the Python CLI's detailed logging, or `--quiet` to enforce concise output explicitly.

Example commands:

* Auto-detected control-region plotting with timestamped outputs: `./run_plotter.sh -f histos/plotsCR_Run2.pkl.gz -o ~/www/cr_plots --timestamp`
* Combining Run-3 campaigns in one call: `./run_plotter.sh -f histos/CR2022_combo.pkl.gz -o ~/www/cr_run3 -y 2022 2022EE 2023 2023BPix`
* Enforcing a blinded SR pass with specific variables: `./run_plotter.sh -f histos/plotsTopEFT.pkl.gz -o ~/www/sr -n sr_scan --sr --blind --variables lj0pt ptz`
* Passing additional CLI flags through the wrapper: `./run_plotter.sh -f histos/SR2018.pkl.gz -o ~/www/sr_2018 --unblind -- --no-sumw2`
* Switching the stacked panel to a log scale via the wrapper: `./run_plotter.sh -f histos/plotsCR_Run2.pkl.gz -o ~/www/cr_plots --log-y`

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

