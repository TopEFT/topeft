# Plotting

## Entrypoints and defaults

`analysis/topeft_run2/run_plotter.sh` is the maintained convenience wrapper.
It maps year aliases, can infer CR/SR from the input filename, supplies wrapper
defaults for blinding and output naming, and forwards unsupported options to
`make_cr_and_sr_plots.py`.

`analysis/topeft_run2/make_cr_and_sr_plots.py` is the direct interface. Its
inputs are repeatable `-f` paths or a list file. The default channel output is
`merged`, the default binning view is `processing`, the default worker count is
one, and year coverage defaults to `warn`. `--binning fitting` asks the plotter
to resolve the same exact late aggregation used by cards; it does not rewrite
the source PKL.

Both commands are public supported interfaces. The shell usage block and
`make_cr_and_sr_plots.build_arg_parser()` are option authority. The direct CLI
returns process status; its side effects are output-directory creation, plot
files, optional merge caches/reports, and negative-contribution reports.

### Stable direct option groups

| Group | Type/default/accepted values | Contract |
| --- | --- | --- |
| Inputs | Repeatable `-f/--pkl-file-path` or list file; at least one resolved path | Every PKL requires compatible sidecar/schema/provenance. Inputs are merged through `load_and_merge_histogram_pkls(require_sumw2=True)`. |
| Output | Path/name strings; optional timestamp | Creates or reuses the resolved directory. Condor can enable timestamp tagging to reduce collisions. |
| Years | Individual supported years or `run2`/`run3` aliases | Tokens normalize before region context. Mixed Run 2/Run 3 plotting is rejected. |
| Region/blinding | Filename inference or explicit CR/SR; CR unblinded and SR blinded by default | Ambiguous filenames default to CR with a warning unless overridden. Explicit blind/unblind wins. |
| Channel output | `merged`, `split`, `both`, or `-njets` variants; default `merged` | `-njets` preserves metadata-defined jet bins rather than changing histogram binning. |
| Binning | `processing` or `fitting`; default `processing` | Fitting uses exact family/channel aggregation. `--rebin-plot-vars` is separate integer-factor presentation rebinning. |
| Variables/workers | Optional variable list; workers integer default 1 | Invalid variables and channel coverage fail at context/coverage validation. |
| Uncertainties/reports | Supported uncertainty switches; year coverage `warn`; negative report enabled | Controls presentation/reporting, never source artifact content. |
| Merge/cache | Merge report/cache and merge-only controls | Merge-only stops after successful validation; cached output receives derived provenance. |

## Configuration owners

`topeft/params/cr_sr_plots_metadata.yml` owns maintained plotting metadata,
including:

- data and MC uncertainty drawing options;
- data-driven process prefixes;
- CR/SR channel leaf groups and aliases;
- process grouping patterns and visual metadata;
- per-region skip rules and output behavior;
- per-njet channel groupings used by the `*-njets` output modes;
- stacked-ratio presentation defaults.

An MC process that matches no configured background pattern is retained as a
one-off group named after the raw process, with a warning and fallback colour.
It is not silently removed from the total MC stack.

The histogram-family processing/fitting edges remain owned by
[`topeft.modules.axes`](flexible_binning.md); plot metadata must not duplicate
them.

## Developer surfaces

The source module is signature authority. These contracts avoid making an
extension depend on the monolithic call graph.

| Fully qualified symbol | Kind/status; parameters and return | Contract, side effects, and failures |
| --- | --- | --- |
| `make_cr_and_sr_plots.build_arg_parser` | Public CLI builder; no inputs → `ArgumentParser` | Defines options/defaults. Parser acceptance does not bypass later artifact/year/channel validation. |
| `make_cr_and_sr_plots.run_with_args` | Developer-facing; parsed namespace and parser → integer status | Resolves paths/region/blinding, creates output, validates/merges artifacts, optionally caches, materializes scalar views, and dispatches plots. Parser/config/merge errors stop rendering. |
| `make_cr_and_sr_plots._apply_plot_binning_view` | Internal extension; histogram, family, exact channels, mode → view | Processing returns physical view; fitting resolves a common exact-channel view and rebins. Does not mutate the source artifact. |
| `make_cr_and_sr_plots.RegionContext` | Developer-facing class; region identity, histograms, years, channel/group/sample sets and optional policy/presentation fields → context | Normalizes one region's authority. Required values are normally produced by `build_region_context`; direct construction must supply coherent groups/channels. No I/O. |
| `make_cr_and_sr_plots.build_region_context` | Developer-facing; region, histograms, years, optional unblind and output/binning keywords → `RegionContext` | Accepts CR/SR only; validates years and resolves metadata-owned channels, groups, luminosity, skips, and defaults. |
| `make_cr_and_sr_plots.produce_region_plots` | Developer-facing; context, output, variables, uncertainty/unit/log controls; optional unblind/workers/rebin/report → negative-report rows | Selects eligible variables/categories and writes requested figures. Coverage/schema/rendering errors fail at their owners. |
| `make_cr_and_sr_plots.validate_variable_channel_coverage` | Developer-facing; variable/channel expectations → validation result/`None` | Rejects missing or ambiguous physical channel content instead of guessing. |
| `make_cr_and_sr_plots.validate_channel_group` | Developer-facing; group/leaves and observed authority → validation result/`None` | Rejects invalid, empty, duplicate, or unresolved definitions. |
| `make_cr_and_sr_plots.effective_entries` | Developer-facing diagnostic; values and sumw2 arrays → effective-entry array | Computes yield-squared/sumw2 with source-owned guarded zero/invalid behavior. No I/O. |
| `collect_negative_contribution_rows`, `collect_negative_rows_for_plot_stage` | Developer-facing diagnostics; plot-stage arrays/context → row records | Identify negative MC contributions without changing the stack. |
| `write_negative_weight_report` | Developer-facing writer; rows and destination → written report | Writes only the diagnostic artifact; file errors propagate. |
| `make_cr_and_sr_plots.run_plots_for_region` | Developer-facing high-level renderer | Coordinates context and region sweep. Extend shared region behavior through context/metadata, not copied CR/SR loops. |
| `group_bins`, `make_region_stacked_ratio_fig` | Internal rendering extensions | Own grouped arrays and stacked-ratio construction; not independent public APIs/config owners. |

Important callers are `run_plotter.sh`, direct CLI users, and the Condor
plotting wrapper. Intended extension points are `cr_sr_plots_metadata.yml` for
configuration, context construction for maintained region semantics, and the
relevant renderer for presentation. Histogram axes belong to `axes.py`;
artifact compatibility belongs to merge/sidecar owners.

## Source and test authority

- `analysis/topeft_run2/run_plotter.sh`
- `analysis/topeft_run2/make_cr_and_sr_plots.py`
- `topeft/params/cr_sr_plots_metadata.yml`
- `tests/test_make_cr_and_sr_plots.py`

## Physics-facing view semantics

Plotting consumes already-produced categories, processes, observables,
systematic labels, and artifact provenance. Region context selects the
maintained CR or SR channel authority, process grouping, luminosity/year view,
and eligible variables. A processing or fitting binning view changes the exact
displayed aggregation; it does not redefine the processor observable.

Systematic and validation views expose stored variations and coverage. They do
not create missing templates or assign new nuisance meaning. Negative-weight
and effective-entry diagnostics report properties of the selected content
without changing the histogram stack.

Physics-facing region, grouping, observable, and binning policy is distinct
from scheduling, file paths, rendering mechanics, and presentation styling.
The source establishes displayed behavior and source-owned binning; no
unrecorded aesthetic or physics motivation is assigned here. See
[categories and observables](categories_and_observables.md) and
[histogram artifacts](histogram_artifacts.md).

## Practical defaults and change bridge

The maintained region/group/sample/default authority is
[`cr_sr_plots_metadata.yml`](../../topeft/params/cr_sr_plots_metadata.yml),
while [`axes.py`](../../topeft/modules/axes.py) and
[`axis_binning.py`](../../topeft/modules/axis_binning.py) own the observable and
view. A representative CR view selects CR region context, an `invmass`
histogram, the processing binning view, and metadata-owned process groups. It
does not create an `invmass` observable or CR category.

Use [run and extend CR/SR plotting](../how_to/plotting.md) to change a region,
group, variable, binning view, or coverage gate. Source/artifact and physics-
category changes remain at their owning pages.

## View-bearing output contract

Processing and fitting are distinct plot views. Their output plot paths and
negative-report paths include `_processing` or `_fitting`; callers must keep
those mode-bearing names in separate namespaces to avoid collisions. A plot
skipped for `empty-mc-content` records a rendering condition only. It is a
separate evidence surface from the underlying bin content and cannot be used
as a bin-content or statistical-adequacy oracle.
