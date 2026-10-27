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
The wrapper also composes a sequential SR baseline from a validated TSV via
`--campaign-manifest`, `--campaign-phase all|njets|other`, `-o`, and an explicit
`--dry-run` or `--execute`. This campaign interface calls the same maintained
single-input wrapper path for every row; the direct Python CLI has no campaign
or display-group switch. Ordinary single-invocation wrapper options retain
their behavior.
Maintained nominal campaign child commands select fitting binning explicitly
and omit `--debug-canvas-annotations` unless requested; the generic direct
plotter default remains processing.

`analysis/topeft_run2/plot_campaign_condor.py` is the plotting-owned
shared-filesystem campaign frontend. `prepare` consumes that same TSV and emits
one descriptor per principal PKL, an itemdata file, a worker script and one
vanilla `plots.sub`. It never reads a production PKL, renders, or submits.
An optional `--unit-id` queues exactly one manifest-derived unit for a pilot.
`prepare` binds fitting binning and a default-false diagnostic canvas flag in
each `plot_campaign_unit_v2` descriptor; its explicit
`--debug-canvas-annotations` option binds true. `run-unit` validates and
obeys both modes, rejecting v1, missing or invalid modes before PKL access.
The monitor checks the same descriptor schema and mode fields.
`run-unit` validates its descriptor against the manifest-derived ten-path
campaign input set and checks declared-PKL availability. Prepare-time source
identity is receipt provenance, not an execution gate. It loads the selected
family/channel union once in the parent, then calls `run_with_args` with a
loaded view for each logical row. A mixed unit reduces each njets view using
the same physical-channel patterns as the selective njets path. Render workers receive prepared plotting
state and do not call the PKL loader. An integrity SHA read in the maintained
loader precedes the selective streaming traversal. The v2 receipt records
planned, valid PNG, allowed skip, and unaccounted counts; a path/size PNG
inventory identity; a structured skip inventory identity and records; and no
per-PNG hashes. The plotter owns the skip decision, while the frontend requires
exact one-to-one output identities before writing the receipt.
The structured skip vocabulary is owned by the plotter's drawability branches:
`empty-mc-content`, `empty-data-content`, `empty-or-missing-input`, and
`make_region_stacked_ratio_fig-returned-none`. Other absent files, silent
early returns, and exceptions do not become allowed skips.

The dispatch requests explicit CPUs, memory, disk, runtime and render-worker
count, with `request_cpus >= workers`. SR campaign figures use
`histos/<campaign>/<period>/merged/<category>/<figure>.png`. The period is
`run2`, `run3`, or an individual year (`2016APV`, `2016`, `2017`, `2018`,
`2022`, `2022EE`, `2023`, `2023BPix`). Every produced `njets` figure also has
a regular-file copy at `<period>/merged_njets/<category>/<figure>.png`; the
primary `merged` view retains that figure. Technical unit and variable-group
identities remain in descriptors and receipts, not figure directories. Figure
creation rejects an occupied file, and receipt accounting checks the primary
planned paths for each unit. Receipt-v2 files live in `_metadata/receipts/`;
negative-contribution reports, logs, guards, and
ledger live in `_metadata/diagnostics/`. The optional composition JSON is not
automatically serialized by either SR campaign frontend. No retry, resume,
staging, DAG, or Condor-specific output renaming is supplied.

Campaign dry-run validates the TSV and prints resolved child commands without
creating a root or reading PKL contents. Execute requires a new root under
`topeft/histos/`, verifies distinct selected PKL and sidecar digests, and runs
one child at a time beneath `topeft.modules.node_memory_guard` with a
245000000000-byte total-node-memory threshold and 0.5-second sampling. Each
unit records input/output identity, a visible tee log, guard JSONL and status.
The launcher makes no Git or source-manifest call. A nonzero child or tee result,
missing/ambiguous terminal guard result, or occupied unit stops the remaining
rows. There is no automatic retry or resume. A dry-run command is a validated
configuration, not evidence of produced figures.

### Stable direct option groups

| Group | Type/default/accepted values | Contract |
| --- | --- | --- |
| Inputs | Repeatable `-f/--pkl-file-path` or list file; at least one resolved path | Every PKL requires compatible sidecar/schema/provenance. Inputs are merged through `load_and_merge_histogram_pkls(require_sumw2=True)`. |
| Output | Path/name strings; optional timestamp; `--output-formats` accepts `png`, `pdf`, and `svg`, default `png` | Creates or reuses the resolved directory and writes every requested format for each logical figure. This is renderer capability, not qualification of campaign-wide multi-format staging or promotion. |
| Years | Individual supported years or `run2`/`run3` aliases | Tokens normalize before region context. Mixed Run 2/Run 3 plotting is rejected. |
| Region/blinding | Filename inference or explicit CR/SR; CR unblinded and SR blinded by default | Ambiguous filenames default to CR with a warning unless overridden. Explicit blind/unblind wins. |
| Channel output | `merged`, `split`, `both`, or `-njets` variants; default `merged` | `merged` is aggregate presentation; `merged-njets` preserves metadata-defined jet/category subdivisions. These are presentation modes, not likelihood-bin counts. |
| Binning | `processing` or `fitting`; default `processing` | Fitting uses the exact maintained category-aware aggregation when the variable has a fitting definition and otherwise keeps the identity view; it does not invent fitting edges. `--rebin-plot-vars` is separate integer-factor presentation rebinning. |
| Diagnostic canvas annotations | `--debug-canvas-annotations`; default off | Extra signed numeric canvas text appears only by explicit opt-in. Standard ticks, luminosity, collision energy and physics labels remain visible. Signed yields, sumw2, uncertainty inputs and separate negative-weight CSV/Markdown diagnostics retain their semantics. |
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

For SR, every present process must have exactly one flat display owner;
overlapping or unknown process identities fail explicitly. Locally absent
members and one-component categories are valid. CR retains its established
grouping behavior. `make_jvm_reviewer_panels.py` reads only
`LUMI_COM_PAIRS` from the same YAML, so its consumed schema is unchanged.

The histogram-family processing/fitting edges remain owned by
[`topeft.modules.axes`](flexible_binning.md); plot metadata must not duplicate
them.

## Developer surfaces

The source module is signature authority. These contracts avoid making an
extension depend on the monolithic call graph.

| Fully qualified symbol | Kind/status; parameters and return | Contract, side effects, and failures |
| --- | --- | --- |
| `make_cr_and_sr_plots.build_arg_parser` | Public CLI builder; no inputs → `ArgumentParser` | Defines options/defaults. Parser acceptance does not bypass later artifact/year/channel validation. |
| `make_cr_and_sr_plots.run_with_args` | Developer-facing; parsed namespace, parser, optional `channel_allowlist`, `loaded_view`, or `outcome_collector` → integer status | Resolves paths/region/blinding, creates output, validates/merges artifacts or consumes an already loaded selected mapping, materializes a consumer-local scalar view, and dispatches plots. The optional collector receives planned relative figure paths and structured maintained undrawable skips; the Condor frontend requests PNG only. A loaded view requires one input and render-only options. The allowlist is SR njets only, validated against base names and aliases; omission selects all channels. An output timestamp is added only on explicit request. |
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
- `analysis/topeft_run2/test/test_run_plotter.py`
- `analysis/topeft_run2/make_cr_and_sr_plots.py`
- `topeft/modules/plot_uncertainty_contract.py`
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
and effective-entry diagnostics report properties of the selected content;
they do not mutate the source artifact.

The raw signed nominal and sumw2 remain source-owned. Ordinary SR process
groups are filled, colored stack components with matching legend entries.
Scientific totals and nuisance responses are calculated before display
grouping. Where negative group content would inflate a separately clipped
stack, the display allocates `max(sum(raw groups), 0)` among its positive
components in proportion to their positive content. Only groups with negative
content receive auxiliary signed outlines; their numeric value labels require
`--debug-canvas-annotations`. Nominal plots have no extra diagnostic numeric
canvas annotations. The Data/MC
denominator remains `max(sum(raw groups), 0)`; statistical variance and
systematic band inputs remain attached to the original signed content. CR
keeps its established drawing policy. Zero displayed denominators are masked.

Data markers use exact central Neyman/Garwood intervals at 68.27% confidence.
On a logarithmic axis, a zero Data count is represented by its Garwood upper
limit rather than by a fake epsilon-valued point.

The SR total uncertainty is a two-sided prefit envelope. For each card nuisance
identity, signed process and era Up/Down responses are added first. The upward
excursion is `max(total Up shift, total Down shift, 0)` and the downward
excursion is `max(-total Up shift, -total Down shift, 0)`. Independent nuisance
excursions and the retained MC statistical variance combine in quadrature on
each side. The band is recentered on the visible MC total. It visualizes the
card nuisance scheme; it is not a full card likelihood, a postfit band, or a
profiled covariance. A shape
mismatch between the raw systematic reference and displayed MC total raises an
explicit error; the renderer does not truncate, copy, or zero-pad either array
to manufacture agreement.

The scientific owner is
[`plot_uncertainty_contract.py`](../../topeft/modules/plot_uncertainty_contract.py),
which uses the maintained card `DatacardMaker.GROUP`, shape identity resolver,
annual luminosity factors, and run-specific rate reader from
[`datacard_tools.py`](../../topeft/modules/datacard_tools.py). Rate numbers
and PDF/QCD correlation tags remain in `topeft/params/rate_systs_run2.json`
and `rate_systs_run3.json`; the CR plot rate helper uses the separate
`rate_systs.json`. Producer PKLs own nominal, sumw2, and shape templates.
See [datacards and scalings](datacards_and_scalings.md) for the card policy.
The comparison domain is the same selected processes, eras, bins, flow,
normalization and parameter point. Data is excluded from MC nuisance inputs;
data-driven components follow the card's process-specific rate and luminosity
exclusions. SR display colors, labels, order and group splits never define
nuisance correlation.
The YAML display map is validated when SR context is built, but its graphical
composition is applied only after physical process, era and channel selections,
card rates and shapes, signed nuisance responses, and the total uncertainty
have been prepared. Physical/card process identity remains distinct from a
display label. Splitting or renaming a valid display group therefore leaves
the raw total, variance and nuisance-keyed total band unchanged.

Card-grouped ttll includes inclusive semileptonic and dileptonic ttbar, even
when ttbar is drawn in a separate green group. For example, a common renorm
nuisance moving the two components by +3 and -2 has net +1, whether one or two
colors show them. Treating them as independent would instead give `sqrt(13)`.
ISR uses card gg/qq/qg partitions, renorm/fact use card process identities,
PDF/QCD use run-specific JSON tags and exact down/up factors, and luminosity
uses the card's annual shape components without a scalar global addition.
Applicable Tau/year shape names follow the card resolver. The excluded DY,
single-top, W+jets and ZGamma auxiliaries remain in nominal and statistical
totals; card grouping is applied where defined and their source variations
are used when present. Their absent card JSON rate rows are an intentional
model boundary: no rate is inferred from a display group. The card excludes
`JES_Total` per process and era, including when it is the only JES source;
complete component pairs remain applicable. The default card profile does not
enable nominal-like Tau suppression, so those pairs remain applicable too.
`TTto4Q` remains an exact physical identity for its four Run 3 periods and maps
to itself for card nuisance naming outside the selected card process domain.
For `njets`, the card suppresses its jet-dependent `diboson_njets` and
`missing_parton` rate rows; the SR plot follows that rule. For a different
observable, the SR consumer retains each physical channel and its jet count
through the card's Diboson `diboson_njets` or tllq/tHq `missing_parton` rate
evaluation before category aggregation. A missing required physical channel or
factor fails the total-band gate. CR keeps its existing rate and shape handling.

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
# Persistent plot campaign state and receipt contract

The profile CLI is `run_plotter.sh --campaign-profile PROFILE --campaign-tag
TAG -o ROOT` with `--plan PLAN --dry-run|--execute`, `--resume`, optional
`--selected-unit ID`, or `--activate-pending-plan PLAN
--expected-plan-revision N`. Profile mode is mutually exclusive with the
legacy TSV campaign mode. Campaign state is implemented in
`analysis/topeft_run2/plot_campaign_manifest.py` and does not change rendering,
histogram production, or Condor job behavior.

Plan v1 binds a complete unit catalogue. Unit source fields are `unit_id`,
`component`, `producer_unit_id`, `input_path`, `sidecar_path`,
`qualification_source` (accepted status, evidence locator, accepted input and
sidecar SHA-256 values), and `period_view`. Derived fields are unchanged
sidecar histogram families, accepted input identity (sidecar SHA-256 and size),
accepted sidecar SHA-256, and deterministic first private root. No pre-run
output inventory, coverage matrix, or collision prediction is serialized.

Each component state is schema v1 and lives in
`.<sr_crlike_run2|sr_crlike_run3>_plot_campaign_state_v1.json` under its
component root. Statuses are `planned`, `running`, `success`, `failed`, and
`interrupted`. Attempts bind ID, times, child argv identity, private root,
stored provenance through the state, guard/child/tee results, receipt
identity, materialization status, result, and failure classification. State
writes are atomic. A planned unit has no attempt; every other status has exactly
one attempt with the matching result. Success also requires a completion receipt
identity and complete materialization. Incoherent persisted state fails closed
without retry. The root coordinator binds the full plan; combined status
is derived from separate component states.

Completion receipt v1 is created atomically after successful terminal
guard/child/tee readback and private-root observation. It binds profile, tag,
plan revision, unit, attempt, accepted input/sidecar identities, stored
provenance, private root, lifecycle and UTC times. It inventories only
actual emitted PNG relative paths, size, SHA-256, total count, per-variable
counts, and selected renderer messages. Counts are observations and may be
zero for any requested variable. Region-dependent absence is neutral. A
regular renderer log is scanned in full; `renderer_log_status` records
`present`, `missing`, `unreadable`, or `non_regular`, plus full-scan line and
message counts when present. The stored `renderer_messages` summary is capped
at 200 lines. Log presence and wording are evidence, never lifecycle authority.

Before the higher-level state transition, the guarded-child supervisor writes
`lifecycle_receipt_v1.json` atomically after guard, child, and tee terminate.
It binds attempt ID, command identity, private root, terminal guard result and
guard-log hash, three return codes, and terminal UTC time. A running attempt
with a valid lifecycle receipt reconstructs success or failure without a child
rerun. Guard-terminal evidence without the lifecycle receipt yields
`interrupted`; missing terminal evidence leaves `running` and blocks every
component from launching a new child. A known failure before guard creation
instead leaves a terminal failed attempt and permits unrelated planned units to
proceed. Malformed or contradictory state and
terminal evidence fail closed.

Canonical materialization uses only receipt-listed paths. The component state
first records path ownership and expected bytes, then atomically links a
completed copy into an absent destination. Reconciliation verifies receipt
and private/canonical bytes and finishes only missing copies; it does not
rerun a child. A destination with no ownership record or a mismatched record
or bytes is a collision. Unit success requires terminal lifecycle, valid
receipt, and completed materialization. Existing root or private directories
alone never promote status. Resume skips success and does not retry failed or
interrupted units. Revision N+1 appends units and retains all earlier bindings
and history; it fails closed on revision mismatch or rewrite.

New campaigns record a provenance-not-collected marker in the existing state
field. Older Git/source-manifest values are preserved as historical provenance;
coordinator, component, and receipt bindings must agree, but no current Git or
source identity is queried or compared on resume or revision. Reconciliation
of every running unit in all components precedes any new child. An invocation
that completes orchestration returns zero even with failed, interrupted, or
pending units. The JSON summary reports `campaign_status`, total
`planned_count`, `success_count`, `failed_count`, `interrupted_count`,
`running_count`, `pending_count`, and `waiting_on_active_child`. `waiting`
means a possible live child prevented all new launches. Neither full nor
selected mode silently retries failed or interrupted units.

The local backend runs one child at a time with one renderer worker, fitting
binning, merged channels, total uncertainty, PNG output, SR/unblind flags and
debug canvas annotations off. The memory guard and visible tee record terminal
lifecycle evidence. Runtime success establishes that a child completed; inspect
the rendered figures separately for visual and scientific quality.
