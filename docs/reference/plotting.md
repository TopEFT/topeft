# Plotting

## Entrypoints and defaults

`analysis/topeft_run2/run_plotter.sh` is the sole public plotting control wrapper.
It bootstraps Python and forwards to `plot_campaign_manifest.py`. The controller
owns ordinary CLI normalization (years, filename CR/SR inference, blinding defaults)
and persistent campaign plan/state/recovery/materialization. The direct scientific
CLI `make_cr_and_sr_plots.py` remains available; its default binning is processing,
channel output merged, workers one, year coverage warn and format PNG.

The execution boundary is `plot_campaign_backend.py`: one immutable argv and one
private attempt directory, with objective execution evidence only. The local backend
runs a guarded renderer child. `plot_campaign_condor.py` supplies only descriptor,
single-job submit and exact job/token observation. The Condor worker runs the same
supervisor without importing the controller or constructing a scientific view.
No standalone Condor campaign CLI or TSV campaign language remains.

`--backend local|condor` defaults to local. `--resource-observer cgroup_v2|node_pressure`
defaults to cgroup_v2; `--hard-limit-bytes` defaults to 245000000000. Cgroup v2 uses
allocation-local `memory.current` with finite `memory.max`; absent/unlimited data
fails closed. Node pressure is the explicit whole-node `MemTotal - MemAvailable`
policy, never job RSS. There is no automatic fallback. Probe with
`python -m topeft.modules.node_memory_guard --probe --observer cgroup_v2`.

State schema v1 remains readable without migration. New attempts carry an
`execution_contract` version 1 with backend, exact argv and resource policy.
Historical attempts lacking this field retain their original wrapper-command
identity for recovery verification only. They are not dispatched through it.
The backend writes one lifecycle receipt; only the controller determines campaign
success from that evidence, observed PNGs and collision-safe canonical materialization.
Submission ambiguity, unknown job identity and missing terminal evidence
prevent retry of the affected attempt. Independent planned Condor units can
still dispatch. A held job needs bound lifecycle evidence to establish a
terminal worker failure. Resume observes active attempts and selects only the
bounded recovery paths below; it does not blindly resubmit an attempt.
An exclusive worker-start marker prevents another renderer on a scheduler restart.

`plot_output_layout.py` alone defines human paths:
`<campaign>/<period>/merged/<category>/<figure>.png`, with an additional
`merged_njets` copy for produced njets figures. Backends never add namespaces.
Persistent lifecycle/guard/log evidence stays in each private attempt's `_metadata`.
No real scheduler qualification follows from synthetic adapter tests.

### Stable direct option groups

| Group | Type/default/accepted values | Contract |
| --- | --- | --- |
| Inputs | Repeatable `-f/--pkl-file-path` or list file; at least one resolved path | Every PKL requires compatible sidecar/schema/provenance. Inputs are merged through `load_and_merge_histogram_pkls(require_sumw2=True)`. |
| Output | Path/name strings; optional timestamp; `--output-formats` accepts `png`, `pdf`, and `svg`, default `png` | Creates or reuses the resolved directory and writes every requested format for each logical figure. This is renderer capability, not qualification of campaign-wide multi-format staging or promotion. |
| Years | Individual supported years or `run2`/`run3` aliases | Tokens normalize before region context. Mixed Run 2/Run 3 plotting is rejected. |
| Region/blinding | Filename inference or explicit CR/SR; CR unblinded and SR blinded by default | Ambiguous filenames default to CR with a warning unless overridden. Explicit blind/unblind wins. |
| Channel output | `merged`, `split`, `both`, or `-njets` variants; default `merged` | `merged` is aggregate presentation; `merged-njets` preserves metadata-defined jet/category subdivisions. These are presentation modes, not likelihood-bin counts. |
| Binning | `processing` or `fitting`; default `processing` | Fitting uses the exact maintained category-aware aggregation when the variable has a fitting definition and otherwise keeps the identity view; it does not invent fitting edges. `--rebin-plot-vars` is separate integer-factor presentation rebinning. |
| Diagnostic canvas annotations | `--debug-canvas-annotations`; default off | Extra signed numeric canvas text requires both this flag and `--sr-display-diagnostics`. Standard ticks, luminosity, collision energy and physics labels remain visible. Signed yields, sumw2, uncertainty inputs and separate negative-weight CSV/Markdown diagnostics retain their semantics. |
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
| `make_cr_and_sr_plots.run_with_args` | Developer-facing; parsed namespace, parser, optional `channel_allowlist`, `loaded_view`, or `outcome_collector` → integer status | Resolves paths/region/blinding, creates output, validates/merges artifacts or consumes an already loaded selected mapping, materializes a consumer-local scalar view, and dispatches plots. The optional collector receives planned relative figure paths and structured maintained undrawable skips; persistent campaign commands request PNG only. A loaded view requires one input and render-only options. The allowlist is SR njets only, validated against base names and aliases; omission selects all channels. An output timestamp is added only on explicit request. |
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

Important callers are the controller-selected local/Condor workers and direct CLI users. Intended extension points are `cr_sr_plots_metadata.yml` for
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
content receive auxiliary signed outlines only with `--sr-display-diagnostics`;
their numeric value labels additionally require `--debug-canvas-annotations`.
Nominal plots have no extra diagnostic numeric canvas annotations. The Data/MC
denominator remains `max(sum(raw groups), 0)`; statistical variance and
systematic band inputs remain attached to the original signed content. CR
keeps its established drawing policy. Zero displayed denominators are masked.

Nominal SR process grouping follows the maintained datacard process resolver.
Both inclusive dileptonic and semileptonic ttbar belong to nominal `ttll`;
there is no separate nominal `ttbar` display. The four fully-hadronic
`TTto4Q_central` samples (2022, 2022EE, 2023, 2023BPix) are excluded from
SR stacks, MC totals, uncertainty inputs and ratio denominators.

`--sr-display-diagnostics` is optional and default-off. It partitions the
retained card-grouped `ttll` into inclusive `ttbar` (dileptonic plus
semileptonic) and the `ttll` remainder, without changing scientific process
identity or nuisance treatment. It never restores TTto4Q. Signed outline
lines are also diagnostic-only: each legend entry describes the net signed
contribution of that display group in the plotted binning, rather than a
count of negative-weight events. Their legend handles preserve the line style.
Signed-total arithmetic stays active with diagnostics off. Diagnostic
presentation preserves Data, binning, uncertainties and Data/MC ratios.
`--debug-canvas-annotations` alone does not activate outlines.

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
when diagnostics draw ttbar in a separate green group. For example, a common renorm
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

The public launcher defaults to concise human output for campaign status and
execution. `-o ROOT` names the final canonical root. The controller alone
publishes receipt-verified observed plots through
`topeft/modules/plot_output_layout.py`; the backend writes only into private
attempt staging. `--status -o ROOT` is read-only and gets profile/tag from the
coordinator. It reports execution, publication and integrity separately,
shows the exact root and recorded scheduler job IDs, and never queries the
scheduler. Historical publication requires coherent persisted success, receipt,
and materialization provenance, independently of current canonical files.
Ordinary status checks current canonical presence and recorded sizes without
hashing; `--verify-published` additionally hashes available recorded PNGs
against materialization SHA-256 values. Missing or unsafe paths report
`unavailable`, while changed sizes report `content_mismatch`; neither erases
historical publication or its recorded canonical paths. Human status labels
historical backends separately from the policy for future attempts.
`--format json` retains the existing summary
keys and adds `summary_schema_version: 2`, `execution_policy`, `action_scope`,
active/blocked facts, and per-unit `publication` and `integrity` fields;
`--format yaml` presents the same derived summary. Authoritative state and
receipts remain JSON.

The public Condor `--execute` and `--resume` paths are one-shot by default.
Execute submits eligible units and performs one initial reconciliation per
new attempt. Resume reconciles selected active attempts before dispatch and
reads back newly submitted attempts once. Both return while jobs are active or
unresolved, after printing current status and one shell-quoted
`./analysis/topeft_run2/run_plotter.sh --status -o ROOT --observe-scheduler`
handoff. It assumes execution from the TOPEFT checkout with the intended Conda
environment already active; only the campaign root is resolved to an absolute
path. JSON/YAML stdout retains its schema; the handoff goes to stderr.
Only `--follow` opts into repeated foreground observation/reconciliation, with
the lock released between passes. It stops when only finalization remains.
Ctrl-C ends that observation without cancelling jobs or changing attempts. Fresh
execution resolves one future-attempt policy from code-owned defaults
(`local`, `cgroup_v2`, `245000000000` bytes) and stores it in the coordinator.
Bound resumes reuse it; contradictory explicit values fail before new attempts.
Legacy status and reconcile-only resume do not bind a policy. A legacy planned
local launch needs an explicit backend and may use resource defaults; a legacy
planned Condor launch needs explicit backend, observer and hard limit once.
The policy governs future attempts only. Condor passes reconcile all active
units, then submit every eligible planned unit independently and reconcile each
new attempt once. Local passes retain one active child. `--selected-unit` narrows
only the requested unit. Resume selects existing bounded recovery for a proven
failed attempt, a proven local Condor submit rejection, or a terminal
missing-lifecycle case. Ambiguity never becomes a retry. The query path binds
pool, schedd, exact job, token and owner, limits history to one ad and uses a
finite timeout; timeout leaves the attempt
unresolved.

A backend-proven terminal failure closes the attempt as failed even without a
worker guard file. A verified nonzero lifecycle receipt also closes a failed
worker whose scheduler job remains held or whose scheduler query is unavailable.
An unbound or contradictory receipt cannot establish this terminality. A
terminal-success attempt waits for `--finalize`, which can reconcile an existing
attempt whose persisted state is still `running` when bound scheduler, guard,
and lifecycle evidence independently prove terminal success. It then creates or
verifies the completion receipt and materializes/publishes canonical PNGs in the
same invocation. `--resume` is not a prerequisite for this finalization. Missing
or ambiguous evidence never becomes successful publication or a retry.
A backend-proven terminal success with incomplete worker evidence remains
`interrupted` / `publish pending`, available for later reconciliation without
launching another child. It is not counted as an active child.

The profile CLI is `run_plotter.sh --campaign-profile PROFILE --campaign-tag
TAG -o ROOT` with `--plan PLAN --dry-run|--execute`, `--resume`, `--finalize`,
optional `--selected-unit ID`, or `--activate-pending-plan PLAN
--expected-plan-revision N`. The former TSV campaign mode is retired. Campaign state is implemented in
`analysis/topeft_run2/plot_campaign_manifest.py` and does not change rendering,
histogram production, or renderer science; backend identity is recorded only on new attempts.

The public lifecycle is: planned coverage dispatches through `--execute` or
`--resume`; active attempts are observed by `--resume`; a proven failed attempt
uses the existing one-replacement recovery, and a proven local submit rejection
rearms the same attempt; a terminal missing-lifecycle case records recovery
provenance and waits; terminal success waits for `--finalize`; ambiguous
evidence stays unresolved. `--finalize` accepts selected or all eligible
terminal-success units, including proven stale `running` units, reuses the
existing receipt/materialization machinery, and never starts a worker or retries.
Already published units are skipped
unless a verified legacy output binding needs migration.
There is no separate `--publish`, `--reconcile`, public `--recover-*`, or
`--one-pass` action. `--status --observe-scheduler -o ROOT` remains the
canonical read-only monitoring command.

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
writes are atomic. A planned unit has no attempt; other statuses have one
attempt, or two after the sole verified failed-Condor replacement. In the latter
case, `attempt_0001` remains failed and `attempt_0002` carries the current
result; no third attempt is permitted. Success also requires a completion
receipt identity and complete materialization. Incoherent persisted state fails closed
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
`interrupted`; missing terminal evidence leaves `running` and blocks further
local children. Independent planned Condor units may still start. A known
failure before guard creation instead leaves a terminal failed attempt and permits unrelated planned units to
proceed. Malformed or contradictory state and
terminal evidence fail closed.

Canonical human materialization uses only receipt-listed paths below the global
campaign root; component roots retain private attempts and state. The component state
first records path ownership and expected bytes, then atomically links a
completed copy into an absent destination. Reconciliation verifies receipt
and private/canonical bytes and finishes only missing copies; it does not
rerun a child. A destination with no ownership record or a mismatched record
or bytes is a collision, except that a same-byte destination can be rebound to
a verified receipt. Unit success requires a terminal lifecycle or separately
verified missing-lifecycle recovery provenance, a valid receipt, and completed
materialization. Recovery leaves unavailable lifecycle fields null and never
creates a lifecycle receipt. Existing root or private directories
alone never promote status. Resume skips success. A failed Condor unit may
receive its sole replacement only after the nonzero lifecycle, old scheduler
identity and unchanged qualified input pass the recovery checks. A proven
local submit rejection may rearm the same attempt. A terminal interrupted
unit with missing lifecycle evidence can use the separate no-launch recovery
proof; ambiguity permits no retry. Revision N+1 appends units and retains
all earlier bindings and history; it fails closed on revision mismatch or rewrite.

New campaigns record a provenance-not-collected marker in the existing state
field. Older Git/source-manifest values are preserved as historical provenance;
coordinator, component, and receipt bindings must agree, but no current Git or
source identity is queried or compared on resume or revision. Reconciliation
of every active unit in all components precedes new dispatch. An invocation
that completes orchestration returns zero even with failed, interrupted, or
pending units. The JSON summary reports `campaign_status`, total
`planned_count`, `success_count`, `failed_count`, `interrupted_count`,
`running_count`, `pending_count`, and `waiting_on_active_child`.
`active_child_present` and `dispatch_blocked` are separate; `waiting` means
the selected local dispatch is blocked by an active child. Neither full nor
selected mode permits a retry without the bounded recovery proofs above.

The local backend runs one child at a time with one renderer worker, fitting
binning, merged channels, total uncertainty, PNG output, SR/unblind flags and
debug canvas annotations off. The memory guard and visible tee record terminal
lifecycle evidence. Runtime success establishes that a child completed; inspect
the rendered figures separately for visual and scientific quality.

The Condor worker and submit intent markers sync their parent directories before
external effects. Missing cgroup interfaces are checked before local guard launch.
The low-level guard API/CLI retains its named node_pressure compatibility default;
public campaign commands always pass their selected resource policy explicitly.
Missing-lifecycle recovery that checks local PIDs is limited to local attempts;
remote process liveness is not inferred from a submit-host PID. An ambiguous
submission without a proven scheduler identity remains waiting and requires
separately authorized reconciliation evidence; resume never guesses or resubmits.

# Scheduler observations in campaign status

`--status --observe-scheduler -o ROOT` is a one-shot, read-only projection through
the campaign controller and its backend. Default `--status` makes no scheduler
calls. The option is valid only with status, and can be combined with human,
JSON or YAML output and `--verify-published`. It does not change the persisted
campaign or recover scheduler identity to disk.

`--selected-unit UNIT_ID` scopes the status projection and both optional
scheduler observation and published-content hashing to exactly one existing
unit. Counts, aggregate status, historical backends and active/dispatch facts
describe that projection; campaign profile/tag, plan revision, canonical root
and future execution policy retain their campaign identity. Unknown or
nonunique unit IDs fail before observation or content hashing. Without a
selector, status and both checks remain campaign-wide. Selection does not
reconcile, persist recovered identity, create attempts or publish.

Each Condor unit's optional `scheduler` object contains `state`, `backend_state`,
`source`, `observed_at`, `attention`, `reason`, `mem_mb`, `rss_kib`, and
`raw_classad`. Queue miss falls back to history; unresolved or contradictory
identity is UNKNOWN with attention. HELD, removal, nonzero exit and signal
termination require attention. A scheduler observation never overrides the
separate execution, publication, integrity or dispatch axes. MEM_MB and RSS_KIB
in JSON/YAML (`mem_mb` and `rss_kib`) preserve raw scheduler values, including
zero; unavailable values are null. Human `MEM_MB` uses finite nonnegative numeric
`MemoryUsage` first, then numeric `ResidentSetSize` in KiB rounded up to whole
MiB (1024 KiB), else `-`. Booleans and expression strings are not numeric memory
sources; expression text is never evaluated. `RSS_KIB` remains the raw value.

Human `Campaign state (persisted)` counts retain the meaning of the stable
machine count fields, including `running_count` and `success_count`, under
`summary_schema_version: 2`. The per-unit `CAMPAIGN VIEW` is the existing
controller execution view and may reflect a bound lifecycle receipt without
persisting a transition. `Scheduler observed` is a separate aggregate of exactly
the non-null per-unit `scheduler` objects rendered below it. COMPLETED, RUNNING,
IDLE and HELD are counted individually; every other state, including UNKNOWN
and unresolved evidence, contributes to OTHER/UNKNOWN. This presentation makes
no extra scheduler query and performs no reconciliation or state write.

`plot_campaign_condor.py` is an internal observation/submission adapter. Its
retired standalone command exits with a migration message; legacy diagnostics
and submission maps are not implicitly adopted by a persistent campaign.

### Dedicated datacard campaign profiles

`sr_datacard_run2`, `sr_datacard_run3`, and `sr_datacard_run2_run3` use the
existing schema-v1 qualified-input plan and campaign controller. Each component
owns its own `.sr_datacard_<run>_plot_campaign_state_v1.json`; the combined
profile keeps independent states under `components/run2` and `components/run3`.
The plan binds the input roster; the profile binds the run components. Variables
remain derived from each accepted sidecar's `runtime_histogram_families`.
No category or variable selection field is added. Existing `sr_crlike_*` roots
and state names retain their meaning and are not migrated.
