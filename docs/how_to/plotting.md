# Run and extend CR/SR plotting

Use `run_plotter.sh` for the maintained convenience interface and
`make_cr_and_sr_plots.py` when you need the complete plotting CLI. Both consume
existing histogram PKLs; they do not run the processor or mutate the input
artifact.

| Interface | Owns | Delegates or derives | Default source | Downstream outputs |
| --- | --- | --- | --- | --- |
| `run_plotter.sh` | Python bootstrap and forwarding to plot_campaign_manifest.py | controller owns CLI normalization and backend dispatch | controller: one worker; renderer supplies unset defaults | same figures/reports as direct CLI |
| `make_cr_and_sr_plots.py` | artifact merge/coverage checks, region context, process grouping, blinding, systematics, binning view, rendering and reports | reads plotting metadata and canonical axis definitions | argparse: output `.`, name `plots`, merged channels, processing binning, one worker, year coverage `warn`, negative report enabled | plot directories, HTML indexes, optional zero-yield report, negative-contribution CSV/Markdown |

The controller forwards unrecognized ordinary plotting tokens, but forwarding does not make shell
code the semantic owner of the option. Exact direct options stay in the Python
parser and [plotting reference](../reference/plotting.md).

## Plot with `run_plotter.sh`

Run from `analysis/topeft_run2`:

```bash
./run_plotter.sh -f /path/to/final_np.pkl.gz \
  -o /absolute/path/to/plots -y run3 --sr \
  --variables lj0pt ptz ptll lt --channel-output merged --dry-run
```

The controller validates paths and year tokens, expands `run2`/`run3`, resolves CR
or SR from an explicit flag or filename, and forwards plotting options. It does
not own channel maps, process grouping, styles, or binning definitions. Keep
`--dry-run` until the resolved command and output destination are correct.
The wrapper's dry run checks input path readability and prints the resolved
command without creating the output directory or reading the PKL contents.

An explicit `--cr` or `--sr` is preferable for production plots. Filename
detection is a convenience: if both `CR` and `SR` appear, the controller warns and
uses its control-region fallback unless explicitly overridden. `--blind` and
`--unblind` override the direct plotter's region-dependent data choice.
`--channel-output` accepts `merged`, `split`, `both`, and their `-njets`
variants; the latter retain per-jet bins defined by plotting metadata.

Extend ordinary CLI normalization in the controller and science options at the
renderer. Keep the shell wrapper as bootstrap/forwarding only. Validate controller
CLI behavior and backend descriptors in their focused tests; do not add a second
campaign parser or state owner.

## Plot directly

```bash
python make_cr_and_sr_plots.py \
  -f /path/to/final_np.pkl.gz -o /absolute/path/to/plots -n run3_sr \
  -y 2022 2022EE 2023 2023BPix --sr \
  --variables lj0pt ptz ptll lt --binning fitting \
  --year-coverage-policy error
```

Repeat `-f` for coherent fragments or use `--pkl-list-file`. Direct plotting
defaults to the processing-binning view, merged channel output, one worker,
warning-only year coverage, no extra diagnostic numeric canvas annotations,
and enabled negative-weight reports. The direct CLI accepts
`--sr-display-diagnostics` as the display diagnostic opt-in;
`--debug-canvas-annotations` adds numeric labels only when those diagnostics
are active; signed yields, uncertainties and separate negative-weight reports do not depend on it. A mixed Run 2
and Run 3 year selection is rejected.

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

The plotter validates and merges repeated inputs before rendering. Choose
`--year-coverage-policy error` for a production validation pass; `warn` records
structural gaps but continues, and `off` suppresses that check. More than one
worker parallelizes variables and then categories when slots remain. The parent
loads the PKL before creating render workers; worker memory use still grows
with the retained plot state and multiprocessing start method.

## Change plotting configuration at its owner

`topeft/params/cr_sr_plots_metadata.yml` owns the maintained plotting metadata:

- `CR_CHAN_DICT` and `SR_CHAN_DICT` map displayed category names to histogram
  channel labels;
- `CR_GRP_MAP` and `SR_GRP_MAP` group raw processes for stacked plots; SR
  groups are flat display ownership applied after physical rates and shapes;
- `REGION_PLOTTING` owns region transformations, removals, opt-in skips, and
  blinding mechanics;
- `STACKED_RATIO_STYLE` owns figure, axis, tick, and legend presentation.

Do not copy those values into `run_plotter.sh`. A metadata change can affect
which channels/processes are shown, how yields are combined, blinding, and all
produced figures. Update the plotting tests covering the changed block and
validate at least one representative CR or SR configuration statically or with
the permitted lightweight test scope.

When adding or changing a plotting category:

1. Confirm the physical channel labels present in the producer artifact and the
   intended merged/split presentation.
2. Edit `CR_CHAN_DICT` or `SR_CHAN_DICT` and the appropriate group/region block
   in `cr_sr_plots_metadata.yml`; avoid an equivalent hard-coded Python or shell
   list.
3. Decide whether the change is selection, grouping, transformation, skip,
   blinding, or styling. Keep it in that existing metadata block.
4. Validate unknown/missing channels, merged and split output, relevant year
   coverage, blinding, sumw2/systematics, and category-specific removal/skip
   behavior.
5. Update the focused `tests/test_make_cr_and_sr_plots*.py` owner. If a tau
   merged category changes, include
   `test_make_cr_and_sr_plots_1l_tau_merged.py`; if fallback/validation changes,
   use the corresponding channel tests.

An axis or binning change instead belongs to `axes.py` and must keep nominal,
sumw2, and any EFT payloads aligned. A new SR display group can change the
stack and legend but not the signed total or nuisance response; a new physical channel or
processing family usually starts upstream and may require new artifacts.

## Select the SR njets wave in Python

The maintained programmatic route is `run_with_args(args, parser,
channel_allowlist=...)`. It accepts configured `SR_CHAN_DICT` base names or
their aliases, validates unknown names, limits the njets loader to the selected
physical channels, and filters the rendered category map. Omit the keyword for
the normal complete SR selection. It requires `--sr --variables njets`; there
is no channel allowlist CLI option. For example, after importing
`analysis.topeft_run2.make_cr_and_sr_plots as plotter`:

```python
wave_one = (
    "2lss_4t_m", "2lss_4t_p", "2lss_m", "2lss_p",
    "2lss_fwd_m", "2lss_fwd_p", "2los_onZ_1tau",
    "3l_1tau_1b", "3l_1tau_2b", "3l_onZ_1b", "3l_onZ_2b",
)
parser = plotter.build_arg_parser()
args = parser.parse_args([
    "--sr", "--variables", "njets", "-y", "2022", "-f", input_pkl,
    "-o", output_parent, "-n", "wave1_run3_2022",
])
plotter.run_with_args(args, parser, channel_allowlist=wave_one)
```

This programmatic allowlist remains available for an explicitly selected njets
subset. Use the campaign interface above for the full principal-input baseline
catalogue. For combined Run 2 select UL16APV, UL16, UL17 and UL18; for combined
Run 3 select 2022, 2022EE, 2023 and 2023BPix. Each separate view selects one
period. Source-period filtering applies before aggregation. Check input
identities and sidecars against the
[histogram artifact contract](../reference/histogram_artifacts.md).

Histogram processing/fitting edges belong to `topeft/modules/axes.py`, not the
plot metadata. See [change binning](flexible_binning.md). Exact plotting
interfaces and metadata keys are in the
[plotting reference](../reference/plotting.md).

## Execution backends and resource observation

Use the persistent campaign interface below. Fresh execution defaults to
`--backend local`; select `--backend condor` for a Condor campaign. Local
execution is sequential; independent Condor units may overlap.
The public wrapper is never a batch payload. The Condor adapter sends a single
immutable renderer command to `plot_campaign_backend.py --worker`; it has no
standalone campaign CLI, plan parser, scientific view selection or output roster.
The former TSV campaign interface and standalone Condor launchers are retired.
Use the qualified persistent JSON plan, not a converted expected-output roster.

Local and Condor workers use the same guard and lifecycle evidence. The controller
alone interprets receipts and materializes observed figures. Condor submission
can leave active jobs while independent planned units remain dispatchable; a
later resume observes those same attempts.
A nonzero, timed-out or ambiguous submit response does not prove that no job was
created. It never causes resubmission or a local fallback. Missing scheduler
identity requires separate reconciliation evidence. A worker start marker prevents
rerender after scheduler eviction/restart. A held job remains unresolved unless
its bound lifecycle receipt proves a terminal worker failure.

Condor requires a shared checkout, interpreter and attempt directory. Each job
requests one CPU, disk 10240 MB, runtime 86400 seconds and memory rounded up from
`--hard-limit-bytes` to MiB. Those requests and fake-scheduler tests do not establish
site/runtime qualification. No production execution is implied by these examples.

The default `--resource-observer cgroup_v2` reads the current allocation's
`memory.current` and requires a finite positive `memory.max`. This is allocation
memory charge, not process RSS. Before a later authorized local run, use:

```bash
python -m topeft.modules.node_memory_guard --probe --observer cgroup_v2
```

Choose an explicit positive `--hard-limit-bytes` no larger than the observed
allocation limit, with suitable headroom. The default hard limit is
245000000000 bytes and sampling is every 0.5 seconds. Missing or unlimited
cgroup accounting fails closed; there is no silent fallback. An explicitly
approved `--resource-observer node_pressure` instead observes whole-node
`MemTotal - MemAvailable`. It requires a separately appropriate node-pressure
threshold and must never be presented as job RSS. Open OnDemand is an environment
for the **local** backend, not a third backend.

## Common failures

| Symptom | Correct owner or action |
| --- | --- |
| wrapper cannot infer CR/SR | pass one explicit region flag; do not rename a PKL as evidence of contents |
| requested channel is absent | inspect artifact channel labels and plotting metadata; do not silently drop a required production category |
| combined channels resolve incompatible fitting axes | correct exact channel binning definitions or plot separately |
| missing/ambiguous sumw2 dense axis | correct the companion artifact/schema; do not use a nominal histogram as variance |
| memory growth with `--workers` | reduce workers; parallel processes each load input state |
| Condor job starts but cannot import/read | inspect worker-visible checkout/environment/storage and execution evidence; do not blindly resubmit |

To add a physics observable or category, use
[categories and observables](categories_and_observables.md) before updating the
plot metadata. To change only region context, group membership, plotted
variables, a supported binning view, or coverage validation, keep the change in
this guide and the canonical
[`cr_sr_plots_metadata.yml`](../../topeft/params/cr_sr_plots_metadata.yml).

## Output names and diagnostic boundary

Select the processing or fitting view explicitly. Plot outputs and negative
reports carry the corresponding `_processing` or `_fitting` suffix, so the two
views must not share an output namespace. The plotter's local `--workers`
option controls local rendering work; it is unrelated to external Work Queue
worker provisioning.

An `empty-mc-content` skip is a rendering diagnostic for the selected plot
input. It does not establish the content of a bin, a statistical conclusion, or
the adequacy of a fitting view.
# Persistent SR CR-like plotting campaigns

For a campaign, `-o ROOT` is the **final canonical output root**. The controller
publishes verified plots under `<ROOT>/<period>/merged/<category>/<filename>.png`
and, where the layout owner requires it, `merged_njets` beside `merged`. Private
`components/` directories are automatic staging and are not the final plot
destination. Execution completion and publication are separate: a finished
Condor job can show `completed / publish pending` while persisted campaign state
still says `running`. For an already terminal existing child, `--finalize` itself
verifies the bound scheduler and lifecycle evidence, records terminal success,
and creates or verifies the completion receipt in one invocation. A preparatory
`--resume` is not required; `--resume` remains for recovery and resumption.
`--finalize` then copies the observed plots without overwriting another owner's
bytes. The controller uses `topeft/modules/plot_output_layout.py` for final paths.

For an existing campaign, use the read-only status command:

```bash
./analysis/topeft_run2/run_plotter.sh --status -o /absolute/path/to/topeft/histos/my_campaign
```

For a current-content check, add `--verify-published` to that status command.

Add `--selected-unit UNIT_ID` to report one existing unit. Its counts, status,
rows and optional `--observe-scheduler` / `--verify-published` checks cover only
that unit; campaign identity, plan revision, root and future execution policy
remain available. An unknown unit fails before scheduler or content checks.
Omitting the selector preserves campaign-wide status and checks.

For one live scheduler observation, add `--observe-scheduler`. This explicitly
queries each bound Condor job in the status scope, then history on a queue miss. It
does not reconcile, publish, submit, retry, or write recovered scheduler IDs.
Missing or contradictory evidence requires attention; HELD remains nonterminal.
Human output labels persisted lifecycle counts as `Campaign state (persisted)`
and shows a separate `Scheduler observed` aggregate computed from the same
per-unit observations as the scheduler rows. A persisted Running count of 10
can coexist with COMPLETED 9 and RUNNING 1: status does not reconcile state.
`OTHER/UNKNOWN` includes all observed states outside COMPLETED, RUNNING, IDLE
and HELD, including unresolved observations. Only units with an observation
contribute to that aggregate. The `CAMPAIGN VIEW` column retains the controller's
execution view, which can also reflect an existing lifecycle receipt.

The scheduler row includes HELD/removal reasons, exit/signal evidence, and
`MEM_MB` and `RSS_KIB` (`ResidentSetSize`) as distinct quantities. Human `MEM_MB`
uses numeric `MemoryUsage` when available; otherwise it rounds numeric RSS in
KiB up to whole MiB (1024 KiB), or shows `-` when neither is usable. ClassAd
expression text is never evaluated. Zero is a valid numeric value.
JSON and YAML retain the requested raw ClassAd attributes under each unit's
`scheduler.raw_classad` and the existing raw `mem_mb`/`rss_kib` values; scheduler
completion alone never establishes publication. Machine campaign count fields
continue to describe persisted controller state.
Scheduler reads require the relevant site/network authorization.

Refresh externally when those reads are authorized:

```bash
watch -n 5 -t './analysis/topeft_run2/run_plotter.sh --status -o /absolute/path/to/topeft/histos/my_campaign --observe-scheduler'
```

The former `plot_campaign_condor.py monitor --diagnostics-dir ...` route returns
a migration error. Use the persistent canonical root with the status command
above and `--format json` for machine callers. Historical TSV submission maps
remain evidence and are not persistent campaign roots; they are not silently
converted. Ordinary status remains offline.

It reads campaign identity from the root and shows the exact tag, plan
revision, final root, separate execution, publication and integrity states,
recorded job IDs, and action-needed warnings. Human status distinguishes
historical backends from the policy bound for future attempts, or says that a
legacy campaign remains unbound. Ordinary status derives publication from
persisted success, receipt, and materialization provenance, then checks current
canonical presence and recorded sizes without hashing PNG content.
Add `--verify-published` to hash recorded canonical PNGs against their stored
SHA-256 values. It is read-only, and historical publication remains `published`
even if a current file is missing, truncated, or unsafe; its recorded canonical
path remains listed, while current integrity reports `unavailable` or
`content_mismatch`. `--format yaml` gives a bounded readable
summary with exact published plot paths. `--format json` is the stable machine
route and retains the established count/status fields. Status does not contact
the scheduler or modify campaign files.

Condor campaigns can submit independent planned units in parallel. A normal
public `--execute` submits the selected planned units, performs one initial
readback/reconciliation for each submitted attempt, prints current status and a
copy-pastable monitoring command, then exits even while jobs remain active or
unresolved. A normal `--resume` on a policy-bound Condor root reconciles every
selected active unit, records proven terminal success as publish pending,
submits every eligible planned unit once, reads back the newly submitted
attempts once, and exits. Terminal nonzero results become failed attempts,
including workers held by OnExitHold with a verified failure receipt. `--resume`
selects the existing guarded recovery for eligible failed or interrupted attempts. A failed Condor
attempt may create only one replacement after its nonzero lifecycle and retired
old scheduler identity are verified; a proven local submit rejection may rearm
only the same attempt. Ambiguous evidence never authorizes retry. `--finalize`
alone publishes proven terminal success. Local execution remains foreground
and sequential.

From the TOPEFT checkout with the intended Conda environment already active,
use the printed command for a separate, read-only scheduler observation:

```bash
./analysis/topeft_run2/run_plotter.sh \
  --status -o "$ROOT" --observe-scheduler
```

The handoff uses the relative launcher and shell-quoted campaign output root.
It is printed on stderr for JSON/YAML so stdout remains machine-readable.
Status never reconciles or publishes. The existing `--follow` option explicitly
opts into foreground observation and reconciliation; only this option polls.
It prints changed status and releases the lock between polls. Ctrl-C stops
observation without cancelling jobs. `--poll-seconds N` (1–300, default 15)
controls this optional cadence. `--selected-unit` remains a bounded action;
follow uses the same bounded resume recovery rules and stops when only
terminal-success finalization remains.

```bash
./analysis/topeft_run2/run_plotter.sh --resume -o /absolute/path/to/topeft/histos/my_campaign
```

For a machine caller, request JSON. A single controller pass is the default:

```bash
./analysis/topeft_run2/run_plotter.sh --resume -o /absolute/path/to/topeft/histos/my_campaign \
  --format json
```

An existing root supplies profile and tag. A fresh campaign requires a qualified
plan, profile, tag and root. It binds the resolved future-attempt policy once:
local, `cgroup_v2`, and `245000000000` bytes by default. A policy-bound root
reuses it on resume; matching explicit values are accepted and contradictory
values fail before dispatch. Legacy roots without this field can run status or
reconcile a selected completed/running unit without resource flags. To launch
a planned unit on a legacy root, explicitly select `--backend local` or
`--backend condor`. Local resolves missing resource values from the fresh
defaults. Condor requires `--backend condor`, `--resource-observer` and
`--hard-limit-bytes` together once; they are stored before the first submit.
A selected unit that cannot launch does not inherit resource requirements from
unrelated planned units. Historical attempt contracts are unchanged. Explicit
profile or tag values must agree with the root.

For the first planned Condor launch on a legacy root, supply the chosen policy
once; later resumes need only the root:

```bash
./analysis/topeft_run2/run_plotter.sh --resume -o /absolute/path/to/topeft/histos/my_campaign \
  --backend condor --resource-observer cgroup_v2 --hard-limit-bytes 192000000000
```

The three persistent CR-like profiles of `analysis/topeft_run2/run_plotter.sh` are:
`sr_crlike_run2`, `sr_crlike_run3`, and `sr_crlike_run2_run3`. A fresh plan is a
JSON object with `schema_version: 1`, `campaign_profile`, `campaign_tag`,
`plan_revision: 1`, and a nonempty `units` array. Each unit has exactly
`unit_id`, `component` (`run2` or `run3`), `producer_unit_id`, absolute
`input_path`, absolute `sidecar_path` (`input_path + ".metadata.json"`), a
`qualification_source` object, and `period_view`
(`combined` or an individual year token). Combined plans contain both
components. `qualification_source` has `status: "accepted"`, a nonempty
`locator` for the accepted producer return, and accepted `input_sha256` and
`sidecar_sha256` values. The controller checks those values against the bound
sidecar without rehashing the PKL. Unit IDs and tags are safe path components. The sidecar's
`sumw2_storage_provenance.runtime_histogram_families` supplies variable names
unchanged; its `artifact` binds the PKL basename, accepted size, and accepted
SHA-256. The planner checks input existence and size but does not deserialize
or rehash the large PKL. The qualification locator must identify an actually
accepted producer return; the launcher checks its declared identity but does
not establish scientific qualification from the plan's claim alone.

Use a fresh absolute root beneath `topeft/histos`:

```bash
./analysis/topeft_run2/run_plotter.sh --campaign-profile sr_crlike_run2_run3 \
  --campaign-tag my_campaign -o /absolute/path/to/topeft/histos/my_campaign \
  --plan /absolute/path/to/plan.json --dry-run
```

The dry run prints all bound units, variables, accepted identities, first
private attempt roots, and resolved child arguments. It creates no campaign
root/state, reads no PKL content, and does not predict categories, filenames,
PNG counts, or a variable by region coverage matrix. Add `--selected-unit ID`
to show or execute one unit while preserving the complete plan. Replace
`--dry-run` with `--execute` only when you are authorized to run the campaign
in that environment, after verifying its plan, inputs, output root, and
resources. Fresh execution requires the root to be absent. For an existing
campaign use `--resume` with the existing root. It reconciles active attempts,
skips published successes, and runs eligible planned units. A possibly live
local child blocks other local launches. Independent Condor attempts do not
block other planned Condor units. A terminal failed or interrupted unit does
not block an eligible planned unit elsewhere. An ineligible or ambiguous attempt
is never retried. A successful controller invocation exits zero
even when units remain planned, failed, or interrupted. Its JSON summary reports
`campaign_status` (`complete`, `partial`, `in_progress`, or `waiting`), total
`planned_count`, per-status counts, and `waiting_on_active_child`. The derived
`active_child_present` and `dispatch_blocked` fields separate job presence
from dispatch blocking; `waiting_on_active_child` means dispatch is blocked.
Only the local backend blocks other launches. Malformed state or plan still
makes the invocation fail.

Each attempt writes only below its deterministic private
`.campaign_units/<unit_id>/attempt_0001/` root. The local memory guard and
visible tee record terminal status and renderer messages there. After terminal
guard, child, and tee results, the supervisor atomically writes
`lifecycle_receipt_v1.json` before updating campaign state. Resume reconstructs
terminal state from that receipt without rerunning the child. If it is missing
but guard evidence conclusively shows the child ended, resume marks the attempt
interrupted. Without conclusive terminal evidence, it remains running; local
work waits, while independent planned Condor units may start. After terminal
success, resume leaves terminal postprocessing pending. `--finalize` creates
or verifies the separate atomic `completion_receipt_v1.json`, which records
the exact observed PNG paths, counts, sizes, SHA-256 values, and counts by
requested variable. Sparse or zero output for a variable in a region is
ordinary renderer applicability; the controller does not call that absence a
failure. Renderer log presence and
message content are review evidence: a regular log is scanned in full and a
bounded message summary is stored; a missing log is recorded explicitly in
`renderer_log_status`. Neither changes successful guard/child/tee lifecycle
status. The receipt proves operational child completion, not visual or
scientific acceptance.

The controller makes no Git or runtime source-manifest check. New state records
that source provenance was not collected. Existing campaign provenance remains
historical evidence and is internally bound across coordinator, component state,
and receipts, but it is never compared with the current checkout. Git and source
drift are review concerns and do not prevent safe resume or revision.

Receipt-listed PNGs are copied without overwrite below the global campaign
root as `<period>/merged/<category>/<figure>.png` (and `merged_njets` when
required by the layout owner). A combined campaign keeps separate private
`components/run2` and `components/run3` attempt roots and state files. The state binds
each destination to its unit, receipt, size, and SHA-256 before copying it.
Finalize can finish missing copies from a bound receipt without rerunning the
child. Same-byte canonical destinations can be rebound to a verified receipt;
different-owner or different-byte destinations stop the affected component.
Directory existence alone never means success.

For a terminal attempt whose guard records normal exit zero but whose supervisor
never wrote a lifecycle receipt, use `--resume --selected-unit UNIT_ID` after
checking that the managed process is gone. The existing recovery proof requires
the bound guard, complete observed PNG inventory, and renderer completion
summary. Resume writes distinct recovery provenance and leaves publication
pending; the missing lifecycle receipt remains missing. `--finalize` then
creates or verifies the completion receipt with unavailable guard/child/tee
results left null and materializes the canonical PNGs. It never starts a
renderer or advances planned units. For an already successful legacy unit,
finalize also reuses the existing verified component-to-canonical PNG migration
when its old human-output binding remains. Private attempts and logs stay put.

Use `--finalize -o ROOT` for all eligible terminal-success units, or add
`--selected-unit UNIT_ID` to require one eligible unit. A stale persisted
`running` unit is finalized only when its existing attempt passes the bound
terminal-success proof. Planned, actively running, failed, and ambiguous units
are not finalized. An already published unit is left alone unless its verified
legacy output binding needs migration. There
is no separate `--publish` or `--reconcile` action.

To append formally qualified units, submit a full plan whose revision is one
greater and whose existing units are byte-equivalent in their source fields:

```bash
./analysis/topeft_run2/run_plotter.sh --campaign-profile sr_crlike_run2_run3 \
  --campaign-tag my_campaign -o /absolute/path/to/topeft/histos/my_campaign \
  --activate-pending-plan /absolute/path/to/revised_plan.json \
  --expected-plan-revision 1
```

The guarded revision preserves existing unit states, attempts, receipts, and
materialization history. A mismatch or attempted rewrite stops. Revisions do
not run a child; `--resume` executes newly planned units. Direct single-PKL
use remains available through controller dispatch. The former TSV interface is
retired; persistent JSON plans are the only campaign plan language.

The low-level guard CLI retains its explicitly named `node_pressure` compatibility
default. Campaign backends always pass the selected observer explicitly; the public
campaign default is `cgroup_v2`. No fallback follows a failed cgroup probe.

### Plan a separate datacard campaign

Use `--campaign-profile sr_datacard_run2_run3` with a new campaign tag and an
absolute fresh output root to keep datacard inputs separate from CR-like
campaigns. Bind five qualified inputs per run in the reference campaign plan;
use `--dry-run --plan /absolute/path/to/plan.json` to validate it without creating
state. The single-run profiles are `sr_datacard_run2` and `sr_datacard_run3`.
The combined profile preserves each run's durable state and failure boundary.
A fit-target manifest and its reviewer mask select future delivered figures;
extra native renderer outputs remain technical byproducts. Validate required
fit variables as a subset of sidecar families, without overriding native variables.
