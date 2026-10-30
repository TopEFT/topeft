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
`--debug-canvas-annotations` as an explicit diagnostic opt-in; signed yields,
uncertainties and separate negative-weight reports do not depend on it. A mixed Run 2
and Run 3 year selection is rejected.

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

Use the persistent campaign interface below with `--backend local` or
`--backend condor`. Both execute exactly one controller-selected unit at a time.
The public wrapper is never a batch payload. The Condor adapter sends a single
immutable renderer command to `plot_campaign_backend.py --worker`; it has no
standalone campaign CLI, plan parser, scientific view selection or output roster.
The former TSV campaign interface and standalone Condor launchers are retired.
Use the qualified persistent JSON plan, not a converted expected-output roster.

Local and Condor workers use the same guard and lifecycle evidence. The controller
alone interprets receipts and materializes observed figures. Condor submission
returns a waiting campaign; a later explicit resume observes that same execution.
A nonzero, timed-out or ambiguous submit response does not prove that no job was
created. It never causes resubmission or a local fallback. Missing scheduler
identity requires separate reconciliation evidence. A worker start marker prevents
rerender after scheduler eviction/restart; held jobs remain unresolved.

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
Condor job can show `completed / publish pending` until the controller verifies
its receipt and copies the observed plots without overwriting another owner's
bytes. The controller uses `topeft/modules/plot_output_layout.py` for final
paths.

For an existing campaign, use the read-only status command:

```bash
./analysis/topeft_run2/run_plotter.sh --status -o /absolute/path/to/topeft/histos/my_campaign
```

It reads campaign identity from the root and shows the exact tag, plan
revision, final root, unit execution and publication states, recorded job IDs,
and action-needed warnings. `--format yaml` gives a bounded readable summary
with exact published plot paths. `--format json` is the stable machine route and
retains the established count/status fields. Status does not contact the
scheduler or modify campaign files.

Condor campaigns can submit independent planned units in parallel. A normal
public `--resume --backend condor` reconciles every active unit, publishes each
valid terminal success, submits every eligible planned unit once, and stays in
the foreground to observe further completions. It prints only changed status
and releases the campaign lock between polls. Ctrl-C stops observation without
cancelling jobs; the same resume command continues their existing attempts.
Use `--poll-seconds N` (1–300, default 15) to change observation cadence. Local
execution remains sequential. `--selected-unit` is an advanced bounded action,
and neither normal resume nor follow retries failed or interrupted units.

```bash
./analysis/topeft_run2/run_plotter.sh --resume -o /absolute/path/to/topeft/histos/my_campaign \
  --backend condor --resource-observer cgroup_v2 --hard-limit-bytes 192000000000
```

For a machine caller, request one JSON pass explicitly:

```bash
./analysis/topeft_run2/run_plotter.sh --resume -o /absolute/path/to/topeft/histos/my_campaign \
  --backend condor --resource-observer cgroup_v2 --hard-limit-bytes 192000000000 \
  --format json --one-pass
```

An existing root supplies profile and tag. Resume still needs `--backend` when
historical attempts do not identify one unambiguously. When planned Condor
units remain, give `--resource-observer` and `--hard-limit-bytes` explicitly:
attempt records bind earlier jobs but do not declare a campaign-wide resource
policy for new units. Explicit profile or tag values must
agree with the root. A fresh campaign still requires its qualified plan,
profile, tag, root, and runtime choices.

`analysis/topeft_run2/run_plotter.sh` owns three persistent profiles:
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
not block an eligible planned unit elsewhere;
it is never silently retried. A successful controller invocation exits zero
even when units remain planned, failed, or interrupted. Its JSON summary reports
`campaign_status` (`complete`, `partial`, `in_progress`, or `waiting`), total
`planned_count`, per-status counts, and `waiting_on_active_child`. An unresolved
possibly live child returns `waiting`; only the local backend blocks other
launches. Malformed state or plan still makes the invocation fail.

Each attempt writes only below its deterministic private
`.campaign_units/<unit_id>/attempt_0001/` root. The local memory guard and
visible tee record terminal status and renderer messages there. After terminal
guard, child, and tee results, the supervisor atomically writes
`lifecycle_receipt_v1.json` before updating campaign state. Resume reconstructs
terminal state from that receipt without rerunning the child. If it is missing
but guard evidence conclusively shows the child ended, resume marks the attempt
interrupted. Without conclusive terminal evidence, it remains running; local
work waits, while independent planned Condor units may start. After terminal
success, a separate atomic `completion_receipt_v1.json`
records the exact observed PNG
paths, counts, sizes, SHA-256 values, and counts by requested variable. Sparse
or zero output for a variable in a region is ordinary renderer applicability;
the controller does not call that absence a failure. Renderer log presence and
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
Resume can finish missing copies from a bound receipt without rerunning the
child. Same-byte canonical destinations can be rebound to a verified receipt;
different-owner or different-byte destinations stop the affected component.
Directory existence alone never means success.

For a terminal attempt whose guard records normal exit zero but whose supervisor
never wrote a lifecycle receipt, use the separate `--recover-terminal-unit
--selected-unit UNIT_ID` action after checking that the managed process is gone.
It never starts a renderer or advances planned units. It requires the bound
guard, complete observed PNG inventory, and renderer completion summary, then
writes distinct recovery provenance and a completion receipt with unavailable
guard/child/tee lifecycle results left null. The missing lifecycle receipt
remains missing. The same action migrates successful human PNG copies from
component-relative paths, verifies canonical byte identity, rebinds state to
the global root, and removes only verified redundant old PNG copies. Private
attempts, logs, state, and receipts stay put.

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
