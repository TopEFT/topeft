# Create cards and build datacard packages

For the complete package lifecycle and fit-input handoff, use
[build and use datacard packages](datacard_packages.md). This page covers card
production and matrix execution in detail.

`analysis/topeft_run2/make_cards.py` produces cards directly from one or more
compatible histogram PKLs—normally the nonprompt-transformed products—and
writes individual text cards, ROOT template files, `selectedWCs.txt`, and
`scalings-preselect.json`. For multiple production rows, generate a matrix
manifest and use `analysis/topeft_run2/run_datacard_matrix_resumable.sh` to
run and resume it.

| Interface | Purpose | Inputs and defaults | Outputs and checks |
| --- | --- | --- | --- |
| `make_cards.py` | Merge histogram inputs and select WCs, channels, and variables | Fitting binning, year coverage `warn`, Asimov data, no nuisances or MC-stat opt-in | Direct mode writes TXT/ROOT and scaling records; `--condor` submits matrix dispatch; `--condor --prepare-only` writes material without submission |
| `make_datacard_matrix_manifest.py` | Prepare standard Run 2 or Run 3 execution units | Era, `--input-pkl ROLE=PATH` bindings, runtime paths, and fresh manifest path | Writes a `topeft_datacard_matrix_v3` manifest; does not produce cards |
| `build_per_era_datacard_package.py build` | Build one Run 2 or Run 3 package | Era, one or more matrix manifests with completed rows, and a fresh absolute output | Requires exactly the targets declared by the manifest rows, checks recorded source files, and writes cards, WCs, scalings, and per-era `chN` |
| `build_combined_datacard_package.py build` | Build a cards-only Run 2 + Run 3 package | Two per-era packages with the same physical target set, fresh absolute output, analysis, package date, and version | Writes the combined mapping, ordered card list, and scalings; checks the package against its inputs; does not run Combine |
| EFTFit/Combine | Combine cards and construct the statistical model | Packaged cards and scalings plus fit configuration | Creates `combinedcard.txt` and the workspace after package construction |

The region → distribution → binning choices come from repository configuration:
physical regions and jet populations are defined in `topeft/channels/ch_lst.json`;
the standard matrix profile assigns each region's card distribution (`lj0pt`,
`ptz`, `ptll`, `ptz_wtau`, or `lt`); `topeft/modules/axes.py` supplies processing/fitting
edges. A matrix manifest records the selected production rows; it does not
redefine the region or binning configuration.

## Path 1: create cards and templates directly

This path needs PKLs and `make_cards.py`, with no matrix manifest. It suits
individual or interactive card production. Per-era packaging uses the matrix
workflow described below.

From `analysis/topeft_run2`:

```bash
python make_cards.py /path/to/final_np.pkl.gz \
  --out-dir /absolute/path/to/cards \
  --var-lst lj0pt ptz ptll ptz_wtau lt \
  --ch-lst '^2lss_.*' '^3l_.*' '^4l_.*' \
  --binning fitting --year-coverage-policy error
```

Repeat positional PKLs for coherent fragments, or use `--pkl-list-file` for a
long list; do not provide both. The default uses Asimov data, fitting binning,
no nuisance insertion, and warning-only year coverage. `--unblind`,
`--do-nuisance`, `--do-mc-stat`, and `--keep-negative-bins` are explicit
choices. Both modes write `data_obs`; its presence in a ROOT file does not
identify whether observations came from data or Asimov construction.
`--merge-only` stops after load/merge validation, and
`--merge-report PATH` retains the diagnostic report.
The optional `--lepton-mva-correlation {per-run,cross-run}` defaults to
`per-run`. It changes only the final electron and muon MVA nuisance identities
in the TXT card and ROOT keys; see the [nuisance reference](../reference/datacards_and_scalings.md).
`cross-run` writes unsuffixed `lepSF_elec_mva` and `lepSF_muon_mva` names in
both runs; `per-run` writes the corresponding run-energy suffixes.
For matrix work, select the same option when generating each canonical manifest;
its rows carry the choice in `producer_args`, including the explicit default.
Older v3 rows without this option still mean `per-run` under their historical
producer; replay also remains subject to the existing runtime fingerprint checks.
`--suppress-nominal-like-tau-shapes` is a separate opt-in, disabled by default.
With nuisances enabled, it omits eligible Tau Up/Down template pairs only when
both directions meet the writer's fixed, strict relative `1e-5` comparison
with nominal in every nonzero regular bin. The suppressed pair is absent from
the ROOT templates and its process/category entry in the text card is `-`.
Other applications keep their nuisance names and correlations; no suppressed
shape is converted to `lnN`. Other shapes, including protected Run 2
genuine-tau energy-scale variations, remain in place. The option is
unavailable with `--use-AAC`; see the
[reference](../reference/datacards_and_scalings.md#final-nuisance-identities)
for the exact eligibility and zero-bin conditions.

For direct card production from `analysis/topeft_run2`:

```bash
python make_cards.py /path/to/final_np.pkl.gz \
  --out-dir /absolute/path/to/cards \
  --var-lst lj0pt --ch-lst '^2lss_.*' \
  --do-nuisance --suppress-nominal-like-tau-shapes
```

Before card production, confirm that every input belongs to one compatible
production family and has the nominal/sumw2 companions required by the selected
card channels and variables. A filename or campaign directory is not
compatibility evidence.

`make_cards.py` validates and merges all PKLs before constructing
`DatacardMaker`. It rejects positional inputs combined with `--pkl-list-file`,
mixed incompatible schemas/policies, and missing required companions.
`--merge-only` is the lowest-cost way to exercise that boundary. A cached
merged PKL written with `--cache-merged-pkl` is a new artifact and sidecar when
the input schema supports it; preserve its lineage and merge report.

Channel arguments are selection patterns interpreted by the existing
`regex_match` helper against actual histogram channel labels. They do not add a
channel to the registry or repair an absent artifact category. Variables must
exist as histogram families in the merged input.

## Modify card selection or configuration

- Select variables and channels through `--var-lst` and `--ch-lst`; keep
  physical channel definitions in `topeft/channels/ch_lst.json`.
- Choose processing or fitting edges through `--binning`; change definitions at
  `topeft/modules/axes.py`, following the [binning guide](flexible_binning.md).
- Use `--miss-parton-file` for an exact missing-parton payload override. The
  card consumer infers its maintained layout from the payload; it has no
  `--sr-registry` option. The separate payload producer selects its output
  layout with `--missing-parton-layout-key`.
- `--rate-syst-json` overrides the run-era rate-systematics JSON path. An
  explicit value is forwarded to `DatacardMaker` as `rate_systs_path`; when it
  is omitted, `DatacardMaker` selects its maintained Run 2 or Run 3 default.
- `--use-selected FILE` reads the reviewed JSON for card construction,
  canonicalizes its signal-only representation, and materializes that
  representation as `<out_dir>/selectedWCs.txt` without modifying `FILE`.
- When extending the CLI, update parsing, canonical manifest arguments where
  applicable, `DatacardMaker` construction, output/provenance behavior, and
  focused tests.

`make_cards.py --condor` submits execution units from an existing canonical
v3 manifest before any PKL loading or producer setup. Put producer options in
that workload; the frontend accepts only manifest, dispatch destination,
resource, and row-selection options. Local and worker execution pass the same
recorded producer arguments through the matrix runner. The former Condor option
reconstruction and `--chunks` interface have been retired; execution
partitioning remains owned by the profile and manifest generator.

Add repeatable `--row-id ROW_ID` to dispatch only those original
rows; omission retains whole-manifest dispatch. Duplicate or unknown IDs are
rejected before dispatch output is created. `--prepare-only` remains an
ownership-free no-submit inspection, including with row selection.

For one selected row whose prior Condor submission has an ambiguous outcome,
`--rescue-ambiguous --row-id ROW_ID` explicitly transfers its reservation to a
fresh token under the row lease. An active scheduler job or held producer lease
blocks rescue. An empty or unavailable scheduler query does not prove that the
old job terminated; the old token is fenced before the new submission attempt.
If the new submit outcome is uncertain, retain its reservation and inspect H2
state before any further operator action. A valid H2 receipt is the only row
completion authority. The manifest and its pinned producer fingerprints must
remain unchanged for that campaign; a stale producer fails before reservation.

To add a supported selection/configuration control:

1. Identify its existing owner: physical channels in `ch_lst.json`, axes in
   `axes.py`, currently default-selected rate-systematic JSON in
   `DatacardMaker`, missing-parton payload through its exact-path option,
   or WC selection through selected-WC inputs. Use
   `--rate-syst-json` only to select an existing supported rate-systematics
   JSON.
2. Add a CLI selector only when choosing among existing supported authorities;
   do not copy the configuration into `make_cards.py`.
3. Validate choices before output creation and thread the resolved value to
   `DatacardMaker` once.
4. Preserve local/Condor equivalence or explicitly fail a mode that cannot
   represent the option.
5. Update `tests/test_make_cards_multi_pkl.py`, the focused option/physics
   contract test, `tests/test_split_datacard_boundary.py`, and late-rebin or
   selective-sumw2 tests when those surfaces are affected.

Card changes can affect template shapes, nuisance content, WC selection, the
preselected scaling records, and every later EFT fit. Validate the card/template
pair together rather than checking the text card alone.

## Path 2: run a datacard matrix resumably

For standard multi-row production and packaging, bind each input role in the
selected era's profile to a new PKL. Run 2 uses `block1` through `block5`;
Run 3 uses `2l_mixed`, `3l_m_offz`, `3l_p_offz`, `3l_onz_tau`, and `3l_fwd`.
These roles reflect the PKL blocks supplied to the standard rows. From the
repository root, a Run 3 invocation is:

```bash
python analysis/topeft_run2/make_datacard_matrix_manifest.py \
  --era run3 \
  --input-pkl 2l_mixed=/absolute/path/to/2l-mixed.pkl.gz \
  --input-pkl 3l_m_offz=/absolute/path/to/3l-m-offz.pkl.gz \
  --input-pkl 3l_p_offz=/absolute/path/to/3l-p-offz.pkl.gz \
  --input-pkl 3l_onz_tau=/absolute/path/to/3l-onz-tau.pkl.gz \
  --input-pkl 3l_fwd=/absolute/path/to/3l-fwd.pkl.gz \
  --python-executable /absolute/path/to/python \
  --missing-parton-file /absolute/path/to/missing-parton.root \
  --lepton-mva-correlation per-run \
  --runtime-contract-id run3-cards-001 \
  --output-root /absolute/path/to/row-outputs \
  --control-root /absolute/path/to/runner-control \
  --manifest-output /absolute/path/to/run3-matrix.json
```

For Run 2, use `--era run2`, five `--input-pkl` bindings named `block1` through
`block5`, and the Run 2 missing-parton file. The helper combines the maintained
channel registry with the human-authored
`analysis/topeft_run2/datacard_matrix_profiles.yml`, selects the era's years,
and writes a `topeft_datacard_matrix_v3` JSON manifest. The full profile has 11
logical rows in either era. Its maintained execution partition produces 11
Run 2 units and 34 Run 3 units. A logical row may map to several execution
units; each keeps its input role and distribution, and their physical channels
have exactly the logical row's union. The generated units select channels in
`make_cards.py` with `--ch-lst` and use year coverage `error`.
It records the selected Python interpreter and hashes `make_cards.py`; repeat
`--runtime-file` for other files to include in the runner's runtime check.
The manifest path must be new. For a maintained restricted fit, use
`--channel-set-key OFFZ_SPLIT_CH_LST_SR`. For a one-off subset, repeat
`--physical-target <channel>_<distribution>` as an advanced generator option.
These manifest filters choose production targets; the generated `--ch-lst`
arguments select channels in `make_cards.py`, while
`--missing-parton-layout-key` selects only the separate payload producer's
output layout.
Only PKL roles used by the selected rows are required. Make the same physical
subset in the Run 2 and Run 3 manifests before running either matrix: Run 2
manifest X and Run 3 manifest X produce packages with the same surface X.

The generator records nuisance and MC-stat insertion, the chosen MVA mode,
years and channel selection in each row's producer arguments; the input PKL
is recorded separately as `input_pkl`. Add
`--suppress-nominal-like-tau-shapes` to the generator when that
opt-in card policy is intended for every row. Add `--unblind` only when the
selected PKLs contain the observed `data` process for every selected channel;
the generator forwards it to every row in that manifest. Without it, each row
uses Asimov observations. Keep independently generated manifests, row outputs,
and package inputs consistent with the intended observation mode when
assembling packages; the package builders do not choose that mode for you.

The current manifest uses the `topeft_datacard_matrix_v3` schema. Each
execution-unit row records its logical row ID, input PKL, literal channel
arguments, distribution, years, output paths,
`make_cards.py` arguments, and log/snapshot locations. The runner checks the
declared runtime files before execution and writes successful row JSON receipts;
the analyst does not create receipt files. Direct `make_cards.py` use does not
require a matrix manifest.

`topeft_datacard_matrix_v2` is a historical schema, not a corrupt manifest.
The current datacard workflow does not accept it. Generate a new v3 manifest
with `make_datacard_matrix_manifest.py`; there is no migration or conversion
path.

A minimal invocation is:

```bash
analysis/topeft_run2/run_datacard_matrix_resumable.sh --plan-only /path/to/manifest.json
analysis/topeft_run2/run_datacard_matrix_resumable.sh --status /path/to/manifest.json
analysis/topeft_run2/run_datacard_matrix_resumable.sh /path/to/manifest.json
```

The public wrapper locates its companion Python engine relative to its own
absolute directory, so it does not depend on the checkout location or the
caller's current working directory. It prefers `python` and falls back to
`python3` for the runner engine bootstrap.

Tip: A persistent terminal session such as `tmux` can be useful for
long-running jobs that should continue after the terminal disconnects.
Inspect `--plan-only` and
`--status` first, and keep the manifest unchanged during an
attempt. The runner validates the schema and runtime fingerprints before plan
or execution, acquires ownership for each execution unit, and directly runs
the declared Python argv with no shell intermediary. Sequential orchestration
stops on the first
command or output-check failure.

After a successful row command, the runner checks the declared outputs, retains
the row-specific merge report, snapshots `selectedWCs.txt`,
`scalings-preselect.json`, and the merge report, hashes the primary TXT/ROOT
outputs and every row-local control artifact (merge report, snapshots, and log).
Producer writes go first into a private numbered physical-attempt directory
beside the canonical row output. After validation the runner writes an
attempt-local completion intent, publishes canonical files without overwriting
existing bytes, verifies the published sizes and SHA-256 values, then publishes
the execution receipt. Receipt validation rechecks both size and SHA-256;
these checks do not assess the physics content.

On restart, a row is skipped only when its receipt matches the current manifest,
argv, paths, runtime contract, and byte-bound artifacts. A valid immutable
receipt permits that skip even if the row's historical input PKL or
missing-parton file is no longer available. Rows that are about to execute
still require the current runtime contract, a readable regular input PKL, and
a readable regular missing-parton file. The manifest binds
`missing_parton_path` to exactly one matching `--miss-parton-file` producer
argument, so the checked file is the file passed to `make_cards.py`.

`--plan-only` does not mutate or execute. For each `not_started` row it reports
runtime and execution-input availability and marks the overall plan not
launch-ready when a required input is unavailable; completed rows are not
penalized for unavailable historical execution inputs.

The manifest `attempt_id` remains the public receipt and package
`--selected-unit` identity. Physical attempt numbers are internal runner
retries under that identity. There is no `--resume` flag and no automatic retry
within a failing invocation; passing the same manifest again is the explicit
rerun/reconciliation action. A valid receipt skips without a new physical
attempt. An incomplete attempt is preserved, receives no completion authority,
and a later explicit invocation uses a new noncolliding directory when the
canonical row directory remains absent.

Owner metadata is updated atomically under the unit's output-domain lock.
`started_at` and `current_row_started_at` describe that unit's ownership and
physical producer start. `--status` reports `active` when the held lock and
owner metadata identify the row. Different units may be active together;
local, manual and Condor attempts on the same canonical output root exclude
each other. Owner metadata carries backend and optional scheduler identity as
observations. The dashboard renders every active unit and its staged log.
A free lock makes matching owner metadata stale; rerun still passes through
the runner's transaction classification and logical reservation check. A live
Condor job keeps its logical reservation even when no process holds this lock.

Before each row launch, the held runner rechecks the runtime contract and
freshly classifies the row. `not_started`, `stale_owner_free_lock`, and
`partial_staging_attempt` can launch a new physical attempt; valid receipts
skip. `complete_validated_staging_not_promoted` is revalidated and published
without running the producer. `promoted_incomplete` completes only missing
files whose existing bytes match the attempt intent.
`promoted_complete_without_receipt` publishes the missing receipt only after
all canonical and control bytes match the validated attempt. A stale owner
cannot override a held lock. An invalid receipt, unmarked canonical directory,
mismatched bytes, malformed intent, or unknown lock ownership blocks with
`invalid_receipt`, `canonical_collision_or_ambiguous_state`, or
`held_lock_active_or_unknown`. Preserve these bytes and investigate the
reported paths; do not delete them or rely on file counts. A crash between
creating the canonical directory and linking its attempt marker is intentionally
ambiguous and fails closed. A runtime fingerprint change blocks before the next
producer launch; reconciliation uses its recorded successful runtime binding.

### Execute one unit or prepare Condor material

The global v3 manifest defines workload topology. Each existing manifest row
is one execution unit. A derived `topeft_datacard_execution_unit_v1` descriptor
contains `parent_manifest`, `parent_integrity`, and `row_id` alongside `schema`.
It references the original manifest; it contains no independent channel plan.
The integrity hash detects changed dispatch input. Public row/attempt and
receipt/package bindings continue to use the original manifest bytes.

Whole-manifest local execution is a sequential loop over the same unit
executor used by workers. To inspect or execute a single existing row, use its
exact ID from the manifest:

```bash
analysis/topeft_run2/run_datacard_matrix_resumable.sh --derive-units /path/to/manifest.json
analysis/topeft_run2/run_datacard_matrix_resumable.sh --plan-only --row-id "$row_id" /path/to/manifest.json
analysis/topeft_run2/run_datacard_matrix_resumable.sh --row-id "$row_id" /path/to/manifest.json
```

This scheduler contract covers TopEFT datacard production and H2 publication.
EFTFit/Combine runtime and fit scheduling belong to the package consumer.

Submit from a fresh dispatch directory after choosing resources and retry/start
budgets for your site and workload. The frontend uses the installed HTCondor
configuration for default scheduler discovery; it has no schedd/pool flags.
Set the resource and budget variables below explicitly; the software supplies
no production resource or retry defaults:

```bash
python analysis/topeft_run2/make_cards.py --condor \
  --matrix-manifest /absolute/path/to/manifest.json \
  --condor-output /absolute/path/to/fresh_dispatch \
  --request-cpus "$request_cpus" \
  --request-memory "$request_memory" \
  --request-disk "$request_disk" \
  --max-retries "$max_retries" \
  --max-starts "$max_starts"
```

This writes `units/*.json`, `items.txt`, `worker.sh`, `datacards.sub`, and an
empty `logs/` directory, then invokes the installed `condor_submit` through
the maintained reservation transaction. Each queue item identifies exactly one unit. The
worker enters `datacard_matrix_runner.py --unit DESCRIPTOR`; its producer runs
without `--condor`. Add `--prepare-only` to the command above for a no-submit
inspection run. Material generation is ownership-free: prepare-only creates no
row reservation, queries no scheduler, and cannot block later local execution.
Existing dispatch destinations are rejected. Items initially carry an
`unreserved` placeholder, which cannot authorize a worker. Already complete
rows retain their queue item and skip production by valid H2 receipt.

Submission ownership belongs to `submission_transaction()`, called by the
internal Condor adapter after preparation. It reconciles target rows, creates
fresh reservation tokens and binds itemdata while holding all row leases. The
adapter passes those lease descriptors into `condor_submit` and waits for its
result. Operators use `make_cards.py --condor`; prepared material must not be
submitted directly or replayed outside the transaction.

A definite process-creation failure rolls back newly created reservations.
Command exit zero retains them. A timeout, nonzero exit or interruption after
possible scheduler side effects is `AMBIGUOUS_SUBMISSION_OUTCOME`: reservations
remain until the next invocation automatically reconciles them by token and
then applies H2 recovery as needed. The command's stdout and any reported
ClusterId.ProcId are informational; they do not establish scheduler ownership.

Normal ownership attempts reconcile automatically. The explicit maintenance
command is optional and proactive; it is not required for backend switching:

```bash
analysis/topeft_run2/run_datacard_matrix_resumable.sh \
  --reconcile-reservations /absolute/path/to/manifest.json
```

This queries current ownership and reconciles H2 without starting a producer.
Use `--row-id "$row_id"` to select one row. Reconciliation cannot acquire a
physical lease while its producer or submission transaction is active. Status
queries remain read-only. Never delete reservation or lock files to bypass
unresolved ownership.

The adapter currently requires a shared filesystem: original manifests,
inputs, runtime, dispatch and output paths must be visible at the same absolute
locations on every worker. Dispatch destination, interpreter and runner paths
support letters, digits, `_`, `.`, `/`, `+` and `-`; whitespace and submit
metacharacters fail before output creation. Worker/site qualification is a
separate step. Material generation alone does not qualify submission.

`getenv = false` disables inheritance of the interactive submitter environment.
The worker invokes the manifest's absolute Python interpreter and the absolute
maintained runner; no Conda activation, inherited `PYTHONPATH`, credentials or
EFTFit bootstrap is serialized. The interpreter's installed packages must
resolve TopEFT and its dependencies. The adapter resolves an absolute
`condor_q` for submit-side reconciliation only. Workers do not run it or contact
the schedd to prove identity. They require `_CONDOR_JOB_AD`: local literal
`DatacardUnit` and `DatacardReservation` must match the submitted row/token and
shared reservation. Missing, malformed and conflicting local context have
specific diagnostics. A sanitized submit-host import/CLI probe does not
qualify the execute-node environment or the site's job wrapper. First qualify
the shared absolute interpreter; if that fails while common Conda paths are
visible, evaluate an explicit `conda run -p <resolved-prefix>` bootstrap before
considering a frozen environment archive.

CPU requests are positive integers. Memory/disk are positive quantities with
optional HTCondor unit suffixes; explicit suffixes avoid unit ambiguity.
Optional `--max-runtime "$allowed_execute_duration"` emits native
`allowed_execute_duration` seconds. HTCondor 25.0.1 parses this control. It
limits one activation, excludes file-transfer time and resets for a later
activation after eviction/restart. `RemoteWallClockTime` instead accumulates
remote allocation across attempts. A value of `21600` represents six hours per
activation exactly; it is supported, not a universal workload recommendation.
The first successful lifecycle canary must allow at least 21600 seconds per
activation. Exceeding the native limit causes a Held transition followed by
periodic removal; inspect its event-log reason.
Use one common resource request initially and measure it before normal
scheduling. Cards per unit come from workload decomposition; jobs in flight
come from operator/site policy. This adapter sets no concurrency cap.

`--max-retries` is a nonnegative terminal executable retry budget; zero disables
those retries. Native success exit code zero removes the job normally. A
nonzero/signal exit can retry within the budget, then native removal takes it
out of the live queue. There is no unconditional failure hold. Evictions do not
consume terminal-exit retries, so positive `--max-starts` separately caps
`NumJobStarts`. A match guard prevents another activation at the cap; the final
permitted activation may finish. Its terminal exit removes the job, and an
evicted job left Idle at the cap is removed on the scheduler's next periodic
policy evaluation. Exhausted retries/starts are not deliberately parked Held
or permanently Idle. Every Held state is terminal for the current TopEFT job,
including duration, memory/disk, process-creation, input/output, site and manual
holds. `JobStatus == 5` triggers periodic removal regardless of reason code;
there is no automatic release or replacement scheduler job. Hold-to-removal
timing follows scheduler policy evaluation.

The canonical row/output domain identifies the workload. A unique reservation
token is the current operational correlation key, not workload or completion
authority. `ClusterId.ProcId` is only a scheduler locator for status and logs;
ProcId never defines a row or an H2 physical attempt. A changed cached locator
does not invalidate a reservation. Local worker metadata uses its job-ad locator.

Submit-side ownership queries select `DatacardReservation == "<safe token>"`
and cross-check `DatacardUnit`. One live match means active ownership; multiple
matches fail closed with all locators. A failed or insufficient query reports
`SCHEDULER_OBSERVABILITY_INCOMPLETE` or
`SCHEDULER_OBSERVABILITY_INSUFFICIENT` with query diagnostics and preserves the
reservation unless another covered query or terminal proof establishes
disappearance. An empty query without coverage of the previously observed
owner is not disappearance proof.

Logical reservations survive live Idle, Running, Held-pending-removal,
Suspended, Transferring and post-eviction Idle states. Eviction, job removal and
physical flock release do not directly free a row. Local/manual unit execution,
full-workload execution and Condor reservation all reconcile before a new
claim. Raw `make_cards.py` holds the output-domain lease through production and
refuses reserved or H2-published roots; use the matrix runner for those domains.
Unit output roots must be disjoint; nested ownership domains are rejected.

On the next ownership attempt after scheduler disappearance, the runner
automatically acquires the physical lease and inspects H2. A valid receipt
means complete. Otherwise validated staging is reconciled, or an incomplete
attempt is preserved and eligible only if H2 permits it. Unavailable physical
ownership, malformed state or collisions block retirement. A surviving producer
retains its inherited lease even after manager death. Only safe H2 reconciliation
permits another backend to claim the unchanged workload. Neither disappearance
nor exit zero establishes completion.

Shared-filesystem persistence remains `should_transfer_files = NO`: worker ->
H2 private staging -> validation -> no-overwrite publication -> receipt last.
There is no authoritative HTCondor output transfer or wrapper `cp`/`rsync` step.
Scheduler logs are `logs/<row>__<ClusterId>.<ProcId>.out`, `.err`, and
`.events.log`. H2 producer logs remain separately scoped to physical attempts;
restarts of the same scheduler job can share its scheduler logs. The event log
is primary durable evidence of submission, executions/evictions, holds with
reason information, removal and termination as applicable. The fault canary
must prove useful hold reasons survive removal. Live status also exposes
`HoldReason`, `HoldReasonCode` and `HoldReasonSubCode` when present. Optional
post-mortem history may add removal/terminal details; unavailable history does
not prevent H2 reconciliation when another covered query or event-log removal
proves disappearance.

Runner `--status` and the dashboard expose live reservations and prominently
show Held/unknown ownership, even when a valid H2 receipt already exists.
Scheduler observations can also be supplied separately to either runner `--status`
or the dashboard with `--scheduler-status FILE`. They are displayed as
operational metadata with an observation time; they never change completion
or permit retry. Scheduler exit zero without a valid receipt is incomplete.
See the [reference](../reference/datacards_and_scalings.md#execution-unit-and-dispatch-interfaces)
for the observation schema and maintained APIs.

The steps are:

1. Run the producer rows and retain their logs, snapshots, and receipts.
2. Review the completed v3 manifests and source artifacts.
3. Build each per-era package with `build_per_era_datacard_package.py build`.
   The runner does not launch either package builder.

## Build per-era packages

The resumable runner produces individual TXT/ROOT pairs
and row-local `selectedWCs.txt` and `scalings-preselect.json`. The last row's
shared metadata files are not a complete era package. Supply the completed
v3 manifests to the per-era builder; repeat
`--matrix-manifest` for each contributing manifest. Run the following
builder command from the repository root:

```bash
python analysis/topeft_run2/build_per_era_datacard_package.py build \
  --era run2 \
  --matrix-manifest /absolute/path/to/run2-matrix.json \
  --output /absolute/path/to/new-run2-package \
  --analysis TOP-26-006
```

Repeat with `--era run3`, its completed manifest(s), and a distinct fresh
output. The builder takes its requested physical targets from the supplied
manifest rows. It rechecks receipt-bound source files and exact card/template
pairs, assigns deterministic
per-era `chN` labels, consolidates selected WCs and scaling records, and
publishes `cards/`, `selectedWCs.txt`, `scalings.json`,
`physical_to_chN.json`, and `package-provenance.json`. Duplicate scaling
identities are rejected. `combinedcard.txt` is not built here.

For a restricted fit, select the same physical subset in each era's manifest.
Run the two manifests, then build each package without repeating the selection.
The builder requires the completed rows to provide exactly the manifest-declared
set. On a mismatch it lists missing and extra targets separately. For a missing target present in a
supplied manifest, it names the manifest and row and prints the runner's
`--status` and resume commands. Check status before deciding whether to resume;
the builder never runs producer rows.
The combined builder accepts X with X and rejects X with a different subset Y.

## Build and consume a combined Run 2 + Run 3 package

Use the Run 2 and Run 3 per-era package directories as inputs. They must
contain the same set of physical channel/distribution targets; matching
restricted subsets are allowed. A mismatch reports the targets missing from
each era before publishing a combined package. Run the
combined builder from the repository root. It derives the mapping and card
order from their mappings, copies ROOT templates, updates only the card
template reference, and relabels scaling channels. It checks the staged and
published cards-only package against the input packages. It requires a fresh
absolute output path:

```bash
python analysis/topeft_run2/build_combined_datacard_package.py build \
  --run2-package /absolute/path/to/run2-package \
  --run3-package /absolute/path/to/run3-package \
  --output /absolute/path/to/new-combined-package \
  --analysis TOP-26-006 \
  --package-date YYMMDD \
  --package-version v1
```

The package contains `cards/`, `scalings.json`,
`combined_mapping_manifest.json`, `ordered_card_inputs.txt`,
`package-provenance.json`, and a README. It does not contain a
combined `selectedWCs.txt` or `combinedcard.txt`. The `certify` subcommand can
check an existing package against the same Run 2 and Run 3 inputs. Fit
configuration determines the Wilson coefficients used when building the
workspace.

## Standardize a certified combined package

For the current `TOP26006_v1` package schema, run from the repository root:

```bash
python analysis/topeft_run2/standardize_systematics.py \
  --input-package /absolute/path/to/certified-combined-package \
  --output-package /absolute/path/to/fresh-standardized-sibling
```

The input and output must differ, and the destination must be fresh. The input
package is not modified. The mapping row's `run2` or `run3` era selects the
naming map; card filenames and card count do not select it. This operates on
final TXT/ROOT datacards, preserving their already-realized correlation
topology, card order, EFT scalings, and logical mapping. The sibling is still
a datacard package. The standardizer preserves an already shared MVA identity;
it does not create one from per-run inputs. See the [nuisance reference](../reference/datacards_and_scalings.md)
for exact identities and failure boundaries.

`ordered_card_inputs.txt` lists cards in combination order. From the combined
package directory, follow its generated README:

```bash
cd <combined-package-root>
mapfile -t cards < ordered_card_inputs.txt
combineCards.py "${cards[@]}" > combinedcard.txt
```

The list follows the packaged channel mapping; do not use shell-glob order.
This command combines the individual cards. Workspace construction and fitting
follow in EFTFit/Combine.

## Diagnose card/finalization failures

| Failure | Correct response |
| --- | --- |
| merge policy/schema/companion mismatch | fix or reproduce the upstream artifact; do not concatenate dictionaries manually |
| selected variable/channel absent | inspect the merged histogram axes and source registry; a regex cannot create missing content |
| fitting edges not exactly representable | correct the fitting-bin definition or produce compatible processing-binned PKLs |
| selected-WC mismatch | review the new selection and reference before rerunning |
| per-era builder missing physical target | inspect the reported manifest row with the displayed runner `--status` command; resume through the runner when appropriate, or supply a manifest defining the target |
| per-era builder unexpected physical target | use the matching requested target set or correct the supplied manifests before packaging |
| combined builder unequal physical targets | select or rebuild both era packages with the same physical channel/distribution set |
| preselect scaling has no selected physical label | determine whether the process intentionally has no external EFT morph or the producer output is incomplete |
| destination already exists | choose a fresh package directory; there is no supported resume/merge behavior |

Exact schemas and option contracts are in the
[datacards/scalings reference](../reference/datacards_and_scalings.md). The
[EFTFit boundary explanation](../explanation/datacards_and_eftfit.md) describes
why package construction and card combination remain separate responsibilities.

Use [categories and observables](categories_and_observables.md) before changing
the physical category or distribution that a card consumes, and use
[corrections, weights, and systematics](corrections_weights_and_systematics.md)
before changing the upstream variation. This guide owns the card-facing rate,
applicability, fitting-view, selected-WC, and scaling-export changes only.
