# Datacards and scalings

For the end-to-end package procedure, artifact ladder, inspection checklist,
and fit-input handoff, see [build and use datacard packages](../how_to/datacard_packages.md).

## Card-production interface

`analysis/topeft_run2/make_cards.py` consumes one or more coherent histogram
artifact pairs. It accepts positional PKLs or a list file, channel/variable
selection, selected-WC controls, nuisance controls, missing-parton controls,
and local or Condor execution. The default binning view is `fitting`; year
coverage defaults to `warn`; and observed data are blinded through the default
Asimov behavior unless the supported option changes that choice.

`topeft.modules.datacard_tools.load_and_merge_histogram_pkls(...)` owns
multi-artifact compatibility and defaults to requiring sumw2. `DatacardMaker`
owns process/systematic selection, late exact aggregation, individual card and
ROOT-template contents, statistical-uncertainty policy, selected-WC discovery,
and EFT scaling extraction.

The normal WC-discovery path writes individual card/template pairs,
`selectedWCs.txt`, and `scalings-preselect.json` to the direct output directory.
With `--use-selected FILE`, `make_cards.py` reads the supplied WC selection and
materializes the same canonical signal-only representation as output-side
`selectedWCs.txt` without modifying the caller's file. A preselect record
contains a physical `channel`, a producer-owned `process`, `parameters`, and
scaling coefficient payload. A consolidated metadata set must not contain
multiple records for one physical channel/process identity.

`make_cards.build_arg_parser()` is the public option authority. Stable groups
are:

| Group | Type/default/accepted values | Contract |
| --- | --- | --- |
| Histogram input | Zero or more positional paths or one `--pkl-list-file`; exactly one source form after resolution | All inputs are sidecar-validated and merged with required sumw2. `--merge-only` stops after these checks; optional cache/report outputs retain merge diagnostics. |
| Output/selection | `--out-dir .`; repeatable variables/channels; optional ignore/drop lists | `--ch-lst` patterns are regex selectors over physical channel names. This is distinct from exact fitting override keys. |
| Years/coverage | Optional supported year list; coverage `warn`, `error`, or `off`, default `warn` | Mixed input identities and structural year gaps follow the selected fail/warn policy. |
| Binning | `fitting` or `processing`, default `fitting` | Fitting performs exact late aggregation per selected physical channel. |
| WCs | POI list, reference/selected-WC files, select-only/check controls, optional WC values/scaling order | WC selection uses the card-facing fitting view; selected files must match the producer's supported process/WC structure. `--use-selected` materializes canonical signal-only `selectedWCs.txt` in `--out-dir`. |
| Nuisances | Nuisance and MC-stat switches, `--rate-syst-json`, `--miss-parton-file`, `--skip-missing-parton-rate-syst`, and optional `--suppress-nominal-like-tau-shapes` | Run era selects maintained defaults when the override is omitted. An explicit `--rate-syst-json` is forwarded as `rate_systs_path`; worker arguments come from the canonical manifest. The missing-parton consumer infers the maintained payload layout and has no `--sr-registry` option. |
| Data/negative bins | Asimov by default; `--unblind` for observed data; crop negative bins by default | Both modes write `data_obs`. Unblinded production requires a `data` process in each selected channel; object presence does not identify the observation mode. |
| Execution | Direct local default; `--condor` requires `--matrix-manifest`, `--condor-output`, explicit CPU/memory/disk requests, `--max-retries`, `--max-starts`; optional `--max-runtime` and `--prepare-only`. Scheduler queries use installed default discovery. | Submits one queue item per manifest row after token reservation. Prepare-only emits ownership-free material and invokes no scheduler. |
| Existing-row dispatch | Repeat `--row-id ROW_ID` with `--condor`; `--rescue-ambiguous` requires exactly one selected row and excludes `--prepare-only` | Uses original v3 row descriptors and public manifest/attempt identity. Guarded rescue replaces an ambiguous reservation token under the row lease; known active jobs and held leases block it. |

In direct mode, `make_cards.main()` resolves inputs, validates/merges them,
constructs one `DatacardMaker`, selects or loads WCs, calls `run_local()`, and
writes `scalings-preselect.json`. It writes canonical signal-only
`selectedWCs.txt` for both derived and `--use-selected` selection paths.
`run_local()` selects physical channels by regex and calls `analyze()` per
channel/distribution. `--condor` dispatches before these producer operations to
`datacard_matrix_condor.prepare_material()` and, unless `--prepare-only` was
selected, `submit_material()`. The legacy `run_condor()` and
`--chunks` partitioning interface have been removed. Supported worker producer
options are recorded in canonical manifest rows, never reconstructed by the
scheduler frontend.

## `topeft.modules.datacard_tools.DatacardMaker`

**Kind/status:** class, developer-facing. **Purpose:** transform one coherent
histogram family into card-facing process/systematic views, individual text
cards, ROOT templates, selected-WC information, and preselection scaling
records. **Signature authority:** class constants, `__init__`, `read`, and the
methods in `topeft/modules/datacard_tools.py`. Its `**kwargs` constructor is
weak machine-near documentation; only the curated keys below are supported.

Exactly one of `pkl_path` (path-like) or `hists` (built-in histogram dictionary)
is required. Supplying both or neither raises `ValueError`; non-dict `hists`
raises `TypeError`.

| Constructor key | Type/default | Semantics |
| --- | --- | --- |
| `binning_mode` | `fitting` default; also `processing` | Selects card-facing view. Unknown values fail. |
| `year_lst` | Sequence, default empty/all | Supported values are `UL16`, `UL16APV`, `UL17`, `UL18`, `2022`, `2022EE`, `2023`, `2023BPix`. Determines Run 2/3 rate-systematic defaults and missing-parton era. |
| `do_sm` | Boolean, default false | Selects SM-only card behavior where consumed. |
| `do_nuisance` | Boolean, default false | Loads/writes supported nuisance contracts. |
| `drop_syst` | Sequence, default empty | Removes named template systematics after supported loading/selection. |
| `skip_missing_parton_rate_syst` | Boolean, default false | Disables only the missing-parton nuisance. |
| `out_dir` | Path string, default `.` | Destination for text/ROOT card artifacts. |
| `var_lst` | Sequence, default empty | Kinematic family selection. |
| `do_mc_stat` | Boolean, default false | Enables card `autoMCStats` behavior; stored template variances still follow process policy. |
| `wcs` | Sequence, default empty | Restricts WC consideration during selection/decomposition. |
| `unblind` | Boolean, default false | Keeps real data; otherwise data processes are removed and Asimov output is used. |
| `suppress_nominal_like_tau_shapes` | Boolean, default false | Omits only qualifying complete local Tau shape pairs after the card-facing view; unsupported with `use_AAC`. |
| `verbose` | Boolean, default true at class level | Controls progress diagnostics. The CLI resolves its own default. |
| `use_AAC` | Boolean, default false | Enables the AAC template convention. |
| `wc_scalings` | Sequence, default empty | Requested WC ordering/selection for scaling production. |
| `rate_systs_path` | Path, default Run-2/Run-3 JSON selected from years | Rate-systematic registry relative to `topeft_path`. |
| `missing_parton_path` | Optional exact path | Overrides run-era payload selection; empty string and mixed-era use fail. |
| `ignore` | Sequence, default empty | Additional pre-grouping process names to exclude. |

The constructor key is the plural `rate_systs_path`. The direct CLI forwards
that key only for an explicit `--rate-syst-json`; omission preserves the
constructor's era-selected default.

Construction reads/validates the input, loads `params/wc_ranges.json`, selects
the rate-systematic registry, optionally loads missing-parton payload content,
groups/prunes processes/systematics, and prints resolved output/read timing. It
therefore has file-read side effects even before `analyze()`. Writing occurs in
`analyze()`. Unknown leftover kwargs are not a supported extension mechanism.

Important failure boundaries include invalid/mixed years, incompatible
artifacts or WC order, missing required sumw2, unsupported SR application-axis
labels, unresolved sparse axes that would duplicate ROOT template names,
invalid exact rebinning, missing/mismatched shape pairs, and a payload that
matches no maintained layout or matches more than one.

## Datacard matrix manifest and runner

`analysis/topeft_run2/make_datacard_matrix_manifest.py` writes a new
`topeft_datacard_matrix_v3` JSON manifest. Required options are
`--era {run2,run3}`, `--manifest-output`, `--output-root`, `--control-root`,
`--python-executable`, `--missing-parton-file`, and
`--runtime-contract-id`. Repeat `--input-pkl ROLE=PATH` to bind the selected
profile's current PKL blocks. A full Run 2 manifest requires `block1`,
`block2`, `block3`, `block4`, and `block5`; a full Run 3 manifest requires
`2l_mixed`, `3l_m_offz`, `3l_p_offz`, `3l_onz_tau`, and `3l_fwd`.
`--make-cards-path` defaults to this checkout's `make_cards.py`;
`--working-directory` defaults to the repository root; `--attempt-id`
defaults to `attempt_01`. Repeat `--runtime-file` to hash additional runtime
files. `--channel-set-key` defaults to `ALL_CH_LST_SR` and selects a maintained
channel set. As an advanced optional filter, repeat
`--physical-target <channel>_<distribution>` to produce an exact one-off
subset. Supply the PKL roles used by the selected rows.
`--lepton-mva-correlation {per-run,cross-run}` defaults to `per-run` and is
recorded in every selected row's producer arguments. Optional
`--suppress-nominal-like-tau-shapes` and `--unblind` are likewise forwarded to
every selected row; neither is enabled by default. A manifest therefore has
one requested observation mode across its rows. Package builders do not
establish consistency among independently produced manifests or packages.

Physical targets and their distributions come from the human-authored
`analysis/topeft_run2/datacard_matrix_profiles.yml`, checked against
`ch_lst.json`. The helper writes 11 logical profile rows for either full era.
The maintained execution partition yields 11 Run 2 execution units or 34 Run 3
execution units, each serialized as a manifest row. Every unit retains its
logical row's input role
and distribution; units belonging to one logical row have exactly its physical
channel union without duplication. `make_cards.py --ch-lst` selects each unit's
physical channels. Run 2 uses years `UL16APV UL16 UL17 UL18`; Run 3 uses
`2022 2022EE 2023 2023BPix`. Generated producer arguments set
`--year-coverage-policy error`. Input files and runtime paths must exist;
manifest, output, and control paths must be absolute, and the manifest must
not already exist. The helper creates neither cards nor receipts.

`topeft_datacard_matrix_v2` is historical, not corrupt. The current runner and
per-era builder do not accept v2 manifests. Generate a new v3 manifest with
`make_datacard_matrix_manifest.py`; no migration or conversion path is provided.
Select a restricted physical subset once at manifest generation using
`--channel-set-key` or the advanced `--physical-target` filter. The helper
filters the maintained execution partition; it does not regroup units. The
per-era builder does not reselect targets.

`run_datacard_matrix_resumable.sh --plan-only MANIFEST` checks the plan;
`--status MANIFEST` reports row state without running cards; passing
`MANIFEST` runs or reconciles eligible rows. There is no `--resume` flag.
`attempt_id` in a v3 manifest remains the public successful-unit identity used
by receipts and `--selected-unit MANIFEST_SHA256:ROW_ID:ATTEMPT_ID`. The runner
derives a private numbered physical-attempt directory beside each row output
directory; a failed physical attempt never acquires selected-unit authority.
Already valid v2 receipts remain valid and are skipped without a new attempt.
Direct `make_cards.py` production has no manifest requirement, but holds the
output-domain lease and refuses logical reservations or H2-published roots.

The runner redirects row-owned producer output, merge report, and log writes
into that private directory. After a zero return code it checks the declared
nonempty TXT/ROOT outputs and control artifacts, snapshots the selection,
scalings, and merge report, records sizes and SHA-256 values, and writes an
attempt-local completion intent. It then creates the absent canonical row
directory, marks it with a hardlink to that intent, publishes files without
overwriting existing destinations, checks canonical bytes against the intent,
and finally writes the existing receipt format. Publication is a checked
sequence, not an atomic directory rename on the deployed NFS filesystem.
The intent and marker distinguish recoverable process-crash windows; an
unmarked or mismatched canonical directory fails closed. The byte checks do
not establish physics validity or power-loss durability.

`--status` is the row-state authority. It distinguishes `complete`,
`not_started`, `active`, `partial_staging_attempt`,
`complete_validated_staging_not_promoted`, `promoted_incomplete`,
`promoted_complete_without_receipt`, `stale_owner_free_lock`,
`held_lock_active_or_unknown`, `invalid_receipt`, and
`canonical_collision_or_ambiguous_state`. A later explicit invocation starts a
new private attempt after incomplete staging, or reconciles a complete
validated attempt without rerunning the producer. A valid receipt remains the
only completion authority. The dashboard renders runner status and receipt
counts plus live reservation observations; it does not independently validate
completion, physics, resources, or topology suitability.

### Execution-unit and dispatch interfaces

| Maintained interface | Contract |
| --- | --- |
| `datacard_matrix_runner.derive_units(path)` | Loads the original global v3 workload and deterministically returns one descriptor per existing row, in manifest order. New execution requires disjoint canonical output roots. |
| `resolve_unit(descriptor)` | Validates exact descriptor fields and parent integrity, returning the unchanged full parent, its byte hash and selected row. Unknown rows and changed parent bytes fail closed. |
| `execute_unit(manifest, manifest_hash, row, backend=..., scheduler_id=..., reservation_token=...)` | Canonical execution primitive: acquire physical lease, check/reconcile logical ownership, then apply H2 skip/reconcile/execute. Condor checks shared row/token ownership against its local job ad without a scheduler query. |
| `execute(manifest, manifest_hash, ...)` | Sequential orchestration over `execute_unit`; stops at a blocked/failed unit. No second completion state machine. |
| Runner `--unit FILE` or `MANIFEST --row-id ID` | Selects one unit without rewriting its parent or public identity. `--plan-only` and `--status` remain read-only. `--derive-units MANIFEST` prints descriptors only. |
| Runner `--reconcile-reservations MANIFEST [--row-id ID]` | Optional proactive reconciliation under each physical lease. Every new ownership attempt also reconciles automatically. Never runs a producer. |
| `datacard_matrix_condor.prepare_material(manifest_path, destination, request_cpus=..., request_memory=..., request_disk=..., max_retries=..., max_starts=..., query_executable=..., max_runtime=...)` | Fresh-only shared-filesystem material with no row reservations. Resources and budgets are required; runtime optional. The absolute query executable defaults to resolved `condor_q`, and scheduler endpoint discovery follows the installed configuration. No producer, submission or package builder. |
| `datacard_matrix_condor.submission_transaction(manifest_path, destination, scheduler)` | Context manager: validate prepared descriptors, reconcile target rows, create reservations and bind itemdata under all row leases. Yields transaction state; never invokes a submitter. |
| `datacard_matrix_condor.submit_material(manifest_path, destination, scheduler)` | Internal process adapter used by `make_cards.py --condor`; invokes installed `condor_submit` with argv and inherited row lease descriptors, and classifies command outcomes. Operators do not call it directly. |
| `datacard_matrix_runner.reserve_condor(manifest, manifest_hash, rows, scheduler)` | Maintained reservation transaction context underlying dispatch. Reconciles stale ownership/H2 before claims; rolls back only definite no-start outcomes. |

Descriptor fields are exactly `schema` (`topeft_datacard_execution_unit_v1`),
`parent_manifest` (absolute path), `parent_integrity` (SHA-256 of those current
bytes), and `row_id`. Descriptors are generated dispatch artifacts. Workload
semantics remain in the global v3 manifest and its profile/channel owners.
Receipt validation continues to use the original parent hash, public row and
attempt, exact runtime/argv and recorded artifacts. Existing receipt and
selected-unit interfaces are unchanged; historical manifest loading is not
restricted by the new execution-layout check.

The canonical output root determines a persistent sibling
`.<output_name>.datacard.lock` and `.<output_name>.datacard.owner.json`.
Hash, backend and row-name aliases cannot create another lease for that root.
A short `.datacard_lease_setup.lock` at the nearest existing parent serializes
path initialization and is released before execution. This prevents a nested
request from creating an active owner's absent canonical directory. Ancestor
ownership is checked before any directory creation and again under setup.
Lock inodes persist and must never be unlinked while the workflow may run.
The legacy manifest lock is held shared to exclude an older exclusive runner;
unit execution and completion state remain entirely row-owned. Producers
inherit both lifetime lock descriptors, so manager death alone cannot authorize
a duplicate producer. Unknown/held ownership fails closed.

`.<output_name>.datacard.reservation.json` is an atomic per-domain logical
reservation, not a global registry or completion record. Its existing schema
remains `topeft_datacard_reservation_v1`; fields are `schema`, `row_id`,
`output_root`, `backend`, `owner_id`, `created_at`, `scheduler`, and
`scheduler_id`. The last two are null for local/manual ownership. Condor's
scheduler object records explicit `pool`, `schedd` and absolute
`query_executable`. `scheduler_id` caches only the last observed locator; it is
never a query fallback or equality gate. `owner_id` is fresh UUID-style hex:
exactly 32 lowercase hex digits, checked before constraint interpolation.
It is current operational correlation, not historical workload authority.
Submitted ads carry `DatacardReservation`, `DatacardUnit` and
`DatacardParentIntegrity`; canonical workload and H2 receipt identity remain
unchanged. Native counters do not become another completion record.

Material generation is not submission ownership. `prepare_material()` does not
reserve or reconcile rows and writes `unreserved` itemdata placeholders.
`submission_transaction()` replaces these with newly allocated tokens under
`reserve_condor()`'s physical row leases. It validates prepared descriptors and
retains all leases through the caller's submission boundary, preventing another
claimant from observing zero jobs before submission and reclaiming a row.

The yielded transaction exposes `reservations`, `lease_fds`, and
`mark_submission_started()`. The internal adapter calls that marker immediately
before process creation, passes `lease_fds` to its subprocess, and awaits
subprocess termination before leaving the context. A zero command exit
and ambiguous timeout/nonzero/interruption preserve reservations. Exiting
without possible effects rolls back newly created reservations, including
partial acquisition/material-binding failure. If process creation itself
raises `OSError` and no submit executable started, the adapter uses
`record_definite_spawn_failure(error)` to restore rollback eligibility. Its
proof obligation excludes later I/O/status errors, terminal nonzero exits and
timeouts. `submission_transaction()` itself never runs `condor_submit`;
`submit_material()` owns that process. Prepared itemdata must not be replayed
outside this transaction. Command stdout, including ClusterId.ProcId, is
operator metadata only. A later frontend invocation automatically reconciles
an ambiguous reservation by its token.

All reservation mutation is serialized under the output-domain lease. Worker
activations acquire the same lease. An unusually early worker may encounter
the still-active submission lease and fail within the bounded start/retry
policy; live qualification must check this ordering. Logical ownership remains
while a job is live, including Held pending removal. Canonical row/output,
reservation token, scheduler locator, physical lease, physical attempt and
valid receipt remain distinct. Neither ProcId order nor a historical manifest
hash defines the row; changed scheduler locators do not invalidate ownership.

Submit-side queries use installed default discovery (or an observed endpoint
hint for an existing reservation), `-allusers`, exact
`DatacardReservation == "<safe token>"` constraint, JSON projection and a
30-second timeout. One match with matching row/token has `ownership: ACTIVE`;
multiple matches raise `DUPLICATE_SCHEDULER_OWNERSHIP` with all locators. An
empty reply proves `GONE` only when the queried endpoint covers the previously
observed owner; terminal history or a matching removal event log can also prove
disappearance. Otherwise failed or insufficient queries raise
`SCHEDULER_OBSERVABILITY_INCOMPLETE` or
`SCHEDULER_OBSERVABILITY_INSUFFICIENT` with query diagnostics, preserving
ownership. Contradictory or malformed ads fail closed with specific diagnostics.
History is terminal evidence, never live ownership authority.

Workers never invoke `condor_q` or contact a schedd for identity. They require
`_CONDOR_JOB_AD` and parse only literal `DatacardUnit`, `DatacardReservation`,
`ClusterId`, and `ProcId`, with case-insensitive attribute names. Row/token must
match submitted itemdata and the shared reservation. Other ClassAd expressions
are not evaluated. Missing context, malformed/missing/duplicate literals and
row/token conflicts have distinct diagnostics. The local job-ad Cluster/Proc
is recorded as scheduler metadata; differences from submitted/cached locators
are not blockers. The installed HTCondor distribution supports the local ad
path; synthetic fixtures validate parsing, while execute-node availability
remains a live qualification obligation.

After a successful query proves disappearance, existing H2 classification and
reconciliation run automatically under the lease before reservation retirement.
Local direct-unit execution, full orchestration, Condor reservation and explicit
maintenance all use this gate; the manual reconciliation CLI is optional for
backend switching. Valid receipts mean complete; partial staging can become
eligible and validated staging can publish without another producer. Invalid
receipts, collisions, ambiguity or an unavailable physical lease block. Local
stale ownership also uses H2 and the inherited lease, never cross-host PID
inference. Raw producer calls without a manifest cannot reconcile H2 records
and refuse reserved or H2-marked roots; H2 workers use private staging domains.

Status rows carry unit `lock_path`, `lock_held`, `lock_probe_error` and
`owner_metadata`; aggregate `lock_held` means any unit is held, or unknown if
none is held and a probe failed. The legacy top-level `owner_metadata` may be
null. Consumers must use row metadata for multiple active units. Owner backend
and scheduler ID are observations, not receipt fields.
Rows with logical ownership additionally expose `reservation`, including
backend, observed scheduler locator/state and observation time. `HELD`,
`UNKNOWN` and `GONE` remain visible independently of H2 completion. Error codes
and exact query diagnostics are displayed, as are available live hold fields.
Read-only status does not retire reservations or persist locator observations.

`--scheduler-status FILE` on runner status or dashboard accepts an externally
collected observation with exactly these keys:

```json
{
  "schema": "topeft_datacard_scheduler_status_v1",
  "parent_integrity": "<current parent manifest SHA-256>",
  "observed_at": "<observation timestamp>",
  "units": [{"row_id": "<existing row>", "scheduler_id": "<cluster.proc>", "state": "<observed scheduler state>"}]
}
```

Unknown/duplicate rows, duplicate job IDs or a mismatched parent fail. The runner attaches
`scheduler_metadata` with `completion_authority: false` and preserves its own
row status. Observations may cover a subset; with `--row-id`, supply only that
row. This supplied overlay cannot replace live reservation queries or authorize
execution. Exit zero alone provides no producer completion authority.

Generated material consists of `units/*.json`, `items.txt`, `worker.sh`,
`datacards.sub`, and `logs/`. Queue items bind row IDs to descriptor byte hashes
and initially invalid `unreserved` placeholders; the submission transaction
binds fresh reservation tokens. Scheduler logs use
`logs/$(row_id)__$(Cluster).$(Process).out`, `.err`, and `.events.log`;
each row/current job has distinct paths. H2 physical-attempt logs remain separate.
The worker checks descriptor/parent/runner integrity and enters the same unit
executor, with the producer's recorded arguments and no recursive Condor mode.
The submit file is published last using the existing no-overwrite copy helper.
Existing destinations and invalid resource quantities are rejected. Generated
paths have the restricted character set documented in the operator guide.

This is TopEFT producer orchestration; EFTFit/Combine runtime and fit scheduling
remain outside this interface. `getenv = false` disables wholesale submitter
inheritance. Absolute manifest interpreter/runner paths construct the TopEFT
worker runtime without serializing shell state, secrets or credentials. A
sanitized import/CLI probe verifies the submit-host installation; execute-node
packages, site wrapper environment, local job-ad availability and shared mounts
still require qualification. Submit-side reconciliation requires scheduler query
access; worker execution does not. Qualify shared absolute interpreter paths
first; if insufficient with common Conda paths visible, evaluate explicit
bootstrap before considering an environment archive. No additional
`environment` variables are currently required or emitted.

`should_transfer_files = NO` preserves shared-filesystem H2 private staging,
validation, no-overwrite publication and receipt-last authority. Absolute paths
must be available on workers. No authoritative `transfer_output_files`, wrapper
copy or post-producer publication path is introduced. Native scheduler logs are
operational evidence only.

`request_cpus` is a positive integer; memory/disk accept positive integer
quantities and optional unit suffixes. Optional positive `max_runtime` maps to
native `allowed_execute_duration`, parsed by HTCondor 25.0.1. Its seconds limit
one activation, exclude transfer time and reset for a later activation after
eviction/restart. `21600` is representable exactly and is the minimum for the
first successful lifecycle canary. It is not cumulative `RemoteWallClockTime`; native timeout
causes a Held transition followed by periodic removal.

For explicit terminal retry budget R >= 0 and total-start budget N > 0, the
adapter emits:

```text
max_retries = R
success_exit_code = 0
requirements = MY.NumJobStarts < N
on_exit_remove = NumJobStarts >= N
periodic_remove = (JobStatus == 5) || (JobStatus == 1 && NumJobStarts >= N)
```

Native HTCondor combines the removal condition with
`NumJobCompletions > JobMaxRetries || ExitCode =?= JobSuccessExitCode` and
retains its ordinary resource requirements. No unconditional `on_exit_hold`
intercepts failed terminal exits. Native retry exhaustion removes the job.
Evictions count toward starts separately from terminal retries: the match guard
prevents another activation at N, the last Running activation can finish, its
terminal exit removes the job, and capped Idle jobs leave the queue at the next
periodic evaluation. The guard alone would strand an Idle job; periodic removal
alone would permit starts before evaluation. Every Held state is terminal for
the current TopEFT scheduler job, independent of HoldReasonCode: native duration,
resource, process creation, I/O, site/policy and manual holds all satisfy removal.
There is no automatic release or automatic replacement job. Removal waits for
scheduler periodic evaluation and does not delete the reservation.

The unique scheduler event log is the primary durable hold record, including
reason information, with execution/eviction/removal/termination events as
applicable. Live query projection includes `HoldReason`, `HoldReasonCode` and
`HoldReasonSubCode`. Optional history lookup by token/locator may retain these
and `RemoveReason`/terminal status; history availability is not an H2 safety
gate. The live fault canary must prove hold evidence persists through removal
and that the next ownership attempt only reuses the row after safe H2
reconciliation.

There are no production resource/retry defaults or jobs-in-flight caps. One
common configuration covers the existing rows, including valid-receipt skips.
Real eviction, removal, start-limit enforcement, resources and worker runtime
remain separate live validation; this interface never submits a job.

## Per-era package interface

`analysis/topeft_run2/build_per_era_datacard_package.py build` builds a
Run 2 or Run 3 package. Required options are
`--era {run2,run3}`, repeatable `--matrix-manifest`, fresh absolute
`--output`, and `--analysis`. The requested target surface is the union of the
supplied manifest rows. There is no separate per-era target selector. The
inputs are matrix manifests with receipts for source card/template pairs and
metadata snapshots. The builder checks the
recorded source files, requires the completed rows to provide exactly the
requested physical targets, rejects duplicate scaling identities, and assigns
deterministic per-era `chN` labels.

For a recovered full-era campaign whose source manifests declare overlapping
targets, repeat `--selected-unit MANIFEST_SHA256:ROW_ID:ATTEMPT_ID` for every
authoritative successful unit. The manifest SHA is the hash of the original
manifest bytes under which that unit's receipt was written. Supply each
original manifest with `--matrix-manifest`; do not regenerate a restricted
manifest from completed rows, because receipts bind to the original manifest
SHA. The selected-unit route requires one valid current receipt for every
listed unit, rejects duplicate selected targets and units, and requires exact
coverage of the maintained `ALL_CH_LST_SR` full-era target set. Omitted or
unknown selected units fail. Without `--selected-unit`, the original manifest
surface behavior remains in effect and overlapping declarations are an error.
The package provenance retains the actual source manifest hashes and selected
unit count; the explicit invocation identifies the selected row/attempt receipts.

For missing manifest-declared targets, the error identifies each applicable
manifest and row ID and prints the runner's read-only `--status` and normal
resume commands as guidance. Extra completed targets are listed separately.
No final package is published on a
coverage mismatch, and the builder does not execute the runner.

The published package contains only `cards/`, `selectedWCs.txt`,
`scalings.json`, `physical_to_chN.json`, and
`package-provenance.json`. Its provenance records source manifest hashes,
source unit count, builder identity, card/template counts, and hashes of the
three metadata files; it does not record per-card/template hashes or exact
selected receipt keys. Preserve original manifests, successful receipts, and
the selected-unit invocation/build report for receipt-level lineage; see the
[package procedure](../how_to/datacard_packages.md). The package does not contain
`scalings-preselect.json` or `combinedcard.txt`. The builder writes and
verifies a private sibling staging tree, then publishes to a previously absent
output directory. The per-era scaling payload preserves producer-owned fields
while using the package mapping for the channel label; an absent exact
channel/process record means no external EFT morph for that pair.

## Combined package interface

`analysis/topeft_run2/build_combined_datacard_package.py build` consumes
Run 2 and Run 3 per-era package directories. It requires
`--run2-package`, `--run3-package`, a fresh absolute `--output`,
`--analysis`, `--package-date` in `YYMMDD` form, and
`--package-version` in `vN` form. Both sources must have the same
analysis identity and the same set of physical targets from their
`physical_to_chN.json` mappings. Their local `chN` labels need not match.
Matching restricted subsets are valid. A mismatch reports
`missing_from_run2` and `missing_from_run3` before copying or publishing.
The builder derives the combined channel mapping, copies
cards/templates, relabels scaling channels, and checks both the staged and
published packages against their inputs.

The cards-only package contains `cards/`, `scalings.json`,
`combined_mapping_manifest.json`, `ordered_card_inputs.txt`,
`package-provenance.json`, and `README.md`. It contains neither
`combinedcard.txt` nor a combined `selectedWCs.txt`. The generated README
lists the cards in `ordered_card_inputs.txt` for combination in that order:

```bash
mapfile -t cards < ordered_card_inputs.txt
combineCards.py "${cards[@]}" > combinedcard.txt
```

Run this command from the combined package directory;
the builder does not run Combine or create the combined card. The `certify`
subcommand reads an existing package against `--run2-package` and
`--run3-package`, with required `--package-root` and optional
`--report-json`. It does not mutate the package.

## Developer surfaces

| Fully qualified symbol | Kind/status; parameters and return | Stable contract |
| --- | --- | --- |
| `datacard_tools.process_retains_stat_uncertainty` | Developer-facing function; process string → Boolean | Only exact `fakes` or names containing `close` retain stored bin variance in ROOT output; all other templates write zero stored variance. |
| `datacard_tools.load_and_merge_histogram_pkls` | Developer-facing; paths; `require_sumw2=True`, required-family iterable empty, year policy `off` → `(histograms, merge_report)` | Validates sidecars, schema, identities, axes, disjoint contributions, policy, companion requirements, and year coverage before merge. Reads files; does not write them. |
| `datacard_tools.RateSystematic` | Developer-facing class; name and optional `all=False`, required `unc` when all | Stores process→uncertainty values. `add_process` is invalid for all-process instances; `get_process` returns the value or `-`. |
| `datacard_tools.JetScale` | Developer-facing `RateSystematic` subclass | Adds jet-indexed uncertainty lookup, symmetric low/high formatting, and a 0.01 lower bound. Missing process returns `-`; absent jet keys propagate lookup failure. |
| `datacard_tools.MissingParton` | Developer-facing `RateSystematic` subtype/marker | Holds the maintained channel-map role used while constructing the missing-parton rate systematic; payload loading/lookup is owned by `DatacardMaker` and `missing_parton_contract`. |
| `DatacardMaker.binning_view` | Histogram, family, exact physical channel → card-facing histogram | Processing returns input; fitting calls exact resolver/rebin. No regex is used here. |
| `DatacardMaker._scaling_histogram_for_json` | Channel histogram, channel, process → category-projected `HistEFT` | Internal extension. Selects nominal systematic and exact final SR application; any remaining categorical axis fails. |
| `DatacardMaker.get_selected_wcs` | Family and optional exact channel subset → process→WC sets | Inspects signal coefficient terms in fitting bins, ignoring flow, using class tolerance and optional WC restriction. No files written. |
| `DatacardMaker.make_scalings_json` | Existing record list, physical channel, family, process, WC names, scaling array → same appended list | Emits physical `<channel>_<family>`, `<process>_sm`, formatted parameters, and per-bin scaling excluding underflow. |
| `DatacardMaker.analyze` | Family, channel, selected-WC map, negative-bin policy, WC values → card result/`None` | Writes `ttx_multileptons-{channel}_{family}.txt/.root`, appends scaling records, applies fitting view to nominal and sumw2, and validates physical scaling edges. Unknown family/channel currently reports and returns `None`; deeper contract failures raise. |
| `build_per_era_datacard_package.build_per_era_package_from_units` | Era, analysis, physical names, validated units, output, source manifest hashes → provenance | Builds and verifies one per-era package from the supplied units; the CLI resolves v3 completions. |
| `topeft/channels/ch_lst.json` | Developer-facing configuration registry | Owns named topology blocks and physical channel/jet membership. Different consumers select explicit blocks; it is not a global claim that every block is current analysis scope. |

`DatacardMaker` extension points are the maintained process/systematic
registries, category registry, binning resolver, selected-WC policy, and
individual writer methods. Changing one requires its focused shape, sumw2,
application-axis, merge, and late-rebin tests.

See [flexible binning](flexible_binning.md) and
[missing-parton payloads](missing_parton_payloads.md).

## Source and test authority

- `analysis/topeft_run2/make_cards.py`
- `analysis/topeft_run2/make_datacard_matrix_manifest.py`
- `analysis/topeft_run2/datacard_matrix_profiles.yml`
- `analysis/topeft_run2/datacard_matrix_runner.py`
- `analysis/topeft_run2/datacard_matrix_condor.py`
- `tests/test_datacard_matrix_execution_units.py`
- `tests/test_datacard_matrix_condor.py`
- `analysis/topeft_run2/build_per_era_datacard_package.py`
- `analysis/topeft_run2/build_combined_datacard_package.py`
- `topeft/modules/datacard_tools.py`
- `topeft/modules/datacard_packaging.py`
- `tests/test_build_per_era_datacard_package.py`
- `tests/test_build_combined_datacard_package.py`
- `tests/test_make_datacard_matrix_manifest.py`
- `tests/test_make_cards_multi_pkl.py`
- `tests/test_datacard_late_rebin.py`
- `tests/test_datacard_tools_selective_sumw2.py`

## Physics effects and authorities

Card construction maps selected process and category content to nominal rates,
shape templates, rate/shape nuisances, MC-stat terms, and EFT scaling records.
Rate-systematic files own configured values and applicability; correction
variation labels own the candidate shape inputs; the missing-parton contract
owns its category/payload layout; sumw2 owns statistical companions; and the
binning resolver owns the exact processing-to-fitting view.

### Final nuisance identities

The [generated exhaustive nuisance-policy and standardization catalog](systematics_nuisance_catalog.md)
lists the current static DatacardMaker policy and every standardizer mapping row.
The explanation below gives the interpretation and ownership boundary.

The source variation label is an input, not necessarily the final card nuisance
identity. `DatacardMaker` applies `syst_year_corr` in `correlate_years(...)`
while merging periods, then writes the final TXT nuisance rows and paired ROOT
templates. For Run 3 DeepTau2018v2p5 `VSmu`, its source/period check rejects
ambiguous or mismatched legacy labels; the maintained final groups are
2022 with 2022EE and 2023 with 2023BPix, separately for each eta bin. Run 3
electron-fake tau energy scales (`genElectron`, decay modes 0, 1, 10, 11)
use the same two year groups. Run 2 DeepTau2017v2p1 `VSe`, `VSmu`, genuine-tau
energy scales, and electron-fake tau energy scales for decay modes 0 and 1
group 2016 with 2016APV; 2017 and 2018 retain separate year identities.
Other tau families have their own final identities; inspect the focused test
before inferring a shared year domain from a family name.

The optional `--suppress-nominal-like-tau-shapes` check acts after the
card-facing binning view on eligible SM processes. It considers only uniquely
paired, fully specified Tau Up/Down shapes with valid finite arrays, a
positive nominal total, nonnegative raw bins, unchanged physical underflow,
and exact agreement in nominal-zero bins. Both directions must differ from
nominal by strictly less than `1e-5` relative in every other serialized bin.
It excludes processes retaining statistical variance, data and flip processes,
and protects Run 2 genuine-tau energy-scale shapes from suppression. This is
a narrowly gated local shape omission, not a general near-nominal nuisance
policy. It is off by default and incompatible with `--use-AAC`.

The default `--lepton-mva-correlation per-run` gives `lepSF_elec_mva` and
`lepSF_muon_mva` separate `13TeV` Run 2 and `13p6TeV` Run 3 identities.
`--lepton-mva-correlation cross-run` instead gives them the unsuffixed shared
identities `lepSF_elec_mva` and `lepSF_muon_mva` in both original and
standardized TXT/ROOT cards. Electron and muon remain
separate nuisances. `lepSF_elec_non_mva` and `lepSF_muon_non_mva` remain
run-specific in either mode. The [generated nuisance catalog](systematics_nuisance_catalog.md)
lists both modes from the card writer's naming resolver. `PU` becomes `CMS_pileup_13TeV` or
`CMS_pileup_13p6TeV`; `MET_UnclusteredEnergy` becomes
`CMS_scale_met_Unclustred_energy_13TeV` or
`CMS_scale_met_Unclustred_energy_13p6TeV` (the spelling is maintained).
`PreFiring` maps to `CMS_l1_ecal_prefiring` in the naming map; this identity
does not assert a nonzero effect in every era or package. The luminosity
identities include Run 2 `lumi_13TeV_1516_l`, `lumi_13TeV_151617_l`, and
`lumi_13TeV_15161718_l`, and Run 3 `lumi_1` and `lumi_2`. Theory rate and
shape aliases are specified by the era entries in `SYST_NAMING_MAP`; their
names alone do not establish a blanket process or year correlation rule.

These are current implementation contracts in
[`datacard_tools.py`](../../topeft/modules/datacard_tools.py) and
[`standardize_systematics.py`](../../analysis/topeft_run2/standardize_systematics.py),
with focused checks in
[`test_tau_datacard_correlation.py`](../../tests/test_tau_datacard_correlation.py),
[`test_standardize_systematics_tau_year_correlation.py`](../../tests/test_standardize_systematics_tau_year_correlation.py),
and [`test_standardize_systematics_release_contract.py`](../../tests/test_standardize_systematics_release_contract.py).

### Package standardizer

`standardize_systematics.py` accepts a certified `TOP26006_v1` combined
datacard package through `--input-package` and a distinct, absent destination
through `--output-package`. Its `SYST_NAMING_MAP` selects final names from each
mapping row's `run2` or `run3` era. It renames TXT shape and rate nuisance rows
and matching ROOT systematic keys, rejecting nuisance and ROOT-key collisions.
It copies card order and EFT scalings, preserves logical mapping rows, and
writes a fresh package with updated provenance. It does not reconstruct
correlations from PKLs, choose scientific year groups, or construct a workspace.
The card maker has already realized the correlation topology before this
rename. The standardizer infers MVA topology from the source identities:
unsuffixed names remain shared and run-appropriate energy suffixes remain
per-run. It rejects mixed identities for either flavor and wrong-run or
retired double-energy aliases before creating the output package.
The package schemas are unchanged; full package and fit comparison of the new
mode requires a separate qualification. The [package procedure](../how_to/datacard_packages.md) and
[artifact boundary](../explanation/datacards_and_eftfit.md) give the operating
context.

When the source combined package has `config/TOP_26_006.yml`, the standardized
sibling copies its bytes and records its path and SHA-256 in `fit_config` within
`package-provenance.json`. Its standardization inventory also includes that
file. The original combined builder's `certify` compares an original package
with its per-era inputs; it does not certify the rewritten sibling against
those inputs. `build_fit_input_package.py` accepts the sibling with an explicit
`--fit-config` path to its package-local YAML, reads both inputs without
changing them, and copies selected TXT/ROOT pairs, ordered inputs, scalings
and config to a fresh fit-input package. It validates the selected consumer
contract and records copied-byte hashes; it does not combine cards, build a
workspace or run a fit.

The ineffective legacy rate/envelope override is not an active default.
Current era-selected rate-systematic files and the implemented consumer remain
authority. Numeric inputs and current effects can be documented without
inferring their scientific derivation.

Category-to-distribution behavior starts from
[categories and observables](categories_and_observables.md). EFT projection
uses the shared `HistEFT` mechanism under the analysis policy described in
[HistEFT](histeft.md).

## Practical defaults and change bridge

Current numerical/applicability authorities include
[`rate_systs_run2.json`](../../topeft/params/rate_systs_run2.json),
[`rate_systs_run3.json`](../../topeft/params/rate_systs_run3.json), the
[`missing_parton_contract.py`](../../topeft/modules/missing_parton_contract.py)
registry/payload layout, and the fitting views in
[`axis_binning.py`](../../topeft/modules/axis_binning.py). The selected-WC and
scaling record are category/process dependent, so one representative record is
not a universal default.

A concrete chain is physical category + distribution → fitting view → nominal
and systematic card templates → selected WCs → physical-channel scaling
record. Use
[create cards and finalize EFT scalings](../how_to/datacards_and_scalings.md)
to change rate files, applicability, binning, selected WCs, or export. Use the
[missing-parton payload guide](../how_to/missing_parton_payloads.md) for that
separately owned payload family.
