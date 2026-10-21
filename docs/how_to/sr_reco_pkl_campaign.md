# Produce SR-reco extension PKLs

`run2_sr_ext` and `run3_sr_ext` are the maintained SR-reco profiles in
`analysis/topeft_run2/run_cr.sh`. Each uses the eight histograms `fwd0pt`,
`j0pt`, `b0pt`, `met`, `l0conept`, `l1conept`, `invmass`, and `tau0Tpt` where
they apply. `fwd0pt` is restricted to forward-labelled SR channels and
`tau0Tpt` to tight-tau channels. `b0pt` is the leading tagged b-jet pT
histogram; its numerical bins match `j0pt` while its key and label remain
distinct. Fit observables around b-tag classification are a separate analysis.

The supplied [default plan](../../analysis/topeft_run2/sr_reco_default_plan.json)
contains 12 units per profile. `--sr-plan /absolute/plan.json` selects another
complete, valid initial partition for a fresh campaign. Its unit count depends
on the supplied plan; it does not change the default. The plan has one `units`
array; each unit has a unique snake-case `id`, `category_groups` list and
`histograms` list.
Each unit describes a category-group × histogram product. The launcher checks
that its applicable physical category × variable coverage is exact and unique
against the live channel registry. It derives the target size from that
registry. It does not accept unknown groups, keys, duplicates or gaps.
The default splits `l0conept` from the `j0pt b0pt` mixed-category chunk:
the largest category × axis-bin proxy is 2,112, versus 2,684 for the
otherwise equivalent 11-unit packing. This proxy is not a measured memory
or output-size limit.

## Start a full Run 3 campaign with an explicit plan

Choose a complete plan that covers the current SR category-group × histogram
target exactly once. Use an absolute path for `--sr-plan`, a fresh absolute
output directory, and a campaign tag. From `analysis/topeft_run2`, start the
whole plan without `--selected-fragment` or `--resume`:

```bash
./run_cr.sh --production-profile run3_sr_ext \
  --output-dir "$campaign_dir" --campaign-tag "$campaign_tag" \
  --sr-plan "$plan_file" --chunksize "$chunk_size" \
  --env-file "$worker_archive" \
  --wq-staging-base "$staging_base" --wq-quota-root "$quota_root" \
  --wq-free-warning-bytes "$warning_bytes" \
  --wq-min-free-bytes "$minimum_free_bytes" \
  --wq-poll-seconds "$poll_seconds"
```

Set each shell variable to the corresponding absolute path or value before
running the command. For an SR extension profile, choose a `--chunksize` of
100000, 75000, 50000 or 25000; omitting it selects 100000. For an ordinary
`--resume`, repeat the persisted campaign's chunksize so pending units keep
the same choice. The five `--wq-*` options shown are required staging guard
parameters for a real Work Queue campaign. The launcher also accepts optional
`--wq-treereduction`, `--wq-processing-cores`, `--wq-accumulating-memory-mb`
and `--wq-accumulating-cores`; the example sets none of them. See
[Work Queue staging and stop conditions](#work-queue-staging-and-stop-conditions)
for their constraints and defaults. The staging base must be an existing owned
directory on a non-root filesystem, and the minimum free-space value
must be at least 10 GB (decimal). Supply `--env-file` only when using a
prebuilt worker archive as an explicit snapshot; its path must be absolute.
Omit that option to resolve or create the current checkout's worker environment.
The archive and its adjacent manifest are validated before campaign work,
without resolving an environment request from the current checkout.
The snapshot freezes the worker environment, while the manager runs from the
current checkout and transfers `analysis_processor.py` separately. It does not
freeze all runtime code or establish worker compatibility by itself.

For a new worker archive, follow the native builder procedure in the
[production guide](production.md). Inspect the resolved plan, environment
identity, staging resources and output paths before starting production.

Each SR-reco profile owns its own fresh campaign root. An internal plan
revision stays inside that profile's existing root. No output is copied or
adopted from another root.

## Selected units and continuation

A selected-unit invocation uses the complete persisted plan. For example,
`--selected-fragment run2_sr_ext_a_1` runs one ordinary unit in the default
Run 2 plan rather than launching every unit. A later `--resume` recovers an
existing campaign in the same root and skips verified completed units. Run 2
and Run 3 now write the source artifact first and perform `_np`
in a separate local child. The source child retains `--do-np` to certify the
nonprompt product and sumw2 policy; `--defer-np` changes when postprocessing
runs. A normal successful source child publishes a checked source receipt
before `_np` starts. A failed `_np` leaves that receipt and source attempt
unchanged, and a selected-unit resume retries only `_np`. The campaign is
complete only after both stages have verified receipts for all applicable
coverage.

The state file is `.<profile>_campaign_state.json` in the campaign root. It
stores immutable plan snapshots, the active revision, per-unit source and
nonprompt status, artifact/sidecar receipts, environment and source contract,
and a coverage target. `campaign_summary.tsv` and `campaign_summary.md` read
that same state. A source or `_np` PKL's existence alone is not completion
proof. Keep the state, both sidecars, native Work Queue logs, output paths and
command transcript together for review.

After the run, inspect `.<profile>_campaign_state.json`,
`campaign_summary.tsv`, and `campaign_summary.md` in the output directory.
Check each planned unit's source and nonprompt status, its expected PKLs,
adjacent sidecars, and recorded receipts. The required artifact count follows
the persisted plan and its revisions; a shell exit or a fixed PKL count alone
does not establish completion.

When a new source or `_np` output is accepted, the native owner hashes its PKL
bytes and checks the digest and size against its sidecar before recording a
receipt. Ordinary `render-active`, `consumer-summary`, and same-campaign resume
check the stored receipts against file type, size, sidecars, lineage and bound
state without reading accepted PKL payloads again. This fast readback cannot
detect a same-size change to accepted PKL bytes. For a deliberate full-byte
integrity check, run `python ./sr_reco_manifest.py audit-integrity
/absolute/existing/campaign/.<profile>_campaign_state.json` from
`analysis/topeft_run2`. It rehashes every accepted source and `_np` PKL and
fails if any payload differs from its recorded receipt. A full audit is not a
condition for ordinary resume or summary.

An occupied schema-6 campaign can opt into per-pending-unit runtime choices
through one explicit state transition in the native manifest owner. With the
campaign quiescent, the exact historical commit string recorded in the state,
unchanged bound input/sample/channel/axis files and worker archive, valid
completed receipts, and collision-free pending outputs must pass before the
atomic transition. The commit string is an identity check against the state,
not a live Git check:

```text
python ./sr_reco_manifest.py upgrade-runtime-policy \
  /absolute/existing/campaign/.run2_sr_ext_campaign_state.json \
  HISTORICAL_TOPEFT_COMMIT_FROM_STATE
```

Run this command from `analysis/topeft_run2` in the normal production shell.
It takes the existing campaign lock and keeps the original schema-6 contract,
completed receipts, plan revisions, source commands and output names unchanged.
The upgraded state records a separate scientific/input contract for new units.
The original `sr_reco_chunk_size` remains historical provenance. A selected
untouched pending unit may then use `--chunksize 25000` while another uses the
earlier chunk size. Each new source attempt retains its runtime snapshot; the
receipt binds the latest successful attempt's resolved chunk size,
tree reduction, processing core maximum, accumulating minima, worker archive
identity, manager file hashes and plan revision. These manager hashes describe
the new attempt; they do not claim a historical commit or freeze later pending
units. Resume still checks the historical source and `_np` sidecars and
their lineage. A historical Run 3 `source_ready` unit keeps its original source
receipt and contract while its separate `_np` stage records its own command and
completion provenance. Changed bound inputs are rejected.

New SR campaigns use schema 8 and record the supported separate-stage policy.
The historical `nonprompt_mode` and its contract hash remain unchanged when an
older campaign is explicitly activated. The current scientific contract
excludes only that technical timing choice; it still binds the campaign,
environment, sample inputs, selections and coverage. Schema 6 and 7 states
remain readable. An older inline campaign must be quiescent and activate the
stage policy before resume:

```text
python ./sr_reco_manifest.py activate-stage-policy \
  /absolute/existing/campaign/.run2_sr_ext_campaign_state.json \
  EXACT_ACTIVE_PLAN_REVISION
```

Activation takes the campaign lock, checks historical contracts, accepted
receipts and output collisions, then atomically publishes schema 8. Repeating
the same activation leaves state bytes unchanged. It does not adopt a failed
source or launch a child. A source produced by a failed historical inline
command requires a separate explicit `adopt-source` transition, with an
owner-accepted JSON evidence record binding campaign, revision, unit, command,
attempt, runtime, environment, scientific contract, source receipt, completed
publication and historical composite outcome. The transition verifies the
source and sidecar, CFG identities, namespace and absence of `_np` output;
the evidence may request a fresh full payload SHA-256 check. It preserves the
historical composite exit and signal while recording the independently
unobserved source-only exit as unknown. Adoption never starts source or `_np`.
An incomplete or conflicting record leaves state unchanged. Production
recovery still needs its own authorization and selected-unit resume decision.

## Repartition only untouched coverage

Prepare a JSON plan containing **only** still-uncovered units. Any valid
category-group × histogram repartition is accepted if its union is exactly
remaining physical coverage and every new unit ID and output path is unused.
All completed units remain in the original plan revision with unchanged
artifacts and receipts. The new plan is appended to the same state file and
activated by one atomic state replacement:

```text
./run_cr.sh --production-profile run2_sr_ext \
  --output-dir /absolute/existing/campaign --campaign-tag chosen_tag \
  --resume --activate-pending-plan /absolute/pending.json \
  --expected-plan-revision r1_exact_observed_id
```

Activation is a separate quiescent operation; it submits no work. Read the
current `active_sr_reco_revision` from state immediately before it. The
campaign lock must be available. Failed, interrupted, in-flight or otherwise
ambiguous units, stale revision IDs, changed source/input/binning contracts,
missing or changed artifact receipts, and output collisions reject the
replacement before changing state. A finished Run 3 source with a verified
receipt and pending local `_np` is retained; it is not called fully complete.
An interrupted atomic write leaves either the old state or the entire new
state authoritative. Do not edit state or receipts by hand.

After activation, ordinary `--resume` runs the retained deferred stages and
active pending units in the same root. `--selected-fragment` can filter a
single active unit without altering the full plan. The launcher checks
coverage across historical completed units and the active pending revision;
it does not decide completion from only the latest revision's rows.

## Work Queue staging and stop conditions

Every real Work Queue route, including CR and direct `run_analysis.py`, needs
an explicit guarded staging base. Through `run_cr.sh`, provide
`--wq-staging-base`, `--wq-quota-root`, `--wq-free-warning-bytes`,
`--wq-min-free-bytes 10000000000`, and `--wq-poll-seconds`. A direct
`run_analysis.py -x work_queue` invocation requires `--wq-filepath`,
`--wq-guard`, a unique invocation ID and the same free-space values. The
base must be an existing owned writable directory on a non-root filesystem;
implicit `/tmp` and root-backed staging are rejected by mount identity.
`--wq-treereduction` is optional and requires separate resource review.
`--wq-processing-cores 1` is a separate opt-in processing-category reservation
maximum. `--wq-accumulating-memory-mb 80000 --wq-accumulating-cores 2`
optionally sets an accumulating-category minimum after Coffea declares its
categories. Both positive integer values are required together. This does not
change automatic resource mode, processing reservations, or the default tree
reduction of 10. Two reserved cores isolate a reducer on a fully two-core
worker; a four-core worker may still run another task. Minimum memory is not a
peak RSS cap or an OOM guarantee. Worker eligibility and packing depend on
actual advertised resources.

Before launch, the guard requires effective available bytes strictly above the
selected minimum, calculated as the smaller of filesystem free availability
and group quota remaining, plus 1,000 free inodes. The minimum is at least 10
GB (decimal). The `Work Queue staging health` message reports available
filesystem/quota space in bytes and free inodes, not worker RAM. During
execution the guard reports these values without scanning the staging tree.
Only a valid current measurement at or below
`--wq-min-free-bytes` requests one controlled `SIGTERM` to the manager after
its PID/start identity and owned marker are verified and the critical-space
reason, observed bytes, threshold and timestamp are written to the marker and
log. No `SIGKILL` or `SIGINT` is sent. Low inode counts, staging growth,
unavailable telemetry, watcher loss and other warnings do not signal the
manager. The final manager readback rejects a dead watcher or changed ownership
marker before recording source success; ambiguous staging remains for inspection. The older
`--wq-warning-bytes`, `--wq-stop-bytes` and `--wq-reserve-bytes` options are
retired. Native logs are copied and checked before only the completed owned
staging is removed. Cleanup refusal retains staging and prints a warning; a
valid source receipt remains valid. A failed source stops later guarded units. Preserve
an ambiguous state and inspect the exact process, scheduler, artifact and log
ownership before any separate recovery decision.

An explicit worker archive is a frozen snapshot, not proof of processor or
worker compatibility. Qualify the archive against the intended analysis before
production. Confirm staging capacity, scheduler readiness and worker resources
for the selected plan. An occupied campaign is resumed only after its native
state, receipts, staging and collision checks pass.

`--dry-run` resolves commands without creating a campaign root or submitting
workers. It does not qualify an archive, site capacity, physics content or
production outputs.
