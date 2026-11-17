# Produce SR extension PKLs

`run2_sr_ext` and `run3_sr_ext` produce the approved CR-like distributions
in SR. Their scientific default remains
[the 12-unit plan](../../analysis/topeft_run2/sr_reco_default_plan.json).
The eight approved histograms are `fwd0pt`, `j0pt`, `b0pt`, `met`,
`l0conept`, `l1conept`, `invmass`, and `tau0Tpt`. The channel registry
determines where forward and tight-tau observables apply.

The same [campaign engine](../../analysis/topeft_run2/campaign_manifest.py)
owns state, plans, revisions, attempts, receipts and recovery for SR extension,
production and statistical diagnostic component profiles. New campaigns write
schema 9 to `.<profile>_campaign_state_v9.json`. The original schema-8 state
file, if present, is never rewritten by this engine.

## Review a campaign plan

From `analysis/topeft_run2`, inspect a fresh campaign before scheduling work:

```bash
./run_cr.sh --production-profile run3_sr_ext \
  --output-dir "$campaign_dir" --campaign-tag "$campaign_tag" \
  --plan "$plan_file" --chunksize "$chunk_size" \
  --env-file "$worker_archive" --dry-run
```

Set each variable to an absolute path or value. Omit `--plan` to use the
12-unit scientific default. A supplied JSON plan has one `units` array;
each unit has a unique snake-case `id`, `category_groups`, `histograms`,
and optionally `years`. The engine rejects added, omitted or overlapping
scientific coverage. It permits a different partition of the same approved
coverage. `--chunksize` is any positive integer accepted by the downstream
source command; it is recorded for each source attempt and does not change
campaign science.

A real Work Queue run requires the guarded staging options described below.
The `--dry-run` form resolves commands without creating a campaign root or
submitting work.

## Use the shared launcher for combined profiles

`run_cr.sh` also owns combined Run 2/Run 3 production and statistical diagnostic
profiles. Start or resume a combined profile with one top-level invocation;
the launcher runs the Run 2 component before the Run 3 component. Each child
has its own state and output root under the combined output directory, and the
launcher derives separate component tags from the supplied campaign tag.
Plan selection and revision remain component-only operations.

For an existing combined production campaign, use the same profile, absolute
root and tag that established its state. Supply both manager prefixes together
when prefix-derived names are needed:

```bash
./run_cr.sh --production-profile run2_run3_prod --resume \
  --output-dir "$existing_campaign_root" --campaign-tag "$same_campaign_tag" \
  --env-file "$worker_archive" \
  --run2-wq-manager-prefix "$run2_prefix" \
  --run3-wq-manager-prefix "$run3_prefix" --wq-port 0 \
  --wq-staging-base "$staging_base" \
  --wq-free-warning-bytes "$warning_bytes" \
  --wq-min-free-bytes "$minimum_bytes" --wq-poll-seconds "$poll_seconds"
```

Set the variables to the existing campaign identity and a dedicated local
staging path selected for this run. Choose the warning, minimum and poll values
for that path. Supported common operational options, including `--resume`,
`--env-file`, `--wq-port` and the Work Queue guard settings, are forwarded to
each child. The paired prefixes select separate Run 2 and Run 3 manager-name
families; a prefix-derived source name ends with its unit ID and attempt number.
The manager name is distinct from its TCP port. `--wq-port 0` asks Work Queue
to allocate an available manager port; inspect the port reported by that
invocation rather than treating zero as the listening endpoint. Omit the flag
to retain the lower layer's default port behavior.

Resume validates the existing component state and receipts. Reusable source or
completed work is retained; an eligible failed or incomplete stage receives a
new attempt while earlier attempt records remain in state. An ambiguous running
writer or unexpected output requires inspection before another invocation.
Do not start a fresh campaign at an existing output root.

## Select a unit and resume

`--selected-fragment run2_sr_ext_a_1` executes one named unit while the
state retains the complete approved plan. A later `--resume` validates the
scientific contract, scientific coverage, state structure, artifact receipts
and exclusive writer evidence. A different chunksize, manager resource choice,
log location or relocated byte-identical environment archive creates new
operational provenance; it does not change scientific identity. Campaign
science and approved coverage remain fixed. The frozen archive SHA-256 is the
source worker implementation authority: a different archive SHA requires a
new campaign root and state (`NEW_CAMPAIGN_REQUIRED`), including during retry,
plan revision or schema-8 import. Live checkout and manager fingerprints are
operational provenance, not source reuse gates.

The ordered stages are `source` and `data_driven`. The source child retains
`--do-np --defer-np` and publishes a verified source receipt before
`run_data_driven.py` starts. If the downstream stage fails, a resume reuses
the verified source and retries only downstream. A failed source with no
verified artifact may be retried. Under the exclusive campaign lock, an
interrupted running attempt with no current writer is recorded as interrupted
and returned to a recoverable stage. When its PKL and maintained metadata
sidecar validate against the recorded attempt, the engine adopts that source
or downstream result without rerunning it. Source recovery validates the real
processor sidecar, including its exact CFG set, applicability, sample contract,
layout, stat policy and artifact identity. Wrapper spelling remains provenance,
not a recovery gate. Both SR and CR require nonprompt and charge-flip
preparation in the source artifact. The ordinary deferred
`_np.pkl.gz` result must be `nonprompt_output`; a flips or processor sidecar
cannot stand in for it. Downstream recovery reads the maintained transformed
sidecar, including lineage, product and transformation contracts, sumw2
manifest, sample contract, applicability, layout and PKL identity. An
unverified output or uncertain writer ownership stops recovery without
deleting or overwriting the candidate.

The state stores campaign, source-stage and data-driven-stage scientific
contracts plus a separate approved coverage digest. Source-stage implementation
identity is the frozen archive SHA-256. The local data-driven stage binds its
estimator, nonprompt policy, product-name rule, application-region mapping,
resolved prompt alias list, entry-point implementation and exact source
artifact identity. Generic containers, serialization and provenance helpers
remain runtime dependencies. A downstream-only science change can retain a
valid source receipt, while an old downstream result cannot be reused under
that changed calculation. The active plan revision and partition digest
describe how that coverage is grouped. Source and downstream receipts bind the contract,
coverage, unit coverage, revision, attempt and artifact lineage. The summary
files `campaign_summary.tsv` and `campaign_summary.md` derive from the
same state. PKL existence alone is not completion evidence.

For a deliberate full payload check, run the maintained engine's
`audit-integrity` action in the normal production environment:

```text
python ./campaign_manifest.py audit-integrity \
  /absolute/existing/campaign/.run3_sr_ext_campaign_state_v9.json
```

This hashes accepted PKLs and checks them against their receipts. Ordinary
resume checks stored receipts, size, sidecars and lineage without rereading
accepted PKL payload bytes.

## Revise only pending coverage

An explicit plan revision uses `--resume --activate-pending-plan PLAN
--expected-plan-revision REVISION`. It takes the exclusive lock, compares the
active revision, verifies exact approved coverage and keeps completed or
source-ready unit coverage frozen. Untouched pending or failed source units
without reusable source may be repartitioned. The transition is atomic and
does not submit work. A stale revision, output collision, changed coverage or
ambiguous writer blocks activation.

## Recover an immediate schema-8 predecessor

The schema-8 SR extension state is historical and is not a normal schema-9
resume target. `campaign_manifest.py import-schema8` is an explicit,
non-destructive recovery route for a current SR extension predecessor. It
requires the original state, a distinct schema-9 target, matching scientific
policies and coverage, the historical fingerprints actually recorded, and
verified completed receipts. A recorded `source_running` attempt with a valid
real producer sidecar can be reconciled after import; planned units get no
invented historical attempts. The original state bytes remain unchanged.
New schema-9-only downstream fingerprints bind when downstream first runs.
When required historical evidence is missing, the import reports
`recovery_evidence_insufficient`; it does not infer compatibility.
Schemas 5, 6 and 7 and retired public profile names
have no maintained import or resume alias.

## Work Queue staging and stop conditions

Every real Work Queue route, including CR and direct `run_analysis.py`, needs
an explicit guarded staging base. Choose a dedicated local directory for the
run. Through `run_cr.sh`, provide `--wq-staging-base PATH`,
`--wq-free-warning-bytes`, `--wq-min-free-bytes` (at least 10000000000),
and `--wq-poll-seconds`. A direct `run_analysis.py -x work_queue`
invocation requires `--wq-filepath`,
`--wq-guard`, a unique invocation ID and the same free-space values. The
guard requires an existing absolute path that resolves to a directory owned
and usable by the current user. It rejects symlink paths and temporary
locations under `/tmp` or `/var/tmp`; it also checks that enough free bytes
and inodes remain for the configured thresholds.
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

Before launch, the guard requires available bytes strictly above the selected
minimum and at least 1,000 free inodes. The minimum is at least 10 GB
(decimal). The `Work Queue staging health` message reports available staging
space in bytes and free inodes, not worker RAM. A site that explicitly
configures quota telemetry may use that information in its capacity reading;
check the site's quota configuration and measurement behavior before relying
on it. During execution the guard reports these values without scanning the
staging tree.
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
