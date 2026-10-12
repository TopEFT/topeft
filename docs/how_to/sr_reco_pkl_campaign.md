# Produce and qualify SR-reco extension PKLs

Use `run2_sr_ext` and `run3_sr_ext` in `analysis/topeft_run2/run_cr.sh`
for the maintained SR-reco extension source campaigns. The current
`run2_sr_ext` and `run3_sr_ext` profiles each materialize 15 fragments from
five SR category groups and three variable chunks per group. This is the
current profile grouping, not a fixed fragment count for every future SR
campaign. The exact block and artifact names come from the maintained profile
configuration; see
[production configuration](../reference/production_configuration.md) for the
category/variable matrix.

This guide specifies launch and evidence requirements. It does not certify a
particular campaign or replace the artifact schemas in
[histogram artifacts](../reference/histogram_artifacts.md).

## Resolve source and worker authorities

The manager and local postprocessing execute from the live checkout.
`analysis_processor.py` is separately transferred to Work Queue as
live-checkout-controlled source. Run 2 performs inline local
`DataDrivenProducer` processing during `run_analysis.py`; Run 3 runs the local
`run_data_driven.py` stage after its source stage completes. Worker-only
imports, runtime libraries, and correction payloads come from the explicitly
selected frozen worker environment. Thus the worker archive's TOPEFT and
TOPCOFFEA source provenance and the manager/local TOPCOFFEA checkout are
distinct authorities.

Prepare an environment archive through the canonical
`run_analysis.py --prepare-env-only` path, backed by
`topcoffea.modules.remote_environment.get_environment` (documented in
TOPCOFFEA's `docs/environment_archive_contract.md`), and select its absolute
path with `run_cr.sh --env-file`. Record the resolved
environment request/fingerprint, archive path and SHA-256, frozen TOPEFT source
identity, worker TOPCOFFEA identity, and live manager/local source identities.
An explicit archive is a frozen snapshot: validate its integrity and
provenance; do not interpret its selection as proof that it matches the current
checkout. No particular historical tarball is the permanent worker
environment.

Before each launch, assess compatibility against the maintained contracted
live-manager dependencies, the live local data-driven and transferred
processor source, the manager/local TOPCOFFEA authority, and the explicit
frozen archive. Equality of the whole live TOPEFT HEAD to the frozen TOPEFT
commit is not the launch gate. A change in a contracted dependency requires
review before execution. If the campaign namespace encodes a worker-frozen
source identity, keep that identity consistent with the selected archive;
the workflow does not prescribe a universal date or short-SHA spelling.

## Run 2, then Run 3

Use the maintained `run_cr.sh` workflow and one campaign output root. Keep the
terminal transcript and exact process exit status for each profile. Workers
are provisioned separately with the manager name printed for each Work Queue
source invocation.

1. Before `run2_sr_ext`, rerun the applicable launch-compatibility preflight;
   require the fresh output root to be absent; inspect the scheduler situation;
   then run the Run 2 command with the explicit `--env-file`, absolute
   `--output-dir`, and `--campaign-tag`.
2. Start `run3_sr_ext` only after the Run 2 process returns zero. Rerun the
   compatibility preflight and inspect the scheduler situation immediately
   before Run 3. Use the same campaign namespace and archive. There is no
   second fresh-root absence gate: Run 2 has already established the
   shared output root.

A zero process status is necessary, but campaign state, receipts, artifacts,
sidecars, and logs still need post-run review. In Run 2, each fragment's source
and `_np` artifacts are produced by the inline path. In Run 3, the local
`run_data_driven.py` child follows successful source publication. Do not start
the deferred stage from an incomplete source artifact.

## Native Work Queue logs and recovery

The campaign assigns each source invocation a native log directory:

```text
<output_root>/work_queue_logs/<production_component>/<block_id>/source_native/
```

The four native files are `debug.log` (manager diagnostics), `tr.log`
(transactions), `stats.log` (statistics), and `tasks.log` (task history).
After verified archival, the campaign records the corresponding `source/`
directory in state. An existing owned native log prevents a new invocation;
checkout-global log files are not campaign-owned state. See
[Work Queue operations](work_queue.md) for direct-run defaults and archive
behavior.

A nonzero return, interruption, lost terminal, unclear completion, ambiguous
owned or stale native logs, or ambiguous campaign state is a stop for
classification. Preserve the transcript, state, receipts, artifacts, and
native logs. Inspect process and scheduler ownership and the exact output
paths before choosing any recovery action. Do not automatically retry,
resume, delete, overwrite, restart, or resubmit a fragment or stage. The
presence of a source PKL alone does not establish that its sidecar or deferred
`_np` stage completed.

## Return evidence and qualify artifacts

For each profile, retain its command transcript and exact exit status, final
`run_cr.sh` campaign state and block receipts, any campaign summary or failure
snapshot, and the exact source/`_np` PKL and sidecar inventory with paths and
sizes. Return the worker-archive identity and live-source provenance bound to
the campaign, plus native Work Queue logs when needed to explain completion or
recovery. Use the runner's serialized state as the primary completion record;
do not reconstruct it from filenames alone.

After successful Run 2 and Run 3 production, qualify the extension campaign
with the checks below against the maintained profile configuration:

- Match expected fragment membership for each current 15-fragment profile,
  including source and `_np` realization. Confirm Run 2 inline and Run 3
  deferred local data-driven stage order from state, receipts, and sidecars.
- Reconcile each PKL with its adjacent sidecar, source-to-`_np` lineage,
  campaign state, and receipt. Check path ownership, nonempty size, artifact
  digest, and the sidecar's recorded size and digest.
- Compare the recorded family/category inventory and applicability with the
  profile plan. `fwd0pt` and `fwd0eta` have support only in forward-labelled SR
  categories; `tau0Tpt` has support only in categories with at least one tight
  tau. `tau0Fpt` is absent from the SR-reco extension variable set. Distinguish
  an intentionally inapplicable channel from a missing expected fill.
- Check the applicable systematic variations, including expected Up/Down
  templates; the resolved production `sumw2` companions and data-driven
  contract; and the raw event-count setting recorded by the profile. These
  extension profiles use systematics, implicit production `sumw2`, and no
  raw-count recording.

Qualification is a separate post-production step. This checklist does not imply
that any current production run or artifact set has passed it.
