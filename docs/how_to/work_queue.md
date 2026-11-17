# Run with Work Queue

`analysis/topeft_run2/run_analysis.py` is the maintained Work Queue manager
entry point. It prepares the Coffea `WorkQueueExecutor`, constructs or accepts a
worker environment, stages repository inputs, and waits for independently
launched workers. There is no maintained `work_queue_run.py` entry point in the
current workflow.

`run_analysis.py --wq-manager-name NAME` passes a nonempty name unchanged to
the Work Queue executor. Without it, the name remains
`USER-workqueue-outname`. The option fails for other executors. `run_cr.sh`
forwards the same option through the unchanged `fullR3_run.sh` argument path,
after confirming that exactly one component source manager is due to start.
The manager name selects operational routing and attempt provenance; it does
not enter the campaign or stage scientific contracts.

Use Work Queue only after a small `futures` or `--pretend` check has established
that the samples, options, and output identity are correct. Distributed
execution changes where tasks run; it does not change the processor, histogram,
sumw2, or artifact contracts.

## Know the manager/worker boundary

| Concern | Owner/default |
| --- | --- |
| manager identity | `run_analysis.py` derives `${USER}-workqueue-${outname}` |
| port | parser default `9164-9170`; the resolved first port is passed to Coffea unless configured otherwise |
| worker environment | current archive is automatically built/resolved and validated, or an exact `--env-file` is validated |
| executor mechanics | `run_analysis.py` constructs `WorkQueueExecutor` with retries 15, compression 1, automatic measured resources, split-on-exhaustion, and `/tmp` executor workspace |
| manager logs | `debug.log`, `tr.log`, `stats.log`, and `tasks.log` under an owned `source_native/` directory; campaign execution archives them under `source/` |
| workers/site submission | operator or site factory; no wrapper launches them |
| processor and artifact meaning | unchanged `AnalysisProcessor`, policy, and sidecar contracts |

`run_cr.sh` and `fullR3_run.sh` may select/forward Work Queue and a frozen
environment, but `run_analysis.py` remains the executor configuration owner.
The worker environment transports code/dependencies; it does not replace the
repository commit and environment identity recorded for the campaign.

## Read-only operator dashboard

From `analysis/topeft_run2`, run `python3 wq_dashboard.py`, or refresh it with
`watch -n 10 python3 wq_dashboard.py`. It uses only the Python 3.9+ standard
library and read-only Work Queue, Condor, history and local process queries.
Set `OWNER` and `SCHEDD` when their defaults are unsuitable. Local factory
inspection is available only when `OWNER` is the current user. The global catalog
section lists records that explicitly name that owner, with each record's age;
stale records are labeled. A selected manager has a separate direct-query view.

Set `MANAGER_PROJECT` to one exact catalog project to resolve a fresh endpoint.
Zero or multiple fresh owner matches leave selected-manager detail unavailable
while owner-wide sections continue. Without `MANAGER_PROJECT`, the existing
`MANAGER_HOST` and `MANAGER_PORT` endpoint path remains available. Catalog and
direct results are separate snapshots and can time out independently.

`TASK_DETAILS=0` is the default. Set `TASK_DETAILS=1` to request the selected
manager's full task payload; `TASK_QUERY_TIMEOUT` bounds that query.
`MAX_ACTIVE_TASKS` limits displayed rows only and does not bound the RPC
payload. Condor worker profiles and pools cover identified owner WQ jobs across
all managers. A wrapped worker is identified by its `condor.sh` or
`with_oasis_certs` invocation, regardless of scratch-directory naming. Pool
origin matching a visible factory scratch directory is provenance correlation,
not proof of a worker's Work Queue connection. Condor `RUNNING` likewise does
not prove connection or task activity. A missing
visible factory PID may reflect process-namespace isolation.

Recent worker exits come from an owner-scoped history query with default
`HISTORY_LOOKBACK_SECONDS=300` and `HISTORY_MAX_JOBS=500`; the query also has
a scan cap and a separate bounded `HISTORY_QUERY_TIMEOUT` (default 20 seconds).
`UNAVAILABLE` means history could not be read, not zero exits.
Set `CGROUP_JOB_IDS` to explicit comma- or space-separated Condor IDs to request
read-only cgroup-v2 memory values for those jobs. Only `RUNNING` jobs in the
current owner Condor snapshot are eligible. `MAX_CGROUP_PROBES=4` and
`CGROUP_PROBE_TIMEOUT=8` bound the optional SSH reads. With the ID variable
unset or empty, no cgroup SSH probe runs. Host or task matching alone does not
authorize a probe.
The live-factory table reports JSON `timeout` as `worker_timeout` and CLI
`--factory-timeout` as `factory_timeout`; `?` means that value was not observed.
Exit codes and observed active jobs above a readable factory maximum are
observations, not diagnoses or automatic repair instructions. The dashboard
does not submit, remove, resize, start or stop jobs or factories.

## Start the manager

From `analysis/topeft_run2`, choose the executor explicitly and give the output
a unique name. This name is also used in the manager identity:

```bash
python run_analysis.py \
  ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
  --executor work_queue --chunksize 128000 \
  --outpath /absolute/path/to/output --outname wq_check
```

`run_analysis.py` accepts `--port PORT` or `--port PORT_MIN-PORT_MAX`; it
validates and resolves that setting before creating the executor. Read the
printed manager name and port from this exact invocation rather than reusing a
name from an older guide.

For a maintained campaign, `run_cr.sh --wq-port PORT` accepts the literal `0`
or decimal `1` through `65535` without leading zeros, then forwards it through
`fullR3_run.sh` to the direct `--port` option. Zero requests
automatic allocation of an available manager port; it is not the final
listening port. A positive value requests an explicit port. With no
`--wq-port`, the direct parser's default remains in effect. The manager name
used for worker discovery is separate from this TCP port; read both from the
current manager invocation.

For a direct Work Queue invocation, the default native log directory is
`<outpath>/work_queue_logs/<outname>/source_native/`. `--wq-log-dir PATH`
selects an explicit native log directory. The runner refuses to overwrite any
of the four owned log filenames, including existing symlinks.

### Guarded processing admission

Both guarded Work Queue launchers install the same queue-local Coffea 0.7.22
admission coordinator through `custom_init`. It is idempotent across the
preprocessing and processing callbacks. Unguarded Work Queue and other
executors retain their prior behavior. The coordinator never selects a policy
from a manager name, port, project, campaign, dataset, or launcher name.

The initial ready reserve is `max(10, ceil(2 × C_processing))`, where
`C_processing` is a conservative one-slot-per-compatible-connected-worker
estimate from native worker resource summaries, including busy workers. It is
not a limit on running work or a request for Condor jobs. At zero connected
workers the minimum provides a bounded bootstrap; an unavailable summary uses
a valid recent estimate for at most 30 seconds, then the labeled bootstrap.
Memory and disk summaries are rounded down by CCTools. Declared Coffea and
launcher resource bounds are checked against each connected profile; borderline
rounded profiles remain unknown. Unspecified proportional dimensions are
worker-relative and appear as `null` in telemetry. The native category
`first_allocation` is calculated on the largest worker and is never applied
as a universal threshold. A profile that meets declared bounds while leaving
dimensions to automatic allocation contributes to a labeled compatibility
estimate, since learned native requirements may change; native matching decides actual
placement. Cores and GPUs are checked as well.
The coordinator admits at most the ready deficit, estimated unsubmitted input,
512 new tasks, and 0.25 seconds of submission work in one pass. It then yields
to Coffea's wait/result loop. The task-count estimate is a provisioning hint
derived from remaining event count and current chunksize; input exhaustion
publishes zero. Retry and split children bypass ordinary new-input admission,
retain Coffea's event accounting, and are counted separately.

The native global `hungry()` is not an additional processing gate. Native
resource matching remains authoritative. At most 32 default-priority
processing tasks per 30-second window receive priority 1 before submission
to make them visible through a large reduction backlog; explicit negative and higher
pre-existing
priorities are preserved. This preference is bounded and does not create a
separate scheduler or route categories to workers. Reduction groups retain
Coffea's size order but are admitted one feasible group at a time. Inputs
remain owned until a successful reducer output replaces them. At finalization,
deferred groups drain cooperatively through the native queue before Coffea's
final local merge. The admission policy does not fail a healthy running reducer
after a fixed silence interval. An external test harness may impose a bounded
test time; operator early termination can still return partial results.

Admission reuses the staging guard's filesystem and quota free-space
measurement. It caches measurements for five seconds between results, but
invalidates the cache after every collected result, including failures and
cleanup. This prevents a new output from being excluded by a pre-result
measurement. Actual measurement count and elapsed time are reported in
telemetry; result-heavy operation may invoke quota more often than every five
seconds. Observed free bytes already include the cost
of retained files, so the decision subtracts only estimated *additional*
outputs from active processing/accumulation and a reducer transient reserve.
The cache timestamp is the beginning of the filesystem/quota read; the two
observations can be acquired at different instants. Freshness is checked when
the full read returns. If its oldest possible age exceeds 15 seconds, no new
processing or reducer task is admitted from that sample. A later valid quick
read permits submission again.
Result arrival and input cleanup invalidate cached headroom so a later pass
measures files already materialized before releasing reservations.
The output estimate begins at 64 MiB per task and grows to twice the largest
observed processing result; it is a planning estimate, not a maximum output
size. New processing stops when the warning floor, inode floor, unknown
measurement makes further admission unsafe. With no proven reducer capacity,
Coffea-held intermediates have a bounded budget; queued reducers are already
accounted for by their transient space and do not veto processing globally.
Eligible reductions may still submit above the guard's
critical floor, using their own transient allowance, and a deferred group is
rechecked after wait timeouts without consuming or releasing its inputs.
The guard's independent critical-space behavior remains active. Shared
filesystem headroom is best effort across managers; this coordinator holds no
cross-manager reservation.

`admission.jsonl` lives beside the configured native logs. Schema version 3
records UTC and monotonic time, invocation correlation, effective config,
capacity and quality, ready/running/waiting-retrieval/retrieved processing,
held Coffea results, estimated unsubmitted work, new and recovery submissions,
useful completion counters, measured retained bytes when bounded inspection
is complete, predicted additional bytes, free-space freshness, and a decision
reason. A transition labels the old reason and its duration separately. A
2 MiB current file and one previous file bound long-run retention. Records
are periodic at most every 30 seconds plus coalesced changes; an I/O error
disables telemetry without changing scientific task
success. Compare ready-reserve starvation with native transfer timings and
dispatch counters in `tr.log`/`stats.log`, and compare useful completions,
intermediate bytes, retries and compatible idle capacity together. Later
production analytics and parameter tuning require a separate round.

Use `--prepare-env-only` to create/validate the current archive and exit before
analysis setup. `--env-file` pins a prebuilt archive; `--snapshot` permits an
explicit historical archive after integrity validation and is not a current
compatibility claim. `--rebuild-env` forces current archive recreation.
Disabling remote packaging with `--no-remote-env` makes the operator responsible
for an equivalent worker-node environment.

Wrappers such as `fullR3_run.sh` and `run_cr.sh` ultimately forward executor
selection to `run_analysis.py`. They do not launch workers.

The separate `--phase-telemetry` opt-in records processor and manager phase
observations; it is disabled by default and distinct from `admission.jsonl`.
See [phase telemetry](production.md#optional-phase-telemetry) for its
`phase_telemetry.jsonl` persistence and attempt archive paths.

## Launch workers

Launch workers from another terminal using the manager identity and resource
limits appropriate to the tasks and site. A local diagnostic worker has the
form:

```bash
work_queue_worker -dall --cores 1 --memory 8000 --disk 8000 \
  -M USER-workqueue-wq_check
```

Replace `USER-workqueue-wq_check` with the manager name printed by
`run_analysis.py`. Site batch systems can submit workers with
`condor_submit_workers` or a Work Queue factory. Worker allocation is an
operator/site responsibility; the analysis repository does not own cluster
credentials, quotas, or site submission policy.

For a factory, make the manager name match exactly and choose bounded worker
resources. JSON must not contain a trailing comma:

```json
{
  "manager-name": "USER-workqueue-wq_check",
  "max-workers": 10,
  "min-workers": 0,
  "workers-per-cycle": 10,
  "cores": 4,
  "memory": 16000,
  "disk": 16000,
  "timeout": 900
}
```

Then start the factory with the backend appropriate to the site, for example:

```bash
work_queue_factory -Tcondor -C factory.json
```

## Worker environment ownership

The manager packages the runtime environment through the current
`run_analysis.py` remote-environment support. The worker must see compatible
`topeft`, `topcoffea`, Coffea, correction payloads, and processor source. Use a
current environment file or the maintained environment construction path; do
not rely on a global `PYTHONPATH` or assume the worker sees the login node's
editable checkout.

`--wq-filepath` is an owned staging base when paired with `--wq-guard`; the
guard creates a private invocation directory and passes that directory to the
Coffea executor. It is not a general remote scratch/configuration interface.

When extending Work Queue configuration:

- keep executor and environment assembly in `run_analysis.py` rather than a
  wrapper profile;
- add a CLI/YAML option only if its precedence and validation are explicit;
- preserve staging cleanup and worker-exception reporting;
- update `tests/test_workqueue.py` and the relevant CLI/preflight coverage;
- review whether the option is also meaningful for TaskVine before sharing a
  code path.

The Work Queue executor mapping is currently assembled directly in
`run_analysis.py`. To add a supported option, define its CLI/YAML type and
default, validate it before environment/processor construction, pass it once to
the executor, show the resolved value in pretend/help diagnostics, and test the
default plus override. Do not hide executor settings in `run_cr.sh` profile
arrays. Changes can affect retries, resource scheduling, transfer/staging,
manager discovery, log paths, and campaign runtime, but must not change the
histogram/sidecar contract for otherwise identical processing.

Validation owners include `tests/test_workqueue.py`,
`tests/test_run_analysis_cli_help.py`,
`tests/test_run_analysis_preflight.py`, and environment/profile tests for any
archive behavior. A local manager/worker smoke test checks connectivity and
packaging only; compare produced artifact validation with the equivalent
bounded `futures` path when executor invariance is in scope.

## Troubleshoot a stalled or failed run

1. Confirm that the manager printed a listening port and remains alive.
2. Confirm the worker uses the exact printed manager name and can reach the
   manager host/port through site networking.
3. Compare requested task resources with the worker's cores, memory, and disk.
4. Inspect manager and worker logs for environment transfer/import failures,
   missing staged inputs, and the worker-side exception summary.
5. Verify that the current environment contains the expected `topcoffea` data
   payloads; a successful manager startup is not proof that every worker can
   import or read them.
6. Retry only after identifying whether the failure belongs to connectivity,
   resources, environment staging, input access, or analysis code.

Additional fail-closed checks:

- no input files or zero requested chunks exits before useful work;
- an invalid/unsupported executor name is rejected;
- malformed port ranges are rejected before executor construction;
- environment archive integrity/current-compatibility failures stop manager
  setup rather than launching mismatched workers;
- worker-side exceptions are surfaced after executor execution and must not be
  treated as successful artifact publication; and
- a PKL without the expected sidecar/readback is incomplete even if Work Queue
  reports completed tasks.

The [entry-point reference](../reference/entrypoints.md) is authoritative for
current options and defaults. The [architecture explanation](../explanation/architecture.md)
places the executor inside the full production responsibility model.

## Campaign-owned versus externally provisioned resources

Worker provisioning remains external to the analysis CLI. The production
wrapper does not own or forward a Work Queue worker-count option. Before a
campaign, prepare one validated environment archive and reuse its exact frozen
identity for the campaign rather than rebuilding it per task.

The Work Queue executor uses `retries=15` with `resource_monitor=measure`.
Those internal task retries are distinct from schema-9 campaign recovery and
the campaign's stage-attempt history. Native `debug.log`, `tr.log`, `stats.log`,
and `tasks.log` belong to
`<output_root>/work_queue_logs/<production_component>/<block_id>/source_native/`
for a campaign source stage. The runner copies them to the corresponding
`source/` directory, verifies the copies, binds the archive to state, and only
then removes the native files. A copy failure preserves the native files.
Keep stdout and stderr directly on the terminal; native logs and campaign
state are the durable evidence surfaces. See the
[SR-reco campaign guide](sr_reco_pkl_campaign.md) for its launch and recovery
sequence.

## Processing admission and candidate monitor

The guarded Work Queue launchers share one queue-local admission adapter. Its
initial `ready_minimum=10` and `ready_factor=2` set the processing **ready**
reserve to `max(10, ceil(2 * estimated compatible connected capacity))`.
Running tasks and retained results have separate accounting. The adapter
admits at most 512 new input tasks or 0.25 seconds of admission work per pass,
subject to remaining input and effective staging headroom. A zero connected
capacity has a ten-task bootstrap. A failed headroom measurement pauses new
input until a fresh measurement succeeds. Accumulation keeps Coffea's group
size, size order, and final drain rules; the native queue still places tasks.
This is an initial qualified policy, not a factory worker target or a measured
production optimum. The parameters are independent of manager and campaign
names and do not change native `q.tune` options.

The invocation log directory contains a bounded `admission.jsonl` file with
schema version 3, as described under guarded processing admission above.
Each record identifies the invocation, elapsed time, configuration,
compatible-capacity estimate and quality, ready/running/native retrieval
states, held grouping inputs, estimated remaining input, admission and recovery
counts, retained/predicted bytes, effective quota headroom, measurement age,
and pause reason/duration. Frequent unchanged decisions are coalesced; a
2 MiB current file rotates to one `.previous` file. An I/O error disables
this diagnostic sink without certifying or changing scientific task success.
Compare these records with native transfer and task logs by timestamp. A full
ready reserve with idle compatible capacity points to native dispatch or
transfer delay; low headroom with growing retained bytes points to downstream
pressure. Repeated short pause/resume intervals and useful completed work
should be reviewed together before changing the two parameters. No automatic
retuning or production analytics job is installed.

An isolated CCTools monitor candidate contains a fatal signal path that exits
without trying to format another summary, plus a separate negative-receive
correction. A future manager invocation must explicitly select and verify its
ELF with `CCTOOLS_RESOURCE_MONITOR` before enabling monitoring. Existing
managers, workers, cached monitor copies, and environments do not change when
the candidate file is built. Keep task result and Coffea success checks after
monitor failure; a parseable old summary is not proof that the attempt was
successfully monitored.

## Factory capacity and timeout disposition for a later deployment

The pinned CCTools 7.15.10 factory source computes demand from current
`tasks_waiting`, `tasks_left`, and `tasks_on_workers`; the last is omitted for
foremen in one path. It applies the configured maximum and minimum to that
aggregate, then subtracts its own submitted jobs and any excess connected
workers. `tasks_dispatched` is cumulative and is not current demand. This
calculation cannot by itself give three factories distinct processing,
128 GB, and 192 GB capacity floors. The processing example of min 500,
max 800, and 50 workers per cycle describes availability while relevant work
remains; it does not authorize 500 processing tasks when only 20 inputs remain.
The three memory profiles are distinct. A compatible multi-profile worker is
one allocation, and an attributable manual worker counts toward the pool's
total rather than an extra factory request. Pending Condor jobs, running jobs,
connected workers, and held jobs are different states. The existing factory
knows only aggregate connected workers when deducting external capacity; a
failed or stale manager query does not prove zero demand. A living finished
manager is not a permanent reason for a floor, while a transient admission
pause is not campaign completion.

The source-backed next action is a separate, bounded **CCTools status-schema
correction**, owned by `work_queue/src/work_queue.c`: expose the already
recorded `factory_name` provenance alongside each worker's existing resource
and identity fields in `worker_status`, marking absent provenance unknown.
Its inputs are manager-owned worker records; its only write path is the
read-only status response. A focused manager test should connect one tagged
worker and one untagged worker, verify stable distinct identities and resource
units, then verify disconnect removes each record. This closes a specific
attribution gap before any profile-aware actuator design. It does not by
itself qualify a multi-factory controller or reserve a shared worker for one
profile; a later capacity decision must still select one writer, define
overlap ownership, and prove backend effects. No factory JSON writer or
worker action is part of the admission or monitor candidates.

At this version, factory `timeout` becomes worker `-t`, which sets **both**
connect and idle timeouts. `worker-extra-options` is appended after `-t`, so
explicit `--connect-timeout` and `--idle-timeout` values can override them in
the worker's left-to-right parser; their resolved argv must be checked for a
specific deployment. Connect timeout bounds finding a manager; idle timeout
is reset by worker task activity and does not recover a stuck active monitor.
`factory-timeout` concerns time without managers. Monitor/task failure is a
fourth, separate path. A 1,800,000-second manual value is about 20.8 days and
is not an approved value for both timers. The factory rereads its own one
config file each cycle, retaining old values after a bad reread; three files
have no group transaction. This version waits when it computes excess jobs,
but that observation does not establish safe live lowering across backends.
No new value changes already running worker argv. Do not remove busy workers
on the basis of WAITING, age, CPU zero, or status timeout.
