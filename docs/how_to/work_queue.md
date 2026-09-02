# Run with Work Queue

`analysis/topeft_run2/run_analysis.py` is the maintained Work Queue manager
entry point. It prepares the Coffea `WorkQueueExecutor`, constructs or accepts a
worker environment, stages repository inputs, and waits for independently
launched workers. There is no maintained `work_queue_run.py` entry point in the
current workflow.

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
| manager logs | hard-coded `debug.log`, `tr.log`, `stats.log`, and `tasks.log` in the manager working context |
| workers/site submission | operator or site factory; no wrapper launches them |
| processor and artifact meaning | unchanged `AnalysisProcessor`, policy, and sidecar contracts |

`run_cr.sh` and `fullR3_run.sh` may select/forward Work Queue and a frozen
environment, but `run_analysis.py` remains the executor configuration owner.
The worker environment transports code/dependencies; it does not replace the
repository commit and environment identity recorded for the campaign.

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

Use `--prepare-env-only` to create/validate the current archive and exit before
analysis setup. `--env-file` pins a prebuilt archive; `--snapshot` permits an
explicit historical archive after integrity validation and is not a current
compatibility claim. `--rebuild-env` forces current archive recreation.
Disabling remote packaging with `--no-remote-env` makes the operator responsible
for an equivalent worker-node environment.

Wrappers such as `fullR3_run.sh` and `run_cr.sh` ultimately forward executor
selection to `run_analysis.py`. They do not launch workers.

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

`--wq-filepath` is not a general remote scratch/configuration interface. The
current code prepares a staging-directory path but passes `/tmp` as the Coffea
executor `filepath`. Treat any change to that behavior as an executor
implementation change with tests, not a documented site override that is not
currently effective end to end.

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
Those internal task retries are distinct from campaign-level recovery or a
profile-specific resume decision. Native `debug.log`, `tr.log`, `stats.log`,
and `tasks.log` are copied to the campaign evidence location, verified, bound
to state, and only then removed from their native location. Keep stdout and
stderr directly on the terminal; native logs and campaign state are the durable
evidence surfaces. See [the production runbook](production.md) for profile and
failure-domain rules.
