# Run and extend histogram production

Run current production commands from `analysis/topeft_run2` in an activated
environment containing the matching `topeft` and `topcoffea` checkouts. The
three supported entry levels expose progressively less automation:

| Entry point | Choose it when | It owns | You still own |
| --- | --- | --- | --- |
| `run_cr.sh` | running or resuming a maintained multi-block campaign | public profile resolution, block command construction, output namespace and child execution | selecting the profile, output root, campaign tag, and reviewing the plan |
| `campaign_manifest.py` | state or recovery operations driven by the launcher | schema-9 plan, revision, attempt, receipt and reconciliation checks | inspecting its state and report evidence |
| `fullR3_run.sh` | running one Run 2 or Run 3 block | year/cfg bundle selection and construction of one `run_analysis.py` command | campaign layout, state, resume, and downstream lifecycle |
| `run_analysis.py` | a focused direct run or CLI development | sample loading, executor setup, processor construction, artifact/sidecar publication, and optional nonprompt dispatch | every input, option, output identity, and reproducibility record |

The wrappers delegate downward; they do not become alternate owners of sample
definitions, channel registries, histogram definitions, sumw2 policy, or
processor behavior.

## Know which layer owns a setting

| Concern | Owner | Derived or hard-coded behavior | Protected by |
| --- | --- | --- | --- |
| public profile, block matrix and child commands | `run_cr.sh` | Run 3 SR production defaults to five blocks; all component profiles accept an exact-coverage plan; output and campaign tag have no implicit production value | `test_campaign_production_profiles.py`, `test_campaign_profile_handoff.py` |
| plan, schema-9 state, receipts and recovery | `campaign_manifest.py` | approved scientific coverage is independent of the active partition; attempts retain operational provenance | `test_campaign_manifest.py` |
| cfg bundle, years, region, histogram list, tag-derived output name | `fullR3_run.sh` | year aliases and cfg paths are derived; absent year/tag still have script defaults, while its default output path is site-specific | `test_fullr3_run_wrapper.py`, profile tests |
| sample loading, CLI/YAML resolution, executor, processor and artifacts | `run_analysis.py` | parser defaults are `work_queue`, 8 workers, chunksize 100000, `histos/plotsTopEFT`; active channels/histograms are derived from registries and options | CLI, preflight, producer, policy, and artifact tests |
| event selection and histogram filling | `AnalysisProcessor` | constructed from the resolved run contract | processor/selection/nominal-schema tests |

This separation matters when extending the workflow. A setting belongs at the
highest layer that truly owns it: a campaign-only block belongs in a production
profile; a repeatable one-block selection belongs in `fullR3_run.sh`; a direct
runtime option or processor input belongs in `run_analysis.py`; a physics or
histogram definition belongs at its module/source authority.

## Run a maintained `run_cr.sh` campaign

The wrapper derives its working directory and repository root from the invoked
script. Before starting or resuming a campaign, verify that this checkout is
the one you intend to execute.

Inspect a fresh TOP-26-006 Run 3 plan before launching it:

```bash
./run_cr.sh --production-profile run3_prod \
  --output-dir /absolute/path/to/fresh_run3_campaign \
  --campaign-tag run3_campaign --dry-run
```

Run the same command without `--dry-run` after reviewing the block matrix,
inputs, outputs, and environment archive. Resume the same approved scientific
coverage and any validated active partition:

```bash
./run_cr.sh --production-profile run3_prod \
  --output-dir /absolute/path/to/fresh_run3_campaign \
  --campaign-tag run3_campaign --resume
```

`campaign_manifest.py` owns the schema-9 state, with generic `source` and
`data_driven` stages. Do not point a fresh campaign at an existing output
directory or edit state to force resume. A resume compares the resolved
campaign and stage-local scientific contracts and approved coverage, then
checks receipts and writer ownership. An operational commit, chunksize or
manager-resource change is recorded on a new attempt. A downstream-only
science change can retain a valid source receipt; a source science change
blocks source reuse.

For maintained profiles, omit `--env-file` to resolve or create the current
worker environment. An absolute `--env-file` selects a validated snapshot;
combined profiles forward it to both children. A resume may use a relocated
archive only when its SHA-256 is unchanged. That frozen byte digest is the
source worker implementation authority; a different digest requires a new
campaign. Checkout and manager fingerprints remain attempt provenance.

The campaign plan's `unit.outputs` owns each durable source and data-driven PKL
path. Existing plans therefore write to their canonical
`/groups` paths. The launcher refuses a command whose output contradicts that
path. If persistent-output capacity thresholds are configured, pass all four
positive `--pkl-min-free-bytes`, `--pkl-min-free-inodes`,
`--pkl-safety-reserve-bytes` and `--pkl-safety-reserve-inodes` values. Select
them during production preflight as capacity thresholds for the output location:
the minimum free bytes and inodes needed before an attempt, plus the reserve
that must remain when the result is published. These values do not set a hard
size limit for an individual PKL. The launcher passes them explicitly to the
source and data-driven executors. The artifact helper checks the actual output
parent before an attempt is recorded; at publication it checks the remaining
reserve. The pre-attempt inode threshold must account for the pending
directory, PKL and sidecar; hard links reuse their source inodes. Source and
data-driven publication requests no-clobber independently
of capacity settings. A completed same-attempt pending pair sits beside the
durable output and remains until the final pair and state receipt are durable.
Exclusive resume validates and publishes it without recomputation. A collision
or invalid pending pair stops recovery with its evidence preserved.

Work Queue staging is a separate configured capacity domain. Supply
`--wq-staging-base PATH` with a warning threshold for low staging space, the
minimum free-space threshold required to start, and a poll interval for the
run.
Use an existing directory owned and writable by the current user, outside
`/tmp` and `/var/tmp`, with no symlink components. The guard checks the path,
creates a private invocation directory, and measures available bytes and
inodes. No filesystem-type selection or extra staging authorization is needed.
An explicitly configured site quota can supply additional capacity information
for that site; its measurement behavior needs to be checked in that
environment.
The staging base, durable PKL root and campaign control root have separate
capacity and ownership checks.

For an invocation that will start exactly one component source manager, pass
`--wq-manager-name EXACT_NAME` to select its Work Queue name. A selected unit
with a reusable source, or an invocation with multiple pending sources, rejects
the override before changing campaign state or starting a source child. Combined
profiles do not accept this override; select a component invocation. The name
is recorded in the source command and attempt identity, not the campaign,
source, or data-driven scientific identity.

After the dry run, inspect the five `run3_prod` block rows, especially their
category groups, histogram families, source paths, and `_np` destinations. The
profile continues to later blocks after one block fails and records each result,
so the final summary and state—not the last shell exit alone—are the campaign
completion evidence.

### Modify or extend a profile

Keep a profile change inside the existing ownership model:

1. Locate the profile and block declarations in `run_cr.sh`.
   `production_block_ids`, the profile category arrays, and the associated
   variable-set arrays form one ordered plan.
2. Decide whether the change belongs to the campaign matrix or to the delegated
   `fullR3_run.sh`/`run_analysis.py` interface. Do not duplicate lower-level
   configuration in the profile.
3. Update every parallel profile array and the mechanical packing checks
   together. A block ID, category set, variable set, output tag, source PKL,
   transformed PKL, and state row must stay one-to-one.
4. Preserve unique block/output identities, approved scientific coverage,
   archive checks, and the two-stage source/data-driven model. A repartition
   may change the active plan revision only through a checked transition.
5. Forward a lower-level option instead of reimplementing its semantics. The
   profile should construct `fullR3_run.sh` arguments, not select sample JSONs
   or interpret processor settings itself.
6. Update the directly affected production-profile and campaign-manifest tests.
7. Dry-run a fresh namespace and a matching resume fixture. Check block count,
   ordering, command construction, output uniqueness, environment identity,
   interruption handling, and refusal of stale/missing artifacts.

`rebin_fine` is retired from the current CLI. A new campaign family that cannot fit the existing
profile/block/state invariants is a design decision, not a local documentation
or shell edit.

## Run one block with `fullR3_run.sh`

```bash
./fullR3_run.sh -y 2022 2022EE 2023 2023BPix -t run3_block --sr \
  --hist-vars njets lj0pt ptz ptll ptz_wtau lt \
  --do-np --defer-np -p /absolute/path/to/output --dry-run
```

Remove `--dry-run` after inspecting the printed command. Use
`--sample-json FILE` or `--cfg-override FILE` for one explicit input authority.
Unrecognized options are forwarded to `run_analysis.py`. The wrapper does not
create campaign state or coordinate several blocks.

The wrapper requires exactly one of `--cr` or `--sr`. It expands `run2` to
`UL16 UL16APV UL17 UL18` and `run3` to
`2022 2022EE 2023 2023BPix`, removes duplicate year tokens, and then derives
the cfg bundle for each era and region. Run 2 uses one aggregate cfg bundle;
Run 3 uses per-year NDSkim cfgs. `--sample-json` and `--cfg-override` replace
that derived bundle and are mutually exclusive.

If omitted, the script currently supplies year `2022`, the tag defined by
`DEFAULT_TAG` in `fullR3_run.sh` (`rebin_lj0pt350GeV` at the time of writing),
chunksize 100000, and the region histogram shorthand (`cr` or `ana`). Its
built-in output root is a Glados-specific
`/groups/klannon/$USER/` path. For a reproducible invocation, set years, tag,
histograms, and `--outpath` explicitly rather than inheriting those convenience
values. The output name is derived from normalized years, region, and tag.

`--do-np` is forwarded like any direct option. `--defer-np` adds
`--np-postprocess=defer`; it does not enable nonprompt by itself. The wrapper
records its identity through `--sample-universe-wrapper` and defaults the
ttgamma sample-role policy to `split`.
For maintained SR and CR campaigns, `--do-np --defer-np` certifies the complete
nonprompt and charge-flip input contract before the separate data-driven stage.
The recorded wrapper name is provenance; recovery checks the resolved
scientific inputs and the final artifact sidecar.

When modifying `fullR3_run.sh`, keep Run 2 aggregate and Run 3 per-year cfg
selection centralized, keep CR/SR selection explicit, and classify every new
option as wrapper-owned or forwarded. It must not be both. Preserve the
relationships checked by `tests/test_fullr3_run_wrapper.py`.

For a wrapper-owned option, add help text, parsing, missing-value validation,
conflict checks, derivation, dry-run visibility, and a focused test. For a
forwarded option, preserve its token/value grouping in `EXTRA_ARGS` and let
`run_analysis.py` validate semantics. Test a similar unknown option to prove
that forwarding still works. Any change to cfg selection, histogram defaults,
year normalization, region flags, output identity, sample-universe provenance,
or deferred nonprompt affects the PKL/sidecar identity and downstream campaign
reproducibility.

## Run `run_analysis.py` directly

```bash
python run_analysis.py \
  ../../input_samples/cfgs/NDSkim_2022_background_samples.cfg \
  --executor futures --years 2022 --nworkers 8 \
  --hist-list njets lj0pt ptz ptll lt \
  --category-groups 2los_CRZ \
  --outpath /absolute/path/to/output --outname run3_direct
```

The positional input is a sample JSON or comma-separated cfg expression
understood by the sample loader. A sample JSON describes dataset files and
metadata; a sample cfg bundles several JSONs. Their resolved contents—not the
filename—define the active sample universe.

`--options FILE` loads one top-level YAML mapping of supported option names. In
the current implementation, recognized YAML values are applied after argparse
values and therefore replace the corresponding parser-derived value. Do not set
the same option in both places expecting the CLI to win. There is no supported
`FILE:KEY` selector in this CLI, and unrecognized top-level keys fail before
processing. Use `--pretend` to resolve samples, policies, categories, and output
planning without processing events.

Important parser defaults are executor `work_queue`, 8 workers, chunksize
100000, output directory `histos`, output name `plotsTopEFT`, tree `Events`,
analysis mode `standard`, and nonprompt mode `inline` when `--do-np` is active.
Defaults are owned in the parser and option-overlay code; wrappers may choose
different explicit values for a particular route.

For a direct run, record at least the exact input expression, option file,
years, category groups, histogram list, executor settings, output directory,
output name, current commit, and environment identity. The direct route does
not choose a campaign matrix, protect a fresh namespace, or resume stages.

The direct entry point delegates event processing to `AnalysisProcessor` and
execution to the selected Coffea executor. It owns preflight, sample/option
resolution, sidecar provenance, source-PKL publication, and the inline/deferred
nonprompt dispatch decision. It does not own the category registry, axis
definitions, correction payloads, or downstream plotting/cards. A successful
run produces a source PKL plus its adjacent metadata sidecar; with inline
nonprompt it also produces the distinct transformed artifact and sidecar.

### Extend direct options or configuration

- Add a CLI option only when it controls run orchestration or supplies an input
  to an existing processor contract.
- Keep channel membership in `topeft/channels/ch_lst.json`, histogram and
  binning definitions in `topeft/modules/axes.py`, and sumw2 policy in
  `topeft/modules/sumw2_policy.py`.
- Thread a new value through parsing, option-file precedence, validation, and
  processor construction explicitly. Fail before executor startup on invalid
  combinations.
- Decide whether the value changes artifact identity, only executor mechanics,
  or physics/selection behavior. If it changes identity or compatibility,
  serialize it at the existing provenance owner and validate readback; do not
  create a parallel metadata registry.
- Keep derived data derived: sample identity comes from loaded JSON metadata,
  active category groups from `ch_lst.json`, runtime families from the resolved
  histogram list, and required companions from consumer contracts.
- Update focused CLI/preflight tests such as
  `tests/test_run_analysis_cli_help.py`,
  `tests/test_run_analysis_preflight.py`, and the test for the affected
  contract.
- Review sidecar/provenance fields whenever the new option changes artifact
  identity or downstream compatibility.
- Validate `--pretend`, a bounded direct route, wrapper forwarding where
  applicable, malformed YAML/type behavior, and the downstream consumer whose
  contract is affected.

For exact options and defaults, use the
[entry-point reference](../reference/entrypoints.md). For why the three layers
are separated, see the [architecture explanation](../explanation/architecture.md).

## Optional phase telemetry

Phase telemetry is disabled by default. Add `--phase-telemetry` explicitly
to `run_cr.sh`, `run_analysis.py`, or `run_data_driven.py` to request
diagnostics. The campaign launcher forwards it to both source and separate
data-driven stages, including through combined profiles. The direct analysis
YAML key `phase_telemetry` accepts a boolean and follows the normal
YAML-over-CLI precedence.

Processor stdout records start with `[phase_telemetry]`. They contain
process-local wall/CPU clocks and RSS observations, reached phases and
object-systematic iterations. Fill counters are grouped by histogram family,
weight label and lepton category; selection counts are category memberships,
not unique event totals. Submitted fill entries can count jets rather than
events. Operation timing includes synchronous shared correction calls:
inclusive totals overlap children, exclusive totals subtract immediate timed
children, and deferred lazy work is not forced or attributed.

For Work Queue, the manager persists accepted scalar `processor_*` records
from successful processing tasks, plus manager-local observations, in
`phase_telemetry.jsonl` beside its native logs. Task identity and local clock
domains must be respected when comparing rows. During campaign log archival,
the telemetry inode is renamed within `source_native/` to an attempt-specific
name and the `source/` or `source_resume_N/` archive receives a relative
symlink. Keep the archive and native target together. This optional stream
does not use the regular native-log copy path.

The direct data-driven helper also writes
`<output-pkl>.phase_telemetry.jsonl` for the latest enabled invocation.
That file retains fixed phase/clock/RSS fields; stdout carries additional
context. These diagnostics are separate from the required
`.metadata.json` artifact sidecar. Optional persistence or telemetry-archive
errors warn and do not replace scientific stage status; required artifact,
receipt and regular native-log checks retain their failure contracts.
Telemetry presence, timing or absence does not establish scientific success,
physical equivalence or production qualification.

## Route physics-policy changes to their owner

Do not extend a wrapper merely to avoid the owning analysis contract:

| Intended change | Owning route |
| --- | --- |
| Object threshold, working point, trigger, filter, or overlap policy | [Objects, selections, and triggers](objects_selections_and_triggers.md) |
| Correction tag/payload, event weight, variation, or forward-JER policy | [Corrections, weights, and systematics](corrections_weights_and_systematics.md) |
| Category group, observable, or wrapper histogram matrix | [Categories and observables](categories_and_observables.md) |
| Sample metadata, role, active universe, or normalization input | [Sample roles and normalization](sample_roles_and_normalization.md) |
| EFT sample treatment, coefficient input, or consumer | [EFT](eft.md) |

For a new `run_cr.sh` profile, update the frozen year/region/category/histogram/
nonprompt matrix, its fail-closed validation, and profile tests together. For a
new `fullR3_run.sh` policy, preserve exactly one CR/SR choice and the sample
JSON/cfg exclusivity. For a new direct option, classify it as execution,
selection, correction, sample semantics, or expert/diagnostic in the
[entrypoint reference](../reference/entrypoints.md).

The diboson and sum-of-weights CLIs are maintained specialist paths. Extend
their own parser, processor/config authority, and focused tests without adding
their defaults to the core wrapper chain. See
[specialist interfaces](../reference/specialist_interfaces.md).

## Maintained profile and recovery boundary

Use a maintained profile: `run2_prod`, `run3_prod`, `run2_prod_CR`,
`run3_prod_CR`, `run2_run3_prod`, `run2_run3_prod_CR`, the corresponding
`stat_diagnostic` SR/CR component and combined profiles, or `run2_sr_ext` and
`run3_sr_ext`. Supply a fresh absolute
output directory and a campaign tag for a new campaign. The SR-reco extension
profiles have a separate [operator guide](sr_reco_pkl_campaign.md) for explicit
plans, worker archives, and post-run qualification. That guide also describes
shared `run_cr.sh` combined-profile resume and Work Queue port selection.

The combined Run 2/Run 3 production SR route is `run2_run3_prod`; it
orchestrates separate component states. With no `--env-file`, the wrapper resolves or
creates the current worker environment. With `--env-file X`, it validates and
uses exactly `X` as an intentional snapshot without requiring current-checkout
compatibility, and both children of a combined profile receive that same
archive. Capture the environment mode, resolved path, SHA-256, environment
fingerprint, and source identity from campaign state. Qualify an explicit
archive for the intended processor and worker runtime; snapshot validation
alone does not establish that compatibility.

The wrapper validates explicit archive integrity and provenance and may reuse
the same archive bytes at a new path. When no archive is
supplied for a fresh maintained campaign, it delegates current archive creation
or reuse to `run_analysis.py --prepare-env-only`. To make a new archive from a
chosen build workspace, use the native builder:

```bash
python /absolute/path/to/topeft/analysis/topeft_run2/run_analysis.py \
  --prepare-env-only --rebuild-env
```

The reported archive path depends on that workspace; keep its adjacent
`.manifest.json` with the archive. The maintained profile uses chunksize
`100000` and leaves stdout/stderr on the controlling terminal; do not add a
`tee` collector as a substitute for native evidence paths.

Each new campaign uses a fresh namespace. A known component-local failure may
leave its evidence intact while an independent component proceeds; a shared
unsafe blocker fails closed. An ambiguous interruption is neither success nor a
safe retry signal: inspect saved state, the output namespace, and native logs
before deciding on a follow-up. This describes procedure only, not production
completion or artifact acceptance.
