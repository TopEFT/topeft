# Run and extend histogram production

Run current production commands from `analysis/topeft_run2` in an activated
environment containing the matching `topeft` and `topcoffea` checkouts. The
three supported entry levels expose progressively less automation:

| Entry point | Choose it when | It owns | You still own |
| --- | --- | --- | --- |
| `run_cr.sh` | running or resuming a maintained multi-block campaign | production-profile block selection, output namespace, state, resume, environment archive, and artifact checks | selecting the profile, output root, campaign tag, and reviewing the frozen plan |
| `fullR3_run.sh` | running one Run 2 or Run 3 block | year/cfg bundle selection and construction of one `run_analysis.py` command | campaign layout, state, resume, and downstream lifecycle |
| `run_analysis.py` | a focused direct run or CLI development | sample loading, executor setup, processor construction, artifact/sidecar publication, and optional nonprompt dispatch | every input, option, output identity, and reproducibility record |

The wrappers delegate downward; they do not become alternate owners of sample
definitions, channel registries, histogram definitions, sumw2 policy, or
processor behavior.

## Know which layer owns a setting

| Concern | Owner | Derived or hard-coded behavior | Protected by |
| --- | --- | --- | --- |
| campaign block matrix, state, resume, and frozen environment | `run_cr.sh` production profile | `run3_full` has five fixed blocks; `rebin_fine` has six; output and campaign tag have no implicit production value | `test_run3_full_production_profile.py`, `test_rebin_fine_resume_and_environment.py` |
| cfg bundle, years, region, histogram list, tag-derived output name | `fullR3_run.sh` | year aliases and cfg paths are derived; absent year/tag still have script defaults, while its default output path is site-specific | `test_fullr3_run_wrapper.py`, profile tests |
| sample loading, CLI/YAML resolution, executor, processor and artifacts | `run_analysis.py` | parser defaults are `work_queue`, 8 workers, chunksize 100000, `histos/plotsTopEFT`; active channels/histograms are derived from registries and options | CLI, preflight, producer, policy, and artifact tests |
| event selection and histogram filling | `AnalysisProcessor` | constructed from the resolved run contract | processor/selection/nominal-schema tests |

This separation matters when extending the workflow. A setting belongs at the
highest layer that truly owns it: a campaign-only block belongs in a production
profile; a repeatable one-block selection belongs in `fullR3_run.sh`; a direct
runtime option or processor input belongs in `run_analysis.py`; a physics or
histogram definition belongs at its module/source authority.

## Run a maintained `run_cr.sh` campaign

The current wrapper is non-portable: it hard-codes the managed correction-lib
checkout for both its `topeft/analysis/topeft_run2` working directory and its
topeft commit readback, rather than deriving either path from the invoked
script. Invoking a copied script or a different checkout still uses that fixed
managed checkout. Before starting or resuming a campaign, verify that the
managed checkout is the one you intend to execute and record; making the
wrapper relocatable requires an executable workflow change.

Inspect a fresh TOP-26-006 Run 3 plan before launching it:

```bash
./run_cr.sh --production-profile run3_full \
  --output-dir /absolute/path/to/fresh_run3_campaign \
  --campaign-tag run3_campaign --dry-run
```

Run the same command without `--dry-run` after reviewing the block matrix,
inputs, outputs, and environment archive. Resume only that frozen plan:

```bash
./run_cr.sh --production-profile run3_full \
  --output-dir /absolute/path/to/fresh_run3_campaign \
  --campaign-tag run3_campaign --resume
```

`run_cr.sh` owns the campaign state file and records source and nonprompt stages
separately. Do not point a fresh campaign at an existing output directory, and
do not edit the state file to force a resume.

For `run3_full`, omission of `--env-file` asks the wrapper to resolve and
validate one current remote environment archive before campaign state is
created. `rebin_fine` requires an explicit absolute `--env-file`. A resume uses
the exact environment path, hash, fingerprint, repository commits, block plan,
and output identities frozen in the existing state; a command-line environment
that disagrees with state is rejected.

After the dry run, inspect the five `run3_full` block rows, especially their
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
4. Preserve unique block/output identities, the frozen-plan hash, environment
   archive checks, and the two-stage source/nonprompt state model. Never make a
   changed live plan silently compatible with old state.
5. Forward a lower-level option instead of reimplementing its semantics. The
   profile should construct `fullR3_run.sh` arguments, not select sample JSONs
   or interpret processor settings itself.
6. Update `tests/test_run3_full_production_profile.py`; for `rebin_fine`, also
   use `tests/test_rebin_fine_resume_and_environment.py`.
7. Dry-run a fresh namespace and a matching resume fixture. Check block count,
   ordering, command construction, output uniqueness, environment identity,
   interruption handling, and refusal of stale/missing artifacts.

`rebin_fine` is a specialist Run 2/Run 3 profile and requires an explicit
current `--env-file`. A new campaign family that cannot fit the existing
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

If omitted, the script currently supplies year `2022`, tag
`fec79a60_PNet`, chunksize 100000, and the region histogram shorthand (`cr` or
`ana`). Its built-in output root is a Glados-specific
`/groups/klannon/$USER/` path. For a reproducible invocation, set years, tag,
histograms, and `--outpath` explicitly rather than inheriting those convenience
values. The output name is derived from normalized years, region, and tag.

`--do-np` is forwarded like any direct option. `--defer-np` adds
`--np-postprocess=defer`; it does not enable nonprompt by itself. The wrapper
records its identity through `--sample-universe-wrapper` and defaults the
ttgamma sample-role policy to `split`.

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
