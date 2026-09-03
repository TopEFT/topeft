# Entrypoints and wrappers

This page identifies supported entrypoints and the responsibility that each
layer retains. It is a lookup page, not a production recipe. See the
[production how-to](../how_to/production.md) for commands in context.

## Supported entrypoints

| Component | Kind and status | Inputs | Defaults and outputs | Failure boundary |
| --- | --- | --- | --- | --- |
| `analysis/topeft_run2/run_cr.sh` | Maintained production-profile wrapper | `--production-profile`, a fresh absolute `--output-dir`, and `--campaign-tag`; optional `--env-file`, `--resume`, and `--dry-run` | Six public Run 2/Run 3 SR+CR profiles; campaign state plus source/transformed PKLs and sidecars | Rejects missing or mismatched state, a reused fresh namespace, invalid archives, ambiguous interruption, and incomplete outputs |
| `analysis/topeft_run2/fullR3_run.sh` | Maintained command/configuration wrapper | Years, exactly one of `--cr` or `--sr`, optional histogram/input/output overrides, and forwarded analysis options | Year/config-dependent `run_analysis.py` command | Rejects conflicting input overrides, missing cfg/JSON inputs, and unsupported region/year combinations |
| `analysis/topeft_run2/run_analysis.py` | Direct developer CLI | Sample JSON/cfg expression and CLI or YAML options | Executor `work_queue`; 8 workers; chunksize 100000; `histos/plotsTopEFT.pkl.gz` | Validates executor, active sample universe, categories, environment, policies, and artifact contracts at their owning boundaries |
| `analysis/topeft_run2/run_data_driven.py` | Direct transformed-artifact CLI | A validated source PKL/sidecar and requested data-driven product | Streaming input; output name derived from the input when omitted | Rejects missing/incompatible sidecars, uncertified input policy, invalid product requests, and incomplete transformed companions |
| `analysis/topeft_run2/run_plotter.sh` | Maintained plotting wrapper | Readable PKL, output directory, and years | Forwards to the direct plotter; can infer CR/SR from the filename; dry-run prints the resolved command without creating the output directory | Rejects missing inputs/years; the direct plotter owns artifact and metadata validation |
| `analysis/topeft_run2/make_cr_and_sr_plots.py` | Direct plotting CLI | Repeatable `-f` inputs or a list file, output, years, and plot controls | Processing binning; merged channels; one worker; year coverage `warn` | Rejects incoherent artifacts, ambiguous channel authority, invalid binning, and mixed Run 2/Run 3 inputs |
| `analysis/topeft_run2/make_cards.py` | Direct card CLI | Positional PKLs or a list file, variables/channels, and card controls | Fitting binning; year coverage `warn`; Asimov data | Rejects incoherent artifacts, invalid exact aggregation, incomplete shape pairs, and selected-WC/coverage failures according to options |
| `analysis/topeft_run2/datacards_post_processing.py` | Direct topology/scaling finalizer | Datacard directory plus exactly one topology selector | Selected copy directory and final `scalings.json` | Rejects selector count, missing inputs, output collisions, text/ROOT asymmetry, and source-grounded exact counts; `-f` has no hard-coded exact count |

All rows above are `public_supported`. The executable file and its usage/parser
block are signature authority. Normal success is exit status 0; parser,
preflight, child-process, validation, or publication failures return nonzero or
raise before completion.

## Production CLI contracts

### `analysis/topeft_run2/run_cr.sh`

Purpose: execute a named maintained block plan and retain enough local campaign
state to resume without confusing partial, stale, or mismatched output.

| Input | Type/requirement/default | Semantics |
| --- | --- | --- |
| `--production-profile` | Required public profile: `run2_full`, `run3_full`, `run2_full_CR`, `run3_full_CR`, `run2_run3_full`, or `run2_run3_full_CR` | Selects the hard-coded maintained block plan. It is not a free-form config path. `rebin_fine` remains a specialist legacy profile with its own explicit environment requirement. |
| `--output-dir` | Required fresh absolute path for a new campaign | Owns state, logs, and produced artifacts. Existing state is accepted only through compatible resume behavior. |
| `--campaign-tag` | Required non-empty string | Portable campaign identity recorded with state. |
| `--env-file` | Optional path | Forwarded environment archive; validated according to the child environment contract. |
| `--resume` | Boolean, default false | Reopens compatible campaign state and mechanically selects incomplete blocks/stages. |
| `--dry-run` | Boolean, default false | Skips campaign-block execution, but is not side-effect-free: the wrapper first resolves and validates the environment, and `run3_full` without `--env-file` may prepare an environment archive. It then prints the resolved plan without running the campaign blocks. |

The wrapper derives full `fullR3_run.sh` invocations, records profile/block
state, and, for the maintained deferred path, invokes `run_data_driven.py` only
after the heavy processor child exits. It does not own selections, processor
physics, bin edges, or artifact schemas. Extend a profile by changing its one
block-plan owner and the production-profile tests; do not copy its defaults into
another wrapper.

### `analysis/topeft_run2/fullR3_run.sh`

Purpose: translate a year bundle plus CR/SR intent into one direct analysis
command. Required inputs are the requested year/bundle and exactly one of
`--cr`/`--sr`. Optional `--sample-json` and `--cfg-override` are mutually
exclusive input authorities. Histogram-family, output, executor, environment,
nonprompt, and other unknown analysis options are forwarded to
`run_analysis.py` after wrapper-owned validation.

The wrapper derives current NDSkim cfgs, a CR `cr` or SR `ana` hist list when no
override is supplied, and output/region naming. It writes no histogram itself.
Its side effect is executing the constructed direct command. Missing inputs,
invalid region/year combinations, conflicting cfg/JSON overrides, or child
failure stop the wrapper. Current cfg resolution is detailed in
[production configuration](production_configuration.md).

### `analysis/topeft_run2/run_analysis.py`

Purpose: validate one active sample/configuration universe, construct
`AnalysisProcessor`, execute the selected coffea executor, and publish a
processor artifact pair; optionally publish an inline nonprompt transformed
pair.

| Stable input group | Type/default/accepted values | Semantics |
| --- | --- | --- |
| Positional `jsonFiles` | Optional path/expression, default empty string | Sample JSON or cfg authority. Higher wrappers normally supply it. |
| Executor/resources | `--executor work_queue` by default; workers 8; chunksize 100000; optional chunks/prefix/tree | Executor names are validated later by the execution branch; parser acceptance alone is not support. |
| Output | `--outname plotsTopEFT`, `--outpath histos` | Publishes `<outpath>/<outname>.pkl.gz` and adjacent metadata. Existing/output safety is owned by the writer path. |
| Scope | Optional years, histogram list, category groups, WC list | Filters the resolved input/processor family universe. Names are validated before or during processor construction. |
| Analysis flags | Off-Z, tau, forward, or all-analysis; default none | Mutually exclusive mode selectors. `--analysis-mode` accepts `standard` or `taufitter`, default `standard`. |
| Sumw2/systematics | Modern YAML `sumw2_storage`; deprecated `--no-sumw2`; `--do-systs` | Modern policy is resolved before processing. Legacy statistical inputs have explicit compatibility/conflict handling. |
| Nonprompt | `--do-np`; `--np-postprocess` in `inline`, `defer`, `skip`, default `inline` | Inline publishes a transformed artifact; defer prints a direct follow-up command; skip omits transformation. |
| Environment | Optional archive/rebuild/prepare/snapshot/no-remote controls | Owns worker-environment preparation/validation, not package installation. |
| YAML overlay | `--options FILE` | One mapping loaded after CLI-derived values; recognized YAML values replace corresponding CLI values. See the exact caveat in production configuration. |

`--pretend` reads/resolves inputs without executing analysis; `--test` bounds
the event/chunk run but is still execution. The unsupported renormalization-
envelope flag exits before analysis work. The direct command fails closed on
sample/profile, category, sumw2, data-driven, environment, processor, and
artifact-publication errors. It returns no library value; its durable return is
the published artifact pair and exit status.

## Transformation CLI contract

### `analysis/topeft_run2/run_data_driven.py`

Purpose: consume one already-complete validated processor artifact and publish
one separately transformed artifact with lineage.

`--input-pkl` is required. `--output-pkl` is optional and otherwise derives an
`_np` or `_np_nominal_reference` name. `--only-flips` drops nonprompt processes
from the transformed result; it does not invoke an independent flips producer.
Streaming input is the default; `--legacy-dict-mode` is the explicit
materialized-dictionary compatibility choice. Data-driven/memory reports,
heartbeat/quiet controls, and `--nominal-only-reference` affect diagnostics or
the explicit output kind but do not weaken sidecar requirements. The deprecated
envelope option fails before reading the input.

`main(argv=None)` returns a process status and writes only through
`write_histogram_artifact`, binding the new sidecar to the validated input
lineage and transformation contract. Missing source certification, unsupported
product/applicability, output collision, transformation failure, or incomplete
nominal/sumw2 content prevents publication. See
[histogram artifacts](histogram_artifacts.md) for exact schemas.

## Layering contract

`run_cr.sh` expands a maintained production profile into calls to
`fullR3_run.sh`. `fullR3_run.sh` resolves the years, region, maintained sample
cfg files, and histogram family, then constructs `run_analysis.py`. Calling a
lower layer transfers its omitted responsibilities to the user; it does not
silently reconstruct the higher-level campaign record.

`run_plotter.sh` is a convenience layer over `make_cr_and_sr_plots.py`.
Dry-run validates and prints the resolved Python command without creating the
requested output directory; normal execution creates it immediately before
launching the plotter.
`make_cards.py` is already the direct supported card interface and delegates
card/template construction to `topeft.modules.datacard_tools.DatacardMaker`.
`datacards_post_processing.py` finalizes selected scaling records after the
individual cards and templates exist; it is not a card producer.

See [production configuration](production_configuration.md),
[plotting](plotting.md), and
[datacards and scalings](datacards_and_scalings.md) for exact owned contracts.

## Operator records

Scripts with campaign, date, site, user, or immutable evidence identifiers in
their names are not supported merely because they are executable. In
particular, `run_make_cards_run3_yawen_matrix.sh` is a DATACARD023-qualified
operator record with fixed local paths, branch and input hashes, and a recorded
off-Z input that predates the required `ptll` schema. It is useful archival
evidence, but it is not a reusable current card entrypoint. The durable
region-to-distribution contract belongs to [flexible binning](flexible_binning.md)
and its source/test authorities.

## Signature and validation authority

Shell usage blocks and Python `build_arg_parser()` definitions are the exact
option authority. Focused ownership tests include
`tests/test_run3_full_production_profile.py`,
`tests/test_run_analysis_preflight.py`, `tests/test_run_data_driven.py`,
`tests/test_make_cr_and_sr_plots.py`, and
`tests/test_make_cards_multi_pkl.py`.

## Physics-policy control map

The maintained processor-facing controls fall into five semantic classes. The
classification describes downstream meaning; it does not change parser
authority.

### Physics selection

Analysis mode, category groups, histogram groups, tau/forward/all-analysis
selection, CR/SR inclusion, flavor splitting, off-Z splitting, and the Run-3
MVA path change which objects, masks, categories, or observables are processed.
They are not interchangeable with executor or file-selection controls.

### Physics corrections

Systematic activation controls whether registered object and weight variations
are produced. The forward stochastic-JER suppression and forward eta-band pT
policy must be read together: the former controls a shared jet-factory hook and
the latter resolves concrete forward-object application. Their reusable
mechanism belongs to `topcoffea`; their activation and default analysis policy
belong to `topeft`.

### Sample semantics

Input year/sample configuration, EFT/WC inputs, the ttgamma sample-role policy,
and production-profile sample matrices change how the input dataset is
interpreted. The maintained production path uses the split ttgamma role
contract. `lo_xsec_samples` remains a role set, not numeric cross-section data.

### Expert or diagnostic controls

`--ecut` is an upper event-energy diagnostic cut. It is not an electron
threshold and is not part of ordinary object-selection guidance. Statistical
storage controls such as the sumw2 opt-out affect downstream uncertainty
capability without defining a physics nuisance.

### Execution only

Executor, worker, chunk, prefix, output-path, output-name, dry-run, and resume
controls govern scheduling, coverage, storage, or recovery. They should not be
used to explain a physics choice even when an incomplete test run naturally
changes the amount of data processed.

## Active, conflicting, and inactive controls

YAML option values replace corresponding CLI-derived values in the implemented
configuration path. Parser help and this reference describe the same ordering.

The legacy `--do-renormfact-envelope` entry is deprecated and has no active
processor effect. It is not a supported alternative to the maintained scale-
weight variations. Active physics controls, diagnostic controls, and inactive
compatibility entries must remain separated in examples and operating guides.

The source-derived processor stages are in
[analysis processor physics map](analysis_processor.md). Diboson and
sum-of-weights entrypoints are maintained specialist interfaces described in
[specialist interfaces](specialist_interfaces.md).

## Physics-control defaults and practical bridge

| Control | Class | Default or required behavior | Physics consequence | Change route |
| --- | --- | --- | --- | --- |
| `--analysis-mode` | physics selection | `standard`; `taufitter` is specialist | Selects top-level processor mode and compatible category/variation contracts | [categories/observables](../how_to/categories_and_observables.md) |
| `--do-systs` | physics correction | false | Adds applicable object and weight variation templates | [corrections/systematics](../how_to/corrections_weights_and_systematics.md) |
| `--noRun3MVA` | physics selection | Run-3 MVA enabled | Disables the maintained Run-3 MVA object-selection branch | [objects/selections](../how_to/objects_selections_and_triggers.md) |
| `--ecut` | expert/diagnostic | unset | Applies an upper event-energy diagnostic cut; not an electron threshold | Extend at parser and processor consumer with focused diagnostics |
| `--ttgamma-sample-role-policy` | sample semantics | `split` | Changes ttgamma source-role masks; alternate is constrained Run-2 diagnostic use | [sample roles](../how_to/sample_roles_and_normalization.md) |
| `--no-sumw2` | expert/statistical | sumw2 enabled | Removes statistical companions and may make variance-dependent consumers inadmissible | [sumw2](../how_to/sumw2.md) |
| `--no-suppress-forward-eta-stochastic-jer` | physics correction | suppression enabled | Disables the default forward stochastic-JER hook policy | [corrections/systematics](../how_to/corrections_weights_and_systematics.md) |
| `--fwd-eta-band-pt-apply` | physics correction | `auto` | Resolves eta-band pT policy by era; interacts with forward JER and categories | [corrections/systematics](../how_to/corrections_weights_and_systematics.md) |
| `--category-groups` | physics selection | all groups in each active block | Filters maintained category groups | [categories/observables](../how_to/categories_and_observables.md) |
| `--hist-list` | physics selection | analysis-selected list | Filters observable families built and filled | [categories/observables](../how_to/categories_and_observables.md) |
| tau/forward/all-analysis, SR/CR, flavor/off-Z flags | physics selection | inactive unless selected | Enables or partitions the corresponding category surface | [objects](../how_to/objects_selections_and_triggers.md) and [categories](../how_to/categories_and_observables.md) |
| `fullR3_run.sh --cr|--sr` | physics selection | exactly one required | Selects wrapper region behavior and forwarded skip policy | [production](../how_to/production.md) |
| `fullR3_run.sh --year`, sample JSON, or cfg override | sample semantics | wrapper year/config authority | Selects sample universe, era, payloads, triggers, and normalization | [production](../how_to/production.md) and [sample roles](../how_to/sample_roles_and_normalization.md) |
| `fullR3_run.sh --hist-vars` and forwarded analysis controls | physics selection | region-owned list unless overridden | Selects observables/categories/modes for all enabled variations | [production](../how_to/production.md) |
| `run_cr.sh --production-profile` | sample semantics | required; no implicit profile | Selects a frozen years/regions/categories/histograms/nonprompt matrix | [production](../how_to/production.md) |
| `run_cr.sh` category/histogram and nonprompt/systematic policy | selection/correction | profile-owned | Defines each campaign block's physics output and transformation schedule | [production](../how_to/production.md) |

Representative direct invocation:

```bash
python analysis/topeft_run2/run_analysis.py <sample.json> \
  --analysis-mode standard \
  --skip-sr --category-groups 2los_CRZ \
  --hist-list invmass --do-systs
```

This is a semantic example, not a production authorization. Executor, input,
and output details belong to [the production guide](../how_to/production.md).
If `--options` is supplied, recognized YAML values replace corresponding CLI-
derived values in the implemented path.

## `run_cr.sh` production interface

The public production profiles are `run2_full`, `run3_full`, `run2_full_CR`,
`run3_full_CR`, `run2_run3_full`, and `run2_run3_full_CR`; no arguments retain
the legacy `run2_full` alias. `rebin_fine` is a specialist legacy profile and
requires an explicit environment archive. Public profiles require fresh output
namespaces, an absolute output directory, and a campaign tag.

For the maintained profiles, `run_cr.sh` validates the exact pinned archive and
records the resolved archive identity rather than constructing an archive on
behalf of the campaign. Resume is profile-specific: it is not a generic retry
operation, and an interrupted state must be inspected before a resume is
requested. Worker provisioning remains an external Work Queue concern; this
wrapper does not forward a worker-count option.
