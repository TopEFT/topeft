# Specialist interfaces

These interfaces support focused inspection or maintenance. They are not
alternative production routes.

## Histogram inspection

`analysis/topeft_run2/inspect_histeft_pkl.py` is the read-only artifact
inspection CLI. It reports axes, WC metadata, populated categories, nominal
evaluations, and value/variance availability without replacing artifact
validation. The detailed in-memory API is documented in
[HistEFT and SparseHist](histeft.md); artifact/sidecar validation belongs to
[histogram artifacts](histogram_artifacts.md).

This is a public supported inspection CLI. The required positional `pkl_path`
accepts `.pkl` or `.pkl.gz`. `--hist` selects one exact top-level key;
otherwise at most `--max-hists` histogram-like objects are summarized (default
5). `--max-labels` limits displayed labels/edges (default 20). Both limits must
be positive integers. `--yield-summary` additionally attempts an available
nominal total and variance summary. Success returns 0 and writes only stdout;
the tool does not modify the input. Missing/unreadable input, unknown requested
key, invalid limits, or incompatible pickle content fails. Loading a pickle is
code-deserialization and should be limited to trusted repository artifacts.

## Nominal schema utilities

`topeft.modules.nominal_schema` owns the split scalar/EFT/sumw2 sibling layout,
canonical keys, compatibility validation, merge behavior, WC evaluation, and
materialized consumer views. Its public developer functions include
`get_nominal_components()`, `validate_nominal_family()`,
`validate_nominal_mapping()`, `canonicalize_nominal_keys()`,
`merge_nominal_mappings()`, `evaluate_nominal_at_wc()`, and the explicit
materialization helpers.

The complete symbol, parameter, return, and failure table is in
[histogram artifacts](histogram_artifacts.md#topeftmodulesnominal_schema).

## Data-driven product utilities

`topeft.modules.data_driven_products` owns generated-process naming, requested
and resolved product contracts, preflight certification, sumw2 requirement
validation, and bounded compatibility readback. These functions support
`run_data_driven.py`; they do not authorize a consumer to synthesize a missing
second moment.

The complete record and function table is in
[histogram artifacts](histogram_artifacts.md#topeftmodulesdata_driven_products).

## Missing-parton contract utilities

`topeft.modules.missing_parton_contract` owns the current category and terminal
jet-bin layout used by the payload producer, consumer, and schema tests. See
[missing-parton payloads](missing_parton_payloads.md) for the installed files
and consumer selection.

Developer-facing constants are `SUPPORTED_SR_REGISTRIES`,
`DEFAULT_SR_REGISTRY`, current base/final channel counts, and the bounded legacy
branch/layout constants. Typed record classes model channel application labels,
parsed jet tokens, per-category payload layout, and the full registry layout.
Stable functions are:

| Symbol | Parameters and return | Contract |
| --- | --- | --- |
| `normalize_sr_registry` | Optional registry name → canonical string | Defaults to `ALL_CH_LST_SR`; unknown registries fail. |
| `load_or_validate_selected_registry` | Optional name/config path → selected registry and config | Loads the category JSON and validates exact supported structure. Reads one file. |
| `parse_analysis_njet_token`, `parse_sr_njet_token` | Registry token string → parsed semantic tuple | Enforce direct versus terminal jet-bin syntax; malformed tokens fail. |
| `build_registry_payload_layout` | Selected registry mapping/context → immutable layout | Derives category order, physical bins, and terminal-bin coverage. |
| `load_registry_payload_layout` | Registry/config options → immutable layout | File-reading convenience around the builder. |
| `build_channel_appl_contract`, `load_missing_parton_channel_contract` | Registry/layout/config inputs → channel/application contract | Own exact final SR application labels used by cards; no label guessing. |
| `legacy_missing_parton_payload_lengths` | No required inputs → immutable expected-length mapping | Exposes only the accepted legacy schema. |
| `validate_legacy_missing_parton_values`, `validate_legacy_missing_parton_payload` | Legacy arrays/payload → `None` | Bounded compatibility validation; never converts legacy content into current production authority. |

Tests in `tests/test_missing_parton_contract.py`,
`tests/test_missing_parton_registry_layout.py`, and
`tests/test_missing_parton_sr_registry.py` own these invariants.

Specialist analysis guides are indexed from the main
[documentation landing page](../README.md), not duplicated here.

## Maintained specialist processors

The diboson and sum-of-weights processors and their direct CLIs are maintained
specialist interfaces. They are neither the core
`run_cr.sh` → `fullR3_run.sh` → `run_analysis.py` chain nor historical code.
Their status is supported by current runners, consumers, configuration, or
tests; a processor-like filename alone is not sufficient evidence.

### Diboson processor and CLI

`analysis_processor_diboson.AnalysisProcessor` interprets diboson sample and
era metadata, applies its maintained correction/weight and object-selection
path, selects categories from `topeft/channels/ch_lst_diboson.json`, constructs
observables, and fills ordinary or EFT-aware histogram content. Its EFT algebra
and correction factories are shared `topcoffea` mechanisms; its sample,
category, and application policy remain `topeft` specialist policy.

`run_analysis_diboson.py` selects the specialist input universe, systematics,
WC set, categories, and observables. Executor, worker, chunk, prefix, and output
controls remain execution-only. Its physics controls must not be mixed into the
main `run_analysis.py` contract as though the two processors had identical
category registries or defaults.

### Sum-of-weights processor and CLI

`sow_processor.AnalysisProcessor` accumulates generator normalization sums for
the selected MC samples. It does not perform the main selected-event analysis
or define CR/SR categories. `run_sow.py` owns the specialist input and output
boundary; its executor and chunk controls affect execution or completeness,
while the selected sample metadata determines which normalization sums are
produced.

The resulting sums are consumed through the metadata and normalization flow in
[sample roles and normalization](sample_roles_and_normalization.md).

### Excluded surfaces

Historical, test-only, dead/unreachable-candidate, and uncertain-reachability
surfaces are not promoted by this page. In particular, role-list membership and
registered-but-unfilled diagnostics remain at their actual maintenance class.

## Specialist defaults and examples

| Interface/control | Class | Default or authority | Consequence |
| --- | --- | --- | --- |
| `run_analysis_diboson.py` input JSON | sample semantics | positional maintained sample JSON | Selects diboson sample, era, normalization, and EFT applicability |
| Diboson `--do-systs` | physics correction | false | Adds specialist object/weight templates |
| Diboson `--wc-list` | sample semantics | sample/configured WC list | Selects coefficient content for compatible EFT samples |
| Diboson `--hist-list` and category options | physics selection | specialist processor defaults and `ch_lst_diboson.json` | Changes produced specialist channels/observables |
| Diboson executor/workers/chunks/test/path/output controls | execution only | parser defaults | Changes scheduling, bounded coverage, or storage, not intended physics policy |
| `run_sow.py` input JSON/cfg | sample semantics | required input | Selects samples whose generator normalization sums are accumulated |
| `run_sow.py --wc-list` | sample semantics | inferred from sample metadata when omitted | Selects named EFT normalization sums where available |
| Sum-of-weights executor/chunks/max-files/output controls | execution only | parser defaults | Changes execution/completeness/storage, not the definition of a sum |

Representative specialist invocations are:

```bash
python analysis/topeft_run2/run_analysis_diboson.py <diboson.json> \
  --hist-list njets --do-systs

python analysis/topeft_run2/run_sow.py <sample.json> \
  --outname sowTopEFT --outpath histos
```

Use [production](../how_to/production.md) for the specialist extension
boundary, [testing](../how_to/testing.md) for focused validation, and
[sample roles](../how_to/sample_roles_and_normalization.md) for normalization
metadata. These snippets show maintained interfaces but do not authorize heavy
execution.
