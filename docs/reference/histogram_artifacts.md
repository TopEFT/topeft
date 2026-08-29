# Histogram artifacts and provenance

The maintained on-disk unit is a histogram PKL and its adjacent metadata
sidecar. Consumers validate the pair; a PKL filename alone is not provenance.

## Artifact kinds and schemas

| Contract | Current owner | Current value or role |
| --- | --- | --- |
| Metadata sidecar schema | `topeft.modules.histogram_artifact` | Version 2 |
| Nominal split-family schema | `topeft.modules.nominal_schema` | Version 2, `split_sibling_v1` layout |
| Sumw2 content manifest | `histogram_artifact.build_sumw2_content_manifest` | Version 1 |
| Transformation contract | `topeft.modules.histogram_artifact` | Version 3; version 2 has bounded read compatibility |
| Processor artifact | `artifact_kind == "processor_output"` | Direct processor output with production/sample and sumw2 contracts |
| Transformed artifact | nonprompt or flips artifact kind | New payload derived from a validated input and recorded lineage |

`metadata_sidecar_path(pkl_path)` appends `.metadata.json` without removing the
PKL suffix. The sidecar records artifact identity, kind, nominal layout, sumw2
provenance and content, production sample contract, data-driven applicability,
and lineage where applicable.

## Publication and readback interfaces

All symbols below are developer-facing; typed source in
`topeft.modules.histogram_artifact` is signature/schema authority.

| Fully qualified symbol | Parameters and return | Contract, side effects, and failures |
| --- | --- | --- |
| `histogram_artifact.metadata_sidecar_path` | PKL path-like → adjacent path | Appends `.metadata.json` without stripping the PKL suffix. No I/O. |
| `histogram_artifact.derive_data_driven_applicability` | Region labels → product applicability mapping | Recognizes only maintained application-region semantics; malformed input fails during validation. |
| `histogram_artifact.build_sumw2_content_manifest` | Histogram mapping plus required provenance/artifact kind and optional transformed requirements → JSON-ready mapping | Recomputes nominal/companion process content per family. Processor requirements derive from policy; transformed requirements are independently derived. |
| `histogram_artifact.lineage_input_from_sidecar` | Validated sidecar mapping → compact lineage-input mapping | Projects content/identity fields needed to bind a derived artifact to its source. No I/O. |
| `histogram_artifact.required_sumw2_processes_from_transformation_contract` | Transformation contract plus family/product context → required process set | Reads exact contributor/output authority; unsupported versions or malformed contracts fail. |
| `histogram_artifact.derive_transformed_required_sumw2_processes` | Input sidecar, resolved product contract, and transformed context → per-family requirements | Derives transformed companion requirements rather than copying source requirements blindly. |
| `histogram_artifact.read_histogram_sidecar` | PKL or sidecar path → parsed mapping | Reads JSON and validates the record. Missing, malformed, unsupported, or identity-inconsistent metadata raises a sidecar/artifact error. |
| `histogram_artifact.validate_processor_output` | Histogram mapping and sidecar/context → `None` | Requires processor kind, valid nominal layout, policy/content equality, and current production sample certification where applicable. |
| `histogram_artifact.validate_nonprompt_output` | Histogram mapping and sidecar/context → `None` | Requires the nonprompt contract, exact generated outputs, lineage, and independently complete companions. |
| `histogram_artifact.validate_nonprompt_nominal_reference_output` | Mapping and sidecar/context → `None` | Validates the explicit nominal-reference kind; it is not interchangeable with a full transformed product. |
| `histogram_artifact.validate_flips_output` | Mapping and sidecar/context → `None` | Enforces flips contributors, outputs, applicability, lineage, and companion content. |
| `histogram_artifact.validate_histogram_artifact` | Payload path/mapping plus validation context → validated sidecar/result | Dispatches by exact artifact kind. Unknown kinds and payload/metadata disagreement fail closed. |
| `histogram_artifact.merge_histogram_sidecars` | Sequence of validated sidecars plus merge context → merged sidecar mapping | Requires compatible schema, requested-product source and products, policy, production identity, transformation version, lineage, axes, and contributions. Requested-product warning lists are diagnostic rather than compatibility gates: the merger retains their sorted unique union, then validates the composed contracts. It does not merge payload objects. |
| `histogram_artifact.write_histogram_sidecar` | Output path, artifact kind, histograms, provenance/contracts, optional validated input context → JSON-ready sidecar mapping | Writes JSON atomically. Transformation/lineage are derived from input evidence; an unrelated caller-authored original contract is rejected. |
| `histogram_artifact.write_histogram_artifact` | Output path plus exactly one histogram mapping or payload-writer callback, metadata, and contracts → published sidecar mapping | Atomically publishes the payload/sidecar pair and restores prior files after publication failure. Passing both or neither payload source fails. |

The `sumw2_content_manifest` records each family, its dimensionality, scalar and
EFT nominal processes, available companions, and the companions required by the
resolved policy or transformation. Consumers recompute requirements rather
than trusting a positive sidecar label.

The module exposes `histogram_artifact_error` and the specific
`histogram_sidecar_error`, `histogram_content_error`, and
`histogram_merge_error` types. Callers should surface those diagnostics rather
than retry with validation disabled.

## Sidecar field ownership

Common fields identify metadata schema, artifact kind/identity, merge state,
nominal schema/layout, sumw2 provenance, content manifest, and data-driven
request/resolution state. Current processor artifacts additionally require a
certified `production_sample_contract`. Transformed products require lineage
and a transformation contract. Prior transformation version 2 is readable
within established semantics but is not silently mixed with version 3. A
nominal reference has its own explicit kind/reference contract.

Exact required fields are enforced by `_build_sidecar_payload` and the matching
validator. Optionality is artifact-kind dependent; a field accepted on a
processor artifact is not automatically accepted on a transformed artifact.

## `topeft.modules.nominal_schema`

**Kind/status:** typed module API, developer-facing. **Purpose:** own the
schema-v2 split nominal layout, sibling access, validation, merge, EFT
evaluation, and bounded consumer materialization. Source types are useful but
docstrings are shallow; the table below is the curated contract.

`NOMINAL_CONTAINER_SCHEMA_VERSION` is 2 and
`NOMINAL_CONTAINER_LAYOUT` is `split_sibling_v1`. For a 1D family, helpers
derive `<family>__scalar_nominal`, `<family>__eft_nominal`, and
`<family>_sumw2`. The unsplit `<family>` key is forbidden in schema-v2 producer
output. A 2D family remains scalar at `<family>` and may have the sumw2 suffix.
Scalar and companion objects must be exact `SparseHist` with
`storage="Double"`; EFT siblings must be exact `HistEFT`.

| Fully qualified symbol | Parameters and return | Contract and failure boundary |
| --- | --- | --- |
| `nominal_schema.scalar_nominal_key`, `eft_nominal_key`, `sumw2_key`, `family_from_component_key` | Family/component string → canonical key or family | Pure naming helpers; suffix constants are schema authority. |
| `nominal_schema.validate_histogram_compatibility` | Two histograms; required diagnostic key; optional sumw2-name normalization → `None` | Requires exact concrete type, categorical axis types/names, dense type/name/edges/flow, and HistEFT WC order. |
| `nominal_schema.get_nominal_components` | Mapping, family, schema default current → ordered component mapping | Returns scalar/EFT, 2D scalar, or legacy uniform content. Rejects duplicate authority, split siblings under old schema, and unsplit 1D content under v2. |
| `get_scalar_nominal`, `get_eft_nominal` | Same lookup inputs → matching histogram or `None` | Type-selecting accessors; preserve schema errors from `get_nominal_components`. |
| `iter_nominal_components` | Same lookup inputs → iterable of `(component_name, histogram)` | Preserves deterministic component order. |
| `nominal_schema.validate_nominal_family` | Mapping, family; optional schema, companion selection, selected processes → `None` | Rejects absent nominal content, orphan/unselected/missing companion, wrong type/storage, sibling axis mismatch, duplicate process ownership, extra companion processes, and partial required coverage. |
| `nominal_schema.validate_nominal_mapping` | Mapping; runtime family order; optional schema/policy → `None` | Applies family validation and rejects orphan component keys outside the runtime family set. |
| `nominal_schema.canonicalize_nominal_keys` | Mapping plus runtime family order/schema → new built-in dict | Returns deterministic schema order, retaining unrelated keys afterward. Does not mutate inputs. |
| `nominal_schema.merge_nominal_mappings` | Nonempty iterable of mappings plus runtime families; optional schema/policy → ordered merged mapping | Deep-copies content, validates represented families and exact compatibility, adds compatible duplicates, canonicalizes, and revalidates. |
| `nominal_schema.evaluate_eft_histogram_at_wc` | Exact `HistEFT`, optional WC values default SM → scalar `SparseHist` | Evaluates populated categories without changing axes; wrong input type fails. |
| `nominal_schema.evaluate_nominal_at_wc` | Mapping, family, optional WC values/schema → scalar `SparseHist` | Deep-copies scalar content, evaluates EFT content, verifies compatibility, and sums them. Missing/unsupported content fails. |
| `nominal_schema.map_nominal_components` | Mapping, family, callable, optional schema → ordered mapping | Applies one in-memory operation to each nominal sibling and companion. Callable failures propagate. |
| `nominal_schema.materialize_nominal_family` | Mapping, family; optional schema/WC order → uniform compatibility histogram | Builds a consumer-only HistEFT view for split 1D content or copies scalar 2D content. WC order must agree. |
| `nominal_schema.materialize_legacy_histogram_dict` | Mapping; optional families/schema/required companions → ordered uniform view | Bounded compatibility view that does not alter serialized source content. Missing required companions fail. |
| `nominal_schema.materialize_scalar_histogram_dict` | Mapping; runtime families; optional WC values/schema/required companions → ordered scalar view | Evaluates every nominal family for scalar consumers. Companions remain deep-copied scalar histograms. |

These functions perform no file I/O. Artifact writers, mergers, plotters, cards,
and transformations are their important consumers. Validation ownership is
`tests/test_nominal_schema.py` plus artifact/merge consumer tests.

## `topeft.modules.data_driven_products`

**Kind/status:** typed module API, developer-facing. **Purpose:** resolve the
requested nonprompt/flips configuration into exact same-year contributor and
generated-output contracts, then validate transformation/readback compatibility.
`data_driven_product_error` is its fail-closed `ValueError` subtype.

The requested-products schema is 1. The current resolved contract is 4;
precanonical version 3 and legacy version 1 have bounded readback paths.
Registered products are `nonprompt` and `flips`. Artifact kinds are
`nonprompt_output`, `nonprompt_nominal_reference_output`, and `flips_output`.

| Fully qualified symbol | Fields or parameters; return | Stable contract |
| --- | --- | --- |
| `data_driven_products.normalized_process_selector` | Immutable exact `process_names` and `process_prefixes`, both default empty | `to_dict()` emits only populated selector dimensions. |
| `data_driven_products.resolved_generated_output` | Immutable canonical `year` and role→process tuples | Provides role lookup and deterministic required-source union. |
| `data_driven_products.resolved_product` | Enabled state, selectors, resolved contributors/datasets, generated outputs | Provides role/dataset/output queries and exact required-process union. |
| `data_driven_products.resolved_data_driven_products` | Source, metadata path, runtime families, product records, optional certified nonprompt policy, warnings | Provides enabled products, requested provenance, and exact dataset/process/family sumw2 targets. |
| `data_driven_products.parse_process_name` | Canonical process label → `(base_name, year)` | Requires a supported exact year-qualified process. |
| `data_driven_products.generated_process_name` | Product and canonical year → canonical output process label | Rejects an unknown product or year. |
| `data_driven_products.group_contributors_by_generated_output` | Product, role→process mapping; metadata path/source → ordered generated-output records | Groups exact same-year contributors. Nonprompt prompt years without same-year data fail. |
| `required_source_processes_from_generated_outputs` | Serialized or resolved generated outputs → sorted tuple | Deterministic source-process union, never inferred from observed payload categories. |
| `generated_output_processes_from_contract`, `resolved_prompt_processes_from_contract` | Contract mapping, and product where applicable → exact tuples | Read only supported current/precanonical/legacy layouts; unsupported versions fail. |
| `data_driven_products.resolve_data_driven_products` | Requested config/presence plus samples, runtime families, metadata context → resolved record | Validates schemas, fields, selectors, contributors, datasets, and outputs without creating histogram content. |
| `data_driven_products.certify_data_driven_preflight` | Resolved products and resolved sumw2 policy → `(requested_provenance, resolved_contract)` mappings | Requires every exact source dataset/process/family target before processing, then self-validates both serialized records. |
| `validate_generated_output_contract`, `validate_generated_outputs_against_sumw2_policy` | Output contract plus product/policy/content context → `None` | Reject labels, contributors, years, or companion coverage differing from certification. |
| `validate_serialized_data_driven_contract` | Contract mapping and compatibility context → validated result | Enforces exact current, precanonical, or legacy fields; versions are not silently rewritten. |
| `reresolve_nonprompt_policy_from_sidecar` | Sidecar and current context → certified policy | Reopens serialized source authority and rejects incompatible re-resolution. |
| `resolve_requested_product_input`, `validate_requested_product_input` | Product, input histograms/sidecar and context → resolved input/result | Never fabricates a missing second moment or empty product; disabled, inapplicable, and absent remain distinct. |
| `resolved_sumw2_policy_from_sidecar` | Sidecar mapping → `resolved_sumw2_policy` | Parses embedded policy through the sumw2 schema owner. |

Resolution has no storage side effects. `run_analysis.py` consumes the
requested/resolved policy; `run_data_driven.py` and the data-driven producer
consume the certified transformation contract; artifact/merge consumers enforce
it. Test ownership is `tests/test_data_driven_products.py`,
`tests/test_run_data_driven.py`, and sidecar tests.

For the in-memory histogram API and pickle compatibility surface, see
[HistEFT and SparseHist](histeft.md). For companion selection, see
[sumw2 policy](sumw2.md).

## Source and test authority

- `topeft/modules/histogram_artifact.py`
- `topeft/modules/nominal_schema.py`
- `topeft/modules/data_driven_products.py`
- `tests/test_histogram_artifact_sidecars.py`
- `tests/test_nominal_schema.py`
- `tests/test_data_driven_products.py`

## Physics-content boundary

An artifact schema records process, category, observable, systematic, EFT, and
statistical content together with provenance. It does not assign scientific
meaning to a process or region name independently of the processor and
registries that produced it. Source artifacts represent the processor fill;
transformed artifacts additionally represent the certified data-driven
product and its lineage.

Consumers must validate the content family they need. A nominal histogram does
not imply that every systematic variation, sumw2 companion, EFT coefficient,
or data-driven target is present. See the
[analysis processor map](analysis_processor.md),
[data-driven estimation](data_driven_estimation.md), and
[categories and observables](categories_and_observables.md).

## Practical schema and change bridge

Canonical schemas and writers are
[`nominal_schema.py`](../../topeft/modules/nominal_schema.py),
[`histogram_artifact.py`](../../topeft/modules/histogram_artifact.py), and
[`data_driven_products.py`](../../topeft/modules/data_driven_products.py).
There is no single artifact default: source, split nominal/sumw2, transformed,
and bounded compatibility readbacks are distinct families.

For a representative source-to-transformed chain, begin with a processor
artifact and its sidecar, validate it, then let `run_data_driven.py` publish a
new artifact/sidecar with source lineage. Use
[production](../how_to/production.md) for source publication,
[nonprompt](../how_to/nonprompt.md) for transformation, and
[sumw2](../how_to/sumw2.md) for companion-policy changes.
