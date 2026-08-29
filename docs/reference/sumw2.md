# Sumw2 policy

Sumw2 companions store the sum of squared complete event contributions at the
SM point (`WC = 0`) for the concrete dataset/process/family targets selected by
policy. A companion is named `<family>_sumw2` and uses the same processing axes
as its nominal sibling. Nonzero-WC quartic sumw2 is not part of this contract.

## Modes and default

`topeft.modules.sumw2_policy` is the registry and default owner.

| Mode | Contract |
| --- | --- |
| `production` | Required production companions using the private signal sample profile |
| `production_central` | Required production companions using the central signal sample profile |
| `taufitter` | Required companions for the tau-fitter analysis mode |
| `full_diagnostics` | Every concrete runtime target |
| `disabled` | No sumw2 companions |
| `full_custom` | Explicit rule-selected targets |

The current default is `production` when `sumw2_storage` is absent and when a
present mapping omits `mode`. Deprecated `no_sumw2` values map explicitly to
`disabled` or `full_diagnostics` with a warning; they do not establish a second
default.

Rule selectors are `dataset_names`, `dataset_prefixes`, `process_names`,
`process_prefixes`, and `variables`. Unknown modes/families, empty or zero-match
rules, overlapping targets, mode/analysis mismatch, signal-profile mismatch,
and absent consumer requirements fail closed.

## Resolved policy and provenance

The current provenance schema is 2. Schema 1 remains readable within explicit
compatibility limits and is not silently upgraded. A resolved record includes
the configuration source, requested and resolved mode, signal profile,
normalized rules, runtime families, concrete targets, resolved datasets and
processes, warnings, and deterministic identity.

The content manifest is a separate contract: it describes which companion
content is physically present. A resolved mode, its default, provenance schema,
and the content manifest therefore answer different questions.

## Developer interfaces

All symbols below are developer-facing. Their typed definitions in
`topeft.modules.sumw2_policy` are signature authority.

| Fully qualified symbol | Parameters or fields; return | Contract, consumers, and failures |
| --- | --- | --- |
| `sumw2_policy.SUMW2_PROVENANCE_SCHEMA_VERSION` | Integer constant `2` | Current serialized policy schema. Sidecar writers and current transformation certification require it. |
| `sumw2_policy.LEGACY_SUMW2_PROVENANCE_SCHEMA_VERSION` | Integer constant `1` | Read compatibility only; cannot encode `production_central`. |
| `sumw2_policy.SUMW2_MODES` | Frozen set of six strings | Accepted values listed above. `RULE_MODES` contains the modes that accept/require explicit rules; `PRODUCTION_SIGNAL_SAMPLE_PROFILES` maps production modes to private/central. |
| `sumw2_policy.sumw2_target` | Immutable `dataset: str`, `process: str`, `family: str`; `to_dict()` → exact three-field mapping | One concrete storage requirement and deterministic ordering unit. Used by policy, production certification, manifests, and data-driven requirements. |
| `sumw2_policy.sumw2_mode_resolution` | Immutable `source`, requested/resolved modes, signal profile, normalized storage mapping, warnings | Public mode result before any samples/families are selected. |
| `sumw2_policy.normalized_sumw2_rule` | Tuple fields for exact names/prefixes and variables; `variables_wildcard=True` by default | `to_dict()` omits empty selector dimensions and wildcard variables; `identity()` is canonical JSON. Lists must contain unique non-empty strings. |
| `sumw2_policy.resolved_sumw2_policy` | Immutable source/mode/profile/rules/runtime universe/concrete targets/warnings/schema | Query methods test a target/family and return selected families/processes. `to_provenance()` is the sidecar record; `identity()` is canonical JSON used for merge equality. |
| `sumw2_policy.resolve_sumw2_storage_mode` | Optional mapping; presence flag; legacy presence/value flags → `sumw2_mode_resolution` | Defaults to `production`; forbids modern plus explicit legacy configuration, unknown fields, and unknown modes. Emits deprecation warnings for legacy mappings. |
| `sumw2_policy.resolve_sumw2_storage_policy` | Storage mapping plus required sample mapping, ordered runtime families, 1D/2D registries; optional analysis mode, legacy flags, consumer requirements, implicit requirements, prior mode resolution → `resolved_sumw2_policy` | Converts selectors into non-overlapping concrete targets. Requires unique registered family order, taufitter mode coupling, nonempty rule coverage, and inclusion of every declared consumer target. |
| `sumw2_policy.resolved_policy_from_provenance` | Provenance mapping → `resolved_sumw2_policy` | Requires the exact schema-specific field set and canonical deterministic order/content. Validates source enum, modes, profiles, rules, and target shapes. |
| `sumw2_policy.validate_policy_identity` | Two policies or provenance mappings → `None` | Parses both sides and rejects different canonical identities. Used at cross-artifact boundaries. |
| `analysis_processor.calculate_sm_sumw2_weights` | Scalar-weight array and optional EFT coefficient array → squared array | Computes the physical software quantity stored by companions; coefficient and scalar evaluated shapes must match. |

### Serialized policy fields

Schema 2 requires exactly `schema_version`, `source`, `requested_mode`,
`resolved_mode`, `signal_sample_profile`, `normalized_rules`,
`runtime_histogram_families`, `resolved_datasets`, `resolved_processes`,
`resolved_targets`, and `warnings`. Each target has exactly `dataset`,
`process`, and `family`. Unknown or missing fields fail. Schema 1 omits
`resolved_mode` and `signal_sample_profile`; its compatible profile is derived
from the requested mode.

Resolution does not write files. Its side effects are bounded to emitted
`UserWarning`s for compatibility/default notices. `run_analysis.py` is the
primary resolver caller; `AnalysisProcessor`, artifact writers, transformations,
plot/card merge consumers, and production certification consume the result or
its provenance.

Adding a mode, changing the default, and changing the provenance schema are
independent maintenance operations. Each has different registry, compatibility,
consumer, and test consequences; none should be implemented as an incidental
side effect of another.

See [histogram artifacts](histogram_artifacts.md) for publication/readback and
[flexible binning](flexible_binning.md) for the shared physical axes.

## Source and test authority

- `topeft/modules/sumw2_policy.py`
- `analysis/topeft_run2/analysis_processor.py`
- `analysis/topeft_run2/run_analysis.py`
- `tests/test_sumw2_policy.py`
- `tests/test_run_analysis_preflight.py`
- `tests/test_run_analysis_hist_outputs.py`

## Physics-facing consequence

Sumw2 content is the statistical second moment of the stored weighted yield.
It is not a detector, modeling, or theory nuisance variation. Retaining it
allows plotting and card consumers to derive variance-dependent quantities
under the certified policy; opting out removes those companions and can make a
downstream consumer inadmissible even when nominal yields are present.

For EFT content, the policy distinguishes scalar companions and the maintained
SM-point statistical treatment from the coefficient-bearing nominal object.
Data-driven transformations must preserve or validate the required companion
families rather than synthesize a missing variance. See
[data-driven estimation](data_driven_estimation.md) and
[histogram artifacts](histogram_artifacts.md).

## Practical bridge

The canonical registry/default/resolver is
[`topeft/modules/sumw2_policy.py`](../../topeft/modules/sumw2_policy.py).
Sumw2 storage is enabled in the direct CLI unless `--no-sumw2` is supplied;
modern YAML `sumw2_storage` policy is resolved before processor construction.
Artifact families can have different applicable companion sets, so there is no
single universal companion key list.

Use [select or change sumw2 storage](../how_to/sumw2.md) for mode, default,
schema, and consumer changes. Low-level resolver/schema entries link to that
owning procedure rather than receiving separate editing recipes.
