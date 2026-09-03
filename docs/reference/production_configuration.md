# Production configuration

## Configuration layers

| Layer | Source owner | Contract |
| --- | --- | --- |
| Sample JSON | `run_analysis.py` and `topcoffea.modules.utils.load_sample_json_file` | Describes one or more datasets and their file/process metadata |
| Sample cfg | `run_analysis.py` cfg loading | Lists sample JSON inputs; the active universe is derived from the resolved contents, not the cfg filename alone |
| Production profile | `run_cr.sh` | Names an ordered maintained set of production blocks and records campaign state |
| Year/region expansion | `fullR3_run.sh` | Chooses maintained cfgs, hist-list defaults, and the `run_analysis.py` invocation |
| Direct options | `run_analysis.py` argparse | Own executor, resources, analysis toggles, output, policy, and provenance inputs |
| YAML option overlay | `run_analysis.py --options FILE` | Supplies recognized runtime values through one top-level YAML mapping |

The maintained public production interface is the six-profile `run_cr.sh`
matrix described below. `run3_full` is the complete maintained Run 3 SR
production profile. `rebin_fine` is a specialized legacy profile for its
declared changed-binning family blocks; it is not a synonym for the public
profile matrix.

## Current NDSkim cfg reachability

`fullR3_run.sh` is the runtime owner of the current default cfg paths. Run 2 SR
selects `mc_signal_samples_NDSkim.cfg`,
`mc_background_samples_NDSkim.cfg`, and `data_samples_NDSkim.cfg`; the Run 2 CR
path additionally selects `mc_background_samples_cr_NDSkim.cfg`. Run 3 uses
the corresponding `NDSkim_${year}_...` background, data, and signal variants
for CR or SR. Explicit `--sample-json` and `--cfg-override` inputs replace that
default selection for one invocation.

Historical sample configs may remain in the repository for reproduction. Their
presence does not make them reachable from the maintained wrappers.

The six maintained public `run_cr.sh` profiles map to the exact 24-cfg Run 2
and Run 3 SR+CR production surface. That surface is the validated Project01
production contract: canonical sample-JSON entries remain `/store/...` paths,
which resolve through `file:///project01/ndcms/apiccine` onto the accepted
migrated input universe. Cfgs outside this maintained production surface are
outside the validated Project01 production contract; this does not make a
claim about the physical availability of individual inputs associated with
other cfgs.

## Direct defaults

The `run_analysis.py` parser owns current defaults. Notable values are
`--executor work_queue`, `--nworkers 8`, `--chunksize 100000`, output name
`plotsTopEFT`, and output directory `histos`. Analysis mode defaults to
`standard`; nonprompt post-processing defaults to `inline` when requested.
Use `run_analysis.py --help` from the checked-out environment for the complete
option set.

Before processing, the direct CLI resolves an active sample universe, production
sample contract, sumw2 mode and policy, data-driven product contract, category
configuration, and environment identity. The produced PKL sidecar records
those resolved contracts. See [histogram artifacts](histogram_artifacts.md) and
[sumw2 policy](sumw2.md).

## YAML option overlay

`run_analysis.py --options FILE` loads one top-level YAML mapping. The current
parser and loader do not support selecting a named submapping. Recognized YAML values are
applied after the parsed CLI values and therefore replace the corresponding CLI
values when both are present. The overlay is configuration, not an arbitrary
function-call interface: use the exact keys popped by the loader and values
accepted by the corresponding downstream owner.

`sumw2_storage`, `data_driven_products`, and their presence flags are resolved
before processor construction. The deprecated `no_sumw2`/`do_errors` inputs
are compatibility inputs with explicit conflict checks; they must not coexist
as an alternative policy owner. Unrecognized top-level YAML keys are rejected
with their names sorted in the diagnostic. Invalid YAML shape, modern/legacy
conflicts, unsupported keys, or values rejected by downstream resolvers stop
the run before event processing.

## `analysis_processor.AnalysisProcessor`

**Kind/status:** class, developer-facing. **Purpose:** construct and run the
coffea processor that owns event selection, category assignment, systematic
variation filling, and the processor-output histogram mapping. **Signature
authority:** `AnalysisProcessor.__init__`, `process`, and `postprocess` in
`analysis/topeft_run2/analysis_processor.py`. Its constructor is weakly typed;
the curated groups below are the stable contract rather than permission to pass
arbitrary keywords.

### Constructor contract

| Parameter group | Type, requirement, and default | Semantics |
| --- | --- | --- |
| `samples` | Mapping; required | Dataset-keyed sample metadata. Runtime records require `histAxisName`, `isData`, and `WCnames`; additional selection/correction metadata is consumed downstream. |
| `wc_names_lst` | Sequence of strings; default empty | Global WC order for EFT `HistEFT` siblings. Source coefficients are remapped to this order before treatment. |
| `hist_lst` | Optional sequence; default all registered families | Base family names or companion names. Unknown names fail; companion requests resolve through their base family. |
| `ecut_threshold` | Optional numeric; default `None` | Event-level upper-energy diagnostic cut. |
| `fill_sumw2_hist` | Boolean; default `True` | Compatibility switch used only when no resolved `sumw2_policy` is supplied. A supplied policy is the concrete target authority. |
| `do_systematics`, `split_by_lepton_flavor`, `skip_signal_regions`, `skip_control_regions` | Booleans; all default `False` | Filling and category controls. |
| `muonSyst` | String; default `nominal` | Muon systematic selection consumed by correction/selection logic. |
| `dtype` | NumPy dtype; default `numpy.float32` | EFT coefficient/storage numerical dtype where used. |
| `offZ_split`, `tau_h_analysis`, `fwd_analysis`, `all_analysis` | Booleans; all default `False` | Mutually exclusive analysis-mode flags. `all_analysis` enables all three extension blocks; more than one true flag is rejected. |
| `useRun3MVA` | Boolean; default `True` | Selects the maintained Run 3 lepton-MVA path rather than alternative cuts. |
| `tau_run_mode` | `standard` or `taufitter`; default `standard` | Controls tau-specific correction/selection behavior. Unknown modes fail at their owner. |
| `sr_category_dict`, `cr_category_dict` | Optional mappings | Explicit already-resolved category dictionaries; deep-copied when supplied. Default category block names come from the analysis-mode flags. |
| `suppress_forward_eta_stochastic_jer` | Boolean; default `False` at class level | Processor-level forward stochastic-JER suppression control. The direct CLI resolves and passes its own maintained default. |
| `fwd_eta_band_pt_apply` | `auto`, `on`, or `off`; default `auto` | Controls forward-band jet-pT tightening. |
| `ttgamma_sample_role_policy` | Registered policy string; default `split` | Resolves conversion-overlap sample roles; invalid values fail through the policy owner. |
| `sumw2_policy` | Optional `resolved_sumw2_policy` | Concrete runtime family/target authority. Its family order must equal the processor family order. |

Construction returns an `AnalysisProcessor` instance and allocates an
accumulator mapping. One-dimensional nominal families use split
`<family>__scalar_nominal` (`SparseHist`) and
`<family>__eft_nominal` (`HistEFT`) siblings according to active sample
metadata. Selected `<family>_sumw2` companions are `SparseHist` objects on
matching processing axes. Two-dimensional families remain scalar
`SparseHist`. `process(events)` consumes NanoEvents with a dataset metadata key
and returns the filled mapping; `postprocess(accumulator)` returns it unchanged.

Construction has no file-output side effect. `run_analysis.py`, not the
processor, serializes the accumulator and sidecar. Important failure boundaries
include invalid mode combinations, unknown histogram families, a sumw2-policy
family-order mismatch, unsupported EFT treatment, absent required EFT branches,
and incompatible coefficient shapes. `run_analysis.py` is the primary caller;
processor construction and output invariants are covered by
`tests/test_run_analysis_preflight.py`, `tests/test_run_analysis_hist_outputs.py`,
`tests/test_sumw2_policy.py`, and `tests/test_ptll_semantic_contract.py`.

### Processor helper symbols

| Fully qualified symbol | Parameters and return | Contract and failures |
| --- | --- | --- |
| `analysis_processor.validate_analysis_mode_flags` | Four flag-like values → mapping of normalized booleans | Rejects more than one of off-Z, tau, forward, and all-analysis flags. Called during processor construction. |
| `analysis_processor.evaluate_eft_coefficients_at_sm` | Array-like coefficients with quadratic-term final dimension → SM factor array | Derives the WC count and evaluates all WCs at zero. Rejects missing coefficient dimension or a non-quadratic term count. |
| `analysis_processor.calculate_sm_sumw2_weights` | Scalar-weight array and optional EFT-coefficient array → squared complete SM contributions | Without EFT coefficients returns `scalar_weights**2`; with EFT multiplies the scalar weight by the evaluated SM factor first. Shapes must match. |
| `analysis_processor.resolve_eft_treatment` | Sample metadata and optional sample name → `None` or `sm_only` | Only `sm_only` is registered. It is MC-only and requires a non-empty unique string `WCnames` list. |
| `analysis_processor.project_eft_coefficients_for_treatment` | Coefficients, treatment, optional sample name → original or projected array | `None` preserves coefficients. `sm_only` keeps the SM value in term zero and zeros other terms; it requires coefficients. |
| `analysis_processor.prepare_eft_coefficients` | Coefficients, native/global WC orders, treatment, sample name → prepared array | Remaps coefficient order when needed, then applies the explicit treatment. Propagates remap/treatment failures. |
| `analysis_processor.prepare_event_eft_coefficients` | Events, sample metadata, global WC order, treatment, sample name → array or `None` | Reads `EFTfitCoefficients` only when present/needed; a required treated source without the branch fails. |
| `analysis_processor.derive_analysis_enable_toggles` | Four normalized mode flags → off-Z/tau/forward enable mapping | `all_analysis` enables all extension blocks. Caller must first enforce exclusivity. |
| `analysis_processor.resolve_category_dict_names` | Four mode flags → `(sr_block_name, cr_block_name)` | Chooses `ALL`, `OFFZ_SPLIT`, `TAU`, `FWD`, or TOP-22 SR registry and tau/default CR registry. This is selection, not a claim that TOP-22 is the current analysis identity. |
| `analysis_processor.load_category_config` | Optional path, default packaged `channels/ch_lst.json` → decoded mapping | Reads UTF-8 JSON; file and JSON errors propagate. The caller owns semantic block validation. |

The extension point for a new maintained analysis mode is the coordinated
flag/config/category/processor/test contract. Adding only a constructor flag or
only a `ch_lst.json` block is incomplete.

## `topeft.modules.production_sample_profile`

**Kind/status:** typed module API, developer-facing. **Purpose:** freeze the
exact input universe and certify that selected private/central signal variants,
sumw2 policy, and data-driven contributors are mutually consistent before
processing or transformation. `production_sample_profile_error` is its
fail-closed `ValueError` subtype.

| Symbol | Parameters and return | Stable contract |
| --- | --- | --- |
| `signal_variant_group` | Fields `name`, `years`, `private_bases`, `central_bases`; immutable dataclass | Defines one validated private/central equivalence group. `VALIDATED_SIGNAL_VARIANT_GROUPS` is deliberately narrower than all signal samples. |
| `active_sample_universe` | Fields `wrapper_identity`, canonical `cfg_identities`, sorted `datasets`, sorted unique `processes` | Immutable exact input identity. `serialized_cfg_identities()` returns JSON-ready records. |
| `build_active_sample_universe` | `samples` mapping; keyword `input_paths=()` and `wrapper_identity="run_analysis.py"` → `active_sample_universe` | Requires non-empty dataset keys and `histAxisName` labels. Hashes exact cfg/JSON bytes; direct library input receives a deterministic generated identity. |
| `validate_active_sample_profile` | Universe, `sumw2_mode_resolution`; optional data-driven mapping/presence and metadata path → `None` | Checks configured contributor names/prefixes, private/central duplication, and mode/profile agreement. Raises coded profile errors before policy resolution. |
| `certify_production_sample_contract` | Universe, resolved sumw2 policy, resolved data-driven products → JSON-ready mapping | Requires identical cfg/policy universes and complete contributor sumw2 targets; writes schema version 1 and a content SHA-256 identity, then validates it. |
| `validate_production_sample_contract` | Contract mapping and resolved policy or provenance mapping → `None` | Requires the exact field set, current sumw2 schema, canonical unique cfg identities, valid digest, certified flag, and signal variants matching provenance. |
| `require_data_driven_profile_certification` | Sidecar mapping → `None` | A new transformation requires current sumw2 provenance plus a production sample contract. Older artifacts may be reopened read-only but cannot authorize transformation. |

The schema-v1 production contract fields are `contract_version`,
`wrapper_identity`, `cfg_identities`, `resolved_mode`,
`signal_sample_profile`, `active_signal_variants`,
`compatibility_validated`, and `contract_identity_sha256`. The processor sidecar
is its consumer and storage owner. Tests are in
`tests/test_production_sample_profile.py`.

## Source and test authority

- `analysis/topeft_run2/run_cr.sh`
- `analysis/topeft_run2/fullR3_run.sh`
- `analysis/topeft_run2/run_analysis.py`
- `topeft/modules/production_sample_profile.py`
- `tests/test_run3_full_production_profile.py`
- `tests/test_run_analysis_preflight.py`
- `tests/test_production_sample_profile.py`

## Current `run_cr.sh` profiles

`run_cr.sh` exposes six maintained public profiles: `run2_full`, `run3_full`,
`run2_full_CR`, `run3_full_CR`, `run2_run3_full`, and
`run2_run3_full_CR`. The combined profiles place their Run 2 and Run 3
components in separate child namespaces. The no-argument invocation remains a
legacy alias for the fixed `run2_full` campaign. `rebin_fine` remains a
specialist legacy profile for fitting families whose bins changed; it is not a
replacement for the public profile matrix.

The public profiles require a fresh absolute output directory and campaign tag,
use the maintained frozen environment archive, and do not expose a profile-level
Work Queue worker-count setting. See [the production runbook](../how_to/production.md)
for the environment and recovery contract.
