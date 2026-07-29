# HistEFT API contract and replacement parity specification

## 1. Purpose and scope

This document is the CL007AD compatibility contract for the current
`HistEFT`/`SparseHist` implementation and for the way `topeft` actually uses it.
It is intended to gate any future EFT-aware histogram replacement, including a
possible `scikit-hist` backend.

This is not an implementation plan for the replacement. It does not change
physics behavior, `HistEFT`, `SparseHist`, processors, plotting, datacard logic,
runner scripts, sample JSON/CFG files, payloads, or production outputs. It
extracts the existing contract from source inspection and small current-behavior
tests.

The central distinction is:

- `Implemented`: behavior present in `topcoffea.modules.histEFT.HistEFT` or
  inherited from `topcoffea.modules.sparseHist.SparseHist`.
- `Used by topeft`: behavior directly or practically used by processors,
  plotting, datacards/yields, pkl helpers, or existing tests.
- `Replacement priority`: whether a future backend must reproduce the behavior
  immediately, only for old pkl compatibility, only if a consumer is not
  migrated, or can defer it.

## 2. Source map

Implementation sources inspected read-only:

| Source | Role |
| --- | --- |
| `../topcoffea/topcoffea/modules/histEFT.py:23-388` | `HistEFT` class, WC metadata, fill, evaluation, pickle reduce, scaling |
| `../topcoffea/topcoffea/modules/sparseHist.py:15-529` | sparse categorical storage, slicing, grouping, views, arithmetic, pickle reduce |
| `../topcoffea/topcoffea/modules/eft_helper.py:10-266` | quadratic-term counts, EFT evaluation, quartic sumw2 helpers, coefficient remapping |
| `../topcoffea/topcoffea/modules/compat.py:13-51` | import compatibility helpers for old HistEFT pickles and hist utilities |
| `../topcoffea/topcoffea/modules/utils.py:399-477` | gzip/cloudpickle pkl writing and materialized pkl loading wrapper |
| `../topcoffea/topcoffea/modules/hist_utils.py:36-293` | streaming/lazy pkl reading, empty-hist filtering |

`topeft` usage sources inspected:

| Source | Role |
| --- | --- |
| `analysis/topeft_run2/analysis_processor.py:233-343,620-631,1896-1924` | main processor histogram declaration, EFT coefficient remapping, HistEFT/SparseHist fills |
| `analysis/topeft_run2/analysis_processor_diboson.py:72-110,297-304,1291-1334` | diboson processor HistEFT declaration and fills |
| `analysis/topeft_run2/run_analysis.py:1539-1595,1760-1784` | WC-list aggregation, processor construction, gzip/cloudpickle pkl writing, pkl postprocess load/write |
| `analysis/topeft_run2/make_cr_and_sr_plots.py:49-110,732-807,1263-1321,5810-5832,6231-6277,6461-6531,6742-6785,7802-7816` | pkl load compatibility, axis labels, filtering, grouping, SM eval, SparseHist 2D views |
| `topeft/modules/datacard_tools.py:39-172,175-302,788-854,947-1130,1176-1210,1335-1553` | pkl merge validation, axis/WC checks, grouping/pruning, scaling extraction, datacard decomposition |
| `topeft/modules/yield_tools.py:306-345,430-482,505-582,816-827` | axis labels, channel-label restoration, integration, yield evaluation, values printing |
| `analysis/topeft_run2/inspect_histeft_pkl.py:20-345` | read-only pkl inspection, axis/WC discovery, nominal eval/value/variance summaries |
| `topeft/modules/axes.py:1-260` | dense axis definitions used by processors |
| `tests/test_group_bins.py:20-80`, `tests/test_make_cards_multi_pkl.py:11-104`, `tests/test_data_driven_streaming.py:17-99` | existing lightweight HistEFT/SparseHist usage in tests |

Prior documentation context:

- `docs/histeft_pkl_tutorial.md:992-1365` documents student-facing API notes and
  a broad future replacement mapping. This CL007AD document turns that material
  into an explicit implemented-versus-used matrix and parity-test gate.

## 3. Implementation inventory

### HistEFT inventory

`HistEFT` is defined as `class HistEFT(SparseHist, family=_family)` in
`../topcoffea/topcoffea/modules/histEFT.py:23`.

Constructor and core attributes:

- `HistEFT.__init__(*args, wc_names=None, **kwargs)`, source
  `histEFT.py:74-126`.
- Supports only `storage="Double"` and rejects `rebin=True`.
- Requires named axes.
- Requires one user dense axis, last among user axes, of type
  `Regular`, `Variable`, or `Integer`.
- Creates or accepts a dense internal `quadratic_term` axis.
- Rejects reserved axis names `quadratic_term`, `sample`, `weight`, and
  `thread`.
- Stores `_wc_names`, `_wc_count`, `_quad_count`, `_coeff_axis`,
  `_dense_axis`, and `_init_args_eft`.
- Delegates to `SparseHist.__init__` with user axes plus the hidden coefficient
  axis.

Public or practically public `HistEFT` methods and properties:

| Method/property | Source | Contract summary |
| --- | --- | --- |
| `empty_from_axes(*args, **kwargs)` | `histEFT.py:128-131` | Preserve `wc_names` when reconstructing empty HistEFT objects |
| `wc_names` | `histEFT.py:133-135` | Return WC names in evaluation order |
| `index_of_wc(wc)` | `histEFT.py:137-138` | Return WC index, raising `KeyError` for unknown names |
| `quadratic_term_index(*wcs)` | `histEFT.py:140-163` | Map two factors, including `"sm"`, to lower-triangle coefficient index |
| `should_rebin()` | `histEFT.py:165-166` | Current HistEFT always reports false |
| `dense_axis` | `histEFT.py:169-170` | Return the user physics dense axis, not `quadratic_term` |
| `fill(eft_coeff=None, **values)` | `histEFT.py:197-249` | Fill weighted EFT coefficients into hidden coefficient axis |
| `eval(values)` | `histEFT.py:271-284` | Evaluate populated sparse blocks at a WC point |
| `as_hist(values)` | `histEFT.py:286-305` | Evaluate and materialize a regular `hist.Hist` without the coefficient axis |
| `__reduce__()` | `histEFT.py:307-319` | Pickle categorical axes, dense axis, init args, WC args, and `_dense_hists` |
| `make_scaling(flow="show", wc_list=None)` | `histEFT.py:321-353` | Produce EFT scaling coefficients, optionally remapped and flow-handled |
| `_read_from_reduce(...)` | `histEFT.py:355-356` | Delegate pickle reconstruction to `SparseHist` |
| `calc_eft_weights(q_coeffs, wc_values)` | `histEFT.py:361-388` | Implement local coefficient evaluation over coefficient-axis flow layout |

Private helpers that affect public behavior:

| Helper | Source | Behavioral impact |
| --- | --- | --- |
| `_fill_flatten(a, n_events)` | `histEFT.py:172-188` | Accept scalar-like, 1D, or `(n_events, 1)` arrays and repeat by `_quad_count`; raise on incompatible shape |
| `_fill_indices(n_events)` | `histEFT.py:190-195` | Repeat coefficient-term indices for each event |
| `_wc_for_eval(values)` | `histEFT.py:251-269` | Normalize `None`, mapping, or array-like WC inputs; missing WCs default to zero; unknown mapping keys raise `LookupError` |

### SparseHist inventory

`SparseHist` is defined as `class SparseHist(hist.Hist, family=hist)` in
`../topcoffea/topcoffea/modules/sparseHist.py:15`.

Constructor and storage:

- `SparseHist.__init__(*args, **kwargs)`, source `sparseHist.py:18-39`.
- Splits axes into categorical and dense groups.
- Stores `_categorical_axes`, `_dense_axes`, `_dense_hists`, `_init_args`, and
  a namedtuple type `_tuple_t`.
- Calls `hist.Hist.__init__` on categorical axes only.
- Exposes `self.axes` as categorical axes plus dense axes.

Public or practically public inherited methods:

| Method/property | Source | Contract summary |
| --- | --- | --- |
| `empty_from_axes` | `sparseHist.py:60-70` | Construct empty object preserving dense/categorical axis split |
| `make_dense` | `sparseHist.py:72-73` | Construct dense `hist.Hist` block |
| `__copy__`, `__deepcopy__` | `sparseHist.py:75-83` | Copy sparse histogram with deep-copied dense blocks |
| `categories_to_index`, `index_to_categories` | `sparseHist.py:103-109` | Convert between category labels and namedtuple sparse keys |
| `categorical_axes`, `dense_axes`, `categorical_keys` | `sparseHist.py:112-122` | Expose axis metadata and populated sparse keys |
| `fill(**kwargs)` | `sparseHist.py:124-139` | Create dense block per categorical key and fill dense values |
| `__getitem__`, `__setitem__` | `sparseHist.py:273-325` | Mapping/tuple selection and assignment; can return sparse hist, dense hist, or scalar |
| `values(flow=False)` | `sparseHist.py:327-350` | Return values over populated sparse blocks as awkward arrays |
| `counts(flow=False)` | `sparseHist.py:352-353` | Return counts over populated sparse blocks |
| `reset()`, `empty()` | `sparseHist.py:359-443` | Reset dense blocks; test whether all dense values are zero |
| `view(flow=False, as_dict=True)` | `sparseHist.py:362-371` | Return mapping from sparse key to dense view; `as_dict=False` raises |
| `integrate(name, value=None)` | `sparseHist.py:373-376` | Slice by axis; `value=None` maps to `sum` |
| `group(axis_name, groups)` | `sparseHist.py:378-406` | Merge categorical labels into group labels |
| `remove(axis_name, bins)`, `prune(axis, to_keep)` | `sparseHist.py:408-433` | Remove or keep categorical labels |
| `scale(factor)` | `sparseHist.py:435-437` | Mutate in place through `self *= factor`, return self |
| arithmetic and merge operators | `sparseHist.py:445-522` | Scalar and SparseHist binary ops, including `+=`, `+`, `*=`, `*`, `/` |
| `__reduce__`, `_read_from_reduce` | `sparseHist.py:466-483` | Pickle reconstruction from axes, init args, and dense hist blocks |
| `identity()` | `sparseHist.py:524-529` | Deprecated old-coffea accumulator compatibility: empty copy |

Not implemented directly in `SparseHist`/`HistEFT` but observed or mentioned in
`topeft`:

- `replace_axis(...)` is called by `yield_tools.restore_split_channel_labels` at
  `topeft/modules/yield_tools.py:430-463`; this appears to rely on inherited
  `hist.Hist` behavior or a path that needs runtime validation with SparseHist.
- `rebin(...)` is called by `datacard_tools.read` when `h.should_rebin()` is true
  at `topeft/modules/datacard_tools.py:832-834`, but `HistEFT.should_rebin()`
  returns false and `HistEFT.__init__` rejects `rebin=True`.
- `variances(...)` is probed by `inspect_histeft_pkl.py:192-199`; HistEFT does
  not define it, and current workflow uses `_sumw2` companions.
- `.sum(...)`, `.project(...)`, and old-coffea `.identifiers()` appear in older
  or non-primary scripts, not in the current processor/plotter/datacard path
  inspected for this contract.

## 4. Implemented versus used feature matrix

This is the required implemented-versus-used feature matrix. `Used by topeft`
means source inspection found a direct or practical consumer in the current
`topeft` codebase. `Implemented, apparently unused` means the feature exists in
`HistEFT`/`SparseHist` but no current inspected topeft call site was found.

| Feature / method | Implemented where | Used by processor? | Used by plotter? | Used by datacards/yields? | Used by pkl helper? | Replacement priority | Evidence | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `HistEFT.__init__` | `histEFT.py:74-126` | Yes | Indirect | Indirect | Indirect | Required immediately | `analysis_processor.py:268-292`; `analysis_processor_diboson.py:93-110` | Constructor shape is the processor-facing creation API. |
| `HistEFT.fill(eft_coeff=...)` | `histEFT.py:197-249` | Yes | No | No | No | Required immediately | `analysis_processor.py:1900-1924`; `analysis_processor_diboson.py:1297-1334` | Must preserve weight times coefficient storage and scalar categorical labels. |
| `HistEFT.fill(eft_coeff=None)` | `histEFT.py:214-224` | Yes | No | No | No | Required immediately | Non-EFT samples can produce `eft_coeffs_cut=None` before fill at `analysis_processor.py:1909-1911` | Needed for non-EFT samples stored in HistEFT 1D hists. |
| `HistEFT.eval(values)` | `histEFT.py:271-284` | No | Yes | Yes | Yes | Required immediately | `make_cr_and_sr_plots.py:6231-6277`; `datacard_tools.py:1494-1553`; `yield_tools.py:548-582`; `inspect_histeft_pkl.py:177-189` | This is the primary evaluated-yield API. |
| `HistEFT.as_hist(values)` | `histEFT.py:286-305` | No | Yes | Existing tests | No | Required if plotting is not migrated | `make_cr_and_sr_plots.py:6742-6785`; `tests/test_group_bins.py:42-80` | Plotter expects evaluated regular histograms for plotting arrays/edges. |
| `HistEFT.wc_names` | `histEFT.py:133-135` | No | Indirect | Yes | Yes | Required immediately | `datacard_tools.py:94-165,1062-1130,1335-1341`; `inspect_histeft_pkl.py:140-156` | WC order is metadata, validation, and datacard behavior. |
| `HistEFT.dense_axis` | `histEFT.py:169-170` | No | No | Yes | No | Required if datacards/yields are not migrated | `datacard_tools.py:1176-1185` | Datacards use dense-axis extent after channel selection. |
| `HistEFT.make_scaling` | `histEFT.py:321-353` | No | No | Yes | No | Required if datacards/yields are not migrated | `datacard_tools.py:1335-1341` | Used for EFT scaling output. |
| `HistEFT.should_rebin` | `histEFT.py:165-166` | No | No | Yes | No | Required if datacards/yields are not migrated | `datacard_tools.py:832-834` | Currently always false; datacard rebin branch is effectively disabled for HistEFT. |
| `HistEFT.__reduce__` / `_read_from_reduce` | `histEFT.py:307-356` | Indirect | Yes | Yes | Yes | Required for old-pkl compatibility | `run_analysis.py:1760-1765`; `make_cr_and_sr_plots.py:49-110`; `utils.py:399-477` | Required to load existing HistEFT pkls without conversion. |
| `HistEFT.quadratic_term_index` | `histEFT.py:140-163` | No | No | No direct call | No | Nice to have | Source only; tests in `tests/test_histeft_api_contract.py` | Public helper, useful for coefficient inspection; no topeft runtime call found. |
| `HistEFT.index_of_wc` | `histEFT.py:137-138` | No | No | No | No | Probably unused | Source only | Implemented, apparently unused by topeft except through `quadratic_term_index`. |
| `HistEFT.calc_eft_weights` instance method | `histEFT.py:361-388` | No | No | No | No | Probably unused | `HistEFT.eval` calls `efth.calc_eft_weights` at `histEFT.py:281` | Implemented, apparently unused by topeft. |
| `SparseHist.__init__` | `sparseHist.py:18-39` | Yes for 2D | Yes | Indirect | Indirect | Required immediately for 2D SparseHist outputs | `analysis_processor.py:293-342`; `make_cr_and_sr_plots.py:6461-6531` | Required if 2D histograms stay in current format. |
| `SparseHist.fill` | `sparseHist.py:124-139` | Yes for 2D; inherited by HistEFT | No | No | No | Required immediately | `analysis_processor.py:1900-1924` via `HistEFT.fill`; `analysis_processor.py:293-342` for 2D | Fill bookkeeping creates dense blocks per sparse category. |
| `.axes` metadata | `SparseHist.__init__` sets at `sparseHist.py:38` | Indirect | Yes | Yes | Yes | Required immediately | `make_cr_and_sr_plots.py:732-807`; `datacard_tools.py:39-172`; `yield_tools.py:306-345`; `inspect_histeft_pkl.py:234-268` | Labels and bin edges are consumer-visible. |
| `categorical_axes`, `dense_axes`, `categorical_keys` | `sparseHist.py:112-122` | No | Yes | Yes | No | Required if plotting/datacards are not migrated | `make_cr_and_sr_plots.py:6461-6531`; `datacard_tools.py:39-172,976-1060` | Datacards use populated keys directly. |
| `values(flow=...)` | `sparseHist.py:327-350` | No | Yes | Yes | Yes | Required if consumers not migrated | `make_cr_and_sr_plots.py:6231-6531`; `yield_tools.py:816-827`; `inspect_histeft_pkl.py:177-189` | For HistEFT raw values are coefficient arrays, not evaluated yields. |
| `view(flow=..., as_dict=True)` | `sparseHist.py:362-371` | No | Yes | Yes | No | Required immediately | `HistEFT.eval` uses it at `histEFT.py:280`; `make_cr_and_sr_plots.py:1263-1294,6522-6531`; `datacard_tools.py:1062-1130` | Raw coefficient-axis view is datacard-visible for selected-WC discovery. |
| `integrate(name, value)` | `sparseHist.py:373-376` | No | Yes | Yes | Yes | Required immediately | `make_cr_and_sr_plots.py:1310-1321`; `datacard_tools.py:1062-1210`; `yield_tools.py:505-582`; `inspect_histeft_pkl.py:159-174` | Central category selection API. |
| `__getitem__` with mapping and `sum` sentinel | `sparseHist.py:299-325` | No | Yes | Yes | No | Required immediately | `make_cr_and_sr_plots.py:6742-6785`; `datacard_tools.py:1062-1130` | Current code uses `h[{"process": sum}]` and similar category summing. |
| `group(axis_name, groups)` | `sparseHist.py:378-406` | No | Yes | Yes | No | Required if plotting/datacards are not migrated | `make_cr_and_sr_plots.py:5810-5832`; `datacard_tools.py:947-973` | Process grouping depends on this exact role. |
| `remove(axis_name, bins)` | `sparseHist.py:408-428` | No | Yes | Yes | No | Required if plotting/datacards are not migrated | `make_cr_and_sr_plots.py:786-807`; `datacard_tools.py:788-854` | Used to drop data/MC/signal groups and nuisances. |
| `prune(axis, to_keep)` | `sparseHist.py:430-433` | No | No direct primary plotter call | Yes | No | Required if datacards/yields are not migrated | `datacard_tools.py:788-854,1062-1130` | Note one `yield_tools` path may ignore returned value. |
| `scale(factor)` | `sparseHist.py:435-437` | No | Yes | No | No | Required if plotting is not migrated | `make_cr_and_sr_plots.py:3522-3544` | Used for unit-normalized plots. |
| `empty()` | `sparseHist.py:439-443` | No | Yes | Yes | Indirect | Required if consumers not migrated | `make_cr_and_sr_plots.py:1263-1294`; `datacard_tools.py:788-854`; `hist_utils.py:204-211` | Empty filtering and plotting guards use it. |
| `copy`, `__copy__`, `__deepcopy__` | `sparseHist.py:75-83` | No | Indirect | Yes | No | Required if datacards/yields are not migrated | `yield_tools.py:475-482`; binary ops use copy at `sparseHist.py:461-464` | Needed by utility transformations and non-mutating arithmetic. |
| `__iadd__`, `__add__` | `sparseHist.py:485-493` | Indirect via coffea accumulation | Yes via merge | Yes | Yes | Required immediately | `datacard_tools.py:175-302`; `corrections.py:1695`; `corrections_240414.py:1389` | Pkl merge/add behavior is a core compatibility gate. |
| `__imul__`, `__mul__`, scalar arithmetic | `sparseHist.py:501-514` | No | Yes through `scale` | No | No | Required if plotting is not migrated | `scale` delegates to `self *= factor` at `sparseHist.py:435-437` | Other scalar ops are implemented but no direct runtime call was found. |
| `counts`, `reset`, `identity` | `sparseHist.py:352-359,524-529` | No | No | No | Indirect only | Probably unused | No primary topeft call found | Implemented, apparently unused by topeft except internal/legacy accumulator behavior. |
| `replace_axis` | Inherited/unknown | No | No | Yes | No | Unknown | `yield_tools.py:430-463` | Needs runtime validation if channel-label restoration remains in scope. |
| `rebin` | Not implemented by HistEFT current path | No | Legacy scripts | Guarded datacard branch | No | Unknown / can defer for current HistEFT | `datacard_tools.py:832-834`; `histEFT.py:89-91,165-166` | Constructor rejects rebin. Do not promise replacement rebin without consumer migration. |
| `variances` | Not implemented by HistEFT | No | Dense regular hist paths | No | Probed | Can defer for HistEFT replacement if `_sumw2` preserved | `inspect_histeft_pkl.py:192-199`; `_sumw2` fills at `analysis_processor.py:1913-1924` | Current HistEFT uncertainty convention is top-level `_sumw2`, not `variances`. |
| `.sum`, `.project`, `.identifiers` | Not implemented by SparseHist | No current primary processor | Older/non-primary scripts | No | No | Probably unused for current replacement | Search found older validation/training scripts, not current primary flow | Can defer unless those legacy workflows are explicitly supported. |

The replacement does not need to reproduce every implemented method if the
method is not used by topeft and old-pkl compatibility is not required. It does
need to reproduce any feature above marked `Required immediately` before it can
replace current runtime output.

## 5. HistEFT public API contract

### `HistEFT.__init__(*args, wc_names=None, **kwargs)`

Source: `../topcoffea/topcoffea/modules/histEFT.py:74-126`.

- Purpose: create an EFT-aware sparse histogram with one user dense axis and one
  internal dense `quadratic_term` axis.
- Arguments: named categorical axes followed by one physics dense axis;
  `wc_names` list without SM; optional `storage`, `label`, and other `hist.Hist`
  args that survive through `SparseHist`.
- Returns: new `HistEFT`.
- Side effects: constructs WC metadata, coefficient axis, dense-axis metadata,
  and sparse storage dictionary.
- Mutation: constructor only.
- Hidden-axis dependency: yes, creates or accepts `quadratic_term`.
- WC metadata dependency: yes, computes `_quad_count` from WC count.
- Used by topeft: yes, processors instantiate it directly.
- Replacement exactness: Required immediately for processor compatibility.

### `HistEFT.fill(eft_coeff=None, **values)`

Source: `histEFT.py:197-249`.

- Purpose: fill one sparse category and one dense value array with EFT
  coefficient arrays.
- Arguments: scalar categorical values, dense variable keyword, optional
  `weight`, optional `eft_coeff` shaped `(n_events, n_quad_terms)`.
- Return value: implementation does not return `self` despite annotation.
- Side effects: mutates `_dense_hists` by creating/filling the matching sparse
  category block.
- Mutation: mutates self.
- Hidden-axis dependency: yes, repeats dense values by `_quad_count` and fills
  `quadratic_term` indices.
- WC metadata dependency: yes, validates coefficient length against `_quad_count`
  indirectly through array shape and dense fill.
- Used by topeft: yes, processor and diboson processor fill all 1D histograms.
- Replacement exactness: Required immediately.

Current fill semantics:

1. Determine `n_events` from the user dense axis value.
2. If `eft_coeff is None`, create SM-only coefficients `[1, 0, ..., 0]` per
   event.
3. Repeat dense values once per quadratic term.
4. Repeat coefficient-term indices once per event.
5. Pop `weight`; if present, repeat weights and multiply them into flattened
   EFT coefficients.
6. Delegate to `SparseHist.fill` with `quadratic_term=<indices>` and
   `weight=<weighted coefficients>`.

### `HistEFT.eval(values)`

Source: `histEFT.py:271-284`.

- Purpose: evaluate raw stored coefficient arrays at a WC point.
- Arguments: `None`, mapping from WC names to numeric values, or array-like
  values in `wc_names` order.
- Return value: dictionary from sparse categorical key to NumPy array over the
  user dense axis, including flow bins.
- Side effects: none expected.
- Mutation: should not mutate source histogram.
- Hidden-axis dependency: yes, calls `view(flow=True, as_dict=True)` and removes
  coefficient-axis flow slots with `[..., 1:-1]`.
- WC metadata dependency: yes, mapping keys are checked against `wc_names`.
- Used by topeft: yes, plotter, datacards, yields, and pkl helper.
- Replacement exactness: Required immediately.

Mapping behavior:

- `eval({})` and `eval(None)` evaluate the SM point.
- Missing WC names are set to zero.
- Unknown WC names raise `LookupError`.
- Array-like input is accepted as-is and interpreted in `wc_names` order; length
  checks are not explicit in `_wc_for_eval`.

### `HistEFT.as_hist(values)`

Source: `histEFT.py:286-305`.

- Purpose: evaluate at a WC point and materialize a regular `hist.Hist` with
  categorical axes plus the user dense axis.
- Return value: regular `hist.Hist`, filled by sparse key.
- Side effects: none expected on source.
- Mutation: returns a new object.
- Hidden-axis dependency: removes hidden coefficient axis from materialized
  output.
- WC metadata dependency: uses `eval`.
- Used by topeft: yes, plotter and existing tests.
- Replacement exactness: Required if plotting is not migrated.

### `HistEFT.make_scaling(flow="show", wc_list=None)`

Source: `histEFT.py:321-353`.

- Purpose: expose EFT scaling coefficients for datacard-style output.
- Arguments: flow policy `"show"` or `"sum"`; optional target `wc_list`.
- Return value: raw coefficient array transformed to scaling coefficients.
- Side effects: none expected.
- Mutation: returns arrays, not self.
- Hidden-axis dependency: reads raw coefficient axis through `values(flow=True)`
  and removes coefficient flow columns.
- WC metadata dependency: optional remapping through `efth.remap_coeffs`.
- Used by topeft: yes, datacard code.
- Replacement exactness: Required if datacards/yields are not migrated.

### `HistEFT.__reduce__()` and `_read_from_reduce`

Sources: `histEFT.py:307-319`, `histEFT.py:355-356`,
`sparseHist.py:477-483`.

- Purpose: pickle/unpickle HistEFT objects.
- Return value: reduce tuple with reconstruction function, categorical axes,
  dense axis, init args, WC args, and `_dense_hists`.
- Side effects: none during reduce; reconstructing unpickle populates dense
  hist dictionary.
- Hidden-axis dependency: hidden axis is recreated from WC metadata unless an
  explicit `quadratic_term` axis was pickled in the constructor args.
- WC metadata dependency: `wc_names` must survive.
- Used by topeft: yes through pkl writing/loading.
- Replacement exactness: Required for old-pkl compatibility. For new runs only,
  an explicit converter or new serialization contract could replace it.

### WC helpers and current usage

- `wc_names` is Required immediately because merge validation, datacards, and pkl
  inspection use it.
- `quadratic_term_index` is a public coefficient-order helper, but no current
  primary topeft runtime call was found. It is Nice to have and useful for parity
  tests.
- `index_of_wc` is Implemented, apparently unused by topeft except internally by
  `quadratic_term_index`.
- `calc_eft_weights` as an instance method is Implemented, apparently unused by
  topeft; `HistEFT.eval` uses `eft_helper.calc_eft_weights`.

## 6. SparseHist inherited behavior contract

A replacement for HistEFT must either inherit/reproduce these `SparseHist`
behaviors or migrate every consumer that relies on them:

| Behavior | Required behavior | Mutation/return | Evidence |
| --- | --- | --- | --- |
| sparse categorical storage | Only populated category tuples need dense blocks | Mutates on fill | `sparseHist.py:124-139`; processors fill scalar `process/channel/systematic/appl` |
| `.axes` | Ordered categorical axes plus dense axes, name-addressable | Metadata | `sparseHist.py:38`; consumers in plotter/datacards/yields/pkl helper |
| `view(flow=True, as_dict=True)` | Dict keyed by sparse category namedtuple with dense raw views | Non-mutating | `HistEFT.eval`; `datacard_tools.py:1062-1130`; `make_cr_and_sr_plots.py:6522-6531` |
| `values(flow=...)` | Return block values over populated sparse keys | Non-mutating | `make_cr_and_sr_plots.py:6231-6531`; `inspect_histeft_pkl.py:177-189` |
| `integrate` and mapping `__getitem__` | Select labels, lists, slices, and `sum` sentinel by axis | Return selected object | `datacard_tools.py:1062-1210`; `make_cr_and_sr_plots.py:1310-1321` |
| `group` | Merge categorical labels into new group labels | Return new hist | `make_cr_and_sr_plots.py:5810-5832`; `datacard_tools.py:947-973` |
| `remove` / `prune` | Drop or keep categorical labels | Return new hist | `make_cr_and_sr_plots.py:786-807`; `datacard_tools.py:788-854` |
| `scale` | Scalar in-place multiplication, return self | Mutates self | `make_cr_and_sr_plots.py:3522-3544` |
| `copy`, `+`, `+=` | Compatible non-mutating and in-place merging | Both forms | `datacard_tools.py:175-302`; `sparseHist.py:461-493` |
| `empty` | All dense views zero means empty | Non-mutating | `hist_utils.py:204-211`; plotter/datacard guards |
| pickle reduce | Reconstruct axes/init args/dense blocks | Serialization | `sparseHist.py:466-483`; plotter fast-patch at `make_cr_and_sr_plots.py:49-110` |

Unknown inherited behavior:

- `replace_axis` is used in one yield-tool channel-label restoration path, but
  is not implemented in `SparseHist`. A replacement should not promise this path
  until a runtime fixture proves the current behavior.

## 7. EFT semantics contract

### Mathematical object

For each populated sparse category key and each physics dense bin, `HistEFT`
stores the coefficients of a quadratic polynomial in Wilson coefficients.

Given WC names in order:

```text
wc_names = [c_1, c_2, ..., c_n]
```

define the factor vector:

```text
w_0 = 1
w_i = c_i for i = 1..n
```

The stored bin content is an ordered coefficient vector:

```text
a = [a_00, a_10, a_11, a_20, a_21, a_22, ..., a_n0, ..., a_nn]
```

and evaluation is:

```text
yield_bin(c) = sum_{i=0..n} sum_{j=0..i} a_ij * w_i * w_j
```

Equivalently:

```text
yield_bin(c) = a_00
             + sum_i a_i0 * c_i
             + sum_i a_ii * c_i * c_i
             + sum_{i>j>0} a_ij * c_i * c_j
```

Source: `eft_helper.calc_eft_weights` loops over `i` and `j <= i` at
`../topcoffea/topcoffea/modules/eft_helper.py:10-39`. The number of stored
terms is:

```text
n_quad_terms(n_wc) = (n_wc + 2) * (n_wc + 1) / 2
```

Source: `eft_helper.py:42-46`.

For `wc_names = ["ctG", "cpt"]`, the order is:

```text
0: SM*SM
1: ctG*SM
2: ctG*ctG
3: cpt*SM
4: cpt*ctG
5: cpt*cpt
```

This order is also checked by `tests/test_histeft_api_contract.py`.

### Coefficient arrays and fill

Event-level coefficient arrays come from `events["EFTfitCoefficients"]` when the
branch exists. The main processor converts to NumPy and remaps sample WC order to
the processor WC order at `analysis/topeft_run2/analysis_processor.py:620-626`.
The diboson processor uses the same pattern at
`analysis/topeft_run2/analysis_processor_diboson.py:297-304`.

`eft_helper.remap_coeffs` prepends `SM`, maps lower-triangle terms from a current
WC list into a target WC list, drops omitted WCs, and zero-fills missing target
WCs. Source: `../topcoffea/topcoffea/modules/eft_helper.py:208-266`.

During fill:

```text
stored_coeff[event, term] = event_weight[event] * eft_coeff[event, term]
```

The event weight is multiplied into coefficients during `HistEFT.fill`, not
during later evaluation. Source: `histEFT.py:230-245`.

If `eft_coeff` is missing or `None`, the current implementation stores SM-only
coefficients:

```text
eft_coeff[event] = [1, 0, ..., 0]
```

Source: `histEFT.py:214-224`.

### Evaluation behavior

- `eval({})` and `eval(None)` evaluate the SM point.
- A mapping with one WC evaluates that WC while unspecified WCs remain zero.
- A mapping with two WCs includes linear, pure quadratic, and cross terms.
- Unknown WC names raise `LookupError`.
- Array-like values are interpreted in current `wc_names` order.
- Repeated evaluation is expected not to mutate stored coefficients; this is part
  of the parity-test specification.

### Systematics and sumw2

Systematic variations are categorical labels on the `systematic` sparse axis,
not extra EFT polynomial dimensions. The processors fill nominal and variations
under the same `HistEFT` object with different `systematic` labels. Source:
`analysis_processor.py:642-719,1744-1760,1900-1924`.

`_sumw2` companion histograms are separate top-level scalar `SparseHist`
objects with dense axis names suffixed by `_sumw2`. Their allocation is
policy-selected by concrete dataset/process/family targets. The main processor
fills a selected companion only for the nominal producer path. For an EFT
event, the EFT factor is evaluated at WC=0 and folded into the event weight
before squaring; the companion fill has no `eft_coeff` argument. Thus the
companion is a complete-event SM/WC=0 second moment, not an EFT-polynomial
variance object.

The codebase contains quartic/squared-weight helpers, but nonzero-WC quartic
sumw2 is outside the maintained storage and consumer contract.

## 8. Pickle and pkl compatibility contract

### Top-level pkl structure

Schema-v2 processor output is a gzip-compressed pickle containing a dictionary
whose 1D nominal content is split by source type:

```text
{
  "<family>__scalar_nominal": SparseHist,  # when scalar content exists
  "<family>__eft_nominal": HistEFT,        # when EFT content exists
  "<family>_sumw2": SparseHist,            # when policy-selected
  ...
}
```

The original unsplit 1D `<family>` key is forbidden in schema-v2 producer
output. A 2D family remains a scalar `SparseHist` at `<family>` and can have
an optional `<family>_sumw2` companion. See
[`histeft_pkl_tutorial.md`](histeft_pkl_tutorial.md) for artifact inspection.

Evidence:

- `analysis/topeft_run2/run_analysis.py:1760-1765` writes `output` with
  `gzip.open(..., "wb")` and `cloudpickle.dump`.
- `../topcoffea/topcoffea/modules/utils.py:399-405` writes pkl.gz payloads with
  gzip and cloudpickle.
- `../topcoffea/topcoffea/modules/hist_utils.py:274-293` materializes pkl files
  as dictionaries when requested.
- `topeft/modules/datacard_tools.py:175-302` requires base histogram keys and
  matching `_sumw2` companions by default.
- `analysis/topeft_run2/make_cr_and_sr_plots.py:5347-5351,7802-7816` discovers
  `_sumw2` companions and loads/merges pkl payloads.

### Hist object graph

Each current `HistEFT` pkl must preserve:

- categorical axes and labels;
- user dense axis name, label, type, and binning;
- WC names and order;
- hidden coefficient-axis storage after reconstruction;
- `_dense_hists` dense blocks keyed by sparse categorical namedtuples;
- class import path `topcoffea.modules.histEFT.HistEFT` for old pickle loading.

`HistEFT.__reduce__` stores categorical axes, the user dense axis, init args,
WC args, and `_dense_hists` at `histEFT.py:307-319`. `SparseHist.__reduce__`
does the analogous storage at `sparseHist.py:466-475`.

### Old-pkl compatibility hooks

- `analysis/topeft_run2/make_cr_and_sr_plots.py:49-110` monkey-patches
  `SparseHist._read_from_reduce` for faster pkl reconstruction.
- `analysis/topeft_run2/inspect_histeft_pkl.py:20-28` attempts to call
  `ensure_histEFT_py39_compat`.
- `topeft/modules/yield_tools.py:7-14` calls the same compatibility helper before
  importing HistEFT/SparseHist.
- `../topcoffea/topcoffea/modules/compat.py:13-51` provides compatibility shims
  for old import/type expectations.

### Replacement pkl requirements

Required for new runs only:

- New output must be loadable by the current plotter/datacard/yield path unless
  those consumers are migrated in the same change.
- Top-level dictionary keys and `_sumw2` naming must remain compatible unless
  consumers are migrated.
- Axis names and labels must remain discoverable.
- WC metadata must remain discoverable.
- Histograms must be mergeable after pkl load.

Required for old-pkl compatibility:

- Old module/class import paths must remain importable, or an explicit converter
  must be supplied.
- Old `__reduce__` state shape must be readable.
- Old `_dense_hists` block layout and sparse key namedtuple behavior must be
  translated correctly.
- Old WC metadata and hidden coefficient-axis state must reconstruct enough to
  support `eval({})`, nonzero WC eval, grouping, integration, and pkl merge.

Optional / legacy:

- Preserving the plotter's monkey-patched fast loader is optional if replacement
  pkl loading is already fast and old pkls are handled by a converter.
- Preserving `identity()` is only required for old accumulator compatibility if a
  still-supported workflow uses it.

## 9. Consumer contract from current topeft usage

| Consumer | File | HistEFT/SparseHist feature used | Required behavior | Evidence | Replacement priority |
| --- | --- | --- | --- | --- | --- |
| Main processor | `analysis/topeft_run2/analysis_processor.py` | `HistEFT.__init__`, `SparseHist.__init__`, `HistEFT.fill`, `SparseHist.fill` | Construct 1D EFT hists and 2D non-EFT sparse hists; fill scalar categories, dense arrays, weights, optional `eft_coeff` | `analysis_processor.py:245-343,620-631,1900-1924` | Required immediately |
| Diboson processor | `analysis/topeft_run2/analysis_processor_diboson.py` | `HistEFT.__init__`, `HistEFT.fill` | Construct/fill HistEFT and `_sumw2` for all axes | `analysis_processor_diboson.py:72-110,297-304,1297-1334` | Required immediately |
| Runner/output writer | `analysis/topeft_run2/run_analysis.py` | pickleability, dict of histograms | Output must serialize with gzip/cloudpickle and retain class state | `run_analysis.py:1539-1595,1760-1765` | Required immediately |
| Plotter pkl loader | `analysis/topeft_run2/make_cr_and_sr_plots.py` | pkl compatibility, `_read_from_reduce`, merge helper | Load old/new pkl dicts and `_sumw2` companions | `make_cr_and_sr_plots.py:49-110,7802-7816` | Required for old-pkl compatibility |
| Plotter axis/filter helpers | `make_cr_and_sr_plots.py` | `.axes`, `.remove`, `.group`, `.integrate`, `__getitem__`, `.empty` | Axis label discovery, sample filtering, grouping, nominal integration | `make_cr_and_sr_plots.py:732-807,1263-1321,5810-5832` | Required if plotting is not migrated |
| Plotter value extraction | `make_cr_and_sr_plots.py` | `HistEFT.eval({})`, `as_hist({})`, `SparseHist.view`, `values` | SM values with flow handling; 2D sparse view extraction | `make_cr_and_sr_plots.py:6231-6277,6461-6531,6742-6785` | Required if plotting is not migrated |
| Datacard merge | `topeft/modules/datacard_tools.py` | `axes`, `wc_names`, `+=`, `+`, `_sumw2` pairing | Validate axes/WC metadata and merge multiple pkl payloads | `datacard_tools.py:94-302` | Required immediately if datacards consume new output |
| Datacard read/group | `datacard_tools.py` | `.empty`, `.remove`, `.prune`, `.group`, `should_rebin` | Drop nuisances/processes and group process labels | `datacard_tools.py:788-973` | Required if datacards are not migrated |
| Datacard EFT logic | `datacard_tools.py` | `wc_names`, `view`, `integrate`, `make_scaling`, `eval`, `dense_axis` | Select WCs, scaling extraction, SM/linear/quadratic decomposition | `datacard_tools.py:1062-1553` | Required if datacards are not migrated |
| Yield tools | `topeft/modules/yield_tools.py` | `.axes`, `.copy`, `.integrate`, `eval`, `values`, possibly `replace_axis` | Label discovery, integration, yield sums, channel-label restoration | `yield_tools.py:306-345,430-582,816-827` | Required if yields are not migrated; `replace_axis` Unknown |
| Pkl inspector | `analysis/topeft_run2/inspect_histeft_pkl.py` | `.axes`, `wc_names`/`_wc_names`, `.integrate`, `.eval`, `.values`, `.variances` probe | Read-only summary of top-level keys, axes, labels, and nominal yield | `inspect_histeft_pkl.py:20-345` | Nice to have for replacement introspection; `variances` can defer if `_sumw2` preserved |
| Pkl utilities | `topcoffea` utils/hist_utils | gzip/cloudpickle dict load/write; empty filtering | Load materialized pkl dictionaries and optionally stream entries | `utils.py:399-477`; `hist_utils.py:36-293` | Required for old-pkl/new-output compatibility unless serialization is migrated |

## 10. Replacement requirements

### Required immediately

- Constructor compatibility for processor-created 1D HistEFT objects:
  evidence `analysis_processor.py:268-292`,
  `analysis_processor_diboson.py:93-110`.
- `fill(..., eft_coeff=..., weight=..., process=..., channel=...,
  systematic=..., appl=..., <dense axis>=...)`: evidence
  `analysis_processor.py:1900-1924`.
- SM-only fill when `eft_coeff is None`: evidence `histEFT.py:214-224` and
  non-EFT processor path can pass `None`.
- `eval({})` and named WC `eval` with current polynomial semantics: evidence
  `histEFT.py:271-284`, `datacard_tools.py:1494-1553`,
  `make_cr_and_sr_plots.py:6231-6277`.
- Axis metadata: `process`, `channel`, `systematic`, `appl`, dense axis names,
  labels, and edges: evidence `datacard_tools.py:94-165`,
  `yield_tools.py:306-345`.
- `wc_names` metadata and ordering: evidence `datacard_tools.py:94-165,1062-1130`.
- Addition and in-place addition for pkl merge: evidence
  `datacard_tools.py:175-302`.
- Pickleability for new output and compatibility with top-level dict plus
  `_sumw2` companions: evidence `run_analysis.py:1760-1765`,
  `datacard_tools.py:175-302`.
- `SparseHist` behavior for current 2D non-EFT outputs if those outputs remain
  in scope: evidence `analysis_processor.py:293-342` and
  `make_cr_and_sr_plots.py:6461-6531`.

### Required for old-pkl compatibility

- Read old `HistEFT.__reduce__` state: evidence `histEFT.py:307-319`.
- Read old `SparseHist.__reduce__` state: evidence `sparseHist.py:466-483`.
- Preserve or translate `_dense_hists` storage and sparse key names.
- Preserve old module/class import paths or provide a converter.
- Support the plotter/yield compatibility helpers or remove the need for them
  with a documented migration.

### Required if plotting is not migrated

- `as_hist({})`, `.axes`, `.integrate`, `__getitem__` with `sum`, `.group`,
  `.remove`, `.empty`, `.scale`, `.values`, and `.view`.
- Flow-bin behavior compatible with `_values_with_flow_or_overflow` and
  `_eval_without_underflow`.
- Existing process/channel/systematic/appl labels and group maps.

### Required if datacards/yields are not migrated

- `make_scaling`, `dense_axis`, raw coefficient `view(flow=True, as_dict=True)`,
  `categorical_keys`, `.prune`, `.group`, `.remove`, `should_rebin`.
- `replace_axis` remains Unknown and needs a fixture if
  `restore_split_channel_labels` remains supported.

### Nice to have

- `quadratic_term_index` for public coefficient-order inspection.
- `index_of_wc` if `quadratic_term_index` is preserved.
- Pkl inspector-friendly fallback `values` and `variances` probes, as long as
  the official variance contract is explicit.

### Can defer

- Native `variances` for HistEFT if `_sumw2` companions remain the source of
  squared-weight uncertainty.
- Native HistEFT `rebin` if current `should_rebin()` remains false and datacard
  paths do not require HistEFT rebinning.
- Streaming pkl optimizations if new pkl loading is small/fast in the first
  prototype and old pkl compatibility is out of scope.

### Probably unused

- `HistEFT.index_of_wc` direct runtime calls.
- `HistEFT.calc_eft_weights` instance method.
- `SparseHist.counts`, `reset`, and `identity` direct runtime calls.
- `.sum`, `.project`, and old `.identifiers()` for the current primary
  processor/plotter/datacard flow.

### Unknown

- `replace_axis` with SparseHist/HistEFT in `yield_tools.restore_split_channel_labels`.
- Whether old non-primary scripts under `analysis/topeft_run2/make_cr_and_sr_plots_v*`,
  `analysis/extreme_events_study`, and validation/training directories are still
  supported replacement consumers.
- Whether quartic `calc_w2_coeffs` should become the official sumw2 EFT
  convention in a future physics review. Current processors compute but do not
  use `eft_w2_coeffs`.

## 11. HistEFT parity-test suite specification

Any future histogram backend must pass these tests before replacing HistEFT.
Tests may use current `HistEFT` as the reference implementation.

### Constructor parity

- Same sparse axis names, labels, growth behavior, and order.
- Same dense physics axis metadata: name, label, type, bin edges, flow policy.
- Same WC list and order.
- Same quadratic-term count from `n_quad_terms`.
- Same hidden `quadratic_term` behavior where old consumers inspect raw views.
- Same failure modes for unsupported storage, unnamed axes, invalid dense-axis
  position/type, reserved axis names, and `rebin=True` if replacement is claiming
  drop-in compatibility.

### Fill parity

- One bin, no WC list, `eft_coeff=None` gives SM-only coefficient storage.
- One WC coefficient array: SM, linear, quadratic terms.
- Two WC coefficient array: SM, both linears, both pure quadratics, cross term.
- Multiple events in different dense bins.
- Weighted fills.
- Negative weights.
- Categorical labels for `process/channel/systematic/appl`.
- Multiple systematic labels under the same object.
- Fill without required `eft_coeff` for non-EFT samples.
- Shape failures for incompatible coefficient arrays and dense value arrays.

### Evaluation parity

- SM point `eval({})`.
- `eval(None)`.
- Single nonzero WC.
- Two nonzero WCs, including cross terms.
- Zero point with explicit zeros.
- Large coefficient point.
- Missing WC handling: unspecified WCs are zero.
- Unknown WC handling: `LookupError`.
- Array-like WC input in `wc_names` order.
- Repeated eval does not mutate source values or metadata.

### Histogram operation parity

- `values(flow=False)` and `values(flow=True)` raw coefficient views.
- `view(flow=True, as_dict=True)` shape and sparse keys.
- `integrate` by scalar label, list of labels, and `None`.
- `__getitem__` with `{axis: sum}`.
- `group`, including dropping or preserving unspecified labels through helper
  logic.
- `remove` and `prune`.
- `scale` mutation and return value.
- `copy`, `deepcopy`, `empty`, `identity` if old coffea compatibility is claimed.
- Addition and in-place addition for compatible histograms.
- Multiplication by scalar through `scale`.
- Rebin only if the replacement claims support or datacard migration requires it.

### Pickle parity

- Pickle/unpickle current HistEFT and replacement object.
- Evaluate after unpickle at SM and nonzero WC points.
- Merge/add after unpickle.
- Compare WC metadata after unpickle.
- Compare axes and categorical labels after unpickle.
- Preserve the schema-v2 sibling layout and policy-selected scalar companion
  contract for new producer output.
- Load one representative old pkl if a small safe fixture exists, or define a
  converter test that proves old content can be translated.

### Processor/plotter compatibility parity

- Mock a processor-like fill with `process/channel/systematic/appl`, dense axis,
  event weights, and `eft_coeff`.
- Mock non-EFT fill with `eft_coeff=None`.
- Build a schema-v2 top-level dict with 1D scalar/EFT siblings and any selected
  scalar `<hist>_sumw2` companion.
- Run a minimal plotter-like flow: integrate nominal, group processes, sum
  process axis, evaluate `eval({})`, materialize `as_hist({})`, and read values.
- Preserve `process/channel/systematic/appl` labels exactly.
- Validate datacard-like merge for two disjoint process payloads.

### Numerical tolerance policy

- Use exact equality for axis names, category labels, WC names, coefficient term
  order, top-level pkl keys, and error types.
- Use `np.testing.assert_allclose(..., atol=1e-12, rtol=0)` for deterministic
  small coefficients and simple hand-computed fills.
- Use relative tolerance `rtol=1e-10` plus `atol=1e-12` for accumulated weighted
  sums where operation order can differ.
- Use looser tolerances only for real-input workflows with documented floating
  point differences; do not hide label or coefficient-order mismatches with
  numeric tolerances.

## 12. Synthetic fixture design

### Fixture A: one dense axis, one WC

- Purpose: constructor, SM-only fill, one-WC linear/quadratic evaluation.
- Axes: `process`, `channel`, `systematic`, `appl`, dense `x` with two bins.
- WC names: `["ctG"]`.
- Event inputs: one event in first bin with coefficients `[2, 3, 5]`.
- Expected SM eval: first physical bin `2 * weight`.
- Expected nonzero eval at `ctG=2`: `(2 + 3*2 + 5*4) * weight`.
- Metadata: hidden coefficient term count `3`; axis labels preserved.
- Tests: constructor, fill, eval, pickle.

### Fixture B: one dense axis, two WCs

- Purpose: lower-triangle coefficient order and cross-term behavior.
- Axes: same as Fixture A.
- WC names: `["ctG", "cpt"]`.
- Event inputs: coefficients `[1, 2, 3, 5, 7, 11]`, weight `2`.
- Expected SM eval: `2`.
- Expected nonzero eval at `ctG=1`, `cpt=2`: `148`.
- Metadata: term order
  `SM*SM, ctG*SM, ctG*ctG, cpt*SM, cpt*ctG, cpt*cpt`.
- Tests: fill parity, evaluation parity, coefficient-order parity.

### Fixture C: realistic topeft-like sparse axes

- Purpose: processor/plotter sparse-axis contract.
- Axes: `process=["ttH", "ttlnu"]`, `channel=["2lss_p"]`,
  `systematic=["nominal", "JESUp"]`, `appl=["isSR"]`, dense `njets`.
- WC names: `["ctG", "cpt"]`.
- Event inputs: tiny deterministic fills across two processes and two
  systematics.
- Expected SM eval: hand-computed per process and systematic.
- Expected nonzero-WC eval: same sparse keys, changed values.
- Metadata: labels remain exactly as filled.
- Tests: integrate, group, remove, `__getitem__` sum, plotter-like value flow.

### Fixture D: `_sumw2` companion behavior

- Purpose: schema-v2 scalar second-moment companion contract.
- Axes: categorical axes match the selected base source; its dense axis is
  `njets_sumw2`.
- Event inputs: scalar base contribution `w`; EFT event contribution evaluated
  at WC=0 and folded into the scalar event weight before it is squared.
- Expected content: a scalar `SparseHist` second moment, not an EFT polynomial
  that can be evaluated at a nonzero WC.
- Metadata: its top-level key is `njets_sumw2`; schema-v2 1D nominal sources
  are `njets__scalar_nominal` and/or `njets__eft_nominal`.
- Tests: pkl payload, schema/coverage validation, datacard merge validation,
  and plotter uncertainty input.

### Fixture E: pickle round-trip fixture

- Purpose: serialization and old/new pkl boundary.
- Axes: use Fixture B or C.
- WC names: two WCs.
- Event inputs: at least one populated sparse key.
- Expected SM and nonzero eval: identical before and after pickle.
- Expected metadata: class, axes, labels, `wc_names`, and dense axis survive.
- Tests: pickle/unpickle, add after unpickle, old-pkl converter if available.

## 13. Optional tests added or proposed

These tests use current `HistEFT` as the reference behavior and do not require
production files or large pkls. They cover:

- quadratic-term order for two WCs;
- weighted fill and evaluation at SM and a two-WC point;
- SM-only fill when `eft_coeff` is omitted;
- unknown-WC failure;
- integrate/group/copy/add/as_hist consumer operations;
- pickle round-trip and `_sumw2` companion shape.

Proposed future tests before any replacement:

- a datacard-like minimal decomposition fixture using `make_scaling`;
- a plotter-like fixture that runs the actual value-extraction helper on the
  replacement;
- an old-pkl fixture or converter test once a small representative pkl can be
  safely stored.

## 14. Open questions

- Does `yield_tools.restore_split_channel_labels` need to support HistEFT/
  SparseHist long term, and does its `replace_axis` call work on current objects
  in the relevant runtime path?
- Are old `make_cr_and_sr_plots_v*`, validation, training, and extreme-event
  scripts in scope for the replacement, or only the current processor/plotter/
  datacard/yield path?
- Whether a future, separately reviewed nonzero-WC quartic uncertainty contract
  is needed; it is not part of the maintained contract described here.
- Is old pkl compatibility required inside the replacement class, or can it be
  provided by a one-time converter plus clear migration boundary?
- Should `as_hist` output include flow bins exactly as current plotter expects,
  or should plotting be migrated to an explicit flow policy at the same time?

## 15. Selective sumw2 schema and consumer contract

### Source components and selected companions

Schema version 2 uses `split_sibling_v1` for each 1D family. Scalar nominal
content and EFT nominal content are separate, non-overlapping process sources:
`<family>__scalar_nominal` is an exact `SparseHist`, while
`<family>__eft_nominal` is an exact `HistEFT`. Either sibling can be absent
when that content class is empty. The original unsplit 1D `<family>` producer
key is not valid schema-v2 output.

`<family>_sumw2`, when selected, is a scalar `SparseHist` holding a selected SM
second moment. An EFT contribution enters it as the complete event evaluated at
WC=0 before squaring; it is not a quadratic or quartic EFT variance payload.
Nonzero-WC quartic sumw2 is intentionally unsupported. The selection policy
resolves rules over dataset, process, and family and records the resolved
provenance; a selected required target without a companion is invalid.

2D families remain scalar `SparseHist` objects under the base `<family>` key
and may carry an optional `<family>_sumw2` companion. They never use
scalar/EFT split siblings.

### Serialization, merge, and consumer-local views

Serialization retains schema/provenance and validates known families, sibling
types, non-overlapping process labels, companion axes, and selected-process
coverage. Merge validation happens before consumers transform the mapping. The
uniform/scalar dictionaries used by plotting or card making are bounded
consumer-local views, not an alternate producer format.

Legacy uniform pkls may be read through the established compatibility path, but
that does not relax schema-v2 producer requirements. Deprecated `no_sumw2` and
`do_errors` flags are migration mappings, not a promise of broad legacy
production behavior. This contract adds no compatibility shim.

### Application-axis and merge-and-collision boundaries

The producer retains the `appl` axis. Data-driven nonprompt processing may use
AR content transiently at WC=0 while constructing scalar output. At the
DatacardMaker boundary, only the exact metadata-defined final SR application
label is selected; CR/AR card production, label guessing, and fallback are not
implemented.

Collision handling is consumer-specific. A supported merge validates schema and
provenance, permits only the consumer's documented disjoint/allowed source
merge behavior, and reports exact duplicate or incompatible process content.
Remediate a collision by removing the duplicate input, separating incompatible
productions, or using the supported consumer workflow with explicitly
documented collision handling. Do not rename inputs, suppress the diagnostic,
or add an adapter merely to force an unsupported merge.

## 16. Data-driven applicability and transformed artifacts

This section is the normative contract for the maintained nonprompt and charge-
flip (flips) transformations. It complements the operational instructions in
the [Run-2 README](../analysis/topeft_run2/README.md#applicability-aware-data-driven-outputs-and-recovery)
and the practical [sidecar inspection tutorial](histeft_pkl_tutorial.md#18-inspecting-data-driven-applicability).

### Enabled, applicable, and generated

The source-wide data-driven policy and each histogram family answer different
questions:

- **Enabled** means that the source-wide `requested_data_driven_products`
  policy requests a product. It is a policy decision shared by the artifact;
  it does not assert that every family contains the inputs for that product.
- **Applicable** means that one family’s authoritative `appl` axis contains a
  maintained application region consumed by that product’s transformer. It is
  family-specific and is derived from actual source histogram content.
- **Generated** means that the transformed nominal or `_sumw2` payload contains
  the certified output process labels for an enabled-and-applicable product.

Thus, enabled does not imply applicable, applicable is family-specific, and
generated labels are required exactly when the product is both enabled and
applicable. A family with no maintained data-driven application region keeps
its ordinary source content and generates neither product.

The current application-region policy is centralized in
`histogram_artifact.py`:

| Product | Maintained application regions |
| --- | --- |
| nonprompt | `isAR_1l`, `isAR_2lSS`, `isAR_2lOS`, `isAR_3l` |
| flips | `isAR_2lSS_OS` |

This is an explicit mapping. A string merely beginning with `isAR_` is not
automatically a nonprompt region. Unknown application labels are not silently
reclassified; they are not a supported runtime policy override.

For the established 1-lepton/1-tau control-region shape, the physical output
pattern is nonprompt-only when the source contains `isAR_1l` alongside the
`isSR_1l` content, flips-only when `isAR_2lSS_OS` is present, and neither when
there is no maintained data-driven AR. The broader mapping above is the
authoritative rule for other maintained categories.

### Version-3 transformed sidecars

The source processor sidecar remains a schema-v2 `split_sibling_v1` artifact.
The transformed sidecar adds `transformation_contract` with
`contract_version: 3`. Its top-level normalized shape is:

```json
{
  "contract_version": 3,
  "artifact_kind": "nonprompt_output",
  "eft_prompt_projection": {
    "mode": "sm_point",
    "required_processes": [],
    "generated_nonprompt_eft_dependence": false
  },
  "families": {
    "njets": {
      "source_scalar_processes": [],
      "source_eft_processes": [],
      "retained_scalar_processes": [],
      "retained_eft_processes": [],
      "generated_nonprompt_processes": ["nonpromptUL18"],
      "generated_flips_processes": ["flipsUL18"],
      "consumed_source_processes": [],
      "source_application_regions": [
        "isAR_2lSS_OS",
        "isAR_3l",
        "isSR_3l"
      ],
      "applicable_products": {
        "flips": true,
        "nonprompt": true
      }
    }
  }
}
```

The excerpt uses the exact normalized field names and shape emitted by the
maintained producer; a real sidecar contains the complete runtime family order
and complete process lists. `source_application_regions` is producer-derived
evidence from the scalar source sibling. `applicable_products` is recomputed
from that evidence and validated, not accepted as an independent caller
choice. Generated process sets must agree with the enabled-and-applicable
products. `required_sumw2_processes` is independently derived from those roles,
so generated companions are mandatory only for applicable products. In
particular, a nonprompt-only family must not acquire flips companions merely
because the source-wide flips product is enabled.

The maintained reader rejects tampering when applicable booleans contradict
the source regions, generated labels disagree with the certified resolved
product contract, or serialized required companions disagree with the
independently derived requirement. The writer also requires the transformation
context generated by `DataDrivenProducer`; caller-authored replacement
contexts/contracts are rejected. Editing a sidecar by hand cannot redefine the
contract.

### Merge and version-2 compatibility

Version-3 merges union each family’s authoritative
`source_application_regions` deterministically, recompute
`applicable_products`, and then revalidate generated roles and required
sumw2 companions. If merged evidence makes a product applicable, its certified
generated labels and companions remain mandatory. The merged contract must
remain internally consistent with the resolved source-wide data-driven policy.

Version-2 transformed contracts contain the prior maintained role fields but
not the version-3 applicability evidence. They remain readable and mergeable
under their established semantics. Version-2 and version-3 transformed
contracts are not silently mixed or reinterpreted: a merge containing both
versions is rejected. Version 3 changes transformed-artifact provenance and
validation; it does not rewrite the source processor contract or old artifacts.

### Changing the data-driven applicability contract

Use this procedure when the policy itself changes. A category that uses only
already recognized AR labels is not an applicability-contract semantic change;
adding a new AR label, changing an AR’s meaning, or adding a product is.

1. Define the physics meaning of the region and its owning data-driven product.
2. Update the centralized checked-in applicability policy/helper.
3. Update `DataDrivenProducer` transformation logic when the new meaning
   requires it.
4. Update normalization, generated-process validation, and tamper checks.
5. Update exhaustive `ch_lst.json` registry coverage and union tests.
6. Add old/new read and merge compatibility tests.
7. Bump the transformation-contract version when semantics changed.
8. Update this API contract, the [pkl tutorial](histeft_pkl_tutorial.md), and
   the [Run-2 operational README](../analysis/topeft_run2/README.md).
9. Validate at least one representative synthetic or real artifact.
10. Preserve truthful provenance and never fabricate empty products.

The cases are:

- **Existing recognized AR only:** adding or modifying a category that uses
  already recognized labels normally needs no applicability-contract version
  bump.
- **New AR mapped to an existing product:** applicability semantics change;
  introduce the next transformation-contract version and compatibility tests.
- **Changed AR meaning or new product:** this is semantically incompatible;
  introduce a new contract version and update every producer/reader gate.

These are not supported change mechanisms: adding an unknown `isAR_*` label to
`ch_lst.json` alone; editing `applicable_products` or generated-process lists
in a sidecar; introducing an unrecorded CLI/runtime override; or disabling
validation to accept a new region. A future runtime-configurable policy would
need a versioned policy identity serialized in the artifact and independently
validated; no such override is currently supported.
