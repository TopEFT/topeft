# Missing-parton uncertainties

## 1. Purpose and scope

The `missing_parton` uncertainty accounts for the parton-content difference
between the private leading-order `tllq` signal sample and the central
next-to-leading-order `tZq` reference used to derive the correction. A compact
ROOT payload stores one non-negative fractional uncertainty for each supported
analysis category and physical jet population.

This payload is not produced inside the event processor. It is derived after
histogram PKLs have been converted into per-category source datacards. The
derivation is implemented in
[`analysis/topeft_run2/missing_parton.py`](../analysis/topeft_run2/missing_parton.py).
[`DatacardMaker`](../topeft/modules/datacard_tools.py) later reads the final
payload and writes the `missing_parton` `lnN` row into analysis cards.

The nuisance applies to `tllq` and `tHq`. It does not apply to `tZq`, `ttll`,
`ttH`, or unrelated processes.

## 2. End-to-end pipeline

The complete data flow is:

```text
analysis processor
    -> histEFT PKL outputs
    -> DatacardMaker source cards
    -> missing-parton derivation
    -> final correction payload ROOT
    -> DatacardMaker consumer application
    -> final analysis cards
```

There are two different ROOT layers in this chain:

1. **Source-card ROOT files** are paired with TXT datacards. Each
   `ttx_multileptons-<base_category>_njets.root` file contains the nominal and
   shape templates used by the derivation, while the matching TXT file supplies
   process rates and rate-systematic factors.
2. **Final correction payload ROOT files** are
   `topeft/data/missing_parton/missing_parton_run2.root` and
   `topeft/data/missing_parton/missing_parton_run3.root`. They contain only the
   public fractional correction arrays and are consumed later by
   `DatacardMaker`.

The source-card ROOT files are derivation inputs. They are not interchangeable
with the final correction payloads.

## 3. Relationship to the histEFT PKL tutorial

The [histEFT PKL tutorial](histeft_pkl_tutorial.md) explains the upstream
analysis-processor and histogram-PKL production workflow: processor launch,
histogram structure, PKL inspection, and plotting-oriented use of HistEFT.

This guide starts at the missing-parton-specific boundary. It assumes that an
accepted PKL production has already been converted into the required
per-category ROOT/TXT source-card pairs. It does not repeat processor setup,
sample selection, HistEFT coefficient handling, executor configuration, or PKL
inspection. Instead, it explains how those accepted card inputs become the
correction payload and how the later card-making step consumes it.

## 4. Source authority by era

- **Run 2:** the accepted Run 2 all-analysis missing-parton source-card
  production.
- **Run 3:** source cards from the accepted Run 3 fixyield production used by
  the maintained analysis workflow.

The accepted Run 2 source-card production and the corresponding maintained
Run 2 production used the same upstream histogram PKLs. Exact private PKL
hashes are not available, so this statement records source provenance without
claiming byte-for-byte identity between PKLs and cards.

A reproducible source manifest records each ROOT/TXT pair, its era, category,
central or private role, process identity, and content hash. Storage locations
can vary by site; hashes and category/role assignments define the inputs.

## 5. Physics definition

The derivation compares one private `tllq` yield with one central `tZq` yield
for the same base category and physical jet population. All yield and
uncertainty quantities below have units of expected events. The stored result
is dimensionless.

For one population, define:

- `p`: the private nominal `tllq` yield;
- `c`: the central nominal `tZq` yield;
- `sigma_down`: the private directional uncertainty that can move the private
  yield downward;
- `sigma_up`: the private directional uncertainty that can move the private
  yield upward;
- `delta = p - c`: the nominal yield difference.

### 5.1 Directional shape and rate inputs

For every private shape template, the code forms the signed bin shift
`shape_value - p`. Positive shifts accumulate only in the up quadrature;
zero or negative shifts accumulate only in the down quadrature:

```text
shape_up_squared   = sum(positive_shape_shift ** 2)
shape_down_squared = sum(nonpositive_shape_shift ** 2)
```

TXT rate-systematic entries add independent contributions. For an asymmetric
factor `low/high`, the fractional shifts are `1 - low` and `high - 1`. For a
single factor `kappa`, both directions use `kappa - 1`. Each contribution is
multiplied by the private nominal yield before it is added in quadrature:

```text
sigma_down = sqrt(shape_down_squared + sum((p * rate_down) ** 2))
sigma_up   = sqrt(shape_up_squared   + sum((p * rate_up) ** 2))
```

Only TXT entries matching the private `tllq` process are used. Down and up are
independent accumulators; neither direction is seeded from or overwritten by
the other.

### 5.2 Residual amount and stored fraction

The selected directional uncertainty is the one that moves the private yield
toward the central yield:

```text
selected_sigma = sigma_down if delta >= 0 else sigma_up
```

The residual missing-parton amount and stored fractional uncertainty are:

```text
residual_squared = max(delta ** 2 - selected_sigma ** 2, 0)
missing_amount   = sqrt(residual_squared)
stored_fraction  = missing_amount / p
```

`missing_amount` has units of expected events. `stored_fraction` is a
dimensionless, non-negative fraction of the private nominal yield. The
subtraction in quadrature prevents the new nuisance from double-counting the
part of the private/central difference already covered by the selected private
uncertainty. If that uncertainty covers the full difference, the residual and
stored fraction are zero.

### 5.3 Near-zero and invalid yields

The numerical zero threshold is `1e-5` expected events:

- if `abs(p) < 1e-5`, both the amount and stored fraction remain zero;
- if `abs(c) < 1e-5`, the effective central yield is set to zero;
- a private yield at or below `-1e-5` is rejected;
- negative directional errors, non-finite inputs, negative fractions, and
  invalid consumer factors are rejected rather than clipped or repaired.

### 5.4 Consumer factors

The payload stores only `stored_fraction = f`. The consumer first forms
`kappa_up = 1 + f`, then writes:

```text
kappa_down = max(0.01, 1 - f)
kappa_up   = 1 + f
```

The factors are symmetric in their displacement from one until the protective
`0.01` lower floor is reached. The datacard representation is the explicit
two-sided `kappa_down/kappa_up` form; separate down and up fractions are not
stored in the payload.

## 6. Terminal-bin semantics

The registry in
[`topeft/channels/ch_lst.json`](../topeft/channels/ch_lst.json) defines the
physical jet populations:

- `=N` means exactly `N` selected jets;
- `>N` is the maintained inclusive token for physical `njet >= N`.

For an exact category, the formula is evaluated directly at the physical
index. For an inclusive terminal category, the producer selects every source
bin from `N` through the maintained physical overflow bin, then:

1. sums the private nominal yields;
2. sums the central nominal yields;
3. sums each private shape template over the same population;
4. applies each private TXT rate factor to the aggregated private nominal;
5. recomputes independent down and up errors; and
6. applies the scalar residual formula once.

The producer therefore evaluates one uncertainty for the complete `>=N`
population. It does not calculate a fraction in each source bin and add those
fractions or event amounts afterward. These operations are not equivalent:
the formula contains a squared yield difference, directional uncertainty
selection, subtraction in quadrature, a square root, and division by the
private yield, none of which generally commutes with summation.

## 7. Forward-category correction

The current production contract for both
`3l_m_offZ_1b_fwd` and `3l_p_offZ_1b_fwd` is:

```text
array length: 5
index 4: complete physical njet >= 4 population
index 5: absent
```

The supported `run3_test_mmerged` legacy compatibility contract is:

```text
array length: 6
index 4: exactly 4 selected jets
index 5: physical njet >= 5 population
```

The active producer writes only the current contract. Legacy test coverage
does not select or prefer the older production layout.

## 8. Process, era, and correlation policy

The consumer assigns the same per-category correction dictionary to `tllq`
and `tHq`. This is the maintained physics policy for the two target signal
processes; it is not inferred by process-name similarity. `tZq`, `ttll`,
`ttH`, and unrelated processes receive `-` in the `missing_parton` row.

Run-era selection is:

| Requested period | Default payload |
|---|---|
| `UL16`, `UL16APV`, `UL17`, `UL18` | `data/missing_parton/missing_parton_run2.root` |
| `2022`, `2022EE`, `2023`, `2023BPix` | `data/missing_parton/missing_parton_run3.root` |

Both eras use the nuisance name `missing_parton`. When cards from the two eras
are combined, the shared name makes this one correlated nuisance rather than
two era-decorrelated nuisances. One `DatacardMaker` invocation cannot mix Run 2
and Run 3 periods with a single payload source; cards must be produced
separately with the matching era payloads.

## 9. Public ROOT schema and supported contracts

The current consumer-facing schema is:

- exactly one top-level `TTree` for each of the 34 `ALL_CH_LST_SR` base
  categories;
- tree set and order exactly equal to registry key order;
- exactly one branch named `tllq` in each tree;
- branch representation `double`/`float64` one-dimensional array;
- array index equal to physical `njet`;
- an inclusive `>N` terminal stored at index `N` with public length `N + 1`;
- no entry above the terminal threshold; and
- finite, non-negative stored fractions.

The public structure is an interface between registry metadata, the producer,
the installed files, and the consumer lookup. Changing it requires coordinated
producer, consumer, test, and payload updates.

Exactly two schemas have maintained test contracts:

1. the current production schema described above;
2. the `run3_test_mmerged` legacy compatibility schema introduced by immutable
   commit `2469053a8d7ab0b42c86c68000f51b6e7f6dafff`.

No other historical or partially migrated schema is supported. The legacy
validator documents one fixed compatibility layout; it does not make that
layout an accepted production output.

## 10. Payload selection and overrides

With `--do-nuisance`, `DatacardMaker` resolves the default payload from the
requested `--year` values and the `ALL_CH_LST_SR` registry. For example:

```bash
python analysis/topeft_run2/make_cards.py input_histograms.pkl.gz \
  --year 2022 \
  --do-nuisance \
  --var-lst ht \
  --ch-lst 3l_onZ_1b_2j
```

An explicit payload is selected with `--miss-parton-file`:

```bash
python analysis/topeft_run2/make_cards.py input_histograms.pkl.gz \
  --year UL18 \
  --do-nuisance \
  --miss-parton-file data/missing_parton/missing_parton_run2.root \
  --var-lst ht \
  --ch-lst 3l_onZ_1b_2j
```

The explicit string is retained as the selected payload path and resolved
relative to the installed `topeft` package by `topeft_path()`. It must be
non-empty. The consumer does not inspect its layout and then choose another
path, and it does not fall back to the era default if opening the explicit path
fails.

To keep other nuisances but omit this one, use:

```bash
python analysis/topeft_run2/make_cards.py input_histograms.pkl.gz \
  --year 2022 \
  --do-nuisance \
  --skip-missing-parton-rate-syst \
  --var-lst ht \
  --ch-lst 3l_onZ_1b_2j
```

In this mode no missing-parton payload path is resolved or opened. Omitting
`--do-nuisance` also returns before any rate-systematic payload is opened.

## 11. Reproduction workflow

The commands below show the maintained interface with environment-dependent
source locations represented by shell variables. Run them from the repository
root in the configured analysis environment.

### 11.1 Identify and freeze source inputs

Select one accepted central `tZq` and private `tllq` source-card directory for
the era. Build a manifest covering all 34 matching
`ttx_multileptons-<base_category>_njets.root` and `.txt` pairs. Record content
hashes, category names, roles, process identities, and rate inputs before
generation.

### 11.2 Validate the derivation without writing

```bash
central_card_dir=/path/to/accepted/central_tzq
private_card_dir=/path/to/accepted/private_tllq
scratch_payload=/path/to/writable/scratch/missing_parton_run2.root

python analysis/topeft_run2/missing_parton.py \
  --central-card-dir "${central_card_dir}" \
  --private-card-dir "${private_card_dir}" \
  --sr-registry ALL_CH_LST_SR \
  --output-file "${scratch_payload}" \
  --var njets \
  --dry-run
```

The dry run validates the category inventory, ROOT/TXT pairing, process names,
physical eight-bin `njets` axes, numerical inputs, registry layout, and complete
payload plan without writing the ROOT output.

### 11.3 Write a fresh scratch payload

Run the same command without `--dry-run`. Use a fresh output path. If an
existing scratch file is intentionally replaced, add `--overwrite`; the writer
validates a temporary file before atomically replacing the target.

```bash
python analysis/topeft_run2/missing_parton.py \
  --central-card-dir "${central_card_dir}" \
  --private-card-dir "${private_card_dir}" \
  --sr-registry ALL_CH_LST_SR \
  --output-file "${scratch_payload}" \
  --var njets
```

### 11.4 Validate and reconstruct

Before installation:

1. recheck all source hashes;
2. require the exact 34-tree registry set and order;
3. require the `tllq` `double`/`float64` branch and current lengths;
4. require finite, non-negative values and the current forward layout;
5. regenerate twice and compare semantic identities;
6. independently reconstruct every direct and terminal value without calling
   the producer's payload-plan or writer path; and
7. compare the result with the installed semantic identity expected for the
   era.

The checked-in schema validator and focused consumer coverage can be run with:

```bash
python -m pytest -q \
  tests/test_missing_parton_payload_schema.py \
  tests/test_missing_parton_registry_layout.py \
  tests/test_missing_parton_contract.py \
  tests/test_make_cards_missing_parton_option.py \
  tests/test_missing_parton_payload_roundtrip.py
```

### 11.5 Install both eras together

Treat Run 2 and Run 3 as one installation unit. Replace
`missing_parton_run2.root` and `missing_parton_run3.root` only after both pass
source-hash, deterministic regeneration, independent reconstruction, strict
schema, and focused consumer checks. Reopen the installed files and repeat the
schema, semantic-identity, and reconstruction checks.

## 12. Validated properties

The installed payloads have the following validated properties:

- 34 trees per era;
- 211 public values per era: 177 direct and 34 terminal;
- 422 of 422 values independently reconstructed across both eras;
- deterministic semantic regeneration for each era;
- strict current-schema validation;
- strict `run3_test_mmerged` legacy-schema validation;
- cross-rejection between the current and legacy validators;
- no broad schema fallback; and
- focused checks for era selection, exact override behavior, skip behavior,
  physical-jet lookup, target processes, and excluded processes.

The installed semantic digests are:

- Run 2: `4a869bc8ecc56adb491100e50b29d0e600a6916824b2849ff9f9d31c5a09736a`
- Run 3: `6f948c7859a43249dae70e4e679c4439425dc4384991ab2829e0d85c09eed26f`

## 13. Quantitative historical terminal impact

Holding each accepted source-card set fixed, replacing aggregation of
already-derived terminal amounts with source-level aggregation changed 28 of
34 terminal categories in each era.

| Diagnostic quantity | Run 2 | Run 3 |
|---|---:|---:|
| Nonzero terminal categories | 28 of 34 | 28 of 34 |
| Maximum directional effect on private source-card `tllq` yield | approximately 0.2154 events | approximately 0.08152 events |
| Categories at or above 0.01 event | 2 | 6 |
| Categories at or above 0.1 event | 1 | 0 |
| Categories at or above 1 event | 0 | 0 |

These are diagnostic changes to the private source-card `tllq` yield used in
the derivation. They are not direct measurements of final-card yields, fit
parameters, or final fit impact. The corrected forward `>=4` population is a
separate public-layout change and is not included in the pure terminal-
aggregation comparison.

## 14. Bounded downstream comparison

A bounded consumer comparison covered Run 3 period `2022`, five final
channels, and ten `tllq`/`tHq` cells. It found:

- zero unexpected consumer differences;
- a maximum observed selected-versus-reference directional yield difference
  of approximately `0.002692` expected events; and
- zero cells at or above `0.01` expected event.

This was a smoke test of a limited Run 3 selection, not exhaustive coverage of
all eras, categories, channels, or fit behavior.

## 15. Why the historical payloads were replaced

The production contract now requires:

- independent down and up error accumulation;
- source-level terminal aggregation followed by one formula evaluation;
- registry-driven tree ordering and public lengths;
- corrected forward terminal populations; and
- source-backed deterministic reproduction.

The older production layout used cross-coupled directional accumulation,
aggregated already-derived terminal amounts, relied on hard-coded public
ordering, and retained the older forward population split. The explicit legacy
schema remains useful for compatibility tests, but it is not the active output
format.

## 16. Troubleshooting

| Failure | Likely layer | What to inspect |
|---|---|---|
| Tree-order mismatch | public writer or test fixture | Compare top-level keys with `ALL_CH_LST_SR` registry order; do not sort the current contract alphabetically. |
| Wrong forward length | derivation layout or test fixture | Current forward arrays have length 5; only the fixed legacy contract has length 6. |
| Missing category | source cards or derivation inventory | Confirm both ROOT and TXT inputs exist for every registry base category. |
| Extra category | source cards or public writer | Check whether an unused card leaked into the selected inventory or output. |
| Wrong branch type | public writer | Require one `tllq` `double`/`float64` array branch per tree. |
| Non-finite value | source cards or derivation | Inspect nominal, shape, and rate inputs before the scalar formula. |
| Wrong physical-jet lookup | registry metadata or consumer lookup | Confirm `=N`/`>N` parsing, final-channel normalization, and `bin_idx = num_j`. |
| Unexpected process application | consumer lookup | The correction dictionary must be attached only to `tllq` and `tHq`. |
| Payload opened in disabled mode | consumer initialization | Check `do_nuisance` and `skip_missing_parton_rate_syst` before path resolution and loading. |
| Source-manifest drift | source cards | Recompute every ROOT/TXT content hash and stop regeneration until the change is understood. |
| Semantic digest mismatch | derivation, writer, or source cards | Compare source hashes, registry order, array lengths, and every stored value; raw ROOT metadata alone is not a semantic comparison. |

## 17. Maintenance rules

- Keep the accepted source manifest and its hashes with the production record.
- Preserve independent directional quadrature and source-level terminal
  aggregation.
- Derive current tree order and lengths from `ALL_CH_LST_SR`; never discover a
  schema from the payload under test.
- Keep current and `run3_test_mmerged` validators separate and cross-rejecting.
- Do not add automatic schema detection or support partially migrated layouts.
- Treat a semantic digest change as a source, derivation, registry, or writer
  change that requires full reconstruction and consumer checks.
- Keep source-card yield diagnostics distinct from downstream card and fit
  effects.

## 18. Selective-sumw2 input boundary

The maintained standard-analysis workflow uses its resolved selective-sumw2
policy and validated histogram artifact as the upstream input contract. This
guide does not broaden that contract into a new missing-parton physics claim.

`full_custom` is different: the operator owns the configuration and must
explicitly select and validate every family/process companion required by the
intended missing-parton and downstream card consumers. It is not automatically
equivalent to a maintained standard-analysis mode. Use the
[Run 2 operator guide](../analysis/topeft_run2/README.md#selective-sumw2-storage-workflow)
for mode/rule configuration and the
[HistEFT API contract](histeft_api_contract.md#15-selective-sumw2-schema-and-consumer-contract)
for artifact, merge, and consumer boundaries.
