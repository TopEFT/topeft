# Embedded nominal sumw2 (schema version 3)

The main producer and paired consumer chain use schema v3. Source and derived
sidecars certify actual samples, contributors, applicability and lineage.
`run_analysis.py` and `run_analysis_three_top.py` support inline/deferred DD;
`run_data_driven.py` supports streaming and materialized transformations.
Cards and the maintained renderer share the transient view in `nominal_schema`.
Specialist legacy producer/readers retain their separate contracts. See
[the migration status](../../MULTICELL_MIGRATION_STATUS.md) for tested scope,
paired provider and the legacy/v3 boundary.

## Decisions

- There are exactly two storage modes, `embedded` and `none`. One mode applies
  to the whole payload; mixed payloads are invalid.
- 1D families store their scalar and EFT siblings as MultiCell `HistEFT`
  objects. The scalar sibling has no Wilson coefficients. Under `embedded`,
  each bin holds the quadratic EFT coefficients plus one extra cell with the
  nominal sumw2.
- 2D families store a `SparseHist` with `storage="Weight"` under `embedded`
  (value and variance in the same bin) and `storage="Double"` under `none`.
- Separate `<family>_sumw2` companion histograms are rejected. Older artifacts
  that contain them can still be unpickled, but the version 3 validator
  rejects them; regenerate them with the embedded producer.
- Selective legacy policies do not allocate v3 storage. `rules`, `selective`,
  `full_custom` and `taufitter` selection requests fail explicitly on this path.
  V3's content manifest inventories actual component processes and recorded
  embedded moments; its `sumw2_processes` field does not denote sibling keys.

## Recording convention

Storage being available and sumw2 being recorded are different things.

- Nominal statistical sumw2 is recorded for every process: Data, MC and EFT
  MC. For Data with unit weights it equals the event count.
- It is recorded only in sparse bins whose `systematic` label is `nominal`.
  In every other systematic bin the slot means "not recorded", not "zero
  variance". Its numeric content there is not authoritative: a
  `Weight`-storage fill adds w² even in JES/JER bins.
  `nominal_schema.get_nominal_sumw2` therefore omits non-nominal bins, and
  `is_nominal_sumw2_recorded` answers the question per bin.
- For EFT MC, sumw2 is the second moment of the SM-point event weight
  (`(w * c0)**2`, all Wilson coefficients zero). Its Wilson-coefficient
  dependence is neither computed nor stored.
- Selecting away the `systematic` axis preserves bin-local source labels and
  recording state. A varied slice remains unrecorded nominal content. An
  allocated zero without fill evidence remains unknown; recorded physical
  zero is preserved through copy, pickle, grouping and rebin.
- 2D `Weight` histograms are filled with the standard `SparseHist.fill()`,
  which accumulates `w**2` in every systematic bin, including JES/JER and
  weight variations. Those non-nominal variances are an artifact of the
  storage, not recorded statistical uncertainties. Consumers must read 2D
  sumw2 through `get_nominal_sumw2` (or check `is_nominal_sumw2_recorded`) and
  must not use `.variances()` of a non-nominal 2D bin as an uncertainty.
  1D MultiCell `HistEFT` fills pass `fill_sumw2=False` for non-nominal
  variations, so their sumw2 cell stays 0 there.

## Variance arithmetic

For a histogram scaled by a real scalar `c`, sumw scales by `c` and sumw2 by
`c**2`, so scaling by `-1` never produces negative sumw2. Independent
contributions add their variances. Direct subtraction of histograms with
embedded sumw2 is refused, because `Var(A - B)` needs a covariance term for
correlated inputs. The nonprompt estimate therefore keeps its existing pattern:
scale the prompt contribution by `-1`, then add it, which gives
`Var = Var(A) + Var(B)`.

Fake factors and flip rates enter the processor as per-event nominal weights.
Filling sumw2 from the full nominal weight records `sum((w * FF)**2)` exactly,
so no unweighted data count is assumed after event-dependent factors.
Multiplying a histogram by a bin-dependent array is refused, because the
correct variance would need per-event information.

## Backend requirements

- `hist.storage.MultiCell` requires boost-histogram 1.7. The current
  `environment.yml` (Python 3.9) resolves to boost-histogram 1.6.1, which
  lacks it.
- topcoffea must provide MultiCell `HistEFT` with `store_sumw2`,
  `fill(fill_sumw2=...)`, `nominal_sumw2()` and raw-count tracking. As of this
  note that is the `feature/multicell-clean` branch (tested provider 307c15ac, on upstream
  `run3_test_mmerged`). It has no `SparseHist.fill_with_moments` and no
  `HistEFT.yield_coefficients`; TopEFT uses neither.

`tests/test_embedded_sumw2_backend_contract.py` pins the backend behaviour this
contract relies on. `tests/test_nominal_schema_embedded_sumw2.py` covers the
schema contract.
