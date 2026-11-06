# Embedded nominal sumw2 (schema version 3)

**Status:** design note for the in-progress multicell migration. The contract
is implemented in `topeft.modules.nominal_schema` (`EMBEDDED_SUMW2_SCHEMA_VERSION`),
and `AnalysisProcessor` now allocates and fills version 3 histograms. The
runners, sidecars and downstream consumers still expect schema version 2 with
`<family>_sumw2` companions. Until they are migrated, `run_analysis.py` cannot
run end to end, and the diboson specialist processor still writes companions.
Do not rely on version 3 artifacts until the migration is complete.

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
- The selective sumw2 policy (`sumw2_policy.py`) and the sumw2 content
  manifest take no part in version 3. Their code stays in the repository but
  will be disconnected from the active path during the migration.

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
- A histogram without a `systematic` axis (for example after selecting the
  nominal slice) is treated as nominal content.

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
  `nominal_sumw2()` and raw-count tracking. As of this note that is the
  `integration/multicell-current-upstream` branch.

`tests/test_embedded_sumw2_backend_contract.py` pins the backend behaviour this
contract relies on. `tests/test_nominal_schema_embedded_sumw2.py` covers the
schema contract.
