# State of the embedded-sumw2 (multicell) migration

- **TopEFT branch:** `feature/multicell-global-sumw2`
- **Required topcoffea:** branch `feature/multicell-clean`, commit `e92774a`,
  at https://github.com/lazdlie/topcoffea. It is upstream `run3_test_mmerged`
  plus the MultiCell backend, and it keeps Andrea's raw-count tracking (PR #103).
  It needs `boost-histogram` 1.7 or newer, so use `environment-multicell.yml`
  rather than `environment.yml`.

The old design stored statistical uncertainties as separate `<family>_sumw2`
histograms, chosen by `sumw2_policy` and tracked in a manifest. This branch
replaces that with sumw2 embedded in each histogram, for all processes. The
policy and manifest code is commented out with the marker
`SUMW2_POLICY_DISABLED (multicell migration)`, not deleted. Design details are
in `docs/explanation/embedded_sumw2.md`.

## Completed

- **Storage.** 1D scalar and EFT histograms use MultiCell `HistEFT`; 2D
  histograms use `SparseHist` with `Weight` storage. One mode applies to the
  whole output: `embedded` or `none`. Companion `<family>_sumw2` histograms are
  rejected.
- **Sumw2 convention.** Recorded for Data and MC, for the nominal systematic
  only. For EFT samples it is evaluated at the SM point (all Wilson
  coefficients zero); its Wilson-coefficient dependence is not stored.
- **Processor.** `AnalysisProcessor` no longer uses `fill_with_moments()` or
  the selective sumw2 policy. 1D fills pass `fill_sumw2=False` for non-nominal
  variations. 2D fills use the standard `fill()`, so non-nominal 2D bins also
  hold w²; that content is "not recorded", and `get_nominal_sumw2()` returns
  nominal bins only.
- **Raw counts.** Compatible with Andrea's upstream implementation, unchanged
  (only nominal MC is counted; preserved by pickling and rebinning).
- **Statistical operations.** Scaling by c gives c² on sumw2, including
  negative c. Merging adds variances. Subtraction is refused. Rebinning keeps
  EFT coefficients aligned.
- **Launchers.** `run_analysis.py` and `run_analysis_three_top.py` no longer
  resolve or pass the policy. `--no-sumw2` / `sumw2_storage.mode: disabled`
  select `none`; any other mode selects `embedded`. They validate the output
  with the schema-v3 validator.
- **Artifact writing.** `write_histogram_artifact(sumw2_storage=...)` writes
  schema-v3 processor outputs. It keeps the PKL identity (sha256), lineage,
  histogram applicability, requested data-driven products and atomic
  publication, without policy provenance or the sumw2 content manifest.

## Partially implemented

- **v3 sidecars** are written for processor output only. The policy-certified
  production-sample and resolved data-driven contracts are not included yet.
- **Inline nonprompt** (`--np-postprocess inline`) is rejected at startup for
  v3 outputs. `defer` and `skip` still run.

## Remaining blockers

- Reading and merging v3 sidecars are not supported yet
  (`read_histogram_sidecar`, `validate_histogram_artifact`, merge path).
- Data-driven (nonprompt, flips), datacard, plotting and diboson consumers
  still expect schema v2 with companion histograms.
- The histogram-applicability structure check ("Family X is structurally
  absent for applicable channel(s)") fails launcher tests and default-list runs.
  This was already the case before this branch.
- Real-event end-to-end validation against the clean topcoffea backend is
  pending.

## Validation

- 118 focused tests pass against `feature/multicell-clean` (`e92774a`): the
  embedded-sumw2 backend, schema and processor tests, plus the hist-filter,
  ptll and raw-count tests. The v2 artifact sidecar tests also pass.
- The 13 launcher-test failures that remain also fail on the pre-migration
  commit `558662fe`.
- Earlier, against the previous integration backend, the CI ttH sample matched
  the old companion format exactly (yields, sumw2, EFT coefficients). That
  comparison has not been repeated on the clean backend.
