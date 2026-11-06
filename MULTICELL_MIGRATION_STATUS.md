# State of the embedded-sumw2 (multicell) migration

This branch, `feature/multicell-global-sumw2`, starts from the current topeft
code, where statistical uncertainties are stored as separate
`<family>_sumw2` histograms chosen by `sumw2_policy` and tracked in a
manifest. The branch replaces that with the multicell backend: each histogram
stores its own sumw2 inside every bin, for all processes, so no policy or
manifest is needed.

## What's done (Phases 0–2)

- **New format.** The schema "version 3" rules are in
  `topeft/modules/nominal_schema.py`, with helpers to validate it and read
  sumw2 back.
- **Processor.** `AnalysisProcessor` now fills histograms in the new format
  only. 1D histograms use multicell `HistEFT` and 2D histograms use `Weight`
  storage. Sumw2 is recorded for the nominal systematic only, and at the SM
  point for EFT samples.
- **Policy switched off, not deleted.** `sumw2_policy` is disconnected; the
  old code is commented out with the marker
  `SUMW2_POLICY_DISABLED (multicell migration)`.
- **Rebinning.** It works with the new format.
- **Checks.** On the CI ttH sample, yields, sumw2 and EFT coefficients match
  the old format exactly. 232 focused tests pass.

## What's missing (Phases 3–8)

- **Runner.** The processor produces the new histograms, but
  `run_analysis.py` can't write them to a PKL yet. It still passes the old
  policy, which the processor now rejects.
- **Downstream code.** The PKL metadata (sidecars), datacards, the nonprompt
  estimate, plotting and the related tests still expect the old format.
- **Rollout.** CI, documentation and a full reprocessing of the PKLs come
  last.
- Details and open issues are in `docs/explanation/embedded_sumw2.md`.

## Required topcoffea

This topeft branch only works with a topcoffea that has both the multicell
backend and Andrea's raw-count tracking (`track_raw_counts`, upstream PR #103).
That combination is on the `integration/multicell-current-upstream` branch,
commit `ab95b39`, at https://github.com/lazdlie/topcoffea. It also needs
`boost-histogram` 1.7 or newer, so use `environment-multicell.yml` rather than
`environment.yml`.
