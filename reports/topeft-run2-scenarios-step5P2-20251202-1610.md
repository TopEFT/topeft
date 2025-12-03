# Run 2 scenarios.yaml + validation (Step 5P.2)

## Summary
- Added `analysis/metadata/run2_scenarios.yaml` capturing the agreed scenario → group composition for Run 2 (TOP_22_006, tau_analysis, offz_tau_analysis, offz_fwd_analysis, all_analysis).
- Introduced `scratch/validate_run2_scenarios_step5.py`, which loads the scenario mapping plus the canonical group YAMLs and reports approximate per-scenario channel counts.
- Validation confirms each scenario resolves to the expected groups and that the summed channel counts match our design (e.g. TOP_22_006 = 147 + 21, offz_fwd_analysis = 451 + 21, etc.).

## Scenario mapping
```
TOP_22_006:           TOP22_006_CH_LST_SR, CH_LST_CR
tau_analysis:         TAU_CH_LST_SR, TAU_CH_LST_CR, CH_LST_CR
offz_tau_analysis:    OFFZ_TAU_SPLIT_CH_LST_SR, TAU_CH_LST_CR, CH_LST_CR
offz_fwd_analysis:    OFFZ_FWD_SPLIT_CH_LST_SR, CH_LST_CR
all_analysis:         ALL_CH_LST_SR, TAU_CH_LST_CR, CH_LST_CR
```

## Validation results
Command:
```
PYTHONNOUSERSITE=1 PYTHONPATH="" \
  /users/apiccine/work/miniconda3/envs/coffea2025/bin/python \
  topeft/scratch/validate_run2_scenarios_step5.py
```

Summary table:
```
SCENARIO                 GROUP_COUNT   APPROX_CHANNELS
TOP_22_006                         2               168
tau_analysis                       3               275
offz_tau_analysis                  3               237
offz_fwd_analysis                  2               472
all_analysis                       3               587
```

Representative breakdowns:
```
Scenario TOP_22_006:
  - TOP22_006_CH_LST_SR: 147
  - CH_LST_CR: 21
  → total: 168

Scenario offz_fwd_analysis:
  - OFFZ_FWD_SPLIT_CH_LST_SR: 451
  - CH_LST_CR: 21
  → total: 472
```
