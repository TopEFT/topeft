# Run 2 canonical channel groups from ch_lst.json (Step 5P.1)

## Summary
- Updated `metadata_TOP_22_006.yaml` and `metadata_tau_analysis.yaml` so the shared off-Z split block matches the canonical `OFFZ_TAU_SPLIT` layout (added `~fwdjet_mask` and `0tau` tags, dropped redundant on-Z entries).
- Added `analysis/metadata/metadata_run2_groups.yaml` capturing canonical definitions for `OFFZ_TAU_SPLIT_CH_LST_SR`, `ALL_CH_LST_SR`, and the new derived `OFFZ_FWD_SPLIT_CH_LST_SR` group for future scenarios.
- Canonical JSON (`topeft/scratch/ch_lst_step5.json`) is now mirrored by YAML; validation counts line up exactly for all six legacy groups.
- `OFFZ_FWD_SPLIT_CH_LST_SR` encodes the “ALL minus tau bins, no 0tau tag” derived layout to seed forward/off-Z combinations.

## Metadata updates
- `analysis/metadata/metadata_TOP_22_006.yaml`
  - `OFFZ_SPLIT_CH_LST_SR` now enforces `~fwdjet_mask` and explicit `0tau` tagging on every region definition and removes the stray on-Z clones so it mirrors the canonical off-Z split. Baseline `TOP22_006_CH_LST_SR` and `CH_LST_CR` remain unchanged but are now consistent with the JSON counts.
- `analysis/metadata/metadata_tau_analysis.yaml`
  - Applied the same `~fwdjet_mask` / `0tau` enforcement to its off-Z split block so the tau scenario uses the same canonical definitions when those groups are active.
- `analysis/metadata/metadata_run2_groups.yaml`
  - New file containing the canonical data-only representations of `OFFZ_TAU_SPLIT_CH_LST_SR`, `ALL_CH_LST_SR`, and the derived `OFFZ_FWD_SPLIT_CH_LST_SR`. Tags such as `0tau`, `1tau`, `1Ftau`, `fwdjet_mask`, and `~fwdjet_mask` are preserved exactly as in `ch_lst.json`. This file is not yet wired into the workflow but serves as the authoritative source for future scenario wiring.
- `topeft/scratch/validate_run2_groups_step5.py`
  - New helper that compares JSON vs YAML approximate channel counts for the six canonical groups and prints representative channel name samples. Also reports the derived OFFZ_FWD split size.

## OFFZ_FWD_SPLIT_CH_LST_SR
- Derived directly from `ALL_CH_LST_SR` by:
  - Dropping the `2lss_1tau`, `2los_1tau`, and `3l_1tau` lepton categories.
  - Removing the explicit `0tau` tag from the remaining tau-neutral regions while retaining their forward (`fwdjet_mask`) or central (`~fwdjet_mask`) selections.
- Provides a ready-made group for future scenarios that combine off-Z splitting with forward-enriched bins without activating the tau-specific categories.

## Validation results
Output from `validate_run2_groups_step5.py`:

```
GROUP                          JSON_COUNT    YAML_COUNT    MATCH
TOP22_006_CH_LST_SR                    147          147     YES
TAU_CH_LST_SR                          230          230     YES
TAU_CH_LST_CR                           24           24     YES
OFFZ_TAU_SPLIT_CH_LST_SR               192          192     YES
CH_LST_CR                               21           21     YES
ALL_CH_LST_SR                          542          542     YES
Derived group OFFZ_FWD_SPLIT_CH_LST_SR: YAML_COUNT=451
```

All canonical groups now match the JSON-derived counts exactly; the derived group reports only its YAML count by construction.

## Testing
- `PYTHONNOUSERSITE=1 PYTHONPATH="" /users/apiccine/work/miniconda3/envs/coffea2025/bin/python topeft/scratch/validate_run2_groups_step5.py`
