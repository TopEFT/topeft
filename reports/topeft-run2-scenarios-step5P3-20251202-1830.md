# Step 5P.3 – Run 2 scenario loader + validator

## Scenario loader
- Added `topeft/modules/run2_scenarios.py` with a `ScenarioDefinition` dataclass and helpers to parse `analysis/metadata/run2_scenarios.yaml`.
- The loader aggregates channel groups across `metadata_TOP_22_006.yaml`, `metadata_tau_analysis.yaml`, and `metadata_run2_groups.yaml`, building a canonical `channels` mapping trimmed to the groups requested by a scenario.
- Duplicate group names across metadata bundles retain the first definition encountered so shared CR blocks such as `CH_LST_CR` remain deterministic.
- `load_run2_channels_for_scenario()` now supplies a ready-to-use payload for `ChannelMetadataHelper`, including a scoped `scenarios` entry for downstream helpers.

## Scenario validator (scratch)
- Rebuilt `scratch/validate_run2_scenarios_step5.py` to depend solely on `ChannelMetadataHelper` + the new loader.
- Inlined the minimal datacard naming logic (`extract_number`, `_analysis_mode_for_group`, `determine_histogram_suffix`) so no `topcoffea` imports are required.
- Added CLI arguments for scenario selection and sample reporting while keeping the default behaviour focused on all Run 2 scenarios.
- The script now prints a concise table and sample channel names:

```
SCENARIO              GROUPS    SR_CHANNELS
-------------------------------------------
TOP_22_006                 2             43
tau_analysis               3             68
offz_tau_analysis          3             48
offz_fwd_analysis          2            121
all_analysis               3            148
```

Sample channels (first 8 by default):
- `TOP_22_006`: `2lss_4t_m_4j_lj0pt`, `2lss_4t_m_5j_lj0pt`, `2lss_4t_m_6j_lj0pt`, `2lss_4t_m_7j_lj0pt`, ...
- `offz_tau_analysis`: `3l_m_offZ_high_1b_2j_ptz`, `3l_m_offZ_high_1b_3j_ptz`, `3l_m_offZ_high_1b_4j_ptz`, `3l_m_offZ_high_1b_5j_ptz`, ...
- `offz_fwd_analysis`: `2lss_4t_m_4j_lj0pt`, `2lss_4t_m_5j_lj0pt`, `2lss_4t_m_6j_lj0pt`, `2lss_4t_m_7j_lj0pt`, ...
- `all_analysis`: `2los_onZ_1tau_5j_ptz`, `2lss_4t_m_4j_lj0pt`, `2lss_4t_m_5j_lj0pt`, `2lss_4t_m_6j_lj0pt`, ...

## Testing
- `PYTHONNOUSERSITE=1 PYTHONPATH="" $PYTHON_ENV -m compileall topeft/modules/run2_scenarios.py scratch/validate_run2_scenarios_step5.py`
- `PYTHONNOUSERSITE=1 PYTHONPATH="" $PYTHON_ENV scratch/validate_run2_scenarios_step5.py --help`
- `PYTHONNOUSERSITE=1 PYTHONPATH="" $PYTHON_ENV scratch/validate_run2_scenarios_step5.py`
