# Step 3 – scenario metadata wiring

## Summary
- `run_workflow` now requires `RunConfig.metadata_path` to be set by the CLI/options resolver and loads only that YAML, removing the legacy fallback to `params/metadata.yml`.
- `ChannelMetadataHelper` exposes `selected_group_names`, which returns the requested scenario groups when a legacy `channels.scenarios` block exists and otherwise yields every group defined in the per-scenario YAML.
- `ChannelPlanner` retrieves the active groups via this helper and splits SR/CR purely by group names, keeping feature aggregation while staying agnostic to the legacy scenario map.

## Technical details
- The helper still stores scenario definitions when present for backwards compatibility; selecting a scenario validates referenced groups and preserves insertion order.
- When metadata contains no `channels.scenarios`, the helper simply iterates the `channels.groups` keys, so per-scenario YAMLs automatically describe the active regions without additional wiring.
- SR vs CR splitting still hinges on the `_CR` suffix; `ChannelPlanner` records feature flags from every registered group and feeds them into channel dictionaries unchanged.

## Testing
- `PYTHONNOUSERSITE=1 PYTHONPATH="" /users/apiccine/work/miniconda3/envs/coffea2025/bin/python analysis/topeft_run2/run_analysis.py input_samples/cfgs/mc_signal_samples_NDSkim.cfg --outname UL17_SRs_quickstart_test --outpath histos/local_debug --nworkers 1 --summary-verbosity brief --executor iterative --skip-cr --do-systs --log-level INFO -c 1 -s 5 --scenario TOP_22_006 --pretend`
  - PASS: logs include the scenario/metadata pairing and the pretend run completes without missing-scenario errors.
