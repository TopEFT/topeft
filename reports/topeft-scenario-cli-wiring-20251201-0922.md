# Scenario registry and CLI wiring

## Summary
- Added `analysis/topeft_run2/scenario_registry.py` to map `TOP_22_006`, `tau_analysis`, and `fwd_analysis` to their respective `analysis/metadata/metadata_*.yaml` files and expose helpers for the CLI.
- Updated `run_analysis.py` to enforce one scenario per run, infer metadata paths via the registry when `--metadata` is omitted, and log the resolved scenario/metadata pairing alongside the new `--metadata` help text.
- Ensured `ChannelMetadataHelper` can operate without a `channels.scenarios` block by falling back to all known groups in per-scenario YAMLs, keeping planner behaviour intact.
- Kept the workflow loading logic pointed at `config.metadata_path`, so scenario-driven paths flow through untouched.

## Testing
- `PYTHONNOUSERSITE=1 PYTHONPATH="" /users/apiccine/work/miniconda3/envs/coffea2025/bin/python analysis/topeft_run2/run_analysis.py input_samples/cfgs/mc_signal_samples_NDSkim.cfg --outname UL17_SRs_quickstart_test --outpath histos/local_debug --nworkers 1 --summary-verbosity brief --executor iterative --skip-cr --do-systs --log-level INFO -c 1 -s 5 --pretend`
  - Confirms scenario/metadata logging, registry resolution, and planning succeed (pretend mode used to avoid long-running execution in this environment).

## Follow-ups
- Support multi-scenario executions (or the future `all_analysis` meta-scenario) once a combined metadata file is ready and planners can merge groups.
- Expand automated tests around the scenario registry and ChannelMetadataHelper fallback to guard against regressions.
