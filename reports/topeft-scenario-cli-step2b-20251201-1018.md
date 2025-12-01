# Scenario CLI UX refinements (format_update_scenarios_step2)

## Summary
- Added `resolve_scenario_choice` so callers receive both the resolved metadata path and the available scenario names, giving the CLI enough context to emit friendly errors.
- Removed the CLI `--metadata` flag, enforced mutual exclusivity between `--scenario` and `--options`, and bubbled configuration metadata through `_apply_scenario_metadata_defaults` so the caller always sees the effective scenario/metadata pair.
- Logged a single INFO line after logging is configured that reports `Using scenario '<name>' with metadata '<path>'`, with an options-profile suffix when the metadata path was explicitly provided by YAML.
- Documented the new behavior in `docs/run_analysis_cli_reference.md` and `docs/run2_scenarios.md`, including guidance on the scenario registry and the `--options` precedence rules.

## Testing
- `PYTHONNOUSERSITE=1 PYTHONPATH="" /users/apiccine/work/miniconda3/envs/coffea2025/bin/python analysis/topeft_run2/run_analysis.py input_samples/cfgs/mc_signal_samples_NDSkim.cfg --outname UL17_SRs_quickstart_test --outpath histos/local_debug --nworkers 1 --summary-verbosity brief --executor iterative --skip-cr --do-systs --log-level INFO -c 1 -s 5 --scenario TOP_22_006 --pretend`
- `PYTHONNOUSERSITE=1 PYTHONPATH="" /users/apiccine/work/miniconda3/envs/coffea2025/bin/python analysis/topeft_run2/run_analysis.py input_samples/cfgs/mc_signal_samples_NDSkim.cfg --scenario does_not_exist --pretend`
- `PYTHONNOUSERSITE=1 PYTHONPATH="" /users/apiccine/work/miniconda3/envs/coffea2025/bin/python analysis/topeft_run2/run_analysis.py --options /tmp/options_scenario_test.yaml --pretend`
- `PYTHONNOUSERSITE=1 PYTHONPATH="" /users/apiccine/work/miniconda3/envs/coffea2025/bin/python analysis/topeft_run2/run_analysis.py --options /tmp/options_scenario_test.yaml --scenario TOP_22_006 --pretend`

## Follow-ups
- Consider migrating older docs that still reference `--metadata` on `run_analysis.py` so they point readers at the scenario registry or YAML overrides.
