# Run analysis docs cleanup (format_update_scenarios_step2)

## Summary
- Updated the top-level `README.md`, `analysis/topeft_run2/README.md`, `docs/quickstart_top22_006.md`, `docs/quickstart_run2.md`, and `docs/run_analysis_configuration.md` so user-facing instructions explain that `run_analysis.py` selects metadata via the scenario registry or options profiles rather than a `--metadata` CLI flag.
- Replaced CLI examples that previously used `--metadata` with YAML-first workflows (showing where to set the `metadata:` key) and highlighted that quickstart helpers remain the only place where `--metadata` is accepted directly.
- Confirmed the only remaining `--metadata` mentions under `docs/` reference the quickstart helper intentionally.

## Verification
- `rg --no-heading --line-number -- '--metadata' docs analysis`  → quickstart-specific mentions only (`docs/run_analysis_configuration.md:41`, `docs/quickstart_run2.md:165`).
- `PYTHONNOUSERSITE=1 PYTHONPATH="" /users/apiccine/work/miniconda3/envs/coffea2025/bin/python analysis/topeft_run2/run_analysis.py input_samples/cfgs/mc_signal_samples_NDSkim.cfg --scenario TOP_22_006 --executor iterative --pretend`
  - Confirms the CLI still resolves the scenario/metadata pair and completes pretend runs after docs changes. (The default TaskVine run without `--executor iterative` still errors in this environment because no cached remote environment tarball is present.)

## Follow-ups
- None; future doc updates should continue emphasizing the scenario registry plus YAML overrides for custom metadata.
