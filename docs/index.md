# TopEFT documentation map

Use this map to navigate the documentation tracks introduced during the docs
reorganisation. Each section highlights the current source of truth for that
topic so newcomers can follow a single path instead of bouncing between
overlapping quickstarts.

## Landing & quickstart

- [Workflow and YAML hub](workflow_and_yaml_hub.md) – **Start here** for
  prerequisites, YAML merging, and executor choices.
- [Run 2 quickstart pipeline](quickstart_run2.md) – Current end-to-end Run‑2
  walkthrough (to be consolidated with plotting guides in a later step).
- [Run and plot quickstart](run_and_plot_quickstart.md) – Plotting appendix for
  the quickstart; slated for merger into the main quickstart during 6B.2.
- [TOP-22-006 script walkthrough](quickstart_top22_006.md) – Scenario-specific
  quickstart extending the Run‑2 presets.

## Running analyses

- [Run analysis configuration flow](run_analysis_configuration.md) – Narrative
  walkthrough of CLI/YAML merging and workflow helpers.
- [`run_analysis.py` CLI and YAML reference](run_analysis_cli_reference.md) –
  Flag-by-flag lookup table.
- [TaskVine workflow quickstart](taskvine_workflow.md) – Distributed executor
  focus, including environment packaging pointers.
- [Environment packaging](environment_packaging.md) – Maintaining the shared
  TaskVine tarball.

## Metadata & scenarios

- [Run configuration dataclasses and metadata overview](dataclasses_and_metadata.md)
  – How metadata is stored in dataclasses.
- [Run 2 metadata scenarios guide](run2_scenarios.md) – Scenario/feature
  definitions and validator pointers.
- [Sample metadata reference](sample_metadata_reference.md) – JSON manifest
  schema and troubleshooting tips.

## Architecture & internals

- [Workflow and processor reference](analysis_processing.md) – Processor ↔
  workflow architecture and execution flow.
- [Tuple key audit](tuple_key_audit.md) – 5‑tuple conventions for histogram
  keys across the repository.
- [analysis/topeft_run2/DEVELOPER_NOTES.md](../analysis/topeft_run2/DEVELOPER_NOTES.md)
  – Metadata feature flag behaviours distilled from the processor.

## Legacy / archival

- [analysis/topeft_run2/README.md](../analysis/topeft_run2/README.md) – Legacy
  directory README; pending cleanup during later steps.
- [README_WORKQUEUE.md](../README_WORKQUEUE.md) – Archived Work Queue/TaskVine
  transition notes.
- [README_FITTING.md](../README_FITTING.md) – Datacard/fit instructions retained
  for historical reference.
