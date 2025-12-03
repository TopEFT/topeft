# Run configuration dataclasses and metadata overview

This guide explains how the Run 2 workflow records configuration in data classes,
how histogram planning consumes those structures, and how the JSON/YAML metadata
feeds each stage.  It complements the end-to-end walkthrough in
[Run analysis configuration flow](run_analysis_configuration.md) and the
[sample metadata reference](sample_metadata_reference.md).

## `RunConfig` and the builder pipeline

The workflow persists configuration in the `RunConfig` dataclass introduced by
`analysis.topeft_run2.run_analysis_helpers`.  The class groups CLI, YAML, and
default settings into immutable attributes (per [PEP 557](https://peps.python.org/pep-0557/))
so downstream helpers can rely on consistent types and defaults.  Key fields
capture input discovery, metadata switches, executor choices, and futures-tuning
knobs that mirror the CLI flags. 【F:analysis/topeft_run2/run_analysis_helpers.py†L330-L371】

`RunConfigBuilder` is responsible for merging CLI namespaces, YAML option files,
and default values.  The builder maps user-facing keys such as `--scenarios` or
`metadata:` entries to the corresponding dataclass attributes, applying
normalisation helpers (`coerce_bool`, `normalize_sequence`, etc.) before storing
them. 【F:analysis/topeft_run2/run_analysis_helpers.py†L374-L467】  When the
`--options` argument references a YAML document, the builder evaluates the
`defaults`, `profiles`, and pass-through keys in order, making profile overlays
behave predictably. 【F:analysis/topeft_run2/run_analysis_helpers.py†L491-L539】
If no YAML is provided, the builder falls back to CLI overrides, deduplicating
scenario and Wilson-coefficient selections before returning the final
configuration. 【F:analysis/topeft_run2/run_analysis_helpers.py†L541-L599】

Because `RunConfig` is a dataclass, existing code can call `dataclasses.asdict`
for auditing or serialization, and new optional attributes can be added without
breaking existing runs.  The configuration object is passed unchanged through
`SampleLoader`, `ChannelPlanner`, and the executor factory so that each helper
reads the same authoritative settings.

## Histogram planning dataclasses

Histogram orchestration relies on a trio of frozen dataclasses declared in
`analysis.topeft_run2.workflow`:

- `HistogramTask` captures the per-job inputs (sample name, clean channel,
  application, metadata and systematic availability) that will be fed into the
  Coffea processor. 【F:analysis/topeft_run2/workflow.py†L293-L307】
- `HistogramCombination` records the `(sample, channel, variable, application,
  systematic)` tuple for human-readable summaries. 【F:analysis/topeft_run2/workflow.py†L309-L317】
- `HistogramPlan` packages the final task list, the histogram names that were
  activated, and the flattened summary for logging. 【F:analysis/topeft_run2/workflow.py†L320-L326】

`HistogramPlanner` assembles these structures by crossing the selected metadata
channels with variable definitions, available systematics, and sample-specific
variations. 【F:analysis/topeft_run2/workflow.py†L329-L494】  During planning the
class honours channel-specific whitelists/blacklists, handles optional flavour
splitting, and ensures that each combination is represented once in the summary.
The resulting `HistogramPlan` is consumed by the executor factory, which keeps
the dataclass payload intact when scheduling Coffea jobs.

## Sample JSON metadata schema

Per-sample manifests are loaded via the `SampleLoader` helper.  During
`load(...)` the helper enforces the required keys (`xsec`, `nEvents`,
`nGenEvents`, `nSumOfWeights`, `files`, `histAxisName`, `treeName`, `year`, and
`isData`) and coerces each value into the expected Python type before attaching a
redirector prefix. 【F:analysis/topeft_run2/run_analysis_helpers.py†L202-L248】
Any metadata-driven weight variations listed in the manifest are also coerced to
floats so downstream code can perform numeric comparisons without additional
parsing. 【F:analysis/topeft_run2/run_analysis_helpers.py†L243-L247】

The optional variation keys themselves are derived from the metadata YAML via
`weight_variations_from_metadata`, which scans the `systematics` section for
`sum_of_weights` entries. 【F:analysis/topeft_run2/run_analysis_helpers.py†L294-L327】
Supplying `--options` profiles that disable systematics or editing the metadata
file therefore changes both the JSON validation rules and the systematic
combinations enumerated later by `HistogramPlanner`.

For concrete manifest examples and troubleshooting checklists, see the
[sample metadata reference](sample_metadata_reference.md).

## YAML metadata catalogue

The canonical metadata bundle (`topeft/params/metadata.yml`) contains several
sections that drive channel discovery and systematic planning:

- `golden_jsons` maps each year to the luminosity mask used when running on
  collision data. 【F:topeft/params/metadata.yml†L1-L7】
- `channels.groups` defines scenario-specific region lists, including jet-bin
  expansions, histogram include/exclude lists, and separate application tags for
  data vs. MC. 【F:topeft/params/metadata.yml†L9-L111】
- `variables` (further down in the file) describes histogram axes, binning, and
  callable expressions that populate the planner’s `variable_info` payloads.
- `scenarios` binds friendly scenario names to channel groups so the CLI/YAML
  `scenarios` knobs translate into concrete region selections.
- `systematics` enumerates weight and object variations, each of which can
  declare `sum_of_weights` keys that the loader must find in JSON manifests.

`ChannelPlanner` resolves the requested scenarios into concrete region and
feature selections, preserving metadata that the processor needs later.
【F:analysis/topeft_run2/workflow.py†L76-L275】  The planner records per-channel
whitelists, blacklists, application tags, and flavour metadata, which are then
consumed by `HistogramPlanner` when building each `HistogramTask`.

Taken together, the dataclasses form a consistent bridge between user-facing
configuration, metadata catalogues, and the Coffea execution pipeline.  Editing
one layer (for example, adding a new systematic in the YAML) automatically
propagates through the JSON schema validation, histogram planning, and task
summaries without additional glue code.

## Metadata editing checklist

Use this checklist when introducing or modifying Run‑2 metadata:

1. **Edit locations**
   - Clone ``topeft/params/metadata.yml`` to a tracked file (for example
     ``analysis/topeft_run2/configs/metadata_dev.yml``) before making changes.
   - Update scenario/group compositions in
     ``analysis/metadata/run2_scenarios.yaml`` and related metadata files under
     ``analysis/metadata/`` when adding new channel bundles.
   - Point YAML options (such as ``analysis/topeft_run2/configs/fullR2_run.yml``)
     at the new metadata file via the ``metadata`` key so the presets stay in sync.
2. **Validate the updates**
   - Run the validators under ``scratch/`` after each change:
     - ``python scratch/validate_run2_scenarios_step5.py`` – confirms scenario
       definitions and prints per-scenario counts.
     - ``python scratch/validate_run2_groups_step5.py`` – compares YAML group
       counts against the legacy JSON snapshot.
     - ``python scratch/validate_run2_datacard_channels_step5.py`` – checks that
       datacard channel counts match the expected SR totals.
   - These scripts assume the Run‑2 metadata lives under ``analysis/metadata`` and
     should be executed from the repository root so relative imports resolve.
3. **Smoke-test the workflow**
   - Use the [Run 2 quickstart pipeline](quickstart_run2.md) or
     ``analysis/topeft_run2/full_run.sh`` with the updated YAML to confirm the
     new metadata runs end to end (``--dry-run`` first, then a short futures pass).
   - Keep scenario names consistent with ``run2_scenarios.yaml`` so CLI
     invocations and presets (like ``fullR2_run.yml:sr``) continue to work.
4. **Share presets wisely**
   - For experimental configurations, create a dedicated YAML file (for example
     ``fullR2_run_tau_dev.yml``) instead of editing production presets directly.
     Reference the new metadata path in that file and document it alongside the
     change so collaborators can reproduce your setup.
