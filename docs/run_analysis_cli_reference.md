# `run_analysis.py` CLI and YAML reference

This page summarises every command-line flag and YAML key understood by
[`analysis/topeft_run2/run_analysis.py`](../analysis/topeft_run2/run_analysis.py) via
`RunConfigBuilder`.  Use it as a quick lookup when preparing reproducible
profiles or translating ad-hoc commands into a shared configuration file.  The
[Run analysis configuration flow](run_analysis_configuration.md) guide walks
through the broader execution pipeline, including how the metadata and executor
helpers consume these values.

## How to read this reference

* **Precedence** – When `--options` is supplied, the selected YAML profile is the
  single source of truth.  Every CLI flag listed below is ignored so the run
  stays reproducible.  Drop `--options` when you want to experiment with
  temporary command-line overrides.  `RunConfigBuilder` performs this check
  before reading CLI attributes.【F:analysis/topeft_run2/run_analysis_helpers.py†L347-L415】
* **YAML profiles** – Options files support three top-level sections:
  * `defaults`: a mapping applied unconditionally.
  * `profiles`: a mapping of profile names to overrides.  Select one with
    `path.yml:profile`.  When omitted the builder falls back to
    `default_profile` or the only profile in the file.
  * Additional keys outside these sections act as last-minute overrides.

  These sections are merged in the order listed above so that profiles can
  extend the shared defaults.  See the configuration flow guide for a narrative
  walkthrough of profile usage and metadata integration.【F:analysis/topeft_run2/run_analysis_helpers.py†L347-L415】【F:docs/run_analysis_configuration.md†L1-L63】
* **Types** – Values are normalised into the `RunConfig` dataclass.  YAML lists
  and strings are accepted interchangeably for sequence-like fields such as
  `scenarios` and `wc_list`.  Integers and booleans are coerced from strings when
  needed.【F:analysis/topeft_run2/run_analysis_helpers.py†L103-L306】
* **Metadata defaults** – Omitting `metadata` defers to
  `topeft/params/metadata.yml`, matching the quickstart helpers.  The
  [metadata configuration](run_analysis_configuration.md#metadata-configuration)
  section describes how the planners consume the file.【F:analysis/topeft_run2/workflow.py†L934-L1002】【F:docs/run_analysis_configuration.md†L15-L63】

## Command-line flags and YAML keys

The table below lists each CLI flag, the corresponding YAML key (where
applicable), the accepted type after coercion, the default value embedded in the
parser/dataclass, and extra notes.  Keys marked with * are synonyms provided for
backwards compatibility when writing YAML.

| CLI flag(s) | YAML key(s) | Type after coercion | Default | Notes |
| --- | --- | --- | --- | --- |
| `jsonFiles` (positional) | `jsonFiles`, `json_files`* | list of strings | `[]` | Accepts a single path, a comma-separated string, or a list.  Combined with `prefix` during sample discovery.【F:analysis/topeft_run2/run_analysis.py†L32-L207】【F:analysis/topeft_run2/run_analysis_helpers.py†L103-L232】 |
| `--prefix`, `-r` | `prefix` | string | `""` | Applied to every JSON entry unless the manifest overrides it.  See the configuration flow for redirector usage.【F:analysis/topeft_run2/run_analysis.py†L38-L64】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】【F:docs/run_analysis_configuration.md†L64-L134】 |
| `--executor`, `-x` | `executor` | string | `"work_queue"` | Selects the backend created by `ExecutorFactory`.  Refer to the configuration flow overview for executor comparisons.【F:analysis/topeft_run2/run_analysis.py†L32-L207】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】【F:analysis/topeft_run2/workflow.py†L972-L1002】【F:docs/run_analysis_configuration.md†L64-L134】 |
| `--test`, `-t` | `test` | bool | `False` | Enables reduced-event smoke tests.  Distributed backends may reject this combination as described in the troubleshooting checklist.【F:analysis/topeft_run2/run_analysis.py†L38-L85】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】【F:docs/run_analysis_configuration.md†L214-L240】 |
| `--pretend` | `pretend` | bool | `False` | Parses manifests without executing Coffea tasks.【F:analysis/topeft_run2/run_analysis.py†L38-L85】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】 |
| `--nworkers`, `-n` | `nworkers` | int | `8` | Worker count for the selected executor.【F:analysis/topeft_run2/run_analysis.py†L38-L109】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】 |
| `--chunksize`, `-s` | `chunksize` | int | `100000` | Events per task chunk.【F:analysis/topeft_run2/run_analysis.py†L38-L109】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】 |
| `--nchunks`, `-c` | `nchunks` | optional int | `None` | Limits the number of chunks processed when set.【F:analysis/topeft_run2/run_analysis.py†L38-L109】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】 |
| `--outname`, `-o` | `outname` | string | `"plotsTopEFT"` | Histogram filename stem.【F:analysis/topeft_run2/run_analysis.py†L38-L132】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】 |
| `--outpath`, `-p` | `outpath` | string | `"histos"` | Output directory for results.【F:analysis/topeft_run2/run_analysis.py†L38-L132】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】 |
| `--treename` | `treename` | string | `"Events"` | Input TTree name.【F:analysis/topeft_run2/run_analysis.py†L38-L132】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】 |
| `--metadata` | `metadata`, `metadata_path`* | optional string | `None` | When omitted the workflow loads `topeft/params/metadata.yml`.  See the metadata configuration guide for structure details.【F:analysis/topeft_run2/run_analysis.py†L38-L132】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】【F:analysis/topeft_run2/workflow.py†L934-L1002】【F:docs/run_analysis_configuration.md†L15-L63】 |
| `--do-errors` | `do_errors` | bool | `False` | Persists quadratic weights (`w**2`) for uncertainty propagation.【F:analysis/topeft_run2/run_analysis.py†L64-L163】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】 |
| `--do-systs` | `do_systs` | bool | `False` | Enables systematic planning based on metadata definitions.【F:analysis/topeft_run2/run_analysis.py†L64-L163】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】【F:docs/run_analysis_configuration.md†L136-L211】 |
| `--split-lep-flavor` | `split_lep_flavor` | bool | `False` | Splits histogram categories by lepton flavour.  Mentioned in the summary verbosity description for awareness.【F:analysis/topeft_run2/run_analysis.py†L64-L163】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】 |
| `--summary-verbosity` | `summary_verbosity` | string (`"none"`, `"brief"`, `"full"`) | `"brief"` | Controls the textual run summary printed before execution.【F:analysis/topeft_run2/run_analysis.py†L132-L173】【F:analysis/topeft_run2/run_analysis_helpers.py†L188-L238】【F:analysis/topeft_run2/workflow.py†L972-L1002】 |
| `--log-tasks` | `log_tasks` | bool | `False` | Emits a one-line log for each submitted histogram task.【F:analysis/topeft_run2/run_analysis.py†L173-L206】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】 |
| `--scenario` (repeatable) | `scenarios` | list of strings | `[]` (resolved to `['TOP_22_006']` when empty) | Scenarios map to channel groups in the metadata file.【F:analysis/topeft_run2/run_analysis.py†L173-L206】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】【F:analysis/topeft_run2/workflow.py†L972-L1016】【F:docs/run_analysis_configuration.md†L15-L63】 |
| `--skip-sr` | `skip_sr` | bool | `False` | Drops all signal-region categories during planning.【F:analysis/topeft_run2/run_analysis.py†L173-L206】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】 |
| `--skip-cr` | `skip_cr` | bool | `False` | Drops all control-region categories during planning.【F:analysis/topeft_run2/run_analysis.py†L173-L206】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】 |
| `--do-np` | `do_np` | bool | `False` | Requests nonprompt estimation after histogram production.【F:analysis/topeft_run2/run_analysis.py†L173-L206】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】【F:analysis/topeft_run2/workflow.py†L906-L976】 |
| `--do-renormfact-envelope` | `do_renormfact_envelope` | bool | `False` | Requires `do_np` and `do_systs` and applies the renormalisation/factorisation envelope post-processing.【F:analysis/topeft_run2/run_analysis.py†L173-L206】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】【F:analysis/topeft_run2/workflow.py†L918-L965】 |
| `--wc-list` | `wc_list` | list of strings | `[]` | Wilson coefficients to evaluate; duplicates are removed while preserving order.【F:analysis/topeft_run2/run_analysis.py†L173-L206】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L336】 |
| `--ecut` | `ecut` | optional float | `None` | Event-level energy cut in GeV.【F:analysis/topeft_run2/run_analysis.py†L173-L206】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】 |
| `--port` | `port` | string | `"9123-9130"` | Work Queue port or range.  String inputs are passed through, sequences are normalised to `min-max`.【F:analysis/topeft_run2/run_analysis.py†L173-L206】【F:analysis/topeft_run2/run_analysis_helpers.py†L140-L187】【F:analysis/topeft_run2/run_analysis_helpers.py†L240-L318】 |
| `--options` | *(YAML only)* | string | `None` | Selects the YAML options file and optional profile.  When set, the CLI values above are ignored in favour of the YAML content.【F:analysis/topeft_run2/run_analysis.py†L198-L206】【F:analysis/topeft_run2/run_analysis_helpers.py†L319-L415】 |

### YAML-only helper keys

These keys shape how the YAML profile is applied but do not map to explicit CLI
flags.

| YAML key | Type | Default | Notes |
| --- | --- | --- | --- |
| `defaults` | mapping | `{}` | Baseline values merged before profiles and ad-hoc overrides.【F:analysis/topeft_run2/run_analysis_helpers.py†L347-L392】 |
| `profiles` | mapping of mappings | `{}` | Named overlays.  Combine with `default_profile` or `path.yml:profile` to choose one.【F:analysis/topeft_run2/run_analysis_helpers.py†L347-L392】 |
| `default_profile` | string | `None` | Automatically selected when present and `path.yml:profile` is not supplied.【F:analysis/topeft_run2/run_analysis_helpers.py†L360-L383】 |

For executor-specific tuning, extend the builder with additional keys as
outlined in the [Run analysis configuration flow](run_analysis_configuration.md#key-helpers-and-extension-points)
reference section.【F:docs/run_analysis_configuration.md†L200-L276】
