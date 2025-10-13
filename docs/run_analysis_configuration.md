# Run analysis configuration flow

The ``analysis/topeft_run2/run_analysis.py`` entry point stitches together the
command line, metadata bundles, and Coffea execution helpers that drive the Run 2
analysis.  This page documents how the configuration is normalized, how the
helpers cooperate, and which extension points are available when adapting the
workflow to new samples or channels.

The guide follows the same YAML-first workflow presented in the quickstart
examples (``docs/quickstart_run2.md`` and ``docs/quickstart_top22_006.md``).  The
YAML options file is optional, but using it keeps custom runs reproducible and
makes it easy to share presets with collaborators.

## Overview of the configuration pipeline

``run_analysis.py`` executes the steps below when launched from the repository
root::

    python analysis/topeft_run2/run_analysis.py \
        input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
        --options analysis/topeft_run2/examples/options.yml:sr

1. **Argument parsing** – :func:`analysis.topeft_run2.run_analysis.build_parser`
   defines the CLI options that downstream helpers understand.  The parser keeps
   backwards compatibility with the historical ``jsonFiles`` positional argument
   while advertising the YAML-driven ``--options`` knob, which accepts either
   ``path.yml`` or ``path.yml:profile`` when you need a specific preset.
2. **YAML overrides** – :class:`analysis.topeft_run2.run_analysis_helpers.RunConfigBuilder`
   reads the options file (if provided) before evaluating CLI values.  The YAML
   supports three blocks:

   * ``defaults`` – baseline values that are always applied.
   * ``profiles`` – named overlays that can be enabled via ``path.yml:profile``
     or automatically when only one profile exists.
   * Top-level keys – additional overrides that do not fit in the two sections
     above.  They are merged last so that ad-hoc experiments can live next to
     reusable defaults.

   When an options file is supplied, the YAML becomes the single source of truth.
   Drop ``--options`` if you need to experiment with ad-hoc CLI flags; otherwise
   bake the desired configuration into the file before launching the workflow.

3. **Configuration normalization** – the builder converts all inputs into a
   :class:`analysis.topeft_run2.run_analysis_helpers.RunConfig` dataclass.  The
   dataclass mirrors the traditional CLI options but guarantees consistent types
   (lists instead of comma separated strings, ``None`` for unset values, etc.).
4. **Sample discovery** – :class:`analysis.topeft_run2.run_analysis_helpers.SampleLoader`
   expands positional inputs into concrete JSON files, parsing ``.cfg`` bundles
   and directories when necessary.  The loader attaches redirectors, validates
   files, and normalizes numeric metadata.
5. **Metadata planning** – :class:`analysis.topeft_run2.workflow.ChannelPlanner`
   and :class:`analysis.topeft_run2.workflow.HistogramPlanner` translate the
   selected metadata scenarios and variable definitions into the list of
   histogram tasks.  This is where ``--scenario``, ``--skip-sr`` and similar
   knobs take effect.
6. **Execution** – :class:`analysis.topeft_run2.workflow.ExecutorFactory`
   instantiates the selected backend (``futures``, ``work_queue`` or
   ``taskvine``).  Each histogram task yields an
   :class:`analysis.topeft_run2.analysis_processor.AnalysisProcessor` instance
   configured for the corresponding sample and channel.  Progress is summarized
   according to ``summary_verbosity``—``"brief"`` prints bullet lists of the
   unique samples, channel/application pairs, variables, and systematics that
   will be processed, while ``"full"`` prepends the same lists to the
   combination table and structured dump (including a reminder when
   ``split_lep_flavor`` is active).  These details are optionally mirrored in
   the single-line ``log_tasks`` messages.

The helpers are designed so that the resulting :class:`RunConfig` can be stored
or passed around.  For example, the quickstart workflow returns the configuration
it used so that you can plug the same object into :func:`run_workflow` later.

## Systematic handling

Systematic switches are managed collaboratively by the helpers:

* ``RunConfig.do_systs`` enables systematic planning.  The builder honours YAML
  booleans (``true``/``false``) and CLI flags.
* :func:`analysis.topeft_run2.run_analysis_helpers.weight_variations_from_metadata`
  inspects ``topeft/params/metadata.yml`` to identify all available sum-of-weight
  variations.  Supplying a YAML options file with a ``systematics`` block lets
  you add or restrict variations on a per-profile basis.
* :class:`SampleLoader` makes the variations available on each sample entry so
  that :class:`analysis.topeft_run2.workflow.RunWorkflow` can validate the
  metadata before any Coffea tasks are submitted.
* :class:`analysis.topeft_run2.workflow.SystematicsHelper` (instantiated inside
  the workflow) cross-references the metadata with the scenario-provided feature
  flags and the requested year, building the final structure that
  :class:`AnalysisProcessor` consumes.

When ``--do-renormfact-envelope`` is enabled, the workflow enforces the presence
of ``--do-systs`` and ``--do-np`` because the renormalization/factorization
envelope is only meaningful after the non-prompt application integrals are
available.  These checks happen in :meth:`RunWorkflow._validate_config` so that
misconfigurations fail fast.

## Key helpers and extension points

The table below summarises the most common extension hooks:

| Helper | Responsibility | How to extend |
| ------ | -------------- | ------------- |
| :class:`RunConfigBuilder` | Merge CLI, defaults, and YAML into a :class:`RunConfig`. | Subclass the builder and override :meth:`build` to recognise additional YAML keys (for example, executor-specific settings).  The CLI parser can expose the same flag so that existing scripts keep working. |
| :class:`SampleLoader` | Resolve JSON/CFG inputs and normalize metadata. | Provide a custom ``SampleLoader`` to support other manifest formats (for example, CSV).  The replacement object must offer ``collect`` and ``load`` methods returning the same structures. |
| :class:`analysis.topeft_run2.workflow.ChannelPlanner` | Activate channels according to metadata scenarios. | Extend :class:`topeft.modules.channel_metadata.ChannelMetadataHelper` or wrap the planner to insert additional filters (for example, dropping jet categories). |
| :class:`analysis.topeft_run2.workflow.HistogramPlanner` | Enumerate histogram combinations for execution. | Pass a custom planner that rewrites the histogram list (for example, sampling only a subset of variables) before the workflow starts the executor. |
| :class:`analysis.topeft_run2.workflow.ExecutorFactory` | Configure the execution backend. | Supply a factory that sets up distributed resources (for example, a site-specific Work Queue profile).  The factory only needs to return an object with a ``create_runner`` method. |
| :func:`analysis.topeft_run2.workflow.run_workflow` | Convenience wrapper mirroring the CLI. | Import the function and feed it the :class:`RunConfig` returned by the quickstart helpers when you want to programmatically drive the workflow from notebooks or scripts. |

### Adding new configuration values

1. Extend the CLI parser in ``run_analysis.py`` with the new flag.
2. Update the ``field_specs`` mapping in :class:`RunConfigBuilder` so that the
   value is recorded inside :class:`RunConfig`.  Reuse the existing coercion
   helpers when possible (``coerce_bool``, ``coerce_int`` or
   ``normalize_sequence``).
3. Add documentation for the new key to the YAML options file used in your team
   and mention the knob in the quickstart walkthrough if it helps new users.
4. Use the value in either :class:`RunWorkflow` or
   :class:`AnalysisProcessor`.  Because :class:`RunConfig` is a dataclass, adding
   optional attributes is backwards compatible and automatically reflected in the
   quickstart return value.

### Using the YAML quickstart workflow

The quickstart helper (``python -m topeft.quickstart``) produces a YAML snippet
when called with ``--emit-options``.  Saving the output provides a reproducible
baseline that can be tweaked manually before feeding it to ``run_analysis.py``.
Common adjustments include enabling ``do_systs``, tweaking
``summary_verbosity`` to show only the bullet lists or the full table, and
switching executors once the run is ready to scale beyond the local machine.

## Troubleshooting checklist

* If the workflow aborts with ``Missing weight variation`` exceptions, confirm
  that the requested variations are listed under ``systematics`` in the metadata
  bundle and that the JSON files contain the corresponding ``nSumOfWeights``
  entries.
* When ``--test`` is requested with a distributed executor, the workflow raises
  an error because only the local ``futures`` backend is configured for fast
  validation runs.  Drop the flag or switch executors when performing smoke tests
  on remote resources.
* ``FileNotFoundError`` from :class:`SampleLoader` usually means that relative
  paths were provided.  Always invoke ``run_analysis.py`` from the repository
  root or supply absolute paths.

With these pieces in place you can mix and match quickstart presets and YAML
profiles while keeping the run history compact and shareable.  When you need to
experiment with CLI overrides, drop ``--options`` so the command-line values are
honoured for that run.
