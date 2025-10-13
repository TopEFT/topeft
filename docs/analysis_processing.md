# Workflow and processor reference

This page documents how the ``analysis/topeft_run2`` helpers cooperate once the
configuration is built.  It focuses on the execution phase that starts inside
:func:`analysis.topeft_run2.workflow.run_workflow`, covers the systematic
handshake with :class:`analysis.topeft_run2.analysis_processor.AnalysisProcessor`,
and highlights the primary extension points.

The high-level flow is:

1. :class:`analysis.topeft_run2.workflow.RunWorkflow` validates the
   :class:`analysis.topeft_run2.run_analysis_helpers.RunConfig` generated during
   the CLI/YAML merge.
2. :class:`analysis.topeft_run2.run_analysis_helpers.SampleLoader` expands the
   input manifests and returns a normalized ``samplesdict`` together with the
   expanded file list per sample.
3. :class:`analysis.topeft_run2.workflow.ChannelPlanner` resolves metadata
   scenarios and feature tags, producing channel dictionaries that are attached
   to each histogram task.
4. :class:`analysis.topeft_run2.workflow.HistogramPlanner` enumerates histogram
   combinations by crossing samples, channel metadata, Coffea applications, and
   systematic toggles.  The result is a :class:`HistogramPlan` which records the
   ``(sample, channel, variable, application, systematic)`` tuples that will be
   processed.
5. An :class:`analysis.topeft_run2.workflow.ExecutorFactory` creates the
   requested backend runner (``futures``, ``work_queue`` or ``taskvine``).  Each
   histogram task is turned into an :class:`AnalysisProcessor` instance with the
   correct per-sample metadata and systematic configuration before being
   submitted to the executor.

## AnalysisProcessor responsibilities

The :class:`AnalysisProcessor` class is a Coffea processor responsible for
building selections, applying corrections, and filling histograms.  A few key
aspects are worth keeping in mind when extending it:

* **Channel metadata** – the processor expects the ``channel_dict`` passed by
  the workflow to provide ``jet_selection``, ``chan_def_lst``, ``lep_flav_lst``,
  ``appl_region`` and an optional ``features`` collection.  The flags control
  specialised paths such as ``requires_tau`` or ``offz_split``.
* **Systematics** – the ``available_systematics`` mapping is produced by the
  :class:`SystematicsHelper` inside the workflow.  It contains the full matrix of
  weight variations, object shifts, and fake-factor toggles that the metadata
  describes.  The processor caches both the tuple lists (for histogram
  enumeration) and set views (for fast membership checks) so that custom
  systematic categories can be added without rewriting the ``process`` method.
* **Golden JSONs** – when processing data samples the workflow injects the
  appropriate ``golden_json_path`` so that :class:`coffea.lumi_tools.LumiMask`
  can be used directly inside the ``process`` method.  Missing entries raise a
  ``ValueError`` before any event loop is started.
* **Histogram keys** – the processor accepts either a single 5-tuple or a list
  of tuples per systematic label.  The normalization in the constructor ensures
  that the internal representation always uses ordered tuples, making it safe to
  extend the histogram planning logic without breaking call sites.

Because the constructor performs strict validation (checking for ``None``
arguments, verifying tuple lengths, etc.), deviations are caught early.  The
workflow reuses the same processor class for both quickstart and production runs,
so new options should prefer optional keyword arguments with sensible defaults.

## Systematic catalogue

Systematic variations originate from two sources and meet inside the workflow:

* **Weight variations** – collected via
  :func:`analysis.topeft_run2.run_analysis_helpers.weight_variations_from_metadata`
  and attached to each sample during ``SampleLoader.load``.  The workflow checks
  that every MC sample declares the required ``nSumOfWeights`` entries.
* **Application variations** – defined in ``topeft/params/metadata.yml`` under
  the ``systematics`` block.  :class:`SystematicsHelper` projects them onto the
  selected channel features and returns a dictionary with the variations grouped
  by type.  The processor interprets those labels to register weight and object
  shifts.

The workflow always starts the summary with compact bullet lists describing the
unique samples, channel/application pairs, variables, and systematic labels that
will be processed when ``RunConfig.summary_verbosity`` is ``"brief"`` or
``"full"``.  Selecting ``"full"`` retains the per-combination table and
structured YAML (or JSON) payload printed previously.  When
``split_lep_flavor`` is enabled, the detailed table also reminds readers that
flavored channels reuse the processor task constructed for their base channel.
These summaries are particularly useful when validating that the requested
systematics match team expectations after editing metadata.

## Extending the workflow

The helpers were designed to be replaced piecemeal.  A few concrete patterns are
listed below:

* **Custom executor backends** – subclass :class:`ExecutorFactory` or supply an
  object exposing a compatible ``create_runner`` method.  The runner only needs
  ``submit`` and ``finish`` methods, making it straightforward to integrate
  site-specific schedulers.
* **Selective histogram planning** – provide an alternate
  :class:`HistogramPlanner` that filters variables or channels based on the
  :class:`RunConfig`.  Because the planner builds :class:`HistogramTask`
  instances, new metadata (for example, systematic tags) can be attached without
  touching the executor or processor code.
* **Channel overrides** – wrap :class:`ChannelPlanner` to inject or filter
  feature tags programmatically.  This is useful for dry runs that focus on
  validation categories or to introduce new experimental regions while metadata
  changes are still under review.

## Reusing the workflow from Python

While ``run_analysis.py`` remains the canonical entry point, the module-level
API makes it easy to script runs from notebooks or CI tasks:

```python
from analysis.topeft_run2.run_analysis import build_parser
from analysis.topeft_run2.run_analysis_helpers import RunConfigBuilder
from analysis.topeft_run2.workflow import run_workflow

parser = build_parser()
args = parser.parse_args([
    "input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json",
    "--options", "analysis/topeft_run2/examples/options.yml",
])

builder = RunConfigBuilder()
config = builder.build(args, options_path=args.options, options_profile=args.options_profile)
run_workflow(config=config)
```

The ``config`` returned here is the same structure used by the quickstart
utilities.  Persisting it (for example via ``dataclasses.asdict``) provides a
compact audit trail that complements the stored output pickle.
