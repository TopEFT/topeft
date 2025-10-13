# Run 2 quickstart pipeline

The Run 2 quickstart helpers are designed to let new users validate their
software environment without having to digest the entire set of options exposed
by `analysis/topeft_run2/run_analysis.py`.  They configure the
:class:`analysis.topeft_run2.workflow.RunWorkflow` and the
:class:`analysis.topeft_run2.analysis_processor.AnalysisProcessor` with
conservative defaults, load a small histogram list and keep the executor limited
to the local ``futures`` backend.

The quickest way to run the helper is via the dedicated module entry point::

    python -m topeft.quickstart input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
        --prefix root://cmsxrootd.fnal.gov/ \
        --output quickstart-output

The command performs two steps:

1. :func:`analysis.topeft_run2.quickstart.prepare_samples` resolves the JSON file
   (directories and ``.cfg`` files are also accepted), validates that the file is
   reachable and that the requested metadata scenario exists, and prints a
   summary of the number of samples and events discovered.
2. :func:`analysis.topeft_run2.quickstart.run_quickstart` launches the Coffea
   processor with a single histogram (`lj0pt` by default), ``nchunks=2`` and a
   local futures executor.  This keeps the runtime to a couple of minutes even on
   a laptop.

The output pickle is stored in the directory provided through ``--output`` with
an ``outname`` of ``quickstart`` (so you can expect something like
``quickstart-output/quickstart.pkl.gz``).

## Understanding the quickstart inputs

### Samples JSON

The quickstart helpers use the same JSON description as the full workflow.  A
minimal example targeting a public CMS Run 2 file is shown below:

```json
{
  "xsec": 0.2151,
  "year": "2017",
  "treeName": "Events",
  "histAxisName": "ttHJet_UL17_quickstart",
  "files": [
    "/store/mc/RunIISummer20UL17NanoAODv9/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/\n     NANOAODSIM/106X_mc2017_realistic_v9-v1/130000/0A1884EE-9F60-6F4F-B423-13F6179B611D.root"
  ],
  "nEvents": 418590,
  "nGenEvents": 418590,
  "nSumOfWeights": 418590,
  "isData": false
}
```

Saving the snippet as ``ttz_quickstart.json`` allows you to run the helper with
``python -m topeft.quickstart ttz_quickstart.json --prefix root://cmsxrootd.fnal.gov/``.
The prefix ensures that the remote EOS path is opened via XRootD.

### Metadata scenarios and feature tags

The Run 2 metadata (``topeft/params/metadata.yml``) defines multiple channel
scenarios.  The quickstart helpers use the ``TOP_22_006`` scenario by default,
which matches the reinterpretation categories used in the main analysis.  Use
``--scenario`` to activate other bundles or pass the option multiple times to
combine them.  Feature tags (``--feature requires_tau`` for example) can be used
in addition to scenarios to enable dedicated regions advertised by the metadata
file.  For a step-by-step walkthrough of each scenario—including how to combine
bundles and feature tags in YAML or on the command line—see the
[Run 2 metadata scenarios guide](run2_scenarios.md).

During :func:`prepare_samples`, both the scenario names and feature tags are
validated via the :class:`analysis.topeft_run2.workflow.ChannelPlanner`.  Passing
unknown identifiers raises a ``ValueError`` before any processing starts so that
misconfigurations are caught early.

### Next steps {#run2-quickstart-next-steps}

!!! note "Next steps"
    Keep the workflow lightweight while layering on additional validation
    targets.  The [Run 2 metadata scenarios guide](run2_scenarios.md) walks
    through each bundle in detail, including how to mix feature tags.  To reuse
    this helper for the tau- or forward-focused reinterpretations, add the
    scenario switches directly on the command line:

    ```bash
    python -m topeft.quickstart \
        input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
        --prefix root://cmsxrootd.fnal.gov/ \
        --scenario tau_analysis
    ```

    Swap ``tau_analysis`` for ``fwd_analysis`` to validate the forward
    categories, or request both in a single run when you want to confirm the
    combined configuration:

    ```bash
    python -m topeft.quickstart \
        input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
        --prefix root://cmsxrootd.fnal.gov/ \
        --scenario TOP_22_006 \
        --scenario tau_analysis \
        --scenario fwd_analysis
    ```

    To keep the combined scenario close at hand for the full workflow, mirror
    the same list in a YAML override for :mod:`run_analysis.py` before scaling
    up.  Start by copying ``analysis/topeft_run2/configs/fullR2_run.yml`` to
    ``analysis/topeft_run2/configs/fullR2_run_tau_fwd.yml`` so all of the other
    defaults remain aligned, then update the new file's sample list and
    scenarios as shown below:

    ```yaml
    # analysis/topeft_run2/configs/fullR2_run_tau_fwd.yml
    jsonFiles:
      - ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json
    scenarios:
      - TOP_22_006
      - tau_analysis
      - fwd_analysis
    ```

    Launching ``python run_analysis.py --options configs/fullR2_run_tau_fwd.yml``
    keeps the validation fast while aligning the configuration with the
    quickstart dry run.

### Systematic toggles

Systematic variations are disabled by default to keep the run lightweight.
Add ``--do-systs`` when you want to exercise the same code paths used in the
full analysis.  The helper automatically inspects
``topeft/params/metadata.yml`` to determine which sum-of-weights variations are
available and enables them in the Coffea processor.  Because only a single
histogram is processed, enabling systematics keeps the runtime manageable while
still producing variations in the output pickle.

When you are ready to explore more advanced options, you can pass the same
arguments used by the main workflow (``--split-lep-flavor``, ``--skip-sr``,
``--skip-cr`` and custom ``--wc`` lists are all supported by the helper).  Any
run generated through the quickstart utilities is reproducible via the returned
:class:`analysis.topeft_run2.run_analysis_helpers.RunConfig`, which records all
relevant switches.
