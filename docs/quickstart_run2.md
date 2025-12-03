# Run 2 quickstart pipeline

This is the single entry point for running a small Run‑2 job end to end: prepare
an input JSON, launch the helper or wrapper, then make a first plot. It assumes
you have already read the [workflow & YAML hub](workflow_and_yaml_hub.md) and
set up the shared `coffea2025` environment plus the sibling
[`topcoffea`](https://github.com/TopEFT/topcoffea) checkout.

## Prerequisites

1. Follow the “Start here” hub to create/activate the shared environment, install
   both repositories in editable mode, and build the TaskVine environment
   tarball (when needed).
2. Keep `topcoffea` on the `ch_update_calcoffea` branch (or matching tag) so
   cache-free jet/MET corrections match the workflow expectations.
3. Pick a sample manifest such as
   `input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json`. The
   helper accepts directories and `.cfg` bundles as well; see
   [sample_metadata_reference.md](sample_metadata_reference.md) for schema
   details.

With the prerequisites in place, you can validate the workflow locally in a few
minutes using the quickstart module below.

### Cache-free jet/MET corrections

Run 2 processing now assumes the cache-free jet and MET correction interfaces
shipped with the `ch_update_calcoffea` branch of `topcoffea`.  The helper and
full workflow no longer rely on NanoEvents caches when building corrected jets
or propagating jet variations into MET; the updated APIs materialise the
corrections directly.  Make sure your environment uses the matching
`topcoffea` checkout (or a release containing the same cache-free helpers) so
the processor can run end-to-end without attaching a `lazy_cache` to the
NanoEvents object.

Run 2 histogramming expects Awkward's native implementations of helpers such as
`ak.stack` and relies on in-memory arithmetic instead of array-of-arrays
coercions. Ensure your environment pulls a recent Awkward build (the Coffea
2025.7 bundle does) so the masks remain regular arrays; Awkward 2.x is the
supported baseline. B-tag and jet multiplicities are derived directly from
numeric fields (for example, ``ak.num(cleaned_jets.pt)`` with ``ak.fill_none``
and integer casts) rather than cached Records, so custom pre-processing that
produces jagged or optional counts will raise during histogram filling.

## Step 1 – Run the quickstart helper

Launch the lightweight helper module from the repository root:

```bash
python -m topeft.quickstart \
    input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
    --prefix root://cmsxrootd.fnal.gov/ \
    --output quickstart-output
```

The helper:

1. Resolves the manifest (directories and `.cfg` bundles also work), validates
   the requested scenario, and prints a summary of the discovered samples.
2. Runs a trimmed `AnalysisProcessor` job with `nchunks=2`, a local futures
   executor, and a single histogram so the end-to-end test finishes quickly.

The output histogram pickle is stored in the requested directory:
`quickstart-output/quickstart.pkl.gz`.

## Step 2 – Try the unified run wrapper

When you want to exercise the full workflow, use
`analysis/topeft_run2/full_run.sh`. It expands the `run2`/`run3` presets,
resolves cfg/json bundles, and selects sensible defaults for both TaskVine and
futures executors.

```
cd analysis/topeft_run2
./full_run.sh --sr -y UL17 \
    --executor futures \
    --samples ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
    --outdir histos/local_debug \
    --tag quickstart --dry-run
```

Drop `--dry-run` to execute the command. TaskVine is still the default backend;
append `--executor taskvine` (and make sure workers connect with the advertised
manager name) to run distributed jobs. The wrapper honours the `fullR2_run.yml`
profiles by default and selects SR/CR bundles based on the `--sr/--cr` choice.

## Step 3 – Plot the results

Once a helper or wrapper run finishes you will have tuple-keyed histogram
pickles such as `histos/local_debug/plotsTopEFT.pkl.gz` (wrapper) or
`quickstart-output/quickstart.pkl.gz` (module). Use the existing plotting helper
to turn them into quick validation plots:

```bash
cd analysis/topeft_run2
python make_cr_and_sr_plots.py \
    -f histos/local_debug/plotsTopEFT.pkl.gz \
    -o plots/local_debug \
    -n plots \
    -y 2017 \
    --skip-syst
```

For notebook-driven checks you can also materialise the tuple summaries from
Python:

```python
import gzip, pickle
from topeft.modules.runner_output import materialise_tuple_dict

with gzip.open("histos/local_debug/plotsTopEFT.pkl.gz", "rb") as handle:
    tuple_summary = materialise_tuple_dict(pickle.load(handle))
print(list(tuple_summary.items())[:2])
```

Both paths assume the stored histogram tuples follow the canonical
`(variable, channel, application, sample, systematic)` convention described in
[analysis_processing.md](analysis_processing.md).

## Understanding the quickstart inputs

### Samples JSON

The quickstart helpers use the same JSON description as the full workflow.
Refer to the [sample metadata reference](sample_metadata_reference.md) for a
complete breakdown of required keys, optional systematic sums of weights, and
common validation fixes.  A minimal example targeting a public CMS Run 2 file is
shown below:

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

## Metadata configuration

Quickstart jobs lean entirely on ``topeft/params/metadata.yml`` to decide which
regions, histograms, and weight variations to run.  Before editing the file,
consult the [metadata configuration guide](run_analysis_configuration.md#metadata-configuration)
for a full breakdown of each section.  The highlights are:

* ``channels.groups`` enumerate the signal/control regions and include any
  per-region histogram include/exclude lists.
* ``variables`` provides the histogram catalogue (binning, labels, callable
  definitions) consumed by the Coffea processor.
* ``scenarios`` expose friendly names that map to channel bundles and power the
  ``--scenario`` flag used throughout the quickstarts.
* ``systematics`` lists the available weight/object variations.  The helper
  inspects this block whenever ``--do-systs`` is requested.

To test custom metadata, copy ``topeft/params/metadata.yml`` to a new location,
edit the clone, and pass it to the helper via ``--metadata``.  Keeping the clone
under version control (for example ``analysis/topeft_run2/configs/metadata_dev.yml``)
makes it easy to promote the same configuration to the full workflow once the
quickstart validation looks good.  When you are ready to run
``analysis/topeft_run2/run_analysis.py``, reference the same clone from a YAML
profile launched with ``--options`` by adding ``metadata: configs/metadata_dev.yml``
near the top of the file.  This keeps custom metadata tied to the profile while
allowing the CLI to keep relying on the scenario registry for standard runs.

### Metadata scenarios

The Run 2 metadata (``topeft/params/metadata.yml``) defines the channel bundles
exposed by the quickstart helpers.  Channel activation is now controlled solely
through the scenario list:

* ``TOP_22_006`` – Baseline reinterpretation categories, including the refined
  off-Z trilepton split.
* ``tau_analysis`` – Extends the baseline with the tau-enriched regions.
* ``fwd_analysis`` – Adds the forward-jet selections on top of the shared
  control suite.  Forward-jet counts are recomputed as integer per-event
  tallies before histogramming to avoid Awkward broadcasting pitfalls.

Pass ``--scenario`` one or more times to request the combinations you need.  For
example, ``--scenario TOP_22_006 --scenario tau_analysis`` layers the tau
categories onto the baseline job, while adding ``--scenario fwd_analysis`` in
the same command keeps the forward selections active as well.  For a
step-by-step walkthrough of each scenario—including how to mix them in YAML—see
the [Run 2 metadata scenarios guide](run2_scenarios.md).

During :func:`prepare_samples`, the scenario names are validated via the
:class:`analysis.topeft_run2.workflow.ChannelPlanner`.  Passing unknown
identifiers raises a ``ValueError`` before any processing starts so that
misconfigurations are caught early.

### Next steps {#run2-quickstart-next-steps}

!!! note "Next steps"
    Keep the workflow lightweight while layering on additional validation
    targets.  The [Run 2 metadata scenarios guide](run2_scenarios.md) walks
    through each bundle in detail.  To reuse this helper for the tau- or
    forward-focused reinterpretations, add the scenario switches directly on the
    command line:

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
    defaults remain aligned, then update the new file's metadata pointer, sample
    list, and scenarios as shown below:

    ```yaml
    # analysis/topeft_run2/configs/fullR2_run_tau_fwd.yml
    metadata: configs/metadata_dev.yml
    # (clone topeft/params/metadata.yml to configs/metadata_dev.yml before editing)
    jsonFiles:
      - ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json
    scenarios:
      - TOP_22_006
      - tau_analysis
      - fwd_analysis
    ```

    Launching ``python run_analysis.py --options configs/fullR2_run_tau_fwd.yml``
    keeps the validation fast while aligning the configuration with the
    quickstart dry run.  To activate an alternate profile defined in the same
    YAML, append the desired name after a colon—for example,
    ``--options configs/fullR2_run.yml:sr`` selects the signal-region preset.
    Remember that once ``--options`` is provided, command-line flags are ignored
    in favour of the YAML contents, so bake any overrides into the file before
    submitting the run.

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
