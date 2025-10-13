# Sample metadata reference

The Run 2 workflow consumes per-sample JSON manifests to learn where files live,
which histogram axis labels to assign, and how to normalize the resulting event
weights.  This page summarizes the required keys, optional systematic payloads,
and common pitfalls so you can validate inputs without digging through the
loader implementation.

## Required keys

Every JSON manifest must provide the fields listed below.  The
:class:`analysis.topeft_run2.run_analysis_helpers.SampleLoader` enforces types and
fills defaults where appropriate during :func:`collect` and
:func:`load`. 【F:analysis/topeft_run2/run_analysis_helpers.py†L183-L242】

| Key | Type | Purpose | Notes |
| --- | ---- | ------- | ----- |
| ``xsec`` | number | Physics cross section used to scale MC yields. | Required even for data samples; the helper coerces the value to ``float`` before processing. |
| ``year`` | string | Data-taking era used to select metadata scenarios and corrections. | Accepts values such as ``"2016preVFP"``, ``"2017"``, or ``"2018"``. |
| ``treeName`` | string | NanoAOD tree containing the events. | Typically ``"Events"`` for Run 2 NanoAOD samples. |
| ``histAxisName`` | string | Unique identifier for histogram bookkeeping. | Used as the axis name in the Coffea accumulator and must be distinct across the job. |
| ``files`` | list of strings | Relative or absolute paths to NanoAOD ROOT files. | The loader prepends any redirector/prefix before scheduling work. |
| ``nEvents`` | integer | Number of events stored in the manifest. | Converted to ``int`` and reported in run summaries. |
| ``nGenEvents`` | integer | Number of generated events prior to filtering. | Required for MC weighting and recorded as ``int``. |
| ``nSumOfWeights`` | number | Nominal sum of generator weights. | Always coerced to ``float``; for data samples this usually matches ``nEvents``. |
| ``isData`` | boolean | Declares whether the sample should be treated as collision data. | Determines whether systematic sums of weights are required (see below). |

Additional keys—such as ``options``, ``WCnames`` and dataset ``path``
fragments—are carried through untouched so they can be consumed by higher-level
helpers or downstream plotting tools.

## Optional systematic sums of weights

When ``isData`` is ``false`` the loader will look for extra ``nSumOfWeights_*``
entries that correspond to metadata-driven systematic variations.  The keys must
match the variations extracted from ``topeft/params/metadata.yml`` via
:func:`analysis.topeft_run2.run_analysis_helpers.weight_variations_from_metadata`.
Any variation found in the JSON is converted to ``float`` before being attached
to the sample record. 【F:analysis/topeft_run2/run_analysis_helpers.py†L232-L241】

A compact MC manifest with ISR, FSR, and scale variations would therefore look
like:

```json
{
  "xsec": 0.2151,
  "year": "2017",
  "treeName": "Events",
  "histAxisName": "ttHJet_privateUL17",
  "files": [
    "ttHJet_UL17_R1B14_NAOD-00000_10194_NDSkim.root"
  ],
  "nEvents": 7993714,
  "nGenEvents": 40553247,
  "nSumOfWeights": 575.3735616864701,
  "isData": false,
  "nSumOfWeights_ISRUp": 551.6043364015839,
  "nSumOfWeights_ISRDown": 595.296014965334,
  "nSumOfWeights_FSRUp": 575.0603156871065,
  "nSumOfWeights_FSRDown": 576.5027201552521,
  "nSumOfWeights_renormUp": 490.3073273065021,
  "nSumOfWeights_renormDown": 687.0718575833407,
  "nSumOfWeights_factUp": 533.2206309224223,
  "nSumOfWeights_factDown": 625.4158625663422,
  "nSumOfWeights_renormfactUp": 454.65945259538086,
  "nSumOfWeights_renormfactDown": 747.2810760952543
}
```

Omitting a variation that is requested by the workflow results in a
``Missing weight variation`` error before Coffea jobs are launched.  Either drop
the variation from the metadata scenario or record the corresponding
``nSumOfWeights_*`` entry in the JSON to resolve the failure.

## Redirectors and remote paths

``run_analysis.py`` and the quickstart helper accept ``--prefix`` (aliased as
``--redirector``) to prepend a URI before each file.  The loader stores the value
under ``redirector`` and concatenates it with each entry in ``files`` when
building the job list. 【F:analysis/topeft_run2/run_analysis_helpers.py†L198-L224】
Use this mechanism whenever the JSON lists EOS or site-local paths that require
XRootD access.

You can also bake a redirector into a ``.cfg`` manifest by inserting the desired
prefix on its own line before including JSON files.  The parser treats unknown
entries as redirector updates, so the following snippet mixes local files and an
XRootD endpoint:

```
root://cmsxrootd.fnal.gov/
/uscms/home/user/local_samples/
signal_samples.json
```

## Data versus MC differences

Set ``isData`` to ``true`` for collision data.  In this mode the loader skips the
optional systematic sums of weights and relies on the nominal ``nSumOfWeights``
to be identical to ``nEvents``. 【F:analysis/topeft_run2/run_analysis_helpers.py†L232-L235】【F:input_samples/sample_jsons/data_samples/2018/MuonEG_A-UL2018_NDSkim.json†L1-L19】
A representative data manifest looks like:

```json
{
  "xsec": 1.0,
  "year": "2018",
  "treeName": "Events",
  "histAxisName": "dataUL18",
  "files": [
    "/store/user/awightma/skims/data/NAOD_ULv9_new-lepMVA-v2/FullRun2/v3/MuonEG_A_UL2018/output_2047.root"
  ],
  "nEvents": 32958503,
  "nGenEvents": 32958503,
  "nSumOfWeights": 32958503.0,
  "isData": true
}
```

Even though the cross section is unused for data, keeping ``xsec`` at ``1.0``
ensures the schema stays uniform across the manifest collection.

## Common validation errors and quick fixes

The workflow performs several checks before dispatching Coffea tasks.  The table
below outlines frequent failures and the quickest way to recover.

| Error message | Root cause | Fix |
| ------------- | ---------- | --- |
| ``FileNotFoundError: Input file /path/to/sample.json not found!`` | The manifest path is relative to a different working directory or contains a typo. | Run the command from the repository root or switch to an absolute path.  When using ``.cfg`` bundles, prefer paths relative to the bundle itself. 【F:analysis/topeft_run2/run_analysis_helpers.py†L209-L228】 |
| ``Missing weight variation: nSumOfWeights_ISRUp`` | The metadata scenario requests ISR variations but the JSON omits the matching sum of weights. | Add the ``nSumOfWeights_ISRUp`` (and corresponding ``Down``) entries to the manifest or drop the variation from ``systematics`` in the metadata file. |
| ``ValueError: Unsupported input type`` | A ``.cfg`` file references a non-JSON asset (for example, a TXT list). | Convert the list into JSON manifests or expand the ``.cfg`` to point directly at the ``.json`` files. |

### Example fixes

*Missing redirector for remote files*

If the JSON lists EOS paths without a redirector, prepend one via the CLI:

```bash
python -m topeft.quickstart samples.json --prefix root://cmsxrootd.fnal.gov/
```

To make the change permanent, add the prefix line inside the ``.cfg`` bundle so
all downstream scripts inherit the redirector automatically.

*Adding absent weight variations*

```json
{
  "nSumOfWeights": 575.37,
  "nSumOfWeights_ISRUp": 551.60,
  "nSumOfWeights_ISRDown": 595.30
}
```

Recording the up/down sums allows ``--do-systs`` jobs to proceed without editing
any Python code.

With these guidelines in place you can confidently prepare, validate, and share
sample manifests across the quickstart helpers and the full Run 2 workflow.
