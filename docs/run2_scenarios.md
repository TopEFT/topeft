# Run 2 metadata scenarios guide

This guide expands on the quickstart documentation by showing how to run the
Run 2 workflows with the three metadata scenarios distributed in
`topeft/params/metadata.yml`:

- `TOP_22_006`
- `tau_analysis`
- `fwd_analysis`

Each section starts from a clean checkout of the repository with the editable
``topeft`` and ``topcoffea`` packages installed (see the
[TOP-22-006 quickstart](quickstart_top22_006.md#prerequisites) for the
recommended environment setup).  The commands shown below assume that you are in
``analysis/topeft_run2`` unless otherwise noted.

!!! important
    Once ``--options`` is supplied the YAML file becomes authoritative.  Append
    ``:profile`` to select a preset (for example ``configs/fullR2_run.yml:sr``)
    and edit the YAML directly to enable additional scenarios.  Drop
    ``--options`` entirely if you need to experiment with ad-hoc CLI flags.

!!! note
    The ``--channel-feature`` flag has been retired.  Use the scenarios below—on
    their own or in combination—to activate the tau, forward, and off-Z
    variations.

## Running individual scenarios

### TOP_22_006 baseline

The TOP-22-006 scenario powers both the Run 2 quickstart helper and the default
YAML profile.  To launch the full control-region job from the preset YAML, run:

```bash
python run_analysis.py --options configs/fullR2_run.yml
```

Switch to the signal-region profile by appending ``:sr`` to the YAML path:

```bash
python run_analysis.py --options configs/fullR2_run.yml:sr
```

You can reach the same configuration from the quickstart helper with:

```bash
python -m topeft.quickstart \
    ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
    --prefix root://cmsxrootd.fnal.gov/ \
    --scenario TOP_22_006
```

### Tau-enriched workflow

The ``tau_analysis`` bundle adds the dedicated signal and control regions needed
by the tau reinterpretation.  When using ``run_analysis.py`` with the preset
YAML, edit the configuration to include ``tau_analysis`` in the ``scenarios``
list (for example by copying ``configs/fullR2_run.yml`` to
``configs/fullR2_run_tau.yml`` and updating the defaults).  After the edit, run
``python run_analysis.py --options configs/fullR2_run_tau.yml``.

From the quickstart helper the equivalent command is:

```bash
python -m topeft.quickstart \
    ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
    --prefix root://cmsxrootd.fnal.gov/ \
    --scenario tau_analysis
```

### Forward-jet workflow

The ``fwd_analysis`` bundle activates the forward-jet categories while reusing
the shared Run 2 control regions.  Update your YAML preset to list
``fwd_analysis`` under ``scenarios`` (or create a dedicated profile) before
launching ``python run_analysis.py --options <your_yaml>``.

And the matching quickstart invocation is:

```bash
python -m topeft.quickstart \
    ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
    --prefix root://cmsxrootd.fnal.gov/ \
    --scenario fwd_analysis
```

## Combining scenarios

Metadata scenarios can be combined to produce hybrid category selections.  The
planner validates that the requested combinations are compatible before any
processing starts, so invalid bundles fail fast.  The baseline ``TOP_22_006``
scenario already ships with the refined off-Z split; layering ``tau_analysis``
or ``fwd_analysis`` on top keeps those specialised categories active alongside
the shared control regions.

### YAML profiles

To combine scenarios in a YAML profile, set the ``scenario`` key to a list.  The
following snippet extends the default Run 2 configuration with both the tau and
forward regions:

```yaml
# configs/fullR2_run_tau_fwd.yml
infile: ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json
scenario:
  - TOP_22_006
  - tau_analysis
  - fwd_analysis
```

YAML values merge with the base profile when the file is passed through
``--options``.  CLI flags are ignored in this mode, so bake any additional
overrides into the file before launching the workflow.

### Command-line runs without YAML

You can request the same combination directly on the command line by dropping
``--options``.  Each time ``--scenario`` is provided the value is appended to the
active set:

```bash
python run_analysis.py \
    --scenario TOP_22_006 \
    --scenario tau_analysis \
    --scenario fwd_analysis
```

For quickstart tests, the equivalent call is:

```bash
python -m topeft.quickstart \
    ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
    --prefix root://cmsxrootd.fnal.gov/ \
    --scenario TOP_22_006 \
    --scenario tau_analysis \
    --scenario fwd_analysis
```

The planner will echo the resolved scenario list before launching the Coffea
processor, making it easy to confirm the combined configuration.
