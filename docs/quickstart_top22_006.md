# TOP-22-006 Quickstart Guide

This guide walks through the minimal setup needed to reproduce the Run 2
signal-region (SR) and control-region (CR) histograms for the TOP-22-006
analysis using the refactored `run_analysis.py` entrypoint.  It assumes you
are starting from a clean checkout of the repository and would like to reuse
the shared metadata bundles defined in the Run 2 preset YAML configuration.

## Prerequisites

Before running the workflow you will need:

1. **A working Python environment.**  The analysis is developed and tested
   against the conda environment shipped with the repository.  Create and
   activate it with:

   ```bash
   conda env create -f environment.yml
   conda activate coffea202507
   ```

   The packaged environment tracks the Coffea 2025.7 release. Rebuild the TaskVine archive after dependency changes with `python -m topcoffea.modules.remote_environment` so remote workers stay in sync.

2. **The editable `topeft` and `topcoffea` packages.**  From the repository
   root install the local modules with:

   ```bash
   pip install -e .
   ```

   The workflow depends on the companion [`topcoffea`](https://github.com/TopEFT/topcoffea)
   utilities.  Install them in the same environment:

   ```bash
   git clone https://github.com/TopEFT/topcoffea.git
   cd topcoffea
   pip install -e .
   ```

3. **Access to the Run 2 metadata bundles.**  The YAML preset points to
   configuration files already tracked in this repository under
   `input_samples/cfgs/`.  Ensure your checkout includes the following files:

   - `mc_signal_samples_NDSkim.cfg`
   - `mc_background_samples_NDSkim.cfg`
   - `data_samples_NDSkim.cfg`
   - `mc_background_samples_cr_NDSkim.cfg`

   These files enumerate the JSON sample metadata produced for the Run 2
   campaign and do not require any additional downloads for the quickstart
   example.

## Running the workflow with the preset YAML

The Run 2 preset lives at
`analysis/topeft_run2/configs/fullR2_run.yml`.  It mirrors the historic
`fullR2_run.sh` script while setting the default scenario to `TOP_22_006` so
both the SR and CR jobs can be launched from the same configuration file.  When
you are ready to extend the quickstart to the tau or forward-jet regions, refer
to the [Run 2 metadata scenarios guide](run2_scenarios.md) for the additional
scenario combinations.  Channel activation is handled entirely by the metadata
scenarios declared in ``topeft/params/metadata.yml``.

1. Change into the Run 2 analysis directory:

   ```bash
   cd analysis/topeft_run2
   ```

2. Enable TaskVine in the YAML profile before launching the run.  Edit
   ``configs/fullR2_run.yml`` and set the profile you plan to execute to use the
   TaskVine backend:

   ```yaml
   profiles:
     cr:
       # ...existing options...
       executor: taskvine
   ```

   When you want to preserve both backends, copy the file to a new name (for
   example ``fullR2_run_taskvine.yml``) and adjust the ``executor`` there so the
   original futures-based profile stays available for local tests.

3. To reproduce the control-region job (the default profile), run:

   ```bash
   python run_analysis.py --options configs/fullR2_run.yml
   ```

   This executes the workflow with the shared defaults plus the ``cr`` profile,
   which skips all SR categories, splits lepton flavors, and runs a short test
   over the background configuration.  TaskVine will automatically upload the
   packaged environment if the helper tarball is present.

4. To launch the signal-region job, reuse the same YAML but select the ``sr``
   profile by appending it after a colon:

   ```bash
   python run_analysis.py --options configs/fullR2_run.yml:sr
   ```

   The SR profile keeps the shared defaults while swapping in the signal and
   data metadata bundles and disabling the control regions.

5. Adjust any additional options directly inside the YAML file.  Once
   ``--options`` is provided, command-line flags (such as ``--executor`` or
   ``--outname``) are ignored so that the captured configuration remains
   reproducible.  Drop ``--options`` entirely if you need a one-off CLI-driven
   run.

   When you need to test changes to the metadata catalogue, clone
   ``topeft/params/metadata.yml`` and pass the new path with ``--metadata``.  For
   example, to reuse the JSON inputs above with a custom metadata bundle in
   ``configs/metadata_dev.yml`` run::

       python run_analysis.py ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
           --metadata configs/metadata_dev.yml --executor taskvine --nworkers 1

## Next steps {#top22-quickstart-next-steps}

!!! note "Next steps"
    The quickstart profile keeps the run intentionally light so you can iterate
    quickly.  When you are ready to exercise the tau, forward, or off-Z focused
    reinterpretation categories, follow the
    [Run 2 metadata scenarios guide](run2_scenarios.md) and extend the preset
    before launching the workflow.  Because CLI flags are ignored when
    ``--options`` is present, edit the YAML directly (or drop ``--options`` for a
    one-off command-line run such as ``python run_analysis.py \
    input.json --scenario tau_analysis``).

    When you want to track the combination in version control, clone the preset
    into a dedicated YAML override so the workflow stays reproducible.  Copy
    ``analysis/topeft_run2/configs/fullR2_run.yml`` to
    ``analysis/topeft_run2/configs/fullR2_run_tau_fwd.yml`` and adjust the new
    file's metadata pointer and entries as shown below so that the extra
    categories are activated by default:

    ```yaml
    # analysis/topeft_run2/configs/fullR2_run_tau_fwd.yml
    metadata: configs/metadata_dev.yml
    # (clone topeft/params/metadata.yml to configs/metadata_dev.yml before editing)
    jsonFiles:
      - ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json
    scenarios:
      - TOP_22_006  # Baseline reinterpretation with the off-Z split
      - tau_analysis
      - fwd_analysis
    ```

    Launching ``python run_analysis.py --options configs/fullR2_run_tau_fwd.yml``
    keeps the validation loop short while layering on the extra categories.
