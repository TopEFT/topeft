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
   conda activate coffea-env
   ```

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
scenario and feature-tag combinations.

1. Change into the Run 2 analysis directory:

   ```bash
   cd analysis/topeft_run2
   ```

2. To reproduce the control-region job (the default profile), run:

   ```bash
   python run_analysis.py --options configs/fullR2_run.yml
   ```

   This executes the workflow with the shared defaults plus the `cr` profile,
   which skips all SR categories, splits lepton flavors, and runs a short test
   over the background configuration.

3. To launch the signal-region job, reuse the same YAML but select the `sr`
   profile:

   ```bash
   python run_analysis.py --options configs/fullR2_run.yml --options-profile sr
   ```

   The SR profile keeps the shared defaults while swapping in the signal and
   data metadata bundles and disabling the control regions.

4. Add any other command-line flags as needed.  Arguments supplied directly on
   the CLI (for example `--executor work_queue` or `--outname my_test_run`) take
   precedence over the YAML values, making it straightforward to adapt the
   quickstart to your local environment.

## Next steps

The quickstart configuration is intended to mirror the production defaults.
From here you can:

- Update the metadata paths in the YAML if you maintain custom sample lists.
- Override executor settings or chunking parameters on the command line to
  match your compute resources.
- Feed the generated histograms into the datacard machinery documented in the
  Run 2 README once you are ready to progress beyond the initial SR/CR runs.
