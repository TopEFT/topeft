[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5258003.svg)](https://doi.org/10.5281/zenodo.5258002)
[![CI](https://github.com/TopEFT/topcoffea/actions/workflows/main.yml/badge.svg)](https://github.com/TopEFT/topeft/actions/workflows/main.yml)
[![Coffea-casa](https://img.shields.io/badge/launch-Coffea--casa-green)](https://cmsaf-jh.unl.edu/hub/spawn)
[![codecov](https://codecov.io/gh/TopEFT/topcoffea/branch/master/graph/badge.svg?token=U2DMI1C22F)](https://codecov.io/gh/TopEFT/topcoffea)

# topeft
Top quark EFT analyses using the Coffea framework

## Repository contents
The `topeft/topeft` directory is set up to be installed as a pip installable package.
- `topeft/topeft`: A package containing modules and files that will be installed into the environment. 
- `topeft/setup.py`: File for installing the `topeft` package
- `topeft/analysis`: Subfolders with different analyses or studies. 
- `topeft/tests`: Scripts for testing the code with `pytest`. For additional details, please see the [README](https://github.com/TopEFT/topeft/blob/master/tests/README.md) in the `tests` directory.
- `topeft/input_samples`: Configuration files that point to root files to process.

## Getting started

### Setting up
If conda is not already available, download and install it:
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > conda-install.sh
bash conda-install.sh
```
The topeft directory is set up to be installed as a python package. First clone the repository as shown, then run the following commands to set up the environment (note that `environment.yml` is a file that is a part of the `topeft` repository, so you should `cd` into `topeft` before running the command):
```
git clone https://github.com/TopEFT/topeft.git
cd topeft
unset PYTHONPATH # To avoid conflicts.
conda env create -f environment.yml
conda activate coffea202507
pip install -e .
```
The `-e` option installs the project in editable mode (i.e. setuptools "develop mode"). If you wish to uninstall the package, you can do so by running `pip uninstall topcoffea`. 
The `topcoffea` package upon which this analysis also depends is not yet available on `PyPI`, so we need to clone the `topcoffea` repo and install it ourselves.
```
cd /your/favorite/directory
git clone https://github.com/TopEFT/topcoffea.git
cd topcoffea
pip install -e .
cd ../topeft
python -m topcoffea.modules.remote_environment
```
Now all of the dependencies have been installed and the `topeft` repository is ready to be used. The packaged environment targets the Coffea 2025.7 release and can be re-generated at any time with `python -m topcoffea.modules.remote_environment`. The next time you want to use the project, activate the environment via `conda activate coffea202507`.


### TaskVine distributed execution

The packaged archive produced by `python -m topcoffea.modules.remote_environment` is uploaded automatically when runs target the TaskVine executor. Launching a distributed job therefore becomes:

1. Configure the workflow with TaskVine enabled (for example, selecting the Run 2 control preset):

   ```bash
   python run_analysis.py \
       ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
       --options configs/fullR2_run.yml \
       --executor taskvine
   ```

2. Start one or more workers that match the manager name advertised by the run. A minimal local worker looks like:

   ```bash
   vine_worker --cores 1 --memory 8000 --disk 8000 -M ${USER}-taskvine-coffea
   ```

   For HTCondor-backed pools, `condor_submit_workers` and other submission helpers ship with TaskVine; point them at the same manager string. Refer to the [remote environment maintenance guide](docs/environment_packaging.md) for details on rebuilding the tarball whenever dependencies change.

Work Queue remains supported for legacy deployments. The historic configuration guide is preserved in [README_WORKQUEUE.md](README_WORKQUEUE.md).

### To run an example job

The Run 2 workflow now ships with YAML presets under
`analysis/topeft_run2/configs/`.  They mirror the historic shell scripts while
allowing you to toggle signal and control regions from the command line.

1. `cd` into `analysis/topeft_run2` and resolve the example input:

   ```bash
   cd analysis/topeft_run2
   wget -nc http://www.crc.nd.edu/~kmohrman/files/root_files/for_ci/ttHJet_UL17_R1B14_NAOD-00000_10194_NDSkim.root
   ```

2. Launch the control-region (CR) pass of the reinterpretation with the shared
   YAML configuration:

   ```bash
   python run_analysis.py ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
       --options configs/fullR2_run.yml
   ```

3. Switch to the signal-region (SR) profile by pointing to the same YAML bundle
   and appending `:sr` to select the preset:

   ```bash
   python run_analysis.py ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
       --options configs/fullR2_run.yml:sr
   ```

Both commands emit verbose task summaries before execution so you can confirm
the samples, regions, and histograms that will be processed.  When `--options`
is present the YAML file becomes the single source of truth—CLI flags such as
`--executor` or `--summary-verbosity` are ignored, so embed any overrides
directly in the configuration or drop `--options` for ad-hoc command-line runs.
You can still run without YAML by calling the scripts in
`analysis/topeft_run2/examples/`, but the config-based approach keeps SR/CR
toggles, output names, and executor settings in a single place.

#### Supported metadata scenarios

The default metadata bundle matches the TOP-22-006 reinterpretation categories,
but the YAML and CLI entry points can target additional scenarios defined in
`topeft/params/metadata.yml`:

- `TOP_22_006` – Baseline Run 2 reinterpretation with the shared control suite.
- `tau_analysis` – Adds the tau-enriched signal/control regions needed for the
  dedicated tau study.
- `fwd_analysis` – Enables the forward-jet categories while reusing the common
  control regions.

Channel activation is controlled entirely by these scenario selections.  Scenarios
can be combined by copying the relevant blocks in your YAML profile or—when
running without `--options`—by passing `--scenario` multiple times on the CLI.
Detailed instructions for running each bundle
individually and mixing them in YAML are provided in the
[Run 2 metadata scenarios guide](docs/run2_scenarios.md).  For a guided
walkthrough of the Run 2 workflow—including environment setup, metadata
bundles, and extended examples—see the [TOP-22-006 quickstart
guide](docs/quickstart_top22_006.md) and the [Run 2 quickstart
overview](docs/quickstart_run2.md).

Additional reference material for the module structure and configuration helpers
is available in the [analysis processing primer](docs/analysis_processing.md),
the [YAML configuration guide](docs/run_analysis_configuration.md), and the
[`run_analysis.py` CLI/YAML reference](docs/run_analysis_cli_reference.md).

### Metadata configuration

The Run 2 helpers, quickstarts, and processors all read from
`topeft/params/metadata.yml`.  This YAML file is the single source of truth for
which regions run, which histogram variables are kept, and which systematic
variations are evaluated.  Key sections include:

| Metadata key | Controls | Processor impact |
| --- | --- | --- |
| `channels.groups[].regions` | Channel definitions, lepton/jet binning, and application tags. | Drives channel activation inside `ChannelPlanner` and the per-task metadata passed to the Coffea processor. |
| `channels.groups[].histogram_variables` | Include/exclude lists for variables per region. | Filters the histogram catalogue built by `HistogramPlanner` so the processor only schedules the variables you request. |
| `variables` | Bin edges, labels, and callable definitions for histograms. | Supplies the axis configuration for every histogram task. |
| `scenarios` | Scenario bundles that map friendly names to channel groups. | Enables CLI/YAML `--scenario` toggles and the quickstart presets. |
| `systematics` | Weight and object variations (with optional year/scenario guards). | Determines which sum-of-weights keys and shape variations the processor evaluates when `--do-systs` is set. |
| `golden_jsons` | Year-tagged certified-luminosity files. | Guides data-quality filtering when running over data JSONs. |

When you need a custom configuration, copy the metadata file to a new name (for
example `analysis/topeft_run2/configs/metadata_myteam.yml`) so your edits stay
isolated.  Both the quickstart helper and `run_analysis.py` accept the clone via
`--metadata`, removing the need to swap files in `topeft/params/metadata.yml`.
When you are driving the workflow through a YAML profile, set a top-level
`metadata: configs/metadata_myteam.yml` entry so the override lives alongside
the rest of your configuration knobs.
For example, the following command launches the full workflow with a bespoke
metadata bundle kept alongside your analysis configs:

```bash
python run_analysis.py ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
    --metadata configs/metadata_myteam.yml --executor taskvine --nworkers 1 --chunksize 128000
```

The [metadata configuration guide](docs/run_analysis_configuration.md#metadata-configuration)
expands on the available keys and shows how the planners consume them.

### Migration note: scenario-only channel activation

As of this release, the ``--channel-feature`` flag has been retired.  Channel
activation is now driven entirely by the metadata scenarios documented in
``topeft/params/metadata.yml``.  Use ``--scenario`` (either repeated on the
command line or listed in a YAML profile) to enable the tau, forward, or other
specialised selections.  The [Run 2 metadata scenarios guide](docs/run2_scenarios.md)
collects end-to-end examples of the recommended combinations.

If you prefer a minimal smoke test before running the full configuration,
consider the quickstart helper:

```bash
python -m topeft.quickstart input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
    --prefix root://cmsxrootd.fnal.gov/ \
    --output quickstart-output
```

The helper resolves your samples, validates the requested scenario, and launches
a short futures-based run.  Detailed explanations of each switch are covered in
[docs/quickstart_run2.md](docs/quickstart_run2.md).  When you are ready to scale
out, switch to the TaskVine executor described above so the packaged environment
can be reused across distributed resources.  The legacy Work Queue workflow is
still documented in [README_WORKQUEUE.md](README_WORKQUEUE.md) for historical
setups that require it.


## How to contribute

If you would like to push changes to the repo, please make a branch and open a PR and ensure that the CI passes. Note that if you are developing on a fork, the CodeCov CI will fail.

Note, if your branch gets out of date as other PRs are merged into the master branch, you may need to merge those changes into your brnach and fix any conflicts prior to your PR being merged. 

If your branch changes anything that is expected to causes the yields to change, please run the following to updated the reference yields:
```bash
cd analysis/topEFT/
sh remake_ci_ref_yields.sh
sh remake_ci_ref_datacard.sh
```
The first script remakes the reference `json` file for the yields, and the second remakes the reference `txt` file for the datacar maker. If you are sure these change are expected, commit and push them to the PR.

## Installing and running pytest locally
To install `pytest` for local testing, run:
```bash
conda install -c conda-forge pytest pytest-cov
```
where `pytest-cov` is only used if you want to locally check the code coverage.

The `pytest` commands are run automatically in the CI. If you would like to run them locally, you can simply run:
```bash
pytest
```
from the main topcoffea directory. This will run _all_ the tests, which will take ~20 minutes. To run a subset, use e.g.:
```bash
pytest -k test_futures
```
where `test_futures` is the file/test you would like to run (check the `tests` directory for all the available tests, or write your own and push it!). If you would also like to see how the coverage changes, you can add `--cov=./ --cov-report=html` to `pytest` commands. This will create an `html` directory that you can then copy to any folder which you have web access to (e.g. `~/www/` on Earth) For a better printout of what passed and failed, add `-rP` to the `pytest` commands.



## To reproduce the TOP-22-006 histograms and datacards

The [v0.5 tag](https://github.com/TopEFT/topcoffea/releases/tag/v0.5) was used to produce the results in the TOP-22-006 paper.

1. Run the processor to obtain the histograms (from the skimmed naod files). Use the `fullR2_run.sh` script in the `analysis/topEFT` directory.
    ```
    time source fullR2_run.sh
    ```

2. Run the datacard maker to obtain the cards and templates from SM (from the pickled histogram file produced in Step 1, be sure to use the version with the nonprompt estimation, i.e. the one with `_np` appended to the name you specified for the `OUT_NAME` in `fullR2_run.sh`). This step would also produce scalings-preselect.json file which the later version is necessary for IM workspace making. Note that command option `--wc-scalings` is not mandatory but to enforce the ordering of wcs in scalings. Add command `-A` to include all EFT templates in datacards for previous AAC model. Add option `-C` to run on condor.
    ```
    time python make_cards.py /path/to/your/examplename_np.pkl.gz --do-nuisance --var-lst lj0pt ptz -d /scratch365/you/somedir --unblind --do-mc-stat --wc-scalings cQQ1 cQei cQl3i cQlMi cQq11 cQq13 cQq81 cQq83 cQt1 cQt8 cbW cpQ3 cpQM cpt cptb ctG ctW ctZ ctei ctlSi ctlTi ctli ctp ctq1 ctq8 ctt1
    ```

3. Run the post-processing checks on the cards to look for any unexpected errors, to grab the right set of ptz and lj0pt templates/cards used in TOP-22-006, and to get final version of scalings.json file. The script will copy the relevant cards/templates/ and create the json file to a directory called `ptz-lj0pt_withSys` that it makes inside of the directory you pass that points to the cards and templates made in Step 2. This `ptz-lj0pt_withSys` is the directory that can be copied to wherever you plan to run the `combine` steps (e.g. PSI). Can also run this on condor with `-c`.
    ```
    time python datacards_post_processing.py /scratch365/you/somedir -s
    ```

4. Check the yields with `get_datacard_yields.py` script. This scrip will read the datacards in the directory produced in Step 3 and will dump the SM yields (summed over jet bins) to the screen (the text is formatted as a latex table). Use the `--unblind` option if you want to also see the data numbers.
    ```
    python get_datacard_yields.py /scratch365/you/somedir/ptz-lj0pt_withSys/ --unblind
    ```

5. Proceed to the [Steps for reproducing the "official" TOP-22-006 workspace](https://github.com/TopEFT/EFTFit#steps-for-reproducing-the-official-top-22-006-workspace) steps listed in the EFTFit Readme. Remember that in addition to the files cards and templates, you will also need the `selectedWCs.txt` file. 


